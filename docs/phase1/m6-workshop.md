# Воркшоп M6 · Профиль игрока и retention: pandas, потом Polars

<span class="lecture-meta">Воркшоп к модулю M6 · ориентир 5-7 ч</span>

## Что отрабатываем

Этот воркшоп прогоняет руками всё ядро модуля M6 на одних и тех же данных игроков: `groupby`/`agg` с именованными агрегатами, `merge` с защитой от разъезда строк, `pivot_table` для когортной матрицы, `resample`/`rolling` по времени. Затем ты переписываешь ключевой пайплайн на Polars в ленивом режиме и честно меряешь, что выиграл — скорость и читаемость.

Понятия, которые отрабатываем: split-apply-combine, `agg` vs `transform`, `validate=` в merge, широкий/длинный формат, временные ряды, eager vs lazy, projection/predicate pushdown.

Артефакт на выходе: один файл с двумя версиями главного пайплайна — pandas и Polars lazy — плюс замер времени и проверка, что обе дают одинаковый результат.

## Бизнес-кейс

!!! example "Ситуация"

    Ты — аналитик в команде iGaming-продукта. К тебе приходит **Head of Retention**: дашборд показывает, что новые когорты «тают» быстрее обычного, но цельной картины нет — никто не считал retention по неделям жизни и не знает, кто на самом деле приносит кассу. Цифры по ситуации (иллюстративные):

    - **Retention W1** в мартовских когортах просел с 26% до 19% — это значит, что из условных 20k регистраций до второй недели доживает на ~1 400 игроков меньше, и при среднем GGR ~1 200 ₽ на удержанного игрока это ориентировочно **1.5-1.7 млн ₽** недополученной выручки в месяц.
    - Маркетинг параллельно просит понять, **какие каналы и страны** гонят «одноразовых» игроков, а какие — китов: бюджет закупки на следующий квартал (порядка **4 млн ₽**) перераспределяется по этим выводам.
    - От твоего ответа зависит решение: **запускать ли welcome-цепочку удержания** на самые отваливающиеся когорты и **куда сместить закуп** — в каналы с длинным хвостом активных игроков или резать те, что льют отвал.

    Ограничение: данных по реальным депозитам пока нет, работаешь по ставкам и регистрациям; ответ нужен **к планёрке через 2 дня**, поэтому пайплайн должен быть воспроизводимым и быстрым (отсюда и перенос на Polars — пересчитывать когорты придётся не раз).

!!! tip "Окружение"

    Заведи изолированный проект через uv. Это стек 2026 из модуля.

    ```bash
    uv init m6-workshop && cd m6-workshop
    uv add pandas polars pyarrow numpy
    ```

    Код ниже запускай через `uv run python script.py`.

## Данные

Никаких внешних файлов. Генерируем синтетику с фиксированным seed и сохраняем в Parquet — ровно тот формат, который модуль называет рабочим. Две таблицы: `registrations` (по строке на игрока) и `bets` (много строк на игрока).

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_PLAYERS = 20_000

channels = np.array(["facebook", "google", "organic", "affiliate"])
countries = np.array(["UZ", "KZ", "RU", "TR"])

reg_ts = pd.Timestamp("2026-03-01") + pd.to_timedelta(
    rng.integers(0, 60 * 24 * 3600, N_PLAYERS), unit="s"
)
registrations = pd.DataFrame({
    "player_id": [f"u{i}" for i in range(N_PLAYERS)],
    "reg_ts": reg_ts,
    "country": rng.choice(countries, N_PLAYERS, p=[0.4, 0.3, 0.2, 0.1]),
    "channel": rng.choice(channels, N_PLAYERS, p=[0.35, 0.3, 0.2, 0.15]),
})

# у каждого игрока разное число ставок (gamma -> длинный хвост китов)
n_bets = rng.gamma(shape=1.5, scale=8, size=N_PLAYERS).astype(int) + 1
reg_repeat = registrations.loc[registrations.index.repeat(n_bets)].reset_index(drop=True)

life_days = rng.exponential(scale=14, size=len(reg_repeat))
bet_ts = reg_repeat["reg_ts"] + pd.to_timedelta(life_days, unit="D")
stake = np.round(rng.gamma(2.0, 15.0, len(reg_repeat)), 2)
payout = np.round(stake * rng.uniform(0.0, 1.8, len(reg_repeat)), 2)

bets = pd.DataFrame({
    "player_id": reg_repeat["player_id"],
    "bet_ts": bet_ts,
    "bets": stake,
    "payouts": payout,
    "channel": reg_repeat["channel"],
})

registrations.to_parquet("registrations.parquet")
bets.to_parquet("bets.parquet")
print("registrations:", registrations.shape, "| bets:", bets.shape)
```

Получилось: ~20k игроков и порядка 250-300k ставок с реалистичным длинным хвостом (немного китов, много мелких). `bet_ts` всегда позже `reg_ts`, поэтому когорты считаются корректно.

## Ход работы

### Шаг 1: профиль игрока через groupby + named aggregation

Зачем: отрабатываем split-apply-combine и именованные агрегаты — основу любого LTV-профиля. Разбиваем по `player_id`, к каждой группе применяем набор агрегатов, собираем в таблицу.

```python
import pandas as pd

bets = pd.read_parquet("bets.parquet")
bets["ggr"] = bets["bets"] - bets["payouts"]

player_stats = (
    bets
    .groupby("player_id", as_index=False)
    .agg(
        total_ggr=("ggr", "sum"),
        bet_count=("ggr", "size"),
        active_days=("bet_ts", lambda s: s.dt.normalize().nunique()),
        last_bet=("bet_ts", "max"),
    )
    .sort_values("total_ggr", ascending=False)
)
print(player_stats.head())
```

Что получилось: таблица «один игрок — одна строка» с суммарным GGR, числом ставок, числом активных дней и датой последней ставки. Строк ровно столько, сколько игроков делали ставки — `agg` свернул каждую группу в одну строку.

### Шаг 2: transform — доля игрока внутри страны

Зачем: это место, где видно разницу `agg` и `transform`. Нужна групповая характеристика (сумма GGR по стране), приклеенная обратно к каждой строке. `agg` бы схлопнул таблицу, `transform` сохраняет длину.

```python
prof = player_stats.merge(
    registrations[["player_id", "country", "channel"]],
    on="player_id", how="left", validate="one_to_one",
)
prof["country_ggr"] = prof.groupby("country")["total_ggr"].transform("sum")
prof["share_in_country"] = prof["total_ggr"] / prof["country_ggr"]
print(prof.nlargest(10, "share_in_country")[["player_id", "country", "share_in_country"]])
```

Что получилось: к каждому игроку приклеена доля его GGR в его стране. Топ-10 — это «киты», на которых держится касса региона.

!!! question "Проверь себя"

    1. Почему `active_days` нельзя посчитать обычным `"count"`?
    2. Что вернул бы `groupby("country")["total_ggr"].agg("sum")` вместо `transform("sum")` по форме?
    3. Зачем здесь `validate="one_to_one"`?

??? success "Ответы"

    1. `count` посчитал бы число ставок, а не число уникальных дней; нужен `nunique` по нормализованной дате.
    2. `agg` вернул бы одну строку на страну (схлопнул таблицу); `transform` вернул результат той же длины, что и группа — по строке на игрока.
    3. `player_stats` и `registrations` оба по строке на игрока: ждём строго один-к-одному, и pandas упадёт, если внезапно появится дубль.

### Шаг 3: merge и тихий баг разъезда строк

Зачем: главный молчаливый баг аналитики из модуля. Специально создаём неуникальный ключ справа и смотрим, как `left join` раздувает сумму, а `validate=` ловит это.

```python
dep = registrations[["player_id"]].copy()
dep["deposit"] = 100.0
dep_bad = pd.concat([dep, dep.iloc[:500]], ignore_index=True)  # 500 дублей

m = registrations.merge(dep_bad, on="player_id", how="left")
print("строк до:", len(registrations), "| после merge:", len(m),
      "| сумма депозитов:", m["deposit"].sum())  # раздулась

try:
    registrations.merge(dep_bad, on="player_id", how="left", validate="one_to_many")
except Exception as e:
    print("validate поймал:", type(e).__name__, e)
```

Что получилось: без `validate` сумма депозитов оказалась больше истинной (500 лишних строк по 100), и это нигде не упало — отчёт бы молча соврал. С `validate="one_to_many"` pandas сразу падает с понятной ошибкой. Вывод в привычку: при каждом merge ставь `validate=` и сверяй `len()`.

### Шаг 4: временные ряды — resample и rolling

Зачем: отрабатываем `datetime`-операции. Считаем дневной GGR (`resample`), сглаживаем 7-дневным скользящим средним (`rolling`) и считаем рост день-к-дню через `shift`.

```python
ts = bets.set_index("bet_ts").sort_index()

daily = ts["ggr"].resample("D").sum()
daily_smooth = daily.rolling(window=7, min_periods=1).mean()
daily_growth = daily / daily.shift(1) - 1

trend = pd.DataFrame({"ggr": daily, "ggr_7d": daily_smooth, "growth": daily_growth})
print(trend.head(10))
```

Что получилось: дневной ряд GGR, сглаженная кривая тренда без шума и относительный прирост к прошлому дню. `min_periods=1` убирает `NaN` в первые дни ряда.

### Шаг 5: когортная retention-матрица через pivot_table

Зачем: финальная сборка из модуля — длинный формат превращаем в широкую матрицу. Это и есть инструмент cohort-анализа: строки — когорта по неделе регистрации, столбцы — неделя жизни.

```python
m = bets.merge(registrations[["player_id", "reg_ts"]],
               on="player_id", how="inner", validate="many_to_one")
m["cohort_week"] = m["reg_ts"].dt.to_period("W").dt.start_time
m["week_offset"] = (m["bet_ts"] - m["reg_ts"]).dt.days // 7

cohort = (
    m.groupby(["cohort_week", "week_offset"])["player_id"]
     .nunique()
     .reset_index(name="active_players")
)
retention = cohort.pivot_table(
    index="cohort_week", columns="week_offset",
    values="active_players", fill_value=0,
)
retention_pct = retention.div(retention[0], axis=0).round(3)
print(retention_pct.iloc[:, :6])
```

Что получилось: матрица retention в долях от нулевой недели. По строкам читается, как когорта тает по неделям жизни — главный отвал обычно между W0 и W1.

!!! question "Проверь себя"

    1. Почему `pivot_table`, а не голый `pivot`?
    2. Что именно делит `retention.div(retention[0], axis=0)`?
    3. Зачем `validate="many_to_one"` на этом merge?

??? success "Ответы"

    1. На паре (cohort_week, week_offset) бывают дубли до агрегации; `pivot_table` агрегирует через `aggfunc`, а `pivot` упал бы на неуникальности.
    2. Каждую строку-когорту на её же значение нулевой недели — получаем долю выживших игроков относительно старта когорты.
    3. Много ставок (`bets`) к одной регистрации: ждём many-to-one, pandas проверит, что справа ключ уникален.

### Шаг 6: главный пайплайн на Polars lazy

Зачем: тот самый перенос pandas → Polars из модуля. Берём профиль игрока (Шаг 1) как ключевой пайплайн и пишем его в ленивом режиме: `scan_parquet` не читает файл сразу, `collect` запускает оптимизированный план с pushdown и параллелизмом.

```python
import polars as pl

player_stats_pl = (
    pl.scan_parquet("bets.parquet")
    .with_columns((pl.col("bets") - pl.col("payouts")).alias("ggr"))
    .group_by("player_id")
    .agg(
        pl.col("ggr").sum().alias("total_ggr"),
        pl.len().alias("bet_count"),
        pl.col("bet_ts").dt.date().n_unique().alias("active_days"),
        pl.col("bet_ts").max().alias("last_bet"),
    )
    .sort("total_ggr", descending=True)
    .collect()
)
print(player_stats_pl.head())
```

Что получилось: структурно тот же отчёт, но без индекса и в выражениях `pl.col(...)`. За `scan_parquet` Polars прочитает с диска только нужные столбцы (projection pushdown) и распараллелит агрегацию по ядрам.

### Шаг 7: замер скорости и проверка совпадения

Зачем: модуль обещает выигрыш Polars — проверяем руками, а не на веру. И обязательно сверяем, что результаты совпадают: быстрый, но неверный пайплайн бесполезен.

```python
import time

def bench(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out

def pandas_pipe():
    b = pd.read_parquet("bets.parquet")
    b["ggr"] = b["bets"] - b["payouts"]
    return (b.groupby("player_id", as_index=False)
             .agg(total_ggr=("ggr", "sum"), bet_count=("ggr", "size")))

def polars_pipe():
    return (pl.scan_parquet("bets.parquet")
              .with_columns((pl.col("bets") - pl.col("payouts")).alias("ggr"))
              .group_by("player_id")
              .agg(pl.col("ggr").sum().alias("total_ggr"),
                   pl.len().alias("bet_count"))
              .collect())

t_pd, r_pd = bench(pandas_pipe)
t_pl, r_pl = bench(polars_pipe)
print(f"pandas: {t_pd*1000:.1f} ms | polars: {t_pl*1000:.1f} ms | speedup x{t_pd/t_pl:.1f}")

check = (r_pd.sort_values("player_id").reset_index(drop=True)["total_ggr"].round(2)
         .equals(r_pl.sort("player_id").to_pandas()["total_ggr"].round(2)))
print("результаты совпадают:", check)
```

Что получилось: Polars стабильно быстрее (на этом объёме обычно в 2-6 раз, разрыв растёт с данными за счёт pushdown и многопоточности), и проверка `equals` подтверждает идентичность чисел. Это и есть твой артефакт — две версии одного пайплайна плюс доказательство корректности.

!!! tip "Посмотреть план оптимизации"

    Замени `.collect()` на `.explain()` в любом lazy-пайплайне — Polars напечатает физический план. Увидишь строки PROJECT и SELECTION: это и есть projection/predicate pushdown из модуля, проталкивающие выбор столбцов и фильтр к чтению Parquet.

## Критерий готовности

- [ ] Сгенерированы `registrations.parquet` и `bets.parquet` с фиксированным seed
- [ ] Профиль игрока построен через named aggregation; различаешь `size`/`count`/`nunique`
- [ ] Доля внутри страны сделана через `transform`, и объясняешь, почему не `agg`
- [ ] Воспроизвёл разъезд строк после merge и поймал его через `validate=`
- [ ] Построены дневной GGR, 7-дневное скользящее и рост день-к-дню
- [ ] Retention-матрица собрана `pivot_table` и переведена в проценты от W0
- [ ] Ключевой пайплайн переписан на Polars lazy (`scan_parquet` + `collect`)
- [ ] Замерено время обоих движков и доказано совпадение результатов через `equals`

## Бизнес-вывод

Технический артефакт (retention-матрица + профиль игрока) сам по себе Head of Retention ничего не говорит. Переведи его в решение: не «W1 = 19%», а «теряем 1.5 млн ₽ в месяц, вот что делаем». Собери короткий вывод на одну страницу по чек-листу:

- [ ] **Рекомендация (что делать).** Конкретное действие из данных: на какие когорты/каналы/страны включить welcome-цепочку удержания и где притормозить закуп. Например: «отвал между W0 и W1 максимален в когортах канала X — туда welcome-бонус; в стране Y держится на 5-10 китах — отдельный VIP-трек».
- [ ] **Эффект в деньгах или метриках.** Оцени вилкой: подъём W1 retention с 19% до 24% возвращает ~N игроков и ~M ₽ GGR в месяц; перераспределение Z% закупочного бюджета из каналов-отвала. Не абсолютная точность, а порядок величины и направление.
- [ ] **Риски и допущения.** Считали по ставкам, а не депозитам; когорты за один месяц (сезонность не учтена); часть «отвала» может быть естественной для гемблинга. Назови это прямо, чтобы решение не приняли как факт.
- [ ] **Следующий шаг.** Что проверить дальше: A/B welcome-цепочки на одной когорте, дотянуть депозитные данные, пересчитать retention после изменения закупа (пайплайн на Polars это позволяет делать быстро).
- [ ] **Как подать стейкхолдеру.** Язык решений, не метрик: один слайд — «теряем X ₽, причина в каналах A/B, предлагаю Y, цена вопроса Z, риск W». Матрицу и графики — в приложение, не в заголовок.

## Развитие

- Перенеси на Polars не только профиль, но и Шаг 5 (retention): `join` + `dt.truncate("1w")` + `group_by` + `pivot`. Сравни читаемость с pandas-версией.
- Добавь третий движок: тот же когортный запрос на DuckDB прямо по Parquet (`duckdb.sql(...).pl()`), без загрузки в память, и сравни время с pandas и Polars.
- Раздуй синтетику до 5-10 млн ставок (увеличь `N_PLAYERS` и `scale`) и пересними бенчмарк — увидишь, как разрыв pandas/Polars растёт с объёмом.
- Заверни pandas-пайплайн в единый method-chaining без промежуточных переменных и проверь через `.pipe(lambda d: (print(d.shape), d)[1])`, что формы на каждом шаге те, что ждёшь.
