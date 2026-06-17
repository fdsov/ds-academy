# Воркшоп M7 · Пайплайн очистки грязного лога ставок

<span class="lecture-meta">Воркшоп к модулю M7 · ориентир 4-6 ч</span>

## Что отрабатываем

В модуле M7 был тезис: грязные данные почти никогда не падают с исключением, они дают правдоподобный, но неверный ответ. Этот воркшоп переводит тезис в навык. Вы возьмёте сырой лог ставок и депозитов гемблинг-продукта, в котором сидят все классы загрязнений из теории (пропуски, полные дубли, выбросы, кривые типы, битая кодировка, события из будущего, рассогласование `country`/`currency`), и построите из него профессиональный артефакт.

Что именно отрабатываем из M7:

- M7.6 — классификация загрязнений и диагностика профилированием.
- M7.7 — выбор стратегии для пропусков через механизм MCAR/MAR/MNAR, а не наугад. Принцип «никогда не `fillna(0)` для денег».
- M7.8 — поиск выбросов методом Тьюки (IQR) и винзоризация вместо удаления.
- M7.9 — контракт на данные через Pandera (polars-бэкенд), падение пайплайна на нарушении.
- M7.10 — шесть измерений качества как числа с SLA.
- M7.11 — идемпотентный детерминированный пайплайн bronze -> silver, сырьё неизменно.

Артефакт на выходе: чистая функция `clean_bets()` плюс контракт `BetsSchema` плюс отчёт о качестве данных до/после по шести измерениям. Всё запускается одной командой у любого, кто склонировал репозиторий.

## Данные

Никаких внешних загрузок. Генерируем синтетический лог с фиксированным seed, чтобы у всех байт-в-байт совпадало, и тут же портим его всеми способами из M7.6. Это и есть ваша bronze-зона.

Окружение через uv (стек 2026):

```bash
uv init m7-workshop && cd m7-workshop
uv add polars pandera "pandas>=2" pyarrow
```

Генератор грязного лога. Запустите его один раз — он создаст `bronze/bets_raw.csv` (сырьё как пришло от партнёра, в CSV с битой кодировкой по части строк):

```python
# gen_data.py
import polars as pl
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
N = 10_000
Path("bronze").mkdir(exist_ok=True)

user_ids = [f"u{rng.integers(1, 1500):05d}" for _ in range(N)]
amounts = np.round(rng.lognormal(mean=4.0, sigma=1.0, size=N), 2)  # логнормаль ~ как реальные суммы
currency = rng.choice(["USD", "usd", "EUR", "eur", "RUB", "rub"], size=N)
country = rng.choice(["US", "DE", "RU"], size=N)
etype = rng.choice(["bet", "deposit", "withdrawal"], size=N, p=[0.6, 0.25, 0.15])
ts = pl.datetime_range(
    pl.datetime(2026, 1, 1), pl.datetime(2026, 6, 1),
    interval="26m", eager=True
)[:N]

df = pl.DataFrame({
    "event_id": [f"e{i:07d}" for i in range(N)],
    "user_id": user_ids,
    "amount": amounts,
    "currency": currency,
    "country": country,
    "event_type": etype,
    "ts": ts,
})

# --- порча по M7.6 ---
# 1. суммы как строки с запятой-разделителем и пробелами (формат партнёра)
df = df.with_columns(
    pl.col("amount").map_elements(lambda x: f"{x:,.2f}".replace(",", " ").replace(".", ","),
                                  return_dtype=pl.Utf8)
)
# 2. 5% пропусков в amount — причём чаще у deposit (потом проверим механизм)
miss_mask = (rng.random(N) < np.where(df["event_type"] == "deposit", 0.15, 0.02))
df = df.with_columns(
    pl.when(pl.Series(miss_mask)).then(None).otherwise(pl.col("amount")).alias("amount")
)
# 3. десяток сумм-артефактов на миллион (тестовые транзакции)
art_idx = rng.choice(N, size=12, replace=False)
amt = df["amount"].to_list()
for i in art_idx:
    amt[i] = "10 000 000,00"
df = df.with_columns(pl.Series("amount", amt))
# 4. события из будущего — битые часы клиента
fut_idx = rng.choice(N, size=20, replace=False)
ts2 = df["ts"].to_list()
for i in fut_idx:
    ts2[i] = pl.datetime(2099, 1, 1)
df = df.with_columns(pl.Series("ts", ts2).cast(pl.Datetime))
# ts в строку формата партнёра DD.MM.YYYY HH:MM
df = df.with_columns(pl.col("ts").dt.strftime("%d.%m.%Y %H:%M"))

# 5. 2% полных дублей от ретраев загрузки
dup = df.sample(fraction=0.02, seed=7)
df = pl.concat([df, dup])

df.write_csv("bronze/bets_raw.csv")
print("bronze rows:", df.height)
```

Колонки сырья: `event_id`, `user_id`, `amount` (строка `"1 234,50"`, есть `null`), `currency` (разный регистр), `country`, `event_type`, `ts` (строка `DD.MM.YYYY HH:MM`, есть из будущего). Дубли и артефакты внутри.

## Ход работы

### Шаг 1: профилирование сырья (M7.6, M7.10)

Зачем. Прежде чем чистить, надо знать, с чем имеешь дело — это дисциплина недоверия из M7.1. Считаем шесть измерений качества как числа, чтобы потом сравнить до/после.

```python
# pipeline.py
import polars as pl

def quality_report(df: pl.DataFrame, amount_col: str = "amount_num") -> dict:
    n = df.height
    has_num = amount_col in df.columns
    return {
        "rows": n,
        # полнота: доля заполненных amount
        "completeness_amount": round(1 - df["amount"].null_count() / n, 4),
        # уникальность: нет дублей event_id
        "uniqueness_event_id": round(df["event_id"].n_unique() / n, 4),
        # валидность валюты: из справочника после нормализации
        "validity_currency": round(
            df.filter(pl.col("currency").str.to_uppercase().is_in(["USD", "EUR", "RUB"])).height / n, 4
        ),
        # согласованность: ts не из будущего (битые часы)
        "future_ts": df.filter(
            pl.col("ts").str.strptime(pl.Datetime, "%d.%m.%Y %H:%M", strict=False)
            > pl.datetime(2026, 12, 31)
        ).height if df["ts"].dtype == pl.Utf8 else
        df.filter(pl.col("ts") > pl.datetime(2026, 12, 31)).height,
        # выбросы-артефакты на сумме (если уже распарсили)
        "outliers_amount": (
            df.filter(pl.col(amount_col) > 1_000_000).height if has_num else None
        ),
    }

raw = pl.read_csv("bronze/bets_raw.csv")
print("BEFORE:", quality_report(raw))
```

Что получилось. `completeness_amount` около 0.95, `uniqueness_event_id` около 0.98 (2% дублей), `validity_currency` = 1.0 только после `to_uppercase` (это и есть сигнал, что регистр кривой), `future_ts` около 20. Это ваша точка отсчёта.

!!! question "Проверь себя"

    1. Почему `uniqueness_event_id` меньше 1.0 ещё до всякой чистки и какой класс загрязнения за этим стоит?
    2. Поле `currency` заполнено в 100% строк. Почему это не делает его валидным?

??? success "Ответы"

    1. В сырьё добавлены 2% полных дублей от ретраев загрузки (M7.6). `n_unique / n < 1` ловит именно повтор `event_id`. Полнота тут ни при чём — строки есть, они просто дублируются.
    2. Полнота и валидность — разные измерения (M7.10). `"usd"` заполнено, но не входит в справочник ISO в верхнем регистре, который ждёт схема. Значение есть, но формат нарушен.

### Шаг 2: диагностика механизма пропуска (M7.7)

Зачем. Главная ошибка новичка — `fillna(0)` или `dropna()` без вопроса «почему пропущено». Механизм MCAR/MAR/MNAR определяет, смещены ли выводы. Проверяем гипотезу: пропуск `amount` зависит от `event_type`?

```python
miss_by_type = (
    raw.with_columns(pl.col("amount").is_null().alias("is_miss"))
       .group_by("event_type")
       .agg(pl.col("is_miss").mean().alias("miss_rate"))
       .sort("miss_rate", descending=True)
)
print(miss_by_type)
```

Что получилось. У `deposit` доля пропусков около 0.15, у `bet`/`withdrawal` около 0.02. Пропуск зависит от наблюдаемой переменной `event_type`, но (по построению) не от самой скрытой суммы внутри группы — это MAR, не MNAR. Вывод для стратегии: listwise deletion сместил бы выборку (выкинул бы непропорционально много депозитов), поэтому мы НЕ удаляем строки. Берём решение из M7.7: флаг пропуска `amount_was_missing` плюс сентинел, а не `fillna(0)`. Факт отсутствия сохраняем как сигнал, ложное число не вносим.

!!! tip "Почему не MNAR здесь"

    MNAR был бы, если бы пропущены оказывались именно крупные суммы (хайроллеры прячут депозит). Тогда ни удаление, ни медиана не спасают. Здесь пропуск объясняется наблюдаемым `event_type`, значит MAR — и флаг плюс честная импутация по группе корректны. Различить MAR и MNAR кодом нельзя, только пониманием механизма генерации.

### Шаг 3: контракт Pandera на выходе (M7.9)

Зачем. Профессиональный сдвиг из M7.9 — не чистить руками каждый раз, а декларативно описать, какими данные должны быть. Контракт собирает все ожидания в одном месте; нарушение ловится на входе, а не всплывает в отчёте через неделю.

```python
import pandera.polars as pa
from pandera.typing.polars import Series

class BetsSchema(pa.DataFrameModel):
    event_id: Series[str] = pa.Field(unique=True)
    user_id: Series[str] = pa.Field(nullable=False)
    amount: Series[float] = pa.Field(ge=0, le=1_000_000, nullable=True)
    amount_was_missing: Series[bool] = pa.Field()
    currency: Series[str] = pa.Field(isin=["USD", "EUR", "RUB"])
    country: Series[str] = pa.Field(isin=["US", "DE", "RU"])
    event_type: Series[str] = pa.Field(isin=["bet", "deposit", "withdrawal"])
    ts: Series[pl.Datetime] = pa.Field(nullable=False)

    class Config:
        strict = True  # лишние колонки запрещены
```

Что получилось. Формальное описание silver-таблицы: `amount` неотрицательна и не больше миллиона (артефакты на 10 млн обязаны быть погашены до валидации), валюта и страна из справочников, `event_id` уникален. Если `clean_bets` не дочистит — `validate` упадёт громко.

### Шаг 4: чистая функция clean_bets (M7.6, M7.8, M7.11)

Зачем. Собираем детерминированный идемпотентный пайплайн bronze -> silver. Каждый шаг — воспроизводимое правило, сырьё не мутируем, на выходе валидация. Винзоризация хвоста вместо удаления (M7.8): артефакт гасим, строку храним.

```python
def clean_bets(raw: pl.DataFrame) -> pl.DataFrame:
    before = raw.height
    df = (
        raw
        # 1. типы: "1 234,50" -> 1234.50, нормализация валюты, парс даты
        .with_columns(
            pl.col("amount").str.replace_all(" ", "").str.replace(",", ".")
              .cast(pl.Float64, strict=False).alias("amount"),
            pl.col("currency").str.to_uppercase(),
            pl.col("ts").str.strptime(pl.Datetime, "%d.%m.%Y %H:%M", strict=False),
        )
        # 2. полные дубли от ретраев -> по event_id, keep first
        .unique(subset=["event_id"], keep="first")
        # 3. согласованность: события из будущего = битые часы клиента
        .filter(pl.col("ts") <= pl.datetime(2026, 12, 31))
        # 4. флаг пропуска вместо молчаливого fillna(0) (MAR, M7.7)
        .with_columns(pl.col("amount").is_null().alias("amount_was_missing"))
        # 5. винзоризация по верхней границе Тьюки, а не удаление (M7.8)
    )
    q1 = df["amount"].quantile(0.25)
    q3 = df["amount"].quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)
    df = df.with_columns(
        pl.when(pl.col("amount").is_not_null())
          .then(pl.col("amount").clip(upper_bound=upper))
          .otherwise(None).alias("amount")
    )
    print(f"[clean] {before} -> {df.height} строк "
          f"(дублей убрано: {before - df.height - 0}, верхний ус Тьюки: {upper:.2f})")

    BetsSchema.validate(df, lazy=True)  # контракт на выходе
    return df
```

Что получилось. Функция логирует число затронутых строк (M7.11), приводит типы, бьёт дубли по `event_id`, выкидывает только доказанный артефакт (будущие `ts`), помечает пропуски флагом и винзоризует хвост по границе Тьюки $Q_3 + 1.5 \cdot IQR$. Артефакты на 10 млн прижаты к верхнему усу, поэтому проходят `le=1_000_000` в контракте. Сырьё `raw` не изменено — `clean_bets` возвращает новый DataFrame.

!!! question "Проверь себя"

    1. Почему будущие `ts` мы `filter`-уем (удаляем строки), а суммы-артефакты на 10 млн — `clip`-аем (винзоризуем)?
    2. Что в этой функции делает её идемпотентной — почему повторный прогон на том же сырье даст тот же результат?

??? success "Ответы"

    1. Событие из будущего — доказанный артефакт (битые часы, дата физически невозможна), его честно удалить. Крупная сумма может быть и реальным хайроллером, и тестом; молча удалять её нельзя (M7.8), поэтому гасим влияние винзоризацией, но строку сохраняем. Решение зависит от того, доказан ли артефакт.
    2. Все шаги детерминированы (фиксированные правила, `keep="first"`, квантили считаются от данных), функция не зависит от внешнего состояния и не мутирует вход. Прогон `clean_bets(clean_bets_input)` дважды на одинаковом `raw` даёт байт-в-байт одинаковый выход.

### Шаг 5: отчёт до/после и запуск пайплайна (M7.10, M7.11)

Зачем. Финальный артефакт — таблица качества до/после, переводящая «данные плохие» в конкретные числа с SLA. Плюс сохранение в Parquet, а не в CSV (M7.4).

```python
import os

if __name__ == "__main__":
    raw = pl.read_csv("bronze/bets_raw.csv")
    before = quality_report(raw)              # на сырье amount ещё строка
    clean = clean_bets(raw)
    after = quality_report(clean, amount_col="amount")

    os.makedirs("silver", exist_ok=True)
    clean.write_parquet("silver/bets_clean.parquet")

    csv_mb = os.path.getsize("bronze/bets_raw.csv") / 1e6
    pq_mb = os.path.getsize("silver/bets_clean.parquet") / 1e6

    print("\n| метрика | before | after |")
    print("|---|---|---|")
    for k in before:
        print(f"| {k} | {before[k]} | {after.get(k)} |")
    print(f"\nCSV {csv_mb:.2f} MB -> Parquet {pq_mb:.2f} MB "
          f"(сжатие x{csv_mb / pq_mb:.1f})")
```

Что получилось. `completeness_amount` остаётся прежней (пропуски не удаляли — это правильно, мы их пометили флагом), но `uniqueness_event_id` поднимается до 1.0, `future_ts` падает до 0, `outliers_amount` (суммы > 1 млн) становится 0 после винзоризации, `validity_currency` = 1.0. Parquet в несколько раз меньше CSV (коэффициент $C$ из M7.4). Запуск всего пайплайна — одна команда:

```bash
uv run gen_data.py && uv run pipeline.py
```

!!! question "Проверь себя"

    1. После чистки `completeness_amount` НЕ выросла. Это баг пайплайна или правильное поведение?
    2. Почему отчёт о качестве считается и до, и после — что даёт именно сравнение, а не одна финальная цифра?

??? success "Ответы"

    1. Правильное поведение. Мы диагностировали пропуск как MAR и сознательно не заполняли `amount` ложным значением, а пометили флагом `amount_was_missing` (M7.7). Поднять полноту до 1.0 здесь означало бы внести ложь. Полнота честно остаётся прежней, а факт отсутствия сохранён как сигнал.
    2. Сравнение до/после показывает, что именно пайплайн починил и не сломал ли он что-то (например, не подтянул ли полноту искусственно). Одна финальная цифра не отличает «данные были чистыми» от «мы их вычистили» и скрывает регрессии.

## Критерий готовности

- [ ] `gen_data.py` генерирует `bronze/bets_raw.csv` с фиксированным seed (у всех одинаково)
- [ ] Механизм пропуска `amount` диагностирован через разрез по `event_type`, вывод (MAR) обоснован в комментарии
- [ ] `clean_bets()` — чистая функция: не мутирует вход, детерминирована, идемпотентна, логирует число строк на шагах
- [ ] Пропуски помечены флагом `amount_was_missing`, нигде нет `fillna(0)` для `amount`
- [ ] Выбросы-артефакты винзоризованы по границе Тьюки, события из будущего удалены, дубли сбиты по `event_id`
- [ ] Контракт `BetsSchema` (Pandera, polars-бэкенд) валидирует выход; при поломке пайплайн падает
- [ ] Отчёт о качестве по шести измерениям посчитан до и после, оформлен таблицей
- [ ] Silver-таблица сохранена в Parquet, сжатие относительно CSV измерено
- [ ] Весь пайплайн запускается одной командой и даёт тот же результат при повторном прогоне

## Развитие

- Замените простой флаг пропуска на честную импутацию по группе: для MAR корректно заполнить `amount` медианой внутри `event_type` (sklearn `IterativeImputer` или групповая медиана в Polars), сохранив флаг. Сравните распределение `amount` до и после — заметьте, как импутация занижает дисперсию (M7.7).
- Добавьте контракт на входе через Pydantic: распарсите каждую сырую запись как модель на границе и отсейте структурно битые строки до DataFrame-чистки (M7.9). Сравните, что ловит Pydantic построчно, а что Pandera таблично.
- Реализуйте проверку согласованности `country`/`currency` как межколоночное правило Pandera (`@pa.dataframe_check`): запретите `country=US` при `currency=RUB` и измерьте долю нарушений как седьмую кастомную метрику качества.
- Доведите до medallion: вынесите `gen_data` в bronze-зону неизменным сырьём, `clean_bets` в silver, и добавьте gold-шаг — агрегат `deposit`-сумм по дням, который читает silver-Parquet через DuckDB прямо по файлу без загрузки в память (M7.11, M7.12).
