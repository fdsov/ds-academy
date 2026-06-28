# W5 · Прогноз LTV и выручки

<span class="lecture-meta">Воркшоп · ориентир 8-12 ч · Продвинутый</span>

## Что ты построишь

К концу воркшопа у тебя на руках будет два рабочих артефакта и понимание, почему их нельзя путать.

Первый артефакт — **модель раннего LTV**: регрессор, который по поведению игрока за первые $N$ дней (депозиты, сессии, ставки, дни активности) предсказывает его пожизненную ценность. Это то, что нужно маркетингу: оценить качество трафика по источнику уже на 7-й день, а не ждать полгода.

Второй артефакт — **прогноз выручки по когортам на горизонт с интервалами неопределённости**. Это то, что нужно финансам: сказать совету директоров "в следующем квартале эти когорты принесут X ± Y миллионов". Здесь работают не индивидуальные признаки, а кривая дожития депозитов (retention/decay) и агрегатная динамика.

Ключевая мысль, ради которой весь воркшоп: оба прогноза упираются в **цензурирование** — молодые когорты ещё не дожили до конца наблюдения, и наивно усреднять их с дозревшими нельзя. Ты научишься это видеть, моделировать и честно сообщать неопределённость.

!!! info "Как устроен этот воркшоп"

    Это не лекция с готовым кодом, а задачник. Каждый содержательный шаг построен по схеме **Задача → Критерий → Решение**:

    - **Задача** — что сделать руками, с явными именами выходных переменных (их проверяет критерий — если назовёшь иначе, получишь `NameError`).
    - **Критерий шага** — блок с `assert`, который запускаешь после своего решения. Зелёный прогон = шаг сдан. Это твой локальный авто-грейдер. Там, где результат зависит от внешней динамики данных и устойчивого числа нет, вместо `assert` идёт **self-check** чек-лист.
    - **Решение** спрятано под спойлер `Решение` — открывай после своей попытки, чтобы сверить подход, а не списать.
    - **Числовые подзадачи** проверяются прямо на странице: посчитай число, впиши в поле, нажми «Проверить».

    Вся синтетика детерминирована при `seed=42`. Не меняй seed — числа в критериях рассчитаны на него.

!!! info "Контекст домена"

    LTV (Lifetime Value, иногда CLV/CLTV) в гемблинге считается в терминах нетто-выручки оператора — GGR (Gross Gaming Revenue) минус бонусы, или NGR. Депозит игрока сам по себе не выручка: часть вернётся выплатами. В этом воркшопе для простоты целевая переменная — суммарный net revenue от игрока за горизонт жизни. В проде уточни определение с финансами до первой строки кода.

## Бизнес-кейс

!!! example "Ситуация"

    Ты — аналитик в команде iGaming-продукта. К тебе приходит **руководитель закупки трафика (CMO)** перед защитой квартального бюджета. Команда планирует залить в следующий квартал около 30 млн ₽ на привлечение, но текущая разбивка бюджета между источниками построена на «выручке на сегодня» — а свежие когорты по этой метрике выглядят убыточными просто потому, что ещё не дожили. Есть риск зарезать хороший канал и перелить в плохой.

    - **Проблема в цифрах (иллюстративно):** по сырой наблюдаемой выручке source `social` и `ppc` выглядят слабее `seo` и `affiliate`, и финансы давят «отключить social». Но молодым когортам social всего 20-40 дней жизни против 300+ у дозревших — сравнение нечестное. Если ошибиться в ранжировании каналов на ~15% бюджета, это ориентировочно 4-5 млн ₽ в квартал, залитых не туда.
    - **Что зависит от ответа:** решение, какие источники масштабировать, а какие резать, и какое число выручки CMO понесёт на совет директоров как прогноз на квартал. Точечное число без коридора создаёт ложную уверенность и подставит команду, если факт уедет.
    - **Ограничение:** решение о качестве трафика нужно принимать на **7-й день** жизни когорты, а не через полгода ожидания; доступны только данные депозитов и поведения, дозревшие когорты старше года ограничены.

## Предпосылки

Нужны два модуля. Из **M14 (supervised learning)** — регрессия, разбиение train/test, метрики ошибки, работа с градиентным бустингом. Из **M28 (time series)** — понятие тренда и сезонности, экспоненциальное сглаживание, оценка прогноза на горизонте и почему ошибка может расти с дальностью. Если эти модули ещё не пройдены — пройди до старта, здесь они предполагаются как фундамент.

Окружение собираем через `uv` — это актуальный на 2026 стандарт: быстрее pip, lock-файл из коробки, не надо вручную возиться с venv.

```bash
uv init w5-ltv-forecast
cd w5-ltv-forecast
uv add pandas numpy scikit-learn lightgbm matplotlib statsmodels pyarrow
uv add --dev jupyterlab
```

Проверка, что всё встало:

```bash
uv run python -c "import pandas, lightgbm, sklearn, statsmodels; print('ok')"
```

Весь код воркшопа запускается как `uv run python step_XX.py` или в `uv run jupyter lab`. Зависимости от прода нет — данные генерим сами.

## Данные

Нам нужен реалистичный когортный датасет: игроки приходят в разные дни, делают депозиты во времени, и — главное — частота депозитов затухает (decay), а сумма депозита разнородна между игроками. Плюс мы сознательно встроим **цензурирование**: дата "сегодня" фиксирована, и поздние когорты физически не успели прожить столько же, сколько ранние.

Это setup-код, а не задача — скопируй и запусти как есть. Дальше ты работаешь с файлами `players.parquet` и `deposits.parquet`.

!!! note "Зачем синтетика, а не сразу прод"

    Синтетика даёт то, чего нет в реальных данных — известную "истинную" LTV. Мы знаем, сколько игрок принёс бы за полный горизонт, потому что сами это сгенерировали. Это позволяет честно измерить ошибку модели на цензурированных когортах. На реальных данных истину ты не знаешь никогда, поэтому учиться надо там, где она есть.

Модель генерации игрока:

- Дата регистрации (`signup_date`) — равномерно за последние **2 года** до точки наблюдения. Окно шире горизонта жизни (365 дней) намеренно: иначе ни одна когорта не успеет «дозреть», и обучать модель будет не на ком.
- Латентное "качество" игрока $q \sim \text{LogNormal}$ — задаёт и частоту, и размер депозитов. Так мы получаем тяжёлый правый хвост: немного китов дают основную выручку.
- Депозиты во времени — пуассоновский процесс с убывающей интенсивностью: $\lambda(t) = \lambda_0 \cdot q \cdot e^{-t/\tau}$, где $t$ — дни с регистрации, $\tau$ — характерное время жизни. Это и есть кривая дожития.
- Сумма каждого депозита — LogNormal, масштаб зависит от $q$.
- Net revenue с депозита — депозит, умноженный на hold (маржу оператора) с шумом.

```python
# data_gen.py
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
HORIZON = 365                          # горизонт "полной" жизни игрока в днях
OBS_DATE = pd.Timestamp("2026-01-01")  # точка наблюдения "сегодня"
N_PLAYERS = 8000

def generate_players(n=N_PLAYERS):
    # signup за последние 2 года: окно ШИРЕ горизонта, чтобы были дозревшие когорты
    signup_offset = RNG.integers(1, 730, size=n)
    signup_date = OBS_DATE - pd.to_timedelta(signup_offset, unit="D")
    quality = RNG.lognormal(mean=0.0, sigma=0.9, size=n)  # латентное качество, тяжёлый хвост
    source = RNG.choice(["seo", "ppc", "affiliate", "social"], size=n,
                        p=[0.30, 0.30, 0.25, 0.15])
    src_mult = pd.Series({"seo": 1.25, "ppc": 0.85, "affiliate": 1.05, "social": 0.7})
    return pd.DataFrame({
        "player_id": np.arange(n),
        "signup_date": signup_date,
        "quality": quality * pd.Series(source).map(src_mult).values,
        "source": source,
    })

def generate_deposits(players, tau=70.0, lam0=0.04, hold=0.06):
    rows = []
    for p in players.itertuples():
        # ожидаемое число депозитов за полный горизонт при decay-интенсивности
        expected = lam0 * p.quality * tau * (1 - np.exp(-HORIZON / tau))
        n_dep = RNG.poisson(expected)
        if n_dep == 0:
            continue
        # времена депозитов из экспоненциального decay (через инверсию CDF)
        u = RNG.uniform(0, 1, size=n_dep)
        t = -tau * np.log(1 - u * (1 - np.exp(-HORIZON / tau)))
        t = np.sort(t)
        amount = RNG.lognormal(mean=2.8, sigma=0.7, size=n_dep) * (0.5 + p.quality)
        net = amount * hold * RNG.normal(1.0, 0.15, size=n_dep).clip(0.2, None)
        for ti, ai, ni in zip(t, amount, net):
            rows.append((p.player_id, p.signup_date + pd.Timedelta(days=float(ti)),
                         float(ai), float(ni), int(ti)))
    dep = pd.DataFrame(rows, columns=["player_id", "deposit_date", "amount",
                                      "net_revenue", "day_since_signup"])
    return dep

if __name__ == "__main__":
    players = generate_players()
    deposits = generate_deposits(players)
    # цензурирование: видим только то, что произошло до OBS_DATE
    deposits["observed"] = deposits["deposit_date"] <= OBS_DATE
    players.to_parquet("players.parquet")
    deposits.to_parquet("deposits.parquet")
    print(f"Игроков: {len(players)}, депозитов всего: {len(deposits)}, "
          f"из них наблюдаемых: {deposits['observed'].sum()}")
```

Запуск:

```bash
uv run python data_gen.py
```

**Критерий шага** — запусти после генерации:

```python
import pandas as pd
players = pd.read_parquet("players.parquet")
deposits = pd.read_parquet("deposits.parquet")
assert len(players) == 8000, "ожидаем 8000 игроков"
assert 33000 <= len(deposits) <= 34500, "число депозитов около 33-34 тысяч"
assert 30000 <= deposits["observed"].sum() <= 31500, "наблюдаемых депозитов около 30-31 тысячи"
assert set(players["source"].unique()) == {"seo", "ppc", "affiliate", "social"}
print("OK: данные сгенерированы, есть наблюдаемые и цензурированные депозиты")
```

!!! tip "Замена на реальные данные"

    Вместо синтетики можно взять публичный датасет транзакций. На Kaggle — "Online Retail II" (UCI) с колонками InvoiceDate / CustomerID / Quantity*Price: классика для CLV, по которой считают даже в учебниках по lifetimes. На Hugging Face ищи датасеты с тегом transactions / e-commerce. Маппинг: CustomerID → player_id, InvoiceDate → deposit_date, сумма строки → net_revenue. Гемблинг-специфики (hold, бонусы) там нет, но механика цензурирования и раннего LTV переносится один в один.

## Разминка: цензурирование на пальцах

Прежде чем писать код, прогрей основную интуицию воркшопа арифметикой. Эти числа понадобятся на шагах 5 и 8. Считай в уме или калькулятором, ответ впиши в поле.

```text
TASK: Когорта прожила половину горизонта и накопила к этому возрасту c(a)=0.50 своего полного LTV. Наблюдаемая выручка когорты сейчас = 3000 у.е. Оцени ПОЛНУЮ выручку когорты по формуле R_full = R_obs / c(a). Ответ - целое число у.е.
ANSWER: 6000
TOL: 50
UNIT: у.е.
PLACEHOLDER: целое число
EXPLAIN: R_full = 3000 / 0.50 = 6000. Это и есть подтягивающий (maturation) множитель: видимую выручку молодой когорты делят на долю прожитой жизни. Чем моложе когорта, тем меньше c(a) и тем сильнее домножение - и тем выше неопределённость оценки.
---
TASK: Доля полного LTV, накопленная к 7-му дню жизни игрока, примерно c(7)=0.11. Во сколько РАЗ надо домножить наблюдаемую раннюю выручку молодой когорты, чтобы грубо оценить её полную ценность? Это множитель 1/c(7). Округли до 0.1.
ANSWER: 9.1
TOL: 0.5
PLACEHOLDER: 0.0
EXPLAIN: 1 / 0.11 = 9.09 = 9.1. За первую неделю игрок приносит лишь около десятой части пожизненной ценности - оставшиеся ~90% впереди. Поэтому судить о качестве трафика по сырой выручке 7-го дня нельзя: надо либо домножать на 1/c(7), либо учить ML-модель, которая делает это умнее (Шаги 3-4).
```

## Ход работы

### Шаг 1: Увидеть цензурирование глазами

**Зачем.** Прежде чем что-то моделировать, надо убедиться, что проблема реальна. Самый частый провал джуна — посчитать "средний LTV по всем игрокам" и удивиться, почему число занижено. Оно занижено, потому что в среднем сидят молодые когорты, у которых жизнь ещё не прожита. Покажем это на данных.

**Задача.** Загрузи `players.parquet` и `deposits.parquet`. Посчитай возраст когорты `age_days = OBS_DATE - signup_date`, добавь месячную когорту `cohort_month`. По наблюдаемым депозитам посчитай выручку на игрока и собери сводную таблицу по месячным когортам `by_cohort` с колонками: число игроков, средний возраст `avg_age`, средняя наблюдаемая выручка `avg_rev_observed`. Положи результат в `by_cohort`.

??? tip "Подсказка"

    Наблюдаемые депозиты — это `deposits[deposits["observed"]]`. Сгруппируй их по `player_id`, просуммируй `net_revenue`, приджойни к `players` (заполни пропуски нулём — у кого не было депозита). Дальше `players.groupby("cohort_month").agg(...)`.

**Критерий шага:**

```python
assert {"players", "avg_age", "avg_rev_observed"} <= set(by_cohort.columns)
ages = by_cohort["avg_age"]
youngest = by_cohort.loc[ages.idxmin()]
oldest = by_cohort.loc[ages.idxmax()]
# у самой молодой когорты наблюдаемая выручка заметно ниже, чем у дозревшей
assert youngest["avg_rev_observed"] < oldest["avg_rev_observed"], \
    "молодая когорта обязана выглядеть беднее дозревшей — это и есть цензурирование"
assert youngest["avg_age"] < 40 and oldest["avg_age"] > 600
print("OK: цензурирование видно — молодые когорты занижены по возрасту, а не по качеству")
```

??? success "Решение"

    ```python
    # step1_censoring.py
    import pandas as pd

    players = pd.read_parquet("players.parquet")
    deposits = pd.read_parquet("deposits.parquet")
    OBS_DATE = pd.Timestamp("2026-01-01")

    players["age_days"] = (OBS_DATE - players["signup_date"]).dt.days
    players["cohort_month"] = players["signup_date"].dt.to_period("M")

    obs = deposits[deposits["observed"]]
    rev_observed = obs.groupby("player_id")["net_revenue"].sum()
    players = players.merge(rev_observed.rename("rev_observed"),
                            left_on="player_id", right_index=True, how="left").fillna({"rev_observed": 0})

    by_cohort = players.groupby("cohort_month").agg(
        players=("player_id", "count"),
        avg_age=("age_days", "mean"),
        avg_rev_observed=("rev_observed", "mean"),
    )
    print(by_cohort.round(1))
    ```

    **Почему так.** У старых когорт `avg_age` около 600-700 дней и наблюдаемая выручка высокая (~20-30 у.е.), а у свежих — возраст 16-46 дней и выручка близка к нулю (~4 у.е.). Если усреднить `avg_rev_observed` по всем — получишь смесь дозревших и недозревших. Это не LTV, это "выручка на сегодня". Запомни это наивное среднее — мы вернёмся к нему на Шаге 8 и покажем, насколько оно врёт.

!!! warning "Главная ловушка домена"

    Наблюдаемая выручка молодой когорты не маленькая потому, что игроки плохие. Она маленькая потому, что у них впереди ещё сотни дней жизни, которых мы не видим. Сравнивать качество источников трафика по сырой наблюдаемой выручке — значит штрафовать недавно закупленный трафик за то, что он недавний.

### Шаг 2: Собрать целевую переменную (true LTV) и фичи раннего поведения

**Зачем.** Для обучения модели раннего LTV нам нужны две вещи: признаки из первых $N$ дней (то, что в проде известно рано) и целевая переменная — полная ценность за горизонт. На синтетике у нас есть роскошь: мы можем взять **все** депозиты игрока за `HORIZON` дней (включая ненаблюдаемые) как истинную цель. Но обучать модель будем только на тех игроках, кто уже прожил полный горизонт, — иначе их "истина" сама цензурирована. Это ключевое решение: train только на дозревших, иначе модель учится на занижённых таргетах.

**Задача.** Собери датасет `df` (индекс — `player_id`) с:

- 7-дневными фичами раннего поведения: `early_dep_count`, `early_dep_sum`, `early_dep_mean`, `early_dep_max`, `early_net_sum`, `early_active_days`, `first_dep_day` (по депозитам с `day_since_signup < 7`);
- целевой переменной `ltv_true` — сумма `net_revenue` по всем депозитам игрока с `day_since_signup <= HORIZON`;
- булевым флагом `mature = age_days >= HORIZON` (прожил ли игрок полный горизонт).

Пропуски по фичам и `ltv_true` заполни нулём; `first_dep_day` у игроков без депозита в окне поставь равным `N_DAYS`. Сохрани `df` в `features.parquet`.

??? tip "Подсказка"

    Фичи — `deposits[deposits["day_since_signup"] < 7].groupby("player_id").agg(...)`. Таргет — отдельный groupby по `net_revenue` с фильтром `<= HORIZON`. Соедини через `players.set_index("player_id").join(feat).join(target)`. Не забудь `fillna(0)` и замену `first_dep_day` нулей.

**Критерий шага:**

```python
assert "mature" in df.columns and "ltv_true" in df.columns
assert len(df) == 8000
assert 3800 <= df["mature"].sum() <= 4100, "дозревших примерно половина (~3940)"
mat = df.loc[df["mature"], "ltv_true"]
# тяжёлый правый хвост: среднее сильно выше медианы (киты)
assert mat.mean() > mat.median() * 3, "ltv_true обязан иметь тяжёлый правый хвост"
print(f"OK: dataset собран, mature={df['mature'].sum()}, хвост подтверждён")
```

??? success "Решение"

    ```python
    # step2_features.py
    import numpy as np
    import pandas as pd

    N_DAYS = 7
    HORIZON = 365
    OBS_DATE = pd.Timestamp("2026-01-01")

    players = pd.read_parquet("players.parquet")
    deposits = pd.read_parquet("deposits.parquet")
    players["age_days"] = (OBS_DATE - players["signup_date"]).dt.days

    early = deposits[deposits["day_since_signup"] < N_DAYS]
    feat = early.groupby("player_id").agg(
        early_dep_count=("amount", "count"),
        early_dep_sum=("amount", "sum"),
        early_dep_mean=("amount", "mean"),
        early_dep_max=("amount", "max"),
        early_net_sum=("net_revenue", "sum"),
        early_active_days=("day_since_signup", "nunique"),
        first_dep_day=("day_since_signup", "min"),
    )

    full = deposits[deposits["day_since_signup"] <= HORIZON]
    target = full.groupby("player_id")["net_revenue"].sum().rename("ltv_true")

    df = players.set_index("player_id").join(feat, how="left").join(target, how="left")
    num_cols = ["early_dep_count", "early_dep_sum", "early_dep_mean", "early_dep_max",
                "early_net_sum", "early_active_days", "first_dep_day"]
    df[num_cols] = df[num_cols].fillna(0)
    df["ltv_true"] = df["ltv_true"].fillna(0)
    df["first_dep_day"] = df["first_dep_day"].replace(0, N_DAYS)  # не было депозита в окне

    df["mature"] = df["age_days"] >= HORIZON
    df.to_parquet("features.parquet")
    print(f"Всего: {len(df)}, дозревших (mature): {df['mature'].sum()}")
    print(df.loc[df["mature"], ["early_dep_sum", "early_net_sum", "ltv_true"]].describe().round(1))
    ```

    **Почему так.** Получаешь ~3940 дозревших игроков из 8000. У `ltv_true` медиана ~4.5, а среднее ~25 — тяжёлый правый хвост (киты тянут среднее вверх). Это определяет выбор метрики и log-трансформации таргета дальше. Флаг `mature` отделяет тех, на ком честно учить, от молодых, кого мы будем только предсказывать.

Проверь понимание:

```text
Q: Почему обучать модель раннего LTV на молодых игроках (которые ещё не дожили до горизонта) - ошибка?
[ ] Молодых игроков мало, выборка нерепрезентативна
[x] Их ltv_true обрезан цензурированием - модель выучит заниженную зависимость и будет недооценивать всех
[ ] Молодые игроки в среднем хуже по качеству
> У молодого игрока часть жизни ещё впереди и не попала в таргет. Обучение на таком таргете занижает прогноз систематически. Поэтому учим только на дозревших когортах (на синтетике это mature, в проде - когорты старше горизонта).
---
Q: Почему окно фич = 7 дней, а не 90?
[ ] 7 дней дают точнее прогноз, чем 90
[x] Короткое окно = раннее решение: маркетинг режет плохой трафик на первой неделе, а не через квартал
[ ] statsmodels требует ровно 7 дней
> Чем короче окно, тем раньше принимается решение и тем ценнее модель для закупки - но тем выше неопределённость прогноза. Это осознанный компромисс ранний-сигнал против точности.
---
Q: Что произойдёт с распределением ltv_true, если убрать латентное quality и сделать всех игроков одинаковыми?
[ ] Появится сезонность
[x] Хвост схлопнется, распределение станет почти симметричным, среднее сравняется с медианой
[ ] ltv_true станет отрицательным
> Тяжёлый хвост создаётся разбросом quality (киты). Без него задача станет лёгкой и нереалистичной - в гемблинге киты решают всё, тяжёлый хвост обязателен.
```

### Шаг 3: Базлайн и модель раннего LTV

**Зачем.** Любую ML-задачу начинаем с тупого базлайна, чтобы было с чем сравнивать. Базлайн здесь — "LTV = ранний net revenue, умноженный на константу" (наивная экстраполяция, как часто делают в Excel). Потом обучаем LightGBM и смотрим, бьёт ли он базлайн. Целевую переменную логарифмируем ($\log(1+y)$) — из-за тяжёлого хвоста это стабилизирует обучение и не даёт китам доминировать в лоссе.

**Задача.** На дозревших игроках (`df[df["mature"]]`) разбей данные на train/test (`test_size=0.25, random_state=42`). Построй наивный базлайн: множитель `ratio = сумма ltv_true train / сумма early_net_sum train`, прогноз = `early_net_sum * ratio`. Обучи `LGBMRegressor` на `np.log1p(ltv_true)`, прогноз верни через `np.expm1(...).clip(0)`. Посчитай MAE обоих подходов: положи в `base_mae` и `ml_mae`.

??? tip "Подсказка"

    `from sklearn.metrics import mean_absolute_error`. MAPE считай только по `true > 1` (иначе деление на ~0 взрывается). Параметры LightGBM: `n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1`.

**Критерий шага:**

```python
assert base_mae > 0 and ml_mae > 0
assert ml_mae < base_mae, "LightGBM обязан побить наивный множитель по MAE"
print(f"OK: baseline MAE={base_mae:.1f}, lightgbm MAE={ml_mae:.1f} — модель бьёт базлайн")
```

??? success "Решение"

    ```python
    # step3_ltv_model.py
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import lightgbm as lgb

    df = pd.read_parquet("features.parquet")
    mature = df[df["mature"]].copy()

    features = ["early_dep_count", "early_dep_sum", "early_dep_mean", "early_dep_max",
               "early_net_sum", "early_active_days", "first_dep_day"]
    X = mature[features]
    y = mature["ltv_true"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    # базлайн: множитель из train (полный LTV / ранний net revenue)
    ratio = y_tr.sum() / (X_tr["early_net_sum"].sum() + 1e-9)
    base_pred = X_te["early_net_sum"] * ratio

    # модель на log-таргете
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
    model.fit(X_tr, np.log1p(y_tr))
    ml_pred = np.expm1(model.predict(X_te)).clip(0)

    def report(name, true, pred):
        mae = mean_absolute_error(true, pred)
        mask = true > 1  # MAPE только по ненулевым, иначе деление на ~0
        mape = (np.abs(true[mask] - pred[mask]) / true[mask]).mean() * 100
        print(f"{name:10s} MAE={mae:8.1f}  MAPE={mape:6.1f}%")
        return mae

    base_mae = report("baseline", y_te.values, base_pred.values)
    ml_mae = report("lightgbm", y_te.values, ml_pred)

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nВажность фич:\n", imp)
    ```

    **Почему так.** LightGBM (MAE ~15.7) заметно обходит базлайн (MAE ~17.5): наивный множитель одинаков для всех, а модель ловит нелинейности (игрок с одним крупным депозитом в день 1 ≠ игрок с пятью мелкими). По важности фич лидируют `early_net_sum` и `early_dep_sum` — ранние деньги предсказывают поздние. Это твой первый артефакт: модель, превращающая 7 дней поведения в прогноз годовой ценности. (Точные числа MAE зависят от версии LightGBM, поэтому критерий проверяет факт «модель бьёт базлайн», а не конкретное значение.)

!!! note "Почему MAPE коварен на LTV"

    MAPE делит на истинное значение. У игроков с околонулевым LTV любая абсолютная ошибка даёт гигантский процент, поэтому мы фильтруем `true > 1`. В отчёте всегда показывай обе метрики: MAE говорит о деньгах (на сколько рублей ошиблись), MAPE — об относительной точности. Для бизнеса MAE на агрегате важнее, потому что суммарный прогноз бюджета считается в деньгах, а не в процентах.

### Шаг 4: Применить модель к молодым когортам

**Зачем.** Модель обучена на дозревших. Теперь её ценность — предсказать LTV тех, кто ещё живёт. Берём молодых игроков, считаем их 7-дневные фичи (они уже доступны), прогоняем модель и получаем прогноз полной ценности. Это разрешает проблему из Шага 1: вместо "выручки на сегодня" у молодой когорты появляется обоснованный прогноз итоговой.

**Задача.** Обучи модель заново на всех дозревших (`mature[features]` → `np.log1p(ltv_true)`). Примени к молодым (`df[~df["mature"]]`), положи прогноз в колонку `ltv_pred` (через `expm1(...).clip(0)`). Сгруппируй молодых по `source`, посчитай средний `pred_ltv` и отсортируй по убыванию. Положи результат в `by_src`.

**Критерий шага:**

```python
assert "ltv_pred" in young.columns
order = list(by_src.sort_values("pred_ltv", ascending=False).index)
assert order[-1] == "social", "social должен оказаться слабейшим источником по прогнозу"
assert "seo" in order[:2], "seo должен быть среди двух сильнейших источников"
print("OK: источники ранжированы по прогнозному LTV:", order)
```

??? success "Решение"

    ```python
    # step4_apply.py
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    df = pd.read_parquet("features.parquet")
    features = ["early_dep_count", "early_dep_sum", "early_dep_mean", "early_dep_max",
               "early_net_sum", "early_active_days", "first_dep_day"]

    mature = df[df["mature"]]
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(mature[features], np.log1p(mature["ltv_true"]))

    young = df[~df["mature"]].copy()
    young["ltv_pred"] = np.expm1(model.predict(young[features])).clip(0)

    # сравнение качества источников по ПРОГНОЗУ, а не по сырой выручке
    by_src = young.groupby("source").agg(
        players=("ltv_pred", "count"),
        pred_ltv=("ltv_pred", "mean"),
    ).sort_values("pred_ltv", ascending=False)
    print(by_src.round(1))
    young[["source", "ltv_pred"]].to_parquet("young_predictions.parquet")
    ```

    **Почему так.** Рейтинг получается `seo` > `affiliate` > `ppc` > `social` — модель восстановила порядок, зашитый в генераторе через `src_mult` (1.25 / 1.05 / 0.85 / 0.7), хотя молодые игроки всех источников пока принесли мало денег. Это и есть бизнес-польза: решение о перераспределении бюджета на 7-й день, а не через год.

Проверь понимание:

```text
Q: Мы не подавали source как фичу в модель. Почему модель всё равно различает источники по прогнозу?
[ ] Модель случайно угадала порядок
[x] Источник влияет на quality, а quality управляет ранним поведением - модель видит источник косвенно через ранние депозиты
[ ] LightGBM читает source из имени файла
> Игроки seo в среднем больше депают на первой неделе. Модель ловит это через поведенческие фичи, и порядок источников восстанавливается без явного признака source.
---
Q: Что рискованно при дообучении модели с явным категориальным признаком source?
[ ] Ничего, source всегда улучшает модель
[x] Риск утечки и хрупкости: при смене маркетинг-микса (новая партнёрка) модель с явным source экстраполирует хуже, чем на устойчивых поведенческих фичах
[ ] source нельзя кодировать категориально
> Поведенческие фичи устойчивее к смене источников трафика. Явный source-признак привязывает модель к текущему распределению каналов. Если добавляешь - проверяй стабильность на out-of-time валидации.
```

### Шаг 5: Прогноз выручки по когортам через кривую дожития

**Зачем.** Индивидуальный LTV отвечает на "кто ценный". Финансам нужен другой вопрос — "сколько денег придёт". Здесь правильный объект не игрок, а **когорта**, и инструмент — кривая дожития (retention/decay) депозитной активности. Идея: построить по дозревшим когортам усреднённую накопительную кривую "доля LTV, накопленная к дню $t$", затем для молодых когорт экстраполировать недостающий хвост. Это классический подтягивающий метод (cohort maturation) — он надёжнее, чем гонять ML на агрегатах.

Формально: пусть $c(t)$ — доля полного LTV когорты, накопленная к возрасту $t$ дней. Тогда для молодой когорты возраста $a$ с наблюдаемой выручкой $R_{obs}$ прогноз полной выручки:

$$\hat{R}_{full} = \frac{R_{obs}}{c(a)}$$

**Задача.** Построй накопительную кривую `curve` (массив длины `HORIZON+1`, значения 0..1, монотонно растёт) **только по дозревшим когортам** (`age_days >= HORIZON`). Затем для каждой месячной когорты посчитай наблюдаемую выручку `rev_obs`, фактор зрелости `matur_factor = c(median_age)` и прогноз полной выручки `rev_forecast_full = rev_obs / matur_factor`. Собери таблицу `res`.

??? tip "Подсказка"

    Кривая: `md.groupby("day_since_signup")["net_revenue"].sum().reindex(range(HORIZON+1), fill_value=0).cumsum()`, затем подели на последнее значение. Защити деление: если `curve[a]` около нуля, замени на маленькую константу (`1e-6`).

**Критерий шага:**

```python
import numpy as np
assert len(curve) == 366
assert np.all(np.diff(curve) >= -1e-9), "накопительная кривая обязана монотонно расти"
assert abs(curve[-1] - 1.0) < 1e-6, "к концу горизонта накоплено 100% LTV"
assert 0.06 <= curve[7] <= 0.16, "к 7-му дню накоплена малая доля LTV (~0.11)"
youngest = res.sort_values("age").iloc[0]
assert youngest["matur_factor"] < 0.5, "у самой молодой когорты фактор зрелости мал"
assert youngest["rev_forecast_full"] > youngest["rev_obs"], "прогноз подтягивает молодую когорту вверх"
print("OK: maturation-кривая построена, молодые когорты подтянуты")
```

??? success "Решение"

    ```python
    # step5_cohort_curve.py
    import numpy as np
    import pandas as pd

    players = pd.read_parquet("players.parquet")
    deposits = pd.read_parquet("deposits.parquet")
    OBS_DATE = pd.Timestamp("2026-01-01")
    HORIZON = 365

    players["age_days"] = (OBS_DATE - players["signup_date"]).dt.days
    players["cohort"] = players["signup_date"].dt.to_period("M")
    dep = deposits.merge(players[["player_id", "cohort", "age_days"]], on="player_id")

    # кривая дожития строится ТОЛЬКО по дозревшим когортам (age >= HORIZON)
    mature_pids = players.loc[players["age_days"] >= HORIZON, "player_id"]
    md = dep[dep["player_id"].isin(mature_pids)].copy()

    days = np.arange(0, HORIZON + 1)
    cum_rev = (md.groupby("day_since_signup")["net_revenue"].sum()
                 .reindex(days, fill_value=0).cumsum())
    curve = (cum_rev / cum_rev.iloc[-1]).values  # c(t): 0..1, монотонно растёт

    def maturation_factor(age):
        a = int(min(max(age, 0), HORIZON))
        return curve[a] if curve[a] > 1e-6 else 1e-6

    obs = dep[dep["deposit_date"] <= OBS_DATE]
    rows = []
    for coh, g in players.groupby("cohort"):
        pids = g["player_id"]
        age = g["age_days"].median()
        r_obs = obs.loc[obs["player_id"].isin(pids), "net_revenue"].sum()
        cf = maturation_factor(age)
        rows.append((str(coh), len(g), int(age), round(r_obs, 0),
                     round(cf, 3), round(r_obs / cf, 0)))
    res = pd.DataFrame(rows, columns=["cohort", "players", "age", "rev_obs",
                                      "matur_factor", "rev_forecast_full"])
    res.to_parquet("cohort_forecast.parquet")
    print(res.to_string(index=False))
    ```

    **Почему так.** У старых когорт `matur_factor` ≈ 1.0 (почти всё уже видно), у самой свежей (возраст ~16 дней) — около 0.23, и подтягивающий множитель большой (×4-5). Кривая на 7-й день даёт ~0.11, на 30-й ~0.36, на 90-й ~0.74, на 180-й ~0.94 — ровно та decay-механика, что зашита в генератор. Сумма `rev_forecast_full` по всем когортам — прогноз полной ценности всего пула; сравним его с наивной суммой на Шаге 8.

!!! warning "Где этот метод врёт"

    Maturation-кривая предполагает, что молодые когорты ведут себя как старые. Если ты сменил продукт, гео или маркетинг-микс — форма кривой у новых когорт другая, и экстраполяция поедет. Это допущение надо проверять: строй кривые отдельно по последним дозревшим когортам и смотри, не дрейфует ли форма. Не дрейфует — метод валиден; дрейфует — нужна модель с ковариатами.

### Шаг 6: Прогноз выручки во времени и интервалы неопределённости

**Зачем.** Шаг 5 дал полную ценность когорт. Финансам нужен ещё временной разрез: сколько выручки придёт по дням вперёд. Здесь подключается M28 — временной ряд. Возьмём ежедневную наблюдаемую выручку, выделим тренд экспоненциальным сглаживанием (Holt с демпфированием) и спрогнозируем на горизонт. И главное — дадим не точечный прогноз, а **коридор неопределённости**.

**Задача.** Построй дневной ряд наблюдаемой выручки, возьми последние 180 дней как историю. Обучи `ExponentialSmoothing(trend="add", damped_trend=True, seasonal=None)`, спрогнозируй на `H=90` дней в `point`. Построй 90%-й коридор `lo90`/`hi90` бутстрэпом остатков обучения (ресэмпл остатков, добавь к точечному прогнозу, возьми 5-й и 95-й перцентили). Собери `fc` с колонками `forecast`, `lo90`, `hi90` и шириной `interval_width`.

??? tip "Подсказка"

    Остатки: `(daily - model.fittedvalues).dropna().values`. На каждой из `B=500` симуляций: `point.values + rng.choice(resid, size=H, replace=True)`, клипни в ноль. Перцентили — `np.percentile(sims, 5, axis=0)` и `95`.

**Критерий шага:**

```python
import numpy as np
assert len(fc) == 90
assert (fc["forecast"] >= 0).all()
assert (fc["lo90"] <= fc["forecast"] + 1e-6).all(), "точечный прогноз внутри коридора снизу"
assert (fc["forecast"] <= fc["hi90"] + 1e-6).all(), "точечный прогноз внутри коридора сверху"
assert (fc["interval_width"] > 0).all(), "коридор имеет положительную ширину"
print("OK: дневной прогноз на 90 дней с коридором 90% построен")
```

**Self-check: расширяется ли коридор с горизонтом?**

- [ ] Сравни `interval_width` на дне 1 и дне 90. Расширяется ли он?
- [ ] Объясни наблюдение: почему на этом ряде коридор почти постоянной ширины?

??? success "Решение"

    ```python
    # step6_revenue_ts.py
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    deposits = pd.read_parquet("deposits.parquet")
    OBS_DATE = pd.Timestamp("2026-01-01")

    obs = deposits[deposits["deposit_date"] <= OBS_DATE]
    daily = (obs.set_index("deposit_date")["net_revenue"]
                .resample("D").sum().fillna(0))
    daily = daily.loc[:OBS_DATE].iloc[-180:]  # последние 180 дней как история

    model = ExponentialSmoothing(daily, trend="add", damped_trend=True,
                                 seasonal=None, initialization_method="estimated").fit()

    H = 90
    point = model.forecast(H)

    resid = (daily - model.fittedvalues).dropna().values
    B = 500
    sims = np.empty((B, H))
    rng = np.random.default_rng(42)
    for b in range(B):
        noise = rng.choice(resid, size=H, replace=True)
        sims[b] = (point.values + noise).clip(0)
    lo = np.percentile(sims, 5, axis=0)
    hi = np.percentile(sims, 95, axis=0)

    fc = pd.DataFrame({"forecast": point.values.clip(0), "lo90": lo, "hi90": hi})
    fc["interval_width"] = fc["hi90"] - fc["lo90"]
    print(fc.iloc[[0, 29, 59, 89]].round(1).to_string())
    print(f"Ширина: день 1 = {fc['interval_width'].iloc[0]:.0f}, "
          f"день 90 = {fc['interval_width'].iloc[-1]:.0f}")
    ```

    **Почему так.** Точечный прогноз ~221/день, коридор шириной ~170-190. И вот честный, неочевидный результат: **на этом ряде коридор почти не расширяется** (ширина день 1 ≈ 179, день 90 ≈ 188). Причина — агрегатная дневная выручка фиксированного пула прошлых регистраций почти стационарна (уровень + шум), поэтому одношаговый остаток описывает неопределённость почти на любом горизонте, и iid-ресэмпл даёт коридор постоянной ширины. Расширение коридора с горизонтом проявляется на **трендовых рядах или случайном блуждании**, где неопределённость накапливается шаг за шагом. Вывод senior-уровня: не предполагай, что коридор расширяется — измерь и пойми динамику ряда.

!!! note "Почему damped_trend, а не обычный"

    Обычный аддитивный тренд экстраполируется линейно в бесконечность — на 90 днях это даёт нереалистичный рост или провал. `damped_trend=True` затухает наклон со временем: прогноз стабилизируется на разумном уровне. Для выручки, которая не растёт линейно вечно, демпфирование — почти всегда правильный выбор по умолчанию.

### Шаг 7: Растёт ли ошибка на горизонте — измерь, не предполагай

**Зачем.** Распространённый тезис M28: прогноз на 1 день вперёд точнее, чем на 90. Это надо не заявлять, а **измерить** через backtest (rolling-origin / walk-forward). Отрезаем последние $H$ дней истории, прогнозируем их с нескольких точек отсчёта, считаем MAE по каждому шагу горизонта и смотрим, есть ли рост ошибки с дальностью. Дисциплина важнее лозунга: иногда роста нет, и это тоже результат.

**Задача.** На полном дневном ряде наблюдаемой выручки сделай backtest: несколько origin-точек, на каждой обучи `ExponentialSmoothing(trend="add", damped_trend=True)` на 180 днях истории и спрогнозируй `H=60` дней. Усредни абсолютную ошибку по шагам горизонта в `mae_by_step` (длина 60). Посчитай корреляцию между номером шага и MAE в `corr`.

**Критерий шага:**

```python
import numpy as np
assert len(mae_by_step) == 60
assert np.all(np.isfinite(mae_by_step)) and np.all(mae_by_step >= 0)
assert -1 <= corr <= 1
print(f"OK: backtest прогнан, MAE по горизонту посчитан, corr(шаг, MAE)={corr:.2f}")
```

**Self-check: интерпретация результата**

- [ ] Посмотри на `corr`. На этом стационарном ряде она близка к нулю — устойчивого роста ошибки с горизонтом нет.
- [ ] Объясни, почему: для трендового ряда или random-walk ошибка росла бы заметно, а у почти стационарного агрегата она быстро выходит на плато.
- [ ] Сделай вывод: коридор неопределённости из Шага 6 всё равно обязателен — но его поведение надо обосновывать измерением, а не лозунгом «ошибка всегда растёт».

??? success "Решение"

    ```python
    # step7_backtest.py
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    deposits = pd.read_parquet("deposits.parquet")
    OBS_DATE = pd.Timestamp("2026-01-01")
    obs = deposits[deposits["deposit_date"] <= OBS_DATE]
    daily = obs.set_index("deposit_date")["net_revenue"].resample("D").sum().fillna(0)
    daily = daily.loc[:OBS_DATE]

    H = 60
    origins = [len(daily) - H - k for k in range(0, 60, 10)]  # несколько точек отсчёта
    errors = np.zeros((len(origins), H))
    for i, o in enumerate(origins):
        train = daily.iloc[max(0, o - 180):o]
        actual = daily.iloc[o:o + H].values
        m = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        pred = m.forecast(H).values.clip(0)
        errors[i] = np.abs(actual - pred)

    mae_by_step = errors.mean(axis=0)
    steps = pd.DataFrame({"horizon_day": np.arange(1, H + 1), "mae": mae_by_step.round(1)})
    print(steps.iloc[[0, 9, 29, 59]].to_string(index=False))
    corr = np.corrcoef(steps["horizon_day"], steps["mae"])[0, 1]
    print(f"\nКорреляция (горизонт, MAE) = {corr:.2f}")
    ```

    **Почему так.** MAE по шагам колеблется в районе 25-57 без устойчивого тренда, и корреляция близка к нулю (~0.06). Это честный эмпирический результат: агрегатная дневная выручка фиксированного пула почти стационарна, поэтому ошибка прогноза быстро выходит на плато, а не растёт линейно. Урок не «ошибка всегда растёт», а «проверяй динамику ряда»: для трендов и случайного блуждания рост был бы выраженным, и тогда расширяющийся коридор обязателен. Никогда не давай финансам точечное число без коридора — даже если backtest показал плато, неопределённость уровня никуда не делась.

Проверь понимание:

```text
Q: Почему backtest делают на нескольких origin-точках, а не на одной?
[ ] Так быстрее считается
[x] Один origin - это одна случайная реализация; несколько точек усредняют ошибку и дают честную оценку, а не подгонку под удачный отрезок
[ ] statsmodels требует минимум 6 origin
> По одной точке отсчёта нельзя судить об устойчивости прогноза. Усреднение по нескольким origin защищает от везения/невезения конкретного отрезка.
---
Q: Чем плоха линейная экстраполяция тренда (без демпфирования) на длинном горизонте?
[ ] Она слишком медленная
[x] Линейный тренд уходит в бесконечность - на 90+ днях даёт абсурд (выручка в минус или в космос), реальные процессы насыщаются
[ ] Линейный тренд нельзя обучить на 180 днях
> Демпфирование затухает наклон и делает прогноз реалистичным на длинном горизонте. Без него экстраполяция ломается тем сильнее, чем дальше прогноз.
---
Q: Если бизнес требует один прогноз выручки на квартал - что сообщить кроме точечного числа?
[ ] Только точечное число, коридор пугает совет директоров
[x] Интервал неопределённости и допущения модели (стабильность маркетинг-микса, отсутствие шоков) - точечное число без коридора создаёт ложную уверенность
[ ] Среднее за прошлый квартал вместо прогноза
> Точечный прогноз без диапазона подставляет команду, если факт уедет. Коридор и явные допущения - обязательная часть честного прогноза для стейкхолдера.
```

### Шаг 8: Свести наивный и честный прогноз — цена цензурирования

**Зачем.** Финальный аккорд — показать в деньгах, насколько наивный подход (усреднить наблюдаемое) занижает оценку относительно подхода с учётом цензурирования. Это тот аргумент, который убеждает руководство тратить силы на корректную методологию.

**Задача.** Посчитай три числа: `naive_total` — сумма наблюдаемой выручки, `forecast_total` — сумма `rev_forecast_full` из Шага 5, `true_total` — сумма всей выручки (вся синтетика, включая ненаблюдаемое будущее). Посчитай, на сколько процентов наив занижает истину, и какова ошибка прогноза.

Сначала прикинь масштаб занижения сам:

```text
TASK: Наблюдаемая выручка пула на сегодня = 167203, истинная полная выручка = 180936. На сколько ПРОЦЕНТОВ наивный подход (только наблюдаемое) занижает истинную ценность? Формула (1 - набл/истина)*100. Округли до 0.1.
ANSWER: 7.6
TOL: 0.5
UNIT: %
PLACEHOLDER: 0.0
EXPLAIN: (1 - 167203/180936) * 100 = (1 - 0.924) * 100 = 7.6%. Это цена цензурирования в процентах: наив теряет невидимое будущее молодых когорт. Прогноз через maturation-кривую (179874) подбирается к истине с ошибкой всего ~0.6% - на порядок точнее наива.
```

**Критерий шага:**

```python
assert naive_total < true_total, "наив обязан занижать истину"
underestimate = (1 - naive_total / true_total) * 100
forecast_err = abs(forecast_total / true_total - 1) * 100
assert underestimate > 3, "занижение наива заметное (порядка нескольких процентов)"
assert forecast_err < 3, "прогноз через дожитие близок к истине (ошибка < 3%)"
assert forecast_err < underestimate, "прогноз точнее наива"
print(f"OK: наив занижает на {underestimate:.1f}%, ошибка прогноза {forecast_err:.1f}%")
```

??? success "Решение"

    ```python
    # step8_compare.py
    import pandas as pd

    deposits = pd.read_parquet("deposits.parquet")
    cohort_fc = pd.read_parquet("cohort_forecast.parquet")
    OBS_DATE = pd.Timestamp("2026-01-01")

    obs = deposits[deposits["deposit_date"] <= OBS_DATE]
    naive_total = obs["net_revenue"].sum()                     # что видно сейчас
    true_total = deposits["net_revenue"].sum()                 # истина (вся синтетика)
    forecast_total = cohort_fc["rev_forecast_full"].sum()      # прогноз через maturation

    print(f"Наивно (наблюдаемое):     {naive_total:12.0f}")
    print(f"Прогноз (с дожитием):     {forecast_total:12.0f}")
    print(f"Истина (полный горизонт): {true_total:12.0f}")
    print(f"Недооценка наива: {(1 - naive_total / true_total) * 100:.1f}%")
    print(f"Ошибка прогноза:  {abs(forecast_total / true_total - 1) * 100:.1f}%")
    ```

    **Почему так.** Наивная сумма (~167k) ниже истины (~181k) на ~7.6% — это цена цензурирования в деньгах. Прогноз через maturation-кривую (~180k) подбирается к истине с ошибкой ~0.6% — на порядок точнее наива. Это финальная демонстрация всего воркшопа: учёт незрелых когорт превращает заниженную "выручку на сегодня" в обоснованный прогноз полной ценности. (Заметь: при более молодом пуле или более коротком окне наблюдения занижение наива было бы в разы больше — здесь оно умеренное, потому что половина когорт уже дозрела.)

## Типичные ошибки

- **Усреднение LTV по смеси когорт.** Самая частая и дорогая ошибка. Среднее по дозревшим и недозревшим занижает оценку и наказывает свежий трафик. Всегда смотри LTV в разрезе возраста когорты.
- **Обучение модели раннего LTV на цензурированных таргетах.** Если взять `ltv` молодых игроков как цель, модель выучит обрезанную зависимость. Обучай только на когортах старше горизонта (на синтетике — на `mature`).
- **MAPE на нулевых LTV.** Большая доля игроков приносит около нуля. MAPE взрывается на делении. Фильтруй ненулевые для MAPE и всегда дублируй MAE в деньгах.
- **Выживший в кривой дожития (survivorship).** Если строить maturation-кривую только по «ещё активным» игрокам, потеряешь отток и завысишь LTV. Кривая должна строиться по всем игрокам когорты, включая ушедших с нулём.
- **Линейная экстраполяция тренда на длинный горизонт.** Без демпфирования прогноз уходит в абсурд. Damped trend по умолчанию для выручки.
- **Точечный прогноз без интервалов.** Senior-ошибка: дать совету директоров одно число на квартал. Даже если backtest показал плато (Шаг 7), неопределённость уровня остаётся — коридор обязателен.
- **Слепое допущение «ошибка/коридор растёт с горизонтом».** Шаги 6-7 показали: на стационарном агрегате роста может не быть. Не лозунг, а измерение: backtest и форма ряда решают, растёт ли неопределённость.
- **Игнор дрейфа когорт (senior-уровень).** Maturation-метод валиден, только пока новые когорты похожи на старые по форме кривой. После смены продукта/гео/оффера форма меняется, и экстраполяция тихо ломается. Периодически переоценивай кривую на последних дозревших когортах. Это самая незаметная ловушка — модель не падает, она просто врёт.
- **Путаница депозита и выручки.** Депозит не равен net revenue. Считать LTV по сумме депозитов без учёта выплат и hold — завышение в разы. Согласуй определение с финансами до кода.

!!! tip "AI-копилот в этом воркшопе"

    Где нейросеть реально ускорит: генерация boilerplate для агрегаций pandas/groupby, написание функции бутстрэпа остатков, объяснение параметров `ExponentialSmoothing`, рефактор Шага 2 в polars для скорости, генерация графиков matplotlib по описанию.

    Где AI подведёт именно здесь: (1) спросишь "посчитай LTV" — даст наивное среднее по всем игрокам без учёта цензурирования (ловушка Шага 1); (2) предложит обучить модель на всех игроках, включая молодых, не заметив обрезанного таргета; (3) построит прогноз ряда без демпфирования и без интервалов, потому что "так короче"; (4) с готовностью заявит "ошибка растёт с горизонтом" как универсальный факт, не проверив его на ряде (Шаг 7 показал обратное); (5) не задаст вопрос про депозит vs net revenue и молча завысит. Вывод: AI хорош на механике (код агрегаций, синтаксис), но методологические решения — что брать в таргет, на ком учить, как считать неопределённость — держи за собой и проверяй каждый кусок против ловушек выше.

## Критерий готовности

- [ ] Синтетика генерируется детерминированно (`seed=42`), есть наблюдаемые и цензурированные депозиты, окно signup шире горизонта.
- [ ] Шаг 1 показывает занижение наблюдаемой выручки у молодых когорт (таблица по месяцам).
- [ ] Фичи раннего поведения собраны за окно 7 дней, таргет — полный горизонт, флаг `mature` отделяет дозревших (~3940).
- [ ] Модель раннего LTV (LightGBM на log-таргете) обучена только на `mature` и бьёт наивный базлайн по MAE.
- [ ] Модель применена к молодым когортам, источники ранжированы по прогнозному LTV (social последний, seo в топе).
- [ ] Построена maturation-кривая по дозревшим когортам (монотонная, c(365)=1), прогноз полной выручки по когортам получен.
- [ ] Прогноз дневной выручки на горизонт с коридором 90% через бутстрэп остатков; поведение ширины коридора объяснено.
- [ ] Backtest прогнан; рост MAE с горизонтом измерен (а не заявлен) и интерпретирован.
- [ ] Финальное сравнение: наив vs прогноз vs истина в деньгах, занижение наива измерено в процентах (~7.6%).
- [ ] В отчёте обе метрики (MAE и MAPE) с пояснением, почему MAPE фильтруется.

## Бизнес-вывод

Технический результат (две метрики MAE/MAPE, рейтинг источников, maturation-кривая, коридор прогноза) сам по себе CMO ничего не говорит. Переведи его в решение на языке денег и действий.

- [ ] **Рекомендация:** какие источники масштабировать, а какие резать — по прогнозному раннему LTV, не по сырой наблюдаемой выручке (например: «social недооценён сырой метрикой меньше, чем кажется, но по прогнозу он действительно слабейший — режем долю осознанно, а не из-за цензурирования»).
- [ ] **Эффект в деньгах:** во сколько ₽ обходится наивное ранжирование — используй сравнение из Шага 8 (занижение наива в %) и переведи в бюджет квартала (порядок 4-5 млн ₽ перелитых не туда при ошибке в ранжировании).
- [ ] **Риски и допущения:** maturation-кривая валидна, пока новые когорты похожи на старые (дрейф продукта/гео/оффера ломает экстраполяцию тихо); поведение ошибки на горизонте проверено backtest'ом, квартальное число даётся только с коридором.
- [ ] **Следующий шаг:** что сделать после защиты — A/B перераспределения бюджета между каналами, либо мониторинг дрейфа кривой на свежих дозревших когортах.
- [ ] **Как подать стейкхолдеру:** один слайд — «прогноз выручки квартала X ± Y млн ₽, рекомендуем перераспределить бюджет так-то, цена бездействия Z млн ₽». Язык решений и денег, а не MAE и log-таргета.

## Развитие

- **Вероятностный LTV через lifetimes / BG-NBD + Gamma-Gamma.** Замени регрессию на классическую probabilistic CLV-модель (пакет `lifetimes`): отдельно частота покупок (BG-NBD) и денежная ценность (Gamma-Gamma). Сравни с ML-подходом по калибровке.
- **Квантильная регрессия для индивидуальных интервалов.** Обучи LightGBM с `objective="quantile"` на 0.1/0.5/0.9 и дай не точечный LTV, а коридор на каждого игрока. Полезно для отсечки рискового трафика.
- **Conformal prediction для гарантированного покрытия.** Оберни модель раннего LTV в conformal-обёртку (`mapie`), чтобы интервалы имели доказанное покрытие без допущений о распределении.
- **Трендовый ряд и расширяющийся коридор.** Сгенерируй версию данных с растущим притоком регистраций (нестационарный дневной ряд) и повтори Шаги 6-7 — убедись, что теперь коридор и MAE растут с горизонтом. Это закрепит урок «динамика ряда решает».
- **Дрейф-монитор maturation-кривой.** Автоматизируй проверку формы кривой дожития на последних дозревших когортах и алерт при значимом сдвиге — операционализируй senior-ловушку из раздела ошибок.

## Что ты закрепил

Ты связал воедино два модуля и два разных взгляда на одну бизнес-задачу. Из **M14** — регрессия с тяжёлым хвостом, log-трансформация таргета, базлайн vs бустинг, MAE/MAPE и их подводные камни, важность фич, честное разделение train/test по зрелости. Из **M28** — кривая дожития и maturation, экспоненциальное сглаживание с демпфированным трендом, прогноз на горизонт, backtest методом rolling-origin и измерение (а не декларация) роста ошибки, интервалы неопределённости через бутстрэп.

Главный навык, который не сводится к коду: ты научился видеть **цензурирование** — структурную причину, по которой наивная аналитика занижает ценность, — строить два независимых, но согласованных артефакта (индивидуальный ранний LTV для маркетинга и когортный прогноз выручки с интервалами для финансов) и честно проверять собственные допущения о неопределённости, а не повторять лозунги. Это ровно та связка, которую в продуктовой и гемблинг-аналитике спрашивают на senior-уровне.
