# W3 · Churn-модель игроков end-to-end

<span class="lecture-meta">Воркшоп · ориентир 10-16 ч · Продвинутый</span>

## Что ты построишь

Ты построишь **полный churn-предиктор для игроков гемблинг-проекта**: от сырого лога событий до обученной модели LightGBM с честной валидацией, бизнес-обоснованным порогом и объяснением драйверов оттока через SHAP.

На выходе три артефакта:

1. **Обученная модель** `churn_model.pkl`, которая по поведению игрока за «окно наблюдения» предсказывает вероятность того, что он уйдёт в отток в ближайшие N дней.
2. **Таблица драйверов оттока** — какие признаки толкают игрока к уходу, с направлением и силой эффекта (SHAP).
3. **Порог принятия решения**, выбранный не по умолчанию 0.5, а по матрице стоимости ошибок: сколько стоит зря дёрнуть удержанием лояльного игрока против того, сколько мы теряем, упустив уходящего.

Это не учебный `model.fit(X, y)` на готовом датасете. Главная ценность воркшопа — научиться **не обманывать себя**: правильно определить таргет, сделать временной сплит, поймать утечки и измерить модель метрикой, которая не врёт при дисбалансе классов.

!!! info "Почему именно churn"

    Удержание игрока в 5-7 раз дешевле привлечения нового. Но кампания удержания (бонус, звонок, пуш) стоит денег и раздражает лояльных. Модель, которая точно показывает, кто реально на грани ухода, напрямую конвертируется в деньги. Это самый «продуктовый» ML-проект в курсе.

## Предпосылки

Нужна вся **Фаза 4**:

- **M13 — основы ML**: train/test, переобучение, bias-variance.
- **M14 — supervised**: классификация, логистическая регрессия, градиентный бустинг.
- **M15 — unsupervised**: пригодится для понимания, но напрямую не используется.
- **M16 — feature engineering**: оконные агрегаты, лаги, recency — ядро этого воркшопа.
- **M17 — подводные камни и интерпретация**: утечки данных, метрики при дисбалансе, SHAP — это половина воркшопа.

Окружение через **uv** (актуальный стандарт 2026):

```bash
uv init churn-w3 && cd churn-w3
uv add pandas numpy scikit-learn lightgbm shap matplotlib pyarrow
uv add --dev jupyterlab
uv run jupyter lab
```

!!! tip "Почему uv, а не pip/conda"

    `uv` ставит зависимости в десятки раз быстрее pip и фиксирует точные версии в `uv.lock`. Для воспроизводимого ML-проекта детерминированное окружение так же важно, как фиксированный seed. Любой, кто склонирует репозиторий и сделает `uv sync`, получит ровно твоё окружение.

## Данные

Реального лога у нас нет, и это правильно: воркшоп должен запускаться у любого. Мы сгенерируем **синтетический поток поведенческих событий** игроков казино, в который заложим реалистичную структуру оттока — чтобы у модели было что находить, но без тривиального читерства.

### Зачем именно событийный лог, а не готовая таблица фич

Реальные данные приходят как **события** (`player_id`, `ts`, `event_type`, `amount`), а не как удобная матрица `признаки × игрок`. Самая частая ошибка джуна — начать с готовой таблицы и не заметить, что фичи и таргет считались на одном и том же временном окне. Мы сознательно стартуем с сырых событий, чтобы пройти весь путь и контролировать границу времени руками.

### Модель данных

Каждый игрок имеет дату регистрации и скрытый «уровень здоровья» (engagement), который дрейфует во времени. Часть игроков получает негативный тренд (надоело, проигрался, нашёл конкурента) и постепенно перестаёт заходить. Депозиты, ставки и сессии генерируются как пуассоновский поток с интенсивностью, зависящей от текущего engagement.

```python
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_PLAYERS = 8000
START = pd.Timestamp("2025-01-01")
HORIZON_DAYS = 240  # общая длина наблюдения

def gen_players(n):
    reg_offset = RNG.integers(0, 90, size=n)  # регистрация в первые 90 дней
    reg_date = START + pd.to_timedelta(reg_offset, unit="D")
    # базовый engagement: лог-нормальный, длинный хвост "китов"
    base_engagement = RNG.lognormal(mean=0.0, sigma=0.6, size=n)
    # у части игроков заложен негативный тренд активности
    decay = RNG.beta(2, 5, size=n)  # 0 = стабилен, ближе к 1 = быстро остывает
    country = RNG.choice(["UZ", "KZ", "RU", "TR", "BR"], size=n,
                         p=[0.30, 0.20, 0.20, 0.15, 0.15])
    device = RNG.choice(["mobile", "desktop"], size=n, p=[0.78, 0.22])
    return pd.DataFrame({
        "player_id": np.arange(n),
        "reg_date": reg_date,
        "base_engagement": base_engagement,
        "decay": decay,
        "country": country,
        "device": device,
    })

players = gen_players(N_PLAYERS)
```

Теперь разворачиваем события. Для каждого игрока проходим по дням его жизни; интенсивность активности падает по экспоненте от `decay`. Это и есть «скрытая» причина оттока, которую модель должна реконструировать по наблюдаемому поведению.

```python
def gen_events(players, horizon_days=HORIZON_DAYS):
    rows = []
    for p in players.itertuples():
        days_alive = horizon_days - (p.reg_date - START).days
        if days_alive <= 0:
            continue
        for d in range(days_alive):
            # текущая активность = база * экспоненциальное затухание + шум
            level = p.base_engagement * np.exp(-p.decay * d / 30.0)
            level = max(level, 0.02)
            # число сессий в этот день ~ Пуассон
            n_sessions = RNG.poisson(level)
            if n_sessions == 0:
                continue
            day = p.reg_date + pd.Timedelta(days=d)
            for _ in range(n_sessions):
                ts = day + pd.Timedelta(minutes=int(RNG.integers(0, 1440)))
                deposit = RNG.random() < 0.35
                dep_amount = float(RNG.lognormal(2.5, 0.8)) if deposit else 0.0
                n_bets = RNG.poisson(8) + 1
                bet_sum = float(RNG.lognormal(2.0, 0.7) * n_bets)
                rows.append((p.player_id, ts, dep_amount, n_bets, bet_sum))
    ev = pd.DataFrame(rows, columns=["player_id", "ts", "deposit", "n_bets", "bet_sum"])
    return ev.sort_values("ts").reset_index(drop=True)

events = gen_events(players)
print(events.shape)
print(events.head())
```

**Что получилось:** примерно 1.5-2 млн строк событий по 8000 игроков за ~8 месяцев. У каждого события — депозит, число ставок и оборот. У части игроков активность визуально затухает к концу окна — это будущий отток.

!!! note "Чем заменить на реальные данные"

    Структура `player_id / ts / amount / event_type` универсальна. Подставить можно:
    - **Kaggle: «Online Gaming / Telco Customer Churn»** — для отработки самого пайплайна (там таргет уже готов, но логику временного сплита всё равно надо натянуть).
    - **Hugging Face: датасеты транзакционных логов e-commerce** — событийная природа та же.
    - В проде Yohoho — таблица событий ставок/депозитов; меняется только источник, весь код ниже работает без изменений.

## Ход работы

### Шаг 1: Операционное определение churn

**Зачем.** «Отток» — не свойство природы, а наше решение. У гемблинга нет подписки, которую отменяют, поэтому churn определяется через **окно бездействия**: если игрок не сделал ни одной ставки за N дней — он в оттоке. Выбор N — бизнес-решение, не статистическое. Слишком маленькое N (7 дней) пометит как ушедших тех, кто просто в отпуске. Слишком большое (90) — мы узнаем об уходе, когда удерживать поздно.

Ещё две даты критичны:
- **observation cutoff** — момент «сейчас», на который мы делаем предсказание. Все фичи считаются **только по событиям до cutoff**.
- **outcome window** — следующие N дней после cutoff, по которым определяется таргет.

```python
CHURN_N_DAYS = 30
CUTOFF = START + pd.Timedelta(days=180)        # "сегодня" для модели
OUTCOME_END = CUTOFF + pd.Timedelta(days=CHURN_N_DAYS)

# Кандидаты: игроки, которые были активны хотя бы раз ДО cutoff
active_before = events[events.ts < CUTOFF].player_id.unique()

# Таргет: была ли активность в окне (CUTOFF, OUTCOME_END]?
in_window = events[(events.ts >= CUTOFF) & (events.ts < OUTCOME_END)]
retained_ids = set(in_window.player_id.unique())

labels = pd.DataFrame({"player_id": active_before})
labels["churn"] = (~labels.player_id.isin(retained_ids)).astype(int)
print(labels.churn.value_counts(normalize=True))
```

**Что получилось:** таргет с дисбалансом — обычно 15-30% оттока. Запомни эту цифру: она определит выбор метрики дальше.

!!! warning "Ловушка выживания"

    Мы берём в кандидаты только игроков, активных **до** cutoff. Иначе в выборку попадут те, кто зарегистрировался вчера или уже давно ушёл — для них предсказание бессмысленно. Это первое решение, которое отсекает мусор и делает таргет осмысленным.

### Шаг 2: Временной сплит (не случайный)

**Зачем.** Если сделать `train_test_split(shuffle=True)`, модель будет учиться на данных из будущего и предсказывать прошлое — в проде такого не бывает. Хуже: при оконных фичах случайный сплит почти гарантированно даёт **утечку через время** (фичи одного игрока размазаны по train и test). Честная оценка = train на раннем cutoff, test на позднем.

Делаем **два cutoff**: train предсказывает с позиции дня 150, test — с позиции дня 180. Окна не пересекаются, test полностью «в будущем» относительно train.

```python
def build_dataset(events, cutoff, n_days=CHURN_N_DAYS):
    outcome_end = cutoff + pd.Timedelta(days=n_days)
    active = events[events.ts < cutoff].player_id.unique()
    window = events[(events.ts >= cutoff) & (events.ts < outcome_end)]
    retained = set(window.player_id.unique())
    lab = pd.DataFrame({"player_id": active})
    lab["churn"] = (~lab.player_id.isin(retained)).astype(int)
    feats = make_features(events, cutoff)            # см. Шаг 3
    df = lab.merge(feats, on="player_id", how="left")
    return df

CUTOFF_TRAIN = START + pd.Timedelta(days=150)
CUTOFF_TEST  = START + pd.Timedelta(days=180)
```

!!! question "Проверь себя"

    1. Почему `shuffle=True` опасен именно для churn-задачи с оконными фичами?
    2. Что произойдёт с метрикой, если train-cutoff будет позже test-cutoff?
    3. Может ли один игрок быть и в train, и в test? Это проблема?

??? success "Ответы"

    1. Оконные агрегаты игрока коррелируют между соседними периодами; при перемешивании похожие строки попадут и в train, и в test, модель «подсмотрит» паттерн конкретного игрока, метрика окажется завышенной, а в проде просядет.
    2. Модель будет учиться на будущем и тестироваться на прошлом — оценка станет оптимистично-бессмысленной, фактически утечка времени.
    3. Да, может — это нормально, потому что фичи и таргет в train и test считаются на **разных временных окнах**. Проблема была бы при перемешивании внутри одного окна. Если хочется строгости — можно дополнительно держать игроков непересекающимися (см. GroupKFold в Шаге 9).

### Шаг 3: Feature engineering

**Зачем.** Сырые события модель не съест. Нужны признаки, описывающие поведение игрока **на момент cutoff**. Ключевая дисциплина: каждая фича считается строго по `events.ts < cutoff`. Опираемся на классический RFM-каркас плюс динамику.

Считаем три семейства признаков:
- **Recency** — сколько дней назад была последняя активность (сильнейший предиктор оттока).
- **Frequency / Monetary** — частота сессий, суммы депозитов и ставок за разные окна (7/30/90 дней).
- **Динамика (лаги/тренды)** — отношение активности последней недели к предыдущему месяцу: если падает — игрок остывает.

```python
def make_features(events, cutoff):
    ev = events[events.ts < cutoff].copy()
    ev["age_days"] = (cutoff - ev.ts).dt.days

    def win_agg(df, days, suffix):
        w = df[df.age_days < days]
        g = w.groupby("player_id").agg(
            sessions=("ts", "count"),
            dep_sum=("deposit", "sum"),
            dep_cnt=("deposit", lambda s: (s > 0).sum()),
            bet_sum=("bet_sum", "sum"),
            n_bets=("n_bets", "sum"),
        )
        return g.add_suffix(f"_{suffix}")

    f7  = win_agg(ev, 7,  "7d")
    f30 = win_agg(ev, 30, "30d")
    f90 = win_agg(ev, 90, "90d")

    base = ev.groupby("player_id").agg(
        recency_days=("age_days", "min"),     # дней с последней активности
        tenure_days=("age_days", "max"),      # как давно с нами
        lifetime_dep=("deposit", "sum"),
        lifetime_bets=("n_bets", "sum"),
    )

    feats = base.join([f7, f30, f90], how="left").fillna(0)

    # динамика: активность последней недели против среднего по месяцу
    feats["trend_sessions"] = feats["sessions_7d"] / (feats["sessions_30d"] / 4 + 1e-6)
    feats["dep_per_session"] = feats["dep_sum_30d"] / (feats["sessions_30d"] + 1e-6)

    # статика игрока
    pl = players.set_index("player_id")[["country", "device"]]
    feats = feats.join(pl, how="left").reset_index()
    return feats
```

**Что получилось:** таблица ~20+ числовых фич плюс категориальные `country`/`device`. `recency_days` и `trend_sessions` — наши главные кандидаты в драйверы.

!!! warning "Самая частая утечка новичка"

    Внутри `make_features` мы фильтруем `ev = events[events.ts < cutoff]` **первой строкой**. Если забыть этот фильтр и посчитать агрегаты по всему логу, в фичи протечёт информация из outcome-окна — например, `recency_days` посчитается уже по будущим событиям. Модель покажет AUC 0.99 на валидации и развалится в проде. Это не теория — так ломается большинство первых churn-моделей.

### Шаг 4: Baseline (правило и логистическая)

**Зачем.** Прежде чем тянуть LightGBM, нужен дешёвый ориентир. Если бустинг не бьёт глупое правило — модель не нужна. Baseline'ов два: тривиальное правило по recency и честная логистическая регрессия.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

train = build_dataset(events, CUTOFF_TRAIN)
test  = build_dataset(events, CUTOFF_TEST)

cat_cols = ["country", "device"]
num_cols = [c for c in train.columns
            if c not in ["player_id", "churn"] + cat_cols]

y_train, y_test = train.churn, test.churn

# Baseline 1: правило "не заходил > 14 дней -> уйдёт"
rule_pred = (test.recency_days > 14).astype(int)
print("rule PR-AUC proxy:", average_precision_score(y_test, test.recency_days))

# Baseline 2: логистическая на числовых фичах (в pipeline, без утечки масштаба)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                           class_weight="balanced"))
logit.fit(train[num_cols], y_train)
p_logit = logit.predict_proba(test[num_cols])[:, 1]
print("logit PR-AUC:", average_precision_score(y_test, p_logit))
print("logit ROC-AUC:", roc_auc_score(y_test, p_logit))
```

**Что получилось:** PR-AUC логистической обычно 0.5-0.65. Это планка, которую LightGBM обязан превзойти, иначе сложная модель не оправдана.

!!! tip "StandardScaler внутри pipeline — не для красоты"

    Если масштабировать фичи `fit_transform` на всём датасете до сплита — параметры масштаба (среднее, std) посчитаются с учётом test. Это **утечка через препроцессинг**. `make_pipeline` гарантирует, что scaler учится только на train. То же правило для любого импьютера, энкодера, отбора фич.

### Шаг 5: LightGBM

**Зачем.** Градиентный бустинг — рабочая лошадь табличного ML 2026: ест категориальные напрямую, устойчив к разномасштабным фичам, ловит нелинейности и взаимодействия. Используем встроенную поддержку категорий и раннюю остановку по валидации.

```python
import lightgbm as lgb

for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c]  = test[c].astype("category")

features = num_cols + cat_cols
dtrain = lgb.Dataset(train[features], label=y_train,
                     categorical_feature=cat_cols)
dvalid = lgb.Dataset(test[features], label=y_test, reference=dtrain)

params = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    "seed": 42,
    "verbose": -1,
}

model = lgb.train(params, dtrain, num_boost_round=2000,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

p_lgb = model.predict(test[features])
print("LGB PR-AUC:", average_precision_score(y_test, p_lgb))
print("LGB ROC-AUC:", roc_auc_score(y_test, p_lgb))
```

**Что получилось:** PR-AUC обычно 0.65-0.8, заметно выше логистической. `scale_pos_weight` компенсирует дисбаланс, ранняя остановка по test-метрике не даёт переобучиться.

!!! note "Почему не XGBoost/CatBoost"

    Все три бустинга дадут сопоставимый результат. LightGBM выбран за скорость и нативную работу с категориями без one-hot. CatBoost удобнее, если категорий много и они высококардинальные. Принцип воркшопа не в выборе библиотеки, а в правильной валидации — она одинакова для любой.

### Шаг 6: Метрика — PR-AUC, а не accuracy/ROC

**Зачем.** При 20% оттока модель «всем предскажу retained» даёт 80% accuracy и при этом бесполезна. Accuracy здесь врёт всегда. ROC-AUC честнее, но при сильном дисбалансе остаётся оптимистичным: огромное число истинно-негативных раздувает знаменатель. Нас интересует **качество среди тех, кого модель назвала уходящими** — это precision и recall, то есть PR-кривая.

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad \text{PR-AUC} = \int_0^1 P(r)\,dr$$

```python
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt

prec, rec, thr = precision_recall_curve(y_test, p_lgb)
baseline = y_test.mean()  # PR-AUC случайной модели = доля позитивов

plt.plot(rec, prec, label=f"LGB (AP={average_precision_score(y_test,p_lgb):.3f})")
plt.axhline(baseline, ls="--", color="gray", label=f"random={baseline:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.show()
```

**Что получилось:** PR-кривая, базовая линия которой — доля оттока. Любая осмысленная модель должна быть выше неё. Именно расстояние от этой линии, а не абсолютная цифра, говорит о пользе.

!!! question "Проверь себя"

    1. Модель предсказала 80% accuracy при 20% оттока. Хорошо?
    2. Почему ROC-AUC = 0.85 может вводить в заблуждение при дисбалансе 5%?
    3. Чему равна PR-AUC случайного предсказателя?

??? success "Ответы"

    1. Нет. Константа «никто не уйдёт» тоже даёт 80% и при этом ловит ноль оттока. Accuracy при дисбалансе бесполезна.
    2. ROC учитывает true-negative rate, а негативов огромное большинство — кривая легко уходит к высоким значениям, не отражая, что среди помеченных «уйдёт» полно ложных. PR-кривая фокусируется именно на позитивном классе.
    3. Доле позитивного класса в выборке (здесь ~доля оттока). Поэтому baseline на графике рисуют горизонталью на этом уровне.

### Шаг 7: Порог по бизнес-стоимости

**Зависит от денег, а не от 0.5.** Дефолтный порог `predict_proba > 0.5` оптимален только если ошибки равноценны. У нас не так. Зададим стоимость:

- **FN** (пропустили уходящего) — теряем будущий LTV игрока, скажем 50 у.е.
- **FP** (зря пометили лояльного) — стоимость кампании удержания + риск раздражения, скажем 5 у.е.

Подбираем порог, минимизирующий суммарную стоимость, а не максимизирующий F1.

```python
import numpy as np

COST_FN = 50.0   # упущенный игрок
COST_FP = 5.0    # зря потраченное удержание

thresholds = np.linspace(0.05, 0.95, 91)
costs = []
for t in thresholds:
    pred = (p_lgb >= t).astype(int)
    fp = ((pred == 1) & (y_test == 0)).sum()
    fn = ((pred == 0) & (y_test == 1)).sum()
    costs.append(fp * COST_FP + fn * COST_FN)

best_t = thresholds[int(np.argmin(costs))]
print(f"Оптимальный порог: {best_t:.2f}")

pred = (p_lgb >= best_t).astype(int)
print(classification_report(y_test, pred, digits=3))
```

**Что получилось:** порог обычно уезжает заметно ниже 0.5 (часто 0.15-0.3), потому что пропустить уходящего в 10 раз дороже, чем зря дёрнуть лояльного — модель должна быть «параноиком». Это и есть перевод ML в деньги.

!!! example "Матрица стоимости как продуктовый разговор"

    Числа `COST_FN`/`COST_FP` — не из кода, а из переговоров с продуктом и финансами. Реальный LTV берётся из когортного анализа (см. воркшоп W1/W5), стоимость кампании — из CRM-бюджета. Меняются цифры — двигается порог. Покажи стейкхолдеру кривую стоимости от порога: это лучший аргумент, почему не 0.5.

### Шаг 8: Интерпретация через SHAP

**Зачем.** Бизнесу мало вероятности — нужен ответ «почему этот игрок уходит» и «что вообще гонит отток». SHAP раскладывает каждое предсказание на вклады признаков, аддитивно и согласованно. Глобально — какие фичи важны; локально — почему именно этот игрок помечен.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test[features])

# Глобально: важность и направление
shap.summary_plot(shap_values, test[features], show=True)

# Топ драйверов одним числом (средний |SHAP|)
import numpy as np
imp = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
print(imp.head(10))

# Локально: объяснение одного игрока
i = int(np.argmax(p_lgb))  # самый "уходящий"
shap.force_plot(explainer.expected_value, shap_values[i], test[features].iloc[i],
                matplotlib=True)
```

**Что получилось:** ожидаемо наверху `recency_days` (давно не заходил — толкает к оттоку) и `trend_sessions` (падающая активность). Это совпадает со скрытой `decay`, которую мы заложили в генератор, — значит модель реконструировала истинную причину, а не шум. Локальный график показывает конкретному игроку: «recency 21 день и падающий тренд дают +0.4 к вероятности оттока».

!!! tip "SHAP как sanity-check утечки"

    Если в топе важности вдруг оказывается технический признак вроде `player_id` или фича, которой по логике не должно быть видно на момент cutoff — это сигнал утечки, а не открытие. SHAP-summary стоит смотреть не только ради интерпретации, но и как детектор «слишком хорошей» фичи.

### Шаг 9: Явная проверка на утечки

**Зачем.** Утечка — причина №1 моделей, которые блестят на валидации и проваливаются в проде. Пройдёмся по трём типам осознанно.

**1. Временная утечка.** Уже закрыта временным сплитом (Шаг 2) и фильтром `ts < cutoff` в фичах. Проверка: метрика на test не должна резко превышать кросс-валидацию внутри train. Аномально высокий AUC (>0.95) на поведенческих данных — почти всегда утечка.

**2. Утечка через препроцессинг.** Закрыта pipeline'ом (Шаг 4): scaler/энкодеры учатся только на train fold. Никаких `fit` на полном датасете до сплита.

**3. Утечка через группы.** Один игрок не должен «обучать» модель сам на себя. Поскольку train и test у нас на разных временных окнах, базовая защита есть. Для строгой оценки внутри одного окна используем **GroupKFold по `player_id`**:

```python
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import numpy as np

gkf = GroupKFold(n_splits=5)
X, y, groups = train[features], y_train, train.player_id
aps = []
for tr_idx, va_idx in gkf.split(X, y, groups):
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx],
                      categorical_feature=cat_cols)
    m = lgb.train(params, dtr, num_boost_round=model.best_iteration or 500)
    p = m.predict(X.iloc[va_idx])
    aps.append(average_precision_score(y.iloc[va_idx], p))
print(f"GroupKFold PR-AUC: {np.mean(aps):.3f} ± {np.std(aps):.3f}")
```

**Что получилось:** если GroupKFold-метрика близка к test-метрике из временного сплита — модель стабильна, утечки групп нет. Большой разрыв (CV сильно выше test) — сигнал, что что-то протекает через время.

!!! question "Проверь себя"

    1. Чем временная утечка отличается от утечки через группы?
    2. Зачем GroupKFold по `player_id`, если уже есть временной сплит?
    3. Ты видишь test PR-AUC 0.97 на поведенческих данных. Радоваться?

??? success "Ответы"

    1. Временная — фичи посчитаны с использованием данных из будущего (после cutoff). Групповая — строки одного игрока попали и в train, и в valid внутри одного окна, и модель запомнила игрока, а не паттерн.
    2. Временной сплит защищает от заглядывания в будущее, но при кросс-валидации **внутри** train игрок может оказаться в обоих фолдах — GroupKFold это исключает, давая более честную оценку обобщения на новых игроков.
    3. Скорее насторожиться. На реальном поведении 0.97 почти всегда означает утечку — ищи фичу, которая косвенно знает таргет (например, посчитана без фильтра по cutoff).

### Шаг 10: Сериализация артефакта

**Зачем.** Модель без сохранённого порога и списка фич — наполовину готовый артефакт. Кладём модель, порог, список фич и метаданные вместе.

```python
import joblib, json, datetime

artifact = {
    "model": model,
    "features": features,
    "cat_cols": cat_cols,
    "threshold": float(best_t),
    "churn_n_days": CHURN_N_DAYS,
    "cutoff_train": str(CUTOFF_TRAIN.date()),
    "metrics": {"pr_auc": float(average_precision_score(y_test, p_lgb))},
    "trained_at": datetime.datetime.utcnow().isoformat(),
}
joblib.dump(artifact, "churn_model.pkl")
print("saved", artifact["metrics"], "threshold", artifact["threshold"])
```

**Что получилось:** один файл `churn_model.pkl`, который можно отдать в сервис скоринга (см. воркшоп W8 про деплой и мониторинг дрейфа). Порог и `churn_n_days` едут вместе с моделью — без них предсказание не интерпретируется.

## Типичные ошибки

- **Фичи посчитаны на всём логе, включая outcome-окно.** Классика. Всегда фильтруй `events.ts < cutoff` первой строкой генератора фич. Симптом — нереально высокий AUC.
- **Случайный train/test split.** Для churn нужен временной. Перемешивание даёт оптимистичную метрику и провал в проде.
- **Accuracy как метрика.** При 20% оттока бесполезна. PR-AUC + бизнес-порог.
- **Порог 0.5 по умолчанию.** Игнорирует асимметрию стоимости ошибок. Подбирай по матрице FN/FP.
- **Scaler/encoder обучен до сплита.** Утечка через препроцессинг. Только pipeline, только fit на train.
- **Senior-уровень: дрейф определения churn.** N=30 на этапе обучения и N=14 в проде — модель учили предсказывать одно, спрашивают другое. Зафиксируй `churn_n_days` в артефакте.
- **Senior-уровень: утечка через будущие справочники.** Если `country`/сегмент игрока обновился ПОСЛЕ cutoff (например, VIP-статус присвоен в outcome-окне) — это утечка статики. Бери атрибуты на момент cutoff, а не текущие.
- **Senior-уровень: непересмотр модели при сезонности.** Праздники, крупные турниры сдвигают базовую активность — модель, обученная на «тихом» периоде, переоценит отток в высокий сезон. Логируй распределение фич и сравнивай (дрейф, W8).
- **Игнор группового лика при повторных cutoff.** Если строишь несколько cutoff'ов для увеличения выборки, один игрок попадает много раз — без GroupKFold кросс-валидация наврёт.

!!! tip "AI-копилот в этом воркшопе"

    Нейросеть сильно ускорит рутину: генерацию синтетики, синтаксис оконных агрегатов pandas, обвязку SHAP-графиков, формулировку `classification_report`. Проси у неё каркас `make_features` и параметры LightGBM — это экономит часы.

    Где копилот подведёт именно здесь: он почти всегда предложит `train_test_split(shuffle=True)` и `accuracy`/`roc_auc` по умолчанию — то есть ровно те две ошибки, которые губят churn-модели. Он не знает твоей матрицы стоимости и предложит порог 0.5. И он не заметит временную утечку, если ты не описал ему границу cutoff явно. Решения про **определение таргета, временной сплит и бизнес-порог** держи на себе — это инженерия, которую модель за тебя не примет. Используй ИИ для кода, не для методологии.

## Критерий готовности

- [ ] Churn определён операционно: зафиксированы `N`, `cutoff`, `outcome window`, кандидаты отфильтрованы по активности до cutoff.
- [ ] Сплит временной (train на раннем cutoff, test на позднем), не случайный, и ты можешь объяснить почему.
- [ ] Все фичи считаются строго по `events.ts < cutoff`; проверено отсутствие будущих данных.
- [ ] Есть два baseline (правило + логистическая), LightGBM их превосходит по PR-AUC.
- [ ] Основная метрика — PR-AUC, на графике нарисована базовая линия = доля оттока.
- [ ] Порог выбран по матрице стоимости FN/FP, а не 0.5; есть кривая стоимости от порога.
- [ ] SHAP-summary построен, топ-драйверы оттока названы и осмыслены.
- [ ] Проверены три типа утечки; GroupKFold-метрика согласуется с временным test.
- [ ] Артефакт `churn_model.pkl` содержит модель, фичи, порог и `churn_n_days`.

## Развитие

1. **Несколько cutoff'ов** — собери обучающую выборку по 4-5 датам cutoff (с шагом 2 недели), увеличив объём и устойчивость; обязательно GroupKFold по `player_id`.
2. **Калибровка вероятностей** — `CalibratedClassifierCV` или isotonic: бустинг даёт «уверенные» вероятности, для расчёта ожидаемых потерь они должны быть калиброванными.
3. **Uplift-моделирование** — предсказывай не «уйдёт ли», а «изменит ли удержание поведение»; не все уходящие реагируют на бонус, часть уйдёт всё равно.
4. **Time-to-churn вместо бинарной метки** — survival-анализ (Cox, lifelines): когда уйдёт, а не только уйдёт ли.
5. **Деплой и мониторинг дрейфа** — оберни артефакт в FastAPI-сервис, добавь логирование распределения фич и алерт на дрейф (см. воркшоп W8).

## Что ты закрепил

Этот воркшоп связал воедино всю **Фазу 4**: основы ML и валидацию (M13), классификацию и градиентный бустинг (M14), feature engineering с оконными агрегатами, лагами и recency (M16), и — самое важное — подводные камни и интерпретацию (M17): три типа утечки, выбор метрики под дисбаланс, бизнес-порог и SHAP.

Главный навык, который ты унёс, — не `model.fit`, а **дисциплина честности**: операционно определить таргет, отделить прошлое от будущего во времени, не дать данным протечь и измерить модель так, чтобы цифра на валидации совпала с пользой в проде. Это то, что отличает ML-инженера от человека, который умеет звать sklearn.
