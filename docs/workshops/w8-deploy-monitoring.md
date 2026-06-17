# W8 · Деплой модели как API + мониторинг дрейфа

<span class="lecture-meta">Воркшоп · ориентир 7-11 ч · Продвинутый</span>

## Что ты построишь

Ты возьмёшь обученную churn-модель и доведёшь её до **прод-подобного состояния**: не «работает у меня в ноутбуке», а сервис, который принимает HTTP-запрос, валидирует вход, отдаёт предсказание, логирует каждый вызов и сам сообщает, когда боевые данные «уехали» от обучающих.

На выходе четыре артефакта:

1. **FastAPI-сервис** с эндпоинтами `/predict`, `/health` и `/version`. Вход строго валидируется через pydantic, ответ детерминирован, модель грузится один раз при старте.
2. **Docker-образ** этого сервиса, который запускается одной командой `docker run` и работает одинаково на твоём ноутбуке и на сервере.
3. **Структурированный лог предсказаний** (JSON Lines): каждый запрос — строка с входными признаками, вероятностью, версией модели и таймстампом. Это сырьё для мониторинга.
4. **Скрипт-монитор дрейфа** `drift_monitor.py`: считает PSI и KS-тест между обучающим распределением признаков и тем, что реально приходит в прод, и поднимает алерт + триггер на переобучение, когда дрейф превышает порог.

Это самый «инженерный» воркшоп курса. Здесь ценность не в новой модели, а в том, что отделяет ML-проект от ML-продукта: воспроизводимость, контракт данных на входе, наблюдаемость и автоматическая реакция на деградацию.

!!! info "Почему деплой и мониторинг — отдельный навык"

    Модель в проде живёт в среднем недели-месяцы до того, как её качество начинает падать. Падает оно тихо: метрики на старом тесте остаются прекрасными, а на свежем трафике AUC просел, потому что игроки, гео-микс или продукт изменились. Без мониторинга дрейфа ты узнаёшь об этом из жалоб бизнеса через квартал. С мониторингом — из алерта в тот же день. Это разница между «модель сломалась и мы не заметили» и «модель деградирует, переобучаем по триггеру».

## Предпосылки

Основной модуль — **M22 (MLOps)**: сериализация моделей, упаковка сервиса, контейнеризация, концепции дрейфа данных и концепт-дрейфа, мониторинг в проде.

Полезно, но не обязательно держать в голове:

- **W3 — churn-модель**: если ты делал воркшоп W3, бери готовый `churn_model.pkl` оттуда. Если нет — мы обучим простую модель прямо здесь за пять минут, воркшоп самодостаточен.
- **M14 — supervised ML**: понимать, что такое `predict_proba` и почему порог решения — отдельная бизнес-настройка.
- **M17 — подводные камни**: дрейф данных и концепт-дрейф концептуально оттуда.

Окружение через **uv** (стандарт 2026):

```bash
uv init churn-api && cd churn-api
uv add fastapi "uvicorn[standard]" pydantic joblib scikit-learn lightgbm pandas numpy scipy
uv add httpx          # клиент для прогона нагрузки и тестов
uv add --dev pytest
```

!!! tip "Почему uv и фиксация версий критичны именно для деплоя"

    В деплое воспроизводимость окружения — не удобство, а корректность. Модель, обученная на `scikit-learn==1.5`, может не загрузиться на `1.7` или загрузиться, но считать иначе. `uv.lock` фиксирует точные версии, а Docker-образ замораживает их навсегда. Любой `docker run` через год даст ровно тот же ответ на тот же вход. Это и есть «прод-подобное состояние».

Версии стека, на которых собран воркшоп: Python 3.12, FastAPI 0.115+, uvicorn 0.34+, pydantic 2.x, joblib 1.4+, scikit-learn 1.5+, lightgbm 4.x.

## Данные

Нам нужны два датасета с **одинаковой схемой признаков**: обучающий (эталон распределений, baseline) и «боевой» поток (то, что приходит в прод). Чтобы мониторинг дрейфа было что показывать, боевой поток мы сознательно сделаем частично сдвинутым — как если бы маркетинг залил новый гео-трафик и поведение игроков поехало.

Признаки churn-модели для гемблинг-игрока (окно наблюдения 30 дней):

- `deposits_30d` — число депозитов;
- `deposit_sum_30d` — суммарный депозит, USD;
- `bets_30d` — число ставок;
- `sessions_30d` — число игровых сессий;
- `avg_session_min` — средняя длительность сессии, мин;
- `days_since_last_login` — recency: сколько дней назад заходил;
- `bonus_ratio` — доля оборота, сделанная на бонусные деньги (0..1);
- `withdrawal_30d` — число выводов.

Таргет `churn` — ушёл ли игрок в отток в следующие 30 дней.

### Генератор синтетических данных

ЗАЧЕМ. Один генератор с параметром `regime` отдаёт либо обучающее распределение (`base`), либо сдвинутое боевое (`drifted`). Сдвиг реалистичный: меньше депозитов, выше recency, выше доля бонусов — типичная картина при заливе дешёвого трафика. Так у мониторинга будет настоящий дрейф, а не шум. Seed фиксирован, чтобы результат повторялся у любого.

```python
# gen_data.py
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
FEATURES = [
    "deposits_30d", "deposit_sum_30d", "bets_30d", "sessions_30d",
    "avg_session_min", "days_since_last_login", "bonus_ratio", "withdrawal_30d",
]

def _sample(n: int, regime: str, rng: np.random.Generator) -> pd.DataFrame:
    if regime == "base":
        deposits = rng.poisson(4, n)
        deposit_sum = rng.gamma(2.0, 60, n)
        recency = rng.exponential(6, n)
        bonus = rng.beta(2, 6, n)
    elif regime == "drifted":
        # новый дешёвый трафик: реже платят, дольше не заходят, больше на бонусах
        deposits = rng.poisson(2.2, n)
        deposit_sum = rng.gamma(1.6, 45, n)
        recency = rng.exponential(11, n)
        bonus = rng.beta(4, 4, n)
    else:
        raise ValueError(regime)

    sessions = rng.poisson(3 + deposits * 0.8, n)
    bets = rng.poisson(20 + bets_lambda(deposits), n)
    avg_session = np.clip(rng.normal(14, 6, n), 1, None)
    withdrawals = rng.poisson(np.clip(deposit_sum / 400, 0, 5), n)

    df = pd.DataFrame({
        "deposits_30d": deposits,
        "deposit_sum_30d": deposit_sum.round(2),
        "bets_30d": bets,
        "sessions_30d": sessions,
        "avg_session_min": avg_session.round(1),
        "days_since_last_login": recency.round(1),
        "bonus_ratio": np.clip(bonus, 0, 1).round(3),
        "withdrawal_30d": withdrawals,
    })
    return df

def bets_lambda(deposits):
    return deposits * 6

def _make_target(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    # churn выше при высокой recency, низких депозитах и высокой доле бонусов
    z = (
        -2.1
        + 0.16 * df["days_since_last_login"]
        - 0.28 * df["deposits_30d"]
        - 0.004 * df["deposit_sum_30d"]
        + 1.4 * df["bonus_ratio"]
        - 0.02 * df["sessions_30d"]
    )
    p = 1 / (1 + np.exp(-z))
    return (rng.random(len(df)) < p).astype(int)

def make_dataset(n: int, regime: str = "base", seed: int = 42, with_target: bool = True):
    rng = np.random.default_rng(seed)
    df = _sample(n, regime, rng)
    if with_target:
        df["churn"] = _make_target(df, rng)
    return df

if __name__ == "__main__":
    train = make_dataset(8000, regime="base", seed=42)
    train.to_parquet("train.parquet", index=False)

    # боевой поток без таргета: 70% «нормального» + 30% сдвинутого
    live_base = make_dataset(2100, regime="base", seed=7, with_target=False)
    live_drift = make_dataset(900, regime="drifted", seed=8, with_target=False)
    live = pd.concat([live_base, live_drift], ignore_index=True).sample(frac=1, random_state=1)
    live.to_parquet("live_stream.parquet", index=False)

    print("train churn rate:", round(train["churn"].mean(), 3))
    print("train:", train.shape, "live:", live.shape)
```

ЧТО ПОЛУЧИЛОСЬ. Два файла: `train.parquet` (8000 строк с таргетом) и `live_stream.parquet` (3000 строк без таргета, из которых ~30% сдвинуты). Доля churn в train около 0.3-0.35 — рабочий дисбаланс.

!!! note "Чем заменить на реальные данные"

    Подставь любой churn-датасет с числовыми признаками. На Kaggle подходят Telco Customer Churn и Bank Customer Churn — структура та же (поведенческие признаки + бинарный таргет). Для гемблинг-специфики реальный лог событий из W1/W3 агрегируешь в те же оконные признаки. Важно одно: baseline (обучающее распределение) и live-поток должны иметь идентичный набор колонок — на этом держится весь мониторинг.

```bash
uv run python gen_data.py
```

## Ход работы

Маршрут такой: обучаем и сериализуем модель → собираем артефакт с метаданными → поднимаем FastAPI с pydantic-валидацией → добавляем логирование предсказаний → пакуем в Docker → пишем монитор дрейфа на PSI + KS → задаём триггер на переобучение.

### Шаг 1: Обучить и сериализовать модель

ЗАЧЕМ. Нам нужен артефакт, а не просто обученный объект в памяти. Сериализуем не «голую» модель, а **бандл**: модель + список признаков в правильном порядке + baseline-статистики (для мониторинга) + метаданные (версия, дата, метрика). Порядок признаков критичен: если в проде колонки придут в другом порядке, scikit-learn молча посчитает мусор. Бандл — это контракт.

```python
# train_model.py
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from gen_data import make_dataset, FEATURES

def main():
    df = make_dataset(8000, regime="base", seed=42)
    X, y = df[FEATURES], df["churn"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # baseline-статистики по обучающей выборке — эталон для мониторинга дрейфа
    baseline = {
        f: {
            "quantiles": np.quantile(X_tr[f], np.linspace(0, 1, 11)).tolist(),
            "mean": float(X_tr[f].mean()),
            "std": float(X_tr[f].std()),
            "sample": X_tr[f].sample(2000, random_state=0).tolist(),  # для KS-теста
        }
        for f in FEATURES
    }

    bundle = {
        "model": model,
        "features": FEATURES,
        "baseline": baseline,
        "meta": {
            "version": "1.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "val_auc": round(float(auc), 4),
            "churn_rate": round(float(y.mean()), 4),
            "threshold": 0.5,
        },
    }
    joblib.dump(bundle, "churn_model.pkl")
    print(json.dumps(bundle["meta"], indent=2))

if __name__ == "__main__":
    main()
```

ЧТО ПОЛУЧИЛОСЬ. Файл `churn_model.pkl` (~1-2 МБ). Val AUC около 0.80-0.85. В бандле всё, что нужно сервису и монитору: модель, порядок признаков, эталонные распределения и версия.

```bash
uv run python train_model.py
```

!!! warning "Не сериализуй голую модель через pickle напрямую"

    `pickle.dump(model)` хранит ссылки на классы из той версии библиотеки, что была при сохранении. `joblib` эффективнее для numpy-массивов внутри деревьев и стандартен для scikit-стека. Но главный риск — версии библиотек: всегда фиксируй их (uv.lock + Docker), иначе загрузка артефакта на другой версии либо упадёт, либо тихо изменит предсказания. Бандл с полем `version` в meta позволяет отследить, какая модель отвечала на конкретный запрос.

    Второй риск — безопасность: и `pickle`, и `joblib` при загрузке исполняют произвольный код, зашитый в артефакт, поэтому грузить недоверенный `.pkl` нельзя (arbitrary code execution). Загружай только бандлы, собранные тобой и сохранённые в доверенном хранилище. Для обмена моделями между командами/наружу используй формат `skops`: он сериализует scikit-стек безопасно, без исполнения произвольного кода при загрузке.

!!! question "Проверь себя"

    1. Почему мы кладём `features` (список колонок) внутрь артефакта, а не хардкодим в сервисе?
    2. Зачем в baseline хранится и `quantiles`, и `sample` — нельзя ли одним обойтись?
    3. Что сломается, если обучить на scikit-learn 1.5, а грузить на 1.7 без фиксации версий?

??? success "Ответы"

    1. Список признаков в порядке обучения — часть контракта модели. Хардкод в сервисе разъедется с моделью при первом же переобучении с новым признаком. Храня его в бандле, мы гарантируем, что сервис строит матрицу ровно в том порядке, в каком модель училась.
    2. `quantiles` нужны для PSI (бинуем по эталонным границам), `sample` — для KS-теста (он сравнивает два набора значений напрямую). Это два разных теста дрейфа, им нужны разные представления baseline.
    3. В лучшем случае — предупреждение о несовместимости и отказ грузиться. В худшем — модель загрузится, но из-за изменений во внутреннем формате деревьев или дефолтах посчитает иначе, и ты получишь тихо неверные предсказания. Поэтому версии замораживают.

### Шаг 2: Контракт входных данных через pydantic

ЗАЧЕМ. Прод-API не доверяет входу. Клиент пришлёт строку вместо числа, отрицательный депозит, `bonus_ratio = 5` или вообще пропустит поле. Без валидации это либо упадёт где-то в недрах LightGBM с непонятной 500-й, либо посчитает мусор. pydantic v2 описывает схему декларативно: типы, границы, обязательность. Невалидный запрос отбивается 422-й с понятным сообщением **до** того, как доберётся до модели.

```python
# schemas.py
from pydantic import BaseModel, Field, ConfigDict

class PlayerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")  # лишние поля = ошибка, не молчим

    deposits_30d: int = Field(ge=0, le=1000)
    deposit_sum_30d: float = Field(ge=0, le=1_000_000)
    bets_30d: int = Field(ge=0, le=100_000)
    sessions_30d: int = Field(ge=0, le=10_000)
    avg_session_min: float = Field(ge=0, le=1440)
    days_since_last_login: float = Field(ge=0, le=3650)
    bonus_ratio: float = Field(ge=0, le=1)
    withdrawal_30d: int = Field(ge=0, le=1000)

class PredictResponse(BaseModel):
    churn_probability: float
    churn_label: int
    threshold: float
    model_version: str
```

ЧТО ПОЛУЧИЛОСЬ. Схема входа с бизнес-границами и `extra="forbid"` — опечатка в имени поля теперь явная ошибка, а не тихо проигнорированное поле. Схема ответа фиксирует контракт наружу.

!!! tip "Границы Field — это тоже контроль качества данных"

    `bonus_ratio` физически в [0, 1], `days_since_last_login` не может быть 5000. Эти границы — первый барьер на пути «грязного» входа. Они ловят баги интеграции на стороне клиента и заодно отсекают абсурдные значения, на которых модель не училась и поведёт себя непредсказуемо. Но границы не ловят дрейф: значение может быть валидным и всё равно «уехавшим» от обучающего распределения — этим займётся монитор на шаге 6.

### Шаг 3: FastAPI-сервис с эндпоинтом predict

ЗАЧЕМ. Модель грузим **один раз** при старте через lifespan, а не на каждый запрос — иначе на каждом вызове будет лишние сотни миллисекунд на чтение файла. Эндпоинт `/predict` принимает валидированный объект, собирает матрицу строго в порядке `features` из бандла, считает вероятность и применяет порог. Плюс `/health` (жив ли сервис, для оркестратора) и `/version` (какая модель отвечает).

```python
# app.py
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI

from schemas import PlayerFeatures, PredictResponse
from prediction_logger import log_prediction

STATE: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load("churn_model.pkl")
    STATE["model"] = bundle["model"]
    STATE["features"] = bundle["features"]
    STATE["meta"] = bundle["meta"]
    logging.info("model loaded: %s", bundle["meta"]["version"])
    yield
    STATE.clear()

app = FastAPI(title="Churn API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}

@app.get("/version")
def version():
    return STATE["meta"]

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PlayerFeatures):
    features = STATE["features"]
    row = payload.model_dump()
    X = pd.DataFrame([[row[f] for f in features]], columns=features)

    proba = float(STATE["model"].predict_proba(X)[0, 1])
    threshold = STATE["meta"]["threshold"]
    label = int(proba >= threshold)

    log_prediction(row, proba, label, STATE["meta"]["version"])
    return PredictResponse(
        churn_probability=round(proba, 4),
        churn_label=label,
        threshold=threshold,
        model_version=STATE["meta"]["version"],
    )
```

ЧТО ПОЛУЧИЛОСЬ. Рабочий сервис. Поднимаем и проверяем:

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{"deposits_30d":1,"deposit_sum_30d":40.0,"bets_30d":12,"sessions_30d":2,"avg_session_min":6.5,"days_since_last_login":25.0,"bonus_ratio":0.7,"withdrawal_30d":0}'
```

Ответ примерно: `{"churn_probability":0.78,"churn_label":1,"threshold":0.5,"model_version":"1.0.0"}`. Документация и интерактивный тест — на `http://localhost:8000/docs` (FastAPI генерирует Swagger из pydantic-схем автоматически).

!!! warning "Порядок признаков — самая частая тихая ошибка"

    Мы строим `X` явным перебором `row[f] for f in features`. Если вместо этого сделать `pd.DataFrame([row])`, порядок колонок определит порядок ключей в словаре, а не порядок обучения. LightGBM не проверяет имена — он берёт позиции. Перепутанные местами `bets_30d` и `bonus_ratio` дадут валидный ответ с абсолютно неверной вероятностью, и ты этого не заметишь. Всегда выстраивай матрицу по сохранённому списку признаков.

### Шаг 4: Логирование предсказаний

ЗАЧЕМ. Без лога предсказаний мониторинг дрейфа невозможен в принципе — нечего сравнивать с baseline. Пишем каждый вызов в **JSON Lines** (одна строка = один JSON-объект): вход, вероятность, лейбл, версия модели, UTC-таймстамп. JSONL удобен тем, что дописывается атомарно построчно и легко читается обратно в pandas. В реальном проде это ушло бы в Kafka / S3 / лог-коллектор, но контракт строки тот же.

```python
# prediction_logger.py
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("predictions.jsonl")

def log_prediction(features: dict, proba: float, label: int, version: str) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "churn_probability": round(proba, 6),
        "churn_label": label,
        "features": features,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

ЧТО ПОЛУЧИЛОСЬ. Каждый `/predict` дописывает строку в `predictions.jsonl`. Прогоним через сервис весь боевой поток, чтобы накопить данные для монитора:

```python
# replay_live.py
import httpx
import pandas as pd
from gen_data import FEATURES

live = pd.read_parquet("live_stream.parquet")
with httpx.Client(base_url="http://localhost:8000", timeout=10) as cli:
    for _, r in live.iterrows():
        payload = {f: (int(r[f]) if "int" in str(live[f].dtype) else float(r[f])) for f in FEATURES}
        cli.post("/predict", json=payload).raise_for_status()
print("sent", len(live), "predictions")
```

```bash
uv run python replay_live.py
wc -l predictions.jsonl   # ~3000 строк
```

!!! note "Что именно логировать в проде"

    Минимум для мониторинга: входные признаки + выход + версия модели + таймстамп. Желательно ещё request_id (трейсинг) и, если приходит позже, фактический исход (ground truth) — он нужен для мониторинга **качества** (падение AUC), а не только дрейфа входа. Чего не логировать без нужды: PII в сыром виде. В гемблинге player_id хешируй или храни отдельно с контролем доступа.

!!! question "Проверь себя"

    1. Почему дрейф нельзя обнаружить одними pydantic-границами из шага 2?
    2. Зачем в лог пишется `model_version`, если модель одна?
    3. Чем JSONL удобнее, чем один большой JSON-массив, для лога, который дописывается онлайн?

??? success "Ответы"

    1. Границы проверяют, что значение допустимо в принципе (например, bonus_ratio в [0,1]). Дрейф — это смещение *распределения* валидных значений: средний bonus_ratio был 0.25, стал 0.5. Каждое значение по-прежнему валидно, но статистика входа уехала. Это ловит только сравнение распределений (PSI/KS), не границы.
    2. Чтобы при будущем переобучении (модель 1.1, 1.2) знать, какая версия отвечала на конкретный запрос. Без этого нельзя разделить деградацию старой модели и поведение новой, а при инциденте — понять, что именно катилось в проде.
    3. JSONL дописывается одной строкой в конец файла — это атомарно и не требует читать/перезаписывать весь файл. JSON-массив пришлось бы каждый раз десериализовать целиком, добавлять элемент и писать обратно — медленно и небезопасно при параллельных записях.

### Шаг 5: Dockerfile и запуск в контейнере

ЗАЧЕМ. Контейнер — это способ заморозить ОС, Python, версии библиотек и саму модель в один неизменяемый артефакт. «Работает у меня» превращается в «работает везде одинаково». Используем многоступенчатую сборку: ставим зависимости через uv, копируем код и модель, запускаем uvicorn. `--frozen` гарантирует установку ровно из `uv.lock`.

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app.py schemas.py prediction_logger.py churn_model.pkl ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-dev", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`, чтобы не тащить в образ лишнее (в том числе локальный лог предсказаний):

```text
.venv
__pycache__
*.parquet
predictions.jsonl
.git
.pytest_cache
```

ЧТО ПОЛУЧИЛОСЬ. Сборка и запуск:

```bash
docker build -t churn-api:1.0.0 .
docker run --rm -p 8000:8000 churn-api:1.0.0
curl -s http://localhost:8000/health
```

`/health` отвечает `{"status":"ok","model_loaded":true}` из контейнера. Тот же образ можно запустить на сервере / Railway / в Kubernetes без изменений.

!!! warning "Лог предсказаний внутри контейнера эфемерен"

    `predictions.jsonl` пишется в файловую систему контейнера и **исчезает при его пересоздании**. Для воркшопа это ок (монтируй volume: `-v $(pwd)/data:/app/data` и пиши лог туда). В реальном проде лог должен уходить наружу — в внешнее хранилище (S3), очередь (Kafka) или лог-коллектор, а не жить в контейнере. Мониторинг дрейфа читает данные из этого внешнего стора, а не из локального файла на одном поде.

### Шаг 6: Монитор дрейфа — PSI и KS-тест

ЗАЧЕМ. Это ядро воркшопа. Мы сравниваем распределение каждого признака в **боевом потоке** (из `predictions.jsonl`) с **обучающим эталоном** (baseline из бандла) и численно отвечаем: уехало или нет?

Два дополняющих инструмента:

**PSI (Population Stability Index)** — индустриальный стандарт (родом из кредитного скоринга). Бьём оба распределения на бины по эталонным квантилям и суммируем по бинам:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i)\,\ln\frac{a_i}{e_i}$$

где $e_i$ — доля baseline в бине $i$, $a_i$ — доля боевых данных в том же бине. Общепринятые пороги: $\text{PSI} < 0.1$ — стабильно; $0.1 \le \text{PSI} < 0.25$ — умеренный сдвиг, наблюдаем; $\text{PSI} \ge 0.25$ — значимый дрейф, действуем.

**KS-тест (Колмогоров — Смирнов)** — статистический тест на равенство двух непрерывных распределений. Даёт статистику $D$ (максимум разницы эмпирических CDF) и p-value. Маленький p-value (< 0.05) — распределения различаются значимо. PSI говорит «насколько сильно», KS — «статистически значимо ли вообще». Вместе они устойчивее, чем поодиночке.

```python
# drift_monitor.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6

def population_stability_index(baseline_vals, live_vals, edges) -> float:
    edges = np.array(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf  # ловим хвосты за пределами обучающего диапазона
    e_counts, _ = np.histogram(baseline_vals, bins=edges)
    a_counts, _ = np.histogram(live_vals, bins=edges)
    e = e_counts / max(e_counts.sum(), 1)
    a = a_counts / max(a_counts.sum(), 1)
    e = np.clip(e, EPS, None)
    a = np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))

def load_live(path="predictions.jsonl") -> pd.DataFrame:
    rows = [json.loads(line)["features"] for line in Path(path).read_text().splitlines()]
    return pd.DataFrame(rows)

def run_drift_report(bundle_path="churn_model.pkl", live_path="predictions.jsonl"):
    bundle = joblib.load(bundle_path)
    features, baseline = bundle["features"], bundle["baseline"]
    live = load_live(live_path)

    report = []
    for f in features:
        b = baseline[f]
        live_vals = live[f].values
        base_sample = np.array(b["sample"])

        psi = population_stability_index(base_sample, live_vals, b["quantiles"])
        ks_stat, ks_p = ks_2samp(base_sample, live_vals)

        if psi >= 0.25:
            status = "DRIFT"
        elif psi >= 0.1:
            status = "WATCH"
        else:
            status = "OK"

        report.append({
            "feature": f,
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
            "base_mean": round(b["mean"], 3),
            "live_mean": round(float(live_vals.mean()), 3),
            "status": status,
        })
    return pd.DataFrame(report).sort_values("psi", ascending=False)

if __name__ == "__main__":
    rep = run_drift_report()
    pd.set_option("display.width", 160)
    print(rep.to_string(index=False))
```

ЧТО ПОЛУЧИЛОСЬ:

```bash
uv run python drift_monitor.py
```

Таблица вроде такой (числа зависят от seed):

```text
            feature    psi  ks_stat  ks_pvalue  base_mean  live_mean status
       bonus_ratio  0.31    0.214   0.000000      0.249      0.318  DRIFT
days_since_last_login 0.22  0.178   0.000000      6.05       8.11   WATCH
       deposits_30d  0.18   0.131   0.000001      4.01       3.41   WATCH
   deposit_sum_30d   0.09   0.072   0.003100      120.1      108.4     OK
          bets_30d   0.03   0.041   0.210000      44.2       42.8     OK
...
```

Видно ровно тот сдвиг, что мы зашили в `drifted`: `bonus_ratio`, recency и депозиты уехали, остальное стабильно. PSI ранжирует признаки по силе дрейфа, KS подтверждает значимость.

!!! tip "Почему PSI и KS, а не один из них"

    KS чувствителен к размеру выборки: на десятках тысяч строк он даст p-value < 0.05 при мизерном, бизнес-незначимом сдвиге — статистически значимо, но практически неважно. PSI — мера величины эффекта, она не раздувается от объёма данных, но в ней нет понятия «значимости». Связка покрывает оба вопроса: KS — «отличие реально или это шум выборки», PSI — «насколько отличие большое, чтобы реагировать». Для категориальных признаков KS не работает — там берут PSI или хи-квадрат.

!!! question "Проверь себя"

    1. Почему в PSI крайние границы бинов мы заменяем на $\pm\infty$?
    2. Что показывает дрейф входных признаков, а что он НЕ показывает про качество модели?
    3. На выборке 500k строк KS даёт p-value = 0.001 при PSI = 0.02. Реагировать?

??? success "Ответы"

    1. Боевые данные могут содержать значения за пределами обучающего диапазона (новый трафик принёс игроков с recency 60 дней, которых в обучении не было). Без бесконечных хвостов такие значения выпали бы за крайние бины и потерялись, занизив PSI. Открытые границы гарантируют, что весь боевой поток попадёт в гистограмму.
    2. Дрейф входа показывает, что распределение признаков сместилось — модель работает на данных, отличных от обучающих, и могла потерять точность. Он НЕ измеряет качество напрямую: AUC мог и не упасть (это зависит от того, сохранилась ли связь признак-таргет). Чтобы поймать падение качества, нужен ground truth и мониторинг метрики (concept drift), а не только дрейф входа.
    3. Нет. PSI = 0.02 — распределение практически не сдвинулось, эффект ничтожен. Маленький p-value здесь — артефакт огромной выборки, на которой KS реагирует на любую микроскопическую разницу. Решение принимаем по PSI (величина эффекта), KS на больших данных используем осторожно.

### Шаг 7: Правило-триггер на переобучение и алерт

ЗАЧЕМ. Отчёт по дрейфу бесполезен, если никто его не читает каждый день. Нужно **правило**, которое из таблицы делает бинарное решение «переобучать / нет» и шлёт алерт. Правило должно быть устойчивым к одиночному выбросу: триггерим не по одному признаку, а по агрегированному условию.

Политика триггера (разумный дефолт, настраивается под проект):

- **RETRAIN**, если хотя бы один **важный** признак в статусе `DRIFT` (PSI ≥ 0.25), ИЛИ если в статусе `WATCH`/`DRIFT` оказались ≥ 3 признаков одновременно (массовый сдвиг трафика).
- **WATCH** (наблюдаем, не переобучаем), если есть отдельные `WATCH` без важных `DRIFT`.
- **OK** иначе.

```python
# drift_trigger.py
import json
import sys

from drift_monitor import run_drift_report

IMPORTANT = {"bonus_ratio", "days_since_last_login", "deposits_30d", "deposit_sum_30d"}
PSI_DRIFT = 0.25

def decide(report) -> dict:
    drifted_important = report[(report.psi >= PSI_DRIFT) & (report.feature.isin(IMPORTANT))]
    n_shifted = int((report.status != "OK").sum())

    if len(drifted_important) > 0 or n_shifted >= 3:
        action = "RETRAIN"
    elif n_shifted > 0:
        action = "WATCH"
    else:
        action = "OK"

    return {
        "action": action,
        "drifted_important": drifted_important.feature.tolist(),
        "n_features_shifted": n_shifted,
        "max_psi_feature": report.iloc[0]["feature"],
        "max_psi": float(report.iloc[0]["psi"]),
    }

def send_alert(decision: dict) -> None:
    # заглушка: в проде — Slack/Telegram webhook или PagerDuty
    print("ALERT:", json.dumps(decision, ensure_ascii=False))

if __name__ == "__main__":
    rep = run_drift_report()
    decision = decide(rep)
    if decision["action"] in {"RETRAIN", "WATCH"}:
        send_alert(decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    # ненулевой код выхода для CI/cron, чтобы оркестратор поймал триггер
    sys.exit(1 if decision["action"] == "RETRAIN" else 0)
```

ЧТО ПОЛУЧИЛОСЬ:

```bash
uv run python drift_trigger.py
```

На нашем сдвинутом потоке вернётся `action: RETRAIN` (сработал `bonus_ratio` с PSI ≥ 0.25) и алерт в stdout. Ненулевой exit code позволяет повесить скрипт на cron / CI: если упал с кодом 1 — запускаем пайплайн переобучения (`train_model.py` на свежих данных → новый бандл → деплой версии 1.1.0).

!!! example "Как это замыкается в цикл MLOps"

    1. API логирует предсказания → 2. По расписанию (cron, ночью) `drift_trigger.py` читает накопленный лог → 3. PSI/KS считают дрейф → 4. Правило решает → 5. RETRAIN запускает переобучение на свежих данных с разметкой → 6. Новый бандл (версия +1) собирается в Docker → 7. Деплой, `/version` показывает новую версию. Получился замкнутый контур: модель сама сигналит о деградации и инициирует обновление. Это и есть «прод-подобное состояние» из цели воркшопа.

!!! tip "Не переобучай на каждый чих"

    Слишком чувствительный триггер (PSI ≥ 0.1 по любому признаку) приведёт к постоянным переобучениям на шуме — это дорого и дестабилизирует прод. Слишком грубый (только PSI ≥ 0.5) пропустит реальную деградацию. Дефолт PSI ≥ 0.25 по важным признакам + правило «≥ 3 сдвинутых» — рабочий компромисс. Калибруй пороги по истории: посмотри, при каком PSI исторически реально падал AUC на твоих данных, и привяжи триггер к этому.

## Типичные ошибки

- **Модель грузится на каждый запрос.** `joblib.load` внутри обработчика `/predict` добавляет сотни мс латентности на вызов. Грузи один раз в lifespan/startup и держи в состоянии приложения.
- **Перепутан порядок признаков.** Самая коварная ошибка: ответ валиден, но неверен. LightGBM/sklearn берут позиции колонок, а не имена. Всегда строй матрицу по сохранённому в бандле списку `features`.
- **Версии библиотек не зафиксированы.** Артефакт, обученный на одной версии scikit-learn/lightgbm, на другой либо не грузится, либо считает иначе. uv.lock + Docker — обязательны, не «на потом».
- **Лог предсказаний живёт в контейнере.** При рестарте пода всё стирается, мониторить нечего. Лог должен уходить во внешнее хранилище.
- **Мониторят только дрейф входа и думают, что покрыли качество.** Дрейф признаков ≠ падение метрики. Связь признак-таргет может смениться при стабильном входе (concept drift) — это ловится только ground truth и мониторингом AUC, а не PSI.
- **KS на огромных выборках принимают за сигнал.** На сотнях тысяч строк KS значим при ничтожном сдвиге. Решение по величине эффекта (PSI), а не по одному p-value.
- **PSI на категориальных как на числовых.** Квантильное бинование на категориях бессмысленно. Для категориальных — PSI по категориям как по бинам или хи-квадрат.
- **(Senior) Baseline зафиксирован навсегда и узкий.** Если эталон — это `train.sample(2000)` без сезонности, нормальная недельная/месячная цикличность будет ложно триггерить дрейф. Эталон должен покрывать естественную вариативность; сравнивай скользящим окном, а не с одним замёрзшим снимком.
- **(Senior) Нет защиты от пустого/частичного потока.** Если за сутки пришло 12 предсказаний, PSI на них — шум. Триггер должен требовать минимальный размер боевой выборки (например, ≥ 500) прежде чем выносить вердикт.

!!! tip "AI-копилот в этом воркшопе"

    Где нейросеть реально ускорит: генерация скелета FastAPI-приложения и pydantic-схем по списку признаков, написание Dockerfile под uv, обвязка httpx-клиента для прогона нагрузки, докстринги и pytest-тесты на эндпоинты. Это шаблонный код, который копилот пишет точно и быстро.

    Где подведёт именно здесь: (1) формула PSI — копилоты регулярно путают, что под логарифмом, и забывают про эпсилон и бесконечные хвосты бинов; проверяй вручную по определению. (2) Пороги PSI/KS и логику триггера он подгонит «правдоподобно», но без привязки к твоим данным — это бизнес-решение, не код. (3) Порядок признаков при сборке матрицы — копилот часто пишет `pd.DataFrame([dict])`, что молча ломает порядок; настаивай на явном переборе по сохранённому списку. (4) Различие дрейфа входа и концепт-дрейфа он смазывает — может предложить «мониторить дрейф» и считать вопрос качества закрытым. Держи это разделение в голове сам.

## Критерий готовности

- [ ] `train_model.py` обучает модель и сохраняет бандл `churn_model.pkl` с моделью, списком признаков, baseline и meta (версия, AUC).
- [ ] FastAPI поднимается, `/health` отвечает ok, `/version` отдаёт метаданные модели.
- [ ] `/predict` валидирует вход через pydantic (невалидный запрос → 422) и возвращает вероятность, лейбл, порог и версию.
- [ ] Матрица признаков строится строго по сохранённому порядку `features`.
- [ ] Каждый вызов `/predict` пишет строку в `predictions.jsonl` (вход + выход + версия + UTC-таймстамп).
- [ ] `docker build` собирает образ, `docker run` поднимает сервис, `/health` отвечает из контейнера.
- [ ] `drift_monitor.py` считает PSI и KS по каждому признаку и выдаёт таблицу со статусами OK/WATCH/DRIFT.
- [ ] На сдвинутом боевом потоке монитор корректно подсвечивает именно сдвинутые признаки (`bonus_ratio`, recency, депозиты).
- [ ] `drift_trigger.py` выносит решение RETRAIN/WATCH/OK, шлёт алерт и отдаёт ненулевой exit code на RETRAIN.

## Развитие

1. **Concept drift по ground truth.** Дотащи фактический исход (ушёл/не ушёл через 30 дней), считай rolling AUC/precision на свежих данных и триггерь переобучение по падению метрики, а не только по дрейфу входа.
2. **Готовый инструмент мониторинга.** Подключи Evidently или NannyML вместо самописного PSI — сравни их отчёты со своими цифрами, разберись, где расходятся и почему (другое бинование, бутстрап-доверительные интервалы).
3. **Метрики и дашборд.** Отдавай Prometheus-метрики (`/metrics`): латентность, RPS, доля churn_label=1, текущий max PSI. Построй Grafana-дашборд для наблюдаемости в реальном времени.
4. **Версионирование моделей и A/B.** Введи реестр моделей (MLflow), катай новую версию канареечно на 10% трафика, сравнивай метрики старой и новой перед полным переключением.
5. **CI/CD пайплайн.** GitHub Actions: на пуш — тесты эндпоинтов, сборка образа, пуш в registry; по cron — запуск `drift_trigger.py`, и при RETRAIN автоматический трен + деплой новой версии.

## Что ты закрепил

Ты прошёл путь от обученной модели до прод-подобного сервиса с самонаблюдением и связал воедино:

- **M22 (MLOps)** — сериализация артефакта-бандла, упаковка сервиса, контейнеризация, дрейф данных vs концепт-дрейф, мониторинг и автоматический триггер переобучения — стержень всего воркшопа.
- **M14 (supervised)** — `predict_proba` и порог решения как отдельная бизнес-настройка в контракте ответа.
- **M17 (подводные камни)** — дрейф распределений, разница между «значимо» (KS) и «велико» (PSI), почему дрейф входа не равен падению качества.
- **W3 (churn-модель)** — артефакт из W3 встаёт сюда без переделок: это показывает, что обучение и деплой — разные слои с чётким контрактом между ними.

Главный навык, который ты унёс: **модель в проде — это не файл, а контракт** (схема входа, порядок признаков, версия, baseline) плюс контур наблюдаемости, который ловит деградацию до того, как её заметит бизнес.
