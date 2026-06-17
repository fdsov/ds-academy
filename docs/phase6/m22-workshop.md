# Воркшоп M22 · От joblib до детектора дрейфа

<span class="lecture-meta">Воркшоп к модулю M22 · ориентир 5-7 ч</span>

## Что отрабатываем

В модуле M22 ты прошёл, что обученная модель — это артефакт, а не продукт. Этот воркшоп закрывает три центральных перехода из теории руками:

- Сериализация (M22.4): заморозить churn-модель в файл через `joblib` и `skops`, понять разницу в безопасности и зафиксировать версии библиотек против «версионной ловушки».
- Online-сервинг на FastAPI (M22.5): поднять REST-эндпоинт `/predict` с валидацией входов через Pydantic и измерением латентности по перцентилям p50/p95/p99 (а не среднему).
- Детекция дрейфа (M22.9): реализовать PSI и тест Колмогорова–Смирнова между референсным (train) и новым (прод) батчем, имитировать маркетинговую кампанию и поймать сдвиг порогом.

Артефакт на выходе: рабочий FastAPI-сервис `serve.py` плюс автономный скрипт мониторинга дрейфа `drift_monitor.py`, который сравнивает свежий батч с референсом и печатает алерт. Всё запускается локально через `uv`, данные синтетические с фиксированным seed.

Не делаем здесь: Docker, MLflow registry, DVC — это отдельные большие темы модуля. Фокус строго на «модель → API → мониторинг».

## Данные

Никаких внешних загрузок. Генерируем синтетический датасет оттока игроков с теми же признаками, что в коде модуля M22.12: `deposits_30d`, `bets_30d`, `avg_bet`, `days_since_last_session`. Связь признаков с оттоком зашита явно, чтобы модель училась на сигнале, а не на шуме.

```python
# gen_data.py
import numpy as np
import pandas as pd

def make_players(n: int, seed: int, campaign: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # campaign=True имитирует маркетинговую кампанию: новая аудитория
    # с другим средним депозитом — это и есть data drift из M22.9
    dep_mean = 60.0 if campaign else 40.0
    deposits = rng.gamma(shape=2.0, scale=dep_mean / 2.0, size=n)
    bets = rng.poisson(lam=deposits / 3.0).astype(int)
    avg_bet = np.where(bets > 0, deposits / np.maximum(bets, 1), 0.0)
    days_idle = rng.integers(0, 45, size=n)

    logit = (
        -0.04 * deposits
        - 0.05 * bets
        + 0.10 * days_idle
        - 1.0
    )
    p_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < p_churn).astype(int)

    return pd.DataFrame({
        "deposits_30d": deposits.round(2),
        "bets_30d": bets,
        "avg_bet": avg_bet.round(2),
        "days_since_last_session": days_idle,
        "churn": churn,
    })

if __name__ == "__main__":
    make_players(8000, seed=42).to_parquet("train.parquet")
    make_players(2000, seed=7).to_parquet("prod_normal.parquet")
    make_players(2000, seed=7, campaign=True).to_parquet("prod_drift.parquet")
    print("Готово: train.parquet, prod_normal.parquet, prod_drift.parquet")
```

```bash
uv init churn-mlops && cd churn-mlops
uv add scikit-learn pandas pyarrow numpy scipy joblib skops fastapi "uvicorn[standard]" httpx
uv run python gen_data.py
```

`prod_normal` — батч без дрейфа (тот же мир), `prod_drift` — батч с искусственной кампанией. На них проверим, что детектор молчит на первом и кричит на втором.

## Ход работы

### Шаг 1: обучить и сериализовать двумя способами

Зачем. Отрабатываем M22.4: `joblib` — де-факто стандарт для sklearn, но небезопасен (исполняет произвольный код при загрузке). `skops` — безопасный формат, который при загрузке проверяет типы и не выполняет код. Берём табличный дефолт из модуля — `HistGradientBoostingClassifier`.

```python
# train.py
import joblib
import skops.io as sio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

FEATURES = ["deposits_30d", "bets_30d", "avg_bet", "days_since_last_session"]

df = pd.read_parquet("train.parquet")
X, y = df[FEATURES], df["churn"]
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
)
model.fit(X_tr, y_tr)
auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"val_auc = {auc:.4f}")

joblib.dump(model, "churn_model.joblib")
sio.dump(model, "churn_model.skops")

# защита от версионной ловушки: фиксируем версию рядом с артефактом
import sklearn, json
with open("model_meta.json", "w") as f:
    json.dump({"sklearn": sklearn.__version__, "val_auc": round(auc, 4)}, f)
```

```python
# проверка загрузки skops: untrusted=True покажет, какие типы требуют доверия
import skops.io as sio
untrusted = sio.get_untrusted_types(file="churn_model.skops")
print("Требуют доверия:", untrusted)
m = sio.load("churn_model.skops", trusted=untrusted)
```

Что получилось. Два артефакта одной модели и `model_meta.json` с зафиксированной версией sklearn. `joblib.load` загрузит что угодно молча; `skops` заставит явно перечислить доверенные типы — это и есть инженерная разница в безопасности из M22.4.

!!! question "Проверь себя"

    1. Почему `skops` безопаснее `joblib` при загрузке модели из недоверенного источника?
    2. Зачем мы пишем версию sklearn в `model_meta.json` рядом с моделью?

??? success "Ответы"

    1. `joblib` (как и pickle) исполняет произвольный код при десериализации; `skops` загружает только разрешённые типы и требует явно подтвердить недоверенные — кода он не выполняет.
    2. Чтобы избежать версионной ловушки M22.4: модель, сохранённая в одной версии sklearn, может не загрузиться или тихо врать в другой. Зафиксированная версия позволяет собрать идентичное прод-окружение.

### Шаг 2: поднять FastAPI с Pydantic-валидацией

Зачем. M22.5: FastAPI — стандарт online-сервинга 2026 из-за асинхронности, автодокументации и валидации входов через Pydantic. Добавляем строгую схему `Player`, чтобы мусорный запрос отклонялся до модели, и измеряем латентность каждого запроса.

```python
# serve.py
import time
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ["deposits_30d", "bets_30d", "avg_bet", "days_since_last_session"]
app = FastAPI(title="Churn API")
model = joblib.load("churn_model.joblib")

class Player(BaseModel):
    deposits_30d: float = Field(ge=0)
    bets_30d: int = Field(ge=0)
    avg_bet: float = Field(ge=0)
    days_since_last_session: int = Field(ge=0, le=365)

@app.post("/predict")
def predict(p: Player):
    start = time.perf_counter()
    row = [[p.deposits_30d, p.bets_30d, p.avg_bet, p.days_since_last_session]]
    proba = float(model.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    return {"churn_proba": round(proba, 4), "latency_ms": round(latency_ms, 2)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

```bash
uv run uvicorn serve:app --port 8000
# в другом терминале — автодокументация: http://localhost:8000/docs
```

`Field(ge=0, le=365)` — это валидация диапазонов: отрицательный депозит или 9999 дней простоя вернут HTTP 422 ещё до инференса. Проверь:

```bash
curl -s -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"deposits_30d": 15, "bets_30d": 3, "avg_bet": 5, "days_since_last_session": 40}'
# {"churn_proba": ..., "latency_ms": ...}

curl -s -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"deposits_30d": -5, "bets_30d": 3, "avg_bet": 5, "days_since_last_session": 40}'
# 422 Unprocessable Entity — Pydantic отклонил до модели
```

Что получилось. Рабочий online-эндпоинт с защитой входов и латентностью в ответе. Pydantic делает то, ради чего FastAPI вытеснил Flask: невалидный объект не доходит до модели.

### Шаг 3: нагрузить 1000 запросов и увидеть хвост латентности

Зачем. M22.8: средняя латентность врёт, боль бизнеса в хвосте (p99). Сами генерируем нагрузку и считаем перцентили, чтобы своими глазами увидеть разрыв между средним и p99.

```python
# load_test.py
import httpx, numpy as np, pandas as pd

df = pd.read_parquet("prod_normal.parquet").head(1000)
lat = []
with httpx.Client(base_url="http://localhost:8000") as c:
    for _, r in df.iterrows():
        payload = {
            "deposits_30d": float(r.deposits_30d),
            "bets_30d": int(r.bets_30d),
            "avg_bet": float(r.avg_bet),
            "days_since_last_session": int(r.days_since_last_session),
        }
        resp = c.post("/predict", json=payload)
        lat.append(resp.json()["latency_ms"])

lat = np.array(lat)
print(f"mean = {lat.mean():.2f} ms")
print(f"p50/p95/p99 = {np.percentile(lat, [50, 95, 99]).round(2)}")
```

Что получилось. Видно, что `mean` ниже `p99`: среднее прячет хвост. Это прямой эксперимент к предупреждению «среднее врёт» из M22.8 — мониторить надо перцентили.

!!! question "Проверь себя"

    1. Что вернёт сервис, если прислать `bets_30d: -1`, и на каком этапе это отсекается?
    2. Почему p99 информативнее среднего для решения о масштабировании сервиса?

??? success "Ответы"

    1. HTTP 422 от Pydantic-валидации (`Field(ge=0)`) — запрос отклоняется до вызова модели.
    2. Среднее маскирует хвост: при среднем в десятки мс каждый сотый запрос (p99) может ждать в разы дольше — именно этот хвост ощущают пользователи и он определяет реальную ёмкость сервиса.

### Шаг 4: детектор дрейфа на PSI и KS

Зачем. M22.9: дрейф детектируют статистическими тестами между референсным (train) и текущим (прод) окном. Для непрерывных признаков — PSI и тест Колмогорова–Смирнова. PSI читаем по порогам теории: `<0.1` нет, `0.1–0.2` умеренный, `>=0.2` сильный сдвиг.

```python
# drift_monitor.py
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

FEATURES = ["deposits_30d", "bets_30d", "avg_bet", "days_since_last_session"]

def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def zone(v):
    return "OK" if v < 0.1 else ("WARN" if v < 0.2 else "ALERT")

def report(ref: pd.DataFrame, cur: pd.DataFrame):
    rows = []
    for f in FEATURES:
        p = psi(ref[f].values, cur[f].values)
        ks = ks_2samp(ref[f].values, cur[f].values)
        rows.append({
            "feature": f, "psi": round(p, 4), "zone": zone(p),
            "ks_stat": round(ks.statistic, 4), "ks_p": round(ks.pvalue, 4),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import sys
    ref = pd.read_parquet("train.parquet")
    cur = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "prod_drift.parquet")
    rep = report(ref, cur)
    print(rep.to_string(index=False))
    fired = rep[rep["zone"] == "ALERT"]
    if len(fired):
        print(f"\nДРЕЙФ: {list(fired.feature)} — кандидаты на переобучение")
    else:
        print("\nДрейф в норме")
```

```bash
uv run python drift_monitor.py prod_normal.parquet   # должен молчать
uv run python drift_monitor.py prod_drift.parquet    # должен поймать deposits_30d
```

Что получилось. На `prod_normal` все признаки в зоне OK/WARN, KS p-value большой. На `prod_drift` `deposits_30d` (и зависимые от него `bets_30d`, `avg_bet`) уходят в ALERT с PSI `>=0.2` и KS p-value около нуля — детектор поймал ту самую «маркетинговую кампанию», которую мы зашили в генераторе. Это полный цикл M22.9: референс vs текущее окно, два теста, пороги, алерт.

!!! question "Проверь себя"

    1. Почему бины PSI мы фиксируем по `expected` (train), а не по `actual`?
    2. Дрейф `deposits_30d` пойман — значит ли это, что модель точно деградировала? Что проверить дальше (M22.9)?

??? success "Ответы"

    1. Чтобы сравнивать оба распределения в одной системе координат: бины — это эталонная разбивка референса, и мы смотрим, как прод-доли перетекают между фиксированными бинами. Перестраивать бины по `actual` означало бы менять линейку при каждом замере.
    2. Нет: дрейф признака не равен деградации модели. Нужно (а) проверить, не сломан ли ETL/источник — частая причина ложного «дрейфа»; (б) связать дрейф с важностью признака и прямыми метриками качества, когда придут labels. Реагировать переобучением — только если падает качество или дрейфует важный признак.

## Критерий готовности

- [ ] `train.py` обучает модель, печатает `val_auc` и сохраняет её в `joblib` и `skops`
- [ ] `model_meta.json` содержит зафиксированную версию sklearn
- [ ] `skops.load` отрабатывает с явным списком доверенных типов
- [ ] FastAPI поднимается, `/predict` возвращает `churn_proba` и `latency_ms`, `/docs` открывается
- [ ] Невалидный вход (отрицательный депозит) возвращает 422 до вызова модели
- [ ] `load_test.py` печатает mean и p50/p95/p99, и p99 заметно выше среднего
- [ ] `drift_monitor.py` молчит на `prod_normal` и даёт ALERT на `prod_drift`
- [ ] Можешь объяснить, почему дрейф признака не равен деградации модели

## Развитие

- Добавь структурированный JSON-лог каждого запроса в `serve.py` (поля `timestamp`, `features`, `proba`, `latency_ms`) и скорми накопленные логи в `drift_monitor.py` как прод-окно — получится замкнутая петля «сервинг → мониторинг» из M22.8–M22.9.
- Расширь детектор на категориальный признак (добавь `geo` в генератор) и посчитай по нему тест хи-квадрат вместо PSI, как советует теория для категорий.
- Замени ручной PSI на отчёт Evidently на тех же трёх parquet и сравни выводы — это пункт 7 практики модуля. Объясни расхождения в порогах.
- Заверни сервис в Docker по Dockerfile из M22.6, зафиксировав версии через `uv lock`, и проверь, что эндпоинт отвечает из контейнера.
- Добавь gate-проверку (M22.10): перед заменой `churn_model.joblib` сравнивай val_auc новой модели с `model_meta.json` и промоуть только при не-ухудшении.
