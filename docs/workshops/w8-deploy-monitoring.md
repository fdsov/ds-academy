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

!!! info "Как устроен этот воркшоп"

    Это не лекция с готовым кодом, а задачник. Каждый шаг построен по схеме **Задача -> Критерий -> Решение**:

    - **Задача** — что именно сделать руками, с явными именами выходных переменных, которые проверяет критерий.
    - **Критерий шага** — способ проверить себя. Где шаг считается локально (обучение, дрейф, валидация схемы) — это `python`-блок с `assert`: зелёный прогон = шаг сдан. Где шаг требует поднятого сервера или Docker — это **чек-лист** «- [ ]», потому что результат зависит от внешней инфраструктуры, а не от чистого вычисления.
    - **Решение** спрятано под спойлер `Решение` — открывай после своей попытки.
    - **Статзадачи** проверяются прямо на странице: посчитай число, впиши в поле, нажми «Проверить».

    Все локальные `assert` рассчитаны на сгенерированные данные с зашитыми сидами (train `seed=42`, live `seed=7/8`). Не меняй сиды, иначе числа поплывут.

## Бизнес-кейс

!!! example "Ситуация"

    Ты — data scientist в команде iGaming-продукта. Полгода назад вы выкатили churn-модель: она каждый день скорит активных игроков, и **Head of Retention** по топ-децилю риска шлёт удерживающие бонусы и пуши. Модель «работает» — но никто не знает, насколько она ещё адекватна боевому трафику.

    Через две недели маркетинг запускает залив нового дешёвого гео. Head of Retention приходит к тебе: «Я трачу бонусный бюджет, опираясь на скоры твоей модели. Если она тихо поедет на новом трафике, я узнаю об этом через квартал — по факту, что бюджет слили, а игроки всё равно ушли. Мне нужно понимать, когда модели больше нельзя верить, в день когда это случилось, а не постфактум».

    - **Цена вопроса.** Retention раздаёт бонусов примерно на 1.5 млн ₽/мес по скорам модели. Если из-за дрейфа трафика точность в топ-дециле падает условно на треть — это порядка 0.5 млн ₽/мес бонусов уходит не тем игрокам, плюс упущенное удержание уходящих «китов», которых модель перестала ловить.
    - **Что зависит от твоего ответа.** Решение: строить ли контур наблюдаемости и автотриггер на переобучение, и при каком пороге дрейфа дёргать дорогостоящее переобучение. Слишком чувствительный триггер — переобучаете на шуме и дестабилизируете прод; слишком грубый — пропускаете реальную деградацию и сжигаете бюджет.
    - **Ограничение.** Фактический исход (ушёл игрок или нет) приходит только через 30 дней, поэтому быстрый сигнал у тебя один — дрейф входного распределения. Срок — успеть до старта залива нового гео, ~2 недели.

## Предпосылки

Основной модуль — **M22 (MLOps)**: сериализация моделей, упаковка сервиса, контейнеризация, концепции дрейфа данных и концепт-дрейфа, мониторинг в проде.

Полезно, но не обязательно держать в голове:

- **W3 — churn-модель**: если ты делал воркшоп W3, бери готовый `churn_model.pkl` оттуда. Если нет — мы обучим простую модель прямо здесь за пять минут, воркшоп самодостаточен.
- **M14 — supervised ML**: понимать, что такое `predict_proba` и почему порог решения — отдельная бизнес-настройка.
- **M17 — подводные камни**: дрейф данных и концепт-дрейф концептуально оттуда.

Окружение через **uv** (стандарт 2026). Это setup, а не задача — выполни как есть:

```bash
uv init churn-api && cd churn-api
uv add fastapi "uvicorn[standard]" pydantic joblib scikit-learn lightgbm pandas numpy scipy pyarrow
uv add httpx          # клиент для прогона нагрузки и тестов
uv add --dev pytest
```

!!! tip "Почему uv и фиксация версий критичны именно для деплоя"

    В деплое воспроизводимость окружения — не удобство, а корректность. Модель, обученная на `scikit-learn==1.5`, может не загрузиться на `1.7` или загрузиться, но считать иначе. `uv.lock` фиксирует точные версии, а Docker-образ замораживает их навсегда. Любой `docker run` через год даст ровно тот же ответ на тот же вход. Это и есть «прод-подобное состояние».

Версии стека, на которых собран воркшоп: Python 3.12, FastAPI 0.115+, uvicorn 0.34+, pydantic 2.x, joblib 1.4+, scikit-learn 1.5+, lightgbm 4.x.

## Данные

Нам нужны два датасета с **одинаковой схемой признаков**: обучающий (эталон распределений, baseline) и «боевой» поток (то, что приходит в прод). Чтобы мониторингу дрейфа было что показывать, боевой поток мы сознательно сделаем частично сдвинутым — как если бы маркетинг залил новый гео-трафик и поведение игроков поехало.

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

Это setup-код, а не задача — скопируй и запусти как есть. Один генератор с параметром `regime` отдаёт либо обучающее распределение (`base`), либо сдвинутое боевое (`drifted`). Сдвиг реалистичный: меньше депозитов, выше recency, выше доля бонусов — типичная картина при заливе дешёвого трафика. Так у мониторинга будет настоящий дрейф, а не шум. Seed фиксирован, чтобы результат повторялся у любого.

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

```bash
uv run python gen_data.py
```

**Критерий шага** — запусти после генерации:

```python
import pandas as pd
from gen_data import make_dataset

train = make_dataset(8000, regime="base", seed=42)
assert train.shape == (8000, 9), "8000 строк, 8 признаков + таргет"
assert "churn" in train.columns
rate = train["churn"].mean()
assert 0.10 <= rate <= 0.14, f"доля churn около 0.12, получили {rate:.3f}"
print(f"OK: train сгенерирован, churn rate = {rate:.3f}")
```

!!! note "Чем заменить на реальные данные"

    Подставь любой churn-датасет с числовыми признаками. На Kaggle подходят Telco Customer Churn и Bank Customer Churn — структура та же (поведенческие признаки + бинарный таргет). Для гемблинг-специфики реальный лог событий из W1/W3 агрегируешь в те же оконные признаки. Важно одно: baseline (обучающее распределение) и live-поток должны иметь идентичный набор колонок — на этом держится весь мониторинг.

## Статразминка: формулы дрейфа руками

Прежде чем писать монитор, прогрей две формулы, на которых он держится. Эти числа понадобятся на шагах 6-7. Считай калькулятором, ответ впиши в поле — проверка мгновенная.

```text
TASK: Боевой поток = 70% базового трафика (средний bonus_ratio = 0.25) + 30% сдвинутого (средний bonus_ratio = 0.50). Чему равно среднее bonus_ratio в смешанном потоке? Округли до 0.01.
ANSWER: 0.325
TOL: 0.02
PLACEHOLDER: 0.00
EXPLAIN: Среднее смеси = взвешенная сумма: 0.7*0.25 + 0.3*0.50 = 0.175 + 0.15 = 0.325. На реальном сгенерированном потоке выйдет около 0.322 - сдвиг налицо: эталонные 0.25 превратились в 0.32. Именно этот сдвиг распределения и ловит монитор дрейфа, хотя каждое отдельное значение остаётся валидным (в диапазоне 0..1).
---
TASK: Посчитай PSI на простом примере из 2 бинов. Эталон (baseline) распределён 50/50: e = [0.5, 0.5]. Боевые данные: a = [0.6, 0.4]. PSI = sum((a_i - e_i) * ln(a_i / e_i)). Округли до 0.001.
ANSWER: 0.0405
TOL: 0.004
PLACEHOLDER: 0.000
EXPLAIN: (0.6-0.5)*ln(0.6/0.5) + (0.4-0.5)*ln(0.4/0.5) = 0.1*ln(1.2) + (-0.1)*ln(0.8) = 0.1*0.1823 + (-0.1)*(-0.2231) = 0.0182 + 0.0223 = 0.0405. Это сильно ниже порога 0.1, то есть распределение почти стабильно. PSI растёт нелинейно: чем дальше боевые доли уезжают от эталонных, тем резче слагаемые (a-e)*ln(a/e). Так формула штрафует большие перекосы сильнее, чем мелкие.
```

## Ход работы

Маршрут такой: обучаем и сериализуем модель -> описываем контракт входа через pydantic -> поднимаем FastAPI -> добавляем логирование предсказаний -> пакуем в Docker -> пишем монитор дрейфа на PSI + KS -> задаём триггер на переобучение. Шаги 1, 2, 6, 7 считаются локально (есть `assert`); шаги 3, 4, 5 требуют поднятого сервера/Docker — там критерий это чек-лист.

### Шаг 1: Обучить и сериализовать модель

**Зачем.** Нам нужен артефакт, а не просто обученный объект в памяти. Сериализуем не «голую» модель, а **бандл**: модель + список признаков в правильном порядке + baseline-статистики (для мониторинга) + метаданные (версия, дата, метрика). Порядок признаков критичен: если в проде колонки придут в другом порядке, scikit-learn молча посчитает мусор. Бандл — это контракт.

**Задача.** Обучи `LGBMClassifier` на `base`-данных, посчитай val AUC и собери словарь `bundle` с ключами `model`, `features`, `baseline`, `meta`. В `baseline[f]` для каждого признака положи `quantiles` (11 квантилей для PSI), `mean`, `std` и `sample` (2000 значений для KS-теста). В `meta` положи `version`, `trained_at`, `val_auc`, `churn_rate`, `threshold`. Сохрани бандл в `churn_model.pkl` через `joblib`. Имена `bundle` и `auc` проверяет критерий.

??? tip "Подсказка"

    Baseline-статистики считай **только по обучающей выборке** `X_tr` (не по всей `X`), иначе эталон протечёт. `sample` бери через `X_tr[f].sample(2000, random_state=0)` — фиксированный random_state нужен, чтобы KS-тест был воспроизводим. Квантили — `np.quantile(X_tr[f], np.linspace(0, 1, 11))`.

**Критерий шага:**

```python
from pathlib import Path
from gen_data import FEATURES

assert set(bundle) >= {"model", "features", "baseline", "meta"}
assert bundle["features"] == FEATURES, "порядок признаков должен совпадать с FEATURES"
assert 0.74 <= bundle["meta"]["val_auc"] <= 0.82, "val AUC около 0.78 на этих данных"
assert 0.10 <= bundle["meta"]["churn_rate"] <= 0.14, "churn rate около 0.12"
assert all({"quantiles", "sample", "mean"} <= set(bundle["baseline"][f]) for f in FEATURES)
assert len(bundle["baseline"]["bonus_ratio"]["quantiles"]) == 11
assert Path("churn_model.pkl").exists(), "бандл должен быть сохранён на диск"
print(f"OK: бандл собран, val AUC = {bundle['meta']['val_auc']}")
```

??? success "Решение"

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
        return bundle, auc

    if __name__ == "__main__":
        bundle, auc = main()
    ```

    ```bash
    uv run python train_model.py
    ```

    **Почему так.** Файл `churn_model.pkl` (~1-2 МБ), val AUC около **0.78** на этих синтетических данных. В бандле всё, что нужно сервису и монитору: модель, порядок признаков, эталонные распределения и версия. `quantiles` пойдут в PSI (бинуем по эталонным границам), `sample` — в KS-тест (он сравнивает наборы значений напрямую). Два разных теста дрейфа — два разных представления baseline.

Сначала посчитай долю оттока и впиши, потом сверь с тем, что выдал `meta`:

```text
TASK: Запусти train_model.py. Какая доля churn в обучающей выборке (поле churn_rate в meta)? Округли до 0.01.
ANSWER: 0.12
TOL: 0.01
PLACEHOLDER: 0.00
EXPLAIN: На сиде 42 доля оттока в train около 0.117. Это рабочий дисбаланс классов: оттока меньше, чем удержания, но не настолько, чтобы класс был редким. AUC такую асимметрию переживает спокойно (он не зависит от порога), а вот точность/полнота в топ-дециле уже чувствительны к ней - поэтому Retention и работает с топ-децилем риска, а не с жёстким лейблом 0/1.
---
TASK: Какой val AUC выдала модель (поле val_auc в meta)? Округли до 0.01.
ANSWER: 0.78
TOL: 0.03
PLACEHOLDER: 0.00
EXPLAIN: На этих данных LightGBM даёт val AUC около 0.779. Это «крепко рабочая, но не идеальная» модель - ровно та ситуация, где мониторинг дрейфа особенно важен: запас качества невелик, и сдвиг трафика быстро столкнёт её в зону, где скоры перестанут разделять уходящих и остающихся.
```

!!! warning "Не сериализуй голую модель через pickle напрямую"

    `pickle.dump(model)` хранит ссылки на классы из той версии библиотеки, что была при сохранении. `joblib` эффективнее для numpy-массивов внутри деревьев и стандартен для scikit-стека. Но главный риск — версии библиотек: всегда фиксируй их (uv.lock + Docker), иначе загрузка артефакта на другой версии либо упадёт, либо тихо изменит предсказания. Бандл с полем `version` в meta позволяет отследить, какая модель отвечала на конкретный запрос.

    Второй риск — безопасность: и `pickle`, и `joblib` при загрузке исполняют произвольный код, зашитый в артефакт, поэтому грузить недоверенный `.pkl` нельзя (arbitrary code execution). Загружай только бандлы, собранные тобой и сохранённые в доверенном хранилище. Для обмена моделями между командами/наружу используй формат `skops`: он сериализует scikit-стек безопасно, без исполнения произвольного кода при загрузке.

Проверь понимание:

```text
Q: Почему список признаков features кладут внутрь артефакта, а не хардкодят в коде сервиса?
[ ] Так артефакт занимает меньше места на диске
[x] Порядок признаков — часть контракта модели; хардкод в сервисе разъедется с моделью при первом же переобучении с новым признаком
[ ] FastAPI не умеет читать списки из кода
> Модель училась на матрице в конкретном порядке колонок. Храня этот порядок в бандле, сервис гарантированно соберёт вход так же, как было при обучении. Хардкод живёт отдельно от модели и рано или поздно разъедется.
---
Q: Что вероятнее всего случится, если обучить на scikit-learn 1.5, а грузить артефакт на 1.7 без фиксации версий?
[ ] Ничего, формат полностью совместим между версиями
[x] В лучшем случае отказ грузиться, в худшем — тихо иные предсказания из-за изменений во внутреннем формате
[ ] Модель автоматически дообучится под новую версию
> Внутренний формат деревьев и дефолты между версиями меняются. Поэтому версии замораживают через uv.lock и Docker - иначе получаешь либо краш, либо незаметно неверные скоры.
```

### Шаг 2: Контракт входных данных через pydantic

**Зачем.** Прод-API не доверяет входу. Клиент пришлёт строку вместо числа, отрицательный депозит, `bonus_ratio = 5` или вообще пропустит поле. Без валидации это либо упадёт где-то в недрах LightGBM с непонятной 500-й, либо посчитает мусор. pydantic v2 описывает схему декларативно: типы, границы, обязательность. Невалидный запрос отбивается 422-й с понятным сообщением **до** того, как доберётся до модели.

**Задача.** Опиши pydantic-модель `PlayerFeatures` со всеми восемью признаками, бизнес-границами через `Field(ge=..., le=...)` и `model_config = ConfigDict(extra="forbid")` (лишнее поле = ошибка). Опиши `PredictResponse` (вероятность, лейбл, порог, версия) — контракт ответа наружу. Имя `PlayerFeatures` проверяет критерий.

**Критерий шага:**

```python
from pydantic import ValidationError

good = dict(deposits_30d=1, deposit_sum_30d=40.0, bets_30d=12, sessions_30d=2,
            avg_session_min=6.5, days_since_last_login=25.0, bonus_ratio=0.7, withdrawal_30d=0)
assert PlayerFeatures(**good).bonus_ratio == 0.7, "валидный вход должен проходить"

bad_cases = [
    {**good, "bonus_ratio": 5},                          # вне [0, 1]
    {**good, "oops": 1},                                 # лишнее поле
    {k: v for k, v in good.items() if k != "bets_30d"},  # пропущенное поле
]
for bad in bad_cases:
    try:
        PlayerFeatures(**bad)
        raise AssertionError(f"должно было отклонить: {bad}")
    except ValidationError:
        pass
print("OK: схема пропускает валидный вход и отбивает три вида невалидного")
```

??? success "Решение"

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

    **Почему так.** `extra="forbid"` превращает опечатку в имени поля из тихо проигнорированного значения в явную 422-ю. Границы `Field` — первый барьер на пути «грязного» входа: `bonus_ratio` физически в [0, 1], `days_since_last_login` не может быть 5000. Они ловят баги интеграции на стороне клиента и отсекают абсурдные значения, на которых модель не училась. Но границы **не ловят дрейф**: значение может быть валидным и всё равно «уехавшим» от обучающего распределения — этим займётся монитор на шаге 6.

### Шаг 3: FastAPI-сервис с эндпоинтом predict

**Зачем.** Модель грузим **один раз** при старте через lifespan, а не на каждый запрос — иначе на каждом вызове будет лишние сотни миллисекунд на чтение файла. Эндпоинт `/predict` принимает валидированный объект, собирает матрицу строго в порядке `features` из бандла, считает вероятность и применяет порог. Плюс `/health` (жив ли сервис, для оркестратора) и `/version` (какая модель отвечает).

**Задача.** Напиши `app.py`: загрузи бандл в `lifespan` и положи `model`/`features`/`meta` в общий `STATE`; реализуй `/health`, `/version` и `POST /predict`. В `/predict` собери `X` **явным перебором** `row[f] for f in features` (не `pd.DataFrame([row])`!), посчитай вероятность, примени порог, залогируй вызов и верни `PredictResponse`. Подними сервис и проверь `curl`-ом.

**Критерий шага (self-check)** — этот шаг требует поднятого сервера, поэтому проверяем руками:

- [ ] `uv run uvicorn app:app --port 8000` стартует без ошибок, в логах видно `model loaded: 1.0.0`;
- [ ] `GET /health` отвечает `{"status":"ok","model_loaded":true}`;
- [ ] `GET /version` отдаёт `meta` с `version`, `val_auc`, `threshold`;
- [ ] `POST /predict` на валидном теле возвращает 4 поля (`churn_probability`, `churn_label`, `threshold`, `model_version`);
- [ ] `POST /predict` с `bonus_ratio: 5` возвращает 422, а не 500;
- [ ] на `http://localhost:8000/docs` Swagger построил схему из pydantic автоматически.

??? success "Решение"

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

    ```bash
    uv run uvicorn app:app --host 0.0.0.0 --port 8000
    ```

    ```bash
    curl -s -X POST http://localhost:8000/predict \
      -H "content-type: application/json" \
      -d '{"deposits_30d":1,"deposit_sum_30d":40.0,"bets_30d":12,"sessions_30d":2,"avg_session_min":6.5,"days_since_last_login":25.0,"bonus_ratio":0.7,"withdrawal_30d":0}'
    ```

    **Почему так.** Ответ вида `{"churn_probability":...,"churn_label":...,"threshold":0.5,"model_version":"1.0.0"}`. Модель в `STATE` — грузится один раз, latency на запрос минимальна. Матрица строится явным перебором по `features`, поэтому порядок колонок гарантированно совпадает с обучением.

!!! warning "Порядок признаков — самая частая тихая ошибка"

    Мы строим `X` явным перебором `row[f] for f in features`. Если вместо этого сделать `pd.DataFrame([row])`, порядок колонок определит порядок ключей в словаре, а не порядок обучения. LightGBM не проверяет имена — он берёт позиции. Перепутанные местами `bets_30d` и `bonus_ratio` дадут валидный ответ с абсолютно неверной вероятностью, и ты этого не заметишь. Всегда выстраивай матрицу по сохранённому списку признаков.

### Шаг 4: Логирование предсказаний

**Зачем.** Без лога предсказаний мониторинг дрейфа невозможен в принципе — нечего сравнивать с baseline. Пишем каждый вызов в **JSON Lines** (одна строка = один JSON-объект): вход, вероятность, лейбл, версия модели, UTC-таймстамп. JSONL удобен тем, что дописывается атомарно построчно и легко читается обратно в pandas. В реальном проде это ушло бы в Kafka / S3 / лог-коллектор, но контракт строки тот же.

**Задача.** Напиши `prediction_logger.py` с функцией `log_prediction(features, proba, label, version)`, дописывающей строку в `predictions.jsonl` (поля `ts` в UTC, `model_version`, `churn_probability`, `churn_label`, `features`). Затем напиши `replay_live.py`, который прогоняет весь `live_stream.parquet` через `POST /predict`, чтобы накопить лог. Запусти сервис, прогони реплей.

**Критерий шага (self-check)** — реплей бьёт в живой сервис, поэтому проверяем руками:

- [ ] сервис из шага 3 запущен в соседнем терминале;
- [ ] `uv run python replay_live.py` отправляет ~3000 запросов без 4xx/5xx ошибок;
- [ ] `wc -l predictions.jsonl` показывает ~3000 строк;
- [ ] первая строка лога — валидный JSON с ключами `ts`, `model_version`, `churn_probability`, `features`;
- [ ] в `features` ровно 8 ключей-признаков, значения числовые.

??? success "Решение"

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

    **Почему так.** Каждый `/predict` дописывает одну строку — это сырьё для монитора. Приведение типов в `replay_live` (`int`/`float`) нужно, чтобы pydantic не споткнулся о `numpy.int64`. После реплея в `predictions.jsonl` лежит ровно тот боевой поток, распределение которого монитор сравнит с baseline.

!!! note "Что именно логировать в проде"

    Минимум для мониторинга: входные признаки + выход + версия модели + таймстамп. Желательно ещё request_id (трейсинг) и, если приходит позже, фактический исход (ground truth) — он нужен для мониторинга **качества** (падение AUC), а не только дрейфа входа. Чего не логировать без нужды: PII в сыром виде. В гемблинге player_id хешируй или храни отдельно с контролем доступа.

Проверь понимание:

```text
Q: Почему дрейф нельзя обнаружить одними pydantic-границами из шага 2?
[ ] Границы проверяют только строки, но не числа
[x] Границы проверяют допустимость каждого значения, а дрейф — это смещение распределения валидных значений (средний bonus_ratio был 0.25, стал 0.32)
[ ] pydantic вообще не умеет работать с распределениями по соображениям производительности
> Каждое значение по-прежнему в [0,1] и проходит валидацию, но статистика входа уехала. Это ловит только сравнение распределений (PSI/KS), а не границы отдельных полей.
---
Q: Зачем писать в лог model_version, если сейчас модель одна?
[ ] Это требование формата JSONL
[x] Чтобы при будущем переобучении (1.1, 1.2) знать, какая версия отвечала на конкретный запрос, и разделять деградацию старой модели от поведения новой
[ ] Чтобы уменьшить размер строки лога
> Без версии при инциденте нельзя понять, что именно катилось в проде, и нельзя сравнить поведение версий. Это базовая трассируемость.
---
Q: Чем JSONL удобнее одного большого JSON-массива для лога, который дописывается онлайн?
[ ] JSONL занимает меньше места на диске
[x] Строку можно дописать в конец файла атомарно, не читая и не перезаписывая весь файл
[ ] JSON-массив нельзя прочитать в pandas
> JSON-массив пришлось бы каждый раз десериализовать целиком, добавлять элемент и писать обратно - медленно и небезопасно при параллельных записях. JSONL дописывается построчно.
```

### Шаг 5: Dockerfile и запуск в контейнере

**Зачем.** Контейнер — это способ заморозить ОС, Python, версии библиотек и саму модель в один неизменяемый артефакт. «Работает у меня» превращается в «работает везде одинаково». Используем многоступенчатую сборку: ставим зависимости через uv, копируем код и модель, запускаем uvicorn. `--frozen` гарантирует установку ровно из `uv.lock`.

**Задача.** Напиши `Dockerfile` (база `python:3.12-slim`, установка через `uv sync --frozen --no-dev`, копирование кода и `churn_model.pkl`, `HEALTHCHECK` на `/health`, запуск uvicorn) и `.dockerignore` (исключи `.venv`, `*.parquet`, `predictions.jsonl`, `.git`). Собери образ и запусти контейнер.

**Критерий шага (self-check)** — нужен установленный Docker, проверяем руками:

- [ ] `docker build -t churn-api:1.0.0 .` собирается без ошибок;
- [ ] `.dockerignore` исключает локальный лог и parquet (образ не тащит лишнее);
- [ ] `docker run --rm -p 8000:8000 churn-api:1.0.0` поднимает сервис;
- [ ] `curl -s http://localhost:8000/health` из хоста отвечает `{"status":"ok","model_loaded":true}`;
- [ ] `docker ps` показывает статус контейнера `healthy` (HEALTHCHECK прошёл).

??? success "Решение"

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

    ```text
    # .dockerignore
    .venv
    __pycache__
    *.parquet
    predictions.jsonl
    .git
    .pytest_cache
    ```

    ```bash
    docker build -t churn-api:1.0.0 .
    docker run --rm -p 8000:8000 churn-api:1.0.0
    curl -s http://localhost:8000/health
    ```

    **Почему так.** Тот же образ запускается на ноутбуке, сервере, Railway или в Kubernetes без изменений. `--frozen` запрещает молчаливое обновление зависимостей — ставится ровно то, что в `uv.lock`. HEALTHCHECK даёт оркестратору сигнал, жив ли сервис.

!!! warning "Лог предсказаний внутри контейнера эфемерен"

    `predictions.jsonl` пишется в файловую систему контейнера и **исчезает при его пересоздании**. Для воркшопа это ок (монтируй volume: `-v $(pwd)/data:/app/data` и пиши лог туда). В реальном проде лог должен уходить наружу — в внешнее хранилище (S3), очередь (Kafka) или лог-коллектор, а не жить в контейнере. Мониторинг дрейфа читает данные из этого внешнего стора, а не из локального файла на одном поде.

### Шаг 6: Монитор дрейфа — PSI и KS-тест

**Зачем.** Это ядро воркшопа. Мы сравниваем распределение каждого признака в **боевом потоке** с **обучающим эталоном** (baseline из бандла) и численно отвечаем: уехало или нет?

Два дополняющих инструмента:

**PSI (Population Stability Index)** — индустриальный стандарт (родом из кредитного скоринга). Бьём оба распределения на бины по эталонным квантилям и суммируем по бинам:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i)\,\ln\frac{a_i}{e_i}$$

где $e_i$ — доля baseline в бине $i$, $a_i$ — доля боевых данных в том же бине. Общепринятые пороги: $\text{PSI} < 0.1$ — стабильно; $0.1 \le \text{PSI} < 0.25$ — умеренный сдвиг, наблюдаем; $\text{PSI} \ge 0.25$ — значимый дрейф, действуем.

**KS-тест (Колмогоров — Смирнов)** — статистический тест на равенство двух непрерывных распределений. Даёт статистику $D$ (максимум разницы эмпирических CDF) и p-value. Маленький p-value (< 0.05) — распределения различаются значимо. PSI говорит «насколько сильно», KS — «статистически значимо ли вообще».

**Задача.** Напиши `drift_monitor.py`: функцию `population_stability_index(baseline_vals, live_vals, edges)` (замени крайние границы бинов на $\pm\infty$, добавь эпсилон от логарифма нуля) и `run_drift_report(...)`, которая для каждого признака считает PSI (по `baseline.sample` против live), KS-тест и присваивает статус OK/WATCH/DRIFT по порогам выше. Верни DataFrame `rep`, отсортированный по PSI убыванию. Для воркшопа можно читать боевой поток прямо из `live_stream.parquet` (он равен залогированному) либо из `predictions.jsonl`, если прошёл шаги 3-4. Имя `rep` проверяет критерий.

??? tip "Подсказка"

    PSI без $\pm\infty$ на краях теряет боевые значения за пределами обучающего диапазона (новый трафик принёс recency, которой в train не было) — они выпадут за крайние бины и занизят PSI. После `np.histogram` нормируй счётчики в доли и обрежь снизу эпсилоном (`np.clip(e, 1e-6, None)`), иначе `ln(a/e)` взорвётся на пустом бине.

**Критерий шага:**

```python
top = rep.iloc[0]
assert top["feature"] == "bonus_ratio", "сильнее всех уехал bonus_ratio (зашитый сдвиг)"
assert 0.15 <= top["psi"] <= 0.25, "PSI bonus_ratio около 0.19 — это WATCH, не DRIFT"
assert top["status"] == "WATCH"
assert (rep["status"] == "DRIFT").sum() == 0, "при 30% сдвинутого трафика до DRIFT не дотягивает никто"
stable = rep[rep["feature"].isin(["avg_session_min", "withdrawal_30d"])]
assert (stable["status"] == "OK").all(), "несдвинутые признаки должны быть OK"
assert (rep["ks_pvalue"] <= 1).all()
print(f"OK: топ-дрейф = {top['feature']} (PSI={top['psi']}, {top['status']})")
```

??? success "Решение"

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

    def load_live_from_log(path="predictions.jsonl") -> pd.DataFrame:
        rows = [json.loads(line)["features"] for line in Path(path).read_text().splitlines()]
        return pd.DataFrame(rows)

    def run_drift_report(bundle_path="churn_model.pkl", live_path="live_stream.parquet"):
        bundle = joblib.load(bundle_path)
        features, baseline = bundle["features"], bundle["baseline"]
        # из лога: load_live_from_log("predictions.jsonl"); для воркшопа — прямо из parquet
        live = pd.read_parquet(live_path)

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
                "feature": f, "psi": round(psi, 4),
                "ks_stat": round(float(ks_stat), 4), "ks_pvalue": round(float(ks_p), 6),
                "base_mean": round(b["mean"], 3), "live_mean": round(float(live_vals.mean()), 3),
                "status": status,
            })
        return pd.DataFrame(report).sort_values("psi", ascending=False)

    if __name__ == "__main__":
        rep = run_drift_report()
        pd.set_option("display.width", 200)
        print(rep.to_string(index=False))
    ```

    ```bash
    uv run python drift_monitor.py
    ```

    Реальный вывод на этих данных:

    ```text
                  feature    psi  ks_stat  ks_pvalue  base_mean  live_mean status
              bonus_ratio 0.1947   0.1847   0.000000      0.250      0.322  WATCH
             deposits_30d 0.0840   0.1245   0.000000      3.998      3.492     OK
                 bets_30d 0.0703   0.1122   0.000000     43.990     41.027     OK
    days_since_last_login 0.0416   0.0708   0.000011      5.970      7.403     OK
          deposit_sum_30d 0.0338   0.0812   0.000000    118.589    106.197     OK
             sessions_30d 0.0301   0.0667   0.000045      6.173      5.822     OK
          avg_session_min 0.0037   0.0130   0.986243     14.228     14.096     OK
           withdrawal_30d 0.0010   0.0137   0.976923      0.293      0.272     OK
    ```

    **Почему так.** Видно ровно тот сдвиг, что мы зашили в `drifted`: `bonus_ratio` уехал сильнее всех (PSI 0.19 -> WATCH, среднее 0.25 -> 0.32), депозиты и recency сдвинулись слабее, а `avg_session_min`/`withdrawal_30d` остались стабильными. Обрати внимание на ловушку KS: у `deposits_30d` p-value = 0.0000 (значимо!), но PSI всего 0.084 — сдвиг статистически реальный, но по величине ниже порога наблюдения. Решение принимаем по PSI, KS используем как подтверждение.

Сначала прикинь PSI top-признака сам:

```text
TASK: По таблице из drift_monitor.py — какой PSI у признака с самым сильным дрейфом (bonus_ratio)? Округли до 0.01.
ANSWER: 0.19
TOL: 0.03
PLACEHOLDER: 0.00
EXPLAIN: PSI bonus_ratio = 0.1947. Это попадает в зону 0.1..0.25 — «умеренный сдвиг, наблюдаем» (WATCH), но НЕ дотягивает до порога значимого дрейфа 0.25 (DRIFT). Важный вывод воркшопа: 30% сдвинутого трафика в смеси разбавляются 70% нормального, поэтому даже самый уехавший признак остаётся в WATCH. Чтобы получить DRIFT, нужен более сильный или более массовый сдвиг.
```

!!! tip "Почему PSI и KS, а не один из них"

    KS чувствителен к размеру выборки: на десятках тысяч строк он даст p-value < 0.05 при мизерном, бизнес-незначимом сдвиге — статистически значимо, но практически неважно (смотри `deposits_30d` в таблице). PSI — мера величины эффекта, она не раздувается от объёма данных, но в ней нет понятия «значимости». Связка покрывает оба вопроса: KS — «отличие реально или это шум выборки», PSI — «насколько отличие большое, чтобы реагировать». Для категориальных признаков KS не работает — там берут PSI или хи-квадрат.

Проверь понимание:

```text
Q: Почему в PSI крайние границы бинов заменяют на ±бесконечность?
[ ] Чтобы PSI всегда был положительным
[x] Чтобы боевые значения за пределами обучающего диапазона попали в крайние бины, а не выпали и не занизили PSI
[ ] Это требование scipy.stats
> Новый трафик приносит значения, которых в обучении не было (recency 60 дней). Без открытых хвостов они выпали бы за крайние бины и потерялись — PSI оказался бы ложно низким.
---
Q: Что показывает дрейф входных признаков, а что он НЕ показывает про качество модели?
[ ] Дрейф входа напрямую измеряет падение AUC
[x] Он показывает, что распределение входа сместилось; падение качества им не доказано — для этого нужен ground truth и мониторинг метрики
[ ] Дрейф входа и падение AUC — это одно и то же
> AUC мог и не упасть, если связь признак-таргет сохранилась. Падение качества (концепт-дрейф) ловится только фактическим исходом и rolling AUC, а не PSI входа.
---
Q: На выборке 500k строк KS даёт p-value = 0.001 при PSI = 0.02. Реагировать?
[ ] Да, p-value значим — это явный дрейф
[x] Нет: PSI = 0.02 означает ничтожный сдвиг, а маленький p-value — артефакт огромной выборки
[ ] Да, но только переобучить, без алерта
> Решение принимаем по величине эффекта (PSI), а не по p-value. На больших данных KS реагирует на любую микроскопическую разницу — это статистически значимо, но практически неважно.
```

### Шаг 7: Правило-триггер на переобучение и алерт

**Зачем.** Отчёт по дрейфу бесполезен, если никто его не читает каждый день. Нужно **правило**, которое из таблицы делает решение «переобучать / наблюдать / ок» и шлёт алерт. Правило должно быть устойчивым к одиночному выбросу: триггерим не по случайному признаку, а по агрегированному условию.

Политика триггера (разумный дефолт, настраивается под проект):

- **RETRAIN**, если хотя бы один **важный** признак в статусе `DRIFT` (PSI ≥ 0.25), ИЛИ если в статусе `WATCH`/`DRIFT` оказались ≥ 3 признаков одновременно (массовый сдвиг трафика).
- **WATCH** (наблюдаем, не переобучаем), если есть отдельные `WATCH` без важных `DRIFT`.
- **OK** иначе.

**Задача.** Напиши `drift_trigger.py`: функцию `decide(report)`, возвращающую словарь с полями `action`, `drifted_important`, `n_features_shifted`, `max_psi_feature`, `max_psi`, и заглушку `send_alert`. Прогони на отчёте из шага 6, положи результат в `decision`. Подумай заранее: какой вердикт вернётся на нашем потоке? Имя `decision` проверяет критерий.

??? tip "Подсказка"

    Сначала прикинь по таблице из шага 6: ни один PSI не дотянул до 0.25, значит `drifted_important` пуст; в статусе не-OK только `bonus_ratio` (один признак), значит условие «≥ 3» тоже не выполнено. Что остаётся по политике?

**Критерий шага:**

```python
assert decision["action"] == "WATCH", "на 30% сдвиге — наблюдаем, до RETRAIN не дотянули"
assert decision["max_psi_feature"] == "bonus_ratio"
assert 0.15 <= decision["max_psi"] <= 0.25
assert decision["n_features_shifted"] == 1, "не-OK только bonus_ratio"
assert decision["drifted_important"] == [], "ни один важный признак не достиг DRIFT"
print(f"OK: вердикт = {decision['action']} (max PSI {decision['max_psi']:.3f} на {decision['max_psi_feature']})")
```

??? success "Решение"

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

    ```bash
    uv run python drift_trigger.py
    ```

    Реальный вывод:

    ```text
    {
      "action": "WATCH",
      "drifted_important": [],
      "n_features_shifted": 1,
      "max_psi_feature": "bonus_ratio",
      "max_psi": 0.1947
    }
    ```

    **Почему так.** На нашем 30%-сдвиге вердикт — **WATCH, а не RETRAIN**: `bonus_ratio` уехал до WATCH (PSI 0.19 < 0.25), а больше ничего из статуса OK не вышло. Это честный и важный результат: умеренный приток нового трафика ещё не повод дёргать дорогое переобучение — его надо наблюдать. Чтобы триггер выдал RETRAIN, нужен либо более сильный сдвиг одного важного признака (PSI ≥ 0.25), либо массовый сдвиг (≥ 3 признака). Поиграй с долей `drifted` в `gen_data.py` (например, 60% вместо 30%) — увидишь, как вердикт переключается на RETRAIN. Ненулевой exit code при RETRAIN позволяет повесить скрипт на cron/CI и запускать пайплайн переобучения автоматически.

!!! example "Как это замыкается в цикл MLOps"

    1. API логирует предсказания -> 2. По расписанию (cron, ночью) `drift_trigger.py` читает накопленный лог -> 3. PSI/KS считают дрейф -> 4. Правило решает (OK / WATCH / RETRAIN) -> 5. RETRAIN запускает переобучение на свежих данных с разметкой -> 6. Новый бандл (версия +1) собирается в Docker -> 7. Деплой, `/version` показывает новую версию. Получился замкнутый контур: модель сама сигналит о деградации. На наших данных контур остановился на WATCH — и это правильное поведение, а не баг.

!!! tip "Не переобучай на каждый чих"

    Слишком чувствительный триггер (PSI ≥ 0.1 по любому признаку) приведёт к постоянным переобучениям на шуме — это дорого и дестабилизирует прод. Слишком грубый (только PSI ≥ 0.5) пропустит реальную деградацию. Дефолт PSI ≥ 0.25 по важным признакам + правило «≥ 3 сдвинутых» — рабочий компромисс. Калибруй пороги по истории: посмотри, при каком PSI исторически реально падал AUC на твоих данных, и привяжи триггер к этому.

## Типичные ошибки

- **Модель грузится на каждый запрос.** `joblib.load` внутри обработчика `/predict` добавляет сотни мс латентности на вызов. Грузи один раз в lifespan/startup и держи в состоянии приложения.
- **Перепутан порядок признаков.** Самая коварная ошибка: ответ валиден, но неверен. LightGBM/sklearn берут позиции колонок, а не имена. Всегда строй матрицу по сохранённому в бандле списку `features`.
- **Версии библиотек не зафиксированы.** Артефакт, обученный на одной версии scikit-learn/lightgbm, на другой либо не грузится, либо считает иначе. uv.lock + Docker — обязательны, не «на потом».
- **Лог предсказаний живёт в контейнере.** При рестарте пода всё стирается, мониторить нечего. Лог должен уходить во внешнее хранилище.
- **Мониторят только дрейф входа и думают, что покрыли качество.** Дрейф признаков ≠ падение метрики. Связь признак-таргет может смениться при стабильном входе (concept drift) — это ловится только ground truth и мониторингом AUC, а не PSI.
- **KS на огромных выборках принимают за сигнал.** На сотнях тысяч строк KS значим при ничтожном сдвиге. Решение по величине эффекта (PSI), а не по одному p-value.
- **Путают WATCH и RETRAIN.** Умеренный дрейф (PSI в зоне 0.1-0.25) — это сигнал наблюдать, а не повод немедленно переобучать. Триггер должен различать «насторожиться» и «действовать», иначе ты либо паникуешь, либо спишь.
- **PSI на категориальных как на числовых.** Квантильное бинование на категориях бессмысленно. Для категориальных — PSI по категориям как по бинам или хи-квадрат.
- **(Senior) Baseline зафиксирован навсегда и узкий.** Если эталон — это `train.sample(2000)` без сезонности, нормальная недельная/месячная цикличность будет ложно триггерить дрейф. Эталон должен покрывать естественную вариативность; сравнивай скользящим окном, а не с одним замёрзшим снимком.
- **(Senior) Нет защиты от пустого/частичного потока.** Если за сутки пришло 12 предсказаний, PSI на них — шум. Триггер должен требовать минимальный размер боевой выборки (например, ≥ 500) прежде чем выносить вердикт.

!!! tip "AI-копилот в этом воркшопе"

    Где нейросеть реально ускорит: генерация скелета FastAPI-приложения и pydantic-схем по списку признаков, написание Dockerfile под uv, обвязка httpx-клиента для прогона нагрузки, докстринги и pytest-тесты на эндпоинты. Это шаблонный код, который копилот пишет точно и быстро.

    Где подведёт именно здесь: (1) формула PSI — копилоты регулярно путают, что под логарифмом, и забывают про эпсилон и бесконечные хвосты бинов; проверяй вручную по определению. (2) Пороги PSI/KS и логику триггера он подгонит «правдоподобно», но без привязки к твоим данным — это бизнес-решение, не код. (3) Порядок признаков при сборке матрицы — копилот часто пишет `pd.DataFrame([dict])`, что молча ломает порядок; настаивай на явном переборе по сохранённому списку. (4) Различие дрейфа входа и концепт-дрейфа он смазывает — может предложить «мониторить дрейф» и считать вопрос качества закрытым. Держи это разделение в голове сам.

## Критерий готовности

- [ ] `train_model.py` обучает модель и сохраняет бандл `churn_model.pkl` с моделью, списком признаков, baseline и meta (версия, AUC ≈ 0.78).
- [ ] FastAPI поднимается, `/health` отвечает ok, `/version` отдаёт метаданные модели.
- [ ] `/predict` валидирует вход через pydantic (невалидный запрос → 422) и возвращает вероятность, лейбл, порог и версию.
- [ ] Матрица признаков строится строго по сохранённому порядку `features`.
- [ ] Каждый вызов `/predict` пишет строку в `predictions.jsonl` (вход + выход + версия + UTC-таймстамп).
- [ ] `docker build` собирает образ, `docker run` поднимает сервис, `/health` отвечает из контейнера.
- [ ] `drift_monitor.py` считает PSI и KS по каждому признаку и выдаёт таблицу со статусами OK/WATCH/DRIFT.
- [ ] На сдвинутом боевом потоке монитор корректно подсвечивает именно сдвинутые признаки (`bonus_ratio` сильнее всех, до WATCH).
- [ ] `drift_trigger.py` выносит решение RETRAIN/WATCH/OK, шлёт алерт и отдаёт ненулевой exit code на RETRAIN; на нашем потоке вердикт — WATCH.

## Бизнес-вывод

Технический результат (`action: WATCH`, таблица PSI/KS) сам по себе ничего не решает — Head of Retention не читает таблицы статистик. Переведи его в решение на языке бизнеса:

- [ ] **Рекомендация.** Одна фраза действия: «Новый трафик уже виден в данных (доля бонусных игроков подросла), но дрейф пока умеренный — модель ещё рабочая. Не переобучаем сейчас, а ставим на усиленное наблюдение и готовим переобучение к моменту, когда сдвиг перейдёт порог».
- [ ] **Эффект в деньгах.** Свяжи дрейф с бюджетом: сдвиг `bonus_ratio` означает риск, что часть бонусов (порядка ~0.5 млн ₽/мес по нашей оценке при сильном дрейфе) начнёт уходить не той аудитории; контур наблюдаемости ловит этот момент в день, а не через квартал. Назови цифру, а не «PSI = 0.19».
- [ ] **Риски и допущения.** Дрейф входа — ранний сигнал, но он НЕ доказывает падение качества напрямую: финальное подтверждение даст фактический отток через 30 дней. WATCH — не «всё хорошо», а «следим внимательнее». Пороги триггера (PSI ≥ 0.25) — стартовый дефолт, калибруется по истории.
- [ ] **Следующий шаг.** Конкретное действие с владельцем и сроком: до старта залива нового гео поднять частоту запуска монитора (ежедневно), а на 30-дневном горизонте подключить мониторинг качества по ground truth (rolling AUC), чтобы ловить и концепт-дрейф, а не только дрейф входа.
- [ ] **Как подать стейкхолдеру.** Говори решениями, не метриками: «когда верить модели, а когда притормозить рассылку», «сколько бюджета на кону», «что делаем и когда» — не «PSI», «KS-статистика», «p-value».

## Развитие

1. **Догнать дрейф до RETRAIN.** Подними долю `drifted` в `gen_data.py` (60-70%) или добавь второй сдвинутый признак — увидишь, как монитор выдаст DRIFT, а триггер переключится на RETRAIN. Это упражнение на калибровку порогов под силу сдвига.
2. **Concept drift по ground truth.** Дотащи фактический исход (ушёл/не ушёл через 30 дней), считай rolling AUC/precision на свежих данных и триггерь переобучение по падению метрики, а не только по дрейфу входа.
3. **Готовый инструмент мониторинга.** Подключи Evidently или NannyML вместо самописного PSI — сравни их отчёты со своими цифрами, разберись, где расходятся и почему (другое бинование, бутстрап-доверительные интервалы).
4. **Метрики и дашборд.** Отдавай Prometheus-метрики (`/metrics`): латентность, RPS, доля churn_label=1, текущий max PSI. Построй Grafana-дашборд для наблюдаемости в реальном времени.
5. **Версионирование моделей и A/B.** Введи реестр моделей (MLflow), катай новую версию канареечно на 10% трафика, сравнивай метрики старой и новой перед полным переключением.
6. **CI/CD пайплайн.** GitHub Actions: на пуш — тесты эндпоинтов, сборка образа, пуш в registry; по cron — запуск `drift_trigger.py`, и при RETRAIN автоматический трен + деплой новой версии.

## Что ты закрепил

Ты прошёл путь от обученной модели до прод-подобного сервиса с самонаблюдением и связал воедино:

- **M22 (MLOps)** — сериализация артефакта-бандла, упаковка сервиса, контейнеризация, дрейф данных vs концепт-дрейф, мониторинг и автоматический триггер переобучения — стержень всего воркшопа.
- **M14 (supervised)** — `predict_proba` и порог решения как отдельная бизнес-настройка в контракте ответа.
- **M17 (подводные камни)** — дрейф распределений, разница между «значимо» (KS) и «велико» (PSI), почему дрейф входа не равен падению качества, и почему умеренный дрейф — это WATCH, а не паника.
- **W3 (churn-модель)** — артефакт из W3 встаёт сюда без переделок: это показывает, что обучение и деплой — разные слои с чётким контрактом между ними.

Главный навык, который ты унёс: **модель в проде — это не файл, а контракт** (схема входа, порядок признаков, версия, baseline) плюс контур наблюдаемости, который ловит деградацию до того, как её заметит бизнес — и трезво отличает «пора действовать» от «пора присмотреться».
</content>
</invoke>
