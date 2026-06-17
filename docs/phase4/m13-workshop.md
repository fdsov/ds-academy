# Воркшоп M13 · Честный каркас оценки ML-модели

<span class="lecture-meta">Воркшоп к модулю M13 · ориентир 4-6 ч</span>

## Что отрабатываем

В модуле M13 главная мысль звучала так: модель ломается не там, где её обучают, а там, где её оценивают. Accuracy 0.97 на churn ничего не значит, random k-fold на временных данных заглядывает в будущее, скейлер до сплита протекает из test в train, а дефолтный порог 0.5 на дисбалансе почти всегда мимо. Этот воркшоп — не про обучение мощной модели, а про сборку воспроизводимого каркаса честной оценки, в котором ни одна из этих ловушек не пройдёт незамеченной.

Отрабатываем конкретные понятия модуля:

- разбиение train/validation/test и роль каждой части (M13.6);
- стратифицированная кросс-валидация и time-series split (M13.9);
- baseline как точка отсчёта (M13.14, шаг 4);
- метрики при дисбалансе: PR-AUC, precision, recall, confusion matrix, и почему accuracy и ROC-AUC льстят (M13.11, M13.12);
- подбор порога под цену ошибок на validation, а не на test (M13.11);
- `Pipeline` как защита от утечки препроцессинга (M13.13, M13.15).

Артефакт на выходе: один скрипт `eval_skeleton.py`, который генерирует данные, делает честный сплит, гоняет baseline и две модели через стратифицированную CV, печатает правильные метрики, подбирает порог под бизнес-цену ошибок и в конце демонстрирует на цифрах, как утечка и random k-fold завышают оценку. Этот скелет переиспользуется в любой реальной задаче — меняется только источник данных.

## Данные

Самодостаточный синтетический churn-датасет с фиксированным seed. Положительный класс (ушедшие) редкий — около 8%, чтобы дисбаланс был настоящим. Добавляем дату регистрации, чтобы был временной разрез для time-series split.

```python
import numpy as np
import pandas as pd

def make_churn(n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    reg_day = rng.integers(0, 180, n)
    df = pd.DataFrame({
        "deposits_7d": rng.poisson(2, n),
        "avg_bet": rng.gamma(2, 5, n),
        "days_since_reg": rng.integers(1, 400, n),
        "geo": rng.choice(["UZ", "KZ", "RU", "TR"], n),
        "reg_date": pd.Timestamp("2026-01-01") + pd.to_timedelta(reg_day, unit="D"),
    })
    logit = -2.4 - 0.5 * df["deposits_7d"] + 0.003 * df["days_since_reg"]
    p = 1 / (1 + np.exp(-logit))
    df["churn"] = (rng.random(n) < p).astype(int)
    return df.sort_values("reg_date").reset_index(drop=True)

df = make_churn()
print("Строк:", len(df), "| доля положительного класса:", round(df["churn"].mean(), 3))
```

Окружение через uv (стек 2026):

```bash
uv venv && source .venv/bin/activate
uv pip install "scikit-learn>=1.7" "pandas>=2.2" lightgbm numpy
```

## Ход работы

### Шаг 1: Честный сплит train/test со стратификацией

**Зачем.** M13.6: test — запечатанный конверт, который вскрывают один раз. Сплит делаем стратифицированным (`stratify=y`), чтобы доля редкого класса в train и test совпадала с генеральной — иначе на 8% можно случайно получить test, где фродеров вдвое меньше, и оценка соврёт.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["churn", "reg_date"])
y = df["churn"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("train доля:", round(y_tr.mean(), 3), "| test доля:", round(y_te.mean(), 3))
```

**Что получилось.** Две доли почти совпадают (около 0.08). Test откладываем и до самого конца к нему не прикасаемся — весь подбор идёт на train через CV.

### Шаг 2: Baseline — точка отсчёта

**Зачем.** M13.14, шаг 4: без baseline любая метрика повисает в воздухе. `DummyClassifier(strategy="most_frequent")` — это и есть та самая модель, которая всегда говорит "не уйдёт". Она даст высокую accuracy и нулевой recall — наглядная иллюстрация, почему accuracy на дисбалансе бесполезна (M13.11).

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
dummy_pred = dummy.predict(X_te)
print("Baseline accuracy:", round(accuracy_score(y_te, dummy_pred), 3))
print("Baseline recall  :", round(recall_score(y_te, dummy_pred), 3))
```

**Что получилось.** Accuracy около 0.92 (выглядит прилично), recall ровно 0.0 — модель не поймала ни одного ушедшего. Любую содержательную модель сравниваем не с нулём, а с этим baseline, и не по accuracy.

!!! question "Проверь себя"

    1. Почему baseline `most_frequent` показывает высокую accuracy, но нулевой recall?
    2. Что в baseline-сравнении заменяет accuracy как осмысленную метрику?

??? success "Ответы"

    1. Положительный класс редкий (8%), поэтому, предсказывая всегда большинство ("не уйдёт"), модель попадает в ~92% случаев — это и есть accuracy. Но среди реальных ушедших она не нашла никого, recall = TP/(TP+FN) = 0.
    2. PR-AUC и recall на положительном классе. Baseline PR-AUC примерно равен доле положительного класса (~0.08) — с этим порогом и сравниваем содержательные модели.

### Шаг 3: Pipeline без утечки + честная CV

**Зачем.** M13.13 и M13.15: весь препроцессинг (импьютер, скейлер, энкодер) обучается на данных, значит должен жить внутри `Pipeline` и фититься только на train-части каждого CV-фолда. Иначе статистики "узнают" про валидацию. CV делаем стратифицированной (`StratifiedKFold`), скоринг — `average_precision`, потому что это и есть PR-AUC (M13.11).

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

num = ["deposits_7d", "avg_bet", "days_since_reg"]
cat = ["geo"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

logreg = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])
gbm = Pipeline([("pre", pre),
                ("clf", LGBMClassifier(class_weight="balanced", n_estimators=300,
                                       learning_rate=0.05, verbose=-1, random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in [("LogReg", logreg), ("LightGBM", gbm)]:
    pr = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="average_precision")
    roc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"{name:9s} PR-AUC={pr.mean():.3f}±{pr.std():.3f}  ROC-AUC={roc.mean():.3f}")
```

**Что получилось.** Обе модели заметно выше baseline PR-AUC ≈ 0.08. ROC-AUC выглядит высоким у обеих (часто 0.7+), но именно PR-AUC отделяет реально полезную модель от посредственной — это центральный урок про дисбаланс. Сравнение валидно, потому что обе модели прошли один и тот же сплит, одни и те же фолды и одну метрику.

!!! question "Проверь себя"

    1. Почему `StandardScaler` стоит внутри `Pipeline`, а не применяется к `X_tr` заранее?
    2. Почему сравнивать LogReg и LightGBM по разным метрикам или на разных сплитах нельзя?

??? success "Ответы"

    1. Внутри CV каждый фолд заново делит train на под-train и под-validation. Если скейлер обучить заранее на всём `X_tr`, он впитает статистику под-validation каждого фолда — это утечка. В Pipeline `fit` скейлера происходит только на под-train фолда.
    2. Метрики несопоставимы между собой (ROC-AUC и PR-AUC на одной модели дают разные числа), а разные сплиты дают разный шум. Корректное сравнение требует фиксированных `random_state`, одинаковых фолдов и одной метрики.

### Шаг 4: Финальная оценка на test и подбор порога на validation

**Зачем.** M13.11: модель выдаёт вероятность, решение требует порога, и подбирать его надо на validation, а не на test. Дефолтные 0.5 на дисбалансе почти всегда мимо. Порог — бизнес-рычаг: задаём цену ошибок (FP бесит честного игрока, FN — упущенный отток, который стоит денег) и ищем порог, минимизирующий суммарную стоимость. Делаем это на отдельном validation-куске, выделенном из train, а test трогаем ровно один раз в самом конце.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score

X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=1
)
gbm.fit(X_fit, y_fit)

val_proba = gbm.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_proba)

cost_fp, cost_fn = 1.0, 8.0          # FN дороже: упущенный отток
best_t, best_cost = 0.5, float("inf")
for t in np.linspace(0.05, 0.95, 91):
    pred = (val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    cost = cost_fp * fp + cost_fn * fn
    if cost < best_cost:
        best_cost, best_t = cost, t
print("Порог под минимум стоимости (validation):", round(best_t, 2))

gbm.fit(X_tr, y_tr)                   # дообучаем на полном train
te_proba = gbm.predict_proba(X_te)[:, 1]
print("TEST ROC-AUC:", round(roc_auc_score(y_te, te_proba), 3))
print("TEST PR-AUC :", round(average_precision_score(y_te, te_proba), 3))

te_pred = (te_proba >= best_t).astype(int)
print(confusion_matrix(y_te, te_pred))
print(classification_report(y_te, te_pred, digits=3))
```

**Что получилось.** Подобранный порог обычно заметно ниже 0.5 (мы готовы ловить больше ушедших ценой ложных тревог, потому что FN дороже FP в 8 раз). Confusion matrix на test показывает реальный расклад: сколько оттока поймали (TP), скольких упустили (FN), скольких честных задели (FP). Test оценили один раз — конверт вскрыт.

!!! question "Проверь себя"

    1. Почему порог подбирается на validation, а не на test?
    2. Как соотношение `cost_fn` к `cost_fp` сдвигает оптимальный порог?

??? success "Ответы"

    1. Подбор порога — это решение, основанное на данных. Если подбирать его на test, мы протечём информацию из test в модель, и финальная оценка станет оптимистично смещённой — test перестанет быть честным экзаменом.
    2. Чем дороже FN относительно FP, тем ниже оптимальный порог: мы охотнее срабатываем (растёт recall, ловим больше ушедших), мирясь с ростом ложных тревог. Если бы дороже был FP, порог уехал бы вверх ради precision.

### Шаг 5: Демонстрация двух ловушек на цифрах

**Зачем.** M13.13 и M13.9: лучший способ запомнить ловушку — измерить, насколько она врёт. Сначала утечка препроцессинга (скейлер на всём датасете до сплита), затем random k-fold на временных данных против `TimeSeriesSplit`.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# (а) утечка: скейлим ВСЁ до сплита
X_num = df[num].copy()
X_leak = StandardScaler().fit_transform(X_num)           # fit на всём датасете
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
    X_leak, y, test_size=0.2, stratify=y, random_state=42)
leak_model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xl_tr, yl_tr)
leak_ap = average_precision_score(yl_te, leak_model.predict_proba(Xl_te)[:, 1])
print("PR-AUC с утечкой скейлера :", round(leak_ap, 3))
print("PR-AUC честный Pipeline   : см. шаг 3 (cross_val)")

# (б) random k-fold против time-series split на данных, упорядоченных по времени
X_time = df.drop(columns=["churn", "reg_date"])
random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ts_cv = TimeSeriesSplit(n_splits=5)
ap_random = cross_val_score(gbm, X_time, y, cv=random_cv, scoring="average_precision")
ap_ts = cross_val_score(gbm, X_time, y, cv=ts_cv, scoring="average_precision")
print("PR-AUC random k-fold :", round(ap_random.mean(), 3))
print("PR-AUC time-series   :", round(ap_ts.mean(), 3))
```

**Что получилось.** Утечка скейлера в этом мягком случае двигает метрику немного, но на признаках с тяжёлыми хвостами или таргет-кодировании разрыв становится драматичным — принцип "всё, что учится на данных, только в Pipeline" не обсуждается. Random k-fold даёт оценку оптимистичнее time-series split, потому что заглядывает в будущее: ближе к реальному поведению в проде именно time-series число. Запомни направление смещения: обе ловушки завышают оценку, и в бою ты получаешь меньше обещанного.

!!! question "Проверь себя"

    1. В какую сторону обе ловушки смещают оценку и почему это особенно опасно?
    2. Какая из двух CV-схем ближе к реальному поведению модели в проде на временных данных?

??? success "Ответы"

    1. Обе завышают качество: утечка даёт модели подсмотреть валидацию, random k-fold обучает на будущем. Опасно, потому что смещение оптимистичное — ты приходишь к продакту с красивой цифрой, а в проде модель работает хуже, и доверие теряется.
    2. `TimeSeriesSplit`: он всегда обучается на прошлом и валидируется на следующем окне, ровно как модель будет работать в проде. Random k-fold смешивает времена и заглядывает в будущее, давая завышенную фантазию.

## Критерий готовности

- [ ] Данные генерируются одним вызовом с фиксированным seed, доля положительного класса около 0.08
- [ ] Сплит train/test стратифицирован, доли классов совпадают, test не используется при подборе
- [ ] Есть baseline (`DummyClassifier`), показан его высокий accuracy при нулевом recall
- [ ] Весь препроцессинг внутри `Pipeline`, CV стратифицированная, скоринг `average_precision` (PR-AUC)
- [ ] Две модели сравниваются на одних фолдах и одной метрике с фиксированным `random_state`
- [ ] Порог подобран на отдельном validation под заданную цену FP/FN, не на test
- [ ] Test оценён ровно один раз: ROC-AUC, PR-AUC, confusion matrix, classification_report
- [ ] Численно показано завышение метрики от утечки скейлера и от random k-fold против time-series split
- [ ] Всё собрано в один воспроизводимый скрипт `eval_skeleton.py`

## Развитие

1. Замени ручной перебор порога на оптимизацию ожидаемой стоимости через `precision_recall_curve` напрямую по массивам `prec, rec, thr`, и построй график "стоимость от порога", отметив минимум.
2. Добавь `imbalanced-learn`: вставь `SMOTE` в `imblearn.Pipeline` так, чтобы ресэмплинг происходил только внутри train-части каждого фолда, и сравни PR-AUC с вариантом `class_weight="balanced"`. Проверь, что SMOTE до сплита (намеренная утечка) завышает метрику.
3. Заверни подбор гиперпараметров LightGBM в `GridSearchCV`/`RandomizedSearchCV` с тем же `StratifiedKFold` и `scoring="average_precision"`, убедившись, что test по-прежнему трогается один раз.
4. Сделай метрики стабильными: прогони весь скелет на трёх разных seed генерации данных и выведи среднее и разброс PR-AUC — это покажет variance оценки и научит не доверять одному прогону.
