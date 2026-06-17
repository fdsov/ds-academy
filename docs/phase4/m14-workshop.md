# Воркшоп M14 · Турнир моделей на одном churn-датасете

<span class="lecture-meta">Воркшоп к модулю M14 · ориентир 5-7 ч</span>

## Что отрабатываем

Модуль M14 утверждает простую вещь: на табличных данных в 2026 нет «лучшей модели вообще» — есть лестница от честного линейного baseline до градиентного бустинга, и профессионал проходит её целиком, чтобы понять, что реально даёт прирост. В этом воркшопе ты пройдёшь её руками на одной churn-задаче.

Понятия модуля, которые отрабатываем напрямую:

- Логистическая регрессия с L1-регуляризацией и масштабированием (M14.4, M14.5) — baseline и интерпретация коэффициентов через **odds ratio** $e^{w_j}$.
- Одиночное дерево решений и его высокий разброс (M14.8).
- Random Forest и **декорреляция** деревьев (M14.9).
- Градиентный бустинг LightGBM как leaf-wise «король табличных данных» (M14.10).
- Метрики при дисбалансе: **PR-AUC вместо accuracy**, выбор **порога под precision** (M14.5).
- Честная важность признаков: встроенная `feature_importances_` против **permutation importance** (M14.10).

Артефакт на выходе: единая таблица сравнения четырёх моделей (ROC-AUC, PR-AUC, recall при precision ≥ 0.6) плюс письменные выводы — какая модель и почему выигрывает, и что говорят коэффициенты логрега и важности бустинга про драйверы оттока.

## Данные

Генерируем реалистичный гемблинг-датасет: одна строка на игрока, бинарный таргет `churn_14d` (~12% оттока, дисбаланс), нелинейные связи и шум. Зависимость зашита явно, чтобы потом проверять, поймали ли её модели.

```bash
uv init m14-workshop && cd m14-workshop
uv add pandas numpy scikit-learn lightgbm
```

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 12_000

first_deposit = rng.gamma(2.0, 40, n)
bets_day1 = rng.poisson(8, n)
sessions_7d = rng.poisson(5, n)
days_since_last_deposit = rng.exponential(6, n)
avg_bet = rng.gamma(2.0, 3, n)
ggr_30d = rng.normal(50, 80, n)
country = rng.choice(["RU", "KZ", "UZ", "DE", "BR"], n, p=[.35, .2, .15, .15, .15])
device = rng.choice(["android", "ios", "web"], n, p=[.55, .25, .2])

# log-odds оттока: растёт от простоя, падает от активности (нелинейно)
z = (
    -1.6
    + 0.16 * days_since_last_deposit
    - 0.22 * sessions_7d
    - 0.012 * first_deposit
    + 0.5 * np.tanh((avg_bet - 6) / 3)
    + np.where(country == "BR", 0.6, 0.0)
    + rng.normal(0, 0.8, n)
)
p = 1 / (1 + np.exp(-z))
churn_14d = rng.binomial(1, p)

df = pd.DataFrame({
    "first_deposit": first_deposit, "bets_day1": bets_day1,
    "sessions_7d": sessions_7d, "days_since_last_deposit": days_since_last_deposit,
    "avg_bet": avg_bet, "ggr_30d": ggr_30d,
    "country": country, "device": device, "churn_14d": churn_14d,
})
print(df["churn_14d"].mean().round(3), "доля оттока")
```

!!! warning "Никакой утечки таргета"

    Все признаки посчитаны на момент строго до окна прогноза. `days_since_last_deposit` здесь — снимок на точку отсчёта, а не «дней до момента, когда игрок уже ушёл». Это ровно та ловушка из M14.11, что даёт фантастический AUC и провал в проде.

## Ход работы

### Шаг 1: Сплит и общий протокол оценки

**Зачем.** Модуль настаивает: сравнение честно только при едином сплите и стратификации (дисбаланс классов). Все четыре модели увидят один и тот же `X_train`/`X_valid`.

```python
from sklearn.model_selection import train_test_split

features = ["first_deposit", "bets_day1", "sessions_7d", "days_since_last_deposit",
           "avg_bet", "ggr_30d", "country", "device"]
cat_features = ["country", "device"]
num_features = [f for f in features if f not in cat_features]

X, y = df[features], df["churn_14d"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
```

Заводим единую функцию оценки — она и наполнит таблицу сравнения.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def evaluate(name, proba, target_precision=0.60):
    roc = roc_auc_score(y_valid, proba)
    pr = average_precision_score(y_valid, proba)
    prec, rec, thr = precision_recall_curve(y_valid, proba)
    mask = prec[:-1] >= target_precision
    rec_at_p = rec[:-1][mask].max() if mask.any() else 0.0
    return {"model": name, "ROC_AUC": round(roc, 4),
            "PR_AUC": round(pr, 4), "recall@P0.6": round(float(rec_at_p), 4)}

results = []
```

**Что получилось.** Готов протокол: один сплит, три метрики на модель. `recall@P0.6` — это бизнес-вопрос «сколько реального оттока поймаем, держа точность тревог не ниже 60%».

### Шаг 2: Логистическая регрессия с L1 — честный baseline

**Зачем.** M14.11 прямо требует: начинай с регуляризованного логрега. Если бустинг не обыграет его заметно — проблема в данных, не в модели. L1 (Lasso) ещё и отберёт признаки. Масштабирование обязательно — регуляризация штрафует веса, а вес зависит от масштаба (M14.4).

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])
logreg = make_pipeline(
    pre,
    LogisticRegression(penalty="l1", solver="liblinear", C=0.5, class_weight="balanced"),
)
logreg.fit(X_train, y_train)
results.append(evaluate("LogReg L1", logreg.predict_proba(X_valid)[:, 1]))
```

Препроцессинг живёт внутри `Pipeline` и `fit` видит только `X_train` — никакой утечки через scaler.

**Что получилось.** Baseline в таблице. На этих данных ожидай ROC-AUC порядка 0.80-0.85 — линейная модель ловит монотонную часть связи, но `tanh` по `avg_bet` ей недоступен.

!!! question "Проверь себя"

    1. Почему `StandardScaler` нельзя «фитить» на всей выборке до сплита?
    2. Зачем здесь `class_weight="balanced"`, если классы 12/88?

??? success "Ответы"

    1. Это утечка через препроцессинг (M14.11): статистики среднего/дисперсии теста просочатся в обучение, метрика на валидации завысится. Скейлер должен учиться только на train — поэтому он внутри Pipeline.
    2. При дисбалансе модель без взвешивания тянет всё к мажоритарному классу (как accuracy-99% модель «всегда не фрод»). Балансировка повышает вклад редкого класса оттока в log-loss.

### Шаг 3: Интерпретация коэффициентов через odds ratio

**Зачем.** Это то, что M14.5 называет признаком аналитика-профессионала: перевести вес в язык продакта. Коэффициент $w_j$ → odds ratio $e^{w_j}$ → «во сколько раз меняются шансы оттока».

```python
clf = logreg.named_steps["logisticregression"]
names = (num_features +
         list(logreg.named_steps["columntransformer"]
              .named_transformers_["cat"].get_feature_names_out(cat_features)))
coef = pd.Series(clf.coef_[0], index=names)
odds = pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)}) \
         .sort_values("odds_ratio", ascending=False)
print(odds.round(3))
```

Признаки масштабированы, поэтому odds ratio здесь — эффект сдвига на одно стандартное отклонение. Какие веса L1 занулил (отбор признаков Lasso) — увидишь сразу: `coef == 0`.

**Что получилось.** Формулировка на языке продакта, например: «рост `days_since_last_deposit` на одно std умножает шансы оттока примерно в 1.6 раза». Запиши 2-3 такие фразы — они пойдут в выводы.

### Шаг 4: Дерево и Random Forest — от разброса к декорреляции

**Зачем.** M14.8-M14.9: одиночное дерево нестабильно (высокий разброс), Random Forest гасит это усреднением плюс **декорреляцией** через случайный отбор признаков в узлах. Деревьям масштабирование и one-hot не нужны — но sklearn не ест строковые категории, поэтому кодируем порядковыми кодами.

```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_train[cat_features])
def encode(d):
    out = d.copy()
    out[cat_features] = enc.transform(d[cat_features])
    return out
Xtr, Xvl = encode(X_train), encode(X_valid)

tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42).fit(Xtr, y_train)
results.append(evaluate("DecisionTree d4", tree.predict_proba(Xvl)[:, 1]))

rf = RandomForestClassifier(
    n_estimators=400, max_features="sqrt", min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=42,
).fit(Xtr, y_train)
results.append(evaluate("RandomForest", rf.predict_proba(Xvl)[:, 1]))
```

`max_features="sqrt"` — это и есть источник декорреляции из M14.9: каждый узел выбирает разбиение из случайного подмножества признаков, ломая доминирование `days_since_last_deposit`.

**Что получилось.** Дерево заметно слабее (ограничено глубиной, иначе переобучится), Random Forest подтягивается к бустингу. Разрыв «одно дерево → лес» — это и есть наглядная цена разброса.

!!! question "Проверь себя"

    1. Что произойдёт с метрикой дерева на валидации, если убрать `max_depth`?
    2. Почему `max_features="sqrt"` важнее для леса, чем число деревьев после некоторого порога?

??? success "Ответы"

    1. Дерево вырастит лист почти под каждый объект, запомнит шум: train-AUC уйдёт к 1.0, valid-AUC просядет — классическое переобучение (M14.8).
    2. Разброс ансамбля упирается в потолок $\rho\sigma^2$; добавление деревьев убирает только слагаемое $\frac{1-\rho}{B}\sigma^2$. Пробить потолок можно лишь снижением корреляции $\rho$ — это и делает отбор признаков (M14.9).

### Шаг 5: LightGBM — leaf-wise бустинг

**Зачем.** M14.10: бустинг строит деревья последовательно, каждое чинит ошибки предыдущих. LightGBM растит деревья **по листьям** (leaf-wise) и быстр на больших выборках; расплата — агрессивнее переобучается, поэтому контролируем `num_leaves` и `min_child_samples`. Категории отдаём нативно (без one-hot).

```python
import lightgbm as lgb

Xtr_c, Xvl_c = X_train.copy(), X_valid.copy()
for c in cat_features:
    Xtr_c[c] = Xtr_c[c].astype("category")
    Xvl_c[c] = pd.Categorical(Xvl_c[c], categories=Xtr_c[c].cat.categories)

lgbm = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.03, num_leaves=31,
    min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=42, verbose=-1,
)
lgbm.fit(
    Xtr_c, y_train, eval_set=[(Xvl_c, y_valid)], eval_metric="auc",
    categorical_feature=cat_features,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
results.append(evaluate("LightGBM", lgbm.predict_proba(Xvl_c)[:, 1]))
```

**Что получилось.** Early stopping сам подобрал число деревьев. Ожидай лучший PR-AUC из четырёх — бустинг ловит и `tanh`-нелинейность, и взаимодействие страны с активностью.

### Шаг 6: Таблица сравнения и честная важность

**Зачем.** Финальный артефакт. И сразу проверяем тезис M14.10: встроенная важность смещена, честнее **permutation importance** на валидации.

```python
from sklearn.inspection import permutation_importance

table = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
print(table.to_string(index=False))

perm = permutation_importance(lgbm, Xvl_c, y_valid, scoring="average_precision",
                              n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.DataFrame({
    "feature": features,
    "builtin_gain": lgbm.feature_importances_,
    "permutation": perm.importances_mean,
}).sort_values("permutation", ascending=False)
print(imp.round(4).to_string(index=False))
```

**Что получилось.** Таблица сравнения готова — это и есть артефакт. Сравни два столбца важности: `builtin_gain` любит признаки с многими уникальными значениями (`ggr_30d`, `first_deposit`), а permutation честно показывает, что метрику реально держат `days_since_last_deposit` и `sessions_7d` — ровно то, что мы зашили в генератор.

!!! question "Проверь себя"

    1. Почему сравнение моделей ведём по PR-AUC, а не по ROC-AUC?
    2. Какой признак встроенная важность может переоценить и почему?

??? success "Ответы"

    1. При дисбалансе ROC-AUC оптимистичен (M14.5): он усредняет по всем порогам, включая бесполезную зону высокого recall. PR-AUC и `recall@P0.6` отражают реальную пользу — поймать отток, не утопив precision.
    2. Непрерывные `ggr_30d`/`first_deposit`: по ним много уникальных значений и потенциальных сплитов, gain-важность раздувается, хотя предсказательной силы меньше, чем у простоя (M14.10).

## Критерий готовности

- [ ] Синтетика генерируется с фиксированным seed, доля оттока ~0.12, запускается у любого через `uv`.
- [ ] Все четыре модели обучены на одном стратифицированном сплите, без утечки через препроцессинг.
- [ ] Логрег масштабирует числовые признаки и использует L1; видно, какие веса занулены.
- [ ] Коэффициенты логрега переведены в odds ratio и сформулированы фразой на языке продакта.
- [ ] Дерево, Random Forest и LightGBM настроены против переобучения (`max_depth` / `min_samples_leaf` / `num_leaves`+`min_child_samples`).
- [ ] Собрана таблица сравнения: ROC-AUC, PR-AUC, recall при precision ≥ 0.6 для всех моделей.
- [ ] Построены два столбца важности (builtin vs permutation), расхождение объяснено.
- [ ] Написаны выводы: какая модель победила по PR-AUC и почему, и 2-3 драйвера оттока.

## Развитие

- Добавь временной сплит вместо случайного: пометь часть игроков «поздней когортой» и проверь, насколько просядут метрики — это честная продуктовая валидация (M14.11).
- Внедри утечку таргета (признак из окна прогноза), зафиксируй неправдоподобный AUC, затем убери её и покажи падение до честного уровня — тренировка детектора «слишком хорошего» результата.
- Откалибруй вероятности LightGBM через `CalibratedClassifierCV` (изотоническая) и сравни reliability-кривую до/после — критично, если скор идёт в денежное решение про ретеншн-бюджет.
- Замени LightGBM на CatBoost с нативными категориями и `auto_class_weights="Balanced"`, сравни по тем же трём метрикам и времени обучения — выбор из «тройки королей» по осям из M14.10.
