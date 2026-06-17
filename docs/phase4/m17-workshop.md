# Воркшоп M17 · Поймать утечку руками и объяснить драйверы

<span class="lecture-meta">Воркшоп к модулю M17 · ориентир 4-6 ч</span>

## Что отрабатываем

В модуле M17 главный тезис: красивая офлайн-метрика — повод не радоваться, а искать утечку. Этот воркшоп прогоняет тебя через полный цикл из теории на собственных руках:

- намеренно строим churn-модель с **target leakage** и **временной/групповой утечкой** и видим завышенный AUC (M17.3, M17.5);
- диагностируем её тремя приёмами из модуля — сравнение с **baseline**, **аблация** подозрительного признака, временной аудит (M17.7);
- чиним: оборачиваем препроцессинг в `Pipeline`, переходим на `GroupKFold` по `player_id`, убираем течь (M17.3, M17.5);
- интерпретируем честную модель через **permutation importance** на тесте и **SHAP** глобально и локально, сравнивая со смещённой встроенной важностью (M17.9, M17.10).

**Артефакт на выходе:** таблица «AUC до/после устранения утечки» с цифрами разрыва, beeswarm и waterfall SHAP, и короткий вывод о драйверах оттока с оговоркой «это объяснение модели, не каузальность».

Стек 2026: `uv`, `pandas`, `scikit-learn`, `lightgbm`, `shap`.

## Данные

Синтетика с временной структурой, повторяющимися игроками и встроенной утечкой. Генератор самодостаточен и воспроизводим (фиксированный seed). Каждый игрок порождает несколько недельных снимков — это даёт и группы (`player_id`), и время (`week`).

```bash
uv venv && source .venv/bin/activate
uv pip install pandas numpy scikit-learn lightgbm shap matplotlib
```

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_PLAYERS = 4000

base = pd.DataFrame({
    "player_id": np.arange(N_PLAYERS),
    "geo": rng.choice(["RU", "KZ", "UZ", "DE"], N_PLAYERS, p=[.4, .25, .2, .15]),
    "device": rng.choice(["mobile", "desktop"], N_PLAYERS, p=[.7, .3]),
    "affiliate": rng.choice([f"aff_{i}" for i in range(20)], N_PLAYERS),
    "skill": rng.normal(0, 1, N_PLAYERS),  # скрытая лояльность игрока
})

rows = []
for _, p in base.iterrows():
    churn_week = rng.integers(2, 12) if rng.random() < 0.35 else 99
    for week in range(1, 11):
        active = week < churn_week
        bets = max(0, rng.poisson(8 + 3 * p.skill) * (1.0 if active else 0.15))
        rows.append({
            "player_id": int(p.player_id), "week": week,
            "geo": p.geo, "device": p.device, "affiliate": p.affiliate,
            "bets_count": bets,
            "avg_stake": max(1, rng.normal(50 + 10 * p.skill, 15)),
            "deposit_freq": max(0, rng.normal(2 + p.skill, 1)),
            "session_len": max(1, rng.normal(20 + 5 * p.skill, 8)),
            # таргет: уйдёт ли игрок в течение 4 недель после этого снимка
            "churned": int(churn_week <= week + 4),
            # ЛОВУШКА: days_since_last_login — следствие таргета (M17.3, пример 1)
            "days_since_last_login": rng.integers(0, 3) if active else rng.integers(15, 40),
        })

df = pd.DataFrame(rows)
df.to_parquet("players.parquet")
print(df.shape, "| churn rate:", round(df.churned.mean(), 3))
```

Признак `days_since_last_login` вычислен из факта оттока — это тавтология. В момент скоринга активного игрока ты не знаешь его будущего, значит значения «15-40 дней» у тебя не будет. Это наш кандидат на target leakage.

## Ход работы

### Шаг 1: Модель с утечкой и завышенный AUC

**Зачем.** Воспроизводим типичную ошибку из M17.5 и M17.3 одновременно: случайный `KFold(shuffle=True)` (игрок попадает и в train, и в test — групповая утечка) плюс протекающий признак `days_since_last_login`. Смотрим, насколько красивой становится метрика.

```python
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_parquet("players.parquet")
cat = ["geo", "device", "affiliate"]
df[cat] = OrdinalEncoder().fit_transform(df[cat])

LEAKY = ["bets_count", "avg_stake", "deposit_freq", "session_len",
         "days_since_last_login"] + cat
X_leak, y = df[LEAKY], df["churned"]

model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           random_state=42, verbose=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_leak = cross_val_score(model, X_leak, y, cv=kf, scoring="roc_auc")
print(f"AUC (утечка, KFold shuffle): {auc_leak.mean():.3f}")
```

**Что получилось.** AUC порядка **0.97-0.99**. По меркам индустрии churn редко предсказывается лучше 0.75-0.82 — это первый звонок из M17.3. Профи здесь не радуется, а идёт искать утечку.

!!! question "Проверь себя"

    1. Здесь сразу две утечки. Какие именно и из какого пункта модуля каждая?
    2. Почему именно `days_since_last_login` течёт, а `bets_count` — пограничный, но скорее нет?

??? success "Ответы"

    1. Target leakage (`days_since_last_login` — следствие таргета, M17.3 вид 1) и групповая утечка (`KFold(shuffle=True)` кладёт строки одного `player_id` и в train, и в test, M17.5).
    2. Значение `days_since_last_login` напрямую вычислено из факта оттока — в момент скоринга активного игрока его нет. `bets_count` известен на момент снимка и легитимен, хотя коррелирует с лояльностью.

### Шаг 2: Диагностика — baseline и аблация

**Зачем.** Модуль M17.7 даёт два инструмента ловли утечки: baseline (если он подозрительно высок на одном признаке — флаг) и аблация (убрать подозреваемого и замерить просадку). Применяем оба.

```python
from sklearn.dummy import DummyClassifier

# Baseline на ОДНОМ подозрительном признаке — если высоко, это target leakage
auc_single = cross_val_score(model, df[["days_since_last_login"]], y,
                             cv=kf, scoring="roc_auc")
print(f"AUC только по days_since_last_login: {auc_single.mean():.3f}")

base = DummyClassifier(strategy="prior")
auc_base = cross_val_score(base, X_leak, y, cv=kf, scoring="roc_auc")
print(f"Baseline (prior): {auc_base.mean():.3f}")

# Аблация: та же схема, но без подозреваемого
NO_LEAK = ["bets_count", "avg_stake", "deposit_freq", "session_len"] + cat
auc_abl = cross_val_score(model, df[NO_LEAK], y, cv=kf, scoring="roc_auc")
print(f"AUC без days_since_last_login: {auc_abl.mean():.3f} | "
      f"просадка: {auc_leak.mean() - auc_abl.mean():.3f}")
```

**Что получилось.** Один признак `days_since_last_login` даёт AUC около **0.95** — почти весь результат держится на нём. Это прямое подтверждение target leakage по правилу M17.7: «baseline на одном подозрительном признаке даёт 0.94 — у тебя точно утечка». Аблация роняет метрику на 0.1-0.15. Диагноз поставлен.

### Шаг 3: Чинить — Pipeline + GroupKFold

**Зачем.** Устраняем обе течи разом (M17.3 главное правило + M17.5 «когда какой сплит»): выкидываем протекающий признак, оборачиваем препроцессинг в `Pipeline` (fit только на train-фолде), переходим на `GroupKFold` по `player_id`, чтобы игрок не пересекал границу train/test.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

df2 = pd.read_parquet("players.parquet")
num = ["bets_count", "avg_stake", "deposit_freq", "session_len"]
cat = ["geo", "device", "affiliate"]
X, y = df2[num + cat], df2["churned"]
groups = df2["player_id"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oe", OrdinalEncoder(handle_unknown="use_encoded_value",
                                            unknown_value=-1))]), cat),
])
clean = Pipeline([("pre", pre),
                  ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                             random_state=42, verbose=-1))])

gkf = GroupKFold(n_splits=5)
auc_clean = cross_val_score(clean, X, y, groups=groups, cv=gkf, scoring="roc_auc")
ap_clean = cross_val_score(clean, X, y, groups=groups, cv=gkf,
                           scoring="average_precision")
auc_base2 = cross_val_score(DummyClassifier(strategy="prior"), X, y,
                            groups=groups, cv=gkf, scoring="roc_auc")

print(f"Честный AUC (GroupKFold, без утечки): {auc_clean.mean():.3f} "
      f"+/- {auc_clean.std():.3f}")
print(f"PR AUC: {ap_clean.mean():.3f} | Baseline: {auc_base2.mean():.3f} | "
      f"lift: {auc_clean.mean() - auc_base2.mean():.3f}")
```

**Что получилось.** Честный AUC падает до **0.70-0.78** — это и есть реалистичный уровень churn из M17.3. Разрыв с шагом 1 (~0.2 AUC) — это ровно то «искусственное завышение», которое в проде превратилось бы в провал. Считаем `lift` над baseline: сложность модели оправдана только если прирост заметный (M17.7).

!!! question "Проверь себя"

    1. Что именно гарантирует `Pipeline` внутри `cross_val_score`, чего не давал ручной `fit` энкодера до сплита?
    2. Почему мы здесь смотрим ещё и PR AUC, а не только ROC AUC?

??? success "Ответы"

    1. На каждом train-фолде `imputer`/`scaler`/`encoder` обучаются только на train этого фолда; статистики теста не зашиваются в трансформацию (M17.3, утечка через препроцессинг).
    2. Классы несбалансированы (отток — редкий класс), и PR AUC честнее фокусируется на меньшинстве, тогда как ROC AUC оптимистична из-за массы истинных негативов (M17.6).

### Шаг 4: Сравнить со встроенной важностью и взять permutation importance

**Зачем.** M17.9: встроенная `feature_importances_` смещена к высокой кардинальности и считается на train. Покажем расхождение на честной модели и возьмём permutation importance на отложенной выборке как надёжный инструмент. Сплит делаем по группам вручную, чтобы тест был честным.

```python
from sklearn.inspection import permutation_importance

uniq = df2["player_id"].unique()
test_ids = set(rng.choice(uniq, size=int(len(uniq) * 0.2), replace=False))
te = df2["player_id"].isin(test_ids)
Xtr, Xte, ytr, yte = X[~te], X[te], y[~te], y[te]

clean.fit(Xtr, ytr)

# Встроенная важность (на train, смещённая)
built_in = pd.Series(clean.named_steps["clf"].feature_importances_,
                     index=num + cat).sort_values(ascending=False)

# Permutation importance на тесте, model-agnostic
pi = permutation_importance(clean, Xte, yte, scoring="roc_auc",
                            n_repeats=20, random_state=42)
perm = pd.Series(pi.importances_mean, index=num + cat).sort_values(ascending=False)

print("Встроенная (train):\n", built_in, "\n")
print("Permutation (test):\n", perm)
```

**Что получилось.** `affiliate` (20 уникальных значений) обычно раздут во встроенной важности из-за высокой кардинальности — это артефакт, а не сигнал (M17.9). В permutation importance на тесте наверх выходят реально предсказательные `bets_count`, `deposit_freq`, `session_len`. Найди признак с максимальным расхождением и объясни его кардинальностью.

### Шаг 5: SHAP — глобально и локально

**Зачем.** M17.10: SHAP — стандарт 2026. Читаем beeswarm для глобальной картины драйверов и waterfall для объяснения одного игрока. Для бустинга берём точный и быстрый `TreeExplainer`.

```python
import shap

fitted = clean.named_steps["clf"]
X_te_trans = clean.named_steps["pre"].transform(Xte)
feat_names = num + cat

explainer = shap.TreeExplainer(fitted)
sv = explainer(X_te_trans)
sv.feature_names = feat_names

shap.plots.beeswarm(sv, show=False)   # глобально: важность + направление
import matplotlib.pyplot as plt
plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=120); plt.clf()

shap.plots.waterfall(sv[0], show=False)  # локально: один игрок
plt.tight_layout(); plt.savefig("shap_waterfall.png", dpi=120)
```

**Что получилось.** На beeswarm видно и важность (разброс точек), и направление (цвет): низкая `bets_count` / `deposit_freq` толкает скор оттока вверх. Waterfall раскладывает скор одного игрока на вклады, которые в сумме с базовым значением дают предсказание (аддитивность из M17.10).

!!! question "Проверь себя"

    1. Формулируя вывод для продакта «низкая частота ставок повышает риск оттока», какую обязательную оговорку добавляешь?
    2. Почему permutation importance и SHAP надёжнее встроенной важности именно здесь?

??? success "Ответы"

    1. Это объяснение поведения МОДЕЛИ, а не каузальность: воздействие на признак не обязательно изменит отток. Каузальность — отдельная дисциплина (uplift, эксперименты), M17.10.
    2. Они считаются на отложенной выборке (обобщение, не запоминание train) и не смещены к высокой кардинальности `affiliate` (M17.9).

## Критерий готовности

- [ ] Сгенерирован воспроизводимый датасет с группами, временем и встроенным `days_since_last_login`
- [ ] Получен завышенный AUC (~0.97+) на `KFold(shuffle=True)` + протекающем признаке
- [ ] Утечка диагностирована: baseline на одном признаке высок, аблация роняет метрику
- [ ] Построен честный пайплайн (`Pipeline` + `GroupKFold`), AUC упал до реалистичных ~0.70-0.78
- [ ] Посчитан lift над `DummyClassifier` и PR AUC рядом с ROC AUC
- [ ] Встроенная важность сопоставлена с permutation importance, найден признак-артефакт кардинальности
- [ ] Построены SHAP beeswarm (глобально) и waterfall (локально) с оговоркой про не-каузальность
- [ ] Записана таблица «до/после» с числом разрыва как ценой утечки

## Развитие

1. **Временная утечка отдельно.** Замени группировку на сплит по `week` (train: недели 1-7, test: 8-10) и сравни с `KFold(shuffle=True)` — покажи, что шаффл во времени тоже завышает (M17.5).
2. **Калибровка.** Оберни честную модель в `CalibratedClassifierCV(method="isotonic")`, построй reliability diagram до/после и посчитай, как меняется ожидаемый LTV-под-риском при калиброванных вероятностях (M17.11).
3. **Learning curve.** Построй `learning_curve` для честной модели и классифицируй форму: переобучение, недообучение или баланс — и обоснуй лечение (M17.4).
4. **LLM-аудит.** Скорми код шага 1 Claude с промптом из M17.14 про поиск четырёх видов утечки и проверь, поймает ли он `days_since_last_login` и шаффл-сплит — затем перепроверь его выбор метрики руками.
