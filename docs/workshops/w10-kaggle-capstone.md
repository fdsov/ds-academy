# W10 · Kaggle capstone end-to-end

<span class="lecture-meta">Воркшоп · ориентир 10-20 ч · Capstone</span>

## Что ты построишь

Ты пройдёшь табличное Kaggle-соревнование целиком: от пустой папки до позиции в leaderboard и честного solution write-up. Не «обучил модель в ноутбуке», а полный соревновательный цикл — тот самый, который отличает человека, умеющего делать ML, от человека, умеющего запускать чужие ячейки.

На выходе четыре артефакта:

1. **Воспроизводимый пайплайн** `pipeline.py` — один файл, который из сырых данных собирает признаки, гоняет кросс-валидацию, обучает финальную модель и пишет `submission.csv`. Запускается одной командой и даёт один и тот же результат на одном seed.
2. **Надёжная схема кросс-валидации**, чей локальный скор коррелирует с public leaderboard. Это центральное умение всего воркшопа: если твой local CV двигается в ту же сторону, что и LB, — у тебя есть приборная панель. Если нет — ты летишь вслепую.
3. **Сабмит в leaderboard** реального (или симулированного локально) соревнования и зафиксированная позиция.
4. **Solution write-up** — разбор: что сработало, что нет, чему научился, и почему public-скор может разойтись с private (shake-up).

Соревнование выбираем **табличное, бинарная классификация с дисбалансом** — это самый частый формат в индустрии и в гемблинге (предсказать конверсию, фрод, отток). Метрика — **ROC-AUC**, потому что она устойчива к дисбалансу классов и совпадает с метрикой большинства таких соревнований.

!!! info "Почему capstone, а не ещё один туториал"

    В предыдущих воркшопах данные были «причёсаны» под задачу. Здесь — нет. Соревнование моделирует реальность: ты не знаешь заранее, какие признаки важны, насколько train похож на test, где спрятан leak и переобучишься ли ты на public-части. Ценность не в финальном AUC, а в дисциплине: построить валидацию, которой можно верить, и принимать решения по ней, а не по красивому числу на public LB.

## Бизнес-кейс

Ты — data scientist в команде iGaming-продукта. Соревновательная механика этого воркшопа (табличная бинарная классификация с дисбалансом и метрикой ROC-AUC) — это ровно тот формат, в котором продукт скорит игроков: вероятность оттока, фрода или конверсии. Возьмём как сквозной сюжет **скоринг оттока для удержания**: модель ранжирует игроков по вероятности уйти, и retention-команда обзванивает/бонусит топ списка.

!!! example "Ситуация"

    К тебе приходит **Head of Retention**. У него проблема: бюджет на удержание ограничен, а текущая «модель» — это эвристика «не заходил 7 дней → шлём бонус веером всем подряд». Деньги на бонусы сгорают, а отток в когортах за последний квартал ориентировочно вырос с 12% до ~14% активной базы.

    - **Что он просит:** ранжированный список игроков по риску оттока, чтобы тратить retention-бюджет на тех, кто действительно на грани, а не на всю базу. Метрика качества — ранжирование (AUC): важно не точное число вероятности, а правильный порядок «кому звонить первым».
    - **Цена решения иллюстративно:** ежемесячный retention-бюджет ~3 000 000 ₽ на бонусы. Веерная раздача конвертит слабо; адресная работа по топ-20% риска при том же бюджете может удержать ориентировочно на 1.5–2x больше игроков. При среднем вкладе удержанного игрока ~8 000 ₽/мес это десятки миллионов ₽ годовой выручки, которую сейчас теряют или жгут вхолостую.
    - **Что зависит от ответа:** запускать ли адресную retention-кампанию на основе скоров и какой порог риска брать. Плохая модель = бюджет снова уходит в шум; хорошая = тот же бюджет даёт кратно больше удержаний.
    - **Ограничение:** 2 недели на пилот; доступны только исторические поведенческие признаки до точки отсечки (никакого «подсматривания» в будущее — это прямой аналог target leak из воркшопа).

## Предпосылки

Основной модуль — **M29 (Kaggle и Hugging Face)**: как устроены соревнования, train/test split, public vs private leaderboard, формат сабмита, kaggle CLI и аутентификация по API-токену.

Нужна вся **Фаза 4** курса:

- **M14 — supervised ML**: что такое `predict_proba`, как работает бинарная классификация.
- **M15 — метрики**: ROC-AUC, почему она, а не accuracy, при дисбалансе.
- **M16 — кросс-валидация**: K-fold, stratified, out-of-fold предсказания (на это опираемся постоянно — см. модуль M16).
- **M17 — подводные камни**: data leakage, target leak, переобучение на валидации.
- **M18 — градиентный бустинг**: LightGBM/XGBoost, ключевые гиперпараметры.

Окружение через **uv** (стандарт 2026):

```bash
uv init kaggle-capstone && cd kaggle-capstone
uv add pandas polars numpy scikit-learn lightgbm xgboost optuna matplotlib
uv add kaggle                 # официальный CLI для скачивания данных и сабмита
uv add --dev jupyter ipykernel
```

Версии, на которых собран воркшоп: Python 3.12, pandas 2.2+, polars 1.x, scikit-learn 1.5+, lightgbm 4.x, optuna 4.x, kaggle 1.6+.

!!! tip "Почему uv и фиксация seed обязательны в соревновании"

    Соревнование — это серия экспериментов, где ты сравниваешь варианты по третьему знаку AUC. Если между запусками плывёт версия sklearn или seed фолдов — ты сравниваешь шум, а не идеи. `uv.lock` замораживает версии, единый `SEED` замораживает разбиение и инициализацию модели. Только так разница «фича помогла / не помогла» становится сигналом, а не случайностью.

## Данные

Нужен табличный датасет для бинарной классификации с дисбалансом. Покажу оба пути: реальное соревнование через kaggle API и **запасной синтетический генератор**, чтобы пайплайн запускался даже без аккаунта Kaggle.

### Реальное соревнование (рекомендуется)

Хорошо подходят табличные playground-соревнования с метрикой AUC и понятным дисбалансом:

- **Playground Series** (ежемесячные, slug вида `playground-series-s6e1` и далее — сейчас идёт сезон S6) — Kaggle специально делает их для обучения: чистый табличный формат, бинарная классификация, метрика AUC.
- **Santander Customer Transaction Prediction** — классика: 200 анонимных числовых признаков, сильный дисбаланс, метрика AUC.
- **Home Credit Default Risk** — сложнее (несколько таблиц), но это настоящая индустриальная задача про дефолт.

Для первого прохождения бери свежий **Playground Series** с бинарным таргетом и метрикой AUC: он самый близкий по духу к этому воркшопу.

Аутентификация kaggle CLI. Современный способ — OAuth-вход прямо из CLI: команда откроет браузер и привяжет аккаунт без ручного файла.

```bash
uv run kaggle auth login          # OAuth: откроет браузер и сохранит токен

# смотрим список соревнований
uv run kaggle competitions list -s tabular

# качаем данные конкретного соревнования (пример slug)
uv run kaggle competitions download -c playground-series-s6e1 -p data/
cd data && unzip -o '*.zip' && cd ..
```

Способ через `kaggle.json` (kaggle.com → Account → Create New API Token, затем `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json`) теперь считается legacy, но всё ещё работает — пригодится в CI и на машинах без браузера.

Что получится: в `data/` появятся `train.csv`, `test.csv` и `sample_submission.csv`. `train.csv` содержит таргет, `test.csv` — нет (его ты и предсказываешь), `sample_submission.csv` задаёт точный формат ответа.

### Запасной генератор (без аккаунта)

Чтобы пайплайн был самодостаточным, есть рабочий генератор синтетического табличного датасета с дисбалансом и фиксированным seed. Он создаёт те же три файла, что и Kaggle, плюс «правду» по тесту — чтобы локально симулировать leaderboard.

ЗАЧЕМ. Нужен датасет, повторяющий структуру соревнования: train с таргетом, test без таргета, дисбаланс ~12% позитивов, информативные + шумовые признаки. И отдельный файл с истинными метками теста, который сыграет роль приватного leaderboard.

```python
# make_dataset.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

SEED = 42

def build():
    X, y = make_classification(
        n_samples=60_000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_repeated=0,
        n_clusters_per_class=3,
        weights=[0.88, 0.12],     # дисбаланс: ~12% позитивного класса
        flip_y=0.02,              # немного шума в метках, как в реальности
        class_sep=0.9,
        random_state=SEED,
    )
    cols = [f"f{i:02d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df.insert(0, "id", np.arange(len(df)))

    # train/test split: test играет роль приватного множества Kaggle
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    test_n = 20_000
    test_idx, train_idx = idx[:test_n], idx[test_n:]

    train = df.iloc[train_idx].reset_index(drop=True)
    test_full = df.iloc[test_idx].reset_index(drop=True)

    test = test_full.drop(columns=["target"])          # как в Kaggle: без таргета
    truth = test_full[["id", "target"]]                # наш локальный private LB
    sample = test[["id"]].copy()
    sample["target"] = 0.0                             # формат сабмита

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    sample.to_csv("data/sample_submission.csv", index=False)
    truth.to_csv("data/_truth.csv", index=False)       # подчёркивание = «не существует» в реальном Kaggle
    print("train:", train.shape, "test:", test.shape,
          "pos_rate:", round(train.target.mean(), 4))

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    build()
```

```bash
uv run python make_dataset.py
```

ЧТО ПОЛУЧИЛОСЬ. `train: (40000, 22) test: (20000, 21) pos_rate: 0.1199`. Дальше весь пайплайн работает с этими файлами, не зная про `_truth.csv`. Файл правды трогаем только в конце, чтобы посмотреть «private» скор — ровно как в настоящем соревновании ты узнаёшь private только после закрытия.

!!! warning "Файл правды — это симуляция, а не чит"

    `_truth.csv` существует только в синтетическом режиме и нужен, чтобы прочувствовать механику public vs private LB на своей машине. В реальном соревновании такого файла нет и быть не может. Если бы он был — это был бы leak, и любое решение, подсматривающее в test-таргет, не имеет ценности. Дисциплина: пока пайплайн строится, ты живёшь только на `train.csv`.

## Ход работы

### Шаг 1: EDA — посмотреть на данные глазами

ЗАЧЕМ. До любой модели нужно понять: какой дисбаланс, есть ли пропуски, как соотносятся распределения признаков в train и test (если они разные — твоя валидация будет врать), нет ли подозрительно «идеального» признака (target leak).

```python
import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
TARGET, ID = "target", "id"
features = [c for c in train.columns if c not in (TARGET, ID)]

print("shape:", train.shape, test.shape)
print("pos_rate:", train[TARGET].mean().round(4))
print("nulls train:", int(train[features].isna().sum().sum()))

# корреляция каждого признака с таргетом — грубый детектор утечки
corr = train[features].corrwith(train[TARGET]).abs().sort_values(ascending=False)
print(corr.head(8))

# train vs test: сравним средние, чтобы заметить ковариативный сдвиг
shift = pd.DataFrame({
    "train_mean": train[features].mean(),
    "test_mean": test[features].mean(),
})
shift["abs_diff"] = (shift.train_mean - shift.test_mean).abs()
print(shift.sort_values("abs_diff", ascending=False).head(5))
```

ЧТО ПОЛУЧИЛОСЬ. Подтверждаем дисбаланс (~12%), отсутствие пропусков, видим топ признаков по связи с таргетом. Если какой-то признак коррелирует с таргетом на 0.99 — это красный флаг утечки, такой признак надо расследовать, а не радоваться ему. Сравнение train/test средних показывает, насколько множества похожи: близкие средние — хороший знак для будущей валидации.

!!! tip "Adversarial validation — детектор «train не похож на test»"

    Если подозреваешь сдвиг между train и test, обучи классификатор отличать одно от другого: склей их, метка `is_test`, прогони CV. Если AUC такого классификатора около 0.5 — множества неразличимы, обычная валидация надёжна. Если сильно выше 0.5 — распределения разъехались, и наивный K-fold переоценит качество. Это первый инструмент, который киглеры применяют, когда local CV и LB не сходятся.

### Шаг 2: Надёжная кросс-валидация — сердце соревнования

ЗАЧЕМ. Public leaderboard считается на маленькой части test и шумит. Твой компас — local CV. Но он полезен только если устроен честно: стратификация по таргету (из-за дисбаланса), фиксированные фолды (чтобы сравнивать эксперименты), и **out-of-fold (OOF) предсказания**, по которым ты меряешь скор на всех train-объектах ровно один раз. Это прямое применение модуля M16.

```python
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# заранее заведём массив OOF-предсказаний — туда фолды пишут свои предсказания
oof = np.zeros(len(train))
folds = list(skf.split(train[features], train[TARGET]))
print(f"{N_SPLITS} фолдов, стратификация по таргету, seed={SEED}")
```

ЧТО ПОЛУЧИЛОСЬ. Фиксированное разбиение: каждый объект train ровно один раз попадает в валидацию. `oof` заполнится предсказаниями «честных» моделей (которые этот объект не видели при обучении). Скор по `oof` — твоя главная цифра. Меняешь фичу или гиперпараметр → пересчитываешь `oof` → сравниваешь. Seed не трогаешь никогда, иначе сравнения развалятся.

!!! danger "Почему совпадение CV и LB важнее самого высокого CV"

    Можно выжать красивый local CV, который не двигает LB, — это значит, что валидация измеряет не то. Цель не «максимальный CV», а **корреляция направлений**: фича подняла CV → она же подняла (или хотя бы не уронила) LB. Когда корреляция есть, ты можешь делать десятки решений по local CV, не тратя дневной лимит сабмитов. Когда её нет — каждый шаг это гадание. Поэтому первым делом после baseline проверяют: сошлись ли CV и LB.

!!! question "Проверь себя"

    1. Зачем при дисбалансе 12% нужен именно `StratifiedKFold`, а не обычный `KFold`?
    2. Что такое out-of-fold (OOF) предсказание и почему скор по нему честнее, чем средний скор по валидационным фолдам?
    3. Почему seed фолдов нельзя менять между экспериментами?

??? success "Ответы"

    1. Обычный `KFold` может случайно собрать фолд почти без позитивов — скор на нём будет шумным и несравнимым. Стратификация сохраняет долю классов в каждом фолде, поэтому 5 оценок сопоставимы между собой.
    2. OOF — предсказание для объекта, сделанное моделью, которая этот объект не видела при обучении. Собрав OOF по всем фолдам, ты получаешь по одному честному предсказанию на каждый train-объект и считаешь метрику разом на всём train. Это устойчивее, чем усреднять пять отдельных fold-AUC.
    3. Меняя seed, ты меняешь разбиение, и разница в скоре между двумя экспериментами становится смесью «эффект идеи + эффект другого разбиения». Сравнение перестаёт быть честным — ты меряешь шум.

### Шаг 3: Baseline-сабмит — закрыть цикл целиком

ЗАЧЕМ. Прежде чем улучшать, надо один раз пройти весь путь: обучить простую модель, получить OOF-скор, собрать `submission.csv` в правильном формате и отправить. Это проверяет, что труба не течёт: формат сабмита верный, метрика считается, CV и LB сопоставимы. Тупой baseline, доехавший до LB, ценнее гениальной модели, застрявшей в ноутбуке.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

test_pred = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(folds):
    X_tr, y_tr = train.iloc[tr_idx][features], train.iloc[tr_idx][TARGET]
    X_va = train.iloc[va_idx][features]

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, C=0.5))
    model.fit(X_tr, y_tr)

    oof[va_idx] = model.predict_proba(X_va)[:, 1]      # честное OOF-предсказание
    test_pred += model.predict_proba(test[features])[:, 1] / N_SPLITS  # усреднение по фолдам

cv_auc = roc_auc_score(train[TARGET], oof)
print(f"baseline OOF AUC: {cv_auc:.5f}")

sub = pd.read_csv("data/sample_submission.csv")
sub[TARGET] = test_pred
sub.to_csv("submission_baseline.csv", index=False)
print(sub.head(3))
```

ЧТО ПОЛУЧИЛОСЬ. OOF AUC порядка ~0.86 (зависит от данных) и файл `submission_baseline.csv` в формате `id,target`. `test_pred` усреднён по 5 фолдам — это уже маленький ансамбль, который стабильнее одной модели. Отправляем:

```bash
# реальный Kaggle:
uv run kaggle competitions submit -c playground-series-s6e1 \
    -f submission_baseline.csv -m "baseline logreg, CV 0.86"
uv run kaggle competitions submissions -c playground-series-s6e1   # посмотреть свой скор на LB
```

Для синтетического режима LB симулируем локально — это и будет твой «public/private»:

```python
def score_submission(path, public_frac=0.30, seed=SEED):
    truth = pd.read_csv("data/_truth.csv")          # существует только в синтет-режиме
    sub = pd.read_csv(path).merge(truth, on="id", suffixes=("_pred", "_true"))
    rng = np.random.default_rng(seed)
    mask = rng.random(len(sub)) < public_frac        # public = 30% теста, как часто на Kaggle
    pub = roc_auc_score(sub.target_true[mask], sub.target_pred[mask])
    prv = roc_auc_score(sub.target_true[~mask], sub.target_pred[~mask])
    return round(pub, 5), round(prv, 5)

print("public/private:", score_submission("submission_baseline.csv"))
```

ЧТО ПОЛУЧИЛОСЬ. Две цифры: public (видишь во время соревнования) и private (узнаёшь в конце). Сравни public с local CV — они должны быть близки. Если разрыв большой, вернись к шагу 2: валидация не совпадает с LB, и до её починки улучшать модель бессмысленно.

### Шаг 4: Feature engineering — итеративно и по CV

ЗАЧЕМ. Сырых признаков обычно мало. Прирост дают производные: агрегаты, взаимодействия, нелинейные преобразования, частотное кодирование категорий. Главное правило — **проверять каждую группу фич по OOF-скору**, а не по интуиции. Фича, не двигающая CV, добавляет шум и риск переобучения.

```python
def add_features(df):
    df = df.copy()
    f = [c for c in df.columns if c.startswith("f")]
    # агрегаты по строке: суммарный «профиль» объекта
    df["row_sum"] = df[f].sum(axis=1)
    df["row_mean"] = df[f].mean(axis=1)
    df["row_std"] = df[f].std(axis=1)
    df["row_max"] = df[f].max(axis=1)
    df["row_min"] = df[f].min(axis=1)
    # взаимодействия топ-признаков (узнали из EDA на шаге 1)
    df["f00_x_f01"] = df["f00"] * df["f01"]
    df["f00_div_f02"] = df["f00"] / (df["f02"].abs() + 1e-6)
    return df

train_fe = add_features(train)
test_fe = add_features(test)
features_fe = [c for c in train_fe.columns if c not in (TARGET, ID)]

def evaluate(frame, feats):
    oof_local = np.zeros(len(frame))
    for tr_idx, va_idx in folds:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, C=0.5))
        m.fit(frame.iloc[tr_idx][feats], frame.iloc[tr_idx][TARGET])
        oof_local[va_idx] = m.predict_proba(frame.iloc[va_idx][feats])[:, 1]
    return roc_auc_score(frame[TARGET], oof_local)

print("baseline feats:", round(cv_auc, 5))
print("with FE       :", round(evaluate(train_fe, features_fe), 5))
```

ЧТО ПОЛУЧИЛОСЬ. Две цифры рядом: было / стало. Оставляешь группу фич, только если OOF вырос. Те же `folds` — поэтому сравнение честное. Это и есть «итеративный feature engineering»: не вываливаешь сто фич разом, а добавляешь группами и смотришь на компас.

!!! warning "Утечка через feature engineering — самая частая ошибка"

    Любая статистика, посчитанная по таргету (например, mean target по категории), должна считаться **только внутри train-фолда**, иначе валидационные объекты «подсматривают» свой ответ. Это раздувает OOF и рушит его связь с LB. Правило: всё, что трогает таргет, — внутри цикла фолдов на `tr_idx`, никогда на полном train. Row-агрегаты выше безопасны: они таргет не видят.

!!! question "Проверь себя"

    1. Признак коррелирует с таргетом на 0.98. Это удача или повод для расследования?
    2. Почему mean-target-encoding, посчитанный на полном train до разбиения на фолды, — это утечка, а `row_sum` по числовым фичам — нет?

??? success "Ответы"

    1. Почти наверняка повод для расследования: такая корреляция обычно означает target leak — признак содержит информацию, недоступную в момент предсказания (например, посчитан после факта). На test/private такого «читерского» сигнала не будет, и модель рухнет. Проверь происхождение признака, прежде чем доверять ему.
    2. Mean-target-encoding на полном train использует таргет валидационных объектов при расчёте их же признака — они подсматривают ответ, OOF завышается. `row_sum` складывает только сами фичи объекта и таргета не касается вообще, поэтому утечки нет.

### Шаг 5: LightGBM + тюнинг через Optuna

ЗАЧЕМ. На табличке градиентный бустинг почти всегда бьёт линейные модели: ловит нелинейности и взаимодействия сам. Но у него много гиперпараметров. Перебирать руками — долго и нечестно (легко переобучиться на одном сплите). Optuna ищет умно (TPE-сэмплер), а оцениваем каждый набор по той же 5-фолдовой OOF-схеме — чтобы тюнинг оптимизировал именно то, что коррелирует с LB.

```python
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def lgb_oof(params, frame, feats):
    oof_local = np.zeros(len(frame))
    for tr_idx, va_idx in folds:
        dtr = lgb.Dataset(frame.iloc[tr_idx][feats], frame.iloc[tr_idx][TARGET])
        dva = lgb.Dataset(frame.iloc[va_idx][feats], frame.iloc[va_idx][TARGET])
        m = lgb.train(
            {**params, "objective": "binary", "metric": "auc",
             "verbosity": -1, "seed": SEED},
            dtr, num_boost_round=2000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_local[va_idx] = m.predict(frame.iloc[va_idx][feats])
    return roc_auc_score(frame[TARGET], oof_local)

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    return lgb_oof(params, train_fe, features_fe)

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=40, show_progress_bar=True)
print("best OOF AUC:", round(study.best_value, 5))
print("best params:", study.best_params)
```

ЧТО ПОЛУЧИЛОСЬ. Лучший набор гиперпараметров по OOF и сам скор — обычно заметно выше линейного baseline. `early_stopping` подбирает число деревьев под каждый фолд автоматически. 40 trials хватает для playground; на больших данных увеличивай или ставь `study.optimize(..., timeout=3600)`. Важно: метрика тюнинга — та же OOF-схема, поэтому победитель Optuna реально полезен на LB, а не только на одном случайном сплите.

!!! tip "Не тюнингуй до посинения — закон убывающей отдачи"

    Первые 20-30 trials дают основной прирост, дальше AUC растёт на третий-четвёртый знак. Это переобучение под валидацию: ты подгоняешь гиперпараметры под конкретные фолды. Лучше потратить время на новую фичу или второй тип модели для ансамбля — там отдача выше. Зафиксируй разумные параметры и двигайся дальше.

### Шаг 6: Финальная модель и сабмит

ЗАЧЕМ. Лучшие параметры найдены — обучаем финальную модель по фолдам, собираем OOF (для записи в write-up) и усреднённое предсказание теста. Усреднение по фолдам — бесплатный bagging, снижает дисперсию.

```python
best = {**study.best_params, "objective": "binary", "metric": "auc",
        "verbosity": -1, "seed": SEED}

oof_final = np.zeros(len(train_fe))
test_final = np.zeros(len(test_fe))
for tr_idx, va_idx in folds:
    dtr = lgb.Dataset(train_fe.iloc[tr_idx][features_fe], train_fe.iloc[tr_idx][TARGET])
    dva = lgb.Dataset(train_fe.iloc[va_idx][features_fe], train_fe.iloc[va_idx][TARGET])
    m = lgb.train(best, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_final[va_idx] = m.predict(train_fe.iloc[va_idx][features_fe])
    test_final += m.predict(test_fe[features_fe]) / N_SPLITS

print("final OOF AUC:", round(roc_auc_score(train_fe[TARGET], oof_final), 5))

sub = pd.read_csv("data/sample_submission.csv")
sub[TARGET] = test_final
sub.to_csv("submission_lgb.csv", index=False)
print("public/private:", score_submission("submission_lgb.csv"))
```

ЧТО ПОЛУЧИЛОСЬ. `submission_lgb.csv` и финальный OOF. В синтет-режиме сразу видим public/private. На реальном Kaggle отправляем тем же `kaggle competitions submit` и сравниваем LB-скор с OOF — они должны быть близко.

### Шаг 7: Ансамблирование — блендинг и стекинг

ЗАЧЕМ. Разные модели ошибаются по-разному. Если усреднить предсказания моделей с похожим скором, но разной природы (бустинг + линейная), итог обычно стабильнее и чуть выше каждой по отдельности. Это почти бесплатный прирост и страховка от shake-up.

```python
from scipy.stats import rankdata

# блендинг по рангам устойчивее к разным масштабам вероятностей у моделей
def rank_blend(preds, weights):
    ranked = [rankdata(p) / len(p) for p in preds]
    return np.average(ranked, axis=0, weights=weights)

# OOF логрега (шаг 4) и OOF lgb (шаг 6) — обе на одних фолдах, сравнимы
logreg_oof = np.zeros(len(train_fe))
logreg_test = np.zeros(len(test_fe))
for tr_idx, va_idx in folds:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))
    m.fit(train_fe.iloc[tr_idx][features_fe], train_fe.iloc[tr_idx][TARGET])
    logreg_oof[va_idx] = m.predict_proba(train_fe.iloc[va_idx][features_fe])[:, 1]
    logreg_test += m.predict_proba(test_fe[features_fe])[:, 1] / N_SPLITS

blend_oof = rank_blend([oof_final, logreg_oof], weights=[0.8, 0.2])
print("lgb OOF   :", round(roc_auc_score(train_fe[TARGET], oof_final), 5))
print("blend OOF :", round(roc_auc_score(train_fe[TARGET], blend_oof), 5))

blend_test = rank_blend([test_final, logreg_test], weights=[0.8, 0.2])
sub[TARGET] = blend_test
sub.to_csv("submission_blend.csv", index=False)
print("public/private:", score_submission("submission_blend.csv"))
```

ЧТО ПОЛУЧИЛОСЬ. Сравнение OOF: блендинг обычно даёт +0.001..0.003 AUC над лучшей одиночной моделью — на LB это десятки позиций. Веса подбираешь по OOF, не по public LB (иначе переобучишься на нём). **Стекинг** — следующий уровень: OOF-предсказания базовых моделей становятся признаками мета-модели (часто logreg), обученной на том же CV. Для capstone достаточно блендинга; стекинг даёт прирост, но усложняет пайплайн и риск переобучения мета-модели.

!!! tip "Веса ансамбля — только по OOF, не по public LB"

    Соблазн крутить веса по public-скору велик, но public — это маленькая шумная выборка. Подгонишь веса под неё — получишь shake-down на private. Веса (и вообще все решения) подбираются по OOF, public LB используется лишь как разовая sanity-проверка, что валидация не разъехалась с реальностью.

### Шаг 8: Public vs private и solution write-up

ЗАЧЕМ. Соревнование заканчивается переключением на private leaderboard. Часто бывает **shake-up**: те, кто переобучился на public, падают, а те, кто доверял CV, поднимаются. Нужно понять механику и зафиксировать выводы в write-up — это и есть рефлексия, превращающая опыт в навык.

```python
for name in ["submission_baseline.csv", "submission_lgb.csv", "submission_blend.csv"]:
    pub, prv = score_submission(name)
    print(f"{name:28s}  public={pub}  private={prv}  shake={round(prv - pub, 5)}")
```

ЧТО ПОЛУЧИЛОСЬ. Таблица public vs private по всем сабмитам. Заметишь, что разрыв небольшой и стабильный, если модели не переобучены, — это плата за дисциплину OOF. Большой разрыв в пользу private у блендинга — типичный сигнал, что ансамбль робастнее.

!!! question "Проверь себя"

    1. Что такое shake-up и почему участники, гнавшиеся за public LB, чаще всего падают на private?
    2. У тебя два финальных сабмита: один с OOF 0.870 и public 0.864, второй с OOF 0.866 и public 0.871. Какой выбрать для финала и почему?

??? success "Ответы"

    1. Shake-up — перестановка участников при переключении на private leaderboard. Public считается на маленькой выборке, и подгонка решений под неё (выбор фич/весов по public-скору) — это переобучение под шум. На private этот шум другой, поэтому переобученные решения падают, а доверявшие CV — поднимаются.
    2. Первый. OOF — оценка на всём train и она надёжнее, чем public на маленькой выборке. Второй сабмит подозрителен: его public выше OOF — вероятно, он словил удачу на public-части, что не воспроизведётся на private. Финал выбирают по OOF, а не по public.

Теперь write-up. Это обязательный артефакт: структурированный разбор решения. Шаблон `WRITEUP.md`:

```markdown
# Solution write-up — <название соревнования>

## Итог
- Public LB: 0.8xxx, private LB: 0.8xxx, место N из M.
- Финальный сабмит: rank-blend(LightGBM 0.8 + LogReg 0.2).

## Валидация
- 5-fold StratifiedKFold, seed=42, метрика OOF ROC-AUC.
- CV и public LB совпали в пределах 0.00x — валидации доверял, решения принимал по OOF.

## Что сработало
- Градиентный бустинг (LightGBM) дал +0.0xx над линейным baseline.
- Row-агрегаты (sum/std/max) и 2 взаимодействия топ-признаков: +0.00x по OOF.
- Optuna (40 trials) по той же CV: +0.00x.
- Rank-blend двух разных моделей: +0.00x и меньше shake-up.

## Что НЕ сработало
- <фичи, которые не двинули OOF — перечислить честно>.
- Тюнинг сверх 30 trials: прирост в шум, выкинул.

## Чему научился
- Главное умение — валидация, коррелирующая с LB. Без неё улучшения — гадание.
- Веса ансамбля только по OOF; public LB — лишь sanity-check.
- Утечку через target-based фичи ловить заранее (всё по таргету — внутри фолда).
```

ЧТО ПОЛУЧИЛОСЬ. `WRITEUP.md` с честными «сработало / не сработало». На реальном Kaggle его публикуют в разделе Discussion — это и портфолио, и вклад в комьюнити. Заполняешь реальными числами своего прогона.

### Шаг 9: Собрать всё в один воспроизводимый пайплайн

ЗАЧЕМ. Разрозненные ячейки — не артефакт. Артефакт — один скрипт, который из `data/` собирает `submission.csv` детерминированно. Так твою работу можно проверить, переиспользовать и поставить на расписание.

```python
# pipeline.py — точка входа capstone
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

SEED, N_SPLITS, TARGET, ID = 42, 5, "target", "id"

def add_features(df): ...        # из шага 4
def main():
    train = add_features(pd.read_csv("data/train.csv"))
    test = add_features(pd.read_csv("data/test.csv"))
    feats = [c for c in train.columns if c not in (TARGET, ID)]
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(train[feats], train[TARGET]))
    # ... обучение по фолдам, OOF, усреднение теста (шаги 5-6)
    # ... rank_blend (шаг 7) -> submission.csv
    print("done: submission.csv")

if __name__ == "__main__":
    main()
```

```bash
uv run python pipeline.py        # один запуск -> один и тот же submission.csv
```

ЧТО ПОЛУЧИЛОСЬ. Воспроизводимый пайплайн. Зафиксированные `SEED`, фолды и `uv.lock` гарантируют идентичный результат на любой машине. Это финальный артефакт capstone вместе с сабмитом, позицией в LB и write-up.

## Типичные ошибки

- **Гонка за public LB.** Подгонка под public — прямой путь к shake-down на private. Доверяй OOF, public — разовая проверка.
- **Меняющийся seed между экспериментами.** Сравниваешь шум вместо идей. Один `SEED` на фолды и модели, навсегда.
- **Утечка через target-based фичи.** Mean target по категории, посчитанный на полном train, раздувает OOF и рушит связь с LB. Всё по таргету — только внутри `tr_idx`.
- **Нет baseline-сабмита.** Полез сразу в сложную модель — не заметил, что формат сабмита неверный или CV не сходится с LB. Сначала тупой baseline до конца трубы.
- **Тюнинг без CV-обёртки.** Optuna на одном train/valid сплите переобучает гиперпараметры под этот сплит. Оценивай каждый trial по полной OOF-схеме.
- **Ансамбль одинаковых моделей.** Усреднять две почти идентичные модели бессмысленно — нужна разная природа ошибок (бустинг + линейная + KNN).
- **Копирование чужого ноутбука без понимания.** Берёшь топ-решение, не понимая, почему оно работает, — не сможешь воспроизвести успех на другой задаче и не пройдёшь разбор на собеседовании.

!!! tip "AI-копилот в этом воркшопе"

    Используй ассистента как ускоритель механики, не как замену мышлению. Хорошие запросы: «сгенерируй каркас StratifiedKFold-цикла с OOF-массивом», «накидай objective-функцию Optuna для LightGBM с early stopping», «предложи 10 производных признаков для строки числовых фич и помечай, какие могут дать утечку», «объясни, почему мой public и CV разошлись на 0.02». Плохой запрос: «реши за меня это соревнование». Главное решение — какой валидации доверять и почему фича помогла — принимаешь ты, и именно его проверяют на интервью. Любую фичу, предложенную ассистентом, прогоняй через свой OOF-компас: помогла по CV — оставил, нет — выкинул, независимо от того, как уверенно её предложил AI.

## Критерий готовности

- [ ] Данные получены: kaggle API (реальное соревнование) или синтетический генератор отработал, в `data/` лежат train/test/sample_submission.
- [ ] Проведён EDA: дисбаланс, пропуски, train-vs-test сдвиг, проверка на target leak.
- [ ] Построена StratifiedKFold-схема с фиксированным seed и OOF-массивом.
- [ ] Baseline-сабмит дошёл до LB (реального или симулированного); CV и public сопоставлены.
- [ ] Хотя бы одна группа фич добавлена и проверена по OOF (оставлена только если подняла CV).
- [ ] LightGBM обучен, гиперпараметры подобраны Optuna по той же CV-схеме.
- [ ] Сделан ансамбль (хотя бы rank-blend двух разных моделей), веса подобраны по OOF.
- [ ] Посчитан public vs private (или получен private после закрытия) и понят shake-up.
- [ ] Написан `WRITEUP.md` с честными «сработало / не сработало / чему научился».
- [ ] Собран `pipeline.py` — один запуск даёт детерминированный `submission.csv`.
- [ ] Зафиксирована позиция в leaderboard.

## Бизнес-вывод

AUC в leaderboard — это ещё не ответ для бизнеса. Head of Retention не примет решение по числу 0.87; он примет его по фразе «вот кого звонить и сколько денег это вернёт». Переведи технический результат в решение для стейкхолдера.

- [ ] **Рекомендация (что делать).** Сформулируй действие, а не метрику: «запускаем адресную retention-кампанию по топ-20% риска из модели вместо веерной раздачи; порог отсечки — такой-то». Без жаргона про OOF и фолды.
- [ ] **Эффект в деньгах/метриках.** Привяжи AUC к деловому смыслу: насколько адресный таргетинг по скорам поднимает долю удержанных при том же бюджете 3 000 000 ₽/мес, и сколько это в ₽ удержанной выручки. Покажи дельту «было/стало», а не абсолютный AUC.
- [ ] **Риски и допущения.** Назови честно: модель валидна, только пока поведение игроков и состав базы стабильны (covariate shift = твой train-vs-test сдвиг и shake-up); скоры — это ранжирование риска, а не гарантия; адресность сужает охват — часть «тихих» оттоков можно пропустить.
- [ ] **Следующий шаг.** Предложи проверяемый план: A/B-пилот на 2 недели (модельный таргетинг против текущей эвристики), guardrail-метрика — стоимость удержания, и мониторинг дрейфа после раскатки.
- [ ] **Как подать стейкхолдеру.** Один слайд на языке решений: «при том же бюджете удерживаем ориентировочно в 1.5–2x больше игроков, риск — деградация при сдвиге базы, проверяем пилотом за 2 недели». Цифры в ₽ и игроках, а не в третьем знаке AUC.

## Развитие

- **Стекинг полноценно.** Сделай OOF-предсказания 3-4 базовых моделей признаками мета-модели на том же CV. Часто +0.001..0.005 над блендингом.
- **polars вместо pandas** на больших данных: feature engineering на сотнях миллионов строк, где pandas задыхается. Перепиши `add_features` на polars lazy API.
- **Pseudo-labeling.** Уверенные предсказания на test добавь в train как метки и переобучи — работает, когда test большой, но легко переобучиться, контролируй по CV.
- **GroupKFold / TimeSeriesSplit.** Если в данных есть группы (user_id) или время — обычный StratifiedKFold течёт. Освой схемы валидации под структуру данных (см. модуль M16).
- **Перенос в прод.** Возьми финальный пайплайн и оберни его в сервис из воркшопа W8 (FastAPI + мониторинг дрейфа) — соревновательная модель станет рабочим API.

## Что ты закрепил

- **Полный соревновательный цикл**: от kaggle API и первого baseline до ансамбля, write-up и воспроизводимого пайплайна.
- **Главное умение ML-практика** — валидация, коррелирующая с leaderboard. Local CV как приборная панель, по которой принимаются все решения, а не красивое число на public.
- **Дисциплина против переобучения**: фиксированный seed, OOF-оценка каждого изменения, веса ансамбля по CV, защита от утечки.
- **Градиентный бустинг и тюнинг**: LightGBM с Optuna в правильной CV-обёртке.
- **Ансамблирование**: rank-blend и понимание, когда он помогает и почему снижает shake-up.
- **Механика public vs private LB** и природа shake-up — почему дисциплинированные поднимаются, а гонящиеся за public падают.
- **Этика и навык**: понимать каждое решение, а не копировать чужой ноутбук вслепую. Именно это превращает участие в соревновании в переносимый навык.
