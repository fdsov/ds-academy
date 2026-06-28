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

!!! info "Как устроен этот воркшоп"

    Это не лекция с готовым кодом, а задачник. Каждый содержательный шаг построен так:

    - **Задача** — что именно сделать руками, с явными именами выходных переменных, которые проверяет критерий.
    - **Критерий шага** — кусок кода с `assert`, который ты запускаешь после своего решения. Зелёный прогон = шаг сдан. Это твой локальный авто-грейдер.
    - Если шаг зависит от внешнего ресурса (аккаунт Kaggle, скачивание данных, реальный сабмит) — вместо `assert` идёт **self-check чек-лист**: проверяешь руками по пунктам.
    - **Решение** спрятано под спойлер `Решение` — открывай после своей попытки, чтобы сверить подход, а не списать.
    - **Числовые прикидки** проверяются прямо на странице: посчитай число, впиши в поле, нажми «Проверить».

    Весь синтетический путь рассчитан на `SEED=42`. Не меняй seed — иначе фолды, инициализация модели и числа в критериях поплывут. Реальное Kaggle-соревнование (раздел «Данные») запускается только при наличии аккаунта — там критерии заменены на self-check.

!!! note "Почему capstone, а не ещё один туториал"

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

Нужен табличный датасет для бинарной классификации с дисбалансом. Покажу оба пути: реальное соревнование через kaggle API и **запасной синтетический генератор**, чтобы пайплайн запускался даже без аккаунта Kaggle. Все критерии-`assert` ниже считаются на синтетическом датасете.

### Реальное соревнование (рекомендуется, но внешнее)

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

!!! warning "Этот блок зависит от сети и аккаунта"

    Команды `kaggle auth login` / `download` / `submit` требуют аккаунт Kaggle и интернет — они не проверяются `assert` в этом воркшопе. Если делаешь capstone на реальном соревновании, сверяйся по self-check в конце шага 3 и шага 8. Если аккаунта нет — иди по синтетическому пути ниже, он самодостаточен и полностью воспроизводим.

### Запасной генератор (без аккаунта) — это SETUP, копируй как есть

Чтобы пайплайн был самодостаточным, есть рабочий генератор синтетического табличного датасета с дисбалансом и фиксированным seed. Он создаёт те же три файла, что и Kaggle, плюс «правду» по тесту — чтобы локально симулировать leaderboard. Это setup-код, а не задача.

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

Печатает `train: (40000, 22) test: (20000, 21) pos_rate: 0.1258`. Дальше весь пайплайн работает с этими файлами, не зная про `_truth.csv`. Файл правды трогаем только в конце, чтобы посмотреть «private» скор — ровно как в настоящем соревновании ты узнаёшь private только после закрытия.

**Критерий шага** — запусти после генерации:

```python
import pandas as pd
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
assert train.shape == (40000, 22), "train: 40000 строк, 22 колонки (id + 20 фич + target)"
assert test.shape == (20000, 21),  "test: 20000 строк, без target"
assert "target" not in test.columns, "в test таргета быть не должно — его ты и предсказываешь"
assert 0.11 <= train["target"].mean() <= 0.14, "дисбаланс должен быть около 12%"
print("OK: датасет сгенерирован, дисбаланс ~", round(train["target"].mean(), 4))
```

!!! warning "Файл правды — это симуляция, а не чит"

    `_truth.csv` существует только в синтетическом режиме и нужен, чтобы прочувствовать механику public vs private LB на своей машине. В реальном соревновании такого файла нет и быть не может. Если бы он был — это был бы leak, и любое решение, подсматривающее в test-таргет, не имеет ценности. Дисциплина: пока пайплайн строится, ты живёшь только на `train.csv`.

## Разминка: прикинь числа руками

Прежде чем писать пайплайн, прогрей две интуиции соревновательного ML руками. Посчитай и впиши ответ — проверка мгновенная.

```text
TASK: Доля позитивного класса (отток) ~12.6%. Тупая модель "предсказываю всем класс 0". Какая у неё будет accuracy в процентах? Округли до 0.1.
ANSWER: 87.4
TOL: 0.6
UNIT: %
PLACEHOLDER: 0.0
EXPLAIN: accuracy = доля правильных = доля негативного класса = 1 - 0.126 = 0.874 = 87.4%. Вот почему accuracy бесполезна при дисбалансе: 87% "точности" не отличают игрока на грани оттока от лояльного. ROC-AUC такой тупой модели = 0.5 - она не ранжирует. Поэтому метрика соревнования - AUC, а не accuracy.
---
TASK: 5-fold кросс-валидация. На какой доле train (в процентах) обучается модель внутри каждого фолда? Целое число.
ANSWER: 80
TOL: 1
UNIT: %
PLACEHOLDER: целое число
EXPLAIN: при K=5 один фолд (1/5 = 20%) уходит в валидацию, остальные 4/5 = 80% - в обучение. Каждый из 5 объектов-блоков ровно один раз бывает валидацией: так собирается OOF-предсказание на 100% train, по одному честному предсказанию на объект.
```

## Ход работы

### Шаг 1: EDA — посмотреть на данные глазами

**Зачем.** До любой модели нужно понять: какой дисбаланс, есть ли пропуски, как соотносятся распределения признаков в train и test (если они разные — твоя валидация будет врать), нет ли подозрительно «идеального» признака (target leak).

**Задача.** Загрузи `data/train.csv` и `data/test.csv`, определи списки `features` (всё кроме `target` и `id`). Посчитай: долю позитивов, число пропусков, абсолютную корреляцию каждого признака с таргетом (в `corr`, отсортируй по убыванию) и сравни средние train vs test. Имена для критерия: `train`, `test`, `features`, `corr`.

??? tip "Подсказка"

    `train[features].corrwith(train[TARGET]).abs().sort_values(ascending=False)` даёт корреляции одной строкой. Для train-vs-test сдвига собери `DataFrame` со средними по обоим множествам и колонкой `abs_diff`. Грубый детектор утечки — признак с корреляцией к таргету близко к 1.0.

**Критерий шага:**

```python
assert "target" in train.columns and "target" not in test.columns
assert len(features) == 20, "20 числовых признаков f00..f19"
assert int(train[features].isna().sum().sum()) == 0, "пропусков в синтетике нет"
assert corr.max() < 0.5, "ни один признак не должен подозрительно сильно коррелировать с таргетом (нет leak)"
print(f"OK: дисбаланс={train['target'].mean():.4f}, max|corr|={corr.max():.3f} — утечки нет")
```

??? success "Решение"

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

    **Почему так.** Подтверждаешь дисбаланс (~0.126), отсутствие пропусков, видишь топ признаков по связи с таргетом — здесь самый сильный около 0.22, что нормально (информативный, но не leak). Если бы какой-то признак коррелировал с таргетом на 0.99 — это красный флаг утечки, такой признак надо расследовать, а не радоваться ему. Близкие train/test средние — хороший знак: множества похожи, обычная валидация будет надёжна.

!!! tip "Adversarial validation — детектор «train не похож на test»"

    Если подозреваешь сдвиг между train и test, обучи классификатор отличать одно от другого: склей их, метка `is_test`, прогони CV. Если AUC такого классификатора около 0.5 — множества неразличимы, обычная валидация надёжна. Если сильно выше 0.5 — распределения разъехались, и наивный K-fold переоценит качество. Это первый инструмент, который киглеры применяют, когда local CV и LB не сходятся.

### Шаг 2: Надёжная кросс-валидация — сердце соревнования

**Зачем.** Public leaderboard считается на маленькой части test и шумит. Твой компас — local CV. Но он полезен только если устроен честно: стратификация по таргету (из-за дисбаланса), фиксированные фолды (чтобы сравнивать эксперименты), и **out-of-fold (OOF) предсказания**, по которым ты меряешь скор на всех train-объектах ровно один раз. Это прямое применение модуля M16.

**Задача.** Заведи `SEED=42`, `N_SPLITS=5`. Построй `StratifiedKFold(shuffle=True, random_state=SEED)`, материализуй разбиение в список `folds = list(skf.split(...))` и создай пустой массив `oof = np.zeros(len(train))`. Стратификацию делай по таргету.

**Критерий шага:**

```python
assert len(folds) == N_SPLITS, "должно быть ровно 5 фолдов"
val_idx = np.concatenate([va for _, va in folds])
assert sorted(val_idx.tolist()) == list(range(len(train))), \
    "каждый объект train ровно один раз попадает в валидацию (основа OOF)"
print(f"OK: {N_SPLITS} фолдов, стратификация по таргету, seed={SEED}")
```

??? success "Решение"

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

    **Почему так.** Фиксированное разбиение: каждый объект train ровно один раз попадает в валидацию. `oof` заполнится предсказаниями «честных» моделей (которые этот объект не видели при обучении). Скор по `oof` — твоя главная цифра. Меняешь фичу или гиперпараметр → пересчитываешь `oof` → сравниваешь. Seed не трогаешь никогда, иначе сравнения развалятся.

!!! danger "Почему совпадение CV и LB важнее самого высокого CV"

    Можно выжать красивый local CV, который не двигает LB, — это значит, что валидация измеряет не то. Цель не «максимальный CV», а **корреляция направлений**: фича подняла CV → она же подняла (или хотя бы не уронила) LB. Когда корреляция есть, ты можешь делать десятки решений по local CV, не тратя дневной лимит сабмитов. Когда её нет — каждый шаг это гадание. Поэтому первым делом после baseline проверяют: сошлись ли CV и LB.

Проверь понимание:

```text
Q: Зачем при дисбалансе 12% нужен именно StratifiedKFold, а не обычный KFold?
[ ] StratifiedKFold обучается быстрее на дисбалансе
[x] Обычный KFold может собрать фолд почти без позитивов, и скор на нём станет шумным и несравнимым; стратификация держит долю классов одинаковой в каждом фолде
[ ] KFold вообще нельзя применять к классификации
> При 12% позитивов случайный фолд легко окажется с 8% или 16% позитивов - пять оценок станут несопоставимы. Стратификация сохраняет долю классов, поэтому фолды сравнимы между собой.
---
Q: Что такое out-of-fold (OOF) предсказание и почему скор по нему честнее среднего по fold-AUC?
[ ] Это предсказание на test-множестве
[x] Это предсказание для объекта моделью, которая его не видела при обучении; собрав OOF по всем фолдам, меряешь метрику разом на всём train — устойчивее, чем усреднять пять отдельных fold-AUC
[ ] Это предсказание финальной модели, обученной на всём train
> OOF даёт по одному честному предсказанию на каждый train-объект. Метрика разом на всём train устойчивее, чем среднее пяти fold-AUC, которое прячет разброс между фолдами.
---
Q: Почему seed фолдов нельзя менять между экспериментами?
[ ] Менять seed запрещено лицензией sklearn
[x] Другой seed меняет разбиение, и разница в скоре станет смесью "эффект идеи + эффект другого разбиения" — сравнение перестанет быть честным
[ ] От seed зависит только скорость, не результат
> Меняя seed, ты меряешь шум вместе с эффектом. Один SEED на фолды навсегда — тогда разница между двумя экспериментами это чистый сигнал об идее.
```

### Шаг 3: Baseline-сабмит — закрыть цикл целиком

**Зачем.** Прежде чем улучшать, надо один раз пройти весь путь: обучить простую модель, получить OOF-скор, собрать `submission.csv` в правильном формате и отправить. Это проверяет, что труба не течёт: формат сабмита верный, метрика считается, CV и LB сопоставимы. Тупой baseline, доехавший до LB, ценнее гениальной модели, застрявшей в ноутбуке.

**Задача.** Прогони логистическую регрессию (`StandardScaler` + `LogisticRegression`) по фолдам: заполни `oof[va_idx]` честными предсказаниями и усредни предсказания теста в `test_pred` (делёж на `N_SPLITS`). Посчитай `cv_auc = roc_auc_score(...)` по `oof`, собери `submission_baseline.csv` из `sample_submission.csv`. Затем определи функцию `score_submission` (симулятор public/private LB) для синтетического режима.

**Критерий шага:**

```python
from pathlib import Path
assert 0.70 <= cv_auc <= 0.80, "линейный baseline на этих данных даёт OOF AUC около 0.75"
assert Path("submission_baseline.csv").exists(), "сабмит должен лечь на диск"
sub_check = pd.read_csv("submission_baseline.csv")
assert list(sub_check.columns) == ["id", "target"], "формат сабмита: id,target"
assert len(sub_check) == len(test), "одна строка на каждый объект test"
pub, prv = score_submission("submission_baseline.csv")
assert abs(pub - cv_auc) < 0.03, "public должен быть близок к local CV — труба не течёт"
print(f"OK: baseline OOF AUC={cv_auc:.5f}, public={pub}, private={prv}")
```

??? success "Решение"

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

    # локальный симулятор LB (только синтетический режим — использует _truth.csv)
    def score_submission(path, public_frac=0.30, seed=SEED):
        truth = pd.read_csv("data/_truth.csv")
        s = pd.read_csv(path).merge(truth, on="id", suffixes=("_pred", "_true"))
        rng = np.random.default_rng(seed)
        mask = rng.random(len(s)) < public_frac        # public = ~30% теста, как часто на Kaggle
        pub = roc_auc_score(s.target_true[mask], s.target_pred[mask])
        prv = roc_auc_score(s.target_true[~mask], s.target_pred[~mask])
        return round(pub, 5), round(prv, 5)

    print("public/private:", score_submission("submission_baseline.csv"))
    ```

    **Почему так.** OOF AUC получается около **0.75** (линейная модель на данных с тремя кластерами на класс ловит не всё — это нормальный baseline). `test_pred` усреднён по 5 фолдам — это уже маленький ансамбль, стабильнее одной модели. Public и local CV близки (оба ~0.74) — значит труба не течёт и валидации можно верить. Если бы разрыв был большой — вернись к шагу 2, до починки валидации улучшать модель бессмысленно.

!!! note "Реальный Kaggle: сабмит через CLI (self-check, не assert)"

    На реальном соревновании после сборки `submission_baseline.csv` отправляешь его и смотришь скор на LB:

    ```bash
    uv run kaggle competitions submit -c playground-series-s6e1 \
        -f submission_baseline.csv -m "baseline logreg, CV 0.75"
    uv run kaggle competitions submissions -c playground-series-s6e1
    ```

    Эти команды зависят от сети и аккаунта — проверяй руками:

    - [ ] сабмит принят (статус `complete`, не `error`);
    - [ ] public-скор на LB отличается от твоего local CV не сильно (в пределах ~0.01–0.02);
    - [ ] если разрыв большой — это сигнал «валидация разъехалась с LB», вернись к шагу 2.

### Шаг 4: Feature engineering — итеративно и по CV

**Зачем.** Сырых признаков обычно мало. Прирост дают производные: агрегаты, взаимодействия, нелинейные преобразования, частотное кодирование категорий. Главное правило — **проверять каждую группу фич по OOF-скору**, а не по интуиции. Фича, не двигающая CV, добавляет шум и риск переобучения.

**Задача.** Напиши `add_features(df)`, добавляющую безопасные row-агрегаты (`row_sum/mean/std/max/min`) и пару взаимодействий топ-признаков. Собери `train_fe`, `test_fe`, список `features_fe`. Напиши `evaluate(frame, feats)`, считающую OOF AUC логрега на тех же `folds`, и сравни: `cv_auc` (без FE) против `fe_auc = evaluate(train_fe, features_fe)`.

??? tip "Подсказка"

    Row-агрегаты считаются по самим признакам объекта и таргета не касаются — это безопасно. Используй те же `folds`, что и на шаге 2, иначе сравнение «было/стало» станет нечестным. Оставляй группу фич, только если OOF вырос.

**Критерий шага:**

```python
assert {"row_sum", "row_mean", "row_std"} <= set(features_fe), "row-агрегаты должны быть добавлены"
assert fe_auc > cv_auc, "FE должен поднять OOF — иначе фичи не нужны"
assert 0.74 <= fe_auc <= 0.80, "после FE линейная модель около 0.756"
print(f"OK: было {cv_auc:.5f} -> стало {fe_auc:.5f} (+{fe_auc - cv_auc:.5f})")
```

??? success "Решение"

    ```python
    def add_features(df):
        df = df.copy()
        f = [c for c in df.columns if c.startswith("f")]
        # агрегаты по строке: суммарный «профиль» объекта (таргет не трогают — безопасно)
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

    fe_auc = evaluate(train_fe, features_fe)
    print("baseline feats:", round(cv_auc, 5))
    print("with FE       :", round(fe_auc, 5))
    ```

    **Почему так.** Две цифры рядом: было ~0.748 / стало ~0.756. Оставляешь группу фич, только если OOF вырос. Те же `folds` — поэтому сравнение честное. Это и есть «итеративный feature engineering»: не вываливаешь сто фич разом, а добавляешь группами и смотришь на компас. На линейной модели прирост скромный; на бустинге (шаг 5) те же фичи отыграют сильнее.

!!! warning "Утечка через feature engineering — самая частая ошибка"

    Любая статистика, посчитанная по таргету (например, mean target по категории), должна считаться **только внутри train-фолда**, иначе валидационные объекты «подсматривают» свой ответ. Это раздувает OOF и рушит его связь с LB. Правило: всё, что трогает таргет, — внутри цикла фолдов на `tr_idx`, никогда на полном train. Row-агрегаты выше безопасны: они таргет не видят.

Проверь понимание:

```text
Q: Признак коррелирует с таргетом на 0.98. Это удача или повод для расследования?
[ ] Удача — добавляем его и радуемся скачку AUC
[x] Повод для расследования: почти всегда это target leak — признак содержит информацию, недоступную в момент предсказания, и на private модель рухнет
[ ] Неважно, бустинг сам разберётся
> Корреляция 0.98 обычно означает, что признак посчитан "после факта" (leak). На test/private такого читерского сигнала не будет. Проверь происхождение признака, прежде чем доверять ему.
---
Q: Почему mean-target-encoding на полном train — это утечка, а row_sum по числовым фичам — нет?
[ ] Оба утечки, row_sum тоже нельзя
[x] Mean-target-encoding использует таргет валидационных объектов при расчёте их же признака (они подсматривают ответ, OOF завышается); row_sum складывает только сами фичи объекта и таргета не касается
[ ] Ни то, ни другое не утечка, encoding безопасен
> Кодирование по таргету на полном train даёт валидационным объектам подсмотреть собственную метку. Считай target-статистики только на tr_idx внутри фолда. row_sum таргет не трогает - утечки нет.
```

### Шаг 5: LightGBM + тюнинг через Optuna

**Зачем.** На табличке градиентный бустинг почти всегда бьёт линейные модели: ловит нелинейности и взаимодействия сам. Но у него много гиперпараметров. Перебирать руками — долго и нечестно (легко переобучиться на одном сплите). Optuna ищет умно (TPE-сэмплер), а оцениваем каждый набор по той же 5-фолдовой OOF-схеме — чтобы тюнинг оптимизировал именно то, что коррелирует с LB.

Прежде чем запускать — прикинь цену тюнинга:

```text
TASK: Optuna делает 40 trials, каждый trial оценивается по 5-фолдовой OOF-схеме. Сколько раз всего обучится модель LightGBM за весь тюнинг? Целое число.
ANSWER: 200
TOL: 1
PLACEHOLDER: целое число
EXPLAIN: 40 trials x 5 фолдов = 200 обучений модели. Поэтому каждое обучение должно быть быстрым (early_stopping обрезает лишние деревья), и поэтому же бессмысленно гнать 500 trials - стоимость линейно растёт, а прирост AUC после ~30 trials уходит в третий знак. Цена тюнинга = trials x folds.
```

**Задача.** Напиши `lgb_oof(params, frame, feats)` — обучает LightGBM по `folds` с `early_stopping` и возвращает OOF AUC. Напиши `objective(trial)` с разумным пространством гиперпараметров и запусти `study` (`direction="maximize"`, `TPESampler(seed=SEED)`, `n_trials=40`). Результат — в `study.best_value` и `study.best_params`.

**Критерий шага:**

```python
assert study.best_value > fe_auc, "бустинг обязан побить линейный baseline с FE"
assert 0.90 <= study.best_value <= 0.97, "на этих данных LightGBM выходит примерно на 0.94-0.95 OOF AUC"
assert isinstance(study.best_params, dict) and "learning_rate" in study.best_params
print(f"OK: best OOF AUC={study.best_value:.5f} (линейный был {fe_auc:.5f})")
```

??? success "Решение"

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

    **Почему так.** Лучший набор даёт OOF AUC около **0.95** — это огромный скачок над линейным baseline (~0.756): бустинг сам выловил нелинейности и взаимодействия трёх кластеров на класс. `early_stopping` подбирает число деревьев под каждый фолд автоматически. 40 trials хватает для playground; на больших данных увеличивай или ставь `study.optimize(..., timeout=3600)`. Важно: метрика тюнинга — та же OOF-схема, поэтому победитель Optuna реально полезен на LB, а не только на одном случайном сплите.

!!! tip "Не тюнингуй до посинения — закон убывающей отдачи"

    Первые 20-30 trials дают основной прирост, дальше AUC растёт на третий-четвёртый знак. Это переобучение под валидацию: ты подгоняешь гиперпараметры под конкретные фолды. Лучше потратить время на новую фичу или второй тип модели для ансамбля — там отдача выше. Зафиксируй разумные параметры и двигайся дальше.

### Шаг 6: Финальная модель и сабмит

**Зачем.** Лучшие параметры найдены — обучаем финальную модель по фолдам, собираем OOF (для записи в write-up) и усреднённое предсказание теста. Усреднение по фолдам — бесплатный bagging, снижает дисперсию.

**Задача.** Собери `best` (лучшие параметры + фиксированные `objective/metric/seed`). По `folds` заполни `oof_final` и усредни `test_final`. Посчитай финальный OOF AUC, запиши `submission_lgb.csv` и (в синтет-режиме) посмотри public/private.

**Критерий шага:**

```python
from pathlib import Path
final_auc = roc_auc_score(train_fe[TARGET], oof_final)
assert 0.90 <= final_auc <= 0.97, "финальный OOF близок к best_value Optuna"
assert Path("submission_lgb.csv").exists()
assert len(pd.read_csv("submission_lgb.csv")) == len(test)
print(f"OK: final OOF AUC={final_auc:.5f}")
```

??? success "Решение"

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

    **Почему так.** `submission_lgb.csv` и финальный OOF около **0.95**. В синтет-режиме сразу видишь public/private (оба ~0.945-0.947) — они близки к OOF, валидация надёжна. На реальном Kaggle отправляешь тем же `kaggle competitions submit` и сравниваешь LB-скор с OOF — они должны быть близко.

### Шаг 7: Ансамблирование — блендинг и стекинг

**Зачем.** Разные модели ошибаются по-разному. Если усреднить предсказания моделей с похожим скором, но разной природы (бустинг + линейная), итог обычно стабильнее и иногда выше каждой по отдельности. Но есть оговорка, которую ты сейчас увидишь числом: блендинг помогает, **только когда модели сопоставимы по силе**.

**Задача.** Собери OOF и test-предсказания логрега на `features_fe` (на тех же `folds`) в `logreg_oof`, `logreg_test`. Напиши `rank_blend(preds, weights)` (блендинг по рангам устойчив к разным масштабам вероятностей). Посчитай `blend_oof` = rank_blend([`oof_final`, `logreg_oof`], [0.8, 0.2]) и сравни его OOF AUC с одиночными моделями. Запиши `submission_blend.csv`.

**Критерий шага:**

```python
lgb_auc    = roc_auc_score(train_fe[TARGET], oof_final)
logreg_auc = roc_auc_score(train_fe[TARGET], logreg_oof)
blend_auc  = roc_auc_score(train_fe[TARGET], blend_oof)
assert blend_auc > logreg_auc, "бленд должен быть выше слабой модели"
assert blend_auc <= lgb_auc + 0.002, "бленд НЕ превзойдёт сильную модель, если вторая заметно слабее"
print(f"OK: logreg={logreg_auc:.5f}  lgb={lgb_auc:.5f}  blend={blend_auc:.5f}")
```

??? success "Решение"

    ```python
    from scipy.stats import rankdata

    # блендинг по рангам устойчивее к разным масштабам вероятностей у моделей
    def rank_blend(preds, weights):
        ranked = [rankdata(p) / len(p) for p in preds]
        return np.average(ranked, axis=0, weights=weights)

    # OOF логрега и OOF lgb — обе на одних фолдах, поэтому сравнимы
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

    **Почему так (важный честный момент).** Здесь блендинг lgb (~0.95) с логрегом (~0.756) даёт OOF около **0.94** — то есть **ниже** чистого LightGBM. Это не баг: вторая модель слишком слабая, и даже с весом 0.2 она тянет ансамбль вниз. Блендинг даёт прирост, только когда модели сопоставимы по силе и ошибаются по-разному (например, два бустинга на разных фичах или LightGBM + CatBoost). Правильный вывод для этого датасета: **в финал идёт чистый LightGBM**, а блендинг с заведомо слабой моделью — отрицательная идея, отвергнутая по OOF. Именно так компас и должен работать: проверил гипотезу числом — и не катишь то, что не помогло.

!!! tip "Веса ансамбля — только по OOF, не по public LB"

    Соблазн крутить веса по public-скору велик, но public — это маленькая шумная выборка. Подгонишь веса под неё — получишь shake-down на private. Веса (и вообще все решения) подбираются по OOF, public LB используется лишь как разовая sanity-проверка, что валидация не разъехалась с реальностью. **Стекинг** — следующий уровень: OOF-предсказания базовых моделей становятся признаками мета-модели (часто logreg) на том же CV; для capstone достаточно блендинга сопоставимых моделей.

### Шаг 8: Public vs private и solution write-up

**Зачем.** Соревнование заканчивается переключением на private leaderboard. Часто бывает **shake-up**: те, кто переобучился на public, падают, а те, кто доверял CV, поднимаются. Нужно понять механику и зафиксировать выводы в write-up — это и есть рефлексия, превращающая опыт в навык.

Прикинь масштаб public-части, по которой считается видимый скор:

```text
TASK: В тесте 20000 объектов, public-leaderboard считается на ~30% из них. Сколько примерно объектов в public-части? Целое число.
ANSWER: 6000
TOL: 300
PLACEHOLDER: целое число
EXPLAIN: 20000 x 0.30 = 6000 (в симуляторе случайная маска даёт ~6015). Именно потому, что public считается на маленькой выборке ~6000, скор на ней шумит, и подгонять под него решения = переобучаться на шум. Private (остальные ~14000) ты узнаёшь только в конце - и доверяешь OOF на всех 40000 train.
```

**Задача.** Прогони `score_submission` по всем трём сабмитам (`baseline`, `lgb`, `blend`) и собери таблицу public / private / shake (= private − public). Затем оформи `WRITEUP.md` по шаблону ниже с реальными числами своего прогона.

**Критерий шага:**

```python
pub_lgb, prv_lgb = score_submission("submission_lgb.csv")
assert abs(prv_lgb - pub_lgb) < 0.02, \
    "при честном OOF разрыв public vs private должен быть маленьким (нет переобучения на public)"
print(f"OK: lgb public={pub_lgb}, private={prv_lgb}, shake={round(prv_lgb - pub_lgb, 5)}")
```

??? success "Решение"

    ```python
    for name in ["submission_baseline.csv", "submission_lgb.csv", "submission_blend.csv"]:
        pub, prv = score_submission(name)
        print(f"{name:28s}  public={pub}  private={prv}  shake={round(prv - pub, 5)}")
    ```

    **Почему так.** Таблица public vs private по всем сабмитам. Разрывы крошечные (порядка ±0.005) — это плата за дисциплину OOF: модели не переобучены на public, поэтому private почти не отличается. У слабого baseline разрыв тоже мал, но сам уровень низкий (~0.74). Сильный LightGBM держит ~0.945 и на public, и на private — никакого shake-down. Большой положительный shake обычно означает, что модель робастна; большой отрицательный — что её подогнали под public-шум.

Теперь write-up. Это обязательный артефакт (текст, проверяется self-check, а не `assert`). Шаблон `WRITEUP.md`:

```markdown
# Solution write-up — <название соревнования>

## Итог
- Public LB: 0.9xx, private LB: 0.9xx, место N из M.
- Финальный сабмит: чистый LightGBM (блендинг с логрегом отвергнут по OOF).

## Валидация
- 5-fold StratifiedKFold, seed=42, метрика OOF ROC-AUC.
- CV и public LB совпали в пределах 0.00x — валидации доверял, решения принимал по OOF.

## Что сработало
- Градиентный бустинг (LightGBM) дал +0.19 над линейным baseline (0.756 -> 0.95).
- Row-агрегаты (sum/std/max) и 2 взаимодействия топ-признаков: небольшой плюс по OOF.
- Optuna (40 trials) по той же CV: довела до best.

## Что НЕ сработало
- Блендинг lgb + logreg: логрег слишком слаб, бленд ушёл НИЖЕ чистого lgb — отверг по OOF.
- Тюнинг сверх ~30 trials: прирост в третий знак, шум.

## Чему научился
- Главное умение — валидация, коррелирующая с LB. Без неё улучшения — гадание.
- Блендинг помогает только для сопоставимых по силе моделей; слабая модель тянет вниз.
- Веса ансамбля только по OOF; public LB — лишь sanity-check.
- Утечку через target-based фичи ловить заранее (всё по таргету — внутри фолда).
```

**Self-check write-up:**

- [ ] заполнены реальные числа твоего прогона (public/private/место), не плейсхолдеры;
- [ ] в «Что сработало» и «Что НЕ сработало» есть конкретные числа по OOF, а не общие слова;
- [ ] зафиксирован финальный сабмит и обоснование выбора по OOF (не по public);
- [ ] есть честный пункт про отвергнутые идеи.

Проверь понимание:

```text
Q: Что такое shake-up и почему гнавшиеся за public LB чаще падают на private?
[ ] Это смена метрики соревнования в последний день
[x] Это перестановка участников при переключении на private; public считается на маленькой выборке, и подгонка решений под неё — переобучение под шум, который на private другой
[ ] Это технический сбой Kaggle при подсчёте финального скора
> Public - маленькая шумная выборка. Выбор фич/весов по public-скору переобучает под её шум. На private шум другой, поэтому переобученные падают, а доверявшие CV поднимаются.
---
Q: Два финальных сабмита: A с OOF 0.870 и public 0.864, B с OOF 0.866 и public 0.871. Какой брать в финал?
[ ] B — у него выше public, а это видимый официальный скор
[x] A — OOF оценён на всём train и надёжнее; у B public ВЫШЕ OOF, это подозрительно (поймал удачу на public-части, что не воспроизведётся на private)
[ ] Любой — разница в третьем знаке несущественна
> Финал выбирают по OOF, а не по public. public у B выше его же OOF - почти наверняка удача на маленькой public-выборке, на private она исчезнет.
```

### Шаг 9: Собрать всё в один воспроизводимый пайплайн

**Зачем.** Разрозненные ячейки — не артефакт. Артефакт — один скрипт, который из `data/` собирает `submission.csv` детерминированно. Так твою работу можно проверить, переиспользовать и поставить на расписание.

**Задача.** Перенеси рабочие куски (шаги 2-6) в один файл `pipeline.py` с функцией `main()`: загрузка → `add_features` → `StratifiedKFold` → обучение по фолдам → OOF и усреднение теста → запись `submission.csv`. Зафиксируй `SEED`, `N_SPLITS` сверху. Запуск — одной командой.

??? tip "Подсказка"

    Не изобретай заново: `add_features` уже написана на шаге 4, цикл по фолдам — на шаге 6. Скелет ниже показывает каркас; тело функций перенеси из соответствующих шагов. Главное — убедиться, что повторный запуск даёт идентичный `submission.csv` (тот же `SEED`, тот же `uv.lock`).

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
    # ... запись submission.csv
    print("done: submission.csv")

if __name__ == "__main__":
    main()
```

```bash
uv run python pipeline.py        # один запуск -> один и тот же submission.csv
```

**Критерий шага (self-check).** Скелет выше содержит `...` — его надо дописать рабочим кодом из шагов 4-6, поэтому авто-`assert` тут нет. Проверь руками:

- [ ] `uv run python pipeline.py` отрабатывает без ошибок и пишет `submission.csv`;
- [ ] два последовательных запуска дают **побайтово** одинаковый `submission.csv` (детерминизм по `SEED`);
- [ ] OOF AUC, который печатает скрипт, совпадает с тем, что ты получил в ноутбуке (~0.95);
- [ ] `uv.lock` закоммичен — версии заморожены, результат воспроизводится на другой машине.

??? success "Решение (что должно получиться)"

    `pipeline.py` — это конденсат шагов 2-6 без EDA и без отвергнутого блендинга: `add_features` из шага 4, `StratifiedKFold` из шага 2, цикл обучения LightGBM с `best`-параметрами и усреднением теста из шага 6, запись `submission.csv`. Зафиксированные `SEED`, фолды и `uv.lock` гарантируют идентичный результат на любой машине. Это финальный артефакт capstone вместе с сабмитом, позицией в LB и write-up.

## Типичные ошибки

- **Гонка за public LB.** Подгонка под public — прямой путь к shake-down на private. Доверяй OOF, public — разовая проверка.
- **Меняющийся seed между экспериментами.** Сравниваешь шум вместо идей. Один `SEED` на фолды и модели, навсегда.
- **Утечка через target-based фичи.** Mean target по категории, посчитанный на полном train, раздувает OOF и рушит связь с LB. Всё по таргету — только внутри `tr_idx`.
- **Нет baseline-сабмита.** Полез сразу в сложную модель — не заметил, что формат сабмита неверный или CV не сходится с LB. Сначала тупой baseline до конца трубы.
- **Тюнинг без CV-обёртки.** Optuna на одном train/valid сплите переобучает гиперпараметры под этот сплит. Оценивай каждый trial по полной OOF-схеме.
- **Блендинг несопоставимых моделей.** Усреднять сильную модель со слабой — это тянуть итог вниз (видел числом на шаге 7). Бленд помогает только для моделей похожей силы и разной природы ошибок.
- **Accuracy при дисбалансе.** «87% точности» у модели, предсказывающей всем класс 0, — иллюзия. Метрика ранжирования AUC честнее при 12% позитивов.
- **Копирование чужого ноутбука без понимания.** Берёшь топ-решение, не понимая, почему оно работает, — не воспроизведёшь успех на другой задаче и не пройдёшь разбор на собеседовании.

!!! tip "AI-копилот в этом воркшопе"

    Используй ассистента как ускоритель механики, не как замену мышлению. Хорошие запросы: «сгенерируй каркас StratifiedKFold-цикла с OOF-массивом», «накидай objective-функцию Optuna для LightGBM с early stopping», «предложи 10 производных признаков для строки числовых фич и помечай, какие могут дать утечку», «объясни, почему мой public и CV разошлись на 0.02».

    Где AI подведёт именно здесь: (1) предложит блендинг сильной и слабой модели «для прироста», не проверив по OOF, — а ты видел, что это тянет вниз; (2) посчитает target-encoding на полном train и раздует OOF утечкой; (3) с радостью покрутит веса по public LB, если ты опишешь сценарий «у меня public выше» — обслужит переобучение, а не остановит. Плохой запрос: «реши за меня это соревнование». Любую фичу или модель, предложенную ассистентом, прогоняй через свой OOF-компас: помогла по CV — оставил, нет — выкинул, независимо от того, как уверенно её предложил AI.

## Критерий готовности

- [ ] Данные получены: kaggle API (реальное соревнование) или синтетический генератор отработал, в `data/` лежат train/test/sample_submission.
- [ ] Проведён EDA: дисбаланс, пропуски, train-vs-test сдвиг, проверка на target leak (max|corr| < 0.5).
- [ ] Построена StratifiedKFold-схема с фиксированным seed и OOF-массивом; каждый объект ровно раз в валидации.
- [ ] Baseline-сабмит дошёл до LB (реального или симулированного); CV и public сопоставлены (~0.74).
- [ ] Хотя бы одна группа фич добавлена и проверена по OOF (оставлена только если подняла CV).
- [ ] LightGBM обучен, гиперпараметры подобраны Optuna по той же CV-схеме (OOF ~0.95).
- [ ] Гипотеза ансамбля проверена по OOF и принято осознанное решение (здесь — отвергнуть бленд со слабой моделью).
- [ ] Посчитан public vs private (или получен private после закрытия) и понят shake-up.
- [ ] Написан `WRITEUP.md` с честными «сработало / не сработало / чему научился».
- [ ] Собран `pipeline.py` — один запуск даёт детерминированный `submission.csv`.
- [ ] Зафиксирована позиция в leaderboard.

## Бизнес-вывод

AUC в leaderboard — это ещё не ответ для бизнеса. Head of Retention не примет решение по числу 0.95; он примет его по фразе «вот кого звонить и сколько денег это вернёт». Переведи технический результат в решение для стейкхолдера.

- [ ] **Рекомендация (что делать).** Сформулируй действие, а не метрику: «запускаем адресную retention-кампанию по топ-20% риска из модели вместо веерной раздачи; порог отсечки — такой-то». Без жаргона про OOF и фолды.
- [ ] **Эффект в деньгах/метриках.** Привяжи AUC к деловому смыслу: насколько адресный таргетинг по скорам поднимает долю удержанных при том же бюджете 3 000 000 ₽/мес, и сколько это в ₽ удержанной выручки. Покажи дельту «было/стало», а не абсолютный AUC.
- [ ] **Риски и допущения.** Назови честно: модель валидна, только пока поведение игроков и состав базы стабильны (covariate shift = твой train-vs-test сдвиг и shake-up); скоры — это ранжирование риска, а не гарантия; адресность сужает охват — часть «тихих» оттоков можно пропустить.
- [ ] **Следующий шаг.** Предложи проверяемый план: A/B-пилот на 2 недели (модельный таргетинг против текущей эвристики), guardrail-метрика — стоимость удержания, и мониторинг дрейфа после раскатки.
- [ ] **Как подать стейкхолдеру.** Один слайд на языке решений: «при том же бюджете удерживаем ориентировочно в 1.5–2x больше игроков, риск — деградация при сдвиге базы, проверяем пилотом за 2 недели». Цифры в ₽ и игроках, а не в третьем знаке AUC.

## Развитие

- **Стекинг полноценно.** Сделай OOF-предсказания 3-4 **сопоставимых** базовых моделей признаками мета-модели на том же CV. Часто +0.001..0.005 над лучшей одиночной — но только если базовые модели сильные и разнообразные.
- **Второй сильный бустинг.** Добавь CatBoost или XGBoost рядом с LightGBM — вот это уже сопоставимые модели, и их бленд обычно помогает (в отличие от связки с логрегом из шага 7).
- **polars вместо pandas** на больших данных: feature engineering на сотнях миллионов строк, где pandas задыхается. Перепиши `add_features` на polars lazy API.
- **Pseudo-labeling.** Уверенные предсказания на test добавь в train как метки и переобучи — работает, когда test большой, но легко переобучиться, контролируй по CV.
- **GroupKFold / TimeSeriesSplit.** Если в данных есть группы (user_id) или время — обычный StratifiedKFold течёт. Освой схемы валидации под структуру данных (см. модуль M16).
- **Перенос в прод.** Возьми финальный пайплайн и оберни его в сервис из воркшопа W8 (FastAPI + мониторинг дрейфа) — соревновательная модель станет рабочим API.

## Что ты закрепил

- **Полный соревновательный цикл**: от kaggle API и первого baseline до ансамбля, write-up и воспроизводимого пайплайна.
- **Главное умение ML-практика** — валидация, коррелирующая с leaderboard. Local CV как приборная панель, по которой принимаются все решения, а не красивое число на public.
- **Дисциплина против переобучения**: фиксированный seed, OOF-оценка каждого изменения, веса ансамбля по CV, защита от утечки.
- **Градиентный бустинг и тюнинг**: LightGBM с Optuna в правильной CV-обёртке (~0.95 OOF против ~0.75 у линейного baseline).
- **Ансамблирование с трезвой головой**: rank-blend помогает только для сопоставимых моделей; слабую модель в бленд не берут — проверено числом.
- **Механика public vs private LB** и природа shake-up — почему дисциплинированные поднимаются, а гонящиеся за public падают.
- **Этика и навык**: понимать каждое решение, а не копировать чужой ноутбук вслепую. Именно это превращает участие в соревновании в переносимый навык.
