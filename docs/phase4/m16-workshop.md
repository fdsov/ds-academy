# Воркшоп M16 · Честный churn-пайплайн на логе ставок

<span class="lecture-meta">Воркшоп к модулю M16 · ориентир 5-7 ч</span>

## Что отрабатываем

Этот воркшоп берёт три блока модуля и собирает их в один сквозной артефакт:

- M16.7 — оконные агрегаты, лаги, recency и отношение окон, посчитанные строго назад от cutoff-даты (без утечки из будущего).
- M16.5 + M16.6 + M16.11 — кодирование категорий через `TargetEncoder` (out-of-fold), robust-масштабирование тяжёлых хвостов, флаг пропуска `is_missing`.
- M16.14 + M16.15 + M16.13 — `Pipeline` + `ColumnTransformer`, где препроцессинг структурно не способен утечь; тюнинг LightGBM через Optuna (TPE + pruner) во вложенной кросс-валидации.

Артефакт на выходе: один объект-пайплайн, словарь лучших гиперпараметров и две оценки качества — оптимистичная (внутренний CV) и честная (внешний цикл вложенной CV). Разрыв между ними — главный результат воркшопа: он показывает, на сколько процентов ты соврал бы себе без nested CV.

Стек 2026: `uv`, `polars` для оконных агрегатов, `scikit-learn` 1.5+ со встроенным `TargetEncoder`, `lightgbm`, `optuna` 4.x.

```bash
uv init m16-churn && cd m16-churn
uv add polars scikit-learn lightgbm optuna numpy
```

## Бизнес-кейс

!!! example "Ситуация"

    Ты — data scientist в iGaming-продукте. К тебе приходит **Head of Retention**: отдел удержания тратит ретеншн-бюджет «вслепую» — бонусы и звонки прилетают всем подряд уже после того, как игрок перестал заходить. По прикидкам, в активной базе ~6000 игроков, отток в горизонте после cutoff около 35%, средний игрок приносит ~8 000 ₽ маржи в квартал. Если научиться ловить уходящих **за 1-2 недели до того, как они окончательно отвалятся**, и точечно работать только по ним, можно вернуть часть из этих ~16-17 млн ₽ квартального риска вместо того, чтобы жечь бюджет на всю базу.

    - **Что зависит от ответа:** на твою модель завяжут таргетинг ретеншн-кампании на следующий квартал — какому списку игроков уйдут бонусы и кому позвонит саппорт. Завысишь качество модели — отдел зальёт бюджет в список, который на самом деле не работает.
    - **Главный риск:** «красивый» offline-AUC из-за утечки из будущего. Модель, обученная на признаках, которые подсматривают поведение после cutoff, в проде проваливается — а решение о бюджете уже принято.
    - **Ограничение:** есть только лог ставок до даты cutoff (`ts < CUTOFF`); никаких данных «из будущего». Срок — спринт, на выходе нужна одна честная цифра качества, которой можно доверять при планировании бюджета.

## Данные

Генерируем синтетический лог ставок с зашитым механизмом оттока: у «уходящих» игроков активность в последние 30 дней до cutoff затухает. Это даёт правильный сигнал именно в отношении окон `bet_7d / bet_30d`, как в M16.7. Seed фиксирован — запускается у любого.

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(42)
N_PLAYERS = 6000
CUTOFF = np.datetime64("2026-06-01")

players = []
for pid in range(N_PLAYERS):
    churn = rng.random() < 0.35
    country = rng.choice(["RU", "KZ", "UZ", "BR", "IN"], p=[.4, .2, .15, .15, .1])
    pay = rng.choice(["card", "crypto", "ewallet", "sbp", "rare_psp"],
                     p=[.45, .2, .2, .14, .01])
    base_rate = rng.uniform(0.3, 4.0)          # ставок в день в норме
    n_days = rng.integers(40, 120)
    for d in range(n_days):
        day = CUTOFF - np.timedelta64(int(n_days - d), "D")
        recent = (n_days - d) <= 30
        rate = base_rate * (0.15 if (churn and recent) else 1.0)
        for _ in range(rng.poisson(rate)):
            ts = day + np.timedelta64(int(rng.integers(0, 24)), "h")
            bet = float(np.exp(rng.normal(3.2, 1.1)))        # логнормальные суммы
            win = bet * rng.choice([0, rng.uniform(1.5, 3)], p=[.55, .45])
            dep = float(np.exp(rng.normal(4.0, 1.0))) if rng.random() < 0.08 else 0.0
            players.append((pid, ts, bet, win, dep, country, pay, int(churn)))

bets = pl.DataFrame(
    players,
    schema=["player_id", "ts", "bet_amount", "win_amount",
            "deposit_amount", "country", "payment_method", "churn"],
    orient="row",
).with_columns(pl.col("ts").cast(pl.Datetime))

labels = bets.group_by("player_id").agg(
    pl.col("churn").first(),
    pl.col("country").first(),
    pl.col("payment_method").first(),
)
print(bets.shape, "событий;", labels["churn"].mean(), "доля оттока")
```

`churn` здесь — это правда о будущем (что было *после* cutoff). В признаки она не попадает; она нужна только как таргет `y`.

## Ход работы

### Шаг 1: оконные признаки строго назад от cutoff

Зачем: отрабатываем M16.7. Каждое окно смотрит только в прошлое относительно `CUTOFF`. Самая частая ошибка джуниора — посчитать агрегаты по всему логу игрока, включая дни после момента предсказания. Здесь мы режем лог по `ts < CUTOFF` один раз и считаем окна 7/30 дней плюс recency.

```python
CUT = pl.lit(CUTOFF).cast(pl.Datetime)
hist = bets.filter(pl.col("ts") < CUT)

def window_aggs(df, days, suffix):
    start = CUT - pl.duration(days=days)
    return (df.filter(pl.col("ts") >= start)
              .group_by("player_id")
              .agg(
                  pl.col("bet_amount").sum().alias(f"bet_sum_{suffix}"),
                  pl.col("bet_amount").mean().alias(f"bet_mean_{suffix}"),
                  pl.col("bet_amount").std().alias(f"bet_std_{suffix}"),   # импульсивность
                  pl.len().alias(f"n_bets_{suffix}"),
                  pl.col("ts").dt.date().n_unique().alias(f"active_days_{suffix}"),
                  pl.col("deposit_amount").sum().alias(f"dep_sum_{suffix}"),
              ))

f7  = window_aggs(hist, 7,  "7d")
f30 = window_aggs(hist, 30, "30d")

recency = (hist.group_by("player_id")
               .agg(pl.col("ts").max().alias("last_ts"))
               .with_columns(
                   (CUT - pl.col("last_ts")).dt.total_days().alias("days_since_last_bet"))
               .select("player_id", "days_since_last_bet"))

feats = (f7.join(f30, on="player_id", how="full", coalesce=True)
           .join(recency, on="player_id", how="full", coalesce=True))
```

Что получилось: таблица признаков на игрока, где ни одна колонка не знает о событиях после cutoff. `std` для игроков с одной ставкой будет `null` — это нормально, заполним на шаге препроцессинга и пометим флагом.

### Шаг 2: отношение окон, циклический час и лаг активности

Зачем: M16.7 говорит, что отношение `активность за 7д / активность за 30д` ловит ранний отток лучше любого абсолютного признака — у ровного игрока оно держится у $7/30 \approx 0.23$, при затухании падает к нулю. Плюс циклический час пика активности (sin/cos, период 24) и доля выигрыша.

```python
peak_hour = (hist.group_by("player_id")
                 .agg(pl.col("ts").dt.hour().mean().alias("peak_hour")))

feats = (feats.join(peak_hour, on="player_id", how="left")
              .with_columns([
                  (pl.col("bet_sum_7d").fill_null(0) /
                   (pl.col("bet_sum_30d").fill_null(0) + 1)).alias("bet_ratio_7_30"),
                  (pl.col("active_days_7d").fill_null(0) /
                   (pl.col("active_days_30d").fill_null(0) + 1)).alias("active_ratio_7_30"),
                  (2 * np.pi * pl.col("peak_hour") / 24).sin().alias("hour_sin"),
                  (2 * np.pi * pl.col("peak_hour") / 24).cos().alias("hour_cos"),
              ]))

data = feats.join(labels.select("player_id", "churn", "country", "payment_method"),
                  on="player_id", how="left")
print(data.select(["bet_ratio_7_30", "days_since_last_bet", "churn"]).describe())
```

Что получилось: 15+ признаков (recency, frequency, monetary, волатильность, два отношения окон, циклический час). Беглый `describe` уже должен показать, что у `churn=1` медиана `bet_ratio_7_30` заметно ниже.

!!! question "Проверь себя"

    1. Почему окна в `window_aggs` фильтруются по `ts >= start` И опираются на уже отрезанный `hist`, а не на полный `bets`?
    2. Что сломается в честности оценки, если `peak_hour` посчитать по всему `bets`, а не по `hist`?

??? success "Ответы"

    1. `hist` отрезает всё после cutoff (`ts < CUT`), а `start` ограничивает окно слева. Вместе это даёт ровно отрезок `[cutoff - days, cutoff)` — признак использует только прошлое. Фильтрация по полному `bets` впустила бы будущее и дала утечку.
    2. `peak_hour` стал бы знать о поведении игрока после cutoff — это утечка из будущего. Offline-AUC взлетит, в проде модель провалится. Любой признак считается строго до cutoff.

### Шаг 3: пайплайн без утечки

Зачем: M16.14 — любая обучаемая трансформация (median impute, robust scale, target encoding) должна фититься только на train-части каждого фолда. Собираем `ColumnTransformer`: числовые → median impute с `add_indicator=True` (флаг `is_missing` из M16.11) + `RobustScaler` (тяжёлые хвосты из M16.6); категории → constant impute + `TargetEncoder` (out-of-fold cross-fitting из коробки, M16.5). Scaler перед LightGBM не ставим осознанно — для деревьев это потерянное время.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

df = data.to_pandas()
y = df["churn"].astype(int).values
num_cols = ["bet_sum_7d", "bet_mean_7d", "bet_std_7d", "n_bets_7d",
            "bet_sum_30d", "bet_std_30d", "dep_sum_30d", "days_since_last_bet",
            "bet_ratio_7_30", "active_ratio_7_30", "hour_sin", "hour_cos"]
cat_cols = ["country", "payment_method"]
X = df[num_cols + cat_cols]

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", RobustScaler()),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encode", TargetEncoder(smooth="auto")),   # cross-fitting только в fit_transform
])
prep = ColumnTransformer([("num", numeric, num_cols),
                          ("cat", categorical, cat_cols)])

pipe = Pipeline([
    ("prep", prep),
    ("model", LGBMClassifier(objective="binary", n_estimators=500,
                             learning_rate=0.05, random_state=42, verbosity=-1)),
])
```

Что получилось: один объект `pipe`. При кросс-валидации он фитится заново на каждом train-фолде и только применяется к val-фолду — `TargetEncoder` и `RobustScaler` физически не видят валидацию.

!!! tip "Почему именно TargetEncoder, а не groupby().mean()"

    Наивный `df.groupby("country").churn.mean()` по всему трейну — это утечка номер один: значение признака для объекта зависит от его собственного таргета. `TargetEncoder` в `fit_transform` делает внутренний cross-fitting (каждый фолд кодируется по остальным), а в `transform` на валидации применяет статистики, выученные на train. Внутри `Pipeline` под CV вызывается ровно правильный путь.

### Шаг 4: baseline-оценка и контроль утечки

Зачем: зафиксировать честную точку отсчёта обычной (не вложенной) CV до тюнинга и проверить, что пайплайн не протекает. Если базовый AUC внезапно около 0.99 — где-то утечка.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"baseline ROC-AUC: {base.mean():.4f} +/- {base.std():.4f}")
```

Что получилось: реалистичный AUC (обычно 0.75-0.88 на этой синтетике). Запомни число — с ним сравним результат тюнинга.

!!! question "Проверь себя"

    1. Зачем `StandardScaler`/`RobustScaler` здесь только в числовой ветке, а перед `LGBMClassifier` его нет?
    2. Что было бы с базовым AUC, если бы `TargetEncoder` мы применили к `X` один раз ДО `cross_val_score`?

??? success "Ответы"

    1. Дерево режет признак по порогу $x < t$; монотонное масштабирование не меняет порядок значений и ни одного сплита — для LightGBM это работа впустую. Robust-scaler оставлен в ветке как часть учебного препроцессинга/на случай линейного бейзлайна, но перед бустингом отдельный scaler не нужен.
    2. Энкодер «увидел» бы таргет всех объектов, включая будущие val-фолды каждого split — утечка. AUC завысился бы (часто к 0.95+), а на проде модель просела бы. Поэтому энкодер живёт внутри пайплайна.

### Шаг 5: тюнинг Optuna во вложенной CV

Зачем: M16.13 + M16.15. Внутренний цикл подбирает гиперпараметры LightGBM через TPE с `MedianPruner`; внешний цикл честно оценивает на фолдах, которых внутренний не видел. Разрыв между `study.best_value` (оптимистичный) и средним по внешним фолдам (честный) — это цена подгонки под валидацию.

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

def make_objective(Xtr, ytr):
    def objective(trial):
        params = {
            "model__num_leaves":       trial.suggest_int("num_leaves", 16, 256),
            "model__max_depth":        trial.suggest_int("max_depth", 3, 12),
            "model__learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "model__subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "model__reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe.set_params(**params)
        sc = cross_val_score(pipe, Xtr, ytr, cv=inner, scoring="roc_auc", n_jobs=-1)
        return sc.mean()
    return objective

outer_scores, best_params_per_fold = [], []
from sklearn.metrics import roc_auc_score

for k, (tr, te) in enumerate(outer.split(X, y)):
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y[tr], y[te]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(make_objective(Xtr, ytr), n_trials=40, show_progress_bar=False)
    pipe.set_params(**{f"model__{k_}": v for k_, v in study.best_params.items()})
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    outer_scores.append(auc)
    best_params_per_fold.append(study.best_params)
    print(f"fold {k}: inner_best={study.best_value:.4f}  outer_honest={auc:.4f}")

print(f"\nЧестная nested-CV AUC: {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")
print(f"Оптимистичный inner-best (среднее): "
      f"{np.mean([s for s in outer_scores]):.4f}")
```

Что получилось: на каждом внешнем фолде `inner_best` (по которому выбирали параметры) систематически чуть выше, чем `outer_honest` (на невиданных данных). Среднее `outer_honest` — это число, которое можно нести бизнесу. Если бы ты отчитался `study.best_value`, ты завысил бы качество на разницу между этими столбцами.

### Шаг 6: финальный артефакт

Зачем: после честной оценки фиксируем рабочую конфигурацию. Финальную модель обучаем на всех данных с параметрами, чаще всего выигрывавшими по фолдам.

```python
from collections import Counter

flat = {kp: np.median([bp[kp] for bp in best_params_per_fold])
        for kp in best_params_per_fold[0]}
final_params = {f"model__{k_}": (int(v) if k_ in ("num_leaves", "max_depth") else v)
                for k_, v in flat.items()}
pipe.set_params(**final_params)
pipe.fit(X, y)

import joblib
joblib.dump(pipe, "churn_pipeline.joblib")
print("Финальные параметры:", final_params)
print("Честная оценка качества:", round(float(np.mean(outer_scores)), 4))
```

Что получилось: три обещанных артефакта — `churn_pipeline.joblib` (препроцессинг + модель в одном объекте), словарь `final_params`, и честная nested-CV оценка качества.

## Критерий готовности

- [ ] Все оконные агрегаты, лаги и recency считаются строго по `hist` (`ts < CUTOFF`) — ни один признак не видит будущего.
- [ ] Построено 15+ признаков, включая `bet_ratio_7_30` и циклический час (sin/cos).
- [ ] `TargetEncoder`, `SimpleImputer` и `RobustScaler` живут внутри `Pipeline`/`ColumnTransformer`, а не фитятся до split.
- [ ] Есть флаг пропуска (`add_indicator=True`), а не только заполнение медианой.
- [ ] Перед LightGBM нет отдельного scaler, и ты можешь объяснить почему (сплит по порогу).
- [ ] Optuna запущена с TPE-семплером и pruner, минимум 40 trials на внутренний фолд.
- [ ] Выведены ОБА числа: оптимистичный inner-best и честный outer-AUC, и виден разрыв между ними.
- [ ] Финальный пайплайн сериализован и переобучается из коробки на новых данных без ручного препроцессинга.

## Бизнес-вывод

Технический артефакт готов — теперь переведи его в решение для Head of Retention на языке бюджета и риска, а не AUC и фолдов. Собери короткий вывод по чек-листу:

- [ ] **Рекомендация.** Сформулируй, что делать: работать ретеншн-кампанией не по всей базе, а только по верхнему сегменту риска оттока (например, топ-N игроков по предсказанной вероятности), и сними нагрузку с остальных.
- [ ] **Эффект в деньгах.** Переведи качество модели в деньги: при честной nested-CV AUC и заданном пороге — сколько уходящих игроков ловим, какую долю из ~16-17 млн ₽ квартального риска покрываем, и насколько сужается список для бюджета против «бонус всем».
- [ ] **Риски и допущения.** Назови честную цифру (среднее `outer_honest`), а не оптимистичный `inner-best`, и проговори разрыв между ними как меру неопределённости. Отметь допущения: синтетический сигнал затухания, фиксированный cutoff, отсутствие данных о реакции игрока на бонус (causal-эффект кампании не измерен).
- [ ] **Следующий шаг.** Предложи проверить модель на реальном логе и закрепить эффект A/B-тестом: одна группа риска получает кампанию, контрольная — нет; меряем разницу в удержании и марже, а не offline-AUC.
- [ ] **Как подать стейкхолдеру.** Один слайд: «Модель отделяет уходящих от остающихся с честным качеством X; точечная работа по группе риска вместо всей базы экономит бюджет и возвращает Y ₽ в квартал; решение подтверждаем A/B-тестом перед раскаткой». Без терминов про энкодеры и фолды.

## Развитие

- Сравни кодировки: прогони `country`/`payment_method` через one-hot, label, частотное и target encoding (M16.5) для логистической регрессии и LightGBM. Покажи, что для линейной модели и для дерева лучшие способы разные.
- Отбор признаков (M16.12): прогони mutual-information фильтр, permutation importance и `RFECV` на своём наборе и сравни топ-списки. Объясни, почему фильтр упускает признаки, сильные только в паре.
- Кривая сходимости: на одном внешнем фолде сравни Optuna, random search и grid search при равном бюджете обучений — построй график «лучший AUC от номера trial».
- Бизнес-интерпретация: посчитай SHAP по финальной модели и объясни нетехническому продакту, почему `bet_ratio_7_30` и `days_since_last_bet` доминируют в предсказании оттока.
