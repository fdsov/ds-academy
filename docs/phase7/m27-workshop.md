# Воркшоп M27 · Аудит справедливости антифрод-модели

<span class="lecture-meta">Воркшоп к модулю M27 · ориентир 3-5 ч</span>

## Что отрабатываем

Этот воркшоп закрепляет руками самое опасное место модуля: модель с отличной агрегатной метрикой может быть систематически несправедлива к подгруппе, и эту несправедливость нельзя «убрать» удалением защищённого признака — она протекает через прокси (redundant encoding из M27.4). Ты построишь антифрод-скоринг на данных со встроенной исторической предвзятостью, измеришь три определения справедливости из M27.5 (demographic parity, equalized odds, predictive parity/калибровка), увидишь теорему о невозможности не как формулу, а как числа на своих данных, попробуешь митигацию порогом по группам через `fairlearn` и применишь простую k-анонимность к квазиидентификаторам из M27.6.

Понятия, которые отрабатываются напрямую:

- историческая предвзятость через разные base rates по группам;
- redundant encoding: защищённый атрибут протекает через коррелированные фичи;
- demographic parity difference, equalized odds difference, групповая калибровка;
- теорема о невозможности (Chouldechova, Kleinberg): при $p_0 \neq p_1$ три критерия несовместимы;
- митигация постобработкой (`ThresholdOptimizer`) и её цена в агрегатной точности;
- k-анонимность и нарушения на квазиидентификаторах.

Артефакт на выходе: один markdown-файл `fairness_audit.md` — fairness & privacy audit модели, который не стыдно показать регулятору: таблица метрик по подгруппам, разрывы до и после митигации, групповая калибровка, отчёт k-анонимности, выбранное определение справедливости с обоснованием и предложение human-in-the-loop.

## Данные

Синтетика с фиксированным seed, запускается у любого. Региональный признак `region` — защищённый атрибут. База фрода в нём различается (историческая предвзятость), а фичи скоринга коррелируют и с фродом, и с регионом — именно так атрибут протекает в модель, даже если сам `region` в обучении не участвует.

Окружение через `uv`:

```bash
uv init m27-fairness && cd m27-fairness
uv add "fairlearn>=0.11" scikit-learn pandas numpy
```

Генератор данных. Сохрани как `data.py` — его импортируют остальные шаги.

```python
import numpy as np
import pandas as pd

def make_data(n: int = 20000, seed: int = 42):
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, size=n)
    base_fraud_rate = np.where(region == 0, 0.10, 0.04)   # историческая предвзятость
    y = (rng.random(n) < base_fraud_rate).astype(int)

    deposit_velocity = rng.normal(0, 1, n) + 0.7 * y + 0.5 * region
    night_activity   = rng.normal(0, 1, n) + 0.5 * y + 0.4 * region
    failed_deposits  = rng.poisson(0.3 + 0.8 * y + 0.3 * region, n).astype(float)

    X = pd.DataFrame({
        "deposit_velocity": deposit_velocity,
        "night_activity": night_activity,
        "failed_deposits": failed_deposits,
    })
    quasi = pd.DataFrame({
        "age_band": rng.choice(["18-25", "26-35", "36-45", "46+"], n),
        "region":   np.where(region == 0, "north", "south"),
        "vip_tier": rng.choice(["bronze", "silver", "gold", "platinum"], n,
                               p=[0.55, 0.30, 0.12, 0.03]),
    })
    return X, y, region, quasi

if __name__ == "__main__":
    X, y, a, q = make_data()
    print("base rate group 0:", round(y[a == 0].mean(), 4))
    print("base rate group 1:", round(y[a == 1].mean(), 4))
```

Запусти и убедись, что base rates разные (примерно 0.10 против 0.04) — это предпосылка теоремы о невозможности. Если бы они совпадали, конфликта определений не было бы.

## Ход работы

### Шаг 1: обучить модель и увидеть, что агрегат лжёт

Зачем. Отрабатываем главную ловушку M27.12: accuracy «в среднем» прячет дискриминацию подгруппы. `region` в обучение не подаётся — проверяем тезис, что fairness нельзя обеспечить удалением входа.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
    demographic_parity_difference, equalized_odds_difference,
)
from data import make_data

X, y, a, quasi = make_data()
X_tr, X_te, y_tr, y_te, a_tr, a_te = train_test_split(
    X, y, a, test_size=0.3, random_state=0, stratify=y
)

model = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
y_pred = (proba >= 0.5).astype(int)

print("Aggregate accuracy:", round(accuracy_score(y_te, y_pred), 4))

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate,
             "TPR": true_positive_rate, "FPR": false_positive_rate},
    y_true=y_te, y_pred=y_pred, sensitive_features=a_te,
)
print("\nПо группам:\n", mf.by_group)
print("\nDP diff:", round(demographic_parity_difference(y_te, y_pred, sensitive_features=a_te), 4))
print("EO diff:", round(equalized_odds_difference(y_te, y_pred, sensitive_features=a_te), 4))
```

Что получилось. Агрегатная accuracy высокая и выглядит благополучно, но `by_group` показывает заметно разный `selection_rate` и разный FPR/TPR между регионами. `region` не был фичей — но `deposit_velocity`, `night_activity` и `failed_deposits` несут его сигнал. Это redundant encoding в чистом виде: модель восстановила защищённый атрибут из коррелятов.

!!! question "Проверь себя"

    1. Почему `selection_rate` различается между регионами, хотя `region` не подавался в модель?
    2. Какой из трёх типов предвзятости (M27.4) встроен в генератор данных через разные base rates?
    3. Что именно скрывает агрегатная accuracy?

??? success "Ответы"

    1. Защищённый атрибут протекает через коррелированные фичи (redundant encoding): все три признака содержат слагаемое `+ k*region`. Удаление колонки `region` не убирает её сигнал из данных.
    2. Историческая предвзятость: $P(Y \mid A)$ зависит от региона по «несправедливой» причине, данные честно это фиксируют.
    3. Разрыв в качестве по подгруппам — разные TPR/FPR и долю положительных решений между группами. Fairness измеряется на выходе по подгруппам, а не агрегатом.

### Шаг 2: посчитать три определения справедливости

Зачем. M27.5 даёт три формализации. DP и equalized odds считает `fairlearn` напрямую, а predictive parity (калибровку) считаем руками через биннинг `predict_proba` — это и есть «надёжность скоров по группам».

```python
import pandas as pd

def group_calibration(proba, y_true, sens, n_bins=5):
    df = pd.DataFrame({"p": proba, "y": y_true, "a": sens})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1),
                       include_lowest=True)
    out = (df.groupby(["bin", "a"], observed=True)
             .agg(mean_pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
             .reset_index())
    return out

calib = group_calibration(proba, y_te.values if hasattr(y_te, "values") else y_te, a_te)
print(calib.to_string(index=False))
```

Что получилось. Для каждого бина скоров сравни столбцы `actual` между группой 0 и 1. Если в бине «скор 0.6-0.8» фактическая доля фродеров в группе north заметно выше, чем в south, — predictive parity нарушена: «0.7» означает разный реальный риск в разных группах. Сейчас модель скорее откалибрована (бустинг на сырых данных тяготеет к калибровке), а DP и equalized odds — нарушены. Запомни эту картину: она перевернётся на следующем шаге.

### Шаг 3: теорема о невозможности на числах

Зачем. M27.5 утверждает: при $p_0 \neq p_1$ нельзя одновременно держать калибровку, равный FPR и равный FNR. Проверим это не формулой, а экспериментом — выровняем equalized odds митигацией и посмотрим, что станет с калибровкой.

```python
from fairlearn.postprocessing import ThresholdOptimizer

postproc = ThresholdOptimizer(
    estimator=model, constraints="equalized_odds",
    prefit=True, predict_method="predict_proba",
)
postproc.fit(X_tr, y_tr, sensitive_features=a_tr)
y_fair = postproc.predict(X_te, sensitive_features=a_te)

mf_fair = MetricFrame(
    metrics={"TPR": true_positive_rate, "FPR": false_positive_rate},
    y_true=y_te, y_pred=y_fair, sensitive_features=a_te,
)
print("EO diff до :", round(equalized_odds_difference(y_te, y_pred, sensitive_features=a_te), 4))
print("EO diff после:", round(equalized_odds_difference(y_te, y_fair, sensitive_features=a_te), 4))
print("accuracy до :", round(accuracy_score(y_te, y_pred), 4))
print("accuracy после:", round(accuracy_score(y_te, y_fair), 4))
print("\nTPR/FPR по группам после митигации:\n", mf_fair.by_group)
```

Подкрепим алгеброй из M27.5 — формулой PPV. Equalized odds фиксирует одинаковые TPR и FPR в группах, но base rates разные, поэтому PPV расходится:

```python
def ppv(p, tpr, fpr):
    return p * tpr / (p * tpr + (1 - p) * fpr)

p0, p1 = y_te[a_te == 0].mean(), y_te[a_te == 1].mean()
tpr, fpr = 0.8, 0.1   # представим, что equalized odds выполнен ровно
print("PPV group0:", round(ppv(p0, tpr, fpr), 4))
print("PPV group1:", round(ppv(p1, tpr, fpr), 4))
```

Что получилось. После `ThresholdOptimizer` EO diff падает почти к нулю — equalized odds теперь держится. Но: агрегатная accuracy просела (осознанная жертва, о которой говорит M27.5), и если пересчитать `group_calibration` на `y_fair`, увидишь, что калибровка разъехалась. Ручной расчёт PPV закрепляет: при $p_0=0.10$ и $p_1=0.04$ одинаковые TPR/FPR дают разные PPV. Нельзя выровнять всё сразу — это и есть теорема о невозможности, не лень инженера, а алгебра.

!!! question "Проверь себя"

    1. Почему после выравнивания equalized odds разъезжается калибровка?
    2. Что в коде является ценой митигации и где её видно?
    3. При каком условии теорема о невозможности перестала бы «срабатывать»?

??? success "Ответы"

    1. При разных base rates фиксация TPR и FPR через формулу $\text{PPV}=\frac{p\,\text{TPR}}{p\,\text{TPR}+(1-p)\,\text{FPR}}$ даёт разный PPV, то есть нарушение predictive parity. Три критерия математически несовместимы.
    2. Падение агрегатной accuracy после `ThresholdOptimizer` — видно в сравнении `accuracy до/после`. Часть точности обменяна на равенство ошибок между группами.
    3. Если бы base rates совпадали ($p_0 = p_1$) или классификатор был идеальным — конфликта определений не было бы.

### Шаг 4: применить k-анонимность к квазиидентификаторам

Зачем. M27.6: удаление прямых идентификаторов не делает данные анонимными, реидентификация идёт по комбинации квазиидентификаторов. Найдём записи-нарушители и применим обобщение.

```python
def k_anon_violations(df, quasi_ids, k):
    sizes = df.groupby(quasi_ids, observed=True).size().rename("n").reset_index()
    return sizes[sizes["n"] < k]

quasi_ids = ["age_band", "region", "vip_tier"]
viol = k_anon_violations(quasi, quasi_ids, k=5)
print("Комбинаций с размером < 5:", len(viol))
print("Записей под риском реидентификации:", int(viol["n"].sum()))

# обобщение vip_tier снижает гранулярность и поднимает анонимность
quasi_gen = quasi.copy()
quasi_gen["vip_tier"] = quasi_gen["vip_tier"].map(
    {"bronze": "mass", "silver": "mass", "gold": "vip", "platinum": "vip"}
)
viol_gen = k_anon_violations(quasi_gen, quasi_ids, k=5)
print("После обобщения vip_tier, комбинаций < 5:", len(viol_gen))
```

Что получилось. До обобщения часть комбинаций `{age_band, region, vip_tier}` встречается реже 5 раз — эти игроки потенциально реидентифицируемы, особенно редкий `platinum`. После схлопывания тиров в `mass`/`vip` число нарушений падает: это компромисс приватность-полезность из M27.6 (обобщение снижает детализацию). Зафиксируй оба числа для аудита.

### Шаг 5: собрать артефакт аудита

Зачем. M27.8 и мини-проект M27.13: аудит — единый документ с метриками по подгруппам, выбранным определением fairness и обоснованием. Соберём всё посчитанное в `fairness_audit.md`.

```python
from datetime import date

dp = demographic_parity_difference(y_te, y_pred, sensitive_features=a_te)
eo_before = equalized_odds_difference(y_te, y_pred, sensitive_features=a_te)
eo_after = equalized_odds_difference(y_te, y_fair, sensitive_features=a_te)

report = f"""# Fairness & Privacy Audit · Антифрод-скоринг
Дата: {date.today()} · Защищённый атрибут: region (north/base 0.10, south/base 0.04)

## 1. Метрики по подгруппам (до митигации)
{mf.by_group.round(4).to_markdown()}

Агрегатная accuracy: {accuracy_score(y_te, y_pred):.4f}
Demographic parity difference: {dp:.4f}
Equalized odds difference: {eo_before:.4f}

## 2. Теорема о невозможности
Base rates различаются (0.10 vs 0.04). Выравнивание equalized odds
(ThresholdOptimizer) дало EO diff {eo_after:.4f}, но калибровка по группам
разошлась, а агрегатная accuracy просела. Одновременно выполнить
demographic parity, equalized odds и predictive parity невозможно.

## 3. Выбранное определение fairness
Решение о блокировке выплаты лишает игрока законного блага, цена ложного
отказа ложится на индивида -> приоритет equal opportunity / equalized odds.
Калибровкой осознанно жертвуем, фиксируем это решение здесь (M27.5).

## 4. k-анонимность
Квазиидентификаторы: {quasi_ids}. Комбинаций с размером < 5 до обобщения:
{len(viol)}; после обобщения vip_tier: {len(viol_gen)}.

## 5. Human-in-the-loop (GDPR ст. 22, EU AI Act)
Антифрод-блокировки = значимое автоматическое решение. Требуется:
ревью человеком до окончательной блокировки выплаты, механизм обжалования,
логирование решений, периодический повторный аудit при дрейфе данных.
"""

with open("fairness_audit.md", "w", encoding="utf-8") as f:
    f.write(report)
print("Аудит записан в fairness_audit.md")
```

Что получилось. Файл `fairness_audit.md` собирает воедино: метрики по подгруппам (а не агрегат), демонстрацию теоремы о невозможности на твоих числах, явно выбранное и обоснованное определение справедливости, отчёт k-анонимности и привязку к регуляции (ст. 22 GDPR, риск-категория EU AI Act из M27.7). Это и есть требуемый артефакт.

## Критерий готовности

- [ ] Модель обучена без признака `region`, но `MetricFrame.by_group` показывает разрыв по группам (redundant encoding продемонстрирован).
- [ ] Посчитаны demographic parity difference, equalized odds difference и групповая калибровка через биннинг.
- [ ] Показано численно, что после выравнивания equalized odds калибровка расходится, а accuracy просела — теорема о невозможности на своих данных.
- [ ] Ручной расчёт PPV для двух групп подтверждает конфликт при разных base rates.
- [ ] Найдены и сокращены нарушения k-анонимности обобщением квазиидентификатора.
- [ ] Создан `fairness_audit.md` с метриками по подгруппам, выбранным определением fairness + обоснованием, k-анонимностью и human-in-the-loop.

## Развитие

- Добавь дифференциально-приватный счётчик из M27.6 (`true_count + np.random.laplace(0, 1/eps)`) и включи в аудит запрос «сколько игроков south со скором выше 0.8» для $\varepsilon \in \{0.1, 1, 10\}$; объясни, что значит «бюджет израсходован» при множественных запросах.
- Сравни постобработку (`ThresholdOptimizer`) с in-processing митигацией `fairlearn.reductions.ExponentiatedGradient` под тем же ограничением и оцени, какой подход дешевле по accuracy.
- Прогони аудит на «дрейфующих» данных: сгенерируй второй срез с другими base rates и покажи, что справедливая вчера модель стала несправедливой — обоснование периодичности аудита из M27.8.
- Расширь защищённый атрибут до небинарного (3-4 региона) и проверь, что разрывы fairness считаются попарно, а не одним числом.
