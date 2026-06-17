# Воркшоп M24 · Retention-модель в деньгах

<span class="lecture-meta">Воркшоп к модулю M24 · ориентир 4-6 ч</span>

## Что отрабатываем

Из модуля M24 берём самое денежное: перевод метрики модели в P&L и причинную оценку эффекта. Конкретно отрабатываем четыре понятия:

- матрица стоимости ошибок и порог под прибыль ($t^\ast$), а не под F1 (M24.3);
- калибровка скора перед умножением на деньги (M24.3, Brier);
- unit-экономика и связь retention → LTV → NGR (M24.4);
- инкрементальность и uplift-таргетинг бонуса вместо churn-ранжирования (M24.6), с ATE и бутстрап-ДИ (M24.5).

Артефакт на выходе: один скрипт, который считает (а) оптимальный порог retention-кампании в деньгах, (б) прирост LTV когорты, (в) инкрементальный эффект бонуса по uplift с доверительным интервалом, и (г) печатает финальный бизнес-вывод на языке решений — запускать или нет, сколько денег, какой риск.

Мы намеренно строим всю цепочку на одном синтетическом датасете с фиксированным seed, чтобы цифры были воспроизводимы у любого.

## Данные

Генерируем гемблинг-когорту: для каждого игрока есть фичи (депозиты, частота ставок, давность), истинная склонность к оттоку и — главное — заложенный причинный эффект бонуса, разный для разных типов игроков (sure things / lost causes / persuadables). Этот заложенный эффект и есть «правда», с которой мы потом сверим оценки.

```bash
uv init m24-workshop && cd m24-workshop
uv add numpy pandas scikit-learn lightgbm
```

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 40_000

deposits = rng.gamma(2.0, 40, N).round(2)
bet_freq = rng.poisson(8, N)
recency = rng.integers(1, 60, N)

# латентный риск оттока: реже ставит, давно не заходил, мало депозитил -> выше риск
z = 0.9 - 0.05 * bet_freq + 0.03 * recency - 0.004 * deposits
churn_p = 1 / (1 + np.exp(-z))

# истинный uplift бонуса: максимален у "середняков" (persuadables),
# близок к нулю у лояльных (sure things) и у безнадёжных (lost causes)
uplift_true = 0.18 * np.exp(-((churn_p - 0.55) ** 2) / (2 * 0.12 ** 2))

# рандомизированный бонус (как в честном A/B): половине дали
treat = rng.integers(0, 2, N)

# исход: остался активен через 30 дней (1 = остался)
stay_p = (1 - churn_p) + treat * uplift_true
stay_p = np.clip(stay_p, 0.01, 0.99)
stayed = (rng.random(N) < stay_p).astype(int)

df = pd.DataFrame({
    "deposits": deposits, "bet_freq": bet_freq, "recency": recency,
    "treat": treat, "stayed": stayed, "uplift_true": uplift_true,
})
df.to_parquet("cohort.parquet")
print(df.head())
print("Базовое удержание:", df.stayed.mean().round(3))
```

ARPU и маржу для unit-экономики возьмём как бизнес-константы: ARPU=50 за период (месяц), маржа $m=0.5$, retention за период $r=0.40$, дисконт $d=0.02$.

## Ход работы

### Шаг 1: Матрица стоимости и порог под прибыль

ЗАЧЕМ. Отрабатываем M24.3. Кампания удержания: модель предсказывает риск оттока, и кому-то мы делаем дорогое вмешательство (звонок саппорта + бонус). Вмешаться в того, кто и так остался бы (FP), — выкинуть стоимость бонуса. Не вмешаться в реально уходящего (FN) — потерять его маржу. Цена ошибок асимметрична, значит порог не 0.5.

Зададим матрицу: удержали уходящего ($V_{TP}$) +50, зря потратили бонус на лояльного ($C_{FP}$) −12, упустили уходящего ($C_{FN}$) −50, верно не трогали лояльного ($V_{TN}$) 0. Сначала обучаем модель риска оттока (целевая — `churn = 1 - stayed` на контрольной группе, чтобы не путать с эффектом бонуса).

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

df = pd.read_parquet("cohort.parquet")
ctrl = df[df.treat == 0].copy()
ctrl["churn"] = 1 - ctrl["stayed"]
X = ctrl[["deposits", "bet_freq", "recency"]]
y = ctrl["churn"]
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=42)

base = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)
base.fit(Xtr, ytr)
p_raw = base.predict_proba(Xval)[:, 1]

def profit_at(y_true, scores, t, v_tp=50, c_fp=-12, c_fn=-50, v_tn=0):
    pred = scores >= t
    tp = np.sum(pred & (y_true == 1)); fp = np.sum(pred & (y_true == 0))
    fn = np.sum(~pred & (y_true == 1)); tn = np.sum(~pred & (y_true == 0))
    return tp*v_tp + fp*c_fp + fn*c_fn + tn*v_tn

grid = np.linspace(0.01, 0.99, 99)
profits = [profit_at(yval.values, p_raw, t) for t in grid]
t_star = grid[int(np.argmax(profits))]
print(f"Порог под прибыль: {t_star:.2f}, прибыль: {max(profits):.0f}")
```

ЧТО ПОЛУЧИЛОСЬ. Оптимальный порог уехал от 0.5 в сторону, продиктованную отношением стоимостей. Сверь его с формулой из M24.3: $t^\ast = \frac{C_{FP}-V_{TN}}{(C_{FP}-V_{TN})+(C_{FN}-V_{TP})}$. Подставив наши числа, получаем ориентир, к которому сетка должна сойтись.

!!! question "Проверь себя"

    1. Почему в кампании удержания FP и FN стоят по-разному и куда это смещает порог?
    2. Что изменится в $t^\ast$, если бонус подорожает с 12 до 40 (FP станет −40)?

??? success "Ответы"

    1. FP — это выброшенная стоимость бонуса (−12), FN — потерянная маржа уходящего (−50). FN дороже, значит выгоднее трогать охотнее — порог ниже 0.5.
    2. Дорогой FP делает блокировку/вмешательство осторожнее: числитель $(C_{FP}-V_{TN})$ растёт по модулю, порог поднимается. Меньше игроков трогаем.

### Шаг 2: Калибровка перед деньгами

ЗАЧЕМ. M24.3 прямо требует: умножать скор на деньги можно только если это честная вероятность. LightGBM ранжирует хорошо, но смещён. Калибруем isotonic-методом и смотрим Brier до/после.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
cal.fit(Xtr, ytr)
p_cal = cal.predict_proba(Xval)[:, 1]

print("Brier raw:", round(brier_score_loss(yval, p_raw), 4))
print("Brier cal:", round(brier_score_loss(yval, p_cal), 4))

profits_cal = [profit_at(yval.values, p_cal, t) for t in grid]
t_star_cal = grid[int(np.argmax(profits_cal))]
print(f"Порог на калиброванной: {t_star_cal:.2f}")
```

ЧТО ПОЛУЧИЛОСЬ. Brier после калибровки ниже (вероятности честнее), и порог под прибыль считается уже от настоящих вероятностей. Если бы мы взяли порог по максимуму F1 — он лёг бы около 0.5 и слил бы прибыль на асимметрии. Сравни три порога: F1, 0.5 и денежный — разница в долларах и есть твой результат.

### Шаг 3: Unit-экономика — retention в LTV

ЗАЧЕМ. M24.4. Прежде чем обещать деньги, нужно уметь протянуть retention → LTV → NGR. Считаем LTV по простой формуле и по когортной с дисконтом, затем смотрим, что даёт рост $r$ на 5 пунктов.

```python
ARPU, m, r, d = 50, 0.5, 0.40, 0.02

ltv_simple = ARPU * m / (1 - r)
ltv_simple_up = ARPU * m / (1 - (r + 0.05))
print(f"LTV r=0.40: {ltv_simple:.1f}, r=0.45: {ltv_simple_up:.1f}, "
      f"прирост {100*(ltv_simple_up/ltv_simple - 1):.1f}%")

R = np.array([1.0, 0.45, 0.30, 0.22, 0.18])
periods = np.arange(len(R))
ltv_cohort = np.sum(ARPU * m * R / (1 + d) ** periods)
ltv_cohort_up = np.sum(ARPU * m * (R + 0.03) / (1 + d) ** periods)
print(f"Когортный LTV: {ltv_cohort:.1f} -> {ltv_cohort_up:.1f} "
      f"(+{100*(ltv_cohort_up/ltv_cohort - 1):.1f}%)")
```

ЧТО ПОЛУЧИЛОСЬ. Рост $r$ нелинеен: знаменатель $1-r$ при росте $r$ уменьшается быстрее, поэтому +5 пунктов retention дают ~+9% LTV. Это и есть мост от «retention вырос» к деньгам когорты. Запомни число — оно пойдёт в финальный вывод.

!!! question "Проверь себя"

    1. Откуда нелинейность прироста LTV по $r$?
    2. Зачем дисконт $(1+d)^t$ в когортной формуле?

??? success "Ответы"

    1. LTV $\propto \frac{1}{1-r}$ — гипербола; около высоких $r$ малый прирост $r$ сильно уменьшает знаменатель, поэтому процентный рост LTV больше прироста $r$ в пунктах.
    2. Деньги завтра дешевле денег сегодня; без дисконта будущие периоды переоценены и LTV завышается.

### Шаг 4: Инкрементальный эффект бонуса (ATE + бутстрап)

ЗАЧЕМ. M24.5–M24.6. У нас честный рандомизированный бонус (`treat`). Считаем ATE как разницу удержания между treatment и control и строим бутстрап-ДИ — это «измерили», а не «ожидаем».

```python
def ate_bootstrap(y_t, y_c, n_boot=10000, seed=42):
    g = np.random.default_rng(seed)
    ate = y_t.mean() - y_c.mean()
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = (g.choice(y_t, len(y_t)).mean()
                   - g.choice(y_c, len(y_c)).mean())
    return ate, np.percentile(boot, [2.5, 97.5])

yt = df[df.treat == 1].stayed.values
yc = df[df.treat == 0].stayed.values
ate, ci = ate_bootstrap(yt, yc)
print(f"ATE удержание: {ate:+.3f}, 95% ДИ [{ci[0]:+.3f}, {ci[1]:+.3f}]")
```

ЧТО ПОЛУЧИЛОСЬ. ATE — это средний инкрементальный эффект бонуса по всем игрокам, размытый sure things и lost causes, на которых эффект около нуля. Джун назвал бы наблюдаемое «среди затронутых остались X%»; корректно — только эта разница с control. ДИ не должен включать ноль, иначе эффект недоказан.

### Шаг 5: Uplift-таргетинг vs churn-ранжирование

ЗАЧЕМ. M24.6, главный денежный шаг. Бюджет ограничен: бонус 5 долларов, бюджет 10 000 → 2000 игроков. Сравним два способа выбрать 2000: по риску оттока (churn) и по uplift (two-model). Покажем, что при равном бюджете uplift удерживает больше.

```python
from lightgbm import LGBMClassifier

feat = ["deposits", "bet_freq", "recency"]
mt = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)
mc = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)
mt.fit(df[df.treat == 1][feat], df[df.treat == 1].stayed)
mc.fit(df[df.treat == 0][feat], df[df.treat == 0].stayed)

tau = mt.predict_proba(df[feat])[:, 1] - mc.predict_proba(df[feat])[:, 1]
churn_score = 1 - mc.predict_proba(df[feat])[:, 1]

budget = 2000
top_uplift = np.argsort(tau)[::-1][:budget]
top_churn = np.argsort(churn_score)[::-1][:budget]

# истинный заложенный эффект на отобранных = реально удержанные сверх базы
gain_uplift = df.uplift_true.values[top_uplift].sum()
gain_churn = df.uplift_true.values[top_churn].sum()
print(f"Удержано доп. игроков uplift: {gain_uplift:.0f}, churn: {gain_churn:.0f}")
print(f"Выигрыш uplift над churn: {gain_uplift - gain_churn:.0f} игроков")
```

ЧТО ПОЛУЧИЛОСЬ. Uplift отбирает persuadables и при том же бюджете удерживает заметно больше игроков, чем churn-ранжирование, которое тратит бонусы на lost causes (высокий риск, нулевой uplift) и sure things. Переводим выигрыш в деньги через LTV из шага 3.

!!! question "Проверь себя"

    1. Почему churn-топ ловит lost causes, а uplift — нет?
    2. Почему качество uplift нельзя мерить обычным AUC?

??? success "Ответы"

    1. Churn ранжирует по вероятности уйти; у lost causes она максимальна, но эффект бонуса ($\tau$) около нуля. Uplift ранжирует прямо по $\tau$ и обходит их.
    2. AUC меряет ранжирование наблюдаемого исхода, а $\tau$ — ненаблюдаемая разница потенциальных исходов. Качество uplift меряют Qini-кривой и uplift@k.

### Шаг 6: Бизнес-вывод на языке решений

ЗАЧЕМ. M24.9. Собираем числа в одно сообщение: решение, деньги с диапазоном, риск, guardrails. Без технических метрик в основном тексте.

```python
ltv = ARPU * m / (1 - r)
extra_players = gain_uplift                 # доп. удержанные за кампанию
money = extra_players * ltv
money_lo = extra_players * ltv * (ci[0] / ate)  # масштаб ДИ переносим на деньги
money_hi = extra_players * ltv * (ci[1] / ate)

print("=== ВЫВОД ДЛЯ БИЗНЕСА ===")
print(f"Решение: запускать uplift-таргетинг бонуса на 100% после A/B с holdout 3%.")
print(f"Эффект: ~{extra_players:.0f} удержанных игроков сверх случайной раздачи,")
print(f"        ~${money:,.0f} доп. NGR за кампанию (диапазон ${money_lo:,.0f}..${money_hi:,.0f}).")
print(f"Альтернатива (churn-таргетинг): удержал бы только {gain_churn:.0f} -> деньги на ветер.")
print(f"Guardrails: отток лояльных (sleeping dogs), доля жалоб, latency скоринга.")
print(f"Риск: эффект подтверждён на симуляции; до прода держим shadow mode + holdout.")
```

ЧТО ПОЛУЧИЛОСЬ. Готовый артефакт: одно решение, эффект в деньгах с диапазоном, явная альтернатива (сколько потеряли бы на наивном churn-таргетинге), guardrails и точка отката. Это и есть перевод модели в P&L, ради которого весь модуль.

## Критерий готовности

- [ ] Сгенерирован датасет с фиксированным seed, базовое удержание выводится.
- [ ] Найден порог под прибыль и сверен с формулой $t^\ast$ из M24.3.
- [ ] Brier после калибровки ниже, чем до; порог считается на калиброванных вероятностях.
- [ ] Посчитан LTV простой и когортной формулой, показан нелинейный прирост от роста $r$.
- [ ] ATE с бутстрап-ДИ посчитан, ДИ не включает ноль.
- [ ] Uplift-таргетинг удерживает больше игроков, чем churn-таргетинг при равном бюджете.
- [ ] Напечатан бизнес-вывод: решение + деньги с диапазоном + альтернатива + guardrails, без технических метрик в основном сообщении.

## Развитие

- Замени two-model на `causalml.UpliftRandomForestClassifier` и оцени отбор по Qini-кривой вместо заложенного `uplift_true` — приблизишься к честной прод-оценке без доступа к «правде».
- Добавь sleeping dogs: задай части игроков отрицательный `uplift_true` и проверь, что uplift-таргетинг их избегает, а churn — нет. Это самый сильный аргумент против наивной churn-рассылки.
- Спроектируй sample size: при каком N бутстрап-ДИ ATE перестаёт накрывать ноль для заложенного эффекта в 2 пункта (свяжи с MDE из M23).
- Прогони сценарии стоимостей: построй кривую «оптимальный порог vs $C_{FP}/C_{FN}$» и покажи менеджеру на салфетке, как дорожание бонуса сдвигает, кого мы трогаем.
