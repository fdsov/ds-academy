# Воркшоп M26 · Кривые ошибки, bias-variance и double descent своими руками

<span class="lecture-meta">Воркшоп к модулю M26 · ориентир 4-6 ч</span>

## Что отрабатываем

Модуль M26 дал язык: истинный риск $R(h)$ против эмпирического $\hat{R}_S(h)$, разложение ошибки на $\text{Bias}^2 + \text{Variance} + \sigma^2$, ёмкость класса гипотез, регуляризацию как контроль ёмкости (раздел M26.7) и феномен double descent (раздел M26.9). В воркшопе ты не доказываешь теоремы — ты их видишь на графиках, построенных руками.

Конкретно отрабатываем:

- кривую обобщения train/val против сложности модели и её U-образный минимум (M26.5);
- прямое измерение $\text{Bias}^2$ и $\text{Variance}$ через bootstrap и упор суммы в неустранимый шум $\sigma^2$ (M26.5);
- регуляризацию как одну ручку $\lambda$, двигающую bias-variance компромисс (M26.7);
- double descent на перепараметризованной сети, где пик стоит у точки интерполяции $p \approx n$ (M26.9).

Артефакт на выходе: набор из четырёх графиков (`m26_curves.png`) плюс таблица bias/variance с письменным разбором каждого в терминах модуля.

## Данные

Всё синтетическое, с фиксированным seed — запускается у любого. Истинная зависимость $y = \sin(2\pi x) + \epsilon$, шум $\sigma = 0.3$, значит теоретическая нижняя граница ошибки $\sigma^2 = 0.09$. Это та же постановка, что в коде M26.11, чтобы числа бились с теорией.

Окружение поднимаем через uv (стек 2026).

```bash
uv init m26-workshop && cd m26-workshop
uv add "numpy>=2.1" "scikit-learn>=1.6" "matplotlib>=3.9" "torch>=2.4"
```

```python
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

def true_f(x):
    return np.sin(2 * np.pi * x)

SIGMA = 0.3

def make_dataset(n, seed):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, size=(n, 1))
    y = true_f(X.ravel()) + r.normal(0, SIGMA, size=n)
    return X, y

X_train, y_train = make_dataset(200, seed=1)
X_test, y_test = make_dataset(2000, seed=999)  # большой тест ≈ истинный риск
```

Большая тестовая выборка (2000 точек) — это наш практический заместитель истинного риска $R(h)$: матожидание по $\mathcal{D}$ мы посчитать не можем (M26.3), но среднее по 2000 i.i.d. точкам близко к нему.

## Ход работы

### Шаг 1: Кривая обобщения против сложности

Зачем. Отрабатываем M26.5: при росте ёмкости train-ошибка монотонно падает (модель всё лучше запоминает, вплоть до подгонки под шум), а test-ошибка идёт по U — сначала падает смещение, потом растёт разброс. Минимум test-кривой — и есть bias-variance компромисс.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

depths = list(range(1, 16))
train_mse, test_mse = [], []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
    train_mse.append(mean_squared_error(y_train, m.predict(X_train)))
    test_mse.append(mean_squared_error(y_test, m.predict(X_test)))

best = depths[int(np.argmin(test_mse))]
print(f"оптимальная глубина по тесту: {best}")
print(f"min test_mse={min(test_mse):.3f}  vs  sigma^2={SIGMA**2:.3f}")
```

Что получилось. `train_mse` стремится к нулю при больших глубинах (дерево вмещает каждую точку), `test_mse` имеет минимум при умеренной глубине и упирается снизу в $\sigma^2 = 0.09$. Запиши значение `best` — оно понадобится для сравнения.

!!! question "Проверь себя"

    1. Почему train-MSE падает монотонно, а test-MSE — нет?
    2. Почему минимальный test-MSE не может опуститься заметно ниже 0.09?

??? success "Ответы"

    1. Рост ёмкости всегда позволяет лучше подогнать обучающую выборку (вплоть до запоминания шума), поэтому $\hat{R}_S$ монотонно падает. Истинный риск — сумма падающего $\text{Bias}^2$ и растущего $\text{Variance}$, поэтому имеет минимум (U-кривая из M26.5).
    2. $\sigma^2 = 0.09$ — неустранимый шум: он заложен в самих данных $y = f(x) + \epsilon$ и не зависит от модели. Это нижняя граница ошибки.

### Шаг 2: Прямое измерение Bias и Variance через bootstrap

Зачем. M26.5 даёт точное разложение $\mathbb{E}_S[(y_0-\hat{f}_S(x_0))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$. Слово $\mathbb{E}_S$ — матожидание по случайным выборкам. Мы его аппроксимируем bootstrap: много раз переобучаем модель на ресэмплах и смотрим, как пляшут предсказания.

```python
from sklearn.base import clone

x_query = np.linspace(0, 1, 100).reshape(-1, 1)
f_true = true_f(x_query.ravel())

def bias_variance(model, n_boot=300):
    preds = np.empty((n_boot, len(x_query)))
    for b in range(n_boot):
        idx = rng.integers(0, len(X_train), len(X_train))
        m = clone(model).fit(X_train[idx], y_train[idx])
        preds[b] = m.predict(x_query)
    mean_p = preds.mean(axis=0)
    bias2 = np.mean((f_true - mean_p) ** 2)
    variance = preds.var(axis=0).mean()
    return bias2, variance

print(f"{'depth':>5} {'bias^2':>8} {'variance':>9} {'sum+sigma^2':>12}")
for depth in [1, 2, 3, 5, 8, 14]:
    b2, v = bias_variance(DecisionTreeRegressor(max_depth=depth, random_state=0))
    print(f"{depth:5d} {b2:8.3f} {v:9.3f} {b2 + v + SIGMA**2:12.3f}")
```

Что получилось. Таблица показывает: малая глубина — высокий $\text{Bias}^2$, низкий $\text{Variance}$ (модель стабильна, но грубая); большая глубина — наоборот. Колонка `sum+sigma^2` должна примерно совпадать с test-MSE из шага 1 для тех же глубин — это и есть проверка разложения на практике. Минимум суммы достигается там же, где минимум U-кривой.

!!! tip "Где расхождение допустимо"

    `sum+sigma^2` и test-MSE из шага 1 совпадут не идеально: bootstrap — оценка $\mathbb{E}_S$ по конечному числу ресэмплов, плюс ресэмпл выборки не то же самое, что свежие выборки из $\mathcal{D}$. Расхождение в пределах ~10-20% — норма, тренд (рост variance с глубиной) важнее точных чисел.

### Шаг 3: Регуляризация как ручка bias-variance

Зачем. M26.7: регуляризация — не запрет сложности, а штраф за неё. Увеличивая $\lambda$ (в scikit-learn это $C = 1/\lambda$ для линейных моделей, или `alpha` для ridge), сужаем эффективный класс гипотез: разброс падает, смещение растёт. Покажем это на полиномиальной ridge-регрессии той же задачи.

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def ridge_model(alpha):
    return make_pipeline(
        PolynomialFeatures(degree=12),
        StandardScaler(),
        Ridge(alpha=alpha),
    )

alphas = np.logspace(-5, 2, 20)
tr, te = [], []
for a in alphas:
    m = ridge_model(a).fit(X_train, y_train)
    tr.append(mean_squared_error(y_train, m.predict(X_train)))
    te.append(mean_squared_error(y_test, m.predict(X_test)))

best_a = alphas[int(np.argmin(te))]
print(f"оптимальная alpha (=lambda): {best_a:.4g}")
```

Что получилось. Полином степени 12 без регуляризации (`alpha→0`) переобучается: train-MSE крошечный, test-MSE взлетает. С ростом `alpha` test-MSE падает до минимума, потом снова растёт (пересиленное смещение — модель стала почти прямой). Один скаляр $\lambda$ двигает весь компромисс — это SRM из M26.7 в действии.

!!! question "Проверь себя"

    1. Почему при $\alpha \to 0$ полином степени 12 даёт большой разрыв $R(h) - \hat{R}_S(h)$?
    2. Что происходит со смещением и разбросом при очень большом $\alpha$?

??? success "Ответы"

    1. Без штрафа класс гипотез богатый (степень 12), эффективная ёмкость высока, модель подгоняется под шум — большой $\text{Variance}$ и большой generalization gap (M26.4, M26.7).
    2. Большой $\alpha$ сжимает веса почти к нулю (норма $B$ мала), эффективный класс сужается до почти константы: $\text{Variance}\to 0$, но $\text{Bias}^2$ растёт — недообучение.

### Шаг 4: Double descent на перепараметризованной сети

Зачем. Кульминация модуля (M26.9). Классика обещает катастрофу при $p > n$. Покажем обратное: тестовая ошибка проходит пик у точки интерполяции $p \approx n$, а затем при $p \gg n$ снова падает. Берём маленький датасет ($n = 40$), плавно растим ширину одного скрытого слоя.

```python
import torch
import torch.nn as nn

torch.manual_seed(SEED)
Xtr = torch.tensor(make_dataset(40, seed=7)[0], dtype=torch.float32)
ytr = torch.tensor(make_dataset(40, seed=7)[1], dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_test, dtype=torch.float32)
yte = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
n = Xtr.shape[0]

def train_width(h, epochs=4000):
    net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=0.0)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return lossf(net(Xtr), ytr).item(), lossf(net(Xte), yte).item()

widths = [1, 2, 5, 10, 20, 40, 60, 100, 200, 400, 800]
print(f"{'width':>6} {'~params':>8} {'train':>8} {'test':>8}")
for h in widths:
    p = 3 * h + 1  # 2*h весов + h + 1 в выходном слое ≈ число параметров
    tr_l, te_l = train_width(h)
    flag = "  <- p≈n" if abs(p - n) < n * 0.5 else ""
    print(f"{h:6d} {p:8d} {tr_l:8.3f} {te_l:8.3f}{flag}")
```

Что получилось. У узких сетей train и test высокие (недообучение). У ширины, где число параметров $p \approx n = 40$ (примерно `width=10..14`), test-ошибка даёт локальный пик — модель вынуждена интерполировать ровно с нулём свободы, разброс взрывается (M26.9). При дальнейшем росте ширины ($p \gg n$) test-ошибка снова падает: среди множества интерполяторов Adam/SGD неявно выбирает решение малой нормы — самое гладкое. Это и есть второй спуск.

!!! tip "Если пик не виден"

    Double descent — тонкий эффект, чувствительный к seed, числу эпох и lr. Если пик смазан: увеличь `epochs` до 8000 (нужна почти полная интерполяция трейна у $p \approx n$), усредни test-ошибку по 3-5 seed на каждую ширину, проверь что у $p \gg n$ train-MSE действительно близок к нулю. Тренд "узко-высоко → пик у $p\approx n$ → снова низко" должен проступить.

### Шаг 5: Собрать артефакт

Зачем. Свести четыре наблюдения в один график — это и есть deliverable, который ты разбираешь словами модуля.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(12, 9))

ax[0, 0].plot(depths, train_mse, "o-", label="train")
ax[0, 0].plot(depths, test_mse, "s-", label="test")
ax[0, 0].axhline(SIGMA**2, ls="--", c="gray", label="sigma^2=0.09")
ax[0, 0].set(title="Шаг 1: U-кривая дерева", xlabel="max_depth", ylabel="MSE")
ax[0, 0].legend()

ax[0, 1].semilogx(alphas, tr, "o-", label="train")
ax[0, 1].semilogx(alphas, te, "s-", label="test")
ax[0, 1].axvline(best_a, ls="--", c="green", label=f"best lambda")
ax[0, 1].set(title="Шаг 3: ridge регуляризация", xlabel="alpha (lambda)", ylabel="MSE")
ax[0, 1].legend()

ax[1, 0].plot(x_query, f_true, "k--", label="истина f(x)")
ax[1, 0].scatter(X_train, y_train, s=10, alpha=0.4, label="train")
for depth, c in [(2, "C0"), (5, "C1"), (14, "C2")]:
    m = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_train, y_train)
    ax[1, 0].plot(x_query, m.predict(x_query), c=c, label=f"depth={depth}")
ax[1, 0].set(title="Шаг 2: смещение vs разброс", xlabel="x", ylabel="y")
ax[1, 0].legend(fontsize=8)

params = [3 * h + 1 for h in widths]
dd_tr, dd_te = zip(*[train_width(h) for h in widths])
ax[1, 1].plot(params, dd_tr, "o-", label="train")
ax[1, 1].plot(params, dd_te, "s-", label="test")
ax[1, 1].axvline(n, ls="--", c="red", label="p≈n (интерполяция)")
ax[1, 1].set(title="Шаг 4: double descent", xlabel="~параметры p", ylabel="MSE", xscale="log")
ax[1, 1].legend()

fig.tight_layout()
fig.savefig("m26_curves.png", dpi=120)
print("saved m26_curves.png")
```

Что получилось. `m26_curves.png` с четырьмя панелями: U-кривая дерева, кривая регуляризации, наглядное "грубое vs дёрганое" предсказание (depth=2 vs depth=14), и double descent. Это твой артефакт.

## Критерий готовности

- [ ] Окружение поднято через uv, все скрипты запускаются без правок
- [ ] Шаг 1: построена U-кривая, найдена оптимальная глубина, min test-MSE сравнён с $\sigma^2 = 0.09$
- [ ] Шаг 2: таблица $\text{Bias}^2$/$\text{Variance}$ для 5+ глубин, проверено что сумма $+\sigma^2$ примерно равна test-MSE и что variance растёт с глубиной
- [ ] Шаг 3: показан рост test-MSE при $\alpha\to 0$ (переобучение) и при больших $\alpha$ (недообучение), найдена оптимальная $\lambda$
- [ ] Шаг 4: видны три фазы — недообучение, пик у $p \approx n$, второй спуск при $p \gg n$
- [ ] Сохранён `m26_curves.png` с четырьмя панелями
- [ ] Письменный разбор: каждый график объяснён терминами модуля ($R$, $\hat{R}_S$, $\text{Bias}^2$, $\text{Variance}$, ёмкость, $\lambda$, интерполяция)

## Развитие

1. Замени синтетику на публичный churn-датасет (например Telco Customer Churn) и повтори шаги 1 и 3 для классификации: `validation_curve` по глубине дерева и по $C$ логистической регрессии. Объясни, ломается ли i.i.d. при случайном сплите, если на игрока несколько строк — и переключись на `GroupKFold` (сеньорская ловушка из M26.11).
2. Добавь в шаг 4 явный `weight_decay` (L2) в Adam и посмотри, как он сглаживает пик double descent — связь неявной и явной регуляризации из M26.9.
3. Измерь не число параметров, а норму весов найденного решения в шаге 4 и построй test-ошибку против нормы. Проверь тезис M26.9: истинная сложность — норма решения, а не подсчёт весов.
4. Воспроизведи проклятие размерности (M26.10): для $d = 2, 10, 50, 200$ сгенерируй равномерные точки и измерь отношение $(\max-\min)/\min$ расстояний до запроса. Покажи, что оно стремится к нулю, и свяжи с тем, почему kNN деградирует.
