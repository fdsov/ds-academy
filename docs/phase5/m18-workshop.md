# Воркшоп M18 · MLP на PyTorch против бустинга

<span class="lecture-meta">Воркшоп к модулю M18 · ориентир 5-7 ч</span>

## Что отрабатываем

В модуле M18 ты разобрал механику нейросети: forward pass как композицию аффинных преобразований и нелинейностей, backpropagation через цепное правило, оптимизаторы (Adam/AdamW), инициализацию, регуляризацию (dropout, weight decay, early stopping, batchnorm). Теория объясняла это на примере скоринга вероятности депозита игрока — этим же и займёмся руками.

Цель воркшопа — собрать полносвязную сеть на PyTorch **с нуля** и довести её до рабочего состояния: свой `nn.Module`, явный обучающий цикл с `AdamW`, `dropout` и `early stopping`, кривые train/val лосса. А затем честно сравнить её с градиентным бустингом (LightGBM) на тех же табличных данных. Модуль прямо предупреждал: на табличке нейросеть часто проигрывает бустингу — это нормальный и важный результат, который ты должен увидеть своими глазами, а не принять на веру.

Артефакт на выходе:

- класс `DepositNet(nn.Module)` с BatchNorm + ReLU + Dropout;
- функция `train()` с валидацией каждую эпоху и early stopping по val-лоссу;
- график двух кривых лосса с отмеченной точкой переобучения;
- таблица сравнения ROC-AUC: логрег / MLP / LightGBM.

## Данные

Генерируем синтетический табличный датасет «скоринг депозита» с фиксированным seed — запускается у кого угодно. Признаки в разном масштабе (ставки в единицах, суммы в тысячах, дни в десятках) специально, чтобы прочувствовать ловушку «не масштабировал признаки» из модуля.

```bash
uv init m18-workshop && cd m18-workshop
uv add torch scikit-learn lightgbm numpy matplotlib
```

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

X, y = make_classification(
    n_samples=20000, n_features=20, n_informative=10, n_redundant=4,
    n_clusters_per_class=3, weights=[0.85, 0.15], flip_y=0.02,
    class_sep=0.9, random_state=SEED,
)

# Имитируем реальные масштабы признаков гемблинг-данных
X[:, 0] = np.abs(X[:, 0]) * 1500      # сумма депозитов, тысячи
X[:, 1] = np.abs(X[:, 1]) * 40        # дни с регистрации
X[:, 2] = np.abs(X[:, 2]) * 3         # число ставок

# Split БЕЗ утечки: сначала режем, потом fit скейлера только на трейне
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

scaler = StandardScaler().fit(X_tr)        # fit ТОЛЬКО на трейне
X_tr_s = scaler.transform(X_tr)
X_val_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_te)

print(X_tr.shape, X_val.shape, X_te.shape, "доля позитива:", y.mean().round(3))
```

Что получилось: 20000 объектов, 20 признаков, дисбаланс ~15% позитива (как у редкого события «депозит»), три части train/val/test без утечки. Скейлер обучен только на трейне — ровно по правилу из ловушек модуля.

!!! question "Проверь себя"

    1. Почему `StandardScaler` мы fit-им только на `X_tr`, а не на всём `X`?
    2. Зачем для нейросети вообще масштабировать признаки, если деревьям бустинга это не нужно?

??? success "Ответы"

    1. Fit на всём датасете подсматривает среднее и std валидации и теста — это утечка данных, метрики будут оптимистично завышены.
    2. Сеть чувствительна к масштабу входа: признаки в тысячах и в единицах дают несбалансированные градиенты и ломают сходимость. Деревья делят по порогам и к монотонному масштабу инвариантны.

## Ход работы

### Шаг 1: Тензоры и DataLoader

Зачем: модуль показывал, что mini-batch — стандарт (стабильность плюс векторизация). `DataLoader` нарезает данные на батчи и перемешивает их каждую эпоху.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(SEED)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def to_ds(Xs, yv):
    return TensorDataset(
        torch.tensor(Xs, dtype=torch.float32),
        torch.tensor(yv, dtype=torch.float32).unsqueeze(1),
    )

train_dl = DataLoader(to_ds(X_tr_s, y_tr), batch_size=256, shuffle=True)
val_dl = DataLoader(to_ds(X_val_s, y_val), batch_size=512, shuffle=False)
```

Что получилось: батчи по 256 на трейне с перемешиванием, по 512 на валидации без него. `unsqueeze(1)` приводит таргет к форме `(N, 1)` — её ждёт `BCEWithLogitsLoss`.

### Шаг 2: Своя сеть — nn.Module и forward

Зачем: отрабатываем определение MLP из M18.5. Стопка `Linear → BatchNorm → ReLU → Dropout`. Последний слой выдаёт **logits без сигмоиды** — сигмоиду возьмёт на себя численно стабильный лосс.

```python
import torch.nn as nn

class DepositNet(nn.Module):
    def __init__(self, n_features: int, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 1),           # logits
        )

    def forward(self, x):
        return self.net(x)
```

Что получилось: сеть со скрытыми слоями 128 и 64. Проверь размерности по цепочке из модуля: вход 20 → `W1` 128×20 → `W2` 64×128 → `W3` 1×64, выход — скаляр-логит на объект.

### Шаг 3: Лосс, оптимизатор и учёт дисбаланса

Зачем: модуль требует `BCEWithLogitsLoss` (стабильно объединяет sigmoid + BCE) и `AdamW` (корректный weight decay). При дисбалансе классов — `pos_weight`, иначе сеть выучит «всегда мажорный класс».

```python
pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32)

model = DepositNet(n_features=X_tr_s.shape[1]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

Что получилось: `pos_weight ≈ 5.7` — позитивный класс штрафуется сильнее пропорционально его редкости. lr=1e-3 — разумный дефолт Adam из модуля.

!!! question "Проверь себя"

    1. Почему последний `Linear` выдаёт logit, а не вероятность через сигмоиду?
    2. Что произойдёт с предсказаниями, если убрать `pos_weight` на этом дисбалансе?

??? success "Ответы"

    1. `BCEWithLogitsLoss` сам применяет сигмоиду внутри в численно устойчивой форме (log-sum-exp), избегая `log(0)` и NaN. Ручная сигмоида плюс `log` — классический источник NaN-лосса.
    2. Сеть смещается к предсказанию мажорного класса (нет депозита): общий лосс минимизируется, но recall по позитивному классу падает почти в ноль.

### Шаг 4: Обучающий цикл с валидацией и early stopping

Зачем: это ядро воркшопа. Собираем полный цикл из M18.11: `zero_grad → forward → backward → step`. Каждую эпоху считаем val-лосс, сохраняем лучший чекпойнт и останавливаемся, если валидация не улучшается `patience` эпох — early stopping из M18.10. Не забыть `model.train()` / `model.eval()` — иначе dropout и batchnorm в неправильном режиме.

```python
import copy

def run_epoch(dl, train: bool):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / n

best_val, best_state, patience, wait = float("inf"), None, 8, 0
hist = {"train": [], "val": []}

for epoch in range(100):
    tr_loss = run_epoch(train_dl, train=True)
    val_loss = run_epoch(val_dl, train=False)
    hist["train"].append(tr_loss)
    hist["val"].append(val_loss)

    if val_loss < best_val - 1e-4:
        best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stop на эпохе {epoch}, лучший val={best_val:.4f}")
            break

model.load_state_dict(best_state)   # откатываемся к лучшему чекпойнту
```

Что получилось: обучение само останавливается, когда val-лосс перестал падать, и в `model` загружены веса лучшей эпохи, а не последней. `torch.set_grad_enabled(False)` на валидации экономит память — autograd не строит граф (привязка к M18.7 про расход VRAM).

### Шаг 5: Кривые обучения и точка переобучения

Зачем: модуль учил диагностировать переобучение по расхождению кривых. Рисуем train и val лосс и отмечаем эпоху, после которой val развернулся вверх.

```python
import matplotlib.pyplot as plt

best_ep = int(np.argmin(hist["val"]))
plt.plot(hist["train"], label="train")
plt.plot(hist["val"], label="val")
plt.axvline(best_ep, ls="--", c="gray", label=f"best val @ {best_ep}")
plt.xlabel("эпоха"); plt.ylabel("BCE loss"); plt.legend()
plt.title("Кривые обучения MLP")
plt.savefig("curves.png", dpi=120, bbox_inches="tight")
```

Что получилось: train-лосс ползёт вниз дальше, чем val; точка `best_ep` — момент, который и поймал early stopping. Если кривые расходятся рано — усиль dropout или weight decay; если обе высоко и плоско — мала ёмкость или lr.

!!! question "Проверь себя"

    1. Почему мы откатываемся к `best_state`, а не берём веса последней эпохи?
    2. На графике train-лосс иногда оказывается ВЫШЕ val-лосса в одной точке. Это баг?

??? success "Ответы"

    1. Последние эпохи уже могут переобучаться: val-лосс там выше минимума. Лучший чекпойнт по валидации — это и есть результат early stopping.
    2. Нет. На train активен dropout (часть нейронов выключена), поэтому train-лосс в моменте может казаться хуже. На val dropout выключен через `model.eval()`. Это прямая ловушка из модуля — «сравнение в неравных условиях».

### Шаг 6: Сравнение с бустингом и baseline

Зачем: финальный и самый честный шаг. Оцениваем MLP на отложенном test через ROC-AUC и сравниваем с логрегрессией и LightGBM. Бустингу масштабирование не нужно — кормим сырые признаки.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

model.eval()
with torch.no_grad():
    logit = model(torch.tensor(X_te_s, dtype=torch.float32).to(device))
    mlp_prob = torch.sigmoid(logit).cpu().numpy().ravel()

logreg = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_s, y_tr)
lr_prob = logreg.predict_proba(X_te_s)[:, 1]

gbm = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=SEED,
).fit(X_tr, y_tr)          # сырые признаки, без скейлера
gbm_prob = gbm.predict_proba(X_te)[:, 1]

for name, p in [("LogReg", lr_prob), ("MLP", mlp_prob), ("LightGBM", gbm_prob)]:
    print(f"{name:9s} ROC-AUC={roc_auc_score(y_te, p):.4f}  PR-AUC={average_precision_score(y_te, p):.4f}")
```

Что получилось: три числа ROC-AUC и PR-AUC на одном test. Типично LightGBM идёт вровень или впереди MLP на такой синтетике, а MLP заметно бьёт логрег. Сделай вывод словами: победила ли сеть и почему — это и есть рефлексия из mini-проекта модуля.

## Критерий готовности

- [ ] Данные разрезаны на train/val/test, `StandardScaler` обучен только на трейне (нет утечки)
- [ ] `DepositNet(nn.Module)` написан сам, последний слой выдаёт logits без сигмоиды
- [ ] Лосс — `BCEWithLogitsLoss` с `pos_weight`; оптимизатор — `AdamW` с weight decay
- [ ] Цикл содержит `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` и переключение `train()/eval()`
- [ ] Early stopping по val-лоссу работает и откатывает модель к лучшему чекпойнту
- [ ] Построен график двух кривых лосса с отмеченной точкой переобучения
- [ ] ROC-AUC и PR-AUC посчитаны на test для LogReg, MLP и LightGBM, сделан честный вывод

## Развитие

- **Абляция нелинейности.** Замени все `ReLU` на `nn.Identity()` и убедись, что качество падает до уровня логрега — наглядное подтверждение M18.4: без активаций глубина схлопывается в один линейный слой.
- **Сетка активаций и оптимизаторов.** Прогони ReLU / Tanh / GELU и SGD / SGD+momentum / AdamW, меняя один фактор за раз при фиксированном seed. Сравни скорость сходимости по кривым.
- **Dropout-свип.** Перебери `p ∈ {0, 0.2, 0.5}` и построй gap между train и val лоссом — увидишь, как растёт регуляризация. Добавь `torch.nn.utils.clip_grad_norm_` и проверь, спасает ли он от NaN при завышенном lr.
- **Эмбеддинги категорий.** Добавь категориальный признак (страна, платёжка) через `nn.Embedding` вместо one-hot — это мостик к тому, где нейросети реально обгоняют бустинг, и подготовка к модулям про эмбеддинги игроков.
