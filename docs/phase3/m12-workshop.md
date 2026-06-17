# Воркшоп M12 · Когортный retention и честный storytelling

<span class="lecture-meta">Воркшоп к модулю M12 · ориентир 3-5 ч</span>

## Что отрабатываем

Этот воркшоп прогоняет руками самые денежные графики продуктовой аналитики из модуля M12 на синтетических данных гемблинг-проекта. Ты отрабатываешь:

- Распределения денег: гистограмма в линейной и лог-шкале плюс ECDF (M12.3, M12.5) — почему линейная шкала прячет структуру, а ECDF не врёт выбором бина.
- Когортную retention-heatmap (M12.5, раздел «Корреляции») — чтение «сверху вниз» (качество когорт) и «слева направо» (отвал со временем), палитра viridis как colorblind-safe sequential (M12.7).
- Retention-кривые (M12.5, «Динамика») с подсветкой худшего канала и приглушением остальных — против графика-спагетти (M12.12, «Ловушки»).
- Boxplot и ECDF для сравнения групп (M12.5, M12.6).
- Один финальный storytelling-график с заголовком-выводом, честными осями и направлением внимания (M12.11).

Артефакт на выходе: папка `m12_out/` с набором EDA-графиков (`dist.png`, `cohorts.png`, `boxplot.png`) и один финальный презентационный график `story_final.png` с заголовком-утверждением и аннотацией. Каждый файл проходит чек «честная ось / colorblind-safe / заголовок делает работу за зрителя».

## Данные

Никаких внешних загрузок. Генерируем синтетику с фиксированным seed — депозиты как лог-нормальное распределение (типично для денег), два канала привлечения A и B с разным качеством когорт. Запускается у любого.

```bash
uv init m12-workshop && cd m12-workshop
uv add numpy pandas matplotlib seaborn
```

```python
# data.py
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_A, N_B = 6000, 6000

def make_channel(n, channel, base_ltv, retain_decay, start_quality_drift):
    reg_week = rng.integers(0, 8, n)                      # неделя регистрации 0..7
    deposit = rng.lognormal(mean=3.0, sigma=1.1, size=n)  # деньги: лог-нормаль
    # качество свежих когорт у A падает (drift>0), у B стабильно
    quality = 1.0 - start_quality_drift * reg_week
    ltv = base_ltv * deposit ** 0.6 * quality * rng.lognormal(0, 0.4, n)
    return pd.DataFrame({
        "channel": channel,
        "reg_week": reg_week,
        "deposit": deposit,
        "ltv_d30": np.clip(ltv, 0, None),
        "retain_decay": retain_decay,
        "quality": quality,
    })

players = pd.concat([
    make_channel(N_A, "A", base_ltv=4.0, retain_decay=0.85, start_quality_drift=0.07),
    make_channel(N_B, "B", base_ltv=5.5, retain_decay=0.93, start_quality_drift=0.00),
], ignore_index=True)

# Лог возвратов: для каждого игрока — вернулся ли он на день d (0..30)
days = np.arange(31)
def retention_events(df):
    rows = []
    for _, r in df.iterrows():
        p_day = r["retain_decay"] ** days * r["quality"]   # шанс вернуться на день d
        returned = rng.random(31) < p_day
        for d in days[returned]:
            rows.append((r["channel"], r["reg_week"], int(d)))
    return rows

# events большой — берём подвыборку игроков для лога возвратов (heatmap всё равно по долям)
sample = players.sample(4000, random_state=1)
events = pd.DataFrame(
    [e for _, r in sample.iterrows()
       for e in [(r["channel"], r["reg_week"], int(d))
                 for d in days[rng.random(31) < (r["retain_decay"] ** days * r["quality"])]]],
    columns=["channel", "reg_week", "day"],
)

if __name__ == "__main__":
    print(players.head())
    print("событий возврата:", len(events))
```

CSV не нужен — модули импортируем напрямую: `from data import players, events, days, sample`.

## Ход работы

### Шаг 1: распределение денег — линейная шкала врёт

Зачем. M12.3: деньги охватывают несколько порядков (закон Вебера-Фехнера), поэтому в линейной шкале депозиты — «столбик у нуля плюс хвост». Отрабатываем согласование шкалы с природой денег и ECDF как самый честный график распределения.

```python
# step1_dist.py
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from data import players

Path("m12_out").mkdir(exist_ok=True)
dep = players["deposit"].to_numpy()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(dep, bins=50, color="#4C72B0"); ax[0].set_title("Линейная шкала: всё у нуля")
ax[1].hist(dep, bins=50, color="#4C72B0"); ax[1].set_xscale("log"); ax[1].set_title("Лог-шкала: читаемый колокол")
sns.ecdfplot(dep, ax=ax[2], color="#C44E52"); ax[2].set_xscale("log"); ax[2].set_title("ECDF (лог-x): доля ниже порога")
for a in ax: a.set_xlabel("депозит")
fig.tight_layout(); fig.savefig("m12_out/dist.png", dpi=150, bbox_inches="tight")
print("медиана:", np.median(dep).round(1), "| 95-й перцентиль:", np.percentile(dep, 95).round(1))
```

Что получилось. Левая панель — почти пустая, вся масса прижата к нулю, форма не читается. Центральная — после `set_xscale("log")` проступает симметричный колокол лог-нормали. ECDF справа отвечает на бизнес-вопрос напрямую: по оси x находишь порог депозита, по оси y читаешь, у какой доли игроков депозит ниже него. Файл `m12_out/dist.png`.

!!! question "Проверь себя"

    1. Какую структуру распределения прячет именно линейная шкала на этих данных?
    2. Почему ECDF не нужно подбирать ширину бина, а гистограмме нужно?

??? success "Ответы"

    1. Лог-нормальную форму (симметричный колокол в логарифме) и реальный разброс «китов» в хвосте — линейная ось сжимает 95% наблюдений в первые бины у нуля.
    2. ECDF строит точную долю наблюдений $\le x$ по формуле $\hat F_n(x)=\frac1n\sum \mathbf 1[X_i\le x]$ без агрегации в интервалы; гистограмма обязана резать ось на бины, и ширина бина произвольно меняет видимую форму (прячет бимодальность или создаёт зубчатый шум).

### Шаг 2: когортная retention-heatmap

Зачем. M12.5: главный график продуктовой аналитики. Строки — неделя регистрации, столбцы — возраст когорты в днях, цвет — доля вернувшихся. Палитра viridis — sequential и colorblind-safe (M12.7). Отрабатываем pivot в матрицу и чтение по двум осям.

```python
# step2_cohorts.py
import matplotlib.pyplot as plt, seaborn as sns
from data import events, sample

# знаменатель: сколько игроков в когорте (неделя регистрации)
cohort_size = sample.groupby(["channel", "reg_week"]).size().rename("size")
# числитель: уникальные вернувшиеся на день d
ret = events.groupby(["reg_week", "day"]).size()  # число событий-возвратов

# для heatmap берём канал A, считаем долю относительно размера когорты
sizeA = sample[sample.channel == "A"].groupby("reg_week").size()
eventsA = events.merge(sample[["channel", "reg_week"]].drop_duplicates(), on="reg_week")  # упрощённо
mat = (events.groupby(["reg_week", "day"]).size()
       .unstack("day").div(sizeA, axis=0)
       .clip(0, 1).iloc[:, ::4])  # каждый 4-й день для читаемости

plt.figure(figsize=(10, 5))
sns.heatmap(mat, annot=True, fmt=".0%", cmap="viridis",
            cbar_kws={"label": "Retention"}, vmin=0, vmax=1)
plt.title("Каждая новая когорта удерживается хуже предыдущей")  # заголовок-вывод
plt.xlabel("Возраст когорты, дней"); plt.ylabel("Неделя регистрации")
plt.tight_layout(); plt.savefig("m12_out/cohorts.png", dpi=150, bbox_inches="tight")
```

Что получилось. Матрица 8 строк (недели регистрации) на ~8 столбцов (возраст). Читаешь слева направо — цвет бледнеет: игроки отваливаются с возрастом когорты. Читаешь сверху вниз — нижние (свежие) строки бледнее верхних: качество приходящих когорт деградирует. Заголовок — утверждение с глаголом, не тема. Файл `m12_out/cohorts.png`.

!!! tip "Если annot загромождает"

    На широких матрицах `annot=True` превращает heatmap в таблицу и убивает data-ink ratio. Тогда annot убирают, оставляя только цвет и colorbar — цвет и есть величина. Аннотируй числами лишь когда строк/столбцов мало и точные значения важны.

### Шаг 3: retention-кривые без спагетти

Зачем. M12.5 «Динамика» + M12.11 «направляй внимание»: line оправдан, потому что по x есть порядок (время). Подсвечиваем худший канал, второй приглушаем в серый, ставим аннотацию на D7 с разницей — это усиление сигнала, не искажение (M12.11).

```python
# step3_curves.py
import numpy as np, matplotlib.pyplot as plt
from data import players, days

def curve(decay, q=1.0):
    return 0.5 * decay ** days * q

a = curve(0.85); b = curve(0.93)
d7_a, d7_b = a[7], b[7]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(days, b, color="#bbbbbb", lw=2, label="Канал B")          # приглушён
ax.plot(days, a, color="#C44E52", lw=2.5, label="Канал A (худший)")  # подсвечен
ax.annotate(f"D7: A={d7_a:.0%} против B={d7_b:.0%}\n(вдвое хуже)",
            xy=(7, d7_a), xytext=(12, d7_a + 0.12),
            arrowprops=dict(arrowstyle="->", color="#333"))
ax.set_ylim(0, None)  # честная ось значений от нуля
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_xlabel("День жизни"); ax.set_ylabel("Retention")
ax.set_title("Канал A теряет игроков вдвое быстрее к D7")
ax.legend(); fig.tight_layout()
fig.savefig("m12_out/curves.png", dpi=150, bbox_inches="tight")
```

Что получилось. Две кривые: красная (A) проваливается, серая (B) держится. Глаз сразу идёт на красную — это и есть направление внимания. Аннотация на D7 называет разницу числом, не заставляя зрителя считать. Заголовок-вывод. Файл `m12_out/curves.png`.

!!! question "Проверь себя"

    1. Почему здесь line оправдан, а соединить линией каналы (A, B) на оси x было бы ошибкой?
    2. Подсветка одной кривой и приглушение второй — это усиление сигнала или искажение? Где граница?

??? success "Ответы"

    1. У дня жизни есть осмысленный порядок (время), линия кодирует непрерывность и тренд. Каналы — номинальные категории без порядка; соединять их линией значит навязать несуществующую динамику, для категорий нужен bar.
    2. Усиление: данные не тронуты, мы лишь направляем внимание цветом и аннотацией (M12.11). Искажением это стало бы, если бы мы обрезали ось y или спрятали канал, который портит историю.

### Шаг 4: сравнение распределений LTV — boxplot плюс ECDF

Зачем. M12.5: boxplot компактен для сравнения групп, но прячет форму; ECDF возвращает честное сравнение (и лежит в основе теста Колмогорова-Смирнова). Деньги — лог-ось (M12.6).

```python
# step4_compare.py
import matplotlib.pyplot as plt, seaborn as sns
from data import players

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=players, x="channel", y="ltv_d30", hue="channel",
            palette="Set2", legend=False, ax=ax[0])
ax[0].set_yscale("log"); ax[0].set_title("Boxplot LTV (лог-y): медианы и хвосты")

sns.ecdfplot(data=players, x="ltv_d30", hue="channel", palette="Set2", ax=ax[1])
ax[1].set_xscale("log"); ax[1].set_title("ECDF LTV (лог-x): A целиком левее B")
fig.tight_layout(); fig.savefig("m12_out/boxplot.png", dpi=150, bbox_inches="tight")
```

Что получилось. Boxplot показывает, что у канала A медиана LTV и верхний квартиль ниже. ECDF усиливает вывод: кривая A целиком лежит левее кривой B — значит при любом пороге LTV доля «недотягивающих» игроков у A выше. Это та самая разница двух ECDF, по которой работает тест Колмогорова-Смирнова. Палитра Set2 — категориальная, colorblind-safe. Файл `m12_out/boxplot.png`.

### Шаг 5: финальный storytelling-график

Зачем. M12.11: собрать один презентационный график, который сам доносит решение. Тезис — «канал A убыточен на горизонте D30». Bar сравнения LTV против CPA по каналам: bar — лучший канал восприятия после положения, ось обязана начинаться с нуля (M12.6).

```python
# step5_final.py
import matplotlib.pyplot as plt, numpy as np
from data import players

cpa = {"A": 9.0, "B": 8.0}  # стоимость привлечения, условные единицы
ltv = players.groupby("channel")["ltv_d30"].mean()
ch = ["A", "B"]
ltv_v = [ltv[c] for c in ch]; cpa_v = [cpa[c] for c in ch]

x = np.arange(len(ch)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, cpa_v, w, label="CPA (затраты)", color="#bbbbbb")
ax.bar(x + w/2, ltv_v, w, label="LTV D30 (отдача)",
       color=["#C44E52" if l < c else "#55A868" for l, c in zip(ltv_v, cpa_v)])
ax.set_ylim(0, None)                      # bar: ось от нуля, не врём длиной
ax.set_xticks(x); ax.set_xticklabels([f"Канал {c}" for c in ch])
for xi, l, c in zip(x, ltv_v, cpa_v):
    if l < c:
        ax.annotate("LTV < CPA: убыток", xy=(xi + w/2, l), xytext=(xi, l + 1.5),
                    ha="center", color="#C44E52",
                    arrowprops=dict(arrowstyle="->", color="#C44E52"))
ax.set_ylabel("у.е. на игрока")
ax.set_title("Канал A убыточен на горизонте 30 дней — сокращаем закупку")
ax.legend(); fig.tight_layout()
fig.savefig("m12_out/story_final.png", dpi=150, bbox_inches="tight")
```

Что получилось. Для канала A столбец отдачи (LTV) короче столбца затрат (CPA) и покрашен в красный с аннотацией «убыток»; для B — наоборот, зелёный. Заголовок — готовое решение. Ось от нуля, длина столбца честно равна величине, смысл продублирован цветом и подписью (не только цвет). Файл `m12_out/story_final.png` — это слайд, который меняет решение, а не «интересные данные».

!!! question "Проверь себя"

    1. Почему именно для этого графика ось y обязана начинаться с нуля, а на шаге 3 (line) — нет?
    2. Чем заголовок «LTV и CPA по каналам» хуже, чем «Канал A убыточен на горизонте 30 дней»?

??? success "Ответы"

    1. Здесь величину кодирует длина столбца (bar) — обрезка оси исказила бы саму длину и раздула мелкую разницу. На шаге 3 line кодирует наклон и положение (динамику), а не длину, поэтому ноль не обязателен (но диапазон нужно подписать).
    2. Первый — заголовок-тема, лишь называет содержимое; зритель сам должен вывести смысл. Второй — заголовок-вывод с глаголом: аудитория читает его за секунду и уже знает решение, график лишь подтверждает.

## Критерий готовности

- [ ] `m12_out/dist.png`: три панели, видно, что линейная шкала прячет лог-нормальную форму; ECDF на лог-оси.
- [ ] `m12_out/cohorts.png`: heatmap читается по двум осям, палитра viridis, заголовок-вывод (утверждение, не тема).
- [ ] `m12_out/curves.png`: худший канал подсвечен, второй приглушён, аннотация на D7 с числом, ось значений от нуля.
- [ ] `m12_out/boxplot.png`: boxplot и ECDF рядом, обе денежные оси логарифмические, категориальная палитра.
- [ ] `m12_out/story_final.png`: bar с осью от нуля, смысл продублирован цветом и подписью, заголовок — готовое решение.
- [ ] Можешь устно объяснить про каждый график: что на $x$, что на цвете, какая геометрия, честна ли ось (грамматика графики из M12.4).
- [ ] Ни один график не кодирует смысл только цветом и не использует радужную/jet-палитру.

## Развитие

- Фасетная heatmap по каналам (`sns.FacetGrid` или `so.Plot(...).facet("channel")`): покажи, что деградацию когорт даёт именно канал A, а у B свежие строки стабильны — это слайд «причина» из кейса M12.11.
- Перепиши шаг 3 на seaborn.objects (`so`) или altair, явно выписав слои грамматики (encode x/y/color, mark_line) — сравни читаемость кода с императивным matplotlib.
- Overplotting (M12.12, челлендж): построй scatter «первый депозит против LTV» на 50 000 точек тремя способами — alpha, hexbin, 2D-density — и аргументируй, какой честнее передаёт плотность.
- Парадокс Симпсона: подмешай в данные сегмент, где агрегатный тренд retention обратен трендам внутри каналов, и покажи обманчивый агрегат против честной разбивки по фасетам.
- Собери четыре графика в дата-историю из M12.11 (контекст → свидетельство → причина → действие), один сделай интерактивным на altair/plotly с tooltip, и прогони через критику осей и colorblind-проверку.
