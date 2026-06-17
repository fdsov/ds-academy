# Воркшоп M4 · Продуктовая аналитика гемблинга на DuckDB

<span class="lecture-meta">Воркшоп к модулю M4 · ориентир 4-6 ч</span>

## Что отрабатываем

Этот воркшоп прогоняет руками всё ядро модуля M4 на одном связном датасете: реляционную модель (`players`, `deposits`, `bets`), порядок выполнения SELECT, агрегаты с GROUP BY/HAVING, JOIN без раздувания строк, CTE, оконные функции (`ROW_NUMBER`, `LAG`, `SUM OVER`), условную агрегацию через CASE, работу с датами и логику NULL. Стек — DuckDB как embedded OLAP-база 2026 года, который читает данные прямо из питона без сервера.

Понятия, которые ты закрепишь предметно:

- DAU через `bet_ts::date` и `COUNT(DISTINCT player_id)` — почему именно DISTINCT.
- FTD (первый депозит) через `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`.
- Retention-когорты D1/D7/D30 через self-join активности с условной агрегацией.
- Нарастающий итог GGR через `SUM(...) OVER (ORDER BY day)`.
- CASE-сегментация игроков на whale/mid/minnow.

Артефакт на выходе: файл `m4_queries.sql` с 7 запросами плюс питон-скрипт `run.py`, который генерирует синтетику, прогоняет запросы через DuckDB и печатает результаты. Запускается у любого, seed фиксирован.

## Данные

Никаких внешних файлов. Генерируем три связанные таблицы синтетического гемблинг-продукта прямо в питоне с фиксированным seed, сохраняем в Parquet — ровно тот формат, который DuckDB читает как таблицу.

```bash
uv init m4-workshop && cd m4-workshop
uv add duckdb pandas numpy
```

```python
# gen.py — генератор синтетики, seed фиксирован
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_PLAYERS = 4000
START = pd.Timestamp("2026-01-01")
COUNTRIES = ["UZ", "KZ", "RU", "BR", "IN", None]  # None даст NULL-страны намеренно
SOURCES = ["seo", "ppc", "affiliate", "referral"]

players = pd.DataFrame({
    "player_id": np.arange(1, N_PLAYERS + 1),
    "signup_date": START + pd.to_timedelta(rng.integers(0, 60, N_PLAYERS), unit="D"),
    "country": rng.choice(COUNTRIES, N_PLAYERS, p=[.30, .20, .20, .12, .15, .03]),
    "source": rng.choice(SOURCES, N_PLAYERS),
})

# Депозиты: не все игроки депозят, у активных — гео-экспоненциальное число пополнений
dep_rows = []
did = 1
for _, p in players.iterrows():
    if rng.random() < 0.45:          # 45% игроков сделали хотя бы один депозит
        n = 1 + rng.poisson(1.8)
        for k in range(n):
            day_off = int(rng.integers(0, 45)) + k * int(rng.integers(1, 10))
            dep_rows.append((did, p.player_id,
                             round(float(rng.gamma(2.0, 60)), 2),
                             p.signup_date + pd.Timedelta(days=day_off,
                                                          hours=int(rng.integers(0, 24)))))
            did += 1
deposits = pd.DataFrame(dep_rows,
    columns=["deposit_id", "player_id", "amount", "deposit_ts"])

# Ставки: активность растянута во времени, payout около 0.9 от stake в среднем
bet_rows = []
bid = 1
for _, p in players.iterrows():
    if rng.random() < 0.70:          # 70% игроков делали ставки
        n = 1 + rng.poisson(12)
        for _ in range(n):
            day_off = int(rng.integers(0, 90))
            stake = round(float(rng.gamma(1.5, 8)), 2)
            payout = round(stake * float(rng.choice([0, 0.5, 0.9, 1.0, 2.4],
                                                    p=[.45, .15, .15, .15, .10])), 2)
            bet_rows.append((bid, p.player_id, stake, payout,
                             p.signup_date + pd.Timedelta(days=day_off,
                                                          hours=int(rng.integers(0, 24)))))
            bid += 1
bets = pd.DataFrame(bet_rows,
    columns=["bet_id", "player_id", "stake", "payout", "bet_ts"])

players.to_parquet("players.parquet")
deposits.to_parquet("deposits.parquet")
bets.to_parquet("bets.parquet")
print(f"players={len(players)} deposits={len(deposits)} bets={len(bets)}")
```

Запусти `uv run gen.py`. Что получилось: три Parquet-файла, связь один-ко-многим (`player_id` — внешний ключ в `deposits` и `bets`), намеренно есть NULL-страны (ловушка из M4.11) и игроки без депозитов (нужны для LEFT JOIN из M4.6).

## Ход работы

Создай `run.py` с каркасом и дальше добавляй в него запросы по шагам.

```python
# run.py
import duckdb
con = duckdb.connect()

def q(sql):
    return con.execute(sql).df()
```

### Шаг 1: DAU по дням

Зачем. Отрабатываем M4.10: обрезаем timestamp до дня через `bet_ts::date` и считаем уникальных игроков. Ключевая ловушка модуля — `COUNT(*)` посчитал бы ставки, а не людей; DAU всегда через `COUNT(DISTINCT player_id)`.

```python
dau = q("""
    SELECT bet_ts::date           AS day,
           COUNT(DISTINCT player_id) AS dau,
           COUNT(*)                  AS bets_cnt
    FROM 'bets.parquet'
    GROUP BY bet_ts::date
    ORDER BY day
""")
print(dau.head(10))
```

Что получилось. Таблица день-DAU. Сравни колонки `dau` и `bets_cnt` — они расходятся в разы, потому что один игрок делает много ставок. Это наглядная иллюстрация, почему DISTINCT не опционален.

### Шаг 2: топ-депозиторы через свёртку

Зачем. M4.5: агрегаты с GROUP BY, `COUNT`/`SUM`/`AVG`, плюс HAVING как фильтр по группам. Считаем по игроку, оставляем только тех, кто суммарно занёс больше 300.

```python
top = q("""
    SELECT player_id,
           COUNT(*)            AS deposits_cnt,
           SUM(amount)         AS total_amount,
           ROUND(AVG(amount),2) AS avg_check
    FROM 'deposits.parquet'
    GROUP BY player_id
    HAVING SUM(amount) > 300
    ORDER BY total_amount DESC
    LIMIT 15
""")
print(top)
```

Что получилось. 15 крупнейших депозиторов. Обрати внимание: фильтр `SUM(amount) > 300` стоит в HAVING, а не в WHERE, потому что сумма известна только после группировки (порядок выполнения из M4.4).

!!! question "Проверь себя"

    1. Почему в Шаге 1 нельзя заменить `COUNT(DISTINCT player_id)` на `COUNT(*)`?
    2. Почему фильтр по сумме депозитов в Шаге 2 стоит в HAVING, а не в WHERE?

??? success "Ответы"

    1. `COUNT(*)` считает все строки-ставки, а один игрок делает их много — получится завышенный показатель активности, а не число уникальных пользователей. DAU — это люди, поэтому DISTINCT.
    2. Сумма по игроку вычисляется агрегатом после GROUP BY. На шаге WHERE строки ещё не сгруппированы, поэтому условие на агрегат ставится в HAVING.

### Шаг 3: FTD через ROW_NUMBER

Зачем. Сердце M4.8. `deposit_no = 1` — это первый депозит (FTD). Оконная функция не схлопывает строки: нумеруем депозиты внутри игрока по времени, потом во внешнем слое оставляем номер 1.

```python
ftd = q("""
    WITH numbered AS (
        SELECT player_id, amount, deposit_ts,
               ROW_NUMBER() OVER (PARTITION BY player_id
                                  ORDER BY deposit_ts) AS deposit_no
        FROM 'deposits.parquet'
    )
    SELECT player_id, amount AS ftd_amount, deposit_ts AS ftd_ts
    FROM numbered
    WHERE deposit_no = 1
    ORDER BY ftd_ts
""")
print(f"FTD-игроков: {len(ftd)}")
print(ftd.head())
```

Что получилось. Ровно одна строка на депозитора — первый депозит каждого. Проверь: `len(ftd)` должно совпадать с числом уникальных `player_id` в `deposits`.

### Шаг 4: нарастающий итог GGR

Зачем. M4.15 + оконный `SUM(...) OVER (ORDER BY day)`. GGR (gross gaming revenue) = `SUM(stake - payout)`. Сначала считаем дневной GGR обычным GROUP BY, затем накопленный итог оконной функцией поверх.

```python
ggr = q("""
    WITH daily AS (
        SELECT bet_ts::date AS day,
               SUM(stake - payout) AS ggr
        FROM 'bets.parquet'
        GROUP BY bet_ts::date
    )
    SELECT day, ggr,
           SUM(ggr) OVER (ORDER BY day
                          ROWS BETWEEN UNBOUNDED PRECEDING
                                   AND CURRENT ROW) AS cumulative_ggr
    FROM daily
    ORDER BY day
""")
print(ggr.tail(10))
```

Что получилось. Дневной GGR и нарастающая кривая выручки. `cumulative_ggr` в последней строке — это суммарный GGR за весь период; сверь его с `SELECT SUM(stake-payout)` по всей таблице.

### Шаг 5: CASE-сегментация игроков

Зачем. M4.9: условная логика и условная агрегация. Сегментируем по суммарному депозиту на whale/mid/minnow, а заодно в одном проходе считаем долю крупных депозитов через `SUM(CASE WHEN ... THEN amount ELSE 0 END)`.

```python
seg = q("""
    WITH per_player AS (
        SELECT player_id,
               SUM(amount) AS total,
               SUM(CASE WHEN amount >= 200 THEN amount ELSE 0 END) AS big_amount
        FROM 'deposits.parquet'
        GROUP BY player_id
    )
    SELECT CASE WHEN total >= 500 THEN 'whale'
                WHEN total >= 150 THEN 'mid'
                ELSE 'minnow' END AS segment,
           COUNT(*)            AS players,
           ROUND(SUM(total),2) AS revenue,
           ROUND(SUM(big_amount) / SUM(total) * 100, 1) AS big_share_pct
    FROM per_player
    GROUP BY 1
    ORDER BY revenue DESC
""")
print(seg)
```

Что получилось. Три сегмента с числом игроков, выручкой и долей крупных депозитов. Типичная картина гемблинга: whale-сегмент мал по числу, но даёт основную выручку.

### Шаг 6: LEFT JOIN и NULL-страны

Зачем. M4.6 (LEFT JOIN, чтобы не потерять страны без депозитов) и M4.11 (NULL-логика). `COALESCE` спасает и сумму (NULL → 0), и саму страну (NULL → 'unknown').

```python
geo = q("""
    SELECT COALESCE(p.country, 'unknown')      AS country,
           COUNT(DISTINCT p.player_id)          AS players,
           COUNT(DISTINCT d.player_id)          AS depositors,
           COALESCE(ROUND(SUM(d.amount),2), 0)  AS revenue
    FROM 'players.parquet' p
    LEFT JOIN 'deposits.parquet' d ON d.player_id = p.player_id
    GROUP BY 1
    ORDER BY revenue DESC
""")
print(geo)
```

Что получилось. Все страны игроков, включая NULL-страну как 'unknown'. С INNER JOIN строки без депозитов исчезли бы. Заметь раздувания строк здесь нет, потому что джойним `players` (одна строка на игрока) с `deposits` по ключу и сразу агрегируем DISTINCT-ом.

!!! question "Проверь себя"

    1. Что вернёт Шаг 6, если заменить LEFT JOIN на INNER JOIN?
    2. Почему `WHERE country <> 'UZ'` не вернул бы строки с `country IS NULL`?

??? success "Ответы"

    1. Исчезнут страны/игроки без единого депозита и строка 'unknown' с revenue 0 — выводы по охвату гео будут заниженными.
    2. Для NULL сравнение «не равно UZ» даёт «неизвестно», а не TRUE, поэтому такие строки выпадают. Нужно явно добавить `OR country IS NULL`.

### Шаг 7: retention-когорты D1/D7/D30

Зачем. Кульминация модуля (M4.8): когорты через self-join активности и условную агрегацию по лагу в днях. Активность = наличие ставки. День 0 когорты — первый день активности игрока.

```python
ret = q("""
    WITH activity AS (
        SELECT player_id, bet_ts::date AS active_day
        FROM 'bets.parquet'
        GROUP BY player_id, bet_ts::date
    ),
    cohort AS (
        SELECT player_id, MIN(active_day) AS cohort_day
        FROM activity
        GROUP BY player_id
    )
    SELECT date_trunc('week', c.cohort_day) AS cohort_week,
           COUNT(DISTINCT c.player_id)      AS cohort_size,
           ROUND(COUNT(DISTINCT CASE WHEN a.active_day - c.cohort_day = 1
                          THEN a.player_id END) * 100.0
                 / COUNT(DISTINCT c.player_id), 1) AS d1_pct,
           ROUND(COUNT(DISTINCT CASE WHEN a.active_day - c.cohort_day = 7
                          THEN a.player_id END) * 100.0
                 / COUNT(DISTINCT c.player_id), 1) AS d7_pct,
           ROUND(COUNT(DISTINCT CASE WHEN a.active_day - c.cohort_day = 30
                          THEN a.player_id END) * 100.0
                 / COUNT(DISTINCT c.player_id), 1) AS d30_pct
    FROM cohort c
    LEFT JOIN activity a ON a.player_id = c.player_id
    GROUP BY 1
    ORDER BY cohort_week
""")
print(ret)
```

Что получилось. Таблица: неделя когорты, её размер, retention D1/D7/D30 в процентах. На синтетике активность размазана равномерно, поэтому кривая будет пологой — на реальных данных она резко падает после D1. Сохрани результат в `m4_queries.sql` и убедись, что все 7 запросов отрабатывают без ошибок.

## Критерий готовности

- [ ] `uv run gen.py` создаёт три Parquet-файла, в `players` есть NULL-страны и игроки без депозитов
- [ ] DAU считается через `COUNT(DISTINCT player_id)`, и видно расхождение с `COUNT(*)`
- [ ] FTD через `ROW_NUMBER()` даёт ровно одну строку на депозитора
- [ ] Нарастающий GGR в последней строке совпадает с `SUM(stake - payout)` по всей таблице
- [ ] CASE-сегментация даёт whale/mid/minnow с долей крупных депозитов
- [ ] LEFT JOIN показывает страну 'unknown' и страны с revenue 0
- [ ] Retention-таблица D1/D7/D30 по неделям когорт считается одним запросом
- [ ] Все 7 запросов сохранены в `m4_queries.sql` и отрабатывают без ошибок

## Развитие

- Замени self-join в Шаге 7 на массивы DuckDB (`list` активных дней) и сравни читаемость — DuckDB-специфичный приём из M4.14.
- Добавь когортный LTV: накопленный депозит по неделе FTD через `SUM(amount) OVER (PARTITION BY cohort_week ORDER BY week_offset)`.
- Построй тепловую карту retention в matplotlib (cohort_week по строкам, лаг по столбцам) — закрой мини-проект модуля.
- Подними те же запросы против PostgreSQL через SQLAlchemy и зафиксируй, какие функции дат пришлось править (`bet_ts::date` vs `date_trunc`), отрабатывая переносимость диалектов из M4.14.
- Прогони `EXPLAIN ANALYZE` на запросе Шага 7, найди самый дорогой шаг и проверь гипотезу про индекс по `player_id` из M4.12-M4.13.
