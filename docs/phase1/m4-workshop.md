# Воркшоп M4 · Продуктовая аналитика гемблинга на DuckDB

<span class="lecture-meta">Воркшоп к модулю M4 · ориентир 4-6 ч</span>

## Что ты построишь

Этот воркшоп прогоняет руками всё ядро модуля M4 на одном связном датасете: реляционную модель (`players`, `deposits`, `bets`), порядок выполнения SELECT, агрегаты с GROUP BY/HAVING, JOIN без раздувания строк, CTE, оконные функции (`ROW_NUMBER`, `LAG`, `SUM OVER`), условную агрегацию через CASE, работу с датами и логику NULL. Стек — DuckDB как embedded OLAP-база 2026 года, который читает данные прямо из питона без сервера.

Понятия, которые ты закрепишь предметно:

- DAU через `bet_ts::date` и `COUNT(DISTINCT player_id)` — почему именно DISTINCT.
- FTD (первый депозит) через `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`.
- Retention-когорты D1/D7/D30 через self-join активности с условной агрегацией.
- Нарастающий итог GGR через `SUM(...) OVER (ORDER BY day)`.
- CASE-сегментация игроков на whale/mid/minnow.

Артефакт на выходе: файл `m4_queries.sql` с 7 запросами плюс питон-скрипт `run.py`, который генерирует синтетику, прогоняет запросы через DuckDB и печатает результаты. Запускается у любого, seed фиксирован.

!!! info "Как устроен этот воркшоп"

    Это не лекция с готовым кодом, а задачник. Каждый шаг построен по схеме **Задача → Критерий → Решение**:

    - **Задача** — что именно написать руками, с явными именами выходных переменных (`dau`, `top`, `ftd`, ...), которые проверяет критерий.
    - **Критерий шага** — кусок кода с `assert`, который ты запускаешь после своего запроса. Зелёный прогон (без ошибки) = шаг сдан. Это твой локальный авто-грейдер, аналог кнопки «Проверить».
    - **Решение** спрятано под спойлер `Решение`. Открывай его только после своей попытки — чтобы сверить подход, а не списать.
    - **Числовые подзадачи** проверяются прямо на странице: посчитай число по результату запроса, впиши в поле, нажми «Проверить».

    Все `assert` и числа рассчитаны на синтетику с `seed=42`. Не меняй seed, иначе числа поплывут.

## Бизнес-кейс

Ты — продуктовый аналитик в команде iGaming-продукта. К тебе приходит продакт-оунер вертикали casino: команда летит вслепую. Сейчас единственный регулярный отчёт — это «сколько денег пришло за вчера», а вопросы про удержание и качество трафика закрываются на глаз. Из-за этого недавно влили бюджет в канал, который казался дешёвым по CPA, но игроки оттуда не возвращались — деньги сгорели, и доказать это постфактум было нечем. Продакту нужен не дашборд из 30 графиков, а набор воспроизводимых SQL-метрик, на которые можно опереться при планировании.

!!! example "Ситуация"

    - Гипотеза команды: ретеншн новых когорт просел, но цифр нет — спор «просел / не просел» идёт на эмоциях уже вторую неделю.
    - На кону решение по перераспределению ~1,5 млн ₽/мес маркетингового бюджета между каналами (`seo` / `ppc` / `affiliate` / `referral`): ориентир — если когортный D7 канала ниже 15%, его доля бюджета режется в пользу каналов с лучшим удержанием.
    - Отдельный вопрос от продакта: whale-сегмент даёт основную выручку, но никто не знает его реальную долю — от этого зависит, ставить ли VIP-менеджера (отдельная ставка в найме).
    - Ограничение: доступ только к трём сырым таблицам (`players`, `deposits`, `bets`), внешней BI-витрины нет, ответ нужен к планёрке через 3 дня. Источник истины — твои запросы.

## Предпосылки

Нужно ядро модуля M4: порядок выполнения SELECT (M4.4), агрегаты и HAVING (M4.5), JOIN (M4.6), CTE и оконные функции (M4.8), CASE (M4.9), работа с датами (M4.10, M4.15), логика NULL (M4.11). Если эти темы плывут — держи конспект модуля рядом, шаги построены ровно по нему.

Окружение собираем через `uv` — стандарт 2026 для воспроизводимых Python-проектов.

```bash
uv init m4-workshop && cd m4-workshop
uv add duckdb pandas numpy pyarrow
```

Проверь, что всё встало:

```bash
uv run python -c "import duckdb, pandas, numpy; print('ok')"
```

!!! note "Почему DuckDB, а не SQLite или Postgres"

    DuckDB — это embedded OLAP-движок: ставится одной строкой, не требует сервера и читает Parquet/CSV прямо как таблицу (`FROM 'file.parquet'`). Для аналитики на одной машине это самый короткий путь от данных к SQL: не нужно поднимать инстанс, грузить данные и настраивать коннект. Синтаксис — почти стандартный SQL, поэтому навык переносится на Postgres и ClickHouse.

## Данные

Никаких внешних файлов. Генерируем три связанные таблицы синтетического гемблинг-продукта прямо в питоне с фиксированным seed, сохраняем в Parquet — ровно тот формат, который DuckDB читает как таблицу.

Это setup-код, а не задача — скопируй в `gen.py` и запусти как есть (`uv run gen.py`). Дальше ты относишься к трём Parquet-файлам как к сырым таблицам продакшена.

!!! warning "Зачем фиксировать seed"

    Без `seed` каждый запуск даст другие числа, и ты не сможешь сверить результат с воркшопом или воспроизвести баг. Все критерии-`assert` и числовые ответы ниже рассчитаны на `seed=42`.

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

После `uv run gen.py` ты получишь три Parquet-файла, связь один-ко-многим (`player_id` — внешний ключ в `deposits` и `bets`), намеренно есть NULL-страны (ловушка из M4.11) и игроки без депозитов (нужны для LEFT JOIN из M4.6). Печать должна показать `players=4000 deposits=4954 bets=37157`.

Прежде чем писать запросы, прикинь одну метрику качества трафика — её ты сверишь уже на шаге 3.

```text
TASK: 45% из 4000 зарегистрированных игроков делают хотя бы один депозит. После запуска gen.py посчитай ФАКТИЧЕСКУЮ долю депозиторов: уникальных player_id в deposits.parquet, делённую на число игроков, в процентах. Округли до 0.1.
ANSWER: 43.7
TOL: 0.4
UNIT: %
PLACEHOLDER: 0.0
EXPLAIN: Уникальных депозиторов 1747, всего игроков 4000, доля = 1747/4000 = 43.68% = 43.7%. Заложено было 45%, но из-за случайности генерации фактическая доля чуть ниже - это нормальный шум. Эта метрика (registration -> deposit) - базовая конверсия воронки, на неё опираются все деньги ниже.
```

## Ход работы

Создай `run.py` с каркасом и дальше добавляй в него запросы по шагам. Каждый запрос складывай в переменную с указанным именем — критерии проверяют именно их.

```python
# run.py
import duckdb
con = duckdb.connect()

def q(sql):
    return con.execute(sql).df()
```

### Шаг 1: DAU по дням

**Зачем.** Отрабатываем M4.10: обрезаем timestamp до дня через `bet_ts::date` и считаем уникальных игроков. Ключевая ловушка модуля — `COUNT(*)` посчитал бы ставки, а не людей; DAU всегда через `COUNT(DISTINCT player_id)`.

**Задача.** Напиши запрос по `bets.parquet`, который вернёт по одной строке на день (`day = bet_ts::date`) с двумя метриками: `dau = COUNT(DISTINCT player_id)` и `bets_cnt = COUNT(*)`. Отсортируй по `day`. Положи результат в `dau`.

**Критерий шага:**

```python
assert {"day", "dau", "bets_cnt"} <= set(dau.columns)
assert (dau["dau"] <= dau["bets_cnt"]).all(), "уникальных не может быть больше, чем строк-ставок"
assert dau["bets_cnt"].sum() == q("SELECT COUNT(*) c FROM 'bets.parquet'")["c"][0]
assert dau["dau"].sum() < dau["bets_cnt"].sum(), "один игрок делает много ставок -> DAU заметно ниже"
print("OK: DAU считается по уникальным игрокам")
```

??? tip "Подсказка"

    Группировать можно прямо по выражению `bet_ts::date` — повторять его в SELECT и GROUP BY не страшно. `COUNT(DISTINCT ...)` и `COUNT(*)` спокойно живут в одном SELECT.

??? success "Решение"

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

    **Почему так.** Колонки `dau` и `bets_cnt` расходятся в разы (среднее число ставок на игрока в день > 1), потому что один игрок делает много ставок. Это наглядная иллюстрация, почему DISTINCT не опционален: без него ты бы рапортовал «активность», а не «людей», и завысил бы аудиторию в несколько раз.

```text
Q: Почему DAU нельзя считать через COUNT(*) вместо COUNT(DISTINCT player_id)?
[ ] COUNT(*) медленнее на больших таблицах
[x] COUNT(*) считает все строки-ставки, а один игрок делает их много -> получится активность, а не число уникальных людей
[ ] COUNT(*) игнорирует строки с NULL и занизит результат
> DAU - это люди. Один игрок за день делает десятки ставок, поэтому COUNT(*) даёт активность, а DISTINCT - аудиторию. Это разные метрики.
```

### Шаг 2: Топ-депозиторы через свёртку

**Зачем.** M4.5: агрегаты с GROUP BY, `COUNT`/`SUM`/`AVG`, плюс HAVING как фильтр по группам. Считаем по игроку, оставляем только тех, кто суммарно занёс больше 300.

**Задача.** По `deposits.parquet` сгруппируй по `player_id` и посчитай `deposits_cnt = COUNT(*)`, `total_amount = SUM(amount)`, `avg_check = ROUND(AVG(amount), 2)`. Оставь только игроков с суммой депозитов больше 300 (через HAVING, не WHERE), отсортируй по `total_amount` убыванием, возьми первые 15. Результат в `top`.

**Критерий шага:**

```python
assert len(top) == 15
assert (top["total_amount"] > 300).all(), "фильтр по сумме должен быть в HAVING"
assert top["total_amount"].is_monotonic_decreasing, "сортировка по total_amount убыванием"
print("OK: топ-15 депозиторов с фильтром по сумме")
```

??? tip "Подсказка"

    Фильтр на агрегат (`SUM(amount) > 300`) нельзя поставить в WHERE — там строки ещё не сгруппированы. Это и есть смысл HAVING. `LIMIT 15` идёт в самом конце.

??? success "Решение"

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

    **Почему так.** Фильтр `SUM(amount) > 300` стоит в HAVING, а не в WHERE, потому что сумма известна только после группировки (порядок выполнения из M4.4: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT). Положи `WHERE amount > 300` — и ты отфильтруешь отдельные крупные депозиты, а не игроков с большой суммой. Это разные вопросы.

```text
Q: Почему фильтр SUM(amount) > 300 стоит в HAVING, а не в WHERE?
[ ] HAVING работает быстрее на агрегатах
[x] Сумма по игроку вычисляется после GROUP BY, а на этапе WHERE строки ещё не сгруппированы
[ ] WHERE нельзя использовать вместе с ORDER BY
> Порядок выполнения: WHERE отрабатывает до группировки и видит только отдельные строки. Условие на агрегат (SUM/COUNT/AVG) ставится в HAVING - после GROUP BY.
```

### Шаг 3: FTD через ROW_NUMBER

**Зачем.** Сердце M4.8. `deposit_no = 1` — это первый депозит (FTD). Оконная функция не схлопывает строки: нумеруем депозиты внутри игрока по времени, потом во внешнем слое оставляем номер 1. Если бы взяли `GROUP BY player_id` с `MIN(deposit_ts)`, пришлось бы вторым джойном доставать сумму того депозита — окно решает это в один проход.

**Задача.** В CTE `numbered` пронумеруй депозиты внутри каждого игрока: `ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY deposit_ts) AS deposit_no`. Во внешнем запросе оставь только `deposit_no = 1` и верни `player_id`, `ftd_amount = amount`, `ftd_ts = deposit_ts`. Результат в `ftd`.

**Критерий шага:**

```python
n_depositors = q("SELECT COUNT(DISTINCT player_id) c FROM 'deposits.parquet'")["c"][0]
assert len(ftd) == n_depositors, "ровно одна строка на депозитора"
assert ftd["player_id"].nunique() == len(ftd), "ни один игрок не задвоился"
print(f"OK: FTD-строк {len(ftd)} = числу уникальных депозиторов")
```

Сколько FTD-игроков получилось — это база для конверсии воронки. Сверь число:

```text
TASK: Сколько строк в результате запроса FTD (len(ftd))? Это число уникальных депозиторов. Ответ - целое число.
ANSWER: 1747
TOL: 0
PLACEHOLDER: целое число
EXPLAIN: Запрос оставляет deposit_no = 1, то есть ровно один первый депозит на игрока. Поэтому число строк в точности равно числу уникальных player_id в deposits = 1747. Это та же цифра, что и в разминке про долю депозиторов (1747/4000 = 43.7%).
```

??? tip "Подсказка"

    `ROW_NUMBER()` присваивает 1, 2, 3... внутри каждой партиции по порядку из `ORDER BY`. Самый ранний депозит игрока получает номер 1. Фильтр `WHERE deposit_no = 1` идёт во ВНЕШНЕМ запросе — в том же SELECT, где определена оконная функция, к её результату обращаться нельзя.

??? success "Решение"

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

    **Почему так.** Ровно одна строка на депозитора — первый депозит каждого. Оконная функция не схлопнула строки (как GROUP BY), а пронумеровала их, поэтому в `numbered` сохранились `amount` и `deposit_ts` именно того первого депозита. `len(ftd)` совпадает с числом уникальных `player_id` в `deposits`.

```text
Q: Чем ROW_NUMBER() OVER (...) принципиально отличается от GROUP BY player_id для задачи FTD?
[ ] ROW_NUMBER работает только в DuckDB, а GROUP BY везде
[x] Оконная функция не схлопывает строки - сохраняет amount и deposit_ts первого депозита в один проход, без второго JOIN
[ ] GROUP BY не умеет сортировать по дате
> GROUP BY свернул бы строки и вернул только агрегаты (MIN/MAX/SUM). Чтобы достать amount именно первого депозита, понадобился бы self-join. Окно нумерует строки, сохраняя все поля, - отсюда выбираешь номер 1.
```

### Шаг 4: Нарастающий итог GGR

**Зачем.** M4.15 + оконный `SUM(...) OVER (ORDER BY day)`. GGR (gross gaming revenue) = `SUM(stake - payout)` — это маржа казино. Сначала считаем дневной GGR обычным GROUP BY, затем накопленный итог оконной функцией поверх, чтобы видеть кривую выручки нарастающим итогом.

**Задача.** В CTE `daily` посчитай дневной GGR: `day = bet_ts::date`, `ggr = SUM(stake - payout)`. Во внешнем запросе добавь `cumulative_ggr = SUM(ggr) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`. Отсортируй по `day`. Результат в `ggr`.

**Критерий шага:**

```python
total_ggr = q("SELECT SUM(stake - payout) s FROM 'bets.parquet'")["s"][0]
assert {"day", "ggr", "cumulative_ggr"} <= set(ggr.columns)
assert abs(ggr["cumulative_ggr"].iloc[-1] - total_ggr) < 0.01, \
    "накопленный итог в последней строке = SUM(stake-payout) по всей таблице"
print(f"OK: нарастающий GGR сходится с общей суммой, итог={ggr['cumulative_ggr'].iloc[-1]:.2f}")
```

Сверь итоговую цифру выручки:

```text
TASK: Чему равен суммарный GGR за весь период - значение cumulative_ggr в ПОСЛЕДНЕЙ строке (оно же SUM(stake - payout) по всей таблице bets)? Округли до целого.
ANSWER: 176670
TOL: 150
PLACEHOLDER: целое число
EXPLAIN: GGR = stake - payout по каждой ставке, просуммированный по всем 37157 ставкам, даёт около 176670. Нарастающий итог в последней строке обязан совпасть с этой суммой - это и есть способ проверить, что окно не потеряло и не задвоило строки.
```

??? tip "Подсказка"

    Сначала сверни ставки в дневной GGR через обычный GROUP BY (это отдельный CTE). Оконная функция применяется уже к свёрнутым дневным строкам: `SUM(ggr) OVER (ORDER BY day ...)`. Рамка `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` означает «от начала и до текущей строки включительно».

??? success "Решение"

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

    **Почему так.** `cumulative_ggr` в последней строке — это суммарный GGR за весь период. То, что он точно совпал с `SELECT SUM(stake-payout)` по всей таблице, доказывает, что окно прошло по всем дням без пропусков и дублей. Дневной GGR показывает волатильность, накопленная кривая — общий тренд маржи.

```text
Q: Что задаёт рамка ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW при ORDER BY day?
[ ] Берёт только текущий день, остальные игнорирует
[x] Суммирует все строки от начала отсортированного набора до текущей включительно - это и есть нарастающий итог
[ ] Суммирует весь набор целиком, одинаково в каждой строке
> UNBOUNDED PRECEDING - от первой строки, CURRENT ROW - до текущей. Получается running total. Без рамки и с ORDER BY поведение по умолчанию такое же, но писать рамку явно - надёжнее и читаемее.
```

### Шаг 5: CASE-сегментация игроков

**Зачем.** M4.9: условная логика и условная агрегация. Сегментируем по суммарному депозиту на whale/mid/minnow, а заодно в одном проходе считаем долю крупных депозитов через `SUM(CASE WHEN ... THEN amount ELSE 0 END)`. Это прямой ответ на вопрос продакта про долю whale-выручки и ставку VIP-менеджера.

**Задача.** В CTE `per_player` посчитай по игроку `total = SUM(amount)` и `big_amount = SUM(CASE WHEN amount >= 200 THEN amount ELSE 0 END)`. Во внешнем запросе через CASE раздели на сегменты (`total >= 500` → whale, `>= 150` → mid, иначе minnow) и посчитай по сегменту: `players = COUNT(*)`, `revenue = ROUND(SUM(total), 2)`, `big_share_pct = ROUND(SUM(big_amount)/SUM(total)*100, 1)`. Отсортируй по `revenue` убыванием. Результат в `seg`.

**Критерий шага:**

```python
assert set(seg["segment"]) == {"whale", "mid", "minnow"}
s = seg.set_index("segment")
assert s.loc["whale", "big_share_pct"] > 50, "у китов крупные депозиты - больше половины оборота"
assert s.loc["minnow", "big_share_pct"] == 0, "у minnow депозитов >= 200 нет по определению порога"
print("OK: три сегмента, доля крупных депозитов посчитана")
```

Ответь продакту на вопрос про VIP-менеджера — долей выручки китов:

```text
TASK: Какую долю ВСЕЙ выручки (SUM revenue по всем сегментам) даёт сегмент whale? Возьми revenue whale, раздели на сумму revenue по трём сегментам, переведи в проценты. Округли до 0.1.
ANSWER: 40.8
TOL: 1.0
UNIT: %
PLACEHOLDER: 0.0
EXPLAIN: revenue whale около 243853, суммарная revenue около 597060, доля = 243853/597060 = 40.8%. Whale - это лишь ~21% игроков-депозиторов, но почти 41% денег. Это и есть аргумент за VIP-менеджера: концентрация выручки в малом сегменте оправдывает персональный сервис.
```

??? tip "Подсказка"

    `big_share_pct` и сегмент — это два разных применения CASE. Первое (`SUM(CASE WHEN amount >= 200 ...)`) считается на уровне депозитов внутри `per_player`. Второе (whale/mid/minnow) — на уровне игрока во внешнем GROUP BY. Группировать по выражению CASE можно через `GROUP BY 1` (по первой колонке SELECT).

??? success "Решение"

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

    **Почему так.** Типичная картина гемблинга: whale-сегмент мал по числу игроков (~21% депозиторов), но даёт ~41% выручки. `big_share_pct` подтверждает природу сегмента: у китов крупные депозиты (>= 200) составляют больше половины оборота, у minnow их нет вовсе. Условная агрегация позволила посчитать «долю крупных денег» в том же проходе, без отдельного запроса.

### Шаг 6: LEFT JOIN и NULL-страны

**Зачем.** M4.6 (LEFT JOIN, чтобы не потерять игроков без депозитов) и M4.11 (NULL-логика). `COALESCE` спасает и сумму (NULL → 0 для игроков без депозитов), и саму страну (NULL → 'unknown'). Это про честный охват гео: с INNER JOIN ты бы тихо потерял часть аудитории.

**Задача.** Сделай LEFT JOIN `players.parquet` к `deposits.parquet` по `player_id`. Сгруппируй по `COALESCE(country, 'unknown')` и посчитай: `players = COUNT(DISTINCT p.player_id)`, `depositors = COUNT(DISTINCT d.player_id)`, `revenue = COALESCE(ROUND(SUM(d.amount), 2), 0)`. Отсортируй по `revenue` убыванием. Результат в `geo`.

**Критерий шага:**

```python
assert "unknown" in set(geo["country"]), "NULL-страна должна стать 'unknown', а не выпасть"
assert geo["players"].sum() == 4000, "LEFT JOIN сохранил всех игроков, включая без депозитов"
assert (geo["depositors"] <= geo["players"]).all()
print("OK: гео-разбивка с unknown, ни один игрок не потерян")
```

??? tip "Подсказка"

    Джойни от `players` (одна строка на игрока) — тогда раздувания не будет: после JOIN ты сразу агрегируешь через `COUNT(DISTINCT ...)`. `COUNT(DISTINCT d.player_id)` автоматически не считает NULL-строки от LEFT JOIN (игроки без депозитов), поэтому даёт честное число депозиторов.

??? success "Решение"

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

    **Почему так.** Все страны игроков попали в результат, включая NULL-страну как 'unknown'. С INNER JOIN строки игроков без депозитов исчезли бы, и охват по гео оказался бы занижен. Раздувания строк здесь нет, потому что джойним `players` (одна строка на игрока) с `deposits` по ключу и сразу агрегируем через DISTINCT.

```text
Q: Что изменится в результате, если заменить LEFT JOIN на INNER JOIN?
[ ] Ничего, результат идентичен
[x] Исчезнут игроки без единого депозита и строки с revenue 0 - охват по гео окажется занижен
[ ] Запрос вернёт ошибку из-за NULL в country
> INNER JOIN оставляет только пары с совпадением в обеих таблицах, то есть только депозиторов. Игроки без депозитов выпадут, и ты недосчитаешь аудиторию по странам.
---
Q: Почему фильтр WHERE country <> 'UZ' не вернёт строки, где country IS NULL?
[ ] Потому что NULL автоматически приравнивается к 'UZ'
[x] Сравнение NULL <> 'UZ' даёт UNKNOWN, а не TRUE, поэтому такие строки не проходят фильтр
[ ] Потому что DuckDB не поддерживает NULL в текстовых колонках
> Любое сравнение с NULL даёт UNKNOWN (трёхзначная логика). WHERE пропускает только TRUE. Чтобы оставить NULL-страны, нужно явно: WHERE country <> 'UZ' OR country IS NULL.
```

### Шаг 7: Retention-когорты D1/D7/D30

**Зачем.** Кульминация модуля (M4.8): когорты через self-join активности и условную агрегацию по лагу в днях. Активность = наличие ставки. День 0 когорты — первый день активности игрока. Это прямой ответ на спор команды «ретеншн просел / не просел» и основа решения по бюджету каналов.

**Задача.** Построй три CTE/слоя: (1) `activity` — уникальные пары `player_id, active_day = bet_ts::date`; (2) `cohort` — `player_id` и `cohort_day = MIN(active_day)`. Затем LEFT JOIN `cohort` к `activity` по игроку и посчитай по неделям когорты (`date_trunc('week', cohort_day)`): `cohort_size = COUNT(DISTINCT player_id)` и доли вернувшихся на D1/D7/D30 через `COUNT(DISTINCT CASE WHEN a.active_day - c.cohort_day = N THEN a.player_id END) * 100.0 / COUNT(DISTINCT c.player_id)`. Результат в `ret`.

**Критерий шага:**

```python
n_bettors = q("SELECT COUNT(DISTINCT player_id) c FROM 'bets.parquet'")["c"][0]
assert {"cohort_week", "cohort_size", "d1_pct", "d7_pct", "d30_pct"} <= set(ret.columns)
assert ret["cohort_size"].sum() == n_bettors, "каждый игрок со ставками попал ровно в одну когорту"
assert (ret["d1_pct"] >= 0).all() and (ret["d7_pct"] >= 0).all()
print(f"OK: retention по {len(ret)} неделям когорт, всего в когортах {ret['cohort_size'].sum()} игроков")
```

??? tip "Подсказка"

    Разница дат `a.active_day - c.cohort_day` в DuckDB даёт целое число дней — это и есть лаг. Условная агрегация `COUNT(DISTINCT CASE WHEN лаг = 7 THEN player_id END)` считает уникальных вернувшихся на 7-й день, а деление на `cohort_size` переводит в процент. Не забудь `* 100.0` (а не `* 100`), иначе целочисленное деление обнулит дробь.

??? success "Решение"

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

    **Почему так.** Таблица: неделя когорты, её размер, retention D1/D7/D30 в процентах. Сумма `cohort_size` по всем неделям равна числу игроков со ставками — значит каждый попал ровно в одну когорту (по своему первому активному дню). На синтетике активность размазана равномерно, поэтому кривая пологая (D1/D7/D30 около 12-17%) — на реальных данных она резко падает после D1. Сохрани все 7 запросов в `m4_queries.sql` и убедись, что они отрабатывают без ошибок.

```text
Q: Как в этом запросе определяется "день 0" когорты игрока?
[ ] Дата регистрации из players.signup_date
[x] Первый день, когда игрок сделал ставку: MIN(active_day) в CTE cohort
[ ] Понедельник недели, в которую игрок зарегистрировался
> Когорта строится по первому ДЕЙСТВИЮ (ставке), а не по дате регистрации. Поэтому cohort_day = MIN(active_day). Лаги D1/D7/D30 считаются от этого дня.
---
Q: Зачем activity сначала сворачивается до уникальных пар (player_id, active_day) через GROUP BY, прежде чем считать retention?
[ ] Чтобы ускорить запрос за счёт меньшего объёма
[x] Иначе несколько ставок в один день раздули бы счётчики, а нам нужен факт активности в день, а не число ставок
[ ] GROUP BY здесь обязателен синтаксически, без него ошибка
> Retention - это вернулся/не вернулся в день N, бинарный факт. Без свёртки до уникального дня COUNT задвоился бы на игроках, сделавших много ставок в один день. DISTINCT в основном запросе тоже страхует, но чистая activity - это явная и читаемая модель.
```

## Типичные ошибки

- **DAU через `COUNT(*)`.** Считает ставки, а не людей, и завышает аудиторию в разы. DAU — всегда `COUNT(DISTINCT player_id)`.
- **Фильтр по агрегату в WHERE.** `WHERE SUM(amount) > 300` не работает: на этапе WHERE строки ещё не сгруппированы. Условие на агрегат — только в HAVING.
- **FTD через `GROUP BY` + `MIN`.** Даст дату первого депозита, но не его сумму без второго JOIN. Оконный `ROW_NUMBER()` достаёт всю строку первого депозита в один проход.
- **Целочисленное деление в retention.** `count * 100 / total` при целых типах обнулит дробь. Пиши `* 100.0`, чтобы получить проценты.
- **INNER JOIN там, где нужен LEFT.** Игроки без депозитов и NULL-страны тихо выпадают — охват гео и аудитории занижается без единой ошибки.
- **Сравнение с NULL через `<>` / `=`.** `country <> 'UZ'` молча выкидывает строки с `country IS NULL` (результат UNKNOWN, а не TRUE). Для NULL-стран нужен `COALESCE` или явное `OR ... IS NULL`.
- **Раздувание строк при JOIN.** Если джойнить `players` с `bets` (много строк на игрока) и потом суммировать без DISTINCT, метрики аудитории задвоятся. Джойни от таблицы-«одного» и агрегируй через DISTINCT.
- **Доверие синтетике как реальной кривой.** На сгенерированных данных retention пологий; на проде D1 падает резко. Методология верна, форма кривой — артефакт генератора.

!!! tip "AI-копилот в этом воркшопе"

    Где нейросеть реально ускорит: вспомнить синтаксис оконной функции (`ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)`), накидать каркас CTE для retention, перевести запрос с одного диалекта на другой, объяснить разницу `date_trunc` vs `::date`.

    Где AI подведёт именно здесь: (1) с готовностью поставит фильтр по агрегату в WHERE или забудет `* 100.0` — проверяй порядок выполнения и типы сам; (2) предложит INNER JOIN «потому что компактнее», молча потеряв игроков без депозитов и NULL-страны; (3) посчитает DAU через `COUNT(*)`, если не уточнить, что нужны уникальные люди; (4) не отличит когорту по регистрации от когорты по первому действию, если ты не зафиксируешь определение «дня 0». Вывод: синтаксис и каркас — AI; определения метрик и логику JOIN/NULL — ты.

## Критерий готовности

- [ ] `uv run gen.py` создаёт три Parquet-файла (`players=4000 deposits=4954 bets=37157`), в `players` есть NULL-страны и игроки без депозитов.
- [ ] DAU считается через `COUNT(DISTINCT player_id)`, и видно расхождение с `COUNT(*)`.
- [ ] Топ-15 депозиторов отфильтрован через HAVING по сумме > 300.
- [ ] FTD через `ROW_NUMBER()` даёт ровно одну строку на депозитора (1747).
- [ ] Нарастающий GGR в последней строке совпадает с `SUM(stake - payout)` по всей таблице (~176670).
- [ ] CASE-сегментация даёт whale/mid/minnow с долей крупных депозитов; доля выручки whale посчитана (~41%).
- [ ] LEFT JOIN показывает страну 'unknown' и сохраняет всех 4000 игроков.
- [ ] Retention-таблица D1/D7/D30 по неделям когорт считается одним запросом, сумма когорт = числу игроков со ставками.
- [ ] Все 7 запросов сохранены в `m4_queries.sql` и отрабатывают без ошибок.

## Бизнес-вывод

Семь запросов — это ещё не ответ продакту. Сырые таблицы DAU и retention для стейкхолдера бесполезны: он принимает решения, а не читает SQL. Переведи технический результат в одну страницу языком решений.

- [ ] **Рекомендация одной фразой:** что делать с бюджетом по каналам (срезать долю каналов с D7 ниже порога, усилить лучшие) — на основании retention-когорт из Шага 7 и гео-разбивки из Шага 6.
- [ ] **Эффект в деньгах и метриках:** на сколько процентных пунктов отличается D7 между лучшим и худшим каналом, какая доля выручки приходится на whale-сегмент (~41%, Шаг 5) — и что это значит для перераспределения 1,5 млн ₽/мес и решения по VIP-менеджеру.
- [ ] **Риски и допущения:** синтетика даёт пологую кривую retention, на реальных данных D1 падает резко; NULL-страны и игроки без депозитов искажают охват, если их не учесть (Шаг 6); короткое окно наблюдения по молодым когортам делает D30 неустойчивым.
- [ ] **Следующий шаг:** какие данные добрать (атрибуция канала по FTD, более длинное окно для D30), нужен ли A/B по каналу до того, как резать бюджет.
- [ ] **Подача стейкхолдеру:** один экран с выводом и цифрой сверху, а не таблица из семи запросов. Говори «канал X удерживает игроков на 40% хуже, предлагаю перевести часть бюджета на Y», а не «D7 равен 12%».

## Развитие

- Замени self-join в Шаге 7 на массивы DuckDB (`list` активных дней) и сравни читаемость — DuckDB-специфичный приём из M4.14.
- Добавь когортный LTV: накопленный депозит по неделе FTD через `SUM(amount) OVER (PARTITION BY cohort_week ORDER BY week_offset)`.
- Построй тепловую карту retention в matplotlib (cohort_week по строкам, лаг по столбцам) — закрой мини-проект модуля.
- Подними те же запросы против PostgreSQL через SQLAlchemy и зафиксируй, какие функции дат пришлось править (`bet_ts::date` vs `date_trunc`), отрабатывая переносимость диалектов из M4.14.
- Прогони `EXPLAIN ANALYZE` на запросе Шага 7, найди самый дорогой шаг и проверь гипотезу про индекс по `player_id` из M4.12-M4.13.

## Что ты закрепил

Ты прошёл всё ядро модуля M4 на одном связном датасете: от обрезки timestamp и `COUNT(DISTINCT)` для DAU — через агрегаты с HAVING, оконный `ROW_NUMBER` для FTD и нарастающий `SUM OVER` для GGR — к условной CASE-сегментации, LEFT JOIN с NULL-логикой и retention-когортам через self-join. Каждый шаг закрывал конкретную ловушку модуля (DISTINCT vs COUNT(*), WHERE vs HAVING, окно vs GROUP BY, INNER vs LEFT, целочисленное деление, сравнение с NULL), а критерии-`assert` доказали корректность числами. Главное, что ты унёс: SQL-метрика — это не запрос ради запроса, а воспроизводимый источник истины для продуктового решения; и определение метрики (кто такой «активный», что такое «день 0 когорты», по чему считать долю) важнее синтаксиса — именно в определениях прячутся ошибки, которые ни один `assert` за тебя не поймает.
