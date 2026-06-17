# Воркшоп M2 · Чистый Python на игровых данных

<span class="lecture-meta">Воркшоп к модулю M2 · ориентир 4-6 ч</span>

## Что отрабатываем

Этот воркшоп закрепляет руками то, что в теории модуля было разобрано на интуиции: модель данных (имена-ярлыки, mutable/immutable), выбор коллекции под задачу по асимптотике (list / tuple / dict / set), comprehensions и генераторы с ленивыми вычислениями, функции с `*args/**kwargs` и ловушкой изменяемого дефолта, декоратор-обёртка, ООП через класс `Player` с dunder-методами и наследованием, обработка исключений конкретного типа.

Никакого pandas. Только чистый Python и стандартная библиотека (`collections`, `itertools`, `datetime`, `functools`, `dataclasses`, `json`). Это принципиально: pandas — надстройка, и пока фундамент шаткий, надстройка рушится.

Артефакт на выходе — один файл `game_core.py`: модуль с решениями всех задач плюс блок мини-тестов на `assert`, который проходит при запуске `uv run python game_core.py`. Это твой первый «питоничный» модуль, который не стыдно показать на ревью.

## Бизнес-кейс

Этот модуль кажется учебным, но руками ты собираешь то, на чём в реальной команде стоят продуктовые отчёты: retention, объём депозитов, LTV игрока, топ-игры. Навык писать такой код корректно и без утечек памяти — это не про синтаксис, это про то, можно ли доверять цифре, по которой примут решение на деньги. Один баг новичка из этого модуля (ярлык вместо копии, дубли в retention, изменяемый дефолт, загрузка всего лога в память) — и стейкхолдер увидит неверное число, не зная об этом.

!!! example "Ситуация"

    Ты — аналитик в команде iGaming-продукта. Продакт-оунер просит собрать переиспользуемый аналитический core-модуль, на котором будут стоять еженедельные отчёты по retention и LTV. Причина срочная: текущие ad-hoc скрипты дают разные числа от запуска к запуску, и на прошлой неделе это уже стоило команде ошибочного решения.

    - Бизнес-проблема: из-за дублей игроков в подсчёте D1 retention показали как ~30% вместо реальных ~22%. На завышенной цифре продакт решил, что новый онбординг «работает», и оставил его как есть — хотя удержание просело. Иллюстративно это около 1.5 млн ₽ в месяц недополученной выручки, которую списали на сезонность.
    - Что зависит от твоего ответа: пойдёт ли core-модуль в основу отчётности и можно ли по нему принимать решения о бюджете на удержание (порядок — сотни тысяч ₽ в месяц). Плюс отдельный риск: лог событий растёт до десятков ГБ, и наивная реализация «упадёт по памяти» в самый нужный момент.
    - Ограничение: один файл без внешних зависимостей кроме стандартной библиотеки, проходящий `ruff` и `mypy`; срок — в рамках одного спринта, данные — синтетический поток событий (депозиты, ставки, выводы).

## Данные

Данные синтетические, генерируются прямо в модуле с фиксированным seed — запустится у любого без скачиваний. Имитируем поток событий гемблинг-платформы: депозиты, ставки, выводы. Формат — список словарей (как распарсенный JSONL-лог).

```python
import random
from datetime import datetime, timedelta

def make_events(n_players: int = 200, days: int = 40, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    games = ["slots", "roulette", "blackjack", "crash", "poker"]
    markets = ["RU", "UZ", "KZ", "TR"]
    start = datetime(2026, 6, 1)
    events: list[dict] = []
    for pid in range(1, n_players + 1):
        market = rng.choice(markets)
        reg_day = rng.randint(0, days - 1)
        n_ev = rng.randint(1, 60)
        for _ in range(n_ev):
            day = reg_day + rng.randint(0, days - 1 - reg_day)
            ts = start + timedelta(days=day, seconds=rng.randint(0, 86399))
            kind = rng.choices(["deposit", "bet", "withdrawal"], weights=[3, 6, 1])[0]
            amount = round(rng.uniform(5, 500), 2)
            ev = {"player_id": pid, "market": market, "type": kind,
                  "game": rng.choice(games) if kind == "bet" else None,
                  "amount": amount, "ts": ts.isoformat()}
            events.append(ev)
    rng.shuffle(events)
    return events
```

При `seed=42` всегда получается один и тот же поток — мини-тесты опираются на конкретные числа. Если меняешь параметры генератора, числа в `assert` тоже поменяются: пересчитай их печатью перед тем, как зашивать в тест.

## Ход работы

Создай проект и открой пустой `game_core.py`. Все функции пишем в нём.

```bash
uv init m2-workshop
cd m2-workshop
uv add --dev ruff mypy
# ruff/mypy понадобятся в конце для проверки стиля и типов
```

### Шаг 1: модель данных и копирование

Зачем. Отрабатываем главную мысль модуля: `b = a` для списка — это второй ярлык, не копия. Половина багов новичка отсюда. Нужно почувствовать разницу руками, а не на словах.

```python
def copy_demo() -> tuple[bool, bool]:
    a = [100, 250, 500]
    alias = a            # тот же объект
    real = a[:]          # поверхностная копия (новый объект)
    alias.append(999)
    return (a == [100, 250, 500, 999], real == [100, 250, 500])
```

Что получилось. Функция возвращает `(True, True)`: мутация через `alias` отразилась на `a` (один объект), а `real` осталась прежней. Запомни идиому копирования `a[:]` или `a.copy()`. Для `is` помни правило: только `None`, `True`, `False`.

!!! question "Проверь себя"

    1. Почему `alias.append(999)` изменил `a`, но не `real`?
    2. Какой идиомой ещё можно сделать поверхностную копию списка кроме `a[:]`?
    3. Можно ли использовать `is` для проверки `a == [100, 250, 500, 999]`? Почему нет?

??? success "Ответы"

    1. `alias` и `a` — два ярлыка на один объект, изменение через любой виден через оба. `real` создан срезом `a[:]` — это отдельный объект с тем же содержимым.
    2. `a.copy()` (или `list(a)`). Для вложенных структур нужна `copy.deepcopy`.
    3. Нет. `is` сравнивает идентичность (тот же ли объект в памяти), а нам нужно сравнить содержимое — это `==`. На списках `is` почти всегда `False`, даже при равном содержимом.

### Шаг 2: коллекции и асимптотика на retention

Зачем. Отрабатываем правило выбора коллекции и эффект `set` против `list` на проверке членства. Это самый частый прирост скорости в продуктовой аналитике.

```python
def day_players(events: list[dict], day_index: int) -> set[int]:
    start = datetime(2026, 6, 1)
    target = (start + timedelta(days=day_index)).date()
    return {e["player_id"] for e in events
            if datetime.fromisoformat(e["ts"]).date() == target}

def d1_retention(events: list[dict], day_index: int) -> float:
    today = day_players(events, day_index)
    tomorrow = day_players(events, day_index + 1)
    if not today:
        return 0.0
    retained = today & tomorrow          # пересечение множеств за O(1) на элемент
    return len(retained) / len(today)
```

Что получилось. `day_players` через set-comprehension сразу дедуплицирует игроков за день. `d1_retention` использует пересечение `&` вместо вложенного цикла `for p in today: if p in tomorrow` — из $O(n \cdot m)$ получился $O(n + m)$. Идиома `if not today` вместо `len(today) == 0` — питоничная проверка пустоты.

### Шаг 3: comprehensions — list, dict, set

Зачем. Закрепляем визитную карточку питоничного кода. Три вида comprehension под три типичные задачи аналитика.

```python
from collections import defaultdict

def comprehension_stats(events: list[dict]) -> dict:
    deposits = [e["amount"] for e in events if e["type"] == "deposit"]   # list
    markets = {e["market"] for e in events}                              # set
    by_market = defaultdict(float)
    for e in events:
        if e["type"] == "deposit":
            by_market[e["market"]] += e["amount"]
    dep_by_market = {m: round(v, 2) for m, v in by_market.items()}       # dict comp
    return {"n_deposits": len(deposits),
            "markets": markets,
            "dep_by_market": dep_by_market}
```

Что получилось. Список сумм депозитов, множество уникальных рынков, словарь рынок→суммарный депозит. `defaultdict(float)` убирает ручную проверку «есть ли ключ» перед `+=`. Финальный dict-comprehension только округляет — читается одним взглядом.

!!! question "Проверь себя"

    1. Какой тип данных вернёт `{e["market"] for e in events}` и какое его главное свойство здесь полезно?
    2. Что в коде делает `defaultdict(float)`, от какой рутины он избавляет?
    3. Почему для `dep_by_market` взят dict, а не list пар `(market, amount)`?

??? success "Ответы"

    1. `set` — множество уникальных рынков. Дедупликация бесплатна: повторяющиеся значения схлопываются автоматически.
    2. Даёт автозначение `0.0` для отсутствующего ключа, поэтому `by_market[e["market"]] += e["amount"]` не падает с `KeyError` на первом обращении — не нужно писать `if key not in d`.
    3. Поиск/обновление по рынку нужен за $O(1)$ и нужна уникальность ключа. dict — хеш-таблица, list пар требовал бы линейного поиска и допускал дубли.

### Шаг 4: генератор для ленивого чтения

Зачем. Самая важная тема модуля для больших данных. Функция с `yield` отдаёт по одному событию и не держит весь лог в памяти. Имитируем чтение JSONL-потока.

```python
import json
from typing import Iterator, Iterable

def to_jsonl_lines(events: list[dict]) -> list[str]:
    return [json.dumps(e, ensure_ascii=False) for e in events]

def read_deposits(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        e = json.loads(line)
        if e["type"] == "deposit":
            yield e                      # отдаём по одной записи, не строим список

def total_deposit_volume(lines: Iterable[str]) -> float:
    return round(sum(e["amount"] for e in read_deposits(lines)), 2)
```

Что получилось. `read_deposits` — генератор: при вызове не исполняется, возвращает объект-генератор, выдаёт депозиты по одному. `total_deposit_volume` суммирует через generator expression `sum(... for ...)` — без материализации списка. На логе в 50 ГБ это разница между «упало по памяти» и «прошло потоком». В реальном файле `lines` был бы `path.open()`, а не список — генератор не знает разницы.

!!! question "Проверь себя"

    1. Что вернёт вызов `read_deposits(lines)` — список депозитов или что-то другое?
    2. Почему `sum(e["amount"] for e in ...)` лучше, чем `sum([e["amount"] for e in ...])` на большом логе?
    3. Если заменить `lines` на открытый файл, что изменится в потреблении памяти?

??? success "Ответы"

    1. Объект-генератор. Тело не исполнится, пока кто-то не начнёт итерировать (`for`, `sum`, `next`).
    2. Вариант с круглыми скобками — generator expression, считает потоком за константную память. Квадратные скобки сначала материализуют весь список депозитов в памяти.
    3. Почти ничего: генератор всё равно держит одну запись за раз. Файл читается построчно лениво, в память не грузится целиком — память не зависит от размера файла.

### Шаг 5: декоратор-таймер

Зачем. Декоратор — обёртка, добавляющая поведение без правки функции. Пишем `@timed`, замеряющий время, с обязательным `functools.wraps`.

```python
import functools
import time

def timed(func):
    @functools.wraps(func)               # сохраняет имя и docstring оригинала
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_seconds = time.perf_counter() - start
        return result
    wrapper.last_seconds = 0.0
    return wrapper

@timed
def aggregate_volume(lines: list[str]) -> float:
    return total_deposit_volume(lines)
```

Что получилось. `@timed` оборачивает `aggregate_volume`: результат тот же, но появился атрибут `wrapper.last_seconds` с длительностью последнего вызова (вместо `print`, чтобы проверять тестом). `*args, **kwargs` в `wrapper` пропускают любые аргументы насквозь. `functools.wraps` сохраняет `__name__` — без него `aggregate_volume.__name__` стало бы `"wrapper"`.

### Шаг 6: класс Player и наследование

Зачем. ООП там, где оно оправдано: объект со состоянием и поведением плюс dunder-методы. Класс `Player` с `__init__`, `__repr__`, методом `ltv()`, и наследник `VIPPlayer`.

```python
class Player:
    def __init__(self, player_id: int, deposits: float, withdrawals: float = 0.0):
        self.player_id = player_id
        self.deposits = deposits
        self.withdrawals = withdrawals

    def ltv(self) -> float:
        return self.deposits - self.withdrawals

    def __repr__(self) -> str:
        return f"Player(id={self.player_id}, ltv={self.ltv():.2f})"


class VIPPlayer(Player):
    def __init__(self, player_id, deposits, withdrawals=0.0, manager="—"):
        super().__init__(player_id, deposits, withdrawals)
        self.manager = manager

    def ltv(self) -> float:              # VIP-бонус 10%
        return super().ltv() * 1.1


def build_players(events: list[dict]) -> dict[int, Player]:
    dep = defaultdict(float)
    wd = defaultdict(float)
    for e in events:
        if e["type"] == "deposit":
            dep[e["player_id"]] += e["amount"]
        elif e["type"] == "withdrawal":
            wd[e["player_id"]] += e["amount"]
    return {pid: Player(pid, round(dep[pid], 2), round(wd[pid], 2)) for pid in dep}
```

Что получилось. `Player` инкапсулирует депозиты/выводы и считает LTV. `VIPPlayer` переопределяет `ltv()` через `super().ltv() * 1.1` — наследование с расширением. `build_players` собирает словарь `id → Player` из потока событий. `__repr__` делает печать объекта читаемой.

!!! question "Проверь себя"

    1. Что делает `super().ltv()` в `VIPPlayer` и почему это лучше, чем переписать формулу заново?
    2. Зачем нужен `__repr__`, если LTV и так можно получить методом?
    3. Почему `withdrawals` задан дефолтом `0.0`, а не пустым списком — связь с ловушкой модуля?

??? success "Ответы"

    1. Вызывает `ltv()` родителя (базовую формулу `deposits - withdrawals`), затем VIP добавляет бонус. Не дублируем логику: если базовая формула изменится, VIP подхватит автоматически.
    2. `__repr__` задаёт текстовое представление объекта при печати и в отладке — видишь `Player(id=5, ltv=320.00)` вместо `<...object at 0x...>`. Это для людей и логов, а не для расчётов.
    3. `0.0` — неизменяемое число, безопасный дефолт. Изменяемый дефолт (`[]`, `{}`) вычисляется один раз при определении и живёт между вызовами — классический баг. Скаляр такой проблемы не создаёт.

### Шаг 7: обработка исключений и финальная агрегация

Зачем. Ловим конкретный тип исключения, а не голый `except`. Считаем безопасное отношение и собираем сводку, устойчивую к битым данным.

```python
def safe_ratio(payout, stake) -> float:
    try:
        return payout / stake
    except ZeroDivisionError:
        return 0.0
    except TypeError:
        raise ValueError(f"нечисловые данные: {payout!r}, {stake!r}")


def summary(events: list[dict]) -> dict:
    players = build_players(events)
    top = max(players.values(), key=lambda p: p.ltv())
    games = Counter(e["game"] for e in events if e["type"] == "bet")
    return {"n_players": len(players),
            "top_player_id": top.player_id,
            "top_ltv": round(top.ltv(), 2),
            "top3_games": games.most_common(3)}
```

Что получилось. `safe_ratio` отлавливает деление на ноль осмысленным дефолтом, а нечисловой вход превращает в говорящий `ValueError` (а не глотает молча). `summary` использует `max` с `key=lambda` для топ-игрока по LTV и `Counter.most_common(3)` для топ-игр. Не забудь `from collections import Counter` вверху модуля.

### Шаг 8: мини-тесты и проверка стиля

Зачем. Артефакт обязан сам себя проверять. Блок `if __name__ == "__main__"` с `assert` — простейший тест без зависимостей. Числа взяты из прогона при `seed=42`.

```python
if __name__ == "__main__":
    ev = make_events()
    lines = to_jsonl_lines(ev)

    assert copy_demo() == (True, True)
    assert 0.0 <= d1_retention(ev, 5) <= 1.0
    vol = total_deposit_volume(lines)
    assert vol == aggregate_volume(lines)
    assert aggregate_volume.last_seconds >= 0.0

    p = Player(1, 300.0, 50.0)
    vip = VIPPlayer(1, 300.0, 50.0, manager="Anna")
    assert p.ltv() == 250.0
    assert round(vip.ltv(), 2) == 275.0
    assert "Player(id=1" in repr(p)

    assert safe_ratio(10, 0) == 0.0
    try:
        safe_ratio("x", 2)
    except ValueError:
        pass
    else:
        raise AssertionError("ожидался ValueError")

    s = summary(ev)
    assert s["n_players"] > 0
    assert len(s["top3_games"]) == 3
    print("OK:", s)
```

Запусти и прогони линтер с типами.

```bash
uv run python game_core.py
uv run ruff check game_core.py
uv run mypy game_core.py
```

Что получилось. При успехе печатается `OK: {...}` и ни один `assert` не падает. `ruff` ловит нарушения PEP 8 и идиом, `mypy` проверяет аннотации. Если `mypy` ругается на `defaultdict` или `Iterator` — добавь импорты типов и аннотации возврата, это и есть тренировка контракта на границах функций.

## Критерий готовности

- [ ] `game_core.py` запускается через `uv run python game_core.py` и печатает `OK: {...}`
- [ ] Все `assert` в `__main__` проходят, ValueError проверяется отдельно (через `else: raise`)
- [ ] `copy_demo` демонстрирует разницу ярлыка и копии; объясняешь её словами
- [ ] retention считается через `&` множеств, а не через вложенный цикл с `in` по списку
- [ ] есть три разных comprehension (list, set, dict) и `defaultdict` без ручной проверки ключа
- [ ] `read_deposits` — генератор с `yield`, объём считается через `sum(... for ...)` без материализации
- [ ] `@timed` использует `functools.wraps` и `*args/**kwargs`
- [ ] `Player`/`VIPPlayer` с `__init__`, `__repr__`, `ltv()` и `super()` в переопределении
- [ ] исключения ловятся по конкретному типу, нет голого `except: pass`
- [ ] нет изменяемых дефолтных аргументов; копии делаются явно
- [ ] `ruff check` и `mypy` проходят без ошибок

## Бизнес-вывод

Код запустился и тесты прошли — это половина работы. Вторая половина: перевести технический результат в решение для продакт-оунера, который не читает Python. Сформулируй ответ на языке решений, а не функций и асимптотики.

- [ ] Рекомендация: что делать. Например — «переводим отчётность по retention/LTV на этот core-модуль, старые ad-hoc скрипты выводим из обращения; решение по онбордингу пересматриваем на корректной цифре D1 ~22%».
- [ ] Эффект в деньгах или метриках: «корректный D1 (22% вместо 30%) меняет вывод об онбординге; на кону ~1.5 млн ₽/мес недополученной выручки, которую раньше списывали на сезонность».
- [ ] Риски и допущения: числа считаны на синтетическом потоке с фиксированным seed; на проде нужна валидация на реальном логе; дедупликация игроков и потоковое чтение убирают два конкретных класса ошибок, но не гарантируют качество входных данных.
- [ ] Следующий шаг: подключить модуль к реальному JSONL-логу через генератор (без загрузки в память), прогнать на полном объёме, сверить retention/LTV с текущей витриной и зафиксировать расхождения.
- [ ] Как подать стейкхолдеру: одно предложение в начале — «прежний отчёт завышал удержание из-за дублей, реальная цифра ниже, решение по онбордингу стоит пересмотреть»; цифры и метод — в приложении, не в первом экране.

## Развитие

1. Декоратор с аргументом `@retry(times=3)`: три уровня вложенных функций, повторяет вызов при исключении до `times` раз и пробрасывает последнюю ошибку. Оберни им `total_deposit_volume`, имитируя случайный сбой парсинга.
2. Контекстный менеджер `timer(label)` через `@contextlib.contextmanager` с `try/finally`, чтобы замерять блоки кода, а не только функции. Сравни с декоратором `@timed` — когда что удобнее.
3. `@dataclass(frozen=True)` для ключа когорты `Cohort(market, month)` и расчёт retention по когортам месяца регистрации: `dict[Cohort, dict[int, float]]`. Проверь, что `Cohort` кладётся в `set` и работает ключом dict.
4. Конвейер генераторов `parse → filter_market → amounts → accumulate`: считай нарастающий объём депозитов по рынку через `itertools.accumulate`, докажи замером (`tracemalloc`), что лог не материализуется целиком. Сравни время наивной версии (списки + `in`) и оптимизированной (множества) на `make_events(n_players=5000)`.
