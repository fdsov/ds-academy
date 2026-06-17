# Воркшоп M1 · Скелет воспроизводимого DS-проекта с нуля

<span class="lecture-meta">Воркшоп к модулю M1 · ориентир 3-5 ч</span>

## Что отрабатываем

Модуль M1 объяснил пять слоёв воспроизводимости и инструменты под каждый: код — git, зависимости — `uv` + lockfile, среда исполнения — виртуальное окружение, данные — фиксация версии, случайность — random seed. Теперь ты закрываешь все пять слоёв руками на одном живом проекте.

Конкретно отрабатываем: `uv init` и `uv add` (M1.6), различие `pyproject.toml` против `uv.lock` (M1.6), стандартную структуру `data/src/notebooks/tests` с неизменяемым `data/raw/` (M1.10), git-цикл commit/branch (M1.7), `.env` и `.gitignore` для секретов (M1.11), перенос логики из ноутбука в `src/` (M1.10) и главный тест зрелости — проверку воспроизводимости через удаление `.venv/` и `uv sync` (M1.12).

Артефакт на выходе: репозиторий `churn-starter`, который заводится у любого человека с трёх команд (clone, `uv sync`, `uv run pytest`) и даёт зелёные тесты. Это тот самый практический критерий из M1.3.

!!! tip "Перед стартом"

    Нужен установленный `uv`. Проверь: `uv --version`. Если нет — `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) и перезапусти терминал. Всё остальное — Python, нужные пакеты — поставит сам `uv`. Системный Python не трогаем (M1.4).

## Данные

Никаких внешних загрузок: генерируем синтетические ставки игроков детерминированно с фиксированным seed. Это сразу закрывает слой «случайность» из M1.3 — у любого, кто запустит скрипт, получится байт-в-байт тот же CSV. Этот файл станет неизменяемым сырьём в `data/raw/`.

```python
# scripts/make_data.py — генератор сырых данных, кладётся ВНЕ data/raw
import numpy as np
import polars as pl

SEED = 42

def generate_bets(n_players: int = 500, n_bets: int = 20_000) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    player_id = rng.integers(1, n_players + 1, size=n_bets)
    day_offset = rng.integers(0, 180, size=n_bets)
    bet_ts = (np.datetime64("2026-01-01") + day_offset.astype("timedelta64[D]"))
    stake = np.round(rng.gamma(shape=2.0, scale=50.0, size=n_bets), 2)
    is_first_deposit = rng.random(n_bets) < 0.03
    return pl.DataFrame({
        "player_id": player_id,
        "bet_ts": bet_ts.astype(str),
        "stake": stake,
        "is_first_deposit": is_first_deposit,
    }).sort(["player_id", "bet_ts"])

if __name__ == "__main__":
    df = generate_bets()
    df.write_csv("data/raw/bets.csv")
    print(f"wrote {df.height} rows, hash check:", df["stake"].sum())
```

Фиксированный seed плюs детерминированный генератор означают: повторный запуск даёт идентичный `bets.csv`. Контрольная сумма `stake` в конце — твой быстрый способ убедиться, что данные те же, что и у соседа.

## Ход работы

### Шаг 1: Инициализируем проект через uv

Зачем: отрабатываем `uv init` (M1.6) — одна команда ставит нужный Python, создаёт `.venv`, `pyproject.toml` и `git`-репозиторий. Это слои «среда исполнения» и зачаток «зависимостей».

```bash
uv init --python 3.13 churn-starter
cd churn-starter
uv add polars scikit-learn marimo numpy
uv add --dev pytest ruff
```

Берём 3.13 как безопасный дефолт (M1.4: не хватаем свежайший релиз вслепую). `uv add` не просто ставит пакеты — он пишет их в `pyproject.toml` и фиксирует точное дерево версий в `uv.lock`.

Что получилось: появились `pyproject.toml`, `uv.lock`, `.venv/` и `.git/`. Открой `uv.lock` и найди строку с polars — там точная версия и хеш. В `pyproject.toml` polars записан как constraint (`>=...`). Это и есть разница «намерение против решения» из M1.6.

### Шаг 2: Разворачиваем структуру DS-проекта

Зачем: отрабатываем стандартную структуру из M1.10 с неизменяемым `data/raw/`. Каждой вещи — своё место.

```bash
mkdir -p data/raw data/interim data/processed src notebooks tests scripts reports
touch src/__init__.py tests/__init__.py
```

Теперь `.gitignore` — чтобы git игнорировал окружение, секреты и данные (M1.10, M1.11). `uv init` уже создал базовый `.gitignore`; дописываем нужное.

```bash
printf '%s\n' '.venv/' '.env' 'data/' 'models/' '__pycache__/' '*.pyc' >> .gitignore
```

Что получилось: дерево каталогов на месте, тяжёлое и секретное исключено из git. Проверишь на шаге 5.

### Шаг 3: Секреты в .env и шаблон .env.example

Зачем: слой безопасности из M1.11. Секрет живёт в `.env` (git его не видит), а в репозиторий идёт только шаблон без значений.

```bash
printf 'YOHOHO_API_KEY=\n' > .env.example
printf 'YOHOHO_API_KEY=test123\n' > .env
uv add python-dotenv
```

Код чтения секрета — упадёт громко, если переменной нет, и это правильно (M1.11):

```python
# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key() -> str:
    return os.environ["YOHOHO_API_KEY"]
```

Что получилось: `.env` с реальным значением на диске, `.env.example` для коммита. Git не должен видеть `.env` — проверим на шаге 5.

!!! question "Проверь себя"

    1. Какой из двух файлов — `.env` или `.env.example` — попадает в git и почему?
    2. Зачем `os.environ["KEY"]`, а не `os.environ.get("KEY")` при чтении обязательного секрета?
    3. Какой слой воспроизводимости из M1.3 закрыл фиксированный seed в генераторе данных?

??? success "Ответы"

    1. В git идёт `.env.example` — шаблон без значений, показывающий нужные переменные. `.env` с реальным секретом внесён в `.gitignore`, иначе ключ утечёт в историю и будет скомпрометирован навсегда.
    2. `os.environ["KEY"]` падает с ошибкой, если переменной нет — проблема видна сразу при старте. `.get()` вернул бы `None`, и баг всплыл бы позже в неожиданном месте.
    3. Слой случайности (random seed). Любой запуск генератора даёт идентичный `bets.csv`.

### Шаг 4: Генерируем данные и выносим логику в src/

Зачем: отрабатываем границу «ноутбук против `src/`» из M1.10 — переиспользуемая логика живёт функцией в `src/`, ноутбук её только импортирует. И заодно защиту от утечки будущего (cutoff), как в примере модуля.

Сначала кладём генератор из секции «Данные» в `scripts/make_data.py` и запускаем:

```bash
uv run python scripts/make_data.py
```

`data/raw/bets.csv` создан — это read-only сырьё, больше его не трогаем. Теперь функция признаков в `src/`:

```python
# src/features.py
import polars as pl

def build_rfm_features(bets: pl.DataFrame, cutoff: str) -> pl.DataFrame:
    """RFM-признаки игроков строго до даты cutoff (защита от утечки будущего)."""
    window = bets.filter(pl.col("bet_ts") < pl.lit(cutoff).str.to_datetime())
    return window.group_by("player_id").agg(
        recency_days=(pl.lit(cutoff).str.to_datetime() - pl.col("bet_ts").max()).dt.total_days(),
        frequency=pl.len(),
        monetary=pl.col("stake").sum(),
    ).sort("player_id")
```

Что получилось: сырьё зафиксировано в `data/raw/`, логика — в импортируемом модуле, а не запертой в ячейке ноутбука. Это и есть признак зрелости из M1.10.

### Шаг 5: Marimo-ноутбук для разведки

Зачем: M1.8 — реактивный ноутбук без проблемы скрытого состояния. Ноутбук импортирует функцию из `src/`, а не дублирует расчёт.

```python
# notebooks/explore.py — открывается через: uv run marimo edit notebooks/explore.py
import marimo as mo
import polars as pl
from src.features import build_rfm_features

bets = pl.read_csv("data/raw/bets.csv").with_columns(
    pl.col("bet_ts").str.to_datetime()
)

rfm = build_rfm_features(bets, cutoff="2026-04-01")
avg_stake = bets["stake"].mean()
```

Запусти и покрути: `uv run marimo edit notebooks/explore.py`. Изменишь `cutoff` — ячейка с `rfm` пересчитается сама, потому что marimo видит зависимость через DAG (M1.8). Файл — обычный `.py`, значит git покажет человеческий дифф.

Что получилось: разведка данных, где состояние на экране всегда соответствует коду. Никакого «запустил ячейки вразнобой».

### Шаг 6: Тест на защиту от утечки

Зачем: M1.10/M1.12 — даже один pytest-тест ловит регрессию и доказывает, что cutoff работает.

```python
# tests/test_features.py
import polars as pl
from src.features import build_rfm_features

def test_cutoff_excludes_future_bets():
    bets = pl.DataFrame({
        "player_id": [1, 1],
        "bet_ts": ["2026-01-01", "2026-12-31"],
        "stake": [100.0, 999.0],
    }).with_columns(pl.col("bet_ts").str.to_datetime())
    out = build_rfm_features(bets, cutoff="2026-06-01")
    assert out["monetary"][0] == 100.0  # декабрьская ставка не должна попасть
```

```bash
uv run pytest
uvx ruff check .
```

Что получилось: зелёный тест и чистый линтер. `uvx` запускает ruff во временном окружении, не засоряя проект (M1.6).

### Шаг 7: Версионируем — git commit, ветка, лог экспериментов

Зачем: слой «код» из M1.3 и git-цикл из M1.7. Коммитим именно `pyproject.toml` + `uv.lock` (оба!), код и заметки — но не `.venv/`, `.env`, `data/`.

Сначала проверь, что секретное и тяжёлое реально игнорируется:

```bash
git status
git check-ignore .env .venv data/raw/bets.csv
```

`git check-ignore` должен напечатать все три пути — значит они исключены. Если `.env` виден в `git status` как untracked — `.gitignore` настроен неверно, чини до коммита (M1.11: предотвратить дешевле, чем отзывать ключ).

Заведём лабораторный журнал из M1.7 и закоммитим:

```bash
printf '# Эксперименты\n\n- cutoff=2026-04-01, RFM по %d игрокам. baseline.\n' 500 > EXPERIMENTS.md

git add pyproject.toml uv.lock src tests scripts notebooks .gitignore .env.example EXPERIMENTS.md README.md
git commit -m "Bootstrap churn-starter: polars + marimo, reproducible env, leak-safe RFM"

git switch -c experiment-frequency-cap
# здесь правил бы гиперпараметры / признаки
git switch main
git log --oneline --graph --all
```

Что получилось: осмысленный коммит (сообщение объясняет «почему», не «что» — M1.7), отдельная ветка под эксперимент, журнал рядом с кодом. Конкретная версия кода, данных и вывода связаны одним коммитом.

!!! question "Проверь себя"

    1. Почему в git нужны оба — `pyproject.toml` и `uv.lock`, и что сломается, если закоммитить только первый?
    2. Что показывает `git check-ignore .env` и почему это важно сделать ДО первого коммита?
    3. Чем плох commit-месседж `fix` и как звучит хороший по логике M1.7?

??? success "Ответы"

    1. `pyproject.toml` декларирует намерение («polars новее X»), `uv.lock` фиксирует точное разрешённое дерево версий с хешами. Без lockfile на чужой машине разрешатся другие версии, и результат поедет — воспроизводимости нет.
    2. Печатает путь, если он игнорируется git. Важно до коммита, потому что утёкший в историю секрет скомпрометирован навсегда — его придётся отзывать и перевыпускать, а не просто удалять из файла.
    3. `fix` не объясняет мотивацию, а дифф и так показывает, что менялось. Хорошо — «Fix LTV leak: exclude deposits after cohort cutoff date»: видно «почему».

### Шаг 8: Главная проверка — воспроизводимость с нуля

Зачем: это критерий готовности из M1.3 и M1.12. Удаляем окружение целиком и пересобираем из lockfile — если тесты зелёные, проект воспроизводим.

```bash
rm -rf .venv
uv sync
uv run python scripts/make_data.py
uv run pytest
```

`uv sync` ставит точь-в-точь версии из `uv.lock` за секунды. Данные пересоздаются детерминированно. Тесты зелёные.

Что получилось: ты симулировал «новый человек на чистой машине». Три команды — и всё работает без вопросов к автору. Это и есть воспроизводимость, ради которой существует весь модуль M1.

!!! question "Проверь себя"

    1. Почему удалить `.venv/` и сделать `uv sync` — корректная проверка воспроизводимости?
    2. Что в этом проекте закрывает каждый из пяти слоёв M1.3?

??? success "Ответы"

    1. `.venv/` — локальная папка, не источник истины; источник — `uv.lock`. Пересоздание из lockfile воспроизводит окружение того, кто коммитил, байт-в-байт. Если после этого тесты падают — значит воспроизводимости не было, просто «работало у меня».
    2. Код — git (шаг 7); зависимости — `uv.lock` (шаги 1, 8); среда исполнения — `.venv` через `uv` с Python 3.13 (шаг 1); данные — детерминированный генератор в `data/raw/` (шаг 4); случайность — `SEED=42` (секция «Данные»).

## Критерий готовности

- [ ] `uv init` создал проект, `uv add` записал зависимости в `pyproject.toml` и `uv.lock`
- [ ] Развёрнута структура `data/{raw,interim,processed}`, `src/`, `notebooks/`, `tests/`, `scripts/`
- [ ] `data/raw/bets.csv` сгенерирован детерминированно (фиксированный seed) и неизменяем
- [ ] `.gitignore` исключает `.venv/`, `.env`, `data/`; `git check-ignore` это подтверждает
- [ ] `.env` с секретом на диске, в git закоммичен только `.env.example`
- [ ] Логика признаков вынесена в `src/features.py`, ноутбук её импортирует, а не дублирует
- [ ] Marimo-ноутбук открывается и реактивно пересчитывается при смене `cutoff`
- [ ] `uv run pytest` зелёный, `uvx ruff check .` без ошибок
- [ ] Есть осмысленный коммит и отдельная ветка под эксперимент; `EXPERIMENTS.md` рядом с кодом
- [ ] После `rm -rf .venv && uv sync && uv run pytest` всё зелёное — воспроизводимость доказана

## Развитие

- Запушь `churn-starter` на GitHub, дай ссылку коллеге и проверь, что у него заводится с тех же трёх команд (clone, `uv sync`, `uv run pytest`). Это переход от reproducibility к командной проверке.
- Добавь `README.md` с ровно тремя командами запуска и одним абзацем «что это» — так, чтобы новый человек не задал ни одного вопроса.
- Возьми реальный «грязный» Jupyter-ноутбук с GitHub, сделай Restart Kernel and Run All, поймай падение и опиши в `EXPERIMENTS.md`, какая переменная «жила» в памяти без кода — проблема скрытого состояния вживую (M1.8).
- Добавь GitHub Actions workflow, который на каждый push делает `uv sync` и `uv run pytest` — CI как автоматический сторож воспроизводимости на чистой машине.
