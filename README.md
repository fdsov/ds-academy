# DS Academy

Личная учебная платформа по Data Science: от полного нуля до research-level. 29 модулей в 8 фазах — мышление, данные, статистика, ML, deep learning, production, research. Актуальный стек 2026 и сквозная линия «как нейросети помогают data scientist».

**Живая версия:** https://fdsov.github.io/ds-academy/

## Стек

- [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- Деплой: GitHub Pages через GitHub Actions (`.github/workflows/deploy.yml`)

## Локальный запуск

```bash
# через uv (рекомендуется)
uv venv
uv pip install -r requirements.txt
uv run mkdocs serve

# открыть http://127.0.0.1:8000
```

Или без установки в окружение:

```bash
uvx --with mkdocs-material --with pymdown-extensions mkdocs serve
```

## Сборка

```bash
mkdocs build          # в каталог site/
mkdocs gh-deploy       # собрать и опубликовать на GitHub Pages
```

## Структура

```
docs/
  index.md            — главная
  methodology.md      — как учиться (методика топ-1%)
  curriculum.md       — полная программа
  phase0..phase7/     — лекции по фазам (M0..M28)
mkdocs.yml            — конфигурация и навигация
.github/workflows/    — авто-деплой
```

## Как добавить лекцию

1. Создай `docs/phaseN/mXX-slug.md`.
2. Добавь её в `nav:` в `mkdocs.yml`.
3. `git push` — GitHub Actions соберёт и опубликует автоматически.
