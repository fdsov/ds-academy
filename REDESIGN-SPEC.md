# DS Academy — Platform Redesign Spec (contract)

Дата: 2026-06-28. Ветка: `redesign/platform-stepper`. Это контракт реализации, НЕ страница сайта (лежит в корне репо, не в `docs/`, поэтому mkdocs её не собирает).

## Цель

Превратить MkDocs-сайт DS Academy в ощущение полноценной учебной платформы:
1. **Шаги разделены** — урок показывается по одной карточке-экрану за раз (степпер-деск), а не длинным скроллом.
2. **Вид платформы** — современная учебная SaaS-эстетика (карточки, мягкие тени, индиго-акцент, light/dark), платформенный хром (прогресс/стрик в шапке, силлабус с кольцами прогресса), лендинг вместо таблицы доков.
3. **Интерактив** — Python в браузере (Pyodide), флешкарты + spaced repetition, геймификация (XP/стрик/бейджи/дашборд), богаче квизы + интерактивные визуализации.

## Жёсткие ограничения

- Остаёмся на MkDocs Material + GitHub Pages. Деплой не меняется.
- **Прозу 528 уроков НЕ трогаем.** Разрешено: новые JS/CSS файлы, правка `mkdocs.yml`, `index.md` (лендинг), новые страницы `docs/dashboard.md` и `docs/review.md`, `overrides/` шаблоны, и точечная вставка `\`\`\`viz` блоков в 2-3 флагманских урока.
- **Не редактировать `docs/stylesheets/extra.css`** (там незакоммиченная WIP-работа другого трека) — вся новая CSS идёт в новые файлы.
- Без эмодзи в UI и коде.

## Design tokens (CSS, владелец — theme.css)

Объявляются в `:root` и переопределяются для тёмной темы через `[data-md-color-scheme="slate"]`. Все модули используют ТОЛЬКО эти переменные, своих цветов не вводят.

```
--dsa-accent:        #5b61d6   (индиго, основной акцент)
--dsa-accent-strong: #3b3f8c
--dsa-accent-soft:   rgba(91,97,214,0.12)
--dsa-ok:            #2e7d32
--dsa-warn:          #ef6c00
--dsa-danger:        #c62828
--dsa-surface:       фон карточки (light: #fff / dark: #1e2030-ish через md vars)
--dsa-surface-2:     вложенная поверхность
--dsa-border:        тонкая граница
--dsa-text / --dsa-text-dim
--dsa-radius: 14px;  --dsa-radius-sm: 10px
--dsa-shadow:   0 2px 10px rgba(15,18,45,.06), 0 8px 30px rgba(15,18,45,.06)
--dsa-shadow-lg:0 10px 40px rgba(15,18,45,.12)
--dsa-gap: 1rem; --dsa-gap-lg: 1.6rem
--dsa-maxw: 52rem (комфортная ширина чтения)
```

Базовый компонентный язык (классы, владелец — theme.css): `.dsa-card`, `.dsa-btn`, `.dsa-btn--primary`, `.dsa-btn--ghost`, `.dsa-chip`, `.dsa-ring` (SVG прогресс-кольцо). Модули переиспользуют их.

## localStorage schema (единый namespace `dsa:`)

| ключ | значение | владелец / писатели |
|---|---|---|
| `dsa:done:<pathname>` | `"1"` | markDone; читают все |
| `dsa:xp` | целое (сумма XP) | gamify |
| `dsa:xp:awarded:<dedupeKey>` | `"1"` | gamify (защита от двойного начисления) |
| `dsa:streak:count` | целое | gamify |
| `dsa:streak:last` | ISO `YYYY-MM-DD` | gamify |
| `dsa:badges` | JSON-массив id | gamify |
| `dsa:deck:<pathname>` | индекс последней карточки | lesson-deck |
| `dsa:fc:<cardId>` | JSON `{ef,interval,due,reps,lapses}` (SM-2-lite) | flashcards |

`pathname` = `location.pathname` (как в текущих progress.js/step-nav.js — совместимость сохраняется).

## window.DSA — глобальный API (владелец — gamify.js, грузится РАНЬШЕ остальных)

gamify.js определяет `window.DSA` синхронно на загрузке, до прочих модулей. Остальные модули вызывают его опционально (`window.DSA && DSA.x(...)`), чтобы работать и без него.

```
DSA.award(amount, dedupeKey)   // +XP один раз на dedupeKey; тихо игнор если уже начислено
DSA.markDone(pathname?)        // отметить шаг пройденным (default = текущий), бамп прогресса + streak
DSA.isDone(pathname?) -> bool
DSA.touchStreak()              // зарегистрировать активность сегодня (обновляет streak)
DSA.progress() -> {done,total,pct}             // по ссылкам в nav (фазы/практикум/воркшопы)
DSA.phaseProgress() -> [{phase, done, total, pct}]
DSA.on(event, cb) / DSA.emit(event, data)      // шина: события 'xp','done','badge'
```

XP-награды (значения — в gamify, прочие модули только зовут award с правильным dedupeKey):
- шаг пройден: dedupeKey `done:<path>`, +10
- верный ответ квиза: `quiz:<path>:<qIdx>`, +5
- верная задача (task): `task:<path>:<i>`, +10
- пройден pyexercise: `pyex:<path>:<i>`, +15
- сессия флешкарт: `fc:session:<date>`, +5

## Файлы и владельцы

Новые JS — в `docs/javascripts/`, новые CSS — в `docs/stylesheets/`.

| Файл | Владелец | Назначение |
|---|---|---|
| `stylesheets/theme.css` | CORE (я) | токены, компонентный язык, платформенный хром, деск-стили, лендинг |
| `javascripts/lesson-deck.js` | CORE (я) | степпер-деск: лента-пилюли + нарезка на карточки + нав + re-typeset MathJax + done/resume. Заменяет step-nav.js |
| `overrides/main.html` (+ partials) | CORE (я) | Material custom_dir: монтаж хедер-виджета прогресс/стрик |
| `index.md` | CORE (я) | лендинг-платформа |
| `javascripts/gamify.js` + `stylesheets/gamify.css` + `docs/dashboard.md` | AGENT C | window.DSA, XP/стрик/бейджи, дашборд-страница |
| `javascripts/pyodide-run.js` + `stylesheets/pyodide.css` | AGENT A | Run на ```python; ```pyexercise с авто-тестами |
| `javascripts/flashcards.js` + `stylesheets/flashcards.css` + `docs/review.md` | AGENT B | сбор пар Q/A, SM-2, страница повторения |
| `javascripts/viz.js` + `stylesheets/viz.css` | AGENT D | движок ```viz + 3 виджета (CI-width~n, FWER p-hacking, Simpson) |
| `javascripts/quiz.js` (extend) | AGENT D | + multi-select (несколько [x]) и режим «Проверить» |
| `javascripts/progress.js` | CORE (я) | консолидирую: делегирую в gamify, оставляю sidebar-метки |

Существующие `quiz.js`, `task.js` сохраняют текущую обвязку (парсинг ```text) — расширяем, не ломаем.

## Конвенции контента (без правки прозы уроков)

- **Run-кнопка**: pyodide-run.js вешает «Запустить» на КАЖДЫЙ отрендеренный ` ```python ` блок (класс `.language-python`). Ленивая загрузка Pyodide по первому клику. numpy/pandas — по запросу через `micropip`/`loadPackage`.
- **pyexercise** (новое, для будущих заданий): fenced ` ```pyexercise ` с секциями `# --- solution ---` (видимый старт-код) и `# --- tests ---` (скрытые assert). Pyodide исполняет solution+tests; зелёно если без исключений.
- **viz** (новое): fenced ` ```viz ` с `TYPE: ci-width|fwer|simpson` + параметры. viz.js заменяет блок на интерактивный виджет.
- **Флешкарты**: flashcards.js собирает карточки из УЖЕ существующих на сайте пар `!!! question "Проверь себя"` (вопросы) ↔ `??? success "Ответы"` (ответы). cardId = хеш(path + индекс вопроса). Контент не пишем.

## Порядок подключения (extra_javascript в mkdocs.yml)

```
javascripts/mathjax.js                 (config)
https://unpkg.com/mathjax@3/.../tex-mml-chtml.js
javascripts/gamify.js                  (определяет window.DSA ПЕРВЫМ из модулей)
javascripts/progress.js
javascripts/quiz.js
javascripts/task.js
javascripts/pyodide-run.js
javascripts/flashcards.js
javascripts/viz.js
javascripts/lesson-deck.js             (ПОСЛЕДНИМ: нарезает на карточки после рендера виджетов)
```

extra_css: theme.css, extra.css (существующий), gamify.css, pyodide.css, flashcards.css, viz.css.

## Степпер-деск — правила (lesson-deck.js)

- Активен на путях `/(phase\d+|practicum|workshops)/`. Короткие страницы (<2 заголовков-границ) остаются одной карточкой.
- Граница нарезки: уроки — по `h3`; воркшопы — по `h2`. H1 + лид-абзац + лента-пилюли = закреплённая шапка, не карточка.
- Виден один `.dsa-deck__card` за раз. «Далее» листает карточки; на последней — переход на следующий под-шаг (footer next link), отметка `markDone`. «Назад» симметрично.
- Прогресс: точки `●─●─○` (карточки внутри под-шага) + существующие пилюли (под-шаги модуля).
- Resume (`dsa:deck:<path>`), клавиши ←/→, скролл вверх при смене.
- **MathJax**: при показе карточки `window.MathJax && card.querySelector('.arithmatex') && MathJax.typesetPromise([card])` — надёжно против неверных метрик в display:none.
- **Гейтинг**: «Далее» НЕ блокируется. Шаг помечается done (и капает XP done) когда (а) на карточках без интерактива пользователь дошёл до последней, либо (б) интерактив (квиз/задача/pyex) на карточке выполнен. Интерактивные карточки подсвечивают «ответь», но не запирают.

## Проверка готовности (CORE)

`./.venv/bin/mkdocs build` без ошибок; ключевые страницы (главная, урок phase0/m0/03, воркшоп w2, dashboard, review) рендерятся; деск листается; MathJax/квизы/Run работают; light/dark ок.
