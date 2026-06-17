# Аудит актуальности DS Academy — июнь 2026

Дата: 2026-06-17. Метод: на каждый currency-критичный файл — агент-аудитор + независимый веб-верификатор (отсев ложных срабатываний). 22 файла. Веб-источники проверены на 17.06.2026.

**Вне scope:** вечнозелёные модули (M0 мышление, M2 Python-ядро, M3 математика, M8-M11 вероятности/статистика, M26 теория обучения) — там вопрос корректности, а не свежести, в currency-аудит не входят.

## Вердикт

Курс актуален. Фундамент вне времени, выбор стека соответствует 2026. Реальные сдвиги, требующие правок, концентрируются в трёх местах: версии (Python, pandas, scikit-learn), быстрый AI/LLM-слой и регуляции. Подтверждено 70+ находок; абсолютное большинство — LOW (номера версий, формулировки). Ниже — приоритизация.

---

## HIGH — фактически неверно на сегодня (править в первую очередь)

### M27 · EU AI Act: high-risk подан как «в фазе применения 2026-2027»
Неверно на июнь 2026. Digital Omnibus (соглашение трилога 6-7 мая 2026) отложил high-risk-требования (Annex III) на **2 декабря 2027** (standalone) и **2 августа 2028** (встроенные в продукты). При этом две фазы УЖЕ действуют, и лекция это не фиксирует:
- запрещённые практики + AI-literacy — со **2 февраля 2025**;
- обязательства для GPAI-моделей — со **2 августа 2025**.

Fix: переписать таймлайн (вступило 1 авг 2024; запреты с фев 2025; GPAI с авг 2025; high-risk отложен на дек 2027 / авг 2028). Добавить отсутствующий US-блок (Colorado AI Act с 30 июня 2026, California frontier-AI с 1 янв 2026, федеральный EO 14365 от 11 дек 2025). Добавить судебную практику GDPR ст.22 (SCHUFA C-634/21, C-203/22) — прямо релевантно автоотказам в выплатах.

---

## MEDIUM — реальное устаревание, заметное студенту

### Версии и базовый стек
- **M1 · Python-версии.** 3.14 вышел 7 окт 2025, к июню 2026 зрелый. Рекомендация «дефолт 3.12 / новый 3.13» отстаёт на релиз -> «дефолт 3.13 / новый 3.14». Free-threading (PEP 779) в 3.14 стал официально поддерживаемым (python3.14t), не дефолт — лекция знает только про экспериментальный 3.13.
- **M6 / M13 · pandas 3.0.** Описан мир pandas 2.x (строки в object, Arrow как опция). В pandas 3.0 дефолт — выделенный str-dtype на PyArrow и Copy-on-Write. Обновить «pandas 2.x на Arrow» -> «pandas 3.x (CoW + PyArrow-строки по умолчанию)».
- **M13 / M14 · scikit-learn.** «1.5+» -> ориентир «1.7+ (актуальная 1.9)».
- **M12 · altair 6** (Vega-Lite 6), а не v5.

### ML-практики
- **M13 · «бустинг по умолчанию бьёт DL на табличке»** — смягчить: с 2025 есть tabular foundation models (TabPFN v2/2.5), на малых/средних выборках уже часто обходят бустинг; на больших проде бустинг остаётся практичным дефолтом.
- **M16 · Polars outer join** — устаревший синтаксис `how="outer"` -> `how="full", coalesce=True`.
- **M22 · MLflow** — стадии Staging/Production устарели, актуально через aliases (@champion/@challenger); rollback = переназначение алиаса. Переписать раздел и практику 5.

### Data engineering и архитектуры
- **M23 · dbt.** Крупные события 2026: dbt Fusion (Rust-движок, до 30x, column-level lineage), dbt Core v2.0 (1 июня 2026), слияние dbt Labs + Fivetran (1 июня 2026). Airflow 3.0 ввёл Assets — граница «Airflow task-центричный vs Dagster asset-центричный» размылась.
- **M19 · State Space Models / Mamba** не упомянуты вообще — заметный пробел в разделе про вытеснение RNN. Добавить подраздел (параллелизуемая селективная рекуррентность).
- **M28 · time-series foundation models** — список неполон/устарел. Обновить: Chronos-2, Moirai-2, TimesFM (Google), TimeGPT, Lag-Llama. Заменить архивный `lifetimes` на `PyMC-Marketing` для LTV.

### LLM-слой (M20) — веб-проверено
- **Контекст Claude.** «200K у Claude» устарело в ~5 раз: с 13 марта 2026 окно 1M в GA у Opus/Sonnet 4.6. Норма топ-моделей 2026 — 200K-1M (Gemini до 2M, Llama 4 Scout до 10M).
- **gpt-4o-mini в коде** — legacy (OpenAI убрала 4o/4.1/o4-mini из ChatGPT в фев 2026). Заменить на gpt-5.x nano/mini. Иронично: соседний блок сам предупреждает «LLM выдаёт устаревшие имена моделей».
- **Пропущены GRPO / RLVR** — определяющий метод post-training 2026 для reasoning-моделей (так учили DeepSeek-R1). Добавить в стадии выравнивания.
- **Пропущен A2A** (Agent2Agent, v1.0 начало 2026) рядом с MCP — multi-agent слой.
- Мелочи: Responses API вместо beta `.parse()`; многоязычные эмбеддинги (Qwen3-Embedding / Gemini Embedding 2 / BGE-M3) для RU-кейсов.

### Прочее MEDIUM
- **M7 · Pandera для Polars** — неверный синтаксис схемы (generic `Series[...]` из pandas-ветки). Поправить на polars-стиль.
- **M21 · GitHub Copilot** подан только как inline-автодополнение — с 2025-26 у него agent mode (issue->PR). Добавить EU AI Act рядом с GDPR в compliance.
- **W8 · pickle/joblib** — добавить про skops и риск arbitrary code execution при загрузке недоверенных артефактов.
- **W10 · kaggle auth** — современный способ `kaggle auth login` (OAuth); kaggle.json как legacy. Slug Playground Series — сезон S6, не S4.

---

## LOW — точечные мелочи (версии, формулировки, ссылки)

Сводно по файлам (полнота — в перечне ниже; всё не критично, но при правке заодно):
- **M5:** опечатка `NMEP 50` -> `NEP 50`; `np.matrix` — pending deprecation, не «deprecated»; ресурс переименован в Scientific Python Lectures.
- **M6:** упомянуть Narwhals (кросс-движковый слой pandas/Polars).
- **M7:** httpx http2=True требует `httpx[http2]`.
- **M12:** seaborn.distplot — deprecated, не «удалён».
- **M13:** издание Géron — новая PyTorch-редакция 2025.
- **M14:** XGBoost умеет leaf-wise (grow_policy=lossguide), не только level-wise; TabPFN -> TabPFN-2.5.
- **M16:** TargetEncoder cross-fitting только в fit_transform; эмбеддинги -> BGE-M3/Qwen3; Optuna -> упомянуть Ray Tune/FLAML.
- **M17:** SHAP/LIME — смягчить «вытеснил»; добавить conformal prediction; sklearn prefit -> FrozenEstimator; TabPFN.
- **M18:** Adam/AdamW -> упомянуть Muon (PyTorch 2.9); GELU -> SwiGLU/SiLU в современных LLM.
- **M22:** FastAPI -> упомянуть BentoML/KServe/Ray Serve для нагруженного инференса.
- **M23:** Iceberg-формулировку смягчить; рядом с DuckDB упомянуть Polars.
- **M29:** Inference API -> Inference Providers (hf-inference); добавить kagglehub; сноска про code-соревнования.
- **W7:** Qwen2.5 -> Qwen3; gpt-4o-mini -> актуальная дешёвая; Qwen3-Embedding/Reranker; смягчить «RAG — самый востребованный».
- **W9:** альтернативы yfinance (Stooq/OpenBB/Tiingo); упомянуть vectorbt в «Развитие».

---

## Что проверено и АКТУАЛЬНО (правок не требует)

uv (стандарт; Astral куплена OpenAI в марте 2026, OSS сохранён), numpy 2.x, DuckDB, векторные БД (Qdrant/Weaviate/Milvus/pgvector), reasoning-модели с test-time compute, LoRA/PEFT, prompt caching, eval-first (RAGAS/Langfuse), reranking/hybrid в RAG, structured outputs как API, fairlearn/SHAP/opacus, дифференциальная приватность, теорема о невозможности fairness, model cards/datasheets, размеры штрафов AI Act (до 7%) и GDPR (до 4%). Фактических ошибок в LLM-концепциях не найдено.

## Итог по слоям

- Фундамент (математика, статистика, теория): вне времени, ~95% — актуальность не вопрос.
- Инструменты: ~85% точно на 2026; основные сдвиги — версии (Python 3.14, pandas 3.0, sklearn 1.9) и data-eng (dbt Fusion/Fivetran, Airflow 3 Assets).
- AI/LLM/регуляции: самый волатильный слой; концепции верны, специфику (контекст, имена моделей, GRPO/RLVR, A2A, даты EU AI Act) надо освежать. Это болезнь любого курса, не дефект конкретно этого.
