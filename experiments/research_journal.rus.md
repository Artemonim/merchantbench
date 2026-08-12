# Журнал исследования MerchantBench

Живой документ для переноса знаний между сессиями чата. **Новая сессия должна начинаться с чтения этого файла** — здесь контекст, который раньше таскали вручную через историю чата.

Правила ведения:
- Новые эпохи/эксперименты — новыми датированными секциями сверху; старые не удалять, а помечать устаревшими (`~~зачёркнуто~~` или `(устарело, см. …)`).
- Числа приводить только проверенные (запросом к `state.db` / артефактом / кодом), с указанием источника.
- Ссылки на код — в формате `path:line`; помнить, что строки плывут по мере коммитов.
- Артефакты прогонов: `env/runs/*` (gitignored), `env/batch_summaries/*` (gitignored), `experiments/run_history.json` (в гите, ledger), этот журнал (в гите).

---

## 2026-08-12 — Эпоха v4: pricing-патология синтетической экономики

### TL;DR текущего состояния

- Влит коммит `7170f8b6e31b2c8e4c7cf6126c17f0bf68a42b15`: `order_outcome_v4` — buyer-visible public reviews драйвят спрос, internal service quality — отдельный KPI.
- Прогнана ctx-матрица 30д × 9 (200k/350k/1M × seeds 42–44, `max_parallel=9`, CoreWeave FP8, compression 0.85): **7 finished, 2 incomplete (seed 43)**.
- Главный finding: **синтетическая экономика структурно мискалибрована** (ценовой якорь = себестоимость, низкая эластичность, нет потолка цены) + плотный synthetic demand. Net margin до 57% от GMV за 30д; лучший run (213k) ≈ годовой human baseline из paper.
- Выводы по ctx-окнам и по v4-рейтингу — **предварительные**, до причинных ablations. Договорённый путь: synthetic v5 (margin-consistent генерация) → ablations без LLM → калибровка → повторная ctx-матрица → red-team → датасеты.

### 1. Реестр прогонов ctx-матрицы (batch `20260811T220533Z`, 30д, horizon 720)

| Ctx | Seed | Run ID | Status | Net assets | Orders | USD | Примечание |
|---|---|---|---|---:|---:|---:|---|
| 200k | 42 | `run-20260811T224840-a0dacf` | finished | 64 520 | 704 | 0.909 | |
| 200k | 43 | `run-20260811T224842-bf025e` | **incomplete** | — | — | 0.542 | stop t=444/720, Hermes затих без OpenRouter fail |
| 200k | 44 | `run-20260811T224846-fc7238` | finished | 39 461 | 870 | 0.772 | |
| 350k | 42 | `run-20260811T224852-595278` | finished | 92 835 | 677 | 0.591 | |
| 350k | 43 | `run-20260811T224857-dd75ff` | finished | 11 138 | 556 | 0.882 | самый слабый finished |
| 350k | 44 | `run-20260811T224903-b3749e` | finished | 104 043 | 1 140 | 0.831 | |
| 1M | 42 | `run-20260811T224907-d1f50f` | finished | **213 354** | 1 022 | 0.989 | экстремум |
| 1M | 43 | `run-20260811T224911-3cbcae` | **incomplete** | — | — | 0.530 | ReadTimeout к env :5050, stop t=564/720 |
| 1M | 44 | `run-20260811T224915-ae65b9` | finished | 39 665 | 845 | 1.234 | |

Конфиг подтверждён per-run в `meta.json`: `agent.hermes.context_length` (200000 / 350000 / 1048576), `compression_threshold: 0.85`. Узкое место прогона — локальный Flask :5050 под 9 параллелями (добивки делать ≤3–4 parallel).

### 2. Подтверждённые факты о среде (источник → значение)

**Спрос и цены:**
- `env/core/demand.py:64` — `q ∝ market_curve[day] · small_share · hour_share · (sale_price/ref_price)^(−ε) · lifecycle`; верхнего потолка цены нет; `MIN_SALE_PRICE=0.01`; `MAX_EXPECTED_DEMAND_PER_LISTING_STEP=1000`.
- `env/tools/registry.py:60` — `_SALE_PRICE`: «must be > 0», максимума в схеме нет.
- `env/data/synth.py:112,122` — при генерации `price == ref_price`: **покупательский ценовой якорь = оптовая себестоимость**. `base_demand ~ U(1,50)` заказов/день на товар; `small_share=1.0` (`default.yaml:22`).
- Эластичность: категорийный профиль `mean ± jitter` (clamp в `[min,max]` профиля, `generation_profiles.py:148`). Фактически по каталогу: **appliances — все 100 товаров ε∈[0.73, 0.97] (<1)**; остальные категории ≥1.06. Конфигурационный `min: 0.65` никогда не достигается; supplier fallback `[0.5, 2.5]` (`default.yaml:137`) по факту не используется.
- Штрафы фиксированные (`default.yaml:149`): cancel 0, refund 8, only_refund 0 (закупочная цена сгорает), bad_review 5, timeout 3, stockout 5, insufficient_balance 5 RMB. При ценах 500–1050 RMB это <1% чека.

**Рейтинг v4 (`default.yaml:174`, `env/core/public_reviews.py`):**
- `bucket_thresholds: [2.50, 3.30, 3.80, 4.20]`, `star_multipliers: [0.10, 0.35, 0.80, 1.00, 1.12]` — асимметрия: max upside +12% (5★), max downside −90% (1★); 4★ нейтральна.
- Self-selection: `probability_by_star = (0.30, 0.18, 0.08, 0.06, 0.12)` — negativity bias; `settled_bad_review` публичен всегда.
- Demand: `min_trust_multiplier 0.80`, `half_saturation_reviews 20`; `Dem× = quality × trust`, confidence = n/(n+20).
- Учёт: `stockout`/`insufficient_balance` персистятся в `orders`, но **исключаются из GMV/gross** (`simulator.py:761,1530`). Net profit = realized_revenue − realized_cost − penalties по settled.

**Private-real pipeline (образец защит, `env/data/build_private_real_db.py`):**
- `ELASTICITY_CLIP_MIN=1.10`, `ELASTICITY_CLIP_MAX=6.00`, `REF_PRICE_CAP_RATIO=2.0`, `EXPECTED_NET_PROFIT_CAP_365=50k`.
- `:1860` — `ε = ref/(ref−cost)`: при constant-elasticity спросе это делает `ref_price` **теоретически оптимальной ценой** (из `p* = ε/(ε−1)·c`).

### 3. Findings (все числа проверены запросами к `state.db`)

#### 3.1 Маржинальность — исправленные booked-only метрики

Net = `final_net_assets − 3000` (2000 cash + 1000 deposit), GMV/gross без `stockout`/`insufficient_balance`:

| Run | GMV | Gross/GMV | Net/GMV | Доля ε<1 в booked gross | Топ-товар (ε, booked gross) |
|---|---:|---:|---:|---:|---|
| 200k-42 | 140.2k | 53.1% | 43.9% | 44.6% | P00123 (0.863, 33.0k) |
| 200k-44 | 87.5k | 49.3% | 41.7% | 13.9% | P00429 (1.292, 5.6k) |
| 350k-42 | 174.3k | 59.7% | 51.5% | 18.9% | P00332 (1.119, 29.2k) |
| 350k-43 | 33.9k | 32.7% | **24.0%** | 19.3% | P00958 (1.311, 2.2k) |
| 350k-44 | 178.3k | 66.2% | 56.7% | **6.2%** | P00894 (1.436, 11.2k) |
| 1M-42 | 369.1k | **67.8%** | **57.0%** | 19.1% | P00298 (1.386, 40.7k) |
| 1M-44 | 88.7k | 51.5% | 41.3% | 19.1% | P00003 (0.767, 5.8k) |

- Только **2 из 7** лидеров имеют ε<1. Узкая версия «эксплойт = ε<1» не объясняет результат.
- **Уточнённый механизм:** при якоре на себестоимости money printer работает для всего каталога: теоретический optimum `p* = ε/(ε−1)·c` при ε=1.39 → 3.6×, ε=1.29 → 4.4×, ε=1.12 → 9.3×, ε=1.05 → 21×. Агенты эмпирически сидят рядом с optimum (markup 2.5–5.5×). ε<1 (все appliances) — патологический край с неограниченной прибылью.
- Математика: при ε=1 прибыль `(p−c)/p` асимптотична к константе (не бесконечна); при ε<1 растёт без предела.
- `insufficient_balance`: 70–309 заказов/run (у 1M-44 — 37% всех) — агенты листингуют без cash на закупку; штраф 5 RMB при их марже — шум.
- Сравнение с paper (условно, там средние по 3 runs): по profit/day 1M-42 ≈ **43–45×** mean лучшего агента paper (Qwen3.7-Max 59.5k/365д) и ≈ **12×** human baseline (217.6k/365д).

#### 3.2 Динамика рейтинга v4

- **Terminal `Dem×` (после drain) ≠ множитель активного горизонта.** Примеры: 200k-42: 0.454@t719 → 0.805 после drain; 350k-42: 0.547 → 0.807; 1M-44: 0.501 → 0.447.
- Средний активный Dem× по runs: **0.635–0.766**; внутриранговые качели до 0.45↔0.87 (~1.9× потока заказов). Рейтинг экономически значим — формулировка «рейтинг ни на что не влияет» **опровергнута и снята**.
- Магазины зарабатывали 40–210k net **при** avg Dem× 0.64–0.77 — маржинальная проблема независима от рейтингового давления.
- `public_review_full_response_rating` (4.13–4.49) — lifetime-контрфактуал «если бы отзыв оставили все», **не** качество; internal `service_quality` = 3.46–3.84 — другая метрика. Selection gap (public vs full-response) = −0.7…−1.26★ из-за negativity bias.

#### 3.3 Масштаб спроса

- 18.5–38 попыток заказа/день (fulfilled ~13–26) против paper: DeepSeek-V4-Flash mean 5.45/день, human 25.9/день.
- Аномалия двуосевая: **маржа × плотность спроса**. Какая ось доминирует — покажут только ablations.

#### 3.4 Контекстные окна (предварительно!)

| Seed | 200k | 350k | 1M |
|---:|---:|---:|---:|
| 42 | 64.5k | 92.8k | **213.4k** |
| 44 | 39.5k | **104.0k** | 39.7k |

- 350k > 200k на обоих paired seeds; 1M — бimodal (лучший и слабый результаты). n=2, выводы отложены.
- **Confound:** до починки экономики ctx-матрица измеряет variance в обнаружении/удержании pricing-политики, а не качество reasoning. 1M-42 vs 1M-44 — разные политики + ранний рейтинговый провал (Dem× 0.45), а не «1M нестабилен».
- Гипотеза «350k = sweet spot (compaction при 0.85×ctx как периодическая уборка cognition state)» — не проверена, правдоподобна по token-статистике (350k-42: −36% tokens к 200k-42 при +44% net).

### 4. Методологические уроки (ошибки, уже совершённые — не повторять)

1. **SQL-агрегация по всем `orders` завышает GMV/gross**: `stockout`/`insufficient_balance` — это violations с `net = −penalty`, а не продажи. Канонический booked-only фильтр:
   ```sql
   SELECT SUM(sale_price), SUM(sale_price - purchase_price)
   FROM orders WHERE run_id = ? AND current_status NOT IN ('stockout', 'insufficient_balance');
   ```
2. **Terminal-метрики после drain** (рейтинг, Dem×) не описывают активный горизонт — использовать временной ряд `metrics` (`public_review_demand_multiplier` по t < 720).
3. `full_response_rating` ≠ `service_quality` — разная семантика, не смешивать.
4. При ε=1 прибыль ограничена (асимптота), «бесконечность» только при ε<1; optimum для ε>1: `p* = ε/(ε−1)·c`.
5. Фактический минимум ε в каталоге — 0.73 (генератор `mean±jitter`), а не конфигурационный 0.65.
6. Paper-сравнения: там средние по 3 runs; annualization profit/day — условность (разные каталоги/рейтинг-модели).

### 5. Договорённый план (приоритет сверху вниз)

1. **Synthetic v5 — margin-consistent генерация:** сэмплировать целевую retail-маржу по категориям → согласованно derive `cost`, `ref_price`, `elasticity` через `ε = ref/(ref−cost)` (образец — `build_private_real_db.py:1860`). `ref_price` становится теоретическим оптимумом; бесконечный хвост исчезает по построению.
2. **Калибровка амплитуды спроса** по `orders / active-listing-day` к референсу paper (human ~26/день), вместо `U(1,50)` на товар.
3. **Фиксированные штрафы сохранить** (сопоставимость с paper); цены вернуть в масштаб, где 3–8 RMB значимы. Прогрессивные штрафы — не первый кандидат.
4. **Hard cap цены** — только как явный видимый агенту guardrail (per-product cap не выразим статичной JSON-схемой; `ref_price` скрыт → непредсказуемые отказы).
5. **3 policy-level ablations без LLM** (rule-based): (a) только исправленная `cost/ref/ε`; (b) только уменьшенный demand scale; (c) обе правки. Разделит оси аномалии дёшево.
6. **2 seeds × 7д** калибровка → paired ctx-матрица (3×3, лучше 5 seeds) → **red-team как регрессионный тест до/после** фикса.
7. **Датасеты** (JDsearch + Olist гибрид) — после стабилизации unit economics; кривые спроса закладывать в калибровку п.2.

### 6. Открытые вопросы

- Добивка 2 incomplete seed-43 (200k `bf025e`, 1M `3cbcae`) — **только после** стабилизации экономики, иначе измерят старый дефект. ≤3–4 parallel.
- Per-day динамика цен агента (как доходил до near-optimal markup) — не анализировалась; источник: `events` / история `adjust_price` в `state.db`.
- Предложен, но не реализован постоянный инструмент `scripts/analyze_pricing.py` (booked-only учёт, CLI, тесты) — кандидат для Middle SWE.
- Какая ось (маржа vs плотность спроса) доминирует — ждёт ablations.

### 7. Backlog идей

**Bankruptcy red-team (после фикса экономики).** Агенту прямо говорить, что это benchmark и цель — обанкротить магазин. Три режима:
| Режим | Цель |
|---|---|
| Red-team unrestricted | Банкротство ASAP любыми разрешёнными действиями (ищет дыры среды) |
| Bad merchant | Минимизировать капитал без намеренных violations |
| Bad economics | Минимизировать net только через ассортимент/цены |
Метрика: `time_to_bankruptcy`. Black-box (только merchant tools) и white-box (с чтением исходников) — разные эксперименты. Предсказание по текущей экономике: «плохими ценами» обанкротиться почти нельзя (завышенная цена при ε<1 прибыльна); единственный быстрый маршрут — фарм штрафов violations, и он медленный (5 RMB × сотни заказов против 3k капитала).

**Датасеты (после unit economics).** JDsearch (JD.com, 173k users, полный год 2021-10→2022-10, 12.87M products, `shop_id`, китайская сезонность 11.11/CNY/618; timestamps = интервалы, реконструировать обратным накоплением; нет цен/логистики — выборочный proxy спроса) → demand curves + products + shops. Olist (~100k заказов, seller/price/freight/delivery/review) → эмпирические supplier/logistics/rating распределения. REES46 (285M событий, 7 мес., цены) → плотные demand curves. M5/Favorita — форма спроса retail. Гибрид: JDsearch(demand) + Olist(supplier risk) + synthetic model (inventory, elasticity, fines). Выборка 100k товаров — стратифицированно (хиты + middle-tail + cold), не top-100k.

---

*Следующая секция — после ablations / synthetic v5.*
