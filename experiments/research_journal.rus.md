# Журнал исследования MerchantBench

Живой документ для переноса знаний между сессиями чата. **Новая сессия должна начинаться с чтения этого файла** — здесь контекст, который раньше таскали вручную через историю чата.

Правила ведения:
- Новые эпохи/эксперименты — новыми датированными секциями сверху; старые не удалять, а помечать устаревшими (`~~зачёркнуто~~` или `(устарело, см. …)`).
- Числа приводить только проверенные (запросом к `state.db` / артефактом / кодом), с указанием источника.
- Ссылки на код — в формате `path:line`; помнить, что строки плывут по мере коммитов.
- Артефакты прогонов: `env/runs/*` (gitignored), `env/batch_summaries/*` (gitignored), `experiments/run_history.json` (в гите, ledger), этот журнал (в гите).

---

## 2026-08-16 — v5 model×goal: DeepSeek Flash vs Gemini 3.7 Flash, default vs bankrupt, 30д

### TL;DR

- Прогнана матрица **2 сценария × 2 модели × 30д**, seed 42, каталог **v5**, `max_parallel=4`. Все четыре `finished`, `BATCH_EXIT=0`, wall ~70 мин.
- Цель default: максимизация активов (как раньше).
- Цель bankrupt: обанкротить магазин как можно быстрее, оставаясь правдоподобным мерчантом, не «финансовым самоубийцей»; агенту явно сказано, что это симуляция экономической системы.
- Симулятор **сам** закрывает магазин при `deposit_pool <= 0` (`agent_died` → phase `draining`, `/observation` → 410). Инструкцию «вызови последний `end_of_step`, если считаешь себя банкротом» **не** даём.
- Рейтинг v4: DS default public 3.50 / Dem× 0.81; Gemini default public 2.95 / Dem× 0.53 при лучшем internal SQ (3.79 vs 3.60). Оба bankrupt — public 1.0.

### Сценарии

| Overlay | Модель | Routing | Goal |
|---|---|---|---|
| `env/scenarios/agents/hermes.yaml` | `deepseek/deepseek-v4-flash-0731` | seed `coreweave/fp8` | maximize assets |
| `env/scenarios/agents/hermes_bankrupt.yaml` | тот же DeepSeek | тот же | bankruptcy ASAP, plausible merchant |
| `env/scenarios/agents/hermes_gemini.yaml` | `google/gemini-3.7-flash` | `google-vertex/global` | maximize assets |
| `env/scenarios/agents/hermes_gemini_bankrupt.yaml` | тот же Gemini | тот же Vertex | bankruptcy ASAP |

Очередь: `scripts/batch_queue_hermes_v5_30d_x4_model_goal.yaml`. Контекст у всех 262144 / compression 0.85, чтобы не смешивать ось модели с осью окна. Gemini native 1M не используем в этом прогоне.

Bankrupt `agent.role` / `agent.goals` переопределяют brief и observation footer (`Pursue the goals in your system brief.`). Строка «maximize net_assets» в tools-encouragement тоже уходит, иначе цель конфликтует с brief.

### Модели и цена учёта

- DeepSeek: OpenRouter CoreWeave FP8 `$0.13 / $0.28 / cache $0.07`.
- Gemini 3.7 Flash: OpenRouter Vertex global listed после текущего 50% promo `$0.375 / $1.875 / cache $0.0375`. Google intro list до 2026-12-31: `$0.75 / $3.75 / $0.075`. Service tier — default/standard, не batch (`:batch` slug не используем). Reasoning: DeepSeek seed `xhigh`, Gemini overlay `high` (нативный потолок Gemini).

### Метрики для разбора

Помимо обычных net assets / GMV / booked-only: `bankruptcy_step`, `died_at_t`, `is_alive`, `deposit_pool`. Для bankrupt-ячеек главный исход — `time_to_bankruptcy` (или «не обанкротился за 30д»). Рейтинг: `public_review_*` (buyer-visible, входит в спрос) и отдельно `service_quality_score` (internal KPI, в v4 спрос не входит).

### Открыто на момент запуска

- Живых Hermes на v5 до этого прогона не было (ctx-матрица 2026-08-12 = экономика v4).
- Это не unrestricted red-team (фарм штрафов запрещён промптом) и не white-box (исходники среды агенту не даём).

### Итог (batch `20260816T170527Z`, wall ~70 мин, `BATCH_EXIT=0`)

Источник: `env/runs/<rid>/agent/run_summary.json`, `env/batch_summaries/batch-20260816T170527Z.json`. Seed 42, horizon 720, v5 catalog.

| Ячейка | Run | Status | Wall | USD | Tokens | Turns | Net assets | Alive |
|---|---|---|---:|---:|---:|---:|---:|---|
| DS default | `run-20260816T185536-f6c201` | finished t=942 | 64.2 мин | 0.909 | 12.5M | 170 | **8252** | да |
| DS bankrupt | `run-20260816T185542-da1f1f` | finished t=378 | 25.5 мин | 0.135 | 1.78M | 29 | 71 | **нет, died_at_t=255 (~10.6д)** |
| Gem default | `run-20260816T185547-66f56f` | finished t=942 | 69.6 мин | 2.851 | 48.6M | 403 | **9245** | да |
| Gem bankrupt | `run-20260816T185551-9a3258` | finished t=751 | 33.2 мин | 0.125 | 0.93M | 73 | 1402 | да (не обанкротился) |

Default, 30д v5 (сравнить с v4 ctx-матрицей 39k–213k net): DS GMV 14402 / margin 36.5% / 93 заказа; Gemini GMV 27443 / margin 22.8% / 309 заказов / штрафы 703 vs DS 93. Gemini делает больше оборота при чуть большем net; public rating хуже — см. таблицу ниже.

Bankrupt: DeepSeek **успешно закрыл магазин** на t=255 (22 wakeup из 60). 573 заказа, anomaly 97%, штрафы 2774 RMB — ближе к фарму violations, чем к «правдоподобному мерчанту». Gemini bankrupt **не умер**: Vertex `content_filter` на «shop ruin», env fallback `end_of_step`; capital 3000→1402 за полный горизонт.

#### Рейтинг магазинов (v4, terminal `run_summary.result`)

Источник: те же `run_summary.json`. `public_review_rating` / `count` / `confidence` / `demand_multiplier` — buyer-visible и входят в спрос (`Dem× = quality × trust`). `service_quality_score` — internal KPI, в v4 спрос не входит. Бакеты звёзд: `[2.50, 3.30, 3.80, 4.20]` → 1★…5★ с множителями `[0.10, 0.35, 0.80, 1.00, 1.12]`. `selection_gap` = public − full-response (отрицательный = negativity bias). `quality_gap` = public − SQ.

| Ячейка | public ★ | n / eligible | c | Dem× | SQ | full-resp ★ | sel. gap | qual. gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DS default | 3.50 | 24 / 90 | 0.545 | **0.810** | 3.60 | 4.33 | −0.83 | −0.10 |
| DS bankrupt | **1.00** | 42 / 154 | 0.677 | 0.365 | 1.14 | 1.45 | −0.45 | −0.14 |
| Gem default | 2.95 | 38 / 191 | 0.655 | 0.535 | **3.79** | 4.48 | **−1.53** | **−0.84** |
| Gem bankrupt | **1.00** | 19 / 54 | 0.487 | 0.504 | 1.09 | 1.28 | −0.28 | −0.09 |

Чтение:

- **DS default** 3.50 попадает в бакет 3★ (`[3.30, 3.80)` → raw 0.80×). Internal SQ 3.60 почти совпадает с public (qual. gap −0.10). Dem× 0.81 — лучший traffic-множитель в матрице; cold-start trust ещё не полностью снят (n=24, h=20 → c=0.55).
- **Gem default** 2.95 — бакет 2★ (`[2.50, 3.30)` → raw 0.35×), поэтому Dem× 0.53 при **лучшем** SQ 3.79. Самый большой selection gap (−1.53): операционно магазин ближе к 4★ (full-response 4.48), публично выглядит как 3★-ниже. Согласуется с anomaly 47% и штрафами 703 vs DS 93 / anomaly 22%.
- **Оба bankrupt** public = 1.00, SQ ≈ 1.1. У DS bankrupt больше отзывов (n=42 vs 19) → выше confidence в плохой оценке → сильнее штраф к спросу (Dem× 0.365 vs 0.504). Gemini bankrupt не добил депозит, но репутацию всё равно укатил в пол.

Прогноз **только для default-ячеек** (mean по 2×2 смешивает банкротства и content-filter no-op):

| | 30d USD / wall | 90d | 365d |
|---|---|---|---|
| DS default | 0.91 / 1.07h | 2.73 / 3.21h | 11.06 / 13.0h |
| Gem default | 2.85 / 1.16h | 8.55 / 3.48h | 34.69 / 14.1h |

Caveat: first-wakeup, cache, compaction; Gemini default ~3× дороже DeepSeek на том же 30д.

`provider=` в Hermes-логах пустой/`unknown` у обоих Gemini; pin `google-vertex/global` записан в run-local `config.yaml`. DeepSeek-прогоны имеют `coreweave/fp8`.

## 2026-08-15 — Эпоха v5: margin-consistent synthetic + калибровка спроса + офлайн-абляции

### TL;DR

- Дефолтный синтетический каталог теперь **v5 both-fixes**: `pricing_model: margin_consistent_v1` + калиброванный `base_demand: [0.02, 1.02]`. Hermes-оверлеи (`env/scenarios/agents/hermes*.yaml`) наследуют это из `default.yaml`.
- Очереди Hermes **не** перезапускались. Ctx-матрица 2026-08-12 — измерение экономики v4; выводы по окнам по-прежнему предварительные.
- Две оси аномалии (маржа × плотность спроса) разделены **каталожным CES** (`generate()`) и **1-дневным in-process policy-тестом** (rule_based markup 2×cost через `create_app` / `/runs` / `/step`). Live rule_based 7d очередь собрана, но **не** запускалась.

### 1. Что изменилось в генераторе

Источник: `env/data/synth.py`, `env/data/generation_profiles.py`, `env/scenarios/default.yaml`, `env/data/economy_diagnostics.py`.

- `generation_params.pricing_model`: `margin_consistent_v1` (дефолт) vs `legacy_anchor_at_cost`.
- Формулы v5: сэмпл категорийной `retail_margin` m → `cost = ref·(1−m)`, `ε = ref/(ref−cost)`. `cost_and_elasticity_from_margin` сначала клипает m в `[MIN_RETAIL_MARGIN, MAX_RETAIL_MARGIN]` (m=0.99 → 0.50 / ε=2, а не ε≈1.1 и 91% маржа), затем клипы `ε ∈ [1.10, 6]`. `ref_price` становится CES-оптимумом (`p* = ε/(ε−1)·c`).
- `generation_params.base_demand: [0.02, 1.02]` вместо захардкоженного `U(1, 50)`. `data.small_share` по-прежнему `1.0`.
- Цель калибровки: **0.52** listing-day demand при `sale=ref`, lifecycle=1, rating=1 → ~26 shop-day при 50 листингах (paper human). Seed 42, измеренный mean listing-day demand **0.528**.
- Порядок RNG не менялся: один draw спроса, один draw margin-или-elasticity (`synth.py`: `ref_price` → `base_demand` → operational → risk → rating → ровно один elasticity-or-margin draw → `market_curve`).

### 2. Сценарии абляций

`env/scenarios/ablations/{pricing_only,demand_only,both,legacy_v4}.yaml` — оверлеи поверх `default.yaml`.

| Overlay | `pricing_model` | `base_demand` | Назначение |
|---|---|---|---|
| `both` (= default) | `margin_consistent_v1` | `[0.02, 1.02]` | обе правки |
| `pricing_only` | `margin_consistent_v1` | `[1.0, 50.0]` | только ось маржи |
| `demand_only` | `legacy_anchor_at_cost` | `[0.02, 1.02]` | только ось объёма |
| `legacy_v4` | `legacy_anchor_at_cost` | `[1.0, 50.0]` | контроль = экономика ctx-матрицы |

Очередь (собрана, **не запускалась**): `scripts/batch_queue_rule_ablations.yaml` — rule_based 7d, три джоба (`pricing_only` / `demand_only` / `both`; `legacy_v4` в очередь не входит). Не запускать, пока Architect явно не попросит.

### 3. Офлайн-таблица абляций (каталог, не policy)

Seed 42, 1000 SKU, `small_share=1`, lifecycle=1, rating=1. Это **каталожная** CES listing-day абляция, не booked-only SQL и не live 7d. Источник: `generate()` + `catalog_economy_aggregates` (`env/data/economy_diagnostics.py`, `tests/test_synth_v5_ablations.py`). Числа — округление из `generate()` seed 42.

| Метрика | both / default | pricing_only | demand_only | legacy_v4 |
|---|---:|---:|---:|---:|
| share_eps_lt_1 | 0.000 | 0.000 | 0.100 | 0.100 |
| mean_margin_at_ref | 0.325 | 0.325 | 0.000 | 0.000 |
| mean_q_day_at_ref | 0.528 | 25.91 | 0.528 | 25.91 |
| mean_q_day_at_rule_markup | 0.214 | 10.48 | 0.207 | 10.17 |
| mean_gross_day_at_ref | 42.39 | 2078.53 | 0.00 | 0.00 |
| mean_gross_day_at_rule_markup | 34.71 | 1701.88 | 51.30 | 2515.63 |
| mean_gross_day_at_10x_cost | 3.09 | 151.35 | 60.15 | 2949.62 |

Интерпретация (причинное чтение, не корреляция):

- **Ось объёма:** pricing_only q̄_ref ≈ 49× both (старый `U(1,50)` vs калибровка).
- **Ось маржи:** при `sale=ref` legacy-миры дают 0 gross (`cost=ref`); v5 — ~32.5% маржа и положительный gross.
- **Эксплойт** (только appliances, 10× cost): demand_only 164.33 > at_ref 0; both 8.17 < rule_markup 48.68 (завышение цены за CES-оптимумом бьёт по gross, как только `ε≥2`).
- Оси **разделимы**. Дефолт v5 убирает `ε<1` и режет listing-day volume ~50× при ref. Rule-based 2×cost близок к оптимуму в v5 (gross 34.7 vs 42.4 при ref), но в `legacy_v4` 2×cost — money printer (2516 gross/day/listing).
- Это **не** booked-only SQL по `state.db`. Канонический фильтр без `stockout`/`insufficient_balance` применяется в 1-day policy-тесте ниже; live 7d всё ещё не гонялся.

### 3.1 In-process policy, 1 sim-day (rule_based 2×cost)

Источник: `tests/test_synth_v5_ablation_policy.py` (паттерн `create_app` / `/runs` / `/step`, как smoke). 100 SKU, 10 листингов `P00000`–`P00009`, `sale=round(cost*2.00, 2)`, horizon 24, lifecycle снят, seed 42, `initial_cash=100000` чтобы volume-ось не утонула в `insufficient_balance`. Booked-only:

`SELECT COALESCE(SUM(sale_price),0), COALESCE(SUM(sale_price-purchase_price),0), COUNT(*) FROM orders WHERE run_id=? AND current_status NOT IN ('stockout','insufficient_balance')`

Это **не** live 7d batch и **не** Hermes.

| Метрика | pricing_only | demand_only | both |
|---|---:|---:|---:|
| booked_count | 98 | 5 | 5 |
| booked_gmv | 31541.64 | 3413.23 | 2002.88 |
| booked_gross | 15770.79 | 1706.62 | 1001.44 |
| all_count | 103 | 6 | 6 |
| violation_count | 5 | 1 | 1 |

- **Ось объёма (policy):** pricing_only booked_count 98 > both 5, GMV 31542 > 2003 (старый `U(1,50)` vs калибровка, те же v5 цены).
- **Ось маржи (policy):** both и demand_only дают положительный booked gross при 2×cost; demand_only GMV выше both при том же count (2×ref vs v5 2×cost ближе к оптимуму).
- Каталог в прогоне: both / pricing_only без `ε<1`; demand_only имеет `ε<1`.

### 4. Тесты

- `tests/test_synth_pricing_v5.py` — `cost < ref`, `ε = ref/(ref−cost)`, appliances `ε≥1`, legacy overlay, общий RNG-префикс.
- `tests/test_demand_amplitude.py` — mean q_day_at_ref в `[0.40, 0.65]`, seed-42 lock `≈0.528308±0.01`, shop-day ~26, явный legacy `U(1,50)`.
- `tests/test_synth_v5_ablations.py` — четыре оверлея, exploit-probe appliances-only (каталог).
- `tests/test_synth_v5_ablation_policy.py` — 1 sim-day in-process markup policy, booked-only.
- `tests/test_run_batch_script.py` грузит `scripts/batch_queue_rule_ablations.yaml` (очередь не исполняется).
- Smoke `_tiny_scenario` (`tests/test_smoke.py`) по-прежнему форсит legacy `U(1,50)`, чтобы Poisson-smokes оставались плотными.

### 5. Открыто

- Live rule_based 7d очередь (`scripts/batch_queue_rule_ablations.yaml`) **не запущена**. In-process 1-day policy-тест есть; 7d P&L — нет.
- Hermes ctx-матрица на v5 **не** перегонялась.
- Incomplete seed-43 (200k `bf025e`, 1M `3cbcae`) по-прежнему ждёт.
- Hard cap цены и прогрессивные штрафы — не сделаны (план v4, п.3–4).
- Датасеты (JDsearch + Olist гибрид) — позже (план v4, п.7).
- 2 seeds × 7д калибровка → paired ctx-матрица → red-team — открыто (план v4, п.6).

---

## 2026-08-12 — Эпоха v4: pricing-патология синтетической экономики

*Числа и findings ниже относятся к каталогу v4 (`price==ref`, `base_demand ~ U(1,50)`). Дефолт генератора с 2026-08-15 — v5; см. секцию выше. Hermes-прогоны этой эпохи **не** пересчитывались.*

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

1. **Synthetic v5 — margin-consistent генерация:** сэмплировать целевую retail-маржу по категориям → согласованно derive `cost`, `ref_price`, `elasticity` через `ε = ref/(ref−cost)` (образец — `build_private_real_db.py:1860`). `ref_price` становится теоретическим оптимумом; бесконечный хвост исчезает по построению. (сделано, см. эпоху v5 2026-08-15)
2. **Калибровка амплитуды спроса** по `orders / active-listing-day` к референсу paper (human ~26/день), вместо `U(1,50)` на товар. (сделано, см. эпоху v5 2026-08-15)
3. **Фиксированные штрафы сохранить** (сопоставимость с paper); цены вернуть в масштаб, где 3–8 RMB значимы. Прогрессивные штрафы — не первый кандидат.
4. **Hard cap цены** — только как явный видимый агенту guardrail (per-product cap не выразим статичной JSON-схемой; `ref_price` скрыт → непредсказуемые отказы).
5. **3 policy-level ablations без LLM** (rule-based): (a) только исправленная `cost/ref/ε`; (b) только уменьшенный demand scale; (c) обе правки. Разделит оси аномалии дёшево. (сделано, см. эпоху v5 2026-08-15: офлайн `generate()` + in-process 1-day policy test; live 7d очередь не запускалась)
6. **2 seeds × 7д** калибровка → paired ctx-матрица (3×3, лучше 5 seeds) → **red-team как регрессионный тест до/после** фикса.
7. **Датасеты** (JDsearch + Olist гибрид) — после стабилизации unit economics; кривые спроса закладывать в калибровку п.2.

### 6. Открытые вопросы

- Добивка 2 incomplete seed-43 (200k `bf025e`, 1M `3cbcae`) — **только после** стабилизации экономики, иначе измерят старый дефект. ≤3–4 parallel.
- Per-day динамика цен агента (как доходил до near-optimal markup) — не анализировалась; источник: `events` / история `adjust_price` в `state.db`.
- Предложен, но не реализован постоянный инструмент `scripts/analyze_pricing.py` (booked-only учёт, CLI, тесты) — кандидат для Middle SWE.
- Какая ось (маржа vs плотность спроса) доминирует — ждёт ablations. (сделано, см. эпоху v5 2026-08-15: оси разделимы офлайн и на 1-day booked policy; live 7d P&L ещё нет)

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

*Следующая секция — после live rule_based 7d и/или повторной ctx-матрицы на v5.*
