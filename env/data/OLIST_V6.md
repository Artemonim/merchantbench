# Olist v6 catalog

Real-catalog track for MerchantBench v6. The synthetic paper-track stays on
`scenarios/default.yaml`. This file is attribution only; the CSVs and the
full sqlite live under `data/private_data/` (gitignored).

## License

Olist Brazilian E-Commerce Public Dataset is published as
**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike).
Do not redistribute a commercial build of this catalog without checking that
license. Attribution: Olist and the public Kaggle release
`olistbr/brazilian-ecommerce`.

## Build

From `env/` (PYTHONPATH is the `env` package root):

```
python -m data.build_olist_v6 --output-db data/private_data/olist_v6.sqlite
```

CSVs are cached in `data/private_data/olist_csv/`. Re-running is idempotent
when those files already have the official headers. Mirrors, in order:

1. HuggingFace `debs-b/ecommerce-brazil` (may 401 without `HF_TOKEN`)
2. GitHub `Kaaykun/OlistAnalysis` raw CSVs
3. jsDelivr of the same repo
4. GitHub `mohamedyounis10/Olist-brazilian-ecommerce-analytics`

If every Olist mirror fails, the CLI tries the UCI Online Retail II zip and
then **stops** rather than inventing SKUs. No Kaggle account is required.

## Mapping

| Product field | Source |
| --- | --- |
| `ref_price` | Median observed `order_items.price` (WTP). Not recomputed from `(c+F)/(1-τ)`. |
| `price` / cost | `ref * (1 - margin)` via `cost_and_elasticity_from_margin` |
| `elasticity` | Same helper, clipped to the v5 band |
| `category` | English Olist name → `default.yaml` `category_pool`; unknown → `home_goods` |
| `name` | Generated marketplace-style title from `data/product_titles.py` (Olist has no titles); separate `derive_rng` key appended after all other draws, so numeric fields are rebuild-stable. Typo injection: `--typo-rate` (default 0.0). Pre-2026-08-23 builds used `English category + short product id` |
| `market_curve` | Daily order counts on the shared calendar, tiled or `resample_periodic_curve` to 365, then scaled into the calibrated demand range. Empty history → constant `0.02` floor, **not** the synth sine. |
| `historical_avg_rating` | Mean review score (1–5); default 4.0 if none |
| `shop_rating` / `return_buyer_rate` / `supplier_age_years` | Per seller, copied to every SKU of that seller |
| `ship_hours` / `logistics_hours` | approved→carrier and carrier→customer, clamped to `supplier_ranges` |
| `cancel_rate` / refund-like rates | Order status + low-star reviews when the SKU has enough history; otherwise sampled rates biased like `apply_risk_trust_coupling` |
| inventory / timeout / delist / price_change | `derive_rng(build_seed, "data_gen", "olist_v6_product", …)` |

`olist_customers_dataset.csv` is optional. When present, `return_buyer_rate`
uses `customer_unique_id` repeats; otherwise it is a monotone map from
`shop_rating` onto `supplier_profile_ranges.return_buyer_rate`.

## Run-time subsample

`data.source: private_real` loads the full pool, then keeps the first
`data.num_products` IDs after

`derive_rng(master_seed, "data_gen", "catalog_subsample")`.

Same seed ⇒ same assortment. Smaller N is a prefix of larger N at that seed.
Synthetic `generate()` already emits exactly N and is not shuffled again.

Scenario: `scenarios/economy_v6.yaml`. Ablation `ablations/economy_v6_both.yaml`
stays on the synthetic catalog.
