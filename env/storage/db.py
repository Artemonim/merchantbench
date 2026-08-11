"""SQLite schema + thin DAO. Runtime uses one SQLite DB per run.

Tables follow spec section 7. JSON columns stored as TEXT.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import urllib.parse
from dataclasses import astuple
from datetime import datetime, timezone
from typing import Iterable, Optional

log = logging.getLogger(__name__)

from core.entities import (
    Agent,
    Cash,
    EventLog,
    Order,
    OrderStatusRow,
    Product,
    StoreListing,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  name TEXT,
  scenario_yaml TEXT,
  master_seed INTEGER,
  current_t INTEGER,
  pending_hook_t INTEGER,
  pending_hook_closed INTEGER DEFAULT 0,
  horizon INTEGER,
  step_hours INTEGER,
  status TEXT,
  started_at TEXT,
  bootstrap_agent TEXT DEFAULT 'none',
  bootstrap_config_json TEXT DEFAULT '{}',
  initial_quantity_provenance TEXT DEFAULT 'native',
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS agents (
  run_id TEXT,
  agent_id TEXT,
  name TEXT,
  created_at TEXT,
  is_alive INTEGER DEFAULT 1,
  died_at_t INTEGER,
  PRIMARY KEY (run_id, agent_id)
);

CREATE TABLE IF NOT EXISTS products (
  run_id TEXT, product_id TEXT,
  name TEXT, quantity INTEGER, initial_quantity INTEGER, quantity_updated_t INTEGER DEFAULT 0,
  price REAL, ref_price REAL, base_price REAL,
  supplier_id TEXT, supplier_name TEXT,
  ship_hours INTEGER, base_ship_hours INTEGER, supplier_ship_hours INTEGER,
  logistics_hours INTEGER,
  category TEXT,
  historical_avg_rating REAL, shop_rating REAL,
  return_buyer_rate REAL, supplier_age_years REAL,
  cancel_rate REAL, refund_rate REAL, only_refund_rate REAL,
  bad_review_rate REAL,
  max_quantity INTEGER, hourly_increment INTEGER,
  timeout_rate REAL, price_change_rate REAL, supplier_delist_rate REAL,
  elasticity REAL,
  is_listed_by_supplier INTEGER, delist_recover_t INTEGER, price_recover_t INTEGER,
  timeout_active INTEGER DEFAULT 0, timeout_recover_t INTEGER,
  market_curve TEXT,
  PRIMARY KEY (run_id, product_id)
);

CREATE INDEX IF NOT EXISTS ix_products_run_category
  ON products(run_id, category, product_id);
CREATE INDEX IF NOT EXISTS ix_products_run_supplier
  ON products(run_id, supplier_id, product_id);
CREATE INDEX IF NOT EXISTS ix_products_run_visible_price
  ON products(run_id, is_listed_by_supplier, price, product_id);
CREATE INDEX IF NOT EXISTS ix_products_run_visible_rating
  ON products(run_id, is_listed_by_supplier, historical_avg_rating, shop_rating, product_id);

CREATE VIRTUAL TABLE IF NOT EXISTS catalog_search_fts USING fts5(
  run_id UNINDEXED,
  product_id UNINDEXED,
  search_text,
  tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS store_listings (
  run_id TEXT, agent_id TEXT, product_id TEXT,
  sale_price REAL,
  cum_sales INTEGER DEFAULT 0, cum_revenue REAL DEFAULT 0,
  listed_at INTEGER,
  first_listed_at INTEGER,
  normal_count INTEGER DEFAULT 0,
  bad_review_count INTEGER DEFAULT 0,
  rating_sum REAL DEFAULT 0,
  rating_count REAL DEFAULT 0,
  promised_ship_hours INTEGER,
  promised_logistics_hours INTEGER,
  PRIMARY KEY (run_id, agent_id, product_id)
);

CREATE TABLE IF NOT EXISTS orders (
  run_id TEXT, order_id TEXT,
  agent_id TEXT,
  product_id TEXT, supplier_id TEXT,
  order_t INTEGER, promised_delivery_t INTEGER,
  sale_price REAL, purchase_price REAL,
  current_status TEXT,
  purchase_t INTEGER, shipped_t INTEGER, delivered_t INTEGER, settled_t INTEGER,
  preset_anomaly TEXT, preset_anomaly_t INTEGER,
  promised_ship_hours INTEGER DEFAULT 0,
  supplier_ship_hours INTEGER DEFAULT 0,
  actual_ship_hours INTEGER DEFAULT 0,
  promised_logistics_hours INTEGER DEFAULT 0,
  actual_logistics_hours INTEGER DEFAULT 0,
  late_t INTEGER,
  realized_revenue REAL DEFAULT 0,
  realized_cost REAL DEFAULT 0,
  total_penalty REAL DEFAULT 0,
  settlement_delay_steps INTEGER DEFAULT -1,
  PRIMARY KEY (run_id, order_id)
);

CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(run_id, current_status);
CREATE INDEX IF NOT EXISTS ix_orders_product ON orders(run_id, product_id);
CREATE INDEX IF NOT EXISTS ix_orders_order_t ON orders(run_id, order_t);
CREATE INDEX IF NOT EXISTS ix_orders_agent ON orders(run_id, agent_id);
CREATE INDEX IF NOT EXISTS ix_orders_run_agent_t_product
  ON orders(run_id, agent_id, order_t, product_id);
CREATE INDEX IF NOT EXISTS ix_orders_run_agent_status
  ON orders(run_id, agent_id, current_status);
CREATE INDEX IF NOT EXISTS ix_orders_run_agent_settled_product
  ON orders(run_id, agent_id, settled_t, product_id);

CREATE TABLE IF NOT EXISTS order_status (
  run_id TEXT, order_id TEXT, t INTEGER, status TEXT,
  PRIMARY KEY (run_id, order_id, t, status)
);

CREATE INDEX IF NOT EXISTS ix_order_status_run_order ON order_status(run_id, order_id);
CREATE INDEX IF NOT EXISTS ix_order_status_run_t ON order_status(run_id, t);

CREATE TABLE IF NOT EXISTS cash_log (
  run_id TEXT, agent_id TEXT, t INTEGER,
  balance REAL, deposit_pool REAL, in_transit REAL, receivable REAL, cumulative_fine REAL,
  PRIMARY KEY (run_id, agent_id, t)
);

CREATE TABLE IF NOT EXISTS events (
  run_id TEXT, t INTEGER, event_type TEXT, entity_id TEXT,
  agent_id TEXT, payload TEXT
);

CREATE INDEX IF NOT EXISTS ix_events_run_t ON events(run_id, t);
CREATE INDEX IF NOT EXISTS ix_events_agent ON events(run_id, agent_id);
CREATE INDEX IF NOT EXISTS ix_events_run_type_t ON events(run_id, event_type, t);
CREATE INDEX IF NOT EXISTS ix_events_run_entity_type_t
  ON events(run_id, entity_id, event_type, t);

CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT, agent_id TEXT, t INTEGER, key TEXT, value REAL,
  PRIMARY KEY (run_id, agent_id, t, key)
);

CREATE INDEX IF NOT EXISTS ix_metrics_lookup ON metrics(run_id, agent_id, key, t);

CREATE TABLE IF NOT EXISTS daily_aggregates (
  run_id TEXT, day INTEGER,
  gmv REAL, anomaly_count INTEGER, fine_total REAL,
  PRIMARY KEY (run_id, day)
);

CREATE TABLE IF NOT EXISTS hourly_dist (
  run_id TEXT, category TEXT, hour INTEGER, w REAL,
  PRIMARY KEY (run_id, category, hour)
);

CREATE TABLE IF NOT EXISTS supplier_events (
  run_id TEXT,
  due_t INTEGER,
  product_id TEXT,
  event_type TEXT,
  seq INTEGER,
  payload TEXT DEFAULT '{}',
  PRIMARY KEY (run_id, due_t, product_id, event_type, seq)
);

CREATE INDEX IF NOT EXISTS ix_supplier_events_due
  ON supplier_events(run_id, due_t);
CREATE INDEX IF NOT EXISTS ix_supplier_events_product_type
  ON supplier_events(run_id, product_id, event_type);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def open_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _drop_legacy_product_search(conn)
    _migrate_runs_schema(conn)
    _migrate_products_schema(conn)
    _migrate_ship_sla_schema(conn)
    _migrate_lifecycle_schema(conn)
    _migrate_listing_rating_schema(conn)
    _ensure_catalog_search_fts(conn)
    _ensure_analyze(conn)
    return conn


def open_db_readonly(path: str) -> sqlite3.Connection:
    """Open a database for read-only operations without schema migrations.

    This is a lightweight alternative to open_db() for read-only operations
    like leaderboard queries. It skips schema creation, migrations, and ANALYZE
    because:
    1. Historical runs already have up-to-date schemas
    2. ANALYZE has already been run on populated databases
    3. These operations are expensive and unnecessary for read-only access

    The connection is still configured with PRAGMAs for proper operation.
    """
    if not os.path.exists(path):
        raise sqlite3.OperationalError(f"database file does not exist: {path}")
    encoded_path = urllib.parse.quote(path, safe='/')
    conn = sqlite3.connect(f"file:{encoded_path}?mode=ro", uri=True,
                           check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_analyze(conn: sqlite3.Connection) -> None:
    """Run ANALYZE if stats are missing and the DB is populated.

    Without sqlite_stat1, SQLite's query planner makes poor index choices
    for per-run databases where run_id is non-selective (only one value).
    """
    has_stat = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
    ).fetchone()
    if has_stat:
        return
    # Only ANALYZE if there's meaningful data (avoid overhead on empty DBs).
    row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
    if (row["n"] if row else 0) > 0:
        conn.execute("ANALYZE")


def _drop_legacy_product_search(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS product_search")


def _migrate_runs_schema(conn: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "pending_hook_t" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN pending_hook_t INTEGER")
    if "pending_hook_closed" not in columns:
        conn.execute(
            "ALTER TABLE runs ADD COLUMN pending_hook_closed INTEGER DEFAULT 0"
        )
        conn.execute(
            "UPDATE runs SET pending_hook_closed=0"
            " WHERE pending_hook_closed IS NULL"
        )
    if "bootstrap_config_json" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN bootstrap_config_json TEXT DEFAULT '{}'")
        conn.execute(
            "UPDATE runs SET bootstrap_config_json='{}'"
            " WHERE bootstrap_config_json IS NULL"
        )
    if "initial_quantity_provenance" not in columns:
        conn.execute(
            "ALTER TABLE runs ADD COLUMN initial_quantity_provenance TEXT"
            " DEFAULT 'legacy_quantity_fallback'"
        )
        conn.execute(
            "UPDATE runs SET initial_quantity_provenance='legacy_quantity_fallback'"
            " WHERE initial_quantity_provenance IS NULL"
        )


def _migrate_products_schema(conn: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(products)").fetchall()}
    if "base_price" not in columns:
        conn.execute("ALTER TABLE products ADD COLUMN base_price REAL")
        conn.execute("UPDATE products SET base_price=price WHERE base_price IS NULL")
    if "base_ship_hours" not in columns:
        conn.execute("ALTER TABLE products ADD COLUMN base_ship_hours INTEGER")
        conn.execute("UPDATE products SET base_ship_hours=ship_hours WHERE base_ship_hours IS NULL")
    if "supplier_ship_hours" not in columns:
        conn.execute("ALTER TABLE products ADD COLUMN supplier_ship_hours INTEGER")
        conn.execute(
            "UPDATE products SET supplier_ship_hours=COALESCE(base_ship_hours, ship_hours)"
            " WHERE supplier_ship_hours IS NULL"
        )
    if "quantity_updated_t" not in columns:
        conn.execute("ALTER TABLE products ADD COLUMN quantity_updated_t INTEGER DEFAULT 0")
        conn.execute("UPDATE products SET quantity_updated_t=0 WHERE quantity_updated_t IS NULL")
    if "initial_quantity" not in columns:
        conn.execute("ALTER TABLE products ADD COLUMN initial_quantity INTEGER")
        conn.execute(
            "UPDATE products SET initial_quantity=quantity WHERE initial_quantity IS NULL"
        )


def _migrate_ship_sla_schema(conn: sqlite3.Connection) -> None:
    listing_columns = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(store_listings)").fetchall()
    }
    if "promised_ship_hours" not in listing_columns:
        conn.execute("ALTER TABLE store_listings ADD COLUMN promised_ship_hours INTEGER")
        if "promised_logistics_hours" in listing_columns:
            conn.execute(
                "UPDATE store_listings SET promised_ship_hours=promised_logistics_hours"
                " WHERE promised_ship_hours IS NULL"
            )

    order_columns = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(orders)").fetchall()
    }
    if "promised_ship_hours" not in order_columns:
        conn.execute("ALTER TABLE orders ADD COLUMN promised_ship_hours INTEGER DEFAULT 0")
        if "promised_logistics_hours" in order_columns:
            conn.execute(
                "UPDATE orders SET promised_ship_hours=promised_logistics_hours"
                " WHERE promised_ship_hours IS NULL OR promised_ship_hours=0"
            )
    if "actual_ship_hours" not in order_columns:
        conn.execute("ALTER TABLE orders ADD COLUMN actual_ship_hours INTEGER DEFAULT 0")
    if "supplier_ship_hours" not in order_columns:
        conn.execute("ALTER TABLE orders ADD COLUMN supplier_ship_hours INTEGER DEFAULT 0")
        conn.execute(
            "UPDATE orders SET supplier_ship_hours=COALESCE(NULLIF(actual_ship_hours, 0), 0)"
            " WHERE supplier_ship_hours IS NULL OR supplier_ship_hours=0"
        )


def _migrate_lifecycle_schema(conn: sqlite3.Connection) -> None:
    listing_columns = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(store_listings)").fetchall()
    }
    if "first_listed_at" not in listing_columns:
        conn.execute("ALTER TABLE store_listings ADD COLUMN first_listed_at INTEGER")
        conn.execute(
            "UPDATE store_listings SET first_listed_at=listed_at"
            " WHERE first_listed_at IS NULL"
        )


def _migrate_listing_rating_schema(conn: sqlite3.Connection) -> None:
    listing_columns = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(store_listings)").fetchall()
    }
    added = False
    if "rating_sum" not in listing_columns:
        conn.execute("ALTER TABLE store_listings ADD COLUMN rating_sum REAL DEFAULT 0")
        added = True
    if "rating_count" not in listing_columns:
        conn.execute("ALTER TABLE store_listings ADD COLUMN rating_count REAL DEFAULT 0")
        added = True
    if added:
        conn.execute(
            "UPDATE store_listings"
            " SET rating_sum=5.0 * COALESCE(normal_count, 0)"
            "              + 1.0 * COALESCE(bad_review_count, 0),"
            "     rating_count=COALESCE(normal_count, 0)"
            "                + COALESCE(bad_review_count, 0)"
            " WHERE COALESCE(rating_count, 0)=0"
            "   AND (COALESCE(normal_count, 0) + COALESCE(bad_review_count, 0)) > 0"
        )


# ---------- runs ----------

def insert_run(conn, run_id: str, name: str, scenario_yaml: str, master_seed: int,
               horizon: int, step_hours: int, started_at: str,
               bootstrap_agent: str = "none",
               bootstrap_config: Optional[dict] = None) -> None:
    bootstrap_config_json = json.dumps(bootstrap_config or {}, ensure_ascii=False)
    conn.execute(
        "INSERT INTO runs(run_id, name, scenario_yaml, master_seed, current_t,"
        " horizon, step_hours, status, started_at, bootstrap_agent,"
        " bootstrap_config_json, initial_quantity_provenance)"
        " VALUES (?, ?, ?, ?, 0, ?, ?, 'pending', ?, ?, ?, 'native')",
        (
            run_id, name, scenario_yaml, master_seed, horizon, step_hours,
            started_at, bootstrap_agent, bootstrap_config_json,
        ),
    )


def get_bootstrap_agent(conn, run_id: str) -> str:
    r = conn.execute("SELECT bootstrap_agent FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return (r[0] if r else "none") or "none"


def get_bootstrap_config(conn, run_id: str) -> dict:
    r = conn.execute(
        "SELECT bootstrap_config_json FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if not r:
        return {}
    try:
        cfg = json.loads(r["bootstrap_config_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def get_initial_quantity_provenance(conn, run_id: str) -> str:
    row = conn.execute(
        "SELECT initial_quantity_provenance FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if not row:
        return "legacy_quantity_fallback"
    return str(row[0] or "legacy_quantity_fallback")


def list_runs(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT run_id, name, master_seed, current_t, horizon, step_hours,"
        " status, started_at, finished_at, bootstrap_agent, bootstrap_config_json"
        " FROM runs ORDER BY started_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_run(conn, run_id: str) -> Optional[dict]:
    r = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(r) if r else None


def get_run_lightweight(path: str, run_id: str) -> Optional[dict]:
    """Read run metadata without schema initialization.

    For list_runs() which needs to scan many run directories quickly.
    Skips schema DDL, migrations, and ANALYZE — just opens a read-only
    connection and queries the runs table directly.

    Note: Uses raw sqlite3 connection instead of LockedConnection because:
    - Read-only mode (no writes that could conflict)
    - Short-lived (single query, immediate close)
    - Not shared across threads (created and used in same call)
    This avoids circular imports between storage.db and web.runner.

    Returns None only for genuinely missing runs (no such table / no row).
    Database corruption is logged and re-raised so callers (e.g. delete_run)
    can still clean up the directory.
    """
    if not os.path.exists(path):
        return None
    try:
        encoded_path = urllib.parse.quote(path, safe='/')
        conn = sqlite3.connect(f"file:{encoded_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            r = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(r) if r else None
        except sqlite3.OperationalError as e:
            # "no such table: runs" — run directory exists but DB was never
            # initialized or was partially deleted. Treat as missing.
            if "no such table" in str(e):
                return None
            log.warning("get_run_lightweight operational error for %s: %s", path, e)
            raise
        except sqlite3.DatabaseError as e:
            # Corruption, not-a-database, WAL recovery failure — surface so
            # callers can distinguish from "not found" and still clean up.
            log.warning("get_run_lightweight database error for %s: %s", path, e)
            raise
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        # Cannot open file at all (e.g. deleted between exists check and open).
        if "unable to open" in str(e):
            return None
        log.warning("get_run_lightweight cannot open %s: %s", path, e)
        raise


def update_run_t(conn, run_id: str, t: int) -> None:
    conn.execute("UPDATE runs SET current_t=? WHERE run_id=?", (t, run_id))


def mark_step_transition_committed(conn, run_id: str, t: int) -> None:
    """Record that transition ``t`` is durable and its hook remains pending."""
    conn.execute(
        "UPDATE runs SET pending_hook_t=?, pending_hook_closed=0 WHERE run_id=?",
        (int(t), run_id),
    )


def mark_step_hook_closed(conn, run_id: str, t: int) -> None:
    """Record that ``t`` completed its hook and only finalization remains."""
    cursor = conn.execute(
        "UPDATE runs SET pending_hook_closed=1"
        " WHERE run_id=? AND pending_hook_t=?",
        (run_id, int(t)),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(
            f"cannot close hook without pending transition: run_id={run_id}, t={t}"
        )


def finish_committed_step(conn, run_id: str, next_t: int) -> None:
    """Advance the durable clock only after the committed step's hook closes."""
    conn.execute(
        "UPDATE runs SET current_t=?, pending_hook_t=NULL, pending_hook_closed=0"
        " WHERE run_id=?",
        (int(next_t), run_id),
    )


def update_run_status(conn, run_id: str, status: str) -> None:
    conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))


def update_run_finished_at(conn, run_id: str, finished_at: str) -> None:
    conn.execute("UPDATE runs SET finished_at=? WHERE run_id=?", (finished_at, run_id))


def mark_run_running(conn, run_id: str) -> None:
    """Start a new active session and clear the previous terminal timestamp."""
    conn.execute(
        "UPDATE runs SET status='running', finished_at=NULL WHERE run_id=?",
        (run_id,),
    )


def mark_run_terminal(conn, run_id: str, status: str, finished_at: str) -> None:
    """Persist a terminal lifecycle transition atomically."""
    if status not in ("stopped", "finished"):
        raise ValueError(f"invalid terminal run status: {status}")
    conn.execute(
        "UPDATE runs SET status=?, finished_at=COALESCE(finished_at, ?)"
        " WHERE run_id=?",
        (status, finished_at, run_id),
    )


def stop_orphaned_run(path: str, run_id: str, stopped_at: str) -> bool:
    """Demote a live state left behind by a previous server process."""
    if not os.path.exists(path):
        return False
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        cur = conn.execute(
            "UPDATE runs SET status='stopped', finished_at=?"
            " WHERE run_id=? AND status IN ('running', 'draining')",
            (stopped_at, run_id),
        )
        return cur.rowcount > 0
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return False
        raise
    finally:
        conn.close()


ACTIVE_ORDER_STATUSES = ("ordered", "late", "shipped", "delivered")


def active_order_status_counts(conn, run_id: str) -> dict[str, int]:
    qmarks = ",".join("?" * len(ACTIVE_ORDER_STATUSES))
    rows = conn.execute(
        f"SELECT current_status, COUNT(*) AS n FROM orders"
        f" WHERE run_id=? AND current_status IN ({qmarks})"
        f" GROUP BY current_status",
        (run_id, *ACTIVE_ORDER_STATUSES),
    ).fetchall()
    return {str(r["current_status"]): int(r["n"]) for r in rows}


def delete_run(conn, run_id: str) -> None:
    """Hard-delete every row associated with this run_id across all tables.
    Child rows first; the `runs` row last so a partial failure leaves the run
    visible in the All-Runs listing for retry."""
    for table in ("supplier_events", "order_status", "orders", "cash_log", "events", "metrics",
                  "daily_aggregates", "hourly_dist", "store_listings",
                  "products", "agents"):
        conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))


# ---------- agents ----------

def insert_agent(conn, run_id: str, agent_id: str, name: str, created_at: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO agents(run_id, agent_id, name, created_at, is_alive, died_at_t)"
        " VALUES (?,?,?,?,1,NULL)",
        (run_id, agent_id, name, created_at),
    )


def list_agents(conn, run_id: str) -> list[Agent]:
    rows = conn.execute(
        "SELECT agent_id, name, created_at, is_alive, died_at_t FROM agents"
        " WHERE run_id=? ORDER BY created_at",
        (run_id,),
    ).fetchall()
    return [Agent(agent_id=r["agent_id"], name=r["name"], created_at=r["created_at"],
                  is_alive=bool(r["is_alive"]), died_at_t=r["died_at_t"]) for r in rows]


def mark_agent_dead(conn, run_id: str, agent_id: str, t: int) -> None:
    conn.execute(
        "UPDATE agents SET is_alive=0, died_at_t=? WHERE run_id=? AND agent_id=?",
        (t, run_id, agent_id),
    )


# ---------- products ----------

_PRODUCT_COLS = (
    "run_id", "product_id", "name", "quantity", "initial_quantity", "quantity_updated_t",
    "price", "ref_price", "base_price", "supplier_id", "supplier_name",
    "ship_hours", "base_ship_hours", "supplier_ship_hours", "logistics_hours", "category",
    "historical_avg_rating", "shop_rating",
    "return_buyer_rate", "supplier_age_years",
    "cancel_rate", "refund_rate", "only_refund_rate",
    "bad_review_rate",
    "max_quantity", "hourly_increment",
    "timeout_rate", "price_change_rate", "supplier_delist_rate",
    "elasticity",
    "is_listed_by_supplier", "delist_recover_t", "price_recover_t",
    "timeout_active", "timeout_recover_t",
    "market_curve",
)


_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _search_tokens(text: str) -> list[str]:
    text = str(text or "").lower()
    tokens = []
    for token in _ASCII_TOKEN_RE.findall(text):
        tokens.append(token)
        if len(token) > 3:
            tokens.extend(token[i:i + 3] for i in range(len(token) - 2))
    for match in _CJK_RUN_RE.finditer(text):
        segment = match.group(0)
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            tokens.extend(token for token in jieba.lcut(segment, cut_all=False) if len(token) >= 2)
        except Exception:
            pass
        tokens.extend(segment[i:i + 2] for i in range(max(0, len(segment) - 1)))
    return tokens


def _query_tokens(text: str) -> list[str]:
    text = str(text or "").lower()
    tokens = _ASCII_TOKEN_RE.findall(text)
    for match in _CJK_RUN_RE.finditer(text):
        segment = match.group(0)
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            tokens.extend(token for token in jieba.lcut(segment, cut_all=False) if len(token) >= 2)
        except Exception:
            pass
        tokens.extend(segment[i:i + 2] for i in range(max(0, len(segment) - 1)))
    return tokens


def _fts_query(query: str) -> str:
    terms = []
    seen: set[str] = set()
    raw_terms = re.findall(r"[\w]+", str(query or "").lower())
    for raw_term in raw_terms:
        for token in _query_tokens(raw_term):
            if token and token not in seen:
                terms.append('"' + token.replace('"', '""') + '"')
                seen.add(token)
    return " OR ".join(terms)


def _product_search_text(p: Product) -> str:
    return _product_search_text_values(
        p.product_id,
        p.name,
        p.category,
        p.supplier_name,
    )


def _product_search_text_values(
    product_id: str,
    name: str,
    category: str,
    supplier_name: str,
) -> str:
    visible_text = " ".join([
        str(product_id or ""),
        str(name or ""),
        str(category or ""),
        str(supplier_name or ""),
    ])
    return " ".join(_search_tokens(visible_text))


def _ensure_catalog_search_fts(conn: sqlite3.Connection) -> None:
    """Backfill the canonical catalog FTS index when opening legacy databases."""
    product_counts = {
        str(row["run_id"]): int(row["n"])
        for row in conn.execute(
            "SELECT run_id, COUNT(*) AS n FROM products GROUP BY run_id"
        ).fetchall()
    }
    fts_counts = {
        str(row["run_id"]): int(row["n"])
        for row in conn.execute(
            "SELECT run_id, COUNT(*) AS n FROM catalog_search_fts GROUP BY run_id"
        ).fetchall()
    }
    if product_counts == fts_counts:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM catalog_search_fts")
        cursor = conn.execute(
            "SELECT run_id, product_id, name, category, supplier_name"
            " FROM products ORDER BY run_id, product_id"
        )
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            conn.executemany(
                "INSERT INTO catalog_search_fts(run_id, product_id, search_text)"
                " VALUES (?, ?, ?)",
                [
                    (
                        row["run_id"],
                        row["product_id"],
                        _product_search_text_values(
                            row["product_id"],
                            row["name"],
                            row["category"],
                            row["supplier_name"],
                        ),
                    )
                    for row in rows
                ],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _replace_product_search_rows(
    conn,
    run_id: str,
    rows: list[tuple[str, str, str]],
) -> None:
    if not rows:
        return
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM products WHERE run_id=?",
        (run_id,),
    ).fetchone()["n"]
    if len(rows) >= int(total or 0):
        conn.execute("DELETE FROM catalog_search_fts WHERE run_id=?", (run_id,))
    else:
        product_ids = [product_id for _run_id, product_id, _text in rows]
        for i in range(0, len(product_ids), 500):
            chunk = product_ids[i:i + 500]
            qmarks = ",".join("?" for _ in chunk)
            conn.execute(
                f"DELETE FROM catalog_search_fts WHERE run_id=? AND product_id IN ({qmarks})",
                (run_id, *chunk),
            )
    conn.executemany(
        "INSERT INTO catalog_search_fts(run_id, product_id, search_text) VALUES (?, ?, ?)",
        rows,
    )


def insert_products(conn, run_id: str, products: Iterable[Product]) -> None:
    rows = []
    search_rows = []
    for p in products:
        rows.append((
            run_id, p.product_id, p.name, p.quantity, p.quantity,
            int(getattr(p, "quantity_updated_t", 0) or 0),
            p.price, p.ref_price, p.base_price, p.supplier_id, p.supplier_name,
            p.ship_hours, int(p.base_ship_hours), int(p.supplier_ship_hours),
            p.logistics_hours, p.category,
            p.historical_avg_rating, p.shop_rating,
            p.return_buyer_rate, p.supplier_age_years,
            p.cancel_rate, p.refund_rate, p.only_refund_rate,
            p.bad_review_rate,
            p.max_quantity, p.hourly_increment,
            p.timeout_rate, p.price_change_rate, p.supplier_delist_rate,
            p.elasticity,
            int(p.is_listed_by_supplier), p.delist_recover_t, p.price_recover_t,
            int(p.timeout_active), p.timeout_recover_t,
            json.dumps(p.market_curve),
        ))
        search_rows.append((run_id, p.product_id, _product_search_text(p)))
    cols_sql = ",".join(_PRODUCT_COLS)
    placeholders = ",".join("?" * len(_PRODUCT_COLS))
    conn.executemany(
        f"INSERT OR REPLACE INTO products({cols_sql}) VALUES ({placeholders})",
        rows,
    )
    _replace_product_search_rows(conn, run_id, search_rows)


def upsert_product_state(conn, run_id: str, p: Product) -> None:
    conn.execute(
        "UPDATE products SET quantity=?, quantity_updated_t=?, price=?, supplier_ship_hours=?,"
        " is_listed_by_supplier=?, delist_recover_t=?, price_recover_t=?,"
        " timeout_active=?, timeout_recover_t=?"
        " WHERE run_id=? AND product_id=?",
        (p.quantity, int(getattr(p, "quantity_updated_t", 0) or 0),
         p.price, int(p.supplier_ship_hours), int(p.is_listed_by_supplier),
         p.delist_recover_t, p.price_recover_t,
         int(p.timeout_active), p.timeout_recover_t,
         run_id, p.product_id),
    )


def upsert_product_states(conn, run_id: str, products: Iterable[Product]) -> None:
    rows = [
        (
            p.quantity, int(getattr(p, "quantity_updated_t", 0) or 0),
            p.price, int(p.supplier_ship_hours), int(p.is_listed_by_supplier),
            p.delist_recover_t, p.price_recover_t,
            int(p.timeout_active), p.timeout_recover_t,
            run_id, p.product_id,
        )
        for p in products
    ]
    if rows:
        conn.executemany(
            "UPDATE products SET quantity=?, quantity_updated_t=?, price=?,"
            " supplier_ship_hours=?, is_listed_by_supplier=?,"
            " delist_recover_t=?, price_recover_t=?,"
            " timeout_active=?, timeout_recover_t=?"
            " WHERE run_id=? AND product_id=?",
            rows,
        )


def _product_from_row(r, *, initial: bool = False) -> Product:
    keys = set(r.keys())
    quantity = r["quantity"]
    if initial and "initial_quantity" in keys and r["initial_quantity"] is not None:
        quantity = r["initial_quantity"]
    price = r["price"]
    if initial and "base_price" in keys and r["base_price"] is not None:
        price = r["base_price"]
    supplier_ship_hours = (
        r["supplier_ship_hours"]
        if "supplier_ship_hours" in keys and r["supplier_ship_hours"] is not None
        else r["ship_hours"]
    )
    if initial and "base_ship_hours" in keys and r["base_ship_hours"] is not None:
        supplier_ship_hours = r["base_ship_hours"]
    return Product(
            product_id=r["product_id"], name=r["name"],
            quantity=quantity,
            quantity_updated_t=r["quantity_updated_t"] if "quantity_updated_t" in keys else 0,
            price=price, ref_price=r["ref_price"],
            base_price=r["base_price"] if "base_price" in keys and r["base_price"] is not None else r["price"],
            supplier_id=r["supplier_id"], supplier_name=r["supplier_name"],
            ship_hours=r["ship_hours"], logistics_hours=r["logistics_hours"],
            category=r["category"],
            historical_avg_rating=r["historical_avg_rating"],
            shop_rating=r["shop_rating"],
            return_buyer_rate=r["return_buyer_rate"],
            supplier_age_years=r["supplier_age_years"],
            cancel_rate=r["cancel_rate"], refund_rate=r["refund_rate"],
            only_refund_rate=r["only_refund_rate"],
            bad_review_rate=r["bad_review_rate"],
            max_quantity=r["max_quantity"], hourly_increment=r["hourly_increment"],
            timeout_rate=r["timeout_rate"], price_change_rate=r["price_change_rate"],
            supplier_delist_rate=r["supplier_delist_rate"],
            elasticity=r["elasticity"],
            base_ship_hours=(
                r["base_ship_hours"]
                if "base_ship_hours" in keys and r["base_ship_hours"] is not None
                else r["ship_hours"]
            ),
            supplier_ship_hours=supplier_ship_hours,
            is_listed_by_supplier=bool(r["is_listed_by_supplier"]),
            delist_recover_t=r["delist_recover_t"],
            price_recover_t=r["price_recover_t"],
            timeout_active=bool(r["timeout_active"]),
            timeout_recover_t=r["timeout_recover_t"],
            market_curve=json.loads(r["market_curve"] or "[]"),
        )


def load_products(conn, run_id: str, *, initial: bool = False) -> list[Product]:
    rows = conn.execute("SELECT * FROM products WHERE run_id=?", (run_id,)).fetchall()
    return [_product_from_row(r, initial=initial) for r in rows]


def load_product(conn, run_id: str, product_id: str, *, initial: bool = False) -> Optional[Product]:
    row = conn.execute(
        "SELECT * FROM products WHERE run_id=? AND product_id=?",
        (run_id, product_id),
    ).fetchone()
    return _product_from_row(row, initial=initial) if row is not None else None


def _effective_quantity_sql(current_t: int, table_alias: str = "") -> str:
    t = int(current_t)
    def col(name: str) -> str:
        return f"{table_alias}.{name}" if table_alias else name

    return (
        f"CASE WHEN {col('is_listed_by_supplier')}=1 THEN"
        f" MIN({col('max_quantity')}, {col('quantity')} + CASE WHEN {t} > COALESCE({col('quantity_updated_t')},0)"
        f" THEN ({t} - COALESCE({col('quantity_updated_t')},0)) * {col('hourly_increment')} ELSE 0 END)"
        f" ELSE {col('quantity')} END"
    )


def search_products_sql(
    conn,
    run_id: str,
    *,
    query: str,
    filters: dict,
    sort_by: str,
    limit: int,
    offset: int,
    current_t: int,
) -> list[dict]:
    """SQL-backed public product search with effective lazy inventory."""

    def _col(name: str, table_alias: str) -> str:
        return f"{table_alias}.{name}" if table_alias else name

    def _where_parts(table_alias: str = "") -> tuple[str, list[str], list]:
        qty_expr = _effective_quantity_sql(current_t, table_alias)
        parts = [f"{_col('run_id', table_alias)}=?", f"{_col('is_listed_by_supplier', table_alias)}=1"]
        params: list = [run_id]
        mapping = [
            ("price_min", f"{_col('price', table_alias)}>=?"),
            ("price_max", f"{_col('price', table_alias)}<=?"),
            ("supplier_rating_min", f"{_col('shop_rating', table_alias)}>=?"),
            ("historical_rating_min", f"{_col('historical_avg_rating', table_alias)}>=?"),
            ("logistics_hours_max", f"{_col('logistics_hours', table_alias)}<=?"),
            ("supplier_ship_hours_max", f"{_col('supplier_ship_hours', table_alias)}<=?"),
            (
                "delivery_hours_max",
                f"({_col('supplier_ship_hours', table_alias)} + {_col('logistics_hours', table_alias)})<=?",
            ),
        ]
        for key, clause in mapping:
            value = filters.get(key)
            if value is not None:
                parts.append(clause)
                params.append(value)
        if filters.get("quantity_min") is not None:
            parts.append(f"{qty_expr}>=?")
            params.append(filters["quantity_min"])
        return qty_expr, parts, params

    query_text = str(query or "")
    fts_expr = _fts_query(query_text)
    if fts_expr:
        qty_sql, parts, params = _where_parts("p")
        order_sql = {
            "relevance": "bm25(catalog_search_fts) ASC, p.product_id ASC",
            "price_asc": "p.price ASC, bm25(catalog_search_fts) ASC, p.product_id ASC",
            "price_desc": "p.price DESC, bm25(catalog_search_fts) ASC, p.product_id ASC",
            "rating": (
                "p.historical_avg_rating DESC, p.shop_rating DESC,"
                " bm25(catalog_search_fts) ASC, p.product_id ASC"
            ),
            "supplier_rating": (
                "p.shop_rating DESC, p.historical_avg_rating DESC,"
                " bm25(catalog_search_fts) ASC, p.product_id ASC"
            ),
            "logistics_speed": (
                "(p.supplier_ship_hours + p.logistics_hours) ASC,"
                " bm25(catalog_search_fts) ASC, p.product_id ASC"
            ),
        }[sort_by]
        rows = conn.execute(
            "SELECT p.product_id, p.name,"
            f" {qty_sql} AS quantity,"
            " p.price, p.supplier_id, p.supplier_name,"
            " p.supplier_ship_hours, p.logistics_hours,"
            " p.category, p.historical_avg_rating, p.shop_rating,"
            " p.supplier_age_years"
            " FROM catalog_search_fts JOIN products p"
            " ON p.run_id=catalog_search_fts.run_id"
            " AND p.product_id=catalog_search_fts.product_id"
            f" WHERE catalog_search_fts MATCH ? AND {' AND '.join(parts)}"
            f" ORDER BY {order_sql}"
            " LIMIT ? OFFSET ?",
            (fts_expr, *params, int(limit), int(offset)),
        ).fetchall()
        return [dict(r) for r in rows]

    if query_text.strip():
        return []

    qty_sql, parts, params = _where_parts()

    order_sql = {
        "price_asc": "price ASC, product_id ASC",
        "price_desc": "price DESC, product_id ASC",
        "rating": "historical_avg_rating DESC, shop_rating DESC, product_id ASC",
        "supplier_rating": "shop_rating DESC, historical_avg_rating DESC, product_id ASC",
        "logistics_speed": "(supplier_ship_hours + logistics_hours) ASC, product_id ASC",
    }.get(sort_by, "category ASC, product_id ASC")

    rows = conn.execute(
        "SELECT product_id, name,"
        f" {qty_sql} AS quantity,"
        " price, supplier_id, supplier_name, supplier_ship_hours, logistics_hours,"
        " category, historical_avg_rating, shop_rating, supplier_age_years"
        " FROM products"
        f" WHERE {' AND '.join(parts)}"
        f" ORDER BY {order_sql}"
        " LIMIT ? OFFSET ?",
        (*params, int(limit), int(offset)),
    ).fetchall()
    return [dict(r) for r in rows]


def list_supplier_products_sql(conn, run_id: str, supplier_id: str, *,
                               limit: int, offset: int, current_t: int) -> list[dict]:
    qty_sql = _effective_quantity_sql(current_t)
    rows = conn.execute(
        "SELECT product_id, name,"
        f" {qty_sql} AS quantity,"
        " price, supplier_id, supplier_name, supplier_ship_hours, logistics_hours,"
        " category, historical_avg_rating, shop_rating, supplier_age_years"
        " FROM products WHERE run_id=? AND supplier_id=? AND is_listed_by_supplier=1"
        " ORDER BY product_id LIMIT ? OFFSET ?",
        (run_id, supplier_id, int(limit), int(offset)),
    ).fetchall()
    return [dict(r) for r in rows]


def get_supplier_profile_sql(conn, run_id: str, supplier_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT supplier_id, supplier_name, shop_rating, return_buyer_rate,"
        " supplier_age_years,"
        " SUM(CASE WHEN is_listed_by_supplier=1 THEN 1 ELSE 0 END) AS product_count"
        " FROM products WHERE run_id=? AND supplier_id=?"
        " GROUP BY supplier_id, supplier_name, shop_rating,"
        " return_buyer_rate, supplier_age_years"
        " ORDER BY product_count DESC LIMIT 1",
        (run_id, supplier_id),
    ).fetchone()
    return dict(row) if row else None


_DASHBOARD_SUPPLIER_SORTS = {
    "product_id": "product_id ASC",
    "name": "name ASC, product_id ASC",
    "category": "category ASC, product_id ASC",
    "supplier_id": "supplier_id ASC, product_id ASC",
    "supplier_name": "supplier_name ASC, product_id ASC",
    "price": "price ASC, product_id ASC",
    "ref_price": "ref_price ASC, product_id ASC",
    "base_price": "base_price ASC, product_id ASC",
    "quantity": "quantity ASC, product_id ASC",
    "logistics_hours": "(supplier_ship_hours + logistics_hours) ASC, product_id ASC",
    "historical_avg_rating": "historical_avg_rating DESC, product_id ASC",
    "shop_rating": "shop_rating DESC, product_id ASC",
    "return_buyer_rate": "return_buyer_rate DESC, product_id ASC",
    "supplier_age_years": "supplier_age_years DESC, product_id ASC",
    "is_listed_by_supplier": "is_listed_by_supplier DESC, product_id ASC",
    "cancel_rate": "cancel_rate DESC, product_id ASC",
    "refund_rate": "refund_rate DESC, product_id ASC",
    "only_refund_rate": "only_refund_rate DESC, product_id ASC",
    "timeout_rate": "timeout_rate DESC, product_id ASC",
    "bad_review_rate": "bad_review_rate DESC, product_id ASC",
    "hourly_increment": "hourly_increment DESC, product_id ASC",
    "price_change_rate": "price_change_rate DESC, product_id ASC",
    "supplier_delist_rate": "supplier_delist_rate DESC, product_id ASC",
    "elasticity": "elasticity DESC, product_id ASC",
}


def _dashboard_supplier_where(run_id: str, query: str) -> tuple[list[str], list]:
    parts = ["run_id=?"]
    params: list = [run_id]
    terms = [t.lower() for t in re.findall(r"[\w]+", str(query or ""))]
    for term in terms:
        like = f"%{term}%"
        parts.append(
            "(lower(product_id) LIKE ? OR lower(name) LIKE ? OR"
            " lower(category) LIKE ? OR lower(supplier_id) LIKE ?)"
        )
        params.extend([like, like, like, like])
    return parts, params


def count_dashboard_supplier_products(conn, run_id: str, *, query: str = "") -> int:
    parts, params = _dashboard_supplier_where(run_id, query)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM products WHERE {' AND '.join(parts)}",
        tuple(params),
    ).fetchone()
    return int(row["n"] or 0)


def list_dashboard_supplier_products(
    conn,
    run_id: str,
    *,
    query: str = "",
    sort_by: str = "product_id",
    sort_dir: str = "asc",
    limit: int = 500,
    offset: int = 0,
    current_t: int = 0,
) -> list[dict]:
    qty_sql = _effective_quantity_sql(current_t)
    parts, params = _dashboard_supplier_where(run_id, query)
    order_sql = _DASHBOARD_SUPPLIER_SORTS.get(sort_by, _DASHBOARD_SUPPLIER_SORTS["product_id"])
    if str(sort_dir).lower() == "desc":
        order_sql = order_sql.replace(" ASC", " __ASC__").replace(" DESC", " ASC").replace(" __ASC__", " DESC")
    rows = conn.execute(
        "SELECT product_id, name, category, supplier_id, supplier_name,"
        " price, ref_price, base_price,"
        f" {qty_sql} AS quantity,"
        " max_quantity, is_listed_by_supplier, timeout_active,"
        " logistics_hours, base_ship_hours, supplier_ship_hours,"
        " historical_avg_rating, shop_rating, return_buyer_rate, supplier_age_years,"
        " cancel_rate, refund_rate, only_refund_rate, timeout_rate, bad_review_rate,"
        " hourly_increment, price_change_rate, supplier_delist_rate, elasticity,"
        " delist_recover_t, price_recover_t, timeout_recover_t"
        " FROM products"
        f" WHERE {' AND '.join(parts)}"
        f" ORDER BY {order_sql}"
        " LIMIT ? OFFSET ?",
        (*params, int(limit), int(offset)),
    ).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_supplier_product(
    conn,
    run_id: str,
    product_id: str,
    *,
    current_t: int = 0,
) -> dict | None:
    qty_sql = _effective_quantity_sql(current_t)
    row = conn.execute(
        "SELECT product_id, name, category, supplier_id, supplier_name,"
        " price, ref_price, base_price,"
        f" {qty_sql} AS quantity,"
        " max_quantity, is_listed_by_supplier, timeout_active,"
        " logistics_hours, base_ship_hours, supplier_ship_hours,"
        " historical_avg_rating, shop_rating, return_buyer_rate, supplier_age_years,"
        " cancel_rate, refund_rate, only_refund_rate, timeout_rate, bad_review_rate,"
        " hourly_increment, price_change_rate, supplier_delist_rate, elasticity,"
        " delist_recover_t, price_recover_t, timeout_recover_t"
        " FROM products WHERE run_id=? AND product_id=?",
        (run_id, product_id),
    ).fetchone()
    return dict(row) if row else None


def load_dashboard_merchant_listings(
    conn,
    run_id: str,
    agent_id: str,
    *,
    current_t: int = 0,
) -> list[dict]:
    """Return merchant listing rows for the dashboard without hydrating env."""
    qty_sql = _effective_quantity_sql(current_t)
    rows = conn.execute(
        "SELECT l.product_id,"
        " p.name, p.category, l.sale_price,"
        " p.price AS supplier_price, p.ref_price, p.base_price,"
        " p.supplier_id, p.supplier_name, l.listed_at,"
        f" {qty_sql} AS quantity,"
        " p.is_listed_by_supplier,"
        " l.promised_ship_hours,"
        " l.promised_logistics_hours,"
        " p.supplier_ship_hours AS supplier_ship_hours,"
        " p.logistics_hours AS supplier_logistics_hours,"
        " p.historical_avg_rating, p.shop_rating,"
        " p.return_buyer_rate, p.supplier_age_years,"
        " p.cancel_rate, p.refund_rate, p.only_refund_rate,"
        " p.timeout_rate, p.bad_review_rate,"
        " p.price_change_rate, p.supplier_delist_rate, p.elasticity,"
        " l.cum_sales, l.rating_sum, l.rating_count"
        " FROM store_listings l"
        " LEFT JOIN products p"
        " ON l.run_id=p.run_id AND l.product_id=p.product_id"
        " WHERE l.run_id=? AND l.agent_id=?"
        " ORDER BY l.product_id",
        (run_id, agent_id),
    ).fetchall()
    out = []
    for r in rows:
        sale_price = float(r["sale_price"] or 0.0)
        supplier_price = r["supplier_price"]
        margin = (
            ((sale_price - float(supplier_price)) / sale_price)
            if sale_price and supplier_price is not None
            else 0.0
        )
        out.append({
            "product_id": r["product_id"],
            "name": r["name"] or "",
            "category": r["category"] or "",
            "sale_price": r["sale_price"],
            "supplier_price": supplier_price,
            "ref_price": r["ref_price"],
            "base_price": r["base_price"],
            "supplier_id": r["supplier_id"],
            "supplier_name": r["supplier_name"],
            "margin_ratio": round(margin, 4),
            "quantity": r["quantity"],
            "is_listed_by_supplier": (
                bool(r["is_listed_by_supplier"])
                if r["is_listed_by_supplier"] is not None
                else None
            ),
            "listed_at": r["listed_at"],
            "promised_ship_hours": r["promised_ship_hours"],
            "promised_logistics_hours": r["promised_logistics_hours"],
            "supplier_ship_hours": r["supplier_ship_hours"],
            "supplier_logistics_hours": r["supplier_logistics_hours"],
            "supplier_log_hour": (
                r["supplier_ship_hours"] + r["supplier_logistics_hours"]
                if r["supplier_ship_hours"] is not None and r["supplier_logistics_hours"] is not None
                else r["supplier_logistics_hours"]
            ),
            "historical_avg_rating": r["historical_avg_rating"],
            "shop_rating": r["shop_rating"],
            "return_buyer_rate": r["return_buyer_rate"],
            "supplier_age_years": r["supplier_age_years"],
            "cancel_rate": r["cancel_rate"],
            "refund_rate": r["refund_rate"],
            "only_refund_rate": r["only_refund_rate"],
            "timeout_rate": r["timeout_rate"],
            "bad_review_rate": r["bad_review_rate"],
            "price_change_rate": r["price_change_rate"],
            "supplier_delist_rate": r["supplier_delist_rate"],
            "elasticity": r["elasticity"],
            "cum_sales": r["cum_sales"],
            "rating_sum": float(r["rating_sum"] or 0.0),
            "rating_count": float(r["rating_count"] or 0.0),
        })
    return out


def load_dashboard_merchant_action_events(
    conn,
    run_id: str,
    agent_id: str,
    t_to: Optional[int] = None,
) -> list[dict]:
    """Order/action events shown as merchant chart marks.

    Force ix_events_run_type_t: for large runs SQLite otherwise prefers the
    run/t index due to ORDER BY t and scans millions of unrelated events.
    """
    parts = [
        "run_id=?",
        "agent_id=?",
        "event_type IN"
        " ('order_created','order_stockout_violation','order_insufficient_balance_violation')",
    ]
    params: list = [run_id, agent_id]
    if t_to is not None:
        parts.append("t<=?")
        params.append(int(t_to))
    rows = conn.execute(
        "SELECT t, event_type, entity_id, payload FROM events"
        " INDEXED BY ix_events_run_type_t"
        f" WHERE {' AND '.join(parts)}"
        " ORDER BY t ASC",
        tuple(params),
    ).fetchall()
    return [dict(r) for r in rows]


def _dashboard_day(t: int, step_hours: int) -> int:
    return int((int(t) * int(step_hours)) // 24) + 1


def _dashboard_sales_grain(
    max_day: int,
    level: Optional[str] = None,
) -> tuple[str, int]:
    if level == "day":
        return "day", 1
    if level == "week":
        return "week", 7
    if max_day <= 60:
        return "day", 1
    if max_day <= 365:
        return "week", 7
    return "month", 30


def _dashboard_sales_buckets(
    max_day: int,
    level: Optional[str] = None,
    min_day: int = 1,
) -> list[dict]:
    grain, bucket_days = _dashboard_sales_grain(max_day, level)
    buckets = []
    min_day = max(1, int(min_day))
    for start_day in range(min_day, max_day + 1, bucket_days):
        end_day = min(max_day, start_day + bucket_days - 1)
        idx = ((start_day - min_day) // bucket_days) + 1
        if grain == "day":
            key = label = f"D{start_day}"
        elif grain == "week":
            key = label = f"W{idx}"
        else:
            key = label = f"M{idx}"
        buckets.append({
            "key": key,
            "label": label,
            "start_day": start_day,
            "end_day": end_day,
        })
    return buckets


_DASHBOARD_SUPPLY_CHAIN_ANOMALY_EVENT_TYPES = (
    "price_change",
    "supplier_delist",
    "supplier_timeout",
)

_DASHBOARD_ORDER_ANOMALY_EVENT_TYPES = (
    "order_late",
    "order_cancelled",
    "order_settled_refund",
    "order_settled_only_refund",
    "order_settled_bad_review",
    "order_stockout_violation",
    "order_insufficient_balance_violation",
)

_DASHBOARD_AGENT_LISTING_EVENT_METRICS = {
    "agent_list_product": "list",
    "agent_delist_product": "delist",
    "agent_adjust_price": "price",
    "agent_set_promised_ship_hours": "promise",
}


def load_dashboard_merchant_daily_sales_by_product(
    conn,
    run_id: str,
    agent_id: str,
    *,
    current_t: int,
    step_hours: int,
    merchant_products: Optional[dict[str, tuple[str, str]]] = None,
    level: Optional[str] = None,
    t_from: Optional[int] = None,
    t_to: Optional[int] = None,
) -> dict:
    """Return day x product metric bars for the merchant dashboard."""
    range_from = max(0, int(t_from) if t_from is not None else 0)
    range_to = min(
        int(current_t),
        int(t_to) if t_to is not None else int(current_t),
    )
    min_day = _dashboard_day(range_from, step_hours)
    max_day = _dashboard_day(max(range_to, 0), step_hours)
    buckets = (
        _dashboard_sales_buckets(max_day, level, min_day=min_day)
        if range_to >= range_from
        else []
    )
    grain, bucket_days = _dashboard_sales_grain(max_day, level)

    def empty_bucket_metrics() -> dict:
        return {
            "orders": 0,
            "value": 0,
            "gmv": 0.0,
            "gross_profit": 0.0,
            "net_profit": 0.0,
            "supply_chain_anomalies": 0,
            "order_anomalies": 0,
        }

    def ensure_product(
        by_product: dict[str, dict],
        pid: str,
        name: str = "",
        category: str = "",
    ) -> dict:
        product = by_product.setdefault(pid, {
            "product_id": pid,
            "name": name or "",
            "category": category or "",
            "by_bucket": {
                str(b["key"]): empty_bucket_metrics()
                for b in buckets
            },
        })
        if name and not product.get("name"):
            product["name"] = name
        if category and not product.get("category"):
            product["category"] = category
        return product

    def bucket_for_t(t: int) -> dict | None:
        day = _dashboard_day(int(t), step_hours)
        if day < min_day or day > max_day:
            return None
        bucket_start = int((day - min_day) / bucket_days) * bucket_days + min_day
        return bucket_by_start.get(bucket_start)

    bucket_by_start = {int(bucket["start_day"]): bucket for bucket in buckets}
    by_product: dict[str, dict] = {}

    if merchant_products is None:
        listing_rows = conn.execute(
            "SELECT sl.product_id, p.name AS product_name, p.category AS category"
            " FROM store_listings sl"
            " LEFT JOIN products p ON sl.run_id=p.run_id AND sl.product_id=p.product_id"
            " WHERE sl.run_id=? AND sl.agent_id=?"
            "   AND (sl.listed_at IS NULL OR sl.listed_at<=?)"
            " ORDER BY sl.product_id ASC",
            (run_id, agent_id, int(range_to)),
        ).fetchall()
        merchant_product_info = {
            str(r["product_id"]): (r["product_name"] or "", r["category"] or "")
            for r in listing_rows
        }
    else:
        merchant_product_info = {
            str(pid): (name or "", category or "")
            for pid, (name, category) in merchant_products.items()
            if pid
        }
    merchant_product_ids = set(merchant_product_info)

    rows = conn.execute(
        "WITH order_days AS ("
        " SELECT o.product_id,"
        "        CAST(((o.order_t * :step_hours) / 24) AS INTEGER) + 1 AS day,"
        "        o.sale_price, o.purchase_price, o.current_status,"
        "        o.settled_t, o.realized_revenue, o.realized_cost, o.total_penalty,"
        "        p.name AS product_name, p.category AS category"
        " FROM orders o INDEXED BY ix_orders_run_agent_t_product"
        " LEFT JOIN products p ON o.run_id=p.run_id AND o.product_id=p.product_id"
        " WHERE o.run_id=:run_id AND o.agent_id=:agent_id"
        "   AND o.order_t>=:range_from AND o.order_t<=:range_to"
        "), order_buckets AS ("
        " SELECT product_id,"
        "        (CAST((day - :min_day) / :bucket_days AS INTEGER)"
        "          * :bucket_days + :min_day)"
        "          AS bucket_start_day,"
        "        product_name, category, sale_price, purchase_price, current_status,"
        "        settled_t, realized_revenue, realized_cost, total_penalty"
        " FROM order_days"
        ")"
        " SELECT product_id, bucket_start_day, product_name, category,"
        "        COUNT(*) AS value,"
        "        SUM(CASE WHEN current_status NOT IN"
        "          ('stockout','insufficient_balance')"
        "          THEN COALESCE(sale_price, 0.0) ELSE 0 END) AS gmv,"
        "        SUM(CASE WHEN current_status NOT IN"
        "          ('stockout','insufficient_balance')"
        "          THEN COALESCE(sale_price, 0.0) - COALESCE(purchase_price, 0.0)"
        "          ELSE 0 END)"
        "          AS gross_profit,"
        "        SUM(CASE WHEN settled_t IS NOT NULL"
        "          THEN COALESCE(realized_revenue, 0.0)"
        "             - COALESCE(realized_cost, 0.0)"
        "             - COALESCE(total_penalty, 0.0)"
        "          ELSE 0 END) AS net_profit"
        " FROM order_buckets"
        " GROUP BY product_id, bucket_start_day, product_name, category"
        " ORDER BY product_id ASC, bucket_start_day ASC",
        {
            "step_hours": int(step_hours),
            "bucket_days": int(bucket_days),
            "run_id": run_id,
            "agent_id": agent_id,
            "range_from": int(range_from),
            "range_to": int(range_to),
            "min_day": int(min_day),
        },
    ).fetchall()
    for r in rows:
        pid = str(r["product_id"])
        merchant_product_ids.add(pid)
        bucket = bucket_by_start.get(int(r["bucket_start_day"]))
        if bucket is None:
            continue
        bucket_key = str(bucket["key"])
        product = ensure_product(
            by_product,
            pid,
            r["product_name"] or "",
            r["category"] or "",
        )
        point = product["by_bucket"].setdefault(bucket_key, empty_bucket_metrics())
        point["orders"] += int(r["value"] or 0)
        point["value"] += int(r["value"] or 0)
        point["gmv"] += float(r["gmv"] or 0.0)
        point["gross_profit"] += float(r["gross_profit"] or 0.0)
        point["net_profit"] += float(r["net_profit"] or 0.0)

    event_rows = []
    supplier_type_qmarks = ",".join(
        "?" for _ in _DASHBOARD_SUPPLY_CHAIN_ANOMALY_EVENT_TYPES
    )
    merchant_product_ids_list = sorted(merchant_product_ids)
    for i in range(0, len(merchant_product_ids_list), 500):
        chunk = merchant_product_ids_list[i:i + 500]
        entity_qmarks = ",".join("?" for _ in chunk)
        event_rows.extend(conn.execute(
            "SELECT t, event_type, entity_id, agent_id, payload"
            " FROM events INDEXED BY ix_events_run_type_t"
            " WHERE run_id=?"
            f" AND entity_id IN ({entity_qmarks})"
            " AND t>=? AND t<=?"
            f" AND event_type IN ({supplier_type_qmarks})"
            " AND (agent_id IS NULL OR agent_id='' OR agent_id=?)"
            " ORDER BY t ASC, event_type ASC, entity_id ASC",
            (
                run_id,
                *chunk,
                int(range_from),
                int(range_to),
                *_DASHBOARD_SUPPLY_CHAIN_ANOMALY_EVENT_TYPES,
                agent_id,
            ),
        ).fetchall())

    order_type_qmarks = ",".join("?" for _ in _DASHBOARD_ORDER_ANOMALY_EVENT_TYPES)
    event_rows.extend(conn.execute(
        "SELECT t, event_type, entity_id, agent_id, payload"
        " FROM events INDEXED BY ix_events_run_type_t"
        " WHERE run_id=? AND t>=? AND t<=?"
        f" AND event_type IN ({order_type_qmarks})"
        " AND (agent_id IS NULL OR agent_id='' OR agent_id=?)"
        " ORDER BY t ASC, event_type ASC, entity_id ASC",
        (
            run_id,
            int(range_from),
            int(range_to),
            *_DASHBOARD_ORDER_ANOMALY_EVENT_TYPES,
            agent_id,
        ),
    ).fetchall())

    parsed_event_rows = []
    seen_event_rows = set()
    event_order_ids: set[str] = set()
    for r in event_rows:
        event_key = (r["t"], r["event_type"], r["entity_id"], r["agent_id"], r["payload"])
        if event_key in seen_event_rows:
            continue
        seen_event_rows.add(event_key)
        event_type = str(r["event_type"])
        try:
            payload = json.loads(r["payload"] or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        parsed_event_rows.append((r, event_type, payload))
        if event_type in _DASHBOARD_ORDER_ANOMALY_EVENT_TYPES:
            order_id = str(payload.get("order_id") or r["entity_id"] or "")
            if order_id:
                event_order_ids.add(order_id)

    order_to_product = {}
    event_order_ids_list = sorted(event_order_ids)
    for i in range(0, len(event_order_ids_list), 500):
        chunk = event_order_ids_list[i:i + 500]
        qmarks = ",".join("?" for _ in chunk)
        order_rows = conn.execute(
            "SELECT order_id, product_id"
            " FROM orders"
            " WHERE run_id=? AND agent_id=? AND order_t<=?"
            f" AND order_id IN ({qmarks})",
            (run_id, agent_id, int(range_to), *chunk),
        ).fetchall()
        for row in order_rows:
            if row["order_id"] and row["product_id"]:
                order_to_product[str(row["order_id"])] = str(row["product_id"])
    merchant_product_ids.update(order_to_product.values())

    order_anomaly_units: dict[tuple[str, str], set[str]] = {}
    stockout_units: dict[tuple[str, str], set[str]] = {}
    for r, event_type, payload in parsed_event_rows:
        bucket = bucket_for_t(int(r["t"]))
        if bucket is None:
            continue
        bucket_key = str(bucket["key"])
        if event_type in _DASHBOARD_SUPPLY_CHAIN_ANOMALY_EVENT_TYPES:
            pid = str(payload.get("product_id") or r["entity_id"] or "")
            if not pid or pid not in merchant_product_ids:
                continue
            name, category = merchant_product_info.get(pid, ("", ""))
            point = ensure_product(
                by_product, pid, name, category
            )["by_bucket"].setdefault(bucket_key, empty_bucket_metrics())
            point["supply_chain_anomalies"] += 1
        if event_type in _DASHBOARD_ORDER_ANOMALY_EVENT_TYPES:
            order_id = str(payload.get("order_id") or r["entity_id"] or "")
            # Agentless legacy events are safe only after their order resolves
            # to this merchant's own order ledger.  Never trust a payload
            # product_id here: another merchant may sell the same product.
            pid = str(order_to_product.get(order_id) or "")
            if not pid or pid not in merchant_product_ids:
                continue
            order_anomaly_units.setdefault((pid, bucket_key), set()).add(order_id)
            if event_type == "order_stockout_violation":
                stockout_units.setdefault((pid, bucket_key), set()).add(order_id)
    for (pid, bucket_key), units in order_anomaly_units.items():
        name, category = merchant_product_info.get(pid, ("", ""))
        point = ensure_product(
            by_product, pid, name, category
        )["by_bucket"].setdefault(bucket_key, empty_bucket_metrics())
        point["order_anomalies"] += len(units)
    for (pid, bucket_key), units in stockout_units.items():
        name, category = merchant_product_info.get(pid, ("", ""))
        point = ensure_product(
            by_product, pid, name, category
        )["by_bucket"].setdefault(bucket_key, empty_bucket_metrics())
        point["supply_chain_anomalies"] += len(units)

    series = []
    for row in sorted(by_product.values(), key=lambda r: str(r["product_id"])):
        series.append({
            "product_id": row["product_id"],
            "name": row["name"],
            "category": row["category"],
            "data": [
                {
                    "bucket": str(bucket["key"]),
                    "label": str(bucket["label"]),
                    "start_day": int(bucket["start_day"]),
                    "end_day": int(bucket["end_day"]),
                    "day": int(bucket["start_day"]),
                    "orders": int(row["by_bucket"][str(bucket["key"])]["orders"]),
                    "value": int(row["by_bucket"][str(bucket["key"])]["value"]),
                    "gmv": round(
                        float(row["by_bucket"][str(bucket["key"])]["gmv"]), 2),
                    "gross_profit": round(
                        float(row["by_bucket"][str(bucket["key"])]["gross_profit"]),
                        2,
                    ),
                    "net_profit": round(
                        float(row["by_bucket"][str(bucket["key"])]["net_profit"]),
                        2,
                    ),
                    "supply_chain_anomalies": int(
                        row["by_bucket"][str(bucket["key"])]["supply_chain_anomalies"]
                    ),
                    "order_anomalies": int(
                        row["by_bucket"][str(bucket["key"])]["order_anomalies"]
                    ),
                }
                for bucket in buckets
            ],
        })
    return {
        "grain": grain,
        "days": [int(bucket["start_day"]) for bucket in buckets],
        "buckets": buckets,
        "series": series,
    }


def load_dashboard_merchant_listing_ops(
    conn,
    run_id: str,
    agent_id: str,
    *,
    current_t: int,
    step_hours: int,
    level: Optional[str] = None,
    t_from: Optional[int] = None,
    t_to: Optional[int] = None,
) -> dict:
    """Return bucketed agent listing-operation series for the merchant chart."""
    range_from = max(0, int(t_from) if t_from is not None else 0)
    range_to = min(
        int(current_t),
        int(t_to) if t_to is not None else int(current_t),
    )
    min_day = _dashboard_day(range_from, step_hours)
    max_day = _dashboard_day(max(range_to, 0), step_hours)
    buckets = (
        _dashboard_sales_buckets(max_day, level, min_day=min_day)
        if range_to >= range_from
        else []
    )
    grain, bucket_days = _dashboard_sales_grain(max_day, level)
    metric_keys = ("list", "delist", "price", "promise")
    rows_by_bucket = {
        int(bucket["start_day"]): {key: 0 for key in metric_keys}
        for bucket in buckets
    }
    if buckets:
        event_types = tuple(_DASHBOARD_AGENT_LISTING_EVENT_METRICS.keys())
        type_qmarks = ",".join("?" for _ in event_types)
        rows = conn.execute(
            "SELECT t, event_type"
            " FROM events"
            " WHERE run_id=? AND agent_id=? AND t>=? AND t<=?"
            f" AND event_type IN ({type_qmarks})"
            " ORDER BY t ASC, event_type ASC, entity_id ASC",
            (run_id, agent_id, int(range_from), int(range_to), *event_types),
        ).fetchall()
        for row in rows:
            day = _dashboard_day(int(row["t"]), step_hours)
            bucket_start = int((day - min_day) / bucket_days) * bucket_days + min_day
            bucket_metrics = rows_by_bucket.get(bucket_start)
            if bucket_metrics is None:
                continue
            metric = _DASHBOARD_AGENT_LISTING_EVENT_METRICS.get(str(row["event_type"]))
            if metric:
                bucket_metrics[metric] += 1

    def series_for(metric: str) -> list[list[int]]:
        return [
            [int(bucket["start_day"]), int(rows_by_bucket[int(bucket["start_day"])][metric])]
            for bucket in buckets
        ]

    ops = [
        [
            int(bucket["start_day"]),
            sum(int(rows_by_bucket[int(bucket["start_day"])][metric]) for metric in metric_keys),
        ]
        for bucket in buckets
    ]
    return {
        "grain": grain,
        "days": [int(bucket["start_day"]) for bucket in buckets],
        "buckets": buckets,
        "series": {
            "ops": ops,
            "list": series_for("list"),
            "delist": series_for("delist"),
            "price": series_for("price"),
            "promise": series_for("promise"),
        },
    }


def load_dashboard_merchant_product_sales_lifecycle(
    conn,
    run_id: str,
    agent_id: str,
    product_id: str,
    *,
    current_t: int,
    step_hours: int,
) -> dict:
    """Return selected-product agent-side sales series and lifecycle event marks."""
    max_day = _dashboard_day(current_t, step_hours)
    days = list(range(1, max_day + 1)) if current_t >= 0 else []
    stats = {
        day: {"new_orders": 0, "booked_gmv": 0.0, "gross_profit": 0.0}
        for day in days
    }
    rows = conn.execute(
        "SELECT order_id, order_t, sale_price, purchase_price, current_status"
        " FROM orders"
        " WHERE run_id=? AND agent_id=? AND product_id=? AND order_t<=?"
        " ORDER BY order_t ASC, order_id ASC",
        (run_id, agent_id, product_id, int(current_t)),
    ).fetchall()
    order_ids = [str(r["order_id"]) for r in rows if r["order_id"]]
    for r in rows:
        day = _dashboard_day(r["order_t"], step_hours)
        point = stats.setdefault(
            day, {"new_orders": 0, "booked_gmv": 0.0, "gross_profit": 0.0})
        sale = float(r["sale_price"] or 0.0)
        cost = float(r["purchase_price"] or 0.0)
        point["new_orders"] += 1
        if r["current_status"] not in ("stockout", "insufficient_balance"):
            point["booked_gmv"] += sale
            point["gross_profit"] += sale - cost

    product_entity_event_types = (
        "price_change", "price_recover", "supplier_delist", "supplier_relist",
        "supplier_timeout", "supplier_timeout_end",
        "order_stockout_violation", "order_insufficient_balance_violation",
        "agent_list_product", "agent_set_promised_ship_hours",
        "agent_adjust_price", "agent_delist_product",
    )
    order_entity_event_types = (
        "order_stockout_violation", "order_insufficient_balance_violation",
        "order_late", "order_cancelled", "order_settled_refund",
        "order_settled_only_refund", "order_settled_bad_review",
    )
    agent_event_types = {
        "agent_list_product", "agent_set_promised_ship_hours",
        "agent_adjust_price", "agent_delist_product",
    }
    lifecycle_event_labels = {
        "agent_adjust_price": "adjust price",
        "agent_delist_product": "delist",
        "agent_list_product": "list",
        "agent_set_promised_ship_hours": "set promise",
        "order_cancelled": "cancel",
        "order_insufficient_balance_violation": "insufficient balance",
        "order_late": "late",
        "order_settled_bad_review": "bad review",
        "order_settled_only_refund": "only refund",
        "order_settled_refund": "refund",
        "order_stockout_violation": "stockout",
        "price_change": "price change",
        "price_recover": "price recover",
        "supplier_delist": "delist",
        "supplier_relist": "relist",
        "supplier_timeout": "timeout",
        "supplier_timeout_end": "timeout end",
    }
    order_anomaly_event_types = set(order_entity_event_types)
    event_group_rank = {
        "supplier_anomaly": 0,
        "order_anomaly": 1,
        "agent_operation": 2,
    }

    def _lifecycle_event_group(event_type: str) -> str:
        if event_type in agent_event_types:
            return "agent_operation"
        if event_type in order_anomaly_event_types:
            return "order_anomaly"
        return "supplier_anomaly"

    def _event_label(event_type: str) -> str:
        return lifecycle_event_labels.get(
            event_type,
            event_type.replace("order_", "").replace("supplier_", "").replace("_", " "),
        )

    def _event_rows_for_entities(entity_ids: list[str], event_types: tuple[str, ...]) -> list:
        out = []
        if not entity_ids:
            return out
        type_qmarks = ",".join("?" for _ in event_types)
        unique_entity_ids = sorted({str(entity_id) for entity_id in entity_ids})
        for i in range(0, len(unique_entity_ids), 500):
            chunk = unique_entity_ids[i:i + 500]
            entity_qmarks = ",".join("?" for _ in chunk)
            out.extend(conn.execute(
                "SELECT t, event_type, entity_id, agent_id, payload"
                " FROM events INDEXED BY ix_events_run_entity_type_t"
                " WHERE run_id=?"
                f" AND entity_id IN ({entity_qmarks})"
                " AND t<=?"
                f" AND event_type IN ({type_qmarks})"
                " AND (agent_id IS NULL OR agent_id='' OR agent_id=?)"
                " ORDER BY t ASC, event_type ASC, entity_id ASC",
                (run_id, *chunk, int(current_t), *event_types, agent_id),
            ).fetchall())
        return out

    event_rows = []
    seen_event_rows = set()
    for row in (
        _event_rows_for_entities([product_id], product_entity_event_types)
        + _event_rows_for_entities(order_ids, order_entity_event_types)
    ):
        key = (row["t"], row["event_type"], row["entity_id"], row["agent_id"], row["payload"])
        if key in seen_event_rows:
            continue
        seen_event_rows.add(key)
        event_rows.append(row)
    events = []
    order_id_set = set(order_ids)
    for r in event_rows:
        event_agent = r["agent_id"]
        if event_agent not in (None, "", agent_id):
            continue
        payload = json.loads(r["payload"] or "{}")
        event_product_id = payload.get("product_id")
        if event_product_id is not None:
            event_product_id = str(event_product_id)
        if (
            r["entity_id"] != product_id
            and str(r["entity_id"]) not in order_id_set
            and event_product_id != product_id
        ):
            continue
        event_group = _lifecycle_event_group(r["event_type"])
        events.append({
            "t": int(r["t"]),
            "day": _dashboard_day(r["t"], step_hours),
            "event_type": r["event_type"],
            "event_group": event_group,
            "entity_id": r["entity_id"],
            "agent_id": event_agent or "",
            "payload": payload,
        })
    listing = get_listing(conn, run_id, agent_id, product_id)
    has_list_event = any(
        event["event_type"] == "agent_list_product" for event in events
    )
    if listing is not None and not has_list_event and int(listing.listed_at or 0) <= int(current_t):
        events.append({
            "t": int(listing.listed_at or 0),
            "day": _dashboard_day(int(listing.listed_at or 0), step_hours),
            "event_type": "agent_list_product",
            "event_group": "agent_operation",
            "entity_id": product_id,
            "agent_id": agent_id,
            "payload": {
                "sale_price": listing.sale_price,
                "synthetic": True,
            },
        })
    events.sort(key=lambda event: (
        int(event["t"]),
        event_group_rank.get(event["event_group"], 99),
        str(event["event_type"]),
    ))

    ordered_days = sorted(stats)
    order_id_set_for_summary = set(order_ids)

    def _order_event_unit(event: dict) -> str:
        payload_order_id = (event.get("payload") or {}).get("order_id")
        if payload_order_id is not None and str(payload_order_id) in order_id_set_for_summary:
            return str(payload_order_id)
        entity_id = str(event.get("entity_id") or "")
        if entity_id in order_id_set_for_summary:
            return entity_id
        return f"{event.get('event_type')}:{event.get('t')}:{entity_id}"

    def _by_type(group_events: list[dict], *, order_unique: bool = False) -> list[dict]:
        counts: dict[str, set | int] = {}
        for event in group_events:
            event_type = str(event["event_type"])
            if order_unique:
                counts.setdefault(event_type, set()).add(_order_event_unit(event))
            else:
                counts[event_type] = int(counts.get(event_type, 0)) + 1
        out = []
        for event_type in sorted(counts):
            raw_count = counts[event_type]
            count = len(raw_count) if isinstance(raw_count, set) else int(raw_count)
            out.append({
                "event_type": event_type,
                "label": _event_label(event_type),
                "count": count,
            })
        return out

    def _lifecycle_summary() -> list[dict]:
        order_events = [e for e in events if e["event_group"] == "order_anomaly"]
        supplier_events = [e for e in events if e["event_group"] == "supplier_anomaly"]
        agent_events = [e for e in events if e["event_group"] == "agent_operation"]
        summary = []
        if order_events:
            affected_orders = {_order_event_unit(event) for event in order_events}
            denominator = len(order_ids)
            count = len(affected_orders)
            rate = (count / denominator) if denominator else None
            summary.append({
                "key": "order_anomaly",
                "label": "订单异常",
                "count": count,
                "event_count": len(order_events),
                "denominator": denominator,
                "rate": round(float(rate), 4) if rate is not None else None,
                "display": (
                    f"{count}/{denominator} · {rate * 100:.1f}%"
                    if rate is not None else f"{count}/{denominator}"
                ),
                "by_type": _by_type(order_events, order_unique=True),
            })
        if supplier_events:
            denominator = len(ordered_days)
            count = len(supplier_events)
            summary.append({
                "key": "supplier_anomaly",
                "label": "供应商异常",
                "count": count,
                "event_count": len(supplier_events),
                "denominator": denominator,
                "denominator_unit": "d",
                "display": f"{count}/{denominator}d",
                "by_type": _by_type(supplier_events),
            })
        if agent_events:
            count = len(agent_events)
            summary.append({
                "key": "agent_operation",
                "label": "agent 操作",
                "count": count,
                "event_count": count,
                "display": f"{count}x",
                "by_type": _by_type(agent_events),
            })
        return summary

    return {
        "days": ordered_days,
        "series": {
            "new_orders": [
                [day, int(stats[day]["new_orders"])] for day in ordered_days
            ],
            "booked_gmv": [
                [day, round(float(stats[day]["booked_gmv"]), 2)]
                for day in ordered_days
            ],
            "gross_profit": [
                [day, round(float(stats[day]["gross_profit"]), 2)]
                for day in ordered_days
            ],
        },
        "events": events,
        "summary": _lifecycle_summary(),
    }


# ---------- listings ----------

def upsert_listing(conn, run_id: str, agent_id: str, l: StoreListing) -> None:
    """Insert or update a listing. On conflict, sale_price is replaced and
    cum_sales / cum_revenue are taken from the incoming row (caller is expected
    to pass current in-memory accumulator values, not deltas)."""
    conn.execute(
        "INSERT INTO store_listings(run_id, agent_id, product_id, sale_price,"
        " cum_sales, cum_revenue, listed_at, first_listed_at,"
        " normal_count, bad_review_count, rating_sum, rating_count,"
        " promised_logistics_hours)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(run_id, agent_id, product_id) DO UPDATE SET"
        " sale_price=excluded.sale_price,"
        " cum_sales=excluded.cum_sales,"
        " cum_revenue=excluded.cum_revenue,"
        " normal_count=excluded.normal_count,"
        " bad_review_count=excluded.bad_review_count,"
        " rating_sum=excluded.rating_sum,"
        " rating_count=excluded.rating_count,"
        " promised_logistics_hours=excluded.promised_logistics_hours",
        (run_id, agent_id, l.product_id, l.sale_price, l.cum_sales, l.cum_revenue,
         l.listed_at, l.first_listed_at, l.normal_count, l.bad_review_count,
         l.rating_sum, l.rating_count, l.promised_logistics_hours),
    )


def delete_listing(conn, run_id: str, agent_id: str, product_id: str) -> None:
    conn.execute(
        "DELETE FROM store_listings WHERE run_id=? AND agent_id=? AND product_id=?",
        (run_id, agent_id, product_id),
    )


def _listing_rating_aggregates_from_row(r: sqlite3.Row) -> tuple[float, float]:
    columns = set(r.keys())
    if "rating_sum" in columns and "rating_count" in columns:
        return float(r["rating_sum"] or 0.0), float(r["rating_count"] or 0.0)
    normal_count = int(r["normal_count"] or 0) if "normal_count" in columns else 0
    bad_review_count = int(r["bad_review_count"] or 0) if "bad_review_count" in columns else 0
    return 5.0 * normal_count + 1.0 * bad_review_count, float(normal_count + bad_review_count)


def list_listings(conn, run_id: str, agent_id: Optional[str] = None) -> list[StoreListing]:
    if agent_id is None:
        rows = conn.execute(
            "SELECT * FROM store_listings WHERE run_id=?", (run_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM store_listings WHERE run_id=? AND agent_id=?",
            (run_id, agent_id),
        ).fetchall()
    out = []
    for r in rows:
        rating_sum, rating_count = _listing_rating_aggregates_from_row(r)
        out.append(StoreListing(product_id=r["product_id"], agent_id=r["agent_id"],
                                sale_price=r["sale_price"],
                                cum_sales=r["cum_sales"], cum_revenue=r["cum_revenue"],
                                listed_at=r["listed_at"],
                                first_listed_at=(
                                    r["first_listed_at"]
                                    if r["first_listed_at"] is not None
                                    else r["listed_at"]
                                ),
                                normal_count=int(r["normal_count"] or 0),
                                bad_review_count=int(r["bad_review_count"] or 0),
                                rating_sum=rating_sum,
                                rating_count=rating_count,
                                promised_logistics_hours=r["promised_logistics_hours"]))
    return out


def get_listing(conn, run_id: str, agent_id: str, product_id: str) -> Optional[StoreListing]:
    r = conn.execute(
        "SELECT * FROM store_listings WHERE run_id=? AND agent_id=? AND product_id=?",
        (run_id, agent_id, product_id),
    ).fetchone()
    if not r:
        return None
    rating_sum, rating_count = _listing_rating_aggregates_from_row(r)
    return StoreListing(product_id=r["product_id"], agent_id=r["agent_id"],
                        sale_price=r["sale_price"],
                        cum_sales=r["cum_sales"], cum_revenue=r["cum_revenue"],
                        listed_at=r["listed_at"],
                        first_listed_at=(
                            r["first_listed_at"]
                            if r["first_listed_at"] is not None
                            else r["listed_at"]
                        ),
                        normal_count=int(r["normal_count"] or 0),
                        bad_review_count=int(r["bad_review_count"] or 0),
                        rating_sum=rating_sum,
                        rating_count=rating_count,
                        promised_logistics_hours=r["promised_logistics_hours"])



# ---------- orders ----------

_ORDER_COLS = (
    "run_id", "order_id", "agent_id", "product_id", "supplier_id",
    "order_t", "promised_delivery_t",
    "sale_price", "purchase_price", "current_status",
    "purchase_t", "shipped_t", "delivered_t", "settled_t",
    "preset_anomaly", "preset_anomaly_t",
    "supplier_ship_hours", "actual_ship_hours",
    "promised_logistics_hours", "actual_logistics_hours", "late_t",
    "realized_revenue", "realized_cost", "total_penalty",
    "settlement_delay_steps",
)


def insert_orders(conn, run_id: str, orders: Iterable[Order]) -> None:
    rows = []
    status_rows = []
    for o in orders:
        rows.append((
            run_id, o.order_id, o.agent_id, o.product_id, o.supplier_id,
            o.order_t, o.promised_delivery_t,
            o.sale_price, o.purchase_price, o.current_status,
            o.purchase_t, o.shipped_t, o.delivered_t, o.settled_t,
            o.preset_anomaly, o.preset_anomaly_t,
            o.supplier_ship_hours, o.actual_ship_hours,
            o.promised_logistics_hours, o.actual_logistics_hours, o.late_t,
            o.realized_revenue, o.realized_cost, o.total_penalty,
            o.settlement_delay_steps,
        ))
        for s in o.status_log:
            status_rows.append((run_id, o.order_id, s.t, s.status))
    if rows:
        cols_sql = ",".join(_ORDER_COLS)
        placeholders = ",".join("?" * len(_ORDER_COLS))
        conn.executemany(
            f"INSERT OR IGNORE INTO orders({cols_sql}) VALUES ({placeholders})",
            rows,
        )
    if status_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO order_status VALUES (?,?,?,?)",
            status_rows,
        )


def update_order_state(conn, run_id: str, o: Order) -> None:
    conn.execute(
        "UPDATE orders SET current_status=?, purchase_t=?, shipped_t=?,"
        " delivered_t=?, settled_t=?, supplier_ship_hours=?,"
        " actual_ship_hours=?,"
        " promised_logistics_hours=?,"
        " actual_logistics_hours=?, late_t=?,"
        " realized_revenue=?, realized_cost=?, total_penalty=?,"
        " settlement_delay_steps=?"
        " WHERE run_id=? AND order_id=?",
        (o.current_status, o.purchase_t, o.shipped_t, o.delivered_t, o.settled_t,
         o.supplier_ship_hours, o.actual_ship_hours,
         o.promised_logistics_hours, o.actual_logistics_hours, o.late_t,
         o.realized_revenue, o.realized_cost, o.total_penalty,
         o.settlement_delay_steps,
         run_id, o.order_id),
    )


def insert_status_row(conn, run_id: str, order_id: str, row: OrderStatusRow) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO order_status VALUES (?,?,?,?)",
        (run_id, order_id, row.t, row.status),
    )


def _row_to_order(r) -> Order:
    return Order(
        order_id=r["order_id"], product_id=r["product_id"], supplier_id=r["supplier_id"],
        agent_id=r["agent_id"],
        order_t=r["order_t"], promised_delivery_t=r["promised_delivery_t"],
        sale_price=r["sale_price"], purchase_price=r["purchase_price"],
        current_status=r["current_status"],
        purchase_t=r["purchase_t"], shipped_t=r["shipped_t"],
        delivered_t=r["delivered_t"], settled_t=r["settled_t"],
        preset_anomaly=r["preset_anomaly"], preset_anomaly_t=r["preset_anomaly_t"],
        actual_ship_hours=(
            r["actual_ship_hours"]
            if "actual_ship_hours" in r.keys()
            else 0
        ) or 0,
        supplier_ship_hours=(
            r["supplier_ship_hours"]
            if "supplier_ship_hours" in r.keys()
            else r["actual_ship_hours"] if "actual_ship_hours" in r.keys() else 0
        ) or 0,
        promised_logistics_hours=r["promised_logistics_hours"] or 0,
        actual_logistics_hours=r["actual_logistics_hours"] or 0,
        late_t=r["late_t"],
        realized_revenue=r["realized_revenue"] or 0.0,
        realized_cost=r["realized_cost"] or 0.0,
        total_penalty=r["total_penalty"] or 0.0,
        settlement_delay_steps=(
            r["settlement_delay_steps"]
            if "settlement_delay_steps" in r.keys() and r["settlement_delay_steps"] is not None
            else -1
        ),
    )


def load_orders(conn, run_id: str, statuses: Optional[list[str]] = None,
                agent_id: Optional[str] = None) -> list[Order]:
    parts = ["run_id=?"]
    params: list = [run_id]
    if statuses:
        qmarks = ",".join("?" for _ in statuses)
        parts.append(f"current_status IN ({qmarks})")
        params.extend(statuses)
    if agent_id is not None:
        parts.append("agent_id=?")
        params.append(agent_id)
    rows = conn.execute(
        f"SELECT * FROM orders WHERE {' AND '.join(parts)}",
        tuple(params),
    ).fetchall()
    return [_row_to_order(r) for r in rows]


def _status_rank_sql(column: str = "status") -> str:
    return (
        f"CASE {column}"
        " WHEN 'ordered' THEN 10"
        " WHEN 'late' THEN 20"
        " WHEN 'shipped' THEN 30"
        " WHEN 'delivered' THEN 40"
        " WHEN 'cancelled' THEN 50"
        " WHEN 'stockout' THEN 50"
        " WHEN 'insufficient_balance' THEN 50"
        " WHEN 'settled_normal' THEN 60"
        " WHEN 'settled_refund' THEN 60"
        " WHEN 'settled_only_refund' THEN 60"
        " WHEN 'settled_bad_review' THEN 60"
        " ELSE 0 END"
    )


def load_due_orders(conn, run_id: str, t: int, normal_delay_steps: int,
                    default_promised: int = 48) -> list[Order]:
    """Load active orders that can transition at timestep ``t``.

    This keeps the order manager's state-machine semantics intact while avoiding
    a full scan/load of every active order on steps where most orders are not due.
    """
    rows = conn.execute(
        "SELECT o.* FROM orders o"
        " LEFT JOIN products p ON p.run_id=o.run_id AND p.product_id=o.product_id"
        " WHERE o.run_id=? AND ("
        "   (o.current_status IN ('ordered','late')"
        "    AND o.purchase_t IS NOT NULL"
        "    AND ("
        "      ? >= o.purchase_t + COALESCE(NULLIF(o.supplier_ship_hours, 0),"
        "          NULLIF(o.actual_ship_hours, 0), p.supplier_ship_hours, p.ship_hours, 0)"
        "      OR (o.current_status='ordered'"
        "          AND ? - o.purchase_t > ?)"
        "    ))"
        "   OR (o.current_status='shipped' AND ("
        "       (o.preset_anomaly='cancel' AND ? >= o.preset_anomaly_t)"
        "       OR (o.shipped_t IS NOT NULL"
        "           AND o.actual_logistics_hours > 0"
        "           AND ? >= o.shipped_t + o.actual_logistics_hours)"
        "   ))"
        "   OR (o.current_status='late'"
        "       AND o.shipped_t IS NOT NULL"
        "       AND o.actual_logistics_hours > 0"
        "       AND ? >= o.shipped_t + o.actual_logistics_hours)"
        "   OR (o.current_status='delivered' AND ("
        "       (o.preset_anomaly IN ('normal','bad_review')"
        "        AND o.delivered_t IS NOT NULL"
        "        AND ? >= o.delivered_t + COALESCE(NULLIF(o.settlement_delay_steps, -1), ?))"
        "       OR (o.preset_anomaly IN ('refund','only_refund')"
        "           AND ? >= o.preset_anomaly_t)"
        "   ))"
        " )",
        (
            run_id,
            int(t),
            int(t),
            int(default_promised),
            int(t),
            int(t),
            int(t),
            int(t),
            int(normal_delay_steps),
            int(t),
        ),
    ).fetchall()
    return [_row_to_order(r) for r in rows]


def count_orders_by_product(conn, run_id: str, agent_id: str,
                            t_min: int, t_max: int) -> dict[str, int]:
    rows = conn.execute(
        "SELECT product_id, COUNT(*) AS n FROM orders"
        " WHERE run_id=? AND agent_id=? AND order_t BETWEEN ? AND ?"
        " GROUP BY product_id",
        (run_id, agent_id, t_min, t_max),
    ).fetchall()
    return {r["product_id"]: r["n"] for r in rows}


def count_orders_for_agent(conn, run_id: str, agent_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM orders WHERE run_id=? AND agent_id=?",
        (run_id, agent_id),
    ).fetchone()
    return int(row["n"] or 0) if row is not None else 0


def count_orders_for_run(conn, run_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM orders WHERE run_id=?",
        (run_id,),
    ).fetchone()
    return int(row["n"] or 0) if row is not None else 0


def load_order_counts_by_t(
    conn,
    run_id: str,
    agent_id: str,
) -> list[tuple[int, int]]:
    rows = conn.execute(
        "SELECT order_t, COUNT(*) AS n"
        " FROM orders WHERE run_id=? AND agent_id=?"
        " GROUP BY order_t ORDER BY order_t",
        (run_id, agent_id),
    ).fetchall()
    return [
        (int(row["order_t"] or 0), int(row["n"] or 0))
        for row in rows
    ]


def load_order(conn, run_id: str, order_id: str) -> Optional[Order]:
    r = conn.execute("SELECT * FROM orders WHERE run_id=? AND order_id=?", (run_id, order_id)).fetchone()
    if not r:
        return None
    o = _row_to_order(r)
    log_rows = conn.execute(
        "SELECT t, status FROM order_status WHERE run_id=? AND order_id=?"
        f" ORDER BY t, {_status_rank_sql()}",
        (run_id, order_id),
    ).fetchall()
    o.status_log = [OrderStatusRow(t=r2["t"], status=r2["status"]) for r2 in log_rows]
    return o


def load_orders_with_log(conn, run_id: str, limit: int = 200,
                          status: Optional[str] = None,
                          agent_id: Optional[str] = None) -> list[dict]:
    """Return the most recent orders (by order_t DESC) joined with their full
    status_log. Each result is a dict with the order fields + status_log list.
    Filters: optional current_status and agent_id."""
    parts = ["o.run_id=?"]
    params: list = [run_id]
    if status:
        parts.append("o.current_status=?")
        params.append(status)
    if agent_id:
        parts.append("o.agent_id=?")
        params.append(agent_id)
    params.append(limit)
    order_rows = conn.execute(
        "SELECT o.order_id, o.product_id, o.supplier_id, o.agent_id,"
        " o.order_t, o.promised_delivery_t, o.sale_price, o.purchase_price,"
        " o.current_status, o.purchase_t, o.shipped_t, o.delivered_t, o.settled_t,"
        " o.supplier_ship_hours, o.actual_ship_hours,"
        " o.promised_logistics_hours, o.actual_logistics_hours, o.late_t,"
        " o.realized_revenue, o.realized_cost, o.total_penalty,"
        " p.name AS product_name"
        " FROM orders o LEFT JOIN products p"
        " ON o.run_id=p.run_id AND o.product_id=p.product_id"
        f" WHERE {' AND '.join(parts)}"
        " ORDER BY o.order_t DESC, o.order_id LIMIT ?",
        tuple(params),
    ).fetchall()
    if not order_rows:
        return []
    order_ids = [r["order_id"] for r in order_rows]
    qmarks = ",".join("?" for _ in order_ids)
    log_rows = conn.execute(
        f"SELECT order_id, t, status FROM order_status"
        f" WHERE run_id=? AND order_id IN ({qmarks})"
        f" ORDER BY t ASC, {_status_rank_sql()} ASC",
        (run_id, *order_ids),
    ).fetchall()
    log_by_oid: dict[str, list[dict]] = {}
    for r in log_rows:
        log_by_oid.setdefault(r["order_id"], []).append({"t": r["t"], "status": r["status"]})
    out = []
    for r in order_rows:
        rev = float(r["realized_revenue"] or 0.0)
        cost = float(r["realized_cost"] or 0.0)
        pen = float(r["total_penalty"] or 0.0)
        out.append({
            "order_id": r["order_id"], "product_id": r["product_id"],
            "product_name": r["product_name"] or "", "supplier_id": r["supplier_id"],
            "agent_id": r["agent_id"], "order_t": r["order_t"],
            "promised_delivery_t": r["promised_delivery_t"],
            "sale_price": r["sale_price"], "purchase_price": r["purchase_price"],
            "current_status": r["current_status"],
            "purchase_t": r["purchase_t"], "shipped_t": r["shipped_t"],
            "delivered_t": r["delivered_t"], "settled_t": r["settled_t"],
            "supplier_ship_hours": r["supplier_ship_hours"] or 0,
            "actual_ship_hours": r["actual_ship_hours"] or 0,
            "promised_logistics_hours": r["promised_logistics_hours"] or 0,
            "actual_logistics_hours": r["actual_logistics_hours"] or 0,
            "late_t": r["late_t"],
            "realized_revenue": rev,
            "realized_cost": cost,
            "total_penalty": pen,
            "net_profit": rev - cost - pen,
            "status_log": log_by_oid.get(r["order_id"], []),
        })
    return out


def load_order_status_counts_as_of(conn, run_id: str, t_to: int) -> dict[str, int]:
    rows = conn.execute(
        "WITH ranked AS ("
        " SELECT os.order_id, os.status,"
        " ROW_NUMBER() OVER ("
        "   PARTITION BY os.order_id"
        f"   ORDER BY os.t DESC, {_status_rank_sql('os.status')} DESC"
        " ) AS rn"
        " FROM order_status os"
        " WHERE os.run_id=? AND os.t<=?"
        "), latest AS ("
        " SELECT order_id, status FROM ranked WHERE rn=1"
        ")"
        " SELECT status, COUNT(*) AS n FROM latest GROUP BY status",
        (run_id, int(t_to)),
    ).fetchall()
    return {r["status"]: int(r["n"]) for r in rows}


def load_order_status_cum_as_of(conn, run_id: str, t_to: int) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(DISTINCT order_id) AS n"
        " FROM order_status WHERE run_id=? AND t<=? GROUP BY status",
        (run_id, int(t_to)),
    ).fetchall()
    return {r["status"]: int(r["n"]) for r in rows}


def load_order_status_cum_series(
    conn,
    run_id: str,
    statuses: Optional[list[str]] = None,
    t_from: Optional[int] = None,
    t_to: Optional[int] = None,
) -> dict[str, list[tuple[int, int]]]:
    """Return true cumulative first-entry counts per status over time.

    The dashboard's current-status metrics are stock snapshots; deriving
    cumulative counts from their positive deltas undercounts transient statuses
    such as ordered/shipped. This series is based on each order's first entry
    into each status in the append-only order_status log.
    """
    parts = ["run_id=?"]
    params: list = [run_id]
    if statuses:
        qmarks = ",".join("?" for _ in statuses)
        parts.append(f"status IN ({qmarks})")
        params.extend(statuses)
    rows = conn.execute(
        "SELECT status, order_id, MIN(t) AS first_t"
        f" FROM order_status WHERE {' AND '.join(parts)}"
        " GROUP BY status, order_id"
        " ORDER BY first_t, status",
        tuple(params),
    ).fetchall()

    counts_by_status_t: dict[str, dict[int, int]] = {
        status: {} for status in (statuses or [])
    }
    for r in rows:
        first_t = int(r["first_t"])
        if t_to is not None and first_t > int(t_to):
            continue
        status = r["status"]
        by_t = counts_by_status_t.setdefault(status, {})
        by_t[first_t] = by_t.get(first_t, 0) + 1

    out: dict[str, list[tuple[int, int]]] = {}
    min_t = int(t_from) if t_from is not None else None
    for status, by_t in counts_by_status_t.items():
        cum = 0
        series: list[tuple[int, int]] = []
        for t in sorted(by_t):
            cum += by_t[t]
            if min_t is not None and t < min_t:
                continue
            series.append((t, cum))
        out[status] = series
    return out


def load_orders_with_log_as_of(
    conn,
    run_id: str,
    t_to: int,
    limit: int = 200,
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> list[dict]:
    parts = ["o.run_id=?", "o.order_t<=?"]
    where_params: list = [run_id, int(t_to)]
    if status:
        parts.append("latest.status=?")
        where_params.append(status)
    if agent_id:
        parts.append("o.agent_id=?")
        where_params.append(agent_id)
    where_params.append(int(limit))
    order_rows = conn.execute(
        "WITH ranked AS ("
        " SELECT os.order_id, os.status,"
        " ROW_NUMBER() OVER ("
        "   PARTITION BY os.order_id"
        f"   ORDER BY os.t DESC, {_status_rank_sql('os.status')} DESC"
        " ) AS rn"
        " FROM order_status os"
        " WHERE os.run_id=? AND os.t<=?"
        "), latest AS ("
        " SELECT order_id, status FROM ranked WHERE rn=1"
        ")"
        " SELECT o.order_id, o.product_id, o.supplier_id, o.agent_id,"
        " o.order_t, o.promised_delivery_t, o.sale_price, o.purchase_price,"
        " latest.status AS current_status,"
        " CASE WHEN o.purchase_t<=? THEN o.purchase_t ELSE NULL END AS purchase_t,"
        " CASE WHEN o.shipped_t<=? THEN o.shipped_t ELSE NULL END AS shipped_t,"
        " CASE WHEN o.delivered_t<=? THEN o.delivered_t ELSE NULL END AS delivered_t,"
        " CASE WHEN o.settled_t<=? THEN o.settled_t ELSE NULL END AS settled_t,"
        " o.promised_ship_hours, o.supplier_ship_hours, o.actual_ship_hours,"
        " o.promised_logistics_hours, o.actual_logistics_hours,"
        " CASE WHEN o.late_t<=? THEN o.late_t ELSE NULL END AS late_t,"
        " CASE WHEN o.settled_t<=? THEN o.realized_revenue ELSE 0 END AS realized_revenue,"
        " CASE"
        "   WHEN o.settled_t<=? THEN o.realized_cost"
        "   WHEN o.purchase_t<=? THEN o.purchase_price"
        "   ELSE 0"
        " END AS realized_cost,"
        " CASE WHEN o.settled_t<=? THEN o.total_penalty ELSE 0 END AS total_penalty,"
        " p.name AS product_name"
        " FROM orders o"
        " JOIN latest ON latest.order_id=o.order_id"
        " LEFT JOIN products p ON o.run_id=p.run_id AND o.product_id=p.product_id"
        f" WHERE {' AND '.join(parts)}"
        " ORDER BY o.order_t DESC, o.order_id LIMIT ?",
        (
            run_id,
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            int(t_to),
            *where_params,
        ),
    ).fetchall()
    if not order_rows:
        return []
    order_ids = [r["order_id"] for r in order_rows]
    qmarks = ",".join("?" for _ in order_ids)
    log_rows = conn.execute(
        f"SELECT order_id, t, status FROM order_status"
        f" WHERE run_id=? AND t<=? AND order_id IN ({qmarks})"
        f" ORDER BY t ASC, {_status_rank_sql()} ASC",
        (run_id, int(t_to), *order_ids),
    ).fetchall()
    log_by_oid: dict[str, list[dict]] = {}
    for r in log_rows:
        log_by_oid.setdefault(r["order_id"], []).append({"t": r["t"], "status": r["status"]})
    out = []
    for r in order_rows:
        rev = float(r["realized_revenue"] or 0.0)
        cost = float(r["realized_cost"] or 0.0)
        pen = float(r["total_penalty"] or 0.0)
        out.append({
            "order_id": r["order_id"], "product_id": r["product_id"],
            "product_name": r["product_name"] or "", "supplier_id": r["supplier_id"],
            "agent_id": r["agent_id"], "order_t": r["order_t"],
            "promised_delivery_t": r["promised_delivery_t"],
            "sale_price": r["sale_price"], "purchase_price": r["purchase_price"],
            "current_status": r["current_status"],
            "purchase_t": r["purchase_t"], "shipped_t": r["shipped_t"],
            "delivered_t": r["delivered_t"], "settled_t": r["settled_t"],
            "supplier_ship_hours": r["supplier_ship_hours"] or 0,
            "actual_ship_hours": r["actual_ship_hours"] or 0,
            "promised_logistics_hours": r["promised_logistics_hours"] or 0,
            "actual_logistics_hours": r["actual_logistics_hours"] or 0,
            "late_t": r["late_t"],
            "realized_revenue": rev,
            "realized_cost": cost,
            "total_penalty": pen,
            "net_profit": rev - cost - pen,
            "status_log": log_by_oid.get(r["order_id"], []),
        })
    return out


def load_status_log(conn, run_id: str, order_id: str) -> list[OrderStatusRow]:
    rows = conn.execute(
        "SELECT t, status FROM order_status WHERE run_id=? AND order_id=?"
        f" ORDER BY t, {_status_rank_sql()}",
        (run_id, order_id),
    ).fetchall()
    return [OrderStatusRow(t=r["t"], status=r["status"]) for r in rows]


# ---------- cash + events + aggregates ----------

def write_cash_log(conn, run_id: str, agent_id: str, t: int, cash: Cash) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cash_log VALUES (?,?,?,?,?,?,?,?)",
        (run_id, agent_id, t, cash.balance, cash.deposit_pool, cash.in_transit,
         cash.receivable, cash.cumulative_fine),
    )


def load_latest_cash(conn, run_id: str, agent_id: str) -> Optional[Cash]:
    r = conn.execute(
        "SELECT * FROM cash_log WHERE run_id=? AND agent_id=? ORDER BY t DESC LIMIT 1",
        (run_id, agent_id),
    ).fetchone()
    if not r:
        return None
    return Cash(balance=r["balance"], deposit_pool=r["deposit_pool"],
                in_transit=r["in_transit"], receivable=r["receivable"],
                cumulative_fine=r["cumulative_fine"])


def load_latest_cash_at(conn, run_id: str, agent_id: str, t_to: int) -> Optional[Cash]:
    r = conn.execute(
        "SELECT * FROM cash_log WHERE run_id=? AND agent_id=? AND t<=?"
        " ORDER BY t DESC LIMIT 1",
        (run_id, agent_id, int(t_to)),
    ).fetchone()
    if not r:
        return None
    return Cash(balance=r["balance"], deposit_pool=r["deposit_pool"],
                in_transit=r["in_transit"], receivable=r["receivable"],
                cumulative_fine=r["cumulative_fine"])


def load_cash_series(conn, run_id: str, agent_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT t, balance, deposit_pool, in_transit, receivable, cumulative_fine"
        " FROM cash_log WHERE run_id=? AND agent_id=? ORDER BY t",
        (run_id, agent_id),
    ).fetchall()
    return [dict(r) for r in rows]


def write_events(conn, run_id: str, events: Iterable[EventLog]) -> None:
    rows = [(run_id, e.t, e.event_type, e.entity_id, e.agent_id, json.dumps(e.payload)) for e in events]
    if rows:
        conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", rows)


def load_events_at(conn, run_id: str, t: int) -> list[dict]:
    rows = conn.execute(
        "SELECT t, event_type, entity_id, agent_id, payload FROM events WHERE run_id=? AND t=?",
        (run_id, t),
    ).fetchall()
    return [{"t": r["t"], "event_type": r["event_type"], "entity_id": r["entity_id"],
             "agent_id": r["agent_id"], "payload": json.loads(r["payload"])} for r in rows]


def load_events_range(conn, run_id: str, t_from: int, t_to: int,
                      agent_id: Optional[str] = None, limit: int = 500) -> list[dict]:
    parts = ["run_id=?", "t>=?", "t<=?"]
    params: list = [run_id, t_from, t_to]
    if agent_id is not None:
        parts.append("agent_id=?")
        params.append(agent_id)
    params.append(limit)
    rows = conn.execute(
        f"SELECT t, event_type, entity_id, agent_id, payload FROM events"
        f" WHERE {' AND '.join(parts)} ORDER BY t DESC LIMIT ?",
        tuple(params),
    ).fetchall()
    return [{"t": r["t"], "event_type": r["event_type"], "entity_id": r["entity_id"],
             "agent_id": r["agent_id"], "payload": json.loads(r["payload"])} for r in rows]


def load_events_range_by_types(
    conn,
    run_id: str,
    t_from: int,
    t_to: int,
    event_types: Iterable[str],
    *,
    limit: int = 500,
) -> list[dict]:
    types = list(event_types)
    if not types or limit <= 0:
        return []
    qmarks = ",".join("?" for _ in types)
    rows = conn.execute(
        f"SELECT t, event_type, entity_id, agent_id, payload"
        f" FROM events INDEXED BY ix_events_run_type_t"
        f" WHERE run_id=? AND t>=? AND t<=?"
        f" AND event_type IN ({qmarks})"
        f" ORDER BY t DESC LIMIT ?",
        (run_id, t_from, t_to, *types, int(limit)),
    ).fetchall()
    return [{"t": r["t"], "event_type": r["event_type"], "entity_id": r["entity_id"],
             "agent_id": r["agent_id"], "payload": json.loads(r["payload"])} for r in rows]


def load_rating_events(
    conn, run_id: str, event_types: Iterable[str]
) -> list[tuple[str, str, int]]:
    """Return (agent_id, event_type, t) for every rating-relevant event in
    this run, ordered by t ascending. Used only when rehydrating legacy
    Beta-Binomial runs; v2 ratings rebuild from terminal orders instead.
    Rows with NULL agent_id (supplier-side events) are filtered out.
    """
    types = list(event_types)
    if not types:
        return []
    placeholders = ",".join("?" * len(types))
    rows = conn.execute(
        f"SELECT agent_id, event_type, t FROM events"
        f" INDEXED BY ix_events_run_type_t"
        f" WHERE run_id=? AND agent_id IS NOT NULL"
        f" AND event_type IN ({placeholders})"
        f" ORDER BY t ASC",
        (run_id, *types),
    ).fetchall()
    return [(r["agent_id"], r["event_type"], int(r["t"])) for r in rows]


def load_order_rating_rows(
    conn, run_id: str, agent_id: str, cutoff_t: int,
) -> list[tuple[str, str, Optional[int], int]]:
    """Return downstream terminal-order facts before an exclusive cutoff."""
    return [
        (product_id, current_status, late_t, settled_t)
        for (
            _, product_id, current_status, late_t, settled_t
        ) in load_order_feedback_rows(conn, run_id, agent_id, cutoff_t)
    ]


def load_order_feedback_rows(
    conn, run_id: str, agent_id: str, cutoff_t: int,
) -> list[tuple[str, str, str, Optional[int], int]]:
    """Return order identities and terminal feedback facts before a cutoff."""
    rows = conn.execute(
        "SELECT order_id, product_id, current_status, late_t, settled_t"
        " FROM orders INDEXED BY ix_orders_run_agent_settled_product"
        " WHERE run_id=? AND agent_id=? AND settled_t IS NOT NULL"
        " AND settled_t<? ORDER BY settled_t, order_id",
        (run_id, agent_id, int(cutoff_t)),
    ).fetchall()
    return [
        (
            str(r["order_id"]),
            str(r["product_id"]),
            str(r["current_status"]),
            int(r["late_t"]) if r["late_t"] is not None else None,
            int(r["settled_t"]),
        )
        for r in rows
    ]


def upsert_daily_aggregate(conn, run_id: str, day: int, gmv_delta: float, anomaly_delta: int, fine_delta: float) -> None:
    conn.execute(
        "INSERT INTO daily_aggregates(run_id, day, gmv, anomaly_count, fine_total)"
        " VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id, day) DO UPDATE SET"
        " gmv = gmv + excluded.gmv,"
        " anomaly_count = anomaly_count + excluded.anomaly_count,"
        " fine_total = fine_total + excluded.fine_total",
        (run_id, day, gmv_delta, anomaly_delta, fine_delta),
    )


def load_daily_aggregates(conn, run_id: str) -> list[dict]:
    rows = conn.execute("SELECT day, gmv, anomaly_count, fine_total FROM daily_aggregates WHERE run_id=? ORDER BY day", (run_id,)).fetchall()
    return [dict(r) for r in rows]


# ---------- metrics (per-step pre-aggregated time series) ----------

def write_metrics(conn, run_id: str, agent_id: str, t: int, kv: dict) -> None:
    rows = [(run_id, agent_id, t, k, float(v)) for k, v in kv.items()]
    if rows:
        conn.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?)", rows)


def load_metric_series(conn, run_id: str, agent_id: str, key: str,
                       t_from: Optional[int] = None, t_to: Optional[int] = None) -> list[tuple[int, float]]:
    parts = ["run_id=?", "agent_id=?", "key=?"]
    params: list = [run_id, agent_id, key]
    if t_from is not None:
        parts.append("t>=?")
        params.append(t_from)
    if t_to is not None:
        parts.append("t<=?")
        params.append(t_to)
    rows = conn.execute(
        f"SELECT t, value FROM metrics WHERE {' AND '.join(parts)} ORDER BY t",
        tuple(params),
    ).fetchall()
    return [(r["t"], r["value"]) for r in rows]


def load_metric_lasts(
    conn,
    run_id: str,
    agent_id: str,
    keys: list[str],
) -> dict[str, tuple[int, float]]:
    """Fetch several newest metric points with one indexed seek per key."""
    if not keys:
        return {}
    clauses = []
    params = []
    for key in keys:
        clauses.append(
            "SELECT ? AS key, t, value FROM ("
            " SELECT t, value FROM metrics"
            " WHERE run_id=? AND agent_id=? AND key=?"
            " ORDER BY t DESC LIMIT 1"
            ")"
        )
        params.extend((key, run_id, agent_id, key))
    rows = conn.execute(" UNION ALL ".join(clauses), tuple(params)).fetchall()
    return {
        str(row["key"]): (int(row["t"]), float(row["value"]))
        for row in rows
    }


def count_metric_points(conn, run_id: str, agent_id: str, key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM metrics"
        " WHERE run_id=? AND agent_id=? AND key=?",
        (run_id, agent_id, key),
    ).fetchone()
    return int(row["n"] or 0) if row is not None else 0


def sum_metric_values(conn, run_id: str, agent_id: str, key: str) -> Optional[float]:
    row = conn.execute(
        "SELECT SUM(value) AS total FROM metrics"
        " WHERE run_id=? AND agent_id=? AND key=?",
        (run_id, agent_id, key),
    ).fetchone()
    if row is None or row["total"] is None:
        return None
    return float(row["total"])


def load_metrics_bulk_sampled(
    conn,
    run_id: str,
    agent_id: str,
    keys: list[str],
    *,
    max_points_per_key: int,
) -> dict[str, list[tuple[int, float]]]:
    """Load bounded display series while preserving first and last points."""
    if not keys:
        return {}
    limit = max(2, int(max_points_per_key))
    qmarks = ",".join("?" for _ in keys)
    rows = conn.execute(
        "WITH ranked AS ("
        " SELECT t, key, value,"
        " ROW_NUMBER() OVER (PARTITION BY key ORDER BY t) AS rn,"
        " COUNT(*) OVER (PARTITION BY key) AS n"
        " FROM metrics"
        f" WHERE run_id=? AND agent_id=? AND key IN ({qmarks})"
        ")"
        " SELECT t, key, value FROM ranked"
        " WHERE n<=?"
        " OR rn=1 OR rn=n"
        " OR CAST(((rn-1) * (? - 1)) / (n-1) AS INTEGER)"
        "    != CAST(((rn-2) * (? - 1)) / (n-1) AS INTEGER)"
        " ORDER BY key, t",
        (run_id, agent_id, *keys, limit, limit, limit),
    ).fetchall()
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    for row in rows:
        out.setdefault(row["key"], []).append(
            (int(row["t"]), float(row["value"]))
        )
    return out


def load_metrics_bulk_daily_lasts(
    conn,
    run_id: str,
    agent_id: str,
    keys: list[str],
    *,
    step_hours: float,
) -> dict[str, list[tuple[int, float]]]:
    """Load the first point plus the exact last metric point of every sim day."""
    if not keys:
        return {}
    qmarks = ",".join("?" for _ in keys)
    rows = conn.execute(
        "WITH ranked AS ("
        " SELECT t, key, value,"
        " ROW_NUMBER() OVER ("
        "   PARTITION BY key, CAST((t * ?) / 24 AS INTEGER)"
        "   ORDER BY t DESC"
        " ) AS day_rn,"
        " ROW_NUMBER() OVER (PARTITION BY key ORDER BY t) AS series_rn"
        " FROM metrics"
        f" WHERE run_id=? AND agent_id=? AND key IN ({qmarks})"
        ")"
        " SELECT t, key, value FROM ranked"
        " WHERE day_rn=1 OR series_rn=1"
        " ORDER BY key, t",
        (float(step_hours or 1), run_id, agent_id, *keys),
    ).fetchall()
    out: dict[str, list[tuple[int, float]]] = {key: [] for key in keys}
    for row in rows:
        out.setdefault(str(row["key"]), []).append(
            (int(row["t"]), float(row["value"]))
        )
    return out


def load_metric_cumulative_daily_lasts(
    conn,
    run_id: str,
    agent_id: str,
    key: str,
    *,
    step_hours: float,
) -> list[tuple[int, float]]:
    """Return cumulative totals at the first point and every exact sim-day end."""
    rows = conn.execute(
        "WITH cumulative AS ("
        " SELECT t, SUM(value) OVER (ORDER BY t) AS total"
        " FROM metrics"
        " WHERE run_id=? AND agent_id=? AND key=?"
        "), ranked AS ("
        " SELECT t, total,"
        " ROW_NUMBER() OVER ("
        "   PARTITION BY CAST((t * ?) / 24 AS INTEGER)"
        "   ORDER BY t DESC"
        " ) AS day_rn,"
        " ROW_NUMBER() OVER (ORDER BY t) AS series_rn"
        " FROM cumulative"
        ")"
        " SELECT t, total FROM ranked"
        " WHERE day_rn=1 OR series_rn=1"
        " ORDER BY t",
        (run_id, agent_id, key, float(step_hours or 1)),
    ).fetchall()
    return [(int(row["t"]), float(row["total"])) for row in rows]


def load_metrics_bulk(conn, run_id: str, agent_id: str, keys: list[str],
                      t_from: Optional[int] = None, t_to: Optional[int] = None) -> dict[str, list[tuple[int, float]]]:
    """Fetch multiple metric series in one query. Returns {key: [(t, v), ...]}."""
    if not keys:
        return {}
    qmarks = ",".join("?" for _ in keys)
    parts = ["run_id=?", "agent_id=?", f"key IN ({qmarks})"]
    params: list = [run_id, agent_id, *keys]
    if t_from is not None:
        parts.append("t>=?")
        params.append(t_from)
    if t_to is not None:
        parts.append("t<=?")
        params.append(t_to)
    rows = conn.execute(
        f"SELECT t, key, value FROM metrics WHERE {' AND '.join(parts)} ORDER BY t",
        tuple(params),
    ).fetchall()
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    for r in rows:
        out.setdefault(r["key"], []).append((r["t"], r["value"]))
    return out


# ---------- hourly_dist ----------

def write_hourly_dist(conn, run_id: str, hourly_dist: dict) -> None:
    rows = []
    for cat, w in hourly_dist.items():
        for h in range(24):
            rows.append((run_id, cat, h, float(w[h])))
    conn.executemany("INSERT OR REPLACE INTO hourly_dist VALUES (?,?,?,?)", rows)


def load_hourly_dist(conn, run_id: str) -> dict:
    import numpy as np
    rows = conn.execute("SELECT category, hour, w FROM hourly_dist WHERE run_id=?", (run_id,)).fetchall()
    out: dict = {}
    for r in rows:
        cat = r["category"]
        if cat not in out:
            out[cat] = np.zeros(24, dtype=float)
        out[cat][r["hour"]] = r["w"]
    return out


# ---------- supplier event scheduler ----------

def insert_supplier_events(conn, run_id: str, events: Iterable[dict]) -> None:
    rows = [
        (
            run_id,
            int(e["due_t"]),
            str(e["product_id"]),
            str(e["event_type"]),
            int(e.get("seq", 0)),
            json.dumps(e.get("payload") or {}),
        )
        for e in events
    ]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO supplier_events"
            "(run_id, due_t, product_id, event_type, seq, payload)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )


def load_supplier_events_due(conn, run_id: str, t: int) -> list[dict]:
    rows = conn.execute(
        "SELECT due_t, product_id, event_type, seq, payload"
        " FROM supplier_events WHERE run_id=? AND due_t<=?"
        " ORDER BY due_t, product_id, event_type, seq",
        (run_id, int(t)),
    ).fetchall()
    return [
        {
            "due_t": int(r["due_t"]),
            "product_id": r["product_id"],
            "event_type": r["event_type"],
            "seq": int(r["seq"]),
            "payload": json.loads(r["payload"] or "{}"),
        }
        for r in rows
    ]


def delete_supplier_events_due(conn, run_id: str, t: int) -> None:
    conn.execute(
        "DELETE FROM supplier_events WHERE run_id=? AND due_t<=?",
        (run_id, int(t)),
    )


def delete_pending_supplier_events(conn, run_id: str, product_id: str,
                                   event_types: Iterable[str]) -> None:
    types = list(event_types)
    if not types:
        return
    qmarks = ",".join("?" for _ in types)
    conn.execute(
        f"DELETE FROM supplier_events WHERE run_id=? AND product_id=?"
        f" AND event_type IN ({qmarks})",
        (run_id, product_id, *types),
    )


def count_supplier_events(conn, run_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM supplier_events WHERE run_id=?",
        (run_id,),
    ).fetchone()
    return int(row["n"] or 0)
