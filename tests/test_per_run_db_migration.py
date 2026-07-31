import os
import sqlite3

from storage import db as dbm


def test_open_db_drops_legacy_product_search_table(tmp_path):
    path = os.path.join(tmp_path, "legacy.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE VIRTUAL TABLE product_search USING fts5("
            "run_id UNINDEXED, product_id UNINDEXED, name, category, supplier_name)"
        )
        conn.execute(
            "INSERT INTO product_search VALUES (?, ?, ?, ?, ?)",
            ("run-old", "p1", "Legacy product", "office", "Legacy supplier"),
        )

    conn = dbm.open_db(path)
    try:
        legacy_tables = conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE name='product_search' OR name GLOB 'product_search_*'"
        ).fetchall()
        assert legacy_tables == []
    finally:
        conn.close()
