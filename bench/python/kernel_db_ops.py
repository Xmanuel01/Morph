#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path


def run() -> int:
    db_path = Path("bench/results/python_db_ops.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("drop table if exists items")
        cur.execute("create table items(id integer primary key, name text)")
        cur.executemany("insert into items(name) values (?)", [("row",) for _ in range(200)])
        conn.commit()

        cur.execute("select id from items where id = 200")
        row = cur.fetchone()
        return 0 if row is not None else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(run())
