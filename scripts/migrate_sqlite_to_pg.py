"""SQLite → PostgreSQL 数据搬迁脚本。

用法：
    # 1. 先把 PG 的表结构建出来（Alembic 迁移）
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/knowledge_base \\
        .venv/bin/alembic upgrade head

    # 2. 搬迁数据（源默认 ./kb.db，目标从 .env 的 DATABASE_URL 读）
    .venv/bin/python scripts/migrate_sqlite_to_pg.py --dry-run   # 先看要搬什么
    .venv/bin/python scripts/migrate_sqlite_to_pg.py             # 真搬

设计要点（为什么这么写）：
1. **按外键依赖顺序搬**：users → documents → chunks、users → chat_sessions → chat_messages。
   顺序错了会因外键约束失败。
2. **保留原主键 id**：否则 chat_messages.session_id、chunks.document_id 这些引用全部错位。
3. **搬完必须重置序列**：PG 的 SERIAL 序列不会因为显式插入 id 而前进，
   不重置的话下一条 INSERT 会撞主键冲突（这是数据搬迁最经典的坑）。
4. **可重复执行**：默认遇到目标表非空就拒绝，加 --truncate 先清空再搬。
5. **列自动发现**：不硬编码列名，源表有什么列就搬什么列，
   这样将来加字段后这个脚本还能用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Boolean, JSON, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

# 按外键依赖排序：父表在前，子表在后
TABLE_ORDER = [
    "users",
    "documents",
    "chunks",
    "chat_sessions",
    "chat_messages",
    "chat_logs",
    "processing_tasks",
]


def _coerce(value, target_type):
    """★ 把 SQLite 取出的值转成 asyncpg 能接受的 Python 类型。

    为什么必须做（实测踩过）：
    SQLite 没有原生类型，DATETIME 列存的是**字符串**（'2026-10-13 11:05:13'），
    而 asyncpg 对 TIMESTAMP WITH TIME ZONE 参数要求真正的 datetime 对象，
    直接传字符串会报：
        invalid input for query argument: expected a datetime, got 'str'

    同理 BOOLEAN 列在 SQLite 里是整数 0/1，JSON 列是字符串。
    """
    if value is None:
        return None

    if isinstance(target_type, DateTime):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            s = value.strip().replace("T", " ")
            # 兼容带/不带微秒、带/不带时区偏移的多种写法
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None          # 无法解析的时间置空，而不是让整批失败
        return value

    if isinstance(target_type, Boolean):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "t", "yes", "y")
        return bool(value)

    if isinstance(target_type, JSON):
        # ★ 这里必须序列化成**字符串**：
        #   本脚本走 text() 裸 SQL，SQLAlchemy 的 json_serializer 只对
        #   ORM/Core 构造的语句生效，对 text() 不生效。
        #   （应用运行时走 ORM，由 session.py 的引擎 json_serializer 负责，两处不冲突。）
        if isinstance(value, str):
            try:
                json.loads(value)       # 校验是合法 JSON
                return value            # SQLite 里本来就是 JSON 字符串，直接用
            except json.JSONDecodeError:
                return None
        return json.dumps(value, ensure_ascii=False)

    return value


def _target_url() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.core.config import settings

    url = settings.database_url
    if not url.startswith("postgresql"):
        raise SystemExit(
            f"目标库不是 PostgreSQL（当前 {url}）。\n"
            "请先把 .env 的 DATABASE_URL 改成 postgresql+asyncpg://... 再跑本脚本。"
        )
    return url


async def migrate(source_path: Path, *, dry_run: bool, truncate: bool, batch: int) -> None:
    target_url = _target_url()
    print(f"源库:   sqlite+aiosqlite:///{source_path}")
    print(f"目标库: {target_url.split('@')[-1]}")
    print(f"模式:   {'DRY-RUN（只读，不写）' if dry_run else '实际搬迁'}")
    print()

    src = create_async_engine(f"sqlite+aiosqlite:///{source_path}")
    # ★ 目标引擎必须和 app/db/session.py 用同样的参数：
    #   asyncpg 不会自动序列化 JSON 列，缺 json_serializer 会在写 cited_chunk_ids 时报
    #   "descriptor 'encode' for 'str' objects doesn't apply to a 'list' object"
    dst = create_async_engine(
        target_url,
        json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
        json_deserializer=json.loads,
    )

    try:
        async with src.connect() as sconn, dst.connect() as dconn:
            src_tables = set(await sconn.run_sync(lambda c: inspect(c).get_table_names()))
            dst_tables = set(await dconn.run_sync(lambda c: inspect(c).get_table_names()))

            print("=== 表存在性检查 ===")
            for t in TABLE_ORDER:
                mark = "源✓ 目标✓" if t in src_tables and t in dst_tables else (
                    f"源{'✓' if t in src_tables else '✗'} 目标{'✓' if t in dst_tables else '✗'}"
                )
                print(f"  {t:16s} {mark}")
            missing = [t for t in TABLE_ORDER if t in src_tables and t not in dst_tables]
            if missing:
                raise SystemExit(
                    f"\n目标库缺少表: {missing}\n请先执行: alembic upgrade head"
                )
            print()

            # 目标表非空检查
            existing: dict[str, int] = {}
            for t in TABLE_ORDER:
                if t not in dst_tables:
                    continue
                n = (await dconn.execute(text(f'SELECT count(*) FROM "{t}"'))).scalar_one()
                existing[t] = int(n)
            if any(existing.values()):
                print("=== 目标库已有数据 ===")
                for t, n in existing.items():
                    if n:
                        print(f"  {t:16s} {n} 行")
                if not truncate and not dry_run:
                    raise SystemExit(
                        "\n目标库非空。确认要覆盖请加 --truncate（会先清空这些表）。"
                    )
                print()

            if truncate and not dry_run:
                print("=== 清空目标表（按外键逆序）===")
                for t in reversed(TABLE_ORDER):
                    if t in dst_tables:
                        await dconn.execute(
                            text(f'TRUNCATE TABLE "{t}" RESTART IDENTITY CASCADE')
                        )
                        print(f"  已清空 {t}")
                await dconn.commit()
                print()

            print("=== 搬迁数据 ===")
            grand_total = 0
            for table in TABLE_ORDER:
                if table not in src_tables or table not in dst_tables:
                    continue
                src_cols = await sconn.run_sync(
                    lambda c, tb=table: {
                        col["name"]: col["type"] for col in inspect(c).get_columns(tb)
                    }
                )
                dst_types = await dconn.run_sync(
                    lambda c, tb=table: {
                        col["name"]: col["type"] for col in inspect(c).get_columns(tb)
                    }
                )
                common = [c for c in src_cols if c in dst_types]
                skipped = [c for c in src_cols if c not in dst_types]

                col_sql = ", ".join(f'"{c}"' for c in common)
                try:
                    result = await sconn.execute(
                        text(f'SELECT {col_sql} FROM "{table}" ORDER BY rowid')
                    )
                except Exception:                     # noqa: BLE001
                    # 非 SQLite 源（没有 rowid）时退化为无序读取
                    result = await sconn.execute(text(f'SELECT {col_sql} FROM "{table}"'))
                rows = result.mappings().all()

                if not rows:
                    print(f"  {table:16s} 源表为空，跳过")
                    continue

                # ★ 按目标列类型做强制转换（见 _coerce 的说明）
                converted = [
                    {
                        c: _coerce(row[c], dst_types[c])
                        for c in common
                    }
                    for row in rows
                ]

                if not dry_run:
                    placeholders = ", ".join(f":{c}" for c in common)
                    stmt = text(
                        f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'
                    )
                    total = 0
                    for i in range(0, len(converted), batch):
                        await dconn.execute(stmt, converted[i : i + batch])
                        total += len(converted[i : i + batch])
                    await dconn.commit()
                    grand_total += total
                    note = f"（跳过目标不存在的列: {skipped}）" if skipped else ""
                    print(f"  {table:16s} 搬入 {total} 行{note}")
                else:
                    print(f"  {table:16s} 将搬入 {len(rows)} 行")

            if dry_run:
                print("\nDRY-RUN 结束，未写入任何数据。去掉 --dry-run 实际执行。")
                return

            # ★ 重置序列：显式插入 id 不会推进 PG 的 SERIAL 序列
            print("\n=== 重置自增序列 ===")
            for table in TABLE_ORDER:
                if table not in dst_tables:
                    continue
                has_id = await dconn.run_sync(
                    lambda c, tb=table: "id"
                    in [col["name"] for col in inspect(c).get_columns(tb)]
                )
                if not has_id:
                    continue
                max_id = (
                    await dconn.execute(text(f'SELECT coalesce(max(id), 0) FROM "{table}"'))
                ).scalar_one()
                if int(max_id) == 0:
                    print(f"  {table:16s} 无数据，跳过")
                    continue
                # pg_get_serial_sequence 对 SERIAL/IDENTITY 都有效
                await dconn.execute(
                    text(
                        "SELECT setval(pg_get_serial_sequence(:t, 'id'), :v, true)"
                    ),
                    {"t": table, "v": int(max_id)},
                )
                print(f"  {table:16s} 序列 → {max_id}")
            await dconn.commit()

            print(f"\n✅ 搬迁完成，共 {grand_total} 行")

            # 搬迁后校验
            print("\n=== 校验（源 vs 目标行数）===")
            ok = True
            for table in TABLE_ORDER:
                if table not in src_tables or table not in dst_tables:
                    continue
                s = (await sconn.execute(text(f'SELECT count(*) FROM "{table}"'))).scalar_one()
                d = (await dconn.execute(text(f'SELECT count(*) FROM "{table}"'))).scalar_one()
                flag = "✅" if int(s) == int(d) else "❌"
                ok = ok and int(s) == int(d)
                print(f"  {flag} {table:16s} 源 {s} / 目标 {d}")
            if not ok:
                raise SystemExit("\n行数不一致，请检查上面的输出。")
    finally:
        await src.dispose()
        await dst.dispose()


def main() -> None:
    p = argparse.ArgumentParser(description="SQLite → PostgreSQL 数据搬迁")
    p.add_argument("--source", default="kb.db", help="SQLite 文件路径（默认 kb.db）")
    p.add_argument("--dry-run", action="store_true", help="只检查与统计，不写入")
    p.add_argument("--truncate", action="store_true", help="先清空目标表再搬")
    p.add_argument("--batch", type=int, default=200, help="批量插入大小")
    a = p.parse_args()

    path = Path(a.source)
    if not path.exists():
        raise SystemExit(f"源文件不存在: {path}")

    asyncio.run(
        migrate(path, dry_run=a.dry_run, truncate=a.truncate, batch=a.batch)
    )


if __name__ == "__main__":
    main()
