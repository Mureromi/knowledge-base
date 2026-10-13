"""数据库基线初始化：建表 + 给 Alembic 打标。

## 为什么需要这个脚本（重要）

本项目的迁移链**不完整**：阶段 0/1 的表（users / documents / chunks / chat_logs）
是用 `Base.metadata.create_all()` 建的，从来没有生成过基线迁移文件。
唯一的迁移（phase2）只创建了 chat_sessions / chat_messages 并给 chat_logs 加列，
它**假设前面那些表已经存在**。

所以在全新数据库上直接 `alembic upgrade head` 会失败：
    FOREIGN KEY(owner_id) REFERENCES users (id)   →  users 表不存在

正确做法分两步：
    1. 用 ORM 模型建出**完整** schema（模型是事实源，与代码永远一致）
    2. `alembic stamp head` 让 Alembic 认为基线已应用，之后的增量迁移能正常接续

## 用法

    # 全新数据库初始化
    .venv/bin/python scripts/init_db.py --stamp

    # 只建表、不打标（例如你只想验证 schema）
    .venv/bin/python scripts/init_db.py

    # 顺便清空重建（危险：会删掉所有数据）
    .venv/bin/python scripts/init_db.py --drop --stamp

> 注意：`--drop` 会 DROP 所有业务表。生产环境绝对不要用。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.models.base import Base
# ★ 必须导入所有模型模块，否则 Base.metadata 里没有这些表
import app.models  # noqa: F401


def _describe(url: str) -> str:
    if "@" in url:
        return url.split("@", 1)[1]
    return url


async def init(*, drop: bool, stamp: bool, echo: bool) -> None:
    url = settings.database_url
    backend = url.split("+", 1)[0].split(":", 1)[0]
    print(f"目标库: {_describe(url)}")
    print(f"方言:   {backend}")
    print(f"模式:   {'DROP + 重建' if drop else '建表（已存在的表不动）'}")
    print()

    engine = create_async_engine(url, echo=echo)

    try:
        async with engine.begin() as conn:
            existing = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
            model_tables = set(Base.metadata.tables)
            print("=== 表现状 ===")
            for t in sorted(model_tables):
                print(f"  {t:16s} {'已存在' if t in existing else '待创建'}")
            orphan = existing - model_tables - {"alembic_version"}
            if orphan:
                print(f"  （库中另有模型未定义的表: {sorted(orphan)}）")
            print()

            if drop:
                print("=== 删除所有业务表 ===")
                await conn.run_sync(Base.metadata.drop_all)
                print("  已删除")
                print()

            print("=== 创建缺失的表 ===")
            # checkfirst=True 是默认行为：已存在的表会跳过
            await conn.run_sync(Base.metadata.create_all)
            after = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
            created = sorted(model_tables & after)
            print(f"  现在存在 {len(created)} 张表: {created}")

        if stamp:
            print("\n=== 给 Alembic 打标 ===")
            from alembic import command
            from alembic.config import Config

            cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
            cfg.set_main_option("sqlalchemy.url", url)
            # 用 run_sync 跑同步的 alembic API（它内部自己建连接）
            await asyncio.to_thread(command.stamp, cfg, "head")
            print("  已 stamp head：Alembic 现在认为基线已应用，后续增量迁移可正常接续")

        # 最终校验
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
            print(f"\n✅ 完成。库中共 {len(tables)} 张表：")
            for t in sorted(tables):
                n = (await conn.execute(text(f'SELECT count(*) FROM "{t}"'))).scalar_one()
                print(f"  {t:16s} {n} 行")
    finally:
        await engine.dispose()


def main() -> None:
    p = argparse.ArgumentParser(description="数据库基线初始化（建表 + Alembic 打标）")
    p.add_argument("--drop", action="store_true", help="先删除所有业务表（危险）")
    p.add_argument("--stamp", action="store_true", help="建表后执行 alembic stamp head")
    p.add_argument("--echo", action="store_true", help="打印 SQL")
    a = p.parse_args()
    asyncio.run(init(drop=a.drop, stamp=a.stamp, echo=a.echo))


if __name__ == "__main__":
    main()
