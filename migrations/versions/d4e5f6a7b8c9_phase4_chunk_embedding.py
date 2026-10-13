"""phase4: chunks 加 embedding 向量列（pgvector Vector(512)）

Revision ID: d4e5f6a7b8c9
Revises: bcc942ff68e3
Create Date: 2026-10-13 21:30:00.000000

★ 按方言分支（沿用阶段 3 迁移的经验）：
  - PostgreSQL：先 `CREATE EXTENSION IF NOT EXISTS vector`（幂等，保证任何新环境
    一条 `alembic upgrade head` 就够，不用额外记一句 SQL），再原生 add_column。
    chunks 无子表外键，add_column 不触发重建，安全。
  - SQLite：没有 vector 类型，直接跳过。测试环境用 conftest 的 Vector→JSON 类型映射
    走 create_all（docs/项目实现文档-阶段4.md §9.1），不依赖本迁移。

维度 512 与 embedding 模型（bge-small-zh-v1.5）硬绑定；换模型需同步改这里与 config。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'bcc942ff68e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    """Upgrade schema."""
    if _is_sqlite():
        # SQLite 无 vector 类型；测试库靠 conftest 的类型映射，这里直接跳过。
        return
    # ★ 幂等建扩展：本机已手动装过，但写进迁移保证任何全新环境都能一键升级。
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "chunks",
        sa.Column("embedding", Vector(512), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    if _is_sqlite():
        return
    op.drop_column("chunks", "embedding")
