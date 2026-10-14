"""phase5: chat_logs 加 Agent 路径统计三列（is_agent / agent_steps / agent_tools）

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-14 10:00:00.000000

★ 阶段 5（docs/项目实现文档-阶段5.md §3）不新增表，只在 chat_logs 上补三列，
  把「快速路径」与「Agent 路径」的成本/延迟/工具调用分开统计（§7.2 的成本对比 SQL）。

方言说明：
  - 三列都是普通类型（Boolean / Integer / JSON），PostgreSQL 与 SQLite 都原生支持，
    无需像阶段 4 那样按方言分支。
  - server_default 用于**回填历史行**：迁移前已有日志（全部是快速路径），
    补默认值让 §7.2 的 `GROUP BY is_agent` 不会多出一个 NULL 分组。
  - 测试环境仍走 conftest 的 create_all，不依赖本迁移。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # is_agent 带索引：成本对比 SQL 按它分组（模型里 index=True 对应）
    op.add_column(
        "chat_logs",
        sa.Column("is_agent", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(
        op.f("ix_chat_logs_is_agent"), "chat_logs", ["is_agent"], unique=False
    )
    op.add_column(
        "chat_logs",
        sa.Column("agent_steps", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "chat_logs",
        sa.Column("agent_tools", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("chat_logs", "agent_tools")
    op.drop_column("chat_logs", "agent_steps")
    op.drop_index(op.f("ix_chat_logs_is_agent"), table_name="chat_logs")
    op.drop_column("chat_logs", "is_agent")
