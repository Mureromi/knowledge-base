"""phase3b: processing_tasks 保留策略 —— document_id 改 SET NULL + 可空

验收修复（阶段 3 验收报告 §5.2）：
原设计 documents 删除时级联删除 processing_tasks，与文档 §4.2 声称的
"可追溯重试历史、统计失败率"直接矛盾 —— 文档一删，历史就出现缺口。

改为 SET NULL + document_id 可空：任务流水（含失败原因）在文档删除后仍然保留，
可以继续用于失败率统计与排查。这与 chat_logs.session_id 的策略一致
（审计数据与业务数据的生命周期本就不同）。

Revision ID: bcc942ff68e3
Revises: a3f1c7d92b40
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "bcc942ff68e3"
down_revision: Union[str, Sequence[str], None] = "a3f1c7d92b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    """document_id 放开非空 + 外键行为改为 SET NULL。"""
    if _is_sqlite():
        # SQLite 不支持直接改外键，必须走 batch_alter_table 重建表
        with op.batch_alter_table("processing_tasks", schema=None) as batch_op:
            batch_op.alter_column(
                "document_id", existing_type=sa.Integer(), nullable=True
            )
            batch_op.drop_constraint(
                "processing_tasks_document_id_fkey", type_="foreignkey"
            )
            batch_op.create_foreign_key(
                "processing_tasks_document_id_fkey",
                "documents",
                ["document_id"],
                ["id"],
                ondelete="SET NULL",
            )
    else:
        op.alter_column(
            "processing_tasks", "document_id",
            existing_type=sa.Integer(), nullable=True,
        )
        op.drop_constraint(
            "processing_tasks_document_id_fkey",
            "processing_tasks",
            type_="foreignkey",
        )
        op.create_foreign_key(
            "processing_tasks_document_id_fkey",
            "processing_tasks",
            "documents",
            ["document_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """回滚：外键恢复 CASCADE（注意：置空的历史任务需先清理才能恢复非空）。"""
    op.execute("DELETE FROM processing_tasks WHERE document_id IS NULL")
    if _is_sqlite():
        with op.batch_alter_table("processing_tasks", schema=None) as batch_op:
            batch_op.drop_constraint(
                "processing_tasks_document_id_fkey", type_="foreignkey"
            )
            batch_op.create_foreign_key(
                "processing_tasks_document_id_fkey",
                "documents",
                ["document_id"],
                ["id"],
                ondelete="CASCADE",
            )
            batch_op.alter_column(
                "document_id", existing_type=sa.Integer(), nullable=False
            )
    else:
        op.drop_constraint(
            "processing_tasks_document_id_fkey",
            "processing_tasks",
            type_="foreignkey",
        )
        op.create_foreign_key(
            "processing_tasks_document_id_fkey",
            "processing_tasks",
            "documents",
            ["document_id"],
            ["id"],
            ondelete="CASCADE",
        )
        op.alter_column(
            "processing_tasks", "document_id",
            existing_type=sa.Integer(), nullable=False,
        )
