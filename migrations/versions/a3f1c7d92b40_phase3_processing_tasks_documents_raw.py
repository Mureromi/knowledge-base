"""phase3: processing_tasks + documents.raw_content + raw_text 改为可空

Revision ID: a3f1c7d92b40
Revises: 86bf317304ac
Create Date: 2026-10-13 20:15:00.000000

★ 手写而非 autogenerate：
  1. documents.raw_text 从 NOT NULL 改为 NULL —— 异步路径建 pending 文档时全文尚未解析，
     必须允许为空；旧库由 create_all 建成了 NOT NULL，不改会在新代码 INSERT None 时报约束错。
  2. ★ 按方言分支：SQLite 不支持 ALTER COLUMN，走 batch_alter_table（表重建）；
     PostgreSQL 用原生 add_column / alter_column。不能对 PostgreSQL 用 batch——
     documents 有 chunks / processing_tasks 两个子表外键引用它，batch 重建会 DROP 再改名 documents，
     在 PG 上会因依赖而失败（阶段 2 的 batch 只动 chat_logs 那种无子表的表才安全）。
  3. 先改 documents 再建引用它的 processing_tasks。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f1c7d92b40'
down_revision: Union[str, Sequence[str], None] = '86bf317304ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    """Upgrade schema."""
    # ---- documents：加原始字节列 + raw_text 放开非空约束 ----
    if _is_sqlite():
        with op.batch_alter_table('documents') as batch_op:
            batch_op.add_column(sa.Column('raw_content', sa.LargeBinary(), nullable=True))
            batch_op.alter_column('raw_text', existing_type=sa.Text(), nullable=True)
    else:
        op.add_column('documents', sa.Column('raw_content', sa.LargeBinary(), nullable=True))
        op.alter_column('documents', 'raw_text', existing_type=sa.Text(), nullable=True)

    # ---- processing_tasks：任务事件流水表 ----
    op.create_table(
        'processing_tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('job_id', sa.String(length=64), nullable=True),
        sa.Column('attempt', sa.Integer(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('char_count', sa.Integer(), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_processing_tasks_document_id'), 'processing_tasks', ['document_id'], unique=False)
    op.create_index(op.f('ix_processing_tasks_owner_id'), 'processing_tasks', ['owner_id'], unique=False)
    op.create_index(op.f('ix_processing_tasks_status'), 'processing_tasks', ['status'], unique=False)
    op.create_index(op.f('ix_processing_tasks_job_id'), 'processing_tasks', ['job_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_processing_tasks_job_id'), table_name='processing_tasks')
    op.drop_index(op.f('ix_processing_tasks_status'), table_name='processing_tasks')
    op.drop_index(op.f('ix_processing_tasks_owner_id'), table_name='processing_tasks')
    op.drop_index(op.f('ix_processing_tasks_document_id'), table_name='processing_tasks')
    op.drop_table('processing_tasks')

    # 注意：若库中已存在 raw_text 为 NULL 的 pending 文档，回滚会失败——
    # 开发环境需先清空这类记录。生产回滚本就应人工把关。
    if _is_sqlite():
        with op.batch_alter_table('documents') as batch_op:
            batch_op.drop_column('raw_content')
            batch_op.alter_column('raw_text', existing_type=sa.Text(), nullable=False)
    else:
        op.drop_column('documents', 'raw_content')
        op.alter_column('documents', 'raw_text', existing_type=sa.Text(), nullable=False)
