"""phase2: chat_sessions/chat_messages + chat_logs.session_id

Revision ID: 86bf317304ac
Revises:
Create Date: 2026-10-12 19:09:17.274063

★ autogenerate 只给出了 chat_logs.session_id 的增列（当时两张新表已被开发中的
  uvicorn --reload 提前 create_all 出来，diff 里看不到）。这里手工补齐建表，
  保证在全新数据库上 `alembic upgrade head` 也能一次建出完整阶段 2 结构。
★ SQLite 不支持 ALTER TABLE ADD CONSTRAINT，增列 + 外键统一走 batch_alter_table
  （表重建），这是 Alembic 官方推荐的 SQLite 迁移姿势。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '86bf317304ac'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('last_active_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_chat_sessions_last_active_at'), 'chat_sessions', ['last_active_at'], unique=False)
    op.create_index(op.f('ix_chat_sessions_owner_id'), 'chat_sessions', ['owner_id'], unique=False)

    op.create_table(
        'chat_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('cited_chunk_ids', sa.JSON(), nullable=False),
        sa.Column('cache_hit', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_chat_messages_session_id'), 'chat_messages', ['session_id'], unique=False)

    # chat_logs.session_id：SET NULL —— 会话删了，成本记录必须留着（§2.3）
    with op.batch_alter_table('chat_logs') as batch_op:
        batch_op.add_column(sa.Column('session_id', sa.Integer(), nullable=True))
        batch_op.create_index(op.f('ix_chat_logs_session_id'), ['session_id'], unique=False)
        batch_op.create_foreign_key('fk_chat_logs_session_id', 'chat_sessions', ['session_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('chat_logs') as batch_op:
        batch_op.drop_constraint('fk_chat_logs_session_id', type_='foreignkey')
        batch_op.drop_index(op.f('ix_chat_logs_session_id'))
        batch_op.drop_column('session_id')

    op.drop_index(op.f('ix_chat_messages_session_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_index(op.f('ix_chat_sessions_last_active_at'), table_name='chat_sessions')
    op.drop_index(op.f('ix_chat_sessions_owner_id'), table_name='chat_sessions')
    op.drop_table('chat_sessions')
