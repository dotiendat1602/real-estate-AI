"""initial schema

Revision ID: 4417bcccb44c
Revises: 
Create Date: <giữ nguyên>
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from pgvector.sqlalchemy import Vector

revision: str = '4417bcccb44c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    
    # Enable extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create schema
    op.execute("CREATE SCHEMA IF NOT EXISTS ai")
    
    # Langchain collection
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.langchain_pg_collection (
            uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR,
            cmetadata JSON
        )
    """)
    
    # Langchain embeddings
    if 'langchain_pg_embedding' not in inspector.get_table_names(schema='ai'):
        op.create_table(
            'langchain_pg_embedding',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('collection_id', sa.UUID(), nullable=True),
            sa.Column('embedding', Vector(), nullable=True),
            sa.Column('document', sa.String(), nullable=True),
            sa.Column('cmetadata', sa.JSON(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            schema='ai',
        )
    
    # Chat sessions
    if 'chat_sessions' not in inspector.get_table_names(schema='ai'):
        op.create_table('chat_sessions',
            sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column('user_id', sa.BigInteger(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id'),
            schema='ai'
        )
        op.create_index(op.f('ix_ai_chat_sessions_user_id'), 'chat_sessions', ['user_id'], unique=False, schema='ai')
    
    # Chat messages
    if 'chat_messages' not in inspector.get_table_names(schema='ai'):
        op.create_table('chat_messages',
            sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column('session_id', sa.BigInteger(), nullable=False),
            sa.Column('role', sa.Text(), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['session_id'], ['ai.chat_sessions.id']),
            sa.PrimaryKeyConstraint('id'),
            schema='ai'
        )
        op.create_index(op.f('ix_ai_chat_messages_session_id'), 'chat_messages', ['session_id'], unique=False, schema='ai')

def downgrade() -> None:
    op.drop_index(op.f('ix_ai_chat_messages_session_id'), table_name='chat_messages', schema='ai', if_exists=True)
    op.drop_table('chat_messages', schema='ai', if_exists=True)
    op.drop_index(op.f('ix_ai_chat_sessions_user_id'), table_name='chat_sessions', schema='ai', if_exists=True)
    op.drop_table('chat_sessions', schema='ai', if_exists=True)
    op.drop_table('langchain_pg_embedding', schema='ai', if_exists=True)
    op.execute("DROP TABLE IF EXISTS ai.langchain_pg_collection CASCADE")
