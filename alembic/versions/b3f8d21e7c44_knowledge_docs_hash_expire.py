"""knowledge_docs.content_hash / expire_date（内容指纹 + 时效过滤）

Revision ID: b3f8d21e7c44
Revises: a1c7e2f94b30
Create Date: 2026-09-29

为什么需要这两列：
- content_hash：文件内容 SHA-256。同名重传且内容未变时跳过整条解析/嵌入
  管线；不同文件名命中相同 hash 时上传侧 409 提示重复入库。存量行为 NULL
  ——首次重传会走全量入库并顺带回填，无需数据迁移。
- expire_date：失效日期（YYYY-MM-DD，空=永久有效）。过期后检索侧谓词不再
  召回该文档 chunk（写入 Milvus 时换算为 expire_ts 时间戳，见 pipeline.py）。
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b3f8d21e7c44'
down_revision: str | Sequence[str] | None = 'a1c7e2f94b30'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'knowledge_docs',
        sa.Column('content_hash', sa.String(length=64), nullable=True,
                  comment='文件内容 SHA-256 指纹'),
    )
    op.add_column(
        'knowledge_docs',
        sa.Column('expire_date', sa.String(length=10), nullable=True,
                  comment='失效日期 YYYY-MM-DD，空=永久有效'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('knowledge_docs', 'expire_date')
    op.drop_column('knowledge_docs', 'content_hash')
