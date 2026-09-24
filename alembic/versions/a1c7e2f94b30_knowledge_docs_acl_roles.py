"""knowledge_docs.acl_roles（文档级 ACL，检索前过滤用）

Revision ID: a1c7e2f94b30
Revises: 01300a7f50fc
Create Date: 2026-09-23

为什么需要这一列：文档检索的权限谓词（knowledge/acl.py）依据的是
chunk 级 acl_roles（Milvus ARRAY 字段）。SQL 侧存同一份，供文档列表
按角色过滤、回填脚本按 doc 定位、以及审计"这份文档谁能看"。
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1c7e2f94b30'
down_revision: str | Sequence[str] | None = '01300a7f50fc'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'knowledge_docs',
        sa.Column('acl_roles', sa.String(length=255), nullable=True,
                  comment='可见角色列表（逗号分隔）'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('knowledge_docs', 'acl_roles')
