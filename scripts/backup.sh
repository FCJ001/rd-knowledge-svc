#!/usr/bin/env bash
# ============================================================
# 数据备份：PG 逻辑备份 + MinIO 对象镜像 +（可选）Neo4j/Milvus 冷备
#
# 用法:
#   ./scripts/backup.sh [备份目录]            # 默认 ./backups/<时间戳>
#   COLD_BACKUP=1 ./scripts/backup.sh         # 追加 Neo4j/Milvus 数据卷冷备（会短暂停服务）
#
# 行为:
#   - PostgreSQL（rd_knowledge + trulens_eval）: pg_dump 在线一致性备份，随时可跑
#   - MinIO(knowledge-docs 桶): mc mirror 对象级镜像（容器内自带 mc），随时可跑
#   - Neo4j / Milvus: community/standalone 版无在线备份，COLD_BACKUP=1 时
#     stop → tar 数据卷 → start（建议维护窗口执行）；默认跳过并给出警告
#
# 定时（示例，每天 03:30）:
#   30 3 * * * cd /path/to/rd-knowledge-svc && ./scripts/backup.sh >> logs/backup.log 2>&1
#
# 恢复:
#   PG:    gunzip -c rd_knowledge.sql.gz | docker compose exec -T postgres psql -U rdagent -d rd_knowledge
#   MinIO: mc mirror <备份目录>/minio/<桶> <目标别名>/<桶>
#   卷冷备: docker compose stop <服务> → 清空对应卷 → docker run --rm -v <卷>:/data alpine \
#          tar xzf /backup/<卷>.tgz -C /data → docker compose start <服务>
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%F_%H%M)
BACKUP_DIR=${1:-./backups/$STAMP}
COLD_BACKUP=${COLD_BACKUP:-0}
PG_USER=${DB_USER:-rdagent}
MINIO_BUCKET=${MINIO_BUCKET:-knowledge-docs}
mkdir -p "$BACKUP_DIR"

log() { echo "[$(date +%T)] $*"; }

# ---- PostgreSQL（在线，一致性由 pg_dump 保证）----
for db in rd_knowledge trulens_eval; do
  if docker compose exec -T postgres psql -U "$PG_USER" -lqt 2>/dev/null | cut -d'|' -f1 | grep -qw "$db"; then
    docker compose exec -T postgres pg_dump -U "$PG_USER" -d "$db" | gzip > "$BACKUP_DIR/${db}.sql.gz"
    log "PG $db 备份完成 ($(du -h "$BACKUP_DIR/${db}.sql.gz" | cut -f1))"
  else
    log "警告: PG 中不存在 $db，跳过"
  fi
done

# ---- MinIO（对象级镜像，在线）----
docker compose exec -T minio sh -c \
  "mc alias set local http://localhost:9000 \"\$MINIO_ROOT_USER\" \"\$MINIO_ROOT_PASSWORD\" >/dev/null \
   && mc mirror --overwrite \"local/$MINIO_BUCKET\" \"/tmp/mc-mirror\"" \
  && docker compose cp minio:/tmp/mc-mirror "$BACKUP_DIR/minio" \
  && docker compose exec -T minio rm -rf /tmp/mc-mirror
log "MinIO $MINIO_BUCKET 镜像完成"

# ---- 数据卷冷备辅助 ----
vol_name() { docker volume ls --format '{{.Name}}' | grep -m1 "$1\$"; }
tar_volume() { # $1=卷名后缀 $2=输出文件名
  local v; v=$(vol_name "$1")
  if [ -z "$v" ]; then log "警告: 找不到卷 *$1，跳过"; return 0; fi
  docker run --rm -v "$v":/data:ro -v "$(cd "$BACKUP_DIR" && pwd)":/backup alpine \
    tar czf "/backup/$2" -C /data .
  log "$2 备份完成 ($(du -h "$BACKUP_DIR/$2" | cut -f1))"
}

if [ "$COLD_BACKUP" = "1" ]; then
  log "COLD_BACKUP=1：停服务做 Neo4j/Milvus 冷备"
  docker compose stop neo4j milvus
  tar_volume neo4j_data "neo4j_data.tgz"
  tar_volume milvus_data "milvus_data.tgz"
  docker compose start neo4j milvus
else
  log "警告: 跳过 Neo4j/Milvus（community/standalone 无在线备份）。"
  log "      完整备份请在维护窗口执行: COLD_BACKUP=1 ./scripts/backup.sh"
fi

log "全部完成 → $BACKUP_DIR"
