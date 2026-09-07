#!/usr/bin/env bash
set -euo pipefail
polaris_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec docker compose --env-file "$polaris_root/.env" \
  -f "$polaris_root/docker/docker-compose.yml" \
  -f "$polaris_root/docker/docker-compose.codex.yml" "$@"
