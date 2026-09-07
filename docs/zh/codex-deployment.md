# 使用 Codex 订阅部署个人 Polaris

LLM 通过官方 `codex exec` 使用 ChatGPT 订阅登录；向量嵌入和重排继续配置独立 API。
这是个人实例配置，网页与后端只监听本机回环地址。Codex 用量受订阅额度限制。

## 首次部署

需要 Docker 和 Compose 2.24.4 或更新版本。以下命令在仓库根目录执行。

1. 将 `.env.example` 复制为 `.env`，设置 `POLARIS_ENV=prod`、随机的
   `POLARIS_SECRET_KEY`、有效 Fernet `POLARIS_ENCRYPTION_KEY` 和随机邀请码。
   同时修改 `POSTGRES_PASSWORD` 及数据库 URL 中对应的密码。
2. 构建 TeX 基础镜像（已有时可复用），然后构建专用镜像：

   ```bash
   make texbase
   docker/codex-compose.sh build
   docker/codex-compose.sh up -d --wait postgres redis
   docker/codex-compose.sh run --rm --no-deps api alembic upgrade head
   ```

3. 为 Polaris 登录一个独立 Codex 会话。认证保存在 `polaris-codex_codex-auth`
   Docker 卷中，API 与 worker 共用，CLI 可以原地刷新凭据：

   ```bash
   docker/codex-compose.sh run --rm --no-deps api codex login --device-auth
   ```

   按终端中的链接和设备码登录。若设备码返回 403，使用浏览器回调：

   ```bash
   docker run --rm --network host \
     --mount type=volume,source=polaris-codex_codex-auth,target=/srv/codex \
     polaris-local/api:codex-subscription \
     codex -c 'cli_auth_credentials_store="file"' login
   ```

   Linux 下此命令在主机 `localhost:1455` 接收回调。若浏览器在另一台电脑，先在那里
   执行 `ssh -L 1455:127.0.0.1:1455 用户@服务器`，保持连接，再打开登录链接。
   登录只能由账号持有人完成。不要将 `auth.json` 提交到仓库或填入 API Key 输入框。

4. 初始化默认 LLM 路由并启动服务：

   ```bash
   docker/codex-compose.sh run --rm --no-deps api python -m app.cli.setup_codex --model gpt-6-astra
   docker/codex-compose.sh up -d
   ```

   初始化命令可重复执行；已有默认路由时保留原配置。模型名可改为订阅实际支持的模型。
   在浏览器打开 `http://127.0.0.1:8080`，使用 `.env` 中的邀请码注册个人账号。
   当前 Polaris 的已登录用户可以管理模型配置，无需额外分配管理员角色。

5. 在“设置 → 大模型”中检查 Codex 登录并测试模型。添加你自己的 API 提供商，将
   `embedding` 与 `rerank` 分别指向对应模型。其他未单独配置的 LLM 环节跟随默认 Codex。

远程访问网页：`ssh -L 8080:127.0.0.1:8080 用户@服务器`，随后打开本机 8080 端口。
数据库和 Redis 不映射主机端口。不要将此订阅实例作为公开模型接口服务。

如果服务器必须通过代理连接 OpenAI，在 `.env` 设置
`POLARIS_CODEX_PROXY_URL=http://host.docker.internal:1080`（替换实际端口）。此配置仅用于
Codex 推理子进程，不改变嵌入或重排 API 的网络设置。设备码登录时可使用：

```bash
docker/codex-compose.sh run --rm --no-deps \
  -e HTTPS_PROXY=http://host.docker.internal:1080 api codex login --device-auth
```

已有本机 CLI 登录时，也可按官方认证文档将 `auth.json` 安全复制到正在运行的
API 容器的 `/srv/codex/auth.json`，仅复制此文件并保持权限 0600。它保存在专用卷中；
长期运行建议为 Polaris 单独登录，避免两个副本刷新同一会话时互相影响。

## 接入行为

- 普通问答、文献分析、结构化抽取、写作、评审与图片输入走同一个 Codex 提供商。
- CLI 按完整消息输出，等待时页面保持运行状态，完成后显示正文；SSE 接口保持兼容，
  不模拟逐字输出。停止请求会终止子进程组并清理临时文件。
- 工具调用通过受 JSON Schema 约束的响应表达，再由 Polaris 验证和执行。
  工具结果及图片会连同历史传入下一轮。Codex 自身的命令、搜索、插件等执行功能禁用。
- `temperature` 不传给 Codex；`max_tokens` 仅作为回答长度提示，不是严格输出上限。
  推理档位原样传递，不支持的模型或档位会报告错误。
- 多个 API/worker 调用通过共享文件锁串行执行。排队与执行分别默认超时 300 秒，
  可在 `.env` 修改 `POLARIS_CODEX_QUEUE_TIMEOUT_SECONDS` 和 `POLARIS_CODEX_TIMEOUT_SECONDS`。
- 不向 Codex 请求嵌入、重排或语音。未配置嵌入和重排 API 时沿用项目现有检索降级。
- 实验规划、生成代码和分析结果可使用 Codex。实验代码若设置 `eval_model`、要求导出
  可直接调用的模型 API 凭据，会得到 `CODEX_EVAL_API_UNSUPPORTED`：清空该参数，或为
  需要直连 API 的实验配置独立 API 提供商。订阅登录凭据不会发送到实验服务器。

## 检查、升级与恢复

```bash
docker/codex-compose.sh ps
docker/codex-compose.sh exec api codex login status
docker/codex-compose.sh logs --tail=100 api worker
docker/codex-compose.sh restart api worker
```

CLI 固定为 `0.153.4`；升级时修改构建变量 `CODEX_VERSION` 并重新跑兼容测试。
日常升级运行 `docker/codex-compose.sh build`、数据库迁移和 `up -d` 即可。
专用镜像名为 `polaris-local/{api,worker,frontend}:codex-subscription`，Compose 项目名为
`polaris-codex`。停止用 `docker/codex-compose.sh down`；不要加 `-v`，否则会删除数据和登录卷。

| 错误 | 处理方式 |
| --- | --- |
| `CODEX_CLI_NOT_FOUND` | 使用 Codex overlay 构建镜像，检查 `codex --version`。 |
| `CODEX_NOT_LOGGED_IN` | 在专用认证卷重新运行上述登录命令。 |
| `CODEX_RATE_LIMIT` | 等待订阅额度或速率限制恢复；不会自动改用付费 API。 |
| `CODEX_NETWORK_ERROR` | 检查 OpenAI 连通性；容器需要代理时设置 `POLARIS_CODEX_PROXY_URL`。 |
| `CODEX_MODEL_UNAVAILABLE` | 在设置页更换订阅支持的模型或推理档位。 |
| `CODEX_QUEUE_TIMEOUT` | 等待正在执行的任务完成，或调整排队超时。 |
| `CODEX_TIMEOUT` | 本次执行已终止；缩短输入或调整执行超时。 |
| `CODEX_OUTPUT_INVALID` | 回答未符合协议或工具参数无效，检查模型及 CLI 版本后重试。 |

调用日志只记录 Polaris 的输入、回答和脱敏错误，不记录 CLI 原始诊断或登录凭据。
认证目录应与应用数据一起妥善保存；官方认证和非交互文档：
[Authentication](https://learn.chatgpt.com/docs/auth)、
[Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)。
