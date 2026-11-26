# Hy4-Flash SQ-128 · 部署与调研全档

> 本仓库是 **Tencent Hy4-Flash SQ-128** 在 AMD Ryzen AI Max+ 395（Strix Halo / gfx1151 / 128GB 统一内存）上的**专属资料库**。总纲 `AI-MAX-395` 只留基线摘要，详细机制、压测、排障过程全部归这里。
>
> 建立 2026-11-24。引擎 `strata-hy4` 0.1.67，**320k** 上下文（原生上限 1M，受 128 GB 内存约束），已正式投产（替代删除的 Flash-Next）。

---

## 1. 一句话状态

**生产主力**：hy4flash SQ-128 @ **320k**，DFly2 草稿加速 + `HY4_KV_HOST=1` + SSD convcache；systemd `strata.service` 固化，`qie_ai strata` 管理；编程 6/6、TG 26–30 t/s、多轮 cache 复用 471× 加速。

---

## 2. 权重与引擎

### 2.1 权重（ModelScope `strata-engine/Hy4-Flash-Strata`，curl 单流下载，合计 182.16 GB）
| 文件 | 大小 | sha256 校验 |
|---|---|---|
| `Hy4-Flash-SQ-128.strata` | 113,654,603,776 B | `6e8ba2729465f38ed42b5bfa7db02c92beddc25ca0413b7f0c24ab9f1e8b51f4` |
| `Hy4-Flash-Engram-Q8.strata` | 67,584,905,216 B | `ab0cf9b3...` |
| `Hy4-Flash-Vision.strata` | 934,092,800 B | `ef09bce...` |

⚠️ **下载教训**：最初脚本把 SQ-128 的校验哈希抄成 SQ-96 的（`3b1c5755…`），首次校验 FAILED——实算与 HF 模型卡 + ModelScope API 一致，文件完好。**多档位模型要逐档核对哈希归属**。

### 2.2 引擎：`strata-hy4` 0.1.67
- **官方预编译 portable 包**：`BUILD.json` = `{"version":"0.1.67","source":"f-0221-rc1 4da72cca05cc","variant":"halo","shape":"hy4","archs":["gfx1151"],"portable":true}`。**无需源码编译**（对比 Flash-Next 时代要 checkout tag 本机 HIP 编译）。
- 安装位置：`~/.local/share/strata/components/shapes/hy4/engine/0.1.67-linux-x86_64-hip-halo/`。0.1.66 已删。
- 升级 0.1.66→0.1.67 的过程：CLI 下载卡在 96MB 引擎包（GitHub 时通时不通），最终手动 `curl -C -` 断点续传补完 CLI 的 `.part`（sha256 `b16d89d5...` 校验通过）→ 解压到引擎目录 → CLI 识别。
- **DFly2 草稿加速在用**：config `drafting: dfly2`，引擎日志 `[spec]` 活跃，accepted ~2.7 drafts/step、tree_steps 20+。

### 2.3 选型：SQ-128 + 320k
- SQ-128（113.65 GB 权重 + 67.58 GB Engram + 0.93 GB Vision）KL 0.088 最接近原模型。
- **上下文上限订正（2026-11-26）**：权重 manifest 的 `max_position_embeddings` 是 **1048576（原生 1M）**，不是 262144——早期"256k 是训练长度、1M 需 rope scaling 外推"的判断有误；Hy4 引擎也不接受 `--rope-scaling`（那是 Flash-Next 的参数）。真正的约束是内存，不是模型能力。
- 用户判据：「比 27B 快、智商打平就是大赚」「**能复用 cache 才是王道**」——混注意力 KV 很小（256k 时 `kv cache` 仅 3.92 GB），内存余量足以开 convcache。

---

## 3. 256k OOM 的解决（排障全过程）

### 3.1 问题
首次 256k 加载在 `hy4_graph.hip.h:1274/1287` **HIP OOM**（规划 130.4 GB > GTT 124 GiB）。当时判断"256k 不行"，降 128k 部署。后来推翻。

### 3.2 源码里找到的开关（`~/Strata/engine/src/hy4flash/`）
| 开关 | 作用 | 效果 |
|---|---|---|
| `--kv q8` / `HY4_KV_Q8=1` | KV 2.72→1.36 GB | 部分有效 |
| **`HY4_KV_HOST=1`** | 稀疏层 KV/IndexPool 挪到 pinned host memory | **★★★ 关键**，注释明说"128K fits 8GB cards" |
| `HY4_RESERVE_GB` | 手动覆盖 reserve | 精细调节 |

### 3.3 实测：`HY4_KV_HOST=1` 救回 256k
- plan 行：`host mask 0→1`、`reserve 8.91→3.85 GiB`、`[runner] sparse KV cache 0.30 GB in pinned host memory`、`kv cache 3.92 GB`。
- `READY 262144`、`max_model_len 262144`、冒烟 1.07s 出 token。**降 128k 的妥协作废**。
- 注意：CLI `strata launch` **不透传 HY4_KV_HOST**，直连 256k 必 OOM；必须 serve + config（见 §6）。

### 3.4 长上下文压测（验证 host KV 的速度代价）
| 指标 | 100k 档 | **210k 档** | 128k 基线 |
|---|---|---|---|
| prompt token | 101,972 | **212,852** | ~300 |
| 答案（末尾 1+1） | 2 ✅ | **2 ✅** | — |
| prefill 速度 | 618.4 t/s | **577.0 t/s** | 330-460 t/s |
| prefill 耗时 | 165 s | **369 s（6.2 分钟）** | <2 s |
| 生成 TG | 27.8 t/s | **29.8 t/s** | 26-30 t/s |
| wall | 167 s | **371 s** | ~1 s |

**结论**：host KV 几乎不拖慢生成（TG 与 128k 持平）；prefill 线性增长但吞吐稳定；21 万 token 后理解正常。

### 3.5 从 256k 扩到 320k（2026-11-26）

**内存账**（引擎 `[plan]` 实测；KV 按 `--max-context` 全量预分配）：

| ctx | buffers | KV+idx | 实际 `kv cache` | reserve |
|---|---|---|---|---|
| 262144 | 4.27 GB | 5.44 GB | 3.92 GB | — |
| **327680** | **4.80 GB** | **6.80 GB** | **4.83 GB** | **6.76 GiB** |

12 层 sparse MLA KV 各 0.30 GB 常驻 pinned host memory（`HY4_KV_HOST=1`），实测远小于按 34.7 KB/token 的保守估算。**每 +64K ≈ +1.89 GB**：384K 后 reserve 仅 ~5 GiB、448K 起 <3.2 GiB，不建议硬拉；更长的正路是换 SQ-96（footprint 89.43 GB，省 23.6 GB，可上 512K，代价平均位宽 5.7→4.5 bit）。page cache 由内核管理，不手动 drop（`available` 已含可回收部分）。

**为什么需要**：长上下文 agent 的 prompt 中位 **143K**、峰值 **298–303K**，262144 会截断。成本结构不对称：`session-reuse` 命中时 298K prefill 仅 **23 s（12,966 tok/s）**，冷读 143K 需 **232 s（~617 tok/s）**。

---

## 4. 多轮 cache 复用：convcache 机制与排障全过程

### 4.1 两套 cache，别混淆
| | serve 层 Python cache | 引擎 hy4 convcache |
|---|---|---|
| 参数 | `--conversation-cache-mib N` | `--conversation-cache-dir <dir>` |
| 状态显示 | /metrics `conversation_cache.enabled` | 引擎 INFO `conversation_cache=N` |
| 适用 | 老 Qwen 引擎 | **hy4flash 专用** |
| 实测 | enabled=false（对 hy4 无效）| **有效** |

### 4.2 convcache 机制（`~/Strata/engine/src/hy4flash/run/hy4_convcache.h`）
- **本质**：SSD 会话缓存——把驻留序列的 KV / SWA-ring / IndexPool 快照 park 成磁盘文件（`H4CC0001` 格式，512B header），另一 prompt 占 GPU 时换出，原对话回来时 restore；支持回退到公共前缀（live + prompt-end 两套 ring 快照，`rewind up to 8160 tokens`）。
- **关键副作用**：源码 `reuse = session_reuse || cc.on`——**开了 convcache 才启用复用判断链**。
- 参数：`--conversation-cache-dir`（空=off）、`--conversation-cache-gib`、`--conversation-cache-min-tokens`。**当前生产值 240 GiB / 2048**（演进：40 GiB/64 → 120 GiB/2048 → 240 GiB/2048，2026-11-26）。
- 空间账：一条 87 token 对话 park 成 **0.60 GB** 文件（SWA ring 快照固定开销大），40 GiB ≈ 66 条短对话；长条目更大（303K tokens 的会话 park 后 6.89 GB）。
- 命中率实测：120 GiB 时 13h **0 drop** / 命中 21.7%；18h 跑满后 12 drop / 全会话 14.2%（淘汰的是旧会话，非配置过小）；240 GiB 后 2h 内 34 parked / 123.33 GiB。
- live session 优先：同对话连续追问时会话还在 GPU，直接 RESUME（不落盘，最快）；多对话交替才 park。

### 4.3 排障全过程（为什么 `session-reuse 1` 单开没用）
第一轮测试：**五种方式全部 `cached_tokens=0`、prefill 全额重读 28.8s**：
1. 两次完全相同 prompt 2. no_think 正常输出后再问 3. `strata_prefix` 标记长文档 4. 标准多轮（带 assistant 回复）5. 前缀精确延长（符合源码 RESUME 条件）

铁证 `/metrics`：`conversation_cache.enabled: false`、requests=7、`requests_reused: 0`。

**用引擎原生 stdin 协议排除 serve 层**（serve 同款 tokenizer 把 prompt 转 token id，`GEN 32 id,id,...` 连发两轮）：
- 前缀严格一致 + 正常 stop —— 仍 `PP 24 24` 全量重读、**无 RESUME**。
- **根因定位**（`hy4_serve.hip.h`）：RESUME 条件 line456 `if (... && (int)tok.size() >= fwd)`，但生成路径 line592 把 `fwd = pos+1`（含生成 token），而 `tok` 只存 prompt ⇒ 正常问答后 `tok.size() < fwd` 恒不成立 ⇒ 复用分支跳过。
- **升级 0.1.67 未修复**——同样无 RESUME。

**最终解法**：开 `--conversation-cache-dir` 后，原生协议第二轮输出 `RESUME 13`，HTTP 层打通：

| | 第1轮（冷） | 第2轮（相同对话） |
|---|---|---|
| cached_tokens | 0 | **18,996** |
| prefill | 28,139 ms | **60 ms（471×）** |
| 答案 | 2 ✅ | 2 ✅ |

metrics：`requests_reused=1, reused_tokens=18996`。

### 4.4 风险
- convcache 磁盘占用：监控 `~/models/strata-hy4/convcache-prod/` 与 /metrics parks/evictions。
- RAM 贴边：加载后可用 3–6 GiB，`min-free-mib 2560` 空间不足时自动跳过 park（不崩，只是不复用）。

---

## 5. 验收：智商 A/B（生产口径三方对照）

编程客观题同卷（生产配置、xhigh 档、脚本自动判分；原始答卷已清理）：

| 模型 | 得分 | TG |
|---|---|---|
| **hy4flash SQ-128** | **6/6** | 26–30 t/s |
| Flash-Next xhigh（已删） | 5/6 | 36+ t/s |
| 27B Q8 xhigh | 4/6（T3/T4 空卷=思考吃满 32768 额度未交代码）| 14–15 t/s |

⚠️ 判分基准曾手算出错 3 次，已用穷举核准。速度门禁（≥27B）与智商门禁（编程明显不掉档）均过。中途两次死机由 Flash-Next 重启未等 GTT 归零导致，与 hy4flash 无关。

---

## 6. 生产配置（可复现）

### 6.1 启动方式
CLI `strata launch` **不可用**：① 不透传 `HY4_KV_HOST` → 256k OOM ② 触发引擎更新检查。必须 serve + config。

> ⚠️ **config 必备字段 `api_key`（2026-11-25 教训）**：生产 config `hy4-256k-067-convcache.json` 必须带 `api_key`（与 llamacpp 的 `--api-key` 同一个 token，本仓库不存明文）。缺此项时 Strata 的 DNS rebinding 防护会对云端域名返回 `403 Host '<your-domain>' is not allowed`，公网端点全挂；设了 `api_key` 会同时关闭该 host 检查。11-24 换 config 时漏带 → WorkBuddy/OpenCode 经公网全部 403，补回重启后恢复。**换 config 时逐项核对 `api_key`**；本机无 key 401 / 有 key 200 是验收基线。

**strata.service（已固化，qie_ai strata 直接管理）**：
```
WorkingDirectory=/home/zxw/Strata/terminal
Environment=LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib
Environment=HY4_KV_HOST=1
ExecStartPre=/bin/bash /home/zxw/scripts/rocm-wait-devices.sh 90
ExecStartPre=-/bin/bash -c 'p=$(ss -ltnp 2>/dev/null | grep ":8080" | grep -oE "pid=[0-9]+" | cut -d= -f2 | head -1); [ -n "$p" ] && kill $p && sleep 4 || true'
ExecStart=/home/zxw/Strata/.venv/bin/python /home/zxw/Strata/terminal/serve/server.py \
  --engine strata --config /home/zxw/Strata/terminal/hy4-256k-067-convcache.json --port 8080
Restart=on-failure
RestartSec=15
StandardOutput=append:/home/zxw/logs/strata/run.log
StandardError=append:/home/zxw/logs/strata/run.log
```

**run config**（`~/Strata/terminal/hy4-256k-067-convcache.json`，真源在推理机）：
```
exe: /home/zxw/.local/share/strata/components/shapes/hy4/engine/0.1.67-linux-x86_64-hip-halo/strata-hy4
args: --native ~/models/strata-hy4/Hy4-Flash-SQ-128.strata
      --engram ~/models/strata-hy4/Hy4-Flash-Engram-Q8.strata
      --max-context 327680 --session-reuse 1
      --vision ~/models/strata-hy4/Hy4-Flash-Vision.strata
      --conversation-cache-dir ~/models/strata-hy4/convcache-prod
      --conversation-cache-gib 240 --conversation-cache-min-tokens 2048
model_name: hy4-flash-sq-128, port: 8080, drafting: dfly2
```

### 6.2 日志
- server.py 侧 `~/logs/strata/run.log`；引擎全量 `~/.local/state/strata/launch/sq-128-256k.engine.log`（config `log` 字段）。
- 两者均纳入 logrotate（`logrotate-strata`：daily/rotate 14/copytruncate），user 级 `logrotate.timer` 每日 00:10。引擎日志的轮转规则是 2026-11-24 补的（此前是盲区）。

### 6.3 模型名严格校验
serve 层传旧 Flash-Next 名直接 `model_not_found`；必须传 **`hy4-flash-sq-128`**。（早期 README 说"Strata 不校验模型名"是老版本历史表述，已被 serve 层实现推翻。）

---

## 7. 客户端接入

端点统一：`https://<your-domain>/v1`（云端隧道→8080）或局域网 `http://<lan-ip>:8080/v1`；鉴权 Bearer token。

| 客户端 | 配置位置 | 状态 |
|---|---|---|
| **opencode**（WSL，v1.19.6） | `~/.config/opencode/opencode.jsonc` | 模型 `local-llm/hy4-flash-sq-128`，`limit.context 327680`；默认仍 27B，需手动切 |
| **WorkBuddy**（Windows 桌面） | `~/.workbuddy/models.json` | 条目 `hy4-flash-sq-128`，声明 tool call/images/reasoning；context 仍 262144（未同步 320k） |
| **AutoClaw2**（Windows 桌面） | `AppData/Roaming/AutoClaw-official/accounts/<id>/runtime/runtime.sqlite` | 自定义连接已改 hy4-flash-sq-128，但 `active=0`，默认走官方 `zai_auto`，需 UI 手动切 |

**选型**：编码用 opencode、通用 agent 用 WorkBuddy，都能接同一端点，非二选一。WorkBuddy 走公网隧道（~1 Mbps），大上下文/图片交互慢，局域网直连快得多。

⚠️ opencode 升级走 npm（`npm install --prefix ~/.oc-npm opencode-ai@<ver>`，二进制 `~/.oc-npm/node_modules/opencode-linux-x64/bin/opencode`），WSL 里 `opencode upgrade` 因 github 不可达不可用。

---

## 8. 权重碎片化（filefrag 实测）

| 文件 | extents | 评价 |
|---|---|---|
| SQ-128.strata (106 GB) | 89 | 健康（curl 单流下载） |
| Engram-Q8.strata (63 GB) | 44 | 健康 |
| Vision.strata (891 MB) | 2 | 完美 |
| 加载读速 | 4.5–5 GB/s | 满速，**无需 defrag** |

对照：Flash-Next 分片3 曾 130 extents、冷读仅 3.4 GB/s（已删）。单流下载 + 不碎片是本仓库的下载策略验证结论。

---

## 9. 回退

- 27B Q8（llama.cpp）权重与 llamacpp 底座完好，`qie_ai llamacpp` 随时切回；编程 4/6、TG 14–15 t/s，但 prefix cache 实测有效（cache_n 递增）。
- hy4flash 0.1.66 引擎已删；Flash-Next 已删。
- 切换机制：qie-ai-agent 只认 unit 名（`strata`/`llamacpp`/`comfyui`），改 `strata.service` 内容不影响它。
