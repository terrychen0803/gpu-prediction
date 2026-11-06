# O4 七 A × 40×3 工程 Collector：RTX3090 / GB10 / RTX4090

> 版本：`o4_seven_A_40x3_engineering_v1_20261106`（新開發協定）。
> 本工具是完整長度的 **engineering collection pilot**，**不是**已驗證完整性／可直接訓練的 Strict Reference；完整 CUDA/Nsight 節點執行結果需另外確認。不要把它與舊 `o4_gpu_iteration_smoke_v2_20261106` 或 `o4_A_inventory_smoke_v1_20261106` 混用。

## 目前各設備已知進度（依 GitHub 紀錄）

- **GB10**：原版 O4 GEMM/BMM/Conv 27/27 + RN-A 9/9，另有 single 3/3；小算子 GEMM/BMM GI 三程序 spread 未通過 5%，RN-A 三程序 GI 通過；**目前沒有 GitHub 可核對的七 A 全 40×3 成功結果**。完整數據見 `docs/O4_GB10_PILOT_STATUS_20261106.md`。
- **RTX3090**：原版 frozen GEMM/BMM/Conv 27/27 + RN-A 9/9 + single 3/3；新七 A **I2 8 warmup / 1 transition / 3 measured / 1 process** 各 GI/P/R 21/21，14 P/R trace replay PASS；**尚未實測本新版本七 A 40×3**。GEMM R 原版跨程序 spread 29.86% FAIL，保留。
- **RTX4090**：依目前此 repo 文件，未看到新的完整 O4 七 A 40×3 收集紀錄；來源設備 P 必須單獨收集並遵守新確認資料隔離規則。

## 版本與輸入不變性

新程式：`o4_pilot/o4_research/run_formal.py`、`formal_worker.py`、`formal_protocol.py`；原本 `run_smoke.py` 與 `worker.py` 維持不變。七 A Adapter 與原版 I2 使用相同 `frozen_A` manifest／byte-identical weights/input/token IDs；**不得**重新產生、縮 batch、改 dtype、替換模型或改 Qwen cache 長度。

- `frozen_A/ASSETS.json` SHA256：`55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4`。
- RTX3090 舊 I2 根目錄原本是 `$HOME/RTchen/o4-data-3090/frozen_A/ASSETS.json`；`$HOME/RTchen/gpu-prediction-o4-i1i2` 需切換到**新分支**，不要切換原版 frozen `gpu-prediction-o4-frozen` checkout。
- GB10 的 frozen_A 路徑需由節點實際核實；**不要**把 RN-A 原版 `o4_pilot/manifests/assets.json` 的 9 檔資產與七 A 的 11 個唯一引用資產混用。
- `--expect-gpu` 可選 `3090`、`4090`、`GB10`；每個程序只允許一個 CUDA-visible device。

## Protocol

所有七 A 的每一個 `(workload,mode,repeat)` 產生**獨立 GPU Python 子程序**。3 模式 `GI/P/R` × 3 repeats = 每個 A **9 Cells**；七 A 共 **63 Cells**，每 Cell 40 measured + 5 transition，3 × 20 計時視窗暖機檢查、最少 50 / 最多 500 warmup（最早第 60 次通過）。

- 全七 A 累計 **2,520 measured iterations**；P/R 共 **42 Nsight Captures**，同 workload/repeat 的 P/R 以**45 個 Transition+Measured**有序 signatures 對齊。
- GI 無 Profiler，CUDA events 包單次 native forward；P 只有 iter/prepare NVTX scope，R 有 ATen leaf reference。Qwen `adapter.prepare` 在每次 event start **之前**；每次 Decode cache 前態固定 128、後態 129；計時外 prefill 不算 GI。
- 每程序保存全部 40 筆、warmup.csv、GPU 環境、checkpoint/input digest、quality.json。每 `(A,mode)` 的三個程序中位數再取中位數；spread `(max-min)/median`，門檻 5%。無論失敗與否，保存所有嘗試，不刪 outlier。
- 所有 P/R 原樣保存 warning。已知 `Not all NVTX events might have been collected.` **暫時不阻擋 Engineering 探索**，但 Strict Reference 與 `training_eligible` **永遠 false**（另行驗證才能有新資料版的訓練資格）。MV/SW/YO 尚有可能未對應 GPU Activity 的 Memset APIs，不能猜為 zero-byte。
- GB10 與 RTX3090 的 Kernel 分解不同合法；**不得**要求跨設備 kernel signature 相同。只能要求**同一設備**同一 workload/repeat P/R 結構對齊。

## RTX3090 安全部署與測試

先保護舊 I1/I2 固定 commit 與任何本地修改：

```bash
conda activate gpu-o4-3090
export NEW="$HOME/RTchen/gpu-prediction-o4-i1i2"
export ASSETS="$HOME/RTchen/o4-data-3090/frozen_A/ASSETS.json"

cd "$NEW"
git status --short
# 只有無本地修改時才切換：
git fetch origin o4/seven-a-40x3-collector-20261106
git switch --detach FETCH_HEAD
git rev-parse HEAD

cd "$NEW/o4_pilot"
python -m unittest discover -s tests -v
(cd vendor/legacy && python -m unittest discover -s tests -v)
python -m o4_research.verify_assets --manifest "$ASSETS" \
  --expected-manifest-sha256 55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4
```

**先 plan-only**（不執行 GPU），會建立新的 manifest/output 目錄：

```bash
cd "$NEW/o4_pilot"
python -m o4_research.run_formal \
  --out "$HOME/RTchen/o4-data-3090/sevenA_40x3_plan_$(date -u +%Y%m%dT%H%M%SZ)" \
  --assets-manifest "$ASSETS" \
  --expected-manifest-sha256 55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4 \
  --workload RN-A \
  --expect-gpu 3090 \
  --plan-only
```

**先 RN-A Canar​​y**：此單 A 按新協定各模式三程序，共 9 Cells、360 measured，與舊 Frozen RN-A raw 完全不同的 run；輸入 hash 與 protocol 仍要校對。

```bash
nvidia-smi
python -m o4_research.run_formal \
  --out "$HOME/RTchen/o4-data-3090/sevenA_40x3_RN_A_$(date -u +%Y%m%dT%H%M%SZ)" \
  --assets-manifest "$ASSETS" \
  --expected-manifest-sha256 55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4 \
  --workload RN-A \
  --expect-gpu 3090
```

各 run 的 `inventory.json`、`SUMMARY.json`、`aggregate.json`、`pair_summary.json` 分別記錄完整嘗試、GPU 收集狀態、三程序統計及 45-window P/R 對齊；`NVTX_warnings` 不會刪除。接著執行 MV-A、SW-A、UN-A，再 YO-A、QP-A、QD-A（各 run 27 Cells）；每次 `--out` 為新路徑。

**容量風險：** Qwen R 每次約 2,400–2,500 個 ATen × 45 窗口 × 3 processes，NSYS/SQLite/Reference JSON 可能很大；執行前確認 `df -h`、`df -i`，不要把大檔直接推到 Git。預設每 Cell 3600 秒程式 timeout，可用 `--cell-timeout-seconds 0` 取消 cap，任何失敗保留。

## GB10 對齊

同一 Branch、同一 `frozen_A/ASSETS.json` SHA；GB10 本地 CUDA/Nsight 必須各自通過 `--plan-only`、O4 tests 與小量 RN-A canary。把 `--expect-gpu GB10` 與 `--out` 路徑替換為 GB10 新工作目錄；**目前沒有證據可以直接標為 GB10 七 A 40×3 已收集**。

## 關鍵驗收

1. 新版 40×3 需各模式 `measured_iterations=40`、`warmup.pass=true`、3 個獨立工作程序及相同 assets SHA。
2. 每個 P/R trace 可在 Nsight schema 3.28/3.31 結構通過後重新解析。新 Audit 的 scope 集合是 **5 transition + 40 measured**，舊 3-measured I2 仍是 **1+3**；不得因不同協定默默更改舊資料。
3. 3 個 P/R pairs 的 45/45 exact、三模式 process spread gate 分別報告。Pair mismatch、missing NVTX iteration、schema unknown、GPU→API correlation failure 是 review required；已知 NVTX warning 僅 provisional engineering acceptance。
4. `strict_reference_verified=false`、`training_eligible=false`，即使所有 63 Cells completed 也只是**可分析的正式長度工程資料**，不是完成 I3/I4 的模型訓練 dataset。
