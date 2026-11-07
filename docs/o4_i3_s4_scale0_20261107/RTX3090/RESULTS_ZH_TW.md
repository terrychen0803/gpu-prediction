# RTX3090｜O4 I3 S4 scale0 六張 Context Graph：工程收集結果

> **資料來源與驗證層級**：本頁數據根據 RTX3090 節點操作者提供的終端輸出，以及該節點執行的 `final_s4_scale0_3090_v1/SUMMARY.json` 稽核摘要整理。GitHub 更新時**未直接讀取 RTX3090 原始檔案**，因此這是一份具來源聲明的結果登錄，而非另一次獨立 raw replay。原始完整資料由節點封存，**不提交大型 trace 至 Git**。

## 固定環境

- GPU：NVIDIA GeForce RTX 3090
- PyTorch：`2.13.0+cu130`；CUDA：`13.0`；cuDNN：`92000`
- Source release：`o4-i3-s4-scale0-dis-dev-v1-20261107`
- Frozen source commit：`3db5871d43abd87fefbdadc21371beee59ac6b88`
- A local native-D assets SHA256：`6048d5568a99caf6fbfe55de686c82e4df60a593e03ac5d57b682b9c98870236`
- BC local native-D assets SHA256：`3326f71e335ddc40a5b3bdc69e7d9dea81d29b6fc2111ccd0c81fd2e78952091`
- 實驗範圍：`context_scale_0`；**D/I/S 六張受控圖**，非完整六模型 Forward、非 scale1/2。

## 正式 40×3 收集

| 圖 | 工作項目 | Formal cells | Measured iterations | P/R pairs exact | Stable / Review |
|---|---:|---:|---:|---:|---:|
| Conv→BN→ReLU | 5 | 45/45 | 1,800 | 15/15 | 7 / 8 |
| Residual | 9 | 81/81 | 3,240 | 27/27 | 11 / 16 |
| Depthwise→Pointwise | 6 | 54/54 | 2,160 | 18/18 | 7 / 11 |
| SE Gate | 8 | 72/72 | 2,880 | 24/24 | 8 / 16 |
| Resize→Concat→Conv | 6 | 54/54 | 2,160 | 18/18 | 7 / 11 |
| Eager Attention | 6 | 54/54 | 2,160 | 18/18 | 9 / 9 |
| **總計** | **40** | **360/360** | **14,400** | **120/120** | **49 / 71** |

Smoke：120/120 cells、360 measured、40/40 P/R pairs exact。各批次 Correctness、Warmup 與 Trace Engineering Gate 符合 development 收集條件；共 **320 個 P/R trace 的 Engineering Acceptance 為 `warning_accepted`**（Smoke 80 + Formal 240）。各批次沒有 Quality Issues。

## 最終唯讀稽核

使用 RTX3090 本地 `analysis/final_s4_scale0_3090_v1/SUMMARY.json` 所呈現的結果：

- `Status: PASS`，`Work Items: 40`
- `Smoke Cells: 120`；`Formal Cells: 360`；`Formal Measured: 14400`
- `Formal P/R Pairs: 120`
- `Mode Stability: {'review_required': 71, 'stable': 49}`
- `Engineering Acceptance: {'warning_accepted': 320}`
- `R Parameter Rows: 11676`；`R Mismatches: 0`；`Batch Issues: 0`

此稽核涵蓋每個 R 位置的完整 ATen/參數/輸入輸出 Metadata 合約，並非單靠 Kernel 名稱相符。它**尚非**完整 Reference 證明，不能替代 raw replay、submitted-work correlation、clock／overlap、ownership 或 P_CTX 衍生分析。

## 使用限制及研究用途

- `strict_reference_verified=false`、`formal_training_eligible=false`，不可自行改標成 strict PASS。
- 71 個 `review_required` 是跨獨立 Process 的量測波動標記，保留完整 Raw Data 與不確定性；不應直接當作低雜訊監督標籤。
- P Event Duration ≠ per-position P_CTX，後續必須按 position／repeat 計算 P_CTX，並核對 Owner/Correlation。
- 六張圖共用 `context_scale_0`，並非 40 個互相獨立的樣本群；沒有 OOF 或模型 Test 預測成績。
- GB10 對照應使用相同 Shared Base 外部輸入，但**不得**把 GB10 產生的中間 Tensor 冒充 RTX3090 的 Native D Intermediate Inputs。
- D/I/S 耗時差異是**描述性對照**，不能全歸因於純資料依賴（allocator、cache、lifetime、backend 均可能影響）。
- 請依 `COLLECTION_SUMMARY.json` 保存機器可讀狀態。

## GB10 封存交付

使用 `scripts/package_and_scp_rtx3090_i3.sh` 在 **RTX3090** 節點打包並以 SCP 傳送至指定 GB10 目的地。此腳本可選擇同時納入 Core36、S3 與 S4；每項另封裝：

1. `*.analysis.tar.gz`：排除 `*.nsys-rep` 和 `*.sqlite`，保留解析後 Audit、參數、Quality、Inventory、Source Snapshots、Local Assets 與彙整。
2. `*.full.tar.gz`：保留原始 Raw Nsight、SQLite 與上述所有資料。
3. `SHA256SUMS` 與 `MANIFEST.json`：GB10 端必須驗證後才將 staging 目錄正式命名。

使用者提供的 RTX3090 節點資料根目錄為 `~/RTchen/i3-data-3090`。**實際封存大小與 SHA256 必須在節點產生後才能填入，不可根據文字結果推測。** 這個 GitHub 報告不代表已完成 SCP。

相關來源：[S4 Node Runbook](../../O4_I3_S4_NODE_RUNBOOK_20261107.md)、[六張圖原始結果](../README.md)。
