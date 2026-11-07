# O4 節點狀態：2026-11-06

**2026-11-07 S4最新進度：** GB10 scale0六圖D/I/S共40工作已完成；見[結果](o4_i3_s4_scale0_20261107/README.md)及[RTX操作手冊](O4_I3_S4_NODE_RUNBOOK_20261107.md)。A/BC source與shared base分開，各GPU須派生本地D中間輸入。Reference仍未verified，scale1/2與模型訓練未執行；以下舊版狀態保留歷史背景。

本文件區分歷史GPU實測、統一版工程實作、尚未執行與Reference資格。


## 2026-11-07：RTX4090 Unified v2 七組 A 全部完成（7/7）

操作者於 RTX4090 節點 `iccl_resilient@iccl-WHITLEY` 提供 QP-A、SW-A 原始 `SUMMARY.json`，連同本日先前的 RN-A、MV-A、UN-A、YO-A、QD-A 摘要，目前**RTX4090 七組 A 的 40×3 工程收集全部完成**。Collector 既有固定版本為 `c988163a0acf186874ea64e4957284b87f999d43`、Protocol `o4_seven_A_40x3_engineering_v2_20261106`，未因後續文件 Commit 改變。來源為操作者日誌，並非獨立讀取節點原始 SQLite。

| Workload | Session | Cells | Valid iterations | Exact P/R | Structural traces | Stable | Quality | Unmatched API / P或R capture | `du -sh` |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| RN-A | `RN_A_20261106T185740Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | 630 | 133M |
| MV-A | `MV_A_20261106T190225Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | 135 | 119M |
| UN-A | `UN_A_20261106T190759Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | 0 | 43M |
| YO-A | `YO_A_20261106T191227Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | 1035 | 307M |
| QD-A | `QD_A_20261106T191623Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | 0 | 1.2G |
| QP-A | `QP_A_20261106T192142Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | 0 | 847M |
| SW-A | `SW_A_20261106T192527Z` | 9/9 | 360 | 3/3 | 6/6 | PASS | accepted | **1845** | 206M |
| **總計** | 7 個 Session | **63/63** | **2,520** | **21/21** | **42/42** | **7/7** | **7/7 accepted** | — | 約 2.9G |

QP-A `SUMMARY.json` 記錄 `utc_end=2026-11-06T19:24:31.358064+00:00`；SW-A `utc_end=2026-11-06T19:27:02.606514+00:00`，兩份皆 `status=completed_engineering`、`failed_cells=0`、`all_three_process_spreads_pass=true`、`stability_failed_groups=[]`、`unmatched_work_API_captures` 分別為空與六份各 1845 筆。兩份的 P/R 各六次 Capture 均保留 `Not all NVTX events might have been collected.`。

**不可升級 Reference**：全部 Summary 仍為 `strict_reference_verified=false`、`training_eligible=false`、`models_trained=0`，已知 NVTX 警告仍在，不能用 42/42 structural passes 取代完整 CUDA submission／op reference 證明。

Unmatched 調查更新：RN-A 各 630、MV-A 的 P/R repeat 1 各 135、YO-A 六份各 1035 在既有調查中已確認為 `cudaMemsetAsync_v3020`；**SW-A 六份各 1845 尚未分類**。這些只是 unmatched API 名稱與數量，不能推定 `count=0` 或補填 GPU Activity，需獨立 CUPTI Callback Byte Count 驗證。

當時 GPU 節點根檔案系統尚有 **158G** 可用；七組 Session 總原始量約 2.9G。建議接續封包原始 Trace、frozen assets、執行版本與 SHA256 清單，傳 GB10 後重驗完整性，並進行 source→target 分析。所有 session 原始資料應保留；本次只更新文件，不建立新分支。

---

## 2026-11-07：RTX4090 QD-A 40×3 完成與 YO-A Memset API 分類

操作者在 `iccl_resilient@iccl-WHITLEY` 提供 QD-A（Qwen Decode）正式工程收集 `SUMMARY.json` 與容量紀錄：

| 項目 | RTX4090 QD-A |
|---|---|
| Session | `/home/iccl_resilient/RTchen/o4-data-4090/unified_v2/QD_A_20261106T191623Z` |
| Protocol | `o4_seven_A_40x3_engineering_v2_20261106` |
| 結束時間 UTC | `2026-11-06T19:19:45.920235+00:00` |
| Status / quality | `completed_engineering` / `engineering_accepted` |
| Cells | 9/9 completed；0 failed |
| Valid measured iterations | 360（GI / P / R 各 120） |
| Exact P/R pairs | 3/3 |
| Trace structural passes | 6/6 |
| Three-process stability | `all_three_process_spreads_pass=true`，無失敗組 |
| P/R unmatched Work APIs | 摘要清單為空 |
| Warning | 六份 P/R 皆有已知 `Not all NVTX events might have been collected.` |
| Reference 資格 | `strict_reference_verified=false`，`training_eligible=false` |
| 原始資料容量 | `du -sh` 為 1.2G |
| 磁碟剩餘 | 當時約 159G |

YO-A（`YO_A_20261106T191227Z`）進一步取得六份 P/R `audit/unmatched_work_APIs.csv` 的名稱分布：**每份各 1,035 筆全部是 `cudaMemsetAsync_v3020`**。與 RN-A（各 630 筆）及 MV-A（各 135 筆）類型一致。以 5 次 Transition + 40 次 Measured 計算，YO-A 平均每次 23 筆，RN-A 平均 14 筆，MV-A 平均 3 筆。這是總數平均，**不代表已查驗每次迭代均相同，更不代表 Memset count=0 或 GPU Activity 未漏記**。沒有收到 4090 相應的 CUPTI Callback byte-count 直接證據，故仍保留 unmatched，沒有修補 trace 或提升 Reference 資格。

目前 RTX4090 **5/7 組**（RN-A、MV-A、UN-A、YO-A、QD-A）、**45/45 completed Cells、1,800 valid measured iterations、15/15 exact P/R pairs、30/30 P/R structural passes**；已回報五組皆工程驗收通過。待收 QP-A、SW-A。本段僅根據操作者 SUMMARY、CSV 名稱計數與 `du` 日誌，不代表對原始 SQLite 做了額外獨立稽核。Collector 仍可使用本次實際收集的固定 commit；本次僅更新現有 GitHub 分支文件，未建立新分支。

---

## 2026-11-07：RTX4090 YO-A 40×3 正式工程收集

操作者於 `iccl_resilient@iccl-WHITLEY` 提供 YO-A（YOLO26n）Unified v2 `40×3` 全量 SUMMARY 與容量：

- 原始 Session：`/home/iccl_resilient/RTchen/o4-data-4090/unified_v2/YO_A_20261106T191227Z`（`utc_end=2026-11-06T19:14:08.303457+00:00`）。
- `protocol=o4_seven_A_40x3_engineering_v2_20261106`；`status=completed_engineering`；`quality_status=engineering_accepted`。
- `completed_cells=9/9`；`failed_cells=0`；`measured_valid_iterations=360`；`P_R_pairs_exact=3/3`；`trace_structural_passes=6/6`；`all_three_process_spreads_pass=true`。
- 六份 P/R Capture 各有已知 `Not all NVTX events might have been collected.` 警告；六份各 **1,035 unmatched Work API**，尚未取得名稱分類（不能假設全是 Memset），平均 1,035/45 = 23 未匹配 Work API／Iteration，非逐 Iteration 實際分布保證。
- `strict_reference_verified=false`、`training_eligible=false`、`models_trained=0`。原始 Session `du -sh` 為 307M。當時 `df -h` 顯示剩餘約 160G。
- RTX4090 最新正式工程收集 **4/7**（RN-A、MV-A、UN-A、YO-A），**36/36 completed Cells、1,440 measured valid iterations、12/12 exact P/R pairs、24/24 P/R structural passes**。下一組建議 QD-A；其後 QP-A、SW-A。

此記錄僅根據操作者貼出的 SUMMARY／容量資訊，尚未對 YO-A raw SQLite 獨立重稽核；不因工程收集通過而提升 Reference 等級。此次只更新文件，收集程式不變，也未建立新分支。

---

## 2026-11-07：RTX4090 UN-A 40×3 完成與 MV-A Memset 分類

操作者於 `iccl_resilient@iccl-WHITLEY` 完成 UN-A（U-Net）正式 `40×3` 工程收集，原始目錄 `/home/iccl_resilient/RTchen/o4-data-4090/unified_v2/UN_A_20261106T190759Z`（輸出 SUMMARY `utc_end=2026-11-06T19:09:22.426010+00:00`）。

| 指標 | UN-A 4090 |
|---|---|
| Protocol | `o4_seven_A_40x3_engineering_v2_20261106` |
| Status | `completed_engineering` |
| Completed / failed | 9/9 / 0 |
| Measured valid | 360（GI / P / R 各 120） |
| Exact P/R | 3/3 |
| Trace structural passes | 6/6 |
| Stability | `all_three_process_spreads_pass=true`；無失敗 groups |
| Unmatched Work API | **0 captures** |
| Known NVTX warning | P/R 六份皆保留 |
| Quality | `engineering_accepted` |
| Training | `strict_reference_verified=false`、`training_eligible=false` |
| Size | `du -sh` 43M |

MV-A `MV_A_20261106T190225Z` 的 P/R repeat 1 CSV 進一步確認各 **135 筆 unmatched Work API 全部屬 `cudaMemsetAsync_v3020`**。RN-A 六份 P/R 各 630 筆也已確認為 `cudaMemsetAsync_v3020`。以 5 transition + 40 measured 計算，兩者分別平均每 iteration 14 與 3 筆 unmatched。此結論只是 **API 類別**，並不能證明參數 `count=0`，也不能直接判定 GPU Activity 漏記；需要額外 Callback Observer／Bytes 參數佐證。MV-A 目前只取 P/R repeat 1 的具名 CSV 分布，其他 repeat 的 unmatched *數量* 一致，但名稱分布尚未逐份查核。

至此 RTX4090 正式工程收集 **3/7 workloads：RN-A、MV-A、UN-A；27/27 cells，1,080 measured valid，9/9 P/R exact pairs，18/18 trace structural passes**。其餘 YO-A、QD-A、QP-A、SW-A 尚未有成功交付。來源為操作者貼出的 SUMMARY、CSV 摘要及 `du`，不是從節點對 raw SQLite 再獨立 re-audit。歷史 tag 未改，新收集不因這項文件更新改變 Collector commit。

---
## 2026-11-07：RTX4090 MV-A 40×3 與 RN-A unmatched Memset 診斷

RTX4090 新增 MV-A（MobileNetV3）`MV_A_20261106T190225Z` 正式 40×3 工程實測（操作者在 2026-11-07 提供 `SUMMARY.json`、Collector 輸出與 `du`）：

| 指標 | RTX4090 MV-A |
|---|---|
| Status / quality | `completed_engineering` / `engineering_accepted` |
| Cells | 9/9 complete；0 failed |
| Measured | 360（GI/P/R 各 120） |
| P/R exact pairs | 3/3 |
| Trace structural passes | 6/6 |
| Spread | `all_three_process_spreads_pass=true` |
| Nsight | 2026.5.2.323，SQLite Schema 3.29.0 |
| P/R work | 六份 capture 各 13,455 個 GPU records、各 135 個 unmatched Work API |
| Diagnostics | 六份 P/R 皆有已知 `Not all NVTX events might have been collected.`；工程接受 `warning_accepted`，0 blockers |
| 資料大小 | 119 MB |
| Reference | `strict_reference_verified=false`、`training_eligible=false` |

此時 RTX4090 七組 A 已有 **RN-A、MV-A 共 2/7 models、18/18 Cells、720 measured iterations、6/6 exact P/R pairs、12/12 structural passes**。其他五組尚未有正式 40×3 成功交付。

原 RN-A `RN_A_20261106T185740Z` 六份 P/R 各 630 筆 unmatched API，操作者新提供 `audit/unmatched_work_APIs.csv` 的名稱分布顯示**全部為 `cudaMemsetAsync_v3020`，且 API `returnValue=0`**。630/45 = 14 unmatched API per transition/measured iteration。MV-A 的 135/45 = 3 unmatched API per iteration，**但尚未交付 MV-A API 名稱分類，不能推斷皆為 Memset**。RN-A API 成功返回並不代表 Memset `count=0`；對這些 API 尚無直接 Bytes 的 Callback 證據，保留 unmatched，不補事件、不升級 Reference。與 GB10 RN-A op140 零 Bytes 的歷史證據也不得混同不同裝置的實際行為。

資料來源為操作者日誌，GitHub 未獨立存放 raw Trace；原始診斷與成功資料應留在 4090 節點。建議接續 UN-A 收集並優先做 RN-A Memset count observer 診斷，不阻塞其他工程收集。

---

## 2026-11-07：RTX4090 RN-A 40×3 首組正式工程收集

操作者於 `iccl_resilient@iccl-WHITLEY` 使用既有分支 `o4/unified-engineering-20261106`（節點 commit `c988163a0acf186874ea64e4957284b87f999d43`）、PyTorch 2.8.0+cu128、Nsight Systems 2026.5.2.323 及相同 frozen_A，完成 RN-A 正式 40×3：

| 項目 | 結果 |
|---|---|
| 協定 | `o4_seven_A_40x3_engineering_v2_20261106` |
| 完成 | 9/9 Cells；Failed 0 |
| 有效時間量測 | 360（GI/P/R 各 120） |
| P/R 精確配對 | 3/3 |
| 結構通過 | P/R 6/6 |
| 穩定度 | `all_three_process_spreads_pass=true` |
| 工程品質 | `completed_engineering`、`engineering_accepted` |
| 原始大小 | 133 MB（`RN_A_20261106T185740Z`） |
| 診斷警告 | 六份 P/R 均保留已知 `Not all NVTX events might have been collected.` |
| Unmatched API | 六份 P/R **各 630**，待分類，不能視為已證明漏記 Kernel 或零 Bytes Memset |
| Reference / 訓練 | `strict_reference_verified=false`、`training_eligible=false`、`models_trained=0` |

這份資料**已完成工程驗收，不等於正式逐算子 Reference 驗證完成**。RTX4090 接下來優先分析 unmatched Work API 的 API name/correlation/bytes 與事件對應，再決定是否擴大 RN-A 類似的來源追蹤。其他六組正式 40×3 目前尚無新實測交付。以上來自操作者貼出的 `SUMMARY.json` 與 `du`，尚未從 RTX4090 節點獨立讀取 raw data。

---

## 2026-11-07：RTX4090 RN-A Smoke 與 Nsight 3.29.0 結構相容性

RTX4090 目前為 `iccl_resilient@iccl-WHITLEY`、Ubuntu 22.04、RTX4090、NVIDIA Driver 580.178.04。使用 Nsight Systems 2026.5.2.323（以使用者目錄解壓套件執行）、PyTorch 2.8.0+cu128、TorchVision 0.23.0+cu128、Transformers 4.57.1、Ultralytics 8.4.55。操作者已提供 82/82 O4 tests 與 54/54 Legacy tests 通過、frozen assets Verify 和 Preflight 通過的日誌；Manifest SHA256 `55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4`。

RTX4090 RN-A inventory Smoke 首次結果為 GI completed，P/R 因 `Unknown Nsight export schema 3.29.0` 被 Parser 拒絕；**不是已證明 P/R Trace 有漏記**。使用者對失敗的 P、R SQLite 各自唯讀檢查：`PRAGMA quick_check=ok`，`EXPORT_SCHEMA_VERSION=3.29.0`，必要/可選/feature 欄位沒有缺漏，均記錄 1,380 個 Kernel、12 個 Memset；P NVTX 8 筆、R NVTX 708 筆；都有已知 `Not all NVTX events might have been collected.` 警告，並確認 `CUDA legacy/software instrumented trace was collected`。上述為現場欄位檢查，**不等於 v5 audit／P-R exact pairing／提交工作完整性通過**。

已在既有分支 `o4/unified-engineering-20261106` 將 `3.29.0` 新增至明確 Schema allowlist，擴增 offline tests，更新 `unified_release.json` SHA256 source index（`guard-retry-schema329-patch2`）；未修改原固定 tag、未增分支、未放寬 warning／CUDA drop／Reference gate。**下一步需由操作者在 RTX4090 更新程式、執行測試與對原 Smoke P/R SQLite 進行唯讀 re-audit，確認工程 acceptance 再新建正式 collection output。** 目前 RTX4090 40×3 尚未開始，strict reference 仍未驗證。


## 2026-11-07：RTX3090 Unified v2 七組 A 正式長度工程量測更新

操作者已交付七組 A `40×3` 摘要：**7/7 工作負載、63/63 Cells、2,520 valid measured iterations、21/21 P/R exact pairs、42/42 P/R trace structural passes**。六組 `engineering_accepted`；UN-A 全部9 Cells完成但 P repeat spread **5.7697%**，因此 `quality_status=review_required`。

| Node | v2 七組 A 工程實測進度 | 嚴格 Reference |
|---|---|---|
| RTX3090 | RN-A、UN-A、MV-A、YO-A、QD-A、QP-A、SW-A 都已完成 9/9；詳見獨立收集報告 | `strict_reference_verified=false`；未證明 training eligible |
| GB10 | 此批無新增 v2 40×3 實測證據 | 仍依原報告 |
| RTX4090 | 此批無新增 v2 40×3 實測證據 | 仍依原報告 |

所有 P/R capture 保留既知 NVTX 警告；MV-A、YO-A、SW-A 的 unmatched Work API 仍需分類。RN-A 已在最新 GPU Guard cooldown 修正版收集 9/9；原失敗 session 不刪除。QD-A 有一次 SSH 中斷的 0/9 session、後來以 tmux 成功重收 9/9。GB10 大型 raw 傳輸與獨立 re-audit 尚待操作者執行。來源為操作者提供的 `SUMMARY.json`，**不是**直接讀取 RTX3090 節點或 GB10 原始 trace 的獨立驗證。

詳細表格、品質限制和分析交接：**[RTX3090 七組 A 收集結果與 GB10 交接](O4_RTX3090_SEVEN_A_COLLECTION_20261107.md)**。

---


| 節點 | 已記錄 GPU 實測 | 統一版 v2 GPU canary | Reference |
|---|---|---|---|
| GB10 | Native software48/48；hardware0/80；七A短版21cells；舊single/three/RN-A pilot | 未執行 | strict0；NVTX/clock/overlap/部分API待解 |
| RTX3090 | GitHub節點報告：七A短版21cells；14trace replay；舊single/three/RN-A39cells；GEMM R spread29.86%失敗保留 | 未執行 | strict0；MV/SW/YO unmatched APIs待解 |
| RTX4090 | 此批未提供O4實測交付 | 未執行 | 未驗證 |

統一版工程實作：新工程驗收、全資產/來源驗證、環境紀錄、共用CLI、離線測試。沒有訓練辨識器/N0/ITER，沒有新確認集。3090數據來自節點交付摘要，未因Git整合變成本機獨立GPU驗證。

歷史實測報告見 O4_I1_REFERENCE_I2_ADAPTER_STATUS_20261106.md、O4_RTX3090_I1_I2_20261106.md、O4_FROZEN_RELEASE_DOC_RECONCILE_20261106.md。舊文件中「尚未收集」段落是當時狀態，最新交付以各報告末段與本表為準。統一執行入口見 [Quickstart](O4_QUICKSTART_ZH_TW.md)。
