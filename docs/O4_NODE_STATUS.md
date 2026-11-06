# O4 節點狀態：2026-11-06

本文件區分歷史GPU實測、統一版工程實作、尚未執行與Reference資格。


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
