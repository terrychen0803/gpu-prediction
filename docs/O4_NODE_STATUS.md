# O4 節點狀態：2026-11-06

本文件區分歷史GPU實測、統一版工程實作、尚未執行與Reference資格。

| 節點 | 已記錄 GPU 實測 | 統一版 v2 GPU canary | Reference |
|---|---|---|---|
| GB10 | Native software48/48；hardware0/80；七A短版21cells；舊single/three/RN-A pilot | 未執行 | strict0；NVTX/clock/overlap/部分API待解 |
| RTX3090 | GitHub節點報告：七A短版21cells；14trace replay；舊single/three/RN-A39cells；GEMM R spread29.86%失敗保留 | 未執行 | strict0；MV/SW/YO unmatched APIs待解 |
| RTX4090 | 此批未提供O4實測交付 | 未執行 | 未驗證 |

統一版工程實作：新工程驗收、全資產/來源驗證、環境紀錄、共用CLI、離線測試。沒有訓練辨識器/N0/ITER，沒有新確認集。3090數據來自節點交付摘要，未因Git整合變成本機獨立GPU驗證。

歷史實測報告見 O4_I1_REFERENCE_I2_ADAPTER_STATUS_20261106.md、O4_RTX3090_I1_I2_20261106.md、O4_FROZEN_RELEASE_DOC_RECONCILE_20261106.md。舊文件中「尚未收集」段落是當時狀態，最新交付以各報告末段與本表為準。統一執行入口見 [Quickstart](O4_QUICKSTART_ZH_TW.md)。
