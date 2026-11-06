# GB10 O4 Native Controls 與 v4 品質 Gate 實測（2026-11-06）

> **狀態：GB10 本地實際 GPU 實驗的摘要。** 來源為使用者提供的本地 `REPORT_ZH_TW1.md`。本 GitHub 目前未附完整 `.nsys-rep`、SQLite、`serial_mre.cu`、v4 audit 程式與全部原始證據；因此下表是由本地報告轉述，尚非 GitHub 獨立重算，也不是新 O4 量測協定或訓練完成證明。

關聯：[O4 I0～I7 計畫](O4_GPU_ITERATION_PLAN_ZH_TW.md) · [品質 v3 假說](O4_GB10_TRACE_QUALITY_UPDATE_20261106.md) · [三節點交接](O4_NODE_ALIGNMENT_20261106.md)。

## 1. 最小 Native Control 實測及重要結論

- 原生 float32、單 stream、**無 PDL attribute**：每 iteration 兩個真正依賴的 kernel、64B memset 與 0B memset；8 warmup＋8 measured／capture、output 256 個元素全為 3.0。這不是 O4 GI 或 ResNet FP16 的替代數值。
- **Software tracing** `cuda-sw`：A1 P、A2 R、B1 P 三份 capture 均記錄 16/16 kernel 與 8/8 64B memset；合計 48/48 kernels、24/24 positive memset。24 iterations 的 0B API 沒有 GPU activity，符合明知 0B 的控制程式。
- **Hardware tracing** `cuda`：A3 P、A4 R、B3 P、C3 P、D3 P 五份 capture 均只收 8/8 64B memset、**0/16 kernel**；合計 **0/80 已知 kernel**。每份 Nsight diagnostic 明報 16 incomplete CUPTI events dropped（停止前沒有 GPU timestamp）。真正的 kernel API 與正確計算輸出都存在；這是此 **GB10／Nsight 2026.4.1.191／原生 harness 組合**的可重現收集失敗，不能推到所有 GPU／所有 HES 版本。
- C3 顯式 CLI flush 請求後仍失敗，但 SQLite 內 metadata 顯示 `CUDA_FLUSH_ON_CUDA_PROFILER_STOP=false`，不能說有效 flush 已被測試且無效；D3 在計時外 200ms host drain 仍失敗，僅排除「增加這個等待就能恢復」。
- `cuda-sw` 中 17 kernel start 早於對應 CPU API entry，最多 1.488 µs；end-before-entry 為 0 **不表示 clock gate 已通過**。所有八份 native captures 仍有 NVTX／CUDA 完整性警告。Hardware P/R 即使 8/8 ordered signature 一致，兩者依然共同缺 kernel。
- v4 唯讀重新解析原 RN-A 六份 raw：270 iterations 的 event-index 分段結構仍相容，union-based closure 全為 0；原 123 對同 default stream overlap、1,579 end-before-entry、270 筆未知 `cudaMemsetAsync` 與 NVTX 警告 **全部仍在**。Native 已知 0B control 不能證明 RN-A op140 是 0B。
- 本地 v4 診斷：19 新測試＋22 O4 凍結測試＋54 legacy 測試，共 95 項（94 pass／1 optional skip）；本 GitHub 尚未取得 v4 原始碼／產物，**不可稱本遠端已重跑**。

## 2. Reference 與資料使用資格

| Gate | Native SW | Native HW | 歷史 RN-A v4 |
|---|---|---|---|
| `recorded_event_integrity` | 已記錄事件唯一關聯 | 剩餘 memset 唯一關聯，但缺 kernel | 已記錄事件結構一致 |
| `submitted_work_coverage` | 已知正工作與 0B no-op 可核對 | **failed**：16 kernels／capture 缺 | **unverified**：op140 memset 真實 count 不明 |
| `reference_semantic_alignment` | warning-accepted | 已知工作缺失／warning | warning-accepted |
| `timestamp_consistency` | **unverified**：17 start-before-entry | **unverified**：kernel 缺失 | **unverified**：clock／PDL 待驗 |
| `event_index_partition_supported` | 已知 native 分段可核對 | 缺已知工作，不接受完整分段 | recorded 結構相容 |
| `nonoverlap_sum_supported` | native recorded 無 overlap | 缺 kernels，不具代表性 | 五份 trace 有 overlap |
| `overlap_aware_closure_supported` | 已記錄區間可閉合 | 只可閉合剩餘記錄，不證明完整 | 270/270 recorded-interval 閉合 |
| `strict_reference_verified` | **unverified** | **unverified** | **unverified** |

**`strict verified = 0` 是合格 Reference 數量／資格仍為零，不是 GI accuracy = 0、程式未執行或所有資料都不能做 Pilot。** 任何未能證明 submitted work 覆蓋及 NVTX/clock 語意的資料，保留 warning-accepted/exploratory；不重新標示歷史 verified，也不向 MLP 提供錯誤標籤。

## 3. 實驗路徑與下一步

1. GB10 **暫停將 hardware `cuda` trace 作為正式資料**；保持明確 `cuda-sw` 用於工程 pilot，但每份 trace 仍必須通過實際的個別品質 gate。不要在 production collector 直接加任意 sleep、固定 clock offset 或無效 flush。
2. I1 GB10 待解：RN-A op140 真正的 memset byte count、NVTX warning 事件漏記、CPU/GPU timestamp／callback 語意、PDL actual launch attributes；只對每個具體假說執行新的最小實驗。已收集 RN-A raw 只讀，不覆寫。
3. RTX4090／RTX3090 先使用本地 portable MRE／runbook 驗證實際 backend／submitted-work coverage，再執行 single smoke；兩個 CLI 字串未必真的代表兩個不同 backend。**Portable MRE 原始碼仍在 GB10，本 GitHub 尚未有可執行路徑**。
4. I2 的六任務／七群 A 配置 Adapter 與 Inventory 可以作 warning-accepted 工程開發；正式 Operator/N0 訓練及 OOF ITER 要等可信 Reference、足夠 taxonomy 與訓練群數。六任務仍為最終研究目標。
5. 如果將 local v4 code 推至 GitHub，使用新 branch／可重現 hash，另作 O4 frozen collector 及 legacy 回歸；量測協定實際變更必須新 protocol ID。

## 4. 資料保護

本次沒有修改原 O1/O2/O3/O4 collector、舊模型或 old quality；Native v1／flush／drain 為獨立診斷版本。GB10 本地 `captures/<cell>`、`historical_overlay_v4/`、`reports/final_v4_3/`、`FINAL_VALIDATION.json` 等不在本次 GitHub 內，保存原始資料與 hash，日後補交可重跑診斷後再提升證據等級。