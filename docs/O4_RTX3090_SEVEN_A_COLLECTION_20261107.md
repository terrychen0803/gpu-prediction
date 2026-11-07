# RTX 3090 — Unified v2 七組 A 全量收集狀態（2026-11-07）

> **資料來源：** RTX3090 操作者在 ChatGPT 提供的各個 `SUMMARY.json` 與 Collector 輸出摘要，**尚非從 GB10 獨立讀取 raw／重新執行 audit 的驗證結果**。本文件只記錄實測交付與待核對項目，不代表 strict reference 已通過，也不代表模型已訓練。

## 1. 執行範圍與版本

- GPU：RTX 3090，驅動 580.178.04（2026-11-07 節點 `nvidia-smi` 輸出），CUDA Driver API 顯示 13.0。
- 統一入口：`o4_pilot/scripts/o4_node.sh collect --gpu 3090 --workload <ID>`。
- Protocol：`o4_seven_A_40x3_engineering_v2_20261106`。
- 每工作負載 `GI/P/R × 3` 獨立程序，每 Cell 40 measured iterations；每個 model 共 9 Cells／360 measured。
- 一共 7 個 A 工作負載、63/63 Cells、2,520 measured（各模式 840）、21/21 exact P/R pairs、42/42 completed P/R trace structural passes。
- 原 frozen assets manifest SHA-256：`55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4`。
- RN-A 及其後的部分資料使用 2026-11-07 在既有 `o4/unified-engineering-20261106` 分支直接修改的 GPU Guard cooldown 修正；正式分析時應以各個 run 中的 `environment_snapshot.json`、`inventory.json`、`source_release` 與 `code_hashes` 確認每筆實際 commit／版本。原固定 tag 不應回填成修正版。
- 本次沒有變更七 A workload 的模型／輸入／計時範圍定義。

## 2. 每工作負載

| Workload | 功能 | Completed | Valid measured | P/R exact pairs | Trace structure | 工程品質 |
|---|---|---:|---:|---:|---:|---|
| RN-A | ResNet-50 | 9/9 | 360 | 3/3 | 6/6 | `engineering_accepted`；GI/P/R process spread 1.05%/0.96%/1.16% |
| UN-A | U-Net | 9/9 | 360 | 3/3 | 6/6 | `review_required`；P repeat spread 5.7697%，略高於 5% |
| MV-A | MobileNetV3 | 9/9 | 360 | 3/3 | 6/6 | `engineering_accepted`；六份 P/R capture 各 360 unmatched Work API |
| YO-A | YOLO26n | 9/9 | 360 | 3/3 | 6/6 | `engineering_accepted`；六份 P/R capture 各 720 unmatched Work API |
| QD-A | Qwen3-0.6B Decode | 9/9 | 360 | 3/3 | 6/6 | `engineering_accepted`；unmatched Work API 列表空 |
| QP-A | Qwen3-0.6B Prefill | 9/9 | 360 | 3/3 | 6/6 | `engineering_accepted`；unmatched Work API 列表空 |
| SW-A | Swin-T | 9/9 | 360 | 3/3 | 6/6 | `engineering_accepted`；六份 P/R capture 各 945 unmatched Work API |

RN-A 這輪實測 Median（不是 GI/P/R 可互換的時間標籤）：

| mode | Median (µs) | Three-process spread |
|---|---:|---:|
| GI | 2961.376 | 1.05% |
| P | 4691.680 | 0.96% |
| R | 10739.552 | 1.16% |

所有 42 個 P/R capture 的摘要均記錄已知 `Not all NVTX events might have been collected.` warning，並被工程政策 warning-accepted；**這不證明 Reference 完整**。以上 unmatched Work API 數量是每份 capture 的摘要計數，不應直接解讀為實際遺失 kernel 或零長度 Memset。

`strict_reference_verified=false`、`training_eligible=false`、`models_trained=0`、`new_confirmation_dataset=false`。工程收集完整不等於正式 I3 operation label 已可訓練。

## 3. 已知失敗與版本沿革

- RN-A：在舊版單次 `nvidia-smi` >5% Guard 下，首輪 5/9、重收 8/9；保留兩輪資料。新 Guard 版本重收 9/9、Spread 全過。
- UN-A：舊版 Guard 收 3/9，其餘 GPU Utilization 83%／`process_query` 無 Compute Process；新 Guard 修正版收 9/9，但 P spread 5.7697% 尚需 Review。
- QD-A：第一輪 SSH 中斷，`inventory.json` 遺留 `running`，沒有完成 Cell；重新使用 tmux 之後，最新 QD-A 收 9/9。
- MV-A、YO-A、QP-A、SW-A 的 40×3 收集均完成；保留所有 P/R diagnostic warning 與 unmatched API 數字，不覆寫原始資料。

## 4. GB10 匯入與分析要求

- RTX3090 來源目錄：`/home/iccl/RTchen/o4-data-3090/unified_v2/`；資產：`/home/iccl/RTchen/o4-data-3090/frozen_A/`；程式碼為 `/home/iccl/RTchen/gpu-prediction-o4-unified`。
- GB10 SSH：`good_egg@140.113.179.4`（`gx10-c206`）。建議傳輸位置：`/home/good_egg/RTchen/RTX3090_archives/`，分析資料解壓後放獨立子目錄，不覆蓋 GB10 原本的 O4 工作。
- **完整交付建議含** `unified_v2` 下所有成功／失敗／中斷 session（可在分析時挑選最新完整 run）、完整 `frozen_A`、實際 Collector git commit/source snapshot、Run index、SHA256 清單。包含 `.nsys-rep`、`.sqlite` 與原始 log／audit，不只 `SUMMARY.json`。
- 原始資料與權重 **不放 GitHub**；本文件是結果索引。先完成可重現的傳輸完整性驗證，再建立 Operation/Shape/Kernel Coverage、R→P 對應與 unresolved API 診斷報表。
- 不把同一個 A 的三次程序／40 iteration 當成三個／120 個獨立 shape 配置；本批七組 A 是七個 workload anchor，不足以單獨支撐具有未見 shape 泛化聲明的新 I3 MLP 訓練。

## 5. 後續

1. 在 GB10 校驗傳入資料與 SHA256，依 `SUMMARY.json` 為七組 A 選擇最新完整 run；保留舊版失敗 session 供故障分析。
2. 優先完成 **Operation coverage／unmatched API／event coverage**，將 CTX Reference 與 GI label 分開。
3. 對既有 O3 受控三類 N0 及辨識器做合法、明確標註限制的 zero-shot transfer；再按缺口追加新的受控 Shape／Operation 類別。
4. RTX4090→RTX3090／GB10 的正式比較仍需核對模型、權重、輸入、精度、forward window、各方向合法交集；測試目標時間僅 evaluator 可讀。

相關：[O4 節點進度](O4_NODE_STATUS.md)、[收集政策](O4_COLLECTION_POLICY_20261107.md)、[O4 Quickstart](O4_QUICKSTART_ZH_TW.md)、[O3 結果](O3_Q_ABLATION_STATUS_20261106.md)。
