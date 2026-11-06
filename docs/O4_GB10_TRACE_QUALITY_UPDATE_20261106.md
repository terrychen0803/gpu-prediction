# GB10 O4 Trace 品質診斷補充與規劃修訂（2026-11-06）

> **狀態：唯讀診斷的公開摘要，未新增 GPU 實測。** 本文件根據 GB10 本地 Codex 的 `REPORT_ZH_TW.md`，以 GitHub `43a04a3689a27bc14f3b96aedcf7aef2d3c4853f` 為基準撰寫。完整 SQL、figures、版本化 quality overlay、native MRE 原始碼與原始 `.nsys-rep`／`.sqlite` 仍在 GB10 本地，**不在本次 GitHub 交付中**；本文是報告彙整，並非對原始 trace 的遠端獨立重算。

關聯：[O4 主計畫](O4_GPU_ITERATION_PLAN_ZH_TW.md) · [數學規格](O4_GPU_ITERATION_MATH.md) · [既有 GB10 Pilot](O4_GB10_PILOT_STATUS_20261106.md) · [跨節點對齊](O4_NODE_ALIGNMENT_20261106.md)。

## 1. 已觀察到的事實及限制

- RN-A 六份 P/R captures：各含 45 iterations（5 transition＋40 measured），共 **270 iteration**、**94,500** 個已記錄 GPU 工作事件（90,180 kernel＋4,320 memset；無已記錄 copy）。
- **123 對 GPU interval overlap**：全部同 device0／context1／default stream7、kernel→kernel，跨 ATen 邊界；左側 `cudaLaunchKernel_v7000`、右側 `cuLaunchKernelEx`，後繼為 `nvjet_sm121_*`。ReLU→Conv 104、MaxPool→Conv 12、BatchNorm→Conv 7。P 98 對、R 25 對，**25/270** iterations 有 overlap。所有相關 API 提交及 GPU start 的事件順序都未反轉。此模式與 Programmatic Dependent Launch (PDL) 相容，**但缺少真正的 programmatic launch attribute 證據，不能宣稱已證實 PDL**。
- Pair overlap：median 384 ns、P90 11.194 µs、max 30.336 µs。按觀測到的連續片段作 `U_P = ΣCTX_span + G_positive − O_union`，270 iterations 的閉合誤差全為 0；不扣 overlap 會在 25 iterations 額外多計合計 384.960 µs，單 iteration 最多 49.120 µs，最多約對應 trace span 的 1.416%。**這是 P/R interval 閉合，不是 GI 的誤差或排程證明。**
- **1,579** 個 GPU end 早於其 CPU API start：1,509 memset、70 kernel，median 689 ns、max 1.937 µs。已排除 Python parser 任意修改 timestamp、跨 stream 誤合併與 GPU→API 多解 join 等幾個具體假說；仍不能排除更上游 clock conversion／collector callback 等問題。不能用固定 offset 自行校正。
- **270 個成功返回但未匹配 GPU activity 的 `cudaMemsetAsync`**：每 iteration 一個；R 都屬 op140 的 1×1 Conv。GPU activity 16 個 memset／iteration，但 API 有 17 個。SQLite 未保留 `count`，因此可能是合法 0-byte no-op 或漏記工作，尚不能判斷。
- 所有 94,500 個已記錄 GPU 工作有唯一 API correlation，270/270 iteration scope 完整，R 每個已記錄 GPU Event 恰好分配一次，**P/R 135/135 iteration ordered signatures 一致**（120 measured＋15 transition）。但這些只證明 recorded-event 結構，不證明 submitted work 全部被記錄。
- 原 O4 共 28 capture（26 success＋2 early failed）皆有 NVTX 完整性 warning；成功 RN-A captures 未觀察 CUDA 完整性 warning，早期 exit139 failure 是另一種錯誤。NVTX warning 的實際漏記程度仍未獨立確認。**strict verified = 0 保持不變**。
- GI 是不開 profiler 的 CUDA-event 完整 forward 時間，並非由 SQLite parser 得出；GB10 RN-A GI 1,921.920 µs／process spread 1.89% **保留**。它無法證明 CTX／Reference 完整。

## 2. 品質 gate 必須獨立

| Gate | 內容／判定依據 | 目前 GB10 |
|---|---|---|
| `recorded_event_integrity` | GPU→API correlation、scope、已記錄事件的唯一分配 | 結構一致，不代表提交完整 |
| `submitted_work_coverage` | CUDA API→GPU activity 反查；無 activity 需證明 no-op／0 byte／不產生工作 | 270 個 memset API 原因未明 |
| `reference_semantic_alignment` | ATen／NVTX／API／GPU 事件對應，且沒有不可接受的漏記 | 結構可對齊，NVTX warning 尚在 |
| `timestamp_consistency` | CPU/GPU clock、GPU-GPU intervals、事件錨點及因果邊界 | end-before-entry 根因未確認 |
| `event_index_partition_supported` | 同一事件只屬一個連續分段，無排序逆轉 | 已記錄 RN-A 工作具相容結構 |
| `nonoverlap_sum_supported` | Operator interval 不重疊，可以直接 SUM+gap | 有 overlap 的 iteration 為 false |
| `overlap_aware_closure_supported` | 以 interval union、positive gap、重複計時量檢查閉合 | 在已記錄 RN-A 區間閉合 |
| `strict_reference_verified` | 完整提交、NVTX／事件可信度等資格全部達標 | **false** |

上述 gate 各保存 `passed`／`warning_accepted`／`failed`／`unverified`、證據檔與版本，不用單一 `composition_supported` 混合不同概念。Clock inversion 和 PDL 原因未確認時不得自行修補時間戳或替換已凍結 quality。

## 3. 對辨識器及建模的改動

**保留主研究方向：來源 RTX4090 P trace → event-index Operator 分段與類型 → N0 目標 CTX 時間 → T-ITER 完整目標 GI。**

Semi-Markov 的連續非重疊是 **event-index 集合分割**，不是時間區間互斥。GB10 的 overlap 尚未破壞已記錄的事件順序／R 分段連續性；但多 stream 交錯片段、漏記、fusion 與未知類別仍可能使辨識器不適用，必須標記 coverage／unsupported。舊 N0 僅對 GEMM/BMM/Conv 受控資料驗證，RN-A 需 BN/ReLU/add/pool/Linear、shape 和候選上限 L 重新訓練／驗收。

I5 應保留 `T-SUM`、`T-ITER Base`、`DIRECT` 和 wrapper 對照，再增加 **`T-ITER Overlap` 消融**：特徵從合法的來源 4090 P numeric events 計算 overlap count、duration／union ratio、positive gaps。來源 GB10/3090 test P、target overlap、GI、真實 op/shape 與 reference **僅 evaluator 可讀**。每組模型應在相同資料切分、OOF、訓練預算和有效 GI 失敗分母下比較；overlap feature 不是保證效能提升。

## 4. 實作順序及待做驗證

1. I1：先依 GB10 本地最小對照計畫執行原生 serial/no-PDL、64B/0B memset controls；每節點 SW／auto × P/R 四個 native captures（各 8 measured），驗證**實際 trace backend**（`cuda` 可 fallback）、NVTX、reverse API coverage、CPU/GPU timestamp。GPU 實驗 **尚未執行**；本 GitHub 未附完整 MRE source。需獨立記錄採集命令與 SHA。
2. 如果控制組仍無法判斷，再做 PDL launch attributes、device event completion 錨點、op140 Conv 對照。沒有證據不升級 `strict_reference_verified`。
3. RTX4090／3090 安全取得相同資產與 `o4_pilot` 程式，先 single smoke，再 three／RN-A；不覆寫 O1/O2/O3 舊資料。
4. I2/I3：建立 BN/activation/add/pool/Linear 等 taxonomy 與候選長度的受控資料，凍結 O3 現有 artifacts 作 baseline。
5. I4：ResNet bottleneck→stage→native forward 分別評分盲分段、類型、CTX 對照；正式 Reference 不達標時只做探索性 proxy。
6. I5：使用 union-based P-span 觀測診斷，公平比較 SUM／ITER Base／ITER Overlap／DIRECT／oracle wrapper，分組 OOF 校正；**不得用低 GI 誤差掩蓋漏記或錯誤標籤**。
7. I6/I7：兩方向分開報 GI MdAPE、P90 APE、coverage、typed F1、未完成原因；模型與 criteria 凍結後另外收 confirmation set，依 evaluator-only 目標 overlap／reference 狀態做分層分析。

## 5. 資料與版本原則

- `o4_gpu_iteration_smoke_v2_20261106` 與原本 39 successful cell 保持歷史可追溯性；只讀重解析保存獨立 quality overlay／audit 版本；變更真正 collector 時建立新 protocol ID 與 hash。
- 本文件所述 GB10 數字來自本地報告，原始 trace／診斷證據仍在節點，**尚未在 GitHub 發布完整獨立可重跑證據**；轉交至分析節點後再作二次重算。
- 正常 Overlap 不應自動視為資料品質失敗；但 submitted-work 缺口、NVTX warning、timestamp 可靠性各有獨立 Gate。未解時保持 `warning_accepted` 或 `unverified`，不把 exploratory 結果寫成正式驗證。
