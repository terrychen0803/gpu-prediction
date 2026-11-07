# O4 I3 → I4 → I5 後續實驗路線與務實資料使用原則

更新：2026-11-07（Asia/Taipei）｜性質：**新規劃與開發資料使用政策；不是新增 GPU 實測結果**。

## 0. 目標、來源與適用範圍

- **最終研究問題**：以 RTX4090 的 P-only dry-run GPU Trace 作來源，推估目標 GB10／RTX3090 的 Operation Contextual Time，最後預測無 profiler 的完整 Forward **GI / iteration_device_event_us**。ResNet-50、MobileNetV3、Swin-T、U-Net、YOLO26n、Qwen3-0.6B 六類任務（Prefill／Decode 分為七群）維持原規劃。
- **本次直接依據**：[使用者原文 I3 實驗報告](REPORT_ZH_TW2.md)與[使用者原文 I3 Portable Runbook](PORTABLE_RUNBOOK_ZH_TW.md)。兩份文件逐 byte 原樣保留；其中的 `analysis/`、`evidence/`、`source/` 等相對路徑指向**原本 GB10 本地資料包**，不表示 GitHub 已附上這些檔案。
- 既有 [O4 I0～I7 計畫](../O4_GPU_ITERATION_PLAN_ZH_TW.md)、[三節點收集優先原則](../O4_COLLECTION_POLICY_20261107.md)、[統一工程分支](https://github.com/terrychen0803/gpu-prediction/tree/o4/unified-engineering-20261106) 仍適用：**各 GPU 可先獨立收集有用配置；不要求三台每配置全數對齊才能進下一項工作。** 跨設備監督或評估時才核對可比較的來源—目標配對。
- **使用者本次決策**：品質門檻採「**可用優先、註明不確定性、先作探索性模型**」，不要求所有 `strict_reference_verified` 完全通過才允許研究迭代。此原則不抹除舊 `strict_verified=0`、`formal_training_eligible=false`、NVTX 警告、clock／PDL／未歸屬事件，也不將先前資料追認為 formal ground truth。新增一套 **development-only eligibility**，與原 strict gate 分開。

## 1. 已知基準，不將規劃冒稱實測

2026-11-07 上傳的 I3 報告：I3-0 離線重新解析三台七 A 歷史資料，2,520 matched measured forwards；I3-1 在 **GB10** 以九類受控算子各一個 `_a` 配置完成 GI/P/R **27 個選定 Cells、81 measured**；因 source 可追溯性補跑，總共 **33 Cells、99 measured**，兩個數量不可相加或當成 11 個不同配置。18 個 a/b 固定候選資產已建立，但只測九個 a。RTX4090／3090 **本輪 I3 A0 尚未執行**。

舊 O3 detector/N0 權重與 27 維 normalization hash、特徵重播已核對，但新六任務的輸入幾何／非 Kernel 事件／協定不相容，未實際進行 O3 權重推論。I4／I5 的新模型、OOF、正式驗證集、40×3 **均未完成**。詳見[本次狀態摘要](STATUS_AND_HANDOFF_ZH_TW.md)。

## 2. 五個執行階段：依增量資訊與低成本優先

### Step A｜先把 I3 A0 開發版交給 RTX4090、RTX3090

1. 從 GB10 **另行傳輸**完整 `source/`（包含 `o4_i3/`、相依原 O4 與 tests）及 `assets_A0/`；GitHub 此次只上傳兩份 Markdown，**沒有**提供全部程式、38 operand bytes／18 配置資產或原始 Nsight 檔。依原 Portable Runbook 建立相容環境，不盲目升級 driver、CUDA、PyTorch／Nsight。
2. 比對 `assets_A0/ASSETS.json` 的 SHA256：`02e4fbe40f617d571d42b55d3e343c8ea4272b6f26febefa16f0de8bc4594e69`，並對清單中的 operand 逐檔校驗。不可在 RTX 本機重新產生輸入後仍叫相同 Signature。
3. 各節點先執行 CPU tests → `A0_batchnorm_eval_a` plan-only → BN 的 GI/P/R 小型 Smoke；保存每個 Cell 的 Source Snapshot、環境、GPU guard、actual backend、raw、檢查結果與失敗。
4. 之後可依增益選擇 Conv／GEMM／BMM／Linear／ReLU／Add／MaxPool／AdaptiveAveragePool `_a`；**不要求各節點立刻測滿九類**。不支援、OOM、忙碌或來源缺失留下原因，先跑其他可用工作。
5. 跨設備比較只用定義、權重／輸入 bytes、precision、forward 範圍及時間標籤真正可比的子集。GPU kernel 名稱、實作、數量可因架構不同而不同。

**完成標準（工程）**：至少 BN 的計畫／實測結果能在指定設備重現；其他工作列出 completed／not-run／failed 原因與 hash。這**不是**全節點 Reference verified 或模型訓練完成。

### Step B｜補更多獨立 Operator Signature，而不是重複相同 Shape

優先讓新資料覆蓋 I2 的多任務缺口及歷史 1,404 個參數缺口，受控組合須每組凍結完整 `op+shape+stride+layout+dtype+groups+bias+context+backend` Signature，例如：

| Family | 推薦受控變化（候選，不是已執行） |
|---|---|
| Conv / Depthwise | B／H／W、Ci／Co、1×1/3×3、stride1/2、padding、groups=1/depthwise、layout |
| GEMM / BMM / Linear | M/N/K、batch、bias、transpose／stride、reduction／layout |
| BN / Norm | channels、空間長度、training=false、epsilon、contiguous vs 轉置 view（正確性先驗證） |
| Activation / Add | ReLU／SiLU／HardActivation、broadcast、dtype、shape |
| Pool / Resize / Concat | maxpool/adaptive avg；補 bilinear resize、scale、align_corners、concat／copy |
| Attention / Cache | Q/K/V、softmax、RMSNorm/RoPE、不同 prefill/decode cache 長度、prepare 範圍 |

- 每個 Family 的開發集、驗證集、保留集以**獨立 Signature Group**切分，避免相同基礎 shape 的 repeats 或三台 GPU 被拆到不同集合。兩個 a/b 資產不足以完整做 train/validation/test；切分不足就明示 *development-only*, 不偽造 test 泛化。
- 從目前未歸屬 Qwen 720 個事件、某些長片段、resize 誤分類與單位參數缺口回推最有價值的配置；不要先盲目展開全部 21 DNN 配置／567 Cells。
- 舊 O1/O2 trace 能在時標、feature schema 與品質旗標相容時重用作 development／baseline，不用重新收所有三類。但不把 O2 host G0 改名成 O4 GI，也不回填缺失 stride/layout 成真值。

**完成標準**：各 Family 有一份 Signature／參數矩陣、實收與缺失名單、可用子集及 leakage-safe split feasibility；沒有硬性規定三台都必須有同等樣本數。

### Step C｜建立依賴子圖，觀察 Context 時間

優先按六任務實際需求實作：

1. Conv → BN(eval) → ReLU，含不同 stride／C／batch 與依賴張量；不對每個 op 加同步。
2. Residual Add / skip 以及依賴與獨立對照，對比同 Signature isolated 與 contextual CTX。
3. Depthwise Conv＋SE／HardActivation、U-Net resize＋concat＋Conv、YOLO neck/head。
4. Swin attention／LayerNorm、Qwen RMSNorm／RoPE／cache-update／prefill-decode；定義 backend、準備範圍及 cache state。

**資料型態**：P-only 數值事件特徵、R evaluator 邊界／綁定參數、Recorded GPU interval union/gaps/overlap、無 Profiler GI（若測完整受控 sequence），分別保存 time target。不要把不同的 device event elapsed、profiler CTX、host G0 合併同標籤；不要把 R 真實 shape/type 或目標測試資料洩漏進主 Predictor。

### Step D｜版本化 I3 40×3，從 Smoke 走向可比較的資料集

- 保留現有 `o4_i3_A0_smoke_v1_20261107`：每 Cell 1 個程序、8 warmup、1 transition、3 measured，只作工程驗證。
- **新 protocol ID** 另實作 40 measured ×3 個獨立程序重複（warmup、transition、precision、backend、cache、failure、hash 另凍結）；這是**未實作的建議**，不是可以改一個 CLI 參數就取得的正式結果。
- 以單配置小批次驗收再擴展。穩定性建議同時回報 median、repeat spread、不同進程分布與可疑離群；不因 spread 未達舊門檻就整批丟棄，可保留並加權／分層，但不得刪除失敗組只報通過的組。
- 每次 source 或量測語意改變需新版本、保留前版原始資料；保護既有 O1/O2/O3/O4 raw。

**完成標準**：可在至少一台 GPU 重現小型 40×3 並生成完整 Quality／Provenance 報告；有跨設備可比交集時再擴 source→target。未經執行不得寫成正式資料收齊。

### Step E｜I4 先做探索性模型，再逐步導入 I5

- **I4**：先以 R 的已知 Boundary 做分類可辨識性診斷（oracle boundary），再對 P-only blind segmentation、typed F1／coverage、N0 target CTX 誤差分別測試；比較 **legacy O3 frozen baseline**、新模型 from scratch、舊資料預訓練／相容權重轉移。若舊 27D geometry 不相容，不補零／丟 Event 偽造同介面；新版本特徵或模型另建。
- **I5**：當來源 RTX4090 P-only 特徵與 GB10／RTX3090 的**可比 GI** 有可用交集時，先作 T-SUM／簡單統計 Baseline；有足夠獨立 Group 及上游模型後，使用 **grouped OOF** 的 N0/Recognizer predictions 訓練 T-ITER Base／Overlap 消融。沒有足夠 Group 時可作單一方向的工程 prototype，但不報 held-out generalization。
- 留出 Confirmation 必須在模型／特徵／政策凍結後才使用；目標 test 的 P/GI/reference 僅限 evaluator，不能參與來源特徵選擇／模型調參。

## 3. 本次品質政策：低阻礙，但不混淆證據

**新增開發資格，與舊的嚴格真值資格平行運作**：

| 新建議標記 | 能用在哪裡 | 最低必要條件 |
|---|---|---|
| `development_usable` | exploratory Operator/CTX 訓練、Baseline、可比 GI 映射原型 | 工作定義／hash 可追溯，GPU correctness 與時間 target 合法；必要 P/R 結構存在且事件 Ownership 能定位；無已知致命 dropped/missing activity 破壞該特定標籤 |
| `limited_use` | 單設備 Inventory、時間分布、GI-only 研究、diagnostic、部分不受影響單位 | 至少量測與缺失範圍可界定，缺失 op/event/時序不補造；只評估真正可支援的目標 |
| `reject_for_specific_label` | 不能當成**受影響的** Op/CTX 或 GI 訓練標籤，其他獨立可用資料仍保留 | 已知 API 失敗、標籤超出收集範圍、GPU activity 明確缺失、破壞該標籤的 timestamp/geometry、錯誤輸入或無法追溯版本 |

**實務解釋**：

1. 單獨出現 `Not all NVTX events might have been collected.` **不再自動停止所有 exploratory training**。若預期標記／已記錄事件分段、P/R 結構一致、必要 API/correlations 可核查，將影響記成 `nvtx_warning_accepted=true`，仍只稱 *development*，並做排除 Warning 資料的敏感度對照（如可行）。
2. **不可忽略已確定的 GPU Kernel 遺失**（如 GB10 歷史 Hardware Capture 0/80），也不可把未知事件歸屬的 Operator 當 0µs 真值。對不完整或異常樣本用可解釋的 mask／sample weight／用途限制，保留 raw 和 reason；不是將所有不能 strict verified 的資料整筆刪除。
3. 有合理 GPU Event Time、輸出與 Scope 的 **GI** 可先做迭代時間 Baseline，即使其 **R Reference** 未完成 strict gate；此時不可使用不可信的 R-derived target Op labels。P 來源與 GI 目標在跨設備配對時仍要真的是同一工作定義。
4. 分開回報 `engineering_completed`、`development_usable`、`strict_reference_verified`、`formal_training_eligible`、`models_trained`。不更動歷史的 `strict_verified=0`／`formal_training_eligible=false`，也不修改 frozen protocol 的原始 Gate；新政策作為 **overlay／experiment scope**。
5. 三次樣本可以用於 **API／Collector 開發及單機 exploratory plot**，但不能宣稱三程序穩定性。沒有足夠獨立 Signature Groups 時，可以實驗地 fit 小模型，**不能報成獨立未見配置的泛化成績**。

### 建議每筆開發資料保存的審查欄位（新規劃，不表示已實作）

```text
source_run_sha | protocol | device | workload_signature | input/weight_hash
label_kind | mode | backend_actual | event_count | unmatched_api_count
nvtx_warning | cuda_drop | timestamp_anomaly | missing_ownership
development_use = usable / limited / reject_for_specific_label
allowed_targets | exclusion_reason | strict_reference_verified | formal_training_eligible
```

工程團隊應先產生**唯讀審查清單**並手動/單元測試核對，勿一次改寫所有舊檔 status；模型訓練 pipeline 再明確選開發資格。

## 4. 交付清單與下次 Codex handoff

- 保存本次兩份 [原文報告](REPORT_ZH_TW2.md)、[可攜手冊](PORTABLE_RUNBOOK_ZH_TW.md)及[當前證據摘要](STATUS_AND_HANDOFF_ZH_TW.md)。
- 在 GB10 補齊本地 I3 `source/`、`assets_A0/`、source snapshots、原始 Nsight與 manifest 的備份／節點搬移；**它們目前未隨本文件 PR 提交**。
- RTX4090／3090 每個 Smoke 使用獨立新 output root；不碰舊資料；首先回傳 BN A0 的 Plan/GI/P/R 完整摘要。
- 依實際 Group 覆蓋設計新 Configuration、Context、Quality overlay 和 40×3 protocol，依單 GPU 可用性逐步收集，不設「三台全部通過」作阻塞條件。
- I4/I5 的第一次模型結果要明示 *exploratory*、資料 group 數量、泄漏防護、可用比例與警告；正式 Confirmatory Result 待獨立 split/reference 條件達成。

> **研究紀律**：降低的是「阻止開發的門檻」，不是改寫物理時間定義、忽略已知資料遺失，或宣稱未執行的實驗已完成。