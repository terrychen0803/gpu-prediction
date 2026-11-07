# GB10 S3-lite：36配置收集與品質報告

GPU執行狀態：`completed_all_preselected_batches`。本輪只做S3建議36配置，不是全138；沒有模型訓練或test模型分數。UTC 2026-11-06T23:04:59.833023+00:00～2026-11-06T23:49:12.886320+00:00，GPU收集流程牆鐘 44.22 分鐘（含guard、fresh process、Nsight/export/audit，非純GPU compute；不含後續CPU重播與報告）。

## 已完成 GPU 實測

S3 Smoke 108/108 cells；formal 324/324 cells，保存 12,960 formal measured，含Smoke共13,284 measured。每個formal配置GI/P/R各三次獨立程序，每次40次measured，未減repeats。

| 批次 | split | 配置 | Smoke放行後formal排除數 |
|---|---|---:|---:|
|train_controls|train|12|0|
|validation_01|validation|12|0|
|sealed_test_02|test_exploratory|12|0|

S3 formal模式穩定性：94 stable、14 review；這是GI/P/R CUDA event各自三程序spread，不是P_CTX。實際P_CTX為32 stable／4 review。依新衍生用途gate，S3 P_CTX 31 development_usable／5 limited_use。全部仍不具formal_training_eligible。

## 已完成工程實作與核對

181 CPU tests（v2 18＋原I3 16＋O4 85＋legacy 54＋編排gate 8）通過。collector/worker/parser與前輪165檔source原bytes相同；新編排器只決定既定三批與Smoke放行，不改量測語意。六組預覽不是GPU實測。36配置固定資產含74個operand，SHA見evidence與BUNDLES。

S3重播 288 traces、144 P/R pairs，13,048 衍生artifacts逐byte一致；36個實際P_CTX摘要獨立對照重播標籤一致。合併S1/S2/S3的R參數檢查 10,008/10,008 匹配。Hash／重播一致只證明流程一致，不證明Reference完整。

| 歷史run | 原檔SHA重新核對未變 |
|---|---:|
|o4_i3_s1_gb10_20261106T213722Z|3,545|
|o4_i3_s2_gb10_20261106T220336Z|31,342|

## Signature切分與封存test

合併共有72配置；train配置48、validation12、exploratory test12。Controls與anchor00同群，不能額外算獨立train群；總60群，其中train36群，validation/test合計24群。各家族只有一個validation／test anchor，仍是薄弱的探索配置集；沒有OOF fold或正式確認集成績。

靜態174候選完整定義未見exact跨split碰撞；實際完整ATen arguments/output signature碰撞共0組，其中跨split0組。若非零，必須先處理連通群才可稱未見配置。此稽核不保證所有數學等價或近鄰配置都已辨識，也未排除過往DNN／A0資料對舊模型的曝光；未來主分析需from-scratch controlled-only，歷史預訓練另列。

Test收集前已凍結配置與用途政策；只做correctness、coverage、signature、ownership、穩定性等資料品質稽核，沒有計算模型test誤差、選特徵或調超參數。這是邏輯上的evaluator-only封存，不是加密或權限隔離。完整時間表仍為evaluator資料，不給predictor讀取。

## Reference 未驗證：不能升級正式標籤

S3共有288份P/R raw trace，NVTX warning出現在288份；未匹配工作API 556筆，CPU/GPU clock start-before-API 4669、end-before-API 311筆，overlap 180對。保留原timestamp、overlap與API，未平移／裁切或推斷未匹配memset為0B，也未以kernel名稱證明PDL。

合併recorded ownership：unassigned 0、multiple 0、noncontiguous 0、zero-recorded 0。這些recorded-work檢查、P/R exact或低GI波動均不能單獨證明submitted-work/NVTX/clock完整。strict_reference_verified=false，formal_training_eligible=false。

Train controls 的bias-off可診斷Linear配置差異；不能反向證明舊Linear未匹配API是合法0B。`I3_depthwise_00_control1`的groups=C/2在語意上是grouped convolution；catalog family是分組來源，不當作真實depthwise標籤。R完整參數、真實類型與shape只供evaluator；主predictor仍只用來源P numeric allowlist。

## 僅有規劃／尚未完成

完整S3剩餘102配置沒有執行；Context DAG adapter／CBR／Residual等尚未實作或收集。本輪未跑RTX4090／3090，未訓練辨識器、N0或ITER，未完成DNN transfer、未知類型、未見GPU或完整forward預測。各節點可依有用子集獨立收集，跨設備評估才取可比較方向交集。

## 可重現資料

- [凍結範圍](SCOPE.json)、[切分](SPLIT_MANIFEST.json)、[封存政策](SEALED_TEST_POLICY.json)、[執行命令／時間](EXECUTION_STATUS.json)。
- [合併72配置evaluator表](analysis/summary/CORE72_METRICS.csv)、[P_CTX穩定性](analysis/summary/P_CTX_STABILITY.csv)、[用途限制](analysis/summary/DEVELOPMENT_USE.csv)、[碰撞](analysis/summary/R_COMPONENT_COLLISIONS.json)。
- [重播摘要](analysis/raw_replay/REPLAY.json)、[驗證](FINAL_VALIDATION.json)、[跨節點手冊](NODE_HANDOFF_ZH_TW.md)、[可攜包SHA](BUNDLES.json)。
- 原始raw與失敗保留於collections/；前置catalog split-label斷言失敗與修正見evidence/PREPARATION_FAILURES.json，發生在任何GPU啟動前。

本輪source與資產包尚未發布GitHub，既有Core36 release不包含S3資產。原始資料未覆寫，系統服務未修改。

## 本輪P_CTX受限配置

| 配置 | 三程序P_CTX spread | formal未匹配API |
|---|---:|---:|
|I3_bilinear_resize_00_control1|10.56%|0|
|I3_gemm_00_control1|5.85%|0|
|I3_linear_00_control1|6.09%|0|
|I3_linear_01_base|0.00%|270|
|I3_linear_02_base|5.97%|270|

## GPU與背景工作觀測

432個獨立worker程序皆已結束；432個cells的guard沒有blocked，monitor沒有foreign-work／error。5秒monitor每cell實際取到1筆，共432筆，不能證明採樣間隔內沒有其他工作。

前置預檢可見既有RustDesk compute PID；12筆train-controls Smoke sample仍可見它，所有formal的monitor sample均未見它。收集後該PID已不存在，原因未知；本實驗沒有發出停止／刪除服務、Pod或外部程序的命令。S1／S2與本輪背景狀態不可假設完全相同，跨輪時間差異應分層解讀，不能全歸因於配置變化。見evidence/GPU_POST/PROCESS_AND_MONITOR_REVIEW.json。

Test開始前已另完成6,810筆train/validation完整R參數稽核，12個validation配置均已實際觀測、無參數不符與exact跨split碰撞；當時test尚未啟動，見evidence/PRE_TEST_RUNTIME_AUDIT.json。

跨節點定義／資產身份與用途限制見 [ALIGNMENT_MANIFEST.json](ALIGNMENT_MANIFEST.json)。原SCOPE／catalog內precollection狀態是凍結時快照，當前完成實績以FINAL_VALIDATION／STATUS及本報告為準。
