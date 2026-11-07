# S4-A GB10：CBR／Residual D-I-S 工程收集

## 結果與用途

已完成GPU實測：CBR5、Residual9，共14工作；42 Smoke cells＋126 formal cells，保存5,040 formal measured，含Smoke共5,166。每mode三個獨立程序、每次40 measured。這14項是D／I／逐位置S對照，只屬同一context_scale_0保守群；沒有獨立validation/test、OOF、辨識器／N0／ITER訓練或DNN預測成績。

本輪牆鐘UTC 2026-11-07T00:06:54.186712+00:00～2026-11-07T00:26:11.467422+00:00，約19.29分鐘，**包含對話中斷的空檔、重新確認與恢復**，不能當成連續GPU收集成本。各階段命令／monotonic時間分別見EXECUTION_STATUS.json與RECOVERY_STATUS.json；GPU資產derive另外在量測／capture外進行。

| 圖 | D GI μs | I GI μs | Σ S GI μs | D−I | ΣS−D | 全部GI穩定 |
|---|---:|---:|---:|---:|---:|---|
|CTX_conv_bn_relu_0|36.704|36.832|46.592|-0.35%|+26.94%|False|
|CTX_residual_0|73.120|72.496|101.216|+0.86%|+38.42%|False|

這是同設備描述性時間比較，不是模型預測／誤差，也不是純依賴成本的因果估計。ΣS是多次獨立forward的GI中位數相加，包含各自launch／提交／事件間隔，不能直接當整張DAG真值。D/I雖共享本地實際輸入數值與權重/state，仍有input placement、cache、allocator/workspace與跨iteration張量生命週期等差異；I保留單stream次序，只斷資料邊。

42個formal GI/P/R模式組中36 stable、6 review。實際P_CTX依圖／位置計30項，23 stable／7 review，依完整資料用途gate有23 development_usable。這些位置／模式彼此相依，不能當獨立樣本或宣稱正式泛化成功。

## 已完成工程實作

新增o4_i3_context，原165個source檔與歷史parser保持原bytes。新173檔source快照包含Context Adapter／協定／tests；collector使用新模組，不回填舊single-op協定。196 CPU tests（Context15、原v2 18、A0 16、O4 85、legacy54、編排gate8）通過。初始Context fixture缺少legacy bootstrap import造成失敗，已修正並保留失敗log；首次derive helper命令路徑錯誤亦保留，皆發生正式收集前。

Base固定外部FP16 input、每位置權重與BN state。GB10在capture外先執行D，記錄每條位置輸入及輸出bytes/hash；I/S載入同一份local manifest，逐位置核對native-D輸入／輸出exact及CPU FP32數值容差，BN state不變。GPU/torch/CUDA/cuDNN identity不同不可沿用local-derived資產。其他節點只能共享base，再在當地GPU產生local-derived資產，不能假裝各GPU中間bytes相同。

D按DAG傳遞；I按原位置順序用預先算好的輸入；S只跑一個位置。所有forward輸出保留到下次prepare，清除與同步在timing／iteration scope外。P只有iteration marker，沒有逐算子marker或同步；R才有ATen dispatch與完整參數，真值只給evaluator。

## 中斷與恢復

原CBR formal在42/45 cells後因對話中斷停止。原collections/cbr_formal與中斷cell完整保留，其inventory的running是停止當下快照，不代表仍有程序執行。確認無遺留worker或其他GPU compute工作後，複製至cbr_formal_recovery，把未完成cell副本移到interrupted_cell_copy，僅補原未完成三cells，沒有重跑／挑掉已完成42cells。原中斷資料未刪除或覆寫。

原42完成cells的2,128個檔案在恢復副本逐byte相同。active分析只取cbr_smoke、cbr_formal_recovery、residual_smoke、residual_formal；中斷原目錄不重複計數。恢復前後環境／時間分段保留，不能將複製資料當新獨立repeat。

## 品質與重播證據

active 112份trace、56對P/R重播，共5,068個衍生artifact逐byte一致。完整R位置參數4,170/4,170匹配。另由獨立重播P_CTX逐位置重新聚合30項40×3摘要，全相同。

Recorded ownership：unassigned 0、multiple 0、noncontiguous 0、zero-recorded 0。P operator-span union closure最大算術error 0ns。這些只證明recorded事件／算術對應，不證明submitted-work、NVTX與時鐘完整，sum/span/union/GI保持分開。

NVTX warning出現在112份trace，未匹配工作API 0，GPU start-before-API 7008、end-before-API 49，overlap 0對。保留原timestamp與interval，未加offset／裁切／猜0B，沒有證實PDL。**strict_reference_verified=false；formal_training_eligible=false。**

D/I/S與圖內共用位置有4個重複完整runtime signature key，這是預期對照，全部同context_scale_0群；與既有72 isolated配置exact component相同key數0。Exact比對不排除數學等價、近鄰與歷史DNN/A0曝光，不能靠hash不同宣稱新確認集。

## 舊資料與系統保護

| 歷史run | 原檔SHA核對未變 |
|---|---:|
|o4_i3_s1_gb10_20261106T213722Z|3,545|
|o4_i3_s2_gb10_20261106T220336Z|31,342|
|o4_i3_s3_lite_gb10_20261106T230230Z|34,520|

Active168個worker均已結束；foreign monitor cells=0、monitor error cells=0。每5秒取樣不保證採樣間隔內無干擾；GPU preflight與postflight見evidence。本輪沒有停止／刪除／調整K3s、Pod、Netdata、RustDesk或其他系統服務，也沒有升級driver/CUDA/Nsight。

## 僅有規劃／未完成

S4-B Depthwise/Pointwise／SE、S4-C Resize/Concat／eager Attention及scale1/2尚未實作或GPU收集。未在RTX4090／3090執行；六模型完整DNN、未見設備、unknown與排程效用未驗證。本輪是Adapter／量測與對照工程，未訓練辨識器、N0或ITER，不以低GI差或signature一致稱正式Reference。

## 交付

- [全部工作與協定範圍](SCOPE.json)／[原執行](EXECUTION_STATUS.json)／[恢復記錄](RECOVERY_STATUS.json)。
- [D/I/S描述性比較](analysis/summary/D_I_S_COMPARISON.csv)／[各工作GI與P/R event](analysis/summary/WORK_METRICS.csv)。
- [P_CTX位置用途與穩定性](analysis/summary/P_CTX_STABILITY.csv)／[位置參數核對](analysis/summary/R_POSITION_CHECKS.csv)／[P closure](analysis/summary/P_CLOSURE.csv)。
- [完整驗證](FINAL_VALIDATION.json)／[可攜包SHA](BUNDLES.json)／[RTX交接說明](NODE_HANDOFF_ZH_TW.md)。

新的source與shared/local資產包尚未上傳GitHub；舊Core36／S3 release不包含本Context package。所有新raw與中斷資料均保留本run，不改歷史資料。

## 需要review的模式與位置

| 工作 | mode | 三程序median spread |
|---|---|---:|
|ISO_conv_bn_relu_0_r|GI|10.81%|
|IND_residual_0|GI|9.27%|
|ISO_residual_0_c1|P|6.65%|
|ISO_residual_0_c2|P|5.38%|
|ISO_residual_0_a|GI|7.97%|
|ISO_residual_0_r2|GI|13.67%|

CBR的D和I兩個GI組各自穩定，但S中的ReLU不穩；Residual的I GI不穩，加上部分S GI不穩。因此兩張圖的完整D/I/ΣS比較都保留review，不能將上述百分比當精準收益。

| 工作 | 位置 | P_CTX三程序spread |
|---|---|---:|
|CTX_residual_0|r1|6.25%|
|IND_residual_0|c1|12.41%|
|IND_residual_0|r1|27.94%|
|IND_residual_0|c2|11.84%|
|IND_residual_0|a|27.59%|
|IND_residual_0|r2|17.65%|
|ISO_residual_0_r2|r2|6.25%|

R instrumented-output exact檢查採用worker的最終輸出plain／marked比對；逐位置native-D數值核對是在正式量測外的Adapter correctness執行。R完整位置參數、recorded event歸屬與signature重播另行驗證，但仍不能因此證明reference語意完全正確。
