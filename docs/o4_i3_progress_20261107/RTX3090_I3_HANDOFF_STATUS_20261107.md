# RTX3090 O4 I3｜Core36、S3-lite、S4-scale0 交付狀態

**資料來源**：研究操作者提供的 RTX3090 終端輸出與 ChatGPT 對話紀錄；原始檔案並未直接由本次 GitHub PR 讀取。用途為封存索引及重播指引，不取代 Raw Trace 獨立審計。

## Core36 + S3-lite

- 72 個 distinct configurations：Train **48**、Validation **12**、`test_exploratory` **12**。
- S3-lite：36 個配置、**324/324** Formal Cells、**12,960/12,960** measured iterations、**108/108** exact P/R pairs。
- S3 Formal Mode Stability：**47 stable / 61 review_required**。
- Core36 + S3 final pre/post collection metadata audit：**9,720** 筆 R Runtime Parameter Rows；**0 mismatches**、**0 cross-split exact collisions**。
- 舊版 post-test 稽核輸出的 `Status: REVIEW_REQUIRED` 與表列 `test_exploratory` split 名稱不匹配有關（原腳本僅接受 `test`）；**不能聲稱原稽核已重新執行為 PASS**。其餘輸出的規模與一致性項目均已符合預期。
- S3 Train 中一項 RTX3090 Depthwise 使用單獨註記的 No-cuDNN Backend Variant；不可不加標記地與其他節點的相同配置合併。

## S4-scale0

- A + BC 六圖、40 Work Items（同 `context_scale_0` leakage group），**120/120 Smoke Cells**、**360/360 Formal Cells**、**14,400** Formal Iterations、**120/120** Exact P/R Formal Pairs。
- RTX3090 final audit status `PASS`；**11,676** R Parameter Rows、**0 mismatches**、**0 batch issues**、**49 stable / 71 review_required**、**320 warning_accepted**。
- 具體每張圖指標、堆疊版本和 A/BC SHA256：見 [S4 RTX3090 COLLECTION_SUMMARY.json](../o4_i3_s4_scale0_20261107/RTX3090/COLLECTION_SUMMARY.json) 與 [RESULTS_ZH_TW.md](../o4_i3_s4_scale0_20261107/RTX3090/RESULTS_ZH_TW.md)。

## GB10 收件與分析

RTX3090 節點資料源：

- `~/RTchen/i3-data-3090/Core36_S1_smoke_PBfZ0zgK`
- `~/RTchen/i3-data-3090/S3_lite_B3FiyVTe`
- `~/RTchen/i3-data-3090/S4_scale0_3090_01`

請依 [package_and_scp_rtx3090_i3.sh](../../scripts/package_and_scp_rtx3090_i3.sh) 分資料根目錄產生 `analysis.tar.gz` 和 `full.tar.gz`；接收端完整核對 SHA256，再進行 P_CTX、Position Ownership、Clock／Overlap、跨裝置對照。封存實際 bytes/hash 由 RTX3090 產生，GitHub 不能事先推定。

**研究用途限制**：S3 與 S4 均保留 `strict_reference_verified=false` / `formal_training_eligible=false`；未在本次執行模型訓練、OOF、Test Predictor Validation。Raw NSYS/SQLite 不提交 GitHub。
