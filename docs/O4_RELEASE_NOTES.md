# O4 unified engineering v1 — 2026-11-06

這是一個可審查的整合候選，沒有合併 main 或將任何 Reference 升級。合併保留 PR #2～#6 提交歷史；O1/O2 root方法與 O4 vendor副本仍分開，O2 G0不能替代O4 GI。

- 七A40×3新protocol v2；v1從歷史tag保留。
- 新audit_v5為附加工程驗收；v4數值／品質輸出保留，新增獨立 engineering_acceptance。
- 精確NVTX例外；CUDA drop、未知diagnostic、API return、backend、數值特徵契約各自驗證。
- 全source dependency與完整14檔資產核對、版本snapshot、動態replay數量、獨立collection/quality狀態。
- 共用node入口；native schema3.28/3.31 structural檢查與diagnostic政策，native v3尚未實測。
- 歷史code_release v1/v2不變；新增unified_release分發索引與驗證。
- 新版GPU canary待GB10/3090/4090各自執行；Windows live pending。

離線驗證結果及限制見 [驗證報告](O4_UNIFIED_VALIDATION_20261106.md)。各舊分支封存策略見 [分支清單](BRANCH_ARCHIVE_20261106.md)。
