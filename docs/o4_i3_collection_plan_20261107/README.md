# I3 GB10 優先收集：具體計畫

狀態：planning_only。2026-11-07（Asia/Taipei）。原規劃回合沒有 GPU 執行或模型訓練；此目錄為後續發布副本。可執行的 A0 工具見 [I3 節點手冊](../O4_I3_NODE_RUNBOOK_20261107.md)。

- [主計畫](PLAN_ZH_TW.md)：實驗問題、順序、預算、已知限制。
- [算子矩陣](OPERATOR_MATRIX.csv)／[完整參數](OPERATOR_CONFIGS.json)：144 個基礎配置＋30 個同群控制。
- [子圖與對照](CONTEXT_PLAN.json)／[子圖說明](CONTEXTS_ZH_TW.md)：18 個依賴子圖、18 個斷依賴對照與逐位置 isolated 對照。
- [新 40×3 協定草案](PROTOCOL_ZH_TW.md)及[用途 gate](QUALITY_POLICY.json)。
- [分組規則](SPLITS_ZH_TW.md)、[節點交接與工程工作](IMPLEMENTATION_HANDOFF_ZH_TW.md)。
- [計畫一致性檢查](VALIDATION.json)、[來源](SOURCE_REVIEW.json)、[發布檔案 SHA](PUBLISHED_SHA256.json)。

JSON/CSV 是待實作的設計輸入，不是目前 A0 collector 可直接執行的 request。`build_plan.py` 只產生及檢查計畫，不會啟動 GPU。
