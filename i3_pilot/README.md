# I3 A0 可攜收集工具

節點直接依 [兩節點操作手冊](../docs/O4_I3_NODE_RUNBOOK_20261107.md) 執行。固定 release：`o4-i3-a0-dev-v1-20261107`，從目前 PR #8 分支發布，不需要先合併 main。

`source/` 是 GB10 已驗證交付的155個原檔，逐byte保留，包含 `o4_i3`、其凍結 O4 相依副本、測試與原手冊。凍結副本內較早的README／進度為歷史資訊；目前節點進度與用途以本手冊及 docs/O4_NODE_STATUS.md 為準。

**可立即執行**：18個固定資產候選；GI/P/R A0 Smoke，每cell一個程序、8warmup＋1transition＋3measured。GB10只完成九個a的GPU驗證；b及RTX I3尚待當地執行。不同GPU的TorchDispatch／Nsight相容性由當地smoke驗證。

**規劃待實作**：[174個配置、依賴子圖及新40×3](../docs/o4_i3_collection_plan_20261107/PLAN_ZH_TW.md)。不可將 seven-A 40×3 collector 或重複A0 Smoke冒稱這套新協定。沒有進行新的模型訓練；strict_reference_verified維持false。

固定FP16 operand bytes透過GitHub Release傳輸，包含初始化的控制參數，不是DNN預訓練權重。raw trace／SQLite／模型權重不提交Git。新收集一律使用獨立output path，保留failure及actual backend。
