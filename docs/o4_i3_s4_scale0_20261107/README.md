# S4 scale0六圖：GB10實測與可移植交付

A＋B＋C已完成40個D/I/S工作，120 Smoke cells、360 formal cells、14,400 formal measured。不是原120工作全矩陣：scale1/2未收。六圖是CBR、Residual、Depthwise/Pointwise、SE、Resize/Concat/Conv、eager Attention，仍非六模型完整forward。

共320 trace／160 pairs重播，14,480個artifacts一致；R參數11,676/11,676匹配。P_CTX位置69 stable／15 review。全部只有context_scale_0一個保守群，無OOF或模型test成績。

[A原報告](A/REPORT_ZH_TW.md)／[BC原報告](BC/REPORT_ZH_TW.md)／[六圖描述性GI比較](D_I_S_COMPARISON.csv)／[RTX節點操作](../O4_I3_S4_NODE_RUNBOOK_20261107.md)。

Reference仍全部unverified。NVTX／clock及各種用途限制保留；hash重播／P-R一致／closure算術不能單獨證明完整Reference。沒有訓練辨識器、N0或ITER；RTX本次只有命令預覽，未GPU實測。

A與BC的source/PLAN/schema版本分開，舊release不變。下載共用base資產後，每節點先在當地GPU產生native-D local inputs，再用於當地I/S。GB10_local_inputs archive僅供診斷／重現GB10，不能直接當RTX本地等價控制。

原報告原bytes保留，其中未發布／planned時間語句是閉檔當時狀態，本次後續發布不回寫原run；未複製的raw/evidence連結只在節點原目錄有效。A對話中斷與恢復記錄保留，不重複計數。

D/I/S時間差只作描述性對照，不是純依賴因果或模型收益；所有GI穩定性需一起查看。ΣS是多次獨立forward的中位數相加，不能當D圖真值。
