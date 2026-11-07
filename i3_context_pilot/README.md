# S4 Context D/I/S工程交付

[RTX操作手冊](../docs/O4_I3_S4_NODE_RUNBOOK_20261107.md)／[GB10六圖結果](../docs/o4_i3_s4_scale0_20261107/README.md)。

source_A173檔與source_BC174檔各自是GB10實際執行的原bytes。A與BC的PLAN與asset版本不同，不能交叉混用；source內舊runbook／historical tests保留當時快照，當前入口以上層手冊為準。原i3_config_pilot、A0、舊release與raw不改。

固定shared_base可跨節點下載，GB10_local_inputs只供GB10診斷；RTX須在自己GPU/stack上derive自己的D實際中間輸入供I/S。scripts/derive_node.py在GPU前驗source/base，guard/monitor包住自己的有timeout子程序；--plan-only不啟動GPU。source自帶A／BC active tests；BC的tests_context_A_historical是保留快照，不當作BC active suite。

40工作只屬context_scale_0一群，scale1/2未收；沒有完整DNN／正式Reference或訓練結果。
