# S3-lite 已完成GB10結果

36配置：12 train controls＋12 validation＋12 exploratory test，44.22分鐘。108 Smoke cells、324 formal cells、12,960 formal measured，無失敗／排除。181 CPU tests；288 traces／144 P/R pairs重播，13,048個衍生artifacts逐byte一致。

合併S1/S2共72配置、60個保守group；10,008筆R參數全部匹配，exact完整參數跨split碰撞0。這不排除數學等價／近鄰或歷史曝光。沒有模型test分數、訓練或OOF結果。

S3 P_CTX 32 stable／4 review，用途gate為31 development_usable／5 limited_use；全部Reference未verified。288份NVTX warning、556未匹配API、clock與180對overlap持續保留。

[原報告](REPORT_ZH_TW.md)／[用途表](analysis/summary/DEVELOPMENT_USE.csv)／[完整72配置evaluator表](analysis/summary/CORE72_METRICS.csv)／[其他節點指令](../O4_I3_S3_NODE_RUNBOOK_20261107.md)。固定release `o4-i3-s3-lite-40x3-dev-v1-20261107`。

原報告和FINAL_VALIDATION保留閉檔時的「尚未發布」描述；本次另行發布，沒有回寫原run。未複製的raw/evidence連結仍屬GB10本地；原始raw不進Git。RTX本次未GPU實測。
