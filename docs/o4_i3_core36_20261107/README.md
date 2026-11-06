# GB10 S1＋S2：36配置結果與交接

2026-11-07。已完成GPU實測：12家族、anchor00/03/04，共36個訓練配置；324個formal cells、12,960次formal measured。另有Smoke；沒有validation／test，沒有OOF、DNN結果或新模型訓練。

S2自身：396個Smoke＋formal cells、264份P/R capture；132對重播、11,970衍生artifact bytes一致。歷史S1的3,545檔重新核對未變。S2 CPU checks176項。這些證明流程與保存資料的一致性，不能證明Reference完整。

合併36配置：GI/P/R各模式108組中87 stable／21 review；實際P_CTX為28 stable／8 review。依本輪探索用途條件，27配置development_usable、9個limited_use；全部strict_reference_verified=false、formal_training_eligible=false。NVTX warning未解；S2有834筆未匹配cudaMemsetAsync（Linear），byte count未知；157對Overlap集中GEMM03，不能據此證實PDL。

特別區分GEMM04：P CUDA event跨程序spread58.89%，實際P_CTX spread3.10%。GI、profiled event elapsed、P_CTX不可互換。各模式不穩不能被CTX穩定覆蓋掉。

- [S1原報告](S1/REPORT_ZH_TW.md)／[S2原報告](S2/REPORT_ZH_TW.md)：原bytes保留，未複製的raw／evidence連結只在來源節點存在；報告內「尚未發布」為當時狀態。
- [36配置表](S2/analysis/summary/CORE36_METRICS.csv)、[P_CTX穩定性](S2/analysis/summary/P_CTX_STABILITY.csv)、[用途限制](S2/analysis/summary/DEVELOPMENT_USE.csv)。
- [RTX節點操作](../O4_I3_CORE36_NODE_RUNBOOK_20261107.md)／[S3分批建議](../O4_I3_S3_STAGING_20261107.md)。

新發布包可供RTX4090／3090使用，但本次沒有在RTX執行GPU實驗；不能宣稱兩節點PASS。結果只屬controlled isolated算子，不是完整forward GI模型完成。
