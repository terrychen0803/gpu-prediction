# Native software 控制：統一版 v3

請由 repository 根目錄依 [統一 Quickstart](../../docs/O4_QUICKSTART_ZH_TW.md) 的 `o4_node.sh native` 執行。每個子程序前核對GPU忙碌與owner，未知工作不放行，不終止任何既有程序。

只執行correctness與cuda-sw A1/A2；schema3.28/3.31結構檢查、known work驗收、獨立diagnostic gate。已知NVTX warning可工程暫接受，CUDA完整性/drop或backend不明須停止並保留資料。新v3未GPU實測，歷史GB10/3090結果不回填。

原版操作與hash保存在HISTORICAL_*和BUNDLE_MANIFEST_v2_historical.json。不可複製ARM binary到RTX節點，須當地編譯。
