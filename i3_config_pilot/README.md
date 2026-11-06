# I3 Core36 40×3 開發交付

本目錄的 `source/` 是 GB10 S1/S2 實際執行的165檔原始bytes；舊 `i3_pilot/` A0和舊release不變。

請依 [新節點手冊](../docs/O4_I3_CORE36_NODE_RUNBOOK_20261107.md) 下載兩套固定資產，先CPU tests與plan，再逐批Smoke／formal。`source/I3_V2_RUNBOOK_ZH_TW.md` 是S1當時快照，其「僅3配置／未發布」描述保留歷史狀態；目前支援與指令以上層新手冊為準。Catalog有174配置，不代表174個已GPU驗證。Context adapter尚未完成。

40×3是工程收集協定；Reference全部仍未verified。來源P numeric allowlist、真值R及目標test資料隔離規則不變。不需所有節點跑齊；跨設備只配可比交集。原始trace、模型與results不進Git。

後續[S3固定資產與節點指令](../docs/O4_I3_S3_NODE_RUNBOOK_20261107.md)使用完全相同source；S3已完成GB10 GPU實測。S4尚未實作，見[分批建議](../docs/O4_I3_S4_STAGING_20261107.md)。
