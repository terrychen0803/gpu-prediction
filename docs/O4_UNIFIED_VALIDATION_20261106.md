# 統一工程候選版：離線驗證與限制

日期：2026-11-06，Asia/Taipei。Release `o4-unified-engineering-v1-20261106`；新七 A protocol v2。這份發行沒有新增GPU實測或模型訓練。

| 狀態 | 證據 |
|---|---|
| 已完成工程實作 | 共用node入口、完整14檔資產／O4分發來源驗證、版本snapshot、附加v5工程gate、native v3相容入口 |
| 已完成離線測試 | O4/研究78、vendored legacy54、root O1/O2整合65、portable native14，共211項通過 |
| 已完成既有raw重解析 | GB10七A14份：188個v4輸出逐byte一致，raw SHA未變；新增acceptance均warning-accepted，strict0；另讀舊native A1/A2成功 |
| 已完成規劃驗證 | 同一凍結資產14檔核對；七A計畫63cells、GPU executions=0 |
| 僅有規劃／未執行 | 新v2七A40×3與native v3 GPU canary，GB10/3090/4090均未執行；Windows live pending |
| Reference未驗證 | strict verified=0；NVTX、clock、unmatched API等既有資格不解除；無正式辨識器/N0/ITER成績 |

45-window合成SQLite反例涵蓋CUDA dropped diagnostics、API失敗、geometry缺失與已知NVTX例外；合成fixture不混入實驗資料。新增檢查也驗backend缺失／衝突、query not-ready例外、共同依賴source變動、缺warmup與重複iteration ID拒絕。單元測試不證明新GPU協定已成功。

歷史o4_iteration collector、vendor、protocol/assets manifest與原release v1/v2均保留原bytes；新版修改的是工程collector/overlay/分發驗證。根目錄O1/O2方法從已凍結來源整合，與O4 vendor隔離。

重解析來源仍是GB10 schema3.28資料；沒有RTX3090 schema3.31 raw可供本次獨立重解析，3090先前成績依其節點交付文件。新增native支援schema3.31的程式尚需3090當地驗收。native CUDA完整性警告若被新工程gate阻擋，代表新acceptance結論，舊完成狀態不改寫。

全套本機證據在GB10獨立run `o4_unified_release_20261106T150745Z`。Git只包含程式、測試、協定、hash及整理文件；raw、模型權重、環境值與失敗log留節點。
