# 驗證狀態（2026-10-20）

本輪 E1 的唯一共用輸入基準是倉庫 [inputs/](../inputs/README.md) 的 FP16 檔案。GB10 已以這份固定輸入完成 B1／B8／B32 × G／A／P0／P1 × 三次獨立程序，共 36 個正式 run。每個 run 的 `input_sha256` 與 sidecar 一致；36／36 execution 和 worker quality 通過，三組 G 的 run spread 低於 5%。18／18 trace 可解析，9 份 P1 的 phase kernel 數與 duration 覆蓋率最低均為 100%。GB10 詳細數據見 [現況摘要](../docs/GB10_E1_FIXED_INPUT_20261020.md)。

18 份 trace 全有 Nsight「Not all CUDA events might have been collected」警告，9 份 P1 另有 NVTX 同類警告。`trace_quality.json` 保留 `manual_review_required`；parser 通過不等於事件零遺失，E1 trace 完整性尚未驗收。

3090／4090 已另用相同固定輸入各收 36 個新版 run；本機整合稽核為 108/108 正式 run 通過、54 份 trace 待事件完整性人工審查。舊版 CPU 生成輸入的資料仍只作歷史來源，不與新版資料混合。5090 暫不可用。

目前 26 項 `python -m unittest discover -s tests -v` 通過、1 項需選裝 matplotlib 的圖表測試跳過，包含合成 Nsight SQLite 關聯、分段與缺失事件檢查、跨設備資料切分、固定輸入檔損壞拒絕、新舊輸入協定隔離、E3 三解析度固定檔再生成／hash 驗證、同解析度分組及留出 G 標籤不可洩漏到預測。這些離線測試不能替代 GPU 實測。GB10 的真實實驗環境為 Linux aarch64、PyTorch 2.13.0+cu130、TorchVision 0.28.0+cu130、Nsight Systems 2026.4.1，P0／P1 使用 `cuda-sw`。

E3 新輸入的 R160/B1 與 R320/B32 已各在 GB10 做 G／P0 smoke：四個 run 的 execution、worker quality、reference 正確性通過，兩份 P0 parser 通過；P0 仍出現 CUDA 事件可能未完整收集警告。新 input generator 在兩個獨立輸出目錄重新產生完整三解析度檔，均與 Git manifest hash 一致。此 smoke 不計入正式 E3 grid；完整收集與跨節點比對狀態請以 `docs/E3_NODE_RUNBOOK.md` 和本機分析目錄為準。

其他節點須先執行 doctor，再做 B1 G／P1 smoke，確認模型輸出、輸入與程式 hash、Nsight 後端和 trace schema，最後收正式 session。GPU trace 警告與不同軟體版本需單獨記錄；不放寬 E1 驗收門檻。
