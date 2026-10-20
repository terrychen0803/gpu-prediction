# 驗證狀態（2026-10-20）

本輪 E1 的唯一共用輸入基準是倉庫 [inputs/](../inputs/README.md) 的 FP16 檔案。GB10 已以這份固定輸入完成 B1／B8／B32 × G／A／P0／P1 × 三次獨立程序，共 36 個正式 run。每個 run 的 `input_sha256` 與 sidecar 一致；36／36 execution 和 worker quality 通過，三組 G 的 run spread 低於 5%。18／18 trace 可解析，9 份 P1 的 phase kernel 數與 duration 覆蓋率最低均為 100%。GB10 詳細數據見 [現況摘要](../docs/GB10_E1_FIXED_INPUT_20261020.md)。

18 份 trace 全有 Nsight「Not all CUDA events might have been collected」警告，9 份 P1 另有 NVTX 同類警告。`trace_quality.json` 保留 `manual_review_required`；parser 通過不等於事件零遺失，E1 trace 完整性尚未驗收。

3090／4090 各有一輪 36 個本機收到的真實 run，但當時以各節點 CPU 生成輸入，與此共用檔案的 bytes 不同。它們可用於各設備自身描述分析，不與新版 GB10 計算正式同輸入倍率；須以此檔案及同一程式 hash 重新量測。5090 暫不可用。

目前 20 項 `python -m unittest discover -s tests -v` 通過，包含合成 Nsight SQLite 關聯、分段與缺失事件檢查、跨設備資料切分、固定輸入檔損壞拒絕及新舊輸入協定隔離。這些離線測試不能替代 GPU 實測。GB10 的真實實驗環境為 Linux aarch64、PyTorch 2.13.0+cu130、TorchVision 0.28.0+cu130、Nsight Systems 2026.4.1，P0／P1 使用 `cuda-sw`。

其他節點須先執行 doctor，再做 B1 G／P1 smoke，確認模型輸出、輸入與程式 hash、Nsight 後端和 trace schema，最後收正式 session。GPU trace 警告與不同軟體版本需單獨記錄；不放寬 E1 驗收門檻。
