# 驗證狀態

原始提交完成程式與合成資料驗證。後續在 GB10 已取得實測；3090／4090／5090 仍未量測。GB10 trace 的事件完整性仍待審。

已執行：

- 原始提交有 16 項 unittest 通過；本版加入時鐘偏差與明確丟失事件檢查後，18 項測試通過。測試包含合成 Nsight SQLite 的關聯、分段、缺失事件與 schema 檢查。
- 區間聯集測試確認重疊 kernel 不會被重複計算。
- 跨程序 correlation ID 衝突與同程序模糊關聯測試。
- Leave-one-batch-out 檢查確認保留配置的目標倍率不會用於預測。
- Matplotlib 五類圖表以合成資料實際產生。
- CLI dry-run 產生 12 個唯一 run 的隨機化計畫；沒有進行 GPU 推論。
- doctor 正確回報本機 RTX 3060 Laptop，以及此 Python 環境缺少 PyTorch 和 Nsight Systems。

原始軟體驗證環境為 Windows、Python 3.12；圖表驗證使用隔離環境的 Matplotlib 3.11.2。合成測試資料只存在暫存目錄，並非研究結果。

GB10 實測環境為 Linux aarch64、PyTorch 2.13.0+cu130、TorchVision 0.28.0+cu130、Nsight Systems 2026.4.1。B1/8/32 共 36 個 run 的 execution／worker quality 通過，18 個 trace 的 parser 通過；原始 session 未提交 Git。所有 18 個 trace 均有 Nsight「Not all CUDA events might have been collected」警告，9 個 P1 trace 另有 NVTX 同類警告；沒有明確的不完整 CUPTI 事件丟棄數。這些警告仍需以原始 report 審查，不將 parser 通過寫成事件完整性已證實。

仍需在實驗設備確認：

1. 3090／4090／5090 的 CUDA kernel 支援、FP16 forward 正確性與同協定量測。
2. 四設備 Nsight 版本／trace 方法的可比性；GB10 警告的進一步人工審查。
3. 跨設備倍率與泛化；GB10 單機資料不能支持這些結論。

新設備最初執行仍請依 README 先跑 batch 1 的 G 與 P1 模式，再執行 12-run pilot。
