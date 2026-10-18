# 驗證狀態

本版已完成程式實作與本機軟體驗證，尚未完成目標 GPU 的實測。

已執行：

- 16 項 unittest 通過，包含合成 Nsight SQLite 的關聯、分段、缺失事件與 schema 檢查。
- 區間聯集測試確認重疊 kernel 不會被重複計算。
- 跨程序 correlation ID 衝突與同程序模糊關聯測試。
- Leave-one-batch-out 檢查確認保留配置的目標倍率不會用於預測。
- Matplotlib 五類圖表以合成資料實際產生。
- CLI dry-run 產生 12 個唯一 run 的隨機化計畫；沒有進行 GPU 推論。
- doctor 正確回報本機 RTX 3060 Laptop，以及此 Python 環境缺少 PyTorch 和 Nsight Systems。

驗證環境：Windows、Python 3.12；圖表驗證使用專案內隔離目錄的 Matplotlib 3.11.2。合成測試資料只存在暫存目錄，並非研究結果。

仍需在實驗設備確認：

1. CUDA 版框架對 GB10／4090／5090 的 kernel 支援，以及 FP16 forward 正確性。
2. 當地 Nsight CLI 的參數支援、實際 SQLite schema 與資料完整性。
3. 無 profiler 的時間穩定性、標記與 profiler 的實測差異。
4. 實際 trace 的 phase 覆蓋率，以及 profiler 診斷是否存在事件遺失。

最初執行請依 README 先跑 batch 1 的 G 與 P1 模式，再執行 12-run pilot。
