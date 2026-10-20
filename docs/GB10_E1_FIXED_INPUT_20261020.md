# GB10 固定輸入 E1 現況（2026-10-20）

目前以本倉庫 [共用輸入檔](../inputs/README.md) 為 E1 的輸入基準。GB10 已完成 ResNet50 FP16 eager、224 × 224、B1／B8／B32、G／A／P0／P1 各三次獨立程序，共 **36 個正式 run**。量測程式 `code_sha256` 為 `28ceb3e58e25d5d24954ece221087ed55f756598dd495df5c560313d444479b8`，模型權重 SHA-256 為 `72f27a9ddb2f3c57258816e525b12701cb0216f0e17192b780111c45ca8a8597`。原始 session、模型權重與逐次量測留在 GB10，不提交 Git。

36／36 execution 與 worker quality 通過，三組 G 的獨立 run spread 均低於 5%；18／18 P0／P1 trace 可解析，P1 的 phase kernel 數與 duration 覆蓋率最低都是 100%。Nsight 在 **18／18 trace** 提示 CUDA 事件可能未完整收集，9 份 P1 還提示 NVTX，同步及解析通過仍不能證明事件零遺失，因此 trace 完整性保留 `manual_review_required`。

| Batch | G 時間：三個 run 中位數的中位數 | G run spread |
|---:|---:|---:|
| 1 | 1,987.316 µs | 0.822% |
| 8 | 6,066.736 µs | 2.650% |
| 32 | 28,203.644 µs | 1.829% |

輸入載入在計時前；每個 iteration 量測 host forward 到 GPU 最後同步，數字是整個 batch，不除以影像數。詳細本機稽核位於 `/home/good_egg/RTchen/gpu-prediction/analysis_runs/gb10_fixed_input_e1_20261020/REPORT.md`，原始 session 位於 `/home/good_egg/RTchen/artifacts/runs/gpu_prediction_gb10_fixed_input_full_20261020/`。

既有 3090／4090 的舊 session 使用不同的輸入 bytes，不能與此 GB10 session 作嚴格同輸入倍率。兩台需要用本倉庫的固定輸入檔和相同 `inference_pilot/*.py` 版本重新量測；先核對 manifest 的 `input_sha256`、`code_sha256`、權重 hash 與框架版本。5090 暫不可用，E1 四設備矩陣尚未完成。
