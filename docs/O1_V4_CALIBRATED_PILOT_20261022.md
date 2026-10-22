# O1 v4 GB10 校準 pilot：2026-10-22

## 狀態與協定

這是獨立發布的 GB10 v4 **方法分支** `o1-gb10-pilot-v4-20261022`，protocol `o1_operator_pilot_v4_calibrated_repetitions_cuda_reset_20261022`。RTX 3090／4090 進行中的測試仍使用 v2 固定 commit `8e6dac845bd6a1162f3beb92d6d4497bc19f0eb3`，依 [v2 手冊](O1_NODE_RUNBOOK.md) 完成。v2、v3、v4 不可當作同一計時協定合併。v4 的程式與 GB10 校準方法已發布，尚未完成三重複 pilot 或正式 P 逐算子標籤。

v3 的 BMM P/R 五組時長低於 100 ms。v4 從已完成的 GB10 v2 G0/GE 三重複及 v3 P 一次 pilot 的**各組最短 host 時長**校準。按 `ceil(8192 × 125 ms / 最短時長 / 256) × 256` 固定每組呼叫數；P/R 同配置共用 P 的次數，R 另外量測。上限 16,384 次。GB10 校準檔為 [`o1_operator/calibration_gb10_20261022.json`](../o1_operator/calibration_gb10_20261022.json)，SHA-256 `48d9d803a18b13a7cea16c03ae77e721fff506b2d2a5385c30150280683be3d1`，內含來源 session、plan/CSV hash、每配置次數及 `expect_gpu=GB10`。runner 拒絕 GPU 預期不符，並把校準檔複製至 session。

| 配置 | G0 | GE | P=R |
| --- | ---: | ---: | ---: |
| gemm_64_64_64 | 10,496 | 5,888 | 11,520 |
| gemm_256_1024_256 | 8,704 | 5,888 | 15,360 |
| gemm_1024_1024_1024 | 3,328 | 2,816 | 3,840 |
| bmm_2_32_64_64 | 11,264 | 6,144 | 13,568 |
| bmm_8_64_128_64 | 11,008 | 6,144 | 13,056 |
| bmm_4_256_256_256 | 7,936 | 5,376 | 13,312 |
| conv_1_16_32_32_k3_s1_p1 | 6,400 | 4,864 | 6,912 |
| conv_4_64_64_56_k1_s1_p0 | 8,192 | 5,632 | 14,080 |
| conv_1_64_128_56_k3_s2_p1 | 5,376 | 4,352 | 6,912 |

## 實作與驗證

v4 延續 v3 的 P/R worker 收尾：五組量測同步後，先對該獨立 worker 程序執行 `cudaDeviceReset()`，再執行 `cudaProfilerStop()`。R trace 稽核新增預期 scope ID 全集合檢查，每個 ID 形如 `o1.op/gXX/iYYYYYY`；隨後以 scope 包含的 CPU CUDA launch API 與 correlation ID 映射 kernel。已記錄路徑相同只表示兩份 capture 中可見事件相同。GPU 正確性、五組原始時長、Nsight SQLite 診斷仍各自保存。

GB10 九配置 × G0/GE/P/R × 一重複原始 session：`artifacts/runs/o1_gb10_v4_nine_config_smoke_20261022_v1/`。分析：`analysis_runs/o1_gb10_v4_nine_config_smoke_20261022_v1/`，其中 `group_duration_audit.csv`、`pair_event_audit.csv` 及 `audit_provenance.json` 保存獨立稽核。該 session 使用 v4 早期程式 hash `dde6631333a4be57d225209960f57744ffb6aeb6f345b9d8fcb50edddfd0a16d`、初版校準檔 hash `e2a146a33f6d5dbdd92327d4b6ae75476efe3a158c13732f839e5e11dbd1a301`；初版檔保留於該分析目錄。測得 36/36 run 完成且正確性通過，180/180 組至少 100 ms，最短 123.366 ms。九份 P 無 Nsight 警告；九份 R 的 scope ID 數量與全集合、各 scope 到 kernel 的唯一映射、P/R 已記錄 kernel 路徑均通過，但九份 R 各有 `Not all NVTX events might have been collected.`。因此九個配對均是 `manual_review_required_event_loss`。

最終程式／GB10 校準檔再做 BMM P/R 一次短測：`artifacts/runs/o1_gb10_v4_final_bmm_pr_smoke_20261022_v2/`，分析 `analysis_runs/o1_gb10_v4_final_bmm_pr_smoke_20261022_v2/`。其程式 hash `276a75eb4e43225fe248cceb8fadf591cd3fffb16609c29d67d20507e45b54c2`。2/2 run 完成、正確性通過，各 65,280 次呼叫；P 最短組 121.593 ms、無警告，R 最短組 135.497 ms、scope ID 全集合通過，但 NVTX 警告仍在。第一次在受限 sandbox 中試跑的輸出 `artifacts/runs/o1_gb10_v4_final_bmm_pr_smoke_20261022_v1/` 因 Nsight 無法開本機 socket 而失敗，已保留失敗紀錄；v2 短測在一般使用者身分、允許 Nsight 所需本機 socket 的環境成功。

執行 41 個 Python 單元／離線測試，40 個通過、1 個可選 matplotlib 測試略過；`git diff --check`、Python 編譯通過。測試涵蓋 P/R 次數一致、異 GPU 校準檔拒絕、scope ID 缺失拒絕。一次重複不足以評估三次 run spread，因此 `aggregate.csv` 尚無時間聚合。

## NVTX 警告診斷與界線

額外試過 R500 的完整程序 capture、獨立 NVTX domain、root 執行與 NVTX capture range；來源分別保存在 `artifacts/runs/o1_gb10_v3_r500_fullprocess_diag_20261022_v1/`、`o1_gb10_nvtx_domain_diag_20261022_v1/`、`o1_gb10_v3_root_r500_nvtx_diag_20261022_v1/`、`o1_gb10_v3_nvtx_capture_r500_diag_20261022_v1/`。root 執行是使用者在終端輸入 sudo 密碼，並把 `/tmp/gb10_o1_v3_root_r500_nvtx_diag_20261022_v1` 原始結果複製至 artifacts。該次 CLI 沒印警告，但 SQLite 的目標 PID 仍有 NVTX 事件可能未收齊警告；trace SHA-256 `bc4369a23fac33b25b65d970f6f500dfd0bfa9ee731dbb4e854774cd02e227ac`。完整程序與獨立 domain 也未消除警告。NVTX capture range 試作沒有生成 `.nsys-rep`，保留為失敗診斷。這些實驗不屬於正式 O1。

不能憑 root 指令輸出無警告或預期 scope 數齊全，推定 Nsight 的事件完整性。根據 [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)，capture 收尾方式有明確控制；本專案的收尾變更消除了 v2 的 CUDA 警告，但 GB10 R 的 NVTX 警告仍未解決。現階段不把 P `events_allowlist.jsonl` 當正式逐算子訓練標籤，也不擴收九配置 × 三重複作正式 pilot。其他節點的 trace 品質須各自審核。

## GB10 重現

在本候選 checkout、CUDA 可見且已安裝 Nsight Systems 的 GB10 上，輸出目錄須全新：

```sh
python -m o1_operator calibrate \
  --g0ge-root /home/good_egg/RTchen/artifacts/runs/o1_gb10_g0ge_fixed_v2_20261021 \
  --p-root /home/good_egg/RTchen/artifacts/runs/o1_gb10_v3_reset_smoke_20261022_v1 \
  --p-root /home/good_egg/RTchen/artifacts/runs/o1_gb10_v3_reset_family_smoke_20261022_v1 \
  --p-root /home/good_egg/RTchen/artifacts/runs/o1_gb10_v3_p_calibration_20261022_v1 \
  --out /path/to/new_calibration_manifest.json
python -m o1_operator run --out /path/to/new_gb10_pr_session \
  --expect-gpu GB10 --configs bmm_8_64_128_64 --modes P R --repeats 1 \
  --calibration-manifest o1_operator/calibration_gb10_20261022.json
python -m o1_operator summarize --root /path/to/new_gb10_pr_session \
  --out /path/to/new_gb10_pr_analysis
```

## RTX 3090／4090 如何採用同一校準方法

先完成各節點既有的 v2 session，不更動或重新標記原始資料。另建 v4 checkout 並記錄實際 commit：

```sh
git clone https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-o1-v4
cd gpu-prediction-o1-v4
git checkout --detach origin/o1-gb10-pilot-v4-20261022
git rev-parse HEAD
python -m unittest discover -s tests -v
```

v4 校準需要**該節點**九配置、每組 8192 次的已完成 G0／GE 與 P 來源。G0／GE 可用 v2 的九配置三重複 session；P 需九配置至少一次、正確性通過、P trace 沒有 Nsight 警告。若現有 v2 P 只有三配置，先用獨立 v2 或 v3 checkout 補齊九配置 P source，保留原始 session 和 code hash。此分支中的 v3 CUDA 收尾版本提交為 `3a8f966`，其 P 模式仍每組 8192 次；是否適用須以當地 trace 審查為準。來源警告或 GPU 預期字串不一致時，校準命令會失敗，不能借用 GB10 的 JSON 來跳過。

```sh
python -m o1_operator calibrate \
  --g0ge-root /path/to/this_node_v2_g0ge_session \
  --p-root /path/to/this_node_completed_nine_config_p_session \
  --out /path/to/new_this_node_v4_calibration.json
python -m o1_operator run --out /path/to/new_this_node_v4_smoke \
  --expect-gpu 3090 --configs gemm_64_64_64 bmm_8_64_128_64 conv_1_64_128_56_k3_s2_p1 \
  --modes G0 GE P R --repeats 1 \
  --calibration-manifest /path/to/new_this_node_v4_calibration.json
python -m o1_operator summarize --root /path/to/new_this_node_v4_smoke \
  --out /path/to/new_this_node_v4_analysis
```

RTX 4090 將 `--expect-gpu` 改為 `4090`。先審查五組時長、正確性、P/R 呼叫數、Nsight SQLite diagnostics 和 `rp_alignment.csv`；有警告時保留 `manual_review_required_event_loss`，不擴收三重複或建立正式 P 標籤。v4 各節點需保存各自的校準檔 SHA-256、來源 session 和 measurement code hash；不同協定的時間結果分開報告。
