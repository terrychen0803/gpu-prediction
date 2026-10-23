# O2 混合序列收集：GB10／RTX 3090／RTX 4090 節點手冊

本手冊對應 GitHub 分支 `o2-mixed-sequence-method-20261024`、協定
`o2_mixed_sequence_gb10_v3_chain_norm_checked_20261024`。協定名稱記錄首個
實測節點 GB10；3090／4090 使用**同一份程式與模板**，透過 `--expect-gpu`
驗證實際設備。任何運算、同步、模板或 parser 政策變更，都需要新的協定 ID，
不可把不同節點的資料混成 v3。這份分支只發布方法，不含原始 trace、輸入
bytes、環境檔或本機分析結果。

## 1. 固定版本與環境

請使用新的 clone 或 worktree，先保護各節點已有的未提交工作。以下命令都在
repository 根目錄執行；記下 `git rev-parse HEAD`，三節點須固定同一提交。
需要 Python 3.10+、符合本機 GPU 的 CUDA PyTorch、`nsys`、`nvidia-smi`。
本 CLI 要求恰好一張 CUDA 可見 GPU；多卡節點先以
`CUDA_VISIBLE_DEVICES` 選定設備，再留存實際 GPU 名稱／版本於當地 raw run。

```sh
git clone --branch o2-mixed-sequence-method-20261024 \
  https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-o2
cd gpu-prediction-o2
git rev-parse HEAD
python -m unittest discover -s tests -v
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
nsys --version
nvidia-smi -L
python -c 'from o2_sequence.runner import code_hash; print(code_hash())'
```

GB10 已實測的 `code_hash()` 是
`d9ca6fdd0442f386629c2b5ab459dd6f4162a72277907ed205ad16fb63524f1f`。
`plan.json` 另保存 frozen catalog SHA-256
`3696763ee19d886146bec1c5a3f054b0d94c8e86ce2a225c921630b23787d0b4`
與 O1 split SHA-256
`96b41d65914e152a7573b8a2cba1eb627c2ddc1fb71ca4186b1bb8022efe384d`。
跨設備收集前先比對這三項；輸入格式固定
`o1_shake256_lut_fp16_le_v1`，執行後再對比每個 `input_inventory.json` 的
operand SHA-256。輸入由 SHAKE256+FP16 LUT 產生，不使用各機隨機數實作。

## 2. 先跑小型 pilot

下面以 RTX 3090 為例；RTX 4090 設 `GPU_TAG=4090`，GB10 設
`GPU_TAG=GB10`。請把輸出放在節點本地資料區，且每個 `--out` 都用**不存在的
新目錄**。先跑一個 16-step 獨立模板；`--dry-run` 只產生計畫，不能算 GPU
實測。

```sh
GPU_TAG=3090
NODE_ID=rtx3090
python -m o2_sequence run --matrix pilot --template pilot_independent \
  --repeats 1 --expect-gpu "$GPU_TAG" --dry-run \
  --out "../o2-runs/${NODE_ID}_pilot_plan_001"
python -m o2_sequence run --matrix pilot --template pilot_independent \
  --repeats 1 --expect-gpu "$GPU_TAG" \
  --out "../o2-runs/${NODE_ID}_pilot_independent_001"
python -m o2_sequence review \
  --root "../o2-runs/${NODE_ID}_pilot_independent_001" \
  --out "../o2-review/${NODE_ID}_pilot_independent_001"
```

查看 `plan.json`、每個 `execution.json`／`quality.json`、P/R 的
`trace_audit.json`、`profile.log`、`export.log`，以及 review 的
`summary.json`。三個模式必須真正完成；P 不得有 O2 逐步 NVTX scope；R
的 scope 清單與 CUDA API→GPU kernel correlation 必須完整。CUDA 記錄或
SQLite schema 在其他節點可能不同，失敗時保留 raw log，不能改用 CPU
結果或補造 kernel。警告不會阻止**收集**，但必須使標籤維持待審；
`exact_ordered_signature` 不等於無事件遺失證明。

再跑含 tensor 依賴尾段的完整 pilot（兩模板 × 三模式，共 6 個程序），
檢查 CPU FP32→FP16 參考值、P/R 路徑及 Nsight 警告：

```sh
python -m o2_sequence run --matrix pilot --repeats 1 \
  --expect-gpu "$GPU_TAG" --out "../o2-runs/${NODE_ID}_pilot_full_001"
python -m o2_sequence review \
  --root "../o2-runs/${NODE_ID}_pilot_full_001" \
  --out "../o2-review/${NODE_ID}_pilot_full_001"
```

若 pilot 的執行或結構 audit 失敗，先保存該目錄並診斷，不把它算作通過；
修正收集／解析政策須改協定、建立新目錄並重新 pilot。若僅有已知的 NVTX
警告，可繼續收集，但不可把 P 逐步標籤升格為已驗證。

## 3. 正式 60 模板

pilot 執行與結構檢查通過後，執行凍結的 60 模板 × G0/P/R × 3 repeats，
每節點 **540 個獨立程序**，並對 180 組 P/R 做 evaluator-only 審查：

```sh
python -m o2_sequence run --matrix formal --repeats 3 \
  --expect-gpu "$GPU_TAG" --out "../o2-runs/${NODE_ID}_formal60_001"
python -m o2_sequence review \
  --root "../o2-runs/${NODE_ID}_formal60_001" \
  --out "../o2-review/${NODE_ID}_formal60_001"
```

`plan.json` 需列出 60 模板、5 folds 各 12 模板、540 jobs，並保存實際
source/catalog/split hash。每個 job 是 5 groups × 8 完整序列；獨立模板
16 steps，依賴尾段模板 20 steps；G0／P／R 都不在逐步邊界同步。G0 是
整序列 batch-amortized host 時間，不能混稱單算子 GE。P 只有無標記 CUDA
runtime/kernel trace；R 的 NVTX 類型、配置與 scope 只供 evaluator。

`review` 只在**完整有序** kernel signature、grid/block、shared memory、
stream 與 R GPU event 順序一致時，建立候選 R→P 對應；不按絕對 timestamp
或相近 duration 猜標籤。任一 P/R 路徑不同、事件缺失或警告，應保存
`unverifiable` 並留在 coverage 分母。R 去標記資料只是
sanitized-reference，不是真正 P 的部署證據。若本地 profiler
權限不足，記錄命令、錯誤與設備條件；不要將憑證寫入 artifact。

## 4. GB10 參考結果與回傳

GB10 的 v3 收集：540/540 完成、180/180 P/R 完整記錄序列一致、
129,600 個 R logical steps；但 180 個 R trace 都有
`Not all NVTX events might have been collected.`，故嚴格可驗證 P
逐步標籤為 **0/129,600**。3090／4090 應各自檢查，不套用 GB10 的警告
結論。GB10 正式 raw run／完整分析報告留在 GB10 本機，不在 GitHub。

節點回傳可公開的 commit、protocol、三項 hash、input inventory 的
checksum 對照、GPU／PyTorch／CUDA／driver／Nsight 版本、planned/completed/
failed 數、P/R event 與 scope 數、路徑一致率、警告、可驗證 coverage 及
G0/R/P 描述統計。完整 raw `.nsys-rep`、SQLite、輸入 bytes、GPU UUID、
PID／路徑等只留在本地；需要中央重審時另行傳輸，先檢查敏感欄位。

實作細節、輸出 schema 與已知限制見 [O2 README](../o2_sequence/README.md)。
