# O1 九配置 pilot：RTX 3090／RTX 4090 節點操作

此手冊對應 `o1_operator_pilot_v2_fixed_repetitions_20261021`。GB10 已完成 G0/GE 九配置三重複，以及 P/R 三家族各一次診斷；結果與限制見 [GB10 狀態](O1_GB10_PILOT_STATUS_20261021.md)。其餘節點先用同版程式建立可比較的資料，不把 GB10 的 trace 品質結論套到其他 GPU。O1 是已知單算子 pilot，尚無 O2 無標記混合序列分段或逐算子時間預測模型。

## 1. 取得凍結版本與檢查環境

在**新的 checkout** 操作，以免覆蓋當地 E4 或未提交工作。Git branch `o1-gb10-pilot-v2-20261021` 指向本次量測版本；checkout 後記下實際 commit SHA 與下方量測程式 hash，跨節點均固定同一提交。若先前已有 repo，先保存 `git status` 與當地變更。以下命令在 Linux 節點執行：

```sh
git clone https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-o1
cd gpu-prediction-o1
git checkout --detach origin/o1-gb10-pilot-v2-20261021
git rev-parse HEAD
python -m unittest discover -s tests -v
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
nsys --version
```

需 Python 3.10+、可在目標 GPU 執行的 CUDA PyTorch 與 Nsight Systems CLI。O1 不需 TorchVision、模型權重或管理員 GPU Metrics 權限；P/R profiler 需當地 `nsys` 能運作。`--expect-gpu` 必須是裝置名稱內的字串，例如 `3090` 或 `4090`。程式要求剛好一張 CUDA 可見 GPU；多卡節點先設 `CUDA_VISIBLE_DEVICES`，並把選擇與 GPU UUID、driver、PyTorch、CUDA/cuDNN、Nsight 版本留在 session 環境檔。不同軟體版本需分組報告。

## 2. 先確認固定輸入與程式 hash

輸入由 seed 42 的 SHAKE256 byte stream 和固定 FP16 LUT 生成，不靠各機 `torch.randn`。九配置共 18 個 operand 的預期 SHA-256 放在 [`o1_operator/pilot_input_sha256.json`](../o1_operator/pilot_input_sha256.json)。可在 CPU 上核對，不會啟動 GPU：

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path
from o1_operator.spec import BY_ID, input_bytes, operand_shapes
from o1_operator.runner import code_hash

expected = json.loads(Path('o1_operator/pilot_input_sha256.json').read_text())
assert expected['input_format'] == 'o1_shake256_lut_fp16_le_v1'
assert expected['seed'] == 42
for config_id, operands in expected['operands'].items():
    for name, digest in operands.items():
        shape = operand_shapes(BY_ID[config_id])[name]
        data, _ = input_bytes(config_id, name, shape)
        assert hashlib.sha256(data).hexdigest() == digest, (config_id, name)
print('input hashes: 18/18 matched')
print('measurement code SHA-256:', code_hash())
PY
```

凍結版本的量測程式 hash 應為 `aa513968c3a8cd9f1d9baf5c74c8119cdfdebc78a6f3d0863a8ff56fab064182`。`plan.json` 也會保存它。`input_inventory.json` 保存該次實際生成檔的 SHA-256；跨節點比較前須與上述 manifest 相符。

## 3. 當地 smoke 與正式時間 pilot

下例為 RTX 3090；RTX 4090 換成 `NODE_ID=rtx4090`、`GPU_TAG=4090`。每個 `--out` 必須是**不存在的新目錄**。輸出可放在 repo 外的當地資料碟；原始 trace、環境檔、完整分析目錄不提交 Git。

```sh
NODE_ID=rtx3090
GPU_TAG=3090
python -m o1_operator run --out "../o1-runs/${NODE_ID}_smoke_g0ge_001" --expect-gpu "$GPU_TAG" --configs gemm_64_64_64 --modes G0 GE --repeats 1
python -m o1_operator summarize --root "../o1-runs/${NODE_ID}_smoke_g0ge_001" --out "../o1-analysis/${NODE_ID}_smoke_g0ge_001"
```

先看 `summary.json`、`run_inventory.csv`、每 run 的 `quality.json`、`environment.json`、`groups.csv` 與 `worker.log`。要求 execution 完成、finite／shape／FP32 reference allclose 通過，且 `groups.csv` 保留實際組時長。**smoke 一次重複不產生三重複穩定性結論**。若通過，再收九配置 G0/GE：

```sh
python -m o1_operator run --out "../o1-runs/${NODE_ID}_g0ge_v2_001" --expect-gpu "$GPU_TAG" --modes G0 GE --repeats 3
python -m o1_operator summarize --root "../o1-runs/${NODE_ID}_g0ge_v2_001" --out "../o1-analysis/${NODE_ID}_g0ge_v2_001"
```

預期 54 個獨立程序。每 run 五組，每組在所有模式固定 8,192 次，故 P/R 配對各預期 40,960 個 logical call；三 run spread 候選門檻為 `(max−min)/median ≤ 5%`。程式不會為不同節點或模式自動改呼叫數。`group_target_100ms_met=false`、超過 5% 或 correctness 失敗都應如實保留；不要只重跑較慢或較快的一次來挑好結果。

## 4. P/R trace 品質診斷

G0/GE 通過後，先測 GEMM、BMM、Conv 各一配置的一次 P/R 配對：

```sh
python -m o1_operator run --out "../o1-runs/${NODE_ID}_pr_v2_001" --expect-gpu "$GPU_TAG" --configs gemm_64_64_64 bmm_8_64_128_64 conv_1_64_128_56_k3_s2_p1 --modes P R --repeats 1
python -m o1_operator summarize --root "../o1-runs/${NODE_ID}_pr_v2_001" --out "../o1-analysis/${NODE_ID}_pr_v2_001"
```

檢查 `rp_alignment.csv`、每個 `trace_audit.json`、`profile.log`、`export.log` 與 `trace.sqlite` 的 `DIAGNOSTIC_EVENT`。`recorded_sequence_equal=True` 只說明**已記錄的** kernel 名稱、geometry、順序相同。P 是無逐算子標記的獨立 capture；R 的 NVTX scope 經 CPU CUDA API 與 correlation ID 對應已記錄 kernel。兩個 capture 不按絕對時間對齊，也不以相近 duration 猜逐算子標籤。Nsight 有 CUDA／NVTX 事件可能未收齊警告時，保持 `manual_review_required_event_loss`，不把 P 的 `events_allowlist.jsonl` 當可驗證訓練標籤。

GB10 的六份 P/R trace 全有上述警告，三個配對雖然 recorded path 相同，仍未通過真正 P 的標籤關卡。其他節點可能不同，必須各自審核。若當地 P/R 警告、call 數、路徑或組時長未達要求，先保存診斷並停止擴收到九配置三重複。若品質條件真正滿足，再用**相同 commit 與政策**建立新目錄做完整 `--modes G0 GE P R --repeats 3` pilot；既有 session 不回填。

## 5. 回傳清單與限制

回傳 commit SHA、`plan.json` 的 `code_sha256`、九配置 input SHA 比對、環境版本、`summary.json`、`run_inventory.csv`、`aggregate.csv`、`rp_alignment.csv`、trace 警告與失敗清單。需要中央重審時，另外傳保留 checksum 的原始 session；先檢查 `environment.json` 與 profiler log 是否含私人資訊。原始 `.nsys-rep`／SQLite、GPU UUID、PID 等只留在節點資料區，不推 Git。

本 v2 採每組 8,192 次，是為修正 v1 P/R 自適應呼叫數不同而凍結的 pilot 政策，並不保證每張 GPU 的組時長約 100 ms。若要按設備／配置重新校準次數，必須先共同制定 P/R 一致的新固定政策、換新 protocol ID，不能把 v2 舊資料改標成新版本。PM metrics、正式 115/139 配置、O2 混合序列辨識與 O3 時間模型均未包含在這個 CLI。
