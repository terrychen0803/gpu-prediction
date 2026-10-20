# E2 節點操作手冊：RTX 3090／RTX 4090

這份手冊給兩台節點上的操作人員或本地 Codex 執行。E2 量的是各 GPU 的 Compute、Memory、Launch、Host／transfer 校正曲線，**不是重跑 E1 ResNet50 推論**；不需要 E1 的固定輸入或模型權重。GB10 已完成 `e2_calibration_v3`，RTX 3090／4090 尚待各自在本機實測。5090 目前不在這次工作範圍。

本次以 GitHub 上的 `e2_calibration/` 原始碼為基準。GB10 已量測版本的 `kernels.cu` SHA-256 是 `48bd47e6e7fa7ccebebcf17ff98c34981961cf81bf989b21126477a1e4b32083`，`kernels.cu` 加 `__main__.py` 的連續 bytes SHA-256 是 `29c3878d0a9559c250851cb8f8e5f8632e87b0818ba2717bf63f86a7c067a705`。節點的兩個值應一致；各節點編譯後的 `libe2.so` hash、GPU UUID、`software_hash` 可以不同，並須保留實際值。

## 1. 取得程式與檢查本機環境

在**要量測的節點本機**操作。若尚無 checkout：

```sh
git clone https://github.com/terrychen0803/gpu-prediction.git
cd gpu-prediction
```

若已有 checkout，先在該目錄確認未提交工作；有修改時請先自行保存，不要 `reset --hard` 或刪除原資料。接著：

```sh
git fetch origin
git switch --detach origin/main
git rev-parse HEAD
git status --short --branch
python3 --version
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
nvcc --version
nsys --version
```

記錄實際 commit。需 Python 3.10+、能使用目標 GPU 的 CUDA 版 PyTorch 與 `nvcc`。`nsys` 只用於**正式量測之後的短 kernel 路徑驗證**。先使用節點既有、已驗證的 Python 環境；若需安裝套件，選用與該節點 driver、GPU 架構相容的 CUDA 版 PyTorch。不要把 GB10 的 aarch64 wheel 直接裝到 x86 節點。E2 不依賴 TorchVision、E1 權重或固定輸入。沒有 `nvcc`、CUDA PyTorch 或 GPU 權限時停止並保留錯誤紀錄，不能改用 CPU 結果。

在同一 Python 環境檢查：

```sh
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA PyTorch 未就緒'
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, getattr(p, 'uuid', 'unavailable'), torch.cuda.get_device_capability(i))
PY
sha256sum e2_calibration/kernels.cu
python - <<'PY'
import hashlib
from pathlib import Path
h = hashlib.sha256()
for name in ('e2_calibration/kernels.cu', 'e2_calibration/__main__.py'):
    h.update(Path(name).read_bytes())
print('E2 code_sha256:', h.hexdigest())
PY
```

若兩個來源 hash 與本頁開頭不同，先查明版本，**不要將該結果宣稱與 GB10 是同一版 E2 協定**。Git commit、來源 hash、PyTorch／CUDA／nvcc／driver 版本都要記錄。

## 2. 選定單張 GPU

3090 使用 `E2_LABEL=rtx3090`、`E2_EXPECT=3090`；4090 使用 `E2_LABEL=rtx4090`、`E2_EXPECT=4090`。下列命令在同一個 shell 執行，以免環境變數遺失：

```sh
E2_LABEL=rtx3090
E2_EXPECT=3090
# RTX 4090 節點改成：E2_LABEL=rtx4090; E2_EXPECT=4090
```

多卡節點請從 `nvidia-smi -L` 選一張**沒有其他工作負載**的目標卡，再設 `CUDA_VISIBLE_DEVICES` 為它的完整 GPU UUID。不要猜測 `fastfetch` 的 GPU 編號與 CUDA 編號相同。下面是格式示意，`<實際 GPU UUID>` 必須換成查到的值：

```sh
export CUDA_VISIBLE_DEVICES='<實際 GPU UUID>'
python - <<'PY'
import torch
assert torch.cuda.device_count() == 1, '應只看到一張 CUDA GPU'
p = torch.cuda.get_device_properties(0)
print('選定 GPU:', p.name, 'UUID:', getattr(p, 'uuid', 'unavailable'))
PY
```

單卡節點可不設 `CUDA_VISIBLE_DEVICES`，但仍核對實際裝置名稱／UUID。後續均使用可見的 `cuda:0`；程式的 `--expect-gpu` 也會核對名稱。若沒有獨佔空檔或 GPU 記憶體不足，先協調使用時間，保留失敗紀錄，不更換工作集大小來「湊齊」協定。

## 3. Smoke，再收正式 100 cell

每次輸出目錄都用新的名稱；程式會拒絕覆寫。以下範例可在兩台節點共用：

```sh
E2_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
E2_SMOKE="results/e2_${E2_LABEL}_smoke_${E2_STAMP}"
E2_FULL="results/e2_${E2_LABEL}_full_${E2_STAMP}"
python -m e2_calibration --label "$E2_LABEL" --expect-gpu "$E2_EXPECT" --device cuda:0 --smoke --out "$E2_SMOKE"
```

先檢查 `"$E2_SMOKE/plan.json"` 的 `status=completed`、`smoke=true`，以及 `run01/environment.json` 的 GPU 名稱與 UUID。Smoke 只有五格、一個程序、一群，**不能納入正式曲線**。若失敗，看 `plan.json` 和 `run01.log`；保留失敗目錄，用新目錄重試。

確認無其他 GPU 工作後，採集前後裝置快照並執行正式收集。E2 程式要求 `--out` **事先不存在**，所以先把採集前快照放在結果目錄同層，量完再複製進去：

```sh
nvidia-smi -q > "${E2_FULL}_nvidia_smi_before.txt"
python -m e2_calibration --label "$E2_LABEL" --expect-gpu "$E2_EXPECT" --device cuda:0 --out "$E2_FULL"
nvidia-smi -q > "$E2_FULL/nvidia_smi_after.txt"
cp "${E2_FULL}_nvidia_smi_before.txt" "$E2_FULL/nvidia_smi_before.txt"
python -m e2_calibration.summarize "$E2_FULL"
```

**不要同時啟動 E1、其他 benchmark 或 Nsight**。正式 E2 是 100 cell × 三個獨立程序 × 每格五群，即 1,500 筆原始計時。每群目標約 100 ms，最低 80 ms。Compute 9 格、Memory 42 格、Launch 16 格、Host／transfer 33 格。主程式會寫 `plan.json`、三份 `runNN/environment.json`、`runNN/microbench_samples.csv` 與 `runNN.log`；彙整後才有 `microbench_samples.csv`、`curves.csv`、`hardware_profile.json`。正式量測**不使用 profiler**。如果正式命令失敗，保留該完整目錄，先讀 `plan.json`、`runNN.log` 和 `runNN/environment.json`，在新目錄重做；不可把部分成功資料與重跑資料拼接。

## 4. 驗收正式資料

在同一 shell 用以下檢查。`hardware_profile.json` 的 `status=completed` 表示沒有不足 80 ms 的群組；`completed_with_duration_warnings` 需要另外調查，不能直接當作相同品質。`process_spread_fraction` 是三程序 `(max−min)/median`；>10% 只作**診斷標記**，不是自動刪除資料的門檻。

```sh
E2_FULL="$E2_FULL" python - <<'PY'
import csv, json, os
from pathlib import Path
p = Path(os.environ['E2_FULL'])
plan = json.loads((p / 'plan.json').read_text())
profile = json.loads((p / 'hardware_profile.json').read_text())
assert plan['protocol'] == 'e2_calibration_v3'
assert plan['status'] == 'completed' and plan['smoke'] is False
assert plan['processes'] == 3 and plan['groups_per_cell'] == 5
assert profile['status'] == 'completed' and not profile['short_duration_groups']
assert profile['cells'] == 100 and profile['sample_rows'] == 1500
assert plan['code_sha256'] == '29c3878d0a9559c250851cb8f8e5f8632e87b0818ba2717bf63f86a7c067a705'
with (p / 'curves.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 100
for i in range(1, 4):
    env = json.loads((p / f'run{i:02d}' / 'environment.json').read_text())
    assert env['status'] == 'completed' and env['sample_rows'] == 500
    assert env['gpu_uuid'] == profile['gpu_uuid']
outliers = [(r['cell'], round(float(r['process_spread_fraction']) * 100, 2))
            for r in rows if float(r['process_spread_fraction']) > 0.10]
print('E2 formal structure passed; GPU:', profile['gpu_name'], profile['gpu_uuid'])
print('profile_id:', profile['profile_id'], 'spread >10%:', outliers)
PY
```

兩台節點的 cell、來源 hash、計時政策應相同；PyTorch、CUDA toolkit、driver、`nvcc` 版本若不同，保留並標示，先核對可比性。`curves.csv` 的有效 FLOP/s 和邏輯 byte/s 不是 GPU 型錄峰值、實際 DRAM 流量或 cache hit rate。`launch` 主要以包含 Python／ctypes 開銷的 host completion 計時。`pageable` transfer event span 可能包含 host 阻塞；GB10 的傳輸路徑不能直接按離散 GPU PCIe 解讀。

## 5. 獨立的 Nsight kernel 路徑驗證

正式收集及彙整**完成後**，另以極短 trace 確認 native kernel 確實出現在 GPU 軌跡，不拿這份 trace 當作 100 cell 的計時。先看 `nsys profile --help` 是否支援 `cuda-sw`；若不支援，記下版本與錯誤，以新檔名改試 `--trace=cuda`，並把不同 backend 分開標示。

```sh
nsys profile --trace=cuda-sw --sample=none --output="$E2_FULL/kernel_validation" \
  python -m e2_calibration.trace_smoke --library "$E2_FULL/libe2.so"
nsys export --type=sqlite --output="$E2_FULL/kernel_validation.sqlite" \
  "$E2_FULL/kernel_validation.nsys-rep"
```

標準輸出應包含 `trace_smoke_passed empty=10 short=10 copy=10`。在 GUI 的 Diagnostics Summary 或匯出的 SQLite 核對 `e2_empty_kernel`、`e2_short_kernel`、`e2_copy_kernel` 各有 10 筆；若 Nsight 警告事件可能遺失，照實記錄，不能只因 trace 有圖就宣稱完整。若 SQLite schema 不同，檢查表與欄位後再查詢，不猜測。短 trace 失敗不應修改已完成的正式樣本；可在新 `kernel_validation_retry...` 前綴補做並保留第一次錯誤。

GB10 使用的 Nsight SQLite schema 有 `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` 與 `StringIds`；節點若相同，可直接檢查三種 kernel 各 10 筆：

```sh
E2_FULL="$E2_FULL" python - <<'PY'
import os, sqlite3
from pathlib import Path
p = Path(os.environ['E2_FULL']) / 'kernel_validation.sqlite'
with sqlite3.connect(p) as db:
    rows = db.execute('''
        SELECT s.value, COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON k.demangledName = s.id
        WHERE s.value LIKE 'e2_%'
        GROUP BY s.value
    ''').fetchall()
counts = {name.split('(')[0]: count for name, count in rows}
print(counts)
expected = {'e2_empty_kernel': 10, 'e2_short_kernel': 10, 'e2_copy_kernel': 10}
assert all(counts.get(name) == n for name, n in expected.items()), '短 trace kernel 數量不符'
PY
```

查詢錯誤或筆數不符時，先查看 `sqlite_master`／`PRAGMA table_info`、Nsight 診斷與 trace backend；不要把缺少的事件補成 0 或假設整份正式矩陣已獲得事件完整性證明。

## 6. 完整封存與交接

先將人工判讀記到結果目錄外的交接筆記，包括：節點、Git commit、GPU UUID、所有版本、smoke 路徑、正式結果路徑、100／1,500／short groups 檢查、spread >10% 的 cell、Nsight backend／警告與任何重跑原因。原始結果、失敗目錄及 `.nsys-rep` **不要提交 GitHub**。

在確定不會再改動結果目錄後，對每個檔案建立並驗證 SHA-256，製作壓縮檔：

```sh
(cd "$E2_FULL" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$E2_FULL" && sha256sum -c SHA256SUMS)
E2_ARCHIVE="results/${E2_LABEL}_e2_${E2_STAMP}.tar.gz"
tar -czf "$E2_ARCHIVE" -C results "$(basename "$E2_FULL")"
sha256sum "$E2_ARCHIVE" > "$E2_ARCHIVE.sha256"
sha256sum -c "$E2_ARCHIVE.sha256"
```

將 `.tar.gz`、`.tar.gz.sha256` 及交接筆記以節點之間已核准的傳輸方式帶回 GB10 的 `/home/good_egg/RTchen/`，並提供當地路徑供整合分析。封存前先檢查筆記、log 與快照不含登入資訊或憑證。保留節點原件，接收端先驗證外層與內層 checksum，再做跨設備彙整。若結果太大，可仍先保留完整原件，再另做小型摘要供溝通；不要用摘要代替原始資料。

## 可直接交給節點本地 Codex 的任務

> 請在這台 RTX 3090（若是 4090 則改為 RTX 4090）節點上，閱讀 `AGENTS.md`、`docs/E2_NODE_RUNBOOK.md` 與 `e2_calibration/README.md`，按手冊完成 `e2_calibration_v3`。先盤點 GPU、driver、Python、CUDA PyTorch、`nvcc`、Nsight 和 Git commit，核對 E2 來源 hash，選一張空閒的目標 GPU；先跑 smoke，通過後用新目錄跑正式 100 cell／三程序／五群，執行 summarize 與品質檢查，最後做獨立的 Nsight kernel 短 trace。保存每次失敗與原始資料；不要修改協定、覆寫結果或將量測資料提交 Git。將完整目錄附 SHA-256 封存，回報壓縮檔及 checksum 的絕對路徑、Git commit、GPU UUID、版本、驗收結果、異常 cell 與 Nsight 警告。缺少必要環境或節點有其他 GPU 工作時，停止正式量測並回報實際阻礙。
