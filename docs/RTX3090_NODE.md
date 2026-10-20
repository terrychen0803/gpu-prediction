# RTX 3090 正式矩陣節點操作

RTX 3090 已納入 GB10／3090／4090／5090 的正式比較矩陣，並已使用倉庫內 [固定輸入](../inputs/README.md)完成第一階段 ResNet50 E1 的 36 個正式 run。本文件保留可重做的操作步驟；E2 硬體校正則依 [E2 工具說明](../e2_calibration/README.md)執行。舊版 CPU 隨機輸入 session 不納入新版倍率。

## 1. 取得同一份程式並盤點節點

```sh
git clone https://github.com/terrychen0803/gpu-prediction.git
cd gpu-prediction
git rev-parse HEAD
git status --short --branch
uname -m
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
python3 --version
nsys --version
```

若節點已有 checkout，先保存未提交工作，再執行 `git pull --ff-only`；不使用 `git reset` 清除資料。使用 Python 3.10+、與 RTX 3090 驅動相容的 CUDA 版 PyTorch／TorchVision，並以 `nsys profile --help` 確認當地 Nsight CLI 支援的參數。套件安裝方法見 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/)；記錄實際版本，不假定與 GB10 的 aarch64 build 相同。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
# 依 PyTorch 官方安裝頁，將與本機 driver 相容的 torch/torchvision 安裝到此環境。
python -m unittest discover -s tests -v
python -m inference_pilot doctor --out doctor_3090.json
```

`doctor` 應列出預期的 CUDA GPU 與 `ready_for_G_A`／`ready_for_P0_P1`；若未找到 Nsight，先處理 CLI 安裝，再跑 P0/P1。`doctor` 不代替真實 kernel 試跑。若節點有多張 GPU，先依 `nvidia-smi -L` 選定單張，固定 `CUDA_VISIBLE_DEVICES`（可用該 GPU UUID），之後所有指令都用同一環境及 `--device cuda:0`；檢查每個 `environment.json` 的實際 GPU UUID 一致。不要用標籤推測實際裝置。

## 2. 固定權重與 trace 後端

從 GB10 另行複製 `resnet50_imagenet1k_v2.pth` **和** `resnet50_imagenet1k_v2.pth.json` 到 `assets/`，或使用下列命令準備同一版權重：

```sh
python -m inference_pilot prepare --out assets/resnet50_imagenet1k_v2.pth
sha256sum assets/resnet50_imagenet1k_v2.pth
```

SHA-256 必須是 `72f27a9ddb2f3c57258816e525b12701cb0216f0e17192b780111c45ca8a8597`，且 sidecar JSON 必須同行；不符時先停止並查明來源。權重與原始 trace 不在 Git。

輸入檔已在 Git 的 `inputs/e1_shared_fp16_b32_seed42_20261020.bin`，不需要從 GB10 額外複製，也不要在 3090 上重新生成。先執行 `sha256sum inputs/e1_shared_fp16_b32_seed42_20261020.bin`，必須等於 `0082e5f18717dcc379d22cf44250c8497c3f8ccc41763301ee8f67d7cfe7b152`；每次 `run`／`suite` 都使用 `--input-file`。

GB10 的 P0/P1 使用 `cuda-sw`。先查看當地 `nsys profile --help`，再用下節的 P1 smoke test 驗證 RTX 3090 上是否支援 `--nsys-cuda-trace cuda-sw`。若不支援，可在**新的輸出目錄**改試預設 `cuda`，保存失敗與版本紀錄；把兩種 trace 後端當作不同收集條件，先做可比性分析，不直接混合 trace 特徵。G/A 不受此選項影響。

## 3. B1 單次試跑

以下假設選定 GPU 在目前環境中是 `cuda:0`。每次使用新的輸出目錄；失敗 run 留存供排查。

```sh
python -m inference_pilot run --label rtx3090 --expect-gpu 3090 --device cuda:0 --mode G --batch 1 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --out results/3090_b1_fixed_g_smoke
python -m inference_pilot run --label rtx3090 --expect-gpu 3090 --device cuda:0 --mode P1 --batch 1 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/3090_b1_fixed_p1_smoke
```

逐一檢查 `execution.json` 為 completed、`quality.json` 為 passed；P1 還要 `trace_quality.json` 為 passed，閱讀 `trace_diagnostics.json` 與 Nsight 警告。`manual_review_required` 不會因 parser 通過自動解除。核對 `manifest.json` 的 `code_sha256` 與 [GB10 固定輸入摘要](GB10_E1_FIXED_INPUT_20261020.md) 一致，B1 `input_sha256` 為 `c6105d609934198afaeaf5a6badca3d8bf057321e31a3de07ed95840a94d29e2`，權重 hash 相同；並記錄 `environment.json` 的 GPU UUID、框架與 CUDA/cuDNN 版本。若 correctness、warm-up、trace 或相容性檢查失敗，先排查再做正式 12 run。

## 4. 正式收集與交接

先以 `--dry-run` 檢查 batch 1 的 12-run 排程；它只產生 `planned_only`，不是 GPU 結果。然後在新目錄執行正式收集。以下命令僅在 P1 的 `cuda-sw` smoke test 通過後使用：

```sh
python -m inference_pilot suite --label rtx3090 --expect-gpu 3090 --device cuda:0 --batches 1 --repeats 3 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/3090_b1_fixed_plan --dry-run
python -m inference_pilot suite --label rtx3090 --expect-gpu 3090 --device cuda:0 --batches 1 --repeats 3 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/3090_b1_fixed_session01
```

確認 12 run 的品質、G run spread 與 trace 狀態後，再用新目錄完成 B8/B32。CLI 可用 `--batches 8 32`；若當地 CLI 語法不同，以 `python -m inference_pilot suite --help` 為準。已收的 B1 不要複製成第二批獨立樣本。

```sh
python -m inference_pilot suite --label rtx3090 --expect-gpu 3090 --device cuda:0 --batches 8 32 --repeats 3 --input-file inputs/e1_shared_fp16_b32_seed42_20261020.bin --nsys-cuda-trace cuda-sw --out results/3090_b8_b32_fixed_session01
```

正式 E1 預計 12 + 24 = 36 run；若某 batch 不可行，保存失敗資料，標記 infeasible，不能默默縮小 batch 或放寬門檻。分享資料時另行傳遞完整 `results/` session 與 SHA-256 清單，記錄 commit、GPU UUID、版本、權重 hash、trace 後端、所有失敗或排除理由。原始 session 不提交 Git。GB10 與 RTX 3090 的主機、框架或 trace 後端不同時，跨設備倍率與 trace 模型結果要先通過協定相容性檢查；未達成時先分組報告。
