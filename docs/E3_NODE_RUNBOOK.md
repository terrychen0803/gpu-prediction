# E3 節點手冊：GB10／RTX 3090／RTX 4090

> 本文件保留 E3 的凍結命令與 hash，供既有量測重現。2026-10-21 起新增研究的優先順序依 [O 系列設計](NEXT_EXPERIMENTS.md) 及 [節點交接](CODEX_HANDOFF.md)；不將既有六 phase 當算子標籤。

這份手冊供三台節點的本地 Codex 各自在本機執行。E3 量測 ResNet50 的 batch×resolution 配置，收集無 profiler 的 **G** 標籤及來源設備的 **P0** 數值 trace。每台目標矩陣是 resolution={160,224,320} × batch={1,4,8,32} × mode={G,P0} × 三個獨立程序，即 **72 個正式 run**；三台合計 216 run。E2 已收完的硬體曲線另存，不需重跑。5090 尚未可用，不阻礙前三台的設備內 E3。

本輪 E3 使用新量測程式與新輸入協定。**舊 E1 的 224²、torch.randn 固定檔不能混入 E3 grid**：新檔以純 Python SHAKE256 byte stream 和固定 FP16 lookup table 產生，三台按同一 manifest 可驗出完全相同的 bytes。這是控制計算工作量的合成輸入，並非真實影像準確率測試。資料不在 Git，產生程式與 [三解析度 hash manifest](../inputs/e3_grid_manifest.json) 在 Git。正式程式 `inference_pilot/*.py` 合併 hash 預期為 `7743298b9e32fc40b1467174bd193d124e30fd00adcbe4ef626b8214a19b1eb9`；任何修改都須另立協定，原結果保留。

## 1. 節點準備

在當地 `gpu-prediction` checkout 操作。先保存未提交工作，不用 `git reset --hard` 清資料，再取得 GitHub 最新 main：

```sh
git fetch origin
git switch --detach origin/main
git rev-parse HEAD
git status --short --branch
python -m unittest discover -s tests -v
python -m inference_pilot prepare-grid-set
python - <<'PY'
from inference_pilot.common import code_hash
expected = '7743298b9e32fc40b1467174bd193d124e30fd00adcbe4ef626b8214a19b1eb9'
actual = code_hash()
print('inference_pilot code hash:', actual)
assert actual == expected
PY
nvidia-smi -L
```

記下 `git rev-parse HEAD` 的實際 commit；以量測模組 hash、輸入／權重 hash 鎖定資料契約。E3 量測程式最初發布於 `402239b57aa0e6134e7bafd47529c9fc2c9ce4bc`，後續文件修訂不改變上方預期量測模組 hash。`prepare-grid-set` 產生 `inputs/e3_grid_v1/` 下三份 `.bin` 與 `.bin.json`，若已存在則驗證，不覆寫；它逐一核對 Git 中的全檔與 B1／B4／B8／B32 前綴 SHA-256。產生的 160／224／320 檔案全檔 hash 依序為 `5bd2e54bc60cdb718ba82b6fd39acffc2ad6dbef8778047bce2c5e98c2382af1`、`2f10ae79f0fc26dbb51349c60dff8951ea462bba2f1fdaee6e2184eb09c7c64a`、`9326162bd95cca3253f57bac2836d9d41f3cb0a65be4e1eb74774aff256bbbc8`。不一致即停止，查明版本與檔案，不改檢查值。正式權重仍使用 E1 的 `assets/resnet50_imagenet1k_v2.pth` 及 sidecar；檔案 SHA-256 應為 `72f27a9ddb2f3c57258816e525b12701cb0216f0e17192b780111c45ca8a8597`。

使用節點既有且已驗證的 CUDA PyTorch／TorchVision 與 Nsight CLI。多卡節點先用 `nvidia-smi -L` 取得目標 GPU 完整 UUID，再設 `CUDA_VISIBLE_DEVICES=<實際 GPU UUID>`；不要依 fastfetch 編號猜 CUDA index。正式收集期間選沒有其他重負載的單張卡，後續指令均用 `cuda:0`。檢查 `python -m inference_pilot doctor --out <新檔案>`，記錄 GPU UUID、driver、torch、CUDA、cuDNN、nsys 版本與 Git commit。GB10 已驗證 `cuda-sw`；3090／4090 的新版 P0 先由 smoke 確認當地 Nsight 支援。不同 backend 必須分組記錄。

## 2. 先在兩端點 smoke

在 3090 節點設 `E3_LABEL=rtx3090`、`E3_EXPECT=3090`；4090 設 `rtx4090`／`4090`；GB10 設 `gb10`／`GB10`。以下命令在同一 shell，輸出目錄名每次換新：

```sh
set -euo pipefail
E3_LABEL=rtx3090
E3_EXPECT=3090
E3_SMOKE="results/e3_${E3_LABEL}_smoke_001"
python -m inference_pilot run --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --mode G --batch 1 --resolution 160 --input-file inputs/e3_grid_v1/resnet50_fp16_b32_r160_seed42.bin --out "$E3_SMOKE/r160_b1_G"
python -m inference_pilot run --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --mode P0 --batch 1 --resolution 160 --input-file inputs/e3_grid_v1/resnet50_fp16_b32_r160_seed42.bin --nsys-cuda-trace cuda-sw --out "$E3_SMOKE/r160_b1_P0"
python -m inference_pilot run --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --mode G --batch 32 --resolution 320 --input-file inputs/e3_grid_v1/resnet50_fp16_b32_r320_seed42.bin --out "$E3_SMOKE/r320_b32_G"
python -m inference_pilot run --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --mode P0 --batch 32 --resolution 320 --input-file inputs/e3_grid_v1/resnet50_fp16_b32_r320_seed42.bin --nsys-cuda-trace cuda-sw --out "$E3_SMOKE/r320_b32_P0"
```

逐 run 核對 `execution.json=completed`、`quality.json=passed`、`correctness.passed=true`、`manifest.json` 的 batch／resolution／`input_scheme=e3_shake256_lut_fp16_le_nchw_v1`／code hash。P0 還要 `trace_quality.json=passed` 且 `features.csv` 行數等於 measured iteration。查看 `trace_diagnostics.json` 與 Nsight 警告；parser 通過不表示無事件遺失。記錄警告，不把疑似遺失事件補零。GB10 的兩個端點 smoke 已通過執行、正確性與 parser，但 P0 仍有「可能未完整收集 CUDA 事件」警告；正式 E3 特徵目前為探索性。

## 3. 正式 72 run／節點

**每個 resolution 一個新 suite 目錄**，因一份固定輸入檔只對應一個解析度。建議先各做 `--dry-run` 看 24 run 排程；`planned_only` 不算實測。正式命令範例：

```sh
set -euo pipefail
for E3_RES in 160 224 320; do
  python -m inference_pilot suite \
    --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --device cuda:0 \
    --modes G P0 --batches 1 4 8 32 --resolutions "$E3_RES" --repeats 3 \
    --input-file "inputs/e3_grid_v1/resnet50_fp16_b32_r${E3_RES}_seed42.bin" \
    --nsys-cuda-trace cuda-sw \
    --out "results/e3_${E3_LABEL}_r${E3_RES}_session01"
done
```

現有 session 不覆寫。若某個 suite 失敗，保留其完整失敗目錄與 log；以新的輸出目錄重做失敗配置，不把部分成功和重跑樣本默默拼接。`--keep-going` 只允許在保留失敗狀態的前提下收集其他格。若 B32/R320 OOM，標記 infeasible，後續比較採三台共同可行子集，不在單台私自改 batch 或精度。最小與最大配置各另做一個 P1 稽核，查看 phase coverage 與 NVTX/CUDA 警告；它們另存，不進 E3 的 G/P0 主矩陣：

```sh
E3_P1="results/e3_${E3_LABEL}_p1_audit01"
python -m inference_pilot run --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --mode P1 --batch 1 --resolution 160 --input-file inputs/e3_grid_v1/resnet50_fp16_b32_r160_seed42.bin --nsys-cuda-trace cuda-sw --out "$E3_P1/r160_b1_P1"
python -m inference_pilot run --label "$E3_LABEL" --expect-gpu "$E3_EXPECT" --mode P1 --batch 32 --resolution 320 --input-file inputs/e3_grid_v1/resnet50_fp16_b32_r320_seed42.bin --nsys-cuda-trace cuda-sw --out "$E3_P1/r320_b32_P1"
```

驗收時每台應有 12 個配置，各 G×3、P0×3，合計 72 個 completed run；每 run 的 correctness 通過、輸入前綴與權重／程式 hash 一致、G 的三程序中位數 spread ≤5%。P0 的 parser 必須通過，且警告要逐 run 留存；現有工具無法從「可能未收齊」警告證明事件無損。若正式資料仍有警告，E3 模型分析明確標為探索性，另做 trace 品質敏感度分析。

## 4. 節點本地 pilot 與交回資料

三個正式 suite 都完成後，可在本地跑**只含該設備**的 E3 C0/C1 流程檢查：

```sh
python -m e3_calibration --out "results/e3_${E3_LABEL}_analysis01" \
  "results/e3_${E3_LABEL}_r160_session01" \
  "results/e3_${E3_LABEL}_r224_session01" \
  "results/e3_${E3_LABEL}_r320_session01"
```

`source_features.csv` 與 `g_labels_evaluator_only.csv` 分開；`split_manifest.json` 留出完整 configuration，而非個別 iteration；`quality.json` 會保留 trace 警告。此時即使有 12 個配置，也不要在單節點依 pilot 的 C0/C1 誤差直接宣稱 E3 已正式成功。跨三設備整合與品質審查須在 GB10 收到三台原始 session 後進行；C2 多特徵正則化仍是後續實作。

封存三個正式 suite、smoke、P1 稽核、當地環境／log、分析輸出及每個檔案的 SHA-256 清單，另對 `.tar.gz` 建外層 `.sha256`。保留原件並將封存及 checksum 以已核准的方式帶回 `/home/good_egg/RTchen/`，提供兩個檔案的絕對路徑。原始 trace、權重及量測結果不提交 GitHub；不要把其他節點的資料放進本地 Codex 的 predictor 輸入中。

## 可貼給節點本地 Codex 的任務

> 請先讀 `AGENTS.md`、GitHub `main` 最新的 `docs/E3_NODE_RUNBOOK.md`、`e3_calibration/README.md` 與 `inputs/e3_grid_manifest.json`。在本節點只選一張空閒 RTX 3090／RTX 4090，記錄 Git commit、使用已驗證的 CUDA PyTorch 與 Nsight。執行 `prepare-grid-set` 並核對三解析度 SHA-256、權重及 `inference_pilot` code hash 必須等於 `7743298b9e32fc40b1467174bd193d124e30fd00adcbe4ef626b8214a19b1eb9`；先做 R160/B1 和 R320/B32 的 G/P0 smoke，檢查 correctness、trace parser 和警告，再按手冊以新目錄完成三個 resolution 的 G/P0 各三次，共 72 個正式 run；另做兩個 P1 端點稽核。保留所有失敗與警告，不改 batch、精度、同步政策、輸入或門檻。完成後以 E3 CLI 做設備內 pilot，核對 12 配置／72 run、G spread 與 P0 品質，將完整原始目錄附 checksum 封存；回報封存檔與 `.sha256` 的絕對路徑、Git commit、GPU UUID、版本、完成／失敗格與每個 trace 警告。若環境不足、目標 GPU 忙碌或最大配置不可行，先回報實際阻礙並保留資料。
