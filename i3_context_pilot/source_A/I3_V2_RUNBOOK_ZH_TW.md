# I3 新配置 S1：GB10工程收集

版本：o4_i3_config_context_40x3_v1_20261107（context DAG介面尚未實作）。新package `o4_i3_v2`，不修改原A0、O4 parser、v4/v5 gate。新目錄收集，固定asset／source SHA。

目前adapter支援計畫174個isolated配置的12家族，CPU shape／stride／logical-value tests驗證。**本次實際資產與GPU範圍僅BN、GEMM、Resize三個anchor00。** 不因catalog有174筆就稱已測全部。Formal是40×3工程協定，不等於strict reference。

在本source目錄，以既有相容CUDA Python設定：

```bash
export I3_PYTHON=/absolute/path/to/cuda-python
export PYTHONPATH="$PWD:$PWD/o4_pilot"
export I3_ASSETS=/absolute/path/to/assets_S1/ASSETS.json
"$I3_PYTHON" -m o4_i3_v2.assets verify --path "$I3_ASSETS"
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests_v2 -v
```

真正GPU工作前先確認無外部compute owner。collector每cell有guard、每5秒host側唯讀活動sample；若觀察外部工作，保留該cell並停止後續，不殺外部程序。採樣不能證明間隔內沒有其他工作。系統memory欄位在GB10可能N/A，照實記錄，不假裝取得獨立GPU記憶體容量。

新out、三配置、每mode一process的8warm＋1transition＋3measured smoke：

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu GB10 \
  --assets-manifest "$I3_ASSETS" \
  --config I3_batchnorm_eval_00_base I3_gemm_00_base I3_bilinear_resize_00_base \
  --out /new/path/smoke
```

用相同清單先加`--plan-only`產生formal的27cells預覽；實測另用新out，`--kind formal`，每mode三個獨立process，每process暖機50～500（最早60次穩定）＋5transition＋40measured。R的warmup啟用同一dispatch metadata路徑。所有模式stream／精度／assets／prepare一致；GI與profiled event elapsed分開。

freeze source snapshot在啟動第一個worker前完成，實際worker從snapshot執行，每cell前後檢查code及asset。錯誤exit非零，失敗原folder保留，連續同cause兩次則defer該config；不自動改門檻、裁timestamp、抹overlap或選較快repeat。環境錯誤後須另立有原因的新診斷batch。`--resume`僅在無failed cells的暫停情況，驗同protocol/assets/configs並跳過completed；不覆寫failed／interrupted資料。

R只供reference/evaluator；P numeric沿原SAFE allowlist。每iteration保存bound ATen parameters與recorded ownership、API reverse coverage、P/R signature。未知no-op byte、NVTX、clock與semantic未驗證均保留；沒有activity不自動當0us。worker的current-stream檢查不足以獨立證明隱藏stream join，需同配置raw observed-stream審查，無法支援完整範圍時降低用途資格。

穩定性每mode分開，三process median的spread≤5%標stable；超過只標review、不刪值。保存40全樣本、P10/P90/MAD及tail>20%review。只有1process的smoke不評三process spread。

報告與獨立CPU重播（新out，不改raw）：

```bash
"$I3_PYTHON" -m o4_i3_v2.report --run /absolute/path/to/run \
  --out /absolute/path/to/run/analysis/replay_report --replay
```

只作本輪GB10 S0＋S1；其他節點未執行，MLP／N0／ITER沒有訓練。新程式／資產尚未發布GitHub，不可在RTX用舊release假裝支援此protocol。
