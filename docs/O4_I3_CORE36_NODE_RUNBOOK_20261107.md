# RTX4090／RTX3090：Core36 40×3 節點手冊

使用固定tag `o4-i3-core36-40x3-dev-v1-20261107`。這是development prerelease，不是verified reference。不同節點可先收可執行子集；kernel名稱／數量不用相同。請保留版本差異、失敗與raw，再核對可比交集。

## 取得程式與固定資產

```bash
git clone --branch o4-i3-core36-40x3-dev-v1-20261107 --depth 1 https://github.com/terrychen0803/gpu-prediction.git
cd gpu-prediction
```

下方在repo根目錄執行。需要已安裝相容CUDA PyTorch的Python及Nsight `nsys`，不自動安裝／升級系統工具。

```bash
mkdir i3-core36-downloads
gh release download o4-i3-core36-40x3-dev-v1-20261107 --repo terrychen0803/gpu-prediction --dir i3-core36-downloads
(cd i3-core36-downloads && sha256sum -c SHA256SUMS)
mkdir i3-core36-assets
tar -xzf i3-core36-downloads/o4_i3_S1_assets_v1_20261107.tar.gz -C i3-core36-assets
tar -xzf i3-core36-downloads/o4_i3_S2_assets_v1_20261107.tar.gz -C i3-core36-assets
export I3_REPO="$PWD"
export I3_PYTHON=/absolute/path/to/your/cuda-python
export I3_GPU=4090  # RTX3090節點改3090
export PYTHONPATH="$I3_REPO/i3_config_pilot/source:$I3_REPO/i3_config_pilot/source/o4_pilot"
export I3_OUT=/absolute/new/path/core36_node_run
"$I3_PYTHON" -m o4_i3_v2.assets verify --path "$I3_REPO/i3-core36-assets/assets_S1/ASSETS.json"
"$I3_PYTHON" -m o4_i3_v2.assets verify --path "$I3_REPO/i3-core36-assets/assets_S2/ASSETS.json"
```

固定input/weight bytes直接下載；不要重新seed產生後宣稱相同資產。source bundle與Git中的165檔相同，無須兩份一起執行。SHA清單見 `i3_config_pilot/manifests/`。`source/I3_V2_RUNBOOK_ZH_TW.md` 是S1歷史快照，當前發布狀態以本手冊為準。

## CPU預檢

```bash
cd "$I3_REPO/i3_config_pilot/source"
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests_v2 -v
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s o4_pilot/tests -v
(cd o4_pilot/vendor/legacy && CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests -v)
```

確認GPU沒有其他使用者的compute工作，不改既有服務。collector有逐cell guard與5秒monitor；採樣不能保證間隔無干擾。新out不可覆寫；失敗保留原因再選獨立新out，不自動降精度／改shape／backend。

## 四批收集

以下都是plan-only，不啟動GPU。核對後，使用 `$I3_OUT/collections/<batch>_smoke` 新out並移除 `--plan-only` 執行Smoke。每批最多12配置。

### S1（3配置）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-core36-assets/assets_S1/ASSETS.json" \
  --config I3_batchnorm_eval_00_base I3_gemm_00_base I3_bilinear_resize_00_base \
  --out "$I3_OUT/plans/S1_smoke" --plan-only
```

### anchor_00（9配置）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-core36-assets/assets_S2/ASSETS.json" \
  --config I3_conv_00_base I3_depthwise_00_base I3_bmm_00_base I3_linear_00_base I3_relu_00_base I3_add_00_base I3_maxpool_00_base I3_adaptive_average_pool_00_base I3_concat_00_base \
  --out "$I3_OUT/plans/anchor_00_smoke" --plan-only
```

### anchor_03（12配置）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-core36-assets/assets_S2/ASSETS.json" \
  --config I3_conv_03_base I3_depthwise_03_base I3_gemm_03_base I3_bmm_03_base I3_linear_03_base I3_batchnorm_eval_03_base I3_relu_03_base I3_add_03_base I3_maxpool_03_base I3_adaptive_average_pool_03_base I3_bilinear_resize_03_base I3_concat_03_base \
  --out "$I3_OUT/plans/anchor_03_smoke" --plan-only
```

### anchor_04（12配置）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-core36-assets/assets_S2/ASSETS.json" \
  --config I3_conv_04_base I3_depthwise_04_base I3_gemm_04_base I3_bmm_04_base I3_linear_04_base I3_batchnorm_eval_04_base I3_relu_04_base I3_add_04_base I3_maxpool_04_base I3_adaptive_average_pool_04_base I3_bilinear_resize_04_base I3_concat_04_base \
  --out "$I3_OUT/plans/anchor_04_smoke" --plan-only
```

每批Smoke後檢查inventory.json：每配置GI/P/R三cell皆completed、monitor_review為false、唯一P/R pair的measured_exact=3且all_windows_exact=true；同時核對correctness／warmup及engineering gate。只有通過的配置才可進formal。未通過者從下一命令config清單移除、保存原因，其他獨立配置可繼續。GPU guard暫停應先處理外部工作，不能關閉它。

通過後，沿用該批實際通過的config和資產路徑，改 `--kind formal`、新out `$I3_OUT/collections/<batch>_formal`，先plan-only（新out `$I3_OUT/plans/<batch>_formal`）確認每配置9cells，再另新out移除plan-only實收40×3。三個mode各三次獨立程序；不要因只想縮短時間改成40×1。

Formal後每配置各mode三process中位數spread≤5%才標該mode stable；超過保留review，不能刪慢repeat。核對每對P/R有40個measured exact、runtime ATen bound args與預期配置、ownership及submitted-work缺項；不要只看ordered signature就verified。

collector依賴明確cuda-sw；若RTX實際backend不同或不支援，保存實際backend與完整錯誤，另外建立對照，不默默fallback。硬體端不要求相同kernel實作。尚未確認memset byte count不能判合法0B。NVTX／Clock／Overlap未解不改時間戳或裁事件。

## 交回與分析

保留每批完整out（protocol、CODE_SHA256、environment、commands、assets manifest、inventory、iterations、quality、monitor、raw .nsys-rep／SQLite、audit及失敗）。另存 `git rev-parse HEAD`、`nsys --version`、實際Python/torch/CUDA與driver資訊、GPU owner預檢。不得提交憑證。

把每批放在獨立節點run的 `collections/` 下，使用新out執行 `python -m o4_i3_v2.report --run NODE_RUN --out NEW_ANALYSIS --replay`。此報告提供重播與P_CTX標籤；profiled event穩定性不是P_CTX穩定性，聚合後另核對實際P_CTX。GB10本輪專用彙總腳本依賴原節點目錄，未當作RTX工具發布；RTX可先交回逐批原始out與重播結果再彙整。

不啟動辨識器、N0或ITER訓練；當前36配置全為train，不能作未見配置成績。R真實類型／shape只供evaluator，predictor只讀P numeric allowlist；test目標trace/GI不輸入predictor。Reference全部維持unverified，development_usable也不等於formal_training_eligible。
