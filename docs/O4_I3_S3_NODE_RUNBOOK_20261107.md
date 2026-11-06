# S3-lite RTX4090／RTX3090 節點手冊

固定tag：`o4-i3-s3-lite-40x3-dev-v1-20261107`。本tag含S1/S2與S3摘要；新Release附件只含相同source及S3的36配置資產，S1/S2資產仍從Core36 release取得。已完成Core36者不需因換tag重收；source165檔bytes相同。

```bash
git clone --branch o4-i3-s3-lite-40x3-dev-v1-20261107 --depth 1 https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-s3
cd gpu-prediction-s3
mkdir i3-s3-downloads
gh release download o4-i3-s3-lite-40x3-dev-v1-20261107 --repo terrychen0803/gpu-prediction --dir i3-s3-downloads
(cd i3-s3-downloads && sha256sum -c SHA256SUMS)
mkdir i3-s3-assets
tar -xzf i3-s3-downloads/o4_i3_S3_lite_assets_v1_20261107.tar.gz -C i3-s3-assets
export I3_REPO="$PWD"
export I3_PYTHON=/absolute/path/to/compatible/cuda-python
export I3_GPU=4090  # 3090節點改3090
export I3_OUT=/absolute/new/path/S3_node_run
export PYTHONPATH="$I3_REPO/i3_config_pilot/source:$I3_REPO/i3_config_pilot/source/o4_pilot"
"$I3_PYTHON" -m o4_i3_v2.assets verify --path "$I3_REPO/i3-s3-assets/assets_S3/ASSETS.json"
```

相容CUDA PyTorch與nsys需由當地既有環境提供，不自動升級系統。固定input/weight bytes直接下載，不要各自重新seed後冒稱相同資產。source附件與Git source相同，不必重複解包執行。

依 [Core36手冊CPU預檢](O4_I3_CORE36_NODE_RUNBOOK_20261107.md#cpu預檢) 跑四組tests後，確認沒有其他使用者GPU compute工作。每cell guard與5秒monitor保留，不改現有K3s／Pod／Netdata／RustDesk。記錄當地source/commit、Python/torch/CUDA、driver、nsys、GPU owner及資產hash。

## 三批plan與Smoke

以下命令都是plan-only，沒有GPU執行。核對後移除plan-only，out換成 `$I3_OUT/collections/<batch>_smoke` 新路徑，才實際跑Smoke。每批最多12配置，不要36個ID一次提交。

### train_controls（train）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-s3-assets/assets_S3/ASSETS.json" \
  --config I3_conv_00_control1 I3_conv_00_control2 I3_conv_00_kernel1 I3_conv_00_groups4 I3_depthwise_00_control1 I3_gemm_00_control1 I3_bmm_00_control1 I3_linear_00_control1 I3_batchnorm_eval_00_control2 I3_add_00_control1 I3_bilinear_resize_00_control1 I3_concat_00_control2 \
  --out "$I3_OUT/plans/train_controls_smoke" --plan-only
```

### validation_01（validation）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-s3-assets/assets_S3/ASSETS.json" \
  --config I3_conv_01_base I3_depthwise_01_base I3_gemm_01_base I3_bmm_01_base I3_linear_01_base I3_batchnorm_eval_01_base I3_relu_01_base I3_add_01_base I3_maxpool_01_base I3_adaptive_average_pool_01_base I3_bilinear_resize_01_base I3_concat_01_base \
  --out "$I3_OUT/plans/validation_01_smoke" --plan-only
```

### sealed_test_02（test_exploratory）

```bash
"$I3_PYTHON" -m o4_i3_v2.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$I3_REPO/i3-s3-assets/assets_S3/ASSETS.json" \
  --config I3_conv_02_base I3_depthwise_02_base I3_gemm_02_base I3_bmm_02_base I3_linear_02_base I3_batchnorm_eval_02_base I3_relu_02_base I3_add_02_base I3_maxpool_02_base I3_adaptive_average_pool_02_base I3_bilinear_resize_02_base I3_concat_02_base \
  --out "$I3_OUT/plans/sealed_test_02_smoke" --plan-only
```

## Smoke放行與40×3

每配置GI/P/R三cell都須completed，mode不可重複、repeat=1；correctness/warmup與Trace engineering gate通過，monitor_review=false，且唯一P/R pair的measured_exact=3、all_windows_exact=true，才進formal。GPU guard暫停就先等外部工作結束，不能關掉外部程序。失敗與排除原因保留，新out補跑；其他獨立配置可繼續。

通過後用同資產與「實際通過的」config清單，將kind改formal，先用新out `$I3_OUT/plans/<batch>_formal` 加plan-only確認9cells/config；再新out `$I3_OUT/collections/<batch>_formal` 移除plan-only實收。每mode三個獨立程序、每程序40 measured；不能用40×1替代。三process spread>5%標review，不刪慢值或挑最快repeat。GI、P event、R event與實際P_CTX分開檢查。

每formal配置應有三對P/R、各40個measured exact，並核對完整ATen bound參數、ownership、missing work／NVTX／clock等品質。Exact signature、Scope數或GI穩定不能單獨證明Reference完整。Memset byte count未知仍unverified；overlap不裁切、不平移timestamp、不直接宣稱PDL。

GB10本輪觀察到的限制只是參考，不可抄成RTX判定。預期cuda-sw；實際backend不同／不支援要保存原因，不silent fallback。各GPU不用相同kernel名字／數量，跨設備評估才按方向取可比工作／資產／scope／label交集。

## split與test封存

S1/S2原36配置是train；S3的12 controls仍與anchor00同train leakage_group，不能算12個新獨立group。Anchor01是validation、02是test_exploratory。完整清單見 `docs/o4_i3_s3_lite_20261107/SPLIT_MANIFEST.json`；同配置跨設備／mode／repeat／日期仍同群。

在test前以已完成train/validation的R完整參數稽核同配置與跨split碰撞，忽略device／seed／bytes等不能代表新穎性的差異。若碰撞阻礙獨立切分，保存原manifest與raw，另作版本化group決策，不用模型test分數搬群。近鄰與數學等價仍可能需人工審查。已有歷史DNN/A0曝光不能被這個exact稽核排除。

Test只做correctness、穩定性、覆蓋／signature等資料品質檢查，先封存，不算模型test分數、挑特徵或調參。它是exploratory split，每家族只有一個test配置，不能當正式確認集。這是evaluator用途隔離，未加密。R真實類型／shape與target test trace/GI不進predictor；來源P numeric allowlist不變。

## 回傳

保留每批完整out，含raw .nsys-rep、SQLite、source_snapshot、environment、實際命令、protocol、quality、monitor、iterations與所有失敗；勿含憑證。三批放同一新run的collections/下，使用相同source執行：

```bash
"$I3_PYTHON" -m o4_i3_v2.report --run "$I3_OUT" --out "$I3_OUT/analysis/replay_v1" --replay
```

每次analysis用新out，禁止覆寫raw。這個report提供實際P_CTX標籤，需另按config/repeat聚合三process及spread，不能拿P event spread冒充P_CTX。本輪Git只含整理報告；完整raw保留各節點再交回。GB10專用run_s3.py含本機Python路徑，未作RTX入口發布。

本release沒有Context Adapter；不要拿目前o4_i3_v2 CLI執行S4 DAG。辨識器／N0／ITER未訓練。所有Reference仍unverified；warning-accepted development與formal_training_eligible分開。
