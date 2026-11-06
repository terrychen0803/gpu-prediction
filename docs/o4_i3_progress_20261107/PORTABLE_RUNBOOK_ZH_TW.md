# I3 A0 portable development smoke

版本：`o4_i3_dev_v1_20261107`，協定`o4_i3_A0_smoke_v1_20261107`。這是新增獨立工具，未取代原O4七A collector／歷史品質gate。原Nsight parser與v4/v5程式保持原樣；新R另外保存完整ATen綁定參數與tensor metadata。

所有結果為development，正式Reference與訓練資格維持false。GI是無profiler整個受控單位的CUDA event elapsed，P/R的event elapsed另名；P的CTX另外從GPU事件計算。

## 環境及資產

`source/`包含新`o4_i3/`、原`o4_pilot/`相依來源及新`tests/`。使用節點上相容的Python/PyTorch，記錄實際版本，不要求升級成GB10版本。Linux、單一CUDA-visible GPU；Nsight必須支援明示cuda-sw，沒有silent fallback。

先取得GB10所產生的完整`assets_A0/`，共18個候選配置的固定FP16 operands與ASSETS.json。這些是新的initialized performance-only受控工作，不代表任何DNN的預訓練權重。用SHA比對傳輸；**各節點不重新生成隨機資產**。不同輸入bytes、工作參數或精度須新variant，不能混為同配置配對。

在`source/`下設定（換成節點實際路徑）：

```bash
export I3_PYTHON=/absolute/path/to/python
export I3_ASSETS=/absolute/path/to/assets_A0/ASSETS.json
export PYTHONPATH="$PWD:$PWD/o4_pilot"
"$I3_PYTHON" -m o4_i3.controlled verify --path "$I3_ASSETS"
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s o4_pilot/tests -v
```

在`source/o4_pilot/vendor/legacy/`另執行其tests。所有命令均需看return code；有不支援、資產缺失或schema錯誤就保留紀錄，不改manifest繞過。

## 先列計畫，再小型執行

```bash
"$I3_PYTHON" -m o4_i3.collect \
  --assets-manifest "$I3_ASSETS" --expect-gpu 4090 \
  --config A0_batchnorm_eval_a --out /new/path/bn_plan --plan-only
```

真正執行前確認沒有其他GPU工作；collector會每cell保存GPU guard／owner證據。guard不得停止他人程序。各輸出目錄必須全新。

```bash
"$I3_PYTHON" -m o4_i3.collect \
  --assets-manifest "$I3_ASSETS" --expect-gpu 4090 \
  --config A0_batchnorm_eval_a --out /new/path/bn_smoke
```

3090改`--expect-gpu 3090`，GB10改`GB10`。預設GI/P/R各一個獨立程序、8warmup＋1transition＋3measured。這不是40×3穩定性量測；工具目前只實作smoke，不支援以參數偷偷改成正式矩陣。

BN通過後可依價值選擇`A0_conv_a`、`A0_gemm_a`、`A0_bmm_a`、`A0_linear_a`、`A0_relu_a`、`A0_add_a`、`A0_maxpool_a`、`A0_adaptive_average_pool_a`。`_b`是batch變化的第二候選，未因資產存在就算已收集。不要求各台全跑；配對分析按方向取交集。

每cell保存request、protocol、environment、source hash、asset hash、UTC/monotonic iteration、correctness、GPU guard、命令與失敗；P/R另存原`.nsys-rep`、SQLite、audit、完整diagnostics與R參數。跨設備不要求kernel名稱／數量相同；同設備P/R exact也不代表工作無遺失。

## 驗收與交付

- 每cell correctness依固定CPU FP32 parity門檻，R另查instrumented output exact。shape/finite不是唯一判準。
- 工程gate需記錄actual backend、CUDA drop／未知warning、API return、schema／必要特徵；已知NVTX warning只能warning-accepted。
- R metadata不能進predictor；predictor只能讀`audit/predictor/*.jsonl`的既有allowlist。標籤／configuration ID只供join/evaluator。
- 記錄每iteration GPU ownership、unmatched API、P/R signature及候選長度；未歸屬、超L、noncontiguous、multi-stream不能丟棄或補0。
- 保留原timestamp與overlap；union閉合、低GI誤差、warning消失不能單獨證明Reference完整。
- 交付完整raw、assets、source與hash，保持失敗／中斷紀錄。大檔與權重不放Git。此輪沒有要求對hardware backend追加相同flush測試。

後續40×3、更多signature與依賴子圖需新工程版本／協定再實作。只有兩個配置/subtype的A0資產不夠train/validation/test，不能拆3次iteration當獨立配置。
