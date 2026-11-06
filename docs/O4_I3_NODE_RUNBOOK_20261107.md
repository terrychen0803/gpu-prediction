# RTX4090／RTX3090：從 GitHub 取得 I3 工具並同步收集

固定版本 **`o4-i3-a0-dev-v1-20261107`**。本次發布可執行的GB10 A0工具與38個operand檔／18配置固定資產。**現在可以在兩個RTX節點各自執行A0 Smoke。新174配置、子圖及I3 40×3另有具體計畫，尚未實作，不可直接啟動。** 詳見 [計畫](o4_i3_collection_plan_20261107/PLAN_ZH_TW.md)。

## 1. 各節點建立新的 checkout

先保護現有資料與未提交工作；使用新目錄，不在原實驗目錄reset/clean。於適合的新父目錄執行：

```bash
git clone --branch o4-i3-a0-dev-v1-20261107 --single-branch \
  https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-i3-a0
cd gpu-prediction-i3-a0
git rev-parse HEAD
export I3_PYTHON=/absolute/path/to/existing/cuda-python
```

使用Linux、現有相容CUDA版PyTorch及Nsight。不要為對齊GB10而升級driver/CUDA/Torch或重開機；不需要DNN下載或安裝Transformers來跑本次A0。保持一張實體可見GPU，collector的guard按整個host的nvidia-smi查詢；多GPU主機目前會拒絕，不繞過guard。記錄實際版本與unsupported原因。

## 2. 下載同一份資產，不能在各台重新seed生成

本包約21.4MB，保存GB10原bytes；使用GitHub CLI或從 [Release頁面](https://github.com/terrychen0803/gpu-prediction/releases/tag/o4-i3-a0-dev-v1-20261107) 下載同名檔案。以下命令在repo根目錄：

```bash
mkdir i3-downloads
gh release download o4-i3-a0-dev-v1-20261107 \
  --repo terrychen0803/gpu-prediction \
  --pattern o4_i3_assets_A0_v1_20261107.tar.gz --dir i3-downloads
(cd i3-downloads && sha256sum -c ../i3_pilot/manifests/ASSETS_SHA256SUMS)
mkdir i3-assets
tar -xzf i3-downloads/o4_i3_assets_A0_v1_20261107.tar.gz -C i3-assets
export I3_ASSETS="$PWD/i3-assets/assets_A0/ASSETS.json"
bash i3_pilot/scripts/i3_node.sh verify
bash i3_pilot/scripts/i3_node.sh test
```

資產archive SHA256：`dbe8738ff76a5a9ecd3b4bb430b17c0d407970ea9f3cadd79ccdf5acfba372a1`；ASSETS.json SHA256：`02e4fbe40f617d571d42b55d3e343c8ea4272b6f26febefa16f0de8bc4594e69`。verify另逐檔核對38 operands及凍結source，不能修改manifest放寬correctness門檻。這些initialized performance-only tensors不代表DNN預訓練模型。

## 3. 先plan，再跑BN

每台選自己的參數；輸出根目錄在repo外，替換為新路徑。4090例子：

```bash
export I3_NODE=4090
export I3_OUTPUT_ROOT=/absolute/new/path/rtx4090_i3_a0_session01
mkdir "$I3_OUTPUT_ROOT"
bash i3_pilot/scripts/i3_node.sh plan --expect-gpu "$I3_NODE" \
  --config A0_batchnorm_eval_a --out "$I3_OUTPUT_ROOT/bn_plan"
bash i3_pilot/scripts/i3_node.sh bn --expect-gpu "$I3_NODE" \
  --out "$I3_OUTPUT_ROOT/bn_smoke"
```

3090使用`I3_NODE=3090`及獨立的`rtx3090_i3_a0_session01`目錄，其餘工作定義／assets相同。GB10對應`I3_NODE=GB10`。先唯讀確認沒有他人GPU工作；guard拒絕時保留failure後延期，不停止任何現有Pod／服務／程序。

BN的GI/P/R各為一個獨立process，8warmup＋1transition＋3measured，合計3cells／9measured。actual cuda-sw、Nsight schema／warnings、API coverage與P/R映射須當地審查；不能以guard／單元測試通過替代GPU實測。每cell保存source_snapshot並從副本執行。

## 4. BN完成後收其餘八種a

兩個節點可以獨立執行這份同配置清單：

```bash
bash i3_pilot/scripts/i3_node.sh collect --expect-gpu "$I3_NODE" \
  --config A0_conv_a A0_gemm_a A0_bmm_a A0_linear_a \
           A0_relu_a A0_add_a A0_maxpool_a A0_adaptive_average_pool_a \
  --out "$I3_OUTPUT_ROOT/remaining_eight_a"
```

新增24cells／72measured，連BN共27cells／81measured。某項失敗保留原因，其他項可繼續；任何一項失敗collector最後以非零exit回報。需要重試使用全新out，只選失敗config，不覆寫原attempt。b配置可另選，但它們尚未在GB10 GPU驗證，不應一開始當必跑配額。相同工作跨設備不要求kernel名字、事件數或軟體版本一致。

## 5. 回傳內容與品質

交回完整新output root（source_snapshot、CODE_SHA、asset verification、environment、commands、guard、iterations、correctness、raw nsys-rep／SQLite、audit／predictor、R metadata、pair與failure），外加`git rev-parse HEAD`、節點及資產manifest SHA。原始trace保持在節點／交接archive，Git只放索引及整理報告。raw timestamp與overlap不裁切、不平移；未知API／ownership不補0。

所有本次資料先標development／warning-accepted，strict_verified=0；known GPU dropped/missing activity仍阻擋受影響label。P/R exact、低GI誤差與known NVTX warning不構成完整reference證明。新務實用途政策不會把原frozen gate改成PASS。P-only predictor只讀audit/predictor allowlist，R的真實type/shape/邊界不進特徵。

兩台已完成I2 seven-A 40×3是不同協定，不能計為本次I3完成；新174配置與子圖要等新工程版本通過GB10三sentinel驗收後再交接。此release沒有執行RTX GPU、訓練辨識器、N0或ITER。
