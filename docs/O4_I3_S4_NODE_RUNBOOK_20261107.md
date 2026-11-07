# RTX4090／3090：S4 scale0 Context收集

固定tag：`o4-i3-s4-scale0-dis-dev-v1-20261107`。先按節點收有用子集，跨設備再對可比方向交集，不要求同kernel名字／數量或全矩陣完成。

```bash
git clone --branch o4-i3-s4-scale0-dis-dev-v1-20261107 --depth 1 https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-s4
cd gpu-prediction-s4
mkdir i3-s4-downloads
gh release download o4-i3-s4-scale0-dis-dev-v1-20261107 --repo terrychen0803/gpu-prediction --dir i3-s4-downloads
(cd i3-s4-downloads && sha256sum -c SHA256SUMS)
mkdir i3-s4-assets
tar -xzf i3-s4-downloads/o4_i3_S4A_shared_base_v1_20261107.tar.gz -C i3-s4-assets
mv i3-s4-assets/base_assets i3-s4-assets/base_A
tar -xzf i3-s4-downloads/o4_i3_S4BC_shared_base_v1_20261107.tar.gz -C i3-s4-assets
mv i3-s4-assets/base_assets i3-s4-assets/base_BC
export S4_REPO="$PWD"
export I3_PYTHON=/absolute/path/to/compatible/cuda-python
export I3_GPU=4090  # 3090節點改3090
export S4_OUT=/absolute/new/path/S4_node_run
```

兩個base archive內都叫base_assets，必須如上分開解包，不要覆寫。Source archive與Git相同，不需再解包執行。GB10_local_inputs附件只供GB10診斷／重現，**RTX不得拿它當自己的D/I等價中間輸入**；不需解包這兩個附件。原Core36/S3 single-op CLI不支援此Context protocol。

## 每階段先CPU tests

先選A，完成再切BC：

```bash
export S4_STAGE=A  # 或BC，兩階段分開測／收
export S4_SOURCE="$S4_REPO/i3_context_pilot/source_$S4_STAGE"
export PYTHONPATH="$S4_SOURCE:$S4_SOURCE/o4_pilot"
cd "$S4_SOURCE"
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests_context -v
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests_v2 -v
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s o4_pilot/tests -v
(cd o4_pilot/vendor/legacy && CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests -v)
```

BC的tests_context_A_historical是原A fixture快照，不是BC active tests。當地需已安裝相容CUDA PyTorch與nsys，不自動升級driver/CUDA/Nsight或改服務。

## 本地D中間輸入：先預覽，再GPU derive

```bash
"$I3_PYTHON" "$S4_REPO/i3_context_pilot/scripts/derive_node.py" \
  --stage "$S4_STAGE" --base "$S4_REPO/i3-s4-assets/base_$S4_STAGE/BASE.json" \
  --expect-gpu "$I3_GPU" --out "$S4_OUT/derive_$S4_STAGE" --plan-only
```

plan-only驗source/base，不執行GPU。確認沒有其他使用者GPU compute工作後，同一命令移除plan-only，在尚不存在的新out執行。helper有逐source SHA、GPU guard、5秒monitor、自己的子程序timeout與清理，只清自己的子程序，不停止外部服務。derive不是formal量測，發生在capture外；保存每位置native-D input/output bytes與hash。

```bash
export S4_LOCAL="$S4_OUT/derive_$S4_STAGE/assets_context_local/ASSETS.json"
"$I3_PYTHON" -m o4_i3_context.assets verify --path "$S4_LOCAL"
```

只共享外部input/weights/state的base，兩GPU中間結果可因實作略有不同，各自derive並記錄差異。GPU/torch/CUDA/cuDNN改變需新derive與out；禁止在舊manifest上回填hash。GPU guard或monitor review先排除外部工作，不能停掉其他使用者程序或服務。

## 六批plan／Smoke

A含CBR5＋Residual9，BC含DW6＋SE8＋Resize6＋Attention6，各批cap12。切換stage後重新設定source/PYTHONPATH/S4_LOCAL，不能混用。以下都plan-only；真正Smoke另用新out `$S4_OUT/<stage>/collections/<batch>_smoke` 移除plan-only。

### A：cbr

```bash
# 先確認 S4_STAGE=A，S4_SOURCE/PYTHONPATH/S4_LOCAL指向同階段
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$S4_LOCAL" --config CTX_conv_bn_relu_0 IND_conv_bn_relu_0 ISO_conv_bn_relu_0_c ISO_conv_bn_relu_0_n ISO_conv_bn_relu_0_r \
  --out "$S4_OUT/A/plans/cbr_smoke" --plan-only
```

### A：residual

```bash
# 先確認 S4_STAGE=A，S4_SOURCE/PYTHONPATH/S4_LOCAL指向同階段
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$S4_LOCAL" --config CTX_residual_0 IND_residual_0 ISO_residual_0_c1 ISO_residual_0_n1 ISO_residual_0_r1 ISO_residual_0_c2 ISO_residual_0_n2 ISO_residual_0_a ISO_residual_0_r2 \
  --out "$S4_OUT/A/plans/residual_smoke" --plan-only
```

### BC：depthwise_pointwise

```bash
# 先確認 S4_STAGE=BC，S4_SOURCE/PYTHONPATH/S4_LOCAL指向同階段
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$S4_LOCAL" --config CTX_depthwise_pointwise_0 IND_depthwise_pointwise_0 ISO_depthwise_pointwise_0_dw ISO_depthwise_pointwise_0_n ISO_depthwise_pointwise_0_hs ISO_depthwise_pointwise_0_pw \
  --out "$S4_OUT/BC/plans/depthwise_pointwise_smoke" --plan-only
```

### BC：se_gate

```bash
# 先確認 S4_STAGE=BC，S4_SOURCE/PYTHONPATH/S4_LOCAL指向同階段
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$S4_LOCAL" --config CTX_se_gate_0 IND_se_gate_0 ISO_se_gate_0_pool ISO_se_gate_0_reduce ISO_se_gate_0_r ISO_se_gate_0_expand ISO_se_gate_0_gate ISO_se_gate_0_scale \
  --out "$S4_OUT/BC/plans/se_gate_smoke" --plan-only
```

### BC：resize_concat_conv

```bash
# 先確認 S4_STAGE=BC，S4_SOURCE/PYTHONPATH/S4_LOCAL指向同階段
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$S4_LOCAL" --config CTX_resize_concat_conv_0 IND_resize_concat_conv_0 ISO_resize_concat_conv_0_up ISO_resize_concat_conv_0_cat ISO_resize_concat_conv_0_c ISO_resize_concat_conv_0_r \
  --out "$S4_OUT/BC/plans/resize_concat_conv_smoke" --plan-only
```

### BC：attention_eager

```bash
# 先確認 S4_STAGE=BC，S4_SOURCE/PYTHONPATH/S4_LOCAL指向同階段
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu "$I3_GPU" \
  --assets-manifest "$S4_LOCAL" --config CTX_attention_eager_0 IND_attention_eager_0 ISO_attention_eager_0_qk ISO_attention_eager_0_scale ISO_attention_eager_0_softmax ISO_attention_eager_0_av \
  --out "$S4_OUT/BC/plans/attention_eager_smoke" --plan-only
```

## Smoke gate與40×3

每工作GI/P/R三cell都completed，mode唯一／repeat1，correctness/warmup通過、monitor_review=false、Trace engineering accepted，P/R唯一pair measured_exact=3且all_windows_exact=true才放行。另核對每位置完整ATen參數、次序、input/output與ownership，不僅看名字或signature。失敗和排除原因保留，其他獨立工作可繼續；補跑另立新out與原因。

以通過的config清單、相同local資產，把kind改formal；先新out做plan-only（9cells/work），再另新out `$S4_OUT/<stage>/collections/<batch>_formal` 移除plan-only收40×3。每mode三獨立程序、每程序40 measured；不能減成40×1或只取快repeat。三程序spread>5%保留review，GI、P/R event與實際位置P_CTX分開。

## 回傳與Reference

每階段另立run根目錄，例如NODE_STAGE_RUN=$S4_OUT/A或$S4_OUT/BC（A與BC不同source，不混用原report模組的manifest）；將該階段batch放collections/下，保存raw Nsight/SQLite、protocol、source_snapshot、環境、命令、asset manifest、quality、guard/monitor、iterations、失敗與R完整dispatch。可用該source的 `python -m o4_i3_v2.report --run NODE_STAGE_RUN --out NEW_ANALYSIS --replay` 重播基本trace／位置標籤；新out，不改raw。此report按saved protocol迭代，並非替換Context collector為single-op collector。實際P_CTX需按位置／repeat另聚合spread；GB10專用analysis_source快照需原資料路徑，不是通用RTX入口。

D/I/S同一graph/scale所有位置與repeat/devices共用context_scale_0；共用Conv/BN/ReLU等也合群，40工作不是40獨立配置。它是train尺度工程pilot，不構成validation/test或OOF；不拿本輪資料宣稱六模型完整DNN、未見context或未見硬體能力。

NVTX、missing submitted API、clock、overlap與backend實際差異保留。預期cuda-sw，不silent fallback；不平移timestamp／裁overlap、不猜未配對API為0B，不以低GI差／P-R一致／closure=0單獨證明Reference。Strict verified與formal_training_eligible仍false。P只有iteration scope與numeric allowlist；R真實類型/shape/邊界、target test trace/GI只給evaluator。沒有辨識器／N0／ITER訓練。

原A的中斷恢復資料完整保留在GB10，整理結果只計active完整版本一次。其他節點記錄自己的狀態，不複製GB10的PASS、警告或耗時當本機實測。
