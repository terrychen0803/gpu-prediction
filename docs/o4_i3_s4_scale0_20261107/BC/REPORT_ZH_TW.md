# S4-B／C GB10：scale0四圖D/I/S工程結果

已完成GPU實測：Depthwise/Pointwise6、SE8、Resize/Concat6、eager Attention6，共26工作。Smoke78 cells＋formal234 cells，formal measured9,360，含Smoke9,594。UTC 2026-11-07T02:14:11.829112+00:00～2026-11-07T02:46:19.393775+00:00，收集牆鐘約32.13分鐘，含guard、fresh processes、profile/export/audit，非純GPUcompute且不含前置工程或後續獨立重播。

## 已完成工程與限制

新增B/C精確op adapter與新protocol/schema ID，不修改A原目錄、舊raw或parser。B/C source174檔：保留歷史A context tests快照、加入15項B/C CPU tests；原tests_v2 18、A0 16、O4 85、legacy54、orchestrator8共196 tests通過。taxonomy只新增out-of-place aten.hardswish.default精確overload，未知仍unmapped、不模糊fallback。

四圖各位置依原plan完整shape／stride／groups／bias／broadcast／softmax dim和dtype執行。Base外部input/weights/BN state固定；本節點在capture外執行D產生local native inputs/outputs供I/S，逐位置CPU FP32 parity與native input/output exact。不同GPU不可沿用GB10中間值冒稱本地D/I同輸入，須共享base並在當地重新derive。

I移除data edge但保留原single-stream次序；D/I輸出保留到下次prepare，S保留該位置輸出到同樣時機。input placement、cache、allocator/workspace、lifetime仍可能影響差異，不把D/I比較稱純依賴因果。Attention是受控eager兩次BMM＋scalar scale＋FP16 softmax，不是完整Swin/Qwen、SDPA fused或LLM forward。

## 描述性GI比較

| 圖 | D μs | I μs | ΣS μs | D−I | ΣS−D | 全部GI穩定 |
|---|---:|---:|---:|---:|---:|---|
|CTX_depthwise_pointwise_0|35.712|35.840|45.568|-0.36%|+27.60%|False|
|CTX_se_gate_0|43.792|42.992|58.432|+1.86%|+33.43%|False|
|CTX_resize_concat_conv_0|49.360|49.984|68.688|-1.25%|+39.16%|True|
|CTX_attention_eager_0|27.744|27.376|35.728|+1.34%|+28.78%|False|

ΣS是獨立forward GI中位數相加，含各自提交/launch/事件間隔，不是DAG真值或模型預測結果。所有比較保留三程序穩定性與逐組資料，不選快repeat；若全部GI穩定=false，百分比只能作review中的描述性結果。

78個formal GI/P/R模式組：69 stable、9 review。54個圖/位置實際P_CTX：46 stable、8 review；依完整資料用途gate有46 development_usable。這不是Reference升級。

## 重播／位置與Reference品質

208份trace、104對P/R重播，9,412個衍生artifacts逐byte相同；R完整位置參數7,506/7,506匹配。實際P_CTX54項摘要獨立對照重播標籤一致。

Recorded ownership：unassigned=0、multiple=0、noncontiguous=0、zero-recorded=0；operator-span union closure最大算術error=0ns。這只是recorded事件對應與算術檢查，不代表全部submitted工作或reference語意已證明完整。

NVTX warning trace=208；unmatched work APIs=0；GPU start-before-API=7293、end-before-API=17；overlap=0對。保留timestamp與overlap，不裁切、不加offset、不把未匹配memset猜為0B，也不聲稱PDL已證明。**strict_reference_verified=false，formal_training_eligible=false。**

P只用合法iteration marker與numeric allowlist，沒有逐位置marker/同步；R完整ATen型別/shape/真邊界只給evaluator。R worker的plain/marked exact比對在最終輸出，逐位置native數值驗證發生在量測外；不能單以最終parity解除reference警告。

## 分組與舊資料保護

四圖與A兩圖全部是同一context_scale_0/train保守群；D/I/S、同形狀位置、repeats與不同設備不增加獨立群數。B/C內共享runtime component key=17，與前72 isolated及A exact交集key=1，詳見對照清單。Exact hash不排除數學等價、近鄰／歷史DNN曝光，沒有validation/test／OOF或未見context結論。

| 舊run | 原檔SHA未變 |
|---|---:|
|o4_i3_s1_gb10_20261106T213722Z|3,545|
|o4_i3_s2_gb10_20261106T220336Z|31,342|
|o4_i3_s3_lite_gb10_20261106T230230Z|34,520|
|o4_i3_s4a_gb10_20261107T000028Z|16,537|

Active312個獨立worker已結束，monitor foreign cells=0、error cells=0。5秒採樣不保證間隔內沒有干擾；GPU pre/post evidence與逐cell guards保存。沒有改K3s/Pod/Netdata/RustDesk或升級driver/CUDA/Nsight。

## 未完成與交付

A＋B＋C現合計scale0六圖40工作：120 Smoke＋360 formal cells、14,400 formal measured，仍是工程／探索資料，不是六模型完整forward驗收。Scale1/2與完整120工作矩陣不在本次範圍，未進行辨識器、N0或ITER訓練、未見GPU或完整DNN評估；RTX本輪未實測。

source174檔與shared_base／GB10 local-derived bundles分開，local-derived只供GB10本地對照，不當RTX共用輸入。見[BUNDLES.json](BUNDLES.json)／[節點交接](NODE_HANDOFF_ZH_TW.md)。完整raw留collections，所有失敗與warning保留。發布GitHub是後續獨立交付，不回寫舊closed run。

## Review清單

| 工作 | mode | spread |
|---|---|---:|
|ISO_attention_eager_0_scale|GI|11.91%|
|ISO_attention_eager_0_softmax|GI|10.26%|
|CTX_depthwise_pointwise_0|P|5.51%|
|ISO_depthwise_pointwise_0_hs|GI|11.40%|
|IND_se_gate_0|P|9.59%|
|ISO_se_gate_0_r|GI|5.94%|
|ISO_se_gate_0_expand|R|6.35%|
|ISO_se_gate_0_scale|GI|6.36%|
|ISO_se_gate_0_scale|P|12.15%|

| 工作 | 位置 | P_CTX spread |
|---|---|---:|
|CTX_depthwise_pointwise_0|dw|11.67%|
|CTX_depthwise_pointwise_0|hs|27.26%|
|CTX_depthwise_pointwise_0|pw|6.84%|
|IND_se_gate_0|pool|16.88%|
|IND_se_gate_0|scale|29.41%|
|ISO_se_gate_0_scale|scale|18.63%|
|CTX_resize_concat_conv_0|r|16.28%|
|ISO_resize_concat_conv_0_c|c|5.83%|
