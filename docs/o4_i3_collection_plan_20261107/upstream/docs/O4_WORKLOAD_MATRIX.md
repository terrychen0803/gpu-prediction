# O4 六任務工作負載與三配置規劃

**2026-11-07 收集原則更新：以收集有用資料為優先，各設備可獨立選模型／配置，不要求三台跑齊相同矩陣或同步進度。** 跨設備評估時才分方向核對可比配對，缺項與環境差異照實記錄。詳見[收集優先原則](O4_COLLECTION_POLICY_20261107.md)；以下舊版完整矩陣／對齊要求應依此調整。

更新：2026-11-06。**全部為 proposed／planned_only，未在本次執行 GPU，也未完成權重與輸入凍結。** 主目標與 GI/P/R 定義見 [主計畫](O4_GPU_ITERATION_PLAN_ZH_TW.md)。機器可讀清單在 [planned manifest](o4_gpu_iteration_pilot.plan.json)，僅為規劃輸入，不是現有 CLI 設定檔。

**主要預測：RTX4090→GB10、RTX4090→RTX3090。** 不要求每個配置都有這兩個方向，也不要求每台收齊 GI/P/R。21配置共42個配置－方向配對是全矩陣的上限估算；實際分析分方向取可比較交集，重複量測不另算獨立配對。4090提供主 predictor 的工作 trace，目標可用資料依角色供 train supervision／test evaluator；未配對資料仍可作獨立 inventory 與探索。選配反方向不改主結果平均。

## 1. 六任務的具體範圍

| 任務 | 第一版建議實作 | iteration 範圍 | 凍結前要確認 |
|---|---|---|---|
| ResNet-50 | TorchVision resnet50，IMAGENET1K_V2 | device tensor→分類 logits | exact torchvision build、state_dict hash、完整原 forward |
| MobileNetV3 | TorchVision mobilenet_v3_large，IMAGENET1K_V2 | device tensor→分類 logits | large 不是 small；不可默換 variant |
| Swin-T | TorchVision swin_t，IMAGENET1K_V1 | device tensor→分類 logits | v1 不是 Swin V2；window/layout 路徑 |
| U-Net | 建議明定的 2D U-Net（見下節） | device tensor→分割 logits | 原始需求無實作；需先固定 code/weights，不能將任意 U-Net 混稱同工作 |
| YOLO26 | Ultralytics yolo26n detect，固定一個 tensor-forward adapter | device tensor→固定定義的 detection-head 輸出 | 權重、head 路徑、是否內含 top-k/box decode，NMS／wrapper 不入主 iteration |
| LLM | Qwen/Qwen3-0.6B，固定 Transformers build | prefill 或單次 decode→logits/KV 更新 | cache、attention backend、logits 計算範圍、token IDs／positions |

官方程式用於盤點，不代表節點已安裝同版：[ResNet](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)、[MobileNetV3](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html)、[Swin](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/swin_transformer.html)、[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。不要只保存套件名稱而不保存 commit／build／weights checksum。

YOLO26 官方文件目前有 one-to-many/NMS 與 one-to-one/NMS-free 路徑。主 pilot 建議固定 one-to-one 的 tensor forward，**不能假設 `predict(..., nms=False)` 就是只有 GPU forward**；需核對安裝版 adapter 的實際路徑與輸出，將必要 head 內部 GPU 操作全部納入。若該版本無法提供同一路徑，先標 pending_adapter，不能臨時換分支。外部影像 decode、letterbox、輸出搬回 CPU、結果物件與 NMS 都不在主 GI 範圍。[官方模式說明](https://docs.ultralytics.com/guides/end2end-detection)

### 1.1 U-Net 可落地的第一版建議

若沒有使用者指定的實際模型，先以 `unet2d_v1` 建立架構性能 benchmark：1 input channel、2 output classes；encoder channels 32/64/128/256/512；每層兩個 3×3/padding1/bias=True Conv＋ReLU；四次 MaxPool2d(2)；decoder bilinear upsample(scale2, align_corners=False)＋skip concat＋兩次 Conv/ReLU，通道回到 256/128/64/32；最後 1×1 Conv 輸出 logits，無 BatchNorm/softmax。

此規格是建議基準，非已實作、非預訓練的原論文模型。優先採實際任務訓練權重；若暫時只能使用一次產生、跨設備相同 checksum 的初始化權重，要標 `weights_kind=initialized_performance_only`，不能宣稱分割品質或代表任意 U-Net。原始論文見 [U-Net](https://arxiv.org/abs/1505.04597)。替換為任務模型須新 model_signature，不能覆蓋舊基準。

## 2. 每任務三配置：先作流程與範圍盤點

共 5 個影像任務×3=15 個配置；LLM 的 prefill/decode 各3，共6；總計 **21 個配置、六任務、七個 iteration 群**。不能把 prefill＋decode 合成同一筆延遲。

| ID | 模型／iteration | 配置 A（小型） | 配置 B | 配置 C |
|---|---|---|---|---|
| RN-A/B/C | ResNet-50 forward | B1, 3×224×224 | B4, 3×224×224 | B1, 3×320×320 |
| MV-A/B/C | MobileNetV3-Large forward | B1, 3×224×224 | B4, 3×224×224 | B1, 3×320×320 |
| SW-A/B/C | Swin-T forward | B1, 3×224×224 | B4, 3×224×224 | B1, 3×384×384 |
| UN-A/B/C | unet2d_v1 forward | B1, 1×256×256 | B4, 1×256×256 | B1, 1×512×512 |
| YO-A/B/C | YOLO26n fixed head forward | B1, 3×640×640 | B4, 3×640×640 | B1, 3×960×960 |
| QP-A/B/C | Qwen3 prefill | B1, L=128 | B1, L=512 | B1, L=2048 |
| QD-A/B/C | Qwen3 decode | B1, KV before=128, input=1 token | B1, KV before=512, input=1 | B1, KV before=2048, input=1 |

以上尺寸是待節點 smoke 的計畫，不保證記憶體與版本相容。先跑各群 A，不直接啟動全矩陣。所有時間是實測才填；OOM/unsupported 配置仍留名單。若改 batch／尺寸，使用新 ID 與 protocol，不保持原 ID 混算。

這三配置不構成模型訓練／統計泛化的充分資料。尤其只有3個 shape，無法合理支撐多階段複雜 MLP 與多重留出。它們是 inventory 與 pilot 錨點，development 正式矩陣由 inventory 後建立，確認集另凍結。

### 2.1 LLM 的固定狀態

建議使用 Transformers 原生、單 GPU、FP16、eager attention 作第一個可解釋基準。這是執行後端選擇，不代表部署最優；SDPA／FlashAttention 若另測，固定其融合單位與 backend ID，不能混入相同 signature。

- Prefill：device 上固定 token IDs/attention mask/positions；cache 初始為空，use_cache=True；forward 到最後位置 logits 與該步 KV 更新。若模型版本支援只算最後位置 logits，明示設定；不支援時完整 logits 是另一 signature，不能默默切換。
- Decode：forward 前 cache 長度、配置容量、cache 類型／dtype/layout、positions 與單一輸入 token 全固定；輸出包含這一步 KV 更新。建議先固定一種已驗證的 DynamicCache 行為，不將 dynamic/static cache 混用。
- 每次重複前在計時外重新準備同樣的 cache 狀態並完成同步；preparation 會影響 cache 熱度，P/R/GI 同步採相同協定並記錄。嚴禁單純連續 generate 後把不同 cache 長度視為重複量測。
- Tokenizer/chat template、sampling/argmax、文字輸出都在 GI 範圍外。prompt tokens 一次準備並跨設備分享，不依不同 CPU 重新隨機產生。
- 原模型一次 forward 可能由多個 kernel 實作；eager backend 與 fused backend 各自盤點。不能假設 attention 只有兩個 BMM。

KV 狀態的重要性參考 [Transformers cache 文件](https://huggingface.co/docs/transformers/kv_cache)。以上條件需 adapter 正確性測試；無法實現就記明待實作，不填假 benchmark。

## 3. 依資料盤點擴 taxonomy

| 優先順序 | 覆蓋的目標 | 需補的實際工作與變因 |
|---|---|---|
| 1 | ResNet-50 完整流程 | Conv 的實際 shape；BN eval、ReLU、add、pool、Linear/bias；確認 cuDNN 融合 |
| 2 | MobileNetV3 | group/depthwise、stride、通道倍率、Hardswish/Hardsigmoid、SE 的 pooling/mul |
| 3 | U-Net／YOLO26 | resize/interpolation、concat、ConvTranspose（若採用）、SiLU／head 內部操作 |
| 4 | Swin-T | LayerNorm、GELU、softmax、window shift、pad、實際 copy/contiguous、GEMM bias |
| 5 | Qwen3 | RMSNorm、RoPE、embedding/gather、activation/gating、cache update、prefill/decode attention |

按每類在完整 iteration 的 **reference time coverage 與 event coverage** 排優先，而不是僅按數量。0 GPU 工作的 view 不計 kernel 時間；需要 copy 的 reshape/layout 不能當免費。Fused 類型對兩端均可定義才配對，否則保留未對應量。

每新增類至少具備多個獨立配置群，包含計畫範圍的 batch、尺寸、layout、broadcast／bias／groups 等變因。先少量 smoke 確認量測與辨識可行，再擴散；不照架構清單直接宣稱已全部支援。

## 4. 子圖與完整模型的階梯

1. **已知三類依賴子圖**：連續 GEMM、兩層 Conv、連續 BMM；中間張量維度相容，不偷加 ReLU/softmax。與現有 O2 dependent-tail inventory 比較，列出新增上下文。
2. **ResNet bottleneck**：補齊 BN/ReLU/add 等後，block→stage→完整 forward。
3. **MobileNet inverted residual/SE**：短 kernel 與 depthwise 特性。
4. **U-Net encoder/decoder block、YOLO backbone/neck/head**：concat/resize 與輸出 scope。
5. **Swin block、Qwen decoder layer**：normalization、attention、融合與 cache；最後才完整模型。

每一階梯均做固定模型零調整轉移，再單獨報 training-context 校正。不要為每個 block 加同步以方便辨識；這會建立與真實 forward 不同的工作。

## 5. 收集數量與成本估算

配置量測單元 = 一配置×一設備×一模式×一獨立程序 repeat。若未來一程序掃多配置，程序數會不同，必須另外計。

| 範圍 | 配置數 | 三設備×GI/P/R×三程序 | 用途 |
|---|---:|---:|---|
| 每 iteration 群先 A | 7 | 189 單元 | 六任務／七群 inventory pilot |
| 完整展示三配置 | 21 | 567 單元 | 含上述189，不能再相加 |
| GI 每程序40次 | 21×3×3 GI程序 | 7,560 次 measured GI forward | 不含 warmup/transition，非獨立配置數 |

若 P/R 同為40次，完整三模式共22,680次 measured forward；capture、權重準備、reference 修復、E2、子圖、正式development／confirmation均另計。可選 G0 另增加189單元，不在主567內。

這不是要求立刻跑567單元。先3類簡單算子 smoke，再7個A配置；量實際時間／檔案大小後決定擴展。總成本用 `Σ(prepare+warmup+measure+export+parse)` 估計，OOF 訓練另計；不以 forward 毫秒乘次數當總工時。

## 6. 正式資料集擴充方法

三配置 pilot 完成後，由節點產生待凍結清單：

- Model-internal：各模型較密的 batch／resolution／length grid，分開內插與外插；先驗證合法尺寸與記憶體。
- Context A：相同 op signature 的新順序／依賴／reuse，與 isolated 對照。
- Signature B：整組留出 operation signature，跨設備、所有上下文與 repeats 同群。
- Model C：完整架構留出，該模型 GI/CTX 不參與 fit；宣告是否曾以獨立算子見過其 shape。
- Confirmation：在模型／來源 view／threshold 全凍結後才讀結果；新日期同配置測的是穩健性，不算新 shape 泛化。

保留適當的獨立配置群，避免所有混合 template 都共用同一少量配置而連成無法切分的巨群。最終配置數由 learning curve／誤差變異與確認目標決定，不能先用三配置承諾充分統計力。
