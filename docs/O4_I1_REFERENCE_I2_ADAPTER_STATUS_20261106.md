# O4 I1 Software Reference Audit／I2 七個A配置交付

日期：2026-11-06（Asia/Taipei）。依repo main `43a04a3`及Draft PR #1 `d01924c2cbc0152c7dd78db55ecf3146824bfc75`的I0～I7規劃。發布基準更新為main `b204ae63ee2dff2bbc161bb5cc8f9aaecfb68c01`；PR #1已於本輪期間由外部合併，本代理沒有合併或改寫其分支。實測仍按開始時的規劃與source hash保存。本地報告以使用者確認的`REPORT_ZH_TW.md`為準；不存在的`REPORT_ZH_TW1.md`未被假造。

## 1. 四種狀態分開

| 狀態 | 本輪實際內容 |
|---|---|
| 已完成GPU實測 | 七個A配置×GI/P/R共21cell；每cell3measured＋1transition；兩版standalone參數控制／RN-A各3forward；兩個software native clock/NVTX控制；YO raw-head診斷 |
| 已完成工程實作 | 七Adapter、software-only collector、v4品質gate、union閉合、顯式prepare scope、CUPTI參數觀測器、hash資產驗證、可攜交接與測試 |
| 僅有規劃／尚未執行 | I3新受控資料、block/stage轉移、辨識器/N0/ITER/OOF、B/C配置、確認集；RTX4090／3090本輪未操作 |
| Reference未驗證 | strict verified=0；NVTX／Clock／PDL等資格未解除，沒有正式逐算子或跨設備模型成績 |

所有新資料在獨立run。舊O1/O2/O3/O4 raw、權重、協定、判定與本地未提交修改保留。不修改K3s／Pod／Netdata／RustDesk；每個GPU程序前重新核對程序owner，只有同使用者RustDesk保留，其他GPU工作會阻擋。環境非獨占，這批不是正式性能比較。

## 2. I1-A：op140已取得哪些直接證據

使用獨立CUPTI Callback observer，**不在Nsight內載入**。在callback有效期間讀取`cudaMemsetAsync`的`size_t count`並立即拷貝；log寫入延後到停止觀測，沒有改API參數／同步GPU／修改原始trace。64B／0B各一個控制驗證layout、enter/exit correlation與成功return。SDK實際CUPTI版本130001（13.0.1），與Nsight bundled13.3分開記錄。

兩個獨立RN-A replica各觀察3次forward，共102次成功memset API；**op140六次count都是0**。每forward17次API，其中16次positive與1次0B；positive分布是384B×4、64B×5、512B×5、128B×2。op140為`aten.conv2d.default`，input `[1,1024,14,14]`、weight `[512,1024,1,1]`、FP16。v2顯式使用O4 precision policy，並保存precision flags與原始callback CSV。

新舊RN-A的175項ATen／shape／dtype有序signature完全相同。這是**相同資產與觀察路徑下的直接no-op重現證據**，比native零長度範例更接近原問題。但是觀測沒有與舊Nsight capture同時發生，舊270筆API仍缺逐筆count。**不回填舊資料為0B、不改歷史submitted_work_coverage與strict gate。** 新replica能判定的是其直接看到的六次呼叫。

細節：[I1摘要](o4_i1_i2_20261106/I1_REFERENCE_SUMMARY.json)、[新舊signature比較](o4_i1_i2_20261106/I1_old_new_signature_comparison.json)。原始CSV與ATen清單保存在`evidence/rna_memset_arguments{,_v2}/`。

## 3. I1-B：NVTX警告範圍已縮小，但未解除

三個最小新控制：

| 控制 | 實際觀察 | 警告 |
|---|---|---|
| CPU-only NVTX、完整程序 | 已知8 push/pop範圍皆輸出，沒有CUDA呼叫 | NVTX warning仍在 |
| cuda-sw native、完整程序 | 32kernel＋16個64B memset＋2copy；含warmup與計時外copy | NVTX warning仍在，未見CUDA完整性warning |
| cuda-sw native、device event tracing | 16kernel＋8memset、16筆非零device completion timestamps | NVTX與CUDA warning仍在 |

因此，**CUDA workload、RN-A、逐op大量標記、cudaProfilerStop capture range都不是此NVTX warning的必要條件**。可能與工具分析／結束／保守完整性政策有關，但還沒有collector內部證據，不能宣稱是假警告。Scope數完整只是已輸出範圍的結構證據，不升級Reference。

本輪14份A配置P/R trace全部有NVTX warning、沒有CUDA完整性warning。這也不能證明它們沒有漏記；Native控制與模型collector的harness／finalization不同，不將其warning差異歸因於單一因素。

## 4. I1-C：時間軸與Overlap

SQLite／API/GPU時間均保存原始整數ns。Correlation用process限定的唯一ID對應，不用CPU/GPU先後篩掉事件。API start/end描述主機呼叫範圍，GPU時間由工具映射到共用時間軸；GPU開始早於API返回可正常，但開始／結束早於API entry仍列出待解的clock/callback問題。

新device-event控制取得16個非零completion timestamps，8iteration的已記錄GPU工作都落在各自start/end event completion之內。這支持**同一profiler內GPU相對邊界**的觀察一致性，仍非獨立CPU/GPU絕對校準，且新旗標會增加instrumentation。未將新timestamp拿來修補舊trace。

沿用一般化union：`U=sum(operator spans)+positive gap−union overlap`；不拿pair overlap相加代替多重重疊的union。Event-index分段與temporal overlap分開。RN-A舊123對overlap／1579筆end-before-entry不改判；本輪其他模型也可觀察到overlap／clock異常。PDL仍只是相容假說，**未取得實際launch attribute，沒有宣稱證實**。

新P/R完整有序signature：七配置各4/4一致（1transition＋3measured）。R→P只轉移事件索引，用P自己的GPU時間算CTX；R timing不是P／GI真值。相同signature不排除共同漏記。

## 5. Hardware Trace明確限制

保留既有五份hardware capture的0/80kernel與incomplete CUPTI events dropped證據。**本輪新增hardware實驗=0**，沒有重做200ms drain或相同flush請求。Nsys2026.4.1／driver580.159.03／bundled CUPTI13.3及實際backend、flush metadata矛盾皆保留。顯式flag不能等同實際flush路徑已驗證。

GB10工程預設只有`cuda-sw`，無silent fallback。Hardware列為獨立toolchain investigation，不阻擋software與I2。下一個hardware控制必須另有明確新假說（例如實際flush／capture lifetime內部證據）並先凍結；本輪不擴測、不升級driver/CUDA/Nsys、不重開機。見[限制摘要](o4_i1_i2_20261106/HARDWARE_TRACE_LIMITATION.json)。

## 6. I2實際GPU smoke與operation inventory

新協定`o4_A_inventory_smoke_v1_20261106`：每A×GI/P/R一個獨立程序，8warmup＋1transition＋3measured。只有七個A配置，**不是完整21配置或567cells**。初始化assets／input與所有必要準備不在forward計時內。Qwen固定eager attention、DynamicCache、last-position logits；QP cache 0→128，QD每次重新準備128再計時128→129。Mask／positions也在計時外，沒有連續generate。

下表GI只是單程序三筆中位數；未通過或宣稱40筆／三程序穩定性gate，不能用來比較舊run性能或當作正式模型訓練標籤。ATen單位不是C++ primitive或DNN layer真值。Counts只用3個measured、成功條件轉移的P事件；transition不併入。

| 配置 | GI µs（development） | ATen單位/forward | 零recorded-GPU單位 | 最大events/op | 超L4單位/forward | measured P/R |
|---|---:|---:|---:|---:|---:|---:|
| RN-A | 1873.824 | 175 | 1 | 5 | 9 | 3/3 |
| MV-A | 1758.688 | 187 | 2 | 3 | 0 | 3/3 |
| SW-A | 2814.464 | 782 | 434 | 3 | 0 | 3/3 |
| UN-A | 1940.192 | 49 | 0 | 6 | 17 | 3/3 |
| YO-A | 4616.256 | 465 | 44 | 16 | 3 | 3/3 |
| QP-A | 16875.584 | 2543 | 720 | 2 | 0 | 3/3 |
| QD-A | 15979.616 | 2403 | 608 | 2 | 0 | 3/3 |

RN-A/MV-A/SW-A使用矩陣指定TorchVision權重。UN-A依矩陣unet2d_v1：1→2classes、32/64/128/256/512、雙3×3 bias Conv/ReLU、4pool、bilinear+concat decoder、無BN/softmax；權重一次初始化後凍結，**initialized_performance_only**，沒有分割品質成績。

YO-A為YOLO26n，native one2one輸出的tensor forward，head內decode/topk納入、外部NMS/preprocessing不納入；原生forward仍計算auxiliary one2many分支，未做backbone fusion，這個具體scope已保存，不冒充已剪除輔助head的更精簡實作。Same-precision native route parity完全相同。CPU FP32與FP16的topk列相對L2約32.90%，不將不同排序的列直接稱同一偵測。額外anchor-aligned raw head診斷：boxes L2約0.1233%、scores約0.0634%，各自scaled-max亦低於2%門檻。**未驗證最終偵測集合等價或task accuracy**；舊smoke的correctness規則與數字保留，未事後調門檻。

Qwen計時外準備仍有GPU工作：QP-A每P/R capture合計8事件；QD-A各7324事件。它們保存在explicit `o4.prepare/...` scopes，獨立反向audit，沒有丟掉或混入GI。主P export只包含iteration內允許數值欄位；model/op/shape/reference不能進主predictor。

新taxonomy診斷涵蓋Conv、BN、activation、pool、resize、concat、GEMM/BMM、normalization/reduction、softmax、embedding/index與cache/layout等。Family分組是evaluator heuristic，不是Classifier；RMSNorm/RoPE等可被多個ATen操作拆開，需I3另定語意單位。CPU static inventory觀察MV-A有15個depthwise Conv module，YO-A有14個；它們是架構證據，不是已完成module↔ATen唯一映射。

新unmatched work API仍有RN-A／MV-A／SW-A，不能用op140例子替其他API自動解釋。全部模型都有NVTX warning；L4明顯不涵蓋UN-A／YO-A等片段。詳見[配置與smoke](o4_i1_i2_20261106/A_SUPPORT_AND_SMOKE.csv)、[family coverage](o4_i1_i2_20261106/FAMILY_COVERAGE.csv)。

## 7. 工程驗證、版本與已保留失敗

初版21個launch在vendored import bootstrap失敗，沒有產生forward延遲資料。全部保留，不併入21個成功smoke；修正後先RN-A三cell再其餘六A。另保留最初C++重複include編譯錯誤與static inventory Tensor序列化錯誤。失敗不覆寫為成功。

回歸包括原O4測試、新SDK參數／資產／cache／guard／allowlist／union風險測試及legacy全套；精確計數見FINAL_VALIDATION.json。14份A trace由獨立Python程序重新讀raw SQLite，174個分析檔逐byte一致。這是重現性證據，並非Reference正確性證明。

GB10實際stack：torch2.13.0+cu130、vision0.28.0+cu130、Ultralytics8.4.55、Transformers4.57.1；後者與依賴只裝在新run隔離venv。原runtime來源local `fb74b6f`／GitHub `43a04a3`；測量源碼hash、observer v1/v2使用檔、native CPP／binary hash、tool版本、asset manifest與每命令UTC/monotonic都保留。發布版增加RTX expect-gpu／cudart discovery／software-only入口，沒有假稱RTX已測。

共享ASSETS.json SHA-256：`55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4`。所有14個資產檔（含manifest、model config與revision）另有SHA index；權重與input不放Git。其他節點不得當地重生初始化權重或input後仍稱同一份資產。

## 8. 交付、I3準備與下一步

- [I1與四狀態摘要](o4_i1_i2_20261106/FOUR_STATE_STATUS.json)、[硬體限制](o4_i1_i2_20261106/HARDWARE_TRACE_LIMITATION.json)、[RTX節點交接](O4_I1_I2_NODE_HANDOFF_20261106.md)。新Git分支／PR只發工程source、manifest、匿名化摘要；raw／權重／環境值留本機。
- I1剩餘：NVTX warning的collector內部原因、CPU/GPU clock/callback解釋、真實PDL attributes；舊270筆API參數不可恢復。只有追加的直接證據可以改新版本gate，不修改歷史標記。
- I3優先用本輪event/time coverage建立多配置受控資料：BN/activation/add/pool、depthwise/SE、resize/concat、norm/attention/cache及超L4/fused片段。不能從七個A直接訓練複雜多階段模型。
- RTX4090／3090完成同資產的software最小控制與RN-A smoke後，才擴其他A；它們未在本輪執行。主4090→GB10/3090預測、N0/ITER/OOF與正式確認集全部尚未完成。目標test P/GI/reference仍限evaluator。

官方背景：[CUPTI13.0.1 Callback API](https://docs.nvidia.com/cupti/13.0.1/api/group__CUPTI__CALLBACK__API.html)、[Nsight User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Qwen3官方model card](https://huggingface.co/Qwen/Qwen3-0.6B)。線上文件可能較本機新；本機headers、版本與執行證據優先。全程沒有正式重新訓練辨識器、N0或ITER。
