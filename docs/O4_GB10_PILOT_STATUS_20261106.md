# O4 GB10 第一輪交付報告

日期：2026-11-06（Asia/Taipei）。**已完成GI收集器、GB10三類smoke、RN-A完整運算盤點及三節點交接工具。工程工作包完成，品質gate未全部通過；未宣稱正式O4跨設備模型完成。**

開發依據GitHub `891d04aeb74043e23e5309bca2856b61e01057a5`；最終實測collector／parser程式版本為本地 `6a99701`（來源provenance，不是此遠端repo的可checkout ref）。舊O2 collector源自`f1f73fe6571b5409b039bd548718f5727a23cb41`，只複製依賴，未修改原工作區。本次新增獨立的 [o4_pilot](../o4_pilot/README.md) 程式快照；節點記錄此GitHub實際checkout SHA及release manifest。

## 1. 已完成內容

- GI為無profiler一次完整native forward的CUDA event elapsed，與CTX、GE、G0、profiled event分開；不以CPU假資料代替GPU。
- GEMM／BMM／Conv × GI/P/R × 三獨立程序：27/27完成，1,080筆measured iteration。
- RN-A（ResNet-50 pretrained IMAGENET1K_V2、B1／224、FP16／eager／原生forward）× GI/P/R × 三程序：9/9完成，360筆measured iteration。
- 另有修正後單一GEMM三模式smoke 3/3完成，共120筆。所有39個成功cell合計1,560筆；主兩批36cell合計1,440筆。
- 完整attempt狀態：{'completed': 39, 'failed': 2, 'not_started': 10}。兩個早期失敗與十個當時未啟動cell保留，不併入成功分數。
- 原始capture、SQLite、命令、環境、UTC／monotonic時間、warmup、correctness、逐筆樣本與warnings均保留。沒有新訓練辨識器／N0／ITER。

## 2. GI結果與穩定性

每程序40筆先取中位數，再取三程序中位數；spread=(max−min)/median，門檻5%。未挑最快程序、未刪慢值。

| 工作 | GI µs | 三程序spread | gate |
|---|---:|---:|---|
| bmm_2_16_64_64 | 10.288 | 9.18% | 未通過 |
| conv_1_16_32_32_k3_s1_p1 | 20.704 | 3.25% | 通過 |
| gemm_16_64_64 | 8.528 | 9.57% | 未通過 |
| RN-A | 1921.920 | 1.89% | 通過 |

RN-A的GI約1.922ms；P/R profiled event約2.591／5.070ms，兩者不是無profiler真值。它們與GI的差異同時反映profiler與R instrumentation，不能把R時間當直接標籤。GI仍含event interval內的host提交空隙，結論屬本平台／harness，不是純kernel busy。

短GEMM/BMM穩定門檻未過，支持後續做量測變異診斷；不由此批量重收O2或改用batch-amortized時間。Conv與RN-A通過本次repeat gate也不是reference正確性證明。

## 3. RN-A完整運算盤點

120個成功匹配的measured iteration（40×3程序）中，每次觀察到175個非巢狀ATen攔截單位：174個有GPU工作、1個flatten無GPU工作。每次共350個工作事件（334 kernel＋16 memset），無copy。三對P/R全部40/40 iteration完整有序signature相同，並非按duration近似配對。

| 類別 | 每次運算單位 | P event比例 | P busy union比例 |
|---|---:|---:|---:|
| Conv | 53 | 35.14% | 63.84% |
| BN eval | 53 | 45.43% | 26.98% |
| ReLU | 49 | 14.00% | 5.02% |
| residual add | 16 | 4.57% | 2.14% |
| pooling | 2 | 0.57% | 0.86% |
| Linear，含bias | 1 | 0.29% | 1.28% |
| flatten | 1，零GPU工作 | 0% | 0% |

各類別的busy union可重疊，因此上表比例加總可略超過100%，不可直接當互斥時間分解。

舊GEMM/BMM/Conv family只關聯其中35.14%事件、63.84%已記錄busy union；其餘需補類別與資料。這不是舊N0已支援63.84%的聲明，Conv shape／layout／配置仍須逐一核對。Linear帶bias另列，不硬歸GEMM。

部分Conv每次有5事件：每forward有9個運算超過舊L=4，涉及12.86%工作事件。舊候選設定不能直接套完整RN-A。零GPU flatten保留null span，不虛構GPU時間。

ATen攔截單位包含conv2d/batch_norm/linear等複合呼叫，不表示所有C++ primitive或任意Python層皆可唯一還原。另存CPU-only static_module_inventory，記錄Conv stride/padding/dilation/groups/bias與TorchVision source hash，只供evaluator。

## 4. Reference與組合gate

主兩批12/12對P/R有序signature相同，所有成功trace可盤點；NVTX事件可能未完整收集警告仍存在。**Strict verified=0；warning-accepted不升級正式reference。** P只有iteration範圍NVTX，沒有逐op marker；原始SQLite／診斷名稱／ATen shape不能進主predictor。

RN-A六份trace有五份記錄GPU工作重疊，共123對相鄰interval重疊；另有1,579次GPU end早於所關聯CPU API start，最大約1.94µs。根因未識別，不能宣稱一定是真並行或工具誤差。Parser保留timestamps，以process限定的唯一correlation ID關聯，不用跨clock時間過濾；重疊按union盤點並標composition_supported=false。

因此本次不套非重疊SUM恆等式，不把重疊裁掉，也不將GI擬合良好當作解析無誤。RN-A的R reference事件集合在本次皆連續，但L與時間重疊仍限制舊模型轉移。

## 5. 開發失敗與修正證據

1. 最初v1 GEMM P完成40筆，但釋放CUDA events晚於自己的context reset造成程序exit139。失敗run完整保留；v2先釋放events/tensors再reset，後續單配置及主批次均正常完成。
2. 第一個RN-A P完成correctness與40筆量測，但舊parser用CPU/GPU先後過濾correlation而拒絕；修正後又發現原先非重疊假設不成立。兩次read-only重解析均保存在新目錄，原始failed quality未覆寫。
3. 以新parser版本重新收集完整RN-A九cell，正確性、原始資料、P-only匯出、reference與composition狀態分開交付。

## 6. 驗證與重現

22項O4離線測試、54項既有測試通過；含event單位／順序、stream改變、未知schema、copy／memset、漏事件、多解correlation、clock inversion、零GPU工作、重疊gate、oracle欄位poison及失敗不覆寫。

三類與RN-A各P/R共4次read-only SQLite重播：RN-A全部audit檔byte-exact；三類所有舊schema欄位相同，新版僅新增明示診斷欄位。原O2的53份來源檔與O3的30份模型權重hash維持不變，成功cell的原始樣本中位數獨立重算一致。整理驗證見 [validation_summary.json](o4_gb10_pilot_20261106/validation_summary.json)；完整raw／重播證據留GB10。

所有run均有artifact SHA-256清單；SHA一致只證明來源／產物一致，不證明reference正確。GPU結束後0%使用率、compute process清單為空。K3s／Pod／Netdata／既有服務未修改，沒有留下本實驗GPU程序。

## 7. 其他節點如何接手

先讀 [節點對齊與執行清單](O4_NODE_ALIGNMENT_20261106.md)，使用本repo的`o4_pilot/`獨立目錄。公開的是實作、manifest、匿名化aggregate與本報告；原始Nsight、完整逐筆reference、權重與105MB bundle未上傳Git。

- [三類aggregate](o4_gb10_pilot_20261106/three_class_aggregate.csv)／[RN-A aggregate](o4_gb10_pilot_20261106/resnet_aggregate.csv)
- [RN-A運算清單](o4_gb10_pilot_20261106/resnet_operation_inventory.csv)／[P-R對應摘要](o4_gb10_pilot_20261106/resnet_pair_summary.json)
- [機器可讀對齊manifest](o4_gb10_pilot_20261106/alignment_manifest.json)／[量測協定](../o4_pilot/manifests/protocol.json)／[資產checksums](../o4_pilot/manifests/assets.json)

RN-A GI約1.922ms不是其他設備應複製的數值；其他節點應對齊工作、模式、資產bytes、計時與品質規則，設備時間可以不同。GI穩定性、reference可信度、候選覆蓋、直接組合能力分開判定。

下一步優先4090／3090同版small smoke與RN-A；診斷NVTX警告、重疊與短GI變異；依RN-A補BN/ReLU/add/pool/Linear等資料。跨設備預測、完整DNN模型、六任務矩陣與新確認集仍待完成。新程式只支援Linux/POSIX live；Windows live待實作／驗證。
