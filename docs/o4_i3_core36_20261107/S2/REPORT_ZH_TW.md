# GB10 I3 S2：核心36配置擴充收集

2026-11-07（Asia/Taipei）。本輪接續S1，收集預先指定12家族×anchors00/03/04中的剩餘33配置，批次為9／12／12工作。既有S1的3配置以來源索引併表，未重收或改寫。所有配置屬train development範圍。GPU本輪仍是contiguous base配置；channels_last／transpose view／groups4等額外控制屬S3，尚未收集。

## 四種狀態

| 狀態 | 實際成果 |
|---|---|
| GPU已執行 | S2 Smoke完成99/99 cells；40×3完成297/297 cells、11,880筆measured；原S1另有27正式cells／1,080measured |
| 工程 | 沿用S1凍結collector／parser／protocol；新33配置固定assets、批次執行器、完整回歸／參數與signature稽核、獨立raw replay、可攜包 |
| 待執行 | S3其餘138配置；S4依賴圖／斷依賴／isolated對照；RTX新協定；I4／N0／ITER訓練；獨立val/test／OOF |
| Reference未驗證 | strict_verified=0、formal_training_eligible=false；所有warning、clock／overlap／unmatched依用途保留；收集完成不等於Reference完整 |

## 批次與樣本分母

S2預選33配置；實際有formal完成資料的配置33個。Smoke與formal合計保存12,177筆measured；被排除或失敗配置不從預選名單移除。全部嘗試見[CELLS](analysis/summary/CELLS.csv)與[FAILURES](analysis/summary/FAILURES.csv)。
連同原S1，核心池共36個設計配置，正式完成324cells／12,960筆measured。Repeats／GPU events不增加獨立配置數。

source的CONFIGURATIONS.json保留規劃當時的asset_status／adapter_status／gpu_status設計欄位以維持catalog hash；實際執行狀態以本run與S1的inventory／STATUS／report為準，不用catalog中的planned字樣否定已收資料，也不把未測配置升級。

## GI與穩定性

GI是無profiler、整個單算子控制工作的一對CUDA events elapsed，先取每process40次median，再取3個process medians的median。不是完整DNN GI、不是O1 GE，也不是P CTX；不同shape不能直接作演算法快慢排名。

| 家族 | anchor00 GI µs | anchor03 GI µs | anchor04 GI µs |
|---|---:|---:|---:|
| adaptive_average_pool | 9.296 | 11.072 | 11.040 |
| add | 7.136 | 7.024 | 7.296 |
| batchnorm_eval | 19.872 | 16.160 | 16.464 |
| bilinear_resize | 15.616 | 17.840 | 32.288 |
| bmm | 10.880 | 11.120 | 12.816 |
| concat | 7.856 | 7.792 | 8.656 |
| conv | 22.880 | 31.808 | 41.120 |
| depthwise | 14.656 | 9.696 | 11.680 |
| gemm | 11.408 | 11.136 | 10.752 |
| linear | 9.600 | 9.728 | 11.024 |
| maxpool | 10.144 | 9.776 | 10.432 |
| relu | 6.928 | 6.832 | 7.056 |

S2的99個formal配置×mode群：stable=78、review=21；合併S1則stable=87、review=21。每mode spread=(max process median−min)/median，5%門檻；超界保留原值，不挑最快repeat、不補跑取代。
完整GI/P/R與spread見[CORE36_METRICS](analysis/summary/CORE36_METRICS.csv)與[MODE_STABILITY](analysis/summary/MODE_STABILITY.csv)。R instrumented時間不能當無profiler標籤。

P_CTX另外從exact P/R對應的P GPU intervals計算，逐process40次median再跨3process評估：36配置中stable=28、review=8。詳見[P_CTX_STABILITY](analysis/summary/P_CTX_STABILITY.csv)，不能拿profiled iteration event elapsed的spread直接當CTX spread。
例：GEMM anchor04的P event elapsed三process medians約12.224／12.144／5.072µs（spread58.89%），但P_CTX為2.048／2.064／2.112µs（spread3.10%）。前者含整個timed forward的提交／等待等，後者是已映射GPU工作的span，不能互換；這不識別唯一波動原因，也不解除clock／Reference限制。
最初analysis/summary_initial_profiled_event_gate保留初步保守審查；最終analysis/summary改用實際P_CTX穩定性判定CTX用途，GI/P/R event穩定性仍另列，raw／收集協定完全未改。

需review的S2 mode：

- I3_depthwise_00_base／GI：spread=5.02%，process medians=[14.783999882638454, 14.655999839305878, 14.047999866306782]。
- I3_bmm_00_base／R：spread=5.20%，process medians=[30.271999537944794, 30.784000642597675, 31.87199868261814]。
- I3_add_00_base／GI：spread=5.16%，process medians=[6.87999976798892, 7.2480000089854, 7.135999854654074]。
- I3_adaptive_average_pool_00_base／GI：spread=5.68%，process medians=[9.55200009047985, 9.02399979531765, 9.296000003814697]。
- I3_conv_03_base／GI：spread=5.84%，process medians=[32.00000151991844, 31.808000057935715, 30.14400042593479]。
- I3_relu_03_base／GI：spread=6.32%，process medians=[6.464000092819333, 6.895999889820814, 6.832000100985169]。
- I3_add_03_base／GI：spread=6.61%，process medians=[7.023999933153391, 7.408000063151121, 6.943999789655209]。
- I3_maxpool_03_base／GI：spread=10.31%，process medians=[8.799999952316284, 9.808000177145004, 9.775999933481216]。
- I3_adaptive_average_pool_03_base／GI：spread=5.06%，process medians=[11.008000001311302, 11.07200002297759, 11.568000074476004]。
- I3_bilinear_resize_03_base／P：spread=7.30%，process medians=[20.8320003002882, 20.447999238967896, 21.968000568449497]。
- I3_bilinear_resize_03_base／R：spread=5.03%，process medians=[37.87200152873993, 38.704000413417816, 36.80000081658363]。
- I3_conv_04_base／P：spread=20.86%，process medians=[36.25600039958954, 45.2479999512434, 45.69600149989128]。
- I3_depthwise_04_base／P：spread=6.19%，process medians=[14.463999774307013, 13.887999579310417, 14.783999882638454]。
- I3_gemm_04_base／P：spread=58.89%，process medians=[12.223999947309494, 12.144000269472599, 5.071999970823526]。
- I3_bmm_04_base／P：spread=27.08%，process medians=[15.984000638127327, 15.95200039446354, 11.663999874144793]。
- I3_linear_04_base／GI：spread=15.53%，process medians=[11.024000123143196, 9.727999567985535, 11.440000031143427]。
- I3_add_04_base／GI：spread=7.24%，process medians=[7.007999811321497, 7.536000106483698, 7.296000141650438]。
- I3_maxpool_04_base／P：spread=8.49%，process medians=[14.607999939471483, 13.75999953597784, 13.439999893307686]。
- I3_adaptive_average_pool_04_base／GI：spread=5.07%，process medians=[11.440000031143427, 11.04000024497509, 10.879999957978725]。
- I3_concat_04_base／GI：spread=5.18%，process medians=[8.655999787151814, 8.36799992248416, 8.816000074148178]。
- I3_concat_04_base／R：spread=5.92%，process medians=[27.82400045543909, 29.504000209271908, 28.36799994111061]。

Formal成功／嘗試cell保存的warmup為60～260次，R啟用相同dispatch路徑後檢查三視窗穩定。

## Reference、參數與事件品質

S2審查264份raw P/R captures；formal 99組P/R，measured exact=3960，分母為預定99pairs×40=3,960視窗。P/R同device event signature一致只支持條件式映射，不證明submitted完整。
NVTX warning=264份；unmatched work API=834（全部來自三個Linear配置的cudaMemsetAsync，byte-count未知）；GPU start-before-associated-API-entry=3183，GPU end-before-entry=26，overlap pairs=157。這些計數含capture內transition與measured，沒有裁切／平移timestamp。
合併S1的recorded ownership：unassigned=0、multiple=0、noncontiguous=0、zero-recorded ops=0。zero-recorded不自動證明合法no-op；unknown提交參數不反推0-byte。
保存的5004筆R reference完整參數／tensor metadata中，與配置定義自動核對相符5004筆。介面不同／scalar／shape／stride不符需另審；原始資料與檢查明細保留在[R_PARAMETERS_CHECKED](analysis/summary/R_PARAMETERS_CHECKED.csv)。
同樣的Conv介面仍需groups等參數才能區分一般／depthwise；不能因ATen name相同就當同類。kernel names只用evaluator signature／診斷。P predictor輸出沿原numeric SAFE allowlist，不包含真實op／shape／reference邊界。
合併S1，P/R audit兩模式匯出的numeric allowlist格式逐筆檢查14,734個event records通過；部署predictor僅使用P。Overlap依事件類型、same/cross-stream與same/cross-iteration分開見[OVERLAP](analysis/summary/OVERLAP.json)。S2的157對重疊全部在GEMM anchor03、同stream且同iteration，沒有跨iteration重疊。沒有launch-attribute直接證據，PDL仍非已證實原因。
GPU域interval／API關聯與ownership合法時，依務實政策可作warning-accepted分類／CTX探索；CPU-GPU混合時間用途受限。缺少必要GPU activity、P/R多解或錯誤標籤的單位不得進受影響label。GI與trace/reference分別判定，低GI誤差或stable不能抵銷缺失。

依[DEVELOPMENT_USE](analysis/summary/DEVELOPMENT_USE.csv)的逐配置用途審查，核心36配置P CTX可在已知限制下development使用27個，limited_use=9個。特別是Linear等若有未匹配memset，沒有本capture直接byte-count證據，submitted-work維持unverified，不因其他Native 0B控制或P/R exact而解除。
## Signature與可支持的切分

本輪與S1共36個配置、12家族，各3個train anchors；沒有validation／test配置。實際R ATen參數／輸出metadata正規化後，跨配置共享component signatures=0組，詳見[R_COMPONENT_COLLISIONS](analysis/summary/R_COMPONENT_COLLISIONS.json)。Hash排除device、repeat、input bytes，避免用裝置或seed假裝新配置。
所有等價component目前都在train，不作held-out泛化或有效OOF主張。S3／Context收集前仍須跨完整候選pool做component碰撞與連通群稽核，不能把相同shape的repeats當新群。

## 測試與重現

原S1 collector／parser及protocol在本輪source與每批source_snapshot逐檔SHA核對，沒有修改收集語意；新增批次執行器的3項Smoke eligibility測試通過。收集完成後重跑新18＋A016＋O485＋legacy54項，證據見logs。
獨立CPU process重解析S2 raw 264份與P/R 132組，共11,970個產物逐byte相同，raw SHA未變。重現性不等於reference正確性。
S2原capture合計40.83MiB，SQLite 103.09MiB；所有source snapshots／audit／失敗資料另保留。外部GPU監測review cells=0。結束檢查僅見同使用者的RustDesk允許背景程序，未停止它，沒有本次worker遺留；見evidence/POST_PROCESS_REVIEW.json。guard與5秒sample均唯讀，不停止Pod／K3s／Netdata／RustDesk；sample間隔無法排除短暫外部活動。

## 下一步

先使用本輪36配置的per-family事件／時間／穩定性與reference缺口判斷補收優先，再實作Conv→BN→ReLU與Residual train尺度的D/I/S小型context對照。S3剩餘138配置另分批；S4完整120工作未執行，不直接啟動全矩陣。未訓練辨識器、N0或ITER。
新source與固定S2 assets打包並附SHA，見[BUNDLES](BUNDLES.json)。GitHub／RTX本輪未更新／執行；跨節點使用exact asset bytes與新協定，不能拿舊A0 3measured release冒稱S2 40×3。主方向仍4090→GB10／3090，分方向取有效交集。
