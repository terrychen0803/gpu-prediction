# GB10 I3 S0＋S1：新配置 40×3 工程收集結果

2026-11-07（Asia/Taipei）。依前一輪具體收集計畫，完成新collector、三sentinel Smoke與40×3；其餘配置／context未啟動。所有資料寫新run，舊O1/O2/O3/O4 raw／模型／結果保留。

## 實際完成與四種狀態

| 狀態 | 本輪證據 |
|---|---|
| GPU實測 | BN／GEMM／Bilinear Resize各GI/P/R×3獨立process×40，27cells／1,080measured；另9個Smoke cells／27measured，總36cells／1,107measured |
| 工程 | 12家族174配置的isolated adapter／CPU正確尺寸與無mutation、固定layout值、新40×3、active R warmup、source snapshot與guard／monitor、報告及可攜包 |
| 僅規劃 | 另外171配置的GPU、S2核心36擴充、依賴DAG/I/S收集、RTX此新協定、I4／N0／ITER訓練 |
| Reference | 所有NVTX／clock限制保留，strict verified=0，formal_training_eligible=false；只作依用途審查的development |

## GI與跨程序穩定性

每個模式先取process內40次median，再取3個process median的median；spread=(max−min)/median。GI是無profiler整個單算子控制工作的event elapsed；不是完整DNN GI，也不是P CTX。P/R event時間另外報，不能當無profiler真值。

| 配置 | GI µs | GI spread | P event µs／spread | R event µs／spread |
|---|---:|---:|---:|---:|
| I3_batchnorm_eval_00_base | 19.872 | 0.24% | 27.552／0.41% | 51.984／0.68% |
| I3_gemm_00_base | 11.408 | 1.96% | 15.280／0.52% | 32.528／1.97% |
| I3_bilinear_resize_00_base | 15.616 | 0.51% | 17.120／1.21% | 35.104／1.46% |

九組formal mode中9組spread≤5%、0組review；不刪樣本或選最快repeat。數值正確性每cell通過原定relative-L2與scaled-max各2%門檻；詳細在[CONFIG_METRICS](analysis/final/CONFIG_METRICS.csv)。
Formal暖機60～140次；R在暖機時啟用完整dispatch路徑。每個cell使用新的OS process/CUDA context，36個不同worker PID；guard／每5秒host活動sample未終止任何他人工作。監測問題數=0，採樣間隔內的外部活動不能由此排除。GB10 memory欄位N/A照實保存。

## Reference與Trace品質

共24份cuda-sw P/R captures、12組P/R pairs；formal 9組×40=360個measured視窗exact，加smoke共369。每個R視窗只有預定一個ATen operation，共417份完整schema／參數／tensor stride與輸出shape已逐筆對照frozen config。所有已記錄event唯一連續歸屬；unassigned=0、multiple=0、noncontiguous=0、unmatched work API=0。
NVTX warning仍有24份；GPU start早於CPU API entry=502，GPU end早於entry=123，temporal overlap pairs=50。原timestamp全部保留，沒有offset／裁切／移除；具體例子見[ANOMALY_EXAMPLES](analysis/final/ANOMALY_EXAMPLES.json)。
50對重疊全部在formal GEMM同iteration同stream的split-K／reduce兩kernel間（P47／R3），沒有跨iteration重疊。實際launch attributes未量測，PDL仍未證實；詳見[CLOCK_DISTRIBUTIONS](analysis/final/CLOCK_DISTRIBUTIONS.json)與[OVERLAP_CLASSIFICATION](analysis/final/OVERLAP_CLASSIFICATION.json)。
本批必要API皆有已記錄對應、GPU interval合法、P/R在同device同工作上exact，支持探索性boundary/class／P CTX用途；CPU/GPU clock mapping用途仍受限。這不能反推歷史RN-A已修復，也不能由NVTX scope數、signature、union閉合或這次三個控制升級strict Reference。單stream假設由current-stream檢查加同配置raw observed streams審查支撐；未profile的GI本身不提供額外隱藏stream證明。
用途採[DEVELOPMENT_USE_REVIEW](analysis/final/DEVELOPMENT_USE_REVIEW.json)：GI為穩定工程標籤；GPU域CTX／event-index分類可warning-accepted探索，CPU/GPU混合時間用途受限。後續分析須另列GEMM及clock異常敏感度，不能用低GI誤差解除警告。
P numeric只沿SAFE allowlist；R真實類型／shape／參數只供evaluator。完整source catalog雖有174配置，實際每family只有一個本輪signature，無有效train/validation/test分組、跨設備配對或新模型泛化結論。

## 測試、重現與成本

173項CPU tests通過（新18、A0 16、O4 85、legacy54）。第一次legacy測試的cwd錯誤導致import失敗，原log保留，切至正確legacy根目錄後54項全通過；沒有改legacy程式。原155個交付檔不變，新配置catalog另凍結。兩份既有BN raw重播29個產物byte一致。
本輪24份raw與12組pair由獨立CPU process重解析，共1086個產物逐byte一致，raw SHA前後不變。這證明可重現，不能證明reference無遺失。source在每cell前後檢查；assets在每cell前及worker載入時校驗，從獨立snapshot執行；收集期間沒有修改worker來源。
本輪capture約3.70MiB、SQLite約9.48MiB；formal collect命令合計77.9s、export0.5s，未包含全部guard／CPU audit。三個小工作估計的P90單工作命令時間27.4s，下一批12工作加25%餘裕約6.9分鐘命令成本；樣本極小且不含大shape／失敗，不作完整S2工期保證。詳見[COST](analysis/final/COST.json)。

## 交接與下一步

已備妥[BUNDLES](BUNDLES.json)列出的新source與8個operand資產包，逐member校驗。這批新protocol尚未上傳GitHub，不能使用先前A0 release假裝可跑此40×3；RTX亦未執行。操作見[source runbook](source/I3_V2_RUNBOOK_ZH_TW.md)。
下一步先檢查以上穩定性與成本，再擴S2的36個train配置（扣除已收三個），依12工作上限分批；同時才挑CBR／Residual train尺度做小型context工程。若R有review，先檢查instrumentation／host成本，不重跑挑最佳值。新資產與協定送RTX時保持exact bytes，軟體堆疊差異分層記錄，不要求三台一次全跑齊。辨識器／N0／ITER仍未訓練。
