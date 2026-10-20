# E3 單設備來源時間校正

E3 用來源 GPU 的 P0 數值 trace 特徵預測**同一 GPU、無 profiler 的 G 完成時間**。目前 `python -m e3_calibration` 是保守的資料契約與 C0／C1 流程 pilot：每個 workload configuration 將三個獨立 G run 的 measured iteration 中位數再取中位數作標籤；P0 特徵先對每 run 取 iteration 中位數，再對三 run 取中位數。一次 configuration 是一個樣本，數百次 iteration 不是數百個獨立工作。G 標籤與 P0 特徵分別輸出；預測時只讀來源特徵，外層留一配置的 G 只由 evaluator 使用。

完整節點收集與交接見 [E3 節點手冊](../docs/E3_NODE_RUNBOOK.md)。現有三配置 E1 的 pilot 範例：

```sh
python -m e3_calibration --out analysis_runs/e3_pilot_local \
  /path/to/gb10_fixed_input_session \
  /path/to/3090_b1_session /path/to/3090_b8_b32_session \
  /path/to/4090_fixed_input_session
```

輸出 `source_features.csv` 保存來源 trace 特徵與必要的配對／切分鍵；**數值 predictor 的輸入僅限預先列出的 trace feature 欄位**，`label`、`config_id`、batch、resolution、UUID、protocol 與 checksum 只供分組與核對，不作 C0/C1 數值特徵。`g_labels_evaluator_only.csv` 與 `predictions_evaluator_only.csv` 是 evaluator 的目標／評估檔，`split_manifest.json` 按完整 batch×resolution 配置留出，`run_inventory.json` 保存每個輸入 run 的路徑與 checksum，`quality.json` 保留 trace 警告與 pilot 狀態。C0 直接用 `gpu_span_us`；C1 在每設備的訓練配置上以 log-linear 模型和固定 ridge 斜率先驗擬合。它們是流程基線，不把 P0 host latency、P1 phase 或測試 G 時間作輸入。下一版 C2 多特徵正則化與 E4 跨設備模型尚未實作。

舊 E1 每台只有三個配置，且 P0 警告事件可能未收齊。即使留一配置流程可執行，也不足以宣稱穩健泛化。擴充資料後仍須檢查 P0 事件品質；`quality.json` 不會把 parser 通過改寫成事件完整性通過。E3 不把 E2 microbenchmark 當 G 標籤，亦不修改原始 E1 session。
