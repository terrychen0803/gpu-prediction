# 本專案的研究與分析約定

接手先讀 README.md、docs/CODEX_HANDOFF.md、docs/NEXT_EXPERIMENTS.md 與 inference_pilot/VALIDATION.md。使用繁體中文報告結果。使用者最新指示優先於舊方案。

- 研究設備為 GB10、RTX 4090、RTX 5090；四任務均為 inference，LLM 為 Qwen3-0.6B。現有程式僅支援 ResNet50 第一階段；不要把計畫中的功能描述成已實作。
- GPU 實測、合成 fixture、CLI planned_only 必須區分。缺資料時列出缺項，不產生假 latency、counter 或成功結果。
- 保存原始 session 與失敗資料。重新 parse 會寫入衍生檔，應使用副本；分析輸出用新的 analysis_runs/ 子目錄，保留來源 checksum 與分析 commit。
- Nsight parser 通過不代表無事件遺失。保留 manual review 狀態並核對 log／report；未知 schema 先調查，不以猜測 join 或補零繞過檢查。
- G 為無 profiler 的 host 完成時間；P0 目前是同步輔助分段。NVTX phase、operator/shape、reference 邊界為研究標籤，主 trace 模型不可自動納入。
- Predictor 只能看測試工作的來源 trace 與允許的設備資料；測試目標時間與 trace 留給 evaluator。訓練 supervision 使用訓練配置的目標 G 是合法的。
- 以工作配置分組切分所有設備、run、日期、裁切和方向配對。校正、特徵處理與調參只 fit 訓練 fold；兩階段模型使用 OOF 來源校正避免輸入品質不一致。
- 不把 kernel busy union 當全設備利用率，不從 Nsys trace 宣稱精確 FLOPs／DRAM 流量／occupancy。Compute／Memory／Launch 模型的係數須按證據解讀。
- 修改量測與 parser 時，補針對實際風險的測試並執行 `python -m unittest discover -s tests -v`；只修改文件時核對命令、欄位、連結與計數即可。
- 不默默更改 batch、精度、同步政策或驗收門檻。協定改變需記錄新版本，既有資料不可回填成新協定。
- 原始 trace、模型權重、results/、analysis_runs/ 不提交 Git。可追溯的程式、計畫與經整理的研究報告可依使用者授權提交。
