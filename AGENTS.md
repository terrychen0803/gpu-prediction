# 本專案的研究與分析約定

接手先讀 README.md、docs/CODEX_HANDOFF.md、docs/NEXT_EXPERIMENTS.md 與 inference_pilot/VALIDATION.md。使用繁體中文報告結果。使用者最新指示優先於舊方案。

- 本輪設備為 RTX 3090、RTX 4090、GB10；5090 留待擴充。四任務均為 inference，LLM 為 Qwen3-0.6B。最新目標是從 trace/metrics 自動辨識算子邊界與類型，再逐算子跨設備預測並組合 request 服務時間。不能以整模型回歸取代此目標。
- 現有 Git 有 E1～E3 與 O1 九配置 pilot 程式；GB10 已完成 G0/GE 時間量測及受限 P/R trace 診斷。O2～O5 尚未實作，不把 O1 單算子資料稱作無標記混合序列辨識。使用者回報 E4 在節點完成，但目前 Git 尚未收錄，接手先保護並盤點本地 E4。其餘節點按 docs/O1_NODE_RUNBOOK.md 使用固定版本 branch、記錄 commit SHA 與 input hash。
- GPU 實測、合成 fixture、CLI planned_only 必須區分。缺資料時列出缺項，不產生假 latency、counter 或成功結果。
- 保存原始 session 與失敗資料。重新 parse 會寫入衍生檔，應使用副本；分析輸出用新的 analysis_runs/ 子目錄，保留來源 checksum 與分析 commit。
- Nsight parser 通過不代表無事件遺失。保留 manual review 狀態並核對 log／report；未知 schema 先調查，不以猜測 join 或補零繞過檢查。
- G 為無 profiler 的 host 完成時間；P0 目前是同步輔助 request 分段。O 系列的 isolated GE、contextual trace proxy、request G0 分別評估，不可混算。NVTX phase、真實 operator/shape、reference 邊界為 oracle，主 trace 模型不可納入；從合法 trace 推估的類型可作模型輸入。
- O 系列另讀 docs/OPERATOR_DATA_CONTRACT.md。允許已知 request capture 範圍，測試 op 邊界由模型推估；P 不插逐算子 marker/同步。R 去標記版與真正無標記 P 分開報；跨 run 對應多解時不可猜標籤。
- 使用者指定MLP為主模型。2026-10-28 修正後的主設計見 docs/T_V2_JOINT_MLP_MODEL.md：共享表示、分段／類型頭、直接目標時間頭，joint loss；獨立 F 並非主方法必要模型。docs/T_V1_MLP_MODEL_SPEC.md 的 F/V 分解保留為對照。首版unknown採拒絕，GE與CTX時間標籤分開；不要把設計或合成數值檢查說成已訓練GPU模型。
- Predictor 只能看測試工作的來源 trace 與允許的設備資料；測試目標時間與 trace 留給 evaluator。訓練 supervision 使用訓練配置的目標 G 是合法的。
- 以工作配置分組切分所有設備、run、日期、裁切和方向配對。校正、特徵處理與調參只 fit 訓練 fold；兩階段模型使用 OOF 來源校正避免輸入品質不一致。
- O 系列還須將等價配置、混合序列模板分組，辨識器/類型 gating 也用 OOF。已知設備六方向與留一設備測試分開；留出設備工作負載不可參與 fit，獨立硬體校正可用。
- 不把 kernel busy union 當全設備利用率，不從 Nsys trace 宣稱精確 FLOPs／DRAM 流量／occupancy。Compute／Memory／Launch 模型的係數須按證據解讀。
- 修改量測與 parser 時，補針對實際風險的測試並執行 `python -m unittest discover -s tests -v`；只修改文件時核對命令、欄位、連結與計數即可。
- 不默默更改 batch、精度、同步政策或驗收門檻。協定改變需記錄新版本，既有資料不可回填成新協定。
- 原始 trace、模型權重、results/、analysis_runs/ 不提交 Git。可追溯的程式、計畫與經整理的研究報告可依使用者授權提交。
