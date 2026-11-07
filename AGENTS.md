# 本專案的研究與分析約定

接手先讀 README.md、docs/O4_GPU_ITERATION_PLAN_ZH_TW.md、docs/O4_GPU_ITERATION_MATH.md、docs/O4_GPU_ITERATION_HANDOFF.md，再參考 docs/CODEX_HANDOFF.md、docs/NEXT_EXPERIMENTS.md 與 inference_pilot/VALIDATION.md 的歷史背景。使用繁體中文報告結果。使用者最新指示優先於舊方案。

- 最新主場景是 RTX4090 來源 trace → GB10／RTX3090，其他方向選配；5090非必要。六任務為ResNet-50、MobileNetV3、Swin-T、U-Net、YOLO26、Qwen3-0.6B，全部inference。從trace/metrics辨識算子邊界／類型，逐片段MLP預測再組合一次固定forward的GPU迭代時間；不能以整模型回歸取代主方法。
- 主線Git有E1～E3/O1工具與後續文件；節點O2/O3及q消融已完成探索，但全部程式／資料未收錄主線。先保護盤點本地E4/O2/O3，不能因主線缺模組覆蓋它們。O4第一個GB10工程工作包已完成，獨立實作在o4_pilot/；接手再讀docs/O4_NODE_ALIGNMENT_20261106.md及docs/O4_GB10_PILOT_STATUS_20261106.md。原21配置planned manifest仍不是現有CLI設定。舊O1重現依原runbook，新的O4依節點對齊清單及o4_pilot操作手冊，記錄實際collector commit與input hash。
- GPU 實測、合成 fixture、CLI planned_only 必須區分。缺資料時列出缺項，不產生假 latency、counter 或成功結果。
- 保存原始 session 與失敗資料。重新 parse 會寫入衍生檔，應使用副本；分析輸出用新的 analysis_runs/ 子目錄，保留來源 checksum 與分析 commit。
- Nsight parser 通過不代表無事件遺失。保留 manual review 狀態並核對 log／report；未知 schema 先調查，不以猜測 join 或補零繞過檢查。
- 新主標籤GI=`iteration_device_event_us`，無profiler的一次forward CUDA event elapsed；含interval內空隙，不是純GPU busy。舊G/G0仍是host完成時間、GE是isolated、CTX是profiler proxy，全部分開，不回填改名。P0舊版是同步輔助request分段。NVTX phase、真實operator/shape/reference邊界為oracle；只允許預先明示iteration scope。
- O 系列另讀 docs/OPERATOR_DATA_CONTRACT.md。允許已知 request capture 範圍，測試 op 邊界由模型推估；P 不插逐算子 marker/同步。R 去標記版與真正無標記 P 分開報；跨 run 對應多解時不可猜標籤。
- 使用者指定MLP。依2026-11-05 q消融，O4開發主候選為固定辨識器＋N0 scalar時間頭，再比較SUM與小型GI校正MLP；joint、q、T-v1 F/V保留對照，F不是必經。依docs/O4_GPU_ITERATION_MATH.md。未知／未覆蓋工作不得補零，時間與語意coverage分開；高softmax不是已校準unknown偵測。原受控L=4不可直接當DNN已支援。
- Predictor 只能看測試工作的來源 trace 與允許的設備資料；測試目標時間與 trace 留給 evaluator。訓練 supervision 使用訓練配置的目標 G 是合法的。
- 以工作配置分組切分所有設備、run、日期、裁切和方向配對。校正、特徵處理與調參只 fit 訓練 fold；兩階段模型使用 OOF 來源校正避免輸入品質不一致。
- O 系列還須將等價配置、混合序列模板分組，辨識器/類型 gating 也用 OOF。主結果為4090→GB10／3090，其他方向選配；已知設備預測與留一設備測試分開。留出設備工作負載不可參與 fit，獨立硬體校正可用。
- 不把 kernel busy union 當全設備利用率，不從 Nsys trace 宣稱精確 FLOPs／DRAM 流量／occupancy。Compute／Memory／Launch 模型的係數須按證據解讀。
- 修改量測與 parser 時，補針對實際風險的測試並執行 `python -m unittest discover -s tests -v`；只修改文件時核對命令、欄位、連結與計數即可。
- 不默默更改 batch、精度、同步政策或驗收門檻。協定改變需記錄新版本，既有資料不可回填成新協定。
- 原始 trace、模型權重、results/、analysis_runs/ 不提交 Git。可追溯的程式、計畫與經整理的研究報告可依使用者授權提交。

- 新 O4 節點統一入口為 docs/O4_QUICKSTART_ZH_TW.md；固定release tag與full source/asset index。歷史分支以同名封存tag保存，`origin/<歷史名>` 改為 `refs/tags/<歷史名>`。不要在節點重建release manifest來繞過驗證。

- 2026-11-07 使用者調整：優先收集有用、可追溯資料，不要求每台設備的每個模型／配置／環境或進度嚴格一致。可先完成可執行的子集，失敗留原因後繼續；跨設備評估時才分方向檢查可比較交集。依 docs/O4_COLLECTION_POLICY_20261107.md；原品質判定、資料隔離與舊資料保存不因放寬收集對齊而自動解除。
