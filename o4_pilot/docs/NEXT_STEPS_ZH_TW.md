# 這個工作包之後的建議

1. 4090／3090執行相同版本single→three→RN-A，交回完整raw與quality。GB10工程收集不等於兩方向模型完成。
2. 將NVTX警告、同stream重疊、CPU/GPU clock inversion分成獨立診斷；先以極小native forward重現，必要時做Nsight後端／版本對照。對照須保存新protocol或明示分析版本，不改舊raw。單純warning消失不足以驗收。
3. Tiny GEMM/BMM的GI三程序spread超5%，先檢查完整樣本與host提交／平台狀態；不批量重收O2、不換成batch-amortized GE、不挑最快程序。可另加較大原O1配置作新ID的工程對照，以評估短event量測極限。
4. 依RN-A盤點補BN eval、ReLU、add、pool/reduction、Linear bias等受控配置；先查實際融合與每op event範圍，建立新taxonomy與多配置群。不能把未知類硬套舊三類。
5. 先完成ResNet bottleneck→stage→完整forward。保留舊模型零調整CTX轉移與training-context校正兩張表；不相容、重疊、非連續或超L片段保留失敗。
6. 資料群足夠後，以OOF上游特徵訓練ITER；同分組比較SUM、ITER、DIRECT、W-B／W-BC／W-BCS。GI test由evaluator持有，先封存預測再評分。這批所有工程資料列development，不作未見confirmation。
7. 方法與品質門檻凍結後，另收新配置確認集；再依需要擴MobileNetV3、U-Net、YOLO26、Swin、Qwen prefill/decode。現有21展示配置不是充分訓練集。
