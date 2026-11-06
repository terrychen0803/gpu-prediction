# O4 工作包後續建議（GB10 v3 Trace 診斷更新）

目前 GB10 GI/P/R 與 RN-A 盤點屬 engineering pilot，**不等於已完成正式 Reference 或完整跨 GPU 時間模型**。見 [GB10 品質修訂摘要](../../docs/O4_GB10_TRACE_QUALITY_UPDATE_20261106.md)。保留所有 O1/O2/O3/O4 原始資料、原 quality 與凍結協定，另存唯讀重新分析結果。

1. **I1 先解決 Reference 資格，不要求消滅合法 Overlap**：同一 default stream 已觀測 123 對跨 ATen kernel-kernel overlap，與 PDL 相容但欠缺 launch attribute 證據；並有 270 個成功卻無 GPU activity 的 memset API、GPU/CPU clock inversion 及 NVTX warning。先在 GB10 按本地最小控制計畫做 serial/no-PDL、64B／0B memset；每節點 SW／auto × P/R 四個 native captures（各 8 measured），核對 actual backend/fallback。這些控制程式目前還沒加入本 GitHub，不能假裝可直接執行。
2. **分開品質 gate**：recorded_event_integrity、submitted_work_coverage（API→GPU reverse）、reference_semantic_alignment、timestamp_consistency、event_index_partition_supported、nonoverlap_sum_supported、overlap_aware_closure_supported、strict_reference_verified。未知 no-op、NVTX warning 與 clock 異常保留 unverified；不要以低 GI 誤差或 P/R Signature 一致升級正式標籤。
3. **RTX4090／3090**：相同 GitHub O4 程式與共用資產 hash，先 single smoke；核對最小控制結果及品質 gate 後，才展開 three→RN-A，每個新 run 獨立保存 raw、warnings、quality、collector／environment hashes。不同 GPU 不必有相同 kernel 實作。
4. **短 GI 穩定性**：GB10 tiny GEMM／BMM 三程序 spread 超過 5%，先核對全樣本及 host launch／GPU 狀態，不能挑最快程序、刪除慢值或用 batch GE 取代 GI。
5. **I2／I3**：event-index 連續唯一分段與 wall-clock Overlap 分開；RN-A 的 BatchNorm、ReLU、Add、Pooling、Linear、額外 shape/context 與大於 L=4 的 Conv 仍需受控資料。保留原 O3 權重、normalizer 作零調整 baseline，新增 taxonomy／feature 另訓練新版本。
6. **I4**：ResNet bottleneck→stage→full forward；分別驗證 blind segmentation、typed F1、CTX 轉移與 unknown/超 L coverage。Reference 未驗證時只能作 warning-accepted／探索性結果。
7. **I5**：使用 union-based P-span 診斷 U = ΣCTX spans + positive gap − union-overlap；它不是無 profiler GI。以同 GI、相同配置群 split、分組 OOF 比較 T-SUM、T-ITER Base、**T-ITER Overlap（僅來源 4090 P features）**、DIRECT、W-B/W-BC/W-BCS。目標 test overlap、GI、trace、reference 僅 evaluator 使用；不能用 ITER 的低誤差掩蓋 Reference／語意失敗。
8. **I6／I7**：兩方向分開評估 GI MdAPE、P90、coverage、typed/semantic coverage 與 failure denominator；目標 overlap 由 evaluator-only 作事後分層分析。模型與 criteria 凍結後另外收新確認集，再擴至其他 DNN 與 Qwen prefill／decode。

**研究主線不變**：4090 P Trace → Operator Segmentation／Classification → N0 CTX Time MLP → O4 ITER MLP → GB10／3090 的無 profiler GI。修改的是品質 gate、Operator 事件／時間假設以及 overlap-aware ITER 消融，不是直接重做所有資料或取消辨識器。
