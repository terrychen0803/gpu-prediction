# Signature、資料隔離與可支持的結論

目前只有**設計候選分組**。資產、實際ATen分解與跨模板碰撞審查尚未完成，不能稱已凍結有效train/validation/test。GB10採先train範圍工程驗證，test只在schema／policy定案後收集並封存；不以目標test時間修正配置選擇。

## 分開三種識別碼

1. **work_definition_signature**：運算語意、完整operand shape／stride／layout／dtype、scalar參數、state、output、拓撲、forward邊界與semantic backend（eager/fused）。不含GPU型號、mode、repeat。JSON中的semantic hash是尚未包含資產與runtime補充的設計hash，不能拿來當最終身份。
2. **asset/execution_signature**：input／weight／state bytes、generator/version、source snapshot、實際runtime／trace backend／GPU等。相同logical工作不同stack可分層比較，不要求kernel名稱一致；變更外部bytes或範圍要新variant。
3. **leakage_group**：保守的資料切分群。同一anchor的batch、stride、layout、bias、eps、groups等控制一起放；日期／GPU／seed／repeats／P views／派生context不增加群數。GPU／input SHA／Nsight版本不能用來把同一shape拆出一個新test。

跨GPU leak check排除tensor device欄位；不可因CUDA device ID不同誤判配置不同。即使backend不同亦保守同群；backend轉移是獨立問題。

## B：未見配置（主要配置切分）

各核心family 12個anchors，預定index：train `{00,03,04,07,08,11}`；validation `{01,06,09}`；exploratory test `{02,05,10}`。因此每family6/3/3；不是按尺寸前半訓練後半測試的純外推切法。anchor00所有controls同train群，不可用controls湊test數。首批只收00/03/04，**沒有獨立test結果可報**。

Context以scale0/1/2分train/val/test；同scale六模板＋所有I/S控制暫合成一個保守群，避免CBR與Residual共享Conv／BN／ReLU跨split。它只有三個context大群，不能做五折OOF或有力的context統計結論。不要用18個DAG ID當18個獨立樣本。

凍結前執行兩層碰撞稽核：

- 以所有預定atomic定義／實際R完整參數建立equivalence keys，忽略input bytes、op位置、設備、權重值等不足以證明未見配置的欄位；保留shape、stride、layout、dtype、groups、op參數與semantic execution粒度。相同與保守近鄰簽章跨split列出。
- 對同源controls、context所有component、isolated counterparts連邊，計connected components。若碰撞把train/val/test連在一起，先合併群或重設候選配置、重新產生manifest，再開始test收集。若群數不足，交付development-only，不偽造三群／五fold。shared scalar-like操作仍要分辨完整tensor shape，不能為了拆群忽略共同operator。

對已收A0及三設備七A歷史development也做overlap清單：新B test若被過去訓練使用的相同配置覆蓋，就不能宣稱對那個模型「未見配置」。保守方案是本輪from-scratch controlled-only B訓練，歷史資料只作另外的預訓練實驗；兩者不混稱。

## A：已見配置、新上下文

獨立的A manifest允許同signature：只用train尺度的isolated positions訓練／校正，再對未用於訓練的dependent拓撲評估。初版可用CBR作context validation，其餘五模板作探索性context test；其isolated配置已見必須明示。這些template共用尺度／算子，不視為互相獨立；指標以模板列出而非產生虛假的狹窄CI。

若已用某個D圖調特徵、L、時間loss，就不能再稱它對該模型是未見context。A與B各自有模型／normalizer／manifest，禁止A的seen-shape訓練資料混進B test。I/S/D的不同bytes與context差異不得自動形成新的B group。

## C與確認集

六模型transfer仍另建C。現有七A已反覆查看，都是development；QP/QD在「模型留出」上屬同Qwen家族。這份新controlled計畫不宣稱預留全新六模型confirmation。I7需要方法與選模規則凍結後另外產生配置／新日期證據，且未拿該test結果調整。

## Predictor隔離與統計

P-only資料提供numeric events、合法來源觀察及獨立硬體校正。config ID只join，不進模型；R/op type/shape/真實segment數、目標test P與GI均在evaluator分區。產生／檢查R用於標籤不等於允許predictor讀它。

source view預設第8個measured作基線，2～8統計為另候選；只在train/val選定。多view correspondence必須P-only可成立，不能用R拼features。40-request capture成本完整報，取7筆不等於實際只profiling7次。

每方向取4090來源與各target有效配對交集，不能強制三GPU交集刪掉單方向有用資料。以configuration/group先聚合40×3，再做群層配對分析；iteration、GPU、seed不能當獨立樣本。每family3個test groups只是探索起點，報逐群點與範圍；是否需要更多群，依train/validation learning curve、coverage、區間精度另定，不用test成績挑收集樣本。

只有通過runtime signature audit與實際可用group數檢查後才能決定OOF fold數。Context現有3群不足五折；需要OOF時先用獨立上游訓練／下游校正分離或擴群，不能借test。本文沒有啟動任何模型訓練。
