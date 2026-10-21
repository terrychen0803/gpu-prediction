# O1 受控算子 pilot（GB10 已實測，3090／4090 待測）

跨節點請先讀 [O1 節點手冊](../docs/O1_NODE_RUNBOOK.md)；GB10 的數字與未通過關卡見 [狀態摘要](../docs/O1_GB10_PILOT_STATUS_20261021.md)。本模組對應 protocol `o1_operator_pilot_v2_fixed_repetitions_20261021`，GB10 真實量測程式 hash 為 `aa513968c3a8cd9f1d9baf5c74c8119cdfdebc78a6f3d0863a8ff56fab064182`。版本 branch `o1-gb10-pilot-v2-20261021` 供其他節點 checkout，並應記錄實際 commit SHA。

目前實作九個固定 FP16 配置：GEMM、BMM、Conv2D 各三個。輸入為 SHAKE256 byte stream 經固定 FP16 LUT 轉換，逐 operand 保存原始 bytes 和 SHA-256；跨節點預期值見 [`pilot_input_sha256.json`](pilot_input_sha256.json)，須核對當地生成檔。GPU 正確性用 CPU FP32 operation 後轉 FP16 作 reference，候選門檻 `rtol=atol=1e-2`，並保存最大絕對誤差與相對 L2。

版本 `o1_operator_pilot_v2_fixed_repetitions_20261021` 的每個配置／模式／repeat 是一個獨立程序。所有模式固定每組 8192 次呼叫、五組；預熱至少 100 次、累積 100 ms，上限 20000 次／45 秒。單組約 100 ms 是目標，未達到時保留 `group_target_100ms_met=false`，不改寫數值。三次 repeat 的 run 內組中位數再取中位數；跨 run spread 門檻為 5%。

- `G0`：每次 out-of-place functional call 後同步，保存 host 完成時間；無 profiler。
- `GE`：每次 call 以 CUDA event 量 device elapsed，另存整批 event 平均作輔助；兩者不混成一個標籤。
- `P`：不含逐算子 marker/event 或逐算子同步的 Nsight `cuda-sw` trace；每組結尾同步。
- `R`：同 P，但每次 call 有 NVTX reference scope；以 CPU scope 包住的 CUDA launch API 再用 correlation ID 對應 kernel。不用 GPU 時間相交猜標籤。

P/R 分別有五個組尾同步。O1 使用已知整個 capture 範圍；`events_allowlist.jsonl` 是 P 的事件欄位草案，模型不可讀原始 SQLite、run 路徑、request、R reference 或品質診斷。這個 pilot 沒有 O2 的盲分段或可用 predictor 模型。Nsight 的可能遺失警告只記為 `manual_review_required`，即使已記錄事件全部可對應，也不宣稱完整 trace。

在可看見 GPU 的節點執行（需 CUDA PyTorch、Nsight Systems，輸出路徑必須全新；下例先收 G0/GE，完成當地品質審查後才按節點手冊收 P/R）：

```sh
python -m o1_operator run --out /path/to/new_o1_g0ge --expect-gpu 3090 --modes G0 GE --repeats 3
python -m o1_operator summarize --root /path/to/new_o1_g0ge --out /path/to/new_analysis
```

`--configs`、`--modes`、`--repeats` 可供 smoke。完整九配置 pilot 預期 `9×4×3=108` 個程序；在 trace 事件完整性問題解決前，不將 P/R 欄位判作逐算子可驗證正式資料。`run_inventory.csv` 保留所有失敗，`aggregate.csv` 只對三個有效 run 給出聚合，`rp_alignment.csv` 分開呈現策略不一致、路徑不一致及事件警告。

先前試作的 `o1_operator_pilot_v1_draft_20261021` 讓各模式依預熱速度選呼叫數，可能令 P/R 不可配對；它的原始 session 保留作診斷，不合併為 v2 資料。

