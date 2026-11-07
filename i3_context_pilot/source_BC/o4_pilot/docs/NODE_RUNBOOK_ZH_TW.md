# O4 三節點操作手冊

日期：2026-11-06。主方向為4090→GB10／3090。此版完成量測與inventory工具，沒有完成跨設備DNN模型。GI/P/R都是實際GPU模式；`--dry-run`不啟動CUDA，只列計畫。

## 1. 分工與版本

GB10完成collector開發、當地GI/P/R及RN-A運算盤點，後續集中分析。4090提供主來源P，3090與GB10提供target訓練／評分資料。三台都收GI/P/R，test target資料不得提供給predictor。

分發完整node bundle，含同一份程式、manifest、三類輸入、RN-A權重與既有B32輸入檔。不可各自在CPU重生隨機輸入；SHAKE固定算子輸入可重現，但本輪仍以bundle bytes/checksum為準。`manifests/assets.json`所有路徑相對工作區，不需沿用GB10絕對路徑。

本版live collector支援Linux／POSIX（process group與libcudart.so finalization）；Windows live collector尚未實作／驗證。若其他節點為Windows，先記錄為pending_platform，不直接套用bash流程。離線解析為Python標準函式庫。

每台可解壓至新的O4資料夾；不要蓋掉既有O1/O2/O3。使用該機已有、支援GPU的CUDA PyTorch/TorchVision與Nsight。先記錄版本差異，不在未驗證前安裝／升級或變更driver。GB10/RTX軟體不同時結論是設備＋執行堆疊轉移。

```bash
export PYTHON=/absolute/path/to/cuda-python
"$PYTHON" scripts/verify_assets.py
"$PYTHON" -m o4_iteration --help
"$PYTHON" -m unittest discover -s tests -v
(cd vendor/legacy && "$PYTHON" -m unittest discover -s tests -v)
bash -n scripts/run_node.sh
```

GB10已使用Python 3.12、torch2.13.0+cu130、torchvision0.28.0+cu130、Nsys2026.4.1及driver580.159.03；這不是要求RTX盲目安裝同版。此版預設`cuda-sw`trace；當地若不支援，先保存錯誤/help，核對`--trace-backend cuda`後以新run明示後端差異，不默默fallback。worker會核對GPU名稱與只有一個CUDA-visible裝置。

## 2. 三階段執行

先執行single，核對reports與失敗；通過後執行three，再執行resnet。每次都建立新的UTC命名run，拒絕覆寫；不會修改K3s、Pod、Netdata或既有服務。

4090：

```bash
bash scripts/run_node.sh 4090 single
bash scripts/run_node.sh 4090 three
bash scripts/run_node.sh 4090 resnet
```

3090：把參數4090換成3090；GB10換成GB10。`single`是GEMM×GI/P/R×1程序，共3cell；`three`是三類×三模式×三獨立程序，共27cell；`resnet`是RN-A×三模式×三程序，共9cell。各程序40 measured＋5 transition，warmup另列。這批與六任務21配置567cell矩陣分開。

只列計畫、不量測：

```bash
"$PYTHON" -m o4_iteration run --out analysis_runs/o4_planned_new_id --expect-gpu 4090 --dry-run
```

直接指定模式／配置例：

```bash
"$PYTHON" -m o4_iteration run --out analysis_runs/o4_rn_gi_new_id --expect-gpu 4090 --workload RN-A --mode GI --repeats 1
```

raw下的request、command、worker、export log與Nsight檔必須一起保留。summary列所有attempt；failed／not_started不隱藏。失敗可以用新run重做，舊失敗不可刪除。每cell前檢查compute processes，無法確認空閒即保留失敗；前後快照不等於連續背景負載保證。

## 3. 時間與reference語意

GI是無profiler、單次完整forward的CUDA event interval。events在capture前初始化；start→native forward→end→end.synchronize；ms乘1000存µs。沒有逐op同步。host forward+wait另外存，不當GI。

P與R使用相同iteration event完成政策，表中的elapsed欄是`profiled_iteration_event_us`，不是GI。P只有iteration NVTX，沒有op標記。R用TorchDispatchMode攔截ATen leaf，包含module內與functional運算，避免父module重複計數；不是Python原始層名恢復。零GPU view存null span，不補成0時間標籤。ATen有融合或多kernel，必須按實際事件清單解讀。

R較大的Python/NVTX開銷可能改變提交空隙，故R時間不能當無干擾真值。P/R按已知iteration、完整有序signature作條件映射，無法比對即不轉標籤；無duration猜測。此signature只在evaluator診斷使用，P數值匯出不含kernel名稱。解析與閉合通過仍不證明無事件遺失。

初始化時warmup最少50，最後三個20筆視窗相鄰中位數差≤5%；因此最早60次通過，上限500。固定5 transition、40 measured。3程序中位數spread≤5%僅作穩定gate，未通過保留所有值。

計時外正確性：小算子與CPU FP32後cast比較，rtol/atol=.01；RN-A對CPU FP32 native forward的relativeL2≤.01與maxabs/max(outputscale,1)≤.01，R instrumentation另檢查FP16輸出exact相同。這是工程tolerance，不代表分類任務accuracy。更多數值診斷保存在quality。

GI的單stream假設由相同adapter的P/R完整GPU工作stream盤點支援，GI本身無trace不能直接觀察side-stream；若P/R觀察多stream則unsupported。首版不處理多stream join。

Profiler完成時釋放本worker的event／tensor、同步、等待1秒，再reset本程序自己的CUDA context並stop；不是`nvidia-smi --gpu-reset`，不動其他程序。這沿用GB10既有finalization，其他平台須先single smoke驗證。任何警告都保留。

## 4. 回傳資料

每台回傳完整`analysis_runs/o4_*`，以及`manifests/`、`artifacts/preflight/`（如有）、`git rev-parse HEAD`與程式release hash。權重與輸入已共有，可用checksum核對而不重傳。不要只交aggregate.csv。

主要輸出：

- `inventory.json`：全部計畫／attempt／失敗、實際collector commit、程式hash。
- `raw/cell_*/`：原始report/SQLite、完整命令、環境、樣本、warmup、correctness。
- `quality/cell_*/`：diagnostics、stream/scope/reference審查、原始格式事件。
- `quality/cell_*/predictor/<opaque>/`：每個P measured iteration的純數值events/scope；主predictor只讀此目錄。
- `reports/`：GI／profiled event分表、process spread、R→P對應、運算清單與family涵蓋率。
- `reports/evaluator_only/`：ATen類型、shape與轉移到P的參考片段；不可送入盲predictor。

不要把整個quality或raw當predictor輸入，這些包含oracle與診斷kernel名稱。尚未建立train/validation/test或新N0預測；這批資料皆為development工程pilot。

## 5. 前進條件

GPU correctness、計時完整性、repeat穩定、trace結構、reference可信度分開列狀態。NVTX warning仍在時保留warning_accepted；即使warning消失，也需人工審查log/report、scope、API correlation及R→P對應才能提升reference狀態。此工具不自動設strict verified=True。

RN-A inventory後按缺失類別、P事件及busy比例補資料，再進block→stage→完整forward。舊三類family相符不代表舊shape／bias／layout模型適用；舊L=4需要重新盤點。未知／未覆蓋不補零。不啟動567cell全矩陣、不在此階段追加q／joint大搜尋。

## 6. RN-A 解析觀察與限制

實測Nsight RN-A有少量GPU end早於對應CPU API start（第一份最大約2µs），parser以process限定的唯一correlation ID關聯，不以CPU/GPU先後時間過濾已唯一的ID，也不平移timestamps。此現象另列clock inversion診斷。

同一observed stream仍記錄部分GPU intervals重疊；目前原因未識別，不能宣稱是真實硬體並行或純工具誤差。`o4_audit_v2_overlap_explicit`保留所有事件、按union盤點busy，標記`composition_supported=false`，不套非重疊SUM恆等式或舊semi-Markov模型。這不妨礙保留GI、ATen清單與事件盤點，但正式上下文加總／零調整轉移仍需另過gate。跨類別busy union可能重疊，比例和不一定100%。
