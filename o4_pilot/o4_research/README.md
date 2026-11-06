# O4 I1／I2 development tools

本套件完成GB10 software-only reference診斷與七個A配置工程smoke；**未完成正式Reference或MLP訓練**。原始o4_iteration／O1/O2/O3結果與協定不被替換。模型／shape／R標籤只在evaluator，P輸出只含允許數值欄位。

## 使用前

先讀repo的`docs/O4_I1_REFERENCE_I2_ADAPTER_STATUS_20261106.md`與`docs/O4_I1_I2_NODE_HANDOFF_20261106.md`。PR #1已由外部合併，當時Draft規劃hash與實測版本保留。Linux/POSIX已在GB10執行；RTX4090／3090仍待當地smoke，Windows live未實作。所有out參數必須是新路徑；保留失敗。

既有CUDA torch可用時，可在新venv加裝`requirements-i1-i2.txt`；不要修改既有環境、driver或系統服務。GB10實測torch2.13.0+cu130／vision0.28.0+cu130，其他stack要記實際版本，不假裝硬體差異可與軟體差異分離。

## I1直接參數觀測（不可在Nsight下執行）

從o4_pilot目錄：

```bash
bash scripts/build_memset_observer.sh /absolute/new/path/libmemset_observer.so
python -m o4_research.observe_memset --case control --expect-gpu 4090 --out /absolute/new/control \
  --library /absolute/new/path/libmemset_observer.so
python -m o4_research.observe_memset --case resnet --expect-gpu 4090 --out /absolute/new/rna_params \
  --library /absolute/new/path/libmemset_observer.so --assets-manifest /shared/frozen_A/ASSETS.json
```

先核對control直接讀到64／0與成功exit，再執行RN-A。若找不到cudart，明示`--cudart-library /absolute/path/libcudart.so`。CUPTI subscribe失敗就停止；不能與Nsight或另一個subscriber硬併。只在callback內讀取functionParams，拷貝count／return code，不保存GPU pointer。SDK觀測本身不產生GPU activity時間，也不能證明Nsight完整。

## I2七個A配置

取得GB10凍結的完整assets，先核對published manifest hash，再驗證所有bytes：

```bash
python -m o4_research.verify_assets --manifest /shared/frozen_A/ASSETS.json \
  --expected-manifest-sha256 55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4
python -m unittest discover -s tests -v
(cd vendor/legacy && python -m unittest discover -s tests -v)
python -m o4_research.run_smoke --out /absolute/new/rna_plan \
  --assets-manifest /shared/frozen_A/ASSETS.json --workload RN-A --expect-gpu 4090 --plan-only
python -m o4_research.run_smoke --out /absolute/new/rna_smoke \
  --assets-manifest /shared/frozen_A/ASSETS.json --workload RN-A --expect-gpu 4090
```

確認RN-A與I1控制結果後，另開新out執行`--workload MV-A SW-A UN-A YO-A QP-A QD-A`。3090換expect-gpu；不以GB10時間或kernel名字驗收。只有cuda-sw路徑，沒有silent backend fallback。每A×GI/P/R×一程序，8warmup＋1transition＋3measured；這是新development協定，不能冒充40筆／三程序穩定性pilot。

`prepare_assets`／`download_assets`僅用於建立新資產版本；其他節點應複製GB10同一份ASSETS，不當地重新初始化U-Net、輸入或token。UN-A只用initialized_performance_only；不宣稱分割品質。YO-A使用native one2one tensor輸出，包含原生head內部decode/topk，仍計算原生auxiliary one2many分支；未做backbone fusion或外部NMS。Qwen固定eager attention、DynamicCache、last-position logits。Decode每次計時外重建128-token cache，不把連續generate混成同配置。

P/R明示`o4.prepare/...`範圍，計時外GPU準備不丟棄也不併入GI。`audit.py`唯一correlation映射、反向API審查與v4八項gate；zero／unknown工作不補時間。只有完整同設備ordered signature相同時，才條件轉移R event indices到P，並保留warning-accepted。

## 僅重解析／盤點

```bash
python -m o4_research.replay --runs /data/rna_smoke /data/other_A_smoke --out /absolute/new/replay
python -m o4_research.inventory --runs /data/rna_smoke /data/other_A_smoke --out /absolute/new/inventory
```

重解析不啟動GPU。Inventory family只是evaluator診斷分組，不是已訓練Classifier；zero recorded work不自動證明真正no-op。任何warning、clock異常與未解API均保留。未見設備泛化、正式辨識器／N0／ITER／OOF及新確認集都未完成。

## Native控制

repo的`native_control/`保留原始CUDA MRE與完整known-work審查；新entrypoint預設只做software P/R。舊四capture手冊另存historical，不執行已無效的GB10硬體flush／drain重試。
