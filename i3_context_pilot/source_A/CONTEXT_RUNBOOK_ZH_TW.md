# Context D/I/S v1工程協定

S4-A只包含scale0的CBR與Residual共14工作，屬同一context_scale_0保守群，不是14個獨立配置群；沒有validation/test或模型訓練。所有Reference仍未verified。既有o4_i3_v2及parser保持原bytes，新package為o4_i3_context。

1. CPU建立共用base_assets：external inputs／各位置weights與BN state固定FP16 bytes；依實際BASE.json散布到其他節點，禁止各自重新seed後冒稱同bytes。
2. 每節點在GPU guard／monitor下、capture外執行本地derive，保存原D每位置實際FP16輸入／輸出。此處GPU是資產派生驗證，不是formal量測。不同GPU可有不同中間hash，不能假裝相同；本地I/S皆使用同一份本地D實際輸入。
3. D/I所有節點輸出保留到iteration後下次prepare清除，S保留該位置輸出到相同生命週期；prepare同步／清除在GI/iteration scope外。I斷data edge但保留相同stream和順序，不等於移除CUDA stream依賴，差異也受input placement/cache/lifetime/workspace影響。
4. 載入local asset時核對GPU／torch／CUDA／cuDNN；stack改變需新derived目錄與hash，不能在舊manifest上回填。
5. CBR5項／Residual9項分批，各先GI/P/R Smoke（8warm/1transition/3measured），審查correctness與warmup、monitor、工程trace gate和exact P/R後，才formal40×3（50～500warm到三窗口穩定、5transition、40measured；每mode三獨立process）。每批cap12。

```bash
export I3_PYTHON=/absolute/path/to/cuda-python
export PYTHONPATH="$PWD:$PWD/o4_pilot"
CUDA_VISIBLE_DEVICES='' "$I3_PYTHON" -m unittest discover -s tests_context -v
"$I3_PYTHON" -m o4_i3_context.assets generate-base --path /new/path/base_assets
# derive前自行保存GPU guard和monitor；只在確認無其他使用者compute工作後執行
"$I3_PYTHON" -m o4_i3_context.assets derive --base /new/path/base_assets/BASE.json --path /new/path/local_assets --expect-gpu GB10
"$I3_PYTHON" -m o4_i3_context.assets verify --path /new/path/local_assets/ASSETS.json
"$I3_PYTHON" -m o4_i3_context.collect --kind smoke --expect-gpu GB10 --assets-manifest /new/path/local_assets/ASSETS.json --config CTX_conv_bn_relu_0 IND_conv_bn_relu_0 ISO_conv_bn_relu_0_c ISO_conv_bn_relu_0_n ISO_conv_bn_relu_0_r --out /new/path/cbr_smoke_plan --plan-only
```

plan-only不是GPU實測；實測須新out移除plan-only。舊A0/Core36/S3 CLI不能收此DAG協定。R完整參數／operator位置／真邊界只供evaluator，P只有iteration marker和numeric allowlist，不逐位置標記或同步。所有raw、失敗、環境、命令、source snapshot、asset SHA保留；不改timestamp／刪overlap、不將unmatched API猜為合法0B。不升級工具鏈、不干擾既有服務。
