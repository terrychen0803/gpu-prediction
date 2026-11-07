# PR #7／#8 合併審查與 I3 發布

本次依使用者要求檢查兩個現有候選分支，**沒有執行merge或改寫main**。I3程式／計畫／資產交付追加到既有PR #8分支，沒有再開第三個工作分支。

## 可否合併

**工程整合可以合併。** 檢查時main `b204ae63ee2dff2bbc161bb5cc8f9aaecfb68c01` 是統一工程分支 `b28144163e1ff08fc22913f5f05e75bdb561eea0` 的祖先，後者又是PR #8原head `1711a5f1d27c282d640218611c01d9948c13cbd6` 的祖先。GitHub兩PR都回報mergeable=true／clean；本地`git merge-tree --write-tree`亦無衝突。

建議沿既有base順序：

1. PR #8（I3進度＋本次計畫／portable交付）合入`o4/unified-engineering-20261106`。
2. 再以PR #7把統一分支合入main。

如此不用retarget，仍只有兩個候選分支。推薦保留ancestry的merge commit；不為了精簡而重寫既有61個工程commits／歷史tag。另一選擇是先#7進main再retarget #8，但需重新檢查merge狀態。真正merge與branch刪除均未在這次「確認可否合併」中執行。

## 比衝突更重要的內容檢查

- PR #7不是純文件，包含O1/O2工具與O4 collector／parser／gate等136檔變更；已在隔離checkout執行root、O4、legacy、Native tests。無GitHub check runs／branch CI並不代表通過，實際本地測試記錄見 [發布驗證](O4_I3_PUBLISH_VALIDATION_20261107.json)。
- PR #7原body及部分凍結README仍有「GPU尚未收」等早期狀態，而較新的NODE_STATUS已有節點交付。原frozen source／manifest不為改說明而重建；repo README與此次I3入口明確導向最新狀態，PR body另補上歷史範圍說明。
- PR #8原本僅文件，這次已擴為可執行I3 A0交付，不能再描述成Markdown-only；原四份上傳文件逐byte保留，新入口說明後續發布。原相對連結仍可能指GB10本地evidence，不當作GitHub已有raw。
- 本次I3內嵌的155個source/test原檔與GB10已驗證包完全相同；其中141個O4基底檔也與統一分支相同。舊collector／parser／權重／raw不被改寫。
- 新發布的A0可用範圍是1process×3measured Smoke；174配置與依賴子圖、新40×3仍待實作。合併程式不代表這些新測試已完成。
- Reference strict verified仍為0；NVTX、clock／overlap、unmatched／missing ownership等限制仍需逐用途審查。這不阻擋工程工具整合，也不自動允許正式Reference主張或完整MLP訓練。
- 原diff有CSV CRLF與一處Markdown空白警告，屬既有交付格式，沒有改動原始報表來清除它們。本次新程式／文件的語法、關鍵連結、hash與離線計畫另驗證。

## 節點現在應用哪版

使用固定`o4-i3-a0-dev-v1-20261107` release tag及 [I3節點手冊](O4_I3_NODE_RUNBOOK_20261107.md)。它包含兩個候選分支的必要依賴，無需為開始A0而等待main合併。不要clone latest後又下載不同日期的資產；校驗release manifest。日後main整合完成，tag繼續指向這次確切版本，不移動。
