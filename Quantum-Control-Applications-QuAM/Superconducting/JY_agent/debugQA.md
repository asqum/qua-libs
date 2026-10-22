# JY_agent debug Q&A / 排錯問答

這份文件只處理 client、MCP、Dashboard 與 tunnel 的啟動問題。診斷不會建立
measurement workflow、不會執行實驗，也不會修改 `state.json`。

排錯時固定由外往內分成五層，避免把不同問題混在一起：

1. client 是否讀到並信任 project config；
2. client-owned STDIO MCP 是否完成 `initialize` 並列出 `jy_*` tools；
3. loopback MCP HTTP／Approval HTTP 是否健康；
4. public Dashboard tunnel 是否健康；
5. 最後才是 measurement workflow／worker／hardware。

## Q1：Codex 出現 `required MCP servers failed to initialize` 與
`connection closed: initialize response`，這次的原因是什麼？

這次確認有兩個 launcher 問題：

- 舊版 `start_mcp_stdio.ps1` 會在 STDIO MCP 回覆 `initialize` 前同步執行
  `ensure_server.ps1`。只要 Dashboard、port 或 Cloudflare quick tunnel 任一層
  失敗，PowerShell 就先退出，client 只看到 connection closed。
- repository-root `jy_agent.ps1` 曾以 array splatting 轉送 `-Python`；當
  `JY_QUALIBRATE_PYTHON` 已設定時，PowerShell 可能把值當成多餘 positional
  argument，launcher 同樣會在 MCP initialize 前退出。

現在 STDIO 啟動已改成 protocol-only：只找 Python 並執行
`python -m jy_agent stdio`，不依賴網站、Cloudflare、port 或網路；root shim 也以
真正的 named parameter 轉送 Python。Codex 的 startup timeout 同時提高為 90 秒，
供第一次冷啟動載入依賴。專案採 `required=false`，避免 JY MCP 初始化失敗時阻斷
一般 repository 工作；但沒有成功連結工具並取得 `jy_enter_*` 回傳時，agent 仍禁止
宣稱已進入量測模式。

修正後必須 reload workspace 或建立新 task，因為失敗 task 不會在原地重新建立
它的 MCP tool catalog。

## Q2：最先該跑哪個不會量測的自我檢查？

在 repository root 執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Doctor
```

也可以直接請尚未掛上 JY MCP 的 coding agent 執行這一行。Doctor 會：

- 解析 Codex、Claude、Cursor 的 root project config；
- 透過真正的 root launcher 做一次 MCP `initialize` 與 `tools/list`；
- 確認 `jy_get_status`、`jy_enter_measurement_mode`、
  `jy_enter_autonomy_mode` 存在；
- 分別檢查 localhost MCP HTTP 與 Approval HTTP health；
- 只輸出去識別化 bootstrap summary，永遠不輸出 Dashboard token。

`"ok": true` 代表 project config 與 STDIO handshake 正常；它不代表網站目前已
啟動，也不代表已進入量測模式。網站層請另外看 `"infrastructure_ok"`；
`"measurement_started": false` 是預期結果。

## Q3：Doctor 的 STDIO 通過，但 `local_services` 是 false，算失敗嗎？

不算 MCP 掛載失敗。這表示 client tools 可以正常初始化，但 HTTP／Dashboard
服務尚未啟動或已被 `結束量測` 關閉。收到正式進入語句後，agent 應執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Ensure
```

Ensure 成功後再呼叫對應的 `jy_enter_*` tool。正常使用時由 agent 自動做，不要求
操作人員自己開終端。反過來說，若 STDIO 不通，重跑 Ensure 不會修好 tool catalog。

## Q4：Cursor 顯示 `Ensure: ready` 和 Dashboard URL，但這個對話只有
`cursor`，沒有 `jy_bringup`，代表什麼？

`Ensure: ready` 只證明本機 HTTP／Dashboard infrastructure 已就緒；它不會把新的
MCP server 動態注入已存在的 Cursor conversation。缺少 `jy_bringup` 表示該 Agent
session 沒有載入、信任、啟用 project `.cursor/mcp.json`，或載入時 handshake
失敗。處理順序是：

1. 確認 Cursor workspace 是 repository 最上層，不必切進 `JY_agent`。
2. 執行 root Doctor；若 STDIO 不通，先修 launcher／Python。
3. 在 Cursor MCP 設定／Available Tools 中確認 `jy_bringup` 已啟用，而不是被
   toggle off。
4. reload Cursor window／project MCP，並建立新的 Agent chat。
5. Cursor CLI 可用 `cursor-agent mcp list` 與
   `cursor-agent mcp list-tools jy_bringup` 檢查設定與工具。

不要反覆建立 quick tunnel，也不要把公開 Dashboard URL 當成 MCP URL；Cursor
需要的是 client-owned STDIO server，公開 URL 只供人員核准、看圖與停止。

## Q5：Claude Code 會不會遇到相同問題？

舊版會。Codex、Cursor、Claude 都共用 root `jy_agent.ps1` 與 nested
`start_mcp_stdio.ps1`，因此上述「先開 tunnel 才回 initialize」及 Python 轉送問題
會同時影響三者。共用 launcher 修正後，client-independent Doctor 已驗證標準 MCP
handshake；但每個 Claude workspace 仍必須單獨信任 project `.mcp.json`。

Claude config 使用 `${CLAUDE_PROJECT_DIR:-.}`，避免啟動 shell 的 working
directory 不同。Claude 的排錯指令／面板為：

```text
claude mcp list
claude mcp get jy_bringup
/mcp
```

若顯示 `Pending approval`，在該 repository 中互動式開啟 Claude 並接受 workspace
trust／project MCP；若顯示 rejected，重新檢查 project choices；若 connected 但舊
chat 沒工具，使用 `/mcp` reconnect 後建立新 session。若診斷機器沒有可呼叫的
`claude` executable，實際 Claude UI 信任步驟必須在安裝 Claude 的 client 上確認；
底層共用 launcher 仍可由 root Doctor 的真實 MCP handshake 覆蓋。

## Q6：Codex 已修好 launcher，為什麼舊 task 還是可能失敗？

Codex 的 project-scoped `.codex/config.toml` 只在 trusted project 中載入，且
`required=false` 讓 enabled server 初始化失敗時不阻斷 startup／resume。MCP tool
catalog 屬於 session 啟動階段；修改 config 或 launcher 後，請 reload trusted
project 或建立新 task。`startup_timeout_sec = 90` 只處理冷啟動較慢，不會修正
程式退出或 stdout 污染。

若仍失敗，先跑 Doctor，再看 `runtime/mcp-stdio-launch.log` 是否出現該次 PID；
沒有紀錄通常代表 client 根本沒有執行到 project launcher，有紀錄但 handshake
失敗則查 stderr／Python import。

## Q7：什麼是 STDIO stdout 污染？如何避免？

STDIO MCP 的 stdout 是 protocol channel。任何 `Write-Host`、Python `print`、啟動
banner、Ensure JSON 或 tunnel 訊息若寫進 stdout，都可能讓 client 無法解析
`initialize` response。`start_mcp_stdio.ps1` 因此只把啟動摘要 append 到
`runtime/mcp-stdio-launch.log`；服務的 protocol stdout 不得加入人工可讀訊息。
診斷資訊請寫 stderr 或檔案。

## Q8：Doctor 顯示找不到 Qualibrate Python，要怎麼修？

先做一次離線安裝／驗證：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Install
```

自動尋找失敗時才明確傳入：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Install -Python C:\path\to\qualibrate_env\python.exe
```

安裝成功後 reload client MCP。不要把一般系統 Python 當成量測環境，也不要用
Ensure 代替 Install。

## Q9：MCP 正常，但 Dashboard／公開網址打不開，要查哪裡？

先確認 Doctor 的 STDIO 已通，再依序檢查：

1. `http://127.0.0.1:8765/healthz` 是否為 `jy-mcp`；
2. `http://127.0.0.1:8766/healthz` 是否為 `jy-approval`；
3. `runtime/server-bootstrap.json` 的 status／PID 是否與 health 一致；
4. `runtime/approval.stderr.log` 與 `runtime/mcp.stderr.log`；
5. 最後看 `runtime/public-tunnel.stderr.log`。

Quick tunnel URL 會隨重新建立而改變，舊 URL 過期是正常的。只使用最新 entry tool
回傳的 `browser_url`。絕對不要把 MCP port 公開，也不要將 public Dashboard URL
設定成 client MCP endpoint。

## Q10：常用 log 在哪裡？分享時要遮掉什麼？

- `runtime/mcp-stdio-launch.log`：client 是否真的啟動 root/nested launcher。
- `runtime/mcp.stderr.log`：loopback MCP HTTP service 錯誤。
- `runtime/approval.stderr.log`：Approval/Dashboard service 錯誤。
- `runtime/public-tunnel.stderr.log`：Cloudflare tunnel 錯誤。
- `runtime/server-bootstrap.json`：service PID、port、status 與 URL metadata。

分享 log 前不要貼出 `runtime/dashboard-password.txt` 或
`public-dashboard-access.token` 的內容；Dashboard 已不使用 `bootstrap_code`、
`pairing_code` 或配對連結。若密碼曾貼進 issue 或公開 log，先更換
`dashboard-password.txt` 的內容（單行 12–256 字元），再完整執行安全停止並重新進入；
`stop_server.ps1` 會刪除舊 access token 並清空公開 URL metadata。重新進入模式會建立
新的 service lifetime，舊瀏覽器的登入 cookie 隨之失效。

## Q11：一行話判斷「該 reload client 還是重跑 Ensure」？

- 工具清單沒有 `jy_bringup`，或 Codex 報 `initialize response`：跑 **Doctor**，
  然後 trust/enable/reload/new session。
- 工具清單已有 `jy_bringup`，但 localhost health／Dashboard 不通：跑 **Ensure**。
- Dashboard 通但 experiment 失敗：才進入 workflow、snapshot、worker、lock 與
  hardware 層排錯。

這三種情況不可互相替代。

## Q12：Agent 為什麼列出多個網址？它們是不同網站嗎？

它們不是三個 server；歷史版本在同一個 Approval HTTP service 上保留了三條 route：
`/approve/{proposal_id}`、`/autonomy/{lease_id}`、`/session/{session_id}`。如果 client
直接呼叫 lease request，而不是完整的 entry tool，舊版本可能沒有把既有 session
重新綁到新 lease，於是舊控制頁與新結果看起來像分裂的網站。

目前 entry result 最外層的 `browser_url` 是唯一應交給使用者的共用 `/` 入口。
它會解析目前 session，再連到五個同 hostname、同 cookie、同 service lifetime 的
分頁：首頁、核准、實驗結果、Token 用量、使用說明。它們不是不同網站。舊 `/approve/...`、
`/autonomy/...`、`/session/...` 只做 303 redirect；direct/replacement lease 也會
重新綁定同一 session。結果下拉紀錄會保留到完整關機。

若新版仍出現舊畫面，先安全停止再重新進入，讓服務載入新原始碼；不要手動改
SQLite lease/session 關聯。接著確認所有分頁是否同 hostname、entry 是否為 `/`、
舊 route 是否 303 導向對應分頁。若不是，再查看 `runtime/approval.stderr.log` 與 client 是否
仍連著修改前啟動的舊 process。

## Q13：桌面可開 Dashboard，但手機用同一網址得到 `403 Forbidden`，為什麼？

這是舊版一次性 `bootstrap_code` 被重播造成的；現在已沒有 bootstrap code 或
pairing link。公開 Dashboard 以固定密碼保護：手機與桌面都直接開啟最新 entry 回傳
的同一個裸 `browser_url`，未登入時會被導到 `/login`，輸入存在量測電腦
`runtime/dashboard-password.txt` 的實驗室密碼即可。每個瀏覽器各自取得 12 小時
cookie；不要把密碼放進網址或聊天訊息。

登入頁的錯誤：`401` 密碼錯誤；`429` 嘗試過多，稍後再試；`503` 密碼檔無法讀取，
在量測電腦檢查 `dashboard-password.txt`。若送出表單時仍得到 `403`，代表該瀏覽器
來源不被信任（例如仍開著舊 tunnel 網址的分頁）；改開最新 `browser_url` 後再操作。

## Q14：`Decision: halted`／頁面「已中止」等於完整 `結束量測` 嗎？

不等於。`halted` 只代表 bounded automation 因 hard-stop 不再排新實驗；為了保留
診斷與安全處理能力，workflow、Dashboard、MCP 與 tunnel 可能仍在。Session 頁面
現在另有「結束量測並關閉所有 JY 服務」按鈕，即使 lease halted 仍會顯示。它會：

1. 對 active run 發出安全停止並等待 terminal；
2. 若 worker 在 grace period 後仍存活，只 force-stop DB 記錄與 command/token 都吻合的 PID；
   接著 server 只關閉本晶片殘留的 QM 與其 job，若已沒有本晶片（或與其共用 port）
   的 QM 且 state hash 未變，就自動釋放 lock（`forced_stop_checkpoint`）；
3. 若 retained hardware lock 仍存在（QM 清理失敗、state 回復失敗、缺少退出證據等），
   永久停止 workflow 並轉為 `recovery_required`，不刪 lock；
4. 確認所有 JY worker 已死亡後，延遲呼叫受驗證的 `stop_server.ps1` 關閉 MCP、Dashboard 與 tunnel。

若最後的 helper 無法啟動，頁面會顯示錯誤並保留可重試按鈕。

## Q15：結束量測被 retained hardware lock 擋住，可以在網頁修復嗎？

可以，但完整關機不再為了等待 recovery 而把公開網站留著。系統會保留 lock 與
`runtime/recovery-required.json`、關閉 tunnel/MCP/Dashboard；下一次執行 Ensure 時
只會開 loopback recovery-only 服務，不會建立公開 URL。Recovery 只限量測電腦，
不能從 public Dashboard 或手機遠端執行。先實際確認 QOP/OPX 與相連儀器沒有 active job/output，再開啟
`http://127.0.0.1:8765/operator`。只有 lock/run/request identity、worker、database
run 與 state recovery evidence 全部通過時，頁面才顯示 recovery form。新版 run
使用 worker 在接觸硬體前持久化的 pre-run/database hash；舊版若因 abrupt exit 根本
來不及寫 DB checkpoint，則只在 configured active state、run active state 與受保護
recovery backup 三者完全同 hash 時採用相容證據。兩條路徑仍都必須勾選實體確認並
輸入完整 `RECOVER HARDWARE LOCK <run-id>`。

成功後系統封存 lock、刪除 recovery marker、寫 receipt/audit；之後可重新輸入進入
量測模式。若任一檢查
失敗，頁面只顯示阻擋原因且 lock 不變。CLI `-Action RecoverLock -RunId <run-id>`
仍保留為相同安全邏輯的 fallback；永遠不要手動刪除 `runtime\hardware.lock`。
Recovery-only 的 `remote_provider=local` 只是隔離狀態，不是永久偏好；lock 正式復原
後，下一次正常進入會重啟服務並恢復以密碼保護的 Cloudflare Dashboard。只有 MCP
listener 與本機 operator console 永遠維持 `127.0.0.1`，不會對外公開。

## Q16：為什麼以前網頁關機後，再輸入 `結束量測` 會說 worker 異常且 lock 阻擋？

舊版先呼叫 Windows `os.kill(pid, CTRL_BREAK_EVENT)`，再寫 shutdown request。當 worker
不是可接收該 console event 的 process-group 狀態時，Windows 會回 `WinError 87`；
request 尚未持久化，watchdog 隨後只看到「worker 消失、沒有 snapshot」，因此把它
分類成 crash 並保留 lock。新版順序相反：先寫
`runtime/shutdown_requests/<session-id>.json` 與 SQLite，再寫含 run/process token 的
`runtime/requests/<run-id>.stop.json`；worker 自行停止並寫 `<run-id>.exit.json`。
可查看 `runs.stop_intent/exit_code/termination_cause` 與 `worker_process_exited` event
判斷究竟是協作停止、grace 後強制停止，還是無意圖的異常退出。

正常的 Pause 不會停止 active worker；正常的 `結束量測` 會走 authenticated
cooperative stop，完成 state cleanup 後自動釋放 lock。grace period 後由 JY 強制停止
的 worker 也不一定留下 lock：server 先關閉本晶片殘留的 QM 與其 job（其他使用者的
QM 一律忽略），清理乾淨且 state hash 未變時自動釋放 lock。只有 Task Manager／作業
系統直接殺死 worker、Python/native crash、斷電、缺少 exit 證據、active node 無法確認
停止、QM 清理失敗或 state 回復失敗時才保留 lock。這種狀況不能在沒有硬體側可驗證
safe-state receipt 的前提下安全地自動解鎖；本機實體確認是最後的安全邊界，不應以
自動刪檔取代。

## Q17：Codex Terra/Sol 為何會結束回合？可以讓自動量測永遠保持運行嗎？

Terra 與 Sol 都受同一個 Codex task/goal 生命週期控制；模型名稱不是常駐開關。
自動模式在支援時使用 session-scoped durable goal，並每次用
`jy_wait_for_autonomy_status` 最多等待 55 秒後繼續，因此不會因等待核准或 worker
就正常結束。Goal 在 workflow complete、operator stop、lease expiry 或 hard stop
才算完成；host 仍可能因使用者中斷、應用關閉或平台限制終止 turn。Claude/Cursor
沒有可由本 repo 強制的通用 durable-goal API，若它們暫停，回原對話輸入 `已核准`
即可從持久化 lease/SQLite 狀態續跑。真正跨 client 無人值守 orchestration 仍需未來
新增 server-side scheduler；目前不宣稱 agent process 能無限常駐。

## Q18：儀器或 QOP/OPX 連不上時，為什麼不再顯示一般 crash？

Worker 會辨識 QOP/OPX、Octave、gRPC 與底層 transport 的連線例外，將 run 記為
`termination_cause=instrument_unreachable`，並把一般模式 workflow 或自動模式 lease
切到 `paused`。Results 頁會顯示「儀器錯誤：量測已暫停」，後續不會再排新實驗。
先檢查儀器電源、QOP/OPX 服務與實驗室網路；恢復後在 AI 對話輸入 `恢復量測`／
`Resume measurement`（自動模式也可在 Home 按「繼續排程」），同一 workflow 與
lease 會以新 run-id 重跑目前 node，不必重新輸入進入模式指令。若該 lease 開啟了
「全自動判斷模式」，server 每 60 秒探測一次 QOP，探測成功後先關閉本晶片殘留的 QM，
再自動繼續同一授權；agent 只要繼續用 `jy_wait_for_autonomy_status` 等待。本 lease
已用完 10 次自動判斷時改回人工 `恢復量測`；lock 留在 quarantine 時（見下段）全自動
判斷不會繼續，須先完成本機 recovery。

只有 exception 能證明 TCP／server discovery 根本沒有建立連線、active state 已回到
pre-run hash 且沒有 restore error 時，worker 才會自動釋放本次 lock。若是在硬體
執行中連線中斷，軟體無法證明 output 已停止，lock 會留在 local safety quarantine；
這不會阻止首頁完整關機關閉 Dashboard、MCP 與 tunnel。普通 fitting timeout 不會
只因為含有 timeout 字樣就被誤判成儀器離線。

## Q19：ChatGPT／Codex／Claude／Cursor 或外部網路突然斷線，人在遠端怎麼處理？

Dashboard 的按鈕由本機 JY service 執行，不依賴 AI 對話持續在線。頁面仍可連時：

1. 想稍後繼續：在 Home 按 Pause。
2. 想停止自動授權但保留診斷頁：按 End automation。
3. 想全部關閉：到 Home 使用完整關機並輸入 `SHUTDOWN <session-id>`。

網路恢復後，可在原對話輸入 `結束量測` 或 `End measurement`，讓 agent 再確認一次
workflow 與 service 都已關閉。不要另開第二個 agent 同時接手同一 workflow。若頁面
也已無法開啟，等 AI 連線恢復後直接輸入 `恢復`／`Recover`。

## Q20：輸入進入模式後，因殘留 process、workflow 或設定而失敗，最短處理方式？

在同一 repository 對話輸入 `恢復`；英文輸入 `Recover`。這個指令不依賴
`jy_bringup` MCP 是否成功掛載。Agent 會執行 repository-root
`jy_agent.ps1 -Action Recover`：先向仍存活且身分吻合的 worker 發 cooperative full
shutdown，關閉 stale workflow／lease／proposal／session，再用 bootstrap nonce、PID、
executable 與 command line 驗證並關閉 MCP、Dashboard、Cloudflare tunnel 與舊 secret。

- `closed`：已回到全部關閉狀態，可以重新輸入進入模式。
- `waiting`：worker 還在安全停止；稍後再輸入 `恢復`。
- `blocked`：listener／worker 身分無法驗證，沒有誤殺程序；在量測電腦執行 Doctor。
- `closed_recovery_required`：所有服務已關，但硬體 safe state 無法由軟體證明，lock
  留在本機 quarantine。
- `closed_configuration_attention`：所有可驗證的 JY 服務已關，但啟動設定檢查失敗；
  先在量測電腦執行 Doctor 修正設定，再重新進入模式。

Terminal fallback 是 `.\jy_agent.ps1 -Action Recover`；正常使用者不必手動輸入。

## Q21：為什麼 `Recover` 不保證自動刪掉每一個 hardware lock？

`Recover` 解決的是「殘留程式、workflow、Dashboard 或 tunnel 關不掉」，不是偽造
儀器安全證明。對已確認根本未連上的 failure，worker 會在 state hash 驗證後自行
釋放 lock；對中途斷線、native crash、斷電或 worker 身分不明的情況，任意刪 lock
可能讓下一個實驗與仍有 output 的硬體重疊。因此 `Recover` 會關閉所有可驗證服務，
但把最後的 lock 留在只能由量測電腦存取的 operator console。這是唯一仍需現場
確認的極端邊界，不是 Dashboard 或 server crash。
