# Public measurement Dashboard / 公開量測 Dashboard

`ensure_server.ps1` 將 Approval/Dashboard 的本機 `127.0.0.1:8766` 透過 Cloudflare
HTTPS quick tunnel forward 到公開網址。MCP 固定留在 `127.0.0.1:8765`，不會被
forward。公開端只提供共用首頁（狀態、四種控制狀態與完整關機）、審閱、核准、
結果圖、歷史選單、晶片歷史結果圖、Token 用量，以及使用說明（範例圖取自
`docs/guide_assets/`）。

`ensure_server.ps1` forwards local Approval/Dashboard port `127.0.0.1:8766`
through a Cloudflare HTTPS quick tunnel. MCP remains on `127.0.0.1:8765` and is
never forwarded. The public surface exposes only the shared Home (status, the
four operator control states, and full shutdown), review, approval, result
plots, history selection, chip-history result images, Token usage, and the
Guide (example images served from `docs/guide_assets/`).

公開網址本身不帶 token 或 bootstrap query。任何手機或瀏覽器都直接開啟當次
`https://…trycloudflare.com/`，再輸入量測電腦上
`JY_agent/runtime/dashboard-password.txt` 內的固定密碼。首次正常啟動若檔案
不存在，Ensure 會安全產生一次、限制檔案 ACL，之後 server 或 tunnel 重啟都沿用
同一密碼；檔案被 `.gitignore` 排除，密碼不會被 commit、寫入 URL 或啟動日誌。

The public URL contains no token or bootstrap query. Every phone or browser opens
the current `https://…trycloudflare.com/` directly and enters the fixed password
stored in `JY_agent/runtime/dashboard-password.txt`. Ensure securely generates
the ignored, ACL-restricted file once when missing and reuses it across server and
tunnel restarts. The password is never committed, placed in a URL, or printed in
startup logs.

登入成功後，每個瀏覽器取得各自 12 小時、`HttpOnly`、`Secure`、
`SameSite=Strict` 的 cookie；不需要 Pair another phone/browser。登入端點有基本
失敗次數限制。可直接修改密碼檔；若要同時讓所有既有 cookie 失效，修改後完整停止
並重新啟動 JY 服務，使 service instance nonce 一併輪替。

After login, each browser receives its own 12-hour `HttpOnly`, `Secure`,
`SameSite=Strict` cookie. There is no device-pairing step, and failed logins are
rate-limited. The password file can be edited directly. To invalidate every
already-issued cookie as well, fully stop and restart JY after changing it so the
service instance nonce rotates.

本機與公開入口如下：

- MCP：`http://127.0.0.1:8765/mcp`（不是量測網站，永不公開）。
- 現場 recovery console：`http://127.0.0.1:8765/operator`（永不公開）。
- 本機 Dashboard：`http://127.0.0.1:8766/`。
- 公開 Dashboard：當次 Ensure 回傳的 `https://…trycloudflare.com/`。

The loopback MCP and operator console remain private. Dashboard is locally served
at `http://127.0.0.1:8766/`; only that port is forwarded to the current
Cloudflare quick-tunnel hostname. A newly created quick tunnel may use a different
hostname, so use the URL returned by the current Ensure/entry result.

手機不需 VPN App。它只需要一般瀏覽器開啟 public URL；AI 對話仍透過手機 remote
回量測電腦的既有 agent task。若 tunnel 建立失敗，JY 不會改用 LAN binding，也
不會把 MCP 暴露到公網。

No VPN app is required on the phone; a normal browser opens the public URL. The
AI conversation remains a remote view of the desktop agent task. If the tunnel
cannot be established, JY does not fall back to a LAN listener and never exposes
MCP publicly.

硬體鎖 recovery 也在同一本機 operator console，但永遠不經 public tunnel。
只有所有軟體檢查通過、現場人員勾選已確認儀器無 active job/output，並輸入完整
確認句時才會封存 lock；不能從手機遠端繞過現場檢查。

Hardware-lock recovery is also available in that local console but never through
the public tunnel. It still requires all software checks, physical inspection,
the hardware attestation, and the exact confirmation phrase.
