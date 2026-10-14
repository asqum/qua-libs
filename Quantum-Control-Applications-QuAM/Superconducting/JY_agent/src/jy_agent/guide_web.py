"""Dashboard guide: how JY is operated and what each node should look like.

Operator request 2026-10-08. The example figures are static copies of runs JY
itself accepted, kept in `docs/guide_assets/<node>.png`. A node without one
(the flux nodes, until they have succeeded here) describes the expected
picture in words instead.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path


GUIDE_ASSET_NAME = re.compile(r"[0-9a-z_]+\.png")


def guide_asset_path(agent_root: Path, name: str) -> Path | None:
    """The guide image `name`, or None when it is not a guide asset."""

    if not GUIDE_ASSET_NAME.fullmatch(name):
        return None
    root = (agent_root / "docs" / "guide_assets").resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        return None
    return path


# (node, zh title, en title, zh purpose, en purpose, zh success, en success, source)
NODES: tuple[tuple[str, str, str, str, str, str, str, str], ...] = (
    (
        "02", "寬頻共振腔掃描", "Broadband spectroscopy",
        "每條讀取線掃 LO ± 400 MHz，扣除電延遲後列出共振腔候選，可依設計頻率指定各 qubit 的讀取 IF（選用）。",
        "Sweeps each readout line over LO ± 400 MHz, removes the electrical delay and lists resonator candidates; with design frequencies it assigns each qubit's readout IF (optional).",
        "平坦基線上出現一根根窄而深的 dip，橘線（選中的候選）與綠色虛線（平移後的設計值）對齊。",
        "Narrow, deep dips on a flat baseline, with the orange lines (selected candidates) lined up on the green dashed lines (shifted design).",
        "2026-10-07 #309",
    ),
    (
        "02x", "裸共振腔頻率", "Bare resonator",
        "以高功率／最小 flux 找出裸共振腔頻率（bare_resonator_freq）。",
        "Finds the bare resonator frequency at high power / minimum flux.",
        "每個 qubit 一個清楚的 dip，擬合曲線貼合且最低點在視窗內部。",
        "One clear dip per qubit, the fit follows it and the minimum sits inside the window.",
        "2026-10-07 #310",
    ),
    (
        "02c", "共振腔 vs 讀取功率", "Resonator vs readout power",
        "掃讀取功率，分辨低功率的 dressed 頻率與高功率的 bare 頻率，並把讀取功率設在 dressed 邊界下 3 dB。",
        "Sweeps readout power, separates the low-power dressed frequency from the high-power bare one, and sets the readout 3 dB below the dressed limit.",
        "低功率處一條穩定的 dressed 線，高功率時跳到另一個（bare）頻率，兩段都清楚可見。",
        "A steady dressed line at low power that jumps to another (bare) frequency at high power, both clearly visible.",
        "2026-10-07 #312",
    ),
    (
        "02a", "Dressed 共振腔微調", "Dressed resonator",
        "在選定的讀取功率下微調讀取頻率（dressed_resonator_freq）。",
        "Fine-tunes the readout frequency at the chosen power.",
        "清楚的 dip（振幅）與對應的相位跳變，擬合點落在 dip 底部。",
        "A clear amplitude dip with the matching phase step; the fitted point sits at the bottom of the dip.",
        "2026-10-07 #315",
    ),
    (
        "02b", "共振腔 vs flux（flux 校正）", "Resonator vs flux (flux calibration)",
        "掃每顆 qubit 的直流 flux，追蹤共振腔頻率並以 cosine 擬合，求出 sweet spot（joint_offset）、flux 週期 φ0，並把讀取 IF 移到該點。只在核准頁選「進行 flux 校正」時執行。",
        "Sweeps each qubit's DC flux, follows the resonator and fits a cosine: the sweet spot (joint_offset), the flux period φ0, and the readout IF at that point. Runs only when the approval page says Yes to flux calibration.",
        "共振腔頻率隨 flux 呈平滑、週期性的 cosine 弧線，紅色虛線（idle offset）落在弧線的極值上且不在掃描邊緣。若頻率完全不隨 flux 變化（平直）表示無 flux 響應，該 qubit 會略過 03b/06a；若某處頻率突然衝高或衝低又從另一側回來，即 avoided crossing，結果頁會提示調整相鄰 coupler 的 flux offset。",
        "A smooth, periodic cosine arc of resonator frequency vs flux, with the red dashed idle offset on an extremum and away from the sweep edges. A trace that does not move at all means no flux response, and that qubit skips 03b/06a. A spot where the frequency shoots up or down and returns from the other side is an avoided crossing; the results page then asks you to tune the neighbouring couplers' flux offsets.",
        "",
    ),
    (
        "03a", "Qubit 頻譜", "Qubit spectroscopy",
        "由寬到窄（coarse → refinement → fine）找出 qubit 頻率，fine 掃描通過才更新驅動頻率。",
        "Finds the qubit frequency from wide to narrow (coarse → refinement → fine); only a passing fine scan updates the drive frequency.",
        "背景平穩、只有一根窄而高的峰，紅點落在峰頂、擬合線貼合。",
        "A flat background with a single narrow, tall peak; the red dot is on the top and the fit follows it.",
        "2026-10-07 #318",
    ),
    (
        "03b", "Qubit 頻譜 vs flux（flux 校正）", "Qubit spectroscopy vs flux (flux calibration)",
        "在 02b 的 offset 附近加 flux 脈衝做 qubit 頻譜，以拋物線擬合找出 sweet spot，更新 joint_offset、驅動頻率與二次項係數。只對 02b 判定可調的 qubit 執行。",
        "Qubit spectroscopy under a flux pulse around the 02b offset; a parabola fit gives the sweet spot and updates joint_offset, the drive frequency and the quadratic term. Runs only for qubits 02b found tunable.",
        "qubit 線在圖上形成一條開口向下的拋物線，紅色虛線貼合，紅星（sweet spot）在頂點且位於 flux 掃描範圍內部。",
        "The qubit line traces a downward-opening parabola, the red dashed fit follows it, and the red star (sweet spot) sits at the top, inside the flux window.",
        "",
    ),
    (
        "04", "Power Rabi", "Power Rabi",
        "掃 π 脈衝振幅，找出 x180 / x90 的振幅。",
        "Sweeps the π-pulse amplitude to find the x180 / x90 amplitudes.",
        "一個完整、平滑的 Rabi 振盪，擬合曲線貼合，最高點（π 點）在掃描範圍內部。",
        "One complete, smooth Rabi oscillation with the fit on top and the maximum (π point) inside the sweep.",
        "2026-10-07 #320",
    ),
    (
        "05", "T1", "T1",
        "量能量弛豫時間 T1。",
        "Measures the energy relaxation time T1.",
        "訊號由高處指數衰減到平穩的平衡值，等待時間涵蓋好幾個 T1。",
        "The signal decays exponentially to a steady equilibrium, and the window covers several T1.",
        "2026-10-07 #322",
    ),
    (
        "07d", "讀取頻率／長度／功率最佳化", "Readout frequency / duration / power",
        "同時掃讀取頻率、長度與振幅，選出保真度最高的讀取設定。",
        "Sweeps readout frequency, duration and amplitude together and picks the highest-fidelity setting.",
        "隨讀取長度增加出現一塊明顯的高保真度（藍色）區域，星號落在該區域內。",
        "A distinct high-fidelity (blue) region that grows with readout length, with the star inside it.",
        "2026-10-07 #323",
    ),
    (
        "07b", "IQ blobs", "IQ blobs",
        "量 |g⟩、|e⟩ 的 IQ 分布，設定判別閾值與讀取保真度；保真度夠高的 qubit 之後改用 active reset。",
        "Measures the |g⟩ and |e⟩ IQ clouds, sets the discrimination threshold and readout fidelity; qubits with high enough fidelity use active reset afterwards.",
        "藍、橘兩團分得開、形狀緊實，閾值線落在兩團之間。",
        "Two compact, well-separated clouds (blue and orange) with the threshold between them.",
        "2026-09-21 #206",
    ),
    (
        "06", "Ramsey", "Ramsey",
        "量 T2* 並微調 qubit 頻率；失諧量依視窗自動取（約四個條紋）。",
        "Measures T2* and fine-tunes the qubit frequency; the detuning follows the window (about four fringes).",
        "清楚的衰減振盪（約數個條紋），擬合線貼合，最後衰減到平衡值。",
        "A clear decaying oscillation (a few fringes) with the fit on top, settling to equilibrium.",
        "2026-10-04 #287",
    ),
    (
        "06a", "Ramsey vs flux（選用，flux 校正）", "Ramsey vs flux (optional, flux calibration)",
        "在不同 flux 脈衝下做 Ramsey，由條紋頻率 vs flux 的拋物線微調 joint_offset 與驅動頻率。需選擇進行 flux 校正並勾選 06a。",
        "Ramsey under different flux pulses; the fringe-frequency-vs-flux parabola fine-tunes joint_offset and the drive frequency. Needs flux calibration = Yes and the 06a tick.",
        "每個 flux 點都有清楚的 Ramsey 條紋；第二張圖中條紋頻率 vs flux 形成一條拋物線，頂點靠近 0 且在視窗中央附近。",
        "Clear Ramsey fringes at every flux point; in the second figure the fringe frequency vs flux forms a parabola whose vertex is near 0, well inside the window.",
        "",
    ),
    (
        "06b", "T2 echo", "T2 echo",
        "以 echo 量 T2。",
        "Measures T2 with an echo.",
        "平滑的指數衰減到 0.5 附近，擬合線貼合。",
        "A smooth exponential decay towards about 0.5, with the fit on top.",
        "2026-10-04 #291",
    ),
    (
        "10a", "單 qubit randomized benchmarking", "Single-qubit randomized benchmarking",
        "以隨機 Clifford 序列量單 qubit 閘保真度（EPG）。",
        "Measures the single-qubit gate fidelity (EPG) with random Clifford sequences.",
        "存活機率隨序列深度平滑衰減，擬合線貼合，圖上標出 1Q gate fidelity。",
        "Survival probability decays smoothly with depth, the fit follows it, and the 1Q gate fidelity is shown.",
        "2026-10-04 #295",
    ),
)


def _ui(language: str, zh: str, en: str) -> str:
    return escape(en if language == "en" else zh)


def guide_section(language: str, *, asset_prefix: str = "/guide/assets") -> str:
    """The guide body: operation steps, node order, and per-node examples."""

    flux_nodes = {"02b", "03b", "06a"}
    order = " → ".join(
        f'<span class="{"flux-node" if node in flux_nodes else ""}">{escape(node)}</span>'
        for node, *_ in NODES
    )
    steps = [
        (
            "在 AI 對話輸入「進入 JY 自動量測模式」（逐次核准則輸入「進入 JY 量測模式」），AI 會回傳本 Dashboard 的網址。",
            "In the AI conversation, send “Enter JY automatic measurement mode” (or “Enter JY measurement mode” for per-run approval). The AI replies with this Dashboard's link.",
        ),
        (
            "到「核准」頁：勾選要做的額外實驗、選擇量測方式（共用 XY 輸出時只能一次一顆）、選擇是否進行 flux 校正（共用 Z 輸出時無法選「是」），輸入核准文字後按「核准」。",
            "On the Approval page: tick the optional experiments, choose the measurement mode (one qubit at a time only, when an XY output is shared), choose whether to calibrate flux (Yes is disabled when a Z output is shared), type the confirmation text and press Approve.",
        ),
        (
            "回到 AI 對話輸入「已核准」，AI 便在授權範圍內自動依序量測，不再逐次詢問。",
            "Return to the conversation and send “已核准”; the AI then measures the sequence automatically within the authorization without asking again.",
        ),
        (
            "「實驗結果」頁會即時顯示每次實驗的圖、判斷與理由；若出現 avoided crossing 等需要人處理的提示，會以橘色卡片標出。",
            "The Experiment results page shows every run's plots, decision and reason as they happen; anything that needs a person, such as an avoided crossing, is flagged in an orange card.",
        ),
        (
            "「首頁」可暫停／繼續排程或結束自動授權；量完後在對話輸入「結束量測」，或在首頁做完整關機。",
            "Home pauses/resumes scheduling or ends the authorization. When done, send “結束量測” (End measurement) in the conversation, or use full shutdown on Home.",
        ),
        (
            "儀器斷線時排程會自動暫停；確認儀器恢復後輸入「恢復量測」。上次殘留程式導致無法進入時輸入「恢復」。",
            "If an instrument disconnects, scheduling pauses on its own; once it is back, send “恢復量測” (Resume measurement). If entry fails because something was left over, send “恢復” (Recover).",
        ),
    ]
    steps_html = "".join(f"<li>{_ui(language, zh, en)}</li>" for zh, en in steps)
    cards = []
    for node, zh_title, en_title, zh_purpose, en_purpose, zh_ok, en_ok, source in NODES:
        if source:
            figure = (
                f'<figure><img src="{escape(asset_prefix)}/{escape(node)}.png" '
                f'alt="{_ui(language, "成功範例", "Successful example")} {escape(node)}" loading="lazy">'
                f'<figcaption>{_ui(language, "成功範例：", "Successful example: ")}{escape(source)}</figcaption></figure>'
            )
        else:
            figure = (
                f'<p class="muted guide-no-image">{_ui(language, "此 node 尚無 JY 成功的範例圖；請依右側文字判斷。", "No JY-accepted example yet for this node; judge by the description.")}</p>'
            )
        badge = (
            f' <span class="guide-flux">{_ui(language, "flux 校正", "flux calibration")}</span>'
            if node in flux_nodes
            else ""
        )
        cards.append(
            f"""<article class="guide-node" id="node-{escape(node)}">
              <div class="guide-figure">{figure}</div>
              <div>
                <h3>{escape(node)} · {_ui(language, zh_title, en_title)}{badge}</h3>
                <p>{_ui(language, zh_purpose, en_purpose)}</p>
                <p class="summary-ok">{_ui(language, "成功時應該看到：", "Success looks like: ")}{_ui(language, zh_ok, en_ok)}</p>
              </div>
            </article>"""
        )
    return f"""<style>
      .guide-order {{ font-family:ui-monospace,monospace; overflow-wrap:anywhere; }}
      .guide-order .flux-node, .guide-flux {{ color:#ffd479; }}
      .guide-flux {{ font-size:.8rem; border:1px solid #ffd479; border-radius:999px; padding:0 .5em; margin-left:.3em; }}
      .guide-node {{ display:grid; grid-template-columns:minmax(0,320px) 1fr; gap:16px; padding:14px 0; border-top:1px solid #21304d; }}
      .guide-node:first-of-type {{ border-top:0; }}
      .guide-node h3 {{ margin:0 0 6px; }}
      .guide-figure img {{ max-height:320px; object-fit:contain; background:white; }}
      @media(max-width:600px) {{ .guide-node {{ grid-template-columns:1fr; }} }}
    </style>
    <section class="card"><h2>{_ui(language, "操作方式", "How to operate JY")}</h2><ol>{steps_html}</ol></section>
    <section class="card"><h2>{_ui(language, "量測順序", "Measurement order")}</h2>
      <p class="guide-order">{order}</p>
      <p class="muted">{_ui(language, "黃色為 flux 校正節點，只在核准頁選「是」時執行（06a 另需勾選）；02、05st、06st 等為可勾選的額外實驗。共用 XY 輸出的晶片從 03a 起一次量一顆 qubit。", "Yellow nodes are flux calibration and run only after Yes on the approval page (06a also needs its tick); 02 and the statistics nodes are optional ticks. A chip with a shared XY output is measured one qubit at a time from 03a on.")}</p>
    </section>
    <section class="card"><h2>{_ui(language, "各節點與成功範例", "Nodes and successful examples")}</h2>{"".join(cards)}</section>"""
