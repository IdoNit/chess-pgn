import {Chessboard, COLOR, INPUT_EVENT_TYPE} from "/static/vendor/cm-chessboard/src/Chessboard.js"
import {Chess} from "/static/vendor/chess.js"

const HEADER_FIELDS = ["White", "Black", "Event", "Site", "Date", "Round", "Opening"]
const CONVERT_TIMEOUT_MS = 15 * 60 * 1000
let STRINGS = {}
const state = {files: [], game: null, selectedPly: null, board: null}

const $ = (id) => document.getElementById(id)
const t = (key) => STRINGS[key] ?? key
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]))
const ltr = (s) => `<bdi dir="ltr">${esc(s)}</bdi>`  // Latin text (names, moves) inside Hebrew lines

function applyI18n(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n) })
}

async function api(method, url, body, isForm = false, timeoutMs = 0) {
    const options = {method}
    if (timeoutMs) {
        const controller = new AbortController()
        setTimeout(() => controller.abort(), timeoutMs)
        options.signal = controller.signal
    }
    if (body !== undefined) {
        options.body = isForm ? body : JSON.stringify(body)
        if (!isForm) options.headers = {"Content-Type": "application/json"}
    }
    let response
    try {
        response = await fetch(url, options)
    } catch (error) {
        throw new Error(error.name === "AbortError" ? "timeout" : "generic")
    }
    const data = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(data.error || "generic")
    return data
}

function errorText(error) {
    const key = `error.${error.message}`
    return STRINGS[key] ? t(key) : t("error.generic")
}

// ---------------------------------------------------------------- routing

function route() {
    const hash = location.hash || "#/"
    for (const view of ["convert", "review", "games"]) $(`view-${view}`).hidden = true
    const gameMatch = hash.match(/^#\/game\/(.+)$/)
    if (gameMatch) {
        $("view-review").hidden = false
        openGame(decodeURIComponent(gameMatch[1]))
    } else if (hash === "#/games") {
        $("view-games").hidden = false
        showGames()
    } else {
        $("view-convert").hidden = false
    }
}

// ---------------------------------------------------------------- convert

function addFiles(fileList) {
    for (const file of fileList) if (file.type.startsWith("image/")) state.files.push(file)
    renderPages()
}

function renderPages() {
    const list = $("pages")
    list.innerHTML = ""
    state.files.forEach((file, index) => {
        const item = document.createElement("li")
        item.draggable = true
        item.dataset.index = index
        item.innerHTML = `<img alt="" src="${URL.createObjectURL(file)}"><div>${index + 1}. ${ltr(file.name)}</div>
            <button class="button secondary" type="button">${esc(t("convert.remove_image"))}</button>`
        item.querySelector("button").onclick = () => { state.files.splice(index, 1); renderPages() }
        item.ondragstart = (e) => e.dataTransfer.setData("text/plain", String(index))
        item.ondragover = (e) => e.preventDefault()
        item.ondrop = (e) => {
            e.preventDefault()
            const from = Number(e.dataTransfer.getData("text/plain"))
            const [moved] = state.files.splice(from, 1)
            state.files.splice(index, 0, moved)
            renderPages()
        }
        list.appendChild(item)
    })
}

async function runConvert() {
    const status = $("convert-status")
    status.classList.remove("error")
    if (state.files.length === 0) { status.textContent = t("convert.no_images"); return }
    const form = new FormData()
    state.files.forEach((file) => form.append("images", file, file.name))
    $("convert-button").disabled = true
    status.textContent = t("convert.working")
    $("results").innerHTML = ""
    try {
        const data = await api("POST", "/api/convert", form, true, CONVERT_TIMEOUT_MS)
        status.textContent = ""
        state.files = []
        renderPages()
        data.games.forEach((game) => $("results").appendChild(resultCard(game, data.games.length > 1)))
        if (data.games.length > 1) {
            const all = document.createElement("a")
            all.className = "button primary"
            all.href = `/api/download?ids=${data.games.map((g) => encodeURIComponent(g.id)).join(",")}`
            all.textContent = t("result.download_all")
            $("results").prepend(all)
        }
    } catch (error) {
        status.textContent = errorText(error)
        status.classList.add("error")
    } finally {
        $("convert-button").disabled = false
    }
}

function statusBadge(flagsOpen) {
    return flagsOpen === 0
        ? `<span class="badge ok">✓ ${esc(t("result.all_checked"))}</span>`
        : `<span class="badge warn">⚠ ${flagsOpen} ${esc(t("result.to_check"))}</span>`
}

function resultCard(game, several) {
    const card = document.createElement("div")
    card.className = "card"
    card.innerHTML = `<h3>${ltr(game.title)}</h3>
        <div>${game.move_count} ${esc(t("result.moves"))} · ${statusBadge(game.flags_open)}
        ${game.continues ? ` · <span class="badge warn">${esc(t("result.continues"))}</span>` : ""}</div>
        <div class="actions">
            <a class="button primary" href="/api/games/${encodeURIComponent(game.id)}/download">${esc(t("result.download"))}</a>
            <a class="button secondary" href="#/game/${encodeURIComponent(game.id)}">${esc(t("result.review"))}</a>
            ${several ? `<button class="button secondary discard" type="button">${esc(t("result.discard"))}</button>` : ""}
        </div>`
    const discard = card.querySelector(".discard")
    if (discard) discard.onclick = async () => {
        await api("DELETE", `/api/games/${encodeURIComponent(game.id)}`)
        card.innerHTML = `<p class="status">${esc(t("result.discarded"))}</p>`
    }
    return card
}

// ---------------------------------------------------------------- review

async function openGame(id) {
    state.game = await api("GET", `/api/games/${encodeURIComponent(id)}`)
    state.selectedPly = firstOpenFlag(0)
    renderReview()
    renderPageViewer()
}

function firstOpenFlag(afterPly) {
    const plies = state.game.flags.filter((f) => !f.reason.startsWith("side:")).map((f) => f.ply).sort((a, b) => a - b)
    return plies.find((p) => p > afterPly) ?? plies[0] ?? null
}

function renderReview() {
    const game = state.game
    $("download").href = `/api/games/${encodeURIComponent(game.id)}/download`
    $("header-fields").innerHTML = HEADER_FIELDS.map((key) => `<label>${esc(t(`header.${key}`))}
        <input name="${key}" value="${esc(game.headers[key] ?? "")}"></label>`).join("")
    renderMoves()
    renderBoardAndPanel()
    const side = game.flags.filter((f) => f.reason.startsWith("side:"))
    $("side-flags").innerHTML = side.length
        ? `<strong>${esc(t("flags.side_title"))}</strong><ul>${side.map((f, i) =>
            `<li>${esc(t("review.in_variation"))} ${Math.ceil(f.ply / 2)} (${esc(t(f.ply % 2 ? "review.white" : "review.black"))}): ${esc(t(`reason.${f.reason}`))}
             <span dir="ltr" class="move-text">${esc(f.token)} → ${esc(f.chosen)}</span>
             <button class="button secondary" type="button" data-side="${i}">${esc(t("review.confirm"))}</button></li>`).join("")}</ul>`
        : ""
    $("side-flags").querySelectorAll("[data-side]").forEach((button) => {
        button.onclick = () => dismiss(side[Number(button.dataset.side)])
    })
}

function renderMoves() {
    const flagged = new Set(state.game.flags.filter((f) => !f.reason.startsWith("side:")).map((f) => f.ply))
    $("moves").innerHTML = state.game.moves.map((m) => {
        const number = m.ply % 2 === 1 ? `<span class="num">${(m.ply + 1) / 2}.</span>` : ""
        const classes = ["mv", flagged.has(m.ply) ? "flag" : "", m.ply === state.selectedPly ? "selected" : ""].join(" ")
        return `${number}<span class="${classes}" data-ply="${m.ply}">${esc(m.san)}</span> `
    }).join("") + `<span class="num">${esc(state.game.headers.Result ?? "")}</span>`
    $("moves").querySelectorAll(".mv").forEach((el) => {
        el.onclick = () => { state.selectedPly = Number(el.dataset.ply); renderMoves(); renderBoardAndPanel() }
    })
}

async function renderBoardAndPanel() {
    const game = state.game
    const move = game.moves.find((m) => m.ply === state.selectedPly)
    const fen = move ? move.fen_before : (game.moves.length ? game.moves[0].fen_before : "start")
    if (!state.board) {
        state.board = new Chessboard($("board"), {
            position: fen,
            orientation: COLOR.white,
            assetsUrl: "/static/vendor/cm-chessboard/assets/",
            style: {pieces: {file: "pieces/standard.svg"}, animationDuration: 150},
        })
    } else {
        await state.board.setPosition(fen, false)
    }
    state.board.disableMoveInput()
    if (move) state.board.enableMoveInput((event) => onBoardMove(event, move))
    renderPanel(move)
}

function onBoardMove(event, move) {
    if (event.type === INPUT_EVENT_TYPE.moveInputStarted) return true
    if (event.type !== INPUT_EVENT_TYPE.validateMoveInput) return true
    const chess = new Chess(move.fen_before)
    try {
        const played = chess.move({from: event.squareFrom, to: event.squareTo, promotion: "q"})
        pin(move.ply, played.from + played.to + (played.promotion ?? ""))
        return true
    } catch {
        $("panel").querySelector(".panel-status").textContent = t("review.illegal")
        return false
    }
}

function renderPanel(move) {
    const panel = $("panel")
    if (!move) { panel.innerHTML = `<p class="hint">${esc(t(state.game.flags.length ? "review.pick_move" : "review.no_flags"))}</p>`; return }
    const flag = state.game.flags.find((f) => f.ply === move.ply && !f.reason.startsWith("side:"))
    const side = t(move.ply % 2 === 1 ? "review.white" : "review.black")
    const number = Math.ceil(move.ply / 2)
    const options = flag ? [flag.chosen, ...flag.suggestions] : []
    panel.innerHTML = `<div><strong>${flag ? "⚠ " : ""}${esc(t("review.move"))} ${number} (${esc(side)})</strong></div>
        ${flag ? `<div>${esc(t(`reason.${flag.reason}`))}</div>` : ""}
        ${flag && flag.token ? `<div>${esc(t("review.book_read"))} <span class="move-text">${ltr(flag.token)}</span></div>` : ""}
        <div>${esc(t("review.chosen"))} <span class="move-text">${ltr(move.san)}</span></div>
        ${options.length ? `<div>${esc(t("review.which"))}</div><div class="suggestions">${options.map((san) =>
            `<button class="button ${san === move.san ? "primary" : "secondary"}" type="button" data-san="${esc(san)}">${esc(san)}</button>`).join("")}</div>` : ""}
        <div class="hint">${esc(t("review.or_drag"))}</div>
        ${flag ? `<button class="button secondary confirm" type="button">✓ ${esc(t("review.confirm"))}</button>` : ""}
        <div class="status panel-status"></div>`
    const confirm = panel.querySelector(".confirm")
    if (confirm) confirm.onclick = () => dismiss(flag)
    panel.querySelectorAll("[data-san]").forEach((button) => {
        button.onclick = () => {
            const chess = new Chess(move.fen_before)
            const played = chess.move(button.dataset.san)
            pin(move.ply, played.from + played.to + (played.promotion ?? ""))
        }
    })
}

async function pin(ply, uci) {
    $("panel").querySelector(".panel-status").textContent = t("review.saving")
    try {
        state.game = await api("POST", `/api/games/${encodeURIComponent(state.game.id)}/pin`, {ply, uci})
        state.selectedPly = firstOpenFlag(ply) ?? ply
        renderReview()
    } catch (error) {
        $("panel").querySelector(".panel-status").textContent = errorText(error)
    }
}

async function dismiss(flag) {
    try {
        state.game = await api("POST", `/api/games/${encodeURIComponent(state.game.id)}/dismiss`, {ply: flag.ply, reason: flag.reason})
        if (!flag.reason.startsWith("side:")) state.selectedPly = firstOpenFlag(flag.ply)
        renderReview()
    } catch (error) {
        $("side-flags").textContent = errorText(error)
    }
}

async function saveHeaders(event) {
    event.preventDefault()
    const body = Object.fromEntries(new FormData($("headers")).entries())
    $("headers-status").textContent = t("review.saving")
    try {
        state.game = await api("PUT", `/api/games/${encodeURIComponent(state.game.id)}/headers`, body)
        renderReview()
        $("headers-status").textContent = t("review.saved")
    } catch (error) {
        $("headers-status").textContent = errorText(error)
    }
}

function renderPageViewer() {
    const viewer = $("page-viewer")
    viewer.innerHTML = state.game.images.map((src) => `<img alt="" src="${esc(src)}">`).join("")
    const view = {scale: 1, x: 0, y: 0}
    const images = [...viewer.querySelectorAll("img")]
    let offset = 0
    images.forEach((img) => img.onload = () => {
        offset = 0
        images.forEach((i) => { i.style.top = `${offset}px`; offset += i.clientHeight + 8 })
    })
    const apply = () => images.forEach((img) => { img.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})` })
    viewer.onwheel = (e) => {
        e.preventDefault()
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
        const rect = viewer.getBoundingClientRect()
        const px = e.clientX - rect.left, py = e.clientY - rect.top
        view.x = px - (px - view.x) * factor
        view.y = py - (py - view.y) * factor
        view.scale *= factor
        apply()
    }
    let drag = null
    viewer.onpointerdown = (e) => { drag = {x: e.clientX - view.x, y: e.clientY - view.y}; viewer.setPointerCapture(e.pointerId) }
    viewer.onpointermove = (e) => { if (drag) { view.x = e.clientX - drag.x; view.y = e.clientY - drag.y; apply() } }
    viewer.onpointerup = () => { drag = null }
}

// ---------------------------------------------------------------- my games

async function showGames() {
    const games = await api("GET", "/api/games")
    $("games").innerHTML = games.length ? games.map((g) => `<li>
        <div><strong>${ltr(g.title)}</strong><div class="meta">${esc(new Date(g.created).toLocaleDateString("he-IL"))} · ${statusBadge(g.flags_open)}</div></div>
        <a class="button secondary" href="#/game/${encodeURIComponent(g.id)}">${esc(t("games.open"))}</a></li>`).join("")
        : `<li class="hint">${esc(t("games.empty"))}</li>`
}

// ---------------------------------------------------------------- start

async function start() {
    STRINGS = await (await fetch("/static/strings.he.json")).json()
    applyI18n()
    const drop = $("drop")
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over") }
    drop.ondragleave = () => drop.classList.remove("over")
    drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files) }
    $("file-input").onchange = (e) => { addFiles(e.target.files); e.target.value = "" }
    $("convert-button").onclick = runConvert
    $("headers").onsubmit = saveHeaders
    window.onhashchange = route
    route()
}

start()
