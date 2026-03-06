// ─── NAV4RAIL BT Generator — Frontend ───────────────────────────────────────

const MAX_TIME_S = 900; // 15 minutes timeout
let generating = false;
let isLocalGeneration = false; // true only on the instance that triggered generate()
let history = [];

document.addEventListener("DOMContentLoaded", () => {
    checkStatus();
    loadExamples();
    fetchHistory();
    connectSSE();
});

// ─── Model status ───────────────────────────────────────────────────────────

async function checkStatus() {
    const badge = document.getElementById("model-status");
    const btn = document.getElementById("btn-generate");
    try {
        const res = await fetch("api/status");
        const data = await res.json();
        if (data.loaded) {
            badge.textContent = "Modele pret";
            badge.className = "status-badge ready";
            btn.disabled = false;
        } else {
            badge.textContent = "Modele non charge";
            badge.className = "status-badge error";
        }
    } catch {
        badge.textContent = "Serveur injoignable";
        badge.className = "status-badge error";
    }
}

// ─── Load examples ──────────────────────────────────────────────────────────

async function loadExamples() {
    try {
        const res = await fetch("api/examples");
        const data = await res.json();
        const container = document.getElementById("example-list");
        data.missions.forEach(mission => {
            const btn = document.createElement("button");
            btn.className = "example-btn";
            btn.textContent = mission;
            btn.onclick = () => {
                document.getElementById("mission").value = mission;
            };
            container.appendChild(btn);
        });
    } catch { /* ignore */ }
}

// ─── Progress bar ───────────────────────────────────────────────────────────

let progressInterval = null;
let progressStart = 0;

function startProgress(serverStartEpoch) {
    progressStart = serverStartEpoch ? serverStartEpoch * 1000 : Date.now();
    const bar = document.getElementById("progress-bar");
    const text = document.getElementById("progress-text");
    const timer = document.getElementById("progress-timer");

    bar.style.width = "0%";
    bar.classList.remove("timeout");
    setStep("step-prompt", "active");
    setStep("step-inference", "");
    setStep("step-validate", "");
    text.textContent = "Preparation du prompt...";

    // After 2s, switch to inference step
    setTimeout(() => {
        setStep("step-prompt", "done");
        setStep("step-inference", "active");
        text.textContent = "Inference en cours...";
    }, 2000);

    progressInterval = setInterval(() => {
        const elapsed = (Date.now() - progressStart) / 1000;
        const pct = Math.min((elapsed / MAX_TIME_S) * 100, 100);

        bar.style.width = pct + "%";

        const elMin = Math.floor(elapsed / 60);
        const elSec = Math.floor(elapsed % 60);
        timer.textContent =
            `${elMin}:${String(elSec).padStart(2, "0")} / ${Math.floor(MAX_TIME_S/60)}:00`;

        // Color transitions
        if (pct > 85) {
            bar.classList.add("timeout");
            text.textContent = "Bientot termine...";
        } else if (pct > 50) {
            text.textContent = "Inference en cours (patience)...";
        }

        if (elapsed >= MAX_TIME_S) {
            clearInterval(progressInterval);
            text.textContent = "Timeout depasse (15 min)";
        }
    }, 1000);
}

function stopProgress(success) {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }

    const bar = document.getElementById("progress-bar");
    const text = document.getElementById("progress-text");

    if (success) {
        bar.style.width = "100%";
        bar.classList.remove("timeout");
        setStep("step-inference", "done");
        setStep("step-validate", "done");
        text.textContent = "Termine !";
    }
}

function setStep(id, cls) {
    const el = document.getElementById(id);
    el.className = "step" + (cls ? " " + cls : "");
}

// ─── Generate BT ────────────────────────────────────────────────────────────

async function generate() {
    const mission = document.getElementById("mission").value.trim();
    if (!mission || generating) return;

    generating = true;
    isLocalGeneration = true;
    const useGrammar = document.getElementById("use-grammar").checked;
    const btn = document.getElementById("btn-generate");

    // Show loading
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");
    btn.disabled = true;
    startProgress();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MAX_TIME_S * 1000);

    try {
        const res = await fetch("api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mission, use_grammar: useGrammar }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!res.ok) {
            const err = await res.json();
            alert(err.error || "Erreur serveur");
            stopProgress(false);
            return;
        }

        const data = await res.json();
        stopProgress(true);

        // Short delay so user sees 100% before result
        await new Promise(r => setTimeout(r, 500));
        displayResult(data, "generate", mission);
    } catch (e) {
        clearTimeout(timeoutId);
        stopProgress(false);
        if (e.name === "AbortError") {
            alert("Timeout : la generation a depasse 4 minutes.");
        } else {
            alert("Erreur de connexion: " + e.message);
        }
    } finally {
        generating = false;
        isLocalGeneration = false;
        document.getElementById("loading").classList.add("hidden");
        btn.disabled = false;
    }
}

// ─── Validate only ──────────────────────────────────────────────────────────

async function validateOnly() {
    const xml = document.getElementById("xml-input").value.trim();
    if (!xml) return;

    try {
        const res = await fetch("api/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ xml }),
        });
        const data = await res.json();
        displayResult({ ...data, xml, generation_time_s: null }, "validate");
    } catch (e) {
        alert("Erreur: " + e.message);
    }
}

// ─── Display result ─────────────────────────────────────────────────────────

function displayResult(data, mode, mission) {
    document.getElementById("result").classList.remove("hidden");

    // Set mode class on validation panel
    const vPanel = document.getElementById("validation-panel");
    vPanel.classList.remove("mode-generate", "mode-validate");
    vPanel.classList.add(mode === "validate" ? "mode-validate" : "mode-generate");

    // XML with syntax highlighting
    document.getElementById("xml-output").innerHTML = highlightXml(data.xml);

    // Generation time
    const genTime = document.getElementById("gen-time");
    if (data.generation_time_s != null) {
        genTime.textContent = `Genere en ${data.generation_time_s}s`;
    } else {
        genTime.textContent = "";
    }

    // Score bar
    const scoreBar = document.getElementById("score-bar");
    const scoreValue = document.getElementById("score-value");
    scoreBar.style.width = (data.score * 100) + "%";
    if (mode !== "validate") {
        scoreBar.style.background = data.score > 0.8 ? "var(--green)"
                                  : data.score > 0.5 ? "var(--yellow)" : "var(--red)";
    } else {
        scoreBar.style.background = "";
    }
    scoreValue.textContent = data.score.toFixed(2);

    // Valid badge
    const badge = document.getElementById("valid-badge");
    badge.textContent = data.valid ? "VALIDE" : "INVALIDE";
    badge.className = "status-badge " + (data.valid ? "valid" : "invalid");

    // Level badges
    const errors = data.errors || [];
    const hasL1 = errors.some(e => e.startsWith("[L1]"));
    const hasL2 = errors.some(e => e.startsWith("[L2]"));
    const hasL3 = errors.some(e => e.startsWith("[L3]"));

    setLevel("level-l1", !hasL1);
    setLevel("level-l2", !hasL2);
    setLevel("level-l3", !hasL3);

    // Warnings
    const warningsList = document.getElementById("warnings-list");
    const warnings = data.warnings || [];
    if (warnings.length > 0) {
        warningsList.classList.remove("hidden");
        warningsList.innerHTML = warnings.map(w =>
            `<div class="warning-item">${escapeHtml(w)}</div>`
        ).join("");
    } else {
        warningsList.classList.add("hidden");
        warningsList.innerHTML = "";
    }

    // Errors
    const errorsList = document.getElementById("errors-list");
    if (errors.length > 0) {
        errorsList.classList.remove("hidden");
        errorsList.innerHTML = errors.map(e =>
            `<div class="error-item">${escapeHtml(e)}</div>`
        ).join("");
    } else {
        errorsList.classList.add("hidden");
        errorsList.innerHTML = "";
    }

}

// ─── History sync ────────────────────────────────────────────────────────────

async function fetchHistory() {
    try {
        const res = await fetch("api/history");
        history = await res.json();
        renderHistory();
    } catch { /* ignore */ }
}

function connectSSE() {
    const source = new EventSource("api/history/stream");
    source.addEventListener("history", (e) => {
        history = JSON.parse(e.data);
        renderHistory();
    });
    source.addEventListener("gen_status", (e) => {
        if (isLocalGeneration) return; // the local instance manages its own UI
        const info = JSON.parse(e.data);
        const loading = document.getElementById("loading");
        const btn = document.getElementById("btn-generate");
        if (info.status === "running") {
            loading.classList.remove("hidden");
            document.getElementById("result").classList.add("hidden");
            btn.disabled = true;
            startProgress(info.started_at);
            const missionLabel = document.getElementById("progress-text");
            if (info.mission) {
                missionLabel.textContent = "Generation en cours: " + info.mission;
            }
        } else {
            stopProgress(true);
            setTimeout(() => { loading.classList.add("hidden"); }, 800);
            btn.disabled = false;
        }
    });
    source.onerror = () => {
        setTimeout(() => connectSSE(), 3000);
        source.close();
    };
}

function renderHistory() {
    const container = document.getElementById("history-list");
    if (history.length === 0) {
        container.innerHTML = '<p class="text-muted">Aucune generation pour le moment.</p>';
        return;
    }
    container.innerHTML = "";
    history.forEach((h, i) => {
        const item = document.createElement("div");
        item.className = "history-item";
        const label = h.mode === "generate"
            ? `<span class="history-mode gen">GEN</span>`
            : `<span class="history-mode val">VAL</span>`;
        const scoreColor = h.score > 0.8 ? "var(--green)" : h.score > 0.5 ? "var(--yellow)" : "var(--red)";
        const xmlId = "history-xml-" + i;
        item.innerHTML = `
            <div class="history-header">
                ${label}
                <span class="history-time">${h.time}</span>
                <span class="history-score" style="color:${scoreColor}">${h.score.toFixed(2)}</span>
                <span class="history-valid ${h.valid ? 'pass' : 'fail'}">${h.valid ? "OK" : "KO"}</span>
                <button class="history-toggle" onclick="event.stopPropagation(); toggleHistoryXml('${xmlId}', this)">XML</button>
            </div>
            <div class="history-mission">${h.mission ? escapeHtml(h.mission) : "<em>Validation manuelle</em>"}</div>
            <div class="history-xml hidden" id="${xmlId}">
                <pre><code>${highlightXml(h.xml)}</code></pre>
            </div>
        `;
        item.onclick = () => loadHistoryEntry(i);
        container.appendChild(item);
    });
}

function toggleHistoryXml(id, btn) {
    const el = document.getElementById(id);
    const visible = !el.classList.contains("hidden");
    el.classList.toggle("hidden");
    btn.classList.toggle("active", !visible);
}

function loadHistoryEntry(index) {
    const h = history[index];
    if (h.mission) {
        document.getElementById("mission").value = h.mission;
    }
    document.getElementById("xml-input").value = h.xml;
}

function setLevel(id, pass) {
    const el = document.getElementById(id);
    el.className = "level-badge " + (pass ? "pass" : "fail");
}

// ─── XML syntax highlighting ────────────────────────────────────────────────

function highlightXml(xml) {
    let s = escapeHtml(xml);
    // Tags: <TagName or </TagName
    s = s.replace(/&lt;(\/?)([\w]+)/g, '&lt;$1<span class="tag">$2</span>');
    // Attributes: name="value"
    s = s.replace(/([\w-]+)=&quot;([^&]*)&quot;/g,
        '<span class="attr">$1</span>=&quot;<span class="val">$2</span>&quot;');
    // Self-closing />
    s = s.replace(/\/&gt;/g, '<span class="tag">/</span>&gt;');
    return s;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Enter key in textarea triggers generation
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
        e.preventDefault();
        generate();
    }
});
