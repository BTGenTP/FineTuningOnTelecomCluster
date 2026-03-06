const MAX_TIME_S = 240;

const MODE_CONFIG = {
    legacy: {
        status: "/api/status",
        examples: "/api/examples",
        generate: "/api/generate",
        validate: "/api/validate",
        missionPlaceholder: "Ex: Navigue jusqu'au km 42 depuis le km 10",
        constraintLabel: "Decodage contraint (GBNF)",
        supportsNav2Options: false,
    },
    nav2: {
        status: "/api/nav2/status",
        examples: "/api/nav2/examples",
        generate: "/api/nav2/generate",
        validate: "/api/nav2/validate/xml",
        missionPlaceholder: "Ex: Navigue vers le goal (Nav2), puis attends 2.0 s.",
        constraintLabel: "Decodage contraint JSON (lm-format-enforcer)",
        supportsNav2Options: true,
    },
};

let progressInterval = null;
let progressStart = 0;

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("mode-select").addEventListener("change", refreshModeUi);
    refreshModeUi();
});

function currentMode() {
    return document.getElementById("mode-select").value;
}

function currentConfig() {
    return MODE_CONFIG[currentMode()];
}

function refreshModeUi() {
    const config = currentConfig();
    document.getElementById("mission").placeholder = config.missionPlaceholder;
    document.getElementById("constraint-label").textContent = config.constraintLabel;
    document.getElementById("nav2-options").classList.toggle("hidden", !config.supportsNav2Options);
    document.getElementById("write-run-label").classList.toggle("hidden", !config.supportsNav2Options);
    checkStatus();
    loadExamples();
}

async function checkStatus() {
    const badge = document.getElementById("model-status");
    const btn = document.getElementById("btn-generate");
    try {
        const res = await fetch(currentConfig().status);
        const data = await res.json();
        if (currentMode() === "nav2") {
            if (data.loaded) {
                badge.textContent = `Modele Nav2 pret (${data.model_key})`;
                badge.className = "status-badge ready";
                btn.disabled = false;
            } else if (data.configured) {
                badge.textContent = "Adapter Nav2 configure, chargement a la demande";
                badge.className = "status-badge loading";
                btn.disabled = false;
            } else {
                badge.textContent = "Adapter Nav2 absent";
                badge.className = "status-badge error";
                btn.disabled = true;
            }
        } else if (data.loaded) {
            badge.textContent = "Modele pret";
            badge.className = "status-badge ready";
            btn.disabled = false;
        } else {
            badge.textContent = "Modele non charge";
            badge.className = "status-badge error";
            btn.disabled = true;
        }
    } catch {
        badge.textContent = "Serveur injoignable";
        badge.className = "status-badge error";
        btn.disabled = true;
    }
}

async function loadExamples() {
    try {
        const res = await fetch(currentConfig().examples);
        const data = await res.json();
        const container = document.getElementById("example-list");
        container.innerHTML = "";
        data.missions.forEach((mission) => {
            const btn = document.createElement("button");
            btn.className = "example-btn";
            btn.textContent = mission;
            btn.onclick = () => {
                document.getElementById("mission").value = mission;
            };
            container.appendChild(btn);
        });
    } catch {
        // ignore
    }
}

function startProgress() {
    progressStart = Date.now();
    const bar = document.getElementById("progress-bar");
    const text = document.getElementById("progress-text");
    const timer = document.getElementById("progress-timer");

    bar.style.width = "0%";
    bar.classList.remove("timeout");
    setStep("step-prompt", "active");
    setStep("step-inference", "");
    setStep("step-validate", "");
    text.textContent = "Preparation du prompt...";

    setTimeout(() => {
        setStep("step-prompt", "done");
        setStep("step-inference", "active");
        text.textContent = "Inference en cours...";
    }, 1500);

    progressInterval = setInterval(() => {
        const elapsed = (Date.now() - progressStart) / 1000;
        const pct = Math.min((elapsed / MAX_TIME_S) * 100, 100);
        bar.style.width = pct + "%";

        const elMin = Math.floor(elapsed / 60);
        const elSec = Math.floor(elapsed % 60);
        timer.textContent = `${elMin}:${String(elSec).padStart(2, "0")} / 4:00`;

        if (pct > 85) {
            bar.classList.add("timeout");
            text.textContent = "Bientot termine...";
        } else if (pct > 50) {
            text.textContent = "Inference en cours (patience)...";
        }

        if (elapsed >= MAX_TIME_S) {
            clearInterval(progressInterval);
            text.textContent = "Timeout depasse (4 min)";
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

async function generate() {
    const mission = document.getElementById("mission").value.trim();
    if (!mission) return;

    const btn = document.getElementById("btn-generate");
    const useConstraint = document.getElementById("use-grammar").checked;

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");
    btn.disabled = true;
    startProgress();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MAX_TIME_S * 1000);

    try {
        const body = currentMode() === "nav2"
            ? {
                mission,
                constrained: useConstraint ? "jsonschema" : "off",
                max_new_tokens: parseInt(document.getElementById("max-new-tokens").value, 10) || 256,
                temperature: parseFloat(document.getElementById("temperature").value || "0"),
                write_run: document.getElementById("write-run").checked,
            }
            : {
                mission,
                use_grammar: useConstraint,
            };

        const res = await fetch(currentConfig().generate, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        clearTimeout(timeoutId);

        const data = await res.json();
        if (!res.ok) {
            alert(data.error || "Erreur serveur");
            stopProgress(false);
            return;
        }

        stopProgress(true);
        await new Promise((resolve) => setTimeout(resolve, 300));
        displayResult(data);
    } catch (e) {
        clearTimeout(timeoutId);
        stopProgress(false);
        if (e.name === "AbortError") {
            alert("Timeout : la generation a depasse 4 minutes.");
        } else {
            alert("Erreur de connexion: " + e.message);
        }
    } finally {
        document.getElementById("loading").classList.add("hidden");
        btn.disabled = false;
    }
}

async function validateOnly() {
    const xml = document.getElementById("xml-input").value.trim();
    if (!xml) return;

    try {
        const res = await fetch(currentConfig().validate, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ xml }),
        });
        const data = await res.json();
        displayResult({ ...data, xml, generation_time_s: null });
    } catch (e) {
        alert("Erreur: " + e.message);
    }
}

function displayResult(data) {
    document.getElementById("result").classList.remove("hidden");

    const xml = data.xml || "";
    document.getElementById("xml-output").innerHTML = xml ? highlightXml(xml) : escapeHtml("(aucun XML genere)");

    const genTime = document.getElementById("gen-time");
    genTime.textContent = data.generation_time_s != null ? `Genere en ${data.generation_time_s}s` : "";

    const runDir = document.getElementById("run-dir");
    if (data.run_dir) {
        runDir.classList.remove("hidden");
        runDir.textContent = `Run: ${data.run_dir}`;
    } else {
        runDir.classList.add("hidden");
        runDir.textContent = "";
    }

    const stepsPanel = document.getElementById("steps-panel");
    const stepsOutput = document.getElementById("steps-output");
    if (data.steps_json) {
        stepsPanel.classList.remove("hidden");
        stepsOutput.textContent = formatJsonBlock(data.steps_json);
    } else {
        stepsPanel.classList.add("hidden");
        stepsOutput.textContent = "";
    }

    const rawStepsPanel = document.getElementById("raw-steps-panel");
    const rawStepsOutput = document.getElementById("raw-steps-output");
    if (data.raw_steps) {
        rawStepsPanel.classList.remove("hidden");
        rawStepsOutput.textContent = data.raw_steps;
    } else {
        rawStepsPanel.classList.add("hidden");
        rawStepsOutput.textContent = "";
    }

    const score = data.score != null ? data.score : (data.valid ? 1.0 : 0.0);
    const scoreBar = document.getElementById("score-bar");
    const scoreValue = document.getElementById("score-value");
    scoreBar.style.width = (score * 100) + "%";
    scoreBar.style.background = score > 0.8 ? "var(--green)"
        : score > 0.5 ? "var(--yellow)" : "var(--red)";
    scoreValue.textContent = score.toFixed(2);

    const badge = document.getElementById("valid-badge");
    badge.textContent = data.valid ? "VALIDE" : "INVALIDE";
    badge.className = "status-badge " + (data.valid ? "valid" : "invalid");

    const errors = data.errors || [];
    const hasL1 = errors.some((e) => e.startsWith("[L1]"));
    const hasL2 = errors.some((e) => e.startsWith("[L2]"));
    const hasL3 = errors.some((e) => e.startsWith("[L3]"));
    setLevel("level-l1", currentMode() === "nav2" ? data.valid : !hasL1);
    setLevel("level-l2", currentMode() === "nav2" ? data.valid : !hasL2);
    setLevel("level-l3", currentMode() === "nav2" ? data.valid : !hasL3);

    const warningsList = document.getElementById("warnings-list");
    const warnings = data.warnings || [];
    if (warnings.length > 0) {
        warningsList.classList.remove("hidden");
        warningsList.innerHTML = warnings.map((w) =>
            `<div class="warning-item">${escapeHtml(w)}</div>`
        ).join("");
    } else {
        warningsList.classList.add("hidden");
        warningsList.innerHTML = "";
    }

    const errorsList = document.getElementById("errors-list");
    if (errors.length > 0) {
        errorsList.classList.remove("hidden");
        errorsList.innerHTML = errors.map((e) =>
            `<div class="error-item">${escapeHtml(e)}</div>`
        ).join("");
    } else {
        errorsList.classList.add("hidden");
        errorsList.innerHTML = "";
    }
}

function setLevel(id, pass) {
    const el = document.getElementById(id);
    el.className = "level-badge " + (pass ? "pass" : "fail");
}

function formatJsonBlock(value) {
    try {
        return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
        return value;
    }
}

function highlightXml(xml) {
    let s = escapeHtml(xml);
    s = s.replace(/&lt;(\/?)([\w]+)/g, "&lt;$1<span class=\"tag\">$2</span>");
    s = s.replace(/([\w-]+)=&quot;([^&]*)&quot;/g,
        "<span class=\"attr\">$1</span>=&quot;<span class=\"val\">$2</span>&quot;");
    s = s.replace(/\/&gt;/g, "<span class=\"tag\">/</span>&gt;");
    return s;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
        e.preventDefault();
        generate();
    }
});
