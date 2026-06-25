/**
 * Scann Pro — ANN 参数实验（嵌入检索页 · 控制区 + 结果区分离）
 */
(function () {
    const DEBOUNCE_MS = 320;

    let controlsRoot = null;
    let resultsRoot = null;
    let state = null;
    let debounceTimer = null;
    let animFrame = null;
    let displayMetrics = { recall: 0, annMs: 0, exactMs: 0, speedup: 1 };

    function esc(s) {
        return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function q(sel) {
        return controlsRoot?.querySelector(sel) || resultsRoot?.querySelector(sel);
    }

    function tweenMetrics(target) {
        const start = { ...displayMetrics };
        const t0 = performance.now();
        const dur = 480;
        if (animFrame) cancelAnimationFrame(animFrame);

        function step(now) {
            const p = Math.min(1, (now - t0) / dur);
            const ease = 1 - Math.pow(1 - p, 3);
            displayMetrics.recall = start.recall + (target.recall - start.recall) * ease;
            displayMetrics.annMs = start.annMs + (target.annMs - start.annMs) * ease;
            displayMetrics.exactMs = start.exactMs + (target.exactMs - start.exactMs) * ease;
            displayMetrics.speedup = start.speedup + (target.speedup - start.speedup) * ease;
            drawGauges();
            if (p < 1) animFrame = requestAnimationFrame(step);
        }
        animFrame = requestAnimationFrame(step);
    }

    function drawGauges() {
        drawRecallRing(q('#pg-recall-canvas'), displayMetrics.recall);
        drawLatencyGauge(q('#pg-latency-canvas'), displayMetrics.annMs, displayMetrics.exactMs);
        const recallEl = q('#pg-recall-text');
        const annEl = q('#pg-ann-ms');
        const exactEl = q('#pg-exact-ms');
        const speedEl = q('#pg-speedup');
        if (recallEl) recallEl.textContent = (displayMetrics.recall * 100).toFixed(1) + '%';
        if (annEl) annEl.textContent = displayMetrics.annMs.toFixed(2);
        if (exactEl) exactEl.textContent = displayMetrics.exactMs.toFixed(2);
        if (speedEl) speedEl.textContent = displayMetrics.speedup.toFixed(2) + '×';
    }

    function drawRecallRing(canvas, recall) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const r = Math.min(w, h) * 0.38;
        ctx.clearRect(0, 0, w, h);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 8;
        ctx.stroke();
        const start = -Math.PI / 2;
        const end = start + Math.PI * 2 * Math.max(0, Math.min(1, recall));
        const grad = ctx.createLinearGradient(0, 0, w, h);
        if (recall >= 0.95) {
            grad.addColorStop(0, '#34d399');
            grad.addColorStop(1, '#10b981');
        } else if (recall >= 0.8) {
            grad.addColorStop(0, '#60a5fa');
            grad.addColorStop(1, '#2563eb');
        } else {
            grad.addColorStop(0, '#fbbf24');
            grad.addColorStop(1, '#f59e0b');
        }
        ctx.beginPath();
        ctx.arc(cx, cy, r, start, end);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        ctx.stroke();
    }

    function drawLatencyGauge(canvas, annMs, exactMs) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        const pad = 10;
        const gw = w - pad * 2;
        const gh = h - pad * 2;
        const maxMs = Math.max(exactMs * 1.15, annMs * 1.5, 5);
        roundRect(ctx, pad, pad, gw, gh, 8);
        ctx.fillStyle = '#f8fafc';
        ctx.fill();
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.stroke();
        const annX = pad + (annMs / maxMs) * gw;
        const exactX = pad + (exactMs / maxMs) * gw;
        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        roundRect(ctx, pad, pad + gh * 0.52, Math.max(4, exactX - pad), gh * 0.2, 4);
        ctx.fill();
        ctx.fillStyle = '#2563eb';
        roundRect(ctx, pad, pad + gh * 0.22, Math.max(4, annX - pad), gh * 0.24, 5);
        ctx.fill();
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    function injectStyles() {
        if (document.getElementById('ann-playground-styles')) return;
        const s = document.createElement('style');
        s.id = 'ann-playground-styles';
        s.textContent = `
@keyframes pg-pulse { 0%,100%{ box-shadow:0 0 0 0 rgba(37,99,235,0); } 50%{ box-shadow:0 0 0 3px rgba(37,99,235,0.12); } }
@keyframes pg-slide-in { from{ opacity:0; transform:translateY(6px); } to{ opacity:1; transform:none; } }
.pg-card-enter { animation: pg-slide-in 0.28s ease forwards; }
.pg-card-new { border-color:#86efac!important; background:#f0fdf4!important; }
.pg-probe-pulse { animation: pg-pulse 0.45s ease; }
.pg-slider { -webkit-appearance:none; width:100%; height:8px; border-radius:999px; background:linear-gradient(90deg,#dbeafe,#2563eb); outline:none; }
.pg-slider::-webkit-slider-thumb { -webkit-appearance:none; width:18px; height:18px; border-radius:50%; background:#fff; box-shadow:0 2px 6px rgba(37,99,235,0.35); cursor:pointer; border:2px solid #2563eb; }
.pg-preset.active { background:#0f172a!important; border-color:#0f172a!important; color:#fff!important; }
.pg-preset.active span { color:#e2e8f0!important; }
`;
        document.head.appendChild(s);
    }

    function buildControlsShell() {
        return `
<div class="pg-controls space-y-3">
  <div id="pg-presets" class="flex flex-col gap-2"></div>
  <div id="pg-slider-wrap" class="p-3 bg-slate-50 rounded-xl border border-slate-100">
    <div class="flex justify-between items-center mb-2">
      <span id="pg-param-label" class="text-[9px] font-black uppercase text-blue-600">param</span>
      <span id="pg-param-value" class="text-lg font-black text-slate-800 tabular-nums">—</span>
    </div>
    <input type="range" id="pg-slider" class="pg-slider" />
  </div>
  <div id="pg-instruments-mini" class="grid grid-cols-3 gap-2 text-center">
    <div class="p-2 bg-blue-50 rounded-xl border border-blue-100">
      <canvas id="pg-recall-canvas" width="56" height="56" class="mx-auto"></canvas>
      <p id="pg-recall-text" class="text-xs font-black text-blue-600">—</p>
    </div>
    <div class="p-2 bg-slate-50 rounded-xl border border-slate-100 flex flex-col justify-center">
      <p class="text-[8px] font-black uppercase text-slate-400">加速</p>
      <p id="pg-speedup" class="text-sm font-black text-slate-800">—</p>
    </div>
    <div class="p-2 bg-emerald-50 rounded-xl border border-emerald-100 flex flex-col justify-center text-[9px]">
      <p><span class="text-slate-400">ANN</span> <strong id="pg-ann-ms" class="text-blue-600">—</strong></p>
      <p><span class="text-slate-400">Ex</span> <strong id="pg-exact-ms" class="text-emerald-600">—</strong></p>
    </div>
  </div>
  <p id="pg-status" class="text-[9px] font-bold text-slate-400 uppercase tracking-wide text-center">Ready</p>
</div>`;
    }

    function buildResultsShell() {
        return `
<div class="pg-results space-y-5">
  <div class="flex flex-wrap justify-between items-center gap-3">
    <p class="text-xs text-slate-500">以 <strong class="text-emerald-600">Exact</strong> 为 Ground Truth，对比当前 ANN 索引在相同参数下的 Top-<span id="pg-k-label">10</span> 差异</p>
    <div id="pg-diff-badges" class="flex gap-2 flex-wrap"></div>
  </div>
  <canvas id="pg-latency-canvas" width="480" height="64" class="w-full rounded-xl border border-slate-100"></canvas>
  <div class="grid grid-cols-2 gap-4">
    <div>
      <p class="text-[10px] font-black uppercase text-emerald-600 mb-2">Exact 基线</p>
      <div id="pg-exact-list" class="space-y-2 max-h-64 overflow-y-auto"></div>
    </div>
    <div>
      <p class="text-[10px] font-black uppercase text-blue-600 mb-2">ANN 实时</p>
      <div id="pg-ann-list" class="space-y-2 max-h-64 overflow-y-auto"></div>
    </div>
  </div>
  <div>
    <p class="text-[10px] font-black uppercase text-slate-400 mb-2">Top-K 变化卡片</p>
    <div id="pg-cards" class="grid grid-cols-2 gap-2"></div>
  </div>
</div>`;
    }

    function renderPresets(cfg) {
        const wrap = q('#pg-presets');
        if (!wrap || !cfg?.presets) return;
        wrap.innerHTML = Object.entries(cfg.presets).map(([key, p]) => `
            <button type="button" data-preset="${key}" data-value="${p.value}"
                class="pg-preset text-left px-3 py-2 rounded-xl border border-slate-200 bg-white hover:border-blue-200 text-xs">
                <span class="font-black text-slate-700">${esc(p.label)}</span>
                <span class="block text-[9px] text-slate-400">${p.value}</span>
            </button>`).join('');
        wrap.querySelectorAll('.pg-preset').forEach(btn => {
            btn.addEventListener('click', () => {
                state.paramValue = Number(btn.dataset.value);
                const slider = q('#pg-slider');
                if (slider) slider.value = state.paramValue;
                const valEl = q('#pg-param-value');
                if (valEl) valEl.textContent = state.paramValue;
                highlightPreset(btn.dataset.preset);
                scheduleProbe();
            });
        });
    }

    function highlightPreset(key) {
        controlsRoot?.querySelectorAll('.pg-preset').forEach(b => {
            b.classList.toggle('active', key && b.dataset.preset === key);
        });
    }

    function bindControlEvents() {
        const slider = q('#pg-slider');
        if (slider) {
            slider.addEventListener('input', e => {
                state.paramValue = Number(e.target.value);
                const valEl = q('#pg-param-value');
                if (valEl) valEl.textContent = state.paramValue;
                highlightPreset(null);
                scheduleProbe();
            });
        }
    }

    function renderLists(data, prevAnnIds) {
        const entered = new Set(data.diff?.entered || []);
        const exited = new Set(data.diff?.exited || []);
        const prevSet = new Set(prevAnnIds || []);
        const badges = q('#pg-diff-badges');
        if (badges) {
            badges.innerHTML = `
                <span class="px-2 py-1 rounded-lg bg-emerald-50 text-emerald-700 text-[9px] font-black border border-emerald-100">重叠 ${data.overlap_count}/${data.k}</span>
                <span class="px-2 py-1 rounded-lg bg-blue-50 text-blue-700 text-[9px] font-black border border-blue-100">新进 ${entered.size}</span>`;
        }
        const rowHtml = (r, kind) => {
            const isNew = kind === 'ann' && entered.has(r.cell_id) && prevSet.size > 0 && !prevSet.has(r.cell_id);
            return `<div class="pg-card-enter flex items-center gap-2 p-2 rounded-lg border text-xs ${isNew ? 'border-emerald-200 bg-emerald-50 pg-card-new' : 'border-slate-100 bg-slate-50'}">
                <span class="font-black text-slate-400 w-4">#${r.rank}</span>
                <div class="min-w-0 flex-1"><p class="font-bold truncate">${esc(r.cell_id)}</p><p class="text-[9px] text-slate-500">${(r.similarity * 100).toFixed(1)}%</p></div>
            </div>`;
        };
        const exactList = q('#pg-exact-list');
        const annList = q('#pg-ann-list');
        if (exactList) exactList.innerHTML = (data.exact_results || []).map(r => rowHtml(r, 'exact')).join('');
        if (annList) annList.innerHTML = (data.ann_results || []).map(r => rowHtml(r, 'ann')).join('');
    }

    function renderCards(data, prevAnnIds) {
        const wrap = q('#pg-cards');
        if (!wrap) return;
        const entered = new Set(data.diff?.entered || []);
        const prevSet = new Set(prevAnnIds || []);
        wrap.innerHTML = (data.ann_results || []).map(r => {
            const isNew = entered.has(r.cell_id) && prevSet.size > 0 && !prevSet.has(r.cell_id);
            return `<div class="pg-card-enter p-3 rounded-xl border text-xs ${isNew ? 'border-emerald-200 bg-emerald-50 pg-card-new' : 'border-slate-100 bg-white'}">
                <div class="flex justify-between"><span class="font-black">#${r.rank}</span><span class="text-blue-600 font-black">${(r.similarity * 100).toFixed(1)}%</span></div>
                <p class="font-bold truncate mt-1">${esc(r.cell_id)}</p>
                ${isNew ? '<p class="text-[8px] text-emerald-600 font-black mt-1">新进 Top-K</p>' : ''}
            </div>`;
        }).join('');
    }

    function applyConfig(cfg, paramValue) {
        state.config = cfg;
        state.paramValue = paramValue ?? cfg.default;
        const slider = q('#pg-slider');
        const label = q('#pg-param-label');
        const valEl = q('#pg-param-value');
        if (label) label.textContent = cfg.param_name;
        if (slider) {
            slider.min = cfg.min;
            slider.max = cfg.max;
            slider.value = state.paramValue;
        }
        if (valEl) valEl.textContent = state.paramValue;
        renderPresets(cfg);
        const match = Object.entries(cfg.presets || {}).find(([, p]) => p.value === state.paramValue);
        highlightPreset(match ? match[0] : null);
    }

    async function runProbe() {
        if (!state || !state.indexId) return;
        const status = q('#pg-status');
        if (status) {
            status.textContent = '探测中…';
            status.className = 'text-[9px] font-bold text-blue-500 uppercase text-center';
        }
        const mini = q('#pg-instruments-mini');
        if (mini) mini.classList.add('pg-probe-pulse');

        const body = {
            index_id: Number(state.indexId),
            cell_id: state.cellId || undefined,
            k: state.topK,
        };
        if (state.config?.param_name === 'ef_search') body.ef_search = state.paramValue;
        else body.nprobe = state.paramValue;

        try {
            const data = await ScannAPI.playgroundProbe(state.datasetId, body);
            const prevIds = (state.lastData?.ann_results || []).map(r => r.cell_id);
            state.lastData = data;
            if (data.query_cell_id) state.cellId = data.query_cell_id;

            if (data.playground_config) {
                if (!state.config || state.config.param_name !== data.playground_config.param_name) {
                    applyConfig(data.playground_config, data.param_value);
                }
            }

            tweenMetrics({
                recall: data.recall_at_k,
                annMs: data.ann_latency_ms,
                exactMs: data.exact_latency_ms,
                speedup: data.speedup,
            });
            renderLists(data, prevIds);
            renderCards(data, prevIds);

            const kLabel = q('#pg-k-label');
            if (kLabel) kLabel.textContent = data.k;

            if (status) {
                status.textContent = `${data.index_type.toUpperCase()} · ${data.param_name}=${data.param_value}`;
                status.className = 'text-[9px] font-bold text-emerald-600 uppercase text-center';
            }
        } catch (e) {
            if (status) {
                status.textContent = e.error || '失败';
                status.className = 'text-[9px] font-bold text-red-500 uppercase text-center';
            }
        } finally {
            setTimeout(() => mini?.classList.remove('pg-probe-pulse'), 450);
        }
    }

    function scheduleProbe() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runProbe, DEBOUNCE_MS);
    }

    let controlsElRef = null;
    let resultsElRef = null;

    function mountEmbedded(controlsEl, resultsEl, opts) {
        if (!controlsEl) return;
        injectStyles();
        unmount();

        controlsElRef = controlsEl;
        resultsElRef = resultsEl;

        state = {
            datasetId: opts.datasetId,
            indexId: opts.indexId,
            topK: opts.topK || 10,
            cellId: opts.cellId || '',
            config: null,
            paramValue: 100,
            lastData: null,
        };

        controlsEl.innerHTML = buildControlsShell();
        controlsRoot = controlsEl.querySelector('.pg-controls') || controlsEl;
        if (resultsEl) {
            resultsEl.innerHTML = buildResultsShell();
            resultsRoot = resultsEl.querySelector('.pg-results') || resultsEl;
            const kLabel = q('#pg-k-label');
            if (kLabel) kLabel.textContent = state.topK;
        }

        bindControlEvents();
        drawGauges();
        runProbe();
    }

    function updateContext(opts) {
        if (!state) return;
        let changed = false;
        if (opts.datasetId != null && String(opts.datasetId) !== String(state.datasetId)) {
            state.datasetId = opts.datasetId;
            changed = true;
        }
        if (opts.indexId != null && String(opts.indexId) !== String(state.indexId)) {
            state.indexId = opts.indexId;
            state.config = null;
            changed = true;
        }
        if (opts.topK != null && opts.topK !== state.topK) {
            state.topK = opts.topK;
            changed = true;
        }
        if (opts.cellId != null && opts.cellId !== state.cellId) {
            state.cellId = opts.cellId;
            changed = true;
        }
        if (changed) {
            state.lastData = null;
            runProbe();
        }
    }

    function unmount() {
        clearTimeout(debounceTimer);
        if (animFrame) cancelAnimationFrame(animFrame);
        if (controlsElRef) controlsElRef.innerHTML = '';
        if (resultsElRef) resultsElRef.innerHTML = '';
        controlsRoot = null;
        resultsRoot = null;
        controlsElRef = null;
        resultsElRef = null;
        state = null;
        displayMetrics = { recall: 0, annMs: 0, exactMs: 0, speedup: 1 };
    }

    window.AnnPlayground = {
        mountEmbedded,
        updateContext,
        unmount,
        refresh: runProbe,
    };
})();
