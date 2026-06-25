/**
 * Scann Pro — PDF 报告导出（检索 / 细胞对比 / Benchmark）
 * 依赖 html2pdf.js（CDN）
 */
(function () {
    const BRAND = 'Scann Pro';
    const FONT = 'system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif';

    function stamp() {
        return new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    }

    function esc(s) {
        return String(s ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function fmtTime() {
        return new Date().toLocaleString('zh-CN', { hour12: false });
    }

    function canvasDataUrl(id) {
        const el = document.getElementById(id);
        if (!el || el.tagName !== 'CANVAS') return null;
        try {
            return el.toDataURL('image/png');
        } catch (_) {
            return null;
        }
    }

    function reportShell(theme, bodyHtml, subtitle) {
        const accent = theme.accent || '#2563eb';
        const accentLight = theme.accentLight || '#eff6ff';
        return `
<div class="scann-pdf-root" style="font-family:${FONT};color:#1e293b;background:#fff;width:794px;padding:0;margin:0;box-sizing:border-box;">
  <div style="position:relative;overflow:hidden;min-height:1120px;padding:36px 40px 48px;">
    <div style="position:absolute;top:42%;left:50%;transform:translate(-50%,-50%) rotate(-32deg);font-size:64px;font-weight:900;color:${accent};opacity:0.055;white-space:nowrap;pointer-events:none;z-index:0;letter-spacing:0.08em;">${BRAND}</div>

    <header style="position:relative;z-index:1;display:flex;justify-content:space-between;align-items:flex-start;border-bottom:3px solid ${accent};padding-bottom:18px;margin-bottom:24px;">
      <div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <div style="width:36px;height:36px;background:${accent};border-radius:10px;color:#fff;font-weight:900;font-style:italic;font-size:18px;display:flex;align-items:center;justify-content:center;">S</div>
          <span style="font-size:22px;font-weight:900;font-style:italic;letter-spacing:-0.03em;">SCANN <span style="color:${accent};">PRO</span></span>
        </div>
        <h1 style="margin:0;font-size:20px;font-weight:800;color:#0f172a;">${esc(theme.title)}</h1>
        ${subtitle ? `<p style="margin:6px 0 0;font-size:12px;color:#64748b;font-weight:600;">${esc(subtitle)}</p>` : ''}
      </div>
      <div style="text-align:right;font-size:10px;color:#94a3b8;line-height:1.6;">
        <div style="font-weight:800;color:${accent};text-transform:uppercase;letter-spacing:0.12em;">Official Report</div>
        <div>${fmtTime()}</div>
      </div>
    </header>

    <main style="position:relative;z-index:1;">${bodyHtml}</main>

    <footer style="position:relative;z-index:1;margin-top:32px;padding-top:14px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:9px;color:#94a3b8;">
      <span>${BRAND} · 单细胞向量检索与分析平台</span>
      <span>机密内部报告 · 请勿外传</span>
    </footer>
  </div>
</div>`;
    }

    function metaGrid(items) {
        const cells = items.filter(Boolean).map(([k, v]) => `
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px;">
        <div style="font-size:9px;font-weight:800;text-transform:uppercase;color:#94a3b8;letter-spacing:0.06em;margin-bottom:4px;">${esc(k)}</div>
        <div style="font-size:13px;font-weight:700;color:#334155;word-break:break-all;">${esc(v)}</div>
      </div>`).join('');
        return `<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:22px;">${cells}</div>`;
    }

    function dataTable(columns, rows, options) {
        const head = columns.map(c => `
        <th style="background:${options?.headBg || '#f1f5f9'};color:#475569;font-size:9px;font-weight:800;text-transform:uppercase;padding:10px 12px;text-align:left;border-bottom:2px solid #e2e8f0;">${esc(c.label)}</th>`).join('');
        const body = rows.map((row, ri) => {
            const tds = columns.map(c => {
                const raw = typeof c.get === 'function' ? c.get(row) : row[c.key];
                const align = c.align || 'left';
                const bg = typeof c.bg === 'function' ? c.bg(row) : (ri % 2 ? '#fafafa' : '#fff');
                return `<td style="padding:9px 12px;font-size:11px;border-bottom:1px solid #f1f5f9;background:${bg};text-align:${align};">${esc(raw)}</td>`;
            }).join('');
            return `<tr>${tds}</tr>`;
        }).join('');
        return `
<table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;margin-bottom:20px;">
  <thead><tr>${head}</tr></thead>
  <tbody>${body || `<tr><td colspan="${columns.length}" style="padding:16px;text-align:center;color:#94a3b8;">暂无数据</td></tr>`}</tbody>
</table>`;
    }

    function sectionTitle(text, color) {
        return `<h2 style="margin:0 0 12px;font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:0.14em;color:${color || '#64748b'};">${esc(text)}</h2>`;
    }

    function metricCards(items) {
        return `<div style="display:grid;grid-template-columns:repeat(${Math.min(items.length, 3)},1fr);gap:12px;margin-bottom:22px;">
      ${items.map(m => `
        <div style="background:${m.bg};border:1px solid ${m.border};border-radius:14px;padding:16px;text-align:center;">
          <div style="font-size:9px;font-weight:800;text-transform:uppercase;color:${m.labelColor};margin-bottom:6px;">${esc(m.label)}</div>
          <div style="font-size:22px;font-weight:900;color:${m.valueColor};">${esc(m.value)}</div>
        </div>`).join('')}
    </div>`;
    }

    function chartBlock(title, dataUrl) {
        if (!dataUrl) return '';
        return `
<div style="margin-bottom:20px;page-break-inside:avoid;">
  ${sectionTitle(title, '#6366f1')}
  <div style="border:1px solid #e2e8f0;border-radius:14px;padding:12px;background:#fafafa;">
    <img src="${dataUrl}" style="width:100%;height:auto;display:block;border-radius:8px;" alt="${esc(title)}" />
  </div>
</div>`;
    }

    async function renderPdf(html, filename) {
        if (typeof html2pdf === 'undefined') {
            alert('PDF 组件未加载，请刷新页面后重试');
            return;
        }
        const host = document.createElement('div');
        host.style.cssText = 'position:fixed;left:-9999px;top:0;z-index:-1;';
        host.innerHTML = html;
        document.body.appendChild(host);
        const el = host.querySelector('.scann-pdf-root') || host;

        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(15,23,42,0.35);z-index:99999;display:flex;align-items:center;justify-content:center;';
        overlay.innerHTML = '<div style="background:#fff;padding:20px 28px;border-radius:16px;font-family:system-ui,sans-serif;font-weight:700;color:#334155;box-shadow:0 20px 40px rgba(0,0,0,0.15);">正在生成 Scann Pro PDF…</div>';
        document.body.appendChild(overlay);

        try {
            await html2pdf().set({
                margin: [8, 8, 12, 8],
                filename: filename + '.pdf',
                image: { type: 'jpeg', quality: 0.92 },
                html2canvas: { scale: 2, useCORS: true, logging: false, letterRendering: true },
                jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
                pagebreak: { mode: ['avoid-all', 'css', 'legacy'] },
            }).from(el).save();
        } catch (err) {
            console.error(err);
            alert('PDF 生成失败，请稍后重试');
        } finally {
            document.body.removeChild(host);
            if (overlay.parentNode) document.body.removeChild(overlay);
        }
    }

    /* ── 1. 检索结果报告 ── */
    async function exportSearch(payload) {
        const rows = payload.results || [];
        if (!rows.length) return alert('暂无检索结果可导出');

        const columns = [
            { key: 'rank', label: '排名', align: 'center' },
            { key: 'cell_id', label: 'Cell ID' },
            { key: 'dataset_id', label: '数据集', get: r => r.dataset_id ?? '—' },
            { key: 'cell_type', label: '细胞类型', get: r => r.cell_type || r.metadata?.cell_type || '—' },
            { key: 'similarity', label: '相似度', get: r => r.similarity != null ? (r.similarity * 100).toFixed(2) + '%' : '—', align: 'right' },
            { key: 'distance', label: '距离', get: r => r.distance?.toFixed?.(4) ?? '—', align: 'right' },
        ];

        const meta = metaGrid([
            ['查询输入', payload.queryInput || '—'],
            ['检索类型', payload.queryType || '—'],
            ['响应耗时', payload.queryTimeMs != null ? payload.queryTimeMs + ' ms' : '—'],
            ['联合检索', payload.isJoint ? '是' : '否'],
            ['结果数量', rows.length + ' 条'],
            ['数据集', payload.datasetName || '—'],
        ]);

        const explainBlock = payload.bioExplain
            ? `<div style="background:#f5f3ff;border:1px solid #ddd6fe;border-radius:14px;padding:16px;margin-bottom:20px;">
                ${sectionTitle('LLM 生物学解读', '#7c3aed')}
                <p style="margin:0;font-size:12px;line-height:1.7;color:#4c1d95;white-space:pre-wrap;">${esc(payload.bioExplain)}</p>
               </div>`
            : '';

        const body = meta + explainBlock + sectionTitle('Top-K 检索结果', '#2563eb') + dataTable(columns, rows, { headBg: '#eff6ff' });
        const html = reportShell(
            { title: '检索结果报告', accent: '#2563eb', accentLight: '#eff6ff' },
            body,
            'Similarity Search · Result List',
        );
        await renderPdf(html, `scann_search_${stamp()}`);
    }

    /* ── 2. 一对一细胞对比报告 ── */
    async function exportCellCompare(payload) {
        const data = payload.compareData;
        if (!data) return alert('暂无对比数据可导出');

        const qId = data.query?.cell_id || payload.queryCellId || '—';
        const tId = data.target?.cell_id || payload.targetCellId || '—';
        const sim = data.metrics?.similarity ?? payload.searchSimilarity;
        const dist = data.metrics?.distance ?? payload.searchDistance;

        const metrics = metricCards([
            {
                label: '相似度',
                value: sim != null ? (sim * 100).toFixed(2) + '%' : '—',
                bg: '#fef2f2', border: '#fecaca', labelColor: '#f87171', valueColor: '#dc2626',
            },
            {
                label: 'L2 距离',
                value: dist != null ? Number(dist).toFixed(4) : '—',
                bg: '#eff6ff', border: '#bfdbfe', labelColor: '#60a5fa', valueColor: '#2563eb',
            },
            {
                label: '向量维度',
                value: (data.metrics?.n_dims_total ?? '—') + (data.metrics?.n_dims_total ? 'D' : ''),
                bg: '#f8fafc', border: '#e2e8f0', labelColor: '#94a3b8', valueColor: '#334155',
            },
        ]);

        const meta = metaGrid([
            ['查询细胞', qId],
            ['对比细胞', tId],
            ['数据集', payload.datasetName || payload.datasetId || '—'],
            ['检索排名', payload.rank != null ? '#' + payload.rank : '—'],
            ['对比模式', data.has_query ? 'Query vs Hit' : 'Hit 详情（无查询细胞）'],
        ]);

        let metaTable = '';
        if (data.metadata_rows?.length) {
            metaTable = sectionTitle('元数据字段对比', '#dc2626') + dataTable(
                [
                    { key: 'field', label: '字段' },
                    { key: 'query_value', label: '查询细胞', get: r => r.query_value ?? '—' },
                    { key: 'target_value', label: '命中细胞', get: r => r.target_value ?? '—' },
                    {
                        key: 'match', label: '一致', align: 'center',
                        get: r => r.match ? '✓' : '≠',
                        bg: r => r.match ? '#fff' : '#fffbeb',
                    },
                ],
                data.metadata_rows,
                { headBg: '#fef2f2' },
            );
        }

        let vectorTable = '';
        if (data.vector_dims?.length) {
            vectorTable = sectionTitle('向量维度预览（前 ' + data.vector_dims.length + ' 维）', '#2563eb') + dataTable(
                [
                    { key: 'dim', label: 'Dim', align: 'center' },
                    { key: 'query', label: 'Query', get: r => r.query != null ? r.query.toFixed(4) : '—', align: 'right' },
                    { key: 'target', label: 'Target', get: r => r.target?.toFixed?.(4) ?? '—', align: 'right' },
                    { key: 'delta', label: '|Δ|', get: r => r.delta != null ? r.delta.toFixed(4) : '—', align: 'right' },
                ],
                data.vector_dims,
                { headBg: '#eff6ff' },
            );
        }

        const charts = [
            chartBlock('嵌入空间位置', payload.chartMap || canvasDataUrl('cmp-map')),
            chartBlock('向量剖面对比', payload.chartVector || canvasDataUrl('cmp-vector')),
            chartBlock('维度差异 Δ', payload.chartDelta || canvasDataUrl('cmp-delta')),
        ].join('');

        const body = metrics + meta + metaTable + vectorTable
            + (charts ? sectionTitle('可视化图表', '#6366f1') + charts : '');

        const html = reportShell(
            { title: '细胞一对一对比报告', accent: '#dc2626' },
            body,
            `${qId}  vs  ${tId}`,
        );
        await renderPdf(html, `scann_cell_compare_${stamp()}`);
    }

    /* ── 3. Benchmark 性能评测报告 ── */
    async function exportBenchmark(payload) {
        const rows = payload.batchResults?.length
            ? payload.batchResults
            : (payload.singleResult ? [payload.singleResult] : []);
        if (!rows.length) return alert('暂无 Benchmark 数据可导出，请先运行评测');

        const columns = [
            { key: 'index_type', label: '索引类型', get: r => (r.index_type || '').toUpperCase() },
            {
                key: 'recall_at_k', label: 'Recall@K', align: 'right',
                get: r => r.recall_at_k != null ? (r.recall_at_k * 100).toFixed(2) + '%' : '—',
                bg: r => {
                    if (r.recall_at_k == null) return '#fff';
                    if (r.recall_at_k >= 0.95) return '#f0fdf4';
                    if (r.recall_at_k >= 0.85) return '#fffbeb';
                    return '#fef2f2';
                },
            },
            { key: 'speedup', label: '加速比', align: 'right', get: r => r.speedup != null ? r.speedup.toFixed(2) + '×' : '—' },
            {
                key: 'avg_ann_time_ms', label: 'ANN 延迟 (ms)', align: 'right',
                get: r => r.avg_ann_time_ms?.toFixed?.(2) ?? r.avg_latency_ms?.toFixed?.(2) ?? '—',
            },
            {
                key: 'avg_exact_time_ms', label: 'Exact 延迟 (ms)', align: 'right',
                get: r => r.avg_exact_time_ms?.toFixed?.(2) ?? '—',
            },
            { key: 'index_size_label', label: '索引大小', get: r => r.index_size_label || (r.index_size_mb != null ? r.index_size_mb + ' MB' : '—') },
            {
                key: 'memory_ratio', label: '内存比', align: 'right',
                get: r => r.memory_ratio != null ? (r.memory_ratio * 100).toFixed(1) + '%' : '—',
            },
        ];

        const bestRecall = rows.reduce((a, b) => (b.recall_at_k ?? 0) > (a.recall_at_k ?? 0) ? b : a, rows[0]);
        const annRows = rows.filter(r => r.index_type !== 'exact');
        const bestSpeed = annRows.length
            ? annRows.reduce((a, b) => (b.speedup ?? 0) > (a.speedup ?? 0) ? b : a, annRows[0])
            : null;

        const summary = metricCards([
            {
                label: '最高 Recall',
                value: bestRecall?.recall_at_k != null ? (bestRecall.recall_at_k * 100).toFixed(1) + '%' : '—',
                bg: '#f0fdf4', border: '#bbf7d0', labelColor: '#4ade80', valueColor: '#15803d',
            },
            {
                label: '最佳加速比',
                value: bestSpeed?.speedup != null ? bestSpeed.speedup.toFixed(2) + '×' : '—',
                bg: '#faf5ff', border: '#e9d5ff', labelColor: '#a78bfa', valueColor: '#7c3aed',
            },
            {
                label: '评测索引数',
                value: String(rows.length),
                bg: '#eff6ff', border: '#bfdbfe', labelColor: '#60a5fa', valueColor: '#2563eb',
            },
        ]);

        const meta = metaGrid([
            ['数据集', payload.datasetName || payload.datasetId || '—'],
            ['Top-K', payload.topK ?? '—'],
            ['准确度基线', 'Exact'],
            ['采样查询数', rows[0]?.n_queries ?? '—'],
        ]);

        let sweepBlock = '';
        const sweep = payload.paramSweep;
        if (sweep?.results?.length) {
            sweepBlock = sectionTitle(`参数扫描 · ${sweep.param_name}`, '#7c3aed') + dataTable(
                [
                    { key: 'param_value', label: sweep.param_name, align: 'center' },
                    { key: 'recall_at_k', label: 'Recall@K', align: 'right', get: r => (r.recall_at_k * 100).toFixed(2) + '%' },
                    { key: 'avg_latency_ms', label: '延迟 (ms)', align: 'right', get: r => r.avg_latency_ms?.toFixed?.(2) ?? '—' },
                ],
                sweep.results,
                { headBg: '#f5f3ff' },
            );
        }

        const charts = [
            chartBlock('四索引 Recall / 加速比对比', payload.chartCompare || canvasDataUrl('bench-compare-chart')),
            chartBlock('ANN 参数扫描曲线', payload.chartParamSweep || canvasDataUrl('bench-param-sweep-chart')),
        ].join('');

        const body = summary + meta
            + sectionTitle('索引性能明细', '#7c3aed')
            + dataTable(columns, rows, { headBg: '#f5f3ff' })
            + sweepBlock
            + (charts.trim() ? sectionTitle('可视化图表', '#6366f1') + charts : '');

        const html = reportShell(
            { title: 'Benchmark 性能评测报告', accent: '#7c3aed' },
            body,
            'ANN Quantization · Recall · Latency · Memory',
        );
        await renderPdf(html, `scann_benchmark_${stamp()}`);
    }

    window.ScannPdf = {
        exportSearch,
        exportCellCompare,
        exportBenchmark,
    };
})();
