/**
 * Scann 报告导出：JSON / CSV / Markdown / HTML
 */
(function () {
    const FORMATS = ['json', 'csv', 'md', 'html', 'pdf'];

    function downloadBlob(content, filename, mime) {
        const blob = new Blob([content], { type: mime + ';charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function stamp() {
        return new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    }

    function escHtml(s) {
        return String(s ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function csvEscape(v) {
        const s = v == null ? '' : String(v);
        return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    }

    function rowsToCsv(rows, columns) {
        const header = columns.map(c => csvEscape(c.label)).join(',');
        const body = rows.map(row =>
            columns.map(c => csvEscape(typeof c.get === 'function' ? c.get(row) : row[c.key])).join(',')
        ).join('\n');
        return header + '\n' + body;
    }

    function wrapHtml(title, bodyHtml) {
        return `<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>${escHtml(title)}</title>
<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1e293b}
h1{font-size:1.5rem} table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;font-size:13px}
th{background:#f8fafc;font-size:11px;text-transform:uppercase;color:#64748b}
.meta{color:#64748b;font-size:13px;margin-bottom:1.5rem}
</style></head><body>
<h1>${escHtml(title)}</h1>
<div class="meta">导出时间：${new Date().toLocaleString('zh-CN')}</div>
${bodyHtml}
</body></html>`;
    }

    const BENCH_COLUMNS = [
        { key: 'index_type', label: '索引类型' },
        { key: 'recall_at_k', label: 'Recall@K', get: r => r.recall_at_k != null ? (r.recall_at_k * 100).toFixed(2) + '%' : '' },
        { key: 'speedup', label: '加速比', get: r => r.speedup != null ? r.speedup.toFixed(2) + '×' : '' },
        { key: 'avg_ann_time_ms', label: 'ANN耗时(ms)', get: r => r.avg_ann_time_ms?.toFixed?.(2) ?? r.avg_latency_ms?.toFixed?.(2) ?? '' },
        { key: 'avg_exact_time_ms', label: 'Exact耗时(ms)', get: r => r.avg_exact_time_ms?.toFixed?.(2) ?? '' },
        { key: 'index_size_label', label: '索引大小', get: r => r.index_size_label || (r.index_size_mb != null ? r.index_size_mb + ' MB' : '') },
        { key: 'memory_ratio', label: '内存比(vs Exact)', get: r => r.memory_ratio != null ? (r.memory_ratio * 100).toFixed(1) + '%' : '' },
        { key: 'n_queries', label: '采样次数' },
        { key: 'k', label: 'K' },
    ];

    const SEARCH_COLUMNS = [
        { key: 'rank', label: '排名' },
        { key: 'cell_id', label: 'Cell ID' },
        { key: 'dataset_id', label: '数据集ID', get: r => r.dataset_id ?? '' },
        { key: 'cell_type', label: '细胞类型' },
        { key: 'similarity', label: '相似度', get: r => r.similarity?.toFixed?.(4) ?? '' },
        { key: 'distance', label: '距离', get: r => r.distance?.toFixed?.(4) ?? '' },
    ];

    const HISTORY_COLUMNS = [
        { key: 'created_at', label: '时间' },
        { key: 'dataset_id', label: '数据集ID' },
        { key: 'query_type', label: '类型' },
        { key: 'query_input', label: '查询' },
        { key: 'index_type', label: '索引' },
        { key: 'query_time', label: '耗时(ms)' },
        { key: 'result_ids', label: '结果IDs', get: r => Array.isArray(r.result_ids) ? r.result_ids.join('; ') : (r.result_ids || '') },
    ];

    function tableHtml(columns, rows) {
        const th = columns.map(c => `<th>${escHtml(c.label)}</th>`).join('');
        const trs = rows.map(row => {
            const tds = columns.map(c => {
                const v = typeof c.get === 'function' ? c.get(row) : row[c.key];
                return `<td>${escHtml(v)}</td>`;
            }).join('');
            return `<tr>${tds}</tr>`;
        }).join('');
        return `<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
    }

    function mdTable(columns, rows) {
        const header = '| ' + columns.map(c => c.label).join(' | ') + ' |';
        const sep = '| ' + columns.map(() => '---').join(' | ') + ' |';
        const body = rows.map(row =>
            '| ' + columns.map(c => {
                const v = typeof c.get === 'function' ? c.get(row) : row[c.key];
                return String(v ?? '').replace(/\|/g, '\\|');
            }).join(' | ') + ' |'
        ).join('\n');
        return header + '\n' + sep + '\n' + body;
    }

    function exportGeneric({ title, filenamePrefix, rows, columns, meta, format }) {
        const fname = `${filenamePrefix}_${stamp()}`;
        if (format === 'json') {
            downloadBlob(JSON.stringify({ title, exported_at: new Date().toISOString(), meta, rows }, null, 2), fname + '.json', 'application/json');
            return;
        }
        if (format === 'csv') {
            downloadBlob(rowsToCsv(rows, columns), fname + '.csv', 'text/csv');
            return;
        }
        const metaMd = meta ? Object.entries(meta).map(([k, v]) => `- **${k}**: ${v}`).join('\n') + '\n\n' : '';
        if (format === 'md') {
            downloadBlob(`# ${title}\n\n${metaMd}${mdTable(columns, rows)}\n`, fname + '.md', 'text/markdown');
            return;
        }
        if (format === 'html') {
            const metaHtml = meta ? '<ul>' + Object.entries(meta).map(([k, v]) => `<li><strong>${escHtml(k)}</strong>: ${escHtml(v)}</li>`).join('') + '</ul>' : '';
            downloadBlob(wrapHtml(title, metaHtml + tableHtml(columns, rows)), fname + '.html', 'text/html');
        }
    }

    function exportBenchmark(payload, format) {
        const rows = payload.batchResults?.length ? payload.batchResults : (payload.singleResult ? [payload.singleResult] : []);
        if (!rows.length) return alert('暂无 Benchmark 数据可导出，请先运行评测');
        if (format === 'pdf') {
            if (!window.ScannPdf) return alert('PDF 模块未加载');
            return ScannPdf.exportBenchmark(payload);
        }
        exportGeneric({
            title: 'Scann Benchmark 报告',
            filenamePrefix: 'scann_benchmark',
            rows,
            columns: BENCH_COLUMNS,
            meta: {
                数据集: payload.datasetName || payload.datasetId || '—',
                TopK: payload.topK ?? '—',
                基线: 'Exact',
            },
            format,
        });
    }

    function exportSearch(payload, format) {
        const rows = payload.results || [];
        if (!rows.length) return alert('暂无检索结果可导出');
        if (format === 'pdf') {
            if (!window.ScannPdf) return alert('PDF 模块未加载');
            return ScannPdf.exportSearch(payload);
        }
        exportGeneric({
            title: 'Scann 检索报告',
            filenamePrefix: 'scann_search',
            rows,
            columns: SEARCH_COLUMNS,
            meta: {
                查询: payload.queryInput || '—',
                类型: payload.queryType || '—',
                耗时: payload.queryTimeMs != null ? payload.queryTimeMs + ' ms' : '—',
                联合检索: payload.isJoint ? '是' : '否',
            },
            format,
        });
    }

    function exportQueryHistory(history, format, meta) {
        if (!history?.length) return alert('暂无查询历史可导出');
        exportGeneric({
            title: 'Scann 查询历史',
            filenamePrefix: 'scann_query_history',
            rows: history,
            columns: HISTORY_COLUMNS,
            meta: meta || {},
            format,
        });
    }

    function exportEvalReports(reports, format, meta) {
        if (!reports?.length) return alert('暂无评测报告可导出');
        if (format === 'pdf') {
            if (!window.ScannPdf) return alert('PDF 模块未加载');
            return ScannPdf.exportBenchmark({
                batchResults: reports,
                datasetName: meta?.筛选数据集,
                topK: reports[0]?.k,
            });
        }
        exportGeneric({
            title: 'Scann 评测报告归档',
            filenamePrefix: 'scann_eval_reports',
            rows: reports,
            columns: BENCH_COLUMNS,
            meta: meta || {},
            format,
        });
    }

    window.ScannExport = {
        FORMATS,
        exportBenchmark,
        exportSearch,
        exportQueryHistory,
        exportEvalReports,
    };
})();
