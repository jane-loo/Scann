/**
 * Benchmark 可视化：四种索引（HNSW / IVF-Flat / IVF-PQ / Exact）Recall@K 与加速比对比
 */
(function () {
    const charts = {};

    const INDEX_ORDER = ['hnsw', 'ivf_flat', 'ivf_pq', 'exact'];
    const INDEX_LABELS = {
        hnsw: 'HNSW',
        ivf_flat: 'IVF-Flat',
        ivf_pq: 'IVF-PQ',
        exact: 'Exact',
    };
    const INDEX_COLORS = {
        hnsw: { recall: 'rgba(51, 65, 85, 0.85)', speedup: 'rgba(51, 65, 85, 0.45)' },
        ivf_flat: { recall: 'rgba(59, 130, 246, 0.85)', speedup: 'rgba(59, 130, 246, 0.45)' },
        ivf_pq: { recall: 'rgba(139, 92, 246, 0.85)', speedup: 'rgba(139, 92, 246, 0.45)' },
        exact: { recall: 'rgba(16, 185, 129, 0.85)', speedup: 'rgba(16, 185, 129, 0.45)' },
    };

    function destroy(id) {
        if (charts[id]) {
            charts[id].destroy();
            delete charts[id];
        }
    }

    function destroyAll() {
        Object.keys(charts).forEach(destroy);
    }

    /**
     * rows: [{ index_type, recall_at_k, speedup, avg_ann_time_ms, present }]
     * options: { topK, datasetName, subtitle }
     */
    function renderCombined(canvasId, rows, options) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !rows?.length) return;

        const topK = options?.topK || 10;
        const labels = rows.map((r) => INDEX_LABELS[r.index_type] || r.index_type);
        const recallColors = rows.map((r) =>
            r.present ? INDEX_COLORS[r.index_type]?.recall : 'rgba(203, 213, 225, 0.6)'
        );
        const speedupColors = rows.map((r) =>
            r.present ? INDEX_COLORS[r.index_type]?.speedup : 'rgba(226, 232, 240, 0.5)'
        );

        const recallData = rows.map((r) =>
            r.present && r.recall_at_k != null ? r.recall_at_k * 100 : null
        );
        const speedupData = rows.map((r) =>
            r.present && r.speedup != null ? r.speedup : null
        );

        const titleParts = ['四种索引 Benchmark 对比'];
        if (options?.datasetName) titleParts.push(options.datasetName);
        titleParts.push(`Recall@${topK} · 加速比 (vs Exact)`);

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: `Recall@${topK} (%)`,
                        data: recallData,
                        backgroundColor: recallColors,
                        borderRadius: 8,
                        borderSkipped: false,
                        yAxisID: 'y',
                        order: 2,
                    },
                    {
                        label: '加速比 (×)',
                        data: speedupData,
                        backgroundColor: speedupColors,
                        borderRadius: 8,
                        borderSkipped: false,
                        yAxisID: 'y1',
                        order: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            padding: 16,
                            font: { size: 12, weight: '700' },
                        },
                    },
                    title: {
                        display: true,
                        text: titleParts.join(' · '),
                        font: { size: 14, weight: '800' },
                        padding: { bottom: 8 },
                    },
                    subtitle: options?.subtitle
                        ? { display: true, text: options.subtitle, font: { size: 11 }, color: '#64748b', padding: { bottom: 12 } }
                        : { display: false },
                    tooltip: {
                        callbacks: {
                            label(ctx) {
                                const row = rows[ctx.dataIndex];
                                if (!row?.present) return `${ctx.dataset.label}: 未评测`;
                                if (ctx.dataset.label.includes('Recall')) {
                                    return `Recall@${topK}: ${(row.recall_at_k * 100).toFixed(1)}%`;
                                }
                                const lines = [`加速比: ${row.speedup?.toFixed(2)}×`];
                                if (row.avg_ann_time_ms != null) {
                                    lines.push(`ANN 延迟: ${row.avg_ann_time_ms.toFixed(2)} ms`);
                                }
                                if (row.avg_exact_time_ms != null && row.index_type !== 'exact') {
                                    lines.push(`Exact 基线: ${row.avg_exact_time_ms.toFixed(2)} ms`);
                                }
                                if (row.index_size_label) {
                                    lines.push(`索引大小: ${row.index_size_label}`);
                                }
                                if (row.memory_ratio != null && row.index_type !== 'exact') {
                                    lines.push(`相对 Exact 内存: ${(row.memory_ratio * 100).toFixed(1)}%`);
                                }
                                return lines;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 12, weight: '700' } },
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        min: 0,
                        max: 100,
                        title: { display: true, text: 'Recall (%)', font: { weight: '600' } },
                        grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        title: { display: true, text: 'Speedup (×)', font: { weight: '600' } },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
        });
    }

    /** 参数扫描：Recall vs 延迟 trade-off 折线图 */
    function renderParamSweep(canvasId, rows, options) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !rows?.length) return;

        const labels = rows.map(r => String(r.param_value));
        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Recall@K (%)',
                        data: rows.map(r => r.recall_at_k * 100),
                        borderColor: 'rgba(59, 130, 246, 1)',
                        backgroundColor: 'rgba(59, 130, 246, 0.15)',
                        yAxisID: 'y',
                        tension: 0.25,
                    },
                    {
                        label: '延迟 (ms)',
                        data: rows.map(r => r.avg_latency_ms),
                        borderColor: 'rgba(139, 92, 246, 1)',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.25,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `参数扫描 · ${options?.paramName || 'param'} · ${options?.indexType || ''}`,
                        font: { size: 14, weight: '800' },
                    },
                    legend: { position: 'top' },
                },
                scales: {
                    x: { title: { display: true, text: options?.paramName || 'param' } },
                    y: {
                        type: 'linear', position: 'left', min: 0, max: 100,
                        title: { display: true, text: 'Recall (%)' },
                    },
                    y1: {
                        type: 'linear', position: 'right', min: 0,
                        title: { display: true, text: 'Latency (ms)' },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
        });
    }

    window.BenchmarkViz = {
        INDEX_ORDER,
        INDEX_LABELS,
        destroy,
        destroyAll,
        renderCombined,
        renderParamSweep,
    };
})();
