/**
 * 检索结果可视化（Chart.js）
 * 散点图：在 2D 嵌入空间中高亮查询细胞与 Top-K 邻居
 * 柱状图：相似度排序、检索结果细胞类型组成
 */
(function () {
    const charts = {};

    function destroy(id) {
        if (charts[id]) {
            charts[id].destroy();
            delete charts[id];
        }
    }

    function destroyAll(prefix) {
        Object.keys(charts).forEach((key) => {
            if (!prefix || key.startsWith(prefix)) destroy(key);
        });
    }

    function cellIdIndexMap(cellIds) {
        const map = {};
        cellIds.forEach((id, i) => {
            const key = String(id);
            map[key] = i;
            map[key.trim()] = i;
        });
        return map;
    }

    function resolveQueryIndex(ids, idxMap, queryCellId, fallbackId) {
        const candidates = [queryCellId, fallbackId].filter(Boolean).map(String);
        for (const c of candidates) {
            if (c === 'Vector') continue;
            if (idxMap[c] !== undefined) return idxMap[c];
        }
        return undefined;
    }

    function hitColor(rank, total) {
        const hue = 220 - (rank - 1) * (120 / Math.max(total, 1));
        return `hsla(${hue}, 82%, 52%, 0.92)`;
    }

    /** 2D 散点：灰点=背景，彩色=命中，红圈=查询细胞；onCellClick(cellId, rank) */
    function renderScatter(canvasId, embed, queryCellId, results, sourceLabel, fallbackQueryId, onCellClick) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !embed) return;

        const xs = embed.umap_x || embed.pca_x;
        const ys = embed.umap_y || embed.pca_y;
        const ids = embed.cell_ids || [];
        if (!xs || !ys || !ids.length) return;

        const idxMap = cellIdIndexMap(ids);
        const queryIdx = resolveQueryIndex(ids, idxMap, queryCellId, fallbackQueryId);
        const resolvedQueryId = queryIdx !== undefined ? String(ids[queryIdx]) : null;
        const hitSet = new Set((results || []).map((r) => String(r.cell_id)));
        const bg = [];
        ids.forEach((id, i) => {
            const sid = String(id);
            if (sid !== resolvedQueryId && !hitSet.has(sid)) {
                bg.push({ x: xs[i], y: ys[i] });
            }
        });

        const hits = (results || [])
            .filter((r) => idxMap[String(r.cell_id)] !== undefined)
            .map((r) => {
                const i = idxMap[String(r.cell_id)];
                return {
                    x: xs[i],
                    y: ys[i],
                    label: `#${r.rank} ${r.cell_id}`,
                    rank: r.rank,
                    cellId: String(r.cell_id),
                };
            });

        const datasets = [
            {
                label: '其他细胞',
                data: bg,
                order: 1,
                pointRadius: 3,
                pointBackgroundColor: 'rgba(148, 163, 184, 0.35)',
                pointBorderWidth: 0,
            },
        ];

        if (hits.length) {
            datasets.push({
                label: 'Top-K 相似细胞',
                data: hits,
                order: 2,
                pointRadius: 9,
                pointBackgroundColor: hits.map((h) => hitColor(h.rank, hits.length)),
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
            });
        }

        let titleSuffix = '';
        if (queryIdx !== undefined) {
            datasets.push({
                label: '查询细胞',
                data: [{ x: xs[queryIdx], y: ys[queryIdx], label: resolvedQueryId }],
                order: 3,
                pointRadius: 11,
                pointStyle: 'circle',
                pointBackgroundColor: 'rgba(239, 68, 68, 0.45)',
                pointBorderColor: '#ef4444',
                pointBorderWidth: 2.5,
            });
        } else if (queryCellId === 'Vector' || fallbackQueryId === null) {
            titleSuffix = '（向量检索，无单一查询细胞位置）';
        } else {
            titleSuffix = '（未在嵌入坐标中找到查询细胞 ID）';
        }

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12,
                            usePointStyle: true,
                            font: { size: 11, weight: '600' },
                            generateLabels(chart) {
                                const defaults = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                                return defaults.map((item) => {
                                    if (item.text === '查询细胞') {
                                        item.pointStyle = 'circle';
                                        item.fillStyle = '#ef4444';
                                        item.strokeStyle = '#ef4444';
                                        item.lineWidth = 2;
                                    }
                                    return item;
                                });
                            },
                        },
                    },
                    title: {
                        display: true,
                        text: `${sourceLabel || '2D 嵌入'} · 查询细胞与邻居空间分布${titleSuffix}`,
                        font: { size: 13, weight: '700' },
                    },
                    tooltip: {
                        callbacks: {
                            label(ctx) {
                                const p = ctx.raw;
                                return p.label || `(${p.x.toFixed(2)}, ${p.y.toFixed(2)})`;
                            },
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: 'Dim 1' }, grid: { color: 'rgba(0,0,0,0.04)' } },
                    y: { title: { display: true, text: 'Dim 2' }, grid: { color: 'rgba(0,0,0,0.04)' } },
                },
                onClick(evt, elements, chart) {
                    if (!elements.length || typeof onCellClick !== 'function') return;
                    const pt = chart.data.datasets[elements[0].datasetIndex].data[elements[0].index];
                    if (pt.cellId) onCellClick(pt.cellId, pt.rank);
                },
            },
        });
        charts[canvasId].resize();
    }

    /** 水平柱状图：Top-K 相似度 */
    function renderSimilarityBars(canvasId, results) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !results?.length) return;

        const sorted = [...results].sort((a, b) => a.rank - b.rank);
        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: {
                labels: sorted.map((r) => `#${r.rank} ${r.cell_id}`),
                datasets: [{
                    label: '相似度 (%)',
                    data: sorted.map((r) => +(r.similarity * 100).toFixed(2)),
                    backgroundColor: sorted.map((r) => hitColor(r.rank, sorted.length)),
                    borderRadius: 6,
                }],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Top-K 细胞相似度（越高越相似）',
                        font: { size: 13, weight: '700' },
                    },
                },
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        title: { display: true, text: 'Similarity %' },
                    },
                },
            },
        });
    }

    /** 检索结果细胞类型组成 */
    function renderCellTypeBars(canvasId, results) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !results?.length) return;

        const counts = {};
        results.forEach((r) => {
            const t = r.cell_type || r.metadata?.cell_type || '未知';
            counts[t] = (counts[t] || 0) + 1;
        });
        const labels = Object.keys(counts);
        const values = labels.map((k) => counts[k]);
        const colors = labels.map((_, i) => `hsla(${200 + i * 35}, 70%, 55%, 0.85)`);

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: '细胞数',
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 8,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: '检索结果 · 细胞类型组成',
                        font: { size: 13, weight: '700' },
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1 },
                        title: { display: true, text: 'Count' },
                    },
                },
            },
        });
    }

    /** 距离柱状图（补充：向量空间 L2 距离，越小越近） */
    function renderDistanceBars(canvasId, results) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !results?.length) return;

        const sorted = [...results].sort((a, b) => a.rank - b.rank);
        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: {
                labels: sorted.map((r) => `#${r.rank} ${r.cell_id}`),
                datasets: [{
                    label: 'L2 距离',
                    data: sorted.map((r) => +r.distance.toFixed(4)),
                    backgroundColor: 'rgba(59, 130, 246, 0.75)',
                    borderRadius: 6,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Top-K 向量距离（越小越相似）',
                        font: { size: 13, weight: '700' },
                    },
                },
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Distance' } },
                },
            },
        });
    }

    async function fetchEmbedData(datasetId) {
        try {
            const umap = await window.ScannAPI.getUmapData(datasetId);
            return { ...umap, source: 'UMAP' };
        } catch (_) {
            const pca = await window.ScannAPI.getPcaData(datasetId);
            return {
                cell_ids: pca.cell_ids,
                umap_x: pca.pca_x,
                umap_y: pca.pca_y,
                cell_types: pca.cell_types,
                source: 'PCA (Dim1×Dim2)',
            };
        }
    }

    window.SearchViz = {
        destroyAll,
        renderScatter,
        renderSimilarityBars,
        renderCellTypeBars,
        renderDistanceBars,
        fetchEmbedData,
    };
})();
