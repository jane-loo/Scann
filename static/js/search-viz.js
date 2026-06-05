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

    /** 元数据多属性分布（横向堆叠百分比条） */
    function renderMetaProfile(canvasId, results) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !results?.length) return;

        // 感兴趣的元数据字段，按展示优先级排列
        const CANDIDATE_FIELDS = [
            'cell_type', 'disease', 'sex', 'AgeGroup',
            'donor_id', 'tissue', 'author_cell_type', 'Phase',
        ];

        // 统计每个字段的各值出现次数
        const fieldCounts = {};
        CANDIDATE_FIELDS.forEach(f => { fieldCounts[f] = {}; });
        results.forEach(r => {
            const meta = r.metadata || {};
            CANDIDATE_FIELDS.forEach(f => {
                let v = meta[f];
                if (v === undefined || v === null || v === 'nan' || v === 'NaN') return;
                v = String(v).trim();
                if (!v) return;
                fieldCounts[f][v] = (fieldCounts[f][v] || 0) + 1;
            });
        });

        // 只保留有数据且唯一值 ≥ 1 的字段
        const fields = CANDIDATE_FIELDS.filter(f =>
            Object.keys(fieldCounts[f]).length > 0
        );
        if (!fields.length) return;

        const total = results.length;

        // 全局唯一值 → 颜色（色调均匀分布）
        const allVals = [...new Set(fields.flatMap(f => Object.keys(fieldCounts[f])))];
        const palette = {};
        allVals.forEach((v, i) => {
            palette[v] = `hsla(${Math.round(i * 360 / allVals.length)}, 62%, 52%, 0.82)`;
        });

        // 每个唯一值一个 dataset
        const datasets = allVals.map(v => ({
            label: v,
            data: fields.map(f => {
                const c = fieldCounts[f][v] || 0;
                return c > 0 ? +(100 * c / total).toFixed(1) : 0;
            }),
            backgroundColor: palette[v],
            borderRadius: 3,
        }));

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: { labels: fields, datasets },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 10,
                            font: { size: 10 },
                            // 图例太多时截断
                            filter: (item) => item.text.length <= 20,
                        },
                    },
                    title: {
                        display: true,
                        text: 'Top-K 结果 · 元数据属性分布（各字段百分比）',
                        font: { size: 13, weight: '700' },
                    },
                    tooltip: {
                        callbacks: {
                            label(ctx) {
                                return ctx.raw > 0
                                    ? `${ctx.dataset.label}: ${ctx.raw}%`
                                    : null;
                            },
                        },
                        filter(item) { return item.raw > 0; },
                    },
                },
                scales: {
                    x: {
                        stacked: true,
                        max: 100,
                        title: { display: true, text: '占比 (%)' },
                        ticks: { callback: v => v + '%' },
                        grid: { color: 'rgba(0,0,0,0.04)' },
                    },
                    y: { stacked: true },
                },
            },
        });
    }

    /** 保留旧接口兼容（内部调用新实现） */
    function renderCellTypeBars(canvasId, results) {
        renderMetaProfile(canvasId, results);
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
        renderMetaProfile,
        renderDistanceBars,
        fetchEmbedData,
    };
})();
