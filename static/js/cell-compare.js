/**
 * 单细胞对比详情：PCA 维度柱状图 + 2D 位置关系图
 */
(function () {
    const charts = {};

    function destroy(id) {
        if (charts[id]) {
            charts[id].destroy();
            delete charts[id];
        }
    }

    function destroyAll() {
        Object.keys(charts).forEach(destroy);
    }

    /** PCA 前 N 维：查询 vs 命中 分组柱状图 */
    function renderVectorBars(canvasId, vectorDims, hasQuery) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !vectorDims?.length) return;

        const labels = vectorDims.map((d) => `PC${d.dim}`);
        const datasets = [{
            label: '命中细胞',
            data: vectorDims.map((d) => d.target),
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderRadius: 4,
        }];
        if (hasQuery) {
            datasets.unshift({
                label: '查询细胞',
                data: vectorDims.map((d) => d.query),
                backgroundColor: 'rgba(239, 68, 68, 0.75)',
                borderRadius: 4,
            });
        }

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11, weight: '600' } } },
                    title: {
                        display: true,
                        text: hasQuery ? 'PCA 维度值对比（前 15 维）' : '命中细胞 PCA 维度分布',
                        font: { size: 12, weight: '700' },
                    },
                },
                scales: {
                    y: { title: { display: true, text: 'Value' } },
                },
            },
        });
    }

    /** 2D 嵌入上两细胞相对位置 */
    function renderPairMap(canvasId, embed2d, hasQuery, sourceLabel) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !embed2d?.target) return;

        const useUmap = embed2d.target.umap_x != null;
        const tx = useUmap ? embed2d.target.umap_x : embed2d.target.pca_x;
        const ty = useUmap ? embed2d.target.umap_y : embed2d.target.pca_y;

        const datasets = [{
            label: '命中细胞',
            data: [{ x: tx, y: ty, label: embed2d.target_label || 'target' }],
            pointRadius: 12,
            pointBackgroundColor: 'rgba(59, 130, 246, 0.85)',
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
        }];

        if (hasQuery && embed2d.query) {
            const qx = useUmap ? embed2d.query.umap_x : embed2d.query.pca_x;
            const qy = useUmap ? embed2d.query.umap_y : embed2d.query.pca_y;
            datasets.unshift({
                label: '查询细胞',
                data: [{ x: qx, y: qy, label: embed2d.query_label || 'query' }],
                pointRadius: 11,
                pointBackgroundColor: 'rgba(239, 68, 68, 0.45)',
                pointBorderColor: '#ef4444',
                pointBorderWidth: 2.5,
            });
        }

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' },
                    title: {
                        display: true,
                        text: `${sourceLabel || '2D'} · 两细胞空间位置`,
                        font: { size: 12, weight: '700' },
                    },
                    tooltip: {
                        callbacks: {
                            label(ctx) {
                                const p = ctx.raw;
                                return p.label || `(${p.x.toFixed(3)}, ${p.y.toFixed(3)})`;
                            },
                        },
                    },
                },
                scales: {
                    x: { title: { display: true, text: 'Dim 1' } },
                    y: { title: { display: true, text: 'Dim 2' } },
                },
            },
        });
    }

    /** 各维度绝对差值（仅在有查询细胞时） */
    function renderDeltaBars(canvasId, vectorDims) {
        destroy(canvasId);
        const el = document.getElementById(canvasId);
        if (!el || !vectorDims?.length || vectorDims[0].delta == null) return;

        charts[canvasId] = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: {
                labels: vectorDims.map((d) => `PC${d.dim}`),
                datasets: [{
                    label: '|Δ| 绝对差',
                    data: vectorDims.map((d) => d.delta),
                    backgroundColor: 'rgba(168, 85, 247, 0.7)',
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: '各 PCA 维度差异（越小越相似）',
                        font: { size: 12, weight: '700' },
                    },
                },
                scales: { y: { beginAtZero: true } },
            },
        });
    }

    window.CellCompare = {
        destroyAll,
        renderVectorBars,
        renderPairMap,
        renderDeltaBars,
    };
})();
