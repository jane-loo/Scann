/**
 * Scann API ??????
 */
const API_BASE = '';

const api = axios.create({
    baseURL: API_BASE,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json'
    }
});

// ??????????????? 401 ?????
api.interceptors.response.use(
    response => response.data,
    error => {
        if (error.response && error.response.status === 401) {
            // ??????????????????????????
            console.warn('?????????????');
        }
        return Promise.reject(error.response ? error.response.data : error);
    }
);

const ScannAPI = {
    // Auth
    login: (username, password) => api.post('/auth/login', { username, password }),
    register: (data) => api.post('/auth/register', data),
    logout: () => api.post('/auth/logout'),
    getCurrentUser: () => api.get('/auth/me'),

    // Datasets
    getDatasets: () => api.get('/api/datasets/'),
    uploadDataset: (formData) => api.post('/api/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    }),
    getDatasetDetail: (id) => api.get(`/api/datasets/${id}`),
    deleteDataset: (id) => api.delete(`/api/datasets/${id}`),
    getDatasetCells: (id) => api.get(`/api/datasets/${id}/cells`),
    getUmapData: (id) => api.get(`/api/datasets/${id}/umap_data`),
    getPcaData: (id) => api.get(`/api/datasets/${id}/pca_data`),
    compareCells: (datasetId, data) => api.post(`/api/datasets/${datasetId}/cells/compare`, data),

    // Indexes
    buildIndex: (data) => api.post('/api/indexes/build', data),
    getIndexes: () => api.get('/api/indexes/'),
    deleteIndex: (id) => api.delete(`/api/indexes/${id}`),

    // Search
    searchByCell: (data) => api.post('/api/search/by_cell_id', data),
    searchByVector: (data) => api.post('/api/search/by_vector', data),
    searchRandom: (data) => api.post('/api/search/random', data),
    getHistory: (limit = 20) => api.get(`/api/search/history?limit=${limit}`),

    // Evaluation
    runEvaluation: (datasetId, data) => api.post(`/api/evaluate/${datasetId}`, data),
    runBatchEvaluation: (datasetId, data) => api.post(`/api/evaluate/${datasetId}/batch`, data),
    getEvaluationReport: (datasetId) => api.get(`/api/evaluate/${datasetId}/report`),

    // Admin
    getUsers: () => api.get('/admin/users'),
    updateUser: (id, data) => api.put(`/admin/users/${id}`, data),
    deleteUser: (id) => api.delete(`/admin/users/${id}`)
};

window.ScannAPI = ScannAPI;
