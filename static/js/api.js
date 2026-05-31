/**
 * Scann API 封装模块
 */
const API_BASE = '';

const api = axios.create({
    baseURL: API_BASE,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json'
    }
});

// 响应拦截器：处理 401 和错误
api.interceptors.response.use(
    response => response.data,
    error => {
        if (error.response && error.response.status === 401) {
            // 可以在这里触发跳转到登录页的操作
            console.warn('未授权，请先登录');
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
    getEvaluationReport: (datasetId) => api.get(`/api/evaluate/${datasetId}/report`),

    // Admin
    getUsers: () => api.get('/admin/users'),
    updateUser: (id, data) => api.put(`/admin/users/${id}`, data),
    deleteUser: (id) => api.delete(`/admin/users/${id}`)
};

window.ScannAPI = ScannAPI;
