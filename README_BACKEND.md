# Scann 检索系统后端模块说明文档 (v2)

本模块是由后端开发的单细胞检索系统核心逻辑，主要涵盖了：**多角色访问控制 (RBAC)**、**带元数据过滤的 ANN 检索**、以及**自动化性能评测加速比计算**。

## 1. 角色与权限体系 (Role System)
系统预设了 5 种角色，权限按等级递增：
1. **Visitor (访客)**: 仅能查看标记为 `demo` 的数据集（`upload_by` 为空的公共数据）。
2. **Normal (普通用户)**: 能查看公共数据及自己上传的数据，可进行基础检索。
3. **Expert (资深专家)**: 拥有普通用户权限，并可触发**性能评测 (Evaluation)** 任务。
4. **LabAdmin (实验室管理员)**: 可管理本实验室（当前为全局）所有数据、用户，具有评测权限。
5. **SysAdmin (系统管理员)**: 拥有所有后台管理权限。

## 2. 核心 API 接口规范

### A. 数据管理 (`/api/datasets/`)
*   **GET `/api/datasets/`**: 获取当前用户可见的数据集列表。
    *   *注：后端已根据 Role 自动过滤。*

### B. 增强搜索 (`/api/search/`)
*   **POST `/api/search/by_cell_id`**: 按细胞 ID 检索相似细胞。
*   **POST `/api/search/by_vector`**: 按向量检索相似细胞。

**输入格式 (JSON):**
```json
{
  "dataset_id": 1,
  "index_id": 2,
  "cell_id": "Cell_001",
  "top_k": 10,
  "filters": {
    "cell_type": ["T cell", "B cell"],
    "disease": ["Normal"]
  }
}
```
**后端逻辑增强:**
- **相似度计算**: 返回 `similarity` 字段，公式为 $1 / (1 + distance)$。
- **排除自身**: 结果中绝不会包含查询细胞本身。
- **过滤机制**: 支持 `filters` 条件过滤，后端会自动扩大搜索范围以保证返回足额的 `top_k` 个结果。

### C. 性能评测 (`/api/evaluate/`)
*   **POST `/api/evaluate/<dataset_id>`**: 触发性能评测。
    *   **权限**: 仅限 `expert`, `labadmin`, `sysadmin`。
    *   **参数**: `{"index_id": 2, "k": 10, "n_queries": 50}`。
    *   **返回**: 包含 `speedup` (加速比 = $T_{exact} / T_{ann}$)。

## 3. 部署与初始化

### 环境安装
```bash
pip install -r requirements.txt
```

### 初始化测试环境
为了方便前端调试，请先运行以下命令初始化数据库和 5 个全角色账号：
```bash
python Scann/init_db.py
```
**默认账号 (密码均为 `pass123`):**
- `visitor`, `normal`, `expert`, `labadmin`, `sysadmin`

### 运行
```bash
python Scann/run.py
```

## 4. 给前端开发的特别建议
1. **认证**: 系统使用 Session 认证，前端请求需携带 `withCredentials: true`。
2. **错误处理**: 如果权限不足，API 会返回 `403 Forbidden` 及 JSON 格式的报错信息。
3. **元数据**: 检索结果中 `metadata` 字段包含该细胞的所有 obs 标签，建议前端动态渲染表格。
4. **加速比展示**: 在可视化报告页面，建议重点展示 `speedup` 和 `recall_at_k` 两个指标。
