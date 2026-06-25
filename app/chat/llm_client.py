"""DeepSeek LLM 客户端（OpenAI 兼容接口）。"""
import os
import json
from flask import current_app


def _client():
    from openai import OpenAI
    api_key = os.environ.get('DEEPSEEK_API_KEY') or current_app.config.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        raise RuntimeError('未配置 DEEPSEEK_API_KEY，请在环境变量或 .env 中设置')
    return OpenAI(api_key=api_key, base_url='https://api.deepseek.com')


def parse_intent(message: str, obs_columns: list, history: list = None,
                 *, use_joint: bool = False) -> dict:
    joint_hint = ''
    if use_joint:
        joint_hint = '\n- use_joint: true（用户希望在多个数据集联合索引中跨库检索）'

    system = f"""你是单细胞数据库查询助手。把用户描述解析为结构化查询条件。

数据集可用的元数据字段：{', '.join(obs_columns) if obs_columns else '未知'}

返回合法 JSON，字段：
- search_mode: "metadata" | "similarity" | "hybrid"
- use_joint: 布尔，若用户提到跨数据集/联合检索/多个库则为 true
- filters: 对象，键为元数据字段名，值为字符串列表
- cell_id: 字符串或 null
- top_k: 整数，默认 10，最大 200
- search_description: 一句话描述查询目的
{joint_hint}

只输出 JSON。"""

    messages = [{'role': 'system', 'content': system}]
    if history:
        messages.extend(history[-6:])
    messages.append({'role': 'user', 'content': message})

    try:
        resp = _client().chat.completions.create(
            model='deepseek-chat',
            messages=messages,
            response_format={'type': 'json_object'},
            temperature=0.1,
            max_tokens=512,
        )
        data = json.loads(resp.choices[0].message.content)
        if 'search_mode' not in data:
            data['search_mode'] = 'similarity' if data.get('cell_id') else 'metadata'
        if use_joint:
            data['use_joint'] = True
        return data
    except json.JSONDecodeError:
        return {
            'search_mode': 'metadata',
            'use_joint': use_joint,
            'filters': {},
            'cell_id': None,
            'top_k': 10,
            'search_description': message,
        }


def analyze_results(message: str, intent: dict, results: list,
                    cell_stats: dict, history: list = None,
                    *, search_mode: str = None, index_type: str = None,
                    query_time_ms: float = 0, seed_cell_id: str = None,
                    is_joint: bool = False, dataset_ids: list = None) -> str:
    """生成检索结果的生物学分析（纯 LLM，不依赖静态知识库）。"""
    stats_lines = []
    for k, v in cell_stats.items():
        top5 = sorted(v.items(), key=lambda x: -x[1])[:5]
        stats_lines.append(f'  {k}: ' + ', '.join(f'{n}({c})' for n, c in top5))

    mode_desc = {
        'vector_ann': f'单库 ANN 向量检索（{index_type or "hnsw"}）',
        'vector_ann_seeded': f'单库 ANN 相似性检索，种子 {seed_cell_id}',
        'joint_vector_ann': f'联合索引跨库 ANN 检索（{index_type or "hnsw"}）',
        'joint_vector_ann_seeded': f'联合索引跨库检索，种子 {seed_cell_id}',
        'metadata': '元数据过滤',
    }.get(search_mode, '检索')

    top_cells = []
    for r in results[:8]:
        sim = r.get('similarity')
        dist = r.get('distance')
        ds_tag = f' [ds{r["dataset_id"]}]' if r.get('dataset_id') is not None else ''
        line = f"  - {r.get('cell_id')}{ds_tag} ({r.get('cell_type', '未知')})"
        if sim is not None and dist is not None:
            line += f", 相似度={sim:.3f}"
        meta_snip = {k: v for k, v in (r.get('metadata') or {}).items() if k != 'cell_type'}
        if meta_snip:
            line += f", {json.dumps(meta_snip, ensure_ascii=False)}"
        top_cells.append(line)

    joint_note = ''
    if is_joint and dataset_ids:
        joint_note = f'\n联合检索覆盖数据集 ID：{dataset_ids}'

    user_content = f"""用户问题：{message}
检索方式：{mode_desc}{joint_note}
解析条件：{json.dumps(intent.get('filters', {}), ensure_ascii=False)}
检索到 {len(results)} 个细胞，属性分布：
{chr(10).join(stats_lines) if stats_lines else '  （无）'}

Top 细胞：
{chr(10).join(top_cells) if top_cells else '  （无）'}

请基于以上检索结果，用中文给出 120-200 字生物学解读：细胞类型特征、跨库分布（如有）、相似性排序含义。不要编造未出现在结果中的具体基因名。"""

    if query_time_ms:
        user_content += f"\n检索耗时：{query_time_ms:.1f} ms"

    messages = [
        {'role': 'system', 'content': '你是单细胞生物学专家，仅根据给定检索结果分析，不引用外部知识库。'},
    ]
    if history:
        messages.extend(history[-4:])
    messages.append({'role': 'user', 'content': user_content})

    resp = _client().chat.completions.create(
        model='deepseek-chat',
        messages=messages,
        temperature=0.7,
        max_tokens=450,
    )
    return resp.choices[0].message.content


def explain_search_results(query_input: str, query_type: str, results: list,
                           *, is_joint: bool = False, filters: dict = None) -> str:
    """检索页可选：对当前 Top-K 结果做 LLM 生物学解释。"""
    if not results:
        return '无检索结果，无法生成解释。'

    lines = []
    for r in results[:10]:
        ds = f' ds={r["dataset_id"]}' if r.get('dataset_id') is not None else ''
        meta = r.get('metadata') or {}
        meta_brief = ', '.join(f'{k}={v}' for k, v in list(meta.items())[:4])
        lines.append(
            f"#{r.get('rank')} {r.get('cell_id')}{ds} "
            f"sim={r.get('similarity', 0):.3f} {meta_brief}"
        )

    prompt = f"""查询：{query_input}（类型：{query_type}）
{'联合检索' if is_joint else '单库检索'}
过滤：{json.dumps(filters or {}, ensure_ascii=False)}

Top 结果：
{chr(10).join(lines)}

请用中文 100-150 字解读：这些相似细胞可能共同的生物学特征、metadata 分布规律。仅依据上述结果，勿编造基因或文献。"""

    resp = _client().chat.completions.create(
        model='deepseek-chat',
        messages=[
            {'role': 'system', 'content': '你是单细胞分析助手，简洁专业。'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.6,
        max_tokens=350,
    )
    return resp.choices[0].message.content
