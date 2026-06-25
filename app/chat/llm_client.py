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


def parse_intent(message: str, obs_columns: list, history: list = None) -> dict:
    """把自然语言解析为结构化查询条件，返回 dict。"""
    system = f"""你是单细胞数据库查询助手。把用户描述解析为结构化查询条件。

数据集可用的元数据字段：{', '.join(obs_columns) if obs_columns else '未知'}

返回合法 JSON，字段：
- search_mode: 字符串，检索模式：
    * "metadata" — 仅按元数据字段过滤（如"找肝脏巨噬细胞"）
    * "similarity" — 按向量相似性检索（如"找与 Cell_X 相似的细胞"）
    * "hybrid" — 先限定细胞类型/组织，再找相似细胞（如"找与 CD4 T 细胞相似的细胞"）
- filters: 对象，键为元数据字段名，值为字符串列表
- cell_id: 字符串或 null，若用户指定具体 Cell ID 则填入
- top_k: 整数，返回数量，未指定时默认 10，最大 200
- search_description: 字符串，一句话描述本次查询目的

只输出 JSON，不要其他文字。示例：
{{"search_mode":"hybrid","filters":{{"cell_type":["T cell"]}},"cell_id":null,"top_k":20,"search_description":"查找与T细胞相似的细胞"}}"""

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
        return data
    except json.JSONDecodeError:
        return {
            'search_mode': 'metadata',
            'filters': {},
            'cell_id': None,
            'top_k': 10,
            'search_description': message,
        }


def analyze_results(message: str, intent: dict, results: list,
                    cell_stats: dict, knowledge: str, history: list = None,
                    *, search_mode: str = None, index_type: str = None,
                    query_time_ms: float = 0, seed_cell_id: str = None) -> str:
    """生成检索结果的生物学分析文本。"""
    stats_lines = []
    for k, v in cell_stats.items():
        top5 = sorted(v.items(), key=lambda x: -x[1])[:5]
        stats_lines.append(f'  {k}: ' + ', '.join(f'{n}({c})' for n, c in top5))

    mode_desc = {
        'vector_ann': f'ANN 向量检索（{index_type or "hnsw"}）',
        'vector_ann_seeded': f'以种子细胞 {seed_cell_id} 做 ANN 相似性检索（{index_type or "hnsw"}）',
        'metadata': '元数据过滤',
    }.get(search_mode, '检索')

    top_cells = []
    for r in results[:5]:
        sim = r.get('similarity')
        dist = r.get('distance')
        line = f"  - {r.get('cell_id')} ({r.get('cell_type', '未知')})"
        if sim is not None and dist is not None:
            line += f", 相似度={sim:.3f}, 距离={dist:.3f}"
        top_cells.append(line)

    user_content = f"""用户问题：{message}
检索方式：{mode_desc}
解析条件：{json.dumps(intent.get('filters', {}), ensure_ascii=False)}
检索到 {len(results)} 个细胞，属性分布：
{chr(10).join(stats_lines) if stats_lines else '  （无统计数据）'}

Top 相似细胞：
{chr(10).join(top_cells) if top_cells else '  （无结果）'}

相关背景知识：
{knowledge if knowledge else '  （无匹配知识）'}

请用中文简洁分析这批细胞的生物学意义（100-200字），重点说明细胞类型特征和分布规律；若使用了向量检索，请提及相似性排序的含义。"""

    if query_time_ms:
        user_content += f"\n\n检索耗时：{query_time_ms:.1f} ms"

    messages = [
        {'role': 'system', 'content': '你是单细胞生物学专家，用中文给出专业但简洁的分析，不要重复问题本身。'},
    ]
    if history:
        messages.extend(history[-4:])
    messages.append({'role': 'user', 'content': user_content})

    resp = _client().chat.completions.create(
        model='deepseek-chat',
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )
    return resp.choices[0].message.content
