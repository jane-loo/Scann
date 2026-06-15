"""内嵌细胞类型知识库（常见免疫/组织细胞的 marker 与功能）。"""

_KB = {
    'T cell': {
        'markers': ['CD3D', 'CD3E', 'CD3G', 'TRAC', 'TRBC1'],
        'subtypes': 'CD4⁺辅助T（CD4, IL7R）、CD8⁺细胞毒T（CD8A, GZMB）、调节性T（FOXP3, IL2RA）',
        'function': '适应性免疫核心，介导细胞免疫应答',
    },
    'plasma cell': {
        'markers': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'IGHM', 'SDC1', 'PRDM1', 'MZB1'],
        'subtypes': '初始B（IGHD）、记忆B（CD27）、浆细胞（IGHG1, MZB1）',
        'function': '产生抗体，介导体液免疫',
    },
    'B cell': {
        'markers': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'IGHM'],
        'subtypes': '初始B（IGHD）、记忆B（CD27）、浆细胞（IGHG1, MZB1）',
        'function': '产生抗体，介导体液免疫',
    },
    'NK cell': {
        'markers': ['NCAM1', 'GNLY', 'NKG7', 'KLRD1', 'GZMB'],
        'subtypes': 'CD56bright（调节型）、CD56dim（细胞毒型）',
        'function': '天然杀伤肿瘤和病毒感染细胞，无需MHC呈递',
    },
    'Monocyte': {
        'markers': ['CD14', 'LYZ', 'CST3', 'FCN1', 'S100A8'],
        'subtypes': '经典单核（CD14⁺CD16⁻）、非经典单核（CD14⁺CD16⁺）',
        'function': '先天免疫、炎症调节、树突细胞前体',
    },
    'Macrophage': {
        'markers': ['CD68', 'MRC1', 'MARCO', 'CSF1R', 'APOE'],
        'subtypes': 'M1促炎型（TNF, IL6↑）、M2抗炎型（MRC1, CD163↑）',
        'function': '吞噬病原体和凋亡细胞，分泌细胞因子',
    },
    'Dendritic cell': {
        'markers': ['FCER1A', 'CLEC10A', 'CD1C', 'ITGAX', 'HLA-DRA'],
        'subtypes': 'cDC1（XCR1⁺）、cDC2（CD1c⁺）、pDC（LILRA4, IRF7）',
        'function': '专职抗原呈递，激活初始T细胞',
    },
    'Neutrophil': {
        'markers': ['FCGR3B', 'CSF3R', 'CXCR2', 'S100A9', 'MPO'],
        'subtypes': '成熟粒细胞、低密度粒细胞',
        'function': '急性炎症第一响应，吞噬和杀灭病原体',
    },
    'Hepatocyte': {
        'markers': ['ALB', 'APOB', 'CYP3A4', 'TTR', 'APOA1'],
        'subtypes': '中心静脉周围（Glul⁺）、门静脉周围（CPS1⁺）',
        'function': '肝脏实质细胞，负责代谢、解毒、合成血浆蛋白',
    },
    'Endothelial cell': {
        'markers': ['PECAM1', 'VWF', 'CDH5', 'ENG', 'KDR'],
        'subtypes': '动脉内皮（GJA4）、静脉内皮（NR2F2）、毛细血管（CA4）',
        'function': '构成血管壁，调节物质交换',
    },
    'Fibroblast': {
        'markers': ['COL1A1', 'COL3A1', 'DCN', 'LUM', 'PDGFRA'],
        'subtypes': '活化成纤维细胞（α-SMA⁺）、静息成纤维细胞',
        'function': '合成细胞外基质，参与组织修复和纤维化',
    },
    'Stem cell': {
        'markers': ['CD34', 'CD38', 'PROM1', 'THY1', 'KIT'],
        'subtypes': '造血干细胞（CD34⁺CD38⁻）、祖细胞（CD34⁺CD38⁺）',
        'function': '自我更新和多向分化，维持组织稳态',
    },
    'Epithelial cell': {
        'markers': ['EPCAM', 'KRT8', 'KRT18', 'CDH1', 'CLDN4'],
        'subtypes': '肠上皮、肺上皮、乳腺上皮等（组织特异性 KRT 亚型）',
        'function': '屏障功能，分泌功能，组织特异性吸收/分泌',
    },
    'Erythrocyte': {
        'markers': ['HBA1', 'HBA2', 'HBB', 'GYPA', 'ALAS2'],
        'subtypes': '成熟红细胞（无核）、网织红细胞',
        'function': '携带氧气，维持血液渗透压',
    },
    'Platelet': {
        'markers': ['ITGA2B', 'ITGB3', 'GP1BA', 'PPBP', 'PF4'],
        'subtypes': '巨噬细胞衍生物',
        'function': '止血与凝血功能，参与初步炎症反应',
    },
    'Smooth Muscle cell': {
        'markers': ['ACTA2', 'TAGLN', 'MYH11', 'CNN1', 'MYL9'],
        'subtypes': '血管平滑肌细胞（vSMC）、各种脏器平滑肌',
        'function': '收缩功能，调节血管阻力和器官蠕动',
    },
    'Stellate cell': {
        'markers': ['RBP4', 'GFAP', 'ACTA2', 'LRAT', 'PDGFRB'],
        'subtypes': '肝星状细胞、胰腺星状细胞',
        'function': '存储维生素A；活化后参与组织纤维化',
    },
    'Cardiomyocyte': {
        'markers': ['TNNT2', 'TNNI3', 'MYH6', 'MYH7', 'TTN'],
        'subtypes': '心房肌、心室肌、传导束系统',
        'function': '心肌收缩，泵血动力来源',
    },
    'Neuron': {
        'markers': ['RBFOX3', 'MAP2', 'SYP', 'DLG4', 'TUBB3'],
        'subtypes': '兴奋性神经元（SLC17A7）、抑制性神经元（GAD1/2）',
        'function': '电信号传导，处理与整合信息',
    },
    'Microglia': {
        'markers': ['AIF1', 'TMEM119', 'P2RY12', 'CX3CR1', 'C1QA'],
        'subtypes': '稳态小胶质细胞、DAM（疾病相关）',
        'function': '中枢神经系统中的专职巨噬细胞',
    },
    'Adipocyte': {
        'markers': ['ADIPOQ', 'LEP', 'PPARG', 'FABP4', 'PLIN1'],
        'subtypes': '白色脂肪、棕色脂肪、米色脂肪',
        'function': '能量储存、分泌瘦素及脂肪因子',
    },
}


def lookup(query: str) -> str:
    """模糊匹配细胞类型，返回知识文本。"""
    if not query:
        return ''
    q = query.strip().lower()
    matched = []
    for name, info in _KB.items():
        if q in name.lower() or name.lower() in q:
            text = (
                f'【{name}】\n'
                f'  Marker基因：{", ".join(info["markers"])}\n'
                f'  亚型：{info["subtypes"]}\n'
                f'  功能：{info["function"]}'
            )
            matched.append(text)
    return '\n\n'.join(matched)


def collect_knowledge(filters: dict) -> str:
    """从 filters 中提取 cell_type 并查询知识库。"""
    cell_types = filters.get('cell_type', [])
    if isinstance(cell_types, str):
        cell_types = [cell_types]
    texts = [t for ct in cell_types if (t := lookup(ct))]
    return '\n\n'.join(texts)
