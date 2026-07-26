"""
关键词评分器（Keyword Scorer）
============================

为 Agentic-RAG 决策提供第二信号，避免单一 reranker 分数左右一切。

核心思想（B-Tree 式分层）：
  - Layer 1: 锚点集粗筛 → 锁定候选文档子集
  - Layer 2: 关键词细查 → 计算 keyword_score
  - Layer 3: 与 reranker 分数融合 → 最终决策

权重体系（按"对答案的决定性"4 层排序，越能决定答案的词权重越高）：
  - answer_core (强名词/答案核心词)：权重 1.0
    直接决定答案的实体词，如"校门"、"食堂"、"图书馆"
    例: "学校的校门在哪里" → "校门"是答案核心词，权重最高
  - suffix_noun (名词后缀词)：权重 0.8
    以"处/室/馆/楼"等结尾的实体词
  - interrogative (疑问代词)：权重 0.4
    如"哪里"、"什么"、"多少"，指明答案类型
    例: "学校的校门在哪里" → "哪里"是疑问代词，权重中等
    不参与锚点命中（出现在太多问题中，无区分度）
  - generic_noun (通用名词)：权重 0.2
    如"学校"、"学生"、"老师"，在所有问题里都出现，无区分度
    例: "学校的校门在哪里" → "学校"是通用名词，权重最低
    不参与锚点命中
  - weak (虚词/通用动词)：权重 0.05
"""

import re
from typing import Tuple, List, Dict, Set

# ────────────────────────────────────────────────────────────
# 词性识别字典
# ────────────────────────────────────────────────────────────

# 答案核心名词（强名词）：直接决定答案的实体词，权重 1.0
# 命中即视为路由关键信号，且不依赖锚点集（锚点集可能因子串去重丢掉短词）
STRONG_NOUNS: Set[str] = {
    # 餐饮生活
    "食堂", "宿舍", "图书馆", "体育馆", "运动场", "运动场所",
    "医务室", "校医院", "警务处", "保卫处", "教务处", "财务处",
    # 学习
    "奖学金", "助学金", "学费", "学籍", "学分", "学时",
    "考勤", "考试", "成绩", "绩点", "选课", "重修", "补考",
    # 入学毕业
    "新生", "入学", "报到", "毕业", "就业", "校招",
    "转专业", "休学", "复学", "保研", "专升本",
    # 校园设施
    "校门", "东门", "南门", "西门", "北门",
    "教学楼", "自习室", "游泳馆", "健身房",
    # 制度
    "处分", "警告", "学工处", "查寝", "旷课",
    # 人物
    "辅导员", "导员", "班长", "委员", "社团",
    # 事务
    "一卡通", "校园卡", "虚拟卡", "电费", "医保",
    "警务", "医务", "图书馆", "全称",
    # 校园场所（补充）
    "警务处", "保卫处", "教务处", "财务处",
    "医务室", "校医院", "图书馆", "体育馆",
    "运动场", "运动场所", "食堂", "宿舍",
    # 入学相关
    "新生", "入学", "报到", "转专业",
}

# 名词后缀：以这些字结尾的 3 字以上词视为名词（权重 0.8）
NOUN_SUFFIXES: Tuple[str, ...] = (
    "处", "室", "馆", "楼", "门", "院", "部", "局", "科",  # 部门/建筑
    "堂", "厅", "场", "站", "园", "吧",                    # 场所
    "员", "生", "师", "长", "主",                          # 人物
    "费", "金", "款", "证", "书", "卡",                    # 事务
    "制", "度", "法", "规", "程", "则",                    # 制度
)

# 疑问代词：指明答案类型（地点/数量/方式），权重 0.4
# 不参与锚点命中（出现在大量问题中，无区分度），但参与权重计算
INTERROGATIVES: Set[str] = {
    "什么", "怎么", "哪", "哪个", "哪些", "哪几个", "几个",
    "多少", "如何", "为什么", "是不是", "有没有",
    "是什么", "在哪", "哪里", "怎么样", "何",
    "何时", "何地", "何人", "何种", "何事",
}

# 通用名词：在所有问题里都出现，对答案无区分度，权重 0.2
# 不参与锚点命中，但参与权重分母（拉低总体命中比例）
GENERIC_NOUNS: Set[str] = {
    "学校", "校园", "学生", "老师", "同学", "大学",
    "学院", "信息", "技术", "职业", "专业",
    "今天", "明天", "昨天", "现在", "目前",
    "本校", "我校", "大家", "我们",
}

# 弱词/虚词/通用动词：权重 0.05（最低，几乎不影响得分）
WEAK_WORDS: Set[str] = {
    # 通用动词
    "是", "有", "在", "能", "可以", "会", "做", "干",
    "请问", "帮我", "我想", "我要", "了解", "知道",
    # 通用虚词
    "的", "了", "吗", "呢", "啊", "吧", "与", "和",
    "这个", "那个", "一个", "一些", "之", "于", "而",
}

# 动词前缀黑名单：用于过滤 ngram 拦腰斩断产生的伪词
BAD_PREFIXES: Set[str] = {
    "出", "进", "到", "近", "扫", "过", "送", "配", "按", "从",
    "去", "管", "馆", "楼", "院", "处", "室", "是", "或",
    "员", "士", "知", "再", "段", "育", "果", "分", "保", "以",
    "他", "学", "委", "园", "细", "副", "定", "圳", "校",
    "班", "体", "如", "的",
}


# ────────────────────────────────────────────────────────────
# ngram 切分（与 anchor_manager 保持一致）
# ────────────────────────────────────────────────────────────

_RE_CJK = re.compile(r"[\u4e00-\u9fff]+")
_RE_ALNUM = re.compile(r"[a-zA-Z0-9]+")


def _extract_ngrams(text: str, n_range: Tuple[int, int] = (2, 4)) -> List[str]:
    """提取 2~4 字中文 ngram + 2字以上英文/数字 token"""
    tokens = []
    for m in _RE_CJK.finditer(text):
        seq = m.group()
        for n in range(n_range[0], n_range[1] + 1):
            if len(seq) >= n:
                tokens.extend(seq[i:i+n] for i in range(len(seq) - n + 1))
    for m in _RE_ALNUM.finditer(text):
        tok = m.group()
        if len(tok) >= 2:
            tokens.append(tok.lower())
    return tokens


# ────────────────────────────────────────────────────────────
# 词性识别 + 权重分配
# ────────────────────────────────────────────────────────────

def classify_token(token: str, anchor_set: Set[str] = None) -> Tuple[str, float]:
    """
    识别 token 的词性类别和权重。

    参数：
        token: 待分类的词
        anchor_set: 锚点集（可选），用于验证后缀词是否真的是知识库中的名词

    返回：(category, weight)
      category: "strong_noun" / "suffix_noun" / "interrogative" / "generic_noun" / "weak" / "other"

    权重层级（按对答案的决定性排序）：
      strong_noun (答案核心词)   → 1.0   例: 校门、食堂、图书馆
      suffix_noun (名词后缀词)   → 0.8   例: 心理咨询室、学生事务处
      interrogative (疑问代词)   → 0.4   例: 哪里、什么、多少
      generic_noun (通用名词)    → 0.2   例: 学校、学生、老师
      weak (虚词/通用动词)       → 0.05  例: 的、了、是、有
      other (未识别)             → 0.1~0.3
    """
    if not token or len(token) < 2:
        return ("other", 0.0)

    # 1. 答案核心名词（最高权重 1.0）
    if token in STRONG_NOUNS:
        return ("strong_noun", 1.0)

    # 2. 疑问代词（权重 0.4，指明答案类型）
    if token in INTERROGATIVES:
        return ("interrogative", 0.4)

    # 3. 通用名词（权重 0.2，在所有问题里都出现，无区分度）
    if token in GENERIC_NOUNS:
        return ("generic_noun", 0.2)

    # 4. 弱词/虚词（最低权重 0.05）
    if token in WEAK_WORDS:
        return ("weak", 0.05)

    # 5. 名词后缀判断（增加多重过滤防止误判）：
    #    条件：3字以上 + 以指定后缀结尾 + 首字符不在黑名单
    #         + 不包含疑问词/通用动词片段（防止"几个门""怎么办"误判）
    #         + 若提供了锚点集，必须在锚点集中才算（最严格的验证）
    if len(token) >= 3 and token.endswith(NOUN_SUFFIXES):
        # 首字符黑名单检查
        if token[0] in BAD_PREFIXES:
            if len(token) == 2:
                return ("other", 0.2)
            elif len(token) == 3:
                return ("other", 0.1)
            else:
                return ("other", 0.05)
        # 包含疑问词/通用动词片段 → 不是名词
        _has_interrogative_fragment = any(
            iw in token for iw in ("几个", "多少", "什么", "怎么", "哪", "为何",
                                   "有没有", "是不是", "能不能", "为什么")
        )
        if _has_interrogative_fragment:
            if len(token) == 2:
                return ("other", 0.2)
            elif len(token) == 3:
                return ("other", 0.1)
            else:
                return ("other", 0.05)
        # 包含通用动词 → 可能是动词短语而非名词
        _has_verb_fragment = any(
            vw in token for vw in ("办理", "使用", "申请", "填写", "提交",
                                   "进行", "需要", "可以", "应该", "能够")
        )
        # 如果提供了锚点集，只有在锚点集中存在才算后缀名词
        if anchor_set is not None:
            if token in anchor_set:
                return ("suffix_noun", 0.8)
            else:
                # 不在锚点集中的后缀词，降级为 other
                if len(token) == 2:
                    return ("other", 0.2)
                elif len(token) == 3:
                    return ("other", 0.1)
                else:
                    return ("other", 0.05)
        # 没有锚点集时，保守判断：不含动词片段才算
        if not _has_verb_fragment:
            return ("suffix_noun", 0.8)

    # 6. 其他（短词或未识别）
    # 权重根据长度调整：2字词(0.2) > 3字词(0.1) > 4字词(0.05)
    # 原因：越长的 ngram 越可能是跨词边界噪声（如"学校的校""的校门在"）
    if len(token) == 2:
        return ("other", 0.2)
    elif len(token) == 3:
        return ("other", 0.1)
    else:
        return ("other", 0.05)


# ────────────────────────────────────────────────────────────
# 关键词评分主函数
# ────────────────────────────────────────────────────────────

def compute_keyword_score(
    question: str,
    anchor_set: Set[str],
    n_range: Tuple[int, int] = (2, 4),
) -> Tuple[float, List[Dict]]:
    """
    计算关键词匹配分数（B-Tree Layer 2 细查）

    策略：
      1. 从问题中提取 ngram
      2. 每个token分类（强名词/弱词/名词后缀）
      3. 检查是否在锚点集中命中
      4. 计算加权命中比例

    参数：
        question: 用户问题
        anchor_set: 锚点集（来自 anchor_manager）
        n_range: ngram 长度范围

    返回：(keyword_score, matched_tokens)
      keyword_score: 0.0 ~ 1.0
      matched_tokens: [{"token": ..., "category": ..., "weight": ..., "in_anchor": ...}]
    """
    if not question or not anchor_set:
        return (0.0, [])

    # 1. 提取所有 ngram
    tokens = _extract_ngrams(question, n_range)
    if not tokens:
        return (0.0, [])

    # 1b. 省略式实体补全：处理"几个X""多少X""有几X"等省略问法
    # 例如"学校有几个门"→"门"是"校门/东门/南门/北门/入校"等的省略
    # 检测模式：量词/疑问词 + 单字核心词，在锚点集中查找以此单字结尾/开头的词
    _QUANTIFIERS = {"个", "扇", "道", "张", "间", "所", "位", "名", "条", "部", "台", "项", "种"}
    _QWORDS = {"几", "多少", "哪", "什么", "何"}
    _extra_hits = set()
    for i, ch in enumerate(question):
        # 检测"量词+核心字"模式（如"几个门"中的"个"后面是"门"）
        is_quant = ch in _QUANTIFIERS
        # 也检测"疑问词+核心字"模式（如"几门"虽然没有量词但"门"是核心词）
        is_qword = ch in _QWORDS
        if (is_quant or is_qword) and i + 1 < len(question):
            core_char = question[i + 1]
            # 跳过单字是虚词的情况
            if core_char in "的了是有在和与或但如以从到为对不也都":
                continue
            # 在锚点集中查找以此字开头或结尾的实体词（2-4字）
            for anchor in anchor_set:
                if len(anchor) < 2 or len(anchor) > 4:
                    continue
                # 只取末尾字匹配或首字匹配的（同一实体概念的简称）
                if anchor[-1] == core_char or anchor[0] == core_char:
                    # 过滤掉明显不是实体的短语（包含动词/虚词）
                    if any(v in anchor for v in ("的","了","有","在","是","和","与","从","到","为","对",
                                                  "进行","办理","使用","申请","需要","可以","有","靠近","外面")):
                        continue
                    _extra_hits.add(anchor)
    for hit in _extra_hits:
        tokens.append(hit)

    # 2. 去重
    unique_tokens = list(set(tokens))

    # 3. 分类并查锚点集
    # 命中规则：
    #   - strong_noun（答案核心词）：直接视为命中（不依赖锚点集）
    #   - suffix_noun（名词后缀词）：必须在锚点集中才算命中
    #   - interrogative / generic_noun / weak / other：不参与命中（不计入分子）
    #     但参与分母（拉低总体命中比例），权重越低的问题词对分数影响越小
    matched_tokens = []
    total_weight = 0.0       # 所有 token 的总权重（分母）
    hit_weight = 0.0        # 命中锚点的 token 权重之和（分子）
    strong_noun_hit = 0     # 命中的强名词数量
    strong_noun_total = 0    # 问题中强名词总数

    for token in unique_tokens:
        category, weight = classify_token(token, anchor_set)

        # 决定是否命中
        if category == "strong_noun":
            # 强名词：直接视为命中（不依赖锚点集，因为锚点集可能因子串去重丢掉短词）
            in_anchor = True
        elif category == "suffix_noun":
            # 名词后缀词：已在 classify_token 中验证过在锚点集中
            in_anchor = True
        else:
            # 疑问词/通用名词/弱词/其他：不参与锚点命中
            in_anchor = False

        # 跳过权重为0的token（极短的或边界 token）
        if weight == 0.0:
            continue

        total_weight += weight

        if category == "strong_noun":
            strong_noun_total += 1
            if in_anchor:
                strong_noun_hit += 1

        if in_anchor:
            hit_weight += weight

        matched_tokens.append({
            "token": token,
            "category": category,
            "weight": weight,
            "in_anchor": in_anchor,
        })

    if total_weight == 0:
        return (0.0, [])

    # 4. 计算 keyword_score（多维度融合）
    # 4a. 加权命中比例
    weighted_ratio = hit_weight / total_weight

    # 4b. 强名词命中加成（强名词是路由命中的关键信号）
    strong_noun_bonus = 0.0
    if strong_noun_total > 0:
        strong_noun_bonus = (strong_noun_hit / strong_noun_total) * 0.3

    # 4c. 最终分数（基础比例 + 强名词加成，封顶 1.0）
    keyword_score = min(1.0, weighted_ratio + strong_noun_bonus)

    return (keyword_score, matched_tokens)


# ────────────────────────────────────────────────────────────
# B-Tree Layer 1：粗筛锁定候选文档
# ────────────────────────────────────────────────────────────

def build_anchor_to_docs_index(
    documents: List[Dict],
    anchor_set: Set[str],
    n_range: Tuple[int, int] = (2, 4),
) -> Dict[str, List[str]]:
    """
    构建反向索引：锚点词 → 文档ID列表（B-Tree Layer 1）

    参数：
        documents: [{"id": "doc_id", "text": "..."}]
        anchor_set: 锚点集
        n_range: ngram 长度范围

    返回：{"食堂": ["03_生活.txt", "07_校园.txt"], ...}
    """
    inverted_index: Dict[str, List[str]] = {}

    for doc in documents:
        doc_id = doc.get("id") or doc.get("file_name") or ""
        text = doc.get("text", "")
        if not doc_id or not text:
            continue

        # 提取文档所有 ngram
        doc_ngrams = set(_extract_ngrams(text, n_range))

        # 与锚点集取交集，建立反向索引
        for token in doc_ngrams & anchor_set:
            if token not in inverted_index:
                inverted_index[token] = []
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)

    return inverted_index


def coarse_filter(
    question: str,
    anchor_set: Set[str],
    inverted_index: Dict[str, List[str]],
    n_range: Tuple[int, int] = (2, 4),
) -> Set[str]:
    """
    B-Tree Layer 1 粗筛：用问题中的强名词锚点锁定候选文档

    返回：候选文档 ID 集合
    """
    if not question or not inverted_index:
        return set()

    # 1. 提取问题 ngram
    tokens = _extract_ngrams(question, n_range)
    unique_tokens = set(tokens)

    # 2. 找出问题中的强名词 + 名词后缀词
    candidate_docs = set()
    for token in unique_tokens:
        if token not in anchor_set:
            continue
        category, _ = classify_token(token)
        # 只用强名词和名词后缀词做粗筛（弱词会引入噪声）
        if category in ("strong_noun", "suffix_noun"):
            docs = inverted_index.get(token, [])
            candidate_docs.update(docs)

    return candidate_docs


# ────────────────────────────────────────────────────────────
# 完善式追问（v0.3.0 新增）
# ────────────────────────────────────────────────────────────
# 旧版追问逻辑：「换个能回答的问题问」→ 等于把球踢回给用户
# 新版追问逻辑：分析用户问题中已有/缺失的信息维度，主动询问
#   补充细节，把模糊问题变成可回答的精确问题。
# ----------------------------------------------------------------

# 疑问意图 → 对应的答案维度
# 注意：长关键词放前面，避免"怎么样"被"怎么"截胡
_INTENT_DIMENSIONS = {
    "怎么样": "状态/评价",
    "怎么": "办理流程/方法",
    "如何": "办理流程/方法",
    "在哪": "位置",
    "哪里": "位置",
    "何地": "位置",
    "位置": "位置",
    "什么时候": "时间",
    "几点": "时间",
    "何时": "时间",
    "时间": "时间",
    "多少": "数量/金额",
    "几": "数量",
    "为什么": "原因/依据",
    "什么": "具体内容",
    "条件": "申请条件",
    "要求": "要求",
}

# 当问题已有实体但缺失疑问词时，列出该实体可问的维度
_ENTITY_DIMENSIONS = {
    "食堂": ["位置", "开放时间", "菜品/价格", "就餐规则"],
    "宿舍": ["位置", "入住条件", "熄灯时间", "水电规则"],
    "图书馆": ["位置", "开放时间", "借阅规则", "座位预约"],
    "校门": ["位置分布", "开放时间", "进出规则"],
    "教学楼": ["位置", "教室分布", "开放时间"],
    "校园卡": ["办理流程", "充值方式", "使用范围", "挂失补办"],
    "一卡通": ["办理流程", "充值方式", "使用范围", "挂失补办"],
    "奖学金": ["申请条件", "金额/等级", "评审流程", "发放时间"],
    "助学金": ["申请条件", "金额/等级", "申请流程"],
    "学费": ["金额", "缴费方式", "缓交政策"],
    "转专业": ["申请条件", "申请时间", "考核流程", "名额限制"],
    "选课": ["时间安排", "操作流程", "学分要求"],
    "考试": ["时间安排", "考场规则", "补考政策"],
    "成绩": ["查询方式", "绩点计算", "复核流程"],
    "电费": ["充值方式", "收费标准", "停电规则"],
    "医务室": ["位置", "开放时间", "就诊流程"],
    "校医院": ["位置", "开放时间", "就诊流程"],
    "警务处": ["位置", "联系方式", "办理业务"],
    "新生": ["报到时间", "报到流程", "所需材料"],
    "社团": ["种类", "加入方式", "活动规则"],
}

# 问题里出现这些词时，提示用户具体化对象
_OBJECT_HINTS = {
    "办理": ["校园卡/一卡通", "奖学金", "助学金", "转专业", "请假"],
    "申请": ["奖学金", "助学金", "转专业", "请假", "休学"],
    "时间": ["报到时间", "开放时间", "截止时间", "考试时间"],
    "位置": ["食堂", "宿舍", "图书馆", "校门", "教学楼", "医务室"],
    "电话": ["警务处", "校医院", "教务处", "辅导员"],
}


def generate_clarification(
    question: str,
    anchor_set: Set[str],
    hit_tokens: List[str] = None,
    top_doc_keywords: List[str] = None,
) -> str:
    """
    生成完善式追问文本。

    策略：
      1. 从问题中提取已有实体（强名词 / 命中锚点的词）
      2. 从问题中提取已有疑问意图（哪里/多少/怎么/什么时候）
      3. 根据"已有实体 vs 已有意图"的组合，识别缺失维度：
         - 有实体 + 无意图  → 问用户想了解该实体的哪个方面
         - 无实体 + 有意图  → 问用户具体是哪个对象
         - 有实体 + 有意图  → 检索词不足，让用户补充场景细节
         - 都没有           → 给出主题列表让用户选
      4. 基于锚点集生成候选选项（最多4个）

    参数：
      question: 用户原始问题
      anchor_set: 锚点集
      hit_tokens: route() 返回的命中锚点词列表（可选）
      top_doc_keywords: 检索 top-K 文档中提取的关键词（可选，用于生成更贴合的选项）

    返回：
      追问文本字符串
    """
    hit_tokens = hit_tokens or []
    top_doc_keywords = top_doc_keywords or []

    # ── 1. 提取已有实体 ──
    # 从 hit_tokens + 问题本身扫描强名词，避免依赖 route() 是否正确返回
    entities = []
    # 1a. 从 hit_tokens 提取
    for token in hit_tokens:
        if token in STRONG_NOUNS:
            entities.append(token)
        elif len(token) >= 2 and token in anchor_set:
            if token not in INTERROGATIVES and token not in GENERIC_NOUNS:
                entities.append(token)
    # 1b. 从问题本身扫描强名词（hit_tokens 可能因 route() 阈值未命中而漏掉）
    for w in STRONG_NOUNS:
        if len(w) >= 2 and w in question and w not in entities:
            entities.append(w)
    # 去重
    entities = list(dict.fromkeys(entities))

    # ── 2. 提取已有疑问意图 ──
    intent = None
    intent_dim = None
    for kw, dim in _INTENT_DIMENSIONS.items():
        if kw in question:
            intent = kw
            intent_dim = dim
            break

    # ── 3. 根据组合识别缺失维度并生成追问 ──
    has_entity = len(entities) > 0
    has_intent = intent is not None

    if has_entity and not has_intent:
        # 情形 A：有实体但无意图 → 问用户想了解哪个方面
        primary = entities[0]
        dims = _ENTITY_DIMENSIONS.get(primary, ["位置", "时间", "办理流程", "相关规定"])
        options = dims[:4]
        opts_str = "\n".join(f"  {i+1}. {primary}的{opt}" for i, opt in enumerate(options))
        return (
            f"🔍 您问到了「{primary}」，请问您具体想了解哪个方面？\n\n"
            f"{opts_str}\n\n"
            f"请回复序号或直接补充您的问题（例如：「{primary}的{options[0]}」）。"
        )

    if has_intent and not has_entity:
        # 情形 B：有意图但无实体 → 问用户具体对象
        # 从 _OBJECT_HINTS 找到与意图匹配的对象候选
        candidates = []
        for kw, objs in _OBJECT_HINTS.items():
            if kw in question:
                candidates.extend(objs)
        # 去重并过滤掉不在锚点集/强名词里的（避免推荐知识库没有的东西）
        candidates = [c for c in dict.fromkeys(candidates) if c in anchor_set or c in STRONG_NOUNS]
        if not candidates:
            # 没有明确候选 → 给主题提示
            candidates = [w for w in [
                "食堂", "宿舍", "图书馆", "校门", "校园卡",
                "奖学金", "转专业", "学费", "电费", "医务室",
            ] if w in anchor_set or w in STRONG_NOUNS][:4]
        if not candidates:
            return (
                f"🔍 您的问题「{question}」缺少具体对象。\n"
                f"请告诉我您想了解的是哪一项事务？"
            )
        opts_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates[:4]))
        intent_hint = f"关于「{intent_dim}」" if intent_dim else ""
        return (
            f"🔍 您想了解{intent_hint}，但没说具体对象。\n\n"
            f"知识库里相关的事务有：\n{opts_str}\n\n"
            f"请回复序号或直接补充您的问题（例如：「{candidates[0]}的{intent_dim}」）。"
        )

    if has_entity and has_intent:
        # 情形 C：有实体 + 有意图，但仍触发追问 → 检索相关度低
        # 让用户补充场景细节（如哪个校区/哪个时段/针对哪类学生）
        primary = entities[0]
        # 从 top_doc_keywords 中找未在问题中出现的词作为补充线索
        extra = [k for k in top_doc_keywords if k not in question and k not in entities][:3]
        hint = ""
        if extra:
            hint = f"\n相关文档提到了：{('、'.join(extra))}，您是否想问这些？"
        return (
            f"🔍 您问「{primary}的{intent_dim}」，但信息仍不够具体。{hint}\n\n"
            f"请补充一下细节，例如：\n"
            f"  - 具体是哪个校区/哪栋楼？\n"
            f"  - 针对哪类学生（新生/在校生/毕业生）？\n"
            f"  - 想了解哪个时段（工作日/周末/节假日）？"
        )

    # 情形 D：既无实体也无意图 → 给出主题列表
    topics = [w for w in [
        "食堂", "宿舍", "图书馆", "校门", "教学楼",
        "校园卡", "一卡通", "奖学金", "助学金", "转专业",
        "医务室", "校医院", "警务处", "教务处", "选课",
        "考试", "学费", "电费", "新生报到", "社团",
    ] if w in anchor_set or w in STRONG_NOUNS][:6]
    if not topics:
        return (
            f"🔍 您的问题「{question}」与知识库匹配度较低。\n"
            f"请告诉我您想了解哪方面的内容？"
        )
    opts_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(topics))
    return (
        f"🔍 您的问题「{question}」与知识库匹配度较低。\n\n"
        f"当前知识库涵盖以下主题：\n{opts_str}\n\n"
        f"请回复序号或直接补充您的问题。"
    )


# ────────────────────────────────────────────────────────────
# 调试 / 自测
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 模拟锚点集
    test_anchor_set = {
        "食堂", "宿舍", "图书馆", "体育馆", "运动场", "运动场所",
        "警务处", "教务处", "校医院", "医务室", "电话", "保卫处",
        "全称", "转专业", "奖学金",
        # 一些噪声词
        "学校", "学生", "什么", "怎么", "在哪",
    }

    test_questions = [
        "学校有几个食堂",
        "图书馆在哪",
        "警务处电话是多少",
        "运动场在哪",
        "医务室在哪",
        "学校全称是什么",
        "怎么转专业",
        "学校有几个校门",
        "今天天气怎么样",
    ]

    print("=" * 70)
    print("关键词评分测试")
    print("=" * 70)
    for q in test_questions:
        score, matched = compute_keyword_score(q, test_anchor_set)
        strong_hits = [m for m in matched if m["category"] == "strong_noun" and m["in_anchor"]]
        weak_hits = [m for m in matched if m["category"] == "weak" and m["in_anchor"]]
        print(f"\n问题: {q}")
        print(f"  keyword_score = {score:.4f}")
        print(f"  强名词命中: {[m['token'] for m in strong_hits]}")
        print(f"  弱词命中: {[m['token'] for m in weak_hits]}")
        print(f"  全部token:")
        for m in matched[:8]:
            mark = "✓" if m["in_anchor"] else " "
            print(f"    {mark} {m['token']:10s} cat={m['category']:12s} w={m['weight']}")
