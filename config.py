"""
RAG-Skeleton 中心化配置
======================

所有领域相关的字符串、提示词、标签均集中于此。
换领域时只需修改此文件（或通过环境变量覆盖），无需改动核心代码。

用法：
    from config import Config

    cfg = Config()           # 使用默认（通用）
    cfg = Config("finance")  # 使用预设领域
    cfg = Config.from_env()  # 从环境变量覆盖
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 领域预设
# ============================================================
# 每个预设定义了该领域的专用配置，换领域时选择对应预设即可。
# 也可以自定义预设或直接传参数覆盖。

DOMAIN_PRESETS = {
    "finance": {
        "app_name": "财务知识库助手",
        "app_title": "财务知识库",
        "app_description": "面向个体工商户的 AI 财务助手，基于检索增强生成（RAG）技术。",
        "system_prompt": (
            "你是专业的财务税务助手。请基于提供的资料回答用户问题。\n"
            "要求：\n"
            "1. 如果资料中有相关信息，优先引用资料内容\n"
            "2. 回答要简洁实用，适合非专业人士理解\n"
            "3. 如果不能从资料中找到答案，诚实说明"
        ),
        "domain_keywords": [
            "股", "基金", "债券", "期货", "外汇", "黄金", "原油",
            "GDP", "CPI", "PMI", "央行", "降息", "加息", "通胀",
            "A股", "港股", "美股", "IPO", "涨停", "跌停", "分红",
            "财报", "营收", "利润", "净利润", "毛利率", "ROE",
            "政策", "监管", "证监会", "银保监", "财政部", "发改委",
            "消费", "零售", "房产", "楼市", "房价", "制造业",
            "科技", "芯片", "新能源", "光伏", "锂电池", "AI",
            "宏观", "微观", "供给侧", "需求侧", "经济", "贸易",
            "人民币", "美元", "汇率", "利率", "存款", "贷款",
            "保险", "证券", "银行", "信托", "私募", "公募",
            "指数", "上证", "深证", "创业板", "科创板", "北交所",
        ],
        "scoring_prompt": (
            "你是一个财经内容审核员。评估以下文章与「财经/商业/经济/投资」主题的相关程度。\n\n"
            "评分标准：\n"
            "- 0-2: 完全无关（娱乐八卦、体育、天气预报等）\n"
            "- 3-4: 弱相关（涉及消费但不涉及经济分析）\n"
            "- 5-6: 一般相关（提及财经但非主要内容）\n"
            "- 7-8: 高度相关（财经是主要内容）\n"
            "- 9-10: 核心内容（专业财经分析/政策解读/市场数据）\n\n"
            "只回复一个 0-10 的整数数字，不要任何解释。\n\n"
            "文章标题: {title}\n"
            "文章内容: {content}\n\n"
            "相关度(0-10):"
        ),
        "excel_category": "经营数据",
        "excel_summary_category": "经营数据概要",
        "excel_source": "本地Excel",
        "category_placeholder": "例如：税务政策、经营数据",
        "default_category": "未知",
        "mcp_server_name": "rag-finance",
        "mcp_instructions": (
            "你连接了用户的专属知识库，里面可能包含各种业务文档："
            "税务政策、会计数据、经营分析、天气影响、竞品调研等。"
            "内容类型不固定，取决于用户上传了什么。"
            "\n\n判断逻辑："
            "\n- 涉及「用户自己的数据/文档/业务」→ 调用 rag_chat"
            "\n- 纯闲聊/通用常识/你确定能答对 → 不调用"
            "\n- 拿不准 → 宁可调用，查了再说"
            "\n\n重要：不要根据话题类型（如'天气''财务'）硬性判断，"
            "因为用户可能上传了任何主题的文档。"
            "如果需要确认知识库里有什么，先调用 rag_files 查看。"
        ),
    },
    "medical": {
        "app_name": "医疗知识库助手",
        "app_title": "医疗知识库",
        "app_description": "AI 医疗咨询助手，基于检索增强生成（RAG）技术。",
        "system_prompt": (
            "你是专业的医疗健康助手。请基于提供的资料回答用户问题。\n"
            "要求：\n"
            "1. 如果资料中有相关信息，优先引用资料内容\n"
            "2. 回答要实用，帮助用户理解，但不替代专业医生诊断\n"
            "3. 如果不能从资料中找到答案，诚实说明"
        ),
        "domain_keywords": [
            "诊断", "治疗", "药物", "手术", "临床", "患者",
            "症状", "病因", "预防", "疫苗", "检查", "检验",
            "内科", "外科", "儿科", "妇科", "心血管", "肿瘤",
            "糖尿病", "高血压", "感染", "抗生素", "激素",
            "CT", "MRI", "X光", "超声", "心电图",
            "医保", "住院", "门诊", "处方", "中药", "西药",
        ],
        "scoring_prompt": (
            "你是一个医疗内容审核员。评估以下文章与「医学/健康/医疗」主题的相关程度。\n\n"
            "评分标准：\n"
            "- 0-2: 完全无关\n"
            "- 3-4: 弱相关（提及健康但不涉及医学知识）\n"
            "- 5-6: 一般相关（提及医学但非主要内容）\n"
            "- 7-8: 高度相关（医学是主要内容）\n"
            "- 9-10: 核心内容（专业医学分析/临床指南/药物说明）\n\n"
            "只回复一个 0-10 的整数数字，不要任何解释。\n\n"
            "文章标题: {title}\n"
            "文章内容: {content}\n\n"
            "相关度(0-10):"
        ),
        "excel_category": "医疗数据",
        "excel_summary_category": "医疗数据概要",
        "excel_source": "本地Excel",
        "category_placeholder": "例如：诊断手册、药品说明、治疗方案",
        "default_category": "未知",
        "mcp_server_name": "rag-medical",
        "mcp_instructions": (
            "你连接了用户的专属知识库，里面可能包含各种医疗健康文档。"
            "内容类型不固定，取决于用户上传了什么。"
            "\n\n判断逻辑："
            "\n- 涉及「用户自己的数据/文档/业务」→ 调用 rag_chat"
            "\n- 纯闲聊/通用常识/你确定能答对 → 不调用"
            "\n- 拿不准 → 宁可调用，查了再说"
        ),
    },
    "legal": {
        "app_name": "法律知识库助手",
        "app_title": "法律知识库",
        "app_description": "AI 法律咨询助手，基于检索增强生成（RAG）技术。",
        "system_prompt": (
            "你是专业的法律助手。请基于提供的资料回答用户问题。\n"
            "要求：\n"
            "1. 如果资料中有相关信息，优先引用资料内容\n"
            "2. 回答要清晰准确，但不替代专业律师意见\n"
            "3. 如果不能从资料中找到答案，诚实说明"
        ),
        "domain_keywords": [
            "法律", "法规", "条例", "司法解释", "判决", "裁定",
            "合同", "侵权", "刑事", "民事", "行政", "诉讼",
            "仲裁", "律师", "法院", "检察院", "公安",
            "知识产权", "专利", "商标", "版权", "著作权",
            "公司法", "劳动法", "婚姻法", "继承法", "物权法",
            "违约", "赔偿", "罚款", "拘留", "有期徒刑",
        ],
        "scoring_prompt": (
            "你是一个法律内容审核员。评估以下文章与「法律/法规/司法」主题的相关程度。\n\n"
            "评分标准：\n"
            "- 0-2: 完全无关\n"
            "- 3-4: 弱相关\n"
            "- 5-6: 一般相关\n"
            "- 7-8: 高度相关\n"
            "- 9-10: 核心内容（法律法规条文/司法解释/案例分析）\n\n"
            "只回复一个 0-10 的整数数字，不要任何解释。\n\n"
            "文章标题: {title}\n"
            "文章内容: {content}\n\n"
            "相关度(0-10):"
        ),
        "excel_category": "法律数据",
        "excel_summary_category": "法律数据概要",
        "excel_source": "本地Excel",
        "category_placeholder": "例如：法律法规、司法解释、案例分析",
        "default_category": "未知",
        "mcp_server_name": "rag-legal",
        "mcp_instructions": (
            "你连接了用户的专属知识库，里面可能包含各种法律文档。"
            "内容类型不固定，取决于用户上传了什么。"
            "\n\n判断逻辑："
            "\n- 涉及「用户自己的数据/文档/业务」→ 调用 rag_chat"
            "\n- 纯闲聊/通用常识/你确定能答对 → 不调用"
            "\n- 拿不准 → 宁可调用，查了再说"
        ),
    },
    "campus": {
        "app_name": "校园助手",
        "app_title": "校园知识库",
        "app_description": "面向校园生活的 AI 助手，基于检索增强生成（RAG）技术。",
        "system_prompt": (
            "你是深圳信息职业技术大学的校园助手「千问深信」，基于提供的校园资料回答用户问题。\n"
            "要求：\n"
            "1. 严格基于参考资料回答，不要编造校园规章或信息\n"
            "2. 如果用户同时问多个问题，请分点（用1.2.3.编号）分别回答，每个问题对应一段\n"
            "3. 回答简洁直接，贴近学生场景，对学生/家长友好\n"
            "4. 如果资料中未找到某个子问题的答案，请明确说明该部分「资料中未找到相关内容」，其他部分正常回答\n"
            "5. 每个问题只回答一次，不要重复编号、不要重复回答、不要复述问题，答完即止\n"
            "6. 同一个信息不要重复表述"
        ),
        "domain_keywords": [
            "校园", "学校", "学院", "大学", "中学", "小学", "高校",
            "教务", "课程", "选课", "排课", "考试", "成绩", "绩点", "GPA",
            "学分", "毕业", "学位", "论文", "答辩", "导师", "导师制",
            "招生", "录取", "入学", "报到", "学籍", "转专业", "休学", "复学",
            "宿舍", "住宿", "食堂", "校园卡", "门禁", "校车",
            "社团", "活动", "学生会", "实践", "志愿者", "志愿时",
            "奖学金", "助学金", "助学贷款", "学费", "缴费",
            "请假", "考勤", "旷课", "违纪", "处分", "申诉",
            "图书馆", "借阅", "自习室", "实验室", "机房",
            "心理咨询", "校医院", "医务室", "保险", "体检",
            "就业", "实习", "校招", "秋招", "春招", "考研", "保研", "留学",
            "教务处", "学工处", "后勤", "保卫处", "财务处",
            "校历", "开学", "放假", "寒暑假", "节假日",
        ],
        "scoring_prompt": (
            "你是一个校园内容审核员。评估以下文章与「校园生活/教学管理/学生事务」主题的相关程度。\n\n"
            "评分标准：\n"
            "- 0-2: 完全无关（财经/医疗/法律等专业领域或无关内容）\n"
            "- 3-4: 弱相关（提及学校但非主要讨论对象）\n"
            "- 5-6: 一般相关（涉及校园生活但信息较少）\n"
            "- 7-8: 高度相关（校园规章/教学事务/学生服务为主要内容）\n"
            "- 9-10: 核心内容（学校官方文件/校规校纪/教务通知/招生简章）\n\n"
            "只回复一个 0-10 的整数数字，不要任何解释。\n\n"
            "文章标题: {title}\n"
            "文章内容: {content}\n\n"
            "相关度(0-10):"
        ),
        "excel_category": "校园数据",
        "excel_summary_category": "校园数据概要",
        "excel_source": "本地Excel",
        "category_placeholder": "例如：校规校纪、教务通知、生活指南",
        "default_category": "未知",
        "mcp_server_name": "rag-campus",
        "mcp_instructions": (
            "你连接了用户的专属校园知识库，里面可能包含各类校园文档："
            "校规校纪、教务通知、招生简章、生活指南、社团活动等。"
            "内容类型不固定，取决于用户上传了什么。"
            "\n\n判断逻辑："
            "\n- 涉及「用户自己校园的规章/数据/文档」→ 调用 rag_chat"
            "\n- 纯闲聊/通用常识/你确定能答对 → 不调用"
            "\n- 拿不准 → 宁可调用，查了再说"
            "\n\n重要：不要根据话题类型硬性判断，"
            "因为用户可能上传了任何主题的校园文档。"
            "如果需要确认知识库里有什么，先调用 rag_files 查看。"
        ),
        "faq_presets": [
            {
                "q": "学校地址在哪？",
                "a": "学校地址：深圳市龙岗区龙翔大道 2188 号。",
            },
            {
                "q": "学校是公办还是民办？",
                "a": "公办全日制职业本科院校，由深圳市人民政府举办。",
            },
            {
                "q": "宿舍是几人间？",
                "a": "统一四人间，上床下桌（床位 0.9×2 米），配空调、风扇、饮水机、热水器、阳台、独立卫生间。每层楼有公用洗衣机（4 元/次）。",
            },
            {
                "q": "学校有几个食堂？",
                "a": "3 个——A 食堂（红棉阁）、B 食堂（杜鹃阁）、C 食堂（紫荆阁）。",
            },
        ],
        "suggested_questions": [
            "图书馆开放时间是什么时候？",
            "宿舍楼怎么分布的？",
            "学校有哪些奖学金？",
            "体育馆有什么设施？",
        ],
    },
}


@dataclass
class Config:
    """RAG-Skeleton 中心化配置

    所有领域相关字符串集中于此。
    支持三种初始化方式：
      1. Config() — 通用默认
      2. Config("finance") — 使用预设
      3. Config.from_env() — 从环境变量覆盖
    """

    # ── 应用元信息 ──
    app_name: str = "知识库助手"
    app_title: str = "知识库"
    app_description: str = "AI 知识库助手，基于检索增强生成（RAG）技术。"

    # ── 系统提示词 ──
    system_prompt: str = (
        "你是专业的知识库助手。请基于提供的资料回答用户问题。\n"
        "要求：\n"
        "1. 如果资料中有相关信息，优先引用资料内容\n"
        "2. 回答要简洁实用\n"
        "3. 如果不能从资料中找到答案，诚实说明"
    )

    # ── 内容清洗相关 ──
    domain_keywords: list = field(default_factory=list)
    scoring_prompt: str = (
        "评估以下文章与知识库主题的相关程度。\n\n"
        "评分标准：\n"
        "- 0-2: 完全无关\n"
        "- 3-4: 弱相关\n"
        "- 5-6: 一般相关\n"
        "- 7-8: 高度相关\n"
        "- 9-10: 核心内容\n\n"
        "只回复一个 0-10 的整数数字，不要任何解释。\n\n"
        "文章标题: {title}\n"
        "文章内容: {content}\n\n"
        "相关度(0-10):"
    )

    # ── Excel 元数据 ──
    excel_category: str = "数据"
    excel_summary_category: str = "数据概要"
    excel_source: str = "本地Excel"

    # ── Web UI ──
    category_placeholder: str = "例如：分类标签"
    default_category: str = "未知"
    page_title: str = "个人助手"
    page_icon: str = "📚"

    # ── 多模态配置 ──
    ocr_language: str = "ch"              # OCR 语言：ch=中英混合, en=英文
    ocr_use_paddle: bool = True           # True=PaddleOCR优先, False=Tesseract优先
    stt_model_size: str = "tiny"          # faster-whisper 模型大小：tiny/base/small/medium
    stt_language: Optional[str] = "zh"    # STT 语言：zh=中文, None=自动检测
    image_description_prompt: str = (     # 图片描述系统提示（仅用 LLM 描述图片时）
        "请描述这张图片中的内容。如果有文字，请完整提取。"
        "如果是图表或数据展示，请详细描述数据和结构。"
    )

    # ── 多轮对话 ──
    session_enabled: bool = True
    session_max_turns: int = 10
    session_db_path: str = ""  # 空=使用默认路径 sessions.db

    # ── MCP Server ──
    mcp_server_name: str = "rag-knowledge"
    mcp_instructions: str = (
        "你连接了用户的专属知识库。内容类型不固定，取决于用户上传了什么。\n\n"
        "判断逻辑：\n"
        "- 涉及「用户自己的数据/文档/业务」→ 调用 rag_chat\n"
        "- 纯闲聊/通用常识/你确定能答对 → 不调用\n"
        "- 拿不准 → 宁可调用，查了再说\n\n"
        "如果需要确认知识库里有什么，先调用 rag_files 查看。"
    )

    # ── 主页常问问题 & 引导问题 ──
    faq_presets: list = field(default_factory=list)
    suggested_questions: list = field(default_factory=list)

    @classmethod
    def from_preset(cls, domain: str) -> "Config":
        """从预设加载某个领域的配置"""
        preset = DOMAIN_PRESETS.get(domain)
        if preset is None:
            available = ", ".join(DOMAIN_PRESETS.keys())
            print(f"[Config] 未知领域 '{domain}'，可用预设: {available}，使用默认配置")
            return cls()
        return cls(**preset)

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量读取配置（自动检测或手动指定）

        环境变量：
          RAG_DOMAIN=finance    → 使用 finance 预设
          RAG_APP_NAME=xxx     → 覆盖 app_name
          RAG_SYSTEM_PROMPT=xxx → 覆盖 system_prompt
          更多见下方映射表
        """
        domain = os.environ.get("RAG_DOMAIN", "")
        if domain and domain in DOMAIN_PRESETS:
            cfg = cls.from_preset(domain)
        else:
            cfg = cls()

        # 环境变量覆盖映射
        overrides = {
            "RAG_APP_NAME": "app_name",
            "RAG_APP_TITLE": "app_title",
            "RAG_APP_DESCRIPTION": "app_description",
            "RAG_SYSTEM_PROMPT": "system_prompt",
            "RAG_EXCEL_CATEGORY": "excel_category",
            "RAG_EXCEL_SUMMARY_CATEGORY": "excel_summary_category",
            "RAG_CATEGORY_PLACEHOLDER": "category_placeholder",
            "RAG_DEFAULT_CATEGORY": "default_category",
            "RAG_MCP_SERVER_NAME": "mcp_server_name",
            "RAG_MCP_INSTRUCTIONS": "mcp_instructions",
            "RAG_PAGE_TITLE": "page_title",
            "RAG_PAGE_ICON": "page_icon",
            "RAG_OCR_LANGUAGE": "ocr_language",
            "RAG_STT_MODEL_SIZE": "stt_model_size",
            "RAG_STT_LANGUAGE": "stt_language",
            "RAG_IMAGE_DESC_PROMPT": "image_description_prompt",
            "RAG_SESSION_ENABLED": "session_enabled",
            "RAG_SESSION_MAX_TURNS": "session_max_turns",
            "RAG_SESSION_DB_PATH": "session_db_path",
        }

        for env_var, attr in overrides.items():
            val = os.environ.get(env_var)
            if val:
                setattr(cfg, attr, val)

        # 关键词也可以从环境变量覆盖（逗号分隔）
        keywords_env = os.environ.get("RAG_DOMAIN_KEYWORDS")
        if keywords_env:
            cfg.domain_keywords = [k.strip() for k in keywords_env.split(",") if k.strip()]

        return cfg


# ============================================================
# 模块级单例（惰性加载）
# ============================================================
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例（首次调用时从环境变量加载）"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config():
    """强制重新加载配置（用于运行时切换领域）"""
    global _config
    _config = Config.from_env()
    return _config
