"""
RAG-Skeleton - Streamlit 前端（API 客户端）v3 — 多轮对话支持
=========================================================

通用 AI 知识库骨架的前端界面。换一个知识库，就是一个新应用。

启动前需要先运行：python server.py（后端服务）
然后运行：streamlit run web.py

架构：
  web.py（前端，本文件） → HTTP API → server.py（后端）

新功能：多轮对话会话管理、锚点判断路由、PDF/Excel 上传
"""
import streamlit as st
import requests
import os
import json as _json
import time as _time
import random

# 导入中心化配置
from config import get_config
_cfg = get_config()

# ---- 主题配置（白色编辑风 · v3 minimalist-ui + design-taste-frontend） ----
# 编辑感纸白底 + 墨黑文字 + 赭石暖色 accent，摒弃渐变与 heavy shadow
bg_primary = "#fdfdfb"        # 主背景：微暖纸白（避免纯白刺眼）
bg_secondary = "#f5f3ef"      # 次级背景：暖米白
bg_sidebar = "#f8f6f2"        # 侧栏背景：略带暖色
text_primary = "#1a1815"      # 主文字：墨黑带暖（非纯黑，编辑感更柔）
text_secondary = "#6a655d"    # 次文字：暖灰
border_color = "#e5e1d8"      # 边框：浅暖灰
border_strong = "#c8c2b4"     # 强调边框（hover/active）
accent_color = "#8b3a1f"       # 赭石红（编辑风暖色，替代 AI 蓝）
accent_dim = "#b86b4a"         # 浅赭石
input_bg = "#ffffff"           # 输入框：纯白
user_bubble = "#f0ece2"        # 用户气泡：暖米色（与白底形成柔对比）
assistant_bubble = "#ffffff"   # 助手气泡：纯白
# minimalist-ui 禁止 heavy shadow；仅以发丝线作为分隔
shadow = "none"
shadow_accent = "none"

# ---- 后端 API 地址（自动适配 ngrok / 本地） ----
_query_params = st.query_params
BACKEND_URL = (
    _query_params.get("backend", [None])[0]
    or os.environ.get("BACKEND_URL")
    or "http://localhost:8000"
)

# ---- 页面配置 ----
st.set_page_config(
    page_title="千问深信",
    page_icon=_cfg.page_icon,
    layout="centered",
    initial_sidebar_state="expanded",
)

theme_css = f"""
<style>
/* ========== 字体加载（Google Fonts：Fraunces 编辑衬线 + Inter 正文 + JetBrains Mono 数字） ========== */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ========== 主题变量 ========== */
:root {{
    --bg-primary: {bg_primary};
    --bg-secondary: {bg_secondary};
    --bg-sidebar: {bg_sidebar};
    --text-primary: {text_primary};
    --text-secondary: {text_secondary};
    --border: {border_color};
    --border-strong: {border_strong};
    --accent: {accent_color};
    --accent-dim: {accent_dim};
    --input-bg: {input_bg};
    --user-bubble: {user_bubble};
    --assistant-bubble: {assistant_bubble};
    --shadow: {shadow};
    --shadow-accent: {shadow_accent};
}}

/* ========== 强制覆盖 Streamlit 原生主题变量（light） ========== */
/* 避免系统/浏览器深色主题与自定义白色变量混搭，所有 Streamlit 内部组件跟随白主题 */
:root, html, html[data-theme="light"], html[data-theme="dark"] {{
    --background-color: {bg_primary} !important;
    --secondary-background-color: {bg_secondary} !important;
    --tertiary-background-color: {bg_sidebar} !important;
    --primary-color: {accent_color} !important;
    --text-color: {text_primary} !important;
    --secondary-text-color: {text_secondary} !important;
    --text-color-rgb: 26, 24, 21 !important;
    --secondary-text-color-rgb: 106, 101, 93 !important;
    --background-color-rgb: 253, 253, 251 !important;
    --secondary-background-color-rgb: 245, 243, 239 !important;
    --primary-color-rgb: 139, 58, 31 !important;
    --border-color: {border_color} !important;
    --border-color-rgb: 229, 225, 216 !important;
    --border-width: 1.5px !important;
    --radius: 8px !important;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    --heading-font: 'Fraunces', serif !important;
    --code-font: 'JetBrains Mono', monospace !important;
    --link-color: {accent_color} !important;
    --link-color-rgb: 139, 58, 31 !important;
    --hover-color: {accent_dim} !important;
    --hover-color-rgb: 184, 107, 74 !important;
    --warning-color: #b85c2c !important;
    --warning-color-rgb: 184, 92, 44 !important;
    --warning-background-color: #fdf2e9 !important;
    --error-color: #8b1a1a !important;
    --error-color-rgb: 139, 26, 26 !important;
    --error-background-color: #fce8e8 !important;
    --success-color: #4a6b3a !important;
    --success-color-rgb: 74, 107, 58 !important;
    --success-background-color: #eef4e8 !important;
    --info-color: {accent_color} !important;
    --info-color-rgb: 139, 58, 31 !important;
    --info-background-color: #fbf1ec !important;
}}

.stApp {{
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-feature-settings: "ss01", "cv11";  /* Inter 优化细节 */
}}

/* 编辑风：纸白纯底，不加纹理与光晕（minimalist-ui 禁止渐变） */

/* ========== 强制覆盖 Streamlit 全局容器背景（防深色主题残留） ========== */
.stApp, .stApp > div, .main, section, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], [data-testid="stHeader"],
[data-testid="stMain"], [data-testid="stAppViewContainer"] > div {{
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}}

[data-testid="stHeader"] {{
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border);
}}

[data-testid="stToolbar"] {{
    background-color: var(--bg-primary) !important;
    color: var(--text-secondary) !important;
}}

/* ========== 侧边栏 ========== */
[data-testid="stSidebar"] {{
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
    box-shadow: none;
}}

[data-testid="stSidebar"] * {{
    color: var(--text-primary) !important;
}}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    letter-spacing: -0.01em;
}}

/* ========== 主区域 ========== */
.main .block-container {{
    background-color: transparent;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 760px;
    position: relative;
    z-index: 1;
}}

/* ========== 标题 ========== */
h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}}

h1 {{
    font-size: 2rem;
    font-weight: 600;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}}

h2 {{
    font-size: 1.25rem;
    font-weight: 500;
    letter-spacing: -0.015em;
}}

h3 {{
    font-size: 1.05rem;
    font-weight: 500;
}}

/* ========== 聊天消息 ========== */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
    animation: msg-in 0.3s ease-out;
}}

@keyframes msg-in {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.stChatMessage {{
    background: transparent !important;
}}

/* 用户消息：暖米色纯色气泡 + 发丝边框 */
[data-testid="stChatMessage"][data-testid="user"] {{
    background: var(--user-bubble) !important;
    border-radius: 12px 12px 4px 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}}

/* 助手消息：纯白气泡 + 发丝边框 */
[data-testid="stChatMessage"][data-testid="assistant"] {{
    background: var(--assistant-bubble) !important;
    border-radius: 12px 12px 12px 4px !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}}

/* 消息头像 */
[data-testid="stChatMessageAvatar"] {{
    border: 1.5px solid var(--border) !important;
    box-shadow: 0 0 0 2px var(--bg-primary);
}}

/* ========== 输入框 ========== */
[data-testid="stChatInput"] textarea {{
    background-color: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.25s ease !important;
    font-family: 'Inter', sans-serif !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,58,31,0.08) !important;
    outline: none !important;
}}

[data-testid="stChatInput"] {{
    background-color: transparent !important;
}}

/* ========== 底部输入区：改造成白底卡片，与页面融为一体但可识别 ========== */
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {{
    background-color: var(--bg-primary) !important;
    background-image: none !important;
}}

[data-testid="stBottom"] {{
    background-color: var(--bg-primary) !important;
    border-top: 1px solid var(--border);
    padding-top: 0.6rem;
    padding-bottom: 0.8rem;
}}

/* chat input 内层容器：把所有可能带深色背景的 div 都变白 */
.stChatInput,
.stChatInput > div,
.stChatInput > div > div,
.stChatInput > div > div > div,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {{
    background-color: var(--input-bg) !important;
    background-image: none !important;
    background: var(--input-bg) !important;
}}

/* chat input 外层容器 */
.stChatInput {{
    background: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.6rem 0.7rem !important;
}}

/* textarea 本身：更醒目的边框/占位提示 */
[data-testid="stChatInput"] textarea {{
    background-color: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 2.6rem !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,58,31,0.08) !important;
    outline: none !important;
}}

/* placeholder 颜色 */
[data-testid="stChatInput"] textarea::placeholder {{
    color: var(--text-secondary) !important;
    opacity: 1 !important;
}}

/* 发送按钮（右下角箭头）—— accent 实心，醒目 */
[data-testid="stChatInput"] [data-testid="stBaseButton-secondary"] {{
    background-color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    width: 2.2rem !important;
    height: 2.2rem !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

[data-testid="stChatInput"] [data-testid="stBaseButton-secondary"]:hover {{
    background-color: var(--accent-dim) !important;
    border-color: var(--accent-dim) !important;
    color: #ffffff !important;
}}

/* 发送按钮禁用态 */
[data-testid="stChatInput"] [data-testid="stBaseButton-secondary"]:disabled {{
    background-color: var(--border-strong) !important;
    border-color: var(--border-strong) !important;
    color: var(--text-secondary) !important;
}}

/* ========== 按钮（全局统一线条风格 · v2 升级） ========== */
.stButton>button,
.stButton>button[kind="primary"],
.stButton>button[kind="secondary"] {{
    background-color: transparent !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em;
    padding: 0.5rem 1rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    box-shadow: none !important;
    outline: none !important;
    text-shadow: none !important;
    position: relative;
}}

.stButton>button:hover,
.stButton>button[kind="primary"]:hover,
.stButton>button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton>button:hover {{
    background-color: rgba(139,58,31,0.04) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px);
    box-shadow: none !important;
}}

.stButton>button:focus,
.stButton>button:focus-visible,
.stButton>button:focus-within,
.stButton>button[kind="primary"]:focus,
.stButton>button[kind="secondary"]:focus {{
    outline: none !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,58,31,0.1) !important;
}}

.stButton>button:active {{
    transform: translateY(0) scale(0.98) !important;
    box-shadow: none !important;
}}

.stButton>button[kind="primary"] {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: rgba(139,58,31,0.04) !important;
}}

.stButton>button[kind="primary"]:hover {{
    background-color: rgba(139,58,31,0.1) !important;
    box-shadow: none !important;
}}

/* ========== 快捷问题卡片按钮（统一高度·精致卡片感） ========== */
/* FAQ 卡片 + 猜你想问按钮：固定高度、居中文字、圆角卡片 */
div[data-testid="column"] .stButton > button {{
    height: 3.2rem !important;
    min-height: 3.2rem !important;
    max-height: 3.2rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 12px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    writing-mode: horizontal-tb !important;
}}

div[data-testid="column"] .stButton > button:hover {{
    background-color: var(--input-bg) !important;
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(139,58,31,0.1) !important;
}}

div[data-testid="column"] .stButton > button:active {{
    transform: translateY(0) scale(0.97) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}}

/* "换一批"小按钮 */
div[data-testid="column"] .stButton > button[kind="secondary"][data-testid*="refresh_sq"] {{
    height: 3.2rem !important;
    min-height: 3.2rem !important;
    max-height: 3.2rem !important;
    padding: 0.25rem 0.4rem !important;
    font-size: 0.78rem !important;
    border-radius: 12px !important;
    background-color: transparent !important;
    border: 1px dashed var(--border-strong) !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    writing-mode: horizontal-tb !important;
    font-weight: 500 !important;
}}

div[data-testid="column"] .stButton > button[kind="secondary"][data-testid*="refresh_sq"]:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: rgba(139,58,31,0.04) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(139,58,31,0.08) !important;
}}

/* ===== 侧边栏按钮全局重设：覆盖 Streamlit 深色主题默认内联样式 ===== */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
    background-color: var(--input-bg) !important;
    background-image: none !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    min-height: 2.2rem !important;
    padding: 0.3rem 0.6rem !important;
    font-size: 0.9rem !important;
    line-height: 1 !important;
    box-shadow: none !important;
}}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
    background-color: rgba(139,58,31,0.06) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: none !important;
}}

/* 侧边栏中"确认导入/新建会话/删除当前会话"主按钮：accent 填充色使其醒目 */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: var(--accent) !important;
    background-image: none !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background-color: var(--accent-dim) !important;
    border-color: var(--accent-dim) !important;
    color: #ffffff !important;
}}

/* 侧边栏文件列表的 ✕ 删除按钮：方形细线框，hover 变赭石 */
[data-testid="stSidebar"] [data-testid="column"]:nth-child(2) .stButton > button,
[data-testid="stSidebar"] [data-testid="column"]:nth-child(2) [data-testid="stBaseButton-secondary"] {{
    width: 2rem !important;
    min-width: 2rem !important;
    height: 2rem !important;
    min-height: 2rem !important;
    padding: 0 !important;
    border-radius: 6px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    background-color: var(--input-bg) !important;
}}

[data-testid="stSidebar"] [data-testid="column"]:nth-child(2) .stButton > button:hover,
[data-testid="stSidebar"] [data-testid="column"]:nth-child(2) [data-testid="stBaseButton-secondary"]:hover {{
    color: var(--accent) !important;
    border-color: var(--accent) !important;
    background-color: rgba(139,58,31,0.06) !important;
}}

/* ========== 选择器 ========== */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [role="combobox"] {{
    background-color: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.25s ease !important;
}}

[data-testid="stSelectbox"] > div > div:focus,
[data-testid="stSelectbox"] [role="combobox"]:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,58,31,0.08) !important;
    outline: none !important;
}}

/* ========== 文本输入 ========== */
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] > div > div > div > div {{
    background-color: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}}

[data-testid="stTextInput"] input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,58,31,0.08) !important;
    outline: none !important;
}}

/* ========== 文件上传 ========== */
[data-testid="stFileUploader"] {{
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    background-color: var(--input-bg) !important;
    box-shadow: none !important;
    transition: border-color 0.25s ease, background-color 0.25s ease !important;
    padding: 0.8rem !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: var(--accent) !important;
    background-color: rgba(139,58,31,0.03) !important;
}}

[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text-secondary) !important;
}}

/* 修复上传区深色子容器（Dropzone、内层 section、提示文字区） */
[data-testid="stFileUploaderDropzone"] {{
    background-color: var(--input-bg) !important;
    min-height: unset !important;
    padding: 0.6rem !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
    color: var(--text-secondary) !important;
    fill: var(--text-secondary) !important;
}}

/* "Drag and drop file here" 文字及小字 */
[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: var(--text-secondary) !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: var(--text-secondary) !important;
}}

/* Upload / Browse files 按钮（Streamlit 默认给它 primary 深色） */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {{
    background-color: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.4rem 0.9rem !important;
    transition: all 0.25s ease !important;
}}

[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {{
    background-color: rgba(139,58,31,0.06) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}}

[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
}}

/* ========== 分割线 ========== */
hr, [data-testid="stSidebar"] hr {{
    border-color: var(--border);
    opacity: 0.5;
    margin: 1.2rem 0;
}}

/* ========== 文本样式 ========== */
.stCaption, [data-testid="stCaption"] {{
    color: var(--text-secondary) !important;
    font-size: 0.78rem;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.005em;
}}

p, span, div, label {{
    color: var(--text-primary);
}}

/* 数字等宽（引用资料百分比、命中数等） */
.stCaption, [data-testid="stCaption"] {{
    font-variant-numeric: tabular-nums;
}}

/* ========== 提示框 ========== */
[data-testid="stAlert"] {{
    border-radius: 8px;
    border: 1px solid var(--border);
    background-color: var(--bg-secondary);
    backdrop-filter: none;
}}

/* ========== Expander ========== */
[data-testid="stExpander"] {{
    border: 1px solid var(--border);
    border-radius: 8px;
    background-color: var(--bg-secondary);
    transition: border-color 0.25s ease;
}}

[data-testid="stExpander"]:hover {{
    border-color: var(--border-strong);
}}

/* ========== Spinner ========== */
.stSpinner > div {{
    border-color: var(--accent) transparent transparent transparent;
}}

/* ========== 来源标签 ========== */
.source-tag {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
    font-family: 'JetBrains Mono', monospace;
}}

/* ========== 路由徽章 ========== */
.route-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139,58,31,0.04);
    border: 1px solid rgba(139,58,31,0.18);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.015em;
    font-variant-numeric: tabular-nums;
    font-family: 'JetBrains Mono', monospace;
}}

/* ========== 滚动条 ========== */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: var(--border);
    border-radius: 10px;
    transition: background 0.2s;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--border-strong);
}}

/* ========== 选中态 ========== */
::selection {{
    background: rgba(139,58,31,0.18);
    color: var(--text-primary);
}}

/* ========== 响应式 ========== */
@media (max-width: 768px) {{
    .main .block-container {{
        padding: 1rem;
    }}
    h1 {{
        font-size: 1.5rem;
    }}
    [data-testid="stChatMessage"] {{
        padding: 0.4rem 0 !important;
    }}
}}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# ---- 支持的格式（线条风格图标） ----
LINE_ICONS = {
    "pdf": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    "txt": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="12" x2="8" y2="12"/><line x1="16" y1="16" x2="8" y2="16"/></svg>',
    "xlsx": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>',
    "image": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    "audio": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
    "default": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>',
}

# ---- UI 图标（统一 stroke-width 1.8，替换原 emoji） ----
UI_ICONS = {
    # sidebar 标题（原 📂💬📤）
    "knowledge": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h6l2 3h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/></svg>',
    "chat": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "upload": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    # 按钮（原 ➕🔄🗑️）
    "plus": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>',
    "refresh": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>',
    "trash": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
    # 状态指示（原 🚀🤔🧠💡📚🔍📝✅❌⚠️）
    "rocket": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>',
    "help": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "brain": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2z"/></svg>',
    "bulb": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.74V17h8v-2.26A7 7 0 0 0 12 2z"/></svg>',
    "book": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    "search": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "edit": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
    "check": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "x": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    "warn": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "bolt": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
}


@st.cache_resource
def check_backend():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _caption_html(icon_svg: str, text: str) -> None:
    """带 SVG 图标的 caption 渲染（st.caption 不支持 HTML，用 markdown+样式替代）"""
    # 统一 SVG 尺寸为 11px，与 caption 字号 0.78rem 协调
    icon = icon_svg.replace('width="12"', 'width="11"').replace('width="14"', 'width="11"')
    icon = icon.replace('height="12"', 'height="11"').replace('height="14"', 'height="11"')
    st.markdown(
        f'<div style="color:var(--text-secondary);font-size:0.78rem;font-family:Inter,sans-serif;'
        f'letter-spacing:0.005em;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:6px;padding:0.2rem 0;">'
        f'{icon}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def api_chat(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
    return resp.json()


def api_chat_stream(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(
        f"{BACKEND_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=120,
    )
    # 用可变容器存储，避免 nonlocal 重绑定导致返回值过期
    ctx = {
        "sources": [],
        "returned_session_id": session_id,
        "history_count": 0,
        "route_info": None,
        "progress_messages": [],
    }

    def token_gen():
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = _json.loads(line[6:])
                t = data["type"]
                if t == "sources":
                    ctx["sources"] = data.get("sources", [])
                elif t == "session":
                    ctx["returned_session_id"] = data["session_id"]
                elif t == "history_count":
                    ctx["history_count"] = data["count"]
                elif t == "route_info":
                    ctx["route_info"] = data
                elif t == "progress":
                    ctx["progress_messages"].append(data.get("message", ""))
                elif t == "token":
                    yield data["content"]
                elif t == "done":
                    break
                elif t == "error":
                    raise RuntimeError(data.get("message", "流式传输错误"))

    return token_gen, ctx


def api_upload(file_bytes, filename, category="未知"):
    resp = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, file_bytes)},
        data={"category": category},
        timeout=180,
    )
    return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def api_list_files():
    resp = requests.get(f"{BACKEND_URL}/files", timeout=5)
    return resp.json()


def api_delete_file(filename):
    api_list_files.clear()
    resp = requests.delete(f"{BACKEND_URL}/files/{filename}", timeout=60)
    return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def api_list_sessions():
    """获取会话列表"""
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions", timeout=3)
        data = resp.json()
        return data.get("sessions", [])
    except Exception:
        return []


def api_create_session(title=""):
    """新建会话"""
    api_list_sessions.clear()
    try:
        resp = requests.post(f"{BACKEND_URL}/sessions", params={"title": title}, timeout=5)
        data = resp.json()
        return data.get("session_id", "")
    except Exception:
        return ""


def api_delete_session(session_id):
    """删除会话"""
    api_list_sessions.clear()
    try:
        resp = requests.delete(f"{BACKEND_URL}/sessions/{session_id}", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "error"}


# ============================================================
# 侧边栏 - 关于（客户端只读）
# ============================================================
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:1.2rem 0 0.6rem;color:var(--text-primary);font-family:Fraunces,serif;font-weight:500;font-size:0.95rem;letter-spacing:-0.01em;">'
        f'{UI_ICONS["knowledge"]}<span>关于千问深信</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "深圳信息职业技术大学 AI 校园助手\n"
        "基于检索增强生成（RAG）技术\n"
        "由 DeepSeek 大模型驱动"
    )

    # 初始化会话状态（保留自动会话，但不暴露管理 UI）
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    st.divider()
    _caption_html(UI_ICONS["bulb"], "如有疑问，直接在对话框提问即可")


# ============================================================
# 主页面
# ============================================================
st.title("千问深信")
st.caption("深圳信息职业技术大学 AI 校园助手 · 基于检索增强生成（RAG）技术。")

# ---- 检查后端连接 ----
if not check_backend():
    st.warning("后端服务未启动！请先运行 `python server.py`")
    st.stop()

# ---- 聊天记录 ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 问题池 ──
_FAQ_PRESETS = [
    {"q": "学校地址在哪？", "a": "学校地址：深圳市龙岗区龙翔大道 2188 号。"},
    {"q": "学校是公办还是民办？", "a": "公办全日制职业本科院校，由深圳市人民政府举办。"},
    {"q": "宿舍是几人间？", "a": "统一四人间，上床下桌（床位 0.9×2 米），配空调、风扇、饮水机、热水器、阳台、独立卫生间。每层楼有公用洗衣机（4 元/次）。"},
    {"q": "学校有几个食堂？", "a": "3 个——A 食堂（红棉阁）、B 食堂（杜鹃阁）、C 食堂（紫荆阁）。"},
]

_SUGGEST_POOL = [
    # 学校基本信息
    "学校什么时候成立的？",
    "学校占地面积多大？",
    "学校有多少在校生？",
    "学校有什么荣誉？",
    # 校园布局
    "学校有几个门？",
    "东门靠近什么？",
    "北门靠近什么？",
    "南门靠近什么？",
    "宿舍楼叫什么？有几段？",
    "宿舍楼有几层？",
    "理论课在哪里上？",
    "专业课在哪里上？",
    "图书馆在哪里？开放时间？",
    "体育馆有什么设施？",
    # 食堂
    "A食堂有什么特色？",
    "B食堂有什么特色？",
    "C食堂有什么特色？",
    "食堂价格多少？",
    "食堂有清真窗口吗？",
    "食堂开放时间？",
    "学校支持外卖吗？",
    # 校园卡
    "校园卡有几种形态？",
    "校园卡可以用来做什么？",
    "校园卡怎么充值？",
    # 费用补贴
    "学校有餐费补贴吗？",
    "学校有电费补贴吗？",
    "学费多少钱？",
    "住宿费多少钱？",
    # 医疗
    "校医院在哪？",
    "校医院挂号多少钱？",
    "校医院能看什么病？",
    "大学生医保去哪报销？",
    # 运动设施
    "学校有什么运动设施？",
    "体育馆在哪里？",
    # 周边交通
    "东门外有什么？",
    "学校周边有什么商圈？",
    "怎么去地铁站？",
    "校园网怎么用？收费吗？",
    # 就业升学
    "学校毕业生就业率如何？",
    "毕业生平均薪资多少？",
    "有多少毕业生入职头部企业？",
    "专升本情况如何？",
]

# ---- 主页常问问题（仅无聊天记录时显示） ----
if not st.session_state.messages:
    if _FAQ_PRESETS:
        st.markdown("#### 常问问题")
        st.caption("点击即出答案，无需等待")
        _cols = st.columns(4)
        for i, faq in enumerate(_FAQ_PRESETS[:4]):
            with _cols[i]:
                if st.button(faq["q"], key=f"faq_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": faq["q"]})
                    with st.chat_message("user"):
                        st.markdown(faq["q"])
                    with st.chat_message("assistant"):
                        _caption_html(UI_ICONS["bolt"], "预设答案 · 秒回")
                        st.markdown(faq["a"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": faq["a"],
                        "sources": [],
                        "route_info": {"route": "preset", "hits": 0, "threshold": 0, "tokens": [], "needs_clarification": False},
                    })
                    st.rerun()
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 历史消息也显示路由徽章
        if msg["role"] == "assistant" and msg.get("route_info"):
            ri = msg["route_info"]
            r = ri.get("route", "fast")
            hits = ri.get("hits", 0)
            threshold = ri.get("threshold", 2)
            needs_clar = ri.get("needs_clarification", False)
            if r == "fast":
                _caption_html(UI_ICONS["rocket"], f"Fast RAG · 命中 {hits}/{threshold} 锚点")
            elif needs_clar:
                _caption_html(UI_ICONS["help"], f"Agentic RAG 追问 · 命中 {hits}/{threshold} 锚点")
            else:
                _caption_html(UI_ICONS["brain"], f"Agentic RAG · 命中 {hits}/{threshold} 锚点")

        st.markdown(msg["content"])

        # 历史消息也用小字展示引用资料
        if msg["role"] == "assistant" and msg.get("sources"):
            src_lines = []
            for i, src in enumerate(msg["sources"][:5], 1):
                s = src.get("score")
                pct = f"{s*100:.1f}%" if s is not None else "?"
                fname = src.get("metadata", {}).get("source") or src.get("metadata", {}).get("filename", "未知")
                src_lines.append(f"[{i}] {fname} ({pct})")
            _caption_html(UI_ICONS["book"], "引用资料: " + " | ".join(src_lines))

            with st.expander("查看详细片段", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("score")
                    pct_display = f"**{score*100:.1f}%**" if score is not None else "**?**"
                    text = src.get("text", "")
                    meta = src.get("metadata", {})
                    fname = meta.get("source", meta.get("filename", "未知文件"))
                    cat = meta.get("category", "未知")
                    st.markdown(f"**来源 {i}** - {fname} | 相关度: {pct_display} | 分类: {cat}")
                    st.markdown(f"> {text}")
                    st.divider()


# 加载动画随机话语
_LOADING_PHRASES = [
    "大模型正在拼命翻找中...",
    "大模型在知识库里打转，给您寻找最合适的答案...",
    "正在翻阅知识库，请稍候...",
    "AI 正在检索相关资料...",
    "正在从知识海洋中捞针...",
    "大模型正在苦思冥想...",
    "知识库精灵正在翻书...",
]


def do_chat(prompt_text):
    """执行一次对话（流式），传入文字内容"""
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        sources = []
        answer = ""
        stream_failed = False
        stream_has_output = False
        returned_session_id = None
        route_info = None
        progress_messages = []

        # 显示随机加载话语
        loading_phrase = random.choice(_LOADING_PHRASES)
        try:
            token_gen, ctx = api_chat_stream(
                prompt_text,
                session_id=st.session_state.current_session_id,
            )
            gen = token_gen()

            # ── Peek 第一个 token，期间显示加载动画 ──
            first_chunk = None
            with st.spinner(loading_phrase):
                try:
                    first_chunk = next(gen)
                except StopIteration:
                    pass

            route_info = ctx["route_info"]
            progress_messages = ctx.get("progress_messages", [])

            # ── 在答案前显示路由徽章 ──
            if route_info:
                r = route_info.get("route", "fast")
                hits = route_info.get("hits", 0)
                threshold = route_info.get("threshold", 2)
                tokens = route_info.get("tokens", [])
                is_clarification = any("追问" in p for p in progress_messages)
                if r == "fast":
                    _caption_html(UI_ICONS["rocket"], f"Fast RAG · 命中 {hits}/{threshold} 锚点: {', '.join(tokens[:5])}")
                elif is_clarification:
                    st.warning(f"Agentic RAG 追问 · 仅命中 {hits}/{threshold} 锚点，检索匹配度低，需要您换个问法")
                else:
                    st.info(f"Agentic RAG · 仅命中 {hits}/{threshold} 锚点，尝试多角度检索回答")

            # ── 流式输出答案 ──
            def combined_gen():
                nonlocal stream_has_output
                if first_chunk:
                    stream_has_output = True
                    yield first_chunk
                for chunk in gen:
                    stream_has_output = True
                    yield chunk

            answer = st.write_stream(combined_gen())
            sources = ctx["sources"]
            returned_session_id = ctx["returned_session_id"]

            # 更新会话 ID（首次提问时自动创建）
            if returned_session_id and returned_session_id != st.session_state.current_session_id:
                st.session_state.current_session_id = returned_session_id

        except Exception:
            stream_failed = True

        # 仅当流式完全没有输出任何 token 时才 fallback 到非流式重试，
        # 如果已经输出了部分内容则不重试，避免内容重复叠加
        if stream_failed and not stream_has_output:
            with st.spinner("正在重试（非流式）..."):
                try:
                    result = api_chat(
                        prompt_text,
                        session_id=st.session_state.current_session_id,
                    )
                    answer = result.get("answer", "服务返回了空回答")
                    sources = result.get("sources", [])
                    route_info = result.get("route_info")
                    sid = result.get("session_id")
                    if sid and sid != st.session_state.current_session_id:
                        st.session_state.current_session_id = sid
                except Exception as e:
                    answer = f"请求失败：{e}"
                    sources = []
            st.markdown(answer)

            # 非流式回退也显示路由徽章
            if route_info:
                r = route_info.get("route", "fast")
                hits = route_info.get("hits", 0)
                threshold = route_info.get("threshold", 2)
                needs_clar = route_info.get("needs_clarification", False)
                if r == "fast":
                    _caption_html(UI_ICONS["rocket"], f"Fast RAG · 命中 {hits}/{threshold} 锚点")
                elif needs_clar:
                    st.warning(f"Agentic RAG 追问 · 仅命中 {hits}/{threshold} 锚点，需要澄清")
                else:
                    st.info(f"Agentic RAG · 仅命中 {hits}/{threshold} 锚点，尝试回答")

        # ── 答案后用小字展示引用资料及匹配度 ──
        if sources:
            src_lines = []
            for i, src in enumerate(sources[:5], 1):
                s = src.get("score")
                pct = f"{s*100:.1f}%" if s is not None else "?"
                fname = src.get("metadata", {}).get("source") or src.get("metadata", {}).get("filename", "未知")
                src_lines.append(f"[{i}] {fname} ({pct})")
            _caption_html(UI_ICONS["book"], "引用资料: " + " | ".join(src_lines))

            # 详细片段（默认折叠，小字摘要已够用）
            with st.expander("查看详细片段", expanded=False):
                for i, src in enumerate(sources, 1):
                    score = src.get("score")
                    pct_display = f"**{score*100:.1f}%**" if score is not None else "**?**"
                    text = src.get("text", "")
                    meta = src.get("metadata", {})
                    fname = meta.get("source", meta.get("filename", "未知文件"))
                    cat = meta.get("category", "未知")
                    st.markdown(f"**来源 {i}** - {fname} | 相关度: {pct_display} | 分类: {cat}")
                    st.markdown(f"> {text}")
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "route_info": route_info,
        })


# ---- 引导问题点击处理（在 do_chat 定义之后执行） ----
if "_pending_sq" in st.session_state:
    sq = st.session_state.pop("_pending_sq")
    st.session_state.messages.append({"role": "user", "content": sq})
    do_chat(sq)
    st.rerun()


# ---- 猜你想问（始终在输入框上方显示，每次随机4个） ----
_SUGGEST_KEY = "_suggested_round"
if _SUGGEST_KEY not in st.session_state:
    st.session_state[_SUGGEST_KEY] = 0
_round = st.session_state[_SUGGEST_KEY]
_seed_val = hash(str(st.session_state.messages)) % 10000 + _round
_rng = random.Random(_seed_val)
_current_suggested = _rng.sample(_SUGGEST_POOL, min(4, len(_SUGGEST_POOL)))

st.caption("💡 猜你想问")

# 4个问题按钮 + 1个换一批按钮，同一行
_all_cols = st.columns([2.3, 2.3, 2.3, 2.3, 1.6])
for i, sq in enumerate(_current_suggested):
    with _all_cols[i]:
        if st.button(sq, key=f"sq_{_round}_{i}", use_container_width=True):
            st.session_state[_SUGGEST_KEY] = _round + 1
            st.session_state["_pending_sq"] = sq
            st.rerun()

with _all_cols[4]:
    if st.button("🔄 换一批", key=f"refresh_sq_{_round}", use_container_width=True):
        st.session_state[_SUGGEST_KEY] = _round + 1
        st.rerun()


# ---- 输入框 ----
if prompt := st.chat_input("输入问题，例如：图书馆几点开馆 / 校园卡怎么办理 / 东门在哪里..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state[_SUGGEST_KEY] = _round + 1
    do_chat(prompt)
