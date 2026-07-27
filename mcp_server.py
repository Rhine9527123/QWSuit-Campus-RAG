"""
RAG MCP Server - Hermes 的知识库翻译官
======================================

作用：把 Hermes 的 MCP 协议调用，翻译成对 RAG FastAPI 的 HTTP 请求

架构：
  用户 → Hermes → MCP(stdio) → 本文件(HTTP) → server.py:8000 → ChromaDB

启动方式：不需要手动启动，Hermes 会自动通过 stdio 启动这个脚本

暴露给 Hermes 的工具：
  1. rag_chat               - 知识库问答（核心）
  2. rag_files              - 查看知识库文件列表
  3. rag_delete             - 移入垃圾桶（30天软删除）
  4. rag_trash              - 查看垃圾桶内容
  5. rag_restore            - 从垃圾桶恢复文件
  6. rag_trash_clean        - 清理过期垃圾桶文件
  7. rag_ocr                - 图片 OCR 文字识别 (PaddleOCR/Tesseract)
  8. rag_transcribe         - 语音转文字 (faster-whisper)
  9. rag_multimodal_status  - 多模态依赖状态检查

领域切换：修改 config.py 或设置 RAG_DOMAIN 环境变量
"""

import json
import os
import requests
from mcp.server.fastmcp import FastMCP

# 导入中心化配置
from config import get_config

# ============================================================
# 配置：RAG FastAPI 服务的地址
# ============================================================
RAG_BASE_URL = "http://localhost:8000"

# 加载领域配置
_cfg = get_config()

# ============================================================
# 创建 MCP Server 实例
# ============================================================
mcp = FastMCP(
    name=_cfg.mcp_server_name,
    instructions=_cfg.mcp_instructions,
)


# ============================================================
# 工具 1：知识库问答（核心）
# ============================================================
@mcp.tool()
def rag_chat(question: str, verify: bool = False) -> str:
    """
    从用户的知识库中检索相关文档并生成回答。

    知识库是用户专属的，内容不固定（可能含任何类型的文档）。
    只要问题涉及「用户的业务、数据或上传的资料」，就应该调用此工具。

    示例：
    - 「这个月利润为什么降了」→ 调用（涉及用户自己的数据）
    - 「这个文件讲了什么」→ 调用（涉及知识库内容）
    - 「你好」「1+1等于几」→ 不调用（纯闲聊/通用常识）

    参数：
    - question: 用户的问题，直接传入原话即可
    - verify: 是否启用 FactGuard 事实核查（对 RAG 回答做可信度验证），默认关闭
    """
    try:
        resp = requests.post(
            f"{RAG_BASE_URL}/chat",
            json={"question": question, "verify": verify},
            timeout=90,
        )
        data = resp.json()

        answer = data.get("answer", "未获取到回答")
        sources = data.get("sources", [])

        # 把来源片段拼到回答后面，方便 Hermes 引用
        result = f"**回答：**\n{answer}"

        if sources:
            result += "\n\n**参考来源：**\n"
            for i, src in enumerate(sources, 1):
                text = src.get("text", "")[:100]  # 每条来源只取前100字
                meta = src.get("metadata", {})
                source_file = meta.get("filename", "未知")
                result += f"{i}. [{source_file}] {text}...\n"

        # FactGuard 核查结果
        factcheck = data.get("factcheck")
        if factcheck and factcheck.get("available"):
            fc = factcheck
            result += "\n\n**🔍 事实核查 (FactGuard)：**\n"
            result += f"  锚点数: {fc.get('total_anchors', 0)} | ✅{fc.get('factual', 0)} ❌{fc.get('unfactual', 0)} ❓{fc.get('uncertain', 0)}\n"
            result += f"  准确率: {fc.get('accuracy', 0):.0%} | 风险: {fc.get('risk_level', '未知')}\n"
            result += f"  结论: {'✅ 通过' if fc.get('verdict') == 'pass' else '⚠️ 有疑问'}\n"
            issues = fc.get("issues", [])
            if issues:
                result += "  存疑锚点:\n"
                for iss in issues:
                    result += f"    - 锚点#{iss['anchor_id']}: {iss['verdict']} ({iss.get('reason', '')[:50]})\n"

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except requests.exceptions.Timeout:
        return "错误：RAG 服务响应超时（90秒），请稍后重试"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 2：查看知识库文件列表
# ============================================================
@mcp.tool()
def rag_files() -> str:
    """
    查看当前知识库中有哪些文件。

    适用场景：用户想知道知识库里有什么文档、上传了哪些文件时调用。

    返回：文件列表，包含文件名、类型、大小、上传时间。
    """
    try:
        resp = requests.get(
            f"{RAG_BASE_URL}/files",
            timeout=10,
        )
        data = resp.json()

        files = data.get("files", [])
        total = data.get("total", 0)

        if total == 0:
            return "知识库当前为空，没有上传任何文件。"

        result = f"知识库共有 {total} 个文件：\n\n"
        for f in files:
            result += (
                f"- {f['filename']} "
                f"({f['file_type']}, {f['size_human']}, "
                f"更新于 {f['modified']})\n"
            )

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 3：移入垃圾桶（软删除）
# ============================================================
@mcp.tool()
def rag_delete(filename: str) -> str:
    """
    将知识库文件移入垃圾桶（软删除），30天后自动永久删除。

    适用场景：用户想要移除某个过时或错误的文档时调用。
    文件会先进入垃圾桶保留30天，期间可随时恢复。

    参数：
    - filename: 要删除的文件名，例如 "tax_policy.txt"

    注意：移动后会自动重建索引，可能需要约30秒。

    返回：删除结果。如果成功，提示文件已移入垃圾桶和恢复方法。
    """
    try:
        resp = requests.delete(
            f"{RAG_BASE_URL}/files/{filename}",
            timeout=60,
        )
        data = resp.json()

        status = data.get("status", "")
        message = data.get("message", "")
        is_trash = data.get("trash", False)

        if status == "ok":
            result = f"操作成功：{message}"
            if is_trash:
                result += "\n\n💡 提示：文件在垃圾桶中保留30天。"
                result += "\n   - 恢复文件：使用 rag_restore 工具"
                result += "\n   - 永久删除：使用 rag_trash_clean 自动清理（仅清理过期文件）"
            return result
        else:
            return f"操作失败：{message}"

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 4：查看垃圾桶
# ============================================================
@mcp.tool()
def rag_trash() -> str:
    """
    查看垃圾桶中的所有文件及其状态。

    适用场景：用户想知道有哪些文件被删除了、还剩几天过期、能否恢复时调用。

    返回：垃圾桶文件列表，包含删除时间、剩余天数、是否过期。
    """
    try:
        resp = requests.get(
            f"{RAG_BASE_URL}/trash",
            timeout=10,
        )
        data = resp.json()

        total = data.get("total", 0)
        items = data.get("items", [])

        if total == 0:
            return "垃圾桶当前为空，没有待处理的文件。"

        result = f"垃圾桶共有 {total} 个文件（{data.get('trash_days', 30)}天保留期）：\n\n"
        for item in items:
            status = "⚠ 已过期" if item.get("expired") else f"⏳ 剩余 {item['days_left']} 天"
            result += (
                f"- {item['filename']} "
                f"({item.get('file_type', '?')}, {item.get('size_human', '?')})\n"
                f"  删除于: {item.get('deleted_at', '?')}  "
                f"过期: {item.get('expires_at', '?')}  "
                f"{status}\n"
            )

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 5：从垃圾桶恢复文件
# ============================================================
@mcp.tool()
def rag_restore(filename: str) -> str:
    """
    从垃圾桶中恢复指定文件到知识库。

    适用场景：用户误删了文件，或者想重新使用某个之前删除的文档时调用。
    恢复后会重建索引，文件重新可被检索。

    参数：
    - filename: 要恢复的文件名（垃圾桶中的名称），例如 "tax_policy.txt"

    返回：恢复结果。
    """
    try:
        resp = requests.post(
            f"{RAG_BASE_URL}/trash/{filename}/restore",
            timeout=60,
        )
        data = resp.json()

        status = data.get("status", "")
        message = data.get("message", "")

        if status == "ok":
            return f"✅ {message}"
        else:
            return f"恢复失败：{message}"

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 6：清理过期垃圾桶文件
# ============================================================
@mcp.tool()
def rag_trash_clean() -> str:
    """
    自动清理垃圾桶中过期的文件（超过保留期限的）。

    适用场景：
    - 用户想立即清理过期文件，不等服务重启时自动清理
    - 用户想释放磁盘空间

    注意：此操作不可逆，清理后文件无法恢复。

    返回：清理结果，包含被清理的文件列表。
    """
    try:
        resp = requests.post(
            f"{RAG_BASE_URL}/trash/auto-clean",
            timeout=30,
        )
        data = resp.json()

        cleaned = data.get("cleaned", 0)
        cleaned_files = data.get("cleaned_files", [])
        remaining = data.get("remaining", 0)
        message = data.get("message", "")

        result = f"🧹 {message}\n"

        if cleaned_files:
            result += "\n已清理的文件：\n"
            for f in cleaned_files:
                result += f"  - {f}\n"

        if remaining > 0:
            result += f"\n垃圾桶中还有 {remaining} 个文件未过期，暂不清理。"

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 工具 7：图片 OCR 文字识别
# ============================================================
@mcp.tool()
def rag_ocr(image_path: str) -> str:
    """
    对本地图片执行 OCR 文字识别并返回提取的文字。

    适用场景：
    - 用户说「帮我识别这张图片里的文字」并给了图片路径
    - 用户想从截图/照片/扫描件中提取文字

    技术选型：PaddleOCR（首选，中文识别好）→ Tesseract（降级兜底）

    参数：
    - image_path: 本地图片文件路径，支持 .png / .jpg / .bmp / .tiff / .webp

    返回：识别出的文字内容。不保存文件，不入知识库。
    """
    try:
        # 检查文件是否存在
        path = os.path.expanduser(image_path)
        if not os.path.isfile(path):
            if not os.path.isfile(image_path):
                return f"文件不存在: {image_path}"

        filename = os.path.basename(path)
        with open(path, "rb") as f:
            file_bytes = f.read()

        resp = requests.post(
            f"{RAG_BASE_URL}/ocr",
            files={"file": (filename, file_bytes)},
            timeout=120,
        )
        data = resp.json()

        if data.get("status") == "ok":
            text = data.get("text", "")
            method = data.get("method", "?")
            chars = data.get("char_count", 0)
            return f"[OCR: {method}] 识别 {chars} 字\n\n{text}"
        else:
            return f"OCR 失败: {data.get('message', '未知错误')}"

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"OCR 错误：{str(e)}"


# ============================================================
# 工具 8：语音转文字
# ============================================================
@mcp.tool()
def rag_transcribe(audio_path: str) -> str:
    """
    对本地音频文件执行语音转文字（STT）并返回转写文本。

    适用场景：
    - 用户说「帮我把这段录音转成文字」并给了音频路径
    - 用户想从会议录音/语音备忘录中提取文字

    技术选型：faster-whisper（CTranslate2 加速，CPU 友好）

    参数：
    - audio_path: 本地音频文件路径，支持 .wav / .mp3 / .m4a / .ogg / .flac / .aac

    返回：转写后的文字内容。不保存文件，不入知识库。
    """
    try:
        # 检查文件是否存在
        path = os.path.expanduser(audio_path)
        if not os.path.isfile(path):
            if not os.path.isfile(audio_path):
                return f"文件不存在: {audio_path}"

        filename = os.path.basename(path)
        with open(path, "rb") as f:
            file_bytes = f.read()

        resp = requests.post(
            f"{RAG_BASE_URL}/transcribe",
            files={"file": (filename, file_bytes)},
            timeout=300,  # Whisper 转录可能需要较长时间
        )
        data = resp.json()

        if data.get("status") == "ok":
            text = data.get("text", "")
            return f"[语音转写] {len(text)} 字\n\n{text}"
        else:
            return f"语音转写失败: {data.get('message', '未知错误')}"

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except requests.exceptions.Timeout:
        return "错误：语音转写超时（5分钟），音频可能过长或模型未加载"
    except Exception as e:
        return f"语音转写错误：{str(e)}"


# ============================================================
# 工具 9：多模态依赖状态检查
# ============================================================
@mcp.tool()
def rag_multimodal_status() -> str:
    """
    检查多模态相关依赖的可用性。

    适用场景：
    - 用户问「OCR 能用吗」「语音转文字装好了没」
    - 部署后验证 PaddleOCR / Tesseract / faster-whisper 是否就绪

    返回：各依赖的可用状态 + 安装建议。
    """
    try:
        resp = requests.get(
            f"{RAG_BASE_URL}/multimodal/status",
            timeout=10,
        )
        data = resp.json()

        def status_icon(ok: bool) -> str:
            return "✅" if ok else "❌"

        result = "🔧 多模态依赖状态\n"
        result += f"{'─' * 30}\n"
        result += f"{status_icon(data.get('paddleocr', False))} PaddleOCR (中文 OCR)\n"
        result += f"{status_icon(data.get('tesseract', False))} Tesseract (降级 OCR)\n"
        result += f"{status_icon(data.get('faster_whisper', False))} faster-whisper (语音转文字)\n"
        result += f"\n{data.get('message', '')}"

        return result

    except requests.exceptions.ConnectionError:
        return "错误：RAG 服务未启动。请先运行 server.py（http://localhost:8000）"
    except Exception as e:
        return f"错误：{str(e)}"


# ============================================================
# 启动 MCP Server（通过 stdio 与 Hermes 通信）
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")
