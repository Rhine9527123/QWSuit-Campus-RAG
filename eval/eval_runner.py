#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
千问深信 · 自动化评测 MVP
=========================
不依赖 Ragas 库，直接用 DeepSeek API 做 judge 打分。
避开 datasets / huggingface-hub 版本冲突。

评测维度：
  1. Faithfulness（答案忠实度）— 答案是否基于检索上下文
  2. Answer Relevancy（答案相关性）— 答案是否直接回答问题
  3. Composite（综合得分）— 前两项均值

用法：
  1. 确保 server.py 已启动（http://localhost:8000）
  2. 设置环境变量：DEEPSEEK_API_KEY=sk-xxx
  3. python eval/eval_runner.py
"""

import json
import os
import re
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
DB_PATH = EVAL_DIR / "eval_results.db"
SERVER_URL = os.environ.get("RAG_SERVER_URL", "http://localhost:8000")
CHAT_URL = f"{SERVER_URL}/chat"
HEALTH_URL = f"{SERVER_URL}/health"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
JUDGE_MODEL = "deepseek-chat"
REQUEST_TIMEOUT = 90

# ── DeepSeek Judge ────────────────────────────────────────

FAITHFULNESS_PROMPT = """你是严格的事实核查员。判断【答案】中的每个声明是否都能从【检索上下文】找到依据。

评分规则：
- 1.0：答案所有信息都能在上下文中找到依据
- 0.5：答案部分依赖上下文，部分依赖外部知识
- 0.0：答案包含上下文中没有的信息

只返回一个数字（0.0 / 0.5 / 1.0），不要其他内容。

【检索上下文】
{context}

【答案】
{answer}

评分："""

ANSWER_RELEVANCY_PROMPT = """你是严格的问题评判员。判断【答案】是否直接回答了【问题】的核心需求。

评分规则：
- 1.0：答案直接、完整地回答了问题
- 0.5：答案只回答了问题的一部分
- 0.0：答案答非所问或完全缺失

只返回一个数字（0.0 / 0.5 / 1.0），不要其他内容。

【问题】
{question}

【答案】
{answer}

评分："""


def call_deepseek(prompt: str) -> float:
    """调用 DeepSeek API，返回 0.0 / 0.5 / 1.0"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=payload, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            m = re.search(r"([01](?:\.\d)?)", text)
            if m:
                return float(m.group(1))
            return 0.0
    except Exception as e:
        print(f"  [WARN] DeepSeek 调用失败: {e}")
        return 0.0


def score_faithfulness(answer: str, context: str) -> float:
    prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:2000])
    return call_deepseek(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer[:2000])
    return call_deepseek(prompt)


# ── RAG 服务调用 ──────────────────────────────────────────

def check_server() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return True
    except Exception:
        return False


def call_rag_chat(question: str) -> dict:
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        CHAT_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    result["_latency_ms"] = int((time.time() - t0) * 1000)
    return result


# ── SQLite 存储 ─────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            knowledge_version TEXT,
            total_questions INTEGER,
            avg_faithfulness REAL,
            avg_answer_relevancy REAL,
            avg_composite REAL,
            judge_model TEXT,
            duration_seconds INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS eval_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            question_id TEXT,
            question TEXT,
            category TEXT,
            answer TEXT,
            reference TEXT,
            faithfulness REAL,
            answer_relevancy REAL,
            composite REAL,
            source_file TEXT,
            latency_ms INTEGER,
            FOREIGN KEY (run_id) REFERENCES eval_runs(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE,
            description TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_run(run_data: dict, items: list) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO eval_runs (run_at, knowledge_version, total_questions,
            avg_faithfulness, avg_answer_relevancy, avg_composite,
            judge_model, duration_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_data["run_at"], run_data["knowledge_version"],
        run_data["total_questions"], run_data["avg_faithfulness"],
        run_data["avg_answer_relevancy"], run_data["avg_composite"],
        run_data["judge_model"], run_data["duration_seconds"],
    ))
    run_id = c.lastrowid
    for item in items:
        c.execute("""
            INSERT INTO eval_items (run_id, question_id, question, category,
                answer, reference, faithfulness, answer_relevancy, composite,
                source_file, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, item["question_id"], item["question"], item["category"],
            item["answer"], item["reference"],
            item["faithfulness"], item["answer_relevancy"], item["composite"],
            item["source_file"], item["latency_ms"],
        ))
    conn.commit()
    conn.close()
    return run_id


# ── HTML 报告生成 ───────────────────────────────────────────

CATEGORY_COLORS = {
    "生活服务": "#4f8cff",
    "学业管理": "#36d399",
    "入学毕业": "#f59e0b",
    "校园设施": "#a78bfa",
}


def generate_html_report(run_id: int, run_data: dict, items: list):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    avg_f = run_data["avg_faithfulness"]
    avg_r = run_data["avg_answer_relevancy"]
    avg_c = run_data["avg_composite"]

    items_html = ""
    for idx, item in enumerate(items, 1):
        cat_color = CATEGORY_COLORS.get(item["category"], "#888")
        f_color = "#36d399" if item["faithfulness"] >= 0.7 else "#f59e0b" if item["faithfulness"] >= 0.4 else "#ef4444"
        r_color = "#36d399" if item["answer_relevancy"] >= 0.7 else "#f59e0b" if item["answer_relevancy"] >= 0.4 else "#ef4444"
        ans = item['answer'][:400] + ('...' if len(item['answer']) > 400 else '')
        items_html += f"""
        <div class="item-card">
          <div class="item-header">
            <span class="item-id">#{idx}</span>
            <span class="item-category" style="background:{cat_color}20;color:{cat_color}">{item['category']}</span>
            <span class="item-id-tag">{item['question_id']}</span>
          </div>
          <div class="item-q">{item['question']}</div>
          <div class="item-scores">
            <span class="score-badge" style="background:{f_color}20;color:{f_color}">Faithfulness: {item['faithfulness']:.1f}</span>
            <span class="score-badge" style="background:{r_color}20;color:{r_color}">Relevancy: {item['answer_relevancy']:.1f}</span>
            <span class="score-badge composite">综合: {item['composite']:.1f}</span>
            <span class="latency">⏱ {item['latency_ms']}ms</span>
          </div>
          <div class="item-body">
            <div class="item-section">
              <div class="section-label">参考答案</div>
              <div class="item-ref">{item['reference']}</div>
            </div>
            <div class="item-section">
              <div class="section-label">系统答案</div>
              <div class="item-ans">{ans if ans.strip() else '<em style="color:#ef4444">（空答案）</em>'}</div>
            </div>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>千问深信 · 评测报告 #{run_id}</title>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d29; --border: #2d3148;
    --text: #e4e6ef; --dim: #8b8fa3;
    --accent: #4f8cff; --green: #36d399; --amber: #f59e0b; --red: #ef4444;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif; line-height: 1.7; }}
  .header {{ background: linear-gradient(135deg,#1a1d29,#0f1117); border-bottom: 1px solid var(--border); padding: 40px 5%; text-align: center; }}
  .header h1 {{ font-size: 26px; background: linear-gradient(135deg,#4f8cff,#a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header .meta {{ margin-top: 12px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }}
  .header .meta span {{ background: var(--card); border: 1px solid var(--border); padding: 6px 16px; border-radius: 20px; font-size: 13px; color: var(--dim); }}
  .header .meta b {{ color: var(--accent); }}
  .container {{ max-width: 1000px; margin: 0 auto; padding: 40px 5%; }}

  .kpi-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 40px; }}
  @media (max-width: 600px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; text-align: center; }}
  .kpi .num {{ font-size: 42px; font-weight: 800; }}
  .kpi .label {{ font-size: 13px; color: var(--dim); margin-top: 4px; }}
  .kpi.blue .num {{ color: var(--accent); }}
  .kpi.green .num {{ color: var(--green); }}
  .kpi.purple .num {{ color: #a78bfa; }}

  .item-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }}
  .item-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
  .item-id {{ font-size: 18px; font-weight: 700; color: var(--accent); }}
  .item-category {{ padding: 3px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; }}
  .item-id-tag {{ font-size: 12px; color: var(--dim); margin-left: auto; }}
  .item-q {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; padding: 12px; background: rgba(79,140,255,0.06); border-radius: 8px; border-left: 3px solid var(--accent); }}
  .item-scores {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }}
  .score-badge {{ padding: 4px 12px; border-radius: 10px; font-size: 13px; font-weight: 600; }}
  .score-badge.composite {{ background: rgba(167,139,250,0.15); color: #a78bfa; font-size: 14px; padding: 4px 14px; }}
  .latency {{ margin-left: auto; color: var(--dim); font-size: 13px; }}

  .item-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .item-body {{ grid-template-columns: 1fr; }} }}
  .item-section {{ }}
  .section-label {{ font-size: 12px; color: var(--dim); margin-bottom: 6px; font-weight: 600; }}
  .item-ref {{ font-size: 14px; color: var(--text); background: rgba(54,211,153,0.06); padding: 12px; border-radius: 8px; border-left: 3px solid var(--green); }}
  .item-ans {{ font-size: 14px; color: var(--dim); background: rgba(245,158,11,0.06); padding: 12px; border-radius: 8px; border-left: 3px solid var(--amber); }}

  .footer {{ text-align: center; padding: 32px; border-top: 1px solid var(--border); color: var(--dim); font-size: 13px; margin-top: 40px; }}
</style>
</head>
<body>
<div class="header">
  <h1>千问深信 · 自动化评测报告</h1>
  <div class="meta">
    <span>Run ID: <b>#{run_id}</b></span>
    <span>时间: <b>{now}</b></span>
    <span>知识库: <b>{run_data['knowledge_version']}</b></span>
    <span>Judge: <b>{run_data['judge_model']}</b></span>
    <span>题量: <b>{run_data['total_questions']}</b></span>
    <span>耗时: <b>{run_data['duration_seconds']}s</b></span>
  </div>
</div>
<div class="container">
  <div class="kpi-grid">
    <div class="kpi blue"><div class="num">{avg_f:.1%}</div><div class="label">Faithfulness<br>（答案忠实度）</div></div>
    <div class="kpi green"><div class="num">{avg_r:.1%}</div><div class="label">Answer Relevancy<br>（答案相关性）</div></div>
    <div class="kpi purple"><div class="num">{avg_c:.1%}</div><div class="label">综合得分</div></div>
  </div>
  {items_html}
</div>
<div class="footer">千问深信 · 自动化评测系统 | 生成时间：{now} | Judge: DeepSeek API</div>
</body>
</html>"""
    return html


# ── 主流程 ──────────────────────────────────────────────────

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    print("=" * 60)
    print("千问深信 · 自动化评测 MVP")
    print("=" * 60)

    # 检查 API Key
    if not DEEPSEEK_API_KEY or len(DEEPSEEK_API_KEY) < 10:
        print("[ERROR] 请设置 DEEPSEEK_API_KEY 环境变量")
        print("  示例：$env:DEEPSEEK_API_KEY='sk-xxx'")
        return
    print(f"[OK] DeepSeek API Key: {DEEPSEEK_API_KEY[:8]}...")

    # 检查 RAG 服务
    if not check_server():
        print(f"[ERROR] RAG 服务未运行 ({SERVER_URL})")
        print("  请先运行: python server.py")
        return
    print(f"[OK] RAG 服务在线: {SERVER_URL}")

    # 加载数据集
    dataset = load_dataset()
    questions = dataset["testset"]
    print(f"[INFO] 加载评测集: {len(questions)} 道题")

    # 初始化数据库
    init_db()

    # 开始评测
    start_time = time.time()
    items = []

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        reference = q.get("reference", "")
        category = q.get("category", "未分类")
        source = q.get("source", "")

        print(f"\n[{i}/{len(questions)}] {qid}: {question[:45]}...")

        # 1. 调用 RAG 拿答案
        try:
            result = call_rag_chat(question)
            answer = result.get("answer", "")
            latency = result.get("_latency_ms", 0)
            contexts = [s.get("text", "") for s in result.get("sources", [])[:3]]
            print(f"  -> 答案: {len(answer)} 字 | 上下文: {len(contexts)} 条 | 耗时: {latency}ms")
        except Exception as e:
            print(f"  [ERROR] RAG 调用失败: {e}")
            answer = ""
            latency = 0
            contexts = []

        if not answer:
            print(f"  [WARN] 空答案，跳过评分")
            items.append({
                "question_id": qid, "question": question, "category": category,
                "answer": "", "reference": reference, "source_file": source,
                "faithfulness": 0.0, "answer_relevancy": 0.0, "composite": 0.0,
                "latency_ms": latency,
            })
            continue

        # 2. 拼接检索上下文
        context = "\n\n".join(contexts) if contexts else answer

        # 3. DeepSeek judge 评分
        print(f"  -> 评分中...", end="", flush=True)
        f_score = score_faithfulness(answer, context)
        r_score = score_answer_relevancy(question, answer)
        composite = (f_score + r_score) / 2
        print(f" Faithfulness={f_score:.1f}, Relevancy={r_score:.1f}, 综合={composite:.1f}")

        items.append({
            "question_id": qid, "question": question, "category": category,
            "answer": answer, "reference": reference, "source_file": source,
            "faithfulness": f_score, "answer_relevancy": r_score,
            "composite": composite, "latency_ms": latency,
        })

    # 4. 计算平均分
    duration = int(time.time() - start_time)
    valid = [i for i in items if i["answer"]]
    avg_f = sum(i["faithfulness"] for i in valid) / len(valid) if valid else 0
    avg_r = sum(i["answer_relevancy"] for i in valid) / len(valid) if valid else 0
    avg_c = sum(i["composite"] for i in valid) / len(valid) if valid else 0

    run_data = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_version": dataset.get("knowledge_version", "unknown"),
        "total_questions": len(valid),
        "avg_faithfulness": round(avg_f, 3),
        "avg_answer_relevancy": round(avg_r, 3),
        "avg_composite": round(avg_c, 3),
        "judge_model": JUDGE_MODEL,
        "duration_seconds": duration,
    }

    # 5. 保存
    run_id = save_run(run_data, items)
    print(f"\n{'=' * 40}")
    print(f"[OK] 评测完成! Run ID: {run_id}")
    print(f"  Faithfulness:   {avg_f:.1%}")
    print(f"  Relevancy:      {avg_r:.1%}")
    print(f"  综合得分:        {avg_c:.1%}")
    print(f"  有效题数:        {len(valid)}/{len(questions)}")
    print(f"  总耗时:          {duration}s")
    print(f"{'=' * 40}")

    # 6. 生成 HTML 报告
    report_path = EVAL_DIR / f"eval_report_{run_id}.html"
    html = generate_html_report(run_id, run_data, items)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[OK] HTML 报告: {report_path}")
    print(f"  用浏览器打开即可查看")


if __name__ == "__main__":
    run_eval()
