#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
千问深信 · 评测面板 API（独立服务，端口 8002）
=============================================
读取 eval_results.db，提供 REST API 给前端面板调用。
与主前端（Streamlit）物理隔离，互不冲突。

启动方式：
  python eval/panel_api.py
  或
  uvicorn eval.panel_api:app --host 0.0.0.0 --port 8002
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── 配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
DB_PATH = EVAL_DIR / "eval_results.db"
PANEL_DIR = EVAL_DIR / "panel"

app = FastAPI(
    title="千问深信 · 评测面板 API",
    description="读取 eval_results.db，提供评测数据给前端面板",
    version="1.0.0",
)

# CORS（允许任何来源访问面板）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 数据库工具 ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row):
    return dict(row) if row else None


# ── 健康检查 ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "eval-panel-api",
        "version": "1.0.0",
        "db_exists": DB_PATH.exists(),
    }


# ── 评测任务列表 ──────────────────────────────────────────

@app.get("/api/runs")
def list_runs(limit: int = 20):
    """获取最近的评测任务列表"""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, run_at, knowledge_version, total_questions,
            avg_faithfulness, avg_answer_relevancy, avg_composite,
            judge_model, duration_seconds
        FROM eval_runs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    """获取单个评测任务的详细信息"""
    conn = get_db()
    run = conn.execute("""
        SELECT * FROM eval_runs WHERE id = ?
    """, (run_id,)).fetchone()
    conn.close()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return row_to_dict(run)


# ── 评测题目详情 ──────────────────────────────────────────

@app.get("/api/runs/{run_id}/items")
def get_run_items(run_id: int):
    """获取单个评测任务的所有题目详情"""
    conn = get_db()
    # 先确认 run 存在
    run = conn.execute("SELECT id FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    items = conn.execute("""
        SELECT * FROM eval_items WHERE run_id = ? ORDER BY id
    """, (run_id,)).fetchall()
    conn.close()
    return [row_to_dict(r) for r in items]


# ── 汇总统计 ──────────────────────────────────────────────

@app.get("/api/summary")
def get_summary():
    """获取所有评测任务的汇总统计"""
    conn = get_db()
    # 总任务数
    total_runs = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    # 总题数
    total_items = conn.execute("SELECT COUNT(*) FROM eval_items").fetchone()[0]
    # 最新一次
    latest = conn.execute("""
        SELECT * FROM eval_runs ORDER BY id DESC LIMIT 1
    """).fetchone()
    # 平均分（所有任务）
    avg = conn.execute("""
        SELECT
            AVG(avg_faithfulness) as avg_f,
            AVG(avg_answer_relevancy) as avg_r,
            AVG(avg_composite) as avg_c
        FROM eval_runs
    """).fetchone()
    # 各分类平均（所有题目）
    category_avg = conn.execute("""
        SELECT category,
            AVG(faithfulness) as avg_f,
            AVG(answer_relevancy) as avg_r,
            AVG(composite) as avg_c,
            COUNT(*) as cnt
        FROM eval_items
        GROUP BY category
        ORDER BY avg_c DESC
    """).fetchall()
    # 趋势（最近 10 次）
    trend = conn.execute("""
        SELECT id, run_at, avg_faithfulness, avg_answer_relevancy, avg_composite
        FROM eval_runs
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()
    conn.close()

    return {
        "total_runs": total_runs,
        "total_items": total_items,
        "latest": row_to_dict(latest),
        "overall_avg": row_to_dict(avg),
        "category_avg": [row_to_dict(r) for r in category_avg],
        "trend": [row_to_dict(r) for r in trend],
    }


# ── 知识库版本管理 ─────────────────────────────────────────

@app.get("/api/knowledge-versions")
def list_versions():
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM knowledge_versions ORDER BY id DESC
    """).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.post("/api/knowledge-versions")
def create_version(version: str, description: str = ""):
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO knowledge_versions (version, description, created_at)
            VALUES (?, ?, ?)
        """, (version, description, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return {"status": "ok", "version": version}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Version '{version}' already exists")


# ── 静态文件（前端面板） ───────────────────────────────────

@app.get("/panel")
def serve_panel():
    """返回面板首页"""
    index = PANEL_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(status_code=404, detail="Panel not found. Run: python eval/panel_builder.py")


@app.get("/api/panel-status")
def panel_status():
    return {
        "panel_built": (PANEL_DIR / "index.html").exists(),
        "db_exists": DB_PATH.exists(),
        "total_runs": get_db().execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
