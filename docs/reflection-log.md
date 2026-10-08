# 千问深信 · Bug 修复 Reflection 日志

> 每次修改 bug 时记录：问题描述 → root cause 分析 → 修复方案 → 验证结果

---

## 2026-07-28：评测面板前端端口访问失败

### 问题描述
用户访问 `http://localhost:8003/` 显示 "localhost 拒绝连接"。

### 修复过程

#### 尝试 1：拆分前后端服务
- **做法**：把前端改成 `http.server 8003`，后端改成 `uvicorn 8002`
- **结果**：❌ 失败
- **Root Cause**：后台 `Start-Process` 启动的 python 子进程静默退出，我没有持续验证服务存活状态

#### 尝试 2：改用可见窗口启动
- **做法**：创建 `启动评测面板.bat`，两个服务各自独立 cmd 窗口
- **结果**：✅ 成功
- **Root Cause 分析**：

| 为什么之前失败 | 为什么这次修复有效 |
|--------------|------------------|
| 后台进程静默退出，错误日志不可见 | 可见 cmd 窗口，退出有提示 |
| 验证一次通过就以为稳定了 | 用户自己也能看到窗口状态 |
| 没有清理旧僵尸进程 | 启动前先清理旧进程 |
| 用户访问地址不明确 |  bat 脚本明确输出访问地址 |

### 关键教训

1. **后台启动 ≠ 服务存活**：`Start-Process` 启动的 python 子进程一旦出错就死了，必须用可见窗口或健康检查持续验证
2. **验证必须闭环**：启动 → 检查端口 → HTTP 探测 → 用户访问，四个环节都通过才算成功
3. **bat 脚本是最可靠的启动方式**：Windows 环境下，可见 cmd 窗口比后台进程稳定得多

### 修复后的架构

```
启动评测面板.bat
  ├── 评测面板 API（端口 8002）— uvicorn 可见窗口
  └── 评测面板前端（端口 8003）— http.server 可见窗口

前端 JS → fetch("http://localhost:8002/api/summary") → SQLite
```

### 验证结果
- 端口 8002：`/health` → 200 OK
- 端口 8003：`/` → 200 OK (20546 bytes)
- 浏览器访问 `http://localhost:8003/` → 正常显示面板

---

## 2026-07-28：server.py 启动失败 - huggingface-hub 版本冲突

### 问题描述
双击"启动RAG后端服务.bat"后报错：
```
ImportError: huggingface-hub>=0.34.0,<1.0 is required... but found huggingface-hub==2.1.1
```

### Root Cause 分析
- **直接原因**：`transformers` 要求 `huggingface-hub<1.0`，但 Ragas 安装时把 `huggingface-hub` 从 `0.36.2` 升级到了 `2.1.1`
- **连锁反应**：
  1. 装 Ragas → `huggingface-hub` 被升级到 `2.1.1`
  2. `transformers` 检测到版本不匹配 → 报 ImportError
  3. `sentence_transformers` 依赖 `transformers` → 整个 LlamaIndex embeddings 模块炸了

### 修复方案
- **第一次尝试**：降级 `huggingface-hub` 到 `0.36.2`
  - 结果：❌ 又引发 `datasets 5.1.0` 要求 `huggingface-hub>=1.31.0`
- **第二次尝试**：降级 `datasets` 到 `3.6.0`
  - 结果：✅ `server.py` 能启动了，但 Ragas 要求 `datasets>=4.0.0`
- **最终方案**：放弃 Ragas 库，直接用 DeepSeek API 做 judge
  - 结果：✅ 完全避开版本冲突

### 关键教训
1. **Ragas 的依赖链太重**：`ragas → datasets → huggingface-hub → transformers → sentence_transformers`，一环炸了全炸
2. **不引入重型依赖也是一种架构选择**：直接用 DeepSeek API 做 judge，逻辑更透明，版本更稳定
3. **MVP 阶段应优先选择最简路径**：不需要为了"标准框架"而引入不可控的版本风险

### 修复后的架构
```
eval_runner.py
  ├── 调用 server.py /chat 拿答案（HTTP）
  ├── 调用 DeepSeek API 做 judge 打分（urllib）
  └── 结果存入 SQLite + 生成 HTML 报告
```

### 验证结果
- 5 道题全部跑通
- Faithfulness: 30%, Relevancy: 90%, 综合: 60%
- HTML 报告正常生成：`eval/eval_report_1.html`

---

## 2026-07-28：README 中文乱码

### 问题描述
README.md 全篇中文乱码，GitHub 上显示为 GBK 乱码。

### Root Cause 分析
- **直接原因**：PowerShell 的 `Get-Content -replace` 默认用 GBK 编码读取 UTF-8 文件，再写出时编码损坏
- **触发条件**：
  1. README 是 UTF-8 编码（BOM: EF BB BF）
  2. PowerShell `Get-Content` 默认用系统编码（GBK）
  3. `-replace` 操作后 `Set-Content` 写出，编码错乱
  4. emoji 和中文全部变成乱码

### 修复方案
- **根本修复**：用 Python 的 `open(encoding="utf-8")` 读写，绕过 PowerShell 的编码问题
- **回滚修复**：`git checkout 3ed10ac -- README.md` 恢复到正确版本

### 关键教训
1. **PowerShell 处理 UTF-8 文件有坑**：`Get-Content` / `Set-Content` / `-replace` 组合是编码杀手
2. **修改前先备份**：用 `git tag v1.0-stable` 打标签，出问题直接回滚
3. **Python 处理文本比 PowerShell 可靠**：跨平台编码一致性更好

### 验证结果
- `[System.IO.File]::ReadAllText(path, [System.Text.Encoding]::UTF8)` 读取正常
- 文件前 3 行：`# 🏫 千问深信 — 校园智能问答助手` 显示正确

---

*日志持续更新中...*
