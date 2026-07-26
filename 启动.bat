@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
:: 确保使用系统 Python（优先于 WorkBuddy Python）
set "PATH=%LOCALAPPDATA%\Programs\Python\Python313;%PATH%"
set USERNAME=User

echo ============================================================
echo          RAG-Skeleton 知识库 - 一键启动
echo ============================================================
echo.

:: ---- 第1步：检测 Ollama ----
echo [1/4] 检测 Ollama 服务...
curl -s --connect-timeout 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo       Ollama 未运行，尝试启动...
    start "" ollama serve
    echo       等待 Ollama 启动（15秒）...
    timeout /t 15 /nobreak >nul
    :: 再次检测
    curl -s --connect-timeout 3 http://127.0.0.1:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo       [警告] Ollama 启动失败，将使用云端 API 模式
        set USE_OLLAMA=false
        goto :start_backend
    )
)
echo       [OK] Ollama 已就绪
set USE_OLLAMA=true

:: 检测 qwen2.5:7b 模型
curl -s http://127.0.0.1:11434/api/tags 2>&1 | findstr "qwen2.5:7b" >nul 2>&1
if %errorlevel% neq 0 (
    echo       [警告] 未找到 qwen2.5:7b 模型，正在拉取（首次约5分钟）...
    ollama pull qwen2.5:7b
)
echo       [OK] qwen2.5:7b 模型可用

:start_backend
:: ---- 第2步：启动 RAG 后端 ----
echo.
echo [2/4] 启动 RAG 后端服务...
:: 先检查 8000 端口是否被占用
curl -s --connect-timeout 2 http://127.0.0.1:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo       [警告] 8000 端口已被占用，跳过后端启动
    goto :start_frontend
)

:: 用单独的 bat 文件启动后端，避免 cmd /c 环境变量丢失问题
echo set USE_OLLAMA=!USE_OLLAMA!> _start_backend_tmp.bat
echo set USERNAME=User>> _start_backend_tmp.bat
echo python "%~dp0server.py">> _start_backend_tmp.bat
start "RAG-Backend" /min cmd /c _start_backend_tmp.bat
del _start_backend_tmp.bat
echo       后端启动中，等待初始化（约60秒）...

:: ---- 第3步：等待后端就绪 ----
echo.
echo [3/4] 等待后端就绪...
set READY=0
for /l %%i in (1,1,30) do (
    if !READY!==0 (
        timeout /t 3 /nobreak >nul
        curl -s --connect-timeout 2 http://127.0.0.1:8000/health >nul 2>&1
        if !errorlevel! equ 0 (
            echo       [OK] 后端服务已就绪
            set READY=1
        )
    )
)
if !READY!==0 (
    echo       [警告] 后端启动超时，前端可能无法正常使用
    echo       请检查终端窗口 "RAG-Backend" 中的错误信息
)

:: ---- 第4步：启动前端 ----
:start_frontend
echo.
echo [4/4] 启动 Streamlit 前端...
echo.
echo ============================================================
if "!USE_OLLAMA!"=="true" (
    echo   模式：Ollama 离线 (qwen2.5:7b)
) else (
    echo   模式：DeepSeek 云端 (deepseek-chat)
)
echo   后端：http://localhost:8000
echo   前端：http://localhost:8501
echo ============================================================
echo.
python -m streamlit run "%~dp0web.py" --server.port 8501
pause
