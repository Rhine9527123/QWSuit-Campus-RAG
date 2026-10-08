@echo off
chcp 65001 >nul
title 千问深信 · 评测面板
echo ========================================
echo   千问深信 · 评测面板启动中...
echo ========================================
echo.

echo [1/2] 启动面板 API（端口 8002）...
start "评测面板 API" cmd /k "cd /d D:\千问深信 && python -m uvicorn eval.panel_api:app --host 0.0.0.0 --port 8002"

timeout /t 3 /nobreak >nul

echo [2/2] 启动面板前端（端口 8003）...
start "评测面板前端" cmd /k "python -m http.server 8003 --directory D:\千问深信\eval\panel"

timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo   面板访问地址：http://localhost:8003
echo   API 地址：http://localhost:8002
echo ========================================
echo.
echo 如需停止服务，关闭对应 cmd 窗口即可。
echo.

pause
