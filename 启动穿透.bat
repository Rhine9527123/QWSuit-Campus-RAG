@echo off
title Cloudflare Tunnel - www.qwsuit.site
echo ========================================
echo  Cloudflare Tunnel 启动中...
echo  公网地址: https://www.qwsuit.site
echo  公网地址: https://qwsuit.site
echo  请勿关闭此窗口
echo ========================================
echo.
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run finance-rag
pause
