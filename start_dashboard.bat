@echo off
title Ariadne Discovery Console
cd /d "%~dp0"
echo Starting the Ariadne Discovery Console...
echo Your browser will open automatically. Keep this window open while you use it.
python scripts\dashboard.py
echo.
echo The console stopped. Press any key to close this window.
pause >nul
