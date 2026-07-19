@echo off
REM Auto-start launcher for Windows (Task Scheduler / Startup folder)
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)

REM Prefer watchdog so crashes auto-restart; continues from DB + saved brain
"%PY%" main.py --watchdog
