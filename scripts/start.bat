@echo off
rem The Windows one command: double-click this or run scripts\start.bat.
rem Forwards to start.ps1 with an execution-policy bypass for this process only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
