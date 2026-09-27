@echo off
rem SharpFrame. Double-click: the browser page. Drop a video onto it: type the times here.
rem Also: SharpFrame.bat job.json | SharpFrame.bat video.mp4 0:12 1:03.2 [--soft ...]
rem Python: the portable one beside this file, else the project's .venv, else the global one.
setlocal
chcp 65001 >nul
set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"
if not exist "%PY%" set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if "%~1"=="" goto web
if /i "%~x1"==".json" goto run
if not "%~2"=="" goto run

echo %~nx1
echo.
set "T="
set /p "T=Моменты через пробел (например 12.5 1:03.2 0:01:03.2): "
if not defined T goto end
"%PY%" "%HERE%sharpframe.py" "%~1" %T% --open-folder
goto end

:web
title SharpFrame
"%PY%" "%HERE%web.py"
goto end

:run
"%PY%" "%HERE%sharpframe.py" %*

:end
echo.
pause
