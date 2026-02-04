@echo off
chcp 65001 >nul
cls

echo ==========================================
echo  IGQK SaaS Platform - System Status
echo ==========================================
echo.

REM Check processes
echo Checking processes...
echo.

set BACKEND_RUNNING=0
set FRONTEND_RUNNING=0

REM Check if port 8000 is in use (Backend)
netstat -ano | findstr ":8000.*LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Backend API is RUNNING on port 8000
    set BACKEND_RUNNING=1
) else (
    echo [ERROR] Backend API is NOT running on port 8000
)

REM Check if port 7860 is in use (Frontend)
netstat -ano | findstr ":7860.*LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Frontend UI is RUNNING on port 7860
    set FRONTEND_RUNNING=1
) else (
    echo [ERROR] Frontend UI is NOT running on port 7860
)

echo.
echo ==========================================

if %BACKEND_RUNNING% equ 1 (
    echo.
    echo Testing Backend Health...
    curl -s http://localhost:8000/api/health
    echo.
    echo.
    echo Backend Statistics:
    curl -s http://localhost:8000/api/stats
    echo.
)

echo.
echo ==========================================
echo  Process Details
echo ==========================================
echo.
netstat -ano | findstr "8000 7860" | findstr "LISTENING"
echo.

if %BACKEND_RUNNING% equ 1 if %FRONTEND_RUNNING% equ 1 (
    echo ==========================================
    echo  SYSTEM IS FULLY OPERATIONAL!
    echo ==========================================
    echo.
    echo Access Points:
    echo   Frontend UI:  http://localhost:7860
    echo   Backend API:  http://localhost:8000
    echo   API Docs:     http://localhost:8000/api/docs
    echo.
    echo Open in browser?
    choice /C YN /M "Open Frontend UI"
    if errorlevel 2 goto end
    if errorlevel 1 start http://localhost:7860
)

:end
echo.
pause
