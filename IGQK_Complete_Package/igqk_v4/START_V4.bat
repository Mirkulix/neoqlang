@echo off
chcp 65001 >nul
cls

echo ================================================================================
echo 🚀 IGQK v4.0 - UNIFIED QUANTUM-CLASSICAL HYBRID AI PLATFORM
echo ================================================================================
echo.
echo Starting interactive menu...
echo.

python START_V4.py

if errorlevel 1 (
    echo.
    echo ❌ Error running START_V4.py
    echo.
    echo Possible solutions:
    echo   1. Install Python: python --version
    echo   2. Install dependencies: pip install -r requirements.txt
    echo   3. Check Python path
    echo.
    pause
)
