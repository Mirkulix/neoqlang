@echo off
chcp 65001 >nul
cls
echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║            🔄 IGQK v3.0 - SYSTEM NEUSTART                          ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo Stoppe alte Prozesse...
echo.

REM Beende alle Python-Prozesse für IGQK
taskkill /F /FI "WindowTitle eq *IGQK*" >nul 2>&1
timeout /t 2 /nobreak >nul

echo ✅ Alte Prozesse gestoppt
echo.
echo Starte System neu...
echo.
echo ════════════════════════════════════════════════════════════════════
echo 🌐 WEB-UI wird gestartet...
echo ════════════════════════════════════════════════════════════════════
echo.
echo URL: http://localhost:7860
echo.
echo Die Web-UI öffnet sich automatisch im Browser...
echo.
echo Drücken Sie STRG+C zum Beenden
echo.
echo ════════════════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"
python web_ui.py

pause
