@echo off
chcp 65001 >nul
echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║            🚀 IGQK v3.0 - SaaS Platform Launcher                   ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo Was möchten Sie starten?
echo.
echo   [1] 🌐 Web-UI (Frontend Only - Empfohlen für Demo!)
echo   [2] 🔧 Backend API (FastAPI Server)
echo   [3] 🚀 Komplett (Backend + Frontend)
echo   [4] 📚 API Dokumentation öffnen
echo   [5] ❌ Beenden
echo.
set /p choice="Ihre Wahl (1-5): "

if "%choice%"=="1" goto webui
if "%choice%"=="2" goto backend
if "%choice%"=="3" goto full
if "%choice%"=="4" goto docs
if "%choice%"=="5" goto ende

echo Ungültige Eingabe!
pause
exit

:webui
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 🌐 STARTE WEB-UI...
echo ═══════════════════════════════════════════════════════════════════
echo.
echo Die Web-UI öffnet sich automatisch im Browser...
echo URL: http://localhost:7860
echo.
echo Drücken Sie STRG+C zum Beenden
echo.
cd /d "%~dp0"
python web_ui.py
pause
exit

:backend
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 🔧 STARTE BACKEND API...
echo ═══════════════════════════════════════════════════════════════════
echo.
echo Backend läuft auf: http://localhost:8000
echo API Docs: http://localhost:8000/api/docs
echo.
echo Drücken Sie STRG+C zum Beenden
echo.
cd /d "%~dp0\backend"
python main.py
pause
exit

:full
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 🚀 STARTE KOMPLETTE PLATFORM...
echo ═══════════════════════════════════════════════════════════════════
echo.
echo Starte Backend in neuem Fenster...
start "IGQK Backend API" cmd /k "cd /d %~dp0\backend && python main.py"
timeout /t 3 /nobreak >nul
echo.
echo Starte Web-UI...
cd /d "%~dp0"
python web_ui.py
pause
exit

:docs
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 📚 ÖFFNE API DOKUMENTATION...
echo ═══════════════════════════════════════════════════════════════════
echo.
echo Starte Backend (falls nicht läuft)...
start "IGQK Backend" cmd /k "cd /d %~dp0\backend && python main.py"
echo.
echo Warte 5 Sekunden...
timeout /t 5 /nobreak >nul
echo.
echo Öffne Browser...
start http://localhost:8000/api/docs
echo.
echo Backend läuft im Hintergrund.
echo Drücken Sie eine Taste zum Beenden...
pause >nul
taskkill /FI "WindowTitle eq IGQK Backend*" /T /F >nul 2>&1
exit

:ende
echo.
echo Auf Wiedersehen! 👋
echo.
exit
