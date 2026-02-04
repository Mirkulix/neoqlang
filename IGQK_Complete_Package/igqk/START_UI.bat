@echo off
chcp 65001 >nul
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║            🎨 IGQK WEB DASHBOARD WIRD GESTARTET                    ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo 📊 Moderne Web-UI für IGQK Training und Visualisierung
echo.
echo ⏳ Einen Moment bitte...
echo.

cd /d "%~dp0"

python ui_dashboard.py

pause
