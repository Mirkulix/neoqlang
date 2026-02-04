@echo off
chcp 65001 >nul
echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║            🎯 IGQK - SUPER EINFACHE DEMO                           ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo Diese Demo zeigt Ihnen GENAU was IGQK macht!
echo.
echo Dauer: ~5 Minuten
echo.
echo Drücken Sie eine Taste zum Starten...
pause >nul

cd /d "%~dp0"
python demo_einfach.py

echo.
echo.
pause
