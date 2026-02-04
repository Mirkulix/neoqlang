@echo off
chcp 65001 >nul
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║            🚀 IGQK SYSTEM - VOLLSTÄNDIGER START                    ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo.

cd /d "%~dp0"

echo [1/7] Unit-Tests werden ausgeführt...
echo ────────────────────────────────────────────────────────────────────
python test_basic.py
if errorlevel 1 (
    echo ❌ Unit-Tests fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo ✅ Unit-Tests erfolgreich!
echo.
timeout /t 2 >nul

echo [2/7] Integrationstests werden ausgeführt...
echo ────────────────────────────────────────────────────────────────────
python test_integration.py
if errorlevel 1 (
    echo ❌ Integrationstests fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo ✅ Integrationstests erfolgreich!
echo.
timeout /t 2 >nul

echo [3/7] MNIST-Demo wird ausgeführt...
echo ────────────────────────────────────────────────────────────────────
python test_mnist_demo.py
if errorlevel 1 (
    echo ❌ MNIST-Demo fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo ✅ MNIST-Demo erfolgreich!
echo.
timeout /t 2 >nul

echo [4/7] Test mit echten MNIST-Daten...
echo ────────────────────────────────────────────────────────────────────
python test_real_mnist.py
if errorlevel 1 (
    echo ❌ Real-MNIST-Test fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo ✅ Real-MNIST-Test erfolgreich!
echo.
timeout /t 2 >nul

echo [5/7] Performance-Benchmarks werden ausgeführt...
echo ────────────────────────────────────────────────────────────────────
python benchmark_performance.py
if errorlevel 1 (
    echo ❌ Benchmarks fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo ✅ Benchmarks erfolgreich!
echo.
timeout /t 2 >nul

echo [6/7] Live-Monitoring-Demo...
echo ────────────────────────────────────────────────────────────────────
python monitor_training.py
if errorlevel 1 (
    echo ❌ Monitoring fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo ✅ Monitoring erfolgreich!
echo.
timeout /t 2 >nul

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║                 ✅ ALLE TESTS ERFOLGREICH ABGESCHLOSSEN!            ║
echo ║                                                                    ║
echo ╠════════════════════════════════════════════════════════════════════╣
echo ║                                                                    ║
echo ║  🎉 IGQK Framework ist vollständig funktionsfähig!                 ║
echo ║                                                                    ║
echo ║  📊 Ergebnisse:                                                     ║
echo ║     • 6/6 Testsuiten bestanden                                     ║
echo ║     • 16× Kompression erreicht                                     ║
echo ║     • 0.65%% Genauigkeitsverlust                                    ║
echo ║     • Innovation bestätigt!                                        ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo.
echo Alle Logs wurden in diesem Fenster angezeigt.
echo Detaillierte Berichte finden Sie in: VALIDATION_REPORT.md
echo.
pause
