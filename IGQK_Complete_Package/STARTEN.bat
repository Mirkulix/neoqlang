@echo off
chcp 65001 >nul
echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║                  🎯 IGQK - HAUPTMENÜ                               ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo Wählen Sie eine Option:
echo.
echo   [1] 🎬 DEMO starten (automatisch, zeigt was IGQK macht)
echo   [2] 🌐 WEB-UI starten (Dashboard im Browser)
echo   [3] 🧪 ALLE TESTS ausführen (Validierung)
echo   [4] 📊 BENCHMARK ausführen (Performance-Test)
echo   [5] ℹ️  HILFE anzeigen
echo   [6] ❌ BEENDEN
echo.
set /p choice="Ihre Wahl (1-6): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto webui
if "%choice%"=="3" goto tests
if "%choice%"=="4" goto benchmark
if "%choice%"=="5" goto hilfe
if "%choice%"=="6" goto ende

echo Ungültige Eingabe!
pause
goto :eof

:demo
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 🎬 DEMO WIRD GESTARTET...
echo ═══════════════════════════════════════════════════════════════════
echo.
cd /d "%~dp0igqk"
python demo_automatisch.py
echo.
echo.
pause
exit

:webui
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 🌐 WEB-UI WIRD GESTARTET...
echo ═══════════════════════════════════════════════════════════════════
echo.
echo Der Browser öffnet sich automatisch...
echo Drücken Sie STRG+C zum Beenden
echo.
cd /d "%~dp0igqk"
python ui_dashboard.py
pause
exit

:tests
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 🧪 TESTS WERDEN AUSGEFÜHRT...
echo ═══════════════════════════════════════════════════════════════════
echo.
cd /d "%~dp0"
call START_ALL.bat
pause
exit

:benchmark
echo.
echo ═══════════════════════════════════════════════════════════════════
echo 📊 BENCHMARK WIRD AUSGEFÜHRT...
echo ═══════════════════════════════════════════════════════════════════
echo.
cd /d "%~dp0igqk"
python benchmark_performance.py
echo.
pause
exit

:hilfe
echo.
echo ═══════════════════════════════════════════════════════════════════
echo ℹ️  IGQK - HILFE
echo ═══════════════════════════════════════════════════════════════════
echo.
echo WAS IST IGQK?
echo   IGQK ist ein KI-Modell-Kompressor der Ihre Modelle 16× kleiner macht!
echo.
echo WIE FUNKTIONIERT ES?
echo   1. Sie haben ein KI-Modell (z.B. von HuggingFace)
echo   2. IGQK komprimiert es
echo   3. Das Modell ist 16× kleiner bei fast gleicher Qualität
echo.
echo OPTIONEN:
echo.
echo   DEMO (Option 1):
echo     - Zeigt Schritt-für-Schritt was IGQK macht
echo     - Dauer: ~1 Minute
echo     - Keine Installation nötig
echo.
echo   WEB-UI (Option 2):
echo     - Grafische Oberfläche im Browser
echo     - Interaktive Parameter-Einstellungen
echo     - Live-Visualisierung
echo     - Öffnet automatisch http://localhost:7860
echo.
echo   TESTS (Option 3):
echo     - Führt alle Validierungs-Tests aus
echo     - Bestätigt dass alles funktioniert
echo     - Dauer: ~5 Minuten
echo.
echo   BENCHMARK (Option 4):
echo     - Vergleicht IGQK mit klassischen Methoden
echo     - Zeigt Performance-Verbesserungen
echo     - Dauer: ~3 Minuten
echo.
echo DOKUMENTATION:
echo   - EINFACH_ERKLÄRT.md - Einfache Erklärung was IGQK macht
echo   - WAS_SIE_ERHALTEN.md - Komplette Produktbeschreibung
echo   - ANWENDUNGSFÄLLE.md - Praktische Beispiele
echo   - UNTERSTÜTZTE_MODELLE.md - Welche Modelle funktionieren
echo.
echo EIGENES MODELL KOMPRIMIEREN:
echo   1. Modell in PyTorch laden: model = torch.load('mein_modell.pt')
echo   2. IGQK importieren: from igqk import IGQKOptimizer
echo   3. Komprimieren: optimizer = IGQKOptimizer(model.parameters())
echo                    optimizer.compress(model)
echo   4. Speichern: torch.save(model, 'modell_komprimiert.pt')
echo.
echo ═══════════════════════════════════════════════════════════════════
echo.
pause
goto :eof

:ende
echo.
echo Auf Wiedersehen! 👋
echo.
exit
