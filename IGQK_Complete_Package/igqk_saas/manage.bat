@echo off
chcp 65001 >nul

:menu
cls
echo ==========================================
echo  IGQK SaaS Platform - Management
echo ==========================================
echo.
echo 1. Start System (Production)
echo 2. Stop System
echo 3. Restart System
echo 4. View Logs (All)
echo 5. View Backend Logs
echo 6. View Frontend Logs
echo 7. System Status
echo 8. Build Images
echo 9. Clean All (Reset)
echo 0. Exit
echo.
set /p choice="Select option: "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto logs_all
if "%choice%"=="5" goto logs_backend
if "%choice%"=="6" goto logs_frontend
if "%choice%"=="7" goto status
if "%choice%"=="8" goto build
if "%choice%"=="9" goto clean
if "%choice%"=="0" goto end
goto menu

:start
echo.
echo Starting IGQK SaaS Platform...
docker-compose up -d
echo.
echo Services started! Waiting for health check...
timeout /t 10 /nobreak >nul
docker-compose ps
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:7860
echo.
pause
goto menu

:stop
echo.
echo Stopping IGQK SaaS Platform...
docker-compose down
echo.
echo Services stopped!
pause
goto menu

:restart
echo.
echo Restarting IGQK SaaS Platform...
docker-compose restart
echo.
echo Services restarted!
pause
goto menu

:logs_all
echo.
echo Showing all logs (Ctrl+C to exit)...
docker-compose logs -f
goto menu

:logs_backend
echo.
echo Showing backend logs (Ctrl+C to exit)...
docker-compose logs -f backend
goto menu

:logs_frontend
echo.
echo Showing frontend logs (Ctrl+C to exit)...
docker-compose logs -f frontend
goto menu

:status
echo.
echo ==========================================
echo  System Status
echo ==========================================
echo.
docker-compose ps
echo.
echo Health Checks:
echo.
curl -s http://localhost:8000/api/health
echo.
echo.
pause
goto menu

:build
echo.
echo Building Docker images...
docker-compose build --no-cache
echo.
echo Build complete!
pause
goto menu

:clean
echo.
echo [WARNING] This will remove ALL containers, images, and volumes!
set /p confirm="Are you sure? (yes/no): "
if not "%confirm%"=="yes" goto menu
echo.
echo Cleaning up...
docker-compose down -v
docker system prune -af
echo.
echo Cleanup complete!
pause
goto menu

:end
echo.
echo Goodbye!
exit
