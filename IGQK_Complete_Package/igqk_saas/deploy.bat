@echo off
chcp 65001 >nul
cls

echo ==========================================
echo  IGQK v3.0 - Production Deployment
echo ==========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Check if .env exists
if not exist ".env" (
    echo [WARNING] .env file not found. Creating from .env.example...
    copy .env.example .env >nul
    echo [WARNING] Please edit .env file with your configuration!
    echo.
    echo Edit .env file and run this script again.
    pause
    exit /b 1
)

echo [OK] .env file found
echo.

REM Stop existing containers
echo Stopping existing containers...
docker-compose down >nul 2>&1
echo [OK] Old containers stopped
echo.

REM Build images
echo Building Docker images...
echo This may take 5-10 minutes on first run...
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [OK] Images built successfully
echo.

REM Start services
echo Starting services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services!
    pause
    exit /b 1
)

echo [OK] Services started
echo.

REM Wait for services
echo Waiting for services to initialize...
timeout /t 15 /nobreak >nul

REM Check backend
echo Checking backend health...
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Backend is healthy
) else (
    echo [WARNING] Backend might still be starting...
)

REM Check frontend
echo Checking frontend...
curl -s http://localhost:7860 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Frontend is healthy
) else (
    echo [WARNING] Frontend might still be starting...
)

echo.
echo ==========================================
echo  DEPLOYMENT COMPLETE!
echo ==========================================
echo.
echo Services:
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/api/docs
echo   Frontend UI:  http://localhost:7860
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:7860
echo.
echo To view logs:
echo   docker-compose logs -f
echo.
echo To stop services:
echo   docker-compose down
echo.
echo [SUCCESS] IGQK SaaS Platform is now running!
echo.
pause
