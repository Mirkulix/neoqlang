#!/bin/bash

# IGQK SaaS Platform - Production Deployment Script

set -e

echo "=========================================="
echo " IGQK v3.0 - Production Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Check if .env exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from .env.example..."
    cp .env.example .env
    print_warning "Please edit .env file with your configuration before deploying!"
    exit 1
fi

print_success ".env file found"

# Build Docker images
echo ""
echo "Building Docker images..."
docker-compose build --no-cache

print_success "Docker images built successfully"

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

print_success "Services started"

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Check backend health
BACKEND_HEALTH=$(curl -s http://localhost:8000/api/health || echo "failed")
if [[ $BACKEND_HEALTH == *"healthy"* ]]; then
    print_success "Backend is healthy"
else
    print_error "Backend health check failed"
    docker-compose logs backend
    exit 1
fi

# Check frontend health
FRONTEND_HEALTH=$(curl -s http://localhost:7860 || echo "failed")
if [[ $FRONTEND_HEALTH != "failed" ]]; then
    print_success "Frontend is healthy"
else
    print_error "Frontend health check failed"
    docker-compose logs frontend
    exit 1
fi

# Display status
echo ""
echo "=========================================="
echo " DEPLOYMENT SUCCESSFUL!"
echo "=========================================="
echo ""
echo "Services:"
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/api/docs"
echo "  Frontend UI:  http://localhost:7860"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To restart services:"
echo "  docker-compose restart"
echo ""
print_success "IGQK SaaS Platform is now running!"
