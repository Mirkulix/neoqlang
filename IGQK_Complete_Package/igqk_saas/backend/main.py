"""
IGQK v3.0 SaaS Platform - Main API Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os

# Add parent directory to path for IGQK imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'igqk'))

app = FastAPI(
    title="IGQK SaaS API",
    description="All-in-One ML Platform: Training + Compression + Deployment",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from api import (
    auth_router,
    datasets_router,
    training_router,
    compression_router,
    models_router,
    deployment_router
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(training_router, prefix="/api/training", tags=["Training"])
app.include_router(compression_router, prefix="/api/compression", tags=["Compression"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(deployment_router, prefix="/api/deployment", tags=["Deployment"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IGQK v3.0 SaaS Platform API",
        "version": "3.0.0",
        "status": "running",
        "features": {
            "create_mode": {
                "description": "Train models from scratch",
                "capabilities": [
                    "Dataset management",
                    "Architecture builder",
                    "Quantum-optimized training",
                    "Auto-compression",
                    "Publishing"
                ]
            },
            "compress_mode": {
                "description": "Compress existing models",
                "capabilities": [
                    "16× compression",
                    "Multiple methods",
                    "A/B testing",
                    "Multi-cloud deployment"
                ]
            }
        },
        "docs": "/api/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "services": {
            "api": "running",
            "igqk_core": "available"
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get platform statistics"""
    return {
        "total_users": 0,  # TODO: Get from database
        "total_models": 0,
        "total_trainings": 0,
        "total_compressions": 0,
        "total_deployments": 0,
        "compression_saved_gb": 0.0
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
