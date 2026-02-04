"""
API routes package
"""

from .auth import router as auth_router
from .datasets import router as datasets_router
from .training import router as training_router
from .compression import router as compression_router
from .models import router as models_router
from .deployment import router as deployment_router

__all__ = [
    'auth_router',
    'datasets_router',
    'training_router',
    'compression_router',
    'models_router',
    'deployment_router'
]
