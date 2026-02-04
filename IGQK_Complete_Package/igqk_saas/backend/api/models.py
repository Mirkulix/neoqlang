"""
Models API endpoints - Model Registry
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.huggingface_service import HuggingFaceService

router = APIRouter()

# Initialize HuggingFace service
hf_service = HuggingFaceService()


class Model(BaseModel):
    """Model registry entry"""
    id: str
    name: str
    description: Optional[str]
    architecture: str
    size_mb: float
    accuracy: Optional[float]
    task: str
    created_at: str
    source: str  # trained, compressed, huggingface


@router.get("/my-models", response_model=List[Model])
async def list_user_models(limit: int = 10):
    """List models owned by current user"""

    # Mock data
    return [
        Model(
            id="model_1",
            name="My CIFAR-10 Model",
            description="Custom trained on CIFAR-10",
            architecture="resnet18",
            size_mb=42.0,
            accuracy=92.5,
            task="classification",
            created_at="2026-02-04",
            source="trained"
        ),
        Model(
            id="model_2",
            name="CIFAR-10 Compressed",
            description="Compressed version with IGQK",
            architecture="resnet18",
            size_mb=2.6,
            accuracy=91.8,
            task="classification",
            created_at="2026-02-04",
            source="compressed"
        )
    ]


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed model information"""

    return {
        "id": model_id,
        "name": "My CIFAR-10 Model",
        "architecture": "resnet18",
        "size_mb": 42.0,
        "accuracy": 92.5,
        "parameters": 11_000_000,
        "download_url": f"/api/models/download/{model_id}"
    }


@router.get("/search/huggingface")
async def search_huggingface_models(query: str, limit: int = 10):
    """Search models on HuggingFace Hub"""
    try:
        # Use real HuggingFace search
        results = hf_service.search_models(query=query, limit=limit)

        # Transform results to match expected format
        return [
            {
                "id": model["id"],
                "name": model["name"],
                "size_mb": 0,  # Size not available from search
                "downloads": model["downloads"] or 0,
                "task": model["task"]
            }
            for model in results
        ]
    except Exception as e:
        # Fallback to mock data if search fails
        print(f"⚠️ HuggingFace search failed: {e}")
        return [
            {
                "id": "bert-base-uncased",
                "name": "bert-base-uncased",
                "size_mb": 0,
                "downloads": 100_000_000,
                "task": "fill-mask"
            }
        ]
