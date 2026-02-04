"""
Datasets API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class Dataset(BaseModel):
    """Dataset model"""
    id: str
    name: str
    description: str
    source: str  # huggingface, kaggle, custom
    size: str
    num_samples: int
    task: str  # classification, detection, etc.
    format: str  # images, text, audio


@router.get("/public", response_model=List[Dataset])
async def list_public_datasets(
    task: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 10
):
    """List public datasets from HuggingFace, Kaggle, etc."""

    # Mock data
    datasets = [
        Dataset(
            id="mnist",
            name="MNIST",
            description="Handwritten digits dataset",
            source="huggingface",
            size="11 MB",
            num_samples=70000,
            task="classification",
            format="images"
        ),
        Dataset(
            id="cifar10",
            name="CIFAR-10",
            description="10-class image classification",
            source="huggingface",
            size="163 MB",
            num_samples=60000,
            task="classification",
            format="images"
        ),
        Dataset(
            id="imdb",
            name="IMDB Reviews",
            description="Sentiment analysis dataset",
            source="huggingface",
            size="80 MB",
            num_samples=50000,
            task="classification",
            format="text"
        )
    ]

    return datasets


@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get detailed dataset information"""

    if dataset_id == "mnist":
        return {
            "id": "mnist",
            "name": "MNIST",
            "description": "Handwritten digits dataset (0-9)",
            "samples": 70000,
            "train_size": 60000,
            "test_size": 10000,
            "image_size": "28x28",
            "channels": 1,
            "classes": 10,
            "download_size": "11 MB"
        }

    raise HTTPException(status_code=404, detail="Dataset not found")
