"""
Compression Benchmark: Compare different compression methods.

Compares IGQK with standard compression techniques on a simple task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from igqk import (
    IGQKOptimizer,
    TernaryProjector,
    BinaryProjector,
    SparseProjector,
    LowRankProjector,
    HybridProjector,
    measure_compression
)


class SimpleNet(nn.Module):
    """Simple CNN for benchmarking."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, optimizer, train_loader, device, epochs=5):
    """Train model."""
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100. * correct / total


def benchmark_compression(projector_name, projector, device):
    """Benchmark a compression method."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {projector_name}")
    print(f"{'='*60}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model
    model = SimpleNet().to(device)
    
    # Train
    print("Training...")
    start_time = time.time()
    
    if projector_name == "IGQK":
        optimizer = IGQKOptimizer(
            model.parameters(),
            lr=0.01,
            hbar=0.1,
            gamma=0.01,
            projector=projector
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, optimizer, train_loader, device, epochs=3)
    train_time = time.time() - start_time
    
    # Evaluate before compression
    acc_before = evaluate_model(model, test_loader, device)
    
    # Save original
    original_model = SimpleNet().to(device)
    original_model.load_state_dict(model.state_dict())
    
    # Compress
    print("Compressing...")
    start_time = time.time()
    
    if projector_name == "IGQK":
        optimizer.compress(model)
    else:
        with torch.no_grad():
            for param in model.parameters():
                param.data = projector.project(param.data)
    
    compress_time = time.time() - start_time
    
    # Evaluate after compression
    acc_after = evaluate_model(model, test_loader, device)
    
    # Measure compression
    stats = measure_compression(original_model, model)
    
    # Inference speed
    print("Measuring inference speed...")
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure
    start_time = time.time()
    n_inferences = 1000
    with torch.no_grad():
        for _ in range(n_inferences):
            _ = model(dummy_input)
    inference_time = (time.time() - start_time) / n_inferences * 1000  # ms
    
    # Results
    print(f"\nResults:")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Compression Time: {compress_time:.4f}s")
    print(f"  Accuracy Before: {acc_before:.2f}%")
    print(f"  Accuracy After: {acc_after:.2f}%")
    print(f"  Accuracy Drop: {acc_before - acc_after:.2f}%")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}×")
    print(f"  Memory (Before): {stats['original_memory_mb']:.2f} MB")
    print(f"  Memory (After): {stats['compressed_memory_mb']:.2f} MB")
    print(f"  Inference Time: {inference_time:.4f} ms")
    
    return {
        'name': projector_name,
        'train_time': train_time,
        'compress_time': compress_time,
        'acc_before': acc_before,
        'acc_after': acc_after,
        'acc_drop': acc_before - acc_after,
        'compression_ratio': stats['compression_ratio'],
        'memory_before': stats['original_memory_mb'],
        'memory_after': stats['compressed_memory_mb'],
        'inference_time': inference_time,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define compression methods to benchmark
    methods = [
        ("Ternary", TernaryProjector(method='optimal')),
        ("Binary", BinaryProjector()),
        ("Sparse (90%)", SparseProjector(sparsity=0.9)),
        ("Low-Rank (50%)", LowRankProjector(rank_ratio=0.5)),
        ("IGQK", TernaryProjector(method='optimal')),
    ]
    
    # Run benchmarks
    results = []
    for name, projector in methods:
        result = benchmark_compression(name, projector, device)
        results.append(result)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Acc Drop':<12} {'Ratio':<10} {'Memory':<15} {'Inference':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<20} {r['acc_drop']:>6.2f}%     {r['compression_ratio']:>6.2f}×   "
              f"{r['memory_after']:>6.2f} MB      {r['inference_time']:>8.4f} ms")
    
    print("\nBest compression ratio:", max(results, key=lambda x: x['compression_ratio'])['name'])
    print("Best accuracy:", min(results, key=lambda x: x['acc_drop'])['name'])
    print("Fastest inference:", min(results, key=lambda x: x['inference_time'])['name'])


if __name__ == '__main__':
    main()
