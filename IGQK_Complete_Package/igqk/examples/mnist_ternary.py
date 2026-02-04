"""
MNIST Classification with IGQK Ternary Compression

Demonstrates training a simple neural network on MNIST using the IGQK optimizer
and compressing to ternary weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from igqk import IGQKOptimizer, TernaryProjector, measure_compression


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST."""
    
    def __init__(self, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_epoch(model, optimizer, train_loader, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, device):
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


def main():
    # Hyperparameters
    batch_size = 128
    epochs = 10
    lr = 0.01
    hbar = 0.1  # Quantum uncertainty
    gamma = 0.01  # Damping
    hidden_size = 128
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SimpleMLP(hidden_size=hidden_size).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = IGQKOptimizer(
        model.parameters(),
        lr=lr,
        hbar=hbar,
        gamma=gamma,
        use_quantum=True,
        projector=TernaryProjector(method='optimal')
    )
    
    print(f"\nIGQK Optimizer:")
    print(f"  Learning rate: {lr}")
    print(f"  Quantum uncertainty (hbar): {hbar}")
    print(f"  Damping (gamma): {gamma}")
    
    # Training
    print("\n" + "="*50)
    print("Training with IGQK")
    print("="*50)
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        # Quantum metrics
        entropy = optimizer.entropy()
        purity = optimizer.purity()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"Quantum Entropy: {entropy:.4f}, Purity: {purity:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'mnist_igqk_best.pth')
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    # Compression
    print("\n" + "="*50)
    print("Compressing to Ternary Weights")
    print("="*50)
    
    # Save original model
    original_model = SimpleMLP(hidden_size=hidden_size).to(device)
    original_model.load_state_dict(model.state_dict())
    
    # Compress
    optimizer.compress(model)
    
    # Evaluate compressed model
    compressed_acc = evaluate(model, test_loader, device)
    print(f"\nCompressed Model Accuracy: {compressed_acc:.2f}%")
    print(f"Accuracy Drop: {best_acc - compressed_acc:.2f}%")
    
    # Measure compression
    stats = measure_compression(original_model, model)
    print(f"\nCompression Statistics:")
    print(f"  Original Memory: {stats['original_memory_mb']:.2f} MB")
    print(f"  Compressed Memory: {stats['compressed_memory_mb']:.2f} MB")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}×")
    print(f"  L2 Distortion: {stats['distortion']:.4f}")
    
    # Save compressed model
    torch.save(model.state_dict(), 'mnist_igqk_compressed.pth')
    print("\nSaved compressed model to 'mnist_igqk_compressed.pth'")


if __name__ == '__main__':
    main()
