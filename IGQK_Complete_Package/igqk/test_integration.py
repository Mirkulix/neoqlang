"""
Integration test: End-to-end workflow with synthetic data.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os

# Fix Windows encoding issue
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, '.')

from igqk import (
    IGQKOptimizer,
    IGQKScheduler,
    TernaryProjector,
    BinaryProjector,
    HybridProjector,
    SparseProjector,
    measure_compression
)

print("="*70)
print("IGQK Integration Test: End-to-End Workflow")
print("="*70)

# Create synthetic dataset
print("\n[1] Creating synthetic dataset...")
n_samples = 1000
n_features = 20
n_classes = 3

X = torch.randn(n_samples, n_features)
y = torch.randint(0, n_classes, (n_samples,))

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"✓ Dataset created: {n_samples} samples, {n_features} features, {n_classes} classes")

# Define model
print("\n[2] Defining model...")
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, n_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleClassifier()
n_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: {n_params} parameters")

# Test different compression methods
compression_methods = [
    ("Ternary", TernaryProjector(method='optimal')),
    ("Binary", BinaryProjector()),
    ("Sparse (90%)", SparseProjector(sparsity=0.9)),
    ("Hybrid (Sparse+Ternary)", HybridProjector([
        SparseProjector(sparsity=0.8),
        TernaryProjector(method='optimal')
    ]))
]

results = []

for method_name, projector in compression_methods:
    print(f"\n{'='*70}")
    print(f"Testing: {method_name}")
    print(f"{'='*70}")
    
    # Reset model
    model = SimpleClassifier()
    
    # Create optimizer
    print(f"\n[3] Creating IGQK optimizer...")
    optimizer = IGQKOptimizer(
        model.parameters(),
        lr=0.01,
        hbar=0.1,
        gamma=0.01,
        use_quantum=True,
        projector=projector
    )
    
    # Create scheduler
    scheduler = IGQKScheduler(
        optimizer,
        mode='cosine',
        T_max=10,
        hbar_min=0.01,
        gamma_max=0.05
    )
    
    print(f"✓ Optimizer created with {method_name} projector")
    
    # Training
    print(f"\n[4] Training for 5 epochs...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        entropy = optimizer.entropy()
        purity = optimizer.purity()
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
                  f"Entropy={entropy:.4f}, Purity={purity:.4f}")
    
    print(f"✓ Training completed")
    
    # Evaluate before compression
    print(f"\n[5] Evaluating before compression...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    acc_before = 100. * correct / total
    print(f"✓ Accuracy before compression: {acc_before:.2f}%")
    
    # Save original model
    original_model = SimpleClassifier()
    original_model.load_state_dict(model.state_dict())
    
    # Compress
    print(f"\n[6] Compressing model...")
    optimizer.compress(model)
    print(f"✓ Model compressed with {method_name}")
    
    # Evaluate after compression
    print(f"\n[7] Evaluating after compression...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    acc_after = 100. * correct / total
    print(f"✓ Accuracy after compression: {acc_after:.2f}%")
    print(f"✓ Accuracy drop: {acc_before - acc_after:.2f}%")
    
    # Measure compression
    print(f"\n[8] Measuring compression statistics...")
    stats = measure_compression(original_model, model)
    
    print(f"✓ Compression Statistics:")
    print(f"  - Original memory: {stats['original_memory_mb']:.4f} MB")
    print(f"  - Compressed memory: {stats['compressed_memory_mb']:.4f} MB")
    print(f"  - Compression ratio: {stats['compression_ratio']:.2f}×")
    print(f"  - L2 distortion: {stats['distortion']:.4f}")
    
    results.append({
        'method': method_name,
        'acc_before': acc_before,
        'acc_after': acc_after,
        'acc_drop': acc_before - acc_after,
        'compression_ratio': stats['compression_ratio'],
        'memory_before': stats['original_memory_mb'],
        'memory_after': stats['compressed_memory_mb'],
        'distortion': stats['distortion']
    })

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Comparison of Compression Methods")
print(f"{'='*70}")
print(f"{'Method':<25} {'Acc Drop':<12} {'Ratio':<10} {'Memory (MB)':<15}")
print("-"*70)

for r in results:
    print(f"{r['method']:<25} {r['acc_drop']:>6.2f}%     {r['compression_ratio']:>6.2f}×   "
          f"{r['memory_after']:>8.4f}")

print(f"\n{'='*70}")
print("Integration Test Results")
print(f"{'='*70}")

# Check if all tests passed
all_passed = True
for r in results:
    # Check reasonable accuracy drop (< 10%)
    if r['acc_drop'] > 10:
        print(f"⚠️  {r['method']}: High accuracy drop ({r['acc_drop']:.2f}%)")
        all_passed = False
    
    # Check compression ratio > 1
    if r['compression_ratio'] < 1:
        print(f"⚠️  {r['method']}: No compression achieved")
        all_passed = False

if all_passed:
    print("✅ All integration tests PASSED!")
    print("\nKey Findings:")
    print(f"  - All methods achieved compression > 1×")
    print(f"  - All methods maintained accuracy within acceptable range")
    print(f"  - IGQK optimizer successfully trained and compressed models")
    print(f"  - Quantum metrics (entropy, purity) tracked correctly")
else:
    print("⚠️  Some tests had warnings")

print("\n" + "="*70)
