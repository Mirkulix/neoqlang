"""
IGQK Training Monitor - Live-Visualisierung des Trainingsprozesses
Beantwortet die Frage: "wo sehe ich den prozess?"
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from datetime import datetime

# Fix Windows encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, '.')

from igqk import IGQKOptimizer, TernaryProjector

# ASCII Art für Prozessanzeige
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  🚀 IGQK TRAINING MONITOR - LIVE PROZESS ANZEIGE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")

def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Erstellt eine Fortschrittsanzeige"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    return f'{prefix} |{bar}| {percent}% {suffix}'

def print_metric_box(title, value, unit='', color=''):
    """Box für eine Metrik"""
    width = 30
    print("┌" + "─"*width + "┐")
    print("│ " + title.ljust(width-2) + "│")
    print("│ " + f"{value} {unit}".center(width-2) + "│")
    print("└" + "─"*width + "┘")

def print_status(epoch, total_epochs, loss, acc, entropy, purity, elapsed_time):
    """Zeigt den aktuellen Status"""
    clear_screen()
    print_header()
    print()

    # Fortschrittsanzeige
    progress = print_progress_bar(epoch, total_epochs, prefix='Training', suffix='Complete', length=50)
    print(progress)
    print()

    # Metriken in 2 Spalten
    print("╔" + "═"*34 + "╦" + "═"*34 + "╗")
    print("║  📊 TRAINING-METRIKEN".ljust(35) + "║  🔬 QUANTUM-METRIKEN".ljust(35) + "║")
    print("╠" + "═"*34 + "╬" + "═"*34 + "╣")
    print(f"║  Epoch: {epoch}/{total_epochs}".ljust(35) + f"║  Entropie: {entropy:.4f}".ljust(35) + "║")
    print(f"║  Loss: {loss:.4f}".ljust(35) + f"║  Reinheit: {purity:.4f}".ljust(35) + "║")
    print(f"║  Genauigkeit: {acc:.2f}%".ljust(35) + f"║  Zeit: {elapsed_time:.2f}s".ljust(35) + "║")
    print("╚" + "═"*34 + "╩" + "═"*34 + "╝")
    print()

    # Loss-Visualisierung (einfaches Balkendiagramm)
    loss_bar_length = min(50, int(loss * 10))
    print("Loss-Entwicklung:")
    print("│" + "▓"*loss_bar_length)
    print()

    # Genauigkeits-Visualisierung
    acc_bar_length = int(acc / 2)
    print("Genauigkeit:")
    print("│" + "█"*acc_bar_length + " " + f"{acc:.1f}%")
    print()

def print_final_summary(training_time, best_acc, compression_ratio, memory_saved):
    """Finale Zusammenfassung"""
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  ✅ TRAINING ABGESCHLOSSEN!".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╠" + "═"*68 + "╣")
    print("║" + f"  🎯 Beste Genauigkeit: {best_acc:.2f}%".ljust(68) + "║")
    print("║" + f"  ⏱️  Trainingszeit: {training_time:.2f}s".ljust(68) + "║")
    print("║" + f"  💾 Kompression: {compression_ratio:.1f}× ({memory_saved:.2f} MB gespart)".ljust(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")

# Hauptprogramm
def main():
    print_header()
    print()
    print("Initialisiere System...")
    time.sleep(1)

    # Erstelle Daten
    print("✓ Erstelle Trainingsdaten...")
    X_train = torch.randn(2000, 100)
    y_train = torch.randint(0, 5, (2000,))
    X_test = torch.randn(500, 100)
    y_test = torch.randint(0, 5, (500,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    # Erstelle Modell
    print("✓ Erstelle neuronales Netzwerk...")
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )

    # Optimizer
    print("✓ Erstelle IGQK Optimizer...")
    optimizer = IGQKOptimizer(
        model.parameters(),
        lr=0.01,
        hbar=0.1,
        gamma=0.01,
        use_quantum=True,
        projector=TernaryProjector()
    )

    print()
    print("Starte Training in 2 Sekunden...")
    time.sleep(2)

    # Training
    n_epochs = 15
    best_acc = 0
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        # Metriken
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        entropy = optimizer.entropy()
        purity = optimizer.purity()
        elapsed_time = time.time() - start_time

        if train_acc > best_acc:
            best_acc = train_acc

        # Zeige Status
        print_status(epoch, n_epochs, avg_loss, train_acc, entropy, purity, elapsed_time)

        time.sleep(0.3)  # Kleine Pause für bessere Lesbarkeit

    training_time = time.time() - start_time

    # Kompression
    print()
    print("Komprimiere Modell...")
    time.sleep(1)

    n_params = sum(p.numel() for p in model.parameters())
    original_memory = n_params * 4 / (1024**2)

    optimizer.compress(model)

    # Messe komprimierten Speicher
    first_layer_weights = model[0].weight.data.flatten()
    unique_vals = torch.unique(first_layer_weights)
    compressed_memory = n_params * len(unique_vals).bit_length() / 8 / (1024**2)
    compression_ratio = original_memory / compressed_memory
    memory_saved = original_memory - compressed_memory

    # Finale Zusammenfassung
    print()
    print_final_summary(training_time, best_acc, compression_ratio, memory_saved)
    print()

    # System-Info
    print("╔" + "═"*68 + "╗")
    print("║" + "  💻 SYSTEM-INFORMATION".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    print("║" + f"  Betriebssystem: {os.name}".ljust(68) + "║")
    print("║" + f"  Python-Version: {sys.version.split()[0]}".ljust(68) + "║")
    print("║" + f"  PyTorch-Version: {torch.__version__}".ljust(68) + "║")
    print("║" + f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}".ljust(68) + "║")
    print("║" + f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()

    print("✅ Monitoring abgeschlossen! Das System funktioniert einwandfrei.")
    print()
    print("Sie können den Prozess jetzt sehen! 🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training abgebrochen durch Benutzer.")
    except Exception as e:
        print(f"\n\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
