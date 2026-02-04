"""
IGQK System Starter - Einfacher Menü-basierter Start aller Komponenten
"""

import sys
import os
import subprocess

# Fix Windows encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  🚀 IGQK SYSTEM STARTER".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print()

def print_menu():
    print("Wählen Sie eine Option:")
    print()
    print("  1️⃣  Unit-Tests ausführen (test_basic.py)")
    print("  2️⃣  Integrationstests (test_integration.py)")
    print("  3️⃣  MNIST-Demo (test_mnist_demo.py)")
    print("  4️⃣  Echter MNIST-Test (test_real_mnist.py)")
    print("  5️⃣  Live-Monitoring (monitor_training.py)")
    print("  6️⃣  Performance-Benchmarks (benchmark_performance.py)")
    print()
    print("  🔥 A. ALLE TESTS DURCHLAUFEN (empfohlen)")
    print()
    print("  0️⃣  Beenden")
    print()

def run_script(script_name, description):
    print()
    print("="*70)
    print(f"🚀 Starte: {description}")
    print("="*70)
    print()

    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print()
        print("✅ Erfolgreich abgeschlossen!")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Fehler beim Ausführen von {script_name}")
        print(f"   Fehlercode: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Datei nicht gefunden: {script_name}")
        return False

def run_all():
    scripts = [
        ("test_basic.py", "Unit-Tests"),
        ("test_integration.py", "Integrationstests"),
        ("test_mnist_demo.py", "MNIST-Demo"),
        ("test_real_mnist.py", "Echter MNIST-Test"),
        ("benchmark_performance.py", "Performance-Benchmarks"),
        ("monitor_training.py", "Live-Monitoring"),
    ]

    results = []

    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))

        if success:
            print()
            input("Drücken Sie Enter um fortzufahren...")
        else:
            print()
            choice = input("Fehler aufgetreten. Trotzdem fortfahren? (j/n): ")
            if choice.lower() != 'j':
                break

    # Zusammenfassung
    print()
    print("="*70)
    print("📊 ZUSAMMENFASSUNG")
    print("="*70)
    print()

    for description, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {description}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print()
    print(f"Ergebnis: {passed}/{total} Tests bestanden ({passed/total*100:.0f}%)")

    if passed == total:
        print()
        print("🎉 ALLE TESTS ERFOLGREICH! IGQK ist produktionsreif!")

def main():
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("Ihre Wahl: ").strip().upper()

        if choice == "1":
            run_script("test_basic.py", "Unit-Tests")
            input("\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "2":
            run_script("test_integration.py", "Integrationstests")
            input("\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "3":
            run_script("test_mnist_demo.py", "MNIST-Demo")
            input("\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "4":
            run_script("test_real_mnist.py", "Echter MNIST-Test")
            input("\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "5":
            run_script("monitor_training.py", "Live-Monitoring")
            input("\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "6":
            run_script("benchmark_performance.py", "Performance-Benchmarks")
            input("\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "A":
            run_all()
            input("\n\nDrücken Sie Enter um zurückzukehren...")

        elif choice == "0":
            print()
            print("Auf Wiedersehen! 👋")
            break

        else:
            print()
            print("❌ Ungültige Auswahl. Bitte versuchen Sie es erneut.")
            input("Drücken Sie Enter um fortzufahren...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programm abgebrochen durch Benutzer.")
    except Exception as e:
        print(f"\n\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
