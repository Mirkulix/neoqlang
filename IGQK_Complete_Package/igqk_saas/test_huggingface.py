"""
Test HuggingFace Download direkt (ohne UI)
"""

import sys
import os

# Fix Windows encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'igqk'))

print("="*70)
print("🤗 HuggingFace Download Test")
print("="*70)
print()

# Test 1: Import Services
print("Test 1: Importing services...")
try:
    from services.huggingface_service import HuggingFaceService
    from services.compression_service import CompressionService
    print("✅ Services imported successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Create service instances
print("Test 2: Creating service instances...")
try:
    hf_service = HuggingFaceService()
    comp_service = CompressionService()
    print("✅ Services created successfully!")
except Exception as e:
    print(f"❌ Service creation failed: {e}")
    sys.exit(1)

print()

# Test 3: Download a small model
print("Test 3: Downloading model from HuggingFace...")
print("Model: distilbert-base-uncased (small, ~268 MB)")
print("This will take 1-2 minutes...")
print()

try:
    result = hf_service.download_model("distilbert-base-uncased")
    print("✅ Model downloaded successfully!")
    print()
    print("Model Info:")
    print(f"  - Identifier: {result['metadata']['identifier']}")
    print(f"  - Parameters: {result['metadata']['parameters']:,}")
    print(f"  - Size: {result['metadata']['size_mb']:.2f} MB")
    print(f"  - Type: {result['metadata']['model_type']}")
    print()
except Exception as e:
    print(f"❌ Download failed: {e}")
    print()
    print("Mögliche Gründe:")
    print("  - Keine Internetverbindung")
    print("  - HuggingFace Hub nicht erreichbar")
    print("  - transformers library nicht installiert")
    sys.exit(1)

# Test 4: Compress the model
print("Test 4: Compressing model with IGQK...")
print("Method: Ternary (16× compression)")
print("This will take ~1 minute...")
print()

try:
    comp_result = comp_service.compress_huggingface_model(
        model_identifier="distilbert-base-uncased",
        method="ternary",
        job_id="test_001"
    )

    if comp_result["status"] == "completed":
        print("✅ Compression completed successfully!")
        print()
        print("Results:")
        print(f"  - Original Size: {comp_result['original']['size_mb']:.2f} MB")
        print(f"  - Compressed Size: {comp_result['compressed']['size_mb']:.2f} MB")
        print(f"  - Compression Ratio: {comp_result['comparison']['compression_ratio']:.1f}×")
        print(f"  - Memory Saved: {comp_result['comparison']['memory_saved_percent']:.1f}%")
        print(f"  - Saved to: {comp_result.get('save_path', 'N/A')}")
        print()
    else:
        print(f"❌ Compression failed: {comp_result.get('error', 'Unknown error')}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Compression failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("="*70)
print("🎉 ALL TESTS PASSED!")
print("="*70)
print()
print("HuggingFace Integration funktioniert perfekt!")
print()
print("Sie können jetzt:")
print("  1. Web-UI öffnen: http://localhost:7860")
print("  2. Tab '🗜️ COMPRESS Mode' wählen")
print("  3. Model Source: HuggingFace Hub")
print("  4. Model: distilbert-base-uncased")
print("  5. Start Compression klicken")
print()
print("="*70)
