#!/usr/bin/env python3
"""
Final import verification test.
"""

import sys

print("="*80)
print("FINAL IMPORT VERIFICATION")
print("="*80)

print("\n[ENVIRONMENT]")
print("Python:", sys.version)
print("Executable:", sys.executable)

# Test 1: LocalFileStore from langchain_classic
print("\n[TEST 1] from langchain_classic.storage import LocalFileStore")
try:
    from langchain_classic.storage import LocalFileStore
    print("  Result: SUCCESS")
    print("  Module:", LocalFileStore.__module__)
    print("  File:", LocalFileStore.__init__.__code__.co_filename if hasattr(LocalFileStore.__init__, '__code__') else 'N/A')
except Exception as e:
    print("  Result: FAILED")
    print("  Error:", str(e))

# Test 2: LocalFileStore from file_system
print("\n[TEST 2] from langchain_classic.storage.file_system import LocalFileStore")
try:
    from langchain_classic.storage.file_system import LocalFileStore as LocalFileStore2
    print("  Result: SUCCESS")
    print("  Module:", LocalFileStore2.__module__)
except Exception as e:
    print("  Result: FAILED")
    print("  Error:", str(e))

# Test 3: InMemoryStore from langchain_core
print("\n[TEST 3] from langchain_core.stores import InMemoryStore")
try:
    from langchain_core.stores import InMemoryStore
    print("  Result: SUCCESS")
    print("  Module:", InMemoryStore.__module__)
except Exception as e:
    print("  Result: FAILED")
    print("  Error:", str(e))

# Test 4: Current broken import
print("\n[TEST 4] from langchain.storage import InMemoryStore, LocalFileStore")
try:
    from langchain.storage import InMemoryStore as IM, LocalFileStore as LFS
    print("  Result: SUCCESS")
    print("  InMemoryStore module:", IM.__module__)
    print("  LocalFileStore module:", LFS.__module__)
except Exception as e:
    print("  Result: FAILED")
    print("  Error:", str(e))

# Test 5: Docling
print("\n[TEST 5] import docling")
try:
    import docling
    print("  Result: SUCCESS")
    print("  Version:", getattr(docling, '__version__', 'unknown'))
    print("  Location:", docling.__file__)
except Exception as e:
    print("  Result: FAILED")
    print("  Error:", str(e))

# Test 6: Docling base models
print("\n[TEST 6] from docling.datamodel.base_models import InputFormat")
try:
    from docling.datamodel.base_models import InputFormat
    print("  Result: SUCCESS")
    print("  Module:", InputFormat.__module__)
except Exception as e:
    print("  Result: FAILED")
    print("  Error:", str(e))

# List package versions
print("\n[INSTALLED PACKAGES]")
import subprocess

packages = ['langchain', 'langchain-core', 'langchain-community', 'langchain-classic', 'docling']

for pkg in packages:
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'show', pkg],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if line.startswith('Version:'):
                version = line.split(':', 1)[1].strip()
                print("  {}: {}".format(pkg, version))
                break
    else:
        print("  {}: NOT INSTALLED".format(pkg))

print("\n" + "="*80)
