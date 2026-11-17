#!/usr/bin/env python3
"""
Import tests in project venv.
"""

import sys

print("="*80)
print("VENV IMPORT VERIFICATION")
print("="*80)

print("\n[ENVIRONMENT]")
print("Python:", sys.version)
print("Executable:", sys.executable)

tests = [
    ("from langchain.storage import InMemoryStore, LocalFileStore", "langchain.storage (both)"),
    ("from langchain_classic.storage import InMemoryStore, LocalFileStore", "langchain_classic.storage (both)"),
    ("from langchain_classic.storage import LocalFileStore", "langchain_classic.storage (LocalFileStore)"),
    ("from langchain_core.stores import InMemoryStore", "langchain_core.stores (InMemoryStore)"),
    ("from docling.datamodel.base_models import InputFormat", "docling.datamodel (InputFormat)"),
]

print("\n[IMPORT TESTS]")
print("-"*80)

for import_stmt, description in tests:
    print("\nTest: {}".format(description))
    print("Statement: {}".format(import_stmt))
    
    try:
        exec(import_stmt)
        print("Result: SUCCESS")
    except Exception as e:
        error_msg = str(e)
        if "No module named" in error_msg:
            module = error_msg.split("'")[1]
            print("Result: FAILED - Module not found: {}".format(module))
        else:
            print("Result: FAILED - {}".format(error_msg[:80]))

# List installed packages
print("\n[INSTALLED PACKAGES]")
print("-"*80)

import subprocess

packages = [
    'langchain',
    'langchain-core',
    'langchain-community',
    'langchain-classic',
    'langchain-qdrant',
    'langchain-text-splitters',
    'docling'
]

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
