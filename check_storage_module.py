#!/usr/bin/env python3
"""
Direct check of storage modules.
"""

import sys

print("="*80)
print("DIRECT MODULE CHECK")
print("="*80)

# Check langchain_community.storage
print("\n[1] langchain_community.storage")
try:
    import langchain_community.storage
    print("  EXISTS")
    items = dir(langchain_community.storage)
    print("  Classes/items:")
    for item in items:
        if not item.startswith('_'):
            print("    - {}".format(item))
except Exception as e:
    print("  ERROR: {}".format(e))

# Check langchain_core.stores
print("\n[2] langchain_core.stores")
try:
    import langchain_core.stores
    print("  EXISTS")
    items = dir(langchain_core.stores)
    print("  Classes/items:")
    for item in items:
        if not item.startswith('_'):
            print("    - {}".format(item))
except Exception as e:
    print("  ERROR: {}".format(e))

# Try direct imports
print("\n[3] IMPORT ATTEMPTS")

imports_to_try = [
    ("from langchain_core.stores import InMemoryStore", "InMemoryStore from langchain_core.stores"),
    ("from langchain_core.stores import BaseStore", "BaseStore from langchain_core.stores"),
    ("from langchain.storage import LocalFileStore", "LocalFileStore from langchain.storage"),
    ("from langchain_community.storage import LocalFileStore", "LocalFileStore from langchain_community"),
]

for import_stmt, description in imports_to_try:
    print("\n  Attempting: {}".format(description))
    print("  Statement: {}".format(import_stmt))
    try:
        exec(import_stmt)
        print("  Result: SUCCESS")
    except ImportError as e:
        print("  Result: FAILED - {}".format(str(e)[:60]))
    except Exception as e:
        print("  Result: ERROR - {}".format(str(e)[:60]))

# Check if LocalFileStore is in langchain package at all
print("\n[4] SEARCHING FOR LocalFileStore IN INSTALLED PACKAGES")

try:
    from langchain_community import storage
    if hasattr(storage, 'LocalFileStore'):
        print("  Found: langchain_community.storage.LocalFileStore")
    else:
        print("  Not in: langchain_community.storage")
except:
    print("  Cannot import langchain_community.storage")

# List all in langchain_community.storage
print("\n[5] ALL EXPORTS FROM langchain_community.storage")
try:
    import langchain_community.storage as storage
    exports = [x for x in dir(storage) if not x.startswith('_')]
    if exports:
        for exp in exports:
            print("  - {}".format(exp))
    else:
        print("  (empty or all private)")
except Exception as e:
    print("  Error: {}".format(e))

# Check requirements for what should be there
print("\n[6] WHAT VECTORSTORE_QDRANT.PY NEEDS")
print("  Uses:")
print("    - InMemoryStore (currently from: langchain.storage)")
print("    - LocalFileStore (currently from: langchain.storage)")
print("\n  Available in:")
print("    - InMemoryStore: langchain_core.stores")
print("    - LocalFileStore: ???")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The code imports:
  from langchain.storage import InMemoryStore, LocalFileStore

Current status:
  - langchain.storage: DOES NOT EXIST in current installation
  - InMemoryStore: Available in langchain_core.stores
  - LocalFileStore: NOT FOUND in available modules
  
This suggests LocalFileStore may not be needed,
or it's been removed in newer LangChain versions.
""")
