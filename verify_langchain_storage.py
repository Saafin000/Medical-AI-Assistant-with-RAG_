#!/usr/bin/env python3
"""
READ-ONLY Verification of LangChain Storage Imports

Inspects:
1. Installed package versions
2. Available import paths for storage classes
3. Usage in vectorstore_qdrant.py
4. Docling availability
"""

import sys
import importlib
import subprocess


def get_package_version(package_name):
    """Get installed version of a package."""
    try:
        module = importlib.import_module(package_name.replace('-', '_'))
        return getattr(module, '__version__', 'unknown')
    except:
        return None


def test_import(import_statement):
    """Test if an import statement works."""
    try:
        exec(import_statement)
        return True, None
    except Exception as e:
        return False, str(e)


def check_pip_version(package_name):
    """Check pip-installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return None
    except:
        return None


def main():
    """Run verification."""
    
    print("\n" + "="*80)
    print("LANGCHAIN STORAGE IMPORTS VERIFICATION")
    print("="*80)
    
    # Step 1: Package Versions
    print("\n[STEP 1] INSTALLED PACKAGE VERSIONS")
    print("="*80)
    
    packages = [
        'langchain',
        'langchain-core',
        'langchain-community',
        'docling'
    ]
    
    versions = {}
    
    for pkg in packages:
        # Try import version
        version = get_package_version(pkg)
        if version is None:
            # Try pip show
            version = check_pip_version(pkg)
        
        if version is None:
            version = "NOT INSTALLED"
        
        versions[pkg] = version
        print("  {}: {}".format(pkg, version))
    
    # Step 2: Import Tests
    print("\n[STEP 2] IMPORT PATH TESTS")
    print("="*80)
    
    import_tests = [
        "from langchain.storage import InMemoryStore, LocalFileStore",
        "from langchain_community.storage import InMemoryStore, LocalFileStore",
        "from langchain_core.stores import InMemoryStore",
        "from langchain_community.storage import LocalFileStore",
        "from langchain.storage import InMemoryStore",
        "from langchain.storage import LocalFileStore",
        "from langchain_community.stores import InMemoryStore",
        "from langchain_community.stores import LocalFileStore"
    ]
    
    results = {}
    
    for import_stmt in import_tests:
        success, error = test_import(import_stmt)
        results[import_stmt] = {
            "success": success,
            "error": error
        }
        
        status = "PASS" if success else "FAIL"
        print("\n  [{}] {}".format(status, import_stmt))
        
        if error:
            print("        Error: {}".format(error[:100]))
    
    # Step 3: Check vectorstore_qdrant.py for storage usage
    print("\n[STEP 3] STORAGE CLASSES USED IN VECTORSTORE_QDRANT.PY")
    print("="*80)
    
    try:
        with open('agents/rag_agent/vectorstore_qdrant.py', 'r') as f:
            content = f.read()
        
        # Find import line
        lines = content.split('\n')
        
        print("\nImport statement (line 9):")
        if len(lines) > 8:
            print("  {}".format(lines[8]))
        
        # Search for usage of InMemoryStore and LocalFileStore
        print("\nUsage of storage classes in the file:")
        
        usage_count = {
            'InMemoryStore': 0,
            'LocalFileStore': 0
        }
        
        for i, line in enumerate(lines, 1):
            if 'InMemoryStore' in line:
                usage_count['InMemoryStore'] += 1
                if usage_count['InMemoryStore'] <= 3:
                    print("  Line {}: {}".format(i, line.strip()[:80]))
            
            if 'LocalFileStore' in line:
                usage_count['LocalFileStore'] += 1
                if usage_count['LocalFileStore'] <= 3:
                    print("  Line {}: {}".format(i, line.strip()[:80]))
        
        print("\nUsage counts:")
        print("  InMemoryStore: {} occurrences".format(usage_count['InMemoryStore']))
        print("  LocalFileStore: {} occurrences".format(usage_count['LocalFileStore']))
    
    except Exception as e:
        print("  Error reading file: {}".format(e))
    
    # Step 4: Check for docling
    print("\n[STEP 4] DOCLING AVAILABILITY")
    print("="*80)
    
    try:
        import docling
        print("  docling: INSTALLED (version: {})".format(getattr(docling, '__version__', 'unknown')))
        docling_available = True
    except ImportError:
        print("  docling: NOT INSTALLED")
        docling_available = False
    
    # Step 5: Analysis
    print("\n[STEP 5] ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    print("\nPackage Versions Summary:")
    for pkg, version in versions.items():
        print("  {}: {}".format(pkg, version))
    
    print("\nImport Compatibility:")
    
    # Find working imports
    working_imports = {
        'InMemoryStore': None,
        'LocalFileStore': None
    }
    
    for import_stmt, result in results.items():
        if result['success']:
            if 'InMemoryStore' in import_stmt and ',' not in import_stmt:
                if working_imports['InMemoryStore'] is None:
                    working_imports['InMemoryStore'] = import_stmt
            if 'LocalFileStore' in import_stmt:
                if working_imports['LocalFileStore'] is None:
                    working_imports['LocalFileStore'] = import_stmt
    
    for class_name, import_stmt in working_imports.items():
        if import_stmt:
            print("  {} : {}".format(class_name, import_stmt))
        else:
            print("  {} : NO WORKING IMPORT FOUND".format(class_name))
    
    # Check if current import works
    current_import = "from langchain.storage import InMemoryStore, LocalFileStore"
    current_works = results[current_import]['success']
    
    print("\nCurrent Import Status:")
    print("  Current: {}".format(current_import))
    print("  Status: {}".format("WORKS" if current_works else "BROKEN"))
    
    if not current_works:
        print("  Error: {}".format(results[current_import]['error'][:100]))
    
    print("\nDocling Status:")
    print("  Status: {}".format("INSTALLED" if docling_available else "NOT INSTALLED"))
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if not current_works:
        # Find best alternative
        alternatives = []
        for import_stmt, result in results.items():
            if result['success'] and ('InMemoryStore' in import_stmt or 'LocalFileStore' in import_stmt):
                alternatives.append(import_stmt)
        
        if alternatives:
            print("\nCurrent import is BROKEN")
            print("Suggested alternatives:")
            for alt in alternatives[:3]:
                print("  - {}".format(alt))
        else:
            print("\nNO WORKING IMPORTS FOUND FOR STORAGE CLASSES")
    else:
        print("\nCurrent import is WORKING")
    
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
