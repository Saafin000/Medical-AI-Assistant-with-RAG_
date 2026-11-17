#!/usr/bin/env python3
"""
Find LocalFileStore import location.
"""

import sys
import pkgutil
import importlib


def find_class_in_module(class_name, module_name):
    """Try to find a class in a module."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            return True
        return False
    except:
        return False


def search_package(package_name, class_name, max_depth=3):
    """Recursively search a package for a class."""
    
    results = []
    
    try:
        package = importlib.import_module(package_name)
        package_path = package.__path__ if hasattr(package, '__path__') else []
        
        # Check current module
        if hasattr(package, class_name):
            results.append(package_name)
        
        # Search submodules
        if package_path and max_depth > 0:
            for importer, modname, ispkg in pkgutil.iter_modules(package_path):
                full_name = package_name + '.' + modname
                
                try:
                    submodule = importlib.import_module(full_name)
                    
                    if hasattr(submodule, class_name):
                        results.append(full_name)
                    
                    if ispkg and max_depth > 1:
                        # Recursively search subpackages
                        sub_results = search_package(full_name, class_name, max_depth - 1)
                        results.extend(sub_results)
                except:
                    pass
    except:
        pass
    
    return results


def main():
    """Search for LocalFileStore."""
    
    print("\n" + "="*80)
    print("SEARCHING FOR LocalFileStore")
    print("="*80)
    
    packages_to_search = [
        'langchain',
        'langchain_core',
        'langchain_community'
    ]
    
    print("\nSearching installed packages...")
    
    for package in packages_to_search:
        print("\nSearching: {}".format(package))
        
        results = search_package(package, 'LocalFileStore', max_depth=4)
        
        if results:
            print("  Found in:")
            for result in results:
                print("    - {}".format(result))
                
                # Try the import
                import_stmt = "from {} import LocalFileStore".format(result)
                try:
                    exec(import_stmt)
                    print("      Import: SUCCESS - {}".format(import_stmt))
                except Exception as e:
                    print("      Import: FAILED - {}".format(str(e)[:50]))
        else:
            print("  Not found in {}".format(package))
    
    # Try some common variations
    print("\n" + "="*80)
    print("TRYING COMMON IMPORT VARIATIONS")
    print("="*80)
    
    variations = [
        "from langchain.schema.storage import LocalFileStore",
        "from langchain.document_loaders.base import LocalFileStore",
        "from langchain_text_splitters import LocalFileStore",
        "from langchain.storage import LocalFileStore",
        "import langchain.storage",
        "import langchain_community.storage",
        "from langchain import storage",
    ]
    
    for import_stmt in variations:
        try:
            exec(import_stmt)
            print("PASS: {}".format(import_stmt))
        except Exception as e:
            # Silently skip failures
            pass
    
    # Check what's in langchain_community.storage
    print("\n" + "="*80)
    print("CONTENTS OF AVAILABLE MODULES")
    print("="*80)
    
    try:
        import langchain_core.stores as stores_module
        print("\nlangchain_core.stores contents:")
        items = [x for x in dir(stores_module) if not x.startswith('_')]
        for item in items[:10]:
            print("  - {}".format(item))
        if len(items) > 10:
            print("  ... and {} more".format(len(items) - 10))
    except Exception as e:
        print("  Error: {}".format(e))
    
    try:
        import langchain_community.storage as storage_module
        print("\nlangchain_community.storage contents:")
        items = [x for x in dir(storage_module) if not x.startswith('_')]
        for item in items[:10]:
            print("  - {}".format(item))
        if len(items) > 10:
            print("  ... and {} more".format(len(items) - 10))
    except Exception as e:
        print("  Error: {}".format(e))
    
    try:
        import langchain_community as lc_module
        print("\nlangchain_community submodules:")
        if hasattr(lc_module, '__path__'):
            import pkgutil
            for importer, modname, ispkg in pkgutil.iter_modules(lc_module.__path__):
                if 'storage' in modname or 'store' in modname:
                    print("  - {}".format(modname))
    except Exception as e:
        print("  Error: {}".format(e))


if __name__ == "__main__":
    main()
