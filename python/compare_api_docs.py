"""
Compare Python APIs from PYTHON_API_REFERENCE.md with USER_GUIDE.md.

This script doesn't require importing the module - it works from documentation files.
"""

import re
import os


def extract_apis_from_api_reference():
    """Extract all classes and functions from PYTHON_API_REFERENCE.md."""
    path = os.path.join(os.path.dirname(__file__), "..", "PYTHON_API_REFERENCE.md")

    with open(path, 'r') as f:
        content = f.read()

    classes = set()
    functions = set()

    # Extract class definitions (### `ga.ClassName`)
    class_pattern = r'###\s+`ga\.([A-Z][A-Za-z0-9_]*)`'
    for match in re.finditer(class_pattern, content):
        classes.add(match.group(1))

    # Extract function definitions (#### `ga.function_name`)
    func_pattern = r'####\s+`ga\.([a-z][a-z0-9_]*)\('
    for match in re.finditer(func_pattern, content):
        functions.add(match.group(1))

    # Also check ## Summary Statistics section
    if "Total Classes:** 52" in content:
        print("  PYTHON_API_REFERENCE.md reports 52 classes")
    if "Total Functions:** 26" in content:
        print("  PYTHON_API_REFERENCE.md reports 26 functions")

    return classes, functions


def extract_apis_from_user_guide():
    """Extract all API symbols from USER_GUIDE.md summary table."""
    path = os.path.join(os.path.dirname(__file__), "..", "USER_GUIDE.md")

    with open(path, 'r') as f:
        content = f.read()

    # Find the summary table section
    summary_start = content.find("## Summary of Python Bindings Coverage")
    if summary_start == -1:
        print("ERROR: Could not find summary section in USER_GUIDE.md")
        return set(), set()

    summary_end = content.find("---", summary_start + 100)
    summary_section = content[summary_start:summary_end]

    classes = set()
    functions = set()

    # Extract symbols like `ga.ClassName` or `ga.function_name`
    pattern = r'`ga\.([A-Za-z0-9_]+)`'
    matches = re.findall(pattern, summary_section)

    for match in matches:
        # Check if it starts with uppercase (class) or lowercase (function/constant)
        if match[0].isupper():
            classes.add(match)
        else:
            functions.add(match)

    return classes, functions


def main():
    print("=" * 80)
    print("COMPARING API DOCUMENTATION: PYTHON_API_REFERENCE.md vs USER_GUIDE.md")
    print("=" * 80)
    print()

    print("Reading PYTHON_API_REFERENCE.md...")
    ref_classes, ref_functions = extract_apis_from_api_reference()
    print(f"  Found {len(ref_classes)} classes")
    print(f"  Found {len(ref_functions)} functions")
    print()

    print("Reading USER_GUIDE.md...")
    guide_classes, guide_functions = extract_apis_from_user_guide()
    print(f"  Found {len(guide_classes)} classes in summary table")
    print(f"  Found {len(guide_functions)} functions in summary table")
    print()

    # Find undocumented classes
    undoc_classes = ref_classes - guide_classes
    if undoc_classes:
        print("-" * 80)
        print(f"CLASSES MISSING FROM USER_GUIDE.md ({len(undoc_classes)})")
        print("-" * 80)
        for cls in sorted(undoc_classes):
            print(f"  ga.{cls}")
        print()

    # Find undocumented functions
    undoc_functions = ref_functions - guide_functions
    if undoc_functions:
        print("-" * 80)
        print(f"FUNCTIONS MISSING FROM USER_GUIDE.md ({len(undoc_functions)})")
        print("-" * 80)
        for func in sorted(undoc_functions):
            print(f"  ga.{func}")
        print()

    # Find extras in USER_GUIDE that aren't in reference
    extra_classes = guide_classes - ref_classes
    if extra_classes:
        print("-" * 80)
        print(f"EXTRA CLASSES IN USER_GUIDE.md (not in reference) ({len(extra_classes)})")
        print("-" * 80)
        for cls in sorted(extra_classes):
            print(f"  ga.{cls}")
        print()

    extra_functions = guide_functions - ref_functions
    if extra_functions:
        print("-" * 80)
        print(f"EXTRA FUNCTIONS IN USER_GUIDE.md (not in reference) ({len(extra_functions)})")
        print("-" * 80)
        for func in sorted(extra_functions):
            print(f"  ga.{func}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Classes missing from USER_GUIDE.md: {len(undoc_classes)}")
    print(f"Functions missing from USER_GUIDE.md: {len(undoc_functions)}")
    print(f"Extra classes in USER_GUIDE.md: {len(extra_classes)}")
    print(f"Extra functions in USER_GUIDE.md: {len(extra_functions)}")

    total_missing = len(undoc_classes) + len(undoc_functions)

    if total_missing > 0:
        print()
        print(f"⚠ {total_missing} APIs are missing from USER_GUIDE.md summary table")
        return 1
    elif len(extra_classes) + len(extra_functions) > 0:
        print()
        print("⚠ USER_GUIDE.md has extra entries not in PYTHON_API_REFERENCE.md")
        return 1
    else:
        print()
        print("✓ USER_GUIDE.md and PYTHON_API_REFERENCE.md are in sync!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
