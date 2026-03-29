"""
Check which sections in USER_GUIDE.md have Python examples.

Identifies sections marked with Python support (✅) but may lack detailed examples.
"""

import re


def main():
    with open("USER_GUIDE.md", 'r') as f:
        content = f.read()

    # Find the feature index table
    table_start = content.find("## Table of Contents — Feature Index")
    table_end = content.find("**Legend:**", table_start)
    table_section = content[table_start:table_end]

    # Parse the table
    lines = table_section.split('\n')
    features = []

    for line in lines:
        if line.startswith('| '):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5 and parts[1].isdigit():
                feature_num = parts[1]
                feature_name = parts[2]
                cpp_status = parts[3]
                python_status = parts[4]

                # Extract section name from markdown link
                match = re.search(r'\[(.*?)\]', feature_name)
                if match:
                    section_name = match.group(1)
                    features.append({
                        'num': feature_num,
                        'name': section_name,
                        'cpp': cpp_status,
                        'python': python_status
                    })

    print("=" * 80)
    print("PYTHON SUPPORT STATUS IN USER_GUIDE.md SECTIONS")
    print("=" * 80)
    print()

    # Group by Python status
    full_support = []
    partial_support = []
    no_support = []

    for feature in features:
        if '✅' in feature['python']:
            full_support.append(feature)
        elif '⚠️' in feature['python']:
            partial_support.append(feature)
        elif '❌' in feature['python']:
            no_support.append(feature)

    print(f"Features with FULL Python support: {len(full_support)}")
    for f in full_support:
        print(f"  {f['num']:2s}. {f['name']}")
    print()

    print(f"Features with PARTIAL Python support: {len(partial_support)}")
    for f in partial_support:
        status_note = f['python'].replace('⚠️', '').strip()
        print(f"  {f['num']:2s}. {f['name']:50s} - {status_note}")
    print()

    print(f"Features NOT exposed to Python: {len(no_support)}")
    for f in no_support:
        print(f"  {f['num']:2s}. {f['name']}")
    print()

    # Now check if sections have "### Python" subsections
    print("=" * 80)
    print("CHECKING FOR PYTHON SUBSECTIONS IN EACH FEATURE")
    print("=" * 80)
    print()

    missing_python_sections = []

    for feature in full_support:
        # Look for the section
        section_pattern = rf"## {feature['num']}\.\s+{re.escape(feature['name'])}"
        section_match = re.search(section_pattern, content)

        if section_match:
            # Find the next section
            next_section_pattern = r"##\s+\d+\."
            next_section = re.search(next_section_pattern, content[section_match.end():])

            if next_section:
                section_content = content[section_match.start():section_match.end() + next_section.start()]
            else:
                section_content = content[section_match.start():]

            # Check if there's a Python subsection
            has_python_section = "### Python" in section_content or "### python" in section_content

            if not has_python_section:
                missing_python_sections.append(feature)
                print(f"⚠ {feature['num']:2s}. {feature['name']:50s} - No '### Python' subsection found")

    print()
    print(f"Total sections with full Python support: {len(full_support)}")
    print(f"Sections missing '### Python' subsections: {len(missing_python_sections)}")

    if missing_python_sections:
        print()
        print("These sections are marked as having Python support but may lack")
        print("detailed Python examples in their own subsections.")


if __name__ == "__main__":
    main()
