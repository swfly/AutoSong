import os

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def collect_python_files(base_dir):
    py_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                py_files.append((rel_path, full_path))
    return sorted(py_files)

def generate_prompt(base_dir):
    prompt_parts = []

    # README
    readme_path = os.path.join(base_dir, 'README.md')
    if os.path.exists(readme_path):
        prompt_parts.append("# 📘 README.md\n")
        prompt_parts.append(read_file(readme_path))
    else:
        prompt_parts.append("# 📘 No README.md found.\n")

    # Python files
    py_files = collect_python_files(base_dir)
    if not py_files:
        prompt_parts.append("\n# ⚠️ No Python files found in the project.\n")
    else:
        for rel_path, full_path in py_files:
            prompt_parts.append(f"\n# 📄 {rel_path}\n")
            prompt_parts.append("```python\n")
            prompt_parts.append(read_file(full_path))
            prompt_parts.append("\n```\n")

    return '\n'.join(prompt_parts)

if __name__ == "__main__":
    base_dir = "."  # or specify another directory
    prompt_text = generate_prompt(base_dir)
    
    output_path = "codebase_review_prompt.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt_text)

    print(f"Review prompt written to {output_path}")
