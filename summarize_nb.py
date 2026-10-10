import nbformat

def summarize_notebook(path):
    print("=========================================")
    print("Notebook:", path)
    print("=========================================")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    for i, cell in enumerate(nb.cells):
        src = cell.source.strip()
        if not src:
            continue
        print(f"\n--- Cell {i} [{cell.cell_type}] ---")
        lines = src.split("\n")
        print("\n".join(lines[:10]))
        if len(lines) > 10:
            print(f"... ({len(lines)-10} more lines)")

summarize_notebook('NLP_ABSA.ipynb')
summarize_notebook('SentiMix_ABSA_Starter.ipynb')
