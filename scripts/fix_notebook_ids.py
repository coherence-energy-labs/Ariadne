"""Add missing cell IDs to all notebooks (nbformat 5.1.4+ requirement)."""
import json
import uuid
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
for nb_path in sorted((repo / "notebooks").glob("*.ipynb")):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    changed = 0
    for cell in nb.get("cells", []):
        if "id" not in cell:
            cell["id"] = uuid.uuid4().hex[:12]
            changed += 1
    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
    print(f"  {nb_path.name}: added {changed} cell IDs")
print("done")
