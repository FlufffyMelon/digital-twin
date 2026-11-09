#!/usr/bin/env python3
"""Strip outputs and execution counts from Jupyter notebooks.

Outputs of this project's notebooks routinely contain chat messages, so they
must never reach git. Works without extra dependencies.

    python3 scripts/nbstrip.py a.ipynb b.ipynb   # strip files in place
    python3 scripts/nbstrip.py < a.ipynb         # stdin -> stdout (git clean filter)
    python3 scripts/nbstrip.py --check a.ipynb   # exit 1 if a notebook has outputs
"""
import json
import sys


def strip(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def has_outputs(nb):
    return any(
        cell.get("outputs") or cell.get("execution_count") is not None
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def dump(nb):
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main(argv):
    if argv and argv[0] == "--check":
        dirty = []
        for path in argv[1:]:
            with open(path, encoding="utf-8") as f:
                if has_outputs(json.load(f)):
                    dirty.append(path)
        for path in dirty:
            print(f"notebook has outputs: {path}", file=sys.stderr)
        return 1 if dirty else 0

    if not argv:
        sys.stdout.write(dump(strip(json.load(sys.stdin))))
        return 0

    for path in argv:
        with open(path, encoding="utf-8") as f:
            nb = strip(json.load(f))
        with open(path, "w", encoding="utf-8") as f:
            f.write(dump(nb))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
