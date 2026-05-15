from pathlib import Path

labels_a = Path(r"C:\Users\shoai\project\thesis\dataset\validation\labels")
labels_b = Path(r"C:\Users\shoai\project\thesis\dataset\validation\validation\labels")

stems_a = {p.stem for p in labels_a.glob("*.txt")}
stems_b = {p.stem for p in labels_b.glob("*.txt")}

matches = stems_a & stems_b

print(f"TXTs in validation/labels/:            {len(stems_a)}")
print(f"TXTs in validation/validation/labels/: {len(stems_b)}")
print(f"Exact stem matches:                    {len(matches)}")
print(f"Only in validation/labels/:            {len(stems_a - stems_b)}")
print(f"Only in validation/validation/labels/: {len(stems_b - stems_a)}")
