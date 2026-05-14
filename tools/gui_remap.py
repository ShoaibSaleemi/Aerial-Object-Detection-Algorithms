"""
Drag-and-drop GUI to force all class IDs in label files to a target class ID.
Requires: pip install tkinterdnd2
"""
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    import sys
    print("tkinterdnd2 is required. Install it with:  pip install tkinterdnd2")
    sys.exit(1)


CLASS_FOLDERS = {"bird", "drone", "unknown", "no_label", "class3"}

def find_label_path(image_path: Path):
    """Given an image path, return the corresponding label .txt path.

    Handles two layouts:
      - dataset/<split>/images/<file>.jpg  ->  dataset/<split>/labels/<file>.txt
      - dataset/<split>/<class_folder>/<file>.jpg  ->  dataset/<split>/labels/<file>.txt
    """
    stem = image_path.stem
    parts = image_path.parts

    # Layout 1: .../images/<file>
    try:
        img_idx = next(i for i, p in enumerate(parts) if p.lower() == "images")
        label_parts = parts[:img_idx] + ("labels",)
        label_path = Path(*label_parts) / (stem + ".txt")
        if label_path.exists():
            return label_path
    except StopIteration:
        pass

    # Layout 2: .../split/<class_folder>/<file>  (created by split_by_class.py)
    if image_path.parent.name.lower() in CLASS_FOLDERS:
        label_path = image_path.parent.parent / "labels" / (stem + ".txt")
        if label_path.exists():
            return label_path

    return None


def process_drop(event):
    paths = root.tk.splitlist(event.data)
    target_id = str(class_id_var.get())
    log("\n" + "-" * 52)
    log(f"Dropped {len(paths)} file(s)  ->  forcing class {target_id}")

    updated = skipped = missing = 0
    for p in paths:
        img_path = Path(p)
        label_path = find_label_path(img_path)

        if label_path is None:
            log(f"  [NO LABEL]  {img_path.name}")
            missing += 1
            continue

        parts_lower = [x.lower() for x in img_path.parts]
        split = next((s for s in ("train", "validation", "test") if s in parts_lower), "?")

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            parts[0] = target_id
            new_lines.append(" ".join(parts) + "\n")

        if lines != new_lines:
            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            log(f"  [OK]       [{split}] {img_path.name}")
            updated += 1
        else:
            log(f"  [SKIP]     [{split}] {img_path.name}  (already class {target_id})")
            skipped += 1

    log(f"Done - updated: {updated}  skipped: {skipped}  no label: {missing}")


def log(msg: str):
    log_text.config(state="normal")
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)
    log_text.config(state="disabled")


# -- GUI ----------------------------------------------------------------------
root = TkinterDnD.Tk()
root.title("Label Class Remapper")
root.minsize(520, 460)

top = tk.Frame(root, pady=10, padx=12)
top.pack(fill="x")
tk.Label(top, text="Force all class IDs to:", font=("Segoe UI", 11)).pack(side="left")
class_id_var = tk.IntVar(value=2)
tk.Spinbox(top, from_=0, to=2, textvariable=class_id_var, width=5,
           font=("Segoe UI", 11)).pack(side="left", padx=8)
tk.Label(top, text="(0=bird  1=drone  2=unknown)", font=("Segoe UI", 9),
         fg="#666").pack(side="left")

drop_frame = tk.LabelFrame(root, text="Drop image files here",
                            font=("Segoe UI", 10), padx=10, pady=10)
drop_frame.pack(fill="both", expand=True, padx=12, pady=(0, 6))

drop_label = tk.Label(drop_frame, text="Drop Images Here",
                       font=("Segoe UI", 16), fg="#555", bg="#f0f0f0", relief="flat")
drop_label.pack(fill="both", expand=True, ipadx=20, ipady=50)
drop_label.drop_target_register(DND_FILES)
drop_label.dnd_bind("<<Drop>>", process_drop)

log_text = scrolledtext.ScrolledText(root, height=12, state="disabled",
                                      font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                                      insertbackground="white")
log_text.pack(fill="both", padx=12, pady=(0, 12))

log("Ready.  Set the target class ID above, then drop image files onto the zone.")

root.mainloop()
