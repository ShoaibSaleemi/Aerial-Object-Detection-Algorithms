"""
Drag-and-drop GUI to find and delete image files whose corresponding YOLO
label file is empty (zero bytes or all blank lines).

Usage:
  1. Drop one or more image files onto the drop zone.
  2. The log shows which images have empty labels (deletion candidates) and
     which are skipped (have annotations, or no label file found).
  3. Click "Delete X file(s)" and confirm to permanently delete both the
     image and its empty label file.

Requires: pip install tkinterdnd2
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext
from pathlib import Path

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    import sys
    print("tkinterdnd2 is required. Install it with:  pip install tkinterdnd2")
    sys.exit(1)


# ── Layout constants (same class-folder set as gui_remap.py) ─────────────────
CLASS_FOLDERS = {"bird", "drone", "unknown", "no_label", "class3"}

# Deletion candidates accumulated across the current scan session
_pending: list[tuple[Path, Path]] = []  # list of (image_path, label_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_label_path(image_path: Path) -> Path | None:
    """Return the corresponding label .txt path for an image, or None.

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


def is_label_empty(label_path: Path) -> bool:
    """Return True if the label file has no annotation lines.

    A file is considered empty when it is zero bytes, or every line contains
    only whitespace characters.
    """
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError:
        return True  # unreadable → treat as empty
    return not any(line.strip() for line in content.splitlines())


def split_of(image_path: Path) -> str:
    """Return the dataset split name (train/validation/test) or '?'."""
    parts_lower = [p.lower() for p in image_path.parts]
    return next((s for s in ("train", "validation", "test") if s in parts_lower), "?")


# ── Log helper ────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    log_text.config(state="normal")
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)
    log_text.config(state="disabled")


# ── Core scan ─────────────────────────────────────────────────────────────────

def process_drop(event) -> None:
    global _pending
    _pending = []
    btn_delete.config(state="disabled", text="Delete 0 file(s)")

    raw_paths = root.tk.splitlist(event.data)
    log("\n" + "─" * 60)
    log(f"Dropped {len(raw_paths)} file(s) — scanning…")

    candidates = skipped_annot = skipped_missing = 0

    for p in raw_paths:
        img_path = Path(p)
        if not img_path.is_file():
            log(f"  [SKIP]    {img_path.name}  (not a file)")
            continue

        label_path = find_label_path(img_path)
        split = split_of(img_path)

        if label_path is None:
            log(f"  [NO LBL]  [{split}] {img_path.name}  (no matching label file found)")
            skipped_missing += 1
            continue

        if is_label_empty(label_path):
            log(f"  [EMPTY]   [{split}] {img_path.name}  ← will delete")
            _pending.append((img_path, label_path))
            candidates += 1
        else:
            log(f"  [OK]      [{split}] {img_path.name}  (has annotations, skipped)")
            skipped_annot += 1

    log(
        f"Scan complete — "
        f"empty: {candidates}  "
        f"has annotations: {skipped_annot}  "
        f"no label file: {skipped_missing}"
    )

    if candidates:
        btn_delete.config(state="normal", text=f"Delete {candidates} file(s)")
    else:
        log("Nothing to delete.")


# ── Deletion ──────────────────────────────────────────────────────────────────

def confirm_and_delete() -> None:
    if not _pending:
        return

    # Build a short preview for the confirmation dialog (max 10 entries shown)
    preview_lines = [f"  • {img.name}" for img, _ in _pending[:10]]
    if len(_pending) > 10:
        preview_lines.append(f"  … and {len(_pending) - 10} more")
    preview = "\n".join(preview_lines)

    answer = messagebox.askyesno(
        title="Confirm deletion",
        message=(
            f"Permanently delete {len(_pending)} image(s) and their empty label file(s)?\n\n"
            f"{preview}\n\n"
            "This cannot be undone."
        ),
        icon="warning",
        default="no",
    )

    btn_delete.config(state="disabled")

    if not answer:
        log("Deletion cancelled.")
        return

    deleted = errors = 0
    for img_path, label_path in _pending:
        try:
            img_path.unlink()
            label_path.unlink()
            log(f"  [DELETED] {img_path.name}  +  {label_path.name}")
            deleted += 1
        except OSError as exc:
            log(f"  [ERROR]   {img_path.name}  —  {exc}")
            errors += 1

    log(f"Done — deleted: {deleted}  errors: {errors}")
    _pending.clear()


# ── Clear log ─────────────────────────────────────────────────────────────────

def clear_log() -> None:
    log_text.config(state="normal")
    log_text.delete("1.0", tk.END)
    log_text.config(state="disabled")
    btn_delete.config(state="disabled", text="Delete 0 file(s)")
    _pending.clear()


# ── GUI layout ────────────────────────────────────────────────────────────────

root = TkinterDnD.Tk()
root.title("Delete Empty Labels")
root.minsize(560, 460)

# Drop zone
drop_frame = tk.LabelFrame(
    root, text="Drop image files here",
    font=("Segoe UI", 10), padx=10, pady=10,
)
drop_frame.pack(fill="both", expand=True, padx=12, pady=(12, 6))

drop_label = tk.Label(
    drop_frame,
    text="Drop Images Here",
    font=("Segoe UI", 16), fg="#555", bg="#f0f0f0", relief="flat",
)
drop_label.pack(fill="both", expand=True, ipadx=20, ipady=50)
drop_label.drop_target_register(DND_FILES)
drop_label.dnd_bind("<<Drop>>", process_drop)

# Log area
log_text = scrolledtext.ScrolledText(
    root, height=14, state="disabled",
    font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
    insertbackground="white",
)
log_text.pack(fill="both", padx=12, pady=(0, 6))

# Button row
btn_row = tk.Frame(root, pady=6)
btn_row.pack(fill="x", padx=12)

btn_delete = tk.Button(
    btn_row,
    text="Delete 0 file(s)",
    state="disabled",
    font=("Segoe UI", 10, "bold"),
    fg="white", bg="#c0392b",
    activebackground="#96281b", activeforeground="white",
    relief="flat", padx=16, pady=6,
    command=confirm_and_delete,
)
btn_delete.pack(side="left")

btn_clear = tk.Button(
    btn_row,
    text="Clear log",
    font=("Segoe UI", 10),
    relief="flat", padx=12, pady=6,
    command=clear_log,
)
btn_clear.pack(side="left", padx=(10, 0))

log("Ready.  Drop image files onto the zone above to check for empty labels.")

root.mainloop()
