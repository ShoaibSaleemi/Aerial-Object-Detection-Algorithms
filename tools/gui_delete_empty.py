"""
Drag-and-drop GUI to find and delete image files whose corresponding YOLO
label file is empty (zero bytes or all blank lines).

Usage:
  1. Drop one or more image files onto the drop zone.
  2. The log shows which images have empty labels (deletion candidates) and
     which are skipped (have annotations, or no label file found).
  3. Click "Delete X file(s)" and confirm to permanently delete both the
     image and its empty label file.

Requires: pip install tkinterdnd2 customtkinter
"""

from pathlib import Path
from tkinter import messagebox

try:
    import customtkinter as ctk
except ImportError:
    import sys
    print("customtkinter is required. Install it with:  pip install customtkinter")
    sys.exit(1)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    import sys
    print("tkinterdnd2 is required. Install it with:  pip install tkinterdnd2")
    sys.exit(1)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Layout constants (same class-folder set as gui_remap.py) ─────────────────
CLASS_FOLDERS = {"bird", "drone", "unknown", "no_label", "class3"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_label_path(image_path: Path) -> Path | None:
    """Return the corresponding label .txt path for an image, or None if the
    layout cannot be determined.  The returned path may not exist on disk —
    callers should use is_label_empty() which treats missing files as empty.

    Handles two layouts:
      - dataset/<split>/images/<file>.jpg  ->  dataset/<split>/labels/<file>.txt
      - dataset/<split>/<class_folder>/<file>.jpg  ->  dataset/<split>/labels/<file>.txt
    """
    stem  = image_path.stem
    parts = image_path.parts

    # Layout 1: .../images/<file>
    try:
        img_idx     = next(i for i, p in enumerate(parts) if p.lower() == "images")
        label_parts = parts[:img_idx] + ("labels",)
        return Path(*label_parts) / (stem + ".txt")
    except StopIteration:
        pass

    # Layout 2: .../split/<class_folder>/<file>  (created by split_by_class.py)
    if image_path.parent.name.lower() in CLASS_FOLDERS:
        return image_path.parent.parent / "labels" / (stem + ".txt")

    return None


def is_label_empty(label_path: Path) -> bool:
    """Return True if the label file has no annotation lines."""
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError:
        return True
    return not any(line.strip() for line in content.splitlines())


def split_of(image_path: Path) -> str:
    """Return the dataset split name (train/validation/test) or '?'."""
    parts_lower = [p.lower() for p in image_path.parts]
    return next((s for s in ("train", "validation", "test") if s in parts_lower), "?")


# ── App ───────────────────────────────────────────────────────────────────────

class App(TkinterDnD.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Delete Empty Labels")
        self.minsize(560, 500)
        self.configure(bg="#ebebeb")

        self._pending: list[tuple[Path, Path]] = []

        # ── Drop zone ─────────────────────────────────────────────────────────
        drop_outer = ctk.CTkFrame(self, corner_radius=10)
        drop_outer.pack(fill="both", expand=True, padx=16, pady=(16, 8))

        self.drop_label = ctk.CTkLabel(
            drop_outer, text="Drop image files here",
            font=ctk.CTkFont(size=20), text_color="#aaaaaa",
        )
        self.drop_label.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        # ── Log ───────────────────────────────────────────────────────────────
        self.log_box = ctk.CTkTextbox(
            self, height=200,
            font=ctk.CTkFont(family="Consolas", size=10),
        )
        self.log_box.pack(fill="both", padx=16, pady=(0, 8))
        self.log_box.configure(state="disabled")

        # ── Button row ────────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 16))

        self.btn_delete = ctk.CTkButton(
            btn_row, text="Delete 0 file(s)", state="disabled",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#e74c3c", hover_color="#c0392b",
            text_color="white", width=160,
            command=self._confirm_and_delete,
        )
        self.btn_delete.pack(side="left")

        ctk.CTkButton(
            btn_row, text="Clear log", width=110,
            fg_color="#d0d0d0", hover_color="#bbbbbb",
            text_color="#333333",
            command=self._clear_log,
        ).pack(side="left", padx=(10, 0))

        self._log("Ready.  Drop image files onto the zone above to check for empty labels.")

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.btn_delete.configure(state="disabled", text="Delete 0 file(s)")
        self._pending.clear()

    # ── Core scan ─────────────────────────────────────────────────────────────

    def _on_drop(self, event) -> None:
        self._pending = []
        self.btn_delete.configure(state="disabled", text="Delete 0 file(s)")

        raw_paths = self.tk.splitlist(event.data)
        self._log("\n" + "─" * 60)
        self._log(f"Dropped {len(raw_paths)} file(s) — scanning…")

        candidates = skipped_annot = skipped_unknown = 0

        for p in raw_paths:
            img_path = Path(p)
            if not img_path.is_file():
                self._log(f"  [SKIP]    {img_path.name}  (not a file)")
                continue

            label_path = find_label_path(img_path)
            split      = split_of(img_path)

            if label_path is None:
                self._log(f"  [SKIP]    [{split}] {img_path.name}  (cannot determine label location)")
                skipped_unknown += 1
                continue

            if is_label_empty(label_path):
                exists_note = "" if label_path.exists() else " (no label file)"
                self._log(f"  [EMPTY]   [{split}] {img_path.name}{exists_note}  ← will delete")
                self._pending.append((img_path, label_path))
                candidates += 1
            else:
                self._log(f"  [OK]      [{split}] {img_path.name}  (has annotations, skipped)")
                skipped_annot += 1

        self._log(
            f"Scan complete — "
            f"will delete: {candidates}  "
            f"has annotations (skipped): {skipped_annot}  "
            f"unknown layout: {skipped_unknown}"
        )

        if candidates:
            self.btn_delete.configure(state="normal", text=f"Delete {candidates} file(s)")
        else:
            self._log("Nothing to delete.")

    # ── Deletion ──────────────────────────────────────────────────────────────

    def _confirm_and_delete(self) -> None:
        if not self._pending:
            return

        preview_lines = [f"  • {img.name}" for img, _ in self._pending[:10]]
        if len(self._pending) > 10:
            preview_lines.append(f"  … and {len(self._pending) - 10} more")
        preview = "\n".join(preview_lines)

        answer = messagebox.askyesno(
            title="Confirm deletion",
            message=(
                f"Permanently delete {len(self._pending)} image(s) and their empty label file(s)?\n\n"
                f"{preview}\n\n"
                "This cannot be undone."
            ),
            icon="warning",
            default="no",
        )

        self.btn_delete.configure(state="disabled")

        if not answer:
            self._log("Deletion cancelled.")
            return

        deleted = errors = 0
        for img_path, label_path in self._pending:
            try:
                img_path.unlink()
                if label_path.exists():
                    label_path.unlink()
                    self._log(f"  [DELETED] {img_path.name}  +  {label_path.name}")
                else:
                    self._log(f"  [DELETED] {img_path.name}  (no label file on disk)")
                deleted += 1
            except OSError as exc:
                self._log(f"  [ERROR]   {img_path.name}  —  {exc}")
                errors += 1
                continue

            # Sync the mirror copy in the opposite folder (images/ <-> class subfolder)
            split_dir   = img_path.parent.parent
            parent_name = img_path.parent.name

            if parent_name == "images":
                # Dropped from images/ → also remove any class-subfolder copies
                for folder in CLASS_FOLDERS:
                    copy = split_dir / folder / img_path.name
                    if copy.exists():
                        try:
                            copy.unlink()
                            self._log(f"  [DELETED] split copy → {folder}/{img_path.name}")
                        except OSError as exc:
                            self._log(f"  [ERROR]   split copy {folder}/{img_path.name}  —  {exc}")
            elif parent_name in CLASS_FOLDERS:
                # Dropped from class subfolder → also remove the original in images/
                orig = split_dir / "images" / img_path.name
                if orig.exists():
                    try:
                        orig.unlink()
                        self._log(f"  [DELETED] images copy → {img_path.name}")
                    except OSError as exc:
                        self._log(f"  [ERROR]   images copy {img_path.name}  —  {exc}")

        self._log(f"Done — deleted: {deleted}  errors: {errors}")
        self._pending.clear()


App().mainloop()
