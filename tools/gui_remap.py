"""
Drag-and-drop GUI to force all class IDs in label files to a target class ID.
Requires: pip install tkinterdnd2 customtkinter
"""
from pathlib import Path

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

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CLASS_OPTIONS = ["0 — bird", "1 — drone", "2 — unknown"]


# ── Label resolution ──────────────────────────────────────────────────────────

def find_label_path(image_path: Path):
    """Given an image path, return the corresponding label .txt path.

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
        label_path  = Path(*label_parts) / (stem + ".txt")
        if label_path.exists():
            return label_path
    except StopIteration:
        pass

    # Layout 2: .../split/<subfolder>/<file>  (created by split_by_class.py)
    # Works for both class-name folders (bird/drone/…) and prefix folders (20190/…)
    label_path = image_path.parent.parent / "labels" / (stem + ".txt")
    if label_path.exists():
        return label_path

    return None


# ── App ───────────────────────────────────────────────────────────────────────

class App(TkinterDnD.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Label Class Remapper")
        self.minsize(560, 520)
        self.configure(bg="#2b2b2b")

        # ── Top bar ───────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(16, 0))

        ctk.CTkLabel(
            top, text="Force all class IDs to:",
            font=ctk.CTkFont(size=13),
        ).pack(side="left")

        self.class_var = ctk.StringVar(value=CLASS_OPTIONS[2])  # default: unknown
        ctk.CTkOptionMenu(
            top, values=CLASS_OPTIONS, variable=self.class_var,
            width=180, font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=12)

        # ── Drop zone ─────────────────────────────────────────────────────────
        drop_outer = ctk.CTkFrame(self, corner_radius=10)
        drop_outer.pack(fill="both", expand=True, padx=16, pady=12)

        ctk.CTkLabel(
            drop_outer, text="Drop image files here",
            font=ctk.CTkFont(size=11), text_color="#888888",
        ).pack(anchor="nw", padx=12, pady=(8, 0))

        self.drop_label = ctk.CTkLabel(
            drop_outer, text="Drop Images Here",
            font=ctk.CTkFont(size=20), text_color="#555555",
        )
        self.drop_label.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        # ── Log ───────────────────────────────────────────────────────────────
        self.log_box = ctk.CTkTextbox(
            self, height=180,
            font=ctk.CTkFont(family="Consolas", size=10),
        )
        self.log_box.pack(fill="both", padx=16, pady=(0, 8))
        self.log_box.configure(state="disabled")

        # ── Button row ────────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkButton(
            btn_row, text="Clear log", width=110,
            fg_color="#3a3a3a", hover_color="#4a4a4a",
            command=self._clear_log,
        ).pack(side="left")

        self._log("Ready.  Set the target class ID above, then drop image files onto the zone.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_target_id(self) -> str:
        return self.class_var.get().split(" ")[0]

    def _log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ── Drop handler ──────────────────────────────────────────────────────────

    def _on_drop(self, event) -> None:
        paths     = self.tk.splitlist(event.data)
        target_id = self._get_target_id()
        self._log("\n" + "─" * 52)
        self._log(f"Dropped {len(paths)} file(s)  →  forcing class {target_id}")

        updated = skipped = missing = 0
        for p in paths:
            img_path   = Path(p)
            label_path = find_label_path(img_path)

            if label_path is None:
                self._log(f"  [NO LABEL]  {img_path.name}")
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
                self._log(f"  [OK]       [{split}] {img_path.name}")
                updated += 1
            else:
                self._log(f"  [SKIP]     [{split}] {img_path.name}  (already class {target_id})")
                skipped += 1

        self._log(f"Done — updated: {updated}  skipped: {skipped}  no label: {missing}")


App().mainloop()
