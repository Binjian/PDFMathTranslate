"""Progress adapters shared by GUI translation modes."""


def update_gui_progress(progress, event) -> None:
    """Report progress from either the legacy tqdm or precise kernel format."""
    if isinstance(event, dict):
        desc = event.get("stage") or "Translating..."
        try:
            completed = float(event.get("overall_progress", 0.0)) / 100.0
        except (TypeError, ValueError):
            completed = 0.0
    else:
        desc = getattr(event, "desc", "") or "Translating..."
        total = getattr(event, "total", 0)
        completed = event.n / total if total else 0.0
    progress(max(0.0, min(completed, 1.0)), desc=desc)
