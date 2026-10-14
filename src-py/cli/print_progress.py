def print_progress(current, total, width=30):
    fraction = min(current / total, 1)
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)

    print(
        f"\r[{bar}] {fraction:.0%} ({current}/{total})",
        end="",
        flush=True,
    )