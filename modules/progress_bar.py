def print_progress(current: int, total: int, message: str = "", elapsed: float = 0):
    bar_length = 40
    progress = current / total if total > 0 else 0
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)

    if elapsed > 0:
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        time_info = f" | {elapsed:.1f}s elapsed, {eta:.1f}s remaining"
    else:
        time_info = ""

    print(f"\r[{bar}] {current}/{total} ({progress * 100:.1f}%) {message}{time_info}", end='', flush=True)