# courtesy to https://stackoverflow.com/a/1094933
def sizeof_fmt(num, suffix="B", digits: int = 0, space: str = " "):
    # units = ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi")
    units = ("", "K", "M", "G", "T", "P", "E", "Z")
    for unit in units:
        if abs(num) < 1024.0:
            return f"{num:3.{digits}f}{space}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
