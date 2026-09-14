"""数据落地核查 · 一眼看清 data/raw 下有哪些可用数据（便于他人接手）

用法：
    PYTHONPATH=src python scripts/list_datasets.py

扫描 data/raw/（含 ms/ 下 ModelScope 数据），输出每个数据集的：
  · 体积、顶层结构、关键文件、是否完整（.incomplete 标记）
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def dir_size(path):
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total


def has_incomplete(path):
    for r, _, fs in os.walk(path):
        for f in fs:
            if f.endswith((".incomplete", ".part", ".tmp")):
                return True
    return False


def show(path, label):
    if not os.path.exists(path):
        return
    gb = dir_size(path) / 1e9
    inc = has_incomplete(path)
    flag = "⚠️ 未完成" if inc else "✅ 完整"
    print(f"\n=== {label}  ({gb:.2f} GB) {flag} ===")
    for e in sorted(os.listdir(path))[:10]:
        p = os.path.join(path, e)
        tag = "/" if os.path.isdir(p) else ""
        print(f"    {e}{tag}")


def main():
    print(f"数据根目录：{RAW}")
    if not os.path.exists(RAW):
        print("（不存在）"); return 1

    # 原始真实数据
    for name in sorted(os.listdir(RAW)):
        if name == "ms":
            continue
        p = os.path.join(RAW, name)
        if os.path.isdir(p):
            show(p, f"[已有真实数据] {name}")

    # ModelScope 数据
    ms = os.path.join(RAW, "ms")
    if os.path.isdir(ms):
        print("\n" + "=" * 60)
        print("ModelScope 数据集（scripts/fetch_datasets.py 下载）")
        print("=" * 60)
        for name in sorted(os.listdir(ms)):
            p = os.path.join(ms, name)
            if os.path.isdir(p):
                show(p, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
