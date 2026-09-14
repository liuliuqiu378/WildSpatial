"""多线程分块下载器（断点续传）

为什么需要它：本项目的数据集服务器（TUM 等）单连接速度常常只有 0.3~1 MB/s，
一个 344MB 的序列要下 1~2 小时。多线程分块（HTTP Range）通常能快数倍。

用法：
    python -m wildspatial.data.download <url> <out_path> [--workers 16]

或作为库：
    from wildspatial.data.download import parallel_download
    parallel_download(url, out, workers=16)
"""

import os
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK = 4 * 1024 * 1024   # 每块 4MB


def _get_file_size(url, timeout=20):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length", 0)), r.headers.get("Accept-Ranges")


def _download_range(url, start, end, out, idx, progress, lock):
    """下载 [start, end] 字节并写入文件的对应偏移。

    ⚠️ 关键：必须校验**实际写入字节数 == 期望字节数**。
    曾经踩过的坑：连接提前关闭时只写了一部分，而文件已 truncate 到完整大小，
    于是中间留下空洞 —— 文件大小看起来完全正确，但 tar 解压会失败。
    这种"静默损坏"比下载失败更危险。
    """
    expected = end - start + 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(4):
        written = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if getattr(r, "status", 206) not in (206, 200):
                    raise IOError(f"期望 206 Partial Content，实际 {r.status}")
                with open(out, "r+b") as f:
                    f.seek(start)
                    while written < expected:
                        buf = r.read(min(256 * 1024, expected - written))
                        if not buf:
                            break                      # 连接提前关闭 → 写少了
                        f.write(buf)
                        written += len(buf)
                        with lock:
                            progress[idx] += len(buf)
            if written == expected:
                return written
            # 没写满：回退进度后重试
            with lock:
                progress[idx] -= written
        except Exception:
            with lock:
                progress[idx] -= written
            if attempt == 3:
                raise
        time.sleep(1.5 * (attempt + 1))
    raise IOError(f"块 {idx} [{start},{end}] 重试 4 次仍不完整")


def parallel_download(url, out, workers=16, resume=True, verbose=True):
    """多线程分块下载，支持断点续传。

    返回下载的总字节数。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    total, accept_ranges = _get_file_size(url)

    if accept_ranges != "bytes" or total == 0:
        # 服务器不支持 Range，退回单线程
        if verbose:
            print(f"[!] 服务器不支持 Range 分块，退回单线程下载（total={total}）")
        urllib.request.urlretrieve(url, out)
        return os.path.getsize(out)

    # 预分配文件
    if not (resume and os.path.exists(out) and os.path.getsize(out) == total):
        with open(out, "wb") as f:
            f.truncate(total)

    blocks = []
    pos = 0
    while pos < total:
        end = min(pos + CHUNK - 1, total - 1)
        blocks.append((pos, end))
        pos = end + 1

    progress = [0] * len(blocks)
    lock = threading.Lock()

    if verbose:
        print(f"[↓] {os.path.basename(out)}  {total/1e6:.1f} MB  "
              f"{len(blocks)} 块 / {workers} 线程")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_range, url, s, e, out, i, progress, lock): i
                for i, (s, e) in enumerate(blocks)}
        for fu in as_completed(futs):
            fu.result()
            done += 1
            if verbose and done % 10 == 0:
                with lock:
                    got = sum(progress)
                sys.stdout.write(f"\r    {got/1e6:.1f}/{total/1e6:.1f} MB "
                                 f"({100*got/total:.0f}%)")
                sys.stdout.flush()
    if verbose:
        print(f"\n[✓] 完成: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return os.path.getsize(out)


def verify_archive(path):
    """校验归档文件是否完整（能完整遍历才说明没损坏）"""
    if path.endswith((".tgz", ".tar.gz")):
        import tarfile
        try:
            with tarfile.open(path) as tf:
                for _ in tf:
                    pass
            return True
        except Exception:
            return False
    if path.endswith(".zip"):
        import zipfile
        try:
            return zipfile.ZipFile(path).testzip() is None
        except Exception:
            return False
    return None      # 未知类型，跳过


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    u, o = sys.argv[1], sys.argv[2]
    w = 16
    if "--workers" in sys.argv:
        w = int(sys.argv[sys.argv.index("--workers") + 1])
    parallel_download(u, o, workers=w)

    v = verify_archive(o)
    if v is True:
        print("[✓] 归档完整性校验通过")
    elif v is False:
        print("[✗] 归档损坏！请删除后重新下载")
        sys.exit(2)
