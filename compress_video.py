"""撮影した動画を、Web 版に上げられる大きさに縮める(解像度を下げて MPEG-4 に変換する)。

R10 の動画は 1920×1080・約115Mbps で、20分なら15GBを超える。光源の検知はもともと幅320に縮めて行い、
明るさも光源の枠の中の明るい画素から取るので、幅960(960×540)に下げても読み取りに要る情報は残る。
コマ数・コマの速さ(fps)は変えない(点滅の長さはコマ数で測るため)。
圧縮は OpenCV に入っている FFmpeg の MPEG-4(mp4v)。暗い部屋で固定して撮った動画なら20分でも数十MBになる。
(Windows の H.264 は画質を指定できず20分で2GBを超えたので使わない)

使い方: python compress_video.py <動画> ... [--width 960] [--out-dir <フォルダ>]
  出力は <元の名前>_small.mp4(--out-dir を指定したときは同じ名前でそのフォルダへ)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

DEFAULT_WIDTH = 960


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """MPEG-4(mp4v)で書く。OpenCV の pip 版には Windows・Linux とも FFmpeg が入っている。"""
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("動画を書き出せませんでした(OpenCV の FFmpeg が使えません)")
    return writer


def compress(src: Path, dst: Path, width: int = DEFAULT_WIDTH, progress=print) -> dict:
    """src を幅 width に縮めて dst に書き、コマ数と大きさを返す。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_w <= width:
        width = src_w
    height = round(src_h * width / src_w / 2) * 2  # 偶数の大きさにそろえる(圧縮方式によっては奇数を受け付けない)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    writer = open_writer(tmp, fps, (width, height))
    n, started, last = 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            n += 1
            if time.time() - last > 5:
                last = time.time()
                progress(f"  {src.name}: {n}/{total}コマ({n / max(total, 1):.0%})")
    finally:
        cap.release()
        writer.release()
    tmp.replace(dst)
    return {"frames": n, "fps": fps, "size": (width, height), "seconds": round(time.time() - started, 1),
            "src_bytes": src.stat().st_size, "dst_bytes": dst.stat().st_size}


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="動画を Web 版に上げられる大きさに縮める")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="縮めたあとの幅[ピクセル]")
    parser.add_argument("--out-dir", help="出力先のフォルダ(指定すると元と同じ名前で書く)")
    args = parser.parse_args()
    for v in args.videos:
        src = Path(v.strip('"'))
        dst = Path(args.out_dir) / src.name if args.out_dir else src.with_name(src.stem + "_small.mp4")
        print(f"変換: {src} → {dst}", flush=True)
        r = compress(src, dst, args.width, progress=lambda m: print(m, flush=True))
        print(f"  完了: {r['frames']}コマ、{r['size'][0]}×{r['size'][1]}、{r['src_bytes'] / 1e9:.2f}GB → "
              f"{r['dst_bytes'] / 1e6:.0f}MB({r['seconds']}秒)", flush=True)


if __name__ == "__main__":
    main()
