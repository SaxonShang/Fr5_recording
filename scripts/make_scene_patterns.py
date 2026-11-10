#!/usr/bin/env python3
"""生成 S 形场景 B、C 的打印图案（笔记本上运行，要 OpenCV）。输出 8 位灰度 PNG，已写入实际打印密度，按 100% 尺寸打印。

    python -B make_scene_patterns.py <输出目录> [--sheets 2] [--period-mm 20]

B：非重复的多尺度随机纹理，几张板连成一整幅，没有重复的区块。
C：严格的二维周期图案。每个周期里一大一小两个方块，只沿 x、y 平移才和自身重合，周期就是 period，没有半周期混叠。
板宽、板高都是周期的整数倍，拼缝处相位连续。B、C 的平均灰度接近（手动曝光不变，画面亮度才相近）。

默认两张 1.2 × 2.4 m 的板，距 S 点光心 0.8 m：相机到板的距离约 0.55–0.86 m，算法工作分辨率（fx ≈ 343.6 px）下 20 mm 的周期是 8–13 px，
落在图块匹配的 ±15 px 搜索窗内。三张板、距 1.0 m 时用 --sheets 3 --period-mm 30。

用显示屏代替打印：--display 3840x2160 --display-width-mm 1660 --period-mm 14，输出和屏幕分辨率一致的两张图，全屏 1:1 显示。
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PX_PER_MM = 5                    # 127 dpi：周期和板宽都是整数像素
SHEET_MM = (1200, 2400)          # 单张板 宽、高
BLACK, WHITE = 16, 239           # 留一点余量给打印机，B、C 用同一对灰度
OCTAVES = ((2.0, 1.0), (4.0, 1.0), (8.0, 0.4))   # B 的纹理：（噪声网格 mm，权重）。细尺度为主，大尺度多了图块在 ±15 px 内容易错配
SEED = 20261109


def band(rng: np.random.Generator, height: int, width: int, spacing: int) -> np.ndarray:
    """特征尺度约 spacing 像素的带限噪声（单位方差），由粗网格的白噪声平滑后放大得到，起点随机，各层网格不对齐。"""
    rows, cols = height // spacing + 4, width // spacing + 4
    grid = cv2.GaussianBlur(rng.standard_normal((rows, cols), dtype=np.float32), (0, 0), 0.7)
    big = cv2.resize(grid, (cols * spacing, rows * spacing), interpolation=cv2.INTER_CUBIC)
    y, x = (int(v) for v in rng.integers(0, spacing * 2, 2))
    out = big[y:y + height, x:x + width]
    return out / out.std()


def random_texture(height: int, width: int, seed: int = SEED, px_per_mm: float = PX_PER_MM) -> np.ndarray:
    """场景 B：多尺度随机纹理，直方图拉平到 [BLACK, WHITE]，整幅没有重复。"""
    rng = np.random.default_rng(seed)
    field = np.zeros((height, width), np.float32)
    for size_mm, weight in OCTAVES:
        field += weight * band(rng, height, width, max(2, int(size_mm * px_per_mm)))
    levels = np.percentile(field[::16, ::16], np.linspace(0.0, 100.0, 1025))
    return np.interp(field, levels, np.linspace(BLACK, WHITE, 1025)).astype(np.uint8)


def periodic_tile(period: int) -> np.ndarray:
    """场景 C 的一个周期：一大一小两个黑方块，黑色约占一半，平均灰度和 B 接近。"""
    tile = np.full((period, period), WHITE, np.uint8)
    big0, big1, small0, small1 = (round(period * v) for v in (0.05, 0.70, 0.72, 0.97))
    tile[big0:big1, big0:big1] = BLACK
    tile[small0:small1, small0:small1] = BLACK
    return tile


def write_png(path: Path, image: np.ndarray, px_per_mm: float = PX_PER_MM) -> None:
    """写 PNG，并加 pHYs 块记下物理密度，打印软件按它取 100% 尺寸。"""
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f"编码失败：{path}")
    raw = data.tobytes()
    density = round(px_per_mm * 1000)
    payload = b"pHYs" + struct.pack(">IIB", density, density, 1)
    chunk = struct.pack(">I", 9) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    path.write_bytes(raw[:33] + chunk + raw[33:])          # 8 字节文件头 + 25 字节 IHDR 之后


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="输出目录")
    parser.add_argument("--sheets", type=int, default=2, help="板的张数（竖放，并排）")
    parser.add_argument("--period-mm", type=int, default=20, help="场景 C 的周期 mm")
    parser.add_argument("--display", help="显示屏分辨率，例如 3840x2160；给了就只生成屏幕用的两张图")
    parser.add_argument("--display-width-mm", type=float, default=1660.0, help="屏幕显示区宽度 mm")
    args = parser.parse_args(argv)
    if args.display:
        args.output.mkdir(parents=True, exist_ok=True)
        width, height = (int(v) for v in args.display.lower().split("x"))
        px_per_mm = width / args.display_width_mm
        period = round(args.period_mm * px_per_mm)
        tile = periodic_tile(period)
        images = {"B_random_display.png": random_texture(height, width, px_per_mm=px_per_mm),
                  "C_periodic_display.png": np.tile(tile, (height // period + 1, width // period + 1))[:height, :width]}
        for name, image in images.items():
            write_png(args.output / name, image, px_per_mm)
            print(f"  {name}  {width}×{height} px，平均灰度 {image.mean():.1f}")
        print(f"屏幕 {px_per_mm:.3f} px/mm，C 的周期 {period} px = {period / px_per_mm:.1f} mm")
        return 0
    sheet_w, sheet_h = SHEET_MM
    if sheet_w % args.period_mm or sheet_h % args.period_mm:
        parser.error(f"板的尺寸 {SHEET_MM} mm 要是周期 {args.period_mm} mm 的整数倍，否则拼缝处相位不连续")
    args.output.mkdir(parents=True, exist_ok=True)
    width, height = sheet_w * PX_PER_MM, sheet_h * PX_PER_MM
    b = random_texture(height, width * args.sheets)
    tile = periodic_tile(args.period_mm * PX_PER_MM)
    c = np.tile(tile, (args.period_mm and sheet_h // args.period_mm, args.sheets * sheet_w // args.period_mm))
    print(f"B 平均灰度 {b[::16, ::16].mean():.1f}，C 平均灰度 {tile.mean():.1f}；每张 {width}×{height} px，{sheet_w}×{sheet_h} mm，{PX_PER_MM * 25.4:.0f} dpi")
    for i in range(args.sheets):
        for name, image in (("B_random", b), ("C_periodic", c)):
            path = args.output / f"{name}_sheet{i + 1}of{args.sheets}.png"
            write_png(path, image[:, i * width:(i + 1) * width])
            print(f"  {path.name}  {path.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
