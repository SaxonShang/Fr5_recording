#!/usr/bin/env python3
"""开工核对里的数值检查：只读录好的 save_data，不连机械臂、不开相机。只要 numpy，Box 和笔记本都能跑。

    python3 startup_checks.py camera  <save_data>           相机连续录制：丢弃队列、帧间隔、IMU 间隔
    python3 startup_checks.py gravity <save_data> [...]     静止段的比力换到基坐标系，和 +z 的夹角（底座是否水平）
    python3 startup_checks.py settle  <save_data> [...]     到位后陀螺幅值按时间分箱：到位后多久回到静止水平
    python3 startup_checks.py legs    <save_data>           多段路径每段的预计和实测（线速度、角速度、超速保护）

判据（实验记录的开工清单）：camera 丢弃队列都是 0、帧间隔 ≤ 40 ms、IMU 间隔 ≤ 10 ms；gravity 相对开发录制的基线
（偏离 +z 0.24°）变化 > 0.5° 就查底座是否水平，并告知评测端重估 g_w；settle 基线是 0.5–0.75 s 回到静止水平，明显变长就
加长最短停留；legs 长段线速度和预计差 ≤ 10%，角速度不超过预计的 1.5 倍。不通过时退出码为 1。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

import fr5_groundtruth as gt

CAMERA_IMU = Path(__file__).resolve().parent / "camera_imu.yaml"
MAX_FRAME_GAP_MS, MAX_IMU_GAP_MS = 40.0, 10.0
HOLDS = ("hold_start", "hold_before", "hold_after", "hold_end")
HOLD_SKIP_START_MS, HOLD_SKIP_END_MS = 800.0, 300.0      # 静止段开头等稳定、结尾离下一段远一点
BASELINE_TILT_DEG, MAX_TILT_CHANGE_DEG = 0.24, 0.5
SETTLE_BINS_S = (0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0)
SETTLED_FACTOR = 1.5                                     # 陀螺幅值 RMS 不超过静止水平的这么多倍算回到静止
BASELINE_SETTLE_S, SETTLE_MARGIN_S = 0.75, 0.25          # 恢复时间超过 基线 + 余量 算明显变长


def load_camera_imu(path: Path = CAMERA_IMU) -> np.ndarray:
    """camera_imu.yaml 的转动部分（v_cam = R · v_imu）。Box 上不保证有 PyYAML，这里直接读那 4 行。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = lines.index("camera_imu_transform:")
    return np.array([[float(v) for v in ln.strip(" -[]").split(",")] for ln in lines[i + 1:i + 5]])[:3, :3]


def load_motion_segments(save_data: Path) -> dict:
    log = json.loads((save_data / "motion_log.json").read_text(encoding="utf-8"))
    return {s["name"]: s for s in log["segments"] if s.get("start_box_ns")}


def load_imu_columns(save_data: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(时间 ms, 比力 N×3, 陀螺 N×3 °/s)。"""
    data = np.loadtxt(save_data / "zed_imu.csv", delimiter=",", skiprows=1, usecols=(0, 3, 4, 5, 6, 7, 8), ndmin=2)
    return data[:, 0] / 1e6, data[:, 1:4], data[:, 4:7]


def check_camera(save_data: Path) -> bool:
    with open(save_data / "img_timeline.csv", newline="", encoding="utf-8") as f:
        t_img = np.array([float(r["zed_timestamp_ms"]) for r in csv.DictReader(f)])
    t_imu = load_imu_columns(save_data)[0]
    frame_gap, imu_gap = float(np.diff(t_img).max()), float(np.diff(t_imu).max())
    log = save_data.parent / "zed_record.log"
    drops = [int(v) for v in re.findall(r"丢弃队列 (\d+)", log.read_text(encoding="utf-8"))] if log.exists() else None
    print(f"图像 {len(t_img)} 帧、{(t_img[-1] - t_img[0]) / 1000:.1f} s，帧间隔最大 {frame_gap:.1f} ms（门槛 {MAX_FRAME_GAP_MS:g}）；"
          f"IMU {len(t_imu)} 行，间隔最大 {imu_gap:.2f} ms（门槛 {MAX_IMU_GAP_MS:g}）；"
          + (f"丢弃队列 {drops}" if drops is not None else f"没有 {log.name}，查不了丢弃队列"))
    ok = frame_gap <= MAX_FRAME_GAP_MS and imu_gap <= MAX_IMU_GAP_MS and bool(drops) and not any(drops)
    print("通过" if ok else "不通过")
    return ok


def up_vectors(save_data: Path, r_cam_imu: np.ndarray) -> np.ndarray:
    """每个静止段的单位“上”方向（比力换到基坐标系）。"""
    segs = load_motion_segments(save_data)
    t_imu, accel, _ = load_imu_columns(save_data)
    tum = np.loadtxt(save_data / "groundtruth_tum.txt", comments="#", ndmin=2)
    out = []
    for name in HOLDS:
        if name not in segs or not segs[name].get("end_box_ns"):
            continue
        a, b = segs[name]["start_box_ns"] / 1e6 + HOLD_SKIP_START_MS, segs[name]["end_box_ns"] / 1e6 - HOLD_SKIP_END_MS
        i = int(np.argmin(np.abs(tum[:, 0] * 1000.0 - (a + b) / 2.0)))
        r = gt.quat_to_matrix(np.array([tum[i, 7], tum[i, 4], tum[i, 5], tum[i, 6]]))
        f = r @ r_cam_imu @ accel[(t_imu > a) & (t_imu < b)].mean(axis=0)
        out.append(f / np.linalg.norm(f))
    return np.array(out).reshape(-1, 3)


def check_gravity(paths: list[Path], camera_imu: Path = CAMERA_IMU) -> bool:
    r_cam_imu = load_camera_imu(camera_imu)
    ups = np.vstack([up_vectors(sd, r_cam_imu) for sd in paths])
    if not len(ups):
        print("没有可用的静止段")
        return False
    mean = ups.mean(axis=0)
    mean /= np.linalg.norm(mean)
    tilt = float(np.degrees(np.arccos(np.clip(mean[2], -1.0, 1.0))))
    spread = float(np.degrees(np.sqrt(np.mean(np.arccos(np.clip(ups @ mean, -1.0, 1.0)) ** 2))))
    ok = abs(tilt - BASELINE_TILT_DEG) <= MAX_TILT_CHANGE_DEG
    print(f"{len(paths)} 条录制、{len(ups)} 个静止段：重力反方向（基坐标系）{np.round(mean, 4).tolist()}，偏离 +z {tilt:.3f}°，"
          f"各静止段散布 {spread:.3f}°；基线 {BASELINE_TILT_DEG}°，变化 {tilt - BASELINE_TILT_DEG:+.3f}°")
    print("通过（评测端仍按新底座的静止段重估 g_w）" if ok else f"不通过：变化超过 {MAX_TILT_CHANGE_DEG}°，查底座是否水平，并告知评测端")
    return ok


def settle_profile(save_data: Path) -> tuple[np.ndarray, float, int]:
    """(到位后各时间箱的陀螺幅值 RMS °/s, hold_start 的静止水平 RMS °/s, 用到的运动段数)。"""
    segs = load_motion_segments(save_data)
    t_imu, _, gyro = load_imu_columns(save_data)
    hold = segs["hold_start"]
    sel = (t_imu > hold["start_box_ns"] / 1e6 + HOLD_SKIP_START_MS) & (t_imu < hold["end_box_ns"] / 1e6 - HOLD_SKIP_END_MS)
    bias = np.median(gyro[sel], axis=0)
    level = float(np.sqrt(np.mean(np.sum((gyro[sel] - bias) ** 2, axis=1))))
    norm = np.linalg.norm(gyro - bias, axis=1)
    moves = [s for n, s in segs.items() if (n in ("translate", "rotate") or n.startswith("leg_")) and s.get("arrived_box_ns")]
    edges = np.array(SETTLE_BINS_S)
    sums, counts = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for s in moves:
        rel = (t_imu - s["arrived_box_ns"] / 1e6) / 1000.0
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            k = (rel >= a) & (rel < b)
            sums[i] += float(np.sum(norm[k] ** 2))
            counts[i] += int(k.sum())
    return np.sqrt(sums / np.maximum(counts, 1)), level, len(moves)


def recovered_after(profile: np.ndarray, level: float) -> float | None:
    """从这个时间箱的左边界起，后面每个箱都不超过静止水平的 SETTLED_FACTOR 倍；到最后一箱都没回到就是 None。"""
    late = np.nonzero(profile > SETTLED_FACTOR * level)[0]
    if len(late) == 0:
        return float(SETTLE_BINS_S[0])
    return None if late[-1] == len(profile) - 1 else float(SETTLE_BINS_S[late[-1] + 1])


def check_settle(paths: list[Path]) -> bool:
    labels = [f"{a:g}–{b:g}" for a, b in zip(SETTLE_BINS_S[:-1], SETTLE_BINS_S[1:])]
    print("到位后时间箱（s）：" + "  ".join(labels))
    ok = True
    for sd in paths:
        profile, level, n = settle_profile(sd)
        after = recovered_after(profile, level)
        good = after is not None and after <= BASELINE_SETTLE_S + SETTLE_MARGIN_S
        ok &= good
        print(f"{sd.parent.name}（{n} 段，静止水平 {level:.2f} °/s）：" + "  ".join(f"{v:.2f}" for v in profile)
              + "；" + (f"{after:g} s 后回到静止水平" if after is not None else f"{SETTLE_BINS_S[-1]:g} s 内没回到静止水平"))
    print("通过" if ok else f"不通过：超过 {BASELINE_SETTLE_S + SETTLE_MARGIN_S:g} s，考虑加长最短停留")
    return ok


def check_legs(save_data: Path) -> bool:
    log = json.loads((save_data / "motion_log.json").read_text(encoding="utf-8"))
    legs = (log.get("waypoint_path") or {}).get("legs")
    if not legs:
        print("这条录制没有多段路径的记录")
        return False

    def fmt(v, spec=".1f"):
        return "—" if v is None else format(v, spec)

    print("段       类型  行程 mm  转角 °    预计 mm/s  实测均/峰     预计 °/s  实测均/峰     用时 s（预计）  保护阈值 mm/s、°/s  实时峰值")
    ok = True
    for leg in legs:
        over = leg.get("overspeed") or {}
        print(f"{leg['name']:<8} {leg['motion']:<4} {leg['distance_mm']:7.1f} {leg['angle_deg']:7.1f}  "
              f"{fmt(leg['expected_mm_s']):>9}  {fmt(leg['mean_mm_s']):>6}/{fmt(leg['max_mm_s']):<6} "
              f"{fmt(leg['expected_deg_s']):>8}  {fmt(leg['mean_deg_s']):>6}/{fmt(leg['max_deg_s']):<6} "
              f"{fmt(leg['duration_s'], '.2f'):>6}（{leg['planned_duration_s']:.2f}）  "
              f"{fmt(over.get('limit_mm_s'))}、{fmt(over.get('limit_deg_s'))}  {fmt(over.get('peak_mm_s'))}、{fmt(over.get('peak_deg_s'))}")
        for text in leg["notes"]:
            print(f"  注意：{leg['name']} {text}")
            ok = False
    print("通过" if ok else "不通过：看上面的注意")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, many in (("camera", False), ("gravity", True), ("settle", True), ("legs", False)):
        sp = sub.add_parser(name)
        sp.add_argument("save_data", type=Path, nargs="+" if many else None)
    sub.choices["gravity"].add_argument("--camera-imu", type=Path, default=CAMERA_IMU)
    args = parser.parse_args(argv)
    try:
        if args.command == "camera":
            ok = check_camera(args.save_data)
        elif args.command == "gravity":
            ok = check_gravity(args.save_data, args.camera_imu)
        elif args.command == "settle":
            ok = check_settle(args.save_data)
        else:
            ok = check_legs(args.save_data)
    except (OSError, ValueError, KeyError) as exc:
        print(f"失败：{exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
