#!/usr/bin/env python3
"""由同一路径的带板录制算路径点修正，写进 waypoint_corrections.json，并做留一检验（笔记本上运行）。

标定板固定不动时，不加修正的真值乘 PnP 得到的 base_T_board（B_k）在每个路径点上应该相同。路径点 k 的修正
ΔT_k = B_0 · B_k⁻¹（基坐标系左乘，ΔT_0 = I）。第 k 段内按进度 s 插值：exp(s·log(ΔT_{k+1} ΔT_k⁻¹)) ΔT_k（fr5_groundtruth）。
每次到访单独一个 ΔT，往返经过同一位置也按来向分开。

    python -B end_pose_fit.py <save_data 目录> [<save_data 目录> ...] [--write]

给的几条必须是同一路径组、同一修正键（trans_level_first 和 trans_tilt_first 共用 trans_two_attitude，按到达点名字对齐）。
每条先用 fr5_groundtruth 重算一份不加修正的真值（临时目录，不改原数据），取 hold_before、各 wp_k 和 hold_after 的静止帧算 ΔT_k。
检验：运动帧和路径点静止帧的标定板偏离，分别看不修正、用这条自己的 ΔT、用其他几条的平均（留一，两条以上才有）——路径点用自己的 ΔT
必然闭合，运动中和留一的数才是检验。--write：把这几条的平均写进 waypoint_corrections.json（同一修正键原有的内容会被替换）。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # 标定仓库约定不留 __pycache__

import numpy as np  # noqa: E402

import fr5_groundtruth as gt  # noqa: E402
from gt_board_check import CALIB_REPO, CAMERA_YAML, TARGET_YAML, MIN_TAGS, board_in_camera, load_camera  # noqa: E402

HANDEYE = Path(__file__).resolve().parent / "handeye_s003_result.json"
HOLD_FRAMES = 15        # 每个静止段取这么多帧（段的中间一半里均匀取）


def mean_pose(ts: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in ts))
    m = np.eye(4)
    m[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    m[:3, 3] = np.mean([t[:3, 3] for t in ts], axis=0)
    return m


def delta_power(delta: np.ndarray, s: float) -> np.ndarray:
    """exp(s·log delta)。"""
    info = {"source": "END_POSE_CORRECTION", "delta_rotvec_deg_mm": gt.matrix_to_delta(delta).tolist()}
    p, q = gt.apply_end_pose_correction(np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]), np.array([s]), info)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = gt.quat_to_matrix(q[0]), p[0]
    return m


def correction(deltas: list[np.ndarray], k: int, s: float) -> np.ndarray:
    """第 k 段、进度 s 处的修正 exp(s·log(ΔT_{k+1} ΔT_k⁻¹)) ΔT_k。"""
    a, b = deltas[k], deltas[k + 1]
    return delta_power(b @ np.linalg.inv(a), s) @ a


def mean_delta(ds: list[np.ndarray]) -> np.ndarray:
    return gt.delta_to_matrix(np.mean([gt.matrix_to_delta(d) for d in ds], axis=0))


def fit_stops(stops: list[list[tuple[np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """每个路径点的静止帧 [(base_T_camera, camera_T_board), ...] → (B_0, [ΔT_0, ΔT_1, ...])。"""
    boards = [mean_pose([t @ c for t, c in frames]) for frames in stops]
    return boards[0], [boards[0] @ np.linalg.inv(b) for b in boards]


def load_recording(save_data: Path, board: dict, det, k: np.ndarray, detect_board) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        report = gt.process(save_data, HANDEYE, Path(tmp), end_correction=False)
        with open(Path(tmp) / "groundtruth_images.csv", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["flag"] == "ok"]
    info = report.get("waypoint_correction", {})
    if not info.get("legs"):
        raise ValueError(f"{save_data}：不是路径表里的多段录制")
    if info["correction_key"] is None:
        raise ValueError(f"{save_data}：{info['path']} 这条路径不做路径点修正")
    flange_t_cam, _ = gt.load_handeye(HANDEYE)
    segs = {s["name"]: s for s in json.loads((save_data / "motion_log.json").read_text(encoding="utf-8"))["segments"]
            if s.get("start_box_ns")}

    def frames(name: str, moving: bool = False) -> list[tuple[float, np.ndarray, np.ndarray]]:
        s = segs[name]
        if moving:
            a, b, n = s["moving_box_ns"] / 1e6, s["arrived_box_ns"] / 1e6, 10 ** 6
        else:
            a, b, n = s["start_box_ns"] / 1e6, s["end_box_ns"] / 1e6, HOLD_FRAMES
            a, b = a + 0.25 * (b - a), b - 0.25 * (b - a)
        sel = [r for r in rows if a <= float(r["zed_timestamp_ms"]) <= b]
        out = []
        for r in sel[:: max(1, len(sel) // n)]:
            c, tags, _ = board_in_camera(save_data / r["filename_left"], board, det, k, detect_board)
            if c is None or tags < MIN_TAGS:
                continue
            t = np.eye(4)
            t[:3, 3] = [float(r[x]) for x in ("tx_m", "ty_m", "tz_m")]
            t[:3, :3] = gt.quat_to_matrix(np.array([float(r[x]) for x in ("qw", "qx", "qy", "qz")]))
            out.append((float(r["zed_timestamp_ms"]), t, c))
        return out

    n = len(info["legs"])
    names = ["hold_before"] + [f"wp_{i}" for i in range(1, n)] + ["hold_after"]
    stops = [frames(name) for name in names]
    empty = [name for name, f in zip(names, stops) if not f]
    if empty:
        raise ValueError(f"{save_data}：这些路径点看不到标定板（至少 {MIN_TAGS} 个 tag）：{empty}")
    b0, deltas = fit_stops([[(t, c) for _, t, c in f] for f in stops])

    # 运动帧的段号和进度（由相机真值换回法兰位姿）
    move = [x for j in range(n) for x in frames(info["legs"][j]["name"], moving=True)]
    check = []
    if move:
        ts = np.array([m[0] for m in move])
        flanges = [t @ np.linalg.inv(flange_t_cam) for _, t, _ in move]
        leg, s = gt.waypoint_progress(ts, np.array([f[:3, 3] for f in flanges]),
                                      gt.matrix_to_quat(np.array([f[:3, :3] for f in flanges])), info)
        check = [(t, c, int(j), float(si)) for (_, t, c), j, si in zip(move, leg, s)]
    stop_check = [(t, c, max(i - 1, 0), 1.0 if i else 0.0) for i, f in enumerate(stops) for _, t, c in f]
    return {"name": save_data.parent.name, "path": info["path"], "path_set": info["path_set"], "key": info["correction_key"],
            "labels": info["labels"], "deltas": dict(zip(info["labels"], deltas)), "b0": b0, "moving": check, "stops": stop_check}


def errors(frames: list, b0: np.ndarray, deltas: list[np.ndarray] | None, centre_b: np.ndarray) -> np.ndarray:
    """每帧 (标定板中心离 B_0 的距离 mm, 标定板姿态离 B_0 的转角 °)。deltas 为 None 时不修正。"""
    out = []
    centre0 = (b0 @ centre_b)[:3]
    for t, c, k, s in frames:
        board = (np.eye(4) if deltas is None else correction(deltas, k, s)) @ t @ c
        turn = np.degrees(gt.quat_to_rotvec(gt.matrix_to_quat(board[:3, :3] @ b0[:3, :3].T)))
        out.append((np.linalg.norm((board @ centre_b)[:3] - centre0) * 1000.0, np.linalg.norm(turn)))
    return np.array(out).reshape(-1, 2)


def summary_line(moving: np.ndarray, stops: np.ndarray) -> str:
    move = (f"运动帧 RMS {np.sqrt(np.mean(moving[:, 0] ** 2)):.2f}、最大 {moving[:, 0].max():.2f} mm，"
            f"姿态 RMS {np.sqrt(np.mean(moving[:, 1] ** 2)):.3f}°；") if len(moving) else "没有运动帧；"
    return move + f"路径点最大 {stops[:, 0].max():.2f} mm、{stops[:, 1].max():.3f}°"


def write_table(recs: list[dict], average: dict[str, np.ndarray], path: Path) -> None:
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    start = recs[0]["labels"][0]
    values = {label: [round(float(v), 5) for v in gt.matrix_to_delta(d)] for label, d in average.items() if label != start}
    table.setdefault(recs[0]["path_set"], {})[recs[0]["key"]] = {
        "fitted_at": dt.datetime.now().isoformat(timespec="seconds"), "fitted_from": [r["name"] for r in recs], "deltas": values}
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_data", type=Path, nargs="+", help="同一路径（同一修正键）的带板录制的 save_data 目录")
    parser.add_argument("--write", action="store_true", help=f"把平均结果写进 {gt.WAYPOINT_CORRECTIONS.name}")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(CALIB_REPO))
    from handeye.board import board_outline, detect_board, detector, read_board
    board = read_board(CALIB_REPO / TARGET_YAML)
    centre_b = np.r_[board_outline(board).mean(axis=0), 1.0]
    det, k = detector(), load_camera(CAMERA_YAML)
    try:
        recs = [load_recording(sd, board, det, k, detect_board) for sd in args.save_data]
    except (OSError, ValueError, KeyError) as exc:
        print(f"失败：{exc}")
        return 1
    if len({(r["path_set"], r["key"]) for r in recs}) != 1:
        print("失败：给的录制不是同一路径组、同一修正键")
        return 1
    if len({frozenset(r["labels"][1:]) for r in recs}) != 1:
        print("失败：给的录制的到达点不一致（有录制中途中止？）")
        return 1
    labels = recs[0]["labels"][1:]
    average = {recs[0]["labels"][0]: np.eye(4), **{x: mean_delta([r["deltas"][x] for r in recs]) for x in labels}}
    print(f"修正键 {recs[0]['key']}，{len(recs)} 条录制，{len(labels)} 个路径点。ΔT（转动向量 °，平移 mm，基坐标系，多条平均）：")
    for x in labels:
        d = gt.matrix_to_delta(average[x])
        print(f"  {x:<20} [{', '.join(f'{v:+.4f}' for v in d[:3])}]°  [{', '.join(f'{v:+.2f}' for v in d[3:])}] mm")
    print("\n标定板偏离（相对这条录制的 B_0）：")
    for i, r in enumerate(recs):
        print(f"  {r['name']}（{r['path']}，运动帧 {len(r['moving'])}）")
        for text, deltas in [("不修正", None), ("用自己的 ΔT", [r["deltas"][x] for x in r["labels"]])]:
            print(f"    {text}：{summary_line(errors(r['moving'], r['b0'], deltas, centre_b), errors(r['stops'], r['b0'], deltas, centre_b))}")
        others = [x for j, x in enumerate(recs) if j != i]
        if others:
            deltas = [np.eye(4)] + [mean_delta([o["deltas"][x] for o in others]) for x in r["labels"][1:]]
            print(f"    用其他几条的平均（留一）：{summary_line(errors(r['moving'], r['b0'], deltas, centre_b), errors(r['stops'], r['b0'], deltas, centre_b))}")
    if args.write:
        write_table(recs, average, gt.WAYPOINT_CORRECTIONS)
        print(f"\n已写进 {gt.WAYPOINT_CORRECTIONS}（修正键 {recs[0]['key']}）")
    else:
        print("\n没有写文件；确认后加 --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
