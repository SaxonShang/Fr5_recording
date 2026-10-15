#!/usr/bin/env python3
"""由同一路径的标定板录制算端点位姿修正（fr5_groundtruth.END_POSE_CORRECTION），并检验（笔记本上运行）。

标定板固定不动时，真值乘 PnP 得到的 base_T_board 在起点和终点应该相同。真值终点还差的刚体变换
ΔT = B_起点 · B_终点⁻¹（基坐标系左乘）就是要补的量；运动中按进度 s 补 exp(s·log ΔT)。

    python -B end_pose_fit.py <save_data 目录> [<save_data 目录> ...]

给的几条录制必须是同一路径组、同一方向（例如两条 pitch_up --tag board）。每条先用 fr5_groundtruth 重算一份
不加端点修正的真值（临时目录，不改原数据），取 hold_before（起点）和 hold_after（终点）的静止帧算 ΔT；
几条取平均，打印可以填进 END_POSE_CORRECTION 的一行。检验：运动段每一帧的标定板中心离起点标定板中心的距离，
分别看不修正、用这条自己的 ΔT、用其他几条的平均（留一法，两条以上才有）——终点用自己的 ΔT 必然闭合，
中途和留一法的数才是检验。
"""

from __future__ import annotations

import argparse
import csv
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


def to_delta(t: np.ndarray) -> np.ndarray:
    """刚体变换 → [rx, ry, rz（°）, tx, ty, tz（mm）]。"""
    return np.r_[np.degrees(gt.quat_to_rotvec(gt.matrix_to_quat(t[:3, :3]))), t[:3, 3] * 1000.0]


def load_recording(save_data: Path, board: dict, det, k: np.ndarray, detect_board, centre_b: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        report = gt.process(save_data, HANDEYE, Path(tmp), end_correction=False)
        with open(Path(tmp) / "groundtruth_images.csv", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["flag"] == "ok"]
    info = report["end_pose_correction"]
    if "kind" not in info:
        raise ValueError(f"{save_data}：不是平移或旋转录制")
    flange_t_cam, _ = gt.load_handeye(HANDEYE)
    segs = {s["name"]: s for s in json.loads((save_data / "motion_log.json").read_text(encoding="utf-8"))["segments"]
            if s.get("start_box_ns")}

    def frames(name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        s = segs[name]
        if name == info["kind"]:
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
            out.append((t, c))
        return out

    start, end, move = frames("hold_before"), frames("hold_after"), frames(info["kind"])
    b0 = mean_pose([t @ c for t, c in start])
    b1 = mean_pose([t @ c for t, c in end])
    # 运动段和终点帧的进度（由相机真值换回法兰位姿）
    cams = [t for t, _ in move + end]
    flanges = [t @ np.linalg.inv(flange_t_cam) for t in cams]
    s = gt.end_pose_progress(np.array([f[:3, 3] for f in flanges]), gt.matrix_to_quat(np.array([f[:3, :3] for f in flanges])),
                             info, flange_t_cam)
    s[len(move):] = 1.0
    return {"name": save_data.parent.name, "direction": info["direction"], "path_set": info["path_set"],
            "delta": b0 @ np.linalg.inv(b1), "centre0": (b0 @ centre_b)[:3],
            "check": [(t, c, si) for (t, c), si in zip(move + end, s)], "move_frames": len(move)}


def residuals(rec: dict, delta: np.ndarray | None, centre_b: np.ndarray) -> np.ndarray:
    """每帧（进度, 标定板中心离起点的距离 mm）。delta 为 None 时不修正。"""
    out = []
    for t, c, s in rec["check"]:
        fix = np.eye(4) if delta is None else delta_power(delta, s)
        out.append((s, np.linalg.norm((fix @ t @ c @ centre_b)[:3] - rec["centre0"]) * 1000.0))
    return np.array(out)


def delta_power(delta: np.ndarray, s: float) -> np.ndarray:
    info = {"source": "END_POSE_CORRECTION", "delta_rotvec_deg_mm": to_delta(delta).tolist()}
    p, q = gt.apply_end_pose_correction(np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]), np.array([s]), info)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = gt.quat_to_matrix(q[0]), p[0]
    return m


def mean_delta(ds: list[np.ndarray]) -> np.ndarray:
    v = np.mean([to_delta(d) for d in ds], axis=0)
    m = np.eye(4)
    m[:3, :3] = gt.quat_to_matrix(gt.rotvec_to_quat(np.radians(v[:3])))
    m[:3, 3] = v[3:] / 1000.0
    return m


def summary_line(res: np.ndarray) -> str:
    bins = []
    for lo in (0.0, 0.25, 0.5, 0.75):
        sel = (res[:, 0] >= lo) & ((res[:, 0] < lo + 0.25) | (lo == 0.75))
        bins.append(f"{res[sel, 1].mean():4.2f}" if sel.any() else "  - ")
    return " / ".join(bins) + f"，最大 {res[:, 1].max():.2f} mm"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_data", type=Path, nargs="+", help="同一路径、同一方向的标定板录制的 save_data 目录")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(CALIB_REPO))
    from handeye.board import board_outline, detect_board, detector, read_board
    board = read_board(CALIB_REPO / TARGET_YAML)
    centre_b = np.r_[board_outline(board).mean(axis=0), 1.0]
    det, k = detector(), load_camera(CAMERA_YAML)
    try:
        recs = [load_recording(sd, board, det, k, detect_board, centre_b) for sd in args.save_data]
    except (OSError, ValueError, KeyError) as exc:
        print(f"失败：{exc}")
        return 1
    if len({(r["path_set"], r["direction"]) for r in recs}) != 1:
        print("失败：给的录制不是同一路径组、同一方向")
        return 1
    print("每条录制的 ΔT（转动向量 °，平移 mm，基坐标系）：")
    for r in recs:
        d = to_delta(r["delta"])
        print(f"  {r['name']}：[{', '.join(f'{v:+.4f}' for v in d[:3])}]°  [{', '.join(f'{v:+.2f}' for v in d[3:])}] mm")
    avg = mean_delta([r["delta"] for r in recs])
    print("\n运动段每帧标定板中心离起点的距离（进度 0–25 / 25–50 / 50–75 / 75–100%）：")
    for i, r in enumerate(recs):
        print(f"  {r['name']}（运动段 {r['move_frames']} 帧）")
        print(f"    不修正：        {summary_line(residuals(r, None, centre_b))}")
        print(f"    用自己的 ΔT：   {summary_line(residuals(r, r['delta'], centre_b))}")
        others = [x["delta"] for j, x in enumerate(recs) if j != i]
        if others:
            print(f"    用其他几条的平均：{summary_line(residuals(r, mean_delta(others), centre_b))}")
    d = to_delta(avg)
    print(f"\n填进 END_POSE_CORRECTION[\"{recs[0]['path_set']}\"]（{len(recs)} 条平均）：")
    print(f'    "{recs[0]["direction"]}": [{", ".join(f"{v:.4f}" for v in d[:3])}, {", ".join(f"{v:.2f}" for v in d[3:])}],')
    return 0


if __name__ == "__main__":
    sys.exit(main())
