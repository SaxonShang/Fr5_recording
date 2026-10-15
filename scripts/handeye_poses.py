#!/usr/bin/env python3
"""给眼在手上标定（手眼外参）选机械臂位姿（笔记本上运行，不连机械臂）。

标定板固定不动。围绕它生成候选相机位姿：离板距离、视线相对板法线的倾角和方位、瞄准点在板面上的偏移、
绕光轴滚转。每个候选用 FR5 名义运动学（标定仓库 handeye.kinematics，和控制器差 < 0.1 mm）从起点关节角
数值逆解，只留下：
- 关节在允许范围内：给定录制里实际走过的关节范围，每边收进 JOINT_MARGIN_DEG（所以构型也和这些录制相同）；
- 法兰在这些录制走过的空间（法兰位置的外接盒）内，每边收进 FLANGE_MARGIN_MM；
- J3、J5 离 0（奇异位形）至少 SINGULAR_MARGIN_DEG；
- 整块板都在画面里（离边缘至少 EDGE_MARGIN_PX），tag 至少 MIN_TAG_PX。
再按相机姿态、板在画面里的位置和距离的差异挑出 --count 个位姿和 --reserve 个备用（最远点采样），
从起点按关节角就近排顺序，并检查相邻两个位姿之间 MoveJ（关节空间直线插值）经过的法兰位置也在外接盒内。
输出 JSON 给 handeye_capture.py。

    python -B handeye_poses.py --board board.json --recordings <save_data 目录> ... --output poses.json

board.json：{"base_T_board": 4×4（米，板坐标系同标定仓库：tag 0 左下角为原点，z 朝外），"start_joints": [6 个 °]}。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 标定仓库约定不留 __pycache__

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import fr5_groundtruth as gt  # noqa: E402
from gt_board_check import CALIB_REPO, CAMERA_YAML, TARGET_YAML, load_camera  # noqa: E402

HANDEYE = Path(__file__).resolve().parent / "handeye_s003_result.json"
IMAGE_W, IMAGE_H = 1920, 1080
JOINT_MARGIN_DEG = 3.0          # 关节范围每边收进
FLANGE_MARGIN_MM = 20.0         # 法兰外接盒每边收进
SINGULAR_MARGIN_DEG = 5.0
EDGE_MARGIN_PX = 40.0           # 整块板离画面边缘至少
MIN_TAG_PX = 28.0               # tag 边长至少
DISTANCES_M = (0.80, 0.90, 1.00, 1.10)
TILTS_DEG = (0.0, 10.0, 20.0, 30.0, 40.0)
AZIMUTHS_DEG = tuple(range(0, 360, 45))
AIM_OFFSETS_M = tuple((a, b) for a in (0.0, -0.22, -0.12, 0.12, 0.22) for b in (0.0, -0.35, -0.2, 0.2, 0.35))   # 瞄准点在板面上的偏移（板坐标 x、y，m），让板出现在画面各处
ROLLS_DEG = (-40.0, -20.0, 0.0, 20.0, 40.0)
PATH_SAMPLES = 20               # 相邻位姿之间 MoveJ 路径的采样点数


def envelope(save_data_dirs: list[Path]) -> dict:
    """录制里实际走过的关节范围和法兰位置外接盒（各收进一点）。"""
    joints, flange = [], []
    for d in save_data_dirs:
        with open(d / "fr5_state.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                joints.append([float(r[f"j{i}_deg"]) for i in range(1, 7)])
                flange.append([float(r[k]) for k in ("x_mm", "y_mm", "z_mm")])
    joints, flange = np.array(joints), np.array(flange)
    return {"recordings": [str(d) for d in save_data_dirs],
            "joint_min_deg": (joints.min(axis=0) + JOINT_MARGIN_DEG).round(1).tolist(),
            "joint_max_deg": (joints.max(axis=0) - JOINT_MARGIN_DEG).round(1).tolist(),
            "flange_min_mm": (flange.min(axis=0) + FLANGE_MARGIN_MM).round(0).tolist(),
            "flange_max_mm": (flange.max(axis=0) - FLANGE_MARGIN_MM).round(0).tolist()}


def look_at(cam: np.ndarray, aim: np.ndarray, roll_deg: float) -> np.ndarray:
    """相机位姿：光轴指向 aim，roll = 0 时相机 y（下）尽量朝世界 −z，再绕光轴转 roll。"""
    z = (aim - cam) / np.linalg.norm(aim - cam)
    down = np.array([0.0, 0.0, -1.0])
    y0 = down - (down @ z) * z
    y0 /= np.linalg.norm(y0)
    x0 = np.cross(y0, z)
    a = np.radians(roll_deg)
    x = np.cos(a) * x0 + np.sin(a) * y0
    t = np.eye(4)
    t[:3, :3] = np.c_[x, np.cross(z, x), z]
    t[:3, 3] = cam
    return t


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", type=Path, required=True, help="board.json（标定板位姿和起点关节角）")
    parser.add_argument("--recordings", type=Path, nargs="+", required=True, help="定允许范围用的录制 save_data 目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=25, help="位姿个数")
    parser.add_argument("--reserve", type=int, default=5, help="备用位姿个数")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(CALIB_REPO))
    from handeye.board import board_outline, read_board, tag_object_corners
    from handeye.kinematics import fk
    from handeye.geometry import fr5_from_matrix, inverse, rotation_vector

    info = json.loads(args.board.read_text(encoding="utf-8"))
    b_t = np.array(info["base_T_board"])
    q_start = np.array(info["start_joints"], dtype=float)
    board = read_board(CALIB_REPO / TARGET_YAML)
    corners_b = np.array([tag_object_corners(i, board) for i in range(board["tagCols"] * board["tagRows"])])
    corners = (b_t[:3, :3] @ corners_b.reshape(-1, 3).T).T + b_t[:3, 3]                     # 144×3（基坐标）
    centre = (b_t @ np.r_[board_outline(board).mean(axis=0), 1.0])[:3]
    e1, e2, normal = b_t[:3, 0], b_t[:3, 1], b_t[:3, 2]
    k = load_camera(CAMERA_YAML)
    x_he, _ = gt.load_handeye(HANDEYE)
    env = envelope(args.recordings)
    lo, hi = np.array(env["joint_min_deg"]), np.array(env["joint_max_deg"])
    fmin, fmax = np.array(env["flange_min_mm"]) / 1000, np.array(env["flange_max_mm"]) / 1000
    if np.any(q_start < lo - JOINT_MARGIN_DEG) or np.any(q_start > hi + JOINT_MARGIN_DEG):
        print("警告：起点关节角不在录制走过的范围内")

    def solve(target: np.ndarray, seeds: list[np.ndarray]) -> np.ndarray | None:
        def residual(q):
            d = inverse(target) @ fk(q)
            return np.r_[d[:3, 3] * 1000.0, rotation_vector(d[:3, :3]) * 1000.0]
        for seed in seeds:
            fit = least_squares(residual, np.clip(seed, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
            if np.max(np.abs(fit.fun)) < 1e-3:
                return fit.x
        return None

    def view(t_cam: np.ndarray) -> dict | None:
        pc = (t_cam[:3, :3].T @ (corners - t_cam[:3, 3]).T).T
        if np.any(pc[:, 2] < 0.1):
            return None
        u = k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2]
        v = k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]
        margin = float(min(u.min(), IMAGE_W - u.max(), v.min(), IMAGE_H - v.max()))
        uv = np.c_[u, v].reshape(-1, 4, 2)
        tag_px = float(np.linalg.norm(uv - np.roll(uv, 1, axis=1), axis=2).min())
        c = t_cam[:3, :3].T @ (centre - t_cam[:3, 3])
        return {"edge_margin_px": margin, "min_tag_px": tag_px,
                "board_centre_px": [float(k[0, 0] * c[0] / c[2] + k[0, 2]), float(k[1, 1] * c[1] / c[2] + k[1, 2])],
                "hull": uv.reshape(-1, 2)}

    # 1. 候选
    cands, solved = [], []
    for d in DISTANCES_M:
        for tilt in TILTS_DEG:
            for az in (AZIMUTHS_DEG if tilt > 0 else (0,)):
                a, t_r = np.radians(az), np.radians(tilt)
                cam = centre + d * (np.cos(t_r) * normal + np.sin(t_r) * (np.cos(a) * e1 + np.sin(a) * e2))
                for ox, oy in AIM_OFFSETS_M:
                    for roll in ROLLS_DEG:
                        t_cam = look_at(cam, centre + ox * e1 + oy * e2, roll)
                        vw = view(t_cam)
                        if vw is None or vw["edge_margin_px"] < EDGE_MARGIN_PX or vw["min_tag_px"] < MIN_TAG_PX:
                            continue
                        t_f = t_cam @ inverse(x_he)
                        if np.any(t_f[:3, 3] < fmin) or np.any(t_f[:3, 3] > fmax):
                            continue
                        near = sorted(solved, key=lambda s: np.linalg.norm(s[0] - t_f[:3, 3]))[:3]
                        q = solve(t_f, [q_start] + [s[1] for s in near])
                        if q is None or min(abs(q[2]), abs(q[4])) < SINGULAR_MARGIN_DEG:
                            continue
                        solved.append((t_f[:3, 3], q))
                        cands.append({"t_cam": t_cam, "t_f": t_f, "q": q, "distance_m": d, "tilt_deg": tilt,
                                      "azimuth_deg": float(az), "aim_offset_m": [ox, oy], "roll_deg": roll, **vw})
    print(f"候选 {len(cands)} 个可行；按倾角：" + "，".join(f"{tl:.0f}° {sum(c['tilt_deg'] == tl for c in cands)}" for tl in TILTS_DEG))
    if len(cands) < args.count + args.reserve:
        print("可行候选不够")
        return 1

    # 2. 最远点采样：相机姿态（相对正对板的姿态，°）、板在画面中的位置（°）、距离（10 cm ≈ 10°）
    r_ref = look_at(centre + 0.95 * normal, centre, 0.0)[:3, :3]
    feats = np.array([np.r_[np.degrees(rotation_vector(r_ref.T @ c["t_cam"][:3, :3])),
                            np.degrees(np.arctan((np.array(c["board_centre_px"]) - k[:2, 2]) / k[0, 0])),
                            (c["distance_m"] - 0.95) * 100.0] for c in cands])
    chosen = [int(np.argmin(np.linalg.norm(feats, axis=1)))]
    dmin = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < args.count + args.reserve:
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(feats - feats[i], axis=1))
    main_set, reserve = chosen[:args.count], chosen[args.count:]

    # 3. 从起点按关节角就近排顺序（备用的排在后面）
    def path_out(q0: np.ndarray, q1: np.ndarray) -> int:
        """q0 → q1 MoveJ（关节直线插值）经过的法兰位置有几个采样点出了外接盒。"""
        path = [fk(q0 + (q1 - q0) * s)[:3, 3] for s in np.linspace(0, 1, PATH_SAMPLES)]
        return sum(bool(np.any(p < fmin - 1e-9) or np.any(p > fmax + 1e-9)) for p in path)

    def order(idx: list[int], q0: np.ndarray) -> list[int]:
        """从 q0 起每次走关节移动最小、并且 MoveJ 路径不出外接盒的下一个位姿（都出界时才走出界最少的）。"""
        out, left, q = [], list(idx), q0
        while left:
            j = min(left, key=lambda i: (path_out(q, cands[i]["q"]), np.abs(cands[i]["q"] - q).max()))
            out.append(j)
            left.remove(j)
            q = cands[j]["q"]
        return out
    seq = order(main_set, q_start)
    seq += order(reserve, cands[seq[-1]]["q"])

    # 4. 相邻位姿之间 MoveJ 路径（关节直线插值）经过的法兰位置；以及标定质量的粗略指标
    problems, poses, q_prev = [], [], q_start
    for n, i in enumerate(seq, start=1):
        c = cands[i]
        out = path_out(q_prev, c["q"])
        if out:
            problems.append(f"位姿 {n}：从上一个位姿 MoveJ 过来，法兰有 {out}/{PATH_SAMPLES} 个采样点出了外接盒")
        poses.append({"index": n, "role": "pose" if n <= args.count else "reserve",
                      "flange_mm_deg": fr5_from_matrix(c["t_f"]), "joints_deg": np.round(c["q"], 3).tolist(),
                      "max_joint_move_deg": round(float(np.abs(c["q"] - q_prev).max()), 1),
                      "camera_m": np.round(c["t_cam"][:3, 3], 4).tolist(), "distance_m": c["distance_m"],
                      "tilt_deg": c["tilt_deg"], "azimuth_deg": c["azimuth_deg"], "aim_offset_m": c["aim_offset_m"],
                      "roll_deg": c["roll_deg"], "board_centre_px": np.round(c["board_centre_px"], 0).tolist(),
                      "edge_margin_px": round(c["edge_margin_px"], 0), "min_tag_px": round(c["min_tag_px"], 1)})
        q_prev = c["q"]
    rots = [cands[i]["t_cam"][:3, :3] for i in seq[:args.count]]
    rel = [np.degrees(rotation_vector(a.T @ b)) for j, a in enumerate(rots) for b in rots[j + 1:]]
    sv = np.linalg.svd(np.array(rel), compute_uv=False)
    grid = np.zeros((IMAGE_H // 20, IMAGE_W // 20), bool)
    for i in seq[:args.count]:
        hull = cands[i]["hull"]
        u0, v0 = np.clip(hull.min(axis=0) // 20, 0, None).astype(int)
        u1, v1 = (hull.max(axis=0) // 20).astype(int)
        grid[v0:v1 + 1, u0:u1 + 1] = True
    quality = {"max_relative_rotation_deg": round(float(max(np.linalg.norm(rel, axis=1))), 1),
               "relative_rotation_singular_values": np.round(sv, 0).tolist(),
               "distance_m": [min(p["distance_m"] for p in poses), max(p["distance_m"] for p in poses)],
               "tilt_deg": [min(p["tilt_deg"] for p in poses), max(p["tilt_deg"] for p in poses)],
               "roll_deg": [min(p["roll_deg"] for p in poses), max(p["roll_deg"] for p in poses)],
               "image_coverage_bbox": round(float(grid.mean()), 2)}
    result = {"board": info, "handeye": str(HANDEYE), "camera": str(CAMERA_YAML), "envelope": env,
              "start_joints": q_start.tolist(), "check_pose": 1, "quality": quality, "problems": problems,
              "poses": poses}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"选了 {args.count} 个位姿 + {args.reserve} 个备用，写进 {args.output}")
    print(f"允许范围：关节 {env['joint_min_deg']} ～ {env['joint_max_deg']}；法兰 {env['flange_min_mm']} ～ {env['flange_max_mm']} mm")
    print("质量：" + json.dumps(quality, ensure_ascii=False))
    for p in poses:
        print(f"  {p['index']:2d}{'*' if p['role'] == 'reserve' else ' '} 距离 {p['distance_m']:.2f} 倾角 {p['tilt_deg']:4.0f} "
              f"方位 {p['azimuth_deg']:5.0f} 滚转 {p['roll_deg']:4.0f}  板中心 {p['board_centre_px']}  "
              f"边距 {p['edge_margin_px']:4.0f} tag {p['min_tag_px']:4.1f}px  关节 {np.round(p['joints_deg'], 1).tolist()}  "
              f"关节最大移动 {p['max_joint_move_deg']}°")
    for m in problems:
        print("问题：" + m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
