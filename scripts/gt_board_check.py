#!/usr/bin/env python3
"""用标定板验证真值（在笔记本上运行：Box 上 OpenCV 导入不了）。

录制时标定板固定不动、始终在画面里。每张左目图：检测 Aprilgrid → PnP 得到 camera_T_board，
再乘上真值 base_T_camera 得到 base_T_board。真值准确的话，所有帧的 base_T_board 都应该相同。

    python gt_board_check.py <save_data 目录>

看三样东西：
1. 不同静止位置之间的差（例如平移前、平移后）：真值的空间误差（手眼外参、机械臂精度、PnP 偏差）。
2. 运动帧和静止帧的离散程度：时间不同步会让运动帧变散。
3. 时间扫描：把真值时间整体平移 δ，找让运动帧最一致的 δ。这是用图像独立检验 Δt 和曝光中心修正，应接近 0。

需要标定仓库（Aprilgrid 检测、s003 的内参和标定板尺寸），只读，不改它。
输出写进 save_data 目录：board_check_frames.csv（每帧）、board_check_report.json（汇总）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 标定仓库约定不留 __pycache__

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import fr5_groundtruth as gt  # noqa: E402

CALIB_REPO = Path(__file__).resolve().parents[2] / "calibration"   # D:\天府绛溪实验室\calibration
CAMERA_YAML = "results/s003/camera.yaml"      # 相对标定仓库
TARGET_YAML = "results/s003/target.yaml"
MIN_TAGS = 8                  # 少于这么多个 tag 的帧不用
MAX_REPROJ_PX = 1.0           # PnP 重投影 RMS 超过它的帧不用
SPEED_HALF_MS = 20.0          # 相机速度按 ±20 ms 差分
STATIC_MM_S, STATIC_DEG_S = 1.0, 0.5      # 低于这两个值算静止帧
MOVING_MM_S, MOVING_DEG_S = 5.0, 2.0      # 高于任一个算运动帧（之间的过渡帧两边都不算）
GROUP_GAP_MS, GROUP_MOVE_MM = 1000.0, 5.0  # 静止帧分组：间隔超过 1 s 或相机移动超过 5 mm 就是新的一组
MIN_MOVING_FRAMES = 20        # 摆动帧少于这么多就不做时间扫描
NEAR_GROUP_MM = 20.0          # 相机离某个静止位置不到 20 mm 的运动帧算“在这个位置摆动”（摆动时光心偏离约 2 mm）


def load_camera(path: Path) -> np.ndarray:
    cam = yaml.safe_load(path.read_text(encoding="utf-8"))
    if any(abs(float(d)) > 0 for d in cam.get("distortion", [])):
        raise ValueError(f"{path}: 内参带畸变，这里按校正后的图（无畸变）处理")
    return np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])


def load_gt_images(save_data: Path) -> list[dict]:
    rows = []
    with open(save_data / "groundtruth_images.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["flag"] == "ok":
                rows.append({"t_ms": float(r["zed_timestamp_ms"]), "file": r["filename_left"],
                             "t": np.array([float(r[k]) for k in ("tx_m", "ty_m", "tz_m")]),
                             "q": np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])})
    return rows


def load_arm_rate(save_data: Path):
    """groundtruth_arm_rate_tum.txt：机械臂每个状态包的相机位姿（已加 Δt）。返回 t_ms、位置 m、四元数 wxyz。"""
    data = np.loadtxt(save_data / "groundtruth_arm_rate_tum.txt", comments="#")
    return data[:, 0] * 1000.0, data[:, 1:4], data[:, [7, 4, 5, 6]]


def camera_poses(t_query_ms, arm):
    """在 t_query_ms 处插值相机位姿：返回 R (N×3×3)、t (N×3, m)、是否在机械臂数据范围内。"""
    t_arm, pos, quat = arm
    p, q, inside, *_ = gt.interpolate(np.asarray(t_query_ms, dtype=float), t_arm, pos, quat)
    return gt.quat_to_matrix(q), p, inside


def camera_speed(t_ms, arm):
    """相机在 t_ms 处的线速度（mm/s）和角速度（°/s），±SPEED_HALF_MS 差分。"""
    r0, p0, _ = camera_poses(t_ms - SPEED_HALF_MS, arm)
    r1, p1, _ = camera_poses(t_ms + SPEED_HALF_MS, arm)
    dt_s = 2.0 * SPEED_HALF_MS / 1000.0
    lin = np.linalg.norm(p1 - p0, axis=1) * 1000.0 / dt_s
    q0, q1 = gt.matrix_to_quat(r0), gt.matrix_to_quat(r1)
    ang = np.degrees(gt.rotation_angle_between(q0, q1)) / dt_s
    return lin, ang


def board_in_camera(image_path: Path, board: dict, det, k: np.ndarray, detect_board):
    """检测 + PnP。返回 (camera_T_board 4×4, tag 数, 重投影 RMS px)；检测不到返回 None。"""
    # cv2.imread 在 Windows 上打不开中文路径，先读字节再解码
    gray = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    obj, img, ids = detect_board(gray, board, det)
    if len(ids) == 0:
        return None, 0, None
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, len(ids), None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, None, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.ravel()
    return pose, len(ids), rms


def mean_rotation(rs: np.ndarray) -> np.ndarray:
    """一组相近旋转的平均（四元数对齐后求和再归一化）。"""
    q = gt.matrix_to_quat(rs)
    q = np.where((q @ q[0])[:, None] < 0, -q, q)
    m = q.sum(axis=0)
    return gt.quat_to_matrix(m / np.linalg.norm(m))


def static_groups(t_ms, cam_t, is_static) -> list[np.ndarray]:
    """按时间顺序把静止帧分组（同一个静止位置的为一组），返回每组的帧下标。"""
    groups, current = [], []
    for i in np.nonzero(is_static)[0]:
        if current and (t_ms[i] - t_ms[current[-1]] > GROUP_GAP_MS
                        or np.linalg.norm(cam_t[i] - cam_t[current[0]]) * 1000.0 > GROUP_MOVE_MM):
            groups.append(np.array(current))
            current = []
        current.append(i)
    if current:
        groups.append(np.array(current))
    return groups


def spread(centres_mm, ref_mm, cam_r=None) -> dict | None:
    """标定板中心相对参考点的偏差：RMS、最大值；给了相机姿态就再分解到相机系（x 右、y 下、z 深度）。"""
    if len(centres_mm) == 0:
        return None
    d = centres_mm - ref_mm
    n = np.linalg.norm(d, axis=1)
    out = {"frames": int(len(d)), "rms_mm": float(np.sqrt(np.mean(n ** 2))), "max_mm": float(n.max())}
    if cam_r is not None:
        local = np.einsum("nji,nj->ni", cam_r, d)          # R_camᵀ · d
        out["rms_cam_xyz_mm"] = np.sqrt(np.mean(local ** 2, axis=0)).round(3).tolist()
    return out


def time_scan(t_ms, offset_ms, p_cam, ref_mm, arm, scan_ms: float, step_ms: float) -> dict:
    """真值时间整体加 δ 后，运动帧标定板中心相对 ref 的 RMS；找最小处（抛物线细化）。"""
    deltas = np.arange(-scan_ms, scan_ms + step_ms / 2, step_ms)
    costs = []
    for delta in deltas:
        r, t, _ = camera_poses(t_ms + offset_ms + delta, arm)
        c = (np.einsum("nij,nj->ni", r, p_cam) + t) * 1000.0
        costs.append(float(np.sqrt(np.mean(np.sum((c - ref_mm) ** 2, axis=1)))))
    costs = np.array(costs)
    i = int(np.argmin(costs))
    best = float(deltas[i])
    if 0 < i < len(deltas) - 1:
        a, b, c = costs[i - 1], costs[i], costs[i + 1]
        denom = a - 2 * b + c
        if denom > 0:
            best += 0.5 * (a - c) / denom * step_ms
    zero = int(np.argmin(np.abs(deltas)))
    return {"best_delta_ms": best, "at_edge": i in (0, len(deltas) - 1),
            "rms_at_0_mm": float(costs[zero]), "rms_at_best_mm": float(costs[i]),
            "deltas_ms": deltas.round(3).tolist(), "rms_mm": costs.round(4).tolist()}


def check(save_data: Path, calib_repo: Path, camera_yaml: Path, target_yaml: Path, output: Path,
          step: int = 1, scan_ms: float = 20.0, scan_step_ms: float = 0.5) -> dict:
    sys.path.insert(0, str(calib_repo))
    from handeye.board import board_outline, detect_board, detector, read_board

    k = load_camera(camera_yaml)
    board = read_board(target_yaml)
    centre_b = np.r_[board_outline(board).mean(axis=0), 1.0]        # 标定板中心（标定板坐标，m）
    det = detector()
    report = json.loads((save_data / "groundtruth_report.json").read_text(encoding="utf-8"))
    offset_ms = float(report["exposure_centre"]["applied_ms"])
    arm = load_arm_rate(save_data)
    rows = load_gt_images(save_data)[::step]
    if not rows:
        raise ValueError("groundtruth_images.csv 里没有 flag=ok 的帧")

    # 用机械臂频率的真值重新插值（时间扫描也用它），先核对它和 groundtruth_images.csv 一致
    t_ms = np.array([r["t_ms"] for r in rows])
    cam_r, cam_t, inside = camera_poses(t_ms + offset_ms, arm)
    csv_diff_mm = float(np.max(np.linalg.norm(cam_t - np.array([r["t"] for r in rows]), axis=1)) * 1000.0)
    lin, ang = camera_speed(t_ms + offset_ms, arm)

    used, frames = [], []
    for i, r in enumerate(rows):
        pose, tags, rms = board_in_camera(save_data / r["file"], board, det, k, detect_board)
        ok = pose is not None and tags >= MIN_TAGS and rms is not None and rms <= MAX_REPROJ_PX and inside[i]
        state = ("static" if lin[i] < STATIC_MM_S and ang[i] < STATIC_DEG_S
                 else "moving" if lin[i] > MOVING_MM_S or ang[i] > MOVING_DEG_S else "transition")
        frames.append({"i": i, "tags": tags, "reproj_px": rms, "state": state, "used": ok,
                       "p_cam": (pose @ centre_b)[:3] if pose is not None else None,
                       "r_board": pose[:3, :3] if pose is not None else None})
        if ok:
            used.append(i)
    if len(used) < 5:
        raise ValueError(f"只有 {len(used)} 帧检测到标定板（至少 {MIN_TAGS} 个 tag、重投影 ≤ {MAX_REPROJ_PX} px），"
                         "标定板在画面里吗？")

    used = np.array(used)
    p_cam = np.array([frames[i]["p_cam"] for i in used])
    centres = (np.einsum("nij,nj->ni", cam_r[used], p_cam) + cam_t[used]) * 1000.0      # 基坐标系，mm
    board_r = np.einsum("nij,njk->nik", cam_r[used], np.array([frames[i]["r_board"] for i in used]))
    states = np.array([frames[i]["state"] for i in used])
    is_static, is_moving = states == "static", states == "moving"
    ref_set = is_static if is_static.sum() >= 5 else np.ones(len(used), bool)
    ref_mm = np.median(centres[ref_set], axis=0)
    ref_r = mean_rotation(board_r[ref_set])
    rot_dev = np.degrees(gt.rotation_angle_between(gt.matrix_to_quat(board_r), gt.matrix_to_quat(ref_r)[None]))

    groups, group_frames = [], [g for g in static_groups(t_ms[used], cam_t[used], is_static) if len(g) >= 3]
    for g in group_frames:
        groups.append({"frames": int(len(g)), "t_start_ms": float(t_ms[used][g[0]]),
                       "camera_mm": (cam_t[used][g].mean(axis=0) * 1000.0).round(3).tolist(),
                       "board_centre_mm": centres[g].mean(axis=0).round(4).tolist(),
                       "spread_rms_mm": float(np.sqrt(np.mean(np.sum((centres[g] - centres[g].mean(axis=0)) ** 2, 1)))),
                       "rotation": mean_rotation(board_r[g])})
    # 两两比较：差值也分解到第一组的相机系（x 右、y 下、z 前），便于和运动方向对照
    first_r = cam_r[used][group_frames[0][0]] if groups else None
    pairs = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            ga, gb = groups[a], groups[b]
            moved = np.subtract(gb["camera_mm"], ga["camera_mm"])
            diff = np.subtract(gb["board_centre_mm"], ga["board_centre_mm"])
            pairs.append({"groups": [a, b],
                          "camera_moved_mm": float(np.linalg.norm(moved)),
                          "camera_moved_cam_xyz_mm": (first_r.T @ moved).round(2).tolist(),
                          "board_diff_mm": float(np.linalg.norm(diff)),
                          "board_diff_cam_xyz_mm": (first_r.T @ diff).round(3).tolist(),
                          "board_diff_deg": float(np.degrees(gt.rotation_angle_between(
                              gt.matrix_to_quat(ga["rotation"]), gt.matrix_to_quat(gb["rotation"]))))})
    for g in groups:
        del g["rotation"]

    # 时间扫描只用“在某个静止位置附近转动”的帧（同步摆动），参考点取同一位置静止组的标定板中心。
    # 平移中的帧不用：它们带着随位置变化的空间误差，会把时间估计带偏。
    scan, scans_by_group = None, []
    if groups:
        group_cam = np.array([g["camera_mm"] for g in groups])
        moving_idx = np.nonzero(is_moving)[0]
        dist = np.linalg.norm(cam_t[used][moving_idx][:, None, :] * 1000.0 - group_cam[None], axis=2)
        near = dist.min(axis=1) < NEAR_GROUP_MM
        sel, owner = moving_idx[near], dist[near].argmin(axis=1)
        refs = np.array([groups[j]["board_centre_mm"] for j in owner]).reshape(-1, 3)
        if len(sel) >= MIN_MOVING_FRAMES:
            scan = time_scan(t_ms[used][sel], offset_ms, p_cam[sel], refs, arm, scan_ms, scan_step_ms)
            scan["frames"] = int(len(sel))
            for j in np.unique(owner):
                m = owner == j
                if m.sum() >= MIN_MOVING_FRAMES // 2:
                    s = time_scan(t_ms[used][sel[m]], offset_ms, p_cam[sel[m]], refs[m], arm, scan_ms, scan_step_ms)
                    scans_by_group.append({"group": int(j), "frames": int(m.sum()), "best_delta_ms": s["best_delta_ms"],
                                           "rms_at_0_mm": s["rms_at_0_mm"], "rms_at_best_mm": s["rms_at_best_mm"]})

    result = {
        "session": str(save_data),
        "camera_yaml": str(camera_yaml), "target_yaml": str(target_yaml),
        "groundtruth": {"dt_ms": report["time_offset"]["dt_ms"], "exposure_applied_ms": offset_ms},
        "check_csv_vs_arm_rate_max_mm": csv_diff_mm,
        "images": {"checked": len(rows), "board_used": int(len(used)),
                   "static": int(is_static.sum()), "moving": int(is_moving.sum())},
        "reproj_px": {"median": float(np.median([frames[i]["reproj_px"] for i in used])),
                      "max": float(np.max([frames[i]["reproj_px"] for i in used]))},
        "board_distance_m": {"min": float(np.linalg.norm(p_cam, axis=1).min()),
                             "max": float(np.linalg.norm(p_cam, axis=1).max())},
        "static": spread(centres[is_static], ref_mm, cam_r[used][is_static]),
        "moving": spread(centres[is_moving], ref_mm, cam_r[used][is_moving]),
        "rotation_deg": {"static_rms": float(np.sqrt(np.mean(rot_dev[is_static] ** 2))) if is_static.any() else None,
                         "moving_rms": float(np.sqrt(np.mean(rot_dev[is_moving] ** 2))) if is_moving.any() else None},
        "static_groups": groups,
        "static_group_pairs": pairs,
        "time_scan": scan,
        "time_scan_by_group": scans_by_group,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "board_check_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(output / "board_check_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zed_timestamp_ms", "filename_left", "state", "tags", "reproj_px", "used",
                    "cam_speed_mm_s", "cam_rate_deg_s", "board_x_mm", "board_y_mm", "board_z_mm",
                    "dev_mm", "dev_deg"])
        dev = dict(zip(used.tolist(), zip(centres, np.linalg.norm(centres - ref_mm, axis=1), rot_dev)))
        for fr in frames:
            i = fr["i"]
            c, d, dd = dev.get(i, (None, None, None))
            w.writerow([f"{t_ms[i]:.0f}", rows[i]["file"], fr["state"], fr["tags"],
                        "" if fr["reproj_px"] is None else f"{fr['reproj_px']:.3f}", int(fr["used"]),
                        f"{lin[i]:.2f}", f"{ang[i]:.3f}",
                        *(["", "", ""] if c is None else [f"{v:.3f}" for v in c]),
                        "" if d is None else f"{d:.3f}", "" if dd is None else f"{dd:.4f}"])
    return result


def summary(r: dict) -> str:
    lines = [f"图像 {r['images']['checked']} 张，用上 {r['images']['board_used']} 张"
             f"（静止 {r['images']['static']}、运动 {r['images']['moving']}）；重投影中位 {r['reproj_px']['median']:.2f} px，"
             f"标定板距离 {r['board_distance_m']['min']:.2f}–{r['board_distance_m']['max']:.2f} m",
             f"真值：Δt {r['groundtruth']['dt_ms']:+.2f} ms，曝光中心 {r['groundtruth']['exposure_applied_ms']:+.2f} ms；"
             f"机械臂频率真值重插值与 groundtruth_images.csv 最大差 {r['check_csv_vs_arm_rate_max_mm']:.4f} mm"]
    for name, label in (("static", "静止帧"), ("moving", "运动帧")):
        s = r[name]
        if s:
            lines.append(f"{label}：标定板中心偏差 RMS {s['rms_mm']:.2f} mm（最大 {s['max_mm']:.2f}），"
                         f"相机系 x/y/z（z 为深度）RMS {s['rms_cam_xyz_mm']} mm")
    rot = r["rotation_deg"]
    lines.append("标定板姿态偏差 RMS：静止 " + ("-" if rot["static_rms"] is None else f"{rot['static_rms']:.3f}°")
                 + "，运动 " + ("-" if rot["moving_rms"] is None else f"{rot['moving_rms']:.3f}°"))
    for g in r["static_groups"]:
        lines.append(f"  静止组：{g['frames']} 帧，相机 {g['camera_mm']} mm，组内离散 {g['spread_rms_mm']:.2f} mm")
    for pr in r["static_group_pairs"]:
        if pr["camera_moved_mm"] < GROUP_MOVE_MM:
            lines.append(f"  静止组 {pr['groups'][0]}↔{pr['groups'][1]}（同一位置）：标定板位置差 {pr['board_diff_mm']:.2f} mm、"
                         f"姿态差 {pr['board_diff_deg']:.3f}°")
        else:
            lines.append(f"  静止组 {pr['groups'][0]}↔{pr['groups'][1]}：相机移动 {pr['camera_moved_cam_xyz_mm']} mm（起点相机系），"
                         f"标定板位置差 {pr['board_diff_mm']:.2f} mm = {pr['board_diff_cam_xyz_mm']}，"
                         f"姿态差 {pr['board_diff_deg']:.3f}°")
    s = r["time_scan"]
    if s:
        lines.append(f"时间扫描（{s['frames']} 个摆动帧，对照同位置静止组）：真值时间再加 {s['best_delta_ms']:+.2f} ms 时最一致"
                     f"（RMS {s['rms_at_0_mm']:.2f} → {s['rms_at_best_mm']:.2f} mm）"
                     + ("；落在扫描边缘，结果不可信" if s["at_edge"] else ""))
        for g in r["time_scan_by_group"]:
            lines.append(f"  静止组 {g['group']} 处的摆动：{g['frames']} 帧，{g['best_delta_ms']:+.2f} ms"
                         f"（RMS {g['rms_at_0_mm']:.2f} → {g['rms_at_best_mm']:.2f} mm）")
    else:
        lines.append(f"静止位置附近的摆动帧少于 {MIN_MOVING_FRAMES}，没做时间扫描")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_data", type=Path, help="录制的 save_data 目录（已经算过真值）")
    parser.add_argument("--calib-repo", type=Path, default=CALIB_REPO, help="标定仓库路径")
    parser.add_argument("--camera", type=Path, help=f"内参，默认 <标定仓库>/{CAMERA_YAML}")
    parser.add_argument("--target", type=Path, help=f"标定板，默认 <标定仓库>/{TARGET_YAML}")
    parser.add_argument("--output", type=Path, help="输出目录，默认 save_data 目录")
    parser.add_argument("--step", type=int, default=1, help="每隔几张图检测一张（加快速度）")
    parser.add_argument("--scan-ms", type=float, default=20.0, help="时间扫描范围 ±ms")
    args = parser.parse_args(argv)
    try:
        r = check(args.save_data, args.calib_repo, args.camera or args.calib_repo / CAMERA_YAML,
                  args.target or args.calib_repo / TARGET_YAML, args.output or args.save_data,
                  step=max(1, args.step), scan_ms=args.scan_ms)
    except (OSError, ValueError, KeyError) as exc:
        print(f"失败：{exc}")
        return 1
    print(summary(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
