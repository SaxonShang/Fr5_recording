"""拖动标记的极限位姿：相对起点分解，给出行程、转角和关节范围的建议，检查录制有没有超出。
由 run_session.py 的 mark / limits / plan / record 调用；不连机械臂，本文件没有可调参数（余量在 run_session.py 开头）。

标记名和录制名相同：forward 是手动拖到最前、操作员认为还安全的位姿，yaw_left 是往左转到的最大角度，……
平移标记取左目光心沿起点相机（前, 左, 上）方向的偏移（和 TRANSLATION_PATHS 同一约定）；
旋转标记取相对起点的方位角、仰角、横滚（正 = 向左 / 向上 / 画面顶部向右，和 pose_span 相同）。
录制只走经过起点的轴（平移）和起点上的转动（旋转），两端都要离标记至少一个余量。
"""

from __future__ import annotations

import math

import numpy as np

import fr5_motion as fm

TRANSLATION_MARKS = {"forward": (0, +1), "backward": (0, -1), "left": (1, +1), "right": (1, -1),
                     "up": (2, +1), "down": (2, -1)}          # 标记名 -> (偏移分量：前 0 / 左 1 / 上 2, 方向)
ROTATION_MARKS = {"yaw_left": (0, +1), "yaw_right": (0, -1), "pitch_up": (1, +1), "pitch_down": (1, -1),
                  "roll_right": (2, +1), "roll_left": (2, -1)}  # 标记名 -> (角：方位 0 / 俯仰 1 / 横滚 2, 方向)
TRANSLATION_PAIRS = {"forward_backward": ("forward", "backward"), "left_right": ("left", "right"),
                     "up_down": ("up", "down")}               # 方向对 -> (正方向标记, 负方向标记)
ROTATION_PAIRS = {"yaw": ("yaw_left", "yaw_right"), "pitch": ("pitch_up", "pitch_down"),
                  "roll": ("roll_right", "roll_left")}
AXIS_LABEL = ("前", "左", "上")
ANGLE_LABEL = ("方位角", "俯仰", "横滚")
ANGLE_KEY = ("heading", "pitch", "roll")                      # pose_span 的键
CROSS_WARN_MM = 30.0       # 平移标记离起点轴超过这么多就提醒（拖动时没有沿一个方向走）
TILT_WARN_DEG = 3.0        # 平移标记的姿态和起点差这么多就提醒（相机没保持水平、朝前）
OFF_AXIS_WARN_DEG = 5.0    # 旋转标记在另外两个角上超过这么多就提醒
CENTRE_WARN_MM = 50.0      # 旋转标记的光心离起点超过这么多就提醒（录制是绕起点光心转的）
PATH_CROSS_TOL_MM = 20.0   # 平移路径离经过起点的轴超过这么多：拖动标记没有验证过，算问题
JOINT_TOL_DEG = 2.0        # 建议关节范围时，在起点和标记的包络外再放宽这么多
BRANCH_SWING_DEG = 100.0   # J1、J4、J6 离起点超过这么多，也当作换了构型（转过了半圈）


def relative(start: np.ndarray, camera: np.ndarray) -> dict:
    """相机位姿相对起点：光心偏移（前, 左, 上）mm、姿态（方位角, 俯仰, 横滚）°、总转角 °。"""
    return {"offset_mm": fm.start_offsets(start, camera[:3, 3]).tolist(),
            "angles_deg": list(fm.orientation_angles(start[:3, :3].T @ camera[:3, :3])),
            "rotation_deg": fm.rotation_deg(start, camera)}


def reach(name: str, rel: dict) -> float:
    """标记在自己方向上到了哪里（带符号，和 TRANSLATION_PATHS / pose_span 同号：backward、yaw_right 等为负）。"""
    if name in TRANSLATION_MARKS:
        return rel["offset_mm"][TRANSLATION_MARKS[name][0]]
    return rel["angles_deg"][ROTATION_MARKS[name][0]]


def mark_warnings(name: str, rel: dict) -> list[str]:
    """标记本身的问题：拖动时没有沿一个方向走、姿态变了、往反方向去了等。"""
    out = []
    if name in TRANSLATION_MARKS:
        k, sign = TRANSLATION_MARKS[name]
        off = np.asarray(rel["offset_mm"])
        cross = float(np.hypot(*np.delete(off, k)))
        if sign * off[k] <= 0:
            out.append(f"{name} 标记在起点的反方向（{AXIS_LABEL[k]} {off[k]:+.0f} mm）")
        if cross > CROSS_WARN_MM:
            out.append(f"{name} 标记离起点的{AXIS_LABEL[k]}轴 {cross:.0f} mm（超过 {CROSS_WARN_MM:g}）：只沿一个方向拖")
        if rel["rotation_deg"] > TILT_WARN_DEG:
            out.append(f"{name} 标记的相机姿态和起点差 {rel['rotation_deg']:.1f}°（超过 {TILT_WARN_DEG:g}）："
                       "录制时姿态和起点一样，关节会和这个标记不同")
    else:
        k, sign = ROTATION_MARKS[name]
        ang = np.asarray(rel["angles_deg"])
        if sign * ang[k] <= 0:
            out.append(f"{name} 标记转到了反方向（{ANGLE_LABEL[k]} {ang[k]:+.1f}°）")
        other = max(abs(a) for i, a in enumerate(ang) if i != k)
        if other > OFF_AXIS_WARN_DEG:
            out.append(f"{name} 标记在另外两个角上最多 {other:.1f}°（超过 {OFF_AXIS_WARN_DEG:g}）：只绕一个轴转")
        centre = float(np.linalg.norm(rel["offset_mm"]))
        if centre > CENTRE_WARN_MM:
            out.append(f"{name} 标记的光心离起点 {centre:.0f} mm（超过 {CENTRE_WARN_MM:g}）：录制绕起点光心转，"
                       "关节会和这个标记不同")
    return out


def branch_problem(joints, start_joints) -> str | None:
    """关节角和起点是不是同一种构型（逆解分支）：肘部（J3 正负）、腕部（J5 正负）和起点相同，
    J1、J4、J6 离起点不超过 BRANCH_SWING_DEG。不同就返回说明：录制用 MoveL，到不了另一种构型。"""
    q, q0 = np.asarray(joints, dtype=float), np.asarray(start_joints, dtype=float)
    out = [f"J{j + 1} {q[j]:+.1f}°（起点 {q0[j]:+.1f}°，{what}）"
           for j, what in ((2, "肘部翻到了另一边"), (4, "腕部翻到了另一边")) if q[j] * q0[j] < 0]
    out += [f"J{j + 1} {q[j]:+.1f}°（起点 {q0[j]:+.1f}°，差 {abs(q[j] - q0[j]):.0f}°）"
            for j in (0, 3, 5) if abs(q[j] - q0[j]) > BRANCH_SWING_DEG]
    return "构型和起点不同：" + "，".join(out) if out else None


def side_limits(marks: dict, pos: str, neg: str, margin: float) -> tuple[float | None, float | None]:
    """一对方向可用的 (下限, 上限)：负方向标记收进 margin、正方向标记收进 margin；没有标记的一侧为 None。"""
    lo = reach(neg, marks[neg]) + margin if neg in marks else None
    hi = reach(pos, marks[pos]) - margin if pos in marks else None
    return lo, hi


def _fit(lo: float, hi: float, cap: float, step: float, grid: float, prefer: float = 0.0) -> tuple[float, float] | None:
    """在 [lo, hi] 里放一段最长的区间：长度是 step 的整数倍、不超过 cap，中点是 grid 的整数倍、尽量靠近 prefer。
    返回 (长度, 中点)，放不下为 None。"""
    length = math.floor(min(hi - lo, cap) / step + 1e-9) * step
    while length > 0:
        a = math.ceil((lo + length / 2.0) / grid - 1e-9) * grid
        b = math.floor((hi - length / 2.0) / grid + 1e-9) * grid
        if a <= b:
            return length, min(max(round(prefer / grid) * grid, a), b)
        length -= step
    return None


def suggest_translation(marks: dict, margin_mm: float, max_stroke_mm: float, step_mm: float = 10.0) -> dict:
    """每对方向：经过起点的轴上、两端标记各收进 margin_mm 的区间里，取最长（不超过 max_stroke_mm）的行程；
    区间更长时行程中点尽量靠近起点。{方向对: {"lo", "hi", "stroke_mm", "mid_mm"}}，缺标记时为 None。"""
    out = {}
    for pair, (pos, neg) in TRANSLATION_PAIRS.items():
        lo, hi = side_limits(marks, pos, neg, margin_mm)
        if lo is None or hi is None:
            out[pair] = None
            continue
        fit = _fit(lo, hi, max_stroke_mm, step_mm, 1.0)            # 中点取整 mm
        out[pair] = {"lo": lo, "hi": hi, "stroke_mm": fit[0] if fit else 0.0, "mid_mm": fit[1] if fit else None}
    return out


def glass_heading_limits(regions, margin_deg: float, step_deg: float = 0.05) -> tuple[float | None, float | None]:
    """相机在起点水平转动（偏航录制）时画面离玻璃至少 margin_deg 的方位角范围 (右限, 左限)；那一侧没有玻璃为 None。"""
    outlines = [fm.region_outline(r) for r in regions]
    normals = fm.frustum_normals()
    out = []
    for sign in (-1, +1):
        h, limit = 0.0, None
        while abs(h) < 90.0:
            r_rel = fm.axis_angle([0.0, -1.0, 0.0], h)              # 起点相机系里绕“上”转 h（正 = 向左）
            if fm.glass_clearance_deg(r_rel, regions, outlines, normals) < margin_deg:
                limit = h - sign * step_deg
                break
            h += sign * step_deg
        out.append(limit)
    return out[0], out[1]


def suggest_rotation(marks: dict, margin_deg: float, caps: dict, glass_right: float | None = None,
                     step_deg: float = 0.5) -> dict:
    """偏航：标记收进 margin_deg、右边再受右窗限制，取最大转角（不超过 caps["yaw"]），中心朝向尽量靠近起点；
    俯仰、横滚：以水平 / 无横滚为中心对称（和录制一致），取两边都够的转角。
    {"yaw": {"lo", "hi", "deg", "heading"}, "pitch": {"lo", "hi", "deg"}, "roll": …}，缺标记时为 None。"""
    out = {}
    for kind, (pos, neg) in ROTATION_PAIRS.items():
        lo, hi = side_limits(marks, pos, neg, margin_deg)
        if lo is None or hi is None:
            out[kind] = None
            continue
        if kind == "yaw":
            if glass_right is not None:
                lo = max(lo, glass_right)
            fit = _fit(lo, hi, caps[kind], step_deg, 0.25)            # 中心朝向取 0.25° 的整数倍
            out[kind] = {"lo": lo, "hi": hi, "deg": fit[0] if fit else 0.0, "heading": fit[1] if fit else None}
        else:
            half = min(hi, -lo)
            deg = math.floor(min(2.0 * half, caps[kind]) / step_deg + 1e-9) * step_deg if half > 0 else 0.0
            out[kind] = {"lo": lo, "hi": hi, "deg": deg}
    return out


def joint_envelope(joint_sets: dict, tol_deg: float = JOINT_TOL_DEG) -> tuple[list, list, list, list]:
    """{名字: 6 个关节角} 的包络，外扩 tol_deg 后取整。返回 (下限, 上限, 下限来自哪个, 上限来自哪个)。"""
    names = list(joint_sets)
    q = np.array([joint_sets[n] for n in names], dtype=float)
    lo_i, hi_i = q.argmin(axis=0), q.argmax(axis=0)
    lo = [float(math.floor(q[lo_i[j], j] - tol_deg)) for j in range(6)]
    hi = [float(math.ceil(q[hi_i[j], j] + tol_deg)) for j in range(6)]
    return lo, hi, [names[i] for i in lo_i], [names[i] for i in hi_i]


def limit_check(name: str, segments: list, start: np.ndarray, f_t_c: np.ndarray, marks: dict,
                margin_mm: float, margin_deg: float) -> tuple[str, list[str]]:
    """录制（不含 via_start）是否在拖动标记以内、离标记至少一个余量。返回 (说明, 问题列表)；
    缺哪一侧的标记就不查那一侧，在说明里写明。name 不是录制名（sync-test）时不查。"""
    moves = [s for s in segments if s.kind == "move" and s.name != "via_start"]
    if name in fm.DIRECTIONS:
        pos, neg = TRANSLATION_PAIRS[fm.TRANSLATION_PAIRS[name]]
        k, margin, unit = TRANSLATION_MARKS[pos][0], margin_mm, "mm"
        pts = np.array([fm.start_offsets(start, (fm.fr5_to_matrix(s.flange) @ f_t_c)[:3, 3]) for s in moves])
        span = (float(pts[:, k].min()), float(pts[:, k].max()))
        label = f"沿{AXIS_LABEL[k]}"
        cross = float(np.hypot(*np.delete(pts, k, axis=1).T).max())
        problems = ([f"路径离经过起点的{AXIS_LABEL[k]}轴最远 {cross:.0f} mm（超过 {PATH_CROSS_TOL_MM:g}）："
                     "拖动标记只验证了这根轴"] if cross > PATH_CROSS_TOL_MM else [])
    elif name in fm.ROTATIONS:
        kind = {"up": "yaw", "x": "pitch", "z": "roll"}[fm.ROTATIONS[name][0]]
        pos, neg = ROTATION_PAIRS[kind]
        k, margin, unit = ROTATION_MARKS[pos][0], margin_deg, "°"
        span = fm.pose_span(segments, start, f_t_c)[ANGLE_KEY[k]]
        label, problems = ANGLE_LABEL[k], []
    else:
        return "", []
    lo, hi = side_limits(marks, pos, neg, margin)
    fmt = (lambda v: f"{v:+.0f}") if unit == "mm" else (lambda v: f"{v:+.1f}")
    allowed = f"{fmt(lo) if lo is not None else '（没有标记）'}～{fmt(hi) if hi is not None else '（没有标记）'}"
    note = f"拖动极限：{label} {fmt(span[0])}～{fmt(span[1])} {unit}，标记收进余量后允许 {allowed} {unit}"
    if lo is not None and span[0] < lo - 1e-6:
        problems.append(f"{label}到 {fmt(span[0])} {unit}，超出 {neg} 标记收进余量后的 {fmt(lo)} {unit}")
    if hi is not None and span[1] > hi + 1e-6:
        problems.append(f"{label}到 {fmt(span[1])} {unit}，超出 {pos} 标记收进余量后的 {fmt(hi)} {unit}")
    return note, problems
