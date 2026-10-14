"""拖动标记 → 录制路径（高速录制）。由 run_session.py 的 mark / limits / plan / record / speed-test 调用；
本文件没有可调参数（余量、行程上下限等在 run_session.py 开头）。

用户在手动模式下把机械臂拖到每个方向能接受的极限位姿，用 mark 记下（12 个：forward … down、yaw_left … roll_right）。
每对标记定一条录制，共 6 条：平移 3 条（最下到最上、最左到最右、最后到最前）、旋转 3 条（偏航、俯仰、横滚各转最大角度）。
录制不回 set-start 的起点，从机械臂当前位姿直接移到这条录制的起点。标记只是大概的极限，不当端点：
- 平移：相机水平、姿态全程不变。前后、左右的方向取两个标记光心连线的水平分量，光轴（前后）或相机 x 轴（左右）
  沿这个方向，路径过两个标记光心的中点、高度取两者平均；上下沿竖直方向，水平位置取两个标记的中点，
  相机朝向取两个标记的平均。
- 旋转：绕两个标记光心的中点转。偏航绕竖直轴，相机水平；俯仰绕水平相机的 x 轴、横滚绕水平相机的光轴，
  朝向取两个标记的平均。转角范围就是两个标记在这个角上的读数。
- 两端先各收进一个余量（LIMIT_MARGIN_MM / LIMIT_MARGIN_DEG），再沿路径用控制器逆解采样（fit_path），往里收到：
  关节在允许范围内（FR5 关节范围和控制器软限位）、构型不变且不跳支、离奇异位形至少一个余量、光心不低于下限、
  画面看不到右窗、两端能做同步摆动、按录制速度走时关节速度不超上限（有 speed-test 的加速度时按梯形速度曲线）。
- 构型（肘部 J3、腕部 J5 的正负）由这一对标记自己定，标记可以是任何构型（用户 10-14 定：拖的时候人会确认腕部、
  线缆等没问题）。一条直线 / 圆弧换不了构型（要经过奇异位形），两个标记构型不同时两种构型各试一次，取能走得长的（fit_best）。

角度都相对 set-start 的水平起点（参考位姿：只用来定玻璃窗的方向，录制不回这里）：
方位角正 = 向左、俯仰正 = 向上、横滚正 = 画面顶部向右（和 fr5_motion.pose_span 相同）。
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import fr5_motion as fm

TRANSLATION_MARKS = ("forward", "backward", "left", "right", "up", "down")
ROTATION_MARKS = ("yaw_left", "yaw_right", "pitch_up", "pitch_down", "roll_right", "roll_left")
MARK_NAMES = (*TRANSLATION_MARKS, *ROTATION_MARKS)
PAIRS = {                       # 标记对 -> (负端标记, 正端标记)；路径参数 s 从负端到正端增大
    "up_down": ("down", "up"),
    "left_right": ("left", "right"),
    "forward_backward": ("backward", "forward"),
    "yaw": ("yaw_right", "yaw_left"),
    "pitch": ("pitch_down", "pitch_up"),
    "roll": ("roll_left", "roll_right"),
}
TRANSLATION_PAIRS = ("up_down", "left_right", "forward_backward")
PAIR_LABEL = {"up_down": "上下", "left_right": "左右", "forward_backward": "前后",
              "yaw": "偏航", "pitch": "俯仰", "roll": "横滚"}
RECORD_PAIR = {                 # 录制名 -> (标记对, +1 从负端走到正端 / -1 反过来)
    "up": ("up_down", +1), "down": ("up_down", -1),
    "right": ("left_right", +1), "left": ("left_right", -1),
    "forward": ("forward_backward", +1), "backward": ("forward_backward", -1),
    "yaw_left": ("yaw", +1), "yaw_right": ("yaw", -1),
    "pitch_up": ("pitch", +1), "pitch_down": ("pitch", -1),
    "roll_right": ("roll", +1), "roll_left": ("roll", -1),
}
RECORDINGS = ("up", "right", "forward", "yaw_left", "pitch_up", "roll_right")   # 每对标记录一条（用户 10-14 定）
ANGLE_LABEL = ("方位角", "俯仰", "横滚")
OFF_LINE_WARN_MM = 30.0    # 平移：两个标记偏离路径（前后、左右的高差，上下的水平距离）超过这么多就提醒
POSE_WARN_DEG = 5.0        # 标记的相机姿态和录制时的姿态（平移）、或另外两个角（旋转）差这么多就提醒
CENTRE_WARN_MM = 50.0      # 旋转：两个标记的光心相距超过这么多就提醒（绕两者中点转）
FIT_STEP_MM = 10.0         # fit_path 沿路径的采样间隔（和 check_plan 的 PATH_STEP_MM 相同）
FIT_STEP_DEG = 1.0
FIT_SPEED_FRACTION = 0.98  # fit_path 把关节速度收到上限的这么多：之后 check_plan 换一组采样点再查，结果会差一点


# ------------------------------------------------------------------ 位姿分解
def angles(reference: np.ndarray, camera: np.ndarray) -> tuple[float, float, float]:
    """相机姿态相对水平参考起点的（方位角, 俯仰, 横滚）°。"""
    return fm.orientation_angles(reference[:3, :3].T @ camera[:3, :3])


def relative(reference: np.ndarray, camera: np.ndarray) -> dict:
    """相机位姿相对参考起点：光心偏移（前, 左, 上）mm、姿态（方位角, 俯仰, 横滚）°。"""
    return {"offset_mm": fm.start_offsets(reference, camera[:3, 3]).tolist(),
            "angles_deg": list(angles(reference, camera))}


def heading_of(reference: np.ndarray, direction) -> float:
    """水平方向 direction（基坐标系）相对参考起点的方位角 °（正 = 向左）。"""
    d = np.asarray(direction, dtype=float)
    return math.degrees(math.atan2(-(d @ reference[:3, 0]), d @ reference[:3, 2]))


def level_pose(reference: np.ndarray, heading_deg: float, position) -> np.ndarray:
    """光心在 position、相机水平、朝向为 heading_deg（相对参考起点）的位姿。"""
    out = fm.yaw_about_vertical(fm.level_camera(reference), heading_deg)
    out[:3, 3] = position
    return out


def mean_heading(values) -> float:
    a = np.radians(values)
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def signed(value: float, digits: int = 1) -> str:
    """带正负号的数，四舍五入后是 0 时写 +0（不写 -0.0）。"""
    v = round(float(value), digits)
    return f"{v + 0.0 if v != 0 else 0.0:+.{digits}f}"


def config_of(joints) -> tuple[int, int]:
    """构型（逆解分支）：肘部 J3、腕部 J5 的正负。直线 / 圆弧运动中途要变号就得经过奇异位形
    （J3 = 0 手臂伸直；J5 = 0 腕部 J4、J6 两根轴共线），控制器走不过去，所以一条录制只能在一种构型里。"""
    q = np.asarray(joints, dtype=float)
    return int(np.sign(q[2])), int(np.sign(q[4]))


def describe_config(joints) -> str:
    q = np.asarray(joints, dtype=float)
    return f"肘部 J3 {signed(q[2])}°、腕部 J5 {signed(q[4])}°"


def singular_problem(joints, margin_deg: float) -> str | None:
    """离奇异位形太近：J3 或 J5 离 0 不到 margin_deg。"""
    q = np.asarray(joints, dtype=float)
    if abs(q[2]) < margin_deg:
        return f"离肘部奇异位形太近（J3 {signed(q[2])}°，手臂快伸直）"
    if abs(q[4]) < margin_deg:
        return f"离腕部奇异位形太近（J5 {signed(q[4])}°）"
    return None


# ------------------------------------------------------------------ 一对标记 → 路径
@dataclass
class MarkPath:
    """一对标记定出的路径。pose(s)：平移时 s 是沿 axis 离 base 光心的 mm，旋转时 s 是绕过 base 光心的 axis 转的 °。
    偏航的 s 就是方位角，俯仰的 s 是俯仰角，横滚的 s 是横滚角（都相对参考起点）。"""
    pair: str
    kind: str                       # "translate" 或 "rotate"
    base: np.ndarray                # s = 0 时的相机位姿
    axis: np.ndarray                # 平移方向 / 转轴（基坐标系单位向量，s 增大的方向）
    heading_deg: float | None       # 录制时相机的朝向（偏航为 None：朝向就是 s）
    marks_s: tuple[float, float]    # 负端、正端标记在路径上的读数
    lo: float                       # 两端各收进余量后的范围
    hi: float
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def unit(self) -> str:
        return "mm" if self.kind == "translate" else "°"

    def pose(self, s: float) -> np.ndarray:
        if self.kind == "translate":
            out = self.base.copy()
            out[:3, 3] += self.axis * s / 1000.0
            return out
        return fm.rotate_about(self.base, self.axis, s)


def derive_path(pair: str, cam_neg: np.ndarray, cam_pos: np.ndarray, reference: np.ndarray,
                margin_mm: float, margin_deg: float, up_down_facing=None) -> MarkPath:
    """由负端、正端标记的相机位姿定出路径（不连机械臂）。reference 是参考起点（水平）。
    up_down_facing：上下录制时相机光轴的水平方向（基坐标系向量）；None 时取两个标记朝向的平均。"""
    neg, pos = PAIRS[pair]
    ref = fm.level_camera(reference)
    c0, c1 = cam_neg[:3, 3], cam_pos[:3, 3]
    mid = (c0 + c1) / 2.0
    warnings, problems = [], []
    if pair in TRANSLATION_PAIRS:
        if pair == "up_down":
            axis = fm.WORLD_UP.copy()
            heading = (mean_heading([angles(ref, cam_neg)[0], angles(ref, cam_pos)[0]]) if up_down_facing is None
                       else heading_of(ref, up_down_facing))
            off = float(np.hypot(*(c1 - c0)[:2]) * 1000.0)
            if off > OFF_LINE_WARN_MM:
                warnings.append(f"{neg}、{pos} 两个标记水平相距 {off:.0f} mm：路径取两者中间的竖直线")
        else:
            d = c1 - c0
            d[2] = 0.0
            if np.linalg.norm(d) < 1e-3:
                d = ref[:3, 2] if pair == "forward_backward" else ref[:3, 0]
                problems.append(f"{neg}、{pos} 两个标记水平方向上几乎重合，定不出方向")
            axis = d / np.linalg.norm(d)
            heading = heading_of(ref, axis) + (90.0 if pair == "left_right" else 0.0)   # 左右：相机 x 轴沿运动方向
            dz = float((c1 - c0)[2] * 1000.0)
            if abs(dz) > OFF_LINE_WARN_MM:
                warnings.append(f"{neg}、{pos} 两个标记高差 {dz:+.0f} mm：路径取中间高度")
        base = level_pose(ref, heading, mid)
        marks_s = (float((c0 - mid) @ axis * 1000.0), float((c1 - mid) @ axis * 1000.0))
        for name, cam in ((neg, cam_neg), (pos, cam_pos)):
            diff = fm.rotation_deg(cam, base)
            if diff > POSE_WARN_DEG:
                warnings.append(f"{name} 标记的相机姿态和录制时（水平、朝向 {heading:+.1f}°）差 {diff:.1f}°："
                                "录制时腕部关节会和这个标记不同")
        margin, kind = margin_mm, "translate"
    else:
        sep = float(np.linalg.norm(c1 - c0) * 1000.0)
        if sep > CENTRE_WARN_MM:
            warnings.append(f"{neg}、{pos} 两个标记的光心相距 {sep:.0f} mm：绕两者中点转")
        a0, a1 = angles(ref, cam_neg), angles(ref, cam_pos)
        if pair == "yaw":
            heading, k = None, 0
            base = level_pose(ref, 0.0, mid)
            axis = fm.WORLD_UP.copy()
        else:
            heading, k = mean_heading([a0[0], a1[0]]), (1 if pair == "pitch" else 2)
            base = level_pose(ref, heading, mid)
            axis = base[:3, 0].copy() if pair == "pitch" else base[:3, 2].copy()
            dh = abs(_wrap(a1[0] - a0[0]))
            if dh > POSE_WARN_DEG:
                warnings.append(f"{neg}、{pos} 两个标记的朝向差 {dh:.1f}°：取平均 {heading:+.1f}°")
        for name, a in ((neg, a0), (pos, a1)):
            other = [(ANGLE_LABEL[i], a[i]) for i in (1, 2) if i != k]
            worst = max(other, key=lambda t: abs(t[1]))
            if abs(worst[1]) > POSE_WARN_DEG:
                warnings.append(f"{name} 标记的{worst[0]} {worst[1]:+.1f}°（超过 {POSE_WARN_DEG:g}°）：只绕一个轴转，"
                                "录制时这个角是 0")
        marks_s = (float(a0[k]), float(a1[k]))
        margin, kind = margin_deg, "rotate"
    if marks_s[1] <= marks_s[0]:
        problems.append(f"{neg} 标记（{marks_s[0]:+.1f}）不在 {pos} 标记（{marks_s[1]:+.1f}）的负方向："
                        "两个标记记反了，或者拖的方向不对")
    lo, hi = marks_s[0] + margin, marks_s[1] - margin
    if not problems and hi <= lo:
        problems.append(f"{neg}、{pos} 两个标记只相距 {marks_s[1] - marks_s[0]:.1f} {'mm' if kind == 'translate' else '°'}，"
                        f"两端各收进余量 {margin:g} 后没有空间")
    return MarkPath(pair, kind, base, axis, heading, marks_s, lo, hi, warnings, problems)


# ------------------------------------------------------------------ 用逆解收两端
@dataclass
class FitLimits:
    joint_min: list
    joint_max: list
    max_joint_step_deg: float       # 每 10 mm / 2° 允许的关节变化（同 check_plan）
    min_camera_z_m: float
    max_joint_speed_deg_s: float
    speed: float                    # 录制速度：平移 mm/s、旋转 °/s（满速，不乘 --scale：演练和正式录同一条路径）
    accel: float | None             # speed-test 实测的加速度（mm/s²、°/s²）；None 时按全程匀速（偏保守）
    min_length: float               # 行程 mm / 转角 ° 的下限、上限
    max_length: float
    glass: Callable[[np.ndarray], float] | None = None   # 相机位姿 -> 画面离玻璃的角距离 °（fm.GlassCheck）
    glass_margin_deg: float = 1.0
    wiggle_deg: float = 0.0         # 平移两端的同步摆动（绕相机竖直轴）；旋转的同步摆动在转动范围以内，不另查
    wiggle_side: int = 0
    singular_margin_deg: float = 5.0  # J3、J5 离 0（奇异位形）至少这么多


@dataclass
class Fit:
    a: float                        # 收完后的范围（路径参数 s）
    b: float
    ends: list                      # [负端为什么停在这里, 正端为什么停在这里]
    max_joint_speed_deg_s: float | None
    problems: list = field(default_factory=list)
    seed: str = ""                  # 按哪个标记的构型（fit_best 填）
    wiggle_side: int | None = None  # 平移两端的同步摆动往哪边（+1 左、-1 右；fit_wiggle_sides 填）
    joints_a: list | None = None    # 两端的关节角（逆解）
    joints_b: list | None = None
    notes: list = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.b - self.a


def _speed_at(s: float, a: float, b: float, v: float, accel: float | None) -> float:
    """梯形速度曲线（从 a 起步、到 b 停下）在 s 处的速度；没有加速度时按全程匀速。"""
    if not accel:
        return v
    return min(v, math.sqrt(2.0 * accel * max(0.0, s - a)), math.sqrt(2.0 * accel * max(0.0, b - s)))


def fit_path(rpc, path: MarkPath, f_t_c: np.ndarray, seed_joints, lim: FitLimits, from_pos_end: bool = False) -> Fit:
    """沿路径在 [lo, hi] 上采样逆解，取能走的最长一段，再收到关节速度和两端的同步摆动都满足。
    seed_joints：逆解的初始参考（一个标记的关节角），路径只走这个构型；from_pos_end：从正端往负端逐点推
    （种子是正端标记时），否则从负端往正端。不动机械臂，只调控制器逆解。"""
    translate = path.kind == "translate"
    step = FIT_STEP_MM if translate else FIT_STEP_DEG
    step_tol = lim.max_joint_step_deg * step / (fm.PATH_STEP_MM if translate else fm.PATH_STEP_DEG)
    if path.problems:
        return Fit(path.lo, path.lo, ["", ""], None, list(path.problems))
    n = max(1, math.ceil((path.hi - path.lo) / step - 1e-9))
    ss = path.lo + (path.hi - path.lo) * np.arange(n + 1) / n
    inv = np.linalg.inv(f_t_c)
    qs: list = [None] * (n + 1)
    bad = [""] * (n + 1)                 # 这个采样点为什么不能走（空 = 能走）
    seed_config = config_of(seed_joints)
    ref = list(map(float, seed_joints))
    for k in (range(n, -1, -1) if from_pos_end else range(n + 1)):
        cam = path.pose(float(ss[k]))
        err, q = fm.inverse_kin(rpc, cam @ inv, ref)
        if q is None:
            bad[k] = f"逆解失败（错误码 {err}）"
            continue
        qs[k], ref = q, q
        out = [j + 1 for j in range(6) if not lim.joint_min[j] - 1e-6 <= q[j] <= lim.joint_max[j] + 1e-6]
        singular = singular_problem(q, lim.singular_margin_deg)
        z = float(cam[2, 3])
        clear = lim.glass(cam) if lim.glass else math.inf
        if out:
            bad[k] = f"关节 J{out} 超出允许范围（{np.round(q, 1).tolist()}）"
        elif config_of(q) != seed_config:
            bad[k] = f"构型变了（{describe_config(q)}）"
        elif singular:
            bad[k] = singular
        elif z < lim.min_camera_z_m:
            bad[k] = f"光心高度 {z:.3f} m 低于下限 {lim.min_camera_z_m} m"
        elif clear < lim.glass_margin_deg:
            bad[k] = f"画面离右窗只有 {clear:.2f}°"
    linked = [False] + [qs[k] is not None and qs[k - 1] is not None                 # 和前一个采样点之间关节连续
                        and max(abs(x - y) for x, y in zip(qs[k], qs[k - 1])) <= step_tol for k in range(1, n + 1)]

    # 能走的最长一段（中间关节连续）
    runs, i = [], None
    for k in range(n + 1):
        if bad[k]:
            i = None
            continue
        if i is None or not linked[k]:
            i = k
            runs.append([k, k])
        else:
            runs[-1][1] = k
    if not runs:
        reasons = sorted({b for b in bad if b})
        return Fit(path.lo, path.lo, ["", ""], None, ["整条路径都不能走：" + "；".join(reasons[:3])])
    centre = (path.lo + path.hi) / 2.0
    i, j = max(runs, key=lambda r: (ss[r[1]] - ss[r[0]], -abs((ss[r[0]] + ss[r[1]]) / 2.0 - centre)))

    def why(k: int) -> str:
        """能走的一段外面紧挨着的采样点 k 为什么不能走（能走却不在这一段里，就是和这一段之间关节跳了一大步）。"""
        if k < 0 or k > n:
            return f"标记收进余量（{signed(path.lo if k < 0 else path.hi)} {path.unit}）"
        return bad[k] or f"关节在 {signed(ss[k])} {path.unit} 处跳了一大步（可能换到另一组逆解）"

    ends = [why(i - 1), why(j + 1)]
    while ss[j] - ss[i] > lim.max_length + 1e-9:          # 超过上限：从离中点远的一头收
        if (ss[i] + ss[j]) / 2.0 > centre:
            j -= 1
            ends[1] = f"行程上限 {lim.max_length:g} {path.unit}"
        else:
            i += 1
            ends[0] = f"行程上限 {lim.max_length:g} {path.unit}"

    wiggle_cache: dict = {}

    def wiggle_problem(k: int) -> str | None:
        """平移端点 k 处的同步摆动：每 2° 逆解，关节在范围内、同一构型、不跳支、看不到右窗。"""
        if not translate or lim.wiggle_deg <= 0:
            return None
        if k not in wiggle_cache:
            cam0 = path.pose(float(ss[k]))
            found = None
            for side in ((+1, -1) if lim.wiggle_side == 0 else (lim.wiggle_side,)):
                prev = qs[k]
                m = max(1, math.ceil(lim.wiggle_deg / fm.PATH_STEP_DEG))
                for t in range(1, m + 1):
                    cam = fm.rotate_about(cam0, fm.camera_up(cam0), side * lim.wiggle_deg * t / m)
                    err, q = fm.inverse_kin(rpc, cam @ inv, prev)
                    if q is None:
                        found = f"同步摆动逆解失败（错误码 {err}）"
                    elif max(abs(x - y) for x, y in zip(q, prev)) > lim.max_joint_step_deg:
                        found = "同步摆动时关节跳了一大步"
                    elif any(not lim.joint_min[jj] - 1e-6 <= q[jj] <= lim.joint_max[jj] + 1e-6 for jj in range(6)):
                        found = f"同步摆动时关节超出允许范围（{np.round(q, 1).tolist()}）"
                    elif config_of(q) != seed_config:
                        found = "同步摆动时换了构型"
                    elif singular_problem(q, lim.singular_margin_deg):
                        found = "同步摆动时" + singular_problem(q, lim.singular_margin_deg)
                    elif lim.glass and lim.glass(cam) < lim.glass_margin_deg:
                        found = f"同步摆动时画面离右窗只有 {lim.glass(cam):.2f}°"
                    if found:
                        break
                    prev = q
                if found:
                    break
            wiggle_cache[k] = found
        return wiggle_cache[k]

    worst_speed = None
    while ss[j] - ss[i] >= lim.min_length - 1e-9 and j > i:
        a, b = float(ss[i]), float(ss[j])
        speeds = []
        for k in range(i + 1, j + 1):
            rate = max(abs(x - y) for x, y in zip(qs[k], qs[k - 1])) / (ss[k] - ss[k - 1])
            mid_s = (ss[k] + ss[k - 1]) / 2.0
            speeds.append((rate * _speed_at(mid_s, a, b, lim.speed, lim.accel), mid_s))
        worst_speed, where = max(speeds)
        if worst_speed > lim.max_joint_speed_deg_s * FIT_SPEED_FRACTION:
            if where - a < b - where:
                i += 1
                ends[0] = f"关节速度（{worst_speed:.0f} °/s，上限 {lim.max_joint_speed_deg_s:g}）"
            else:
                j -= 1
                ends[1] = f"关节速度（{worst_speed:.0f} °/s，上限 {lim.max_joint_speed_deg_s:g}）"
            continue
        moved = False
        for side, k in ((0, i), (1, j)):
            w = wiggle_problem(k)
            if w:
                if side == 0:
                    i += 1
                else:
                    j -= 1
                ends[side], moved = w, True
                break
        if not moved:
            break
    fit = Fit(float(ss[i]), float(ss[j]), ends, worst_speed, joints_a=qs[i], joints_b=qs[j])
    if fit.length < lim.min_length - 1e-9:
        fit.problems.append(f"能走的只有 {max(fit.length, 0.0):.1f} {path.unit}，少于下限 {lim.min_length:g} {path.unit}"
                            f"（负端：{ends[0]}；正端：{ends[1]}）")
    return fit


def fit_best(rpc, path: MarkPath, f_t_c: np.ndarray, neg_joints, pos_joints, lim: FitLimits) -> Fit:
    """按标记的构型收两端。两个标记构型相同：从负端标记的关节角起逆解。不同：一条直线 / 圆弧换不了构型，
    两种构型各试一次（从负端标记起往正端推、从正端标记起往负端推），取没问题、能走得长的那个，并在 notes 里写明。"""
    neg, pos = PAIRS[path.pair]
    if config_of(neg_joints) == config_of(pos_joints):
        fit = fit_path(rpc, path, f_t_c, neg_joints, lim)
        fit.seed = neg
        return fit
    tries = []
    for name, q, from_pos in ((neg, neg_joints, False), (pos, pos_joints, True)):
        f = fit_path(rpc, path, f_t_c, q, lim, from_pos_end=from_pos)
        f.seed = name
        tries.append(f)
    best = max(tries, key=lambda f: (not f.problems, f.length))
    other = pos if best.seed == neg else neg
    got = "；".join(f"按 {f.seed} 的构型" + (f"能走 {f.length:.1f} {path.unit}" if f.length > 0 else "走不了")
                   for f in tries)
    best.notes.append(f"{neg}、{pos} 两个标记构型不同（{neg}：{describe_config(neg_joints)}；"
                      f"{pos}：{describe_config(pos_joints)}），一条录制换不了构型：{got}，取 {best.seed} 的。"
                      f"靠近 {other} 那一端时机械臂的姿势和你拖的不一样，第一次低速演练时盯住")
    return best


def fit_wiggle_sides(rpc, path: MarkPath, f_t_c: np.ndarray, neg_joints, pos_joints, lim: FitLimits,
                     sides=(+1, -1)) -> Fit:
    """平移：两端的同步摆动只往左（+1）或只往右（-1）各收一次，取没问题、行程长的（一样长取前一个），写进
    fit.wiggle_side。摆动方向会挡住哪一端因位置而异（往一侧摆时腕部可能靠近奇异位形或关节限位）。旋转不用。"""
    if path.kind != "translate" or len(sides) == 1:
        lim = dataclasses.replace(lim, wiggle_side=sides[0]) if path.kind == "translate" else lim
        fit = fit_best(rpc, path, f_t_c, neg_joints, pos_joints, lim)
        fit.wiggle_side = sides[0] if path.kind == "translate" else None
        return fit
    tries = []
    for side in sides:
        f = fit_best(rpc, path, f_t_c, neg_joints, pos_joints, dataclasses.replace(lim, wiggle_side=side))
        f.wiggle_side = side
        tries.append(f)
    best = max(tries, key=lambda f: (not f.problems, round(f.length, 6)))
    other = next(f for f in tries if f is not best)
    if round(other.length, 6) != round(best.length, 6) or other.problems != best.problems:
        label = {+1: "往左", -1: "往右"}
        best.notes.append(f"同步摆动只{label[best.wiggle_side]}摆：{label[other.wiggle_side]}摆时"
                          + (f"只能走 {other.length:.1f} {path.unit}" if not other.problems else "走不了")
                          + f"（{'；'.join(e for e in other.ends if e) or '—'}）")
    return best


def start_joints(direction: str, fit: Fit) -> list | None:
    """录制 direction 起点的关节角（逆解）。"""
    return fit.joints_a if RECORD_PAIR[direction][1] > 0 else fit.joints_b


def record_ends(direction: str, path: MarkPath, fit: Fit) -> tuple[np.ndarray, np.ndarray, float]:
    """录制 direction 的起点、终点相机位姿和转角（旋转，°，正；平移为 0）。"""
    sign = RECORD_PAIR[direction][1]
    s0, s1 = (fit.a, fit.b) if sign > 0 else (fit.b, fit.a)
    return path.pose(s0), path.pose(s1), abs(s1 - s0) if path.kind == "rotate" else 0.0


def recording_segments(direction: str, path: MarkPath, fit: Fit, f_t_c: np.ndarray, cfg) -> list:
    """一条录制的全部段（fr5_motion.plan_translation / plan_rotation）：开头的 goto_start 从当前位姿直接移到录制起点。"""
    start, end, deg = record_ends(direction, path, fit)
    if path.kind == "translate":
        return fm.plan_translation(start, end, f_t_c, cfg)
    return fm.plan_rotation(start, path.axis * RECORD_PAIR[direction][1], deg, f_t_c, cfg)


# ------------------------------------------------------------------ 说明文字
def describe_path(path: MarkPath, fit: Fit | None = None) -> str:
    """路径的说明：方向、姿态、标记读数、收进余量后的范围，有 fit 时还有最终范围和两端为什么停在那里。"""
    neg, pos = PAIRS[path.pair]
    u = path.unit

    def mm(v):
        return np.round(np.asarray(v) * 1000.0).astype(int).tolist()

    if path.kind == "translate":
        what = ("竖直" if path.pair == "up_down"
                else f"水平、方向 {np.round(path.axis, 3).tolist()}（基坐标系）")
        head = f"{PAIR_LABEL[path.pair]}平移：{what}，相机水平、朝向 {signed(path.heading_deg)}°"
    else:
        axis = {"yaw": "竖直轴", "pitch": "水平相机的 x 轴", "roll": "水平相机的光轴"}[path.pair]
        head = (f"{PAIR_LABEL[path.pair]}：绕光心 {mm(path.base[:3, 3])} mm 的{axis}"
                + ("" if path.heading_deg is None else f"，朝向 {signed(path.heading_deg)}°"))
    lines = [head,
             f"  标记 {neg} {signed(path.marks_s[0])}、{pos} {signed(path.marks_s[1])} {u}；"
             f"收进余量后 {signed(path.lo)}～{signed(path.hi)} {u}（{max(path.hi - path.lo, 0.0):.1f} {u}）"]
    if fit is not None and fit.length > 0:
        a, b = path.pose(fit.a), path.pose(fit.b)
        where = (f"光心 {mm(a[:3, 3])} → {mm(b[:3, 3])} mm" if path.kind == "translate"
                 else f"{ANGLE_LABEL[{'yaw': 0, 'pitch': 1, 'roll': 2}[path.pair]]} {signed(fit.a)} → {signed(fit.b)}°")
        speed = (f"，关节最快 {fit.max_joint_speed_deg_s:.0f} °/s" if fit.max_joint_speed_deg_s is not None else "")
        lines.append(f"  实际 {signed(fit.a)}～{signed(fit.b)} {u}，共 {fit.length:.1f} {u}（{where}）{speed}")
        for side, name in ((0, neg), (1, pos)):
            lines.append(f"  {name} 端停在这里：{fit.ends[side] or '—'}")
        if fit.seed and fit.joints_a is not None:
            lines.append(f"  构型：和 {fit.seed} 标记相同（{describe_config(fit.joints_a)}）")
        if fit.wiggle_side in (+1, -1):
            lines.append(f"  同步摆动：只往{'左' if fit.wiggle_side > 0 else '右'}摆")
    if fit is not None:
        lines += [f"  注意：{m}" for m in fit.notes]
    return "\n".join(lines)
