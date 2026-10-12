"""相机保持水平的直线平移和同步摆动：规划、逆解检查、执行。由 run_session.py 调用，本文件没有可调参数。

坐标约定：位姿都在 FR5 基坐标系下；camera = base_T_camera（ZED 左目光学系：x 右、y 下、z 前）；
flange = camera · (flange_T_camera)⁻¹。FR5 位姿 [x, y, z mm, rx, ry, rz °]，R = Rz·Ry·Rx，工具 0、工件 0。

速度：MoveL 用百分比速度模式（velAccParamMode=0）。物理速度模式（mm/s）被这台控制器拒绝（182/183）。
百分比和实际速度的换算受 WebApp 速度百分比影响，由 run_session.py speed-test 实测（SpeedModel）。
指令非阻塞下发，再由 20004 状态判断到位，这样运动中随时可以发 StopMotion；
运动中实时检查法兰线速度，超过预期就中止（标定失效或 WebApp 速度改过时的保护）。
"""

from __future__ import annotations

import math
import time
import xmlrpc.client
from dataclasses import dataclass, field

import numpy as np

import fr5_groundtruth as gt

FX_1080 = 734.46            # ZED X HD1080 校正后 fx（像素），估算运动模糊用
WORLD_UP = np.array([0.0, 0.0, 1.0])
DIRECTIONS = {               # 方向名 -> (相机系轴名, 符号)；up/down 用基坐标系竖直方向
    "forward": ("z", +1), "backward": ("z", -1),
    "right": ("x", +1), "left": ("x", -1),
    "up": ("up", +1), "down": ("up", -1),
}
PATH_STEP_MM = 10.0          # 逆解检查的路径采样间隔
PATH_STEP_DEG = 2.0
ARRIVE_POS_MM = 0.5          # 判定到位的位置/姿态误差
ARRIVE_ROT_DEG = 0.05
MIN_PERCENT = 0.1            # MoveL 速度百分比的下限（0 会被控制器拒绝，错误码 172）
DEFAULT_RAMP_S = 0.5         # 没有标定时，估算时长用的加减速时间
SPEED_WINDOW_S = 0.05        # 实时速度和巡航速度都按跨 50 ms 的差分算
OVERSPEED_FACTOR = 1.5       # 实测线速度超过 预期 × 1.5 + 5 mm/s 就中止
OVERSPEED_MARGIN_MM_S = 5.0
STALL_S = 5.0                # 没到位、又连续这么久没有移动，就中止
STALL_MM, STALL_DEG = 0.05, 0.02
STATE_LOST_S = 2.0           # 运动中超过这么久没有新的状态包（20004 断开），就中止
FRESH_S = 0.1                # 发运动指令前，最新状态包必须是这么久之内到达的（正常每 10 ms 一包）
ONSET_MM, ONSET_DEG = 0.1, 0.05   # 实测用时：从离开起点这么远开始算
DAY_MS = 86_400_000


# ------------------------------------------------------------------ 位姿换算
def fr5_to_matrix(p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = gt.euler_to_matrix(p[3], p[4], p[5])
    t[:3, 3] = np.asarray(p[:3], dtype=float) / 1000.0
    return t


def matrix_to_fr5(t: np.ndarray) -> list[float]:
    r = t[:3, :3]
    ry = -math.asin(max(-1.0, min(1.0, r[2, 0])))
    rx = math.atan2(r[2, 1], r[2, 2])
    rz = math.atan2(r[1, 0], r[0, 0])
    return [*(t[:3, 3] * 1000.0).tolist(), math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def level_camera(camera: np.ndarray) -> np.ndarray:
    """同一光心，光轴取当前光轴的水平投影，图像 y 竖直向下（相机水平、无横滚）。"""
    z = camera[:3, 2].copy()
    z[2] = 0.0
    z /= np.linalg.norm(z)
    y = -WORLD_UP
    out = np.eye(4)
    out[:3, :3] = np.c_[np.cross(y, z), y, z]
    out[:3, 3] = camera[:3, 3]
    return out


def yaw_about_vertical(camera: np.ndarray, deg: float) -> np.ndarray:
    """绕过光心的竖直轴转动相机（相机保持水平）。"""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = camera.copy()
    out[:3, :3] = rz @ camera[:3, :3]
    return out


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa, qb = gt.matrix_to_quat(a[:3, :3]), gt.matrix_to_quat(b[:3, :3])
    return math.degrees(float(gt.rotation_angle_between(qa, qb)))


def blur_px(speed_mps: float, exposure_us: float, depth_m: float) -> float:
    """横向平移在 depth_m 处造成的运动模糊（1080p 像素）。前后平移在画面左右边缘约为它的 1.3 倍。"""
    return FX_1080 * speed_mps * exposure_us * 1e-6 / depth_m


# ------------------------------------------------------------------ 规划
@dataclass
class Segment:
    name: str
    kind: str                     # "hold" 或 "move"
    recorded: bool = True         # False：录制开始前的移动（到起点）
    duration_s: float = 0.0       # hold 的时长；move 的预计时长（apply_speed 填，0 表示未知）
    camera: np.ndarray | None = None
    flange: list | None = None
    joints: list | None = None
    speed_mm_s: float = 0.0       # 设计线速度（平移、移到起点）
    rate_deg_s: float = 0.0       # 设计角速度（同步摆动）；非 0 时按摆动换算速度百分比
    fixed_percent: float | None = None  # speed-test 用：固定的速度百分比
    distance_mm: float = 0.0      # 法兰行程、姿态变化（fill_geometry 填）
    angle_deg: float = 0.0
    percent: float | None = None  # MoveL 速度百分比（apply_speed 填）
    expect_mm_s: float | None = None    # 预计法兰线速度；speed_known=False 时是上限
    expect_deg_s: float | None = None
    speed_known: bool = False     # True：按速度标定算出的预计值
    log: dict = field(default_factory=dict)


@dataclass
class MotionConfig:
    speed_mps: float
    stroke_m: float
    hold_start_s: float
    hold_mid_s: float
    hold_end_s: float
    wiggle_deg: float
    wiggle_deg_s: float
    wiggle_cycles: int
    reposition_mps: float
    sync_test_hold_s: float


@dataclass
class SpeedModel:
    """MoveL 速度百分比 1% 对应的实际速度（已包含 WebApp 速度百分比的影响）。
    mm_s_per_pct：纯平移时的法兰线速度；wiggle_deg_s_per_pct：同步摆动（绕光心竖直轴转）时相机的角速度。
    calibrated=False 是没标定时的保守假设：1% = 默认速度的 1%（WebApp 速度百分比不起作用时最快）。"""
    mm_s_per_pct: float
    wiggle_deg_s_per_pct: float | None = None
    ramp_s: float = DEFAULT_RAMP_S
    calibrated: bool = False


def _move(name, camera, f_t_c, speed_mm_s=0.0, rate_deg_s=0.0, recorded=True, fixed_percent=None) -> Segment:
    return Segment(name, "move", recorded, camera=camera,
                   flange=matrix_to_fr5(camera @ np.linalg.inv(f_t_c)),
                   speed_mm_s=speed_mm_s, rate_deg_s=rate_deg_s, fixed_percent=fixed_percent)


def _wiggle(prefix, camera, f_t_c, cfg: MotionConfig) -> list[Segment]:
    """绕竖直轴 +θ、-θ 来回 cycles 次，最后回到原朝向。"""
    targets = []
    for _ in range(cfg.wiggle_cycles):
        targets += [cfg.wiggle_deg, -cfg.wiggle_deg]
    targets.append(0.0)
    return [_move(f"{prefix}_{i + 1}", yaw_about_vertical(camera, deg), f_t_c, rate_deg_s=cfg.wiggle_deg_s)
            for i, deg in enumerate(targets)]


def plan_recording(direction: str, centre: np.ndarray, f_t_c: np.ndarray, cfg: MotionConfig) -> list[Segment]:
    axis_name, sign = DIRECTIONS[direction]
    axis = WORLD_UP if axis_name == "up" else centre[:3, {"x": 0, "z": 2}[axis_name]]
    half = sign * axis * cfg.stroke_m / 2.0
    start, end = centre.copy(), centre.copy()
    start[:3, 3] -= half
    end[:3, 3] += half
    return ([_move("goto_start", start, f_t_c, cfg.reposition_mps * 1000.0, recorded=False),
             Segment("hold_start", "hold", duration_s=cfg.hold_start_s)]
            + _wiggle("wiggle_start", start, f_t_c, cfg)
            + [Segment("hold_before", "hold", duration_s=cfg.hold_mid_s),
               _move("translate", end, f_t_c, cfg.speed_mps * 1000.0),
               Segment("hold_after", "hold", duration_s=cfg.hold_mid_s)]
            + _wiggle("wiggle_end", end, f_t_c, cfg)
            + [Segment("hold_end", "hold", duration_s=cfg.hold_end_s)])


def plan_sync_test(centre: np.ndarray, f_t_c: np.ndarray, cfg: MotionConfig) -> list[Segment]:
    return ([_move("goto_start", centre, f_t_c, cfg.reposition_mps * 1000.0, recorded=False),
             Segment("hold_start", "hold", duration_s=cfg.hold_start_s)]
            + _wiggle("wiggle_start", centre, f_t_c, cfg)
            + [Segment("hold_middle", "hold", duration_s=cfg.sync_test_hold_s)]
            + _wiggle("wiggle_end", centre, f_t_c, cfg)
            + [Segment("hold_end", "hold", duration_s=cfg.hold_end_s)])


def plan_speed_test(centre: np.ndarray, f_t_c: np.ndarray, cfg: MotionConfig,
                    travel_mm: float, test_percent: float) -> list[Segment]:
    """速度标定：在起点竖直上移 travel_mm 再回来，绕光心竖直轴转 wiggle_deg 再回来。
    去程用固定百分比 test_percent；回程的百分比由 run_session 按去程实测换算到录制速度 / 摆动角速度。"""
    up = centre.copy()
    up[:3, 3] += WORLD_UP * travel_mm / 1000.0
    yaw = yaw_about_vertical(centre, cfg.wiggle_deg)
    return [_move("goto_start", centre, f_t_c, cfg.reposition_mps * 1000.0, recorded=False),
            _move("test_up", up, f_t_c, fixed_percent=test_percent),
            _move("test_down", centre, f_t_c, cfg.speed_mps * 1000.0),
            _move("test_yaw", yaw, f_t_c, rate_deg_s=cfg.wiggle_deg_s, fixed_percent=test_percent),
            _move("test_yaw_back", centre, f_t_c, rate_deg_s=cfg.wiggle_deg_s)]


def fill_geometry(segments: list[Segment], current_flange: np.ndarray) -> None:
    """填每段 move 的法兰行程和姿态变化（相对上一段的终点）。"""
    prev = current_flange
    for seg in segments:
        if seg.kind != "move":
            continue
        target = fr5_to_matrix(seg.flange)
        seg.distance_mm = float(np.linalg.norm(target[:3, 3] - prev[:3, 3]) * 1000.0)
        seg.angle_deg = rotation_deg(prev, target)
        prev = target


def apply_speed(segments: list[Segment], model: SpeedModel | None, scale: float) -> list[str]:
    """按速度模型给每段 move 算速度百分比、预计速度和时长（先 fill_geometry）。
    model=None：只按设计速度估时长，不算百分比（plan 在没标定时用）。返回问题列表。"""
    problems = []
    for seg in segments:
        if seg.kind != "move":
            continue
        k_lin = model.mm_s_per_pct if model else None
        k_rot = model.wiggle_deg_s_per_pct if model else None
        p = lin = rot = t = None
        if seg.rate_deg_s > 0:                    # 同步摆动：按摆动标定换算
            p = seg.fixed_percent or (seg.rate_deg_s * scale / k_rot if k_rot else None)
            if model is None:
                rot = seg.rate_deg_s * scale
            elif k_rot and p:
                rot = k_rot * p
            t = seg.angle_deg / rot if rot else None
            lin = seg.distance_mm / t if t else (k_lin * p if k_lin and p else None)
        else:                                     # 平移（含移到起点）；姿态变化的快慢由控制器另外限制，不预估
            p = seg.fixed_percent or (seg.speed_mm_s * scale / k_lin if k_lin else None)
            lin = k_lin * p if k_lin and p else seg.speed_mm_s * scale
            t = seg.distance_mm / lin if lin else None
        seg.percent, seg.expect_mm_s, seg.expect_deg_s = p, lin, rot
        seg.speed_known = model is not None and model.calibrated and t is not None
        seg.duration_s = t + (model.ramp_s if model else DEFAULT_RAMP_S) if t is not None else 0.0
        if p is not None and not MIN_PERCENT <= p <= 100.0:
            problems.append(f"{seg.name}: 需要速度百分比 {p:.2f}%，超出 [{MIN_PERCENT}, 100]"
                            + ("（WebApp 速度百分比太低？调高后重跑 speed-test）" if p > 100 else ""))
    return problems


# ------------------------------------------------------------------ 逆解检查（只读，不动机械臂）
def check_plan(rpc, segments: list[Segment], current_flange: np.ndarray, current_joints, f_t_c,
               joint_min, joint_max, max_joint_step_deg: float, min_camera_z_m: float) -> list[str]:
    """沿 MoveL 的法兰直线路径采样，用控制器逆解检查：有解、关节在范围内、不跳支、相机不低于下限。
    同时填好每段的行程、姿态变化和目标关节角。返回问题列表（空表示通过）。"""
    fill_geometry(segments, current_flange)
    problems = []
    ref = list(map(float, current_joints))
    prev = current_flange
    q_prev = gt.matrix_to_quat(prev[:3, :3])
    for seg in segments:
        if seg.kind != "move":
            continue
        target = fr5_to_matrix(seg.flange)
        q_t = gt.matrix_to_quat(target[:3, :3])
        n = max(1, math.ceil(seg.distance_mm / PATH_STEP_MM), math.ceil(seg.angle_deg / PATH_STEP_DEG))
        for k in range(1, n + 1):
            u = k / n
            sample = np.eye(4)
            sample[:3, 3] = prev[:3, 3] + u * (target[:3, 3] - prev[:3, 3])
            sample[:3, :3] = gt.quat_to_matrix(gt.slerp(q_prev, q_t, np.array(u)))
            cam_z = float((sample @ f_t_c)[2, 3])
            if cam_z < min_camera_z_m:
                problems.append(f"{seg.name}: 相机光心高度 {cam_z:.3f} m 低于下限 {min_camera_z_m} m")
                break
            ret = rpc.GetInverseKinRef(0, matrix_to_fr5(sample), ref)
            err, q = (ret[0], ret[1]) if isinstance(ret, tuple) else (ret, None)
            if err != 0 or q is None:
                problems.append(f"{seg.name}: 路径 {u:.0%} 处控制器逆解失败（错误码 {err}）")
                break
            q = list(map(float, q))
            out = [i + 1 for i in range(6) if not joint_min[i] <= q[i] <= joint_max[i]]
            if out:
                problems.append(f"{seg.name}: 路径 {u:.0%} 处关节 J{out} 超出允许范围：{np.round(q, 1).tolist()}")
                break
            step = max(abs(a - b) for a, b in zip(q, ref))
            if step > max_joint_step_deg:
                problems.append(f"{seg.name}: 路径 {u:.0%} 处关节一步变化 {step:.1f}°，可能跳到另一组逆解")
                break
            ref = q
        seg.joints = ref
        prev, q_prev = target, q_t
    return problems


# ------------------------------------------------------------------ 实测速度
def cruise_rate(t_s, progress, window_s: float = SPEED_WINDOW_S) -> dict | None:
    """巡航段速度：只看总进度 20%–80% 之间，按跨 window_s 的中心差分。progress 是累计路程或转角。"""
    t, s = np.asarray(t_s, dtype=float), np.asarray(progress, dtype=float)
    keep = np.r_[True, np.diff(t) > 0]
    t, s = t[keep], s[keep]
    if len(t) < 10 or s[-1] <= 0:
        return None
    h = window_s / 2.0
    tc = t[(s > 0.2 * s[-1]) & (s < 0.8 * s[-1])]
    tc = tc[(tc - h >= t[0]) & (tc + h <= t[-1])]
    if len(tc) < 3:
        return None
    v = (np.interp(tc + h, t, s) - np.interp(tc - h, t, s)) / window_s
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "ripple": float(v.std() / v.mean()) if v.mean() > 0 else None}


def recent_speed(samples, j: int) -> tuple[int, float | None]:
    """samples 末尾跨至少 SPEED_WINDOW_S 的法兰线速度（mm/s）。j 是上次的窗口起点，返回新的 j 和速度。"""
    def age(i: int) -> float:
        return ((samples[-1][0] - samples[i][0]) % DAY_MS) / 1000.0
    while j + 1 < len(samples) and age(j + 1) >= SPEED_WINDOW_S:
        j += 1
    if age(j) < SPEED_WINDOW_S:
        return j, None
    return j, float(np.linalg.norm(np.subtract(samples[-1][1][:3], samples[j][1][:3]))) / age(j)


def angles_to(rots: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """每个旋转矩阵 rots[i]（N×3×3）相对 ref 的转角（°），向量化。"""
    d = np.einsum("ji,njk->nik", ref, rots)                      # refᵀ · rots[i]
    cos = (np.trace(d, axis1=1, axis2=2) - 1.0) / 2.0
    sin = np.linalg.norm(np.stack([d[:, 2, 1] - d[:, 1, 2], d[:, 0, 2] - d[:, 2, 0],
                                   d[:, 1, 0] - d[:, 0, 1]], axis=1), axis=1) / 2.0
    return np.degrees(np.arctan2(sin, cos))


def motion_stats(samples) -> dict:
    """一段运动的实测。samples = [(控制器时间 ms, 法兰位姿 6 个数), ...]，含运动前后的静止样本。
    全部向量化：2000 个样本也只要几毫秒。逐个样本算会占住 Python 约 1 s，接收线程读不过来，控制器就断开 20004。"""
    ms = np.array([s[0] for s in samples], dtype=np.int64)
    t = ((ms - ms[0]) % DAY_MS) / 1000.0
    arr = np.array([s[1] for s in samples], dtype=float)
    xyz = arr[:, :3]
    rots = gt.euler_to_matrix(arr[:, 3], arr[:, 4], arr[:, 5])
    path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    angle = angles_to(rots, rots[0])
    out = {"samples": len(samples), "path_mm": float(path[-1]), "angle_deg": float(angle[-1]),
           "max_gap_s": float(np.diff(t).max()) if len(t) > 1 else None,   # 状态包最大间隔（断线时变大）
           "linear_mm_s": cruise_rate(t, path) if path[-1] > 2.0 else None,
           "angular_deg_s": cruise_rate(t, angle) if angle[-1] > 0.5 else None,
           "duration_s": None, "ramp_s": None}
    # 用时：从离开起点 onset 算到离终点不到 onset（用到起点/终点的距离，不受静止时位置噪声累积的影响）
    if out["linear_mm_s"]:
        main, onset = out["linear_mm_s"], ONSET_MM
        from_start, to_end = np.linalg.norm(xyz - xyz[0], axis=1), np.linalg.norm(xyz - xyz[-1], axis=1)
    elif out["angular_deg_s"]:
        main, onset = out["angular_deg_s"], ONSET_DEG
        from_start, to_end = angle, angles_to(rots, rots[-1])
    else:
        return out
    on = np.nonzero(from_start > onset)[0]
    off = np.nonzero((to_end < onset) & (np.arange(len(t)) > (on[0] if len(on) else len(t))))[0]
    if len(on) and len(off) and main["mean"] > 0:
        out["duration_s"] = float(t[off[0]] - t[on[0]])
        # 匀速走完同样距离所需时间之外多出来的部分，约等于加速（或减速）用时 v/a
        ideal = (from_start[-1] - 2.0 * onset) / main["mean"]
        out["ramp_s"] = max(0.0, out["duration_s"] - ideal)
    return out


# ------------------------------------------------------------------ 执行
class MotionAborted(RuntimeError):
    pass


def emergency_stop(ip: str) -> None:
    """用新的 XML-RPC 连接发 StopMotion，不受主连接状态影响。"""
    try:
        xmlrpc.client.ServerProxy(f"http://{ip}:20003").StopMotion()
    except Exception as exc:  # noqa: BLE001 - 尽力而为，调用方还会提示人工急停
        print(f"[motion] 发送 StopMotion 失败：{exc}，请立即按急停！")


def state_problem(pkg) -> str | None:
    if pkg is None:
        return "还没收到机械臂状态"
    if pkg.EmergencyStop:
        return "急停已按下"
    if pkg.main_code or pkg.sub_code:
        return f"机械臂报警 main_code={pkg.main_code} sub_code={pkg.sub_code}"
    if pkg.safety_stop0_state or pkg.safety_stop1_state:
        return "安全停止信号有效"
    if pkg.rbtEnableState != 1:
        return f"机械臂未使能（rbtEnableState={pkg.rbtEnableState}）"
    if (pkg.tool, pkg.user) != (0, 0):
        return f"工具号/工件号不是 0（{pkg.tool}/{pkg.user}）"
    return None


def ctrl_ms(pkg) -> int:
    """状态包里的控制器本地时间（当天的毫秒数）。"""
    return ((pkg.hour * 60 + pkg.minute) * 60 + pkg.second) * 1000 + pkg.millisecond


class MotionRunner:
    """逐段执行：hold 只监视状态；move 发 MoveL 后由状态判断到位，运动中做各项保护并记录实测。
    recorder 需提供 wait_state() 和 latest_monotonic（StateRecorder）。"""

    def __init__(self, recorder, rpc, accel_percent: float):
        self.rec, self.rpc, self.accel_percent = recorder, rpc, accel_percent
        self._count = -1

    def latest(self, timeout: float = 0.5):
        """最新的状态包；超时没有新包时返回上一个（state_age() 会变大）。"""
        self._count, pkg = self.rec.wait_state(self._count, timeout)
        return pkg

    def state_age(self) -> float:
        """最新状态包到达至今多久（s），按录制器记下的到达时刻算：
        断线前已到达、但本对象还没取过的包，计数是新的，内容却可能已经过时。"""
        arrived = self.rec.latest_monotonic
        return math.inf if arrived is None else time.monotonic() - arrived

    def fresh(self, what: str):
        """等一个 FRESH_S 内到达的状态包；STATE_LOST_S 内等不到就中止（还没发运动指令）。"""
        end = time.monotonic() + STATE_LOST_S
        while time.monotonic() < end:
            pkg = self.latest(0.2)
            if self.state_age() < FRESH_S:
                return pkg
        raise MotionAborted(f"{what}: {STATE_LOST_S:.0f} s 没收到新的机械臂状态（20004 断开？），没有发运动指令")

    def precheck(self) -> None:
        pkg = self.fresh("开始前")
        problem = state_problem(pkg)
        if problem is None and pkg.robot_state != 1:
            problem = f"机械臂不在停止状态（robot_state={pkg.robot_state}）"
        if problem:
            raise MotionAborted(problem + "，请在 WebApp 处理后再运行")

    def run(self, seg: Segment) -> None:
        seg.log["start_box_ns"] = time.time_ns()
        if seg.kind == "hold":
            self._hold(seg)
        else:
            self._move(seg)
        seg.log["end_box_ns"] = time.time_ns()

    def _hold(self, seg: Segment) -> None:
        end = time.monotonic() + seg.duration_s
        while time.monotonic() < end:
            problem = state_problem(self.latest())
            if problem:
                raise MotionAborted(f"{seg.name}: {problem}")

    def _move(self, seg: Segment) -> None:
        if seg.percent is None or not MIN_PERCENT <= seg.percent <= 100.0:
            raise MotionAborted(f"{seg.name}: 速度百分比无效（{seg.percent}）")
        target = fr5_to_matrix(seg.flange)
        start_pkg = self.fresh(seg.name)
        start_pos = np.array(start_pkg.flange_cur_pos[:3])
        samples = [(ctrl_ms(start_pkg), list(start_pkg.flange_cur_pos))]
        err = self.rpc.MoveL(desc_pos=seg.flange, tool=0, user=0, joint_pos=seg.joints,
                             vel=seg.percent, ovl=100.0, blendR=0.0, oacc=self.accel_percent,
                             velAccParamMode=0)
        seg.log["cmd_box_ns"] = time.time_ns()
        seg.log["movel"] = {"vel_percent": seg.percent, "ovl": 100.0, "oacc": self.accel_percent}
        if err != 0:
            raise MotionAborted(f"{seg.name}: MoveL 返回错误码 {err}")
        # 超时只兜底；没到位又不动由 STALL_S 发现
        timeout = time.monotonic() + (2.0 * seg.duration_s + 60.0 if seg.speed_known else 300.0)
        limit = (seg.expect_mm_s * OVERSPEED_FACTOR + OVERSPEED_MARGIN_MM_S) if seg.expect_mm_s else None
        moving_seen = False
        mark_pose, mark_time = fr5_to_matrix(samples[0][1]), time.monotonic()
        j = 0                                     # 实时速度窗口的起点
        try:
            while True:
                pkg = self.latest()
                problem = state_problem(pkg)
                if problem:
                    raise MotionAborted(f"{seg.name}: {problem}")
                if self.state_age() > STATE_LOST_S:
                    raise MotionAborted(f"{seg.name}: 超过 {STATE_LOST_S:.0f} s 没收到机械臂状态（20004 断开？），已中止")
                pose6 = list(pkg.flange_cur_pos)
                now_ms = ctrl_ms(pkg)
                if (now_ms - samples[-1][0]) % DAY_MS != 0:
                    samples.append((now_ms, pose6))
                pos = np.array(pose6[:3])
                if not moving_seen and (pkg.robot_state == 2 or np.linalg.norm(pos - start_pos) > 0.3):
                    moving_seen = True
                    seg.log["moving_box_ns"] = time.time_ns()
                j, v = recent_speed(samples, j)
                if limit is not None and v is not None and v > limit:
                    raise MotionAborted(f"{seg.name}: 实测法兰线速度 {v:.1f} mm/s 超过上限 {limit:.1f} mm/s，"
                                        "已中止（WebApp 速度百分比改过？重跑 speed-test）")
                current = fr5_to_matrix(pose6)
                arrived = (np.linalg.norm(current[:3, 3] - target[:3, 3]) * 1000.0 < ARRIVE_POS_MM
                           and rotation_deg(current, target) < ARRIVE_ROT_DEG)
                if arrived and pkg.robot_state == 1:
                    seg.log["arrived_box_ns"] = time.time_ns()
                    return
                if (np.linalg.norm(current[:3, 3] - mark_pose[:3, 3]) * 1000.0 > STALL_MM
                        or rotation_deg(current, mark_pose) > STALL_DEG):
                    mark_pose, mark_time = current, time.monotonic()
                elif time.monotonic() - mark_time > STALL_S:
                    raise MotionAborted(f"{seg.name}: {STALL_S:.0f} s 没有移动，也没到位（robot_state={pkg.robot_state}）")
                if time.monotonic() > timeout:
                    raise MotionAborted(f"{seg.name}: 超时没有到位（robot_state={pkg.robot_state}）")
        finally:
            if len(samples) > 1:
                seg.log["measured"] = motion_stats(samples)
