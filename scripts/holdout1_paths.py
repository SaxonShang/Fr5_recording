"""holdout1 路径组的路径表：验证数据集要录的全部路径，数值来自 kalman-fusion-lab 实验记录 13.11（2026-11-09 定稿）。

改路径只改这个文件。plan 不通过时人工改这里，不自动收缩；改过之后要重新录带板验收（同一路径的所有录像用同一组值）。

坐标约定（和 run_session.py 一致）：
  位移 x、y、z 是相对本条参考位置（reference_mm，左目光心在 FR5 基坐标系下的 mm）的 mm，基坐标系：x 前、y 左、z 上。
  朝向零点是基坐标 +X，向左为正；俯仰向上为正；横滚以画面顶部向右为正。
  姿态由水平朝 +X 的相机依次得到：俯仰（绕相机 x）、横滚（绕光轴）、朝向（绕竖直轴）。

一条路径 = 起点 + 若干段。每段是一条 MoveL 或 MoveC，blendR=0，到位后停住（段名 leg_k，停留段名 wp_k，最后一段后面接 hold_after）。
"""

from __future__ import annotations

from dataclasses import dataclass

PATH_SET = "holdout1"
STOP_S = 1.0                  # 路径点最短停留：真值修正取停留段中间一半的 15 帧


@dataclass(frozen=True)
class Pose:
    x: float = 0.0            # mm，相对参考位置
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0      # °
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class Leg:
    target: Pose
    label: str                # 到达点的名字。真值修正按它查：往返经过同一位置也算不同的到访，名字不同
    motion: str = "L"         # "L"：MoveL；"C"：MoveC（绕光心转，位置不变）
    speed_mm_s: float = 0.0   # MoveL：法兰目标线速度
    max_rate_deg_s: float = 0.0   # MoveL：同时平移和转动时，转动不超过这个角速度（0 = 纯平移）
    rate_deg_s: float = 0.0   # MoveC：角速度
    hold_s: float = STOP_S    # 到位后停留（最后一段的停留是 hold_after_s）


@dataclass(frozen=True)
class WaypointPath:
    description: str
    reference_mm: tuple
    seed_joints: tuple        # 定构型用的关节角（J3 > 0、J5 < 0），plan 由它求起点逆解
    legs: tuple
    start: Pose = Pose()
    start_label: str = "start"
    correction_key: str | None = None  # waypoint_corrections.json 里的键；None = 这条路径不做路径点修正
    wiggle_axis: str = "vertical"      # 同步摆动的轴："vertical" 相机竖直轴；"optical" 光轴（横滚）
    hold_before_s: float = 3.0
    hold_after_s: float = 3.0
    board: str = ""           # 带板录制时标定板放哪里（plan 打印）

    @property
    def labels(self) -> list[str]:
        return [self.start_label] + [leg.label for leg in self.legs]


# 各路径的参考位置（13.11），seed 由解析逆解在 J3 > 0、J5 < 0 的构型里算出
ROLL_REF, ROLL_SEED = (789.0, -264.0, 555.0), (142.3, -48.2, 61.0, -192.7, -50.8, 0.0)
TRANS_REF, TRANS_SEED = (470.0, -40.0, 525.0), (138.6, -97.3, 131.5, -214.1, -47.1, 0.0)
S_REF, S_SEED = (550.0, -150.0, 450.0), (131.9, -66.3, 108.3, -221.9, -20.4, -0.1)
EIGHT_REF, EIGHT_SEED = (600.0, -300.0, 550.0), (127.0, -66.0, 90.4, -204.4, -35.5, 0.0)
ACCEL_REF, ACCEL_SEED = (600.0, -300.0, 650.0), (127.0, -70.1, 81.0, -190.8, -35.5, 0.0)

ROLL_DEG_S = 50.0
TRANS_MM_S, TRANS_STROKE_MM, TRANS_TILT_DEG, TRANS_STOP_S = 500.0, 400.0, 25.0, 3.0
S_MM_S, EIGHT_MM_S, LEG_MAX_DEG_S = 300.0, 250.0, 50.0
ACCEL_STATIC_DEG_S, ACCEL_STATIC_STOP_S = 20.0, 7.0
ACCEL_SLOW_DEG_S, ACCEL_FAST_DEG_S = 15.0, 35.0

# ------------------------------------------------------------------ 横滚去回
ROLL_BIDIR = WaypointPath(
    description="横滚去回：中立 → +50° → −50° → 中立，绕左目光心 50 °/s，两端各停 4 s",
    reference_mm=ROLL_REF, seed_joints=ROLL_SEED, correction_key="roll_bidir", wiggle_axis="optical",
    start_label="neutral_start",
    legs=(Leg(Pose(roll=+50.0), "pos50", "C", rate_deg_s=ROLL_DEG_S, hold_s=4.0),
          Leg(Pose(roll=-50.0), "neg50", "C", rate_deg_s=ROLL_DEG_S, hold_s=4.0),
          Leg(Pose(), "neutral_end", "C", rate_deg_s=ROLL_DEG_S)),
    board="板中心在光轴上，距光心约 1.1 m")


# ------------------------------------------------------------------ 两种姿态走同一段平移
def _trans_block(prefix: str, pitch: float) -> tuple:
    """同一姿态下的四段：上移、返回、前移、返回。"""
    up = Pose(z=TRANS_STROKE_MM, pitch=pitch)
    forward = Pose(x=TRANS_STROKE_MM, pitch=pitch)
    home = Pose(pitch=pitch)
    return tuple(Leg(p, f"{prefix}_{name}", speed_mm_s=TRANS_MM_S, hold_s=TRANS_STOP_S)
                 for p, name in ((up, "up"), (home, "back_up"), (forward, "forward"), (home, "back_forward")))


_TO_TILT = Leg(Pose(pitch=TRANS_TILT_DEG), "tilt_home", "C", rate_deg_s=ROLL_DEG_S, hold_s=TRANS_STOP_S)
_TO_LEVEL = Leg(Pose(), "level_home", "C", rate_deg_s=ROLL_DEG_S, hold_s=TRANS_STOP_S)
_TRANS_COMMON = dict(reference_mm=TRANS_REF, seed_joints=TRANS_SEED, correction_key="trans_two_attitude",
                     board="板中心约 z = 975 mm（比路径中点高约 0.25 m，要支架），水平距离 1.3–1.5 m，正对 +X")
TRANS_LEVEL_FIRST = WaypointPath(
    description="两种姿态的平移，水平先走：水平 4 段 → 俯仰 +25° → 同样 4 段 → 转回水平",
    legs=_trans_block("level", 0.0) + (_TO_TILT,) + _trans_block("tilt", TRANS_TILT_DEG) + (_TO_LEVEL,), **_TRANS_COMMON)
TRANS_TILT_FIRST = WaypointPath(
    description="两种姿态的平移，倾斜先走：俯仰 +25° → 4 段 → 转回水平 → 水平 4 段",
    legs=(_TO_TILT,) + _trans_block("tilt", TRANS_TILT_DEG) + (_TO_LEVEL,) + _trans_block("level", 0.0), **_TRANS_COMMON)

# ------------------------------------------------------------------ S 形折线
#            x     y   朝向   横滚
_S_POINTS = {"P0": (0.0, +175.0, +40.0, +20.0), "P1": (0.0, -175.0, 0.0, -20.0),
             "P2": (+100.0, -175.0, +20.0, 0.0), "P3": (+100.0, +175.0, +40.0, -20.0),
             "P4": (+200.0, +175.0, +20.0, 0.0), "P5": (+200.0, -175.0, 0.0, +20.0)}
_S_START = Pose(heading=+20.0)
_S_ORDER = ("P0", "P1", "P2", "P3", "P4", "P5", "P4", "P3", "P2", "P1", "P0")


def _s_legs() -> tuple:
    legs = [Leg(Pose(x=_S_POINTS[n][0], y=_S_POINTS[n][1], heading=_S_POINTS[n][2], roll=_S_POINTS[n][3]),
                f"{i:02d}_{n}", speed_mm_s=S_MM_S, max_rate_deg_s=LEG_MAX_DEG_S)
            for i, n in enumerate(_S_ORDER, 1)]
    return tuple(legs) + (Leg(_S_START, f"{len(legs) + 1:02d}_S", speed_mm_s=S_MM_S, max_rate_deg_s=LEG_MAX_DEG_S),)


S_PATH = WaypointPath(
    description="S 形折线 12 段：MoveL，法兰 0.30 m/s，每段转动不超过 50 °/s，每点停 1 s",
    reference_mm=S_REF, seed_joints=S_SEED, start=_S_START, legs=_s_legs(), correction_key="s_path",
    board="朝向 +20° 的方向，距 S 点光心约 1.2 m")

# ------------------------------------------------------------------ 空间 8 字（仅完整版）
#            x      y      z     俯仰   横滚
_Q_POINTS = {"Q0": (0.0, 0.0, 0.0, 0.0, 0.0),
             "Q1": (+50.0, +100.0, +60.0, +10.0, +10.0), "Q2": (0.0, +200.0, 0.0, 0.0, +15.0),
             "Q3": (-50.0, +100.0, -60.0, -10.0, +10.0), "Q5": (+50.0, -100.0, +60.0, +10.0, -10.0),
             "Q6": (0.0, -200.0, 0.0, 0.0, -15.0), "Q7": (-50.0, -100.0, -60.0, -10.0, -10.0)}
_Q_LOOP = ("Q1", "Q2", "Q3", "Q0", "Q5", "Q6", "Q7", "Q0")
_EIGHT_LOOPS = 3


def _q(name: str) -> Pose:
    x, y, z, pitch, roll = _Q_POINTS[name]
    return Pose(x=x, y=y, z=z, pitch=pitch, roll=roll)


EIGHT = WaypointPath(
    description="空间 8 字 3 圈：MoveL，法兰 0.25 m/s，每段转动不超过 50 °/s，每点停 1 s",
    reference_mm=EIGHT_REF, seed_joints=EIGHT_SEED, correction_key="eight",
    legs=tuple(Leg(_q(n), f"{i:02d}_{n}", speed_mm_s=EIGHT_MM_S, max_rate_deg_s=LEG_MAX_DEG_S)
               for i, n in enumerate(_Q_LOOP * _EIGHT_LOOPS, 1)),
    board="正对 +X，距起点约 1.2 m")

# ------------------------------------------------------------------ 加计辨识
_ACCEL_ATTITUDES = ((0.0, 0.0), (+25.0, 0.0), (-25.0, 0.0), (0.0, +40.0), (0.0, -40.0),
                    (+20.0, +30.0), (+20.0, -30.0), (-20.0, +30.0), (-20.0, -30.0), (0.0, 0.0))     # （俯仰，横滚）
_ACCEL_BOARD = "距光心 0.9–1.4 m"


def _accel_static(attitudes: tuple, description: str) -> WaypointPath:
    return WaypointPath(
        description=description, reference_mm=ACCEL_REF, seed_joints=ACCEL_SEED, board=_ACCEL_BOARD,
        hold_before_s=ACCEL_STATIC_STOP_S, hold_after_s=ACCEL_STATIC_STOP_S,
        legs=tuple(Leg(Pose(pitch=a, roll=b), f"{i:02d}_{a:+.0f}_{b:+.0f}", "C", rate_deg_s=ACCEL_STATIC_DEG_S,
                       hold_s=ACCEL_STATIC_STOP_S)
                   for i, (a, b) in enumerate(attitudes[1:], 1)))


_ACCEL_SWEEP = ((+25.0, 0.0), (-25.0, 0.0), (0.0, 0.0), (0.0, +40.0), (0.0, -40.0), (0.0, 0.0),
                (+20.0, +30.0), (-20.0, -30.0), (0.0, 0.0))


def _accel_rate(rates: tuple, description: str) -> WaypointPath:
    legs, i = [], 0
    for rate in rates:
        for a, b in _ACCEL_SWEEP:
            i += 1
            legs.append(Leg(Pose(pitch=a, roll=b), f"{i:02d}_{a:+.0f}_{b:+.0f}", "C", rate_deg_s=rate))
    return WaypointPath(description=description, reference_mm=ACCEL_REF, seed_joints=ACCEL_SEED, board=_ACCEL_BOARD,
                        legs=tuple(legs))


# ------------------------------------------------------------------ 录制名 → 路径
PATHS = {
    "accel_static_a": _accel_static(_ACCEL_ATTITUDES, "多姿态停驻：10 个姿态每个停 7 s，20 °/s"),
    "accel_static_b": _accel_static(_ACCEL_ATTITUDES[::-1], "多姿态停驻：访问顺序和 a 相反"),
    "accel_rate_slow_first": _accel_rate((ACCEL_SLOW_DEG_S, ACCEL_FAST_DEG_S), "双轴变速：同一序列先 15 °/s 再 35 °/s"),
    "accel_rate_fast_first": _accel_rate((ACCEL_FAST_DEG_S, ACCEL_SLOW_DEG_S), "双轴变速：同一序列先 35 °/s 再 15 °/s"),
    "roll_bidir": ROLL_BIDIR,
    "trans_level_first": TRANS_LEVEL_FIRST,
    "trans_tilt_first": TRANS_TILT_FIRST,
    "s_path": S_PATH,
    "eight": EIGHT,
}

# 评测端 conditions.yaml 的条件名：录制名 + --tag → 条件名（scene 只用于 s_path）
CONDITION_OF = {"roll_bidir": "roll_bidir", "trans_level_first": "trans_two_attitude",
                "trans_tilt_first": "trans_two_attitude", "eight": "eight"}
S_SCENE_TAGS = {"sceneA": "s_path_A", "sceneB": "s_path_B", "sceneC": "s_path_C"}
