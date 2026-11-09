#!/usr/bin/env python3
"""验证数据集（holdout1）一条录制的总控：FR5 运动（fr5_motion）+ 机械臂状态（fr5_record）+ 图像和 IMU（zed_capture）+ 真值（fr5_groundtruth）。
路径表在 holdout1_paths.py，其他可改参数都在本文件开头。

在 Box 自己的终端里运行（人在机械臂旁边，手边有急停）：
    python3 run_session.py plan [名称 ...]      空跑检查，不动机械臂；不给名称时检查全部 holdout1 路径
    python3 run_session.py speed-test           速度标定（会先确认，不开相机）。WebApp 速度百分比变了才要重跑
    python3 run_session.py record <名称>        录制一条（会先确认）
    python3 run_session.py warmup <路径> --minutes 30   开录前预热：重复走一条路径，不开相机、不存数据（会先确认）

录制名称：holdout1 路径 accel_static_a/b、accel_rate_slow_first/fast_first、roll_bidir、trans_level_first/tilt_first、s_path、eight；
开工核对 right、roll_right（路径组 fast2，由 LIMIT_POSES 里的拖动标记定，这份文件是只读输入）。
选项：--scale 0.25  按比例降速（路径不变）；--yes  跳过确认（只在你已经看过计划时用）；
      --tag board  加在录制目录名后面（board 带板、rehearsal 演练、sceneA/B/C 区分 S 形的场景）

每条录制单独一个目录 SESSIONS_DIR/时间_名称/，相机和机械臂数据都在其中的 save_data/ 里。开头从机械臂当前位姿直接移到录制起点。
record 需要先有 speed-test 的结果。
"""

# ================================ 可修改参数 ================================
# 开工核对的两条（right、roll_right）必须和开发录制是同一条路径：SPEED_MPS 到 ROT_DEG_S、GLASS_* 和 LIMIT_* 都是 fast2 的值，不要改。
# holdout1 的路径和速度在 holdout1_paths.py，窗位在 HOLDOUT_WINDOW。
PATH_SET = "fast2"           # 开工核对的路径组，写进 motion_log；真值的行程、转角系数和端点修正按它查（fr5_groundtruth）
SPEED_MPS = 0.5              # 平移速度 m/s
TRANSLATION_WIGGLE_SIDE = 1  # fast2 平移录制的同步摆动（除非 TRANSLATION_WIGGLE_AUTO）：+1 只往左摆
                             # （画面右边缘离右窗更远），-1 只往右，0 两边摆
TRANSLATION_WIGGLE_AUTO = True  # 平移录制的同步摆动往左还是往右由 plan 按行程选：两种各收一次两端，取长的
                                # （一样长取往左）。10-14：左右往右摆 758 mm、往左 509 mm；前后往左 516、往右 228
UP_DOWN_HEADING_BASE_DEG = 0.0  # 上下录制的相机朝向：基坐标系里从 +X 往 +Y 转的角度 °（0 = 朝 +X）；None 时取
                                # 两个标记朝向的平均。用户 10-14 定三条平移都朝 +X（左右、前后由标记连线定，本来就是 +X），
                                # 一块标定板三条都能用；上下的行程不受朝向影响（两端停在最低高度和标记余量）
ROT_DEG_S = 50.0             # 旋转角速度 °/s
# 画面里不能出现的玻璃：只管右侧落地窗（用户 10-13 定，左窗可以拍）。每块是 (方位角起, 方位角止, 仰角下, 仰角上)，单位 °，
# 从参考起点（start_pose.json）的光心看、相对它的水平朝向，方位角正 = 向左、仰角正 = 向上。10-13 全景图量出：右窗玻璃从柱子边缘 −60.3° 往右，
# 仰角约 −6.5～+4°（上面到天花板的白条也算进去）；−87° 以右没拍到，按全是玻璃处理。
GLASS_REGIONS = [
    (-87.0, -60.3, -8.0, 11.0),
    (-180.0, -87.0, -90.0, 90.0),
]
# holdout1 的玻璃窗检查。2026-11-09 用户定这批录制不考虑玻璃，regions 为空表示不检查；None = 没定过，plan 和 record 报问题。
# 要检查时的格式 {"viewpoint_mm": (x, y, z), "regions": [(方位角起, 方位角止, 仰角下, 仰角上), ...], "distance_m": 玻璃最近距离}：
# 从基坐标系的 viewpoint_mm 水平看，方位角零点是基坐标 +X，向左为正，仰角向上为正，单位 °
HOLDOUT_WINDOW = {"viewpoint_mm": (0.0, 0.0, 0.0), "regions": [], "distance_m": 7.0}
GLASS_MARGIN_DEG = 1.0       # 画面离玻璃至少这么多 °（画面左右边缘 1° 约 35 px）
WINDOW_DISTANCE_M = 7.0      # 玻璃离参考起点光心的最近距离 m（窗底仰角 −6.5°、相机离地至少约 0.8 m 估的下限）；光心移动时按它算视差
CAMERA_FPS = 30              # 相机帧率（15 / 30 / 60，传给 zed_record）
EXPOSURE_US = 2500           # 曝光 μs（传给 zed_record）
ANALOG_GAIN_MDB = 13000      # 模拟增益 mdB（传给 zed_record）：2.5 ms、13 dB 的亮度和 10 ms、1 dB 差不多（10-14 实测）
MAX_BLUR_PX = 2.0            # 允许的最大运动模糊（1080p 像素）：50 °/s、2.5 ms 时画面中心约 1.6 px
NEAREST_DEPTH_M = 1.0        # 画面里最近物体的距离 m，用来估算运动模糊
HOLD_START_S = 5.0           # 开头静止 s（同时用来估陀螺零偏）
HOLD_MID_S = 3.0             # 同步摆动和平移之间的静止 s
HOLD_END_S = 5.0             # 结尾静止 s
WIGGLE_DEG = 10.0            # 同步摆动幅度 °（绕相机竖直轴，横滚绕本条的转轴；单侧方向由 plan 选）
WIGGLE_DEG_S = 20.0          # 同步摆动角速度 °/s（真值脚本要求至少 10 °/s）
WIGGLE_CYCLES = 2            # 同步摆动来回次数
REPOSITION_MPS = 0.05        # 录制前从当前位姿移到录制起点的速度 m/s
WEBAPP_SPEED_PCT = 60        # WebApp 里的速度百分比（SDK 读不到，照抄过来）；WebApp 改了这里也要改，并重跑 speed-test。
                             # 60% 时 0.5 m/s 约需 SDK 83%
ACCEL_PCT = 100              # MoveL 加速度百分比（oacc）；实际加速度看 speed-test 结果，起停太猛就调小再重跑 speed-test
SPEED_TEST_PCT = 5           # speed-test 去程的速度百分比；没标定时最快为控制器默认速度 1000 mm/s 的 5% = 50 mm/s
MAX_SPEED_ERROR = 0.10       # 平移实测速度和预计速度相差超过 10% 就警告（speed-test 里也用来检查是否成正比）
MIN_CAMERA_Z_M = 0.18        # 相机光心最低高度（基坐标系 z，m，从底座安装面算）：只是兜底，录制的上下限由拖动标记定。
                             # 10-14 用户拖的 down 标记在 0.197 m，按用户要求从 0.35 改为 0.18
# 拖动标记（mark）：用户手动把机械臂拖到各方向能接受的极限位姿记下来，每对标记定一条录制（fr5_limits）。
# 录制两端先离标记收进这么多（到行程端点时同步摆动还要转 10°，法兰会横着摆出约 25 mm；高速时路径跟随也有误差），
# 再按逆解、关节速度、右窗、同步摆动往里收
LIMIT_MARGIN_MM = 30.0       # 平移：行程两端离标记至少这么多 mm（沿运动方向）
LIMIT_MARGIN_DEG = 3.0       # 旋转：转动范围两端离标记至少这么多 °
MIN_STROKE_M = 0.15          # 平移行程收完后至少这么长，不够就不录（拖得再开一点，或者看 plan 说两端为什么收）
MAX_STROKE_M = 1.0           # 平移行程上限
MIN_ROTATION_DEG = 15.0      # 旋转转角下限（不小于 WIGGLE_DEG：同步摆动在转动范围以内）
MAX_ROTATION_DEG = 180.0     # 旋转转角上限
# 允许的关节范围（°）：FR5 的关节范围，运行时再和控制器读到的关节软限位取交集（各收进 SOFT_LIMIT_MARGIN_DEG）。
# 不另设保守范围：路径不换构型、离奇异位形至少 SINGULAR_MARGIN_DEG；第一次演练时人盯腕部、线缆和机柜。
JOINT_MIN_DEG = [-175.0, -265.0, -160.0, -265.0, -175.0, -175.0]
JOINT_MAX_DEG = [175.0, 85.0, 160.0, 85.0, 175.0, 175.0]
SOFT_LIMIT_MARGIN_DEG = 2.0  # 离控制器关节软限位至少这么多 °
SINGULAR_MARGIN_DEG = 5.0    # 录制路径上 J3（手臂伸直）、J5（腕部两根轴共线）离 0 至少这么多 °
MAX_JOINT_STEP_DEG = 5.0     # 路径上每 10 mm / 2° 允许的最大关节变化（超过说明逆解可能跳支）
MAX_JOINT_SPEED_DEG_S = 150.0  # 录制段按预计速度走时任一关节的最快速度上限 °/s（FR5 额定 180 °/s）
MAX_SPEED_RIPPLE = 0.05      # 平移巡航段速度波动（标准差/均值）超过它就警告“速度不平稳”
LONG_LEG_MM = 200.0          # 多段路径：不短于这个行程的 MoveL 段才比较实测线速度和预计值（短段没有匀速段）
# 开发录制 6 条实际走过的关节范围 °（只提示超出的关节，上机时盯住线缆和腕部，不拒绝）
DEV_JOINT_MIN_DEG = [101.0, -118.0, 20.0, -253.0, -95.0, -80.0]
DEV_JOINT_MAX_DEG = [173.0, -22.0, 152.0, -96.0, -10.0, 84.0]
MIN_FREE_GB = 5.0            # 开始录制前 Box 至少剩余的磁盘空间 GB
ROBOT_IP = "192.168.58.2"
SESSIONS_DIR = "~/ROBOT/sessions_verify"
ZED_RECORD = "~/ROBOT/zed_imu/build/zed_record"
HANDEYE = "~/ROBOT/fr5_verify/handeye_s003_result.json"
START_POSE = "~/ROBOT/fr5_verify/start_pose.json"
SPEED_CAL = "~/ROBOT/fr5_verify/speed_calibration.json"
LIMIT_POSES = "~/ROBOT/fr5_verify/limit_poses.json"   # 开发录制时记下的拖动标记（只读）：定开工核对的路径和 speed-test 的路径

# 允许范围 [下限, 上限]，超出就拒绝运行
RANGES = {
    "SPEED_MPS": (0.02, 0.6),        # 上限还受 MAX_BLUR_PX 和关节速度约束
    "ROT_DEG_S": (1.0, 60.0),        # 上限还受 MAX_BLUR_PX 和关节速度约束
    "TRANSLATION_WIGGLE_SIDE": (-1, 1),
    "GLASS_MARGIN_DEG": (0.5, 5.0),
    "WINDOW_DISTANCE_M": (3.0, 50.0),
    "CAMERA_FPS": (15, 60),          # 只能是 15 / 30 / 60
    "EXPOSURE_US": (1000, 15000),
    "ANALOG_GAIN_MDB": (1000, 16000),
    "MAX_BLUR_PX": (0.3, 2.0),
    "NEAREST_DEPTH_M": (0.3, 20.0),
    "HOLD_START_S": (3.0, 30.0),
    "HOLD_MID_S": (2.0, 30.0),
    "HOLD_END_S": (3.0, 30.0),
    "WIGGLE_DEG": (5.0, 15.0),
    "WIGGLE_DEG_S": (12.0, 40.0),
    "WIGGLE_CYCLES": (1, 4),
    "REPOSITION_MPS": (0.01, 0.10),
    "WEBAPP_SPEED_PCT": (1, 100),
    "ACCEL_PCT": (1, 100),
    "SPEED_TEST_PCT": (1, 10),
    "MAX_SPEED_ERROR": (0.02, 0.3),
    "MIN_CAMERA_Z_M": (0.0, 1.5),
    "LIMIT_MARGIN_MM": (10.0, 100.0),
    "LIMIT_MARGIN_DEG": (1.0, 10.0),
    "MIN_STROKE_M": (0.05, 0.5),
    "MAX_STROKE_M": (0.2, 1.2),
    "MIN_ROTATION_DEG": (5.0, 60.0),
    "MAX_ROTATION_DEG": (20.0, 180.0),
    "MAX_JOINT_STEP_DEG": (1.0, 15.0),
    "SOFT_LIMIT_MARGIN_DEG": (0.0, 10.0),
    "SINGULAR_MARGIN_DEG": (2.0, 20.0),
    "MAX_JOINT_SPEED_DEG_S": (30.0, 170.0),
    "MAX_SPEED_RIPPLE": (0.01, 0.2),
    "LONG_LEG_MM": (50.0, 500.0),
    "MIN_FREE_GB": (1.0, 100.0),
}
# ==========================================================================

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import fr5_groundtruth as gt
import fr5_limits as fl
import fr5_motion as fm
import holdout1_paths as hp
from fr5_record import StateRecorder
from zed_capture import ZedRecord, camera_users

MB_PER_FRAME = 0.8    # zed_record 双目 JPEG 每帧约 0.8 MB
HANGUP_LOG = Path(tempfile.gettempdir()) / "run_session_hangup.log"  # 有录制目录时改成目录里的 run_session.log


class HangUp(BaseException):
    """终端断开（SIGHUP，例如 SSH 掉线）或被 SIGTERM：按 Ctrl+C 的方式收尾。"""


def _on_hangup(signum, frame):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # 收尾期间不再被打断
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    log = open(HANGUP_LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log                   # 终端已经没了，之后的输出写进日志
    print(f"{dt.datetime.now().isoformat(timespec='seconds')} 收到 {signal.Signals(signum).name}，开始收尾")
    raise HangUp(f"终端断开或被终止（{signal.Signals(signum).name}）")


def p(path: str) -> Path:
    return Path(path).expanduser()


def motion_config() -> fm.MotionConfig:
    return fm.MotionConfig(SPEED_MPS, HOLD_START_S, HOLD_MID_S, HOLD_END_S,
                           WIGGLE_DEG, WIGGLE_DEG_S, int(WIGGLE_CYCLES), REPOSITION_MPS,
                           rot_deg_s=ROT_DEG_S, translation_wiggle_side=int(TRANSLATION_WIGGLE_SIDE))


STARTUP_CHECK_NAMES = ["right", "roll_right"]   # 开工核对：fast2 路径上各录一条带板，看手眼和底座搬动后的精度
RECORD_NAMES = [*hp.PATHS, *STARTUP_CHECK_NAMES]


def pose_note(name: str, segs: list[fm.Segment], start_camera: np.ndarray, f_t_c: np.ndarray) -> tuple[str, list[str]]:
    """全程姿态范围（含同步摆动）和画面离玻璃窗的最近角距离，计划里写明；离得太近就是问题。
    光心移动时按 WINDOW_DISTANCE_M 算视差。返回 (说明, 问题列表)。"""
    span = fm.pose_span(segs, start_camera, f_t_c)
    clear, where = fm.glass_clearance(segs, start_camera, f_t_c, GLASS_REGIONS, distance_m=WINDOW_DISTANCE_M)
    parts = []
    for key, label in (("heading", "朝向"), ("pitch", "俯仰"), ("roll", "横滚")):
        lo, hi = span[key]
        if key == "heading" or hi - lo > 0.05:
            parts.append(f"{label} {lo:+.1f}～{hi:+.1f}°")
    note =(f"姿态（相对起点，含同步摆动；朝向正 = 向左，俯仰正 = 向上，横滚正 = 画面顶部向右）：{'，'.join(parts)}；"
            f"画面离玻璃窗最近 {clear:.2f}°（{where}）")
    if clear >= GLASS_MARGIN_DEG:
        return note, []
    return note, [f"画面离玻璃窗最近只有 {clear:.2f}°（{where} 段），少于 GLASS_MARGIN_DEG={GLASS_MARGIN_DEG}°"]


FULL_SPEED_SCALE = 0.95      # speed-test 的 --scale 不小于这个，测的加速度才用于 plan / record
CAL_KEYS = ("WEBAPP_SPEED_PCT", "ACCEL_PCT", "WIGGLE_DEG")  # 这些改了，speed-test 的结果就不能用了


def load_speed_model() -> tuple[fm.SpeedModel | None, str]:
    """读 speed-test 的结果，返回 (速度模型或 None, 说明)。"""
    path = p(SPEED_CAL)
    if not path.exists():
        return None, f"还没有速度标定（{path}），先运行 python3 run_session.py speed-test"
    cal = json.loads(path.read_text(encoding="utf-8"))
    changed = [f"{k} {cal['settings'].get(k)} → {globals()[k]}" for k in CAL_KEYS
               if cal["settings"].get(k) != globals()[k]]
    if changed:
        return None, "speed-test 之后这些参数改过：" + "，".join(changed) + "，请重跑 speed-test"
    m, acc = cal["model"], cal.get("accel") or {}
    # 加速度只用全速 speed-test 测的：控制器的加速度随速度变（10-14 实测 30 mm/s 时约 1.25 m/s²、100 mm/s 时 0.53 m/s²），
    # 降速测的不能外推到录制速度；不用时关节速度按全程匀速估（偏保守）
    full = cal.get("speed_scale", 1.0) >= FULL_SPEED_SCALE
    note = f"速度标定（{cal['created']}）：1% = 平移 {m['mm_s_per_pct']:.3f} mm/s，摆动 {m['wiggle_deg_s_per_pct']:.3f} °/s"
    if acc and not full:
        note += f"；加速度是降速（--scale {cal['speed_scale']}）测的，不用，关节速度按全程匀速估：全速跑一次 speed-test"
    return (fm.SpeedModel(m["mm_s_per_pct"], m["wiggle_deg_s_per_pct"], m["ramp_s"], calibrated=True,
                          accel_mm_s2=acc.get("test_down_mm_s2") if full else None,
                          accel_deg_s2=acc.get("test_yaw_back_deg_s2") if full else None), note)


def conservative_model(rpc) -> fm.SpeedModel:
    """没标定时的速度模型：1% = 控制器默认速度的 1%。WebApp 速度百分比如果也起作用，实际只会更慢。"""
    ret = rpc.GetDefaultTransVel()
    err, vel = ret if isinstance(ret, tuple) else (ret, None)
    if err != 0 or not vel:
        raise fm.MotionAborted(f"读控制器默认速度失败（错误码 {err}）")
    return fm.SpeedModel(float(vel) / 100.0)


def report_move(seg: fm.Segment) -> None:
    m = seg.log.get("measured") or {}
    parts = []
    if m.get("linear_mm_s"):
        parts.append(f"法兰线速度 {m['linear_mm_s']['mean']:.1f} mm/s")
    if m.get("angular_deg_s"):
        parts.append(f"角速度 {m['angular_deg_s']['mean']:.2f} °/s")
    if m.get("duration_s") is not None:
        parts.append(f"用时 {m['duration_s']:.1f} s")
    if seg.log.get("pivot_dev_max_mm") is not None:
        parts.append(f"光心最大偏离 {seg.log['pivot_dev_max_mm']:.2f} mm")
    gap = m.get("max_gap_s") or 0.0
    print(f"    {seg.name}：速度 {seg.percent:.2f}%，实测 " + ("，".join(parts) or "行程太短，没测速度")
          + (f"（期间状态中断 {gap:.1f} s，用时不准）" if gap > 0.1 else ""), flush=True)


def parameter_problems() -> list[str]:
    g = globals()
    out = [f"{k} = {g[k]} 超出允许范围 [{lo}, {hi}]" for k, (lo, hi) in RANGES.items() if not lo <= g[k] <= hi]
    if len(JOINT_MIN_DEG) != 6 or len(JOINT_MAX_DEG) != 6 or any(a >= b for a, b in zip(JOINT_MIN_DEG, JOINT_MAX_DEG)):
        out.append("JOINT_MIN_DEG / JOINT_MAX_DEG 必须各 6 个数，且下限小于上限")
    blur = fm.blur_px(SPEED_MPS, EXPOSURE_US, NEAREST_DEPTH_M)
    if blur > MAX_BLUR_PX:
        v_max = MAX_BLUR_PX * NEAREST_DEPTH_M / (fm.FX_1080 * EXPOSURE_US * 1e-6)
        out.append(f"速度 {SPEED_MPS} m/s、曝光 {EXPOSURE_US} μs 时 {NEAREST_DEPTH_M} m 处模糊 {blur:.2f} px，"
                   f"超过 MAX_BLUR_PX={MAX_BLUR_PX}；这个曝光下速度最高 {v_max:.3f} m/s")
    blur_rot = fm.rotation_blur_px(ROT_DEG_S, EXPOSURE_US)
    if blur_rot > MAX_BLUR_PX:
        w_max = MAX_BLUR_PX / blur_rot * ROT_DEG_S
        out.append(f"旋转 {ROT_DEG_S} °/s、曝光 {EXPOSURE_US} μs 时模糊 {blur_rot:.2f} px，"
                   f"超过 MAX_BLUR_PX={MAX_BLUR_PX}；这个曝光下角速度最高 {w_max:.1f} °/s")
    if any(len(r) != 4 or not (r[0] < r[1] and r[2] < r[3]) for r in GLASS_REGIONS):
        out.append("GLASS_REGIONS 每块必须是 (方位角起, 方位角止, 仰角下, 仰角上)，且起 < 止、下 < 上")
    out += window_format_problems()
    if CAMERA_FPS not in (15, 30, 60):
        out.append(f"CAMERA_FPS = {CAMERA_FPS}，只能是 15、30 或 60")
    if UP_DOWN_HEADING_BASE_DEG is not None and not (isinstance(UP_DOWN_HEADING_BASE_DEG, (int, float))
                                                     and -180.0 <= UP_DOWN_HEADING_BASE_DEG <= 180.0):
        out.append(f"UP_DOWN_HEADING_BASE_DEG = {UP_DOWN_HEADING_BASE_DEG}，只能是 None 或 −180～180 的数")
    if not isinstance(TRANSLATION_WIGGLE_AUTO, bool):
        out.append("TRANSLATION_WIGGLE_AUTO 只能是 True 或 False")
    if TRANSLATION_WIGGLE_SIDE not in (-1, 0, 1):
        out.append(f"TRANSLATION_WIGGLE_SIDE = {TRANSLATION_WIGGLE_SIDE}，只能是 -1、0 或 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", str(PATH_SET)):
        out.append("PATH_SET 只能用字母、数字、_ 和 -，最多 20 个字符")
    if MIN_STROKE_M >= MAX_STROKE_M:
        out.append(f"MIN_STROKE_M = {MIN_STROKE_M} 必须小于 MAX_STROKE_M = {MAX_STROKE_M}")
    if MIN_ROTATION_DEG >= MAX_ROTATION_DEG:
        out.append(f"MIN_ROTATION_DEG = {MIN_ROTATION_DEG} 必须小于 MAX_ROTATION_DEG = {MAX_ROTATION_DEG}")
    if MIN_ROTATION_DEG < WIGGLE_DEG:
        out.append(f"MIN_ROTATION_DEG = {MIN_ROTATION_DEG} 小于 WIGGLE_DEG = {WIGGLE_DEG}：旋转录制的同步摆动会超出转动范围")
    return out


def window_format_problems() -> list[str]:
    if HOLDOUT_WINDOW is None:
        return []
    w = HOLDOUT_WINDOW
    if not isinstance(w, dict) or set(w) != {"viewpoint_mm", "regions", "distance_m"} or len(w["viewpoint_mm"]) != 3:
        return ["HOLDOUT_WINDOW 必须是 None 或 {viewpoint_mm: (x, y, z), regions: [...], distance_m: 距离}"]
    if any(len(r) != 4 or not (r[0] < r[1] and r[2] < r[3]) for r in w["regions"]):
        return ["HOLDOUT_WINDOW 的 regions 每块必须是 (方位角起, 方位角止, 仰角下, 仰角上)，且起 < 止、下 < 上"]
    return []


def load_start() -> dict:
    path = p(START_POSE)
    if not path.exists():
        raise SystemExit(f"没有起点文件 {path}：从 ~/ROBOT/fr5_record 拷过来")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["camera"] = np.array(data["base_T_camera"])
    return data


def current_pose(rec: StateRecorder):
    _, pkg = rec.wait_state(-1, 1.0)
    return fm.fr5_to_matrix(list(pkg.flange_cur_pos)), list(pkg.jt_cur_pos), pkg


def check_segments(rec: StateRecorder, rpc, segs: list[fm.Segment], f_t_c, model, scale: float,
                   joint_limits: tuple[list, list] | None = None, waypoint: bool = False) -> list[str]:
    """从当前位姿出发做逆解检查，并按速度模型换算每段的速度百分比。返回问题列表。
    joint_limits：(下限, 上限)，默认 JOINT_MIN_DEG / JOINT_MAX_DEG。waypoint：多段路径，另查离奇异和构型。"""
    flange, joints, _ = current_pose(rec)
    lo, hi = joint_limits or (JOINT_MIN_DEG, JOINT_MAX_DEG)
    problems = (fm.check_plan(rpc, segs, flange, joints, f_t_c, lo, hi, MAX_JOINT_STEP_DEG, MIN_CAMERA_Z_M,
                              SINGULAR_MARGIN_DEG if waypoint else None)
                + fm.apply_speed(segs, model, scale))
    return problems + fm.joint_speed_problems(segs, MAX_JOINT_SPEED_DEG_S, model)


def joint_limits(rpc) -> tuple[list, list, str]:
    """允许的关节范围：JOINT_MIN/MAX_DEG 和控制器关节软限位（收进 SOFT_LIMIT_MARGIN_DEG）取交集。
    返回 (下限, 上限, 说明)；读不到软限位时只用 JOINT_MIN/MAX_DEG。"""
    lo, hi = [float(v) for v in JOINT_MIN_DEG], [float(v) for v in JOINT_MAX_DEG]
    try:
        err, values = rpc.GetJointSoftLimitDeg()
    except Exception as exc:  # noqa: BLE001 - 读不到就只用 JOINT_MIN/MAX_DEG
        err, values = f"{type(exc).__name__}: {exc}", None
    if err != 0 or not values or len(values) != 12:
        return lo, hi, f"没读到控制器的关节软限位（{err}），只按 JOINT_MIN/MAX_DEG"
    soft = [(float(values[2 * j]), float(values[2 * j + 1])) for j in range(6)]   # [J1 下, J1 上, J2 下, …]
    if not all(a < b for a, b in soft):
        return lo, hi, f"控制器关节软限位的格式和预期不同（{values}），只按 JOINT_MIN/MAX_DEG"
    lo = [max(a, s + SOFT_LIMIT_MARGIN_DEG) for a, (s, _) in zip(lo, soft)]
    hi = [min(b, e - SOFT_LIMIT_MARGIN_DEG) for b, (_, e) in zip(hi, soft)]
    return lo, hi, "控制器关节软限位：" + "，".join(f"J{j + 1} {s:+.0f}～{e:+.0f}" for j, (s, e) in enumerate(soft))


def config_problem(current_joints, start_q) -> str | None:
    """机械臂现在的构型和要去的位姿不同：直线移不过去（要经过奇异位形），要先手动拖过去。"""
    if start_q is None or fl.config_of(current_joints) == fl.config_of(start_q):
        return None
    return (f"goto_start: 机械臂现在是{fl.describe_config(current_joints)}，这条录制的起点是{fl.describe_config(start_q)}，"
            f"构型不同，直线移不过去（要经过奇异位形）；先手动拖到这条录制的起点附近（关节约 {np.round(start_q, 1).tolist()}）")


# ------------------------------------------------------------------ 拖动标记 → 路径
def load_marks(f_t_c: np.ndarray) -> dict:
    """拖动标记 {名字: limit_poses.json 里的记录 + "camera"（按记下的法兰位姿和现在的手眼外参算的相机位姿）}。"""
    path = p(LIMIT_POSES)
    if not path.exists():
        return {}
    marks = json.loads(path.read_text(encoding="utf-8"))["marks"]
    return {n: {**m, "camera": fm.fr5_to_matrix(m["flange"]) @ f_t_c} for n, m in marks.items()}


def pair_path(pair: str, marks: dict, start: dict) -> tuple[fl.MarkPath | None, str]:
    """一对标记定出的路径（不连机械臂）；缺标记时为 (None, 说明)。"""
    neg, pos = fl.PAIRS[pair]
    missing = [n for n in (neg, pos) if n not in marks]
    if missing:
        return None, f"缺拖动标记 {'、'.join(missing)}（python3 run_session.py mark {missing[0]}）"
    facing = (None if UP_DOWN_HEADING_BASE_DEG is None else
              np.array([np.cos(np.radians(UP_DOWN_HEADING_BASE_DEG)), np.sin(np.radians(UP_DOWN_HEADING_BASE_DEG)), 0.0]))
    return fl.derive_path(pair, marks[neg]["camera"], marks[pos]["camera"], start["camera"],
                          LIMIT_MARGIN_MM, LIMIT_MARGIN_DEG, up_down_facing=facing), ""


def fit_pair(rpc, pair: str, marks: dict, start: dict, f_t_c: np.ndarray, model: fm.SpeedModel | None,
             limits: tuple[list, list]):
    """按标记定路径，再用控制器逆解收两端（构型按标记，见 fl.fit_best）。返回 (路径或 None, Fit 或 None, 问题列表)。
    limits：允许的关节范围（joint_limits）。关节速度按满速算（不乘 --scale：降速演练和正式录制走同一条路径），
    有 speed-test 的加速度时按梯形速度曲线。"""
    path, missing = pair_path(pair, marks, start)
    if path is None:
        return None, None, [missing]
    translate = path.kind == "translate"
    accel = (model.accel_mm_s2 if translate else model.accel_deg_s2) if model else None
    lim = fl.FitLimits(
        joint_min=limits[0], joint_max=limits[1],
        max_joint_step_deg=MAX_JOINT_STEP_DEG, min_camera_z_m=MIN_CAMERA_Z_M,
        max_joint_speed_deg_s=MAX_JOINT_SPEED_DEG_S,
        speed=SPEED_MPS * 1000.0 if translate else ROT_DEG_S, accel=accel,
        min_length=MIN_STROKE_M * 1000.0 if translate else MIN_ROTATION_DEG,
        max_length=MAX_STROKE_M * 1000.0 if translate else MAX_ROTATION_DEG,
        glass=fm.GlassCheck(start["camera"], GLASS_REGIONS, distance_m=WINDOW_DISTANCE_M),
        glass_margin_deg=GLASS_MARGIN_DEG,
        wiggle_deg=WIGGLE_DEG if translate else 0.0, wiggle_side=int(TRANSLATION_WIGGLE_SIDE),
        singular_margin_deg=SINGULAR_MARGIN_DEG)
    neg, pos = fl.PAIRS[pair]
    sides = (+1, -1) if TRANSLATION_WIGGLE_AUTO else (int(TRANSLATION_WIGGLE_SIDE),)
    fit = fl.fit_wiggle_sides(rpc, path, f_t_c, marks[neg]["joints"], marks[pos]["joints"], lim, sides)
    return path, fit, list(fit.problems)


def build_recording(rpc, direction: str, marks: dict, start: dict, f_t_c: np.ndarray, model, limits):
    """录制 direction 的全部段。返回 (段或 None, 路径说明, 写进 motion_log 的路径记录, 起点关节角, 问题列表)。"""
    pair = fl.RECORD_PAIR[direction][0]
    path, fit, problems = fit_pair(rpc, pair, marks, start, f_t_c, model, limits)
    if path is None:
        return None, "", None, None, problems
    note = fl.describe_path(path, fit)
    if path.warnings:
        note += "\n" + "\n".join(f"  注意：{w}" for w in path.warnings)
    if problems:
        return None, note, None, None, problems
    cfg = motion_config()
    if fit.wiggle_side is not None:                      # 平移：按收两端时选的摆动方向
        cfg = dataclasses.replace(cfg, translation_wiggle_side=fit.wiggle_side)
    segs = fl.recording_segments(direction, path, fit, f_t_c, cfg)
    neg, pos = fl.PAIRS[pair]
    record = {"pair": pair, "kind": path.kind,
              "marks": {n: {k: v for k, v in marks[n].items() if k != "camera"} for n in (neg, pos)},
              "base_T_camera": path.base.tolist(), "axis": path.axis.tolist(), "heading_deg": path.heading_deg,
              "marks_s": list(path.marks_s), "margin_range": [path.lo, path.hi], "range": [fit.a, fit.b],
              "unit": path.unit, "ends": fit.ends, "fit_max_joint_speed_deg_s": fit.max_joint_speed_deg_s,
              "config_from": fit.seed, "joints_ends": [fit.joints_a, fit.joints_b], "notes": fit.notes,
              "wiggle_side": fit.wiggle_side,
              "joint_min": limits[0], "joint_max": limits[1], "warnings": path.warnings}
    return segs, note, record, fl.start_joints(direction, fit), []


# ------------------------------------------------------------------ 路径表 → 路径（holdout1）
def window_note(segs: list[fm.Segment], f_t_c: np.ndarray) -> tuple[str, list[str]]:
    """holdout1 路径的右窗检查。返回 (说明, 问题列表)。"""
    if HOLDOUT_WINDOW is None:
        return "", ["HOLDOUT_WINDOW 还没设：先量底座搬动后的右窗（看不到窗或已遮住时 regions 写空表）"]
    if not HOLDOUT_WINDOW["regions"]:
        return "右窗：HOLDOUT_WINDOW 的 regions 为空，不检查", []
    view = fm.level_facing_x(np.asarray(HOLDOUT_WINDOW["viewpoint_mm"], dtype=float) / 1000.0)
    clear, where = fm.glass_clearance(segs, view, f_t_c, HOLDOUT_WINDOW["regions"], distance_m=HOLDOUT_WINDOW["distance_m"])
    note = f"画面离右窗最近 {clear:.2f}°（{where}）"
    return note, ([] if clear >= GLASS_MARGIN_DEG else [f"{note}，少于 GLASS_MARGIN_DEG={GLASS_MARGIN_DEG}°"])


def waypoint_note(spec: hp.WaypointPath, side: int, segs: list[fm.Segment], f_t_c: np.ndarray, window: str) -> str:
    """路径说明：姿态范围、离奇异多远、关节最快、超出开发录制走过范围的关节、摆动方向和标定板放哪。"""
    level = fm.waypoint_pose(np.asarray(spec.reference_mm, dtype=float) / 1000.0, hp.Pose(), spec.frame_yaw_deg)
    span = fm.pose_span(segs, level, f_t_c)
    angles = "，".join(f"{label} {span[key][0]:+.1f}～{span[key][1]:+.1f}°"
                      for key, label in (("heading", "朝向"), ("pitch", "俯仰"), ("roll", "横滚")))
    ranges = [s.log["joint_range"] for s in segs if s.recorded and "joint_range" in s.log]
    lo, hi = np.min([r["min"] for r in ranges], axis=0), np.max([r["max"] for r in ranges], axis=0)
    min_abs = np.min([r["min_abs"] for r in ranges], axis=0)
    fastest = max((s.log.get("max_joint_speed_deg_s") or 0.0) for s in segs if s.kind == "move" and s.recorded)
    legs = [s for s in segs if s.name.startswith("leg_")]
    lines = [spec.description,
             f"参考位置（基坐标系）{np.round(level[:3, 3] * 1000.0, 1).tolist()} mm，{len(legs)} 段，同步摆动 {side:+d}；"
             f"{angles}（零点是水平朝路径前方，基坐标朝向 {spec.frame_yaw_deg:+.1f}°）",
             f"离奇异 |J3| ≥ {min_abs[2]:.1f}°、|J5| ≥ {min_abs[4]:.1f}°；关节最快 {fastest:.0f} °/s；{window}"]
    beyond = [f"J{j + 1} {lo[j]:.0f}～{hi[j]:.0f}"
              for j in range(6) if lo[j] < DEV_JOINT_MIN_DEG[j] or hi[j] > DEV_JOINT_MAX_DEG[j]]
    if beyond:
        lines.append("注意：这些关节超出开发录制走过的范围，第一次演练时盯住腕部和线缆：" + "、".join(beyond))
    if spec.board:
        lines.append(f"带板录制时标定板：{spec.board}")
    return "\n".join(lines)


def build_waypoint_recording(rpc, name: str, f_t_c: np.ndarray, model, limits):
    """路径表里的路径 name 的全部段。同步摆动往哪边由这里选（+1 优先），和机械臂当前位姿无关。不自动收缩：不通过就报问题，人工改路径表。
    返回 (段或 None, 路径说明, 写进 motion_log 的路径记录, 起点关节角, 问题列表)。"""
    spec = hp.PATHS[name]
    start_flange = (fm.waypoint_pose(np.asarray(spec.reference_mm, dtype=float) / 1000.0, spec.start, spec.frame_yaw_deg)
                    @ np.linalg.inv(f_t_c))
    err, start_q = fm.inverse_kin(rpc, start_flange, spec.seed_in_base)
    if start_q is None:
        return None, spec.description, None, None, [f"起点控制器逆解失败（错误码 {err}）"]
    if fl.config_of(start_q) != fl.config_of(spec.seed_in_base):
        return None, spec.description, None, None, [f"起点逆解的构型不对：{np.round(start_q, 1).tolist()}"]
    tried = []
    for side in (+1, -1):
        segs = fm.plan_waypoints(spec, f_t_c, motion_config(), side)
        found = (fm.check_plan(rpc, segs, start_flange, start_q, f_t_c, limits[0], limits[1], MAX_JOINT_STEP_DEG,
                               MIN_CAMERA_Z_M, SINGULAR_MARGIN_DEG)
                 + fm.apply_speed(segs, model, 1.0) + fm.joint_speed_problems(segs, MAX_JOINT_SPEED_DEG_S, model))
        window, glass = window_note(segs, f_t_c)
        found += glass
        if not found:
            record = {"path": name, "path_set": hp.PATH_SET, "description": spec.description,
                      "reference_mm": list(spec.reference_mm), "frame_yaw_deg": spec.frame_yaw_deg,
                      "seed_joints": list(spec.seed_in_base), "start_joints": start_q,
                      "wiggle_side": side, "wiggle_axis": spec.wiggle_axis, "correction_key": spec.correction_key,
                      "labels": spec.labels, "window": HOLDOUT_WINDOW, "joint_min": limits[0], "joint_max": limits[1]}
            return segs, waypoint_note(spec, side, segs, f_t_c, window), record, start_q, []
        tried += [f"同步摆动 {side:+d}：{m}" for m in found]
    return None, spec.description, None, start_q, tried


def leg_summary(seg: fm.Segment) -> dict:
    """多段路径一段的预计和实测，写进 motion_log；notes 是和预计差得多的项（13.11 上机清单第 10 项的判据）。"""
    m = seg.log.get("measured") or {}
    lin, ang = m.get("linear_mm_s"), m.get("angular_deg_s")
    notes = []
    if seg.motion == "L" and seg.distance_mm >= LONG_LEG_MM and lin and seg.expect_mm_s:
        error = lin["mean"] / seg.expect_mm_s - 1.0
        if abs(error) > MAX_SPEED_ERROR:
            notes.append(f"线速度比预计差 {error:+.0%}")
    if ang and seg.expect_deg_s and ang["max"] > 1.5 * seg.expect_deg_s:
        notes.append(f"角速度最大 {ang['max']:.1f} °/s，超过预计 {seg.expect_deg_s:.1f} °/s 的 1.5 倍")
    return {"name": seg.name, "motion": seg.motion, "percent": seg.percent, "distance_mm": seg.distance_mm,
            "angle_deg": seg.angle_deg, "expected_mm_s": seg.expect_mm_s, "expected_deg_s": seg.expect_deg_s,
            "planned_duration_s": seg.duration_s, "duration_s": m.get("duration_s"),
            "mean_mm_s": lin["mean"] if lin else None, "max_mm_s": lin["max"] if lin else None,
            "mean_deg_s": ang["mean"] if ang else None, "max_deg_s": ang["max"] if ang else None,
            "pivot_dev_max_mm": seg.log.get("pivot_dev_max_mm"), "overspeed": seg.log.get("overspeed"), "notes": notes}


def confirm_plan(text: str, assume_yes: bool) -> None:
    """列出计划，等操作员输入 yes；没确认就抛 MotionAborted（还没有运动）。"""
    print(text)
    if assume_yes:
        print("（--yes：跳过确认）")
        return
    try:
        ok = input("确认机械臂周围无人、急停在手边后，输入 yes 开始运动：").strip() == "yes"
    except EOFError:
        ok = False
    if not ok:
        raise fm.MotionAborted("没有确认，没有运动")


def describe(segments: list[fm.Segment], pending: dict | None = None, totals: bool = True) -> str:
    """pending：{段名: 说明}，这些段的速度要等前面的实测（speed-test 的回程）。totals：最后一行写录制时长和数据量。"""
    lines = []
    for s in segments:
        if s.kind == "hold":
            lines.append(f"  {s.name:<16} 静止 {s.duration_s:.1f} s")
            continue
        x, y, z = s.flange[:3]
        head = f"  {s.name:<16} {'圆弧' if s.motion == 'C' else '移动'} {s.distance_mm:5.1f} mm / {s.angle_deg:4.1f}°，"
        tail = f"，法兰目标 ({x:.1f}, {y:.1f}, {z:.1f}) mm" + ("（不录制）" if not s.recorded else "")
        if pending and s.name in pending:
            lines.append(head + pending[s.name] + tail)
            continue
        # 预计：按速度标定算的；设计：没有标定（plan）；最快：没标定时的保守上限
        word = "预计" if s.speed_known else ("设计" if s.percent is None else "最快")
        rates = "，".join(t for t in (f"{s.expect_mm_s:.1f} mm/s" if s.expect_mm_s else "",
                                      f"{s.expect_deg_s:.1f} °/s" if s.expect_deg_s and s.angle_deg > 0.05 else "")
                          if t)
        dur = ((f"约 {s.duration_s:.1f} s" if s.speed_known or s.percent is None else f"至少 {s.duration_s:.1f} s")
               if s.duration_s or s.speed_known else "时长待实测")
        lines.append(head + ("" if s.percent is None else f"速度 {s.percent:.2f}%，")
                     + (f"{word} {rates}，" if rates else "") + dur + tail)
    if totals:
        rec_s = sum(s.duration_s for s in segments if s.recorded)
        lines.append(f"  录制约 {rec_s:.0f} s，数据约 {rec_s * MB_PER_FRAME * CAMERA_FPS / 1000:.1f} GB")
    return "\n".join(lines)


# ------------------------------------------------------------------ 命令
def cmd_plan(args, problems: list[str]) -> int:
    for m in problems:
        print(f"参数问题：{m}")
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    names = args.names or list(hp.PATHS)
    unknown = [n for n in names if n not in RECORD_NAMES]
    if unknown:
        print(f"不认识的录制名：{unknown}（可选：{' '.join(RECORD_NAMES)}）")
        return 2
    marks = load_marks(f_t_c)
    print(f"路径组：路径表里的是 {hp.PATH_SET}，开工核对是 {PATH_SET}；相机 {CAMERA_FPS} fps，曝光 {EXPOSURE_US} μs，增益 {ANALOG_GAIN_MDB} mdB")
    model, note = load_speed_model()
    print(note if model else f"注意：{note}；下面没有速度百分比，时长按设计速度估算，关节速度按全程匀速（偏保守）")
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            rpc = rec.open_rpc()
            lo, hi, limits_note = joint_limits(rpc)
            print(f"关节范围：{limits_note}")
            _, current_q, _ = current_pose(rec)
            print(f"机械臂现在：关节 {np.round(current_q, 1).tolist()}（{fl.describe_config(current_q)}）")
            for name in names:
                waypoint = name in hp.PATHS
                if waypoint:
                    segs, path_note, _, start_q, found = build_waypoint_recording(rpc, name, f_t_c, model, (lo, hi))
                else:
                    segs, path_note, _, start_q, found = build_recording(rpc, name, marks, start, f_t_c, model, (lo, hi))
                note = ""
                if segs is not None:
                    note, glass = ("", []) if waypoint else pose_note(name, segs, start["camera"], f_t_c)
                    moved = config_problem(current_q, start_q)
                    found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale,
                                                                        (lo, hi), waypoint) + glass
                print(f"\n=== {name}：{'通过' if not found else '有问题'}")
                for text in (path_note, note):
                    if text:
                        print("  " + text.replace("\n", "\n  "))
                if segs is not None:
                    print(describe(segs))
                for m in found:
                    print(f"  问题：{m}")
                problems += found
        finally:
            rec.stop(write=False)
    print("\n空跑检查" + ("全部通过（机械臂没有动）" if not problems else f"发现 {len(problems)} 个问题（机械臂没有动）"))
    if any(m.startswith("goto_start") for m in problems):
        print("goto_start 的问题是从机械臂现在的位姿直接移过去走不通：先手动拖到那条录制的起点附近（同一种构型）再试")
    return 0 if not problems else 1


def translate_summary(seg: fm.Segment, scale: float) -> dict | None:
    """平移段的实测速度（运动中已算好）和预计、设计速度对照。"""
    m = (seg.log.get("measured") or {}).get("linear_mm_s")
    if not m or not seg.expect_mm_s:
        return None
    error = m["mean"] / seg.expect_mm_s - 1.0
    return {"design_mm_s": SPEED_MPS * 1000.0 * scale, "expected_mm_s": seg.expect_mm_s, "percent": seg.percent,
            "mean_mm_s": m["mean"], "min_mm_s": m["min"], "max_mm_s": m["max"], "ripple": m["ripple"],
            "smooth": m["ripple"] is not None and m["ripple"] <= MAX_SPEED_RIPPLE,
            "error": error, "speed_ok": abs(error) <= MAX_SPEED_ERROR}


def rotate_summary(seg: fm.Segment, scale: float) -> dict | None:
    """旋转段的实测角速度、转角、光心偏离，和预计、设计值对照。"""
    measured = seg.log.get("measured") or {}
    m = measured.get("angular_deg_s")
    if not m or not seg.expect_deg_s:
        return None
    error = m["mean"] / seg.expect_deg_s - 1.0
    return {"design_deg_s": ROT_DEG_S * scale, "expected_deg_s": seg.expect_deg_s, "percent": seg.percent,
            "mean_deg_s": m["mean"], "min_deg_s": m["min"], "max_deg_s": m["max"], "ripple": m["ripple"],
            "smooth": m["ripple"] is not None and m["ripple"] <= MAX_SPEED_RIPPLE,
            "error": error, "speed_ok": abs(error) <= MAX_SPEED_ERROR,
            "angle_deg": measured.get("angle_deg"), "planned_angle_deg": seg.angle_deg,
            "pivot_dev_max_mm": seg.log.get("pivot_dev_max_mm")}


def segment_log(s: fm.Segment) -> dict:
    """写进 motion_log.json 的一段。"""
    return {"name": s.name, "kind": s.kind, "recorded": s.recorded, "planned_duration_s": s.duration_s,
            "motion": s.motion, "flange": s.flange, "joints": s.joints,
            "via_flange": s.via_flange, "via_joints": s.via_joints,
            "arc_axis": s.arc_axis.tolist() if s.arc_axis is not None else None, "arc_deg": s.arc_deg,
            "pivot_m": s.pivot.tolist() if s.pivot is not None else None,
            "design_speed_mm_s": s.speed_mm_s, "design_rate_deg_s": s.rate_deg_s,
            "distance_mm": s.distance_mm, "angle_deg": s.angle_deg,
            "percent": s.percent, "expect_mm_s": s.expect_mm_s, "expect_deg_s": s.expect_deg_s, **s.log}


def zed_failure(info: dict | None, data_dir: Path) -> str | None:
    """zed_record 没有正常结束时返回原因（这时 img_timeline.csv、zed_imu.csv 会缺结尾，time_meta.txt 可能没写）。"""
    if not info or "returncode" not in info:      # 没有启动过
        return None
    problems = []
    rc = info["returncode"]
    if rc != 0:
        try:
            name = f"（{signal.Signals(-rc).name}）" if rc is not None and rc < 0 else ""
        except ValueError:
            name = ""
        problems.append(f"返回码 {rc}{name}")
    if info.get("exited_before_stop"):
        problems.append("没等到停止就自己退出了")
    if not (data_dir / "time_meta.txt").exists():
        problems.append("没有写 time_meta.txt")
    if not problems:
        return None
    return "zed_record 没有正常结束：" + "，".join(problems) + "。图像时间表和 IMU 文件可能缺结尾，这条不能作正式数据，要重录"


def cmd_motion(args, direction: str) -> int:
    busy = camera_users()
    if busy:
        print("相机正被占用，先关掉这些程序：\n  " + "\n  ".join(busy))
        return 1
    waypoint = direction in hp.PATHS
    group = hp.PATH_SET if waypoint else PATH_SET
    f_t_c, handeye = gt.load_handeye(p(HANDEYE))
    start = load_start()
    model, note = load_speed_model()
    if model is None:
        print(note)
        return 1
    root = p(SESSIONS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < MIN_FREE_GB:
        print(f"磁盘只剩 {free_gb:.1f} GB，少于 MIN_FREE_GB={MIN_FREE_GB}")
        return 1
    session_dir = root / (f"{dt.datetime.now():%Y%m%d_%H%M%S}_{direction}" + (f"_{args.tag}" if args.tag else ""))
    session_dir.mkdir()
    data_dir = session_dir / "save_data"
    global HANGUP_LOG
    HANGUP_LOG = session_dir / "run_session.log"

    rec = StateRecorder(data_dir, ROBOT_IP)
    rec.start()
    zed = ZedRecord(p(ZED_RECORD), session_dir, EXPOSURE_US, ANALOG_GAIN_MDB, CAMERA_FPS)
    error, ran, segs, mark_path, waypoint_path = None, False, [], None, None
    try:
        rpc = rec.open_rpc()
        runner = fm.MotionRunner(rec, rpc, ACCEL_PCT)
        runner.precheck()
        lo, hi, _ = joint_limits(rpc)
        limits = (lo, hi)
        if waypoint:
            built, path_note, waypoint_path, start_q, found = build_waypoint_recording(rpc, direction, f_t_c, model, limits)
        else:
            built, path_note, mark_path, start_q, found = build_recording(rpc, direction, load_marks(f_t_c), start,
                                                                          f_t_c, model, limits)
        if built is None:
            raise fm.MotionAborted("路径没定下来，没有运动：\n" + (path_note + "\n" if path_note else "")
                                   + "\n".join(f"  {m}" for m in found))
        segs = built
        pose, found = ("", []) if waypoint else pose_note(direction, segs, start["camera"], f_t_c)
        moved = config_problem(current_pose(rec)[1], start_q)
        found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale, limits, waypoint) + found
        if found:
            raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found)
                                   + ("\n（goto_start 走不通时，先手动拖到这条录制的起点附近再试）"
                                      if any(m.startswith("goto_start") for m in found) else ""))
        header = (f"{direction}：速度比例 {args.scale}，路径组 {group}"
                  f"，{CAMERA_FPS} fps，曝光 {EXPOSURE_US} μs，增益 {ANALOG_GAIN_MDB} mdB"
                  f"\n{note}；WebApp 速度百分比应为 {WEBAPP_SPEED_PCT}%"
                  + (f"\n{path_note}" if path_note else "")
                  + (f"\n{pose}" if pose else ""))
        confirm_plan(header + "\n" + describe(segs), args.yes)
        ran = True
        runner.run(segs[0])                      # 从当前位姿移到录制起点（相机还没开始录）
        report_move(segs[0])
        zed.start()
        zed.wait_grabbing(30.0)
        for seg in segs[1:]:
            print(f"[{seg.name}]", flush=True)
            runner.run(seg)
            if seg.kind == "move":
                report_move(seg)
    except BaseException as exc:  # noqa: BLE001 - 包括 Ctrl+C：先停机械臂，再收尾
        error = f"{type(exc).__name__}: {exc}"
        if ran:
            fm.emergency_stop(ROBOT_IP)
        print(f"\n中止：{error}")
    finally:
        if zed.proc is not None:
            zed.stop()
        rec.stop(write=ran)

    if not ran:
        if HANGUP_LOG.exists() and HANGUP_LOG.parent == session_dir:  # 断线日志留到 /tmp
            shutil.copy2(HANGUP_LOG, Path(tempfile.gettempdir()) / "run_session_hangup.log")
        shutil.rmtree(session_dir, ignore_errors=True)  # 没有运动，也没有录到任何东西
        return 1

    zed_error = zed_failure(zed.info, data_dir)
    parameters = {k: globals()[k] for k in [*RANGES, "JOINT_MIN_DEG", "JOINT_MAX_DEG", "GLASS_REGIONS", "PATH_SET"]}
    parameters["PATH_SET"] = group
    if waypoint:
        parameters["GLASS_REGIONS"] = (HOLDOUT_WINDOW or {}).get("regions")
        waypoint_path["legs"] = [leg_summary(s) for s in segs if s.name.startswith("leg_")]
    log = {
        "kind": "record", "direction": direction, "tag": args.tag, "speed_scale": args.scale,
        "error": "；".join(e for e in (error, zed_error) if e) or None,
        "parameters": parameters,
        "mark_path": mark_path,                  # fast2：这条录制的拖动标记、定出的路径和收两端的结果（fr5_limits）
        "waypoint_path": waypoint_path,          # holdout1：路径表、到达点名字、选的摆动方向和每段的预计与实测
        "speed_model": dataclasses.asdict(model) if model else None,
        "start_pose": {k: start[k] for k in ("base_T_camera", "flange", "joints", "created")},
        "handeye_sha256": handeye["sha256"],
        "zed_record": zed.info,
        "segments": [segment_log(s) for s in segs],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paths_sha256": hashlib.sha256(Path(hp.__file__).read_bytes()).hexdigest(),
    }
    translate = next((s for s in segs if s.name == "translate"), None)
    if translate is not None and error is None:
        log["translate_speed"] = translate_summary(translate, args.scale)
    rotate = next((s for s in segs if s.name == "rotate"), None)
    if rotate is not None and error is None:
        log["rotate_speed"] = rotate_summary(rotate, args.scale)
    (data_dir / "motion_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n数据目录：{session_dir}")
    if waypoint:
        for leg in waypoint_path["legs"]:
            for text in leg["notes"]:
                print(f"注意：{leg['name']} {text}")
    speed = log.get("translate_speed")
    if speed:
        print(f"平移实际速度 {speed['mean_mm_s']:.1f} mm/s（预计 {speed['expected_mm_s']:.1f}，设计 {speed['design_mm_s']:.1f}），"
              f"波动 {speed['ripple']:.1%}"
              + ("" if speed["smooth"] else f"，超过 MAX_SPEED_RIPPLE={MAX_SPEED_RIPPLE:.0%}，速度不平稳")
              + ("" if speed["speed_ok"] else f"；和预计差 {speed['error']:+.0%}，超过 MAX_SPEED_ERROR，重跑 speed-test"))
    rot = log.get("rotate_speed")
    if rot:
        print(f"旋转实际角速度 {rot['mean_deg_s']:.2f} °/s（预计 {rot['expected_deg_s']:.2f}，设计 {rot['design_deg_s']:.2f}），"
              f"波动 {rot['ripple']:.1%}，转角 {rot['angle_deg']:.2f}°（计划 {rot['planned_angle_deg']:.2f}°），"
              f"光心最大偏离 {rot['pivot_dev_max_mm']:.2f} mm"
              + ("" if rot["smooth"] else f"；波动超过 MAX_SPEED_RIPPLE={MAX_SPEED_RIPPLE:.0%}，速度不平稳")
              + ("" if rot["speed_ok"] else f"；和预计差 {rot['error']:+.0%}，超过 MAX_SPEED_ERROR"))
    if error is None and (data_dir / "zed_imu.csv").exists():
        try:
            report = gt.process(data_dir, p(HANDEYE), data_dir)
            lag, imgs = report["time_offset"], report["images"]
            seg = f"（前后段差 {lag['segment_diff_ms']:.2f} ms）" if "segment_diff_ms" in lag else ""
            print(f"真值：Δt {lag['dt_ms']:+.2f} ms{seg}，图像 {imgs['with_groundtruth']}/{imgs['total']} 张有真值")
            for w in report["warnings"]:
                print(f"  警告：{w}")
        except (OSError, ValueError, KeyError) as exc:
            print(f"真值计算失败：{exc}")
    if zed_error:
        print(f"\n录制失败：{zed_error}")
    return 0 if error is None and zed_error is None else 1


def cmd_warmup(args) -> int:
    """开录前让机械臂转起来：重复走路径表里的一条路径，不开相机、不存任何数据。Ctrl+C 随时停。"""
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    model, note = load_speed_model()
    if model is None:
        print(note)
        return 1
    error, ran, loops = None, False, 0
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            rpc = rec.open_rpc()
            runner = fm.MotionRunner(rec, rpc, ACCEL_PCT)
            runner.precheck()
            limits = joint_limits(rpc)[:2]
            segs, path_note, _, start_q, found = build_waypoint_recording(rpc, args.path, f_t_c, model, limits)
            if segs is None:
                raise fm.MotionAborted("路径没定下来，没有运动：\n  " + "\n  ".join(found))
            moved = config_problem(current_pose(rec)[1], start_q)
            found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale, limits, True)
            if found:
                raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found))
            confirm_plan(f"预热 {args.path}：{args.minutes:g} 分钟，速度比例 {args.scale}，不开相机、不存数据\n{note}\n{path_note}\n"
                         + describe(segs, totals=False), args.yes)
            ran = True
            runner.run(segs[0])
            end = time.monotonic() + args.minutes * 60.0
            while time.monotonic() < end:
                for seg in segs[1:]:
                    runner.run(seg)
                loops += 1
                print(f"第 {loops} 圈走完，还剩 {max(0.0, end - time.monotonic()) / 60.0:.1f} 分钟", flush=True)
        except BaseException as exc:  # noqa: BLE001 - 包括 Ctrl+C：先停机械臂
            error = f"{type(exc).__name__}: {exc}"
            if ran:
                fm.emergency_stop(ROBOT_IP)
            print(f"\n中止：{error}")
        finally:
            rec.stop(write=False)
    return 0 if error is None else 1


def _per_pct(seg: fm.Segment, key: str) -> float | None:
    """实测巡航速度 ÷ 速度百分比；没测到返回 None。key：linear_mm_s 或 angular_deg_s。"""
    m = (seg.log.get("measured") or {}).get(key)
    return m["mean"] / seg.percent if m and m["mean"] > 0 and seg.percent else None


def cmd_speed_test(args) -> int:
    """实测 MoveL 速度百分比和实际速度的换算，保存到 SPEED_CAL。不开相机，机械臂状态也不存盘。
    走上下那条录制路径的全程（down / up 两个拖动标记定的，高速时要有足够长的匀速段），在下端做摆动测试；
    同时由加减速用时估出加速度。路径按全程匀速收两端（还没有加速度），比之后 plan 收得多一点。"""
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    cfg = motion_config()
    marks = load_marks(f_t_c)
    segs, by, travel = [], {}, None
    error, ran, vmax = None, False, None
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            rpc = rec.open_rpc()
            runner = fm.MotionRunner(rec, rpc, ACCEL_PCT)
            runner.precheck()
            model = conservative_model(rpc)
            vmax = model.mm_s_per_pct * 100.0
            lo, hi, _ = joint_limits(rpc)
            path, fit, found = fit_pair(rpc, "up_down", marks, start, f_t_c, None, (lo, hi))
            if found:
                raise fm.MotionAborted("上下那条路径没定下来，没有运动：\n  " + "\n  ".join(found))
            travel = fit.length
            segs = fm.plan_speed_test(path.pose(fit.a), f_t_c, cfg, travel, SPEED_TEST_PCT)
            by = {s.name: s for s in segs}
            moved = config_problem(current_pose(rec)[1], fit.joints_a)
            found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale, (lo, hi))
            if found:
                raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found))
            pending = {"test_down": f"速度按 test_up 实测换算，目标 {SPEED_MPS * 1000 * args.scale:.1f} mm/s",
                       "test_yaw_back": f"速度按 test_yaw 实测换算，目标 {WIGGLE_DEG_S * args.scale:.1f} °/s"}
            confirm_plan(f"speed-test：速度比例 {args.scale}；控制器默认速度 {vmax:.0f} mm/s，"
                         f"WebApp 速度百分比应为 {WEBAPP_SPEED_PCT}%，加速度百分比 {ACCEL_PCT}%\n"
                         "只录机械臂状态，不开相机。“最快”是没标定时的上限，实际可能慢很多\n"
                         + fl.describe_path(path, fit) + "\n"
                         + describe(segs, pending, totals=False), args.yes)
            ran = True
            for seg in segs:
                if seg.name in pending:          # 回程：按去程实测换算到录制用的速度
                    if seg.name == "test_down":
                        k = _per_pct(by["test_up"], "linear_mm_s")
                        model = fm.SpeedModel(k, calibrated=True) if k else None
                    else:
                        k = _per_pct(by["test_yaw"], "angular_deg_s")
                        model = fm.SpeedModel(model.mm_s_per_pct, k, calibrated=True) if k else None
                    if model is None:
                        raise fm.MotionAborted(f"{seg.name}: 上一段没测到速度，没法换算")
                    found = fm.apply_speed([seg], model, args.scale)
                    if found and seg.percent > 100.0:   # WebApp 速度百分比太低：用 100% 测
                        print(f"    {seg.name}：目标速度需要 {seg.percent:.0f}% > 100%，改用 100%")
                        seg.fixed_percent = 100.0
                        found = fm.apply_speed([seg], model, args.scale)
                    if found:
                        raise fm.MotionAborted("；".join(found))
                print(f"[{seg.name}]", flush=True)
                runner.run(seg)
                report_move(seg)
        except BaseException as exc:  # noqa: BLE001 - 包括 Ctrl+C：先停机械臂
            error = f"{type(exc).__name__}: {exc}"
            if ran:
                fm.emergency_stop(ROBOT_IP)
            print(f"\n中止：{error}")
        finally:
            rec.stop(write=False)
    if error is not None:
        return 1

    kt_a, kt_b = _per_pct(by["test_up"], "linear_mm_s"), _per_pct(by["test_down"], "linear_mm_s")
    kw_a, kw_b = _per_pct(by["test_yaw"], "angular_deg_s"), _per_pct(by["test_yaw_back"], "angular_deg_s")
    kt, kw = kt_b or kt_a, kw_b or kw_a       # 优先用接近录制速度的回程
    if not kt or not kw:
        print("没测到平移或摆动的速度，没有保存标定")
        return 1
    ramps = [m["ramp_s"] for m in ((by[n].log.get("measured") or {}) for n in ("test_down", "test_yaw_back"))
             if m.get("ramp_s") is not None and (m.get("max_gap_s") or 0.0) <= 0.1]   # 中断过的段用时不准
    # 梯形速度曲线：总用时 = 路程 / 速度 + 速度 / 加速度，多出来的 ramp_s 就是 v / a
    accel = {}
    for n, key, unit in (("test_down", "linear_mm_s", "mm_s2"), ("test_yaw_back", "angular_deg_s", "deg_s2")):
        m = by[n].log.get("measured") or {}
        if m.get(key) and m.get("ramp_s") and (m.get("max_gap_s") or 0.0) <= 0.1:
            accel[f"{n}_{unit}"] = m[key]["mean"] / m["ramp_s"]
    checks = {"translate_ratio_minus_1": kt_b / kt_a - 1.0 if kt_a and kt_b else None,
              "wiggle_ratio_minus_1": kw_b / kw_a - 1.0 if kw_a and kw_b else None,
              "webapp_factor": kt / (vmax / 100.0)}
    cal = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "settings": {k: globals()[k] for k in CAL_KEYS},
        "default_trans_vel_mm_s": vmax,
        "speed_scale": args.scale,
        "model": {"mm_s_per_pct": kt, "wiggle_deg_s_per_pct": kw,
                  "ramp_s": max(ramps) if ramps else fm.DEFAULT_RAMP_S},
        "checks": checks,
        "accel": accel,
        "tests": [{"name": s.name, "percent": s.percent, "distance_mm": s.distance_mm, "angle_deg": s.angle_deg,
                   "measured": s.log.get("measured")} for s in segs],
    }
    p(SPEED_CAL).write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {p(SPEED_CAL)}")
    print(f"  1% = 平移 {kt:.3f} mm/s，摆动 {kw:.3f} °/s；加减速约 {cal['model']['ramp_s']:.2f} s")
    print(f"  相当于默认速度 {vmax:.0f} mm/s × {checks['webapp_factor']:.3f}（WebApp 速度百分比 {WEBAPP_SPEED_PCT}%）")
    for key, label in (("translate_ratio_minus_1", "平移"), ("wiggle_ratio_minus_1", "摆动")):
        r = checks[key]
        if r is not None and abs(r) > MAX_SPEED_ERROR:
            print(f"  警告：{label}去程和回程的“速度 / 百分比”相差 {r:+.0%}，速度和百分比不成正比，"
                  "录制时以实测速度为准")
    a_lin = accel.get("test_down_mm_s2")
    if a_lin and args.scale < FULL_SPEED_SCALE:
        print(f"  平移加速度约 {a_lin / 1000:.2f} m/s²（{by['test_down'].log['measured']['linear_mm_s']['mean']:.0f} mm/s 时）："
              "这是降速测的，控制器的加速度随速度变，不能推到录制速度，plan / record 不用它；全速跑一次 speed-test")
    elif a_lin:
        print(f"  平移加速度约 {a_lin / 1000:.2f} m/s²（{by['test_down'].log['measured']['linear_mm_s']['mean']:.0f} mm/s 时）；"
              f"{SPEED_MPS} m/s 时加减速共走约 {(SPEED_MPS * 1000.0) ** 2 / a_lin:.0f} mm，"
              f"上下这条 {travel:.0f} mm 的匀速段约 {travel - (SPEED_MPS * 1000.0) ** 2 / a_lin:.0f} mm")
        print("  有了加速度，关节速度按梯形速度曲线算，plan 收两端会比现在少：重跑 plan 看每条录制的最终路径")
    if accel.get("test_yaw_back_deg_s2"):
        print(f"  摆动角加速度约 {accel['test_yaw_back_deg_s2']:.0f} °/s²")
    for label, pct in (("平移", SPEED_MPS * 1000.0 / kt), ("移到起点", REPOSITION_MPS * 1000.0 / kt),
                       ("同步摆动", WIGGLE_DEG_S / kw), ("旋转", ROT_DEG_S / kw)):
        print(f"  {label}需要速度 {pct:.2f}%" + ("：超过 100%，调高 WebApp 速度百分比后重跑" if pct > 100 else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("names", nargs="*", help="要检查的录制，默认全部 holdout1 路径；可选 " + " ".join(RECORD_NAMES))
    sub.add_parser("speed-test")
    record = sub.add_parser("record")
    record.add_argument("direction", choices=RECORD_NAMES, metavar="名称", help=" ".join(RECORD_NAMES))
    warmup = sub.add_parser("warmup")
    warmup.add_argument("path", choices=list(hp.PATHS), metavar="路径", help=" ".join(hp.PATHS))
    warmup.add_argument("--minutes", type=float, default=30.0, help="预热时长（分钟，1–120），默认 30")
    for sp in sub.choices.values():
        sp.add_argument("--scale", type=float, default=1.0, help="速度比例 (0, 1]，首次运行建议 0.2")
        sp.add_argument("--yes", action="store_true", help="跳过确认")
        sp.add_argument("--tag", default="", help="加在录制目录名后面的标签：board、rehearsal、sceneA/B/C")
    args = parser.parse_args(argv)
    if not 0 < args.scale <= 1:
        parser.error("--scale 必须在 (0, 1] 之间")
    if args.command == "warmup" and not 1.0 <= args.minutes <= 120.0:
        parser.error("--minutes 必须在 1–120 之间")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9-]{1,20}", args.tag):
        parser.error("--tag 只能用字母、数字和 -，最多 20 个字符")
    signal.signal(signal.SIGHUP, _on_hangup)
    signal.signal(signal.SIGTERM, _on_hangup)
    problems = parameter_problems()
    if args.command == "plan":
        return cmd_plan(args, problems)
    if problems:
        print("参数问题，没有运动：\n  " + "\n  ".join(problems))
        return 1
    if args.command == "speed-test":
        return cmd_speed_test(args)
    if args.command == "warmup":
        return cmd_warmup(args)
    return cmd_motion(args, args.direction)


def main_synced(argv=None) -> int:
    """main() 之后把文件系统缓冲写到盘上（os.sync）：10-14 俯仰演练录完约 1 分钟 Box 断电，最后写的 motion_log、
    机械臂状态、真值、相机日志都成了空文件（图像早写完、还在）。录完马上落盘，断电也不丢。"""
    try:
        return main(argv)
    finally:
        os.sync()


if __name__ == "__main__":
    sys.exit(main_synced())
