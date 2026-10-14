#!/usr/bin/env python3
"""一条录制的总控：FR5 运动（fr5_motion）+ 机械臂状态（fr5_record）+ 图像和 IMU（zed_capture）+ 真值（fr5_groundtruth）。
所有可改参数都在本文件开头，其他模块不放可调参数。

在 Box 自己的终端里运行（人在机械臂旁边，手边有急停）：
    python3 run_session.py set-start        读当前位姿，算出相机水平的参考起点并保存（不动机械臂；10-10 已做，一般不用再做）
    python3 run_session.py mark up          手动拖到某个方向的极限位姿后，记下当前位姿（不动机械臂）；
                                            平移 forward/backward/left/right/up/down，旋转 yaw_left … roll_right
    python3 run_session.py limits           按拖动标记列出每条录制的路径（不连机械臂）
    python3 run_session.py plan             空跑：按标记定路径、用控制器逆解收两端，检查 6 条录制和 sync-test（不动机械臂）
    python3 run_session.py goto-start       低速移到参考起点（会先确认；sync-test、movec-test 在这里做）
    python3 run_session.py speed-test       速度标定：在上下那条路径上实测速度百分比对应的 mm/s 和 °/s（会先确认，不开相机）
    python3 run_session.py sync-test        时间同步验证：只做同步摆动的录制（会先确认）
    python3 run_session.py movec-test       MoveC 探测：在参考起点用 MoveC 和 MoveL 各绕光心竖直轴转 ±10°，对照光心偏离
    python3 run_session.py record up        正式录制一条（会先确认）。每对标记录一条，共 6 条：
                                            平移 up（最下到最上）、right（最左到最右）、forward（最后到最前）；
                                            旋转 yaw_left（偏航，右到左）、pitch_up（俯仰，下到上）、roll_right（横滚，左到右）。
                                            反方向（down、left、backward、yaw_right、pitch_down、roll_left）也能录
选项：--scale 0.25  按比例降速（路径不变）；--yes  跳过确认（只在你已经看过计划时用）；
      --tag board  加在录制目录名后面（区分验证录制和正式录制）

每条录制单独一个目录 SESSIONS_DIR/时间_名称/，相机和机械臂数据都在其中的 save_data/ 里。
录制的路径由这一对拖动标记定（fr5_limits）：标记只是大概的极限，两端先收进 LIMIT_MARGIN_MM / LIMIT_MARGIN_DEG，
再按逆解、关节速度、右窗和同步摆动往里收。不回参考起点：开头从机械臂当前位姿直接移到录制起点。
sync-test 和 record 需要先有 speed-test 的结果；改了 WebApp 速度百分比要重跑 speed-test。
"""

# ================================ 可修改参数 ================================
# 高速录制（high-speed 分支，2026-10-14 起）：平移 0.5 m/s、旋转 50 °/s、30 fps、曝光 2.5 ms，只避开右侧落地窗。
# 10-12 平移、10-13 旋转的低速录制用的是 git 标签 data-v1 的设置（路径组 "base"）。
PATH_SET = "fast2"           # 这一套路径几何的名字，写进 motion_log；真值的行程、转角系数按它查（fr5_groundtruth）。
                             # fast2 = 10-14 起按拖动标记定的路径；重新拖动标记、路径变了就换个名字，免得用错系数
SPEED_MPS = 0.5              # 平移速度 m/s
TRANSLATION_WIGGLE_SIDE = 1  # 平移录制和 sync-test 的同步摆动：+1 只往左摆（画面右边缘离右窗更远），0 两边摆
ROT_DEG_S = 50.0             # 旋转角速度 °/s
# 画面里不能出现的玻璃：只管右侧落地窗（用户 10-13 定，左窗可以拍）。每块是 (方位角起, 方位角止, 仰角下, 仰角上)，单位 °，
# 从参考起点（set-start）的光心看、相对它的水平朝向，方位角正 = 向左、仰角正 = 向上。10-13 全景图量出：右窗玻璃从柱子边缘 −60.3° 往右，
# 仰角约 −6.5～+4°（上面到天花板的白条也算进去）；−87° 以右没拍到，按全是玻璃处理。
GLASS_REGIONS = [
    (-87.0, -60.3, -8.0, 11.0),
    (-180.0, -87.0, -90.0, 90.0),
]
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
WIGGLE_DEG = 10.0            # 同步摆动幅度 °（平移录制和 sync-test 绕相机竖直轴摆；旋转录制绕本条的转轴往内侧摆）
WIGGLE_DEG_S = 20.0          # 同步摆动角速度 °/s（真值脚本要求至少 10 °/s）
WIGGLE_CYCLES = 2            # 同步摆动来回次数
SYNC_TEST_HOLD_S = 10.0      # sync-test 两次摆动之间的静止 s
REPOSITION_MPS = 0.05        # 录制前从当前位姿移到录制起点的速度 m/s
WEBAPP_SPEED_PCT = 60        # WebApp 里的速度百分比（SDK 读不到，照抄过来）；WebApp 改了这里也要改，并重跑 speed-test。
                             # 高速录制用 60%（10-14 用户已在 WebApp 改好；0.5 m/s 约需 SDK 83%）
ACCEL_PCT = 100              # MoveL 加速度百分比（oacc）；实际加速度看 speed-test 结果，起停太猛就调小再重跑 speed-test
SPEED_TEST_PCT = 5           # speed-test 去程的速度百分比；没标定时最快为控制器默认速度 1000 mm/s 的 5% = 50 mm/s
MAX_SPEED_ERROR = 0.10       # 平移实测速度和预计速度相差超过 10% 就警告（speed-test 里也用来检查是否成正比）
MIN_CAMERA_Z_M = 0.35        # 相机光心最低高度（基坐标系 z，m）；按现场桌面和障碍物修改
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
# 高速录制不再另设保守范围（用户 10-14 定）：腕部 J4、J5、线缆（J6）、机柜等由拖动标记保证，拖的时候人会确认；
# 路径只在两个标记之间，并且不换构型、离奇异位形至少 SINGULAR_MARGIN_DEG。
# 低速录制（data-v1）用的是保守范围 [80, −30, −130, −154, −72, −50]～[160, 34, −15, −87, −19, 50]。
JOINT_MIN_DEG = [-175.0, -265.0, -160.0, -265.0, -175.0, -175.0]
JOINT_MAX_DEG = [175.0, 85.0, 160.0, 85.0, 175.0, 175.0]
SOFT_LIMIT_MARGIN_DEG = 2.0  # 离控制器关节软限位至少这么多 °
SINGULAR_MARGIN_DEG = 5.0    # 录制路径上 J3（手臂伸直）、J5（腕部两根轴共线）离 0 至少这么多 °
MAX_JOINT_STEP_DEG = 5.0     # 路径上每 10 mm / 2° 允许的最大关节变化（超过说明逆解可能跳支）
MAX_JOINT_SPEED_DEG_S = 150.0  # 录制段按预计速度走时任一关节的最快速度上限 °/s（FR5 额定 180 °/s）
MAX_SPEED_RIPPLE = 0.05      # 平移巡航段速度波动（标准差/均值）超过它就警告“速度不平稳”
MIN_FREE_GB = 5.0            # 开始录制前 Box 至少剩余的磁盘空间 GB
ROBOT_IP = "192.168.58.2"
SESSIONS_DIR = "~/ROBOT/sessions"
ZED_RECORD = "~/ROBOT/zed_imu/build/zed_record"
HANDEYE = "~/ROBOT/fr5_record/handeye_s003_result.json"
START_POSE = "~/ROBOT/fr5_record/start_pose.json"
SPEED_CAL = "~/ROBOT/fr5_record/speed_calibration.json"
MOVEC_TEST = "~/ROBOT/fr5_record/movec_test.json"
LIMIT_POSES = "~/ROBOT/fr5_record/limit_poses.json"   # mark 记下的拖动标记

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
    "SYNC_TEST_HOLD_S": (5.0, 60.0),
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
    "MIN_FREE_GB": (1.0, 100.0),
}
# ==========================================================================

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import fr5_groundtruth as gt
import fr5_limits as fl
import fr5_motion as fm
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
                           WIGGLE_DEG, WIGGLE_DEG_S, int(WIGGLE_CYCLES), REPOSITION_MPS, SYNC_TEST_HOLD_S,
                           rot_deg_s=ROT_DEG_S, translation_wiggle_side=int(TRANSLATION_WIGGLE_SIDE))


RECORD_NAMES = list(fl.RECORD_PAIR)          # 12 个方向都能录；每对标记录一条，默认的 6 条见 fl.RECORDINGS


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
    return (fm.SpeedModel(m["mm_s_per_pct"], m["wiggle_deg_s_per_pct"], m["ramp_s"], calibrated=True,
                          accel_mm_s2=acc.get("test_down_mm_s2"), accel_deg_s2=acc.get("test_yaw_back_deg_s2")),
            f"速度标定（{cal['created']}）：1% = 平移 {m['mm_s_per_pct']:.3f} mm/s，摆动 {m['wiggle_deg_s_per_pct']:.3f} °/s")


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
    if CAMERA_FPS not in (15, 30, 60):
        out.append(f"CAMERA_FPS = {CAMERA_FPS}，只能是 15、30 或 60")
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


def load_start() -> dict:
    path = p(START_POSE)
    if not path.exists():
        raise SystemExit(f"没有起点文件 {path}：先运行 python3 run_session.py set-start")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["camera"] = np.array(data["base_T_camera"])
    return data


def current_pose(rec: StateRecorder):
    _, pkg = rec.wait_state(-1, 1.0)
    return fm.fr5_to_matrix(list(pkg.flange_cur_pos)), list(pkg.jt_cur_pos), pkg


def check_segments(rec: StateRecorder, rpc, segs: list[fm.Segment], f_t_c, model, scale: float,
                   joint_limits: tuple[list, list] | None = None) -> list[str]:
    """从当前位姿出发做逆解检查，并按速度模型换算每段的速度百分比。返回问题列表。
    joint_limits：(下限, 上限)，默认 JOINT_MIN_DEG / JOINT_MAX_DEG（录制时用这对标记放宽过的范围）。"""
    flange, joints, _ = current_pose(rec)
    lo, hi = joint_limits or (JOINT_MIN_DEG, JOINT_MAX_DEG)
    problems = (fm.check_plan(rpc, segs, flange, joints, f_t_c, lo, hi, MAX_JOINT_STEP_DEG, MIN_CAMERA_Z_M)
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


def config_problem(current_joints, start_q, target: str = "这条录制的起点") -> str | None:
    """机械臂现在的构型和要去的位姿不同：直线移不过去（要经过奇异位形），要先手动拖过去。
    参考起点（goto-start、sync-test、movec-test）按 set-start 时的关节角查：换一种构型到那里，机械臂的姿势没确认过。"""
    if start_q is None or fl.config_of(current_joints) == fl.config_of(start_q):
        return None
    return (f"goto_start: 机械臂现在是{fl.describe_config(current_joints)}，{target}是{fl.describe_config(start_q)}，"
            f"构型不同，直线移不过去（要经过奇异位形）；先手动拖到{target}附近（关节约 {np.round(start_q, 1).tolist()}）")


REFERENCE = "参考起点（set-start 时的构型）"


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
    return fl.derive_path(pair, marks[neg]["camera"], marks[pos]["camera"], start["camera"],
                          LIMIT_MARGIN_MM, LIMIT_MARGIN_DEG), ""


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
    fit = fl.fit_best(rpc, path, f_t_c, marks[neg]["joints"], marks[pos]["joints"], lim)
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
    segs = fl.recording_segments(direction, path, fit, f_t_c, motion_config())
    neg, pos = fl.PAIRS[pair]
    record = {"pair": pair, "kind": path.kind,
              "marks": {n: {k: v for k, v in marks[n].items() if k != "camera"} for n in (neg, pos)},
              "base_T_camera": path.base.tolist(), "axis": path.axis.tolist(), "heading_deg": path.heading_deg,
              "marks_s": list(path.marks_s), "margin_range": [path.lo, path.hi], "range": [fit.a, fit.b],
              "unit": path.unit, "ends": fit.ends, "fit_max_joint_speed_deg_s": fit.max_joint_speed_deg_s,
              "config_from": fit.seed, "joints_ends": [fit.joints_a, fit.joints_b], "notes": fit.notes,
              "joint_min": limits[0], "joint_max": limits[1], "warnings": path.warnings}
    return segs, note, record, fl.start_joints(direction, fit), []


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
def cmd_set_start(args) -> int:
    f_t_c, handeye = gt.load_handeye(p(HANDEYE))
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            rpc = rec.open_rpc()
            flange, joints, _ = current_pose(rec)
            camera = flange @ f_t_c
            level = fm.level_camera(camera)
            target = level @ np.linalg.inv(f_t_c)
            err, q = rpc.GetInverseKinRef(0, fm.matrix_to_fr5(target), joints)
            if err != 0 or q is None:
                print(f"水平起点控制器逆解失败（错误码 {err}）")
                return 1
        finally:
            rec.stop(write=False)
    z = camera[:3, 2]
    tilt, roll = np.degrees(np.arcsin(-z[2])), np.degrees(np.arcsin(camera[2, 0]))
    data = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "base_T_camera": level.tolist(),
        "flange": fm.matrix_to_fr5(target),
        "joints": list(map(float, q)),
        "from_flange": fm.matrix_to_fr5(flange),
        "from_joints": joints,
        "level_correction_deg": {"tilt": float(tilt), "roll": float(roll)},
        "handeye_sha256": handeye["sha256"],
    }
    p(START_POSE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存起点 {p(START_POSE)}")
    print(f"  相机光心 {np.round(level[:3, 3] * 1000, 1).tolist()} mm；当前相机俯仰 {tilt:+.2f}°、横滚 {roll:+.2f}°，起点改为水平")
    print(f"  起点关节 {np.round(q, 2).tolist()}（相对当前 {np.round(np.subtract(q, joints), 2).tolist()}）")
    if max(abs(tilt), abs(roll)) > 5:
        print("  注意：当前相机偏离水平超过 5°，移到起点时姿态变化较大，先用 --scale 0.25 低速执行")
    return 0


def describe_mark(rel: dict) -> str:
    fwd, left, up = (fl.signed(v, 0) for v in rel["offset_mm"])
    head, pitch, roll = (fl.signed(v) for v in rel["angles_deg"])
    return f"光心相对参考起点 前 {fwd}、左 {left}、上 {up} mm，姿态 方位角 {head}°、俯仰 {pitch}°、横滚 {roll}°"


def mark_pair(name: str) -> str:
    return next(pair for pair, ends in fl.PAIRS.items() if name in ends)


def print_pair(pair: str, marks: dict, start: dict, indent: str = "  ") -> None:
    """一对标记定出的路径（还没按逆解收两端）、提醒和问题。"""
    path, missing = pair_path(pair, marks, start)
    if path is None:
        print(f"{indent}{fl.PAIR_LABEL[pair]}：{missing}")
        return
    print(indent + fl.describe_path(path).replace("\n", "\n" + indent))
    neg, pos = fl.PAIRS[pair]
    if fl.config_of(marks[neg]["joints"]) != fl.config_of(marks[pos]["joints"]):
        print(f"{indent}  注意：两个标记构型不同（{neg}：{fl.describe_config(marks[neg]['joints'])}；"
              f"{pos}：{fl.describe_config(marks[pos]['joints'])}）：一条录制换不了构型，plan 两种构型各试一次，取能走得长的")
    for w in path.warnings:
        print(f"{indent}  注意：{w}")
    for m in path.problems:
        print(f"{indent}  问题：{m}")


def cmd_mark(args) -> int:
    """读当前位姿，记成 args.direction 方向的拖动极限，存进 LIMIT_POSES（不动机械臂）。"""
    f_t_c, handeye = gt.load_handeye(p(HANDEYE))
    start = load_start()
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            first, _, _ = current_pose(rec)
            time.sleep(0.5)
            flange, joints, pkg = current_pose(rec)
        finally:
            rec.stop(write=False)
    moved, turned = np.linalg.norm(flange[:3, 3] - first[:3, 3]) * 1000.0, fm.rotation_deg(first, flange)
    if moved > 1.0 or turned > 0.2:
        print(f"0.5 s 内法兰动了 {moved:.1f} mm、{turned:.2f}°：松手，等机械臂停稳后再记")
        return 1
    camera = flange @ f_t_c
    path = p(LIMIT_POSES)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"marks": {}}
    old = data["marks"].get(args.direction)
    data["marks"][args.direction] = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "flange": fm.matrix_to_fr5(flange),
        "joints": list(map(float, joints)),
        "base_T_camera": camera.tolist(),
        "handeye_sha256": handeye["sha256"],
        "main_code": int(pkg.main_code), "sub_code": int(pkg.sub_code),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已记下 {args.direction}（{path}）")
    print(f"  {describe_mark(fl.relative(start['camera'], camera))}")
    print(f"  关节 {np.round(joints, 1).tolist()}（{fl.describe_config(joints)}）")
    if old is not None:
        before = fm.fr5_to_matrix(old["flange"]) @ f_t_c
        print(f"  替换了 {old['created']} 的标记（光心差 {np.linalg.norm(before[:3, 3] - camera[:3, 3]) * 1000:.0f} mm、"
              f"姿态差 {fm.rotation_deg(before, camera):.1f}°）")
    print_pair(mark_pair(args.direction), load_marks(f_t_c), start)
    missing = [n for n in fl.MARK_NAMES if n not in data["marks"]]
    print("还没记：" + "、".join(missing) if missing
          else "12 个标记都记了：python3 run_session.py plan 按逆解收两端、检查全部录制")
    return 0


def cmd_limits(args) -> int:
    """按拖动标记列出每条录制的路径（还没按逆解收两端；不连机械臂）。"""
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    marks = load_marks(f_t_c)
    if not marks:
        print(f"还没有拖动标记（{p(LIMIT_POSES)}）：手动拖到极限位姿后运行 python3 run_session.py mark <方向>")
        return 1
    print(f"余量：平移 {LIMIT_MARGIN_MM:g} mm、旋转 {LIMIT_MARGIN_DEG:g}°；角度相对参考起点（{start['created']}），"
          "方位角正 = 向左、俯仰正 = 向上、横滚正 = 画面顶部向右")
    print("\n拖动标记：")
    for name in fl.MARK_NAMES:
        if name not in marks:
            print(f"  {name:<11} （没有）")
            continue
        m = marks[name]
        print(f"  {name:<11} {describe_mark(fl.relative(start['camera'], m['camera']))}")
        print(f"  {'':<11} 关节 {np.round(m['joints'], 1).tolist()}（{fl.describe_config(m['joints'])}；{m['created']}）")
    print("\n录制路径（plan 再按逆解、关节速度、右窗和同步摆动收两端）：")
    for pair in fl.PAIRS:
        names = [n for n in fl.RECORDINGS if fl.RECORD_PAIR[n][0] == pair]
        print(f"\n  [{' / '.join(names)}]")
        print_pair(pair, marks, start, "  ")
    return 0


def cmd_plan(args, problems: list[str]) -> int:
    for m in problems:
        print(f"参数问题：{m}")
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    cfg = motion_config()
    names = args.names or [*fl.RECORDINGS, "sync-test"]
    unknown = [n for n in names if n not in RECORD_NAMES and n != "sync-test"]
    if unknown:
        print(f"不认识的录制名：{unknown}")
        return 2
    marks = load_marks(f_t_c)
    print(f"路径组 {PATH_SET}；相机 {CAMERA_FPS} fps，曝光 {EXPOSURE_US} μs，增益 {ANALOG_GAIN_MDB} mdB")
    print(f"模糊估算：{NEAREST_DEPTH_M} m 处 左右/上下平移 {fm.blur_px(SPEED_MPS, EXPOSURE_US, NEAREST_DEPTH_M):.2f} px，"
          f"前后平移画面边缘约 {1.3 * fm.blur_px(SPEED_MPS, EXPOSURE_US, NEAREST_DEPTH_M):.2f} px；"
          f"旋转 {fm.rotation_blur_px(ROT_DEG_S, EXPOSURE_US):.2f} px（1080p）")
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
                start_q = None
                if name == "sync-test":
                    segs, path_note, found = fm.plan_sync_test(start["camera"], f_t_c, cfg), "", []
                    start_q = start["joints"]
                else:
                    segs, path_note, _, start_q, found = build_recording(rpc, name, marks, start, f_t_c, model, (lo, hi))
                if segs is not None:
                    note, glass = pose_note(name, segs, start["camera"], f_t_c)
                    moved = config_problem(current_q, start_q, REFERENCE if name == "sync-test" else "这条录制的起点")
                    found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale,
                                                                        (lo, hi)) + glass
                print(f"\n=== {name}：{'通过' if not found else '有问题'}")
                for text in (path_note, note if segs is not None else ""):
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


def cmd_motion(args, kind: str, direction: str | None) -> int:
    busy = camera_users() if kind != "goto-start" else []
    if busy:
        print("相机正被占用，先关掉这些程序：\n  " + "\n  ".join(busy))
        return 1
    f_t_c, handeye = gt.load_handeye(p(HANDEYE))
    start = load_start()
    cfg = motion_config()
    model, note = load_speed_model()
    if model is None and kind != "goto-start":
        print(note)
        return 1
    name = direction or kind
    session_dir = None
    if kind != "goto-start":
        root = p(SESSIONS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(root).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"磁盘只剩 {free_gb:.1f} GB，少于 MIN_FREE_GB={MIN_FREE_GB}")
            return 1
        session_dir = root / (f"{dt.datetime.now():%Y%m%d_%H%M%S}_{name}" + (f"_{args.tag}" if args.tag else ""))
        session_dir.mkdir()
        data_dir = session_dir / "save_data"
        global HANGUP_LOG
        HANGUP_LOG = session_dir / "run_session.log"
    else:
        data_dir = Path(tempfile.mkdtemp(prefix="fr5_goto_"))

    rec = StateRecorder(data_dir, ROBOT_IP)
    rec.start()
    zed = ZedRecord(p(ZED_RECORD), session_dir, EXPOSURE_US, ANALOG_GAIN_MDB, CAMERA_FPS) if session_dir else None
    error, ran, segs, mark_path = None, False, [], None
    try:
        rpc = rec.open_rpc()
        runner = fm.MotionRunner(rec, rpc, ACCEL_PCT)
        runner.precheck()
        if model is None:                        # 只有 goto-start 允许没标定
            model = conservative_model(rpc)
            note = "还没有速度标定，按保守假设换算（实际只会更慢）"
        path_note, start_q, target = "", start["joints"], REFERENCE   # goto-start、sync-test：去参考起点
        lo, hi, _ = joint_limits(rpc)
        limits = (lo, hi)
        if kind == "record":                     # 按这一对拖动标记定路径，用逆解收两端
            built, path_note, mark_path, start_q, found = build_recording(rpc, direction, load_marks(f_t_c), start,
                                                                          f_t_c, model, limits)
            if built is None:
                raise fm.MotionAborted("路径没定下来，没有运动：\n" + (path_note + "\n" if path_note else "")
                                       + "\n".join(f"  {m}" for m in found))
            segs, target = built, "这条录制的起点"
        elif kind == "sync-test":
            segs = fm.plan_sync_test(start["camera"], f_t_c, cfg)
        else:
            segs = fm.plan_sync_test(start["camera"], f_t_c, cfg)[:1]  # 只移到参考起点
        pose, found = pose_note(name, segs, start["camera"], f_t_c)
        moved = config_problem(current_pose(rec)[1], start_q, target)
        found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale, limits) + found
        if found:
            raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found)
                                   + ("\n（goto_start 走不通时，先手动拖到这条录制的起点附近再试）"
                                      if any(m.startswith("goto_start") for m in found) else ""))
        header = (f"{name}：速度比例 {args.scale}，路径组 {PATH_SET}"
                  + (f"，{CAMERA_FPS} fps，曝光 {EXPOSURE_US} μs，增益 {ANALOG_GAIN_MDB} mdB" if zed else "")
                  + f"\n{note}；WebApp 速度百分比应为 {WEBAPP_SPEED_PCT}%"
                  + (f"\n{path_note}" if path_note else "")
                  + (f"\n{pose}" if pose else ""))
        confirm_plan(header + "\n" + describe(segs, totals=zed is not None), args.yes)
        ran = True
        runner.run(segs[0])                      # 从当前位姿移到录制起点（相机还没开始录）
        report_move(segs[0])
        if zed:
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
        if zed and zed.proc is not None:
            zed.stop()
        rec.stop(write=session_dir is not None and ran)

    if session_dir is not None and not ran:
        if HANGUP_LOG.exists() and HANGUP_LOG.parent == session_dir:  # 断线日志留到 /tmp
            shutil.copy2(HANGUP_LOG, Path(tempfile.gettempdir()) / "run_session_hangup.log")
        shutil.rmtree(session_dir, ignore_errors=True)  # 没有运动，也没有录到任何东西
        return 1
    if session_dir is None:
        shutil.rmtree(data_dir, ignore_errors=True)
        print("已到参考起点" if error is None else "")
        return 0 if error is None else 1

    zed_error = zed_failure(zed.info if zed else None, data_dir)
    log = {
        "kind": kind, "direction": direction, "tag": args.tag, "speed_scale": args.scale,
        "error": "；".join(e for e in (error, zed_error) if e) or None,
        "parameters": {k: globals()[k] for k in [*RANGES, "JOINT_MIN_DEG", "JOINT_MAX_DEG", "GLASS_REGIONS", "PATH_SET"]},
        "mark_path": mark_path,                  # 这条录制的拖动标记、定出的路径和收两端的结果（fr5_limits）
        "speed_model": dataclasses.asdict(model) if model else None,
        "start_pose": {k: start[k] for k in ("base_T_camera", "flange", "joints", "created")},
        "handeye_sha256": handeye["sha256"],
        "zed_record": zed.info if zed else None,
        "segments": [segment_log(s) for s in segs],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    translate = next((s for s in segs if s.name == "translate"), None)
    if translate is not None and error is None:
        log["translate_speed"] = translate_summary(translate, args.scale)
    rotate = next((s for s in segs if s.name == "rotate"), None)
    if rotate is not None and error is None:
        log["rotate_speed"] = rotate_summary(rotate, args.scale)
    (data_dir / "motion_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n数据目录：{session_dir}")
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
    if a_lin:
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


def cmd_movec_test(args) -> int:
    """MoveC 探测：起点上用 MoveC 和 MoveL 各绕光心竖直轴转 +θ、-2θ、+θ（θ = WIGGLE_DEG，同步摆动已走过的范围），
    速度都是 SPEED_TEST_PCT。看 MoveC 是否被接受、光心是否不动、速度百分比和 MoveL 是否一样。不开相机，结果存 MOVEC_TEST。"""
    model, note = load_speed_model()
    if model is None:
        print(note)
        return 1
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    segs = fm.plan_movec_test(start["camera"], f_t_c, motion_config(), SPEED_TEST_PCT)
    error, ran = None, False
    # 原始状态包存下来（结束时写一次），事后能逐包查速度和光心；没有运动就不留目录
    state_dir = p(SESSIONS_DIR) / f"{dt.datetime.now():%Y%m%d_%H%M%S}_movec-test"
    rec = StateRecorder(state_dir, ROBOT_IP)
    try:
        rec.start()
        rpc = rec.open_rpc()
        runner = fm.MotionRunner(rec, rpc, ACCEL_PCT)
        runner.precheck()
        moved = config_problem(current_pose(rec)[1], start["joints"], REFERENCE)
        found = ([moved] if moved else []) + check_segments(rec, rpc, segs, f_t_c, model, args.scale,
                                                            joint_limits(rpc)[:2])
        if found:
            raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found))
        confirm_plan(f"movec-test：圆弧（MoveC）和直线（MoveL）各转 +{WIGGLE_DEG:g}°、-{2 * WIGGLE_DEG:g}°、+{WIGGLE_DEG:g}°，"
                     f"速度都是 {SPEED_TEST_PCT}%（预计按摆动标定估算，超过 1.5 倍就中止；"
                     f"圆弧时光心偏离超过 {fm.MAX_PIVOT_DEV_MM} mm 也中止）\n{note}\n只录机械臂状态，不开相机\n"
                     + describe(segs, totals=False), args.yes)
        ran = True
        for seg in segs:
            print(f"[{seg.name}]", flush=True)
            runner.run(seg)
            report_move(seg)
    except BaseException as exc:  # noqa: BLE001 - 包括 Ctrl+C：先停机械臂
        error = f"{type(exc).__name__}: {exc}"
        if ran:
            fm.emergency_stop(ROBOT_IP)
        print(f"\n中止：{error}")
    finally:
        rec.stop(write=ran)
        if not ran and state_dir.is_dir() and not any(state_dir.iterdir()):
            state_dir.rmdir()
    if not ran:
        return 1

    def stats(prefix):
        done = [s for s in segs if s.name.startswith(prefix) and s.log.get("measured")]
        rates = [k for k in (_per_pct(s, "angular_deg_s") for s in done) if k]
        devs = [s.log["pivot_dev_max_mm"] for s in done if s.log.get("pivot_dev_max_mm") is not None]
        return (float(np.median(rates)) if rates else None), (max(devs) if devs else None), len(done)

    (rate_c, dev_c, n_c), (rate_l, dev_l, n_l) = stats("arc"), stats("line")
    result = {"created": dt.datetime.now().isoformat(timespec="seconds"), "error": error,
              "settings": {k: globals()[k] for k in (*CAL_KEYS, "SPEED_TEST_PCT")}, "speed_scale": args.scale,
              "movec": {"segments_done": n_c, "deg_s_per_pct": rate_c, "pivot_dev_max_mm": dev_c},
              "movel": {"segments_done": n_l, "deg_s_per_pct": rate_l, "pivot_dev_max_mm": dev_l},
              "state_dir": str(state_dir), "segments": [segment_log(s) for s in segs]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    p(MOVEC_TEST).write_text(text, encoding="utf-8")
    (state_dir / "movec_test.json").write_text(text, encoding="utf-8")
    print(f"\n已保存 {p(MOVEC_TEST)}；原始状态在 {state_dir}/")

    def fmt(v, spec):
        return "—" if v is None else format(v, spec)

    print(f"  MoveC：完成 {n_c}/3 段，1% = {fmt(rate_c, '.3f')} °/s，光心最大偏离 {fmt(dev_c, '.2f')} mm")
    print(f"  MoveL：完成 {n_l}/3 段，1% = {fmt(rate_l, '.3f')} °/s，光心最大偏离 {fmt(dev_l, '.2f')} mm"
          f"（摆动标定 1% = {model.wiggle_deg_s_per_pct:.3f} °/s）")
    if rate_c and rate_l and abs(rate_c / rate_l - 1.0) > MAX_SPEED_ERROR:
        print(f"  注意：MoveC 和 MoveL 同样百分比下角速度差 {rate_c / rate_l - 1.0:+.0%}，旋转录制的速度要另行换算")
    return 0 if error is None else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("set-start")
    mark = sub.add_parser("mark")
    mark.add_argument("direction", choices=fl.MARK_NAMES, help="拖到的是哪个方向的极限")
    sub.add_parser("limits")
    plan = sub.add_parser("plan")
    plan.add_argument("names", nargs="*", help="要检查的录制，默认 " + " ".join(fl.RECORDINGS) + " 和 sync-test")
    for name in ("goto-start", "speed-test", "sync-test"):
        sub.add_parser(name)
    sub.add_parser("movec-test")
    record = sub.add_parser("record")
    record.add_argument("direction", choices=RECORD_NAMES)
    for sp in sub.choices.values():
        sp.add_argument("--scale", type=float, default=1.0, help="速度比例 (0, 1]，首次运行建议 0.25")
        sp.add_argument("--yes", action="store_true", help="跳过确认")
        sp.add_argument("--tag", default="", help="加在录制目录名后面的标签，例如 board（标定板验证）")
    args = parser.parse_args(argv)
    if not 0 < args.scale <= 1:
        parser.error("--scale 必须在 (0, 1] 之间")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9-]{1,20}", args.tag):
        parser.error("--tag 只能用字母、数字和 -，最多 20 个字符")
    signal.signal(signal.SIGHUP, _on_hangup)
    signal.signal(signal.SIGTERM, _on_hangup)
    if args.command == "set-start":
        return cmd_set_start(args)
    if args.command == "mark":
        return cmd_mark(args)
    if args.command == "limits":
        return cmd_limits(args)
    problems = parameter_problems()
    if args.command == "plan":
        return cmd_plan(args, problems)
    if problems:
        print("参数问题，没有运动：\n  " + "\n  ".join(problems))
        return 1
    if args.command == "speed-test":
        return cmd_speed_test(args)
    if args.command == "movec-test":
        return cmd_movec_test(args)
    return cmd_motion(args, args.command, getattr(args, "direction", None))


if __name__ == "__main__":
    sys.exit(main())
