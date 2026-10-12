#!/usr/bin/env python3
"""一条录制的总控：FR5 运动（fr5_motion）+ 机械臂状态（fr5_record）+ 图像和 IMU（zed_capture）+ 真值（fr5_groundtruth）。
所有可改参数都在本文件开头，其他模块不放可调参数。

在 Box 自己的终端里运行（人在机械臂旁边，手边有急停）：
    python3 run_session.py set-start        读当前位姿，算出相机水平的起点并保存（不动机械臂）
    python3 run_session.py plan             空跑：检查 6 条录制和 sync-test 的全部目标点（不动机械臂）
    python3 run_session.py goto-start       低速移到水平起点（会先确认）
    python3 run_session.py speed-test       速度标定：实测速度百分比对应的 mm/s 和 °/s（会先确认，不开相机）
    python3 run_session.py sync-test        时间同步验证：只做同步摆动的录制（会先确认）
    python3 run_session.py record forward   正式录制一条：forward/backward/right/left/up/down（会先确认）
选项：--scale 0.25  首次运行按比例降速；--yes  跳过确认（只在你已经看过计划时用）；
      --tag board  加在录制目录名后面（区分验证录制和正式录制）

每条录制单独一个目录 SESSIONS_DIR/时间_名称/，相机和机械臂数据都在其中的 save_data/ 里。
sync-test 和 record 需要先有 speed-test 的结果；改了 WebApp 速度百分比要重跑 speed-test。
"""

# ================================ 可修改参数 ================================
SPEED_MPS = 0.05             # 平移速度 m/s
STROKE_M = 0.30              # 每条录制的行程 m（以 set-start 保存的起点为中心，两边各一半）
EXPOSURE_US = 10000          # 曝光 μs（传给 zed_record）
ANALOG_GAIN_MDB = 1000       # 模拟增益 mdB（传给 zed_record）；缩短曝光后画面偏暗时调大
MAX_BLUR_PX = 1.0            # 允许的最大运动模糊（1080p 像素）
NEAREST_DEPTH_M = 1.0        # 画面里最近物体的距离 m，用来估算运动模糊
HOLD_START_S = 5.0           # 开头静止 s（同时用来估陀螺零偏）
HOLD_MID_S = 3.0             # 同步摆动和平移之间的静止 s
HOLD_END_S = 5.0             # 结尾静止 s
WIGGLE_DEG = 10.0            # 同步摆动幅度 °（绕相机竖直轴转，相机保持水平）
WIGGLE_DEG_S = 20.0          # 同步摆动角速度 °/s（真值脚本要求至少 10 °/s）
WIGGLE_CYCLES = 2            # 同步摆动来回次数
SYNC_TEST_HOLD_S = 10.0      # sync-test 两次摆动之间的静止 s
REPOSITION_MPS = 0.05        # 录制前移到起点的速度 m/s
WEBAPP_SPEED_PCT = 10        # WebApp 里的速度百分比（SDK 读不到，照抄过来）；WebApp 改了这里也要改，并重跑 speed-test
ACCEL_PCT = 100              # MoveL 加速度百分比（oacc）；实际加速度看 speed-test 结果，起停太猛就调小再重跑 speed-test
SPEED_TEST_PCT = 5           # speed-test 去程的速度百分比；没标定时最快为控制器默认速度 1000 mm/s 的 5% = 50 mm/s
SPEED_TEST_MM = 100          # speed-test 在起点竖直上下的行程 mm
MAX_SPEED_ERROR = 0.10       # 平移实测速度和预计速度相差超过 10% 就警告（speed-test 里也用来检查是否成正比）
MIN_CAMERA_Z_M = 0.35        # 相机光心最低高度（基坐标系 z，m）；按现场桌面和障碍物修改
# 允许的关节范围（°）。默认覆盖当前起点附近 ±150 mm 平移和 ±15° 摆动并留有余量，
# 同时离腕部（J5=0）和肘部（J3=0）奇异位形至少 15°，J6 限制在 ±30° 以免扯线。按现场确认修改。
JOINT_MIN_DEG = [80.0, -30.0, -130.0, -175.0, -90.0, -30.0]
JOINT_MAX_DEG = [160.0, 60.0, -15.0, -70.0, -15.0, 30.0]
MAX_JOINT_STEP_DEG = 5.0     # 路径上每 10 mm / 2° 允许的最大关节变化（超过说明逆解可能跳支）
MAX_SPEED_RIPPLE = 0.05      # 平移巡航段速度波动（标准差/均值）超过它就警告“速度不平稳”
MIN_FREE_GB = 5.0            # 开始录制前 Box 至少剩余的磁盘空间 GB
ROBOT_IP = "192.168.58.2"
SESSIONS_DIR = "~/ROBOT/sessions"
ZED_RECORD = "~/ROBOT/zed_imu/build/zed_record"
HANDEYE = "~/ROBOT/fr5_record/handeye_s003_result.json"
START_POSE = "~/ROBOT/fr5_record/start_pose.json"
SPEED_CAL = "~/ROBOT/fr5_record/speed_calibration.json"

# 允许范围 [下限, 上限]，超出就拒绝运行
RANGES = {
    "SPEED_MPS": (0.02, 0.10),       # 下限暂定，待平稳性测试确认；上限还受 MAX_BLUR_PX 约束
    "STROKE_M": (0.05, 0.30),        # ±150 mm 已做离线可达性检查
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
    "SPEED_TEST_MM": (50, 150),       # 起点向上 150 mm 已做离线可达性检查
    "MAX_SPEED_ERROR": (0.02, 0.3),
    "MIN_CAMERA_Z_M": (0.0, 1.5),
    "MAX_JOINT_STEP_DEG": (1.0, 15.0),
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
from pathlib import Path

import numpy as np

import fr5_groundtruth as gt
import fr5_motion as fm
from fr5_record import StateRecorder
from zed_capture import ZedRecord, camera_users

DATA_MB_PER_S = 12.0  # zed_record 双目 JPEG 约 0.8 MB/帧 × 15 fps
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
    return fm.MotionConfig(SPEED_MPS, STROKE_M, HOLD_START_S, HOLD_MID_S, HOLD_END_S,
                           WIGGLE_DEG, WIGGLE_DEG_S, int(WIGGLE_CYCLES), REPOSITION_MPS, SYNC_TEST_HOLD_S)


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
    m = cal["model"]
    return (fm.SpeedModel(m["mm_s_per_pct"], m["wiggle_deg_s_per_pct"], m["ramp_s"], calibrated=True),
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


def check_segments(rec: StateRecorder, rpc, segs: list[fm.Segment], f_t_c, model, scale: float) -> list[str]:
    """从当前位姿出发做逆解检查，并按速度模型换算每段的速度百分比。返回问题列表。"""
    flange, joints, _ = current_pose(rec)
    return (fm.check_plan(rpc, segs, flange, joints, f_t_c, JOINT_MIN_DEG, JOINT_MAX_DEG,
                          MAX_JOINT_STEP_DEG, MIN_CAMERA_Z_M)
            + fm.apply_speed(segs, model, scale))


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
        head = f"  {s.name:<16} 移动 {s.distance_mm:5.1f} mm / {s.angle_deg:4.1f}°，"
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
        lines.append(f"  录制约 {rec_s:.0f} s，数据约 {rec_s * DATA_MB_PER_S / 1000:.1f} GB")
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


def cmd_plan(args, problems: list[str]) -> int:
    for m in problems:
        print(f"参数问题：{m}")
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    cfg = motion_config()
    names = args.names or [*fm.DIRECTIONS, "sync-test"]
    unknown = [n for n in names if n not in fm.DIRECTIONS and n != "sync-test"]
    if unknown:
        print(f"不认识的录制名：{unknown}")
        return 2
    print(f"模糊估算：{NEAREST_DEPTH_M} m 处 左右/上下平移 {fm.blur_px(SPEED_MPS, EXPOSURE_US, NEAREST_DEPTH_M):.2f} px，"
          f"前后平移画面边缘约 {1.3 * fm.blur_px(SPEED_MPS, EXPOSURE_US, NEAREST_DEPTH_M):.2f} px（1080p）")
    model, note = load_speed_model()
    print(note if model else f"注意：{note}；下面没有速度百分比，时长按设计速度估算")
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            rpc = rec.open_rpc()
            for name in names:
                segs = (fm.plan_sync_test(start["camera"], f_t_c, cfg) if name == "sync-test"
                        else fm.plan_recording(name, start["camera"], f_t_c, cfg))
                found = check_segments(rec, rpc, segs, f_t_c, model, args.scale)
                print(f"\n=== {name}：{'通过' if not found else '有问题'}")
                print(describe(segs))
                for m in found:
                    print(f"  问题：{m}")
                problems += found
        finally:
            rec.stop(write=False)
    print("\n空跑检查" + ("全部通过（机械臂没有动）" if not problems else f"发现 {len(problems)} 个问题（机械臂没有动）"))
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

    if kind == "record":
        segs = fm.plan_recording(direction, start["camera"], f_t_c, cfg)
    elif kind == "sync-test":
        segs = fm.plan_sync_test(start["camera"], f_t_c, cfg)
    else:
        segs = fm.plan_sync_test(start["camera"], f_t_c, cfg)[:1]  # 只移到起点

    rec = StateRecorder(data_dir, ROBOT_IP)
    rec.start()
    zed = ZedRecord(p(ZED_RECORD), session_dir, EXPOSURE_US, ANALOG_GAIN_MDB) if session_dir else None
    error, ran = None, False
    try:
        rpc = rec.open_rpc()
        runner = fm.MotionRunner(rec, rpc, ACCEL_PCT)
        runner.precheck()
        if model is None:                        # 只有 goto-start 允许没标定
            model = conservative_model(rpc)
            note = "还没有速度标定，按保守假设换算（实际只会更慢）"
        found = check_segments(rec, rpc, segs, f_t_c, model, args.scale)
        if found:
            raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found))
        header = (f"{name}：速度比例 {args.scale}" + (f"，曝光 {EXPOSURE_US} μs，增益 {ANALOG_GAIN_MDB} mdB" if zed else "")
                  + f"\n{note}；WebApp 速度百分比应为 {WEBAPP_SPEED_PCT}%")
        confirm_plan(header + "\n" + describe(segs, totals=zed is not None), args.yes)
        ran = True
        runner.run(segs[0])                      # 移到起点（相机还没开始录）
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
        print("已到起点" if error is None else "")
        return 0 if error is None else 1

    log = {
        "kind": kind, "direction": direction, "tag": args.tag, "speed_scale": args.scale, "error": error,
        "parameters": {k: globals()[k] for k in [*RANGES, "JOINT_MIN_DEG", "JOINT_MAX_DEG"]},
        "speed_model": dataclasses.asdict(model) if model else None,
        "start_pose": {k: start[k] for k in ("base_T_camera", "flange", "joints", "created")},
        "handeye_sha256": handeye["sha256"],
        "zed_record": zed.info if zed else None,
        "segments": [{"name": s.name, "kind": s.kind, "recorded": s.recorded, "planned_duration_s": s.duration_s,
                      "flange": s.flange, "joints": s.joints, "design_speed_mm_s": s.speed_mm_s,
                      "design_rate_deg_s": s.rate_deg_s, "distance_mm": s.distance_mm, "angle_deg": s.angle_deg,
                      "percent": s.percent, "expect_mm_s": s.expect_mm_s, "expect_deg_s": s.expect_deg_s,
                      **s.log} for s in segs],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    translate = next((s for s in segs if s.name == "translate"), None)
    if translate is not None and error is None:
        log["translate_speed"] = translate_summary(translate, args.scale)
    (data_dir / "motion_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n数据目录：{session_dir}")
    speed = log.get("translate_speed")
    if speed:
        print(f"平移实际速度 {speed['mean_mm_s']:.1f} mm/s（预计 {speed['expected_mm_s']:.1f}，设计 {speed['design_mm_s']:.1f}），"
              f"波动 {speed['ripple']:.1%}"
              + ("" if speed["smooth"] else f"，超过 MAX_SPEED_RIPPLE={MAX_SPEED_RIPPLE:.0%}，速度不平稳")
              + ("" if speed["speed_ok"] else f"；和预计差 {speed['error']:+.0%}，超过 MAX_SPEED_ERROR，重跑 speed-test"))
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
    return 0 if error is None else 1


def _per_pct(seg: fm.Segment, key: str) -> float | None:
    """实测巡航速度 ÷ 速度百分比；没测到返回 None。key：linear_mm_s 或 angular_deg_s。"""
    m = (seg.log.get("measured") or {}).get(key)
    return m["mean"] / seg.percent if m and m["mean"] > 0 and seg.percent else None


def cmd_speed_test(args) -> int:
    """实测 MoveL 速度百分比和实际速度的换算，保存到 SPEED_CAL。不开相机，机械臂状态也不存盘。"""
    f_t_c, _ = gt.load_handeye(p(HANDEYE))
    start = load_start()
    segs = fm.plan_speed_test(start["camera"], f_t_c, motion_config(), SPEED_TEST_MM, SPEED_TEST_PCT)
    by = {s.name: s for s in segs}
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
            found = check_segments(rec, rpc, segs, f_t_c, model, args.scale)
            if found:
                raise fm.MotionAborted("检查没通过，没有运动：\n  " + "\n  ".join(found))
            pending = {"test_down": f"速度按 test_up 实测换算，目标 {SPEED_MPS * 1000 * args.scale:.1f} mm/s",
                       "test_yaw_back": f"速度按 test_yaw 实测换算，目标 {WIGGLE_DEG_S * args.scale:.1f} °/s"}
            confirm_plan(f"speed-test：速度比例 {args.scale}；控制器默认速度 {vmax:.0f} mm/s，"
                         f"WebApp 速度百分比应为 {WEBAPP_SPEED_PCT}%，加速度百分比 {ACCEL_PCT}%\n"
                         "只录机械臂状态，不开相机。“最快”是没标定时的上限，实际可能慢很多\n"
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
    for label, pct in (("平移", SPEED_MPS * 1000.0 / kt), ("移到起点", REPOSITION_MPS * 1000.0 / kt),
                       ("同步摆动", WIGGLE_DEG_S / kw)):
        print(f"  {label}需要速度 {pct:.2f}%" + ("：超过 100%，调高 WebApp 速度百分比后重跑" if pct > 100 else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("set-start")
    plan = sub.add_parser("plan")
    plan.add_argument("names", nargs="*", help="要检查的录制，默认全部：forward backward right left up down sync-test")
    for name in ("goto-start", "speed-test", "sync-test"):
        sub.add_parser(name)
    record = sub.add_parser("record")
    record.add_argument("direction", choices=list(fm.DIRECTIONS))
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
    problems = parameter_problems()
    if args.command == "plan":
        return cmd_plan(args, problems)
    if problems:
        print("参数问题，没有运动：\n  " + "\n  ".join(problems))
        return 1
    if args.command == "speed-test":
        return cmd_speed_test(args)
    return cmd_motion(args, args.command, getattr(args, "direction", None))


if __name__ == "__main__":
    sys.exit(main())
