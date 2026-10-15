#!/usr/bin/env python3
"""眼在手上标定的采集（Box 上运行）。位姿由 handeye_poses.py 在笔记本上选好，写在 poses.json 里。

    python3 handeye_capture.py plan poses.json                       空跑：控制器逆解复核，不动机械臂
    python3 handeye_capture.py capture poses.json [--session DIR]    采集（默认 ~/ROBOT/handeye/s004）

采集顺序：位姿 1…N，第 1 个位姿在中间和结尾各回去拍一张检查图（看相机、标定板有没有动）。
plan 和 capture 开始前都按这个顺序检查：控制器逆解（以上一个位姿的关节角为参考）和笔记本上用名义运动学算的
相差不超过 MAX_IK_DIFF_DEG（否则控制器会选另一组解）；关节在 poses.json 的允许范围和控制器关节软限位内；
构型不变；J3、J5 离 0 至少 SINGULAR_MARGIN_DEG；相邻两个位姿之间 MoveJ（关节直线插值）经过的法兰位置在允许的
外接盒内。

capture 对每个位姿：低速 MoveJ 过去（运动中监视状态：报警、急停、20004 断开、关节速度、关节和法兰超出允许
范围、停住不动、超时，任何一项就发 StopMotion 中止）；到位后再静止 SETTLE_S；调用 ~/zed_snapshot.py 拍一张
校正后左目 PNG（HD1080、FLIP_MODE.OFF、关闭自标定，只在这时存图）；拍照期间收集机械臂状态，法兰变化超过
STILL_MM / STILL_DEG 就中止（说明在动）；取中间那一包的法兰位姿和关节角写进 poses.csv（标定仓库的格式）。
检查图存在 checks/，不进 poses.csv。中断后加 --resume 从没拍的那一张接着拍。Ctrl+C、SSH 断开时先停机械臂。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

import fr5_limits as fl
import fr5_motion as fm
import run_session as rs

MAX_IK_DIFF_DEG = 0.5
SINGULAR_MARGIN_DEG = 5.0
PATH_SAMPLES = 20
MOVEJ_VEL_PCT = 10.0            # MoveJ 速度百分比（WebApp 的速度百分比还会再乘上去）
MAX_JOINT_SPEED_DEG_S = 40.0    # 运动中任一关节超过它就中止（防速度设置出错）
JOINT_TOL_DEG = 2.0             # 运动中关节超出允许范围这么多就中止
FLANGE_TOL_MM = 15.0            # 运动中法兰超出允许的外接盒这么多就中止
ARRIVE_JOINT_DEG = 0.02         # 到位：各关节离目标都在这么多以内，并且 robot_state == 1（停止）
MOVE_TIMEOUT_S = 120.0
SETTLE_S = 3.0                  # 到位后再静止这么久才拍（10-15 实测到位 1 s 时法兰还在微动 0.03 mm）
STILL_MM, STILL_DEG = 0.05, 0.005   # 拍照期间法兰位置、关节的变化上限（静止时 3 s 内约 0.01 mm、0.0007°）
SNAPSHOT = Path("~/zed_snapshot.py").expanduser()
SNAPSHOT_TIMEOUT_S = 60.0
SESSION_DIR = Path("~/ROBOT/handeye/s004").expanduser()
CAMERA_PROGRAMS = ("zed_record", "ZED_Explorer", "ZED_Depth_Viewer", "zed_snapshot")
POSES_HEADER = ["image", "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg",
                "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg"]
# FR5 名义 DH（和标定仓库 handeye/kinematics.py 相同；和控制器报告的法兰位姿差 < 0.1 mm）
FR5_DH = ((0.0, math.pi / 2, 152.0), (-425.0, 0.0, 0.0), (-395.0, 0.0, 0.0),
          (0.0, math.pi / 2, 102.0), (0.0, -math.pi / 2, 102.0), (0.0, 0.0, 100.0))


class Interrupted(KeyboardInterrupt):
    pass


HANGUP_LOG = Path("~/ROBOT/handeye/hangup.log").expanduser()


def _on_signal(signum, frame):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    HANGUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(HANGUP_LOG, "a", encoding="utf-8", buffering=1)   # 终端可能已经没了
    raise Interrupted(f"收到 {signal.Signals(signum).name}")


def fk_position_mm(joints_deg) -> np.ndarray:
    """法兰位置（mm），名义运动学。"""
    t = np.eye(4)
    for theta, (a, alpha, d) in zip(np.radians(np.asarray(joints_deg, dtype=float)), FR5_DH):
        ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
        t = t @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                          [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]])
    return t[:3, 3]


def sequence(plan: dict) -> list[dict]:
    """采集顺序：位姿 1…N，第 1 个位姿在中间和结尾各回去拍一张检查图（存在 checks/，不进 poses.csv）。"""
    poses = [p for p in plan["poses"] if p["role"] == "pose"]
    check = next(p for p in poses if p["index"] == plan.get("check_pose", 1))
    half = len(poses) // 2
    seq = [dict(p, shot=f"images/{p['index']:04d}.png") for p in poses[:half]]
    seq.append(dict(check, shot=f"checks/mid_{check['index']:04d}.png"))
    seq += [dict(p, shot=f"images/{p['index']:04d}.png") for p in poses[half:]]
    seq.append(dict(check, shot=f"checks/end_{check['index']:04d}.png"))
    return seq


def path_out(q0, q1, fmin, fmax) -> int:
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    pts = [fk_position_mm(q0 + (q1 - q0) * s) for s in np.linspace(0.0, 1.0, PATH_SAMPLES)]
    return sum(bool(np.any(p < fmin) or np.any(p > fmax)) for p in pts)


def check_sequence(rpc, plan: dict, seq: list[dict], q_now, soft_lo, soft_hi, verbose: bool = True):
    """按采集顺序逐个用控制器逆解复核。返回 (每一步的目标关节角或 None, 问题列表)。"""
    env = plan["envelope"]
    lo, hi = np.array(env["joint_min_deg"]), np.array(env["joint_max_deg"])
    fmin, fmax = np.array(env["flange_min_mm"]), np.array(env["flange_max_mm"])
    lo_all, hi_all = np.maximum(lo, soft_lo), np.minimum(hi, soft_hi)
    problems, targets = [], []
    if np.any(np.asarray(q_now) < lo) or np.any(np.asarray(q_now) > hi):
        problems.append(f"机械臂现在的关节角不在允许范围内：{np.round(q_now, 1).tolist()}")
    config0, q_prev = fl.config_of(q_now), list(q_now)
    for k, p in enumerate(seq, start=1):
        tag = f"{k:2d}. 位姿 {p['index']:2d}（{p['shot']}）"
        err, q = fm.inverse_kin(rpc, fm.fr5_to_matrix(p["flange_mm_deg"]), q_prev)
        if q is None:
            problems.append(f"{tag}：控制器逆解失败（错误码 {err}）")
            targets.append(None)
            if verbose:
                print(f"  {tag}：逆解失败 {err}")
            continue
        diff = float(np.abs(np.asarray(q) - np.asarray(p["joints_deg"])).max())
        excess = np.maximum(np.maximum(lo_all - q, np.asarray(q) - hi_all), 0.0)
        out = path_out(q_prev, q, fmin, fmax)
        move = float(np.abs(np.asarray(q) - np.asarray(q_prev)).max())
        notes = []
        if diff > MAX_IK_DIFF_DEG:
            notes.append(f"和笔记本算的差 {diff:.2f}°（控制器选了另一组解）")
        if excess.max() > 0:
            notes.append("超出允许范围：" + "，".join(f"J{j + 1} {excess[j]:.1f}°" for j in np.nonzero(excess)[0]))
        if fl.config_of(q) != config0:
            notes.append(f"构型变了（{fl.describe_config(q)}）")
        sing = fl.singular_problem(q, SINGULAR_MARGIN_DEG)
        if sing:
            notes.append(sing)
        if out:
            notes.append(f"MoveJ 过来时法兰有 {out}/{PATH_SAMPLES} 个采样点出了允许的外接盒")
        problems += [f"{tag}：{n}" for n in notes]
        targets.append(q)
        if verbose:
            print(f"  {tag}：关节 {np.round(q, 1).tolist()}，最大移动 {move:5.1f}°，和笔记本差 {diff:.3f}°"
                  + ("" if not notes else "  ← " + "；".join(notes)))
        q_prev = q
    return targets, problems


def open_robot(tmp: str):
    rec = rs.StateRecorder(tmp, rs.ROBOT_IP)
    rec.start()
    rpc = rec.open_rpc()
    soft_lo, soft_hi, note = rs.joint_limits(rpc)
    _, q_now, _ = rs.current_pose(rec)
    print(note)
    return rec, rpc, soft_lo, soft_hi, q_now


def cmd_plan(plan: dict) -> int:
    env = plan["envelope"]
    seq = sequence(plan)
    with tempfile.TemporaryDirectory() as tmp:
        rec, rpc, soft_lo, soft_hi, q_now = open_robot(tmp)
        try:
            print(f"允许范围（poses.json）：关节 {env['joint_min_deg']} ～ {env['joint_max_deg']}；"
                  f"法兰 {env['flange_min_mm']} ～ {env['flange_max_mm']} mm")
            print(f"机械臂现在：{np.round(q_now, 1).tolist()}（{fl.describe_config(q_now)}）")
            print(f"\n采集顺序（{len(seq)} 次拍照）：")
            _, problems = check_sequence(rpc, plan, seq, q_now, soft_lo, soft_hi)
        finally:
            rec.stop(write=False)
    print("\n空跑没有发现问题，机械臂没有动。" if not problems else "\n问题：\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


# ---------------------------------------------------------------- 采集
def movej(rec, rpc, q_target, plan: dict, label: str) -> dict:
    """MoveJ 到 q_target 并监视到到位；出任何问题就 StopMotion 并抛 MotionAborted。返回运动记录。"""
    env = plan["envelope"]
    lo, hi = np.array(env["joint_min_deg"]) - JOINT_TOL_DEG, np.array(env["joint_max_deg"]) + JOINT_TOL_DEG
    fmin, fmax = np.array(env["flange_min_mm"]) - FLANGE_TOL_MM, np.array(env["flange_max_mm"]) + FLANGE_TOL_MM
    q_target = np.asarray(q_target, dtype=float)
    count, pkg = rec.wait_state(-1, 1.0)
    age = time.monotonic() - (rec.latest_monotonic or -math.inf)
    problem = fm.state_problem(pkg)
    if problem is None and age > fm.FRESH_S:
        problem = f"最新的机械臂状态是 {age:.1f} s 前的（20004 断开？）"
    if problem is None and pkg.robot_state != 1:
        problem = f"机械臂不在停止状态（robot_state={pkg.robot_state}）"
    if problem:
        raise fm.MotionAborted(f"{label}：{problem}，没有发运动指令")
    t0 = time.monotonic()
    call = fm.BlockingCall(lambda: rpc.MoveJ(joint_pos=q_target.tolist(), tool=0, user=0, vel=MOVEJ_VEL_PCT,
                                             acc=0.0, ovl=100.0))
    hist = deque()
    peak = 0.0
    mark_q, mark_t = np.array(pkg.jt_cur_pos, dtype=float), time.monotonic()
    try:
        while True:
            count, pkg = rec.wait_state(count, 0.5)
            now = time.monotonic()
            if now - (rec.latest_monotonic or -math.inf) > fm.STATE_LOST_S:
                raise fm.MotionAborted(f"{label}：超过 {fm.STATE_LOST_S:.0f} s 没收到机械臂状态（20004 断开？）")
            problem = fm.state_problem(pkg) or call.problem()
            if problem:
                raise fm.MotionAborted(f"{label}：{problem}")
            q = np.array(pkg.jt_cur_pos, dtype=float)
            pos = np.array(pkg.flange_cur_pos[:3], dtype=float)
            hist.append((now, q))
            while len(hist) > 2 and now - hist[1][0] >= 0.1:
                hist.popleft()
            if now - hist[0][0] >= 0.1:
                speed = float(np.abs(q - hist[0][1]).max() / (now - hist[0][0]))
                peak = max(peak, speed)
                if speed > MAX_JOINT_SPEED_DEG_S:
                    raise fm.MotionAborted(f"{label}：关节速度 {speed:.0f} °/s 超过 {MAX_JOINT_SPEED_DEG_S:.0f} °/s")
            if np.any(q < lo) or np.any(q > hi):
                raise fm.MotionAborted(f"{label}：关节 {np.round(q, 1).tolist()} 超出允许范围")
            if np.any(pos < fmin) or np.any(pos > fmax):
                raise fm.MotionAborted(f"{label}：法兰 {np.round(pos, 0).tolist()} mm 超出允许的空间")
            if float(np.abs(q - q_target).max()) < ARRIVE_JOINT_DEG and pkg.robot_state == 1:
                if not call.done(fm.CALL_RETURN_S):
                    raise fm.MotionAborted(f"{label}：到位 {fm.CALL_RETURN_S:.0f} s 后 MoveJ 还没返回")
                if call.problem():
                    raise fm.MotionAborted(f"{label}：MoveJ {call.problem()}")
                return {"duration_s": round(now - t0, 2), "peak_joint_deg_s": round(peak, 1)}
            if float(np.abs(q - mark_q).max()) > 0.02:
                mark_q, mark_t = q, now
            elif now - mark_t > fm.STALL_S:
                raise fm.MotionAborted(f"{label}：{fm.STALL_S:.0f} s 没有移动，也没到位（robot_state={pkg.robot_state}）")
            if now - t0 > MOVE_TIMEOUT_S:
                raise fm.MotionAborted(f"{label}：{MOVE_TIMEOUT_S:.0f} s 没有到位")
    except BaseException:
        fm.emergency_stop(rs.ROBOT_IP)
        raise


def settle(rec, label: str) -> None:
    """到位后再静止 SETTLE_S，期间状态不能出问题。"""
    count, end = -1, time.monotonic() + SETTLE_S
    while time.monotonic() < end:
        count, pkg = rec.wait_state(count, 0.5)
        problem = fm.state_problem(pkg)
        if problem:
            raise fm.MotionAborted(f"{label}：{problem}")


def snapshot(rec, out_png: Path, label: str) -> tuple[list, list, str]:
    """调 zed_snapshot.py 拍一张，期间收集机械臂状态。返回 (法兰位姿采样, 关节角采样, 工具的输出)。"""
    proc = subprocess.Popen([sys.executable, str(SNAPSHOT), "--output", str(out_png)]
                            + (["--force"] if out_png.exists() else []),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    flange, joints, count, end = [], [], -1, time.monotonic() + SNAPSHOT_TIMEOUT_S
    try:
        while proc.poll() is None:
            count, pkg = rec.wait_state(count, 0.5)
            problem = fm.state_problem(pkg)
            if problem:
                raise fm.MotionAborted(f"{label}：拍照时 {problem}")
            flange.append([float(v) for v in pkg.flange_cur_pos])
            joints.append([float(v) for v in pkg.jt_cur_pos])
            if time.monotonic() > end:
                raise fm.MotionAborted(f"{label}：{SNAPSHOT_TIMEOUT_S:.0f} s 没拍完")
    except BaseException:
        proc.kill()
        raise
    output = proc.stdout.read()
    if proc.returncode != 0 or not out_png.exists():
        raise fm.MotionAborted(f"{label}：zed_snapshot.py 失败（返回码 {proc.returncode}）：{output.strip()[-300:]}")
    return flange, joints, output


def write_poses_csv(path: Path, log: dict) -> None:
    """poses.csv（标定仓库的格式）每次都由 capture_log.json 整体重写；检查图不进表。"""
    shots = sorted((s for s in log["shots"] if s["shot"].startswith("images/")), key=lambda s: s["shot"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# handeye_capture.py：法兰（工具 0）在基坐标系（工件 0）下的位姿和关节角，"
                "取自拍照期间中间那个 20004 状态包\n")
        w = csv.writer(f)
        w.writerow(POSES_HEADER)
        for s in shots:
            w.writerow([s["shot"], *[f"{v:.3f}" for v in s["flange_mm_deg"]], *[f"{v:.3f}" for v in s["joints_deg"]]])


def cmd_capture(plan: dict, session: Path, resume: bool, assume_yes: bool) -> int:
    seq = sequence(plan)
    if not SNAPSHOT.exists():
        print(f"找不到 {SNAPSHOT}（标定仓库 tools/zed_snapshot.py）")
        return 1
    busy = subprocess.run(["pgrep", "-l", "-f", "|".join(CAMERA_PROGRAMS)], capture_output=True, text=True).stdout
    busy = [l for l in busy.splitlines() if "handeye_capture" not in l]
    if busy:
        print("相机正被占用，先关掉：\n  " + "\n  ".join(busy))
        return 1
    poses_csv = session / "poses.csv"
    if session.exists() and any(session.iterdir()) and not resume:
        print(f"{session} 已经有东西了；接着拍加 --resume，重新拍换一个 --session")
        return 1
    (session / "images").mkdir(parents=True, exist_ok=True)
    (session / "checks").mkdir(parents=True, exist_ok=True)
    log_path = session / "capture_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"plan": plan, "shots": []}
    done = {s["shot"] for s in log["shots"]}          # 拍完、检查过、记进 capture_log.json 的才算（图在但没记录的重拍）

    signal.signal(signal.SIGHUP, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    with tempfile.TemporaryDirectory() as tmp:
        rec, rpc, soft_lo, soft_hi, q_now = open_robot(tmp)
        try:
            print(f"机械臂现在：{np.round(q_now, 1).tolist()}（{fl.describe_config(q_now)}）")
            todo = [k for k, p in enumerate(seq) if p["shot"] not in done]
            if not todo:
                print("都拍完了")
                return 0
            first = todo[0]
            # 从现在的位姿出发复核（续拍时从第一张没拍的开始）
            targets, problems = check_sequence(rpc, plan, seq[first:], q_now, soft_lo, soft_hi, verbose=False)
            if problems:
                print("检查没通过，没有动：\n  " + "\n  ".join(problems))
                return 1
            print(f"\n要拍 {len(todo)} 张（共 {len(seq)} 张，已有 {len(done)} 张），存到 {session}。"
                  f"MoveJ 速度 {MOVEJ_VEL_PCT:.0f}%，关节速度超过 {MAX_JOINT_SPEED_DEG_S:.0f} °/s 就停。")
            for k in todo:
                p = seq[k]
                print(f"  {k + 1:2d}. 位姿 {p['index']:2d} → {p['shot']}：距离 {p['distance_m']:.2f} m，"
                      f"倾角 {p['tilt_deg']:.0f}°，滚转 {p['roll_deg']:.0f}°，关节 {np.round(targets[k - first], 1).tolist()}")
            print("\n人在机械臂旁边、手边有急停；标定板和台子不要碰。")
            if not assume_yes and input("输入 yes 开始：").strip() != "yes":
                print("没有动")
                return 1
            for k in todo:
                p, q_target = seq[k], targets[k - first]
                label = f"{k + 1}/{len(seq)} 位姿 {p['index']}"
                print(f"{label}：移动中……", end="", flush=True)
                motion = movej(rec, rpc, q_target, plan, label)
                settle(rec, label)
                print(f"到位（{motion['duration_s']:.1f} s，关节最快 {motion['peak_joint_deg_s']:.0f} °/s），拍照……",
                      end="", flush=True)
                flange, joints, _ = snapshot(rec, session / p["shot"], label)
                f_arr, j_arr = np.array(flange), np.array(joints)
                pos_span = float(np.ptp(f_arr[:, :3], axis=0).max())
                joint_span = float(np.ptp(j_arr, axis=0).max())
                if pos_span > STILL_MM or joint_span > STILL_DEG:
                    raise fm.MotionAborted(f"{label}：拍照期间机械臂在动（法兰变化 {pos_span:.3f} mm，"
                                           f"关节变化 {joint_span:.4f}°），这张作废")
                mid = len(flange) // 2
                log["shots"].append({"step": k + 1, "pose": p["index"], "shot": p["shot"], "time": time.time(),
                                     "flange_mm_deg": flange[mid], "joints_deg": joints[mid],
                                     "planned_joints_deg": [float(v) for v in q_target],
                                     "packets": len(flange), "flange_span_mm": pos_span,
                                     "joint_span_deg": joint_span, **motion})
                log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
                write_poses_csv(poses_csv, log)
                os.sync()
                print(f"好（{len(flange)} 包，法兰变化 {pos_span:.3f} mm）")
            print(f"\n拍完了：{session}")
            return 0
        except (fm.MotionAborted, KeyboardInterrupt) as exc:
            fm.emergency_stop(rs.ROBOT_IP)
            print(f"\n中止：{exc}。机械臂已发 StopMotion；接着拍加 --resume。")
            return 1
        finally:
            rec.stop(write=False)
            os.sync()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="空跑检查（不动机械臂）")
    p.add_argument("poses", type=Path)
    c = sub.add_parser("capture", help="采集")
    c.add_argument("poses", type=Path)
    c.add_argument("--session", type=Path, default=SESSION_DIR)
    c.add_argument("--resume", action="store_true", help="接着拍（跳过已经存在的图）")
    c.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args(argv)
    plan = json.loads(args.poses.read_text(encoding="utf-8"))
    if args.command == "plan":
        return cmd_plan(plan)
    return cmd_capture(plan, args.session.expanduser(), args.resume, args.yes)


if __name__ == "__main__":
    sys.exit(main())
