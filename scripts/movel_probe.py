#!/usr/bin/env python3
"""MoveL 返回错误码诊断：每条指令的目标都是“当前位姿”（零行程），机械臂不会移动。

逐个尝试参数组合，记录控制器的返回码，找出被拒绝的是哪种参数。
任何一条指令后关节变化超过 0.05° 就立即 StopMotion 并退出。

    python3 movel_probe.py                  # 基本组合（百分比速度 / 物理速度 / MoveJ）
    python3 movel_probe.py --set physical   # 物理速度模式下改速度、加速度、ovl，找出允许的取值
"""

import argparse
import sys
import tempfile
import time

import numpy as np

import fr5_motion as fm
from fr5_record import StateRecorder
from run_session import ROBOT_IP

MAX_JOINT_CHANGE_DEG = 0.05

PHYSICAL = dict(blendR=-1.0, velAccParamMode=1)
VARIANT_SETS = {}
VARIANT_SETS["physical"] = [
    ("P1 物理 50 mm/s、oacc 1000，ovl 100", "L", dict(vel=50.0, oacc=1000.0, ovl=100.0, **PHYSICAL)),
    ("P2 物理 50 mm/s、oacc 2000，ovl 100", "L", dict(vel=50.0, oacc=2000.0, ovl=100.0, **PHYSICAL)),
    ("P3 物理 200 mm/s、oacc 1000，ovl 100", "L", dict(vel=200.0, oacc=1000.0, ovl=100.0, **PHYSICAL)),
    ("P4 物理 12.5 mm/s、oacc 1000，ovl 100", "L", dict(vel=12.5, oacc=1000.0, ovl=100.0, **PHYSICAL)),
    ("P5 物理 50 mm/s、oacc 1000，ovl 25", "L", dict(vel=50.0, oacc=1000.0, ovl=25.0, **PHYSICAL)),
    ("P6 物理 50 mm/s、oacc 100，ovl 100", "L", dict(vel=50.0, oacc=100.0, ovl=100.0, **PHYSICAL)),
    ("P7 物理 50 mm/s、oacc 1000，ovl 100，非阻塞", "L",
     dict(vel=50.0, oacc=1000.0, ovl=100.0, blendR=0.0, velAccParamMode=1)),
]
VARIANT_SETS["basic"] = [
    ("A MoveL 百分比速度 20%，阻塞（SDK 默认）", "L", dict(vel=20.0, ovl=100.0, blendR=-1.0)),
    ("B MoveL 同 A，但由控制器自己算逆解", "L", dict(vel=20.0, ovl=100.0, blendR=-1.0, no_joints=True)),
    ("C MoveL 百分比速度 20%，非阻塞 blendR=0", "L", dict(vel=20.0, ovl=100.0, blendR=0.0)),
    ("D MoveL 物理速度 50 mm/s、oacc 200，ovl 100，阻塞", "L",
     dict(vel=50.0, ovl=100.0, blendR=-1.0, oacc=200.0, velAccParamMode=1)),
    ("E MoveL 物理速度 50 mm/s、oacc 200，ovl 25，阻塞", "L",
     dict(vel=50.0, ovl=25.0, blendR=-1.0, oacc=200.0, velAccParamMode=1)),
    ("F MoveL 物理速度 50 mm/s、oacc 200，ovl 25，非阻塞（run_session 现在的用法）", "L",
     dict(vel=50.0, ovl=25.0, blendR=0.0, oacc=200.0, velAccParamMode=1)),
    ("G MoveJ 百分比速度 20%，阻塞", "J", dict(vel=20.0, ovl=100.0)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(VARIANT_SETS), default="basic", help="要试的参数组合")
    variants = VARIANT_SETS[parser.parse_args().set]
    with tempfile.TemporaryDirectory() as tmp:
        rec = StateRecorder(tmp, ROBOT_IP)
        rec.start()
        try:
            rpc = rec.open_rpc()
            _, pkg = rec.wait_state(-1, 1.0)
            problem = fm.state_problem(pkg)
            if problem or pkg.robot_state != 1:
                print(f"状态不对，不发指令：{problem or f'robot_state={pkg.robot_state}'}")
                return 1
            print(f"当前模式：{'手动' if pkg.robot_mode else '自动'}；关节 {np.round(list(pkg.jt_cur_pos), 3).tolist()}")
            print("将依次发送以下零行程指令（目标 = 当前位姿，机械臂不会移动）：")
            for name, _, _ in variants:
                print(f"  {name}")
            try:
                if input("急停在手边后输入 yes 开始：").strip() != "yes":
                    print("没有确认，没有发指令")
                    return 1
            except EOFError:
                return 1
            results = []
            for name, kind, params in variants:
                _, pkg = rec.wait_state(-1, 1.0)
                joints0 = list(pkg.jt_cur_pos)
                flange0 = list(pkg.flange_cur_pos)
                params = dict(params)
                no_joints = params.pop("no_joints", False)
                if kind == "L":
                    ret = rpc.MoveL(desc_pos=flange0, tool=0, user=0,
                                    joint_pos=[0.0] * 6 if no_joints else joints0, **params)
                else:
                    ret = rpc.MoveJ(joint_pos=joints0, tool=0, user=0, desc_pos=flange0, **params)
                time.sleep(1.0)
                _, pkg = rec.wait_state(-1, 1.0)
                change = float(np.max(np.abs(np.subtract(list(pkg.jt_cur_pos), joints0))))
                results.append((name, ret, change, pkg.robot_state, (pkg.main_code, pkg.sub_code)))
                print(f"{name}\n    返回码 {ret}，关节最大变化 {change:.4f}°，robot_state={pkg.robot_state}，"
                      f"报警={pkg.main_code}/{pkg.sub_code}")
                if change > MAX_JOINT_CHANGE_DEG or pkg.robot_state != 1:
                    fm.emergency_stop(ROBOT_IP)
                    print("关节有变化或机械臂没有停下，已发 StopMotion，停止诊断")
                    return 1
            print("\n汇总：" + "；".join(f"{n.split()[0]}={r}" for n, r, *_ in results))
            return 0
        except BaseException as exc:  # noqa: BLE001 - 包括 Ctrl+C
            fm.emergency_stop(ROBOT_IP)
            print(f"中止：{type(exc).__name__}: {exc}")
            return 1
        finally:
            rec.stop(write=False)


if __name__ == "__main__":
    sys.exit(main())
