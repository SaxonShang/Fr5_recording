#!/usr/bin/env python3
"""最普通的 SDK 用法测试：和官方例子 TestMotionCommand.py 一样，只用 Robot.RPC 和 MoveJ，不经过本目录的任何封装。

MoveJ 的目标就是当前关节角（零行程），机械臂不会移动；关节变化超过 0.05° 就 StopMotion。
运行前不要开着 run_session.py / fr5_record.py（它们会占用 20004 端口）。

    python3 plain_sdk_test.py
"""

import time

from fairino import Robot

robot = Robot.RPC('192.168.58.2')
try:
    time.sleep(1.0)
    error, joints = robot.GetActualJointPosDegree(0)
    print("当前关节:", error, [round(v, 3) for v in joints])
    if error != 0 or not any(joints):
        raise SystemExit("读不到关节角，不发指令")
    if input("急停在手边后输入 yes，发送一条零行程 MoveJ：").strip() == "yes":
        error = robot.MoveJ(joints, tool=0, user=0, vel=10)
        print("MoveJ return", error)
        time.sleep(1.0)
        _, after = robot.GetActualJointPosDegree(0)
        change = max(abs(a - b) for a, b in zip(joints, after))
        print(f"关节最大变化 {change:.4f}°")
        if change > 0.05:
            robot.StopMotion()
            print("关节有变化，已发 StopMotion")
    else:
        print("没有确认，没有发指令")
except BaseException as exc:  # noqa: BLE001 - 包括 Ctrl+C
    robot.StopMotion()
    print(f"中止：{type(exc).__name__}: {exc}")
finally:
    robot.CloseRPC()
