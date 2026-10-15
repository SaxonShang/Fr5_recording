"""handeye_capture.py 的离线测试（假状态源、假 RPC，不连机械臂、不开相机）。

    python -B -m unittest test_handeye_capture
"""

import csv
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

import fr5_motion as fm

try:
    import handeye_capture as hc
    SKIP = None
except ImportError as exc:  # 笔记本上没装 fairino 时跳过（Box 上跑）
    hc, SKIP = None, f"需要 FAIRINO SDK：{exc}"


def pkg(q, state=1):
    return types.SimpleNamespace(jt_cur_pos=list(q), flange_cur_pos=[*hc.fk_position_mm(q), 0.0, 0.0, 0.0],
                                 robot_state=state, EmergencyStop=0, main_code=0, sub_code=0, safety_stop0_state=0,
                                 safety_stop1_state=0, rbtEnableState=1, tool=0, user=0)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


class FakeRecorder:
    """每次 wait_state 给出 joints 里的下一包，时钟走 10 ms；给完以后一直给最后一包。"""

    def __init__(self, joints, clock, states=None):
        self.joints, self.clock, self.i = joints, clock, 0
        self.states = states or [1] + [2] * (len(joints) - 2) + [1]
        self.latest_monotonic = clock.t

    def wait_state(self, count, timeout):
        self.clock.t += 0.01
        self.latest_monotonic = self.clock.t
        k = min(self.i, len(self.joints) - 1)
        self.i += 1
        return self.i, pkg(self.joints[k], self.states[min(k, len(self.states) - 1)])


class FakeRPC:
    def __init__(self):
        self.calls = []

    def MoveJ(self, **kw):
        self.calls.append(kw)
        return 0


START = np.array([158.5, -41.1, 52.9, -205.2, -82.5, 1.5])
TARGET = np.array([164.3, -40.8, 71.5, -213.5, -85.9, -19.8])
PLAN = {"envelope": {"joint_min_deg": [104.0, -115.4, 23.3, -250.1, -92.4, -77.0],
                     "joint_max_deg": [170.0, -24.5, 148.8, -98.6, -12.6, 81.2],
                     "flange_min_mm": [-2000.0, -2000.0, -2000.0], "flange_max_mm": [2000.0, 2000.0, 2000.0]}}


def ramp(q0, q1, n, deg_s=None):
    """q0 → q1 的 n 包（每包 10 ms），最后再静止几包。"""
    path = [q0 + (q1 - q0) * s for s in np.linspace(0.0, 1.0, n)]
    return path + [q1] * 5


@unittest.skipIf(SKIP, SKIP)
class MoveJTest(unittest.TestCase):
    def run_movej(self, joints, states=None, target=TARGET):
        clock = Clock()
        rec, rpc = FakeRecorder(joints, clock, states), FakeRPC()
        fake_time = types.SimpleNamespace(monotonic=clock.monotonic, time=lambda: 0.0)
        with unittest.mock.patch.object(hc, "time", fake_time), \
                unittest.mock.patch.object(fm, "emergency_stop") as stop, \
                unittest.mock.patch.object(fm.BlockingCall, "done", lambda self, timeout=0.0: True), \
                unittest.mock.patch.object(fm.BlockingCall, "problem", lambda self: None):
            try:
                result = hc.movej(rec, rpc, target, PLAN, "测试")
                return result, stop.call_count, rpc
            except fm.MotionAborted as exc:
                return exc, stop.call_count, rpc

    def test_arrives(self):
        # 最大移动 21.3°，2 s（200 包）走完：约 11 °/s
        result, stops, rpc = self.run_movej([START] + ramp(START, TARGET, 200))
        self.assertIsInstance(result, dict)
        self.assertEqual(stops, 0)
        self.assertLess(result["peak_joint_deg_s"], 15.0)
        self.assertEqual(rpc.calls[0]["vel"], hc.MOVEJ_VEL_PCT)
        np.testing.assert_allclose(rpc.calls[0]["joint_pos"], TARGET)

    def test_overspeed_stops(self):
        result, stops, _ = self.run_movej([START] + ramp(START, TARGET, 30))       # 0.3 s 走 21°：约 70 °/s
        self.assertIsInstance(result, fm.MotionAborted)
        self.assertIn("关节速度", str(result))
        self.assertEqual(stops, 1)

    def test_out_of_envelope_stops(self):
        bad = TARGET.copy()
        bad[0] = 175.0                                                           # J1 上限 170 + 2° 容差
        result, stops, _ = self.run_movej([START] + ramp(START, bad, 400), target=bad)
        self.assertIsInstance(result, fm.MotionAborted)
        self.assertIn("超出允许范围", str(result))
        self.assertEqual(stops, 1)

    def test_stall_stops(self):
        joints = [START] * 700                                                   # 一直不动：robot_state 2、不到位
        result, stops, _ = self.run_movej(joints, states=[1] + [2] * 699)
        self.assertIsInstance(result, fm.MotionAborted)
        self.assertIn("没有移动", str(result))
        self.assertEqual(stops, 1)

    def test_refuses_when_not_stopped(self):
        result, stops, rpc = self.run_movej([START] * 3, states=[2, 2, 2])
        self.assertIsInstance(result, fm.MotionAborted)
        self.assertIn("没有发运动指令", str(result))
        self.assertEqual(rpc.calls, [])


@unittest.skipIf(SKIP, SKIP)
class SequenceTest(unittest.TestCase):
    def test_order_and_checks(self):
        plan = {"check_pose": 1, "poses": [{"index": i, "role": "pose"} for i in range(1, 26)]
                + [{"index": 26, "role": "reserve"}]}
        seq = hc.sequence(plan)
        shots = [p["shot"] for p in seq]
        self.assertEqual(len(seq), 27)
        self.assertEqual(shots[0], "images/0001.png")
        self.assertEqual(shots[12], "checks/mid_0001.png")
        self.assertEqual(shots[-1], "checks/end_0001.png")
        self.assertEqual(sum(s.startswith("images/") for s in shots), 25)

    def test_poses_csv_from_log(self):
        log = {"shots": [{"shot": "images/0002.png", "flange_mm_deg": [1, 2, 3, 4, 5, 6], "joints_deg": [1] * 6},
                         {"shot": "checks/mid_0001.png", "flange_mm_deg": [0] * 6, "joints_deg": [0] * 6},
                         {"shot": "images/0001.png", "flange_mm_deg": [7, 8, 9, 10, 11, 12], "joints_deg": [2] * 6}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.csv"
            hc.write_poses_csv(path, log)
            with open(path, encoding="utf-8") as f:
                rows = [r for r in csv.reader(l for l in f if not l.startswith("#"))]
        self.assertEqual(rows[0], hc.POSES_HEADER)
        self.assertEqual([r[0] for r in rows[1:]], ["images/0001.png", "images/0002.png"])
        self.assertEqual(rows[1][1:4], ["7.000", "8.000", "9.000"])


if __name__ == "__main__":
    unittest.main()
