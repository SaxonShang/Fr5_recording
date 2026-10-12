"""fr5_motion.py 的离线测试（不连机械臂）。

    cd ~/ROBOT/fr5_record && python3 -B -m unittest test_fr5_motion
"""

import time
import types
import unittest

import numpy as np

import fr5_motion as fm

F_T_C = np.array([[0.999660953447201, 0.0007843776424874251, 0.02602619650942236, -0.058960616228020075],
                  [-0.0007566244143297974, 0.9999991346690702, -0.0010761879977619043, -0.07159469739308227],
                  [-0.026027018125954093, 0.001056131064239924, 0.999660681888933, 0.09927925870705787],
                  [0.0, 0.0, 0.0, 1.0]])
NOW = fm.fr5_to_matrix([484.963, -506.674, 490.858, -90.545, 0.0, -105.264]) @ F_T_C  # 10-10 的相机位姿
CFG = fm.MotionConfig(speed_mps=0.05, stroke_m=0.30, hold_start_s=5, hold_mid_s=3, hold_end_s=5,
                      wiggle_deg=10, wiggle_deg_s=20, wiggle_cycles=2, reposition_mps=0.05, sync_test_hold_s=10)
MODEL = fm.SpeedModel(mm_s_per_pct=1.0, wiggle_deg_s_per_pct=0.5, ramp_s=0.3, calibrated=True)


def _wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


class GeometryTest(unittest.TestCase):
    def test_fr5_round_trip(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            p = [*rng.uniform(-800, 800, 3), rng.uniform(-179, 179), rng.uniform(-89, 89), rng.uniform(-179, 179)]
            back = fm.matrix_to_fr5(fm.fr5_to_matrix(p))
            np.testing.assert_allclose(back[:3], p[:3], atol=1e-9)
            np.testing.assert_allclose(_wrap(np.subtract(back[3:], p[3:])), 0, atol=1e-9)

    def test_level_camera(self):
        level = fm.level_camera(NOW)
        r = level[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.0, places=12)
        self.assertAlmostEqual(r[2, 2], 0.0, places=12)               # 光轴水平
        np.testing.assert_allclose(r[:, 1], [0, 0, -1], atol=1e-12)    # 图像 y 竖直向下
        np.testing.assert_allclose(level[:3, 3], NOW[:3, 3])           # 光心不动
        self.assertLess(fm.rotation_deg(level, NOW), 1.0)              # 现在的相机本来就接近水平

    def test_yaw_keeps_level_and_centre(self):
        level = fm.level_camera(NOW)
        for deg in (-15, 10):
            c = fm.yaw_about_vertical(level, deg)
            self.assertAlmostEqual(c[2, 2], 0.0, places=12)
            np.testing.assert_allclose(c[:3, 1], [0, 0, -1], atol=1e-12)
            np.testing.assert_allclose(c[:3, 3], level[:3, 3])
            self.assertAlmostEqual(fm.rotation_deg(c, level), abs(deg), places=9)

    def test_blur(self):
        self.assertAlmostEqual(fm.blur_px(0.05, 10000, 1.0), 0.367, places=3)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.centre = fm.level_camera(NOW)

    def test_each_direction_is_pure_translation(self):
        expected = {"forward": self.centre[:3, 2], "backward": -self.centre[:3, 2],
                    "right": self.centre[:3, 0], "left": -self.centre[:3, 0],
                    "up": np.array([0, 0, 1.0]), "down": np.array([0, 0, -1.0])}
        for name, axis in expected.items():
            segs = fm.plan_recording(name, self.centre, F_T_C, CFG)
            by = {s.name: s for s in segs}
            start, end = by["goto_start"].camera, by["translate"].camera
            np.testing.assert_allclose(end[:3, 3] - start[:3, 3], axis * CFG.stroke_m, atol=1e-12)
            np.testing.assert_allclose((start[:3, 3] + end[:3, 3]) / 2, self.centre[:3, 3], atol=1e-12)
            self.assertLess(fm.rotation_deg(start, self.centre), 1e-9)   # 平移段姿态不变
            self.assertLess(fm.rotation_deg(end, self.centre), 1e-9)
            self.assertFalse(by["goto_start"].recorded)
            self.assertAlmostEqual(by["translate"].speed_mm_s, 50.0)
            wiggles = [s for s in segs if s.name.startswith("wiggle")]
            self.assertEqual(len(wiggles), 2 * (2 * CFG.wiggle_cycles + 1))
            # 同步摆动结束时回到原朝向
            self.assertLess(fm.rotation_deg(by["wiggle_end_5"].camera, end), 1e-9)

    def test_speed_percent_from_model(self):
        flange0 = self.centre @ np.linalg.inv(F_T_C)            # 已经在起点
        segs = fm.plan_sync_test(self.centre, F_T_C, CFG)
        fm.fill_geometry(segs, flange0)
        self.assertEqual(fm.apply_speed(segs, MODEL, 0.5), [])
        wiggles = [s for s in segs if s.name.startswith("wiggle")]
        for s in wiggles:                                       # 20 °/s × 0.5 ÷ 0.5 °/s/%
            self.assertAlmostEqual(s.percent, 20.0)
            self.assertAlmostEqual(s.expect_deg_s, 10.0)
            self.assertAlmostEqual(s.duration_s, s.angle_deg / 10.0 + 0.3)
            self.assertAlmostEqual(s.expect_mm_s, s.distance_mm / (s.angle_deg / 10.0))
            self.assertTrue(s.speed_known)
        self.assertAlmostEqual(wiggles[0].angle_deg, 10.0, places=9)
        self.assertAlmostEqual(wiggles[1].angle_deg, 20.0, places=9)
        self.assertAlmostEqual(segs[0].distance_mm, 0.0, places=9)  # goto_start 原地
        self.assertAlmostEqual(segs[0].duration_s, 0.3)

        rec = fm.plan_recording("right", self.centre, F_T_C, CFG)
        fm.fill_geometry(rec, flange0)
        self.assertEqual(fm.apply_speed(rec, MODEL, 1.0), [])
        tr = next(s for s in rec if s.name == "translate")
        self.assertAlmostEqual(tr.percent, 50.0)
        self.assertAlmostEqual(tr.expect_mm_s, 50.0)
        self.assertAlmostEqual(tr.distance_mm, 300.0, places=6)
        self.assertAlmostEqual(tr.duration_s, 300.0 / 50.0 + 0.3)

    def test_speed_percent_limits_and_unknown_model(self):
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        rec = fm.plan_recording("up", self.centre, F_T_C, CFG)
        fm.fill_geometry(rec, flange0)
        slow = fm.SpeedModel(0.1, 0.05, calibrated=True)        # 50 mm/s 要 500%
        problems = fm.apply_speed(rec, slow, 1.0)
        self.assertTrue(any(m.startswith("translate:") and "500.00%" in m for m in problems), problems)
        self.assertEqual(fm.apply_speed(rec, None, 1.0), [])    # 没有模型：只按设计速度估时长
        tr = next(s for s in rec if s.name == "translate")
        self.assertIsNone(tr.percent)
        self.assertAlmostEqual(tr.duration_s, 300.0 / 50.0 + fm.DEFAULT_RAMP_S)
        self.assertFalse(tr.speed_known)

    def test_speed_test_plan(self):
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        segs = fm.plan_speed_test(self.centre, F_T_C, CFG, 100.0, 5.0)
        by = {s.name: s for s in segs}
        fm.fill_geometry(segs, flange0)
        self.assertEqual(fm.apply_speed(segs, fm.SpeedModel(10.0), 1.0), [])   # 未标定的保守模型
        self.assertAlmostEqual(by["test_up"].distance_mm, 100.0, places=6)
        self.assertLess(by["test_up"].angle_deg, 1e-9)
        self.assertAlmostEqual(by["test_up"].percent, 5.0)
        self.assertAlmostEqual(by["test_up"].expect_mm_s, 50.0)   # 上限
        self.assertFalse(by["test_up"].speed_known)
        self.assertAlmostEqual(by["test_yaw"].angle_deg, 10.0, places=9)
        self.assertAlmostEqual(by["test_yaw"].percent, 5.0)
        self.assertEqual(by["test_yaw"].duration_s, 0.0)          # 角速度未知
        self.assertIsNone(by["test_yaw_back"].percent)            # 等 test_yaw 实测
        np.testing.assert_allclose(by["test_yaw_back"].camera, self.centre)


def _trapezoid(distance, v, a, dt=0.01):
    """梯形速度曲线下的路程，前后各静止 0.3 s。"""
    ta = v / a
    tc = distance / v - ta
    total = 2 * ta + tc
    t = np.arange(-0.3, total + 0.3, dt)
    tt = np.clip(t, 0, total)
    s = np.where(tt < ta, 0.5 * a * tt ** 2,
                 np.where(tt < ta + tc, 0.5 * a * ta ** 2 + v * (tt - ta),
                          distance - 0.5 * a * (total - tt) ** 2))
    return t, s, total


class MeasureTest(unittest.TestCase):
    def test_motion_stats_linear_and_day_wrap(self):
        t, s, total = _trapezoid(100.0, 50.0, 200.0)
        ms0 = fm.DAY_MS - 1000                                   # 中途跨过控制器的午夜
        samples = [((ms0 + round(ti * 1000)) % fm.DAY_MS, [400.0 + si, -300.0, 500.0, -90.0, 0.0, -100.0])
                   for ti, si in zip(t, s)]
        m = fm.motion_stats(samples)
        self.assertAlmostEqual(m["path_mm"], 100.0, places=6)
        self.assertAlmostEqual(m["linear_mm_s"]["mean"], 50.0, delta=0.5)
        self.assertLess(m["linear_mm_s"]["ripple"], 0.01)
        self.assertIsNone(m["angular_deg_s"])
        onset = np.sqrt(2 * fm.ONSET_MM / 200.0)                # 离开起点 0.1 mm 的时刻
        self.assertAlmostEqual(m["duration_s"], total - 2 * onset, delta=0.02)
        self.assertAlmostEqual(m["ramp_s"], total - 2 * onset - (100.0 - 2 * fm.ONSET_MM) / 50.0, delta=0.02)
        self.assertGreater(m["ramp_s"], 0.15)                   # 真实 v/a = 0.25 s，起点阈值使它略小
        self.assertAlmostEqual(m["max_gap_s"], 0.01, places=6)

    def test_motion_stats_ignores_rest_noise(self):
        rng = np.random.default_rng(1)
        t, s, total = _trapezoid(50.0, 25.0, 1e6)              # 几乎瞬间加速
        t = np.r_[np.arange(-3.0, t[0] - 0.005, 0.01), t]               # 前面多静止 3 s
        s = np.r_[np.zeros(len(t) - len(s)), s]
        samples = [(round(ti * 1000) + 10_000, [400.0 + si + rng.normal(0, 0.005), -300.0, 500.0, -90.0, 0.0, -100.0])
                   for ti, si in zip(t, s)]
        m = fm.motion_stats(samples)
        self.assertAlmostEqual(m["duration_s"], total, delta=0.03)
        self.assertLess(m["ramp_s"], 0.03)
        self.assertAlmostEqual(m["linear_mm_s"]["mean"], 25.0, delta=0.3)

    def test_motion_stats_rotation(self):
        t, s, _ = _trapezoid(10.0, 20.0, 100.0)                 # 转 10°，20 °/s
        samples = [(round(ti * 1000) + 5000, [400.0, -300.0, 500.0, -90.0, 0.0, -100.0 + si]) for ti, si in zip(t, s)]
        m = fm.motion_stats(samples)
        self.assertAlmostEqual(m["angle_deg"], 10.0, places=6)
        self.assertAlmostEqual(m["angular_deg_s"]["mean"], 20.0, delta=0.3)
        self.assertIsNone(m["linear_mm_s"])

    def test_vectorised_angles_match_and_are_fast(self):
        rng = np.random.default_rng(3)
        poses = [[*rng.uniform(-500, 500, 3), *rng.uniform(-170, 170, 3)] for _ in range(50)]
        mats = [fm.fr5_to_matrix(p) for p in poses]
        rots = np.array([m[:3, :3] for m in mats])
        expected = [fm.rotation_deg(mats[7], m) for m in mats]
        np.testing.assert_allclose(fm.angles_to(rots, rots[7]), expected, atol=1e-6)
        samples = [(i * 10, [400.0, -300.0, 500.0 + 0.05 * i, -90.0, 0.0, -100.0 + 0.001 * i]) for i in range(2000)]
        t0 = time.perf_counter()
        fm.motion_stats(samples)
        self.assertLess(time.perf_counter() - t0, 0.2)          # 原来逐个样本算要约 1 s（Box 上）

    def test_recent_speed(self):
        samples = [(i * 10, [i * 0.5, 0, 0, 0, 0, 0]) for i in range(3)]   # 50 mm/s，但还不到 50 ms
        j, v = fm.recent_speed(samples, 0)
        self.assertIsNone(v)
        samples += [(i * 10, [i * 0.5, 0, 0, 0, 0, 0]) for i in range(3, 20)]
        j, v = fm.recent_speed(samples, j)
        self.assertAlmostEqual(v, 50.0)
        self.assertEqual(samples[-1][0] - samples[j][0], 50)


class _FakeArm:
    """假的状态录制器 + RPC：每次 wait_state 前进 10 ms，MoveL 后按 速度百分比 × mm_s_per_pct 匀速直线走向目标。"""

    def __init__(self, start6, mm_s_per_pct, lose_state_after=None):
        self.pos = np.array(start6, dtype=float)
        self.k = mm_s_per_pct
        self.target, self.speed, self.calls, self.ms, self.count = None, 0.0, [], 1000, 0
        self.lose_state_after = lose_state_after   # MoveL 之后再给这么多包就“断线”
        self.frozen = False                        # True：断线，不再有新包
        self.pkg = None
        self.latest_monotonic = None               # 同 StateRecorder：最新一包的到达时刻

    def wait_state(self, last_count, timeout):
        if self.calls and self.lose_state_after is not None:
            if self.lose_state_after <= 0:
                self.frozen = True
            self.lose_state_after -= 1
        if self.frozen:
            time.sleep(min(timeout, 0.05))            # 断线：没有新包，返回上一个
            return self.count, self.pkg
        self.latest_monotonic = time.monotonic()
        self.ms += 10
        state = 1
        if self.target is not None:
            step = self.target[:3] - self.pos[:3]
            d = np.linalg.norm(step)
            move = self.speed * 0.01
            if d <= move:
                self.pos = self.target.copy()
                self.target = None
            else:
                self.pos[:3] += step / d * move
                state = 2
        self.count += 1
        self.pkg = types.SimpleNamespace(
            flange_cur_pos=list(self.pos), robot_state=state, EmergencyStop=0, main_code=0, sub_code=0,
            safety_stop0_state=0, safety_stop1_state=0, rbtEnableState=1, tool=0, user=0,
            hour=self.ms // 3_600_000, minute=self.ms // 60_000 % 60, second=self.ms // 1000 % 60,
            millisecond=self.ms % 1000)
        return self.count, self.pkg

    def MoveL(self, **kw):  # noqa: N802 - SDK 的名字
        self.calls.append(kw)
        self.target = np.array(kw["desc_pos"], dtype=float)
        self.speed = kw["vel"] * self.k
        return 0


class RunnerTest(unittest.TestCase):
    START = [400.0, -300.0, 500.0, -90.0, 0.0, -100.0]

    def _segment(self, distance_mm, percent=None):
        target = fm.fr5_to_matrix(self.START)
        target[:3, 3] += [0.0, 0.0, distance_mm / 1000.0]
        seg = fm.Segment("translate", "move", flange=fm.matrix_to_fr5(target), joints=[0.0] * 6, speed_mm_s=20.0)
        fm.fill_geometry([seg], fm.fr5_to_matrix(self.START))
        fm.apply_speed([seg], MODEL, 1.0)                       # 1 mm/s/% → 20%
        if percent is not None:
            seg.percent = percent
        return seg

    def test_move_uses_percent_mode_and_measures(self):
        arm = _FakeArm(self.START, mm_s_per_pct=1.0)
        seg = self._segment(60.0)
        fm.MotionRunner(arm, arm, accel_percent=50).run(seg)
        call = arm.calls[0]
        self.assertEqual((call["velAccParamMode"], call["vel"], call["ovl"], call["oacc"], call["blendR"]),
                         (0, 20.0, 100.0, 50, 0.0))
        np.testing.assert_allclose(arm.pos, seg.flange, atol=1e-9)
        self.assertAlmostEqual(seg.log["measured"]["linear_mm_s"]["mean"], 20.0, delta=0.5)
        self.assertIn("arrived_box_ns", seg.log)

    def test_overspeed_aborts(self):
        arm = _FakeArm(self.START, mm_s_per_pct=3.0)             # 标定失效：实际快 3 倍
        seg = self._segment(60.0)
        with self.assertRaisesRegex(fm.MotionAborted, "超过上限"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        self.assertIsNotNone(arm.target)                         # 还在半路就中止了

    def test_state_loss_aborts(self):
        arm = _FakeArm(self.START, mm_s_per_pct=1.0, lose_state_after=50)   # 走 0.5 s 后断线
        seg = self._segment(60.0)
        t0 = time.monotonic()
        with self.assertRaisesRegex(fm.MotionAborted, "没收到机械臂状态"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        self.assertLess(time.monotonic() - t0, fm.STATE_LOST_S + 1.0)
        self.assertIn("measured", seg.log)                       # 中止时也留下已测到的部分

    def test_stale_state_blocks_movel(self):
        """断线前到达、但还没被取走的包：计数是新的，内容已过时，不能凭它发运动指令。"""
        arm = _FakeArm(self.START, mm_s_per_pct=1.0)
        runner = fm.MotionRunner(arm, arm, accel_percent=100)
        runner.latest()
        arm.wait_state(arm.count, 0.0)                           # 又到了一包，runner 还没取
        arm.frozen = True
        arm.latest_monotonic -= 1.0                              # 这包其实是 1 s 前到的
        with self.assertRaisesRegex(fm.MotionAborted, "没有发运动指令"):
            runner.run(self._segment(10.0))
        self.assertEqual(arm.calls, [])

    def test_invalid_percent_refused(self):
        arm = _FakeArm(self.START, mm_s_per_pct=1.0)
        for bad in (None, 0.0, 150.0):
            seg = self._segment(10.0)
            seg.percent = bad
            with self.assertRaises(fm.MotionAborted):
                fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        self.assertEqual(arm.calls, [])


class ParametersTest(unittest.TestCase):
    def test_defaults_valid_and_blur_limit_enforced(self):
        try:
            import run_session as rs
        except ImportError as exc:  # 笔记本上没装 fairino 时跳过
            self.skipTest(str(exc))
        self.assertEqual(rs.parameter_problems(), [])
        old = rs.SPEED_MPS
        try:
            rs.SPEED_MPS = 0.2
            self.assertTrue(any("超出允许范围" in m for m in rs.parameter_problems()))
            rs.RANGES["SPEED_MPS"] = (0.02, 0.5)
            self.assertTrue(any("模糊" in m for m in rs.parameter_problems()))
        finally:
            rs.SPEED_MPS = old
            rs.RANGES["SPEED_MPS"] = (0.02, 0.10)


if __name__ == "__main__":
    unittest.main()
