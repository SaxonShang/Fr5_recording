"""holdout1：路径表、多段规划、组合段速度和保护、路径点修正、只用同步摆动估 Δt、开工检查和条件清单。
不连机械臂，用解析逆解（fr5_kin）代替控制器。没有 FAIRINO SDK 时用桩代替 fr5_record。

    python3 -B -m unittest test_holdout1
"""

import argparse
import contextlib
import csv
import dataclasses
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import fr5_groundtruth as gt
import fr5_kin
import fr5_motion as fm
import holdout1_paths as hp
import make_conditions as mc
import startup_checks as sc
import test_fr5_groundtruth as tg
from test_fr5_limits import _FakeRecorder
from test_fr5_motion import _FakeArm

HERE = Path(__file__).resolve().parent
F_T_C, _ = gt.load_handeye(HERE / "handeye_s003_result.json")
MODEL = fm.SpeedModel(5.96, 2.152, 0.459, True, 1084.0, 156.8)
CFG = fm.MotionConfig(0.5, 5.0, 3.0, 5.0, 10.0, 20.0, 2, 0.05, rot_deg_s=50.0, translation_wiggle_side=1)
LIMITS = ([-173.0, -263.0, -158.0, -263.0, -173.0, -173.0], [173.0, 83.0, 158.0, 83.0, 173.0, 173.0])
SOFT = [-175.0, 175.0, -265.0, 85.0, -160.0, 160.0, -265.0, 85.0, -175.0, 175.0, -175.0, 175.0]
EMPTY_WINDOW = {"viewpoint_mm": (600.0, -200.0, 520.0), "regions": [], "distance_m": 7.0}


def import_run_session():
    try:
        import run_session
        return run_session
    except ImportError:                                    # 笔记本上没有 FAIRINO SDK
        stub = types.SimpleNamespace(StateRecorder=object)
        with mock.patch.dict(sys.modules, {"fr5_record": stub}):
            sys.modules.pop("run_session", None)
            import run_session
        return run_session


def start_of(spec) -> tuple[np.ndarray, list]:
    """(起点法兰位姿, 起点关节角)。"""
    flange = fm.waypoint_pose(np.array(spec.reference_mm) / 1000.0, spec.start, spec.frame_yaw_deg) @ np.linalg.inv(F_T_C)
    return flange, fm.inverse_kin(fr5_kin.OfflineRPC(), flange, spec.seed_in_base)[1]


class PathTableTest(unittest.TestCase):
    def test_leg_counts(self):
        expected = {"roll_bidir": 3, "trans_level_first": 10, "trans_tilt_first": 10, "s_path": 12, "eight": 24,
                    "accel_static_a": 9, "accel_static_b": 9, "accel_rate_slow_first": 18, "accel_rate_fast_first": 18}
        self.assertEqual({k: len(v.legs) for k, v in hp.PATHS.items()}, expected)

    def test_labels_unique_and_translation_orders_share_them(self):
        for name, spec in hp.PATHS.items():
            self.assertEqual(len(set(spec.labels)), len(spec.labels), name)
        level, tilt = hp.PATHS["trans_level_first"], hp.PATHS["trans_tilt_first"]
        self.assertEqual(set(level.labels), set(tilt.labels))
        self.assertEqual(level.correction_key, tilt.correction_key)
        by_label = {leg.label: leg.target for leg in level.legs}
        self.assertEqual({leg.label: leg.target for leg in tilt.legs}, by_label)   # 同一个名字是同一个位姿

    def test_sign_conventions(self):
        ref = np.array([0.6, 0.0, 0.5])
        base = fm.waypoint_pose(ref, hp.Pose())
        np.testing.assert_allclose(base[:3, 2], [1, 0, 0], atol=1e-12)              # 朝 +X
        np.testing.assert_allclose(base[:3, 1], [0, 0, -1], atol=1e-12)             # 图像 y 向下
        np.testing.assert_allclose(base[:3, 0], [0, -1, 0], atol=1e-12)             # 图像 x 向右（基坐标 −y）
        self.assertGreater(fm.waypoint_pose(ref, hp.Pose(heading=30.0))[1, 2], 0.0)  # 朝向向左：光轴偏 +y
        self.assertGreater(fm.waypoint_pose(ref, hp.Pose(pitch=20.0))[2, 2], 0.0)    # 俯仰向上：光轴偏 +z
        top = -fm.waypoint_pose(ref, hp.Pose(roll=30.0))[:3, 1]                      # 画面顶部方向
        self.assertLess(top[1], 0.0)                                                 # 横滚为正：顶部向右（−y）
        np.testing.assert_allclose(fm.waypoint_pose(ref, hp.Pose(x=100.0, z=-50.0))[:3, 3], ref + [0.1, 0, -0.05], atol=1e-12)
        turned = fm.waypoint_pose(ref, hp.Pose(), 90.0)                               # 路径坐标系转 90°：前方变成基坐标 +y
        np.testing.assert_allclose(turned[:3, 2], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(turned[:3, 3], [0.0, 0.6, 0.5], atol=1e-12)
        np.testing.assert_allclose(turned[:3, 1], [0, 0, -1], atol=1e-12)             # 仍然水平

    def test_s_and_eight_tables(self):
        s = hp.PATHS["s_path"]
        self.assertEqual([leg.label[3:] for leg in s.legs], ["P0", "P1", "P2", "P3", "P4", "P5", "P4", "P3", "P2", "P1", "P0", "S"])
        self.assertEqual(s.legs[-1].target, s.start)
        self.assertEqual((s.start.heading, s.start.roll), (20.0, 0.0))
        p3 = s.legs[3].target
        self.assertEqual((p3.x, p3.y, p3.heading, p3.roll), (100.0, 175.0, 40.0, -20.0))
        e = hp.PATHS["eight"]
        self.assertEqual([leg.label[3:] for leg in e.legs[:8]], ["Q1", "Q2", "Q3", "Q0", "Q5", "Q6", "Q7", "Q0"])
        self.assertTrue(all(leg.max_rate_deg_s == 50.0 and leg.speed_mm_s == 250.0 and leg.motion == "L" for leg in e.legs))

    def test_accel_b_is_reverse_of_a(self):
        a, b = hp.PATHS["accel_static_a"], hp.PATHS["accel_static_b"]
        visits_a = [hp.Pose()] + [leg.target for leg in a.legs]
        self.assertEqual([leg.target for leg in b.legs], visits_a[-2::-1])
        slow, fast = hp.PATHS["accel_rate_slow_first"], hp.PATHS["accel_rate_fast_first"]
        self.assertEqual([leg.rate_deg_s for leg in slow.legs], [15.0] * 9 + [35.0] * 9)
        self.assertEqual([leg.rate_deg_s for leg in fast.legs], [35.0] * 9 + [15.0] * 9)
        self.assertLessEqual(a.hold_before_s, 7.0)
        self.assertIsNone(a.correction_key)


class WaypointPlanTest(unittest.TestCase):
    def test_segment_names_and_closure(self):
        spec = hp.PATHS["roll_bidir"]
        segs = fm.plan_waypoints(spec, F_T_C, CFG, +1)
        names = [s.name for s in segs]
        self.assertEqual(names[:3], ["goto_start", "hold_start", "wiggle_start_1"])
        i = names.index("hold_before")
        self.assertEqual(names[i + 1:i + 7], ["leg_1", "wp_1", "leg_2", "wp_2", "leg_3", "hold_after"])
        self.assertEqual(names[-1], "hold_end")
        self.assertFalse(segs[0].recorded)
        self.assertEqual([s.motion for s in segs if s.name.startswith("leg_")], ["C", "C", "C"])
        wig = [s for s in segs if s.name.startswith("wiggle_start")]
        self.assertEqual(len(wig), 2 * CFG.wiggle_cycles)
        # 横滚绕光轴摆，光轴方向不变
        np.testing.assert_allclose(wig[0].camera[:3, 2], segs[0].camera[:3, 2], atol=1e-9)
        self.assertLess(fm.rotation_deg(wig[-1].camera, segs[0].camera), 1e-9)

    def test_every_path_passes_offline(self):
        rpc = fr5_kin.OfflineRPC()
        for name, spec in hp.PATHS.items():
            flange, q = start_of(spec)
            segs = fm.plan_waypoints(spec, F_T_C, CFG, +1)
            found = (fm.check_plan(rpc, segs, flange, q, F_T_C, *LIMITS, 5.0, 0.18, 5.0) + fm.apply_speed(segs, MODEL, 1.0)
                     + fm.joint_speed_problems(segs, 150.0, MODEL))
            self.assertEqual(found, [], name)
            for s in segs:
                if s.name.startswith("leg_"):
                    self.assertLessEqual(s.expect_deg_s or 0.0, 50.0 + 1e-6, (name, s.name))
                    self.assertLessEqual(s.percent, 100.0, (name, s.name))
            for s in segs:
                if s.kind == "move" and s.recorded:
                    self.assertGreaterEqual(min(s.log["joint_range"]["min_abs"][2], s.log["joint_range"]["min_abs"][4]), 5.0, (name, s.name))

    def test_frame_yaw_only_shifts_j1(self):
        """整张表绕基座 z 转时，只有 J1 变，其余关节、预计速度都和转之前一样。"""
        rpc = fr5_kin.OfflineRPC()
        for name in ("s_path", "roll_bidir"):
            turned = hp.PATHS[name]
            plain = dataclasses.replace(turned, frame_yaw_deg=0.0)
            results = []
            for spec in (turned, plain):
                flange, q = start_of(spec)
                segs = fm.plan_waypoints(spec, F_T_C, CFG, +1)
                self.assertEqual(fm.check_plan(rpc, segs, flange, q, F_T_C, *LIMITS, 5.0, 0.18, 5.0) + fm.apply_speed(segs, MODEL, 1.0), [], name)
                results.append([s for s in segs if s.kind == "move" and s.recorded])
            for a, b in zip(*results):
                np.testing.assert_allclose(a.joints[1:], b.joints[1:], atol=1e-6, err_msg=f"{name} {a.name}")
                self.assertAlmostEqual((a.joints[0] - b.joints[0] - turned.frame_yaw_deg + 180.0) % 360.0, 180.0, places=6)
                self.assertAlmostEqual(a.percent, b.percent, places=6)

    def test_singular_and_config_problems(self):
        c = fm.level_facing_x([0.6, -0.3, 0.5])
        flange0 = c @ np.linalg.inv(F_T_C)
        seg = fm._move("leg_1", fm.yaw_about_vertical(c, 4.0), F_T_C, speed_mm_s=100.0)

        class Scripted:
            def __init__(self, q):
                self.q = q

            def GetInverseKinRef(self, typ, pose, ref):  # noqa: N802 - SDK 的名字
                return 0, list(self.q)
        wide = ([-1e4] * 6, [1e4] * 6)
        near = fm.check_plan(Scripted([0, 0, 50.0, 0, -3.0, 0]), [seg], flange0, [0, 0, 50.0, 0, -3.0, 0], F_T_C, *wide, 100.0, 0.0, 5.0)
        self.assertTrue(any("离奇异位形" in m for m in near), near)
        flip = fm.check_plan(Scripted([0, 0, 50.0, 0, 30.0, 0]), [seg], flange0, [0, 0, 50.0, 0, -30.0, 0], F_T_C, *wide, 100.0, 0.0, 5.0)
        self.assertTrue(any("构型变了" in m for m in flip), flip)
        # 不传 singular_margin_deg 时不查这两项（开工核对的路径由 fr5_limits 查）
        self.assertEqual(fm.check_plan(Scripted([0, 0, 50.0, 0, 30.0, 0]), [seg], flange0, [0, 0, 50.0, 0, -30.0, 0], F_T_C, *wide, 100.0, 0.0), [])


class CombinedSpeedTest(unittest.TestCase):
    def _leg(self, distance_mm, angle_deg, v=300.0):
        c = fm.level_facing_x([0.6, -0.3, 0.5])
        target = fm.yaw_about_vertical(c, angle_deg)
        target[:3, 3] += [0.0, 0.0, distance_mm / 1000.0]
        flange0 = c @ np.linalg.inv(F_T_C)
        seg = fm._move("leg_1", target, F_T_C, speed_mm_s=v)
        seg.max_rate_deg_s = 50.0
        fm.fill_geometry([seg], flange0)
        return seg

    def test_formula(self):
        for distance, angle in ((350.0, 20.0), (60.0, 40.0)):
            seg = self._leg(distance, angle)
            self.assertEqual(fm.apply_speed([seg], MODEL, 1.0), [])
            t = max(seg.distance_mm / 300.0, seg.angle_deg / 50.0)
            self.assertAlmostEqual(seg.percent, max(seg.distance_mm / (5.96 * t), seg.angle_deg / (2.152 * t)))
            self.assertAlmostEqual(seg.expect_mm_s, seg.distance_mm / t)
            self.assertAlmostEqual(seg.expect_deg_s, seg.angle_deg / t)
            self.assertAlmostEqual(seg.duration_s, t + MODEL.ramp_s)
            self.assertLessEqual(seg.expect_deg_s, 50.0 + 1e-9)
            self.assertLessEqual(seg.expect_mm_s, 300.0 + 1e-9)

    def test_short_leg_does_not_chase_linear_target(self):
        seg = self._leg(60.0, 40.0)
        fm.apply_speed([seg], MODEL, 1.0)
        self.assertLess(seg.expect_mm_s, 300.0)
        self.assertAlmostEqual(seg.expect_deg_s, 50.0, places=6)

    def test_scale(self):
        full, half = self._leg(350.0, 20.0), self._leg(350.0, 20.0)
        fm.apply_speed([full], MODEL, 1.0)
        fm.apply_speed([half], MODEL, 0.5)
        self.assertAlmostEqual(half.percent, full.percent / 2.0)

    def test_without_model_has_no_percent(self):
        seg = self._leg(350.0, 20.0)
        fm.apply_speed([seg], None, 1.0)
        self.assertIsNone(seg.percent)
        self.assertGreater(seg.duration_s, 0.0)


class _TimeLawArm(_FakeArm):
    """MoveL 按“线速度、角速度各有上限，取用时长的”走，speedup 模拟实际比标定快。"""

    def __init__(self, start6, speedup=1.0):
        super().__init__(start6, mm_s_per_pct=MODEL.mm_s_per_pct, deg_s_per_pct=MODEL.wiggle_deg_s_per_pct)
        self.speedup = speedup

    def MoveL(self, **kw):  # noqa: N802 - SDK 的名字
        self.calls.append(("MoveL", kw))
        m0, m1 = fm.fr5_to_matrix(self.pos), fm.fr5_to_matrix(kw["desc_pos"])
        p0, p1 = m0[:3, 3], m1[:3, 3]
        t = max(np.linalg.norm(p1 - p0) * 1000.0 / (kw["vel"] * self.k), fm.rotation_deg(m0, m1) / (kw["vel"] * self.k_rot))
        self._start(kw["desc_pos"], 0.01 * self.speedup / t, lambda u: p0 + u * (p1 - p0))
        return 0


class CombinedGuardTest(unittest.TestCase):
    START = [400.0, -300.0, 500.0, -90.0, 0.0, -100.0]

    def _seg(self, distance_mm, angle_deg):
        m0 = fm.fr5_to_matrix(self.START)
        target = m0.copy()
        target[:3, :3] = fm.axis_angle([0, 0, 1], angle_deg) @ m0[:3, :3]
        target[:3, 3] += [0.0, 0.0, distance_mm / 1000.0]
        seg = fm.Segment("leg_1", "move", flange=fm.matrix_to_fr5(target), joints=[0.0] * 6, speed_mm_s=300.0, max_rate_deg_s=50.0)
        fm.fill_geometry([seg], m0)
        fm.apply_speed([seg], MODEL, 1.0)
        return seg

    def test_normal_run_logs_limits(self):
        arm = _TimeLawArm(self.START)
        seg = self._seg(150.0, 15.0)
        fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        over = seg.log["overspeed"]
        self.assertAlmostEqual(over["limit_mm_s"], seg.expect_mm_s * 1.5 + 5.0)
        self.assertAlmostEqual(over["limit_deg_s"], seg.expect_deg_s * 2.0 + 2.0)
        self.assertGreater(over["peak_mm_s"], 0.0)
        self.assertGreater(over["peak_deg_s"], 0.0)
        self.assertIn("arrived_box_ns", seg.log)

    def test_linear_overspeed_aborts(self):
        arm = _TimeLawArm(self.START, speedup=1.8)
        with self.assertRaisesRegex(fm.MotionAborted, "法兰线速度"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(self._seg(150.0, 15.0))

    def test_angular_overspeed_aborts(self):
        arm = _TimeLawArm(self.START, speedup=2.5)
        with self.assertRaisesRegex(fm.MotionAborted, "角速度"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(self._seg(2.0, 30.0))

    def test_angular_margin_is_wider_than_pure_rotation(self):
        arm = _TimeLawArm(self.START, speedup=1.7)                     # 1.7 倍：纯转动的 1.5 倍会停，组合段不停
        seg = self._seg(2.0, 30.0)
        fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        self.assertIn("arrived_box_ns", seg.log)


# ------------------------------------------------------------------ 路径点修正
def _flange(p_m, euler=(-156.0, -6.0, -168.0)):
    return [*(np.asarray(p_m, dtype=float) * 1000.0).tolist(), *euler]


def _leg(name, start, end, a_ms, b_ms, motion="L", axis=None):
    return {"name": name, "motion": motion, "start_flange": start, "end_flange": end, "axis_base": axis, "start_ms": a_ms, "end_ms": b_ms}


DELTAS = [[0.0] * 6, [0.10, -0.05, 0.02, 1.5, -0.5, 0.8], [-0.04, 0.12, 0.06, -1.0, 2.0, 0.3], [0.03, 0.02, -0.08, 0.4, 0.2, -1.2]]


class WaypointCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.p = [np.array([0.5, 0.0, 0.5]), np.array([0.5, 0.2, 0.5]), np.array([0.7, 0.2, 0.5]), np.array([0.5, 0.0, 0.5])]
        self.q = gt.matrix_to_quat(gt.euler_to_matrix(-156.0, -6.0, -168.0))
        self.info = {"source": gt.WAYPOINT_SOURCE, "deltas": DELTAS, "boundaries_ms": [2500.0, 4500.0],
                     "legs": [_leg(f"leg_{i + 1}", _flange(self.p[i]), _flange(self.p[i + 1]), 1000.0 + 2000.0 * i, 2000.0 + 2000.0 * i)
                              for i in range(3)]}

    def _apply(self, k, s, pos=None):
        pos = np.tile(self.p[0], (len(k), 1)) if pos is None else pos
        return gt.apply_waypoint_correction(pos, np.tile(self.q, (len(k), 1)), np.array(k), np.array(s, dtype=float), self.info)

    def test_single_leg_is_identical_to_end_pose(self):
        info = {"source": gt.WAYPOINT_SOURCE, "deltas": [[0.0] * 6, DELTAS[1]], "boundaries_ms": [], "legs": self.info["legs"][:1]}
        rng = np.random.default_rng(3)
        pos, s = rng.uniform(-1, 1, (200, 3)), rng.uniform(0, 1, 200)
        quat = gt.rotvec_to_quat(rng.uniform(-1, 1, (200, 3)))
        a = gt.apply_waypoint_correction(pos, quat, np.zeros(200, dtype=int), s, info)
        b = gt.apply_end_pose_correction(pos, quat, s, {"source": "END_POSE_CORRECTION", "delta_rotvec_deg_mm": DELTAS[1]})
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_hits_each_waypoint_and_is_continuous_across_legs(self):
        p0 = self.p[0]
        for j in range(3):
            end = self._apply([j], [1.0])
            t = gt.delta_to_matrix(DELTAS[j + 1])
            r0 = gt.quat_to_matrix(self.q)
            np.testing.assert_allclose(end[0][0], t[:3, :3] @ p0 + t[:3, 3], atol=1e-12)
            np.testing.assert_allclose(gt.quat_to_matrix(end[1])[0], t[:3, :3] @ r0, atol=1e-12)
            if j < 2:
                start_next = self._apply([j + 1], [0.0])
                np.testing.assert_allclose(start_next[0], end[0], atol=1e-12)
                np.testing.assert_allclose(gt.quat_to_matrix(start_next[1]), gt.quat_to_matrix(end[1]), atol=1e-12)

    def test_midpoint_matches_screw_interpolation(self):
        half = gt.delta_to_matrix(DELTAS[2]) @ np.linalg.inv(gt.delta_to_matrix(DELTAS[1]))
        d = gt.matrix_to_delta(half)
        step = gt.apply_end_pose_correction(np.zeros((1, 3)), self.q[None] * 0 + [1.0, 0, 0, 0], np.array([0.5]),
                                            {"source": "END_POSE_CORRECTION", "delta_rotvec_deg_mm": d.tolist()})
        expected = np.eye(4)
        expected[:3, :3], expected[:3, 3] = gt.quat_to_matrix(step[1])[0], step[0][0]
        expected = expected @ gt.delta_to_matrix(DELTAS[1])
        got = self._apply([1], [0.5])
        np.testing.assert_allclose(got[0][0], expected[:3, :3] @ self.p[0] + expected[:3, 3], atol=1e-12)

    def test_progress(self):
        pts = np.array([self.p[0], self.p[0] + [0.0, 0.05, 0.0], self.p[1], self.p[1], self.p[1] + [0.1, 0.0, 0.0], self.p[2]])
        t = np.array([900.0, 1250.0, 1400.0, 2800.0, 3250.0, 4000.0])
        k, s = gt.waypoint_progress(t, pts, np.tile(self.q, (6, 1)), self.info)
        self.assertEqual(k.tolist(), [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(s, [0.0, 0.25, 1.0, 0.0, 0.5, 1.0], atol=1e-9)

    def test_arc_progress_uses_angle_and_is_zero_during_wiggle(self):
        euler0 = (-156.0, -6.0, -168.0)
        r0 = gt.euler_to_matrix(*euler0)
        axis = np.array([0.0, 0.0, 1.0])
        r1 = fm.axis_angle(axis, 40.0) @ r0
        end = [100.0, 0.0, 500.0, *np.degrees([math.atan2(r1[2, 1], r1[2, 2]), -math.asin(r1[2, 0]), math.atan2(r1[1, 0], r1[0, 0])])]
        info = {"boundaries_ms": [], "legs": [_leg("leg_1", [100.0, 0.0, 500.0, *euler0], end, 1000.0, 2000.0, "C", axis.tolist())]}
        at = lambda deg: gt.matrix_to_quat(fm.axis_angle(axis, deg) @ r0)                      # noqa: E731
        q = np.array([at(10.0), at(10.0), at(30.0), at(40.0)])
        k, s = gt.waypoint_progress(np.array([500.0, 1100.0, 1500.0, 2500.0]), np.zeros((4, 3)), q, info)
        np.testing.assert_allclose(s, [0.0, 0.25, 0.75, 1.0], atol=1e-9)                       # 第一个样本在同步摆动里


def _line_pose(t_s):
    t = np.asarray(t_s, dtype=float)
    u = np.clip((t - 20.0) / 10.0, 0.0, 1.0)
    u = 3 * u ** 2 - 2 * u ** 3
    return tg.P0 + u[:, None] * np.array([0.3, 0.0, 0.0]), np.broadcast_to(tg.R0, (len(t), 3, 3))


class ProcessTest(unittest.TestCase):
    EULER = (-156.0, -6.0, -168.0)

    def _log(self, d: Path, log: dict) -> None:
        (d / "motion_log.json").write_text(json.dumps(log), encoding="utf-8")

    def _ns(self, s: float) -> int:
        return int((tg.EPOCH_MS + s * 1000.0) * 1e6)

    def test_single_leg_bytes_match_fast2(self):
        """同一段直线、同一个 ΔT：路径点修正和 fast2 的端点修正输出逐字节相同。"""
        delta = [0.10, -0.20, 0.30, 2.0, -3.0, 1.5]
        start, end = _flange(tg.P0, self.EULER), _flange(tg.P0 + [0.3, 0.0, 0.0], self.EULER)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root / "table.json"
            table.write_text(json.dumps({"holdout1": {"k": {"deltas": {"end": delta}}}}), encoding="utf-8")
            outputs = {}
            for name in ("fast2", "holdout1"):
                d = root / name
                d.mkdir()
                tg.write_session(d, np.random.default_rng(0), duration_s=40.0, pose=_line_pose)
                if name == "fast2":
                    log = {"kind": "record", "direction": "right", "parameters": {"PATH_SET": "fast2"},
                           "segments": [{"name": "goto_start", "flange": start}, {"name": "translate", "flange": end}]}
                else:
                    log = {"kind": "record", "direction": "x", "parameters": {"PATH_SET": "holdout1"},
                           "waypoint_path": {"correction_key": "k", "labels": ["start", "end"]},
                           "segments": [{"name": "goto_start", "flange": start},
                                        {"name": "leg_1", "motion": "L", "flange": end, "start_box_ns": self._ns(19.5), "end_box_ns": self._ns(30.5)}]}
                self._log(d, log)
                out = d / "out"
                with mock.patch.dict(gt.TRAVEL_SCALE, {"fast2": {}}), mock.patch.dict(gt.END_POSE_CORRECTION, {"fast2": {"right": delta}}), \
                        mock.patch.object(gt, "WAYPOINT_CORRECTIONS", table):
                    report = gt.process(d, d / "handeye.json", out, dt_ms=0.0)
                outputs[name] = (out, report)
            for f in ("groundtruth_tum.txt", "groundtruth_arm_rate_tum.txt", "groundtruth_images.csv", "groundtruth_tum_uncorrected.txt"):
                self.assertEqual((outputs["fast2"][0] / f).read_bytes(), (outputs["holdout1"][0] / f).read_bytes(), f)
            self.assertNotIn("waypoint_correction", outputs["fast2"][1])
            self.assertEqual(outputs["holdout1"][1]["waypoint_correction"]["source"], gt.WAYPOINT_SOURCE)

    def _wiggle_log(self, group: str) -> dict:
        return {"kind": "record", "direction": "x", "parameters": {"PATH_SET": group},
                "segments": [{"name": "goto_start", "flange": _flange(tg.P0, self.EULER)},
                             {"name": "wiggle_start_1", "start_box_ns": self._ns(3.0), "end_box_ns": self._ns(9.0)},
                             {"name": "wiggle_end_1", "start_box_ns": self._ns(61.0), "end_box_ns": self._ns(67.0)}]}

    def test_holdout_dt_uses_only_wiggles(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tg.write_session(d, np.random.default_rng(0))
            self._log(d, self._wiggle_log("holdout1"))
            with mock.patch.object(gt, "estimate_lag", wraps=gt.estimate_lag) as spy:
                report = gt.process(d, d / "handeye.json", d / "out")
            t_first = spy.call_args_list[0].args[0]
            a, b = [(tg.EPOCH_MS + s * 1000.0) for s in (3.0, 9.0)], [(tg.EPOCH_MS + s * 1000.0) for s in (61.0, 67.0)]
            m = gt.WIGGLE_WINDOW_MARGIN_MS
            inside = ((t_first >= a[0] - m) & (t_first <= a[1] + m)) | ((t_first >= b[0] - m) & (t_first <= b[1] + m))
            self.assertTrue(inside.all())
            lag = report["time_offset"]
            self.assertIn("wiggle only", lag["source"])
            self.assertEqual(set(lag["segments"]), {"first_wiggle", "last_wiggle"})
            self.assertAlmostEqual(lag["dt_ms"], tg.EXPECTED_DT_MS, delta=1.0)

    def test_other_path_sets_use_whole_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tg.write_session(d, np.random.default_rng(0))
            self._log(d, self._wiggle_log("fast2"))
            report = gt.process(d, d / "handeye.json", d / "out")
            self.assertEqual(report["time_offset"]["source"], "gyro cross-correlation")
            self.assertEqual(set(report["time_offset"]["segments"]), {"first_40pct", "last_40pct"})
            self.assertNotIn("waypoint_correction", report)

    def test_missing_table_warns_and_does_not_correct(self):
        start, end = _flange(tg.P0, self.EULER), _flange(tg.P0 + [0.3, 0.0, 0.0], self.EULER)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tg.write_session(d, np.random.default_rng(0), duration_s=40.0, pose=_line_pose)
            self._log(d, {"kind": "record", "direction": "x", "parameters": {"PATH_SET": "holdout1"},
                          "waypoint_path": {"correction_key": "k", "labels": ["start", "end"]},
                          "segments": [{"name": "goto_start", "flange": start},
                                       {"name": "leg_1", "motion": "L", "flange": end, "start_box_ns": self._ns(19.5), "end_box_ns": self._ns(30.5)}]})
            with mock.patch.object(gt, "WAYPOINT_CORRECTIONS", d / "none.json"):
                report = gt.process(d, d / "handeye.json", d / "out", dt_ms=0.0)
            self.assertEqual(report["waypoint_correction"]["source"], "missing")
            self.assertTrue(any("没有路径点修正" in w for w in report["warnings"]))


try:
    import end_pose_fit as epf
    SKIP_FIT = None
except ImportError as exc:                                  # Box 上 OpenCV 导入不了
    epf, SKIP_FIT = None, str(exc)


@unittest.skipIf(SKIP_FIT, SKIP_FIT)
class EndPoseFitTest(unittest.TestCase):
    def test_fit_closes_and_table_round_trips(self):
        board = np.eye(4)
        board[:3, 3] = [1.2, 0.1, 0.6]
        errors = [gt.delta_to_matrix(np.array(d) * -1.0) for d in DELTAS]             # 机械臂报告比真实多出的误差
        frames = []
        for e in errors:                                                               # 报告位姿 × 观测 = 带误差的板位姿
            cam = np.eye(4)
            cam[:3, 3] = [0.5, 0.0, 0.5]
            frames.append([(e @ cam, np.linalg.inv(cam) @ board)] * 5)
        b0, deltas = epf.fit_stops(frames)
        np.testing.assert_allclose(deltas[0], np.eye(4), atol=1e-12)
        for k in range(1, 4):
            np.testing.assert_allclose(deltas[k] @ errors[k] @ board, b0, atol=1e-9)
            np.testing.assert_allclose(epf.correction(deltas, k - 1, 1.0), deltas[k], atol=1e-9)
            np.testing.assert_allclose(epf.correction(deltas, k - 1, 0.0), deltas[k - 1], atol=1e-9)
        labels = ["start", "a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            recs = [{"path_set": "holdout1", "key": "k", "name": "rec1", "labels": labels}]
            epf.write_table(recs, dict(zip(labels, deltas)), path)
            table = json.loads(path.read_text(encoding="utf-8"))["holdout1"]["k"]
            self.assertEqual(sorted(table["deltas"]), ["a", "b", "c"])
            self.assertEqual(table["fitted_from"], ["rec1"])
            np.testing.assert_allclose(gt.delta_to_matrix(table["deltas"]["b"]), deltas[2], atol=2e-4)


class StartupChecksTest(unittest.TestCase):
    def test_camera_imu_rotation(self):
        r = sc.load_camera_imu()
        self.assertEqual(r.shape, (3, 3))
        self.assertAlmostEqual(math.degrees(math.acos((np.trace(r) - 1.0) / 2.0)), 0.606, delta=0.02)

    def test_recovered_after(self):
        level = 0.4
        self.assertEqual(sc.recovered_after(np.array([3.0, 2.0, 1.0, 0.9, 0.5, 0.4, 0.4]), level), 0.25)
        self.assertEqual(sc.recovered_after(np.array([0.4] * 7), level), 0.0)
        self.assertIsNone(sc.recovered_after(np.array([3.0, 2.0, 1.0, 0.9, 0.5, 0.4, 2.0]), level))

    def _camera_session(self, d: Path, drops: int, gap_ms: float = 33.3) -> Path:
        sd = d / "save_data"
        sd.mkdir()
        with open(sd / "img_timeline.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["zed_timestamp_ms", "zed_imu_timestamp_ns", "filename_left", "filename_right"])
            for i in range(50):
                w.writerow([1000 + i * gap_ms, 0, "l", "r"])
        with open(sd / "zed_imu.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["zed_imu_timestamp_ns", "zed_timestamp_ms", "image_sync_trigger", "ax", "ay", "az", "gx", "gy", "gz"])
            for i in range(200):
                w.writerow([int(1e9 + i * 5e6), 0, 0, 0, -9.8, 0, 0, 0, 0])
        (d / "zed_record.log").write_text(f"丢弃队列 0 条\n共保存 50 对，丢弃队列 {drops} 帧\n", encoding="utf-8")
        return sd

    def test_camera(self):
        cases = ((0, 33.3, True), (3, 33.3, False), (0, 45.0, False))
        for drops, gap, expected in cases:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sc.check_camera(self._camera_session(Path(tmp), drops, gap)), expected)


class ConditionsTest(unittest.TestCase):
    def test_condition_names(self):
        for name, expected in (("20261110_100000_roll_bidir", "roll_bidir"), ("20261110_100000_trans_level_first", "trans_two_attitude"),
                               ("20261110_100000_trans_tilt_first", "trans_two_attitude"), ("20261110_100000_s_path_sceneB", "s_path_B"),
                               ("20261110_100000_eight", "eight"), ("20261110_100000_s_path", None), ("20261110_100000_roll_bidir_board", None),
                               ("20261110_100000_accel_static_a", None), ("calib", None)):
            self.assertEqual(mc.condition_of(name), expected, name)

    def test_writes_only_when_complete(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / "calib").mkdir()
            for i in range(2):
                (root / f"20261110_10000{i}_roll_bidir").mkdir()
            self.assertEqual(mc.main([str(root)]), 1)
            self.assertFalse((root / "conditions.yaml").exists())
            (root / "20261110_100002_roll_bidir").mkdir()
            self.assertEqual(mc.main([str(root)]), 0)
            text = (root / "conditions.yaml").read_text(encoding="utf-8")
            self.assertTrue(text.startswith("conditions:\n"))
            self.assertIn("  20261110_100001_roll_bidir: roll_bidir\n", text)
            (root / "20261110_100003_roll_bidir_board").mkdir()
            self.assertEqual(mc.main([str(root)]), 1)


class RunSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = import_run_session()

    def setUp(self):
        rs = self.rs
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        names = ("LIMIT_POSES", "START_POSE", "HANDEYE", "SPEED_CAL", "StateRecorder", "SESSIONS_DIR", "camera_users", "HOLDOUT_WINDOW")
        saved = {k: getattr(rs, k) for k in names}
        self.addCleanup(lambda: [setattr(rs, k, v) for k, v in saved.items()])
        rs.LIMIT_POSES, rs.SPEED_CAL = str(tmp / "limit_poses.json"), str(tmp / "speed.json")
        rs.START_POSE, rs.HANDEYE = str(HERE / "start_pose.json"), str(HERE / "handeye_s003_result.json")
        rs.StateRecorder, rs.camera_users, rs.SESSIONS_DIR = _FakeRecorder, (lambda: []), str(tmp / "sessions")
        rs.HOLDOUT_WINDOW = dict(EMPTY_WINDOW)
        Path(rs.SPEED_CAL).write_text(json.dumps({
            "created": "test", "settings": {k: getattr(rs, k) for k in rs.CAL_KEYS},
            "model": {"mm_s_per_pct": 5.96, "wiggle_deg_s_per_pct": 2.152, "ramp_s": 0.459},
            "accel": {"test_down_mm_s2": 1084.0, "test_yaw_back_deg_s2": 156.8}}), encoding="utf-8")
        self.rpc = fr5_kin.OfflineRPC(SOFT)
        self.rpc.GetDefaultTransVel = lambda: (0, 1000.0)

    def park(self, name):
        flange, q = start_of(hp.PATHS[name])
        _FakeRecorder.ARM.update(flange=fm.matrix_to_fr5(flange), joints=q, ik=self.rpc)

    def run_cmd(self, fn, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fn(*args)
        return code, out.getvalue()

    def test_plan_passes_every_path(self):
        for name in hp.PATHS:
            self.park(name)
            code, text = self.run_cmd(self.rs.cmd_plan, argparse.Namespace(names=[name], scale=1.0), [])
            self.assertEqual(code, 0, text)
            self.assertIn(f"=== {name}：通过", text)
            self.assertIn("同步摆动 +1", text)

    def test_plan_refuses_without_window(self):
        self.rs.HOLDOUT_WINDOW = None
        self.park("s_path")
        code, text = self.run_cmd(self.rs.cmd_plan, argparse.Namespace(names=["s_path"], scale=1.0), [])
        self.assertEqual(code, 1)
        self.assertIn("HOLDOUT_WINDOW 还没设", text)

    def test_plan_reports_window_in_view(self):
        self.rs.HOLDOUT_WINDOW = {"viewpoint_mm": (550.0, -150.0, 450.0), "regions": [(-180.0, 180.0, -90.0, 90.0)], "distance_m": 7.0}
        self.park("s_path")
        code, text = self.run_cmd(self.rs.cmd_plan, argparse.Namespace(names=["s_path"], scale=1.0), [])
        self.assertEqual(code, 1)
        self.assertIn("画面离右窗最近", text)

    def test_unknown_name(self):
        code, text = self.run_cmd(self.rs.cmd_plan, argparse.Namespace(names=["up"], scale=1.0), [])
        self.assertEqual(code, 2)
        self.assertIn("不认识的录制名", text)

    def test_window_format_checked(self):
        self.rs.HOLDOUT_WINDOW = {"regions": []}
        self.assertTrue(any("HOLDOUT_WINDOW" in m for m in self.rs.parameter_problems()))
        self.rs.HOLDOUT_WINDOW = {"viewpoint_mm": (0, 0, 0), "regions": [(10.0, 0.0, -1.0, 1.0)], "distance_m": 7.0}
        self.assertTrue(any("regions" in m for m in self.rs.parameter_problems()))
        self.rs.HOLDOUT_WINDOW = dict(EMPTY_WINDOW)
        self.assertFalse(any("HOLDOUT_WINDOW" in m for m in self.rs.parameter_problems()))

    def test_record_declined_leaves_nothing(self):
        self.park("roll_bidir")
        args = argparse.Namespace(scale=0.2, yes=False, tag="board")
        with mock.patch("builtins.input", return_value="no"):
            code, text = self.run_cmd(self.rs.cmd_motion, args, "roll_bidir")
        self.assertEqual(code, 1)
        self.assertIn("路径组 holdout1", text)
        self.assertIn("leg_1", text)
        self.assertIn("没有确认", text)
        self.assertEqual(list(Path(self.rs.SESSIONS_DIR).iterdir()), [])

    def test_warmup_declined_does_not_move(self):
        self.park("s_path")
        args = argparse.Namespace(path="s_path", minutes=1.0, scale=1.0, yes=False, tag="")
        with mock.patch("builtins.input", return_value="no"), mock.patch.object(fm, "emergency_stop") as stop:
            code, text = self.run_cmd(self.rs.cmd_warmup, args)
        self.assertEqual(code, 1)
        self.assertIn("不开相机、不存数据", text)
        stop.assert_not_called()
        self.assertFalse(Path(self.rs.SESSIONS_DIR).exists())

    def test_leg_summary_flags_large_errors(self):
        seg = fm.Segment("leg_1", "move", motion="L", distance_mm=350.0, angle_deg=20.0, percent=50.0, expect_mm_s=300.0, expect_deg_s=17.0, duration_s=1.6)
        seg.log["measured"] = {"linear_mm_s": {"mean": 250.0, "min": 200.0, "max": 260.0}, "angular_deg_s": {"mean": 18.0, "min": 10.0, "max": 30.0}, "duration_s": 1.7}
        notes = self.rs.leg_summary(seg)["notes"]
        self.assertEqual(len(notes), 2)
        seg.log["measured"] = {"linear_mm_s": {"mean": 295.0, "min": 280.0, "max": 305.0}, "angular_deg_s": {"mean": 17.0, "min": 16.0, "max": 19.0}, "duration_s": 1.6}
        self.assertEqual(self.rs.leg_summary(seg)["notes"], [])


if __name__ == "__main__":
    unittest.main()
