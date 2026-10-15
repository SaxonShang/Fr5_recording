"""fr5_motion.py 的离线测试（不连机械臂）。

    cd ~/ROBOT/fr5_record && python3 -B -m unittest test_fr5_motion
"""

import dataclasses
import json
import signal
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

import fr5_motion as fm

F_T_C = np.array([[0.999660953447201, 0.0007843776424874251, 0.02602619650942236, -0.058960616228020075],
                  [-0.0007566244143297974, 0.9999991346690702, -0.0010761879977619043, -0.07159469739308227],
                  [-0.026027018125954093, 0.001056131064239924, 0.999660681888933, 0.09927925870705787],
                  [0.0, 0.0, 0.0, 1.0]])
NOW = fm.fr5_to_matrix([484.963, -506.674, 490.858, -90.545, 0.0, -105.264]) @ F_T_C  # 10-10 的相机位姿
CFG = fm.MotionConfig(speed_mps=0.05, hold_start_s=5, hold_mid_s=3, hold_end_s=5,
                      wiggle_deg=10, wiggle_deg_s=20, wiggle_cycles=2, reposition_mps=0.05, sync_test_hold_s=10)
MODEL = fm.SpeedModel(mm_s_per_pct=1.0, wiggle_deg_s_per_pct=0.5, ramp_s=0.3, calibrated=True)


def _wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def _shift(camera, vec_m):
    out = camera.copy()
    out[:3, 3] += vec_m
    return out


def _line(centre, axis, stroke_m=0.30, cfg=CFG):
    """以 centre 为中点、沿 axis 的平移录制（同以前 stroke_m 的约定）。"""
    axis = np.asarray(axis, dtype=float)
    return fm.plan_translation(_shift(centre, -axis * stroke_m / 2), _shift(centre, axis * stroke_m / 2), F_T_C, cfg)


ROT_AXIS = {"yaw_left": ("up", +1), "yaw_right": ("up", -1), "pitch_up": ("x", +1), "pitch_down": ("x", -1),
            "roll_right": ("z", +1), "roll_left": ("z", -1)}   # 方向名 -> (转轴, 符号)：右手定则下的正方向


def _turn(centre, name, deg, cfg=CFG):
    """以 centre 为中心姿态、两边各转 deg/2 的旋转录制：偏航绕竖直轴，俯仰绕 centre 的 x 轴，横滚绕它的光轴。返回 (段, 转轴)。"""
    key, sign = ROT_AXIS[name]
    axis = sign * {"up": np.array([0, 0, 1.0]), "x": centre[:3, 0], "z": centre[:3, 2]}[key]
    return fm.plan_rotation(fm.rotate_about(centre, axis, -deg / 2), axis, deg, F_T_C, cfg), axis


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
        self.assertAlmostEqual(fm.rotation_blur_px(5.0, 10000), 0.641, places=3)

    def test_rotate_about_keeps_centre(self):
        level = fm.level_camera(NOW)
        rng = np.random.default_rng(4)
        for _ in range(20):
            axis, deg = rng.normal(size=3), rng.uniform(-40, 40)
            r = fm.axis_angle(axis, deg)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(r @ axis, axis, atol=1e-12)              # 轴不动
            c = fm.rotate_about(level, axis, deg)
            np.testing.assert_allclose(c[:3, 3], level[:3, 3])
            self.assertAlmostEqual(fm.rotation_deg(c, level), abs(deg), places=9)
        np.testing.assert_allclose(fm.yaw_about_vertical(level, 10)[:3, :3],
                                   fm.rotate_about(level, fm.camera_up(level), 10)[:3, :3], atol=1e-15)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.centre = fm.level_camera(NOW)

    def test_each_direction_is_pure_translation(self):
        expected = {"forward": self.centre[:3, 2], "backward": -self.centre[:3, 2],
                    "right": self.centre[:3, 0], "left": -self.centre[:3, 0],
                    "up": np.array([0, 0, 1.0]), "down": np.array([0, 0, -1.0])}
        for name, axis in expected.items():
            segs = _line(self.centre, axis)
            by = {s.name: s for s in segs}
            start, end = by["goto_start"].camera, by["translate"].camera
            np.testing.assert_allclose(end[:3, 3] - start[:3, 3], axis * 0.30, atol=1e-12)
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

        rec = _line(self.centre, self.centre[:3, 0])
        fm.fill_geometry(rec, flange0)
        self.assertEqual(fm.apply_speed(rec, MODEL, 1.0), [])
        tr = next(s for s in rec if s.name == "translate")
        self.assertAlmostEqual(tr.percent, 50.0)
        self.assertAlmostEqual(tr.expect_mm_s, 50.0)
        self.assertAlmostEqual(tr.distance_mm, 300.0, places=6)
        self.assertAlmostEqual(tr.duration_s, 300.0 / 50.0 + 0.3)

    def test_speed_percent_limits_and_unknown_model(self):
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        rec = _line(self.centre, [0, 0, 1.0])
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

    def test_each_rotation_pivots_about_centre(self):
        c = self.centre
        sign_check = {"yaw_left": lambda r: r[:, 2] @ c[:3, 0] < 0, "yaw_right": lambda r: r[:, 2] @ c[:3, 0] > 0,
                      "pitch_up": lambda r: r[2, 2] > 0, "pitch_down": lambda r: r[2, 2] < 0,      # 光轴抬起 / 低下
                      "roll_right": lambda r: r[2, 0] < 0, "roll_left": lambda r: r[2, 0] > 0}     # 相机 x 轴右端向下 / 向上
        self.assertEqual(set(sign_check), set(ROT_AXIS))
        deg = 20.0
        for name, check in sign_check.items():
            segs, axis = _turn(c, name, deg)
            by = {s.name: s for s in segs}
            self.assertNotIn("translate", by)
            start, rot = by["goto_start"].camera, by["rotate"]
            self.assertFalse(by["goto_start"].recorded)
            for cam in (start, rot.camera):
                np.testing.assert_allclose(cam[:3, 3], c[:3, 3], atol=1e-12)        # 光心不动
                self.assertAlmostEqual(fm.rotation_deg(cam, c), deg / 2, places=9)
            self.assertAlmostEqual(fm.rotation_deg(start, rot.camera), deg, places=9)
            self.assertTrue(check(rot.camera[:3, :3]), name)
            self.assertEqual((rot.motion, rot.arc_deg, rot.rate_deg_s), ("C", deg, CFG.rot_deg_s))
            np.testing.assert_allclose(rot.arc_axis, axis, atol=1e-12)
            np.testing.assert_allclose(rot.pivot, c[:3, 3], atol=1e-12)
            # 中间点就是中心姿态（转了一半）
            np.testing.assert_allclose(fm.fr5_to_matrix(rot.via_flange), c @ np.linalg.inv(F_T_C), atol=1e-9)
            # 同步摆动绕同一根转轴、单侧摆，结束时回到原姿态，光心不动
            for prefix, home in (("wiggle_start", start), ("wiggle_end", rot.camera)):
                wig = [s for s in segs if s.name.startswith(prefix)]
                self.assertEqual(len(wig), 2 * CFG.wiggle_cycles)
                for s in wig:
                    np.testing.assert_allclose(s.camera[:3, 3], c[:3, 3], atol=1e-12)
                    if fm.rotation_deg(s.camera, home) > 1e-6:
                        w, v = np.linalg.eig(s.camera[:3, :3] @ home[:3, :3].T)      # 相对起点的转动
                        ax = np.real(v[:, np.argmin(np.abs(w - 1))])
                        self.assertAlmostEqual(abs(ax @ axis), 1.0, places=9)
                self.assertLess(fm.rotation_deg(wig[-1].camera, home), 1e-9)

    def test_inward_wiggle_keeps_rotation_range(self):
        """同步摆动往旋转范围内侧摆：全程姿态范围就是旋转本身的范围；摆动幅度大于总转角时从另一头多出来。"""
        c = self.centre
        expect = {"yaw_left": "heading", "yaw_right": "heading", "pitch_up": "pitch", "pitch_down": "pitch",
                  "roll_left": "roll", "roll_right": "roll"}
        for name, key in expect.items():
            span = fm.pose_span(_turn(c, name, 20.0)[0], c, F_T_C)
            np.testing.assert_allclose(span[key], (-10.0, 10.0), atol=1e-6, err_msg=name)
            for other in {"heading", "pitch", "roll"} - {key}:
                np.testing.assert_allclose(span[other], (0.0, 0.0), atol=1e-6, err_msg=f"{name} {other}")
        span = fm.pose_span(_turn(c, "yaw_left", 6.0)[0], c, F_T_C)     # −3 → +3：开头摆到 −3+10，结尾摆到 +3−10
        np.testing.assert_allclose(span["heading"], (-7.0, 7.0), atol=1e-6)

    def test_level_wiggle_unchanged(self):
        """相机水平时，绕相机竖直轴摆动和原来绕基坐标竖直轴摆动是一回事（平移录制不受影响）。"""
        for s in fm._wiggle("w", self.centre, F_T_C, CFG):
            deg = {1: 10, 2: -10, 3: 10, 4: -10, 5: 0}[int(s.name[-1])]
            np.testing.assert_allclose(s.camera, fm.yaw_about_vertical(self.centre, deg), atol=1e-15)

    def test_rotation_speed_and_arc_length(self):
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        segs, _ = _turn(self.centre, "pitch_up", 30.0)
        fm.fill_geometry(segs, flange0)
        self.assertEqual(fm.apply_speed(segs, MODEL, 1.0), [])
        rot = next(s for s in segs if s.name == "rotate")
        self.assertAlmostEqual(rot.angle_deg, 30.0, places=9)
        self.assertAlmostEqual(rot.percent, 10.0)                  # 5 °/s ÷ 0.5 °/s/%
        self.assertAlmostEqual(rot.expect_deg_s, 5.0)
        self.assertAlmostEqual(rot.duration_s, 30.0 / 5.0 + 0.3)
        # 弧长 = 沿圆弧采样的折线长度
        prev = fm.fr5_to_matrix([s for s in segs if s.name.startswith("wiggle_start")][-1].flange)
        target = fm.fr5_to_matrix(rot.flange)
        pts = np.array([fm.path_flange(rot, prev, target, u, F_T_C)[:3, 3] for u in np.linspace(0, 1, 2001)])
        self.assertAlmostEqual(rot.distance_mm, np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() * 1000, places=3)
        self.assertGreater(rot.distance_mm, 50.0)                  # 法兰离光心约 0.12 m，转 30° 走约 65 mm

    def test_arc_path_keeps_centre(self):
        segs, _ = _turn(self.centre, "yaw_right", 30.0)
        by = {s.name: s for s in segs}
        rot = by["rotate"]
        prev = fm.fr5_to_matrix([s for s in segs if s.name.startswith("wiggle_start")][-1].flange)
        target = fm.fr5_to_matrix(rot.flange)
        for u in np.linspace(0, 1, 31):
            cam = fm.path_flange(rot, prev, target, u, F_T_C) @ F_T_C
            np.testing.assert_allclose(cam[:3, 3], self.centre[:3, 3], atol=1e-12)
        np.testing.assert_allclose(fm.path_flange(rot, prev, target, 1.0, F_T_C), target, atol=1e-12)
        np.testing.assert_allclose(fm.matrix_to_fr5(fm.path_flange(rot, prev, target, 0.5, F_T_C)), rot.via_flange,
                                   atol=1e-9)
        # 同一转动用 MoveL（法兰走直线）时光心在中途鼓出去
        line = fm.Segment("line", "move", flange=rot.flange)
        bulge = max(np.linalg.norm((fm.path_flange(line, prev, target, u, F_T_C) @ F_T_C)[:3, 3] - self.centre[:3, 3])
                    for u in np.linspace(0, 1, 31)) * 1000
        self.assertGreater(bulge, 3.0)

    def test_check_plan_fills_via_joints(self):
        class FakeIK:
            def GetInverseKinRef(self, typ, pose, ref):             # 连续的假逆解：关节 = 位姿 / 10
                return 0, [v / 10.0 for v in pose]
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        wide = ([-1e4] * 6, [1e4] * 6)
        for name in ("yaw_left", "pitch_down"):
            segs, _ = _turn(self.centre, name, 20.0)
            problems = fm.check_plan(FakeIK(), segs, flange0, [0.0] * 6, F_T_C, *wide, 100.0, 0.0)
            self.assertEqual(problems, [])
            rot = next(s for s in segs if s.name == "rotate")
            np.testing.assert_allclose(rot.via_joints, np.array(rot.via_flange) / 10.0, atol=1e-9)
            np.testing.assert_allclose(rot.joints, np.array(rot.flange) / 10.0, atol=1e-9)
        # MoveC 的起点和规划不一致时拒绝
        segs, _ = _turn(self.centre, "yaw_left", 20.0)
        rot_only = [s for s in segs if s.name == "rotate"]
        problems = fm.check_plan(FakeIK(), rot_only, flange0, [0.0] * 6, F_T_C, *wide, 100.0, 0.0)
        self.assertTrue(any("MoveC 的起点" in m for m in problems), problems)

    def test_check_plan_steep_but_continuous(self):
        """一段关节变化得快但连续（每 10 mm 变 8°）：细分后每份都很小，不算跳支，关节速度按快的算；
        真的跳了（某处突然 +40°）：细分后仍然跳，报问题。"""
        centre = fm.level_camera(NOW)
        end = centre.copy()
        end[:3, 3] += centre[:3, 0] * 0.1                         # 往右 100 mm
        seg = fm._move("translate", end, F_T_C, 50.0)
        p0 = (centre @ np.linalg.inv(F_T_C))[:3, 3] * 1000.0

        class Steep:
            def __init__(self, jump=False):
                self.jump = jump

            def GetInverseKinRef(self, typ, pose, ref):
                s = float(np.linalg.norm(np.asarray(pose[:3]) - p0))   # 法兰离起点多少 mm（直线平移，单调）
                q = [0.0, 0.0, -60.0 + 0.8 * abs(s), -90.0, -45.0, 0.0]
                if self.jump and abs(s) > 50.0:
                    q[3] += 40.0
                return 0, q
        flange0 = centre @ np.linalg.inv(F_T_C)
        ok = fm.check_plan(Steep(), [seg], flange0, [0.0, 0.0, -60.0, -90.0, -45.0, 0.0], F_T_C,
                           [-1e4] * 6, [1e4] * 6, 5.0, 0.0)
        self.assertEqual(ok, [])
        self.assertAlmostEqual(seg.joint_rate, 0.8, delta=0.05)  # 每 mm 0.8°
        seg2 = fm._move("translate", end, F_T_C, 50.0)
        bad = fm.check_plan(Steep(jump=True), [seg2], flange0, [0.0, 0.0, -60.0, -90.0, -45.0, 0.0], F_T_C,
                            [-1e4] * 6, [1e4] * 6, 5.0, 0.0)
        self.assertTrue(any("细分后仍有一步" in m for m in bad), bad)

    def test_check_plan_returns_from_outside_range(self):
        """当前就在范围外：移到起点可以往回走（越界只减不增），终点和录制段必须在范围内。"""
        class FakeIK:
            def GetInverseKinRef(self, typ, pose, ref):             # 关节 = 位姿 / 10
                return 0, [v / 10.0 for v in pose]
        centre = self.centre
        away = centre.copy()
        away[:3, 3] += [0.1, 0.0, 0.0]                              # 法兰在 x 方向多 100 mm → J1 多 10
        x_c = fm.matrix_to_fr5(centre @ np.linalg.inv(F_T_C))[0]
        lo, hi = [-1e4] * 6, [1e4] * 6
        hi[0] = x_c / 10.0 + 1.0                                    # 起点在范围内，away 处 J1 超出约 9°

        def run(start_cam, target_cam, recorded):
            seg = fm._move("goto_start", target_cam, F_T_C, 50.0, recorded=recorded)
            flange0 = start_cam @ np.linalg.inv(F_T_C)
            return fm.check_plan(FakeIK(), [seg], flange0, [v / 10.0 for v in fm.matrix_to_fr5(flange0)],
                                 F_T_C, lo, hi, 100.0, 0.0)
        self.assertEqual(run(away, centre, recorded=False), [])                 # 往回走：放行
        self.assertTrue(run(centre, away, recorded=False))                      # 往外走：拒绝
        self.assertTrue(run(away, centre, recorded=True))                       # 录制段：严格
        far = centre.copy()
        far[:3, 3] += [0.05, 0.0, 0.0]                                          # 终点仍在范围外
        self.assertTrue(any("100%" in m for m in run(away, far, recorded=False)))

    def test_one_sided_wiggle(self):
        """translation_wiggle_side=1：同步摆动只往左转（朝向在 0～+10°），光心不动，最后回到原朝向；sync-test 也一样。"""
        cfg = dataclasses.replace(CFG, translation_wiggle_side=1)
        c = self.centre
        for axis in (c[:3, 2], c[:3, 0], np.array([0, 0, 1.0])):
            by = {s.name: s for s in _line(c, axis, 0.45, cfg)}
            start, end = by["goto_start"].camera, by["translate"].camera
            for prefix, base in (("wiggle_start", start), ("wiggle_end", end)):
                wig = [s for s in by.values() if s.name.startswith(prefix)]
                self.assertEqual(len(wig), 2 * cfg.wiggle_cycles)
                for s in wig:
                    z = base[:3, :3].T @ s.camera[:3, 2]                 # 光轴在本段起点相机系里
                    heading = np.degrees(np.arctan2(-z[0], z[2]))         # 正 = 向左
                    self.assertTrue(-1e-9 <= heading <= cfg.wiggle_deg + 1e-9, (s.name, heading))
                    np.testing.assert_allclose(s.camera[:3, 3], base[:3, 3], atol=1e-12)
                self.assertLess(fm.rotation_deg(wig[-1].camera, base), 1e-9)
        sync = [s for s in fm.plan_sync_test(c, F_T_C, cfg) if s.name.startswith("wiggle")]
        self.assertTrue(all(np.degrees(np.arctan2(-(c[:3, :3].T @ s.camera[:3, 2])[0], (c[:3, :3].T @ s.camera[:3, 2])[2]))
                            >= -1e-9 for s in sync))

    def test_joint_speed_check(self):
        """假逆解：关节 = 法兰位姿 / 10。按预计速度算出的关节速度和上限比较；只查录制段。"""
        class FakeIK:
            def GetInverseKinRef(self, typ, pose, ref):
                return 0, [v / 10.0 for v in pose]
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        wide = ([-1e4] * 6, [1e4] * 6)
        segs = _line(self.centre, self.centre[:3, 0])
        self.assertEqual(fm.check_plan(FakeIK(), segs, flange0, [0.0] * 6, F_T_C, *wide, 100.0, 0.0), [])
        self.assertEqual(fm.apply_speed(segs, MODEL, 1.0), [])
        by = {s.name: s for s in segs}
        tr = by["translate"]
        d = np.subtract(tr.flange[:3], by["goto_start"].flange[:3])
        rate = np.abs(d).max() / np.linalg.norm(d) / 10.0          # 变得最快的关节：每 mm 几度
        self.assertAlmostEqual(tr.joint_rate, rate, places=6)
        w = rate * 50.0
        self.assertEqual(fm.joint_speed_problems(segs, 1.01 * w), [])
        self.assertAlmostEqual(tr.log["max_joint_speed_deg_s"], w, places=6)
        found = fm.joint_speed_problems(segs, 0.99 * w)
        self.assertTrue(any(m.startswith("translate:") for m in found), found)
        self.assertFalse(any(m.startswith("goto_start") for m in found))          # 不录制的移动不查
        fm.apply_speed(segs, MODEL, 0.5)                                       # 降速（--scale 0.5）后关节也慢一半
        self.assertEqual([m for m in fm.joint_speed_problems(segs, 0.99 * w) if m.startswith("translate")], [])
        # 有实测加速度：两端在加减速，关节速度按梯形曲线算。这条路径上关节速率处处相同，最快仍是匀速段
        fm.apply_speed(segs, MODEL, 1.0)
        accel = dataclasses.replace(MODEL, accel_mm_s2=1000.0)
        fm.joint_speed_problems(segs, 1e9, accel)
        self.assertAlmostEqual(tr.log["max_joint_speed_deg_s"], w, delta=0.01 * w)
        # 只有终点附近关节变得快（像靠近奇异位形）：匀速估算会超，按减速段算就不超
        tr.joint_rates = [(u, r if u < 0.97 else 3 * r) for u, r in tr.joint_rates]
        tr.joint_rate = 3 * rate
        self.assertTrue(fm.joint_speed_problems([tr], 1.5 * w))                  # 没有加速度：3w > 1.5w
        self.assertTrue(fm.joint_speed_problems([tr], 1.5 * w, accel))           # 加速很快：终点前仍是满速
        slow = dataclasses.replace(MODEL, accel_mm_s2=20.0)                     # 最后 5 mm 处速度 √(2·20·5) ≈ 14 mm/s
        self.assertEqual(fm.joint_speed_problems([tr], 1.5 * w, slow), [])

    def test_movec_test_plan(self):
        segs = fm.plan_movec_test(self.centre, F_T_C, CFG, 5.0)
        flange0 = self.centre @ np.linalg.inv(F_T_C)
        fm.fill_geometry(segs, flange0)
        self.assertEqual(fm.apply_speed(segs, MODEL, 1.0), [])
        arcs = [s for s in segs if s.name.startswith("arc")]
        lines = [s for s in segs if s.name.startswith("line")]
        self.assertEqual([s.motion for s in arcs + lines], ["C"] * 3 + ["L"] * 3)
        for group in (arcs, lines):
            self.assertEqual([round(s.angle_deg, 6) for s in group], [10.0, 20.0, 10.0])
            self.assertLess(fm.rotation_deg(group[-1].camera, self.centre), 1e-9)
        for s in arcs + lines:
            self.assertEqual(s.percent, 5.0)
            self.assertAlmostEqual(s.expect_deg_s, 2.5)             # 5% × 0.5 °/s/%，超速保护用
            np.testing.assert_allclose(s.pivot, self.centre[:3, 3], atol=1e-12)


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


def _circle(p0, p1, p2):
    """过三点的圆：圆心、法向（按 p0→p1→p2 的走向）。"""
    a, b = p1 - p0, p2 - p0
    n = np.cross(a, b)
    centre = p0 + (np.dot(b, b) * np.cross(n, a) + np.dot(a, a) * np.cross(b, n)) / (2.0 * np.dot(n, n))
    return centre, n / np.linalg.norm(n)


class _FakeArm:
    """假的状态录制器 + RPC：每次 wait_state 前进 10 ms。
    MoveL：位置按 速度百分比 × mm_s_per_pct 匀速直线走向目标，姿态同步 slerp；
    MoveC：按 速度百分比 × deg_s_per_pct 的角速度，位置沿过 via 的圆弧、姿态 slerp（movec_as_line=True 时位置走直线）；
    movec_blocks=True 时像真控制器一样走完才返回（10-12 实测），movec_error 非 0 时直接返回这个错误码。"""

    def __init__(self, start6, mm_s_per_pct, lose_state_after=None, deg_s_per_pct=0.5, movec_as_line=False,
                 movec_blocks=False, movec_error=0, pos_lead=1.0):
        self.pos = np.array(start6, dtype=float)
        self.k, self.k_rot, self.movec_as_line = mm_s_per_pct, deg_s_per_pct, movec_as_line
        self.movec_blocks, self.movec_error = movec_blocks, movec_error
        self.pos_lead = pos_lead                   # >1：位置比姿态先走（位置进度 = min(1, u × pos_lead)）
        self.idle = threading.Event()              # 运动走完时置位（阻塞的 MoveC 等它）
        self.target, self.speed, self.calls, self.ms, self.count = None, 0.0, [], 1000, 0
        self.lose_state_after = lose_state_after   # MoveL 之后再给这么多包就“断线”
        self.frozen = False                        # True：断线，不再有新包
        self.pkg = None
        self.latest_monotonic = None               # 同 StateRecorder：最新一包的到达时刻
        self.motion = None                         # (起点矩阵, 目标矩阵, 每包进度, 位置函数)
        self.u = 0.0

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
            m0, m1, du, position = self.motion
            self.u = min(1.0, self.u + du)
            if self.u >= 1.0:
                self.pos = self.target.copy()
                self.target = None
                self.idle.set()
            else:
                pose = np.eye(4)
                pose[:3, 3] = position(min(1.0, self.u * self.pos_lead))
                q0, q1 = fm.gt.matrix_to_quat(m0[:3, :3]), fm.gt.matrix_to_quat(m1[:3, :3])
                pose[:3, :3] = fm.gt.quat_to_matrix(fm.gt.slerp(q0, q1, np.array(self.u)))
                self.pos = np.array(fm.matrix_to_fr5(pose))
                state = 2
        self.count += 1
        self.pkg = types.SimpleNamespace(
            flange_cur_pos=list(self.pos), robot_state=state, EmergencyStop=0, main_code=0, sub_code=0,
            safety_stop0_state=0, safety_stop1_state=0, rbtEnableState=1, tool=0, user=0,
            hour=self.ms // 3_600_000, minute=self.ms // 60_000 % 60, second=self.ms // 1000 % 60,
            millisecond=self.ms % 1000)
        return self.count, self.pkg

    def _start(self, target6, du, position):
        # MoveC 可能在另一个线程里调用：先备好 motion，最后才设 target（wait_state 看到 target 就会用 motion）
        self.idle.clear()
        self.motion = (fm.fr5_to_matrix(self.pos), fm.fr5_to_matrix(target6), du, position)
        self.u = 0.0
        self.target = np.array(target6, dtype=float)

    def MoveL(self, **kw):  # noqa: N802 - SDK 的名字
        self.calls.append(("MoveL", kw))
        m0, m1 = fm.fr5_to_matrix(self.pos), fm.fr5_to_matrix(kw["desc_pos"])
        p0, p1 = m0[:3, 3], m1[:3, 3]                            # m
        d_mm = np.linalg.norm(p1 - p0) * 1000.0
        angle = fm.rotation_deg(m0, m1)
        # 有平移按线速度走；原地转动按角速度走
        du = (kw["vel"] * self.k * 0.01 / d_mm if d_mm > 0
              else kw["vel"] * self.k_rot * 0.01 / angle if angle > 0 else 1.0)
        self._start(kw["desc_pos"], du, lambda u: p0 + u * (p1 - p0))
        return 0

    def MoveC(self, **kw):  # noqa: N802 - SDK 的名字
        self.calls.append(("MoveC", kw))
        if self.movec_error:
            return self.movec_error
        m0, m1 = fm.fr5_to_matrix(self.pos), fm.fr5_to_matrix(kw["desc_pos_t"])
        p0, pv, p1 = m0[:3, 3], fm.fr5_to_matrix(kw["desc_pos_p"])[:3, 3], m1[:3, 3]
        angle = fm.rotation_deg(m0, m1)
        c, n = _circle(p0, pv, p1)
        phi = 2.0 * np.degrees(np.arccos(np.clip(np.dot(p0 - c, pv - c) / np.dot(p0 - c, p0 - c), -1, 1)))

        def position(u):
            if self.movec_as_line:
                return p0 + u * (p1 - p0)
            return c + fm.axis_angle(n, u * phi) @ (p0 - c)
        self._start(kw["desc_pos_t"], kw["vel_t"] * self.k_rot * 0.01 / angle, position)
        if self.movec_blocks:
            self.idle.wait(10.0)                   # 运动由主线程的 wait_state 推进；中止后最多挂 10 s
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
        cmd, call = arm.calls[0]
        self.assertEqual(cmd, "MoveL")
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

    def _rotation(self, name="yaw_left", deg=20.0):
        """规划好的旋转段（从起点姿态开始），关节角用假的（假机械臂不看关节）。"""
        segs, _ = _turn(fm.level_camera(NOW), name, deg)
        by = {s.name: s for s in segs}
        start6 = by["goto_start"].flange
        rot = by["rotate"]
        fm.fill_geometry([rot], fm.fr5_to_matrix(start6))
        fm.apply_speed([rot], MODEL, 1.0)                         # 5 °/s ÷ 0.5 °/s/% = 10%
        rot.joints, rot.via_joints = [0.0] * 6, [0.0] * 6
        return start6, rot

    def test_movec_rotation_keeps_pivot(self):
        start6, rot = self._rotation()
        arm = _FakeArm(start6, mm_s_per_pct=1.0, deg_s_per_pct=0.5)
        fm.MotionRunner(arm, arm, accel_percent=50).run(rot)
        cmd, call = arm.calls[0]
        self.assertEqual(cmd, "MoveC")
        self.assertEqual((call["vel_p"], call["vel_t"], call["blendR"], call["velAccParamMode"], call["oacc"]),
                         (10.0, 10.0, 0.0, 0, 50))
        self.assertEqual((call["tool_p"], call["user_p"], call["tool_t"], call["user_t"]), (0, 0, 0, 0))
        self.assertEqual(call["desc_pos_p"], rot.via_flange)
        self.assertEqual(call["desc_pos_t"], rot.flange)
        self.assertEqual(call["joint_pos_p"], rot.via_joints)
        np.testing.assert_allclose(arm.pos, rot.flange, atol=1e-9)
        self.assertLess(rot.log["pivot_dev_max_mm"], 0.01)
        self.assertAlmostEqual(rot.log["measured"]["angular_deg_s"]["mean"], 5.0, delta=0.1)
        self.assertIn("arrived_box_ns", rot.log)

    def test_movec_off_pivot_aborts(self):
        start6, rot = self._rotation("pitch_down", deg=60.0)     # 当成直线走时光心鼓出约 16 mm
        arm = _FakeArm(start6, mm_s_per_pct=1.0, deg_s_per_pct=0.5, movec_as_line=True)   # 控制器把它当直线走
        with self.assertRaisesRegex(fm.MotionAborted, "光心偏离"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(rot)
        self.assertIsNotNone(arm.target)                         # 半路就停了
        self.assertGreater(rot.log["pivot_dev_max_mm"], fm.MAX_PIVOT_DEV_MM)

    def test_blocking_movec_still_monitored(self):
        """真控制器上 MoveC 走完才返回：放在线程里调用，运动中照样逐包采样、算光心偏离和角速度。"""
        start6, rot = self._rotation()
        arm = _FakeArm(start6, mm_s_per_pct=1.0, deg_s_per_pct=0.5, movec_blocks=True)
        fm.MotionRunner(arm, arm, accel_percent=50).run(rot)
        np.testing.assert_allclose(arm.pos, rot.flange, atol=1e-9)
        m = rot.log["measured"]
        self.assertGreater(m["samples"], 100)                     # 不是只有起点、终点两三个样本
        self.assertLess(m["max_gap_s"], 0.05)
        self.assertAlmostEqual(m["angular_deg_s"]["mean"], 5.0, delta=0.1)
        self.assertLess(rot.log["pivot_dev_max_mm"], 0.01)
        self.assertIsNotNone(rot.log["movec"]["returned_box_ns"])

    def test_blocking_movec_off_pivot_aborts(self):
        start6, rot = self._rotation("pitch_down", deg=60.0)
        arm = _FakeArm(start6, mm_s_per_pct=1.0, deg_s_per_pct=0.5, movec_as_line=True, movec_blocks=True)
        try:
            with self.assertRaisesRegex(fm.MotionAborted, "光心偏离"):
                fm.MotionRunner(arm, arm, accel_percent=100).run(rot)
            self.assertIsNotNone(arm.target)                     # MoveC 还没返回，半路就停了
        finally:
            arm.idle.set()                                       # 放掉挂着的线程

    def test_movec_error_code_aborts(self):
        start6, rot = self._rotation()
        arm = _FakeArm(start6, mm_s_per_pct=1.0, movec_error=112)
        with self.assertRaisesRegex(fm.MotionAborted, "MoveC 返回错误码 112"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(rot)

    def test_rotation_ignores_linear_transient(self):
        """同步摆动（MoveL 绕光心转）起步时位置先走：线速度远超预计，但角速度正常，不应中止。"""
        start = fm.fr5_to_matrix(self.START)
        seg = fm._move("wiggle", fm.rotate_about(start @ F_T_C, [0, 0, 1.0], 10.0), F_T_C, rate_deg_s=5.0, pivot=True)
        fm.fill_geometry([seg], start)
        fm.apply_speed([seg], MODEL, 1.0)
        arm = _FakeArm(self.START, mm_s_per_pct=1.0, deg_s_per_pct=0.5, pos_lead=4.0)
        fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        m = seg.log["measured"]["linear_mm_s"]
        self.assertGreater(m["max"], seg.expect_mm_s * fm.OVERSPEED_FACTOR + fm.OVERSPEED_MARGIN_MM_S)   # 旧规则会误报
        np.testing.assert_allclose(arm.pos, seg.flange, atol=1e-9)

    def test_rotation_overspeed_aborts(self):
        start6, rot = self._rotation()
        arm = _FakeArm(start6, mm_s_per_pct=1.0, deg_s_per_pct=1.5)   # 实际角速度是预计的 3 倍
        with self.assertRaisesRegex(fm.MotionAborted, "超过上限"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(rot)
        self.assertIsNotNone(arm.target)

    def test_angular_guard_alone(self):
        """法兰几乎不平移的转动：线速度保护管不到，角速度保护要能停下。"""
        start = fm.fr5_to_matrix(self.START)
        target = start.copy()
        target[:3, :3] = fm.axis_angle([0, 0, 1], 20.0) @ start[:3, :3]
        seg = fm.Segment("spin", "move", flange=fm.matrix_to_fr5(target), joints=[0.0] * 6, rate_deg_s=5.0)
        fm.fill_geometry([seg], start)
        fm.apply_speed([seg], MODEL, 1.0)
        self.assertEqual(seg.expect_mm_s, 0.0)                   # 行程 0：线速度保护不起作用
        arm = _FakeArm(self.START, mm_s_per_pct=1.0, deg_s_per_pct=1.5)   # 实际角速度是预计的 3 倍
        with self.assertRaisesRegex(fm.MotionAborted, "角速度"):
            fm.MotionRunner(arm, arm, accel_percent=100).run(seg)
        self.assertIsNotNone(arm.target)


class ParametersTest(unittest.TestCase):
    def test_defaults_valid_and_blur_limit_enforced(self):
        try:
            import run_session as rs
        except ImportError as exc:  # 笔记本上没装 fairino 时跳过
            self.skipTest(str(exc))
        self.assertEqual(rs.parameter_problems(), [])

        def problems_with(**kw):
            old = {k: getattr(rs, k) for k in kw}
            try:
                for k, v in kw.items():
                    setattr(rs, k, v)
                return rs.parameter_problems()
            finally:
                for k, v in old.items():
                    setattr(rs, k, v)
        self.assertTrue(any("超出允许范围" in m for m in problems_with(SPEED_MPS=0.7)))
        found = problems_with(EXPOSURE_US=10000)          # 0.5 m/s、50 °/s 时 10 ms 曝光模糊太大
        self.assertTrue(any(m.startswith("速度") and "模糊" in m for m in found), found)
        self.assertTrue(any(m.startswith("旋转") and "模糊" in m for m in found), found)
        self.assertTrue(any("CAMERA_FPS" in m for m in problems_with(CAMERA_FPS=25)))
        self.assertTrue(any("TRANSLATION_WIGGLE_SIDE" in m for m in problems_with(TRANSLATION_WIGGLE_SIDE=0.5)))
        self.assertTrue(any("PATH_SET" in m for m in problems_with(PATH_SET="fast 1")))
        self.assertTrue(any("MIN_STROKE_M" in m for m in problems_with(MIN_STROKE_M=0.4, MAX_STROKE_M=0.3)))
        self.assertTrue(any("同步摆动会超出" in m for m in problems_with(MIN_ROTATION_DEG=8.0)))
        self.assertEqual(list(rs.fl.RECORDINGS), ["up", "right", "forward", "yaw_left", "pitch_up", "roll_right"])

    def test_reduced_speed_accel_not_used(self):
        """降速 speed-test 测的加速度不用（控制器的加速度随速度变）；全速的才用。"""
        try:
            import run_session as rs
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            old = rs.SPEED_CAL
            rs.SPEED_CAL = str(path)
            try:
                for scale, used in ((0.2, False), (1.0, True)):
                    path.write_text(json.dumps({
                        "created": "test", "speed_scale": scale, "settings": {k: getattr(rs, k) for k in rs.CAL_KEYS},
                        "model": {"mm_s_per_pct": 6.0, "wiggle_deg_s_per_pct": 2.16, "ramp_s": 0.19},
                        "accel": {"test_down_mm_s2": 533.0, "test_yaw_back_deg_s2": 55.0}}), encoding="utf-8")
                    model, note = rs.load_speed_model()
                    self.assertAlmostEqual(model.mm_s_per_pct, 6.0)
                    self.assertEqual(model.accel_mm_s2 is not None, used, note)
                    self.assertEqual(model.accel_deg_s2 is not None, used, note)
                    self.assertEqual("降速" in note, not used, note)
            finally:
                rs.SPEED_CAL = old

    def test_joint_limits(self):
        """高速录制不再另设保守的关节范围（用户 10-14 定）：JOINT_MIN/MAX 是 FR5 的关节范围，运行时和控制器软限位取交集。"""
        try:
            import run_session as rs
        except ImportError as exc:
            self.skipTest(str(exc))
        traversed = {1: (106.8, 134.2), 2: (-7.8, 33.6), 3: (-101.9, -22.8),   # 10-10、10-12 录制实测
                     4: (-151.0, -99.2), 5: (-69.5, -22.1), 6: (-0.1, 0.1)}
        for j, (lo, hi) in traversed.items():
            self.assertLessEqual(rs.JOINT_MIN_DEG[j - 1], lo)
            self.assertGreaterEqual(rs.JOINT_MAX_DEG[j - 1], hi)
        self.assertEqual((rs.JOINT_MIN_DEG[1], rs.JOINT_MAX_DEG[1]), (-265.0, 85.0))   # FR5 J2
        self.assertGreaterEqual(rs.SINGULAR_MARGIN_DEG, 2.0)

        class Rpc:
            def __init__(self, ret):
                self.ret = ret

            def GetJointSoftLimitDeg(self):
                if isinstance(self.ret, Exception):
                    raise self.ret
                return self.ret
        soft = [-170.0, 170.0, -260.0, 80.0, -150.0, 150.0, -260.0, 80.0, -170.0, 170.0, -170.0, 170.0]
        lo, hi, note = rs.joint_limits(Rpc((0, soft)))
        self.assertEqual(lo, [-168.0, -258.0, -148.0, -258.0, -168.0, -168.0])        # 软限位收进 2°
        self.assertEqual(hi, [168.0, 78.0, 148.0, 78.0, 168.0, 168.0])
        self.assertIn("J2 -260～+80", note)
        for bad in ((5, None), (0, soft[:10]), (0, [10.0, -10.0] * 6), RuntimeError("no such method")):
            lo, hi, note = rs.joint_limits(Rpc(bad))
            self.assertEqual((lo, hi), (list(rs.JOINT_MIN_DEG), list(rs.JOINT_MAX_DEG)), note)
            self.assertIn("只按 JOINT_MIN/MAX_DEG", note)

    def test_glass_clearance_defaults(self):
        """真实参考起点和手眼外参下：sync-test 离右窗 ≥ GLASS_MARGIN_DEG；左窗可以拍（用户 10-13 定）：
        偏航往左转到 +25° 也通过，往右转到 −10° 碰到右窗（−7.5° 以右不行），被拒绝。"""
        try:
            import run_session as rs
        except ImportError as exc:
            self.skipTest(str(exc))
        import fr5_groundtruth as gt
        here = Path(__file__).resolve().parent
        f_t_c, _ = gt.load_handeye(here / "handeye_s003_result.json")
        start = np.array(json.loads((here / "start_pose.json").read_text(encoding="utf-8"))["base_T_camera"])
        cfg = rs.motion_config()
        note, found = rs.pose_note("sync-test", fm.plan_sync_test(start, f_t_c, cfg), start, f_t_c)
        self.assertEqual(found, [])
        self.assertIn("玻璃窗最近", note)
        up = np.array([0, 0, 1.0])
        for right, ok in ((-7.0, True), (-10.0, False)):
            segs = fm.plan_rotation(fm.yaw_about_vertical(start, right), up, 25.0 - right, f_t_c, cfg)
            self.assertEqual(not rs.pose_note("yaw_left", segs, start, f_t_c)[1], ok, right)


class GlassTest(unittest.TestCase):
    """画面和玻璃区域的角距离（方位角正 = 向左，仰角正 = 向上，相对起点水平相机）。"""

    def setUp(self):
        self.normals = fm.frustum_normals()
        w, h, f, cx, cy = fm.IMAGE_1080
        self.left_half, self.right_half = np.degrees(np.arctan(cx / f)), np.degrees(np.arctan((w - cx) / f))

    def _clear(self, r, region):
        return fm.glass_clearance_deg(r, [region], [fm.region_outline(region)], self.normals)

    def test_level_camera_matches_half_fov(self):
        """水平相机朝向 h：正前方高度上的一小块玻璃，角距离 = 方位角差 − 半视场。"""
        for h in (-5.0, 0.0, 3.0):
            r = fm.axis_angle([0, -1.0, 0], h)               # 绕“上”（-y）转 h：向左为正
            self.assertAlmostEqual(self._clear(r, (60.0, 61.0, -0.5, 0.5)), 60.0 - h - self.left_half, delta=0.02)
            self.assertAlmostEqual(self._clear(r, (-61.0, -60.0, -0.5, 0.5)), 60.0 + h - self.right_half, delta=0.02)

    def test_inside_and_behind(self):
        r = np.eye(3)
        self.assertEqual(self._clear(r, (-30.0, 30.0, -20.0, 20.0)), -90.0)      # 光轴在区域里
        self.assertLess(self._clear(r, (10.0, 12.0, 0.0, 2.0)), 0.0)              # 小块玻璃整个在画面里
        self.assertGreater(self._clear(r, (150.0, 180.0, -10.0, 10.0)), 40.0)     # 在背后

    def test_roll_widens_corners(self):
        """横滚约 29° 时画面左上角转到水平，左边最远到 atan(√(cx² + cy²)/f)，比不转时远约 3.7°。
        角落外的点按两条边平面的距离算，比真实的 2° 小一些（偏保守）。"""
        w, h, f, cx, cy = fm.IMAGE_1080
        diag = np.degrees(np.arctan(np.hypot(cx, cy) / f))
        roll = np.degrees(np.arctan(cy / cx))
        region = (diag + 2.0, diag + 3.0, -0.5, 0.5)
        self.assertAlmostEqual(self._clear(np.eye(3), region), diag + 2.0 - self.left_half, delta=0.02)
        best = min(self._clear(fm.axis_angle([0, 0, 1.0], s * roll), region) for s in (-1, 1))
        self.assertLessEqual(best, 2.0 + 1e-6)
        self.assertGreater(best, 1.5)

    def test_plan_clearance_follows_heading(self):
        """偏航录制往左转得越多，离左窗越近（拍到时为负，等于伸进画面的深度）；结果出在左端的那几段。"""
        c = fm.level_camera(NOW)
        region = [(57.4, 180.0, -90.0, 90.0)]
        near = []
        for heading in (-2.0, 0.0, 2.0):
            segs = fm.plan_rotation(fm.yaw_about_vertical(c, heading - 5.0), [0, 0, 1.0], 10.0, F_T_C, CFG)
            clear, where = fm.glass_clearance(segs, c, F_T_C, region)
            near.append(clear)
            self.assertIn(where, ("wiggle_start_1", "wiggle_start_3", "rotate", "wiggle_end_2", "wiggle_end_4"))
        self.assertGreater(near[0], 0.0)                                  # 左端 +3°：差约 1.1°，没拍到
        self.assertAlmostEqual(near[1], 57.4 - 5.0 - self.left_half, delta=0.02)
        self.assertAlmostEqual(near[1] - near[2], 2.0, delta=0.02)


class GlassParallaxTest(unittest.TestCase):
    def test_translation_parallax(self):
        """右边 −60° 处一小块玻璃，离起点 7 m：相机向右平移 0.3 m 后，它在画面里的方位变成 −atan((7 sin60 − 0.3)/(7 cos60))，
        离画面右边缘更近。不给距离时只看朝向，平移不改变结果。"""
        c = fm.level_camera(NOW)
        right = c.copy()
        right[:3, 3] += c[:3, 0] * 0.3
        segs = [fm._move("goto_start", c, F_T_C, 50.0, recorded=False), fm._move("translate", right, F_T_C, 50.0)]
        region = [(-61.0, -60.0, -0.5, 0.5)]
        w, h, f, cx, cy = fm.IMAGE_1080
        half = np.degrees(np.arctan((w - cx) / f))
        flat, _ = fm.glass_clearance(segs, c, F_T_C, region)
        near, where = fm.glass_clearance(segs, c, F_T_C, region, distance_m=7.0)
        az = np.degrees(np.arctan((7 * np.sin(np.radians(60)) - 0.3) / (7 * np.cos(np.radians(60)))))
        self.assertAlmostEqual(flat, 60.0 - half, delta=0.02)
        self.assertAlmostEqual(near, az - half, delta=0.02)
        self.assertEqual(where, "translate")
        far, _ = fm.glass_clearance(segs, c, F_T_C, region, distance_m=1e6)     # 很远：视差可以忽略
        self.assertAlmostEqual(far, flat, delta=1e-3)


class _FakeZedProc:
    """zed_record 子进程的替身：crash_rc 不为 None 时表示停止前已经退出。"""

    def __init__(self, crash_rc=None):
        self.returncode, self.signals = crash_rc, []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


class ZedRecordTest(unittest.TestCase):
    def _stopped(self, d: Path, proc: _FakeZedProc) -> dict:
        from zed_capture import ZedRecord
        zed = ZedRecord(d / "zed_record_bin", d, 10000, 1000)
        zed.proc = proc
        zed.stop()
        return zed.info

    def test_stop_records_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            proc = _FakeZedProc()
            info = self._stopped(d, proc)
            self.assertEqual(proc.signals, [signal.SIGINT])
            self.assertEqual((info["returncode"], info["exited_before_stop"]), (0, False))
            info = self._stopped(d, _FakeZedProc(crash_rc=-11))
            self.assertEqual((info["returncode"], info["exited_before_stop"]), (-11, True))

    def test_fps_argument(self):
        from zed_capture import ZedRecord
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertNotIn("--fps", ZedRecord(d / "bin", d, 10000, 1000).args)       # 15 fps：旧版 zed_record 也能用
            args = ZedRecord(d / "bin", d, 2500, 13000, 30).args
            self.assertEqual(args[args.index("--fps") + 1], "30")
            self.assertEqual(args[args.index("--exposure-us") + 1], "2500")

    def test_zed_failure(self):
        """10-13 10:50 zed_record 停止时段错误（返回码 -11），没写 time_meta.txt，run_session 却当成正常。"""
        try:
            import run_session as rs
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(rs.zed_failure(None, d))                      # goto-start 不开相机
            self.assertIsNone(rs.zed_failure({"start_box_ns": 1}, d))       # 相机没启动就中止了
            ok = {"returncode": 0, "exited_before_stop": False}
            self.assertIn("time_meta.txt", rs.zed_failure(ok, d))
            (d / "time_meta.txt").write_text("exposure_us=10000\n")
            self.assertIsNone(rs.zed_failure(ok, d))
            msg = rs.zed_failure({"returncode": -11, "exited_before_stop": False}, d)
            self.assertIn("-11（SIGSEGV）", msg)
            self.assertIn("没等到停止", rs.zed_failure({"returncode": 0, "exited_before_stop": True}, d))
            self.assertIn("返回码 1", rs.zed_failure({"returncode": 1, "exited_before_stop": True}, d))


if __name__ == "__main__":
    unittest.main()
