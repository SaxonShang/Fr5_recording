"""fr5_limits.py（拖动标记 → 录制路径）的离线测试（不连机械臂，逆解用假的）。

    cd ~/ROBOT/fr5_record && python3 -B -m unittest test_fr5_limits
"""

import argparse
import contextlib
import io
import json
import math
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import fr5_limits as fl
import fr5_motion as fm
from test_fr5_motion import CFG, F_T_C, NOW

REF = fm.level_camera(NOW)                                   # 参考起点（水平）
REF_Q = [119.5, 14.4, -74.3, -120.0, -44.8, 0.0]             # 参考起点的关节角（常用构型）
UP = np.array([0.0, 0.0, 1.0])


def shifted(camera, fwd=0.0, left=0.0, up=0.0):
    """相机沿参考起点的（前, 左, 上）平移 mm，姿态不变。"""
    out = camera.copy()
    out[:3, 3] += (fwd * REF[:3, 2] - left * REF[:3, 0] + up * UP) / 1000.0
    return out


def heading(camera, deg):
    """绕过光心的竖直轴转 deg（正 = 向左）。"""
    return fm.yaw_about_vertical(camera, deg)


def tilted(camera, kind, deg):
    """绕光心转：pitch 绕相机 x（正 = 向上），roll 绕光轴（正 = 顶部向右）。"""
    return fm.rotate_about(camera, camera[:3, 0] if kind == "pitch" else camera[:3, 2], deg)


class FakeIK:
    """连续的假逆解：关节 = 常用构型 + (法兰位姿 − 参考起点的法兰位姿) / 10（角度差取 ±180°）。
    fail(camera) 为真时返回错误码；warp(q, camera) 可以改关节角。"""

    def __init__(self, fail=None, warp=None):
        self.pose0 = np.array(fm.matrix_to_fr5(REF @ np.linalg.inv(F_T_C)))
        self.fail, self.warp, self.calls = fail, warp, 0

    def GetInverseKinRef(self, typ, pose, ref):
        self.calls += 1
        camera = fm.fr5_to_matrix(pose) @ F_T_C
        if self.fail and self.fail(camera):
            return 112, None
        d = np.asarray(pose, dtype=float) - self.pose0
        d[3:] = (d[3:] + 180.0) % 360.0 - 180.0
        q = np.asarray(REF_Q) + d / 10.0
        if self.warp:
            q = self.warp(q, camera)
        return 0, list(map(float, q))


def limits(**kw):
    base = dict(joint_min=[-1e4] * 6, joint_max=[1e4] * 6, max_joint_step_deg=5.0,
                min_camera_z_m=0.0, max_joint_speed_deg_s=150.0, speed=500.0, accel=None,
                min_length=100.0, max_length=1000.0)
    base.update(kw)
    return fl.FitLimits(**base)


def joints_at(camera):
    return FakeIK().GetInverseKinRef(0, fm.matrix_to_fr5(camera @ np.linalg.inv(F_T_C)), REF_Q)[1]


def updown(down_up=-220.0, top=+250.0):
    """一条上下路径：down、up 两个标记在参考起点左边 300 mm 处的竖直线上。"""
    down, up = shifted(REF, left=300.0, up=down_up), shifted(REF, left=300.0, up=top)
    return fl.derive_path("up_down", down, up, REF, 30.0, 3.0), down, up


class DeriveTest(unittest.TestCase):
    def test_up_down(self):
        down = heading(shifted(REF, fwd=20.0, left=300.0, up=-200.0), 4.0)
        up = heading(shifted(REF, fwd=40.0, left=310.0, up=250.0), 2.0)
        path = fl.derive_path("up_down", down, up, REF, 30.0, 3.0)
        self.assertEqual((path.kind, path.unit), ("translate", "mm"))
        np.testing.assert_allclose(path.axis, UP)
        self.assertAlmostEqual(path.heading_deg, 3.0, places=6)                 # 两个标记朝向的平均
        np.testing.assert_allclose(path.base[:3, 3], (down[:3, 3] + up[:3, 3]) / 2, atol=1e-12)
        np.testing.assert_allclose(path.marks_s, (-225.0, 225.0), atol=1e-9)
        self.assertEqual((path.lo, path.hi), (path.marks_s[0] + 30.0, path.marks_s[1] - 30.0))
        self.assertEqual((path.warnings, path.problems), ([], []))
        for s in (-195.0, 0.0, 195.0):                                           # 竖直平移，姿态不变、相机水平
            cam = path.pose(s)
            np.testing.assert_allclose(cam[:3, 3], path.base[:3, 3] + UP * s / 1000, atol=1e-12)
            np.testing.assert_allclose(fl.angles(REF, cam), (3.0, 0.0, 0.0), atol=1e-9)
        # 拖得歪：水平偏了 60 mm、相机低头 7°
        bad = fl.derive_path("up_down", down, tilted(shifted(up, left=60.0), "pitch", -7.0), REF, 30.0, 3.0)
        self.assertEqual(len(bad.warnings), 2, bad.warnings)
        self.assertEqual(bad.problems, [])

    def test_forward_backward_follows_drag_direction(self):
        """前后：方向取两个标记连线的水平分量，光轴沿这个方向；路径水平地经过两个标记，高度取平均。"""
        back = shifted(REF, fwd=-200.0, left=50.0, up=10.0)
        front = shifted(REF, fwd=250.0, left=50.0 + 450.0 * math.tan(math.radians(8.0)), up=50.0)
        path = fl.derive_path("forward_backward", back, front, REF, 30.0, 3.0)
        self.assertAlmostEqual(path.heading_deg, 8.0, places=6)
        self.assertAlmostEqual(fl.heading_of(REF, path.axis), 8.0, places=6)
        np.testing.assert_allclose(path.base[:3, 2], path.axis, atol=1e-12)     # 光轴沿运动方向
        self.assertAlmostEqual(path.axis[2], 0.0, places=12)
        length = 450.0 / math.cos(math.radians(8.0))
        np.testing.assert_allclose(path.marks_s, (-length / 2, length / 2), atol=1e-6)
        for mark, s in ((back, path.marks_s[0]), (front, path.marks_s[1])):
            d = (path.pose(s)[:3, 3] - mark[:3, 3]) * 1000
            np.testing.assert_allclose(d[:2], 0.0, atol=1e-6)                   # 水平方向上正好过标记
            self.assertAlmostEqual(abs(d[2]), 20.0, places=6)                    # 高度取平均
        self.assertTrue(any("高差 +40 mm" in w for w in path.warnings), path.warnings)
        self.assertTrue(any("朝向 +8.0°" in w for w in path.warnings), path.warnings)   # 标记本身朝前，差 8°

    def test_left_right_camera_x_along_motion(self):
        left, right = shifted(REF, left=200.0, up=30.0), shifted(REF, left=-250.0, up=30.0)
        path = fl.derive_path("left_right", left, right, REF, 30.0, 3.0)
        np.testing.assert_allclose(path.axis, REF[:3, 0], atol=1e-12)             # 从左到右 = 相机 x
        self.assertAlmostEqual(path.heading_deg, 0.0, places=9)
        np.testing.assert_allclose(path.base[:3, :3], REF[:3, :3], atol=1e-12)
        np.testing.assert_allclose(path.marks_s, (-225.0, 225.0), atol=1e-9)
        self.assertEqual(path.warnings, [])

    def test_yaw(self):
        pivot = shifted(REF, left=50.0)
        right = heading(shifted(pivot, fwd=20.0), -15.0)
        left = heading(pivot, 28.0)
        path = fl.derive_path("yaw", right, left, REF, 30.0, 3.0)
        self.assertEqual((path.kind, path.unit, path.heading_deg), ("rotate", "°", None))
        np.testing.assert_allclose(path.marks_s, (-15.0, 28.0), atol=1e-9)
        np.testing.assert_allclose((path.lo, path.hi), (-12.0, 25.0), atol=1e-9)
        np.testing.assert_allclose(path.axis, UP)
        for s in (-12.0, 0.0, 25.0):
            cam = path.pose(s)
            np.testing.assert_allclose(cam[:3, 3], (right[:3, 3] + left[:3, 3]) / 2, atol=1e-12)   # 绕两个光心的中点
            np.testing.assert_allclose(fl.angles(REF, cam), (s, 0.0, 0.0), atol=1e-9)
        self.assertEqual(path.warnings, [])
        sloppy = fl.derive_path("yaw", tilted(right, "pitch", 8.0), shifted(left, left=90.0), REF, 30.0, 3.0)
        self.assertTrue(any("俯仰 +8.0°" in w for w in sloppy.warnings), sloppy.warnings)
        self.assertTrue(any("光心相距" in w for w in sloppy.warnings), sloppy.warnings)

    def test_pitch_and_roll(self):
        centre = heading(shifted(REF, up=40.0), -5.0)
        for pair, kind, lo, hi in (("pitch", "pitch", -12.0, 18.0), ("roll", "roll", -40.0, 45.0)):
            neg = tilted(heading(centre, -1.5), kind, lo)
            pos = tilted(heading(centre, +1.5), kind, hi)
            path = fl.derive_path(pair, neg, pos, REF, 30.0, 3.0)
            self.assertAlmostEqual(path.heading_deg, -5.0, places=6)            # 两个标记朝向的平均
            np.testing.assert_allclose(path.marks_s, (lo, hi), atol=0.05)
            k = 1 if pair == "pitch" else 2
            for s in (path.lo, 0.0, path.hi):
                a = fl.angles(REF, path.pose(s))
                self.assertAlmostEqual(a[k], s, places=6)
                self.assertAlmostEqual(a[0], -5.0, places=6)
                self.assertAlmostEqual(a[3 - k], 0.0, places=6)
                np.testing.assert_allclose(path.pose(s)[:3, 3], centre[:3, 3], atol=1e-12)
            self.assertAlmostEqual(path.axis[2], 0.0, places=12)                  # 俯仰轴、横滚轴都水平
            self.assertEqual(path.warnings, [], pair)

    def test_problems(self):
        path, down, up = updown()
        self.assertEqual(fl.derive_path("up_down", up, down, REF, 30.0, 3.0).problems[0][:4], "down")   # 记反了
        close = fl.derive_path("up_down", down, shifted(down, up=50.0), REF, 30.0, 3.0)
        self.assertTrue(any("没有空间" in m for m in close.problems), close.problems)
        same = fl.derive_path("left_right", down, shifted(down, up=100.0), REF, 30.0, 3.0)
        self.assertTrue(any("定不出方向" in m for m in same.problems), same.problems)

    def test_config(self):
        flipped = [138.8, -58.6, 113.5, -235.4, -84.8, 15.0]                       # 10-14 拖到的另一种构型
        self.assertEqual(fl.config_of(REF_Q), (-1, -1))
        self.assertEqual(fl.config_of(flipped), (1, -1))
        self.assertEqual(fl.config_of([138.8, 30.0, -60.0, -250.0, -30.0, 140.0]), (-1, -1))   # J4、J6 转多少都不算
        self.assertEqual(fl.describe_config(flipped), "肘部 J3 +113.5°、腕部 J5 -84.8°")
        self.assertIsNone(fl.singular_problem(REF_Q, 5.0))
        self.assertIn("肘部", fl.singular_problem([0, 0, -3.0, 0, -40.0, 0], 5.0))
        self.assertIn("腕部", fl.singular_problem([0, 0, -40.0, 0, 4.9, 0], 5.0))


class FitTest(unittest.TestCase):
    def setUp(self):
        self.path, self.down, self.up = updown()
        self.ref = joints_at(self.down)

    def fit(self, ik=None, path=None, **kw):
        return fl.fit_path(ik or FakeIK(), path or self.path, F_T_C, self.ref, limits(**kw))

    def test_whole_path(self):
        fit = self.fit()
        self.assertEqual((fit.a, fit.b, fit.problems), (self.path.lo, self.path.hi, []))
        self.assertTrue(all(e.startswith("标记收进余量") for e in fit.ends), fit.ends)
        self.assertAlmostEqual(fit.max_joint_speed_deg_s, 0.1 * 500.0, places=6)   # 每 mm J3 变 0.1°

    def test_joint_range(self):
        """关节 J3 = 常用构型 + Δz/10：上限卡在路径中间偏上，正端停在上限以内的最后一个采样点。"""
        z_mid = self.path.base[2, 3]
        q3 = joints_at(self.path.pose(100.0))[2]
        fit = self.fit(joint_max=[1e4, 1e4, q3 + 0.05, 1e4, 1e4, 1e4])
        self.assertAlmostEqual(fit.a, self.path.lo)
        self.assertLessEqual(fit.b, 100.0 + 1e-6)
        self.assertGreater(fit.b, 90.0)
        self.assertIn("J[3]", fit.ends[1])
        self.assertGreater(z_mid, 0)

    def test_longest_run_and_jump(self):
        z = lambda s: self.path.pose(s)[2, 3]                                     # noqa: E731
        fit = self.fit(FakeIK(fail=lambda c: z(-120.0) < c[2, 3] < z(-90.0)))    # 下半段有一截逆解失败
        self.assertGreater(fit.a, -90.0)
        self.assertAlmostEqual(fit.b, self.path.hi)
        self.assertIn("逆解失败", fit.ends[0])
        jump = FakeIK(warp=lambda q, c: q + (np.r_[0, 0, 0, 30.0, 0, 0] if c[2, 3] > z(60.0) else 0.0))
        fit = self.fit(jump)                                                      # 上半段 J4 跳了 30°
        self.assertLess(fit.b, 60.0 + 1e-6)
        self.assertIn("跳了一大步", fit.ends[1])

    def test_glass_and_height(self):
        z = lambda s: self.path.pose(s)[2, 3]                                     # noqa: E731
        fit = self.fit(glass=lambda c: 0.5 if c[2, 3] > z(150.0) else 5.0)
        self.assertLessEqual(fit.b, 150.0 + 1e-6)
        self.assertIn("右窗", fit.ends[1])
        fit = self.fit(min_camera_z_m=z(-100.0))
        self.assertGreaterEqual(fit.a, -100.0 - 1e-6)
        self.assertIn("光心高度", fit.ends[0])

    def test_joint_speed_trims_the_fast_end(self):
        """顶上 100 mm 里 J2 变得很快（像靠近奇异位形）：按全程匀速时收掉这一截；加速度小的话那里还在减速，不用收。"""
        top = self.path.pose(self.path.hi - 100.0)[2, 3]
        warp = FakeIK(warp=lambda q, c: q + np.r_[0, 0.4 * max(0.0, c[2, 3] - top) * 1000.0, 0, 0, 0, 0])
        fit = self.fit(warp)                                                      # 0.4 °/mm × 500 mm/s = 200 °/s
        self.assertAlmostEqual(fit.a, self.path.lo)
        self.assertLessEqual(fit.b, self.path.hi - 100.0 + 1e-6)
        self.assertIn("关节速度", fit.ends[1])
        self.assertLessEqual(fit.max_joint_speed_deg_s, 150.0)
        slow = self.fit(warp, accel=100.0)                                        # 最后 100 mm 速度 ≤ √(2·100·100) ≈ 141 mm/s
        self.assertAlmostEqual(slow.b, self.path.hi)

    def test_wiggle_at_ends(self):
        """平移两端要能往左摆 10°：上面一截摆不过去（逆解失败）时，正端往里收到能摆的地方。"""
        top = self.path.pose(80.0)[2, 3]
        base_r = self.path.base[:3, :3]

        def fail(c):
            turned = math.degrees(math.acos(min(1.0, (np.trace(base_r.T @ c[:3, :3]) - 1) / 2)))
            return turned > 5.0 and c[2, 3] > top + 1e-9
        fit = self.fit(FakeIK(fail=fail), wiggle_deg=10.0, wiggle_side=1)
        self.assertLessEqual(fit.b, 80.0 + 1e-6)
        self.assertGreater(fit.b, 70.0)
        self.assertIn("同步摆动", fit.ends[1])
        self.assertAlmostEqual(fit.a, self.path.lo)

    def test_length_limits(self):
        fit = self.fit(max_length=300.0)
        self.assertAlmostEqual(fit.length, 300.0, delta=1e-6)
        self.assertIn("行程上限", fit.ends[0] + fit.ends[1])
        self.assertTrue(self.fit(min_length=500.0).problems)
        bad = fl.derive_path("up_down", self.up, self.down, REF, 30.0, 3.0)
        self.assertTrue(fl.fit_path(FakeIK(), bad, F_T_C, self.ref, limits()).problems)

    def test_yaw_fit_and_config_change(self):
        """偏航：右边到右窗为止（这里假设方位角 < −7.5° 拍到）；关节换了构型的地方不能走。"""
        path = fl.derive_path("yaw", heading(REF, -20.0), heading(REF, 40.0), REF, 30.0, 3.0)
        ref = joints_at(path.pose(path.lo))
        fit = fl.fit_path(FakeIK(), path, F_T_C, ref,
                          limits(min_length=15.0, max_length=180.0, speed=50.0,
                                 glass=lambda c: 0.0 if fl.angles(REF, c)[0] < -7.5 else 5.0))
        self.assertAlmostEqual(fit.a, -7.0, delta=1e-6)
        self.assertAlmostEqual(fit.b, path.hi)
        self.assertIn("右窗", fit.ends[0])
        flipped = FakeIK(warp=lambda q, c: q * np.r_[1, 1, -1 if fl.angles(REF, c)[0] > 30.0 else 1, 1, 1, 1])
        fit = fl.fit_path(flipped, path, F_T_C, ref, limits(min_length=15.0, max_length=180.0, speed=50.0))
        self.assertLessEqual(fit.b, 30.0 + 1e-6)
        self.assertIn("构型", fit.ends[1])

    def test_singular_margin(self):
        """往左转时 J5 往 0 走（腕部奇异位形）：J5 离 0 不到 5° 的地方不能走。"""
        path = fl.derive_path("yaw", heading(REF, -5.0), heading(REF, 40.0), REF, 30.0, 3.0)
        to_zero = FakeIK(warp=lambda q, c: q + np.r_[0, 0, 0, 0, 44.8 * max(0.0, fl.angles(REF, c)[0]) / 37.0, 0])
        fit = fl.fit_path(to_zero, path, F_T_C, joints_at(path.pose(path.lo)),
                          limits(min_length=15.0, max_length=180.0, speed=1.0))
        self.assertIn("腕部奇异", fit.ends[1])
        self.assertTrue(30.0 < fit.b < 34.0, fit.b)                               # J5 ≈ −44.8 + 44.8·s/37 = −5 → s ≈ 33（采样每 1°）

    def test_two_configs_try_both(self):
        """down、up 两个标记构型不同（肘部翻了）：直线换不了构型。下面的构型只能到 −50 mm 处（再往上逆解失败），
        上面的构型能从顶上走到 −150 mm 处：两种都试，取上面的，并写明。"""
        z = lambda s: self.path.pose(s)[2, 3]                                     # noqa: E731

        class TwoBranches(FakeIK):
            def GetInverseKinRef(self, typ, pose, ref):
                elbow_up = ref[2] > 0                                             # 参考关节角在哪个构型，就解哪个
                camera = fm.fr5_to_matrix(pose) @ F_T_C
                if (camera[2, 3] > z(-50.0) + 1e-9) if not elbow_up else (camera[2, 3] < z(-150.0) - 1e-9):
                    return 112, None
                err, q = super().GetInverseKinRef(typ, pose, ref)
                return err, [q[0], q[1], -q[2] if elbow_up else q[2], *q[3:]]
        neg_q = joints_at(self.down)
        pos_q = list(joints_at(self.up))
        pos_q[2] = -pos_q[2]
        fit = fl.fit_best(TwoBranches(), self.path, F_T_C, neg_q, pos_q, limits())
        self.assertEqual(fit.seed, "up")
        self.assertAlmostEqual(fit.b, self.path.hi)
        self.assertTrue(-150.0 - 1e-6 <= fit.a <= -140.0, fit.a)
        self.assertGreater(fit.joints_a[2], 0)                                     # 整条都在 up 的构型里
        self.assertEqual(len(fit.notes), 1)
        self.assertIn("按 down 的构型能走", fit.notes[0])
        self.assertIn("取 up 的", fit.notes[0])
        same = fl.fit_best(TwoBranches(), self.path, F_T_C, neg_q, neg_q, limits(min_length=50.0))
        self.assertEqual((same.seed, same.notes), ("down", []))                   # 构型相同：只试一次
        self.assertLessEqual(same.b, -50.0 + 1e-6)
        self.assertIn("构型：和 up 标记相同", fl.describe_path(self.path, fit))


class RecordingTest(unittest.TestCase):
    def test_translation_direction(self):
        path, _, _ = updown()
        fit = fl.Fit(-150.0, 180.0, ["", ""], None)
        for name, (s0, s1) in (("up", (-150.0, 180.0)), ("down", (180.0, -150.0))):
            by = {s.name: s for s in fl.recording_segments(name, path, fit, F_T_C, CFG)}
            np.testing.assert_allclose(by["goto_start"].camera, path.pose(s0), atol=1e-12)
            np.testing.assert_allclose(by["translate"].camera, path.pose(s1), atol=1e-12)
            self.assertFalse(by["goto_start"].recorded)
            self.assertNotIn("via_start", by)                                    # 不回参考起点

    def test_rotation_direction(self):
        """偏航、俯仰、横滚两个方向：起点在 s0、转到 s1，转角为正，转轴带方向；姿态范围就是 [a, b]。"""
        centre = heading(shifted(REF, up=40.0), -5.0)
        pairs = {"yaw": (heading(centre, -10.0), heading(centre, 30.0)),
                 "pitch": (tilted(centre, "pitch", -15.0), tilted(centre, "pitch", 20.0)),
                 "roll": (tilted(centre, "roll", -40.0), tilted(centre, "roll", 50.0))}
        key = {"yaw": "heading", "pitch": "pitch", "roll": "roll"}
        for pair, (neg, pos) in pairs.items():
            path = fl.derive_path(pair, neg, pos, REF, 30.0, 3.0)
            fit = fl.Fit(path.lo, path.hi, ["", ""], None)
            for name, (p_, sign) in fl.RECORD_PAIR.items():
                if p_ != pair:
                    continue
                segs = fl.recording_segments(name, path, fit, F_T_C, CFG)
                by = {s.name: s for s in segs}
                rot = by["rotate"]
                start = path.pose(fit.a if sign > 0 else fit.b)
                np.testing.assert_allclose(by["goto_start"].camera, start, atol=1e-12)
                np.testing.assert_allclose(rot.camera, path.pose(fit.b if sign > 0 else fit.a), atol=1e-9)
                self.assertAlmostEqual(rot.arc_deg, fit.length, places=9)
                np.testing.assert_allclose(rot.arc_axis, sign * path.axis, atol=1e-12)
                span = fm.pose_span(segs, REF, F_T_C)[key[pair]]
                np.testing.assert_allclose(span, (fit.a, fit.b), atol=1e-6, err_msg=name)
                # 录制名就是转的方向：yaw_left 往左、pitch_up 往上、roll_right 顶部往右（角度增大）
                k = {"yaw": 0, "pitch": 1, "roll": 2}[pair]
                turned = fl.angles(REF, rot.camera)[k] - fl.angles(REF, start)[k]
                self.assertEqual(turned > 0, name in ("yaw_left", "pitch_up", "roll_right"), name)

    def test_defaults(self):
        self.assertEqual(set(fl.RECORD_PAIR), set(fl.MARK_NAMES))
        self.assertEqual(sorted(n for ends in fl.PAIRS.values() for n in ends), sorted(fl.MARK_NAMES))
        self.assertEqual(sorted(fl.RECORD_PAIR[n][0] for n in fl.RECORDINGS), sorted(fl.PAIRS))   # 每对一条
        path, _, _ = updown()
        text = fl.describe_path(path, fl.Fit(path.lo, path.hi, ["标记收进余量", "关节速度"], 120.0))
        self.assertIn("上下平移", text)
        self.assertIn("关节最快 120 °/s", text)


class _FakeRecorder:
    """run_session 用的假录制器：机械臂停在 ARM["flange"]，逆解用 FakeIK（以真实参考起点为常用构型）。"""
    ARM = {}

    def __init__(self, data_dir, ip):
        pass

    def start(self):
        pass

    def stop(self, write=True):
        pass

    def open_rpc(self):
        return self.ARM["ik"]

    @property
    def latest_monotonic(self):
        return time.monotonic()

    def wait_state(self, count, timeout):
        pkg = types.SimpleNamespace(flange_cur_pos=list(self.ARM["flange"]), jt_cur_pos=list(self.ARM["joints"]),
                                    main_code=0, sub_code=0, EmergencyStop=0, safety_stop0_state=0,
                                    safety_stop1_state=0, rbtEnableState=1, tool=0, user=0, robot_state=1)
        return count + 1, pkg


class RunSessionTest(unittest.TestCase):
    """mark → limits → plan 走一遍（假机械臂、假逆解，真实的参考起点、手眼外参和右窗）。要能 import run_session（Box 上）。"""

    def setUp(self):
        try:
            import run_session as rs
        except ImportError as exc:  # 笔记本上没装 fairino 时跳过
            self.skipTest(str(exc))
        import fr5_groundtruth as gt
        self.rs, here = rs, Path(__file__).resolve().parent
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.f_t_c, _ = gt.load_handeye(here / "handeye_s003_result.json")
        self.start = json.loads((here / "start_pose.json").read_text(encoding="utf-8"))
        self.ref = np.array(self.start["base_T_camera"])
        saved = {k: getattr(rs, k) for k in ("LIMIT_POSES", "START_POSE", "HANDEYE", "SPEED_CAL", "StateRecorder",
                                             "JOINT_MIN_DEG", "JOINT_MAX_DEG", "SESSIONS_DIR", "camera_users")}
        self.addCleanup(lambda: [setattr(rs, k, v) for k, v in saved.items()])
        rs.LIMIT_POSES, rs.SPEED_CAL = str(tmp / "limit_poses.json"), str(tmp / "none.json")
        rs.START_POSE, rs.HANDEYE = str(here / "start_pose.json"), str(here / "handeye_s003_result.json")
        rs.StateRecorder = _FakeRecorder
        rs.SESSIONS_DIR, rs.camera_users = str(tmp / "sessions"), lambda: []
        rs.JOINT_MIN_DEG, rs.JOINT_MAX_DEG = [-1e4] * 6, [1e4] * 6   # 假逆解的关节角没有物理意义，只查构型
        flange0 = np.array(fm.matrix_to_fr5(self.ref @ np.linalg.inv(self.f_t_c)))
        q0 = np.array(self.start["joints"])

        class IK:
            def GetDefaultTransVel(self):
                return 0, 1000.0

            def GetInverseKinRef(self, typ, pose, ref):
                d = np.asarray(pose, dtype=float) - flange0
                d[3:] = (d[3:] + 180.0) % 360.0 - 180.0
                return 0, list(map(float, q0 + d / 10.0))
        self.ik = IK()
        self.marks = {
            "down": self.moved(up=-150.0), "up": self.moved(up=300.0, left=20.0),
            "left": self.moved(left=250.0), "right": self.moved(left=-200.0, up=15.0),
            "backward": self.moved(fwd=-150.0), "forward": self.moved(fwd=200.0),
            "yaw_right": fm.yaw_about_vertical(self.ref, -6.0), "yaw_left": fm.yaw_about_vertical(self.ref, 30.0),
            "pitch_down": tilted(self.ref, "pitch", -15.0), "pitch_up": tilted(self.ref, "pitch", 15.0),
            "roll_left": tilted(self.ref, "roll", -45.0), "roll_right": tilted(self.ref, "roll", 45.0)}

    def moved(self, fwd=0.0, left=0.0, up=0.0):
        out = self.ref.copy()
        out[:3, 3] += (fwd * self.ref[:3, 2] - left * self.ref[:3, 0] + up * UP) / 1000.0
        return out

    def park(self, camera):
        flange = camera @ np.linalg.inv(self.f_t_c)
        _FakeRecorder.ARM.update(flange=fm.matrix_to_fr5(flange), ik=self.ik,
                                 joints=self.ik.GetInverseKinRef(0, fm.matrix_to_fr5(flange), [])[1])

    def run_cmd(self, fn, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fn(*args)
        return code, out.getvalue()

    def test_mark_limits_plan(self):
        rs = self.rs
        mark = lambda name: self.run_cmd(rs.cmd_mark, argparse.Namespace(direction=name))   # noqa: E731
        with mock.patch.object(rs.time, "sleep"):
            for name, cam in self.marks.items():
                self.park(cam)
                code, text = mark(name)
                self.assertEqual(code, 0, text)
            self.assertIn("12 个标记都记了", text)
            self.assertIn("横滚：绕光心", text)                                   # 记完一对就列出这一对的路径
            # 另一种构型（10-14 拖到的那种）也记（用户 10-14 定）；这一对构型不同时会提醒
            _FakeRecorder.ARM["joints"] = [138.8, -58.6, 113.5, -235.4, -84.8, 15.0]
            code, text = mark("up")
            self.assertEqual(code, 0, text)
            self.assertIn("肘部 J3 +113.5°", text)
            self.assertIn("两个标记构型不同", text)
            self.park(self.marks["up"])                                       # 再按常用构型记一次，后面的 plan 用它
            self.assertEqual(mark("up")[0], 0)
        data = json.loads(Path(rs.LIMIT_POSES).read_text(encoding="utf-8"))
        self.assertEqual(sorted(data["marks"]), sorted(fl.MARK_NAMES))

        code, text = self.run_cmd(rs.cmd_limits, argparse.Namespace())
        self.assertEqual(code, 0, text)
        for label in ("上下平移", "左右平移", "前后平移", "偏航", "俯仰", "横滚"):
            self.assertIn(label, text)

        self.park(self.marks["up"])                                           # 停在最后拖到的地方，不在参考起点
        code, text = self.run_cmd(rs.cmd_plan, argparse.Namespace(names=[], scale=1.0), [])
        self.assertEqual(code, 0, text)
        self.assertIn("全部通过", text)
        for name in fl.RECORDINGS:
            self.assertIn(f"=== {name}：通过", text)
        # 机械臂在另一种构型（10-14 拖到的那种）：录制和去参考起点（sync-test）都要先手动拖回去
        flipped = [138.8, -58.6, 113.5, -235.4, -84.8, 15.0]
        saved_joints, _FakeRecorder.ARM["joints"] = _FakeRecorder.ARM["joints"], flipped
        code2, text2 = self.run_cmd(rs.cmd_plan, argparse.Namespace(names=["up", "sync-test"], scale=1.0), [])
        self.assertEqual(code2, 1)
        self.assertIn("这条录制的起点是肘部 J3", text2)
        self.assertIn("参考起点（set-start 时的构型）是肘部 J3 -74.3°", text2)
        _FakeRecorder.ARM["joints"] = saved_joints
        self.assertNotIn("via_start", text)
        # 偏航：右边被右窗挡住（拖到 −6°、收进 3° 是 −3°，没到右窗），左边收进 3°
        self.assertIn("方位角 -3.0 → +27.0°", text)
        # 上下：标记 −150 / +300 mm，两端各收进 30 mm：390 mm 的竖直行程
        self.assertIn("共 390.0 mm", text)

        # speed-test 走上下这条路径的全程；record 从当前位姿直接移到录制起点。都在确认处回答 no，没有运动
        args = argparse.Namespace(scale=0.2, yes=False, tag="")
        with mock.patch("builtins.input", return_value="no"):
            code, text = self.run_cmd(rs.cmd_speed_test, args)
            self.assertEqual(code, 1)
            self.assertIn("test_up          移动 390.0 mm", text)
            self.assertIn("没有确认", text)
            Path(rs.SPEED_CAL).write_text(json.dumps({          # 有了速度标定才能 record
                "created": "test", "settings": {k: getattr(rs, k) for k in rs.CAL_KEYS},
                "model": {"mm_s_per_pct": 6.0, "wiggle_deg_s_per_pct": 2.08, "ramp_s": 0.3},
                "accel": {"test_down_mm_s2": 2500.0, "test_yaw_back_deg_s2": 300.0}}), encoding="utf-8")
            code, text = self.run_cmd(rs.cmd_motion, args, "record", "roll_right")
            self.assertEqual(code, 1)
            self.assertIn("横滚 -42.0 → +42.0°", text)
            self.assertIn("rotate           圆弧", text)
            self.assertIn("没有确认", text)
        self.assertEqual(list(Path(rs.SESSIONS_DIR).iterdir()), [])          # 没有运动就不留录制目录


if __name__ == "__main__":
    unittest.main()
