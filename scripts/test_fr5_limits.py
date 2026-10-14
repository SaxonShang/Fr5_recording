"""fr5_limits.py（拖动标记）和 fr5_motion 先回起点（via_start）的离线测试（不连机械臂）。

    cd ~/ROBOT/fr5_record && python3 -B -m unittest test_fr5_limits
"""

import dataclasses
import unittest

import numpy as np

import fr5_limits as fl
import fr5_motion as fm
from test_fr5_motion import CFG, F_T_C, NOW

GLASS = [(-87.0, -60.3, -8.0, 11.0), (-180.0, -87.0, -90.0, 90.0)]   # run_session.py 的默认右窗


def shifted(start, fwd=0.0, left=0.0, up=0.0):
    """起点沿（前, 左, 上）平移 mm，姿态不变。"""
    out = start.copy()
    out[:3, 3] += (fwd * start[:3, 2] - left * start[:3, 0] + up * fm.WORLD_UP) / 1000.0
    return out


def turned(start, kind, deg):
    """起点绕光心转：yaw 绕竖直（正 = 向左），pitch 绕相机 x（正 = 向上），roll 绕光轴（正 = 顶部向右）。"""
    axis = {"yaw": fm.WORLD_UP, "pitch": start[:3, 0], "roll": start[:3, 2]}[kind]
    return fm.rotate_about(start, axis, deg)


class MarkTest(unittest.TestCase):
    def setUp(self):
        self.start = fm.level_camera(NOW)

    def test_relative_and_reach(self):
        for name, kw, value in (("forward", {"fwd": 312.0}, 312.0), ("backward", {"fwd": -150.0}, -150.0),
                                ("left", {"left": 200.0}, 200.0), ("right", {"left": -90.0}, -90.0),
                                ("up", {"up": 120.0}, 120.0), ("down", {"up": -180.0}, -180.0)):
            rel = fl.relative(self.start, shifted(self.start, **kw))
            self.assertAlmostEqual(fl.reach(name, rel), value, places=6)
            self.assertLess(rel["rotation_deg"], 1e-9)
            self.assertEqual(fl.mark_warnings(name, rel), [])
        for name, kind, deg in (("yaw_left", "yaw", 25.0), ("yaw_right", "yaw", -18.0), ("pitch_up", "pitch", 17.0),
                                ("pitch_down", "pitch", -14.0), ("roll_right", "roll", 48.0), ("roll_left", "roll", -50.0)):
            rel = fl.relative(self.start, turned(self.start, kind, deg))
            self.assertAlmostEqual(fl.reach(name, rel), deg, places=6)
            np.testing.assert_allclose(rel["offset_mm"], 0.0, atol=1e-9)
            self.assertEqual(fl.mark_warnings(name, rel), [], name)
        # 和录制的约定一致：roll_right 录制的转轴是 +光轴，转到的横滚为正
        segs = fm.plan_recording("roll_right", self.start, F_T_C, dataclasses.replace(CFG, roll_deg=40.0))
        self.assertGreater(fm.pose_span(segs, self.start, F_T_C)["roll"][1], 19.9)

    def test_mark_warnings(self):
        sloppy = turned(shifted(self.start, fwd=300.0, up=-45.0), "pitch", 6.0)   # 拖的时候往下走了、相机低头
        found = fl.mark_warnings("forward", fl.relative(self.start, sloppy))
        self.assertEqual(len(found), 2, found)
        self.assertTrue(fl.mark_warnings("backward", fl.relative(self.start, shifted(self.start, fwd=50.0))))
        rot = shifted(turned(self.start, "yaw", 20.0), left=80.0)                 # 转的时候光心跑了
        found = fl.mark_warnings("yaw_left", fl.relative(self.start, rot))
        self.assertTrue(any("光心" in m for m in found), found)


class SuggestTest(unittest.TestCase):
    def setUp(self):
        self.start = fm.level_camera(NOW)

    def marks(self, **reach):
        out = {}
        for name, v in reach.items():
            if name in fl.TRANSLATION_MARKS:
                k, _ = fl.TRANSLATION_MARKS[name]
                cam = shifted(self.start, **{("fwd", "left", "up")[k]: v})
            else:
                cam = turned(self.start, name.split("_")[0], v)
            out[name] = fl.relative(self.start, cam)
        return out

    def test_translation(self):
        s = fl.suggest_translation(self.marks(forward=300, backward=-200, left=500, right=-400, up=150), 30.0, 600.0)
        fb = s["forward_backward"]
        self.assertAlmostEqual(fb["lo"], -170.0)
        self.assertAlmostEqual(fb["hi"], 270.0)
        self.assertEqual(fb["stroke_mm"], 440.0)
        self.assertEqual(fb["mid_mm"], 50.0)                     # 区间正好放下，取区间中点
        lr = s["left_right"]                                     # 可用 −370～470：行程到上限 600，中点就近取起点
        self.assertEqual((lr["stroke_mm"], lr["mid_mm"]), (600.0, 0.0))
        self.assertIsNone(s["up_down"])                          # 缺 down
        s = fl.suggest_translation(self.marks(forward=310, backward=-195.5), 30.0, 600.0)["forward_backward"]
        self.assertEqual(s["stroke_mm"], 440.0)                  # 可用 445.5 → 440
        self.assertTrue(s["lo"] + 220 <= s["mid_mm"] <= s["hi"] - 220 and s["mid_mm"] == round(s["mid_mm"]))
        s = fl.suggest_translation(self.marks(up=20, down=-20), 30.0, 600.0)["up_down"]
        self.assertIsNone(s["mid_mm"])                           # 收进余量后没有空间

    def test_glass_heading_limit(self):
        outlines, normals = [fm.region_outline(r) for r in GLASS], fm.frustum_normals()
        right, left = fl.glass_heading_limits(GLASS, 1.0)
        self.assertIsNone(left)                                  # 左边没有玻璃
        self.assertTrue(-9.0 < right < -6.0, right)              # 10-14 plan：偏航转到 −7.25° 时离右窗 1.3°
        clear = [fm.glass_clearance_deg(fm.axis_angle([0, -1, 0], h), GLASS, outlines, normals)
                 for h in (right, right - 0.05)]
        self.assertGreaterEqual(clear[0], 1.0)
        self.assertLess(clear[1], 1.0)

    def test_rotation(self):
        caps = {"yaw": 35.0, "pitch": 30.0, "roll": 90.0}
        m = self.marks(yaw_left=30, yaw_right=-20, pitch_up=20, pitch_down=-16, roll_right=60, roll_left=-55)
        r = fl.suggest_rotation(m, 3.0, caps, glass_right=-7.25)
        self.assertEqual(r["yaw"]["deg"], 34.0)                  # −7.25～27 → 34.25 → 34
        self.assertEqual(r["yaw"]["heading"], 9.75)
        self.assertGreaterEqual(r["yaw"]["heading"] - 17.0, -7.25)
        self.assertLessEqual(r["yaw"]["heading"] + 17.0, 27.0)
        self.assertEqual(r["pitch"]["deg"], 26.0)                # 对称：min(17, 13) × 2
        self.assertEqual(r["roll"]["deg"], 90.0)                 # min(57, 52) × 2 = 104，到上限 90
        r = fl.suggest_rotation(self.marks(yaw_left=30), 3.0, caps)
        self.assertIsNone(r["yaw"])

    def test_branch(self):
        start = [119.5, 14.4, -74.3, -120.0, -44.8, 0.0]
        self.assertIsNone(fl.branch_problem([126.0, 41.0, -60.0, -131.0, -44.0, 0.5], start))   # 往下拖
        found = fl.branch_problem([138.8, -58.6, 113.5, -235.4, -84.8, 15.0], start)      # 10-14 拖动后的实际位姿
        self.assertIn("J3", found)
        self.assertIn("J4", found)                                # −235 离 −120 也超过 100°
        self.assertIn("J6", fl.branch_problem([*start[:5], 120.0], start))

    def test_joint_envelope(self):
        lo, hi, lo_from, hi_from = fl.joint_envelope({"起点": [119.5, 14.4, -74.3, -120.0, -44.8, 0.0],
                                                      "down": [118.0, 41.3, -60.0, -130.0, -44.0, 0.5],
                                                      "roll_left": [119.0, 14.0, -74.0, -120.0, -45.0, -48.2]})
        self.assertEqual(hi[1], 44.0)                            # 41.3 + 2 向上取整
        self.assertEqual(hi_from[1], "down")
        self.assertEqual(lo[5], -51.0)
        self.assertEqual(lo_from[5], "roll_left")


class LimitCheckTest(unittest.TestCase):
    def setUp(self):
        self.start = fm.level_camera(NOW)
        self.marks = {n: fl.relative(self.start, shifted(self.start, fwd=v))
                      for n, v in (("forward", 300.0), ("backward", -200.0))}

    def check(self, name, cfg, marks=None):
        segs = fm.plan_recording(name, self.start, F_T_C, cfg)
        return fl.limit_check(name, segs, self.start, F_T_C, marks or self.marks, 30.0, 3.0)

    def test_translation_inside_and_outside(self):
        cfg = dataclasses.replace(CFG, translation_paths={"forward_backward": ((50.0, 0.0, 0.0), 0.44),
                                                          "left_right": ((0.0, 0.0, 0.0), 0.3),
                                                          "up_down": ((0.0, 0.0, 0.0), 0.3)})
        note, found = self.check("forward", cfg)
        self.assertEqual(found, [], note)
        self.assertIn("-170～+270", note)
        bigger = dataclasses.replace(cfg, translation_paths={**cfg.translation_paths,
                                                             "forward_backward": ((50.0, 0.0, 0.0), 0.46)})
        self.assertEqual(len(self.check("backward", bigger)[1]), 2)   # 两头各超 10 mm
        off_axis = dataclasses.replace(cfg, translation_paths={**cfg.translation_paths,
                                                               "forward_backward": ((50.0, 0.0, 30.0), 0.44)})
        self.assertTrue(any("轴" in m for m in self.check("forward", off_axis)[1]))
        note, found = self.check("left", cfg)                    # 没有左右的标记：不查行程，只写明
        self.assertEqual(found, [])
        self.assertIn("没有标记", note)
        self.assertEqual(fl.limit_check("sync-test", [], self.start, F_T_C, self.marks, 30.0, 3.0), ("", []))

    def test_rotation_range(self):
        cfg = dataclasses.replace(CFG, yaw_deg=30.0, yaw_heading_deg=7.75)     # −7.25～+22.75°
        marks = {n: fl.relative(self.start, turned(self.start, "yaw", v)) for n, v in (("yaw_left", 26.0), ("yaw_right", -11.0))}
        self.assertEqual(self.check("yaw_left", cfg, marks)[1], [])
        marks["yaw_left"] = fl.relative(self.start, turned(self.start, "yaw", 25.0))
        found = self.check("yaw_right", cfg, marks)[1]
        self.assertEqual(len(found), 1)
        self.assertIn("yaw_left", found[0])


class ViaStartTest(unittest.TestCase):
    def setUp(self):
        self.start = fm.level_camera(NOW)
        self.cfg = dataclasses.replace(CFG, translation_paths={"forward_backward": ((50.0, 0.0, 0.0), 0.44),
                                                               "left_right": ((0.0, 0.0, 0.0), 0.3),
                                                               "up_down": ((0.0, 0.0, 0.0), 0.3)})

    def plan(self, name, current):
        segs = fm.plan_recording(name, self.start, F_T_C, self.cfg)
        self.assertEqual([s.name for s in segs[:2]], ["via_start", "goto_start"])
        self.assertFalse(segs[0].recorded)
        np.testing.assert_allclose(segs[0].camera, self.start)
        fm.drop_needless_via(segs, current, self.start)
        return [s.name for s in segs[:2]]

    def test_drop_when_direct_move_stays_on_verified_paths(self):
        for name in ("forward", "left", "up", "yaw_left", "pitch_down"):       # 就在起点
            self.assertEqual(self.plan(name, self.start)[0], "goto_start", name)
        back_end = shifted(self.start, fwd=-170.0)                             # backward 录完停在后端
        self.assertEqual(self.plan("forward", back_end)[0], "goto_start")      # 同一根轴：直接过去
        self.assertEqual(self.plan("left", back_end)[0], "via_start")          # 换轴：先回起点
        self.assertEqual(self.plan("yaw_left", back_end)[0], "via_start")
        after_yaw = turned(self.start, "yaw", 20.0)                            # 偏航录完：光心在起点、朝向变了
        self.assertEqual(self.plan("pitch_up", after_yaw)[0], "goto_start")    # 原地转过去
        self.assertEqual(self.plan("forward", after_yaw)[0], "via_start")      # 先转回起点姿态再平移
        tilted_end = turned(back_end, "pitch", 3.0)
        self.assertEqual(self.plan("forward", tilted_end)[0], "via_start")     # 姿态不对：先回起点

    def test_speed_test_via(self):
        bottom = shifted(self.start, up=-150.0)
        segs = fm.plan_speed_test(bottom, F_T_C, self.cfg, 300.0, 5.0, via=self.start)
        self.assertEqual([s.name for s in segs[:2]], ["via_start", "goto_start"])
        fm.drop_needless_via(segs, self.start, self.start)
        self.assertEqual(segs[0].name, "goto_start")
        self.assertEqual(fm.plan_speed_test(bottom, F_T_C, self.cfg, 300.0, 5.0)[0].name, "goto_start")

    def test_limit_check_ignores_via(self):
        marks = {n: fl.relative(self.start, shifted(self.start, fwd=v)) for n, v in (("forward", 300.0), ("backward", -200.0))}
        segs = fm.plan_recording("forward", self.start, F_T_C, self.cfg)
        segs[0] = fm._move("via_start", shifted(self.start, fwd=-400.0), F_T_C, 50.0, recorded=False)
        self.assertEqual(fl.limit_check("forward", segs, self.start, F_T_C, marks, 30.0, 3.0)[1], [])


if __name__ == "__main__":
    unittest.main()
