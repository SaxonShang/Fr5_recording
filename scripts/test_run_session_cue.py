"""run_session.py 的 --cue 提示（不连机械臂、不开相机、不放声音）。要能 import run_session（Box 上）。

    python3 -B -m unittest test_run_session_cue
"""

import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

try:
    import run_session as rs
    SKIP = None
except ImportError as exc:  # 笔记本上没装 fairino 时跳过（Box 上跑）
    rs, SKIP = None, f"需要 FAIRINO SDK：{exc}"


def segs(move: str) -> list:
    names = (["goto_start", "hold_start"] + [f"wiggle_start_{i}" for i in range(1, 5)] + ["hold_before", move, "hold_after"]
             + [f"wiggle_end_{i}" for i in range(1, 5)] + ["hold_end"])
    dur = {"hold_start": 5.0, "hold_before": 3.0, "hold_after": 3.0, "hold_end": 5.0, move: 1.9}
    return [types.SimpleNamespace(name=n, duration_s=dur.get(n, 0.63)) for n in names]


@unittest.skipIf(SKIP, SKIP)
class CueTest(unittest.TestCase):
    def test_translate_plan(self):
        plan = rs.cue_plan(segs("translate"))
        self.assertEqual(list(plan), ["wiggle_start_1", "hold_before", "translate", "hold_after", "wiggle_end_1"])
        self.assertEqual([plan[k][0] for k in plan], ["ready", "go", None, "leave", None])
        self.assertIn("听到下一声就进画面", plan["wiggle_start_1"][1])
        self.assertIn("3 s 后开始平移", plan["hold_before"][1])
        self.assertIn("3 s 内走出画面", plan["hold_after"][1])

    def test_rotate_plan_and_no_motion(self):
        self.assertIn("旋转", rs.cue_plan(segs("rotate"))["hold_before"][1])
        sync = [s for s in segs("translate") if s.name not in ("translate",)]
        self.assertEqual(rs.cue_plan(sync), {})                                # sync-test 一类没有运动段

    def test_cue_does_not_wait_for_sound(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cue_events.log"
            with unittest.mock.patch.object(rs, "CUE_LOG", str(log)), \
                    unittest.mock.patch.object(rs.subprocess, "Popen") as popen:
                rs.cue("go", "走！")
                rs.cue(None, "只显示")
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(popen.call_args[0][0][1].endswith(rs.CUE_SOUNDS["go"]))
            lines = log.read_text(encoding="utf-8").splitlines()               # 给笔记本 cue_listener.py 的
            self.assertEqual([l.split(" ", 2)[1:] for l in lines], [["go", "走！"], ["-", "只显示"]])
            with unittest.mock.patch.object(rs, "CUE_LOG", str(Path(tmp) / "没有这个目录" / "x.log")), \
                    unittest.mock.patch.object(rs.subprocess, "Popen", side_effect=OSError("没有 paplay")):
                rs.cue("leave", "离开")                                         # 写不了、放不了也不出错


if __name__ == "__main__":
    unittest.main()
