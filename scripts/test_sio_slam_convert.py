"""sio_slam_convert.py 的合成数据测试（只用标准库，笔记本和 Box 上都能运行）。

按 SIO_SLAM DatasetLoader 读 zed_record 格式的方式（逗号分列、按帧号分组 IMU、每帧下限 min(25, 应有个数)）
把转换结果读回来，检查不会有帧被丢掉、IMU 按 (上一帧, 本帧] 分配、时间偏移同时加在帧时间和真值上。

    python -B -m unittest test_sio_slam_convert
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import sio_slam_convert as conv

EPOCH_MS = 1_792_034_772_931
FRAME_MS = [EPOCH_MS + round(i * 1000 / 30) for i in range(20)]
IMU_START_NS = (EPOCH_MS - 1500) * 1_000_000      # IMU 比第一张图早 1.5 s
IMU_PERIOD_NS = 5_000_000                          # 200 Hz


def write_session(d: Path, moving_from_ns: int | None = None, lag_ms: float = -4.5,
                  gap_ns: tuple[int, int] | None = None) -> list[int]:
    """合成一条录制；moving_from_ns 之后陀螺仪读数为 30 °/s；gap_ns=(起, 止) 内没有 IMU。返回 IMU 时间戳（ns）。"""
    (d / "left").mkdir(parents=True)
    (d / "right").mkdir()
    with open(d / "img_timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zed_timestamp_ms", "zed_imu_timestamp_ns", "filename_left", "filename_right"])
        for ms in FRAME_MS:
            w.writerow([ms, ms * 1_000_000, f"left/{ms}.jpg", f"right/{ms}.jpg"])
            (d / "left" / f"{ms}.jpg").write_bytes(b"")
            (d / "right" / f"{ms}.jpg").write_bytes(b"")
    stamps = list(range(IMU_START_NS, (FRAME_MS[-1] + 20) * 1_000_000, IMU_PERIOD_NS))
    if gap_ns is not None:
        stamps = [ns for ns in stamps if not gap_ns[0] < ns < gap_ns[1]]
    with open(d / "zed_imu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zed_imu_timestamp_ns", "zed_timestamp_ms", "image_sync_trigger",
                    "ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"])
        for ns in stamps:
            g = 30.0 if moving_from_ns is not None and ns >= moving_from_ns else 0.1
            w.writerow([ns, 0, 0, "-0.1", "-9.85", "0.03", f"{g}", "0.0", "0.0", 1, 0, 0, 0])
    with open(d / "groundtruth_tum.txt", "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, ms in enumerate(FRAME_MS):
            f.write(f"{ms / 1000:.6f} {0.6 + i * 0.001:.6f} -0.142896 0.187137 -0.5 0.5 -0.5 0.5\n")
    (d / "frame_lag_report.json").write_text(json.dumps({"lag_ms": lag_ms, "ok": True, "verdict": "正常"}))
    return stamps


def load_like_sio_slam(out: Path) -> tuple[list[dict], dict[int, list[float]]]:
    """照 DatasetLoader::LoadZedRecordFrames / LoadZedRecordImu 的解析方式读回来。"""
    frames = []
    lines = (out / "frames.csv").read_text().splitlines()[1:]
    for line in lines:
        v = line.split(",")
        frames.append(dict(id=int(v[0]), st=float(v[1]), end=float(v[2]), left=v[3], right=v[4], expected=int(v[5]),
                           end_text=v[2]))
    imu: dict[int, list[float]] = {}
    for line in (out / "imu.csv").read_text().splitlines()[1:]:
        v = line.split(",")
        assert len(v) >= 9, line          # C++ 读到 values[8]
        imu.setdefault(int(v[0]), []).append(float(v[1]))
    return frames, imu


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loader_keeps_every_frame(self):
        d = self.root / "rec" / "save_data"
        write_session(d)
        meta = conv.convert(d, start_frame=3)
        frames, imu = load_like_sio_slam(d / "sio_slam")
        self.assertEqual(len(frames), len(FRAME_MS) - 3)
        self.assertEqual(meta["warnings"], [])
        for i, fr in enumerate(frames):
            count = len(imu.get(fr["id"], []))
            self.assertEqual(count, fr["expected"])
            dropped = i > 0 and fr["expected"] > 0 and count < min(25, fr["expected"])
            self.assertFalse(dropped, f"第 {i} 帧会被 SIO_SLAM 丢掉")
            # SIO_SLAM 把 dataset_dir（带末尾 /）和 left_path 直接拼起来
            self.assertTrue((d / "sio_slam" / fr["left"]).exists())
            self.assertTrue((d / "sio_slam" / fr["right"]).exists())

    def test_imu_split_by_frame_interval(self):
        d = self.root / "save_data"
        stamps = write_session(d)
        conv.convert(d, start_frame=3, offset_arg="4.7")
        frames, imu = load_like_sio_slam(d / "sio_slam")
        first = frames[0]
        # 起始帧拿到它之前的全部 IMU（足够做静止初始化）
        expected_first = sum(1 for ns in stamps if ns / 1e9 <= first["end"] + 1e-9)
        self.assertEqual(len(imu[first["id"]]), expected_first)
        self.assertGreaterEqual(expected_first, conv.MIN_INIT_IMU)
        for prev, fr in zip(frames, frames[1:]):
            self.assertAlmostEqual(fr["st"], prev["end"], places=6)
            for t in imu[fr["id"]]:
                self.assertTrue(prev["end"] < t <= fr["end"] + 1e-9)
        total = sum(len(v) for v in imu.values())
        self.assertEqual(total, sum(1 for ns in stamps if ns / 1e9 <= frames[-1]["end"] + 1e-9))

    def test_offset_on_frames_and_groundtruth(self):
        d = self.root / "save_data"
        write_session(d)
        conv.convert(d, start_frame=5, offset_arg="4.7")
        frames, _ = load_like_sio_slam(d / "sio_slam")
        gt = [l.split() for l in (d / "sio_slam" / "groundtruth_tum.txt").read_text().splitlines()
              if not l.startswith("#")]
        self.assertEqual(len(gt), len(frames))
        for k, (fr, row) in enumerate(zip(frames, gt)):
            ms = FRAME_MS[5 + k]
            self.assertEqual(fr["end_text"], f"{ms // 1000}.{(ms % 1000) * 1000 + 4700:06d}")
            self.assertEqual(row[0], fr["end_text"])                      # 真值和帧时间同一时间轴
            self.assertEqual(row[1], f"{0.6 + (5 + k) * 0.001:.6f}")       # 位姿不变

    def test_auto_offset_from_frame_lag_report(self):
        d = self.root / "save_data"
        write_session(d, lag_ms=-4.5)
        meta = conv.convert(d, start_frame=3, offset_arg="auto")
        self.assertAlmostEqual(meta["time_offset_ms"], 4.5)
        self.assertEqual(meta["time_offset_source"], "frame_lag_report")

    def test_init_segment_keeps_samples_after_last_gap(self):
        d = self.root / "save_data"
        gap = (IMU_START_NS + 200_000_000, IMU_START_NS + 240_000_000)    # 开头 0.2 s 处缺 40 ms
        stamps = write_session(d, gap_ns=gap)
        meta = conv.convert(d, start_frame=3)
        frames, imu = load_like_sio_slam(d / "sio_slam")
        init = imu[frames[0]["id"]]
        self.assertEqual(meta["warnings"], [])
        self.assertEqual(len(meta["notes"]), 1)
        self.assertGreaterEqual(init[0], gap[1] / 1e9)                    # 缺口之前的都去掉了
        self.assertEqual(len(init), sum(1 for ns in stamps if gap[1] <= ns and ns / 1e9 <= frames[0]["end"] + 1e-9))
        self.assertTrue(all(b - a <= conv.MAX_IMU_GAP_MS / 1000 for a, b in zip(init, init[1:])))
        self.assertGreaterEqual(len(init), conv.MIN_INIT_IMU)

    def test_warns_when_not_static_before_start(self):
        d = self.root / "save_data"
        write_session(d, moving_from_ns=EPOCH_MS * 1_000_000)
        meta = conv.convert(d, start_frame=3)
        self.assertTrue(any("不是静止的" in w for w in meta["warnings"]))

    def test_find_sessions_in_parent_directory(self):
        for name in ("a_up", "b_right"):
            write_session(self.root / name / "save_data")
        (self.root / "calib").mkdir()
        found = conv.find_sessions(self.root)
        self.assertEqual([p.parent.name for p in found], ["a_up", "b_right"])
        self.assertEqual(conv.find_sessions(self.root / "a_up"), [self.root / "a_up" / "save_data"])


if __name__ == "__main__":
    unittest.main()
