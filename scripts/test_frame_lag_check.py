"""frame_lag_check.py 的合成数据测试（在笔记本上运行：要 OpenCV；Box 上会跳过）。

按已知的纯旋转（绕自身竖直轴单侧摆动）用单应变换渲染图像、生成陀螺仪数据：
图像和时间戳对齐时应测出约 0 ms；图像内容整体晚一帧时应测出约 +66.7 ms 并报异常。

    python -B -m unittest test_frame_lag_check
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    import frame_lag_check as fl
    SKIP = None
except ImportError as exc:  # Box 上 OpenCV 导入不了
    SKIP = str(exc)

EPOCH_MS = 1_791_600_000_000.0
PERIOD_MS = 1000.0 / 15
WIGGLES = [(0.5 + i, 1.5 + i) for i in range(4)]     # 4 次单侧摆动 0→10°→0，每次 1 s


def yaw_deg(t_s):
    """相机绕自身竖直轴的转角（°）。"""
    for a, b in WIGGLES:
        if a <= t_s <= b:
            return 10.0 * np.sin(np.pi * (t_s - a) / (b - a)) ** 2
    return 0.0


def yaw_rate_deg_s(t_s):
    for a, b in WIGGLES:
        if a <= t_s <= b:
            return 10.0 * np.pi / (b - a) * np.sin(2 * np.pi * (t_s - a) / (b - a))
    return 0.0


def _ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def write_session(d: Path, k: np.ndarray, frames_late: int = 0, motion_log: bool = True) -> None:
    """frames_late：图像内容比时间戳晚几帧（第 i 张图存的是第 i - frames_late 个时刻的画面）。"""
    rng = np.random.default_rng(0)
    pad_x, pad_y = 400, 300                                       # 参考图比画面大，转 10° 也不出边
    big = np.full((1080 + 2 * pad_y, 1920 + 2 * pad_x), 128, np.uint8)
    for _ in range(1500):                                         # 随机色块：角点多、纹理清楚
        x, y = int(rng.integers(0, big.shape[1])), int(rng.integers(0, big.shape[0]))
        w, h = int(rng.integers(8, 60)), int(rng.integers(8, 60))
        big[y:y + h, x:x + w] = int(rng.integers(0, 256))
    big = cv2.GaussianBlur(big, (0, 0), 1.2)
    k_big = k.copy()
    k_big[0, 2] += pad_x
    k_big[1, 2] += pad_y

    t_s = 0.2 + np.arange(int(5.0 * 15)) / 15.0
    (d / "left").mkdir()
    rows = []
    for i, t in enumerate(t_s):
        j = max(i - frames_late, 0)
        h_mat = k @ _ry(yaw_deg(t_s[j])).T @ np.linalg.inv(k_big)  # 参考像素 → 第 j 个时刻的画面
        image = cv2.warpPerspective(big, h_mat, (1920, 1080), flags=cv2.INTER_LINEAR)
        image = np.clip(image + rng.normal(0, 1.5, image.shape), 0, 255).astype(np.uint8)
        ts = int(round(EPOCH_MS + t * 1000.0))
        name = f"left/{ts}.jpg"
        cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(d / name))
        rows.append([ts, 0, name, name.replace("left", "right")])
    with open(d / "img_timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zed_timestamp_ms", "zed_imu_timestamp_ns", "filename_left", "filename_right"])
        w.writerows(rows)

    t_imu = np.arange(0.0, 5.4, 0.005)
    with open(d / "zed_imu.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zed_imu_timestamp_ns", "zed_timestamp_ms", "gx", "gy", "gz"])
        for t in t_imu:
            gy = -yaw_rate_deg_s(t) + rng.normal(0, 0.05)        # 绕“上”（-y）转，陀螺仪 y 为负
            w.writerow([int(round((EPOCH_MS + t * 1000.0) * 1e6)), 0, rng.normal(0, 0.05), gy, rng.normal(0, 0.05)])
    if motion_log:
        segs = [{"name": f"wiggle_start_{i + 1}", "kind": "move", "start_box_ns": int((EPOCH_MS + a * 1000) * 1e6),
                 "end_box_ns": int((EPOCH_MS + b * 1000) * 1e6)} for i, (a, b) in enumerate(WIGGLES)]
        (d / "motion_log.json").write_text(json.dumps({"kind": "record", "segments": segs}))


@unittest.skipIf(SKIP, SKIP)
class FrameLagTest(unittest.TestCase):
    def setUp(self):
        self.k = fl.load_camera(fl.CAMERA_YAML)

    def _check(self, **kw) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, self.k, **kw)
            r = fl.check(d)
            print("\n" + fl.summary(r))
            self.assertTrue((d / "frame_lag_report.json").exists())
            return r

    def test_aligned_images(self):
        r = self._check()
        self.assertEqual(r["window_source"], "motion_log")
        self.assertGreaterEqual(r["pairs"], fl.MIN_PAIRS)
        self.assertAlmostEqual(r["lag_ms"], 0.0, delta=2.0)
        self.assertTrue(r["ok"])
        self.assertEqual(r["frames_late"], 0)
        self.assertGreater(r["rms_deg_s"]["at_plus_one_frame"], 5 * r["rms_deg_s"]["at_best"])

    def test_one_frame_late(self):
        r = self._check(frames_late=1)
        self.assertAlmostEqual(r["lag_ms"], PERIOD_MS, delta=3.0)
        self.assertFalse(r["ok"])
        self.assertEqual(r["frames_late"], 1)
        self.assertIn("晚约 1 帧", r["verdict"])

    def test_without_motion_log_uses_gyro(self):
        r = self._check(frames_late=1, motion_log=False)
        self.assertEqual(r["window_source"], "gyro")
        self.assertAlmostEqual(r["lag_ms"], PERIOD_MS, delta=3.0)


if __name__ == "__main__":
    unittest.main()
