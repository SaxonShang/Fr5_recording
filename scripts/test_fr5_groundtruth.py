"""fr5_groundtruth.py 的合成数据测试：已知时钟偏差、漂移、各段延迟和 Δt，检查还原出的真值。

    cd ~/ROBOT/fr5_record && python3 -B -m unittest test_fr5_groundtruth
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fr5_groundtruth as gt

EPOCH_MS = 1_791_500_000_000.0  # Box / ZED 时间原点（Unix ms）
CTRL_OFFSET_MS = 127_000.0      # 控制器时钟比 Box 快 127 s
DRIFT = -200e-6                 # 控制器时钟比 Box 慢 200 ppm（实测量级）
LAMBDA_MS = 2.0                 # 控制器状态样本比真实运动晚
MU_MS = 7.0                     # ZED（IMU 和图像）时钟比真实时刻晚
EXPO_LAG_MS = 3.7               # 图像时间戳再比曝光中心晚（实测量级）
EXPOSURE_MS = 10.0
# zed_record 把图像时间戳截到 ms，平均早 0.5 ms，所以估出的“曝光中心 - 时间戳”约为 -(3.7 - 0.5)
EXPECTED_EXPOSURE_OFFSET_MS = -(EXPO_LAG_MS - 0.5)
SEND_PHASE_MS = 1.3             # 10 ms 发包节拍相对 6 ms 采样节拍的相位
NET_MS = 0.2
DURATION_S = 70.0
# 期望 Δt = MU - LAMBDA - 最小(发包 - 采样) - NET，控制器时间取整到 ms 会再带来 <1 ms 的偏差
EXPECTED_DT_MS = MU_MS - LAMBDA_MS - SEND_PHASE_MS - NET_MS
FLANGE_T_CAM = np.array([[0.999660953447201, 0.0007843776424874251, 0.02602619650942236, -0.058960616228020075],
                         [-0.0007566244143297974, 0.9999991346690702, -0.0010761879977619043, -0.07159469739308227],
                         [-0.026027018125954093, 0.001056131064239924, 0.999660681888933, 0.09927925870705787],
                         [0.0, 0.0, 0.0, 1.0]])  # s003 的手眼结果
R0 = gt.euler_to_matrix(-156.0, -6.0, -168.0)
P0 = np.array([-0.018, -0.477, 0.420])


def _rz(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _ry(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rx(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def flange_pose(t_s):
    """静止 → 同步摆动（绕法兰 Z ±15°，0.5 Hz）→ 静止 → 平移加俯仰 → 静止 → 摆动 → 静止。"""
    t = np.asarray(t_s, dtype=float)
    wiggle = lambda a: np.where((t >= a) & (t <= a + 6), 15 * np.sin(np.pi * (t - a)), 0.0)
    theta_z = wiggle(3.0) + wiggle(61.0)
    m = (t >= 12) & (t <= 58)
    u = 2 * np.pi * (t - 12) / 46
    theta_x = np.where(m, 5 * np.sin(5 * u), 0.0)
    p = P0 + np.where(m[:, None], np.c_[0.1 * np.sin(u), 0.05 * np.sin(2 * u), np.zeros_like(t)], 0.0)
    return p, R0 @ _rz(theta_z) @ _rx(theta_x)


def sync_test_pose(t_s):
    """和 run_session sync-test 一样短：静止 → 摆动 → 静止 10 s → 摆动 → 静止，共约 30 s。"""
    t = np.asarray(t_s, dtype=float)
    wiggle = lambda a: np.where((t >= a) & (t <= a + 6), 15 * np.sin(np.pi * (t - a)), 0.0)
    return np.broadcast_to(P0, (len(t), 3)), R0 @ _rz(wiggle(5.0) + wiggle(21.0))


def rotvec(r):
    q = gt.matrix_to_quat(r)
    v = q[:, 1:]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    ang = 2 * np.arctan2(n, q[:, :1])
    return np.where(n > 0, v / np.where(n > 0, n, 1) * ang, 0.0)


def write_session(d: Path, rng, flags: bool = True, duration_s: float = DURATION_S, pose=flange_pose) -> None:
    # 机械臂：6 ms 采样，10 ms 发一包（带最新样本）
    t_send = SEND_PHASE_MS + 10.0 * np.arange(int(duration_s * 100))
    t_sample = 6.0 * np.floor(t_send / 6.0)
    p, r = pose((t_sample - LAMBDA_MS) / 1000.0)
    ctrl = np.floor(EPOCH_MS + CTRL_OFFSET_MS + t_sample * (1 + DRIFT))
    box_ns = np.round((EPOCH_MS + t_send + NET_MS + rng.exponential(0.1, len(t_send))) * 1e6).astype(np.int64)
    ry = -np.rad2deg(np.arcsin(r[:, 2, 0]))
    rx = np.rad2deg(np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    rz = np.rad2deg(np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    with open(d / "fr5_state.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(gt.ARM_COLUMNS)
        for k in range(len(t_send)):
            w.writerow([box_ns[k], int(ctrl[k]), *(p[k] * 1000), rx[k], ry[k], rz[k], 1, 0, 0, 0, 0])

    # 图像：15 Hz；曝光中心在 t_img，时间戳 = 曝光中心 + MU + EXPO_LAG，截到 ms
    t_img = 500.0 + (1000.0 / 15) * np.arange(int((duration_s - 1.5) * 15))
    # ZED IMU：200 Hz，陀螺仪 = 相机体坐标系角速度 + 零偏 + 噪声（°/s）
    t_imu = 0.37 + 5.0 * np.arange(int(duration_s * 200))
    h = 0.5
    _, r_a = pose((t_imu - h) / 1000.0)
    _, r_b = pose((t_imu + h) / 1000.0)
    w_flange = rotvec(np.swapaxes(r_a, 1, 2) @ r_b) / (2 * h / 1000.0)
    w_cam = w_flange @ FLANGE_T_CAM[:3, :3]  # = R_fcᵀ ω
    gyro = np.rad2deg(w_cam) + np.array([0.3, -0.2, 0.1]) + rng.normal(0, 0.05, w_cam.shape)
    imu_ns = np.round((EPOCH_MS + t_imu + MU_MS) * 1e6).astype(np.int64)
    with open(d / "zed_imu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zed_imu_timestamp_ns", "zed_timestamp_ms", "image_sync_trigger",
                    "ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"])
        for k in range(len(t_imu)):
            exposing = flags and np.min(np.abs(t_img - t_imu[k])) <= EXPOSURE_MS / 2
            w.writerow([imu_ns[k], 0, int(exposing), 0, 0, 0, *gyro[k], 1, 0, 0, 0])

    ts = np.floor(EPOCH_MS + t_img + MU_MS + EXPO_LAG_MS).astype(np.int64)
    (d / "time_meta.txt").write_text(f"exposure_us={int(EXPOSURE_MS * 1000)}\n")
    with open(d / "img_timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zed_timestamp_ms", "zed_imu_timestamp_ns", "filename_left", "filename_right"])
        for k in range(len(t_img)):
            w.writerow([ts[k], 0, f"left/{ts[k]}.jpg", f"right/{ts[k]}.jpg"])
    np.save(d / "truth_t_img.npy", t_img)
    (d / "handeye.json").write_text(json.dumps(FLANGE_T_CAM.tolist()))


class RotationTest(unittest.TestCase):
    def test_euler_matches_elementwise_product(self):
        a = np.array([[-156.0, -6.0, -168.0], [10.0, 80.0, -30.0], [179.0, -45.0, 90.0]])
        r = gt.euler_to_matrix(a[:, 0], a[:, 1], a[:, 2])
        ref = _rz(a[:, 2]) @ _ry(a[:, 1]) @ _rx(a[:, 0])
        np.testing.assert_allclose(r, ref, atol=1e-12)

    def test_quaternion_round_trip_all_branches(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=(2000, 4))
        q[:4] = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]  # 180° 和单位阵
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        r = gt.quat_to_matrix(q)
        np.testing.assert_allclose(gt.quat_to_matrix(gt.matrix_to_quat(r)), r, atol=1e-12)

    def test_small_angle_precision(self):
        q0 = gt.matrix_to_quat(gt.euler_to_matrix(-156.0, -6.0, -168.0))
        q1 = gt.matrix_to_quat(gt.euler_to_matrix(-156.0, -6.0, -168.0 + 1e-6))
        self.assertAlmostEqual(np.rad2deg(gt.rotation_angle_between(q0, q1)), 1e-6, delta=1e-9)


class SyntheticSessionTest(unittest.TestCase):
    def test_recovers_clock_lag_and_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            report = gt.process(d, d / "handeye.json", d)
            self.assertEqual(report["warnings"], [])
            self.assertAlmostEqual(report["clock_fit"]["drift_ppm"], -DRIFT * 1e6, delta=5.0)
            lag = report["time_offset"]
            self.assertAlmostEqual(lag["dt_ms"], EXPECTED_DT_MS, delta=1.2)
            self.assertGreater(lag["peak_corr"], 0.99)
            self.assertLess(lag["segment_diff_ms"], 1.0)
            exp = report["exposure_centre"]
            self.assertEqual(exp["source"], "image_sync_trigger")
            # IMU 5 ms 一个样本，15 fps 下相对帧的相位只有 3 种，估计最多偏约 ±0.8 ms
            self.assertAlmostEqual(exp["applied_ms"], EXPECTED_EXPOSURE_OFFSET_MS, delta=1.0)

            with open(d / "groundtruth_images.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(report["images"]["with_groundtruth"], len(rows))
            self.assertTrue(all(r["flag"] == "ok" for r in rows))
            t_img = np.load(d / "truth_t_img.npy")
            p, r = flange_pose(t_img / 1000.0)
            r_cam = r @ FLANGE_T_CAM[:3, :3]
            t_cam = p + r @ FLANGE_T_CAM[:3, 3]
            est_t = np.array([[float(x[k]) for k in ("tx_m", "ty_m", "tz_m")] for x in rows])
            est_q = np.array([[float(x[k]) for k in ("qw", "qx", "qy", "qz")] for x in rows])
            pos_err_mm = np.linalg.norm(est_t - t_cam, axis=1) * 1000
            rot_err_deg = np.rad2deg(gt.rotation_angle_between(est_q, gt.matrix_to_quat(r_cam)))
            print(f"\n合成数据：Δt={lag['dt_ms']:.2f} ms（期望约 {EXPECTED_DT_MS:.1f}），"
                  f"曝光中心 {exp['applied_ms']:+.2f} ms（期望约 {EXPECTED_EXPOSURE_OFFSET_MS:+.1f}），"
                  f"漂移 {report['clock_fit']['drift_ppm']:.1f} ppm，"
                  f"位置误差 RMS {np.sqrt(np.mean(pos_err_mm ** 2)):.3f} / 最大 {pos_err_mm.max():.3f} mm，"
                  f"转角误差 RMS {np.sqrt(np.mean(rot_err_deg ** 2)):.4f} / 最大 {rot_err_deg.max():.4f}°")
            self.assertLess(pos_err_mm.max(), 0.5)
            self.assertLess(rot_err_deg.max(), 0.1)
            self.assertLess(np.sqrt(np.mean(rot_err_deg ** 2)), 0.03)

    def test_short_sync_test_checks_both_halves(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(4), duration_s=30.0, pose=sync_test_pose)
            report = gt.process(d, d / "handeye.json", d)
            self.assertEqual(report["warnings"], [])
            lag = report["time_offset"]
            self.assertEqual(set(lag["segments"]), {"first_40pct", "last_40pct"})
            self.assertLess(lag["segment_diff_ms"], 1.0)
            self.assertAlmostEqual(lag["dt_ms"], EXPECTED_DT_MS, delta=1.2)

    def test_warns_without_sync_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0), flags=False)
            report = gt.process(d, d / "handeye.json", d)
            self.assertEqual(report["exposure_centre"]["applied_ms"], 0.0)
            self.assertTrue(any("image_sync_trigger" in w for w in report["warnings"]))
            manual = gt.process(d, d / "handeye.json", d, exposure_offset_ms=-3.2)
            self.assertEqual(manual["exposure_centre"]["applied_ms"], -3.2)

    def test_refuses_without_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            with open(d / "fr5_state.csv", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            for row in rows[1:]:
                row[5:8] = ["-156", "-6", "-168"]  # 姿态不变，只剩平移
            with open(d / "fr5_state.csv", "w", newline="") as f:
                csv.writer(f).writerows(rows)
            with self.assertRaisesRegex(ValueError, "无法估计"):
                gt.process(d, d / "handeye.json", d)


if __name__ == "__main__":
    unittest.main()
