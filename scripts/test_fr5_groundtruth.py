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


PIVOT = P0 + R0 @ FLANGE_T_CAM[:3, 3]                          # 起点时左目光心（基坐标系）
PITCH_AXIS = R0 @ FLANGE_T_CAM[:3, :3] @ np.array([1.0, 0.0, 0.0])   # 相机 x 轴（基坐标系）


def _about(axis, deg):
    """绕单位轴 axis 转 deg 度（可为数组）的旋转矩阵。"""
    a = np.deg2rad(np.asarray(deg, dtype=float))[..., None, None]
    kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * (kx @ kx)


def pitch_turn_deg(t_s):
    """旋转录制：单侧同步摆动 0→10°→0（3 次）→ 静止 → 转 30° → 静止 → 往回摆 30→20→30（3 次）→ 静止。"""
    t = np.asarray(t_s, dtype=float)
    swing = lambda a: np.where((t >= a) & (t <= a + 3), 10 * np.sin(np.pi * (t - a)) ** 2, 0.0)
    u = np.clip((t - 20.0) / 6.0, 0.0, 1.0)
    return swing(5.0) + 30 * (3 * u ** 2 - 2 * u ** 3) - swing(50.0)


def pitch_pose(t_s):
    """法兰绕左目光心、绕相机 x 轴转（光心不动）。"""
    r = _about(PITCH_AXIS, pitch_turn_deg(t_s))
    return PIVOT + (r @ (P0 - PIVOT)), r @ R0


def rotvec(r):
    q = gt.matrix_to_quat(r)
    v = q[:, 1:]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    ang = 2 * np.arctan2(n, q[:, :1])
    return np.where(n > 0, v / np.where(n > 0, n, 1) * ang, 0.0)


def write_motion_log(d: Path, kind: str, direction: str, start_m, end_m) -> None:
    """只写 fr5_groundtruth 用到的字段：录制类型、方向、路径两端的法兰目标（mm）。"""
    segs = [{"name": "goto_start", "flange": [*(np.asarray(start_m) * 1000), 0, 0, 0]},
            {"name": "translate", "flange": [*(np.asarray(end_m) * 1000), 0, 0, 0]}]
    (d / "motion_log.json").write_text(json.dumps({"kind": kind, "direction": direction, "segments": segs}))


def read_tum(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", ndmin=2)


def write_session(d: Path, rng, flags: bool = True, duration_s: float = DURATION_S, pose=flange_pose,
                  gyro_lag_ms: float = 0.0) -> None:
    """gyro_lag_ms：陀螺仪输出比真实运动晚（模拟标定板验证发现的固定偏移），时间戳不变。"""
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
    _, r_a = pose((t_imu - gyro_lag_ms - h) / 1000.0)
    _, r_b = pose((t_imu - gyro_lag_ms + h) / 1000.0)
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


def pose_errors(d: Path):
    """groundtruth_images.csv 相对合成真值的位置误差（mm）和转角误差（°）。"""
    with open(d / "groundtruth_images.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["flag"] == "ok"]
    t_img = np.load(d / "truth_t_img.npy")
    p, r = flange_pose(t_img / 1000.0)
    r_cam = r @ FLANGE_T_CAM[:3, :3]
    t_cam = p + r @ FLANGE_T_CAM[:3, 3]
    est_t = np.array([[float(x[k]) for k in ("tx_m", "ty_m", "tz_m")] for x in rows])
    est_q = np.array([[float(x[k]) for k in ("qw", "qx", "qy", "qz")] for x in rows])
    pos_err_mm = np.linalg.norm(est_t - t_cam, axis=1) * 1000
    rot_err_deg = np.rad2deg(gt.rotation_angle_between(est_q, gt.matrix_to_quat(r_cam)))
    return pos_err_mm, rot_err_deg


class SyntheticSessionTest(unittest.TestCase):
    def test_recovers_clock_lag_and_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)   # 合成数据里陀螺仪没有滞后
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
            self.assertEqual(report["travel_scale"]["factor"], 1.0)               # 没有 motion_log.json
            pos_err_mm, rot_err_deg = pose_errors(d)
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

    def test_time_correction_removes_gyro_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0), gyro_lag_ms=gt.TIME_CORRECTION_MS)
            report = gt.process(d, d / "handeye.json", d)                        # 默认加时间修正
            self.assertEqual(report["warnings"], [])
            self.assertEqual(report["time_correction"]["applied_ms"], gt.TIME_CORRECTION_MS)
            pos_err_mm, rot_err_deg = pose_errors(d)
            self.assertLess(pos_err_mm.max(), 0.5)
            self.assertLess(rot_err_deg.max(), 0.1)
            gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            _, rot_err_raw = pose_errors(d)
            self.assertGreater(rot_err_raw.max(), 0.15)                          # 47 °/s × 4.27 ms ≈ 0.2°

    def test_travel_scale_along_motion_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            axis = np.array([1.0, 0.0, 0.0])                                     # 合成平移主要沿 x
            write_motion_log(d, "record", "right", P0, P0 + 0.3 * axis)
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            k = gt.TRAVEL_SCALE["base"]["right"]
            self.assertEqual(report["travel_scale"]["factor"], k)
            fixed, raw = read_tum(d / "groundtruth_tum.txt"), read_tum(d / "groundtruth_tum_uncorrected.txt")
            np.testing.assert_array_equal(fixed[:, 0], raw[:, 0])
            diff = fixed[:, 1:4] - raw[:, 1:4]
            np.testing.assert_allclose(raw[:, 4:], fixed[:, 4:], atol=1e-9)    # 姿态不变
            p, _ = flange_pose(np.load(d / "truth_t_img.npy") / 1000.0)
            np.testing.assert_allclose(diff @ axis, (k - 1) * ((p - P0) @ axis), atol=2e-5)
            np.testing.assert_allclose(diff[:, 1:], 0.0, atol=1e-9)             # 垂直方向不动
            self.assertGreater(np.abs(diff @ axis).max(), 0.5e-3)                 # 0.1 m × 0.7%

            write_motion_log(d, "sync-test", "", P0, P0)
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            self.assertEqual(report["travel_scale"]["factor"], 1.0)

            # 旋转录制（主段叫 rotate，没有 translate）：不缩放
            segs = [{"name": "goto_start", "flange": [*(P0 * 1000), 0, 0, 0]},
                    {"name": "rotate", "flange": [*(P0 * 1000), 0, 0, 30]}]
            (d / "motion_log.json").write_text(json.dumps({"kind": "record", "direction": "yaw_left", "segments": segs}))
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            self.assertEqual(report["travel_scale"]["factor"], 1.0)
            self.assertEqual(report["travel_scale"]["source"], "none")
            self.assertEqual(report["rotation_scale"]["source"], "none")              # 没有转轴

    def test_path_set_lookup(self):
        """motion_log 里记了路径组（高速录制）：按组查系数；表里没有的方向不缩放并报警，旧录制（没记）按 base。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            axis = np.array([1.0, 0.0, 0.0])
            segs = [{"name": "goto_start", "flange": [*(P0 * 1000), 0, 0, 0]},
                    {"name": "translate", "flange": [*((P0 + 0.3 * axis) * 1000), 0, 0, 0]}]
            for group, factor, warned in ((None, gt.TRAVEL_SCALE["base"]["right"], False), ("base", gt.TRAVEL_SCALE["base"]["right"], False),
                                          ("fast1", 1.0, True)):
                log = {"kind": "record", "direction": "right", "segments": segs}
                if group:
                    log["parameters"] = {"PATH_SET": group}
                (d / "motion_log.json").write_text(json.dumps(log))
                report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
                ts = report["travel_scale"]
                self.assertEqual((ts["factor"], ts["path_set"]), (factor, group or "base"))
                self.assertEqual(ts["source"], "unverified" if warned else "TRAVEL_SCALE")
                self.assertEqual(any("还没有用标定板验证过行程系数" in w for w in report["warnings"]), warned)
                if warned:
                    np.testing.assert_array_equal(read_tum(d / "groundtruth_tum.txt"),
                                                  read_tum(d / "groundtruth_tum_uncorrected.txt"))
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0, scale_factor=1.002)
            self.assertEqual((report["travel_scale"]["factor"], report["travel_scale"]["source"]), (1.002, "manual"))
            self.assertFalse(any("还没有用标定板验证" in w for w in report["warnings"]))

    def test_rotation_scale_about_pivot(self):
        """俯仰旋转录制：真值姿态相对起点、绕转轴的转角放大 k 倍，其他方向不变，光心不动。"""
        def rotation_log(direction):
            ry = -np.rad2deg(np.arcsin(R0[2, 0]))
            rx, rz = np.rad2deg(np.arctan2(R0[2, 1], R0[2, 2])), np.rad2deg(np.arctan2(R0[1, 0], R0[0, 0]))
            segs = [{"name": "goto_start", "flange": [*(P0 * 1000), rx, ry, rz]},
                    {"name": "rotate", "arc_axis": PITCH_AXIS.tolist(), "pivot_m": PIVOT.tolist()}]
            (d / "motion_log.json").write_text(json.dumps({"kind": "record", "direction": direction, "segments": segs}))

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0), pose=pitch_pose)
            rotation_log("pitch_up")
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            self.assertEqual(report["warnings"], [])
            k = gt.ROTATION_SCALE["base"]["pitch_up"]
            self.assertEqual(report["rotation_scale"]["factor"], k)
            self.assertEqual(report["travel_scale"]["factor"], 1.0)
            fixed, raw = read_tum(d / "groundtruth_tum.txt"), read_tum(d / "groundtruth_tum_uncorrected.txt")
            np.testing.assert_array_equal(fixed[:, 0], raw[:, 0])
            q_fix, q_raw = fixed[:, [7, 4, 5, 6]], raw[:, [7, 4, 5, 6]]
            r_start = R0 @ FLANGE_T_CAM[:3, :3]
            # 未修正真值相对起点的转角 φ（绕转轴）；修正后应多转 (k-1)·φ，且只绕转轴
            phi = np.rad2deg(gt.quat_to_rotvec(gt.quat_mul(q_raw, gt.quat_conj(gt.matrix_to_quat(r_start)))) @ PITCH_AXIS)
            extra = np.rad2deg(gt.quat_to_rotvec(gt.quat_mul(q_fix, gt.quat_conj(q_raw))))
            np.testing.assert_allclose(extra @ PITCH_AXIS, (k - 1) * phi, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(extra - np.outer(extra @ PITCH_AXIS, PITCH_AXIS), axis=1), 0.0, atol=1e-6)
            self.assertGreater(phi.max(), 29.9)
            self.assertAlmostEqual(np.linalg.norm(extra, axis=1).max(), (k - 1) * 30.0, delta=0.002)   # 转满 30° 时约 0.13°
            np.testing.assert_allclose(fixed[:, 1:4], raw[:, 1:4], atol=2e-6)          # 光心不动
            np.testing.assert_allclose(fixed[:, 1:4], np.broadcast_to(PIVOT, (len(fixed), 3)), atol=2e-6)

            rotation_log("yaw_left")                                                    # 偏航不修正
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            self.assertEqual(report["rotation_scale"]["factor"], 1.0)
            self.assertEqual(report["warnings"], [])
            fixed = read_tum(d / "groundtruth_tum.txt")
            np.testing.assert_array_equal(fixed, read_tum(d / "groundtruth_tum_uncorrected.txt"))

            # 高速路径组 fast1：俯仰同一条路径，系数相同；偏航 30° 是新路径，不修正并报警
            for direction, factor, warned in (("pitch_up", gt.ROTATION_SCALE["fast1"]["pitch_up"], False),
                                              ("yaw_left", 1.0, True)):
                rotation_log(direction)
                log = json.loads((d / "motion_log.json").read_text())
                log["parameters"] = {"PATH_SET": "fast1"}
                (d / "motion_log.json").write_text(json.dumps(log))
                report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
                self.assertEqual(report["rotation_scale"]["factor"], factor)
                self.assertEqual(any("还没有用标定板验证过转角系数" in w for w in report["warnings"]), warned)

    def test_duplicate_timestamp_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            with open(d / "img_timeline.csv", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            dup = [rows[1][0], rows[1][1], rows[1][2].replace(".jpg", "_2.jpg"), rows[1][3].replace(".jpg", "_2.jpg")]
            with open(d / "img_timeline.csv", "w", newline="") as f:
                csv.writer(f).writerows([rows[0], rows[1], dup, *rows[2:]])     # ZED 把第一帧抓了两次
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            self.assertEqual(report["images"]["duplicate_timestamp"], 1)
            with open(d / "groundtruth_images.csv", encoding="utf-8") as f:
                flags = [r["flag"] for r in csv.DictReader(f)]
            self.assertEqual(flags[1], "duplicate")
            t = read_tum(d / "groundtruth_tum.txt")[:, 0]
            self.assertEqual(len(t), len(np.unique(t)))

    def test_warns_time_correction_at_other_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            (d / "time_meta.txt").write_text("exposure_us=5000\n")
            report = gt.process(d, d / "handeye.json", d)
            self.assertTrue(any("时间修正" in w for w in report["warnings"]))

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

    def test_truncated_last_line_dropped(self):
        """zed_record 崩溃时 zed_imu.csv、img_timeline.csv 的最后一行只写了一半（10-13 实测）。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, np.random.default_rng(0))
            n_img = len((d / "img_timeline.csv").read_text(encoding="utf-8").splitlines()) - 1
            for name, keep in (("zed_imu.csv", 16), ("img_timeline.csv", 20)):
                raw = (d / name).read_bytes()
                last = raw.rstrip(b"\r\n").rfind(b"\n") + 1
                (d / name).write_bytes(raw[:last + keep])   # 最后一行只剩开头几个字符，没有换行
            report = gt.process(d, d / "handeye.json", d, time_correction_ms=0.0)
            self.assertEqual(sum("最后一行没写完" in w for w in report["warnings"]), 2)
            self.assertEqual(report["images"]["total"], n_img - 1)


if __name__ == "__main__":
    unittest.main()
