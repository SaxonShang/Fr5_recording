"""gt_board_check.py 的合成数据测试（在笔记本上运行：要 OpenCV 和标定仓库；Box 上会跳过）。

用标定仓库 sim/synthetic.py 的渲染器按真实相机位姿画出标定板图像，真值故意晚 4 ms，
检查脚本能从图像里找回这 4 ms，静止帧一致。

    python -B -m unittest test_gt_board_check
"""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fr5_groundtruth as gt

try:
    import cv2
    import gt_board_check as bc
    SKIP = None if (bc.CALIB_REPO / "sim" / "synthetic.py").exists() else "找不到标定仓库"
except ImportError as exc:  # Box 上 OpenCV 导入不了
    SKIP = str(exc)

EPOCH_MS = 1_791_600_000_000.0
TRUTH_LATE_MS = 4.0           # 真值比真实运动晚 4 ms：应该找回 δ ≈ +4 ms


def _ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def camera_in_cam0(t_s):
    """相机相对起始位姿：静止 → 绕自身竖直轴摆 ±10°（1 个来回）→ 静止 → 沿 x 平移 100 mm → 静止。"""
    yaw = 10.0 * np.sin(np.pi * (t_s - 0.8)) if 0.8 <= t_s <= 2.8 else 0.0
    u = np.clip((t_s - 3.3) / 1.0, 0.0, 1.0)
    pose = np.eye(4)
    pose[:3, :3] = _ry(yaw)
    pose[0, 3] = 0.1 * (3 * u ** 2 - 2 * u ** 3)
    return pose


def write_session(d: Path, syn, board, k, rng) -> None:
    base_t_cam0 = np.eye(4)
    base_t_cam0[:3, :3] = gt.euler_to_matrix(-95.0, 3.0, -100.0)
    base_t_cam0[:3, 3] = [0.6, -0.45, 0.55]
    centre_b = bc_board_centre(board)
    cam0_t_board = np.eye(4)
    cam0_t_board[:3, :3] = np.diag([1.0, -1.0, -1.0])            # 标定板正对相机
    cam0_t_board[:3, 3] = np.array([0.05, -0.03, 1.0]) - cam0_t_board[:3, :3] @ centre_b
    base_t_board = base_t_cam0 @ cam0_t_board

    def base_t_cam(t_ms):
        return base_t_cam0 @ camera_in_cam0(t_ms / 1000.0)

    t_arm = np.arange(0.0, 5000.0, 10.0)
    poses = [base_t_cam(t) for t in t_arm]
    with open(d / "groundtruth_arm_rate_tum.txt", "w", newline="\n") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, pose in zip(t_arm, poses):
            q = gt.matrix_to_quat(pose[:3, :3])
            f.write(f"{(EPOCH_MS + t + TRUTH_LATE_MS) / 1000:.6f} {pose[0, 3]:.9f} {pose[1, 3]:.9f} {pose[2, 3]:.9f} "
                    f"{q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {q[0]:.12f}\n")
    arm = (EPOCH_MS + t_arm + TRUTH_LATE_MS, np.array([p[:3, 3] for p in poses]),
           gt.matrix_to_quat(np.array([p[:3, :3] for p in poses])))

    texture, tex_to_board = syn.board_texture(board)
    (d / "left").mkdir()
    t_img = np.round(200.0 + (1000.0 / 15) * np.arange(70))     # 图像时间戳是整 ms
    r_gt, p_gt, _ = bc.camera_poses(EPOCH_MS + t_img, arm)
    with open(d / "groundtruth_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zed_timestamp_ms", "filename_left", "flag", "arm_span_ms",
                    "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"])
        for i, t in enumerate(t_img):
            cam_t_board = np.linalg.inv(base_t_cam(t)) @ base_t_board      # 按真实时刻渲染
            image = syn.render_view(texture, tex_to_board, cam_t_board, k, rng, blur=0.6, noise=2.0)
            name = f"left/{EPOCH_MS + t:.0f}.jpg"
            cv2.imwrite(str(d / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            q = gt.matrix_to_quat(r_gt[i])
            w.writerow([f"{EPOCH_MS + t:.0f}", name, "ok", "10.0", *[f"{v:.9f}" for v in p_gt[i]],
                        f"{q[1]:.12f}", f"{q[2]:.12f}", f"{q[3]:.12f}", f"{q[0]:.12f}"])
    (d / "groundtruth_report.json").write_text(json.dumps(
        {"time_offset": {"dt_ms": 0.0}, "exposure_centre": {"applied_ms": 0.0}}))


def bc_board_centre(board):
    from handeye.board import board_outline
    return board_outline(board).mean(axis=0)


@unittest.skipIf(SKIP, SKIP)
class BoardCheckTest(unittest.TestCase):
    def test_finds_time_error_and_static_consistency(self):
        import sys
        sys.path.insert(0, str(bc.CALIB_REPO))
        from handeye.board import read_board
        spec = importlib.util.spec_from_file_location("synthetic", bc.CALIB_REPO / "sim" / "synthetic.py")
        syn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(syn)
        syn.WIDTH, syn.HEIGHT = 1920, 1080                                  # 录制用 HD1080
        k = bc.load_camera(bc.CAMERA_YAML)
        board = read_board(bc.CALIB_REPO / bc.TARGET_YAML)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_session(d, syn, board, k, np.random.default_rng(0))
            r = bc.check(d, bc.CALIB_REPO, bc.CAMERA_YAML, bc.CALIB_REPO / bc.TARGET_YAML, d,
                         scan_ms=15.0)
            print("\n" + bc.summary(r))
            self.assertLess(r["check_csv_vs_arm_rate_max_mm"], 0.01)
            self.assertEqual(r["images"]["board_used"], 70)
            self.assertGreaterEqual(r["images"]["moving"], bc.MIN_MOVING_FRAMES)
            self.assertLess(r["static"]["rms_mm"], 0.5)
            self.assertEqual(len(r["static_groups"]), 3)
            self.assertTrue(all(p["board_diff_mm"] < 0.5 for p in r["static_group_pairs"]))
            scan = r["time_scan"]
            self.assertFalse(scan["at_edge"])
            self.assertAlmostEqual(scan["best_delta_ms"], TRUTH_LATE_MS, delta=1.0)
            self.assertLess(scan["rms_at_best_mm"], scan["rms_at_0_mm"])
            self.assertTrue((d / "board_check_frames.csv").exists())


if __name__ == "__main__":
    unittest.main()
