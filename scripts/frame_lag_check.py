#!/usr/bin/env python3
"""检查图像内容和时间戳是否错位（在笔记本上运行：Box 上 OpenCV 导入不了）。不需要标定板，也不用机械臂数据。

10-13 发现 ZED 偶尔整条录制的图像比时间戳晚一帧：时间戳本身没错（IMU 的曝光标记正常），但图像内容是上一帧的，
日志里看不出来（实验记录问题 26）。真值按时间戳取位姿，晚一帧时同步摆动处的姿态差约 1.3°。

方法：同步摆动时相机绕光心转动（光心偏离约 1 mm），相邻两帧的特征点（LK 光流）按纯旋转拟合出转角，
得到图像看到的角速度大小；和陀螺仪在同一帧间隔内的平均角速度大小比较，把图像时间整体平移 s，
找最一致的 s。正值表示图像内容比时间戳晚。正常约 0 ms（10-10～10-13 的 12 条在 −0.2～+0.5 ms），晚一帧约 +66.7 ms。

每条录制拷到笔记本后都运行一次（正式录制和标定板验证录制都要）：

    python frame_lag_check.py <save_data 目录> [<save_data 目录> ...]

需要 save_data 里的 img_timeline.csv、left/、zed_imu.csv；有 motion_log.json 时只用同步摆动段（前后各放宽 0.3 s），
没有时用陀螺仪角速度超过 MIN_GYRO_DEG_S 的帧。结果写进 save_data/frame_lag_report.json。有异常时退出码为 1。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 标定仓库约定不留 __pycache__

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import fr5_groundtruth as gt  # noqa: E402
from gt_board_check import CAMERA_YAML, load_camera  # noqa: E402

SCALE = 0.5                     # 检测特征前把图像缩小一半（1080p → 540p），够用且快
WINDOW_MARGIN_MS = 300.0        # 同步摆动段前后各放宽这么多
MIN_GYRO_DEG_S = 5.0            # 没有 motion_log.json 时，用陀螺仪角速度超过它的帧
MAX_FEATURES, MIN_INLIERS = 400, 30
SCAN_MS, SCAN_STEP_MS = 150.0, 0.5
MIN_PAIRS = 40                  # 可用的相邻帧对少于这么多就不下结论
LAG_OK_MS = 20.0                # |延迟| 超过它就算异常（正常约 0 ms，晚一帧约 +66.7 ms）


def read_gray(path: Path) -> np.ndarray:
    # cv2.imread 在 Windows 上打不开中文路径，先读字节再解码
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def bearings(pts: np.ndarray, k: np.ndarray) -> np.ndarray:
    """像素点（已缩放的图像）→ 单位视线方向。"""
    b = np.c_[(pts[:, 0] - k[0, 2]) / k[0, 0], (pts[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(pts))]
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def pure_rotation_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    """两帧中同一批点的视线方向 a、b，按纯旋转 b ≈ R a 拟合（迭代剔除离群点，例如走动的人），返回转角 °。"""
    keep = np.ones(len(a), bool)
    r = np.eye(3)
    for _ in range(3):
        if keep.sum() < MIN_INLIERS:
            return None
        u, _, vt = np.linalg.svd(b[keep].T @ a[keep])
        r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        err = np.degrees(np.arccos(np.clip(np.sum((a @ r.T) * b, axis=1), -1.0, 1.0)))
        keep = err < max(3.0 * np.median(err[keep]), 0.05)
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def frame_pair_rotation(prev: np.ndarray, image: np.ndarray, k: np.ndarray) -> float | None:
    p0 = cv2.goodFeaturesToTrack(prev, MAX_FEATURES, 0.01, 12)
    if p0 is None:
        return None
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, image, p0, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    return pure_rotation_deg(bearings(p0.reshape(-1, 2)[ok], k), bearings(p1.reshape(-1, 2)[ok], k))


def gyro_interval_mean(t_imu: np.ndarray, w_imu: np.ndarray):
    """返回函数 f(a, b)：陀螺仪角速度大小在 [a, b] 内的平均（ms、°/s）。
    按梯形积分：每个样本只代表它自己的时刻；按左矩形积分会把陀螺仪整体推后半个采样间隔（200 Hz 时 2.5 ms）。"""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w_imu[:-1] + w_imu[1:]) * np.diff(t_imu))])
    return lambda a, b: (np.interp(b, t_imu, cum) - np.interp(a, t_imu, cum)) / (b - a)


def wiggle_windows(save_data: Path, t_img: np.ndarray, gyro_mean) -> tuple[np.ndarray, str]:
    """每张图是否在同步摆动段里，以及判断依据。"""
    path = save_data / "motion_log.json"
    if path.exists():
        segs = [s for s in json.loads(path.read_text(encoding="utf-8")).get("segments", [])
                if s.get("kind") == "move" and s.get("name", "").startswith("wiggle") and s.get("start_box_ns")]
        if segs:
            use = np.zeros(len(t_img), bool)
            for s in segs:
                use |= ((t_img >= s["start_box_ns"] / 1e6 - WINDOW_MARGIN_MS)
                        & (t_img <= s["end_box_ns"] / 1e6 + WINDOW_MARGIN_MS))
            return use, "motion_log"
    rate = gyro_mean(t_img - 33.0, t_img + 33.0)
    return rate > MIN_GYRO_DEG_S, "gyro"


def check(save_data: Path, camera_yaml: Path = CAMERA_YAML, output: Path | None = None) -> dict:
    k = load_camera(camera_yaml)
    k[:2] *= SCALE
    with open(save_data / "img_timeline.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError(f"{save_data / 'img_timeline.csv'} 只有 {len(rows)} 帧")
    t_img = np.array([float(r["zed_timestamp_ms"]) for r in rows])
    period = float(np.median(np.diff(t_img)))
    t_imu, gyro, _ = gt.load_imu(save_data / "zed_imu.csv")
    gyro_mean = gyro_interval_mean(t_imu, np.linalg.norm(gyro, axis=1))
    use, source = wiggle_windows(save_data, t_img, gyro_mean)

    t_mid, half, w_img = [], [], []
    prev, prev_t = None, None
    for i, r in enumerate(rows):
        if not use[i]:
            prev = None
            continue
        image = read_gray(save_data / r["filename_left"])
        dt = t_img[i] - prev_t if prev is not None else None
        if dt is not None and 0.5 * period < dt < 1.5 * period:     # 中间丢了帧的不用
            deg = frame_pair_rotation(prev, image, k)
            if deg is not None:
                t_mid.append((t_img[i] + prev_t) / 2.0)
                half.append(dt / 2.0)
                w_img.append(deg / (dt / 1000.0))
        prev, prev_t = image, t_img[i]
    t_mid, half, w_img = map(np.array, (t_mid, half, w_img))

    result = {"session": str(save_data), "camera_yaml": str(camera_yaml), "frame_period_ms": period,
              "window_source": source, "pairs": int(len(w_img))}
    if len(w_img) < MIN_PAIRS:
        result.update(ok=False, verdict=f"可用的相邻帧对只有 {len(w_img)} 对（至少 {MIN_PAIRS}），没法判断")
    else:
        shifts = np.arange(-SCAN_MS, SCAN_MS + SCAN_STEP_MS / 2, SCAN_STEP_MS)
        cost = np.array([np.sqrt(np.mean((gyro_mean(t_mid - half - s, t_mid + half - s) - w_img) ** 2))
                         for s in shifts])
        i = int(np.argmin(cost))
        lag = float(shifts[i])
        if 0 < i < len(shifts) - 1:                                   # 抛物线细化
            a, b, c = cost[i - 1], cost[i], cost[i + 1]
            if a - 2 * b + c > 0:
                lag += float(0.5 * (a - c) / (a - 2 * b + c) * SCAN_STEP_MS)
        frames_late = int(round(lag / period))
        ok = bool(abs(lag) <= LAG_OK_MS)
        result.update(
            lag_ms=lag, frames_late=frames_late, ok=ok, at_scan_edge=i in (0, len(shifts) - 1),
            rms_deg_s={"at_0": float(np.interp(0.0, shifts, cost)), "at_best": float(cost[i]),
                       "at_plus_one_frame": float(np.interp(period, shifts, cost))},
            verdict=("正常" if ok else f"图像比时间戳晚约 {frames_late} 帧" if frames_late else "图像和时间戳错位"),
            shifts_ms=shifts.round(2).tolist(), rms_curve_deg_s=cost.round(4).tolist())
    out = output or save_data
    out.mkdir(parents=True, exist_ok=True)
    (out / "frame_lag_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def summary(r: dict) -> str:
    if "lag_ms" not in r:
        return f"{r['session']}：{r['verdict']}"
    rms = r["rms_deg_s"]
    return (f"{r['session']}：{r['verdict']}。{r['pairs']} 对相邻帧（{'同步摆动段' if r['window_source'] == 'motion_log' else '陀螺仪在转的帧'}），"
            f"图像比时间戳晚 {r['lag_ms']:+.1f} ms 时和陀螺仪最一致（帧间隔 {r['frame_period_ms']:.1f} ms）；"
            f"角速度差 RMS：0 ms 处 {rms['at_0']:.2f}、最佳 {rms['at_best']:.2f}、晚一帧处 {rms['at_plus_one_frame']:.2f} °/s"
            + ("；落在扫描边缘" if r["at_scan_edge"] else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("save_data", type=Path, nargs="+", help="录制的 save_data 目录（可以给多个）")
    parser.add_argument("--camera", type=Path, help=f"内参，默认 {CAMERA_YAML.name}（同目录）")
    args = parser.parse_args(argv)
    bad = 0
    for sd in args.save_data:
        try:
            r = check(sd, args.camera or CAMERA_YAML)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{sd}：失败：{exc}")
            bad += 1
            continue
        print(summary(r))
        bad += not r["ok"]
    if bad:
        print(f"\n{bad} 条有问题：图像晚一帧的要重录")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
