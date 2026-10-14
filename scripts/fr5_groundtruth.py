#!/usr/bin/env python3
"""由同一段录制里的 fr5_record.py 和 zed_record 数据生成相机位姿真值（离线运行，只依赖 numpy）。

    python3 fr5_groundtruth.py SESSION_DIR --handeye handeye_s003_result.json

SESSION_DIR 里需要 fr5_state.csv（fr5_record.py）以及 zed_imu.csv、img_timeline.csv（zed_record）。
--handeye 接受标定仓库的 result.json（取 chosen.flange_T_left_camera），或一个 4x4 矩阵的 JSON。

步骤：
1. 控制器时钟 -> Box 系统时间：对“到达时间 - 控制器时间”每 2 s 取最小值（最小延迟），
   直线拟合出偏差和漂移。
2. 机械臂与相机之间剩下的时间差 Δt：ZED 陀螺仪角速度大小与法兰姿态差分得到的角速度大小
   做互相关。角速度大小与坐标系无关，不需要外参，但录制里要有足够快的转动（同步摆动）。
3. 曝光中心：ZED 把在左目采集期间取到的 IMU 样本标成 image_sync_trigger=1。每帧取这些样本
   时间的中点，减去图像时间戳，整段平均得到“曝光中心 - 图像时间戳”（实测约 -3 ms）。
4. 标定板验证得到的修正（见开头的 TIME_CORRECTION_MS、TRAVEL_SCALE、ROTATION_SCALE）：
   - 时间：Δt 对齐的是陀螺仪，真值仍比图像晚一个固定量，取真值的时刻再晚 TIME_CORRECTION_MS；
   - 行程：机械臂实际行程比报告的长，法兰平移沿本条录制的运动方向、相对路径起点放大（方向和
     路径两端从 motion_log.json 读；不是平移录制就不缩放）；
   - 转角：俯仰旋转录制里机械臂实际转角比报告的大，法兰姿态相对本条起点、绕转轴的转角放大，
     绕转动中心（左目光心）转（转轴和转动中心从 motion_log.json 读；不是旋转录制就不放大）。
5. 相机真值 base_T_camera = base_T_flange(曝光中心 + 时间修正 - Δt) · flange_T_left_camera，
   平移线性插值，旋转 SLERP。输出里的时间戳仍是图像原来的时间戳，方便和 SLAM 结果对齐。
   时间戳重复的图（ZED 偶尔把同一帧抓两次）只保留第一张。

时间约定：输出时间戳都是 ZED / Box 系统时间（Unix 秒）。机械臂样本在 ZED 时间轴上的时刻
= 映射到 Box 时间的控制器时间 + Δt。假设 ZED 图像和 IMU 时间戳在同一时钟上，且 image_sync_trigger
正确标出了左目采集期间的 IMU 样本（ZED X 硬件同步）。

输出（默认写在 SESSION_DIR）：
  groundtruth_tum.txt           每张图一行：timestamp tx ty tz qx qy qz qw（s, m），左目相机在基坐标系下
  groundtruth_tum_uncorrected.txt  同上，但不加第 4 步的修正（对照用）
  groundtruth_arm_rate_tum.txt  每个机械臂状态包一行，格式同上（已加修正）
  groundtruth_images.csv        每张图的真值和标记
  groundtruth_report.json       时钟拟合、Δt、修正、质量检查和警告
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CLOCK_WINDOW_MS = 2000.0   # 时钟拟合时取最小值的窗口
MIN_WINDOW_PACKETS = 20
MAX_GAP_MS = 30.0          # 插值两侧机械臂样本间隔超过它就不给真值
STATIC_DEG_S = 0.3         # 机械臂角速度低于它视为静止，用来估陀螺零偏
MIN_STATIC_MS = 500.0
MIN_MOTION_DEG_S = 10.0    # 估 Δt 至少需要的峰值角速度
MIN_PEAK_CORR = 0.9
MAX_SEGMENT_LAG_DIFF_MS = 2.0
MIN_SPLIT_S = 20.0         # 录制至少这么长，才把前 40% 和后 40% 分开估 Δt、互相核对（sync-test 和正式录制约 30 s）
MIN_EXPOSURE_FRAMES = 20   # 估曝光中心至少需要的带标记帧数
MAX_EXPOSURE_SD_MS = 3.0   # 各帧曝光中心偏差的标准差上限
# 标定板验证（2026-10-12，见 data_recording/实验记录.md）得到的修正
TIME_CORRECTION_MS = 4.27            # 真值比图像晚的固定量（3 条录制、12 个位置平均，±0.26 ms）
TIME_CORRECTION_EXPOSURE_US = 10000  # 上面的值是在这个曝光下测的；曝光不同时报警
# 行程、转角系数按路径组（motion_log.json 里 parameters.PATH_SET；10-12 / 10-13 的录制没有这一项，是 "base"）
# 和录制方向查。表里没有的方向不缩放、并报警（这条路径还没用标定板验证过）。
# 机械臂实际行程 / 报告行程。base：start_pose.json 起点 ±150 mm 的 6 段平移（2026-10-10/12 标定板验证）
TRAVEL_SCALE = {
    "base": {"right": 1.0070, "left": 1.0070, "up": 1.0070, "down": 1.0070,
             "forward": 1.0015, "backward": 1.0015},
    "fast1": {},              # 高速录制的新平移路径（10-14 上午的设计，没有录过）
    "fast2": {},              # 高速录制：按拖动标记定的路径（10-14 起），待标定板验证
}
# 机械臂实际转角 / 报告转角（2026-10-13 标定板验证：pitch_up 3 条 +0.39～+0.46%，平均 +0.43%；
# pitch_down 2 条 +0.50、+0.54%，平均 +0.52%；偏航 10°、横滚 30° 在 ±0.03% 以内，不修正）。
ROTATION_SCALE = {
    "base": {"pitch_up": 1.0043, "pitch_down": 1.0052, "yaw_left": 1.0, "yaw_right": 1.0,
             "roll_left": 1.0, "roll_right": 1.0},
    "fast1": {"pitch_up": 1.0043, "pitch_down": 1.0052},   # 俯仰和 base 同一条路径；偏航 30°、横滚 90° 待验证
    "fast2": {},              # 按拖动标记定的转动中心和转角，俯仰也不再是 base 那条路径，全部待验证
}

ARM_COLUMNS = ["box_realtime_ns", "ctrl_unix_ms", "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg",
               "robot_state", "main_code", "sub_code", "tool", "user"]


# ------------------------------------------------------------------ 旋转（四元数一律 w, x, y, z）
def euler_to_matrix(rx_deg, ry_deg, rz_deg) -> np.ndarray:
    """FR5 姿态（度）-> 旋转矩阵，R = Rz(rz)·Ry(ry)·Rx(rx)。"""
    rx, ry, rz = (np.deg2rad(np.asarray(a, dtype=float)) for a in (rx_deg, ry_deg, rz_deg))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    r = np.empty(np.shape(rx) + (3, 3))
    r[..., 0, 0] = cz * cy
    r[..., 0, 1] = cz * sy * sx - sz * cx
    r[..., 0, 2] = cz * sy * cx + sz * sx
    r[..., 1, 0] = sz * cy
    r[..., 1, 1] = sz * sy * sx + cz * cx
    r[..., 1, 2] = sz * sy * cx - cz * sx
    r[..., 2, 0] = -sy
    r[..., 2, 1] = cy * sx
    r[..., 2, 2] = cy * cx
    return r


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 单位四元数（Shepperd 方法，w >= 0）。"""
    r = np.asarray(r, dtype=float)
    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    cand = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                     1 - m00 + m11 - m22, 1 - m00 - m11 + m22], axis=-1)
    k = np.argmax(cand, axis=-1)
    s = 2.0 * np.sqrt(np.maximum(np.take_along_axis(cand, k[..., None], -1)[..., 0], 1e-300))
    q = np.empty(r.shape[:-2] + (4,))
    variants = [
        (s / 4, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s),
        ((m21 - m12) / s, s / 4, (m01 + m10) / s, (m02 + m20) / s),
        ((m02 - m20) / s, (m01 + m10) / s, s / 4, (m12 + m21) / s),
        ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4),
    ]
    for case, comps in enumerate(variants):
        sel = k == case
        for j in range(4):
            q[..., j] = np.where(sel, comps[j], q[..., j] if case else 0.0)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q[..., i] for i in range(4))
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """四元数 -> 旋转向量（弧度，转角在 [0, π]）。"""
    q = np.where(q[..., :1] < 0, -q, q)
    n = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    return np.where(n > 1e-15, q[..., 1:] / np.where(n > 1e-15, n, 1.0), 0.0) * 2.0 * np.arctan2(n, q[..., :1])


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """旋转向量（弧度） -> 四元数。"""
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(ang > 1e-15, np.sin(ang / 2) / np.where(ang > 1e-15, ang, 1.0), 0.5)
    return np.concatenate([np.cos(ang / 2), k * v], axis=-1)


def rotation_angle_between(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """两个姿态之间的转角（弧度）。用 atan2，小角度也不丢精度。"""
    rel = quat_mul(quat_conj(q0), q1)
    return 2.0 * np.arctan2(np.linalg.norm(rel[..., 1:], axis=-1), np.abs(rel[..., 0]))


def slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where(dot[..., None] < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
    small = theta < 1e-9
    sin_t = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / sin_t)
    w1 = np.where(small, u, np.sin(u * theta) / sin_t)
    q = w0[..., None] * q0 + w1[..., None] * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ------------------------------------------------------------------ 读入
def load_arm(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        missing = [c for c in ARM_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path} 缺少列：{missing}")
        idx = [header.index(c) for c in ARM_COLUMNS]
        rows = [[row[i] for i in idx] for row in reader if row and row[idx[1]] != ""]
    if len(rows) < 10:
        raise ValueError(f"{path} 只有 {len(rows)} 行有效数据")
    data = np.array(rows, dtype=float)
    arm = {name: data[:, i] for i, name in enumerate(ARM_COLUMNS)}
    arm["box_ms"] = arm.pop("box_realtime_ns") / 1e6
    arm["ctrl_ms"] = arm.pop("ctrl_unix_ms")
    # 去掉控制器时间重复或倒退的包（同一个状态样本发了两次）
    keep = np.r_[True, np.diff(arm["ctrl_ms"]) > 0]
    keep &= np.maximum.accumulate(arm["ctrl_ms"]) == arm["ctrl_ms"]
    arm = {k: v[keep] for k, v in arm.items()}
    arm["dropped_duplicates"] = int((~keep).sum())
    return arm


def complete_lines(path: Path, warnings: list[str] | None = None) -> list[str]:
    """文件的各行。zed_record 没有正常退出时最后一行可能只写了一半（没有换行），去掉它。"""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) > 1 and not lines[-1].endswith("\n"):
        lines.pop()
        if warnings is not None:
            warnings.append(f"{path.name} 最后一行没写完（zed_record 可能没有正常退出），已去掉")
    return lines


def load_imu(path: Path, warnings: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """返回 IMU 时间（ms）、陀螺仪（°/s）、image_sync_trigger（没有这一列时为 None）。"""
    lines = complete_lines(path, warnings)
    header = lines[0].strip().split(",")
    names = ["zed_imu_timestamp_ns", "gx", "gy", "gz"]
    has_trigger = "image_sync_trigger" in header
    cols = [header.index(c) for c in names + (["image_sync_trigger"] if has_trigger else [])]
    data = np.loadtxt(lines[1:], delimiter=",", usecols=cols, ndmin=2)
    t_ms = data[:, 0] / 1e6
    t_ms, first = np.unique(t_ms, return_index=True)
    return t_ms, data[first, 1:4], (data[first, 4] if has_trigger else None)


def exposure_centre(t_img: np.ndarray, t_imu: np.ndarray, trigger: np.ndarray | None) -> dict | None:
    """每帧：前后半帧内 trigger=1 的 IMU 样本，取最早和最晚的中点，减去图像时间戳。"""
    if trigger is None or len(t_img) < 3:
        return None
    flagged = t_imu[trigger == 1]
    half = float(np.median(np.diff(t_img))) / 2
    offsets, widths = [], []
    for t in t_img[1:]:  # 第一帧的 IMU 关联不可靠，跳过
        s = flagged[(flagged > t - half) & (flagged < t + half)]
        if len(s) >= 2:
            offsets.append((s.min() + s.max()) / 2 - t)
            widths.append(s.max() - s.min())
    if len(offsets) < MIN_EXPOSURE_FRAMES:
        return None
    o = np.array(offsets)
    keep = np.abs(o - np.median(o)) <= max(3 * 1.4826 * np.median(np.abs(o - np.median(o))), 1.0)
    return {"offset_ms": float(o[keep].mean()), "sd_ms": float(o[keep].std()), "frames": int(keep.sum()),
            "median_flagged_span_ms": float(np.median(widths))}


def read_time_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)


def load_images(path: Path, warnings: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    rows = list(csv.DictReader(complete_lines(path, warnings)))
    return np.array([float(r["zed_timestamp_ms"]) for r in rows]), [r["filename_left"] for r in rows]


def load_handeye(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    info = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(data, list):
        t = np.array(data, dtype=float)
        info["source"] = "4x4 matrix"
    else:
        if data.get("status") != "accepted":
            raise ValueError(f"{path}: 手眼标定结果不是 accepted（status={data.get('status')}）")
        frames = data.get("frames", {})
        if frames.get("pose_moving") != "flange" or frames.get("pose_reference") != "base":
            raise ValueError(f"{path}: 手眼结果的坐标系不是 法兰/基座：{frames}")
        t = np.array(data["chosen"]["flange_T_left_camera"], dtype=float)
        info.update(source="calibration result.json", method=data["chosen"].get("method"),
                    status=data["status"], frames=frames)
    rot = t[:3, :3] if t.shape == (4, 4) else None
    if (rot is None or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or np.linalg.det(rot) < 0):
        raise ValueError(f"{path}: flange_T_left_camera 不是合法的 4x4 刚体变换")
    info["flange_T_left_camera"] = t.tolist()
    return t, info


# ------------------------------------------------------------------ 时间对齐
def fit_clock(ctrl_ms: np.ndarray, box_ms: np.ndarray) -> dict:
    """box ≈ ctrl + offset + slope·(ctrl - c0)，用每个窗口的最小延迟拟合。"""
    d = box_ms - ctrl_ms
    c0 = float(ctrl_ms[0])
    win = np.floor((ctrl_ms - c0) / CLOCK_WINDOW_MS).astype(int)
    order = np.lexsort((d, win))
    first = np.r_[True, win[order][1:] != win[order][:-1]]
    sel = order[first]
    sel = sel[np.bincount(win)[win[sel]] >= MIN_WINDOW_PACKETS]
    x, y = ctrl_ms[sel] - c0, d[sel]
    warnings = []
    if len(sel) < 3:
        offset, slope, keep = float(d.min()), 0.0, np.ones(len(sel), bool)
        warnings.append("录制太短（不到 3 个时钟窗口），没有拟合时钟漂移")
    else:
        keep = np.ones(len(sel), bool)
        for _ in range(5):
            a = np.c_[np.ones(keep.sum()), x[keep]]
            (offset, slope), *_ = np.linalg.lstsq(a, y[keep], rcond=None)
            resid = y - (offset + slope * x)
            mad = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
            new = np.abs(resid) <= max(3 * mad, 0.5)
            if (new == keep).all():
                break
            keep = new
    window_resid = y - (offset + slope * x)
    all_resid = d - (offset + slope * (ctrl_ms - c0))
    if np.abs(window_resid[keep]).max(initial=0) > 1.0:
        warnings.append("时钟拟合残差超过 1 ms，可能有时钟跳变")
    if all_resid.min() < -2.0:
        warnings.append(f"有包比拟合的最小延迟还早 {-all_resid.min():.1f} ms，时钟可能跳变或不是线性漂移")
    return {
        "c0_ctrl_unix_ms": c0,
        "offset_ms": float(offset),
        "drift_ppm": float(slope * 1e6),
        "windows": int(len(sel)),
        "windows_used": int(keep.sum()),
        "window_residual_ms_max": float(np.abs(window_resid[keep]).max(initial=0)),
        "latency_above_floor_ms": {"min": float(all_resid.min()), "median": float(np.median(all_resid)),
                                   "p99": float(np.percentile(all_resid, 99))},
        "warnings": warnings,
    }


def map_ctrl(ctrl_ms: np.ndarray, clock: dict) -> np.ndarray:
    c0 = clock["c0_ctrl_unix_ms"]
    return ctrl_ms + clock["offset_ms"] + clock["drift_ppm"] * 1e-6 * (ctrl_ms - c0)


def arm_angular_speed(t_ms: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """相邻样本之间的角速度大小（°/s），时间取中点；跨越丢包的不要。"""
    dt = np.diff(t_ms)
    ang = np.rad2deg(rotation_angle_between(quat[:-1], quat[1:]))
    ok = (dt > 0) & (dt <= MAX_GAP_MS)
    return ((t_ms[:-1] + t_ms[1:]) / 2)[ok], (ang[ok] / (dt[ok] / 1000.0))


def static_intervals(t_ms: np.ndarray, speed: np.ndarray) -> list[tuple[float, float]]:
    still = speed < STATIC_DEG_S
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    out = []
    for s, e in zip(edges[::2], edges[1::2]):
        if t_ms[e - 1] - t_ms[s] >= MIN_STATIC_MS:
            out.append((float(t_ms[s]), float(t_ms[e - 1])))
    return out


def gyro_bias(t_imu: np.ndarray, gyro: np.ndarray, static: list, margin_ms: float) -> np.ndarray | None:
    mask = np.zeros(len(t_imu), bool)
    for a, b in static:
        mask |= (t_imu >= a + margin_ms) & (t_imu <= b - margin_ms)
    return np.median(gyro[mask], axis=0) if mask.sum() >= 100 else None


def _lag_score(lag, t_imu, g, t_arm, w_arm):
    tq = t_imu - lag
    idx = np.searchsorted(t_arm, tq)
    ok = (idx > 0) & (idx < len(t_arm))
    idx = np.clip(idx, 1, len(t_arm) - 1)
    ok &= (t_arm[idx] - t_arm[idx - 1]) <= MAX_GAP_MS
    if ok.sum() < 200:
        return np.nan
    a = np.interp(tq[ok], t_arm, w_arm)
    b = g[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def estimate_lag(t_imu, g, t_arm, w_arm, max_lag_ms: float) -> dict:
    """在 ±max_lag_ms 内找相关系数最大的 Δt：先 2 ms 粗搜，再 0.1 ms 细搜，最后抛物线插值。"""
    coarse = np.arange(-max_lag_ms, max_lag_ms + 1e-9, 2.0)
    s = np.array([_lag_score(l, t_imu, g, t_arm, w_arm) for l in coarse])
    if np.all(np.isnan(s)):
        raise ValueError("机械臂和 IMU 数据在时间上没有足够重叠")
    best = coarse[np.nanargmax(s)]
    fine = np.arange(best - 4.0, best + 4.0 + 1e-9, 0.1)
    sf = np.array([_lag_score(l, t_imu, g, t_arm, w_arm) for l in fine])
    i = int(np.nanargmax(sf))
    lag = fine[i]
    near = np.abs(fine - lag) <= 1.5
    if near.sum() >= 5 and not np.isnan(sf[near]).any():
        c2, c1, _ = np.polyfit(fine[near] - lag, sf[near], 2)
        if c2 < 0:
            lag += float(np.clip(-c1 / (2 * c2), -1.5, 1.5))
    peak = _lag_score(lag, t_imu, g, t_arm, w_arm)
    side = [_lag_score(lag + o, t_imu, g, t_arm, w_arm) for o in (-10.0, 10.0)]
    return {"dt_ms": float(lag), "peak_corr": peak, "corr_at_pm10ms": side,
            "at_search_edge": bool(abs(best) >= max_lag_ms - 2.0)}


# ------------------------------------------------------------------ 插值与输出
def interpolate(t_query, t_arm, pos, quat):
    idx = np.searchsorted(t_arm, t_query, side="right")
    inside = (idx > 0) & (idx < len(t_arm))
    i1 = np.clip(idx, 1, len(t_arm) - 1)
    i0 = i1 - 1
    span = t_arm[i1] - t_arm[i0]
    u = np.clip((t_query - t_arm[i0]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    p = pos[i0] + u[:, None] * (pos[i1] - pos[i0])
    q = slerp(quat[i0], quat[i1], u)
    return p, q, inside, span, i0, i1


def camera_pose(p_flange, q_flange, flange_t_cam):
    r_f = quat_to_matrix(q_flange)
    r_c = r_f @ flange_t_cam[:3, :3]
    t_c = p_flange + r_f @ flange_t_cam[:3, 3]
    return t_c, matrix_to_quat(r_c)


def path_set(log: dict) -> str:
    """录制用的路径组（run_session.PATH_SET）；10-12 / 10-13 的录制没有记，是 base。"""
    return str((log.get("parameters") or {}).get("PATH_SET") or "base")


def _lookup(table: dict, log: dict, direction, factor: float | None, source: str) -> dict:
    """按路径组和方向查系数。factor 给了就用它（manual）；表里没有这个方向时为 1，标出 unverified。"""
    group = path_set(log)
    if factor is not None:
        return {"factor": float(factor), "source": "manual", "path_set": group}
    known = table.get(group, {})
    if direction not in known:
        return {"factor": 1.0, "source": "unverified", "path_set": group}
    return {"factor": float(known[direction]), "source": source, "path_set": group}


def travel_scale(session: Path, factor: float | None = None) -> dict:
    """本条录制的行程放大：方向、路径起点（法兰，m）和系数。factor=None 时按 TRAVEL_SCALE 取。"""
    path = session / "motion_log.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    segs = {s["name"]: s for s in log.get("segments", [])}
    if log.get("kind") != "record" or "goto_start" not in segs or "translate" not in segs:
        return {"factor": 1.0, "source": "none", "note": "不是平移录制（或没有 motion_log.json），不缩放"}
    start = np.array(segs["goto_start"]["flange"][:3], dtype=float) / 1000.0
    end = np.array(segs["translate"]["flange"][:3], dtype=float) / 1000.0
    direction = log.get("direction")
    return {**_lookup(TRAVEL_SCALE, log, direction, factor, "TRAVEL_SCALE"), "direction": direction,
            "origin_flange_m": start.tolist(), "axis_base": ((end - start) / np.linalg.norm(end - start)).tolist()}


def apply_travel_scale(pos: np.ndarray, scale: dict) -> np.ndarray:
    """法兰位置沿运动方向、相对路径起点放大 factor 倍；垂直方向不动。"""
    if scale["factor"] == 1.0:
        return pos
    origin, axis = np.array(scale["origin_flange_m"]), np.array(scale["axis_base"])
    return pos + (scale["factor"] - 1.0) * ((pos - origin) @ axis)[:, None] * axis


def rotation_scale(session: Path, factor: float | None = None) -> dict:
    """本条录制的转角放大：方向、起点法兰位姿、转轴和转动中心（基坐标系）、系数。factor=None 时按 ROTATION_SCALE 取。"""
    path = session / "motion_log.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    segs = {s["name"]: s for s in log.get("segments", [])}
    rot = segs.get("rotate", {})
    if log.get("kind") != "record" or "goto_start" not in segs or not rot.get("arc_axis") or not rot.get("pivot_m"):
        return {"factor": 1.0, "source": "none", "note": "不是旋转录制（或没有 motion_log.json），不放大转角"}
    direction = log.get("direction")
    axis = np.array(rot["arc_axis"], dtype=float)
    return {**_lookup(ROTATION_SCALE, log, direction, factor, "ROTATION_SCALE"), "direction": direction,
            "origin_flange": [float(v) for v in segs["goto_start"]["flange"]],
            "axis_base": (axis / np.linalg.norm(axis)).tolist(), "pivot_m": [float(v) for v in rot["pivot_m"]]}


def apply_rotation_scale(pos: np.ndarray, quat: np.ndarray, scale: dict) -> tuple[np.ndarray, np.ndarray]:
    """法兰姿态相对本条起点、绕转轴的转角放大 factor 倍，绕转动中心（左目光心）转，光心不动；其他方向的转动不变。"""
    if scale["factor"] == 1.0:
        return pos, quat
    o = scale["origin_flange"]
    q0 = matrix_to_quat(euler_to_matrix(o[3], o[4], o[5]))
    axis, pivot = np.array(scale["axis_base"]), np.array(scale["pivot_m"])
    phi = quat_to_rotvec(quat_mul(quat, quat_conj(q0))) @ axis      # 基坐标系下相对起点、绕转轴的转角（弧度）
    extra = rotvec_to_quat((scale["factor"] - 1.0) * phi[:, None] * axis)
    q = quat_mul(extra, quat)
    q = np.where(np.sum(q * quat, axis=1, keepdims=True) < 0, -q, q)  # 和原来同一半球，SLERP 仍走短路径
    p = pivot + np.einsum("nij,nj->ni", quat_to_matrix(extra), pos - pivot)
    return p, q


def write_tum(path: Path, t_ms, pos, quat) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (Unix s, m; left camera in FR5 base frame)\n")
        for t, p, q in zip(t_ms, pos, quat):
            f.write(f"{t / 1000:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}\n")


def process(session: Path, handeye_path: Path, output: Path, max_lag_ms: float = 300.0,
            dt_ms: float | None = None, exposure_offset_ms: float | None = None,
            time_correction_ms: float | None = None, scale_factor: float | None = None,
            rotation_factor: float | None = None) -> dict:
    warnings = []
    flange_t_cam, handeye = load_handeye(handeye_path)
    arm = load_arm(session / "fr5_state.csv")
    t_imu, gyro, trigger = load_imu(session / "zed_imu.csv", warnings)
    t_img, names = load_images(session / "img_timeline.csv", warnings)
    duplicate = np.ones(len(t_img), dtype=bool)
    duplicate[np.unique(t_img, return_index=True)[1]] = False   # 同一时间戳只保留第一张

    # 曝光中心相对图像时间戳的偏差
    exposure = exposure_centre(t_img[~duplicate], t_imu, trigger)
    if exposure_offset_ms is not None:
        exposure = {**(exposure or {}), "source": "manual", "applied_ms": exposure_offset_ms}
    elif exposure is None:
        warnings.append("没有足够的 image_sync_trigger 标记，无法估曝光中心，按 0 处理（可用 --exposure-offset-ms 指定）")
        exposure = {"source": "none", "applied_ms": 0.0}
    else:
        exposure.update(source="image_sync_trigger", applied_ms=exposure["offset_ms"])
        if exposure["sd_ms"] > MAX_EXPOSURE_SD_MS:
            warnings.append(f"各帧曝光中心偏差的标准差 {exposure['sd_ms']:.1f} ms，偏大")
        exp_us = read_time_meta(session / "time_meta.txt").get("exposure_us")
        if exp_us:
            exposure["exposure_ms"] = float(exp_us) / 1000
            if not exposure["exposure_ms"] - 8 <= exposure["median_flagged_span_ms"] <= exposure["exposure_ms"] + 6:
                warnings.append(f"带标记 IMU 样本的跨度 {exposure['median_flagged_span_ms']:.1f} ms，"
                                f"和曝光 {exposure['exposure_ms']:.1f} ms 对不上")

    meta_path = session / "fr5_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if abs(meta.get("system_clock_step_ms", 0.0)) > 1.0:
            warnings.append(f"录制期间 Box 系统时间被调整过 {meta['system_clock_step_ms']} ms")
        if meta.get("events"):
            warnings.append(f"录制期间有 {len(meta['events'])} 条连接告警，见 fr5_meta.json")
    if (np.unique(arm["tool"]) != 0).any() or (np.unique(arm["user"]) != 0).any():
        warnings.append("录制中工具号或工件号不是 0，法兰位姿可能不在基坐标系下")
    error_rows = (arm["main_code"] != 0) | (arm["sub_code"] != 0)
    if error_rows.any():
        warnings.append(f"{int(error_rows.sum())} 个状态包带机械臂错误码")

    # 1. 控制器时钟 -> Box 时间
    clock = fit_clock(arm["ctrl_ms"], arm["box_ms"])
    warnings += clock["warnings"]
    t_arm = map_ctrl(arm["ctrl_ms"], clock)
    pos = np.c_[arm["x_mm"], arm["y_mm"], arm["z_mm"]] / 1000.0
    quat = matrix_to_quat(euler_to_matrix(arm["rx_deg"], arm["ry_deg"], arm["rz_deg"]))
    # 相邻四元数保持同一半球，SLERP 走短路径
    flip = np.cumsum(np.r_[0, np.sum(quat[1:] * quat[:-1], axis=1) < 0]) % 2 == 1
    quat[flip] *= -1

    # 2. Δt
    t_w, w_arm = arm_angular_speed(t_arm, quat)
    static = static_intervals(t_w, w_arm)
    bias = gyro_bias(t_imu, gyro, static, margin_ms=max_lag_ms / 2)
    if bias is None:
        warnings.append("静止数据不够，没有扣陀螺零偏")
        bias = np.zeros(3)
    g = np.linalg.norm(gyro - bias, axis=1)
    lag = {"source": "manual", "dt_ms": dt_ms} if dt_ms is not None else None
    if lag is None:
        if w_arm.max(initial=0) < MIN_MOTION_DEG_S:
            raise ValueError(f"机械臂最大角速度只有 {w_arm.max(initial=0):.1f} °/s，无法估计 Δt；"
                             "录制里加同步摆动，或用 --dt-ms 手动给定")
        lag = estimate_lag(t_imu, g, t_w, w_arm, max_lag_ms)
        lag["source"] = "gyro cross-correlation"
        if lag["at_search_edge"]:
            warnings.append("Δt 落在搜索范围边缘，增大 --max-lag-ms 再试")
        if not lag["peak_corr"] >= MIN_PEAK_CORR:
            warnings.append(f"Δt 互相关峰值只有 {lag['peak_corr']:.3f}，同步可能不可靠")
        # 前后两段分别估计，检查漂移拟合
        lo, hi = max(t_imu[0], t_w[0]), min(t_imu[-1], t_w[-1])
        segments = {}
        for name, (a, b) in {"first_40pct": (lo, lo + 0.4 * (hi - lo)),
                             "last_40pct": (hi - 0.4 * (hi - lo), hi)}.items():
            sel_w = (t_w >= a) & (t_w <= b)
            sel_i = (t_imu >= a) & (t_imu <= b)
            if (hi - lo) >= MIN_SPLIT_S * 1000.0 and w_arm[sel_w].max(initial=0) >= MIN_MOTION_DEG_S:
                segments[name] = estimate_lag(t_imu[sel_i], g[sel_i], t_w, w_arm,
                                              min(max_lag_ms, 50.0) + abs(lag["dt_ms"]))
        lag["segments"] = segments
        if len(segments) == 2:
            diff = abs(segments["first_40pct"]["dt_ms"] - segments["last_40pct"]["dt_ms"])
            lag["segment_diff_ms"] = diff
            if diff > MAX_SEGMENT_LAG_DIFF_MS:
                warnings.append(f"前后两段的 Δt 相差 {diff:.1f} ms，时钟漂移拟合可能不准")
        else:
            warnings.append(f"开头和结尾没有都做同步摆动（或录制不足 {MIN_SPLIT_S:.0f} s），没能核对前后 Δt 是否一致")
    shift = lag["dt_ms"]

    # 3. 标定板验证得到的修正
    correction = {"applied_ms": TIME_CORRECTION_MS if time_correction_ms is None else time_correction_ms,
                  "source": "TIME_CORRECTION_MS" if time_correction_ms is None else "manual"}
    time_correction_ms = correction["applied_ms"]
    exp_us = read_time_meta(session / "time_meta.txt").get("exposure_us")
    if exp_us:
        correction["exposure_ms"] = float(exp_us) / 1000
        if time_correction_ms and float(exp_us) != TIME_CORRECTION_EXPOSURE_US:
            warnings.append(f"时间修正 {time_correction_ms} ms 是在曝光 {TIME_CORRECTION_EXPOSURE_US / 1000:.0f} ms 下测的，"
                            f"这条录制曝光 {float(exp_us) / 1000:.1f} ms，修正可能不准")
    scale = travel_scale(session, scale_factor)
    pos_c = apply_travel_scale(pos, scale)
    rscale = rotation_scale(session, rotation_factor)
    for s, what in ((scale, "行程"), (rscale, "转角")):
        if s["source"] == "unverified":
            warnings.append(f"路径组 {s['path_set']} 的 {s['direction']} 还没有用标定板验证过{what}系数，没有修正")
    pos_c, quat_c = apply_rotation_scale(pos_c, quat, rscale)

    # 4. 真值（t_arm_cam 对齐陀螺仪，t_arm_img 再对齐到图像）
    output.mkdir(parents=True, exist_ok=True)
    t_arm_cam = t_arm + shift
    t_arm_img = t_arm_cam - time_correction_ms
    cam_t, cam_q = camera_pose(pos_c, quat_c, flange_t_cam)
    write_tum(output / "groundtruth_arm_rate_tum.txt", t_arm_img, cam_t, cam_q)

    t_query = t_img + exposure["applied_ms"]
    p_f, q_f, inside, span, i0, i1 = interpolate(t_query, t_arm_img, pos_c, quat_c)
    img_t, img_q = camera_pose(p_f, q_f, flange_t_cam)
    gap = inside & (span > MAX_GAP_MS)
    bad_robot = inside & (error_rows[i0] | error_rows[i1])
    good = inside & ~gap & ~duplicate
    write_tum(output / "groundtruth_tum.txt", t_img[good], img_t[good], img_q[good])
    p_u, q_u, inside_u, span_u, *_ = interpolate(t_query, t_arm_cam, pos, quat)   # 不加修正，对照用
    raw_t, raw_q = camera_pose(p_u, q_u, flange_t_cam)
    good_u = inside_u & (span_u <= MAX_GAP_MS) & ~duplicate
    write_tum(output / "groundtruth_tum_uncorrected.txt", t_img[good_u], raw_t[good_u], raw_q[good_u])
    with open(output / "groundtruth_images.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zed_timestamp_ms", "filename_left", "flag", "arm_span_ms",
                    "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"])
        for k in range(len(t_img)):
            flag = ("duplicate" if duplicate[k] else "outside" if not inside[k] else "gap" if gap[k]
                    else "robot_error" if bad_robot[k] else "ok")
            vals = ([f"{v:.6f}" for v in img_t[k]] + [f"{v:.9f}" for v in img_q[k][[1, 2, 3, 0]]]
                    if good[k] else [""] * 7)
            w.writerow([f"{t_img[k]:.0f}", names[k], flag, f"{span[k]:.1f}" if inside[k] else ""] + vals)

    report = {
        "session": str(session),
        "handeye": handeye,
        "arm_packets": int(len(t_arm)),
        "arm_dropped_duplicates": arm["dropped_duplicates"],
        "clock_fit": clock,
        "gyro_bias_deg_s": bias.tolist(),
        "static_intervals": len(static),
        "arm_peak_angular_speed_deg_s": float(w_arm.max(initial=0)),
        "time_offset": lag,
        "exposure_centre": exposure,
        "time_correction": correction,
        "travel_scale": scale,
        "rotation_scale": rscale,
        "images": {"total": int(len(t_img)), "with_groundtruth": int(good.sum()),
                   "duplicate_timestamp": int(duplicate.sum()),
                   "outside_arm_data": int((~inside).sum()), "in_arm_gap": int(gap.sum()),
                   "robot_error": int(bad_robot.sum())},
        "warnings": warnings,
        "conventions": [
            "时间戳：ZED / Box 系统时间（Unix 秒）。机械臂样本时刻 = 映射后的控制器时间 + time_offset.dt_ms"
            " - time_correction.applied_ms。",
            "每张图的真值取在曝光中心 = 图像时间戳 + exposure_centre.applied_ms；输出时间戳仍是图像时间戳。",
            "法兰平移沿 travel_scale.axis_base、相对 travel_scale.origin_flange_m 放大 travel_scale.factor 倍"
            "（机械臂实际行程比报告的长）。",
            "法兰姿态相对 rotation_scale.origin_flange、绕 rotation_scale.axis_base 的转角放大 rotation_scale.factor 倍，"
            "绕 rotation_scale.pivot_m 转（机械臂实际转角比报告的大）。groundtruth_tum_uncorrected.txt 不加这三项修正。",
            "位姿：ZED 左目（校正后光学坐标系）在 FR5 基坐标系下，TUM 顺序 tx ty tz qx qy qz qw，单位 m。",
            "base_T_camera = base_T_flange · flange_T_left_camera；FR5 姿态 R = Rz(rz)·Ry(ry)·Rx(rx)。",
            "假设 ZED 图像与 IMU 在同一时钟上、image_sync_trigger 正确标出采集期间，且相机安装自手眼标定后没有变动。",
        ],
    }
    (output / "groundtruth_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="由 fr5_record + zed_record 的录制生成相机位姿真值。")
    parser.add_argument("session", type=Path, help="含 fr5_state.csv、zed_imu.csv、img_timeline.csv 的目录")
    parser.add_argument("--handeye", type=Path, required=True,
                        help="标定仓库的 result.json，或 4x4 flange_T_left_camera 矩阵的 JSON")
    parser.add_argument("--output", type=Path, help="输出目录，默认就是 session 目录")
    parser.add_argument("--max-lag-ms", type=float, default=300.0, help="Δt 搜索范围（±ms）")
    parser.add_argument("--dt-ms", type=float, help="手动指定 Δt，跳过陀螺仪互相关")
    parser.add_argument("--exposure-offset-ms", type=float,
                        help="手动指定“曝光中心 - 图像时间戳”（ms），跳过 image_sync_trigger 估计")
    parser.add_argument("--time-correction-ms", type=float,
                        help=f"真值比图像晚的固定量（ms），默认 TIME_CORRECTION_MS={TIME_CORRECTION_MS}；0 表示不修正")
    parser.add_argument("--scale-factor", type=float,
                        help="行程放大系数，默认按录制方向取 TRAVEL_SCALE；1 表示不缩放")
    parser.add_argument("--rotation-scale-factor", type=float,
                        help="转角放大系数，默认按录制方向取 ROTATION_SCALE；1 表示不放大")
    args = parser.parse_args(argv)
    session = args.session.expanduser()
    try:
        report = process(session, args.handeye.expanduser(), (args.output or session).expanduser(),
                         args.max_lag_ms, args.dt_ms, args.exposure_offset_ms,
                         args.time_correction_ms, args.scale_factor, args.rotation_scale_factor)
    except (OSError, ValueError, KeyError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    clock, lag, imgs = report["clock_fit"], report["time_offset"], report["images"]
    print(f"时钟：控制器比 Box 快 {-clock['offset_ms'] / 1000:.3f} s，漂移 {clock['drift_ppm']:+.1f} ppm，"
          f"窗口残差最大 {clock['window_residual_ms_max']:.2f} ms")
    corr = f"，互相关峰值 {lag['peak_corr']:.3f}" if "peak_corr" in lag else ""
    seg = f"，前后段差 {lag['segment_diff_ms']:.2f} ms" if "segment_diff_ms" in lag else ""
    print(f"Δt = {lag['dt_ms']:+.2f} ms（{lag['source']}{corr}{seg}）")
    exp = report["exposure_centre"]
    print(f"曝光中心 = 图像时间戳 {exp['applied_ms']:+.2f} ms（{exp['source']}"
          + (f"，{exp['frames']} 帧，标准差 {exp['sd_ms']:.2f} ms" if "frames" in exp else "") + "）")
    tc, sc, rc = report["time_correction"], report["travel_scale"], report["rotation_scale"]
    print(f"修正：取真值的时刻再晚 {tc['applied_ms']:+.2f} ms（{tc['source']}）；"
          + (f"行程 ×{sc['factor']:.4f}（{sc['direction']}，{sc['source']}）" if sc["factor"] != 1.0 else "行程不缩放")
          + (f"；转角 ×{rc['factor']:.4f}（{rc['direction']}，{rc['source']}）" if rc["factor"] != 1.0 else "；转角不放大"))
    dup = f"，重复时间戳 {imgs['duplicate_timestamp']} 张已去掉" if imgs["duplicate_timestamp"] else ""
    print(f"图像 {imgs['total']} 张，有真值 {imgs['with_groundtruth']} 张{dup}")
    for w in report["warnings"]:
        print(f"警告：{w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
