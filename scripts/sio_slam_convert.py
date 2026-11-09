#!/usr/bin/env python3
"""把录制（save_data）转换成 SIO_SLAM 的 zed_record 输入格式（frames.csv + imu.csv）。只用标准库，笔记本和跑 SIO_SLAM 的机器上都能运行。

SIO_SLAM（lihongliang0203/SIO_SLAM）的 DatasetLoader 读文件有两种格式：
- 旧格式：相邻两帧之间至少要 25 个 IMU 样本。本数据 IMU 约 200 Hz、相机 30 fps，每帧只有 4–8 个，所有帧都会被丢掉；
- zed_record 格式：dataset_dir 下有 frames.csv，IMU 按帧号分组，每帧样本数下限是 min(25, 这一帧应有的个数)。本脚本输出这种。

时间：图像内容对应的 IMU 时刻比图像时间戳晚约 4.7 ms（使用说明 5.2），这里把图像时间加上这个偏移，统一到 IMU 时间轴，
SIO_SLAM 里的帧时间（end_time）就是加了偏移的值。输出的真值也加同样的偏移（位姿不变），SIO_SLAM 写出的轨迹
可以直接和它按时间戳配对，例如 evo_ape tum groundtruth_tum.txt trajectory_tum.txt --align_origin。

IMU 分配：起始帧拿到它之前的全部 IMU（IMU 比第一张图早开始约 1.5 s，再加上起始帧之前的静止段），SIO_SLAM 用这一帧的
IMU 做静止初始化（至少 200 个样本），这一帧的图像不进入跟踪；之后每帧拿 (上一帧时间, 本帧时间] 内的样本。
最后一帧之后的 IMU 不要。起始帧这一段里如果有超过 10 ms 的缺口（SIO_SLAM 初始化时会因此清空缓存、再等 200 个样本），
只留最后一个缺口之后的样本，例如 20261016_143751_pitch_up_people 开头 0.2 s 处有一个 41.8 ms 的缺口。默认从第 60 帧开始：前 2 s 亮度和白平衡还在变（使用说明 7.1），之后到同步摆动前仍是静止的。

输出（默认 <save_data>/sio_slam/，原有文件不动；再运行会覆盖本目录里转换生成的文件）：
  frames.csv             record_frame_id, st_time, end_time, left_path, right_path, expected_imu_count
                         （帧号是原录制的图像序号；时间为秒；图像路径相对本目录，即 ../left/<ms>.jpg）
  imu.csv                frame_id, timestamp_s, imu_timestamp_ns, ax, ay, az, gx, gy, gz
                         （加速度 m/s²，陀螺仪 °/s，SIO_SLAM 读入时转成 rad/s；第 3 列 SIO_SLAM 不读，留作对照）
  groundtruth_tum.txt    转换范围内每帧的左目真值（FR5 基坐标系），时间戳加了同样的偏移
  run_config.yaml        给 SIO_SLAM 的环境变量 SIO_RUN_CONFIG：数据路径和轨迹输出路径，相对本目录
  convert_meta.json      源目录、起始帧、时间偏移、帧数、每帧 IMU 个数

    python -B sio_slam_convert.py <录制目录或 save_data> [...] [--start-frame 60] [--time-offset-ms 4.7|auto]
    python -B sio_slam_convert.py ..\\data\\recordings_highspeed         # 目录下每条录制都转换

--time-offset-ms auto：用这条录制 frame_lag_report.json 里测得的值（lag_ms 取负号，4.2–4.9 ms）。默认用 6 条的平均 4.7 ms：
相机设置相同，偏移应当相同，各条的差别主要是测量误差。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 标定仓库约定不留 __pycache__

DEFAULT_START_FRAME = 60        # 前 2 s 亮度和白平衡在变（使用说明 7.1）
DEFAULT_TIME_OFFSET_MS = 4.7    # 图像时间 + 偏移 = IMU 时间轴上的时刻（使用说明 5.2）
OUT_NAME = "sio_slam"
MIN_INIT_IMU = 200              # SIO_SLAM 静止初始化要的 IMU 个数（frontend.cpp）
MAX_INIT_GYRO_DEG_S = 3.0       # 初始化段陀螺仪偏离均值超过它，说明起始帧之前不是静止的
MAX_IMU_GAP_MS = 10.0           # SIO_SLAM 初始化段允许的最大 IMU 间隔（isImuDataContinuous）

RUN_CONFIG = """\
# SIO_SLAM 单次运行配置（data_recording/scripts/sio_slam_convert.py 生成）。用法：
#   SIO_RUN_CONFIG=<本文件路径> ros2 run sio_slam run_frontend
# 路径相对本文件所在目录。TensorRT plan、params_file 仍取 SIO_SLAM 的 project_config(.local).yaml，
# params_file 用 config/params_fr5_zedx_highspeed.yaml。
dataset_dir: .
imu_file: imu.csv
trajectory_file: trajectory_tum.txt              # 每帧后验（ESKF / ESIKF）的左目位姿
trajectory_prior_file: trajectory_prior_tum.txt  # 每帧 IMU 先验的左目位姿
"""


def ns_to_s(ns: int) -> str:
    """整数纳秒 → 秒的字符串，不经过浮点。"""
    return f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d}"


def us_to_s(us: int) -> str:
    return f"{us // 1_000_000}.{us % 1_000_000:06d}"


def find_sessions(path: Path) -> list[Path]:
    """录制目录、save_data 或装着多条录制的目录 → save_data 列表。"""
    if (path / "img_timeline.csv").exists():
        return [path]
    if (path / "save_data" / "img_timeline.csv").exists():
        return [path / "save_data"]
    found = sorted(p / "save_data" for p in path.iterdir() if (p / "save_data" / "img_timeline.csv").exists())
    if not found:
        raise FileNotFoundError(f"{path} 里没有录制（找不到 img_timeline.csv）")
    return found


def read_frames(session: Path) -> list[dict]:
    with open(session / "img_timeline.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{session}: img_timeline.csv 是空的")
    stamps = [int(r["zed_timestamp_ms"]) for r in rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{session}: 图像时间戳不是严格递增的")
    return rows


def read_imu(session: Path) -> list[list[str]]:
    """zed_imu.csv 的每行：[ns, ax, ay, az, gx, gy, gz]（数值保持原文本）。"""
    out = []
    with open(session / "zed_imu.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append([r["zed_imu_timestamp_ns"], r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"]])
    stamps = [int(r[0]) for r in out]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{session}: IMU 时间戳不是严格递增的")
    return out


def read_groundtruth(session: Path, frames: list[dict]) -> list[list[str]]:
    """groundtruth_tum.txt 每行 [t, tx, ty, tz, qx, qy, qz, qw]，和图像一一对应。"""
    rows = []
    with open(session / "groundtruth_tum.txt", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                rows.append(line.split())
    if len(rows) != len(frames):
        raise ValueError(f"{session}: 真值 {len(rows)} 行，图像 {len(frames)} 张")
    for r, fr in zip(rows, frames):
        if round(float(r[0]) * 1000) != int(fr["zed_timestamp_ms"]):
            raise ValueError(f"{session}: 真值和图像时间戳对不上（{r[0]} 对 {fr['zed_timestamp_ms']}）")
    return rows


def resolve_offset_ms(session: Path, arg: str) -> tuple[float, str]:
    if arg != "auto":
        return float(arg), "fixed"
    report = session / "frame_lag_report.json"
    if not report.exists():
        raise FileNotFoundError(f"{report} 不存在，先运行 frame_lag_check.py 或用固定的 --time-offset-ms")
    data = json.loads(report.read_text(encoding="utf-8"))
    if not data.get("ok", False):
        raise ValueError(f"{report}: 检查结果不正常（{data.get('verdict')}），这条录制不能用")
    return -float(data["lag_ms"]), "frame_lag_report"


def convert(session: Path, start_frame: int = DEFAULT_START_FRAME, offset_arg: str = str(DEFAULT_TIME_OFFSET_MS),
            out_name: str = OUT_NAME) -> dict:
    frames = read_frames(session)
    imu = read_imu(session)
    gt = read_groundtruth(session, frames)
    if not 0 <= start_frame < len(frames) - 1:
        raise ValueError(f"{session}: 起始帧 {start_frame} 超出范围（共 {len(frames)} 帧）")
    offset_ms, offset_source = resolve_offset_ms(session, offset_arg)
    offset_us = round(offset_ms * 1000)

    # 图像时间换到 IMU 时间轴（微秒），按 (上一帧, 本帧] 分 IMU；起始帧拿之前的全部
    frame_us = [int(fr["zed_timestamp_ms"]) * 1000 + offset_us for fr in frames]
    imu_ns = [int(r[0]) for r in imu]
    groups: dict[int, list[int]] = {}
    j = 0
    for k in range(start_frame, len(frames)):
        end_ns = frame_us[k] * 1000
        idx = []
        while j < len(imu_ns) and imu_ns[j] <= end_ns:
            idx.append(j)
            j += 1
        groups[k] = idx

    # 初始化段有超过 SIO_SLAM 允许间隔的缺口时（它会清空缓存再等 200 个样本），只留最后一个缺口之后的样本；
    # 这些都是起始帧之前的静止段，只用来初始化
    init = groups[start_frame]
    notes, warnings = [], []
    cut = max((a for a in range(1, len(init))
               if (imu_ns[init[a]] - imu_ns[init[a - 1]]) / 1e6 > MAX_IMU_GAP_MS), default=0)
    if cut:
        gap_ms = (imu_ns[init[cut]] - imu_ns[init[cut - 1]]) / 1e6
        notes.append(f"初始化段 IMU 在 {ns_to_s(imu_ns[init[cut - 1]])} s 之后有 {gap_ms:.1f} ms 的缺口，"
                     f"去掉了缺口之前的 {cut} 个样本")
        init = groups[start_frame] = init[cut:]
    if len(init) < MIN_INIT_IMU:
        warnings.append(f"起始帧之前只有 {len(init)} 个 IMU 样本，SIO_SLAM 初始化要 {MIN_INIT_IMU} 个，会往后多吃几帧")
    if init:
        gyro = [[float(v) for v in imu[i][4:7]] for i in init]
        mean = [sum(c) / len(gyro) for c in zip(*gyro)]
        dev = max(max(abs(g - m) for g, m in zip(row, mean)) for row in gyro)
        if dev > MAX_INIT_GYRO_DEG_S:
            warnings.append(f"起始帧之前陀螺仪偏离均值最大 {dev:.1f} °/s，不是静止的，SIO_SLAM 的静止初始化会出错")
    empty = [k for k in groups if k > start_frame and not groups[k]]
    if empty:
        warnings.append(f"{len(empty)} 帧之间没有 IMU 样本（第一个是第 {empty[0]} 帧）")

    out = session / out_name
    out.mkdir(exist_ok=True)
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["record_frame_id", "st_time", "end_time", "left_path", "right_path", "expected_imu_count"])
        for k, idx in groups.items():
            left, right = frames[k]["filename_left"], frames[k]["filename_right"]
            for name in (left, right):
                if not (session / name).exists():
                    raise FileNotFoundError(session / name)
            if k == start_frame:
                st = ns_to_s(imu_ns[idx[0]]) if idx else us_to_s(frame_us[k])
            else:
                st = us_to_s(frame_us[k - 1])
            w.writerow([k, st, us_to_s(frame_us[k]), f"../{left}", f"../{right}", len(idx)])
    with open(out / "imu.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame_id", "timestamp_s", "imu_timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"])
        for k, idx in groups.items():
            for i in idx:
                w.writerow([k, ns_to_s(imu_ns[i]), imu[i][0], *imu[i][1:]])
    with open(out / "groundtruth_tum.txt", "w", encoding="utf-8") as f:
        f.write(f"# timestamp tx ty tz qx qy qz qw  (s, m; left camera in FR5 base frame; "
                f"timestamp = image timestamp + {offset_ms:.3f} ms, on the IMU time axis)\n")
        for k in groups:
            f.write(" ".join([us_to_s(frame_us[k]), *gt[k][1:]]) + "\n")
    (out / "run_config.yaml").write_text(RUN_CONFIG, encoding="utf-8")

    counts = [len(groups[k]) for k in groups if k > start_frame]
    counts_sorted = sorted(counts)
    tracked = [i for k in groups if k > start_frame for i in groups[k]]
    tracked = [groups[start_frame][-1]] + tracked if groups[start_frame] else tracked
    max_gap_ms = max(((imu_ns[b] - imu_ns[a]) / 1e6 for a, b in zip(tracked, tracked[1:])), default=0.0)
    meta = {
        "source": str(session.resolve()),
        "start_frame": start_frame,
        "frames": len(groups),
        "time_offset_ms": offset_ms,
        "time_offset_source": offset_source,
        "imu_total": sum(len(v) for v in groups.values()),
        "imu_init_frame": len(init),
        "imu_per_frame": {"min": counts_sorted[0], "median": counts_sorted[len(counts) // 2], "max": counts_sorted[-1]},
        "imu_max_gap_ms_after_start": round(max_gap_ms, 3),
        "imu_dropped_before_init_gap": cut,
        "imu_dropped_after_last_frame": len(imu_ns) - j,
        "first_end_time": us_to_s(frame_us[start_frame]),
        "last_end_time": us_to_s(frame_us[-1]),
        "notes": notes,
        "warnings": warnings,
    }
    (out / "convert_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="录制 → SIO_SLAM 的 frames.csv + imu.csv")
    p.add_argument("paths", nargs="+", type=Path, help="录制目录、save_data 或装着多条录制的目录")
    p.add_argument("--start-frame", type=int, default=DEFAULT_START_FRAME, help="从第几帧开始（原录制的图像序号）")
    p.add_argument("--time-offset-ms", default=str(DEFAULT_TIME_OFFSET_MS),
                   help="图像时间加多少 ms 到 IMU 时间轴；auto 用 frame_lag_report.json")
    p.add_argument("--out-name", default=OUT_NAME, help="输出子目录名（在 save_data 下）")
    a = p.parse_args()
    if a.time_offset_ms != "auto":
        float(a.time_offset_ms)  # 提前报错

    sessions = [s for path in a.paths for s in find_sessions(path)]
    bad = 0
    for s in sessions:
        meta = convert(s, a.start_frame, a.time_offset_ms, a.out_name)
        name = s.parent.name if s.name == "save_data" else s.name
        print(f"{name}: {meta['frames']} 帧（从第 {meta['start_frame']} 帧），偏移 {meta['time_offset_ms']:+.2f} ms，"
              f"初始化 IMU {meta['imu_init_frame']} 个，每帧 {meta['imu_per_frame']['min']}–{meta['imu_per_frame']['max']} 个"
              f" → {s / a.out_name}")
        for n in meta["notes"]:
            print(f"  说明：{n}")
        for w in meta["warnings"]:
            print(f"  警告：{w}")
        bad += bool(meta["warnings"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
