#!/usr/bin/env python3
"""读出 ZED 左目相机和机身 IMU 之间的外参，以及 IMU 的噪声参数（Box 上运行，只读一次，不录任何数据）。

    python3 zed_imu_extrinsic.py --output camera_imu.yaml

都是 ZED SDK 里的出厂标定：sensors_configuration.camera_imu_transform、accelerometer_parameters、
gyroscope_parameters。坐标系用 IMAGE（x 右、y 下、z 前），和 zed_record 一样（它没有设置，用的是 SDK 默认），
所以 zed_imu.csv 里 IMU 数据的坐标轴和这里一致；平移单位米。先关掉其他占用相机的程序。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyzed.sl as sl


def sensor(p) -> dict:
    return {"available": bool(p.is_available), "unit": str(p.sensor_unit), "sampling_rate_hz": float(p.sampling_rate),
            "range": [float(v) for v in p.sensor_range], "resolution": float(p.resolution),
            "noise_density": float(p.noise_density), "random_walk": float(p.random_walk)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_image_flip = sl.FLIP_MODE.OFF
    init.camera_disable_self_calib = True
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.coordinate_units = sl.UNIT.METER
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"打开相机失败：{err}")
        return 1
    try:
        info = cam.get_camera_information()
        sc = info.sensors_configuration
        t = sc.camera_imu_transform
        r = np.array(t.get_rotation_matrix().r, dtype=float)
        p = np.array(t.get_translation().get(), dtype=float)
        acc, gyro = sensor(sc.accelerometer_parameters), sensor(sc.gyroscope_parameters)
        serial, sdk = int(info.serial_number), str(sl.Camera.get_sdk_version())
        fw = {"camera": int(info.camera_configuration.firmware_version), "sensors": int(sc.firmware_version)}
    finally:
        cam.close()
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, p
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    lines = [
        f"# ZED X（S/N {serial}）左目相机和机身 IMU 之间的外参、IMU 噪声参数：ZED SDK {sdk} 读出的出厂标定",
        f"# （{datetime.now(timezone.utc).isoformat(timespec='seconds')}，data_recording/scripts/zed_imu_extrinsic.py）。",
        "# 坐标系：都是 ZED SDK 的 IMAGE 坐标系（x 右、y 下、z 前），和 zed_imu.csv 里 IMU 数据的坐标轴相同；平移单位 m。",
        "# camera_imu_transform：SDK 的 sensors_configuration.camera_imu_transform（SDK 文档：IMU to left camera），",
        "# 4×4 齐次矩阵，行优先。",
        f"camera_serial: {serial}",
        f"zed_sdk: '{sdk}'",
        f"firmware: {{camera: {fw['camera']}, sensors: {fw['sensors']}}}",
        "coordinate_system: IMAGE",
        "camera_imu_transform:",
        *[f"  - [{', '.join(f'{v:.9f}' for v in row)}]" for row in m],
        f"camera_imu_rotation_deg: {angle:.4f}",
        "# IMU 噪声参数（SDK 的 SensorParameters；noise_density 单位为 unit/√Hz，random_walk 为 unit/s/√Hz，unit 见 unit）",
    ]
    for name, s in (("accelerometer", acc), ("gyroscope", gyro)):
        lines.append(f"{name}:")
        lines += [f"  {k}: {v!r}" if not isinstance(v, str) else f"  {k}: '{v}'" for k, v in s.items()]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
