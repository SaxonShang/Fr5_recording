"""zed_record（相机图像 + IMU）的子进程管理。由 run_session.py 调用，参数由调用方传入。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

GRABBING_MARK = "ZED 捕获"   # zed_record 开始采集时打印的文字


def camera_users() -> list[str]:
    """正在占用相机的程序（同一时间只能有一个程序打开 ZED）。"""
    out = subprocess.run(["pgrep", "-a", "-f", "ZED_Depth_Viewer|ZED_Explorer|ZED_Sensor_Viewer|ZEDfu|"
                          "zed_record|zed_snapshot"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


class ZedRecord:
    """以子进程运行 zed_record。用硬链接放进录制目录，它的 save_data/ 就写在这条录制自己的目录里。"""

    def __init__(self, binary: Path, session_dir: Path, exposure_us: int, analog_gain_mdb: int, fps: int = 15):
        self.binary, self.session_dir = binary, session_dir
        self.args = ["--exposure-us", str(exposure_us), "--analog-gain-mdb", str(analog_gain_mdb)]
        if fps != 15:                     # 默认 15 fps 时不传，旧版 zed_record（没有 --fps）也能用
            self.args += ["--fps", str(fps)]
        self.link = session_dir / "zed_record"
        self.lines, self.proc, self.reader = [], None, None
        self.grabbing = threading.Event()
        self.info = {}

    def start(self) -> None:
        try:
            os.link(self.binary, self.link)
        except OSError:
            shutil.copy2(self.binary, self.link)
        self.info["start_box_ns"] = time.time_ns()
        # 子进程继承“忽略 SIGHUP”：SSH 掉线时 zed_record 不会被直接杀掉，由调用方收尾时用 SIGINT 正常停止
        old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            self.proc = subprocess.Popen([str(self.link), *self.args], cwd=self.session_dir, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
        finally:
            signal.signal(signal.SIGHUP, old)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(f"{time.time_ns()} {line.rstrip()}")
            if GRABBING_MARK in line and not self.grabbing.is_set():
                self.info["grabbing_box_ns"] = time.time_ns()
                self.grabbing.set()

    def wait_grabbing(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        while not self.grabbing.wait(0.2):
            if self.proc.poll() is not None or time.monotonic() > end:
                raise RuntimeError("zed_record 没有开始采集：\n" + "\n".join(self.lines[-10:]))

    def stop(self) -> None:
        if self.proc is not None:
            self.info["exited_before_stop"] = self.proc.poll() is not None   # 没等到停止就自己退出了
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.reader is not None:
            self.reader.join(timeout=5)
        self.info["stop_box_ns"] = time.time_ns()
        self.info["returncode"] = None if self.proc is None else self.proc.returncode
        (self.session_dir / "zed_record.log").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        if self.link.exists():
            self.link.unlink()
