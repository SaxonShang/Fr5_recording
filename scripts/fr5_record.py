#!/usr/bin/env python3
"""FR5 状态录制：独占控制器 20004 状态端口，给每个状态包打 Box 时间戳，停止时一次写盘。

控制器的 20004 端口同时只服务一个连接（新连接会挤掉旧连接，SDK 还会自动重连回来抢），
所以录制和 SDK 运动脚本必须共用同一个连接：

    # 1) 单独录制，机械臂用 WebApp 或示教程序移动
    python3 fr5_record.py --output ~/ROBOT/zed_imu/build/save_data

    # 2) SDK 脚本边动边录
    from fr5_record import StateRecorder
    with StateRecorder(output_dir) as rec:
        robot = rec.open_rpc()     # fairino RPC 子类：状态来自录制器，不另开 20004
        robot.MoveJ(...)

录制器只接收 20004 推送的数据，自己不向控制器发运动指令。数据先存在内存里，
停止时写出 fr5_state.csv 和 fr5_meta.json（先写临时文件再改名），中间不写盘。
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import xmlrpc.client
from pathlib import Path

from fairino import Robot

DEFAULT_IP = "192.168.58.2"
STATE_PORT = 20004
RCVBUF_BYTES = 1 << 20  # 1 MB（Box 的 net.core.rmem_max），内核实际给 2 MB，约 8 s 的状态包
XMLRPC_PORT = 20003
FRAME_SIZE = ctypes.sizeof(Robot.RobotStatePkg)  # SDK V2.2.1 / 控制器 V3.9.1：1236 字节，含帧头和校验和
HEADER = b"\x5a\x5a"
MAX_CTRL_STEP_MS = 13  # 控制器 6 ms 一拍，相邻包差 1–2 拍（实测 5/6/11/12 ms）；更大视为丢包
DAY_MS = 86_400_000

# CSV 列。位置 mm，角度 °，关节速度 °/s。法兰位姿 = 法兰（工具 0）在基坐标系（工件 0）下，
# 姿态 R = Rz(rz)·Ry(ry)·Rx(rx)。
_STATUS = [("frame_cnt", "B"), ("robot_state", "B"), ("robot_mode", "B"), ("program_state", "B"),
           ("main_code", "i"), ("sub_code", "i"), ("emergency_stop", "B"), ("collision", "B"),
           ("enable_state", "i"), ("motion_done", "i"), ("tool", "i"), ("user", "i")]
_JOINTS = [f"j{i}_deg" for i in range(1, 7)]
_FLANGE = ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg"]
_SPEEDS = [f"qd{i}_deg_s" for i in range(1, 7)]
CSV_COLUMNS = (["box_realtime_ns", "box_monotonic_ns", "ctrl_time", "ctrl_unix_ms"]
               + [name for name, _ in _STATUS] + _JOINTS + _FLANGE + _SPEEDS)

# 内存里每包一条定长记录：Box 时间 2 个 + 控制器时间 7 个 + 状态 + 18 个 double
_ROW = struct.Struct("<qq HBBBBBH " + "".join(fmt for _, fmt in _STATUS) + "d" * 18)


def _pack(pkg, t_rt: int, t_mo: int) -> bytes:
    return _ROW.pack(t_rt, t_mo, pkg.year, pkg.mouth, pkg.day, pkg.hour, pkg.minute, pkg.second,
                     pkg.millisecond, pkg.frame_cnt, pkg.robot_state, pkg.robot_mode, pkg.program_state,
                     pkg.main_code, pkg.sub_code, pkg.EmergencyStop, pkg.collisionState,
                     pkg.rbtEnableState, pkg.motion_done, pkg.tool, pkg.user,
                     *pkg.jt_cur_pos, *pkg.flange_cur_pos, *pkg.actual_qd)


def _csv_line(vals: tuple) -> str:
    t_rt, t_mo, y, mo, d, h, mi, s, ms = vals[:9]
    status = vals[9:9 + len(_STATUS)]
    floats = vals[9 + len(_STATUS):]
    ctrl = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}.{ms:03d}"
    try:  # 控制器本地时间，按 Box 的时区解释；两边时钟没同步，离线再拟合
        unix_ms = str(round(dt.datetime(y, mo, d, h, mi, s, ms * 1000).timestamp() * 1000))
    except ValueError:
        unix_ms = ""
    return ",".join([str(t_rt), str(t_mo), ctrl, unix_ms, *map(str, status),
                     *(f"{v:.10g}" for v in floats)])


def _clock_pair() -> dict:
    return {"realtime_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def _ntp_status() -> dict:
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTP", "-p", "NTPSynchronized"],
                             capture_output=True, text=True, timeout=3).stdout
        return dict(item.split("=", 1) for item in out.split())
    except Exception as exc:  # noqa: BLE001 - 只作记录
        return {"error": str(exc)}


class _TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = 2.0
        return conn


def _controller_info(ip: str) -> dict:
    """只读查询控制器版本和 20004 推送周期，失败不影响录制。"""
    proxy = xmlrpc.client.ServerProxy(f"http://{ip}:{XMLRPC_PORT}", transport=_TimeoutTransport())
    info = {}
    for key, name in (("software_version", "GetSoftwareVersion"),
                      ("state_period_ms", "GetRobotRealtimeStateSamplePeriod")):
        try:
            info[key] = getattr(proxy, name)()
        except Exception as exc:  # noqa: BLE001 - 只作记录
            info[key] = f"unavailable: {exc}"
    return info


class _FrameParser:
    """把 TCP 字节流切成完整状态帧：0x5A5A 帧头，长度字段，字节和校验。"""

    def __init__(self):
        self.buf = bytearray()
        self.bad_checksum = 0
        self.bad_length = 0

    def reset(self):
        self.buf.clear()

    def feed(self, data: bytes) -> list[bytes]:
        self.buf += data
        frames = []
        while True:
            i = self.buf.find(HEADER)
            if i < 0:
                del self.buf[:-1]  # 留最后一个字节，帧头可能被拆在两次 recv 之间
                break
            del self.buf[:i]
            if len(self.buf) < 5:
                break
            length = self.buf[3] | (self.buf[4] << 8)
            if length + 7 != FRAME_SIZE:  # 不是帧头，或控制器协议和 SDK 结构体不一致
                self.bad_length += 1
                del self.buf[:2]
                continue
            if len(self.buf) < FRAME_SIZE:
                break
            frame = bytes(self.buf[:FRAME_SIZE])
            del self.buf[:FRAME_SIZE]
            if (sum(frame[:-2]) & 0xFFFF) != (frame[-2] | (frame[-1] << 8)):
                self.bad_checksum += 1
                continue
            frames.append(frame)
        return frames


class StateRecorder:
    """独占 20004 连接，逐包记录，并给 RecordingRPC 提供最新状态。"""

    def __init__(self, output_dir, ip: str = DEFAULT_IP, force: bool = False):
        self.output_dir = Path(output_dir).expanduser()
        self.csv_path = self.output_dir / "fr5_state.csv"
        self.meta_path = self.output_dir / "fr5_meta.json"
        self.ip = ip
        self.force = force
        self.count = 0
        self.error = None
        self._rows = bytearray()
        self._latest = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._sock = None
        self._thread = None
        self._parser = _FrameParser()
        self._last_ctrl_ms = None
        self._gaps = []  # (box_realtime_ns, 控制器时间步长 ms)
        self._duplicates = 0
        self._events = []  # (box_realtime_ns, 说明)
        self._tools, self._users, self._codes = set(), set(), set()
        self._estop_seen = False
        self._rpcs = []
        self._meta = {}
        self._t0 = None
        self.rcvbuf_bytes = None
        self.latest_monotonic = None  # 最新一包到达的 time.monotonic()（秒），判断状态是否新鲜用

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self._event(f"录制期间出现异常：{exc_type.__name__}: {exc}")
        self.stop()
        return False

    # ---------------------------------------------------------------- 控制
    def start(self) -> None:
        for path in (self.csv_path, self.meta_path):
            if path.exists() and not self.force:
                raise FileExistsError(f"{path} 已存在：上一次的数据还没挪走？确认要覆盖再加 --force")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {"start": _clock_pair(), "ntp_start": _ntp_status(),
                      "controller": _controller_info(self.ip)}
        self._connect()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="fr5-state", daemon=True)
        self._thread.start()
        with self._cond:
            self._cond.wait_for(lambda: self.count > 0 or self.error is not None, timeout=3.0)
        if self.count == 0:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._close_socket()
            hint = "（帧长度与 SDK 结构体不一致，SDK 版本和控制器不匹配？）" if self._parser.bad_length else ""
            raise RuntimeError(f"3 s 内没收到有效状态包{hint}：{self.error or ''}")

    def open_rpc(self) -> "RecordingRPC":
        """打开共用本连接的 SDK RPC，用来发运动等指令。"""
        if self._thread is None or self._stop.is_set():
            raise RuntimeError("先 start() 录制器，再打开 RPC")
        rpc = RecordingRPC(self, self.ip)
        self._rpcs.append(rpc)
        return rpc

    def stop(self, write: bool = True) -> None:
        """停止录制；write=False 时只断开、不写文件（只读查看用）。"""
        if self._thread is None:
            return
        for rpc in self._rpcs:
            if not rpc.closeRPC_state:
                rpc.CloseRPC()
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._close_socket()
        with self._cond:
            self._cond.notify_all()
        if not write:
            return
        self._meta["stop"] = _clock_pair()
        self._meta["ntp_stop"] = _ntp_status()
        self._write()

    def wait_state(self, last_count: int, timeout: float):
        """等到有比 last_count 新的状态包；返回 (count, RobotStatePkg 或 None)。"""
        with self._cond:
            self._cond.wait_for(lambda: self.count != last_count or self._stop.is_set(), timeout)
            return self.count, self._latest

    # ---------------------------------------------------------------- 状态
    def status_line(self) -> str:
        elapsed = time.monotonic() - self._t0
        state = self._latest.robot_state if self._latest is not None else "-"
        return (f"{elapsed:6.0f} s：{self.count} 包（{self.count / max(elapsed, 1e-9):.1f} Hz），"
                f"丢包 {len(self._gaps)}，校验失败 {self._parser.bad_checksum}，"
                f"告警 {len(self._events)}，robot_state={state}")

    def summary(self) -> str:
        return (f"{self.count} 包，丢包 {len(self._gaps)}，重复 {self._duplicates}，"
                f"校验失败 {self._parser.bad_checksum}，告警 {len(self._events)}")

    # ---------------------------------------------------------------- 接收线程
    def _connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # 接收缓冲区开大（连接前设置）：本进程忙于计算、接收线程一时读不过来时，由内核先收着。
            # 默认 128 KB 只够约 0.5 s 的状态包，缓冲满了控制器就会断开 20004（2026-10-10 实测复现）。
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
            sock.settimeout(2.0)
            sock.connect((self.ip, STATE_PORT))
        except OSError:
            sock.close()
            raise
        sock.settimeout(1.0)  # 定期醒来检查停止标志
        self.rcvbuf_bytes = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self._sock = sock

    def _close_socket(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()

    def _reconnect(self) -> None:
        self._close_socket()
        self._parser.reset()
        t0, attempts = time.monotonic(), 0
        while not self._stop.wait(0.5):
            attempts += 1
            try:
                self._connect()
            except OSError:
                continue
            self._event(f"已重新连上 20004（断开 {time.monotonic() - t0:.1f} s，第 {attempts} 次尝试）")
            return

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    data = self._sock.recv(65536)
                    t_rt, t_mo = time.time_ns(), time.monotonic_ns()  # 紧跟 recv 打戳
                except socket.timeout:
                    self._event("超过 1 s 没收到状态包")
                    continue
                except OSError as exc:
                    if self._stop.is_set():
                        break
                    self._event(f"20004 接收出错（{exc}），正在重连")
                    self._reconnect()
                    continue
                if not data:
                    if self._stop.is_set():
                        break
                    self._event("20004 连接被控制器断开（可能另有程序连了 20004），正在重连")
                    self._reconnect()
                    continue
                for frame in self._parser.feed(data):
                    self._store(Robot.RobotStatePkg.from_buffer_copy(frame), t_rt, t_mo)
        except Exception as exc:  # noqa: BLE001 - 记下来，由调用方停止
            self.error = f"{type(exc).__name__}: {exc}"
            self._event(f"录制线程出错：{self.error}")
        finally:
            with self._cond:
                self._cond.notify_all()

    def _store(self, pkg, t_rt: int, t_mo: int) -> None:
        ctrl_ms = ((pkg.hour * 60 + pkg.minute) * 60 + pkg.second) * 1000 + pkg.millisecond
        if self._last_ctrl_ms is not None:
            step = (ctrl_ms - self._last_ctrl_ms) % DAY_MS
            if step == 0:
                self._duplicates += 1
            elif step > MAX_CTRL_STEP_MS:
                self._gaps.append((t_rt, step))
        self._last_ctrl_ms = ctrl_ms
        if (pkg.tool, pkg.user) != (0, 0) and (pkg.tool not in self._tools or pkg.user not in self._users):
            self._event(f"注意：工具号={pkg.tool}、工件号={pkg.user}，不是 0/0")
        self._tools.add(pkg.tool)
        self._users.add(pkg.user)
        if pkg.main_code or pkg.sub_code:
            self._codes.add((pkg.main_code, pkg.sub_code))
        self._estop_seen |= bool(pkg.EmergencyStop)
        row = _pack(pkg, t_rt, t_mo)
        with self._cond:
            self._rows += row
            self.count += 1
            self._latest = pkg
            self.latest_monotonic = t_mo / 1e9
            self._cond.notify_all()

    def _event(self, message: str) -> None:
        self._events.append((time.time_ns(), message))
        print(f"[fr5_record] {message}", file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- 写盘
    def _write(self) -> None:
        tmp = self.csv_path.with_name(self.csv_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(CSV_COLUMNS) + "\n")
            for vals in _ROW.iter_unpack(self._rows):
                f.write(_csv_line(vals) + "\n")
        os.replace(tmp, self.csv_path)

        start, stop = self._meta["start"], self._meta["stop"]
        first = _ROW.unpack_from(self._rows, 0) if self.count else None
        last = _ROW.unpack_from(self._rows, (self.count - 1) * _ROW.size) if self.count else None
        duration = (last[1] - first[1]) / 1e9 if self.count > 1 else 0.0
        clock_step_ms = ((stop["realtime_ns"] - stop["monotonic_ns"])
                         - (start["realtime_ns"] - start["monotonic_ns"])) / 1e6
        meta = {
            "tool": "fr5_record.py",
            "format_version": 1,
            "controller_ip": self.ip,
            "controller": self._meta["controller"],
            "box_hostname": socket.gethostname(),
            "sdk_robot_py": Robot.__file__,
            "sdk_robot_py_sha256": hashlib.sha256(Path(Robot.__file__).read_bytes()).hexdigest(),
            "frame_bytes": FRAME_SIZE,
            "rcvbuf_bytes": self.rcvbuf_bytes,
            "start": start,
            "stop": stop,
            "ntp_start": self._meta["ntp_start"],
            "ntp_stop": self._meta["ntp_stop"],
            "system_clock_step_ms": round(clock_step_ms, 3),
            "packets": self.count,
            "duration_s": round(duration, 3),
            "rate_hz": round((self.count - 1) / duration, 2) if duration > 0 else None,
            "gaps": {"count": len(self._gaps),
                     "first": [{"box_realtime_ns": t, "ctrl_step_ms": s} for t, s in self._gaps[:50]]},
            "duplicates": self._duplicates,
            "checksum_failures": self._parser.bad_checksum,
            "length_mismatches": self._parser.bad_length,
            "events": [{"box_realtime_ns": t, "message": m} for t, m in self._events],
            "tools_seen": sorted(self._tools),
            "users_seen": sorted(self._users),
            "error_codes_seen": sorted(self._codes),
            "emergency_stop_seen": self._estop_seen,
            "columns": CSV_COLUMNS,
            "notes": [
                "box_realtime_ns / box_monotonic_ns：Box 收到该包时的系统时间 / 单调时钟（ZED 时间戳也是系统时间）。",
                "ctrl_time：包内控制器本地时间（ms），控制器时钟未与 Box 同步；ctrl_unix_ms 按 Box 时区换算。",
                "位置 mm，角度 °，关节速度 °/s；法兰位姿为法兰在基坐标系下，R = Rz(rz)·Ry(ry)·Rx(rx)。",
                "system_clock_step_ms 不为 0 说明录制期间系统时间被调整过（如 NTP 跳变）。",
            ],
        }
        tmp = self.meta_path.with_name(self.meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)


class _NoSocket:
    """给 SDK 的占位 socket：CloseRPC() 会调用 close()。"""

    def close(self) -> None:
        pass


class RecordingRPC(Robot.RPC):
    """fairino 的 RPC，但不自己连 20004，状态缓存由 StateRecorder 提供。

    运动、IO、查询等指令照常走 SDK 的 XML-RPC（20003 端口）。
    """

    def __init__(self, recorder: StateRecorder, ip: str = DEFAULT_IP):
        self._recorder = recorder
        super().__init__(ip)

    def connect_to_robot(self):  # SDK 在 __init__ 和 reconnect() 里调用
        self.sock_cli_state = _NoSocket()
        self.sock_cli_state_state = True
        return True

    def robot_state_routine_thread(self):  # 替换 SDK 自己的 20004 接收线程
        count = -1
        while not self.stop_event.is_set() and not self.closeRPC_state:
            count, pkg = self._recorder.wait_state(count, timeout=0.5)
            if pkg is not None and not self.closeRPC_state:
                self.robot_state_pkg = pkg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="录制 FR5 状态（20004 端口），停止后一次写盘。")
    parser.add_argument("--output", type=Path, default=Path("~/ROBOT/zed_imu/build/save_data"),
                        help="输出目录，默认和 zed_record 的 save_data 放在一起")
    parser.add_argument("--ip", default=DEFAULT_IP, help="控制器 IP")
    parser.add_argument("--duration", type=float, help="录制秒数；不给就一直录到 Ctrl+C")
    parser.add_argument("--force", action="store_true", help="覆盖输出目录里已有的 fr5_state.csv")
    args = parser.parse_args(argv)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    recorder = StateRecorder(args.output, args.ip, args.force)
    try:
        recorder.start()
    except (OSError, RuntimeError) as exc:
        print(f"无法开始录制：{exc}", file=sys.stderr)
        return 1
    print(f"开始录制 FR5 状态 -> {recorder.csv_path}（Ctrl+C 停止）", flush=True)
    t0 = time.monotonic()
    try:
        while recorder.error is None:
            wait = 5.0
            if args.duration is not None:
                left = args.duration - (time.monotonic() - t0)
                if left <= 0:
                    break
                wait = min(wait, left)
            if stop.wait(wait):
                break
            print(recorder.status_line(), flush=True)
    finally:
        recorder.stop()
    print(f"已写出 {recorder.csv_path} 和 {recorder.meta_path.name}：{recorder.summary()}")
    return 0 if recorder.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
