#!/usr/bin/env python3
"""在笔记本上放 run_session.py --cue 的提示音（Box 的显示器没有喇叭时用）。Windows，只用标准库。

    python cue_listener.py [--host user@10.255.250.132]

通过 SSH 跟踪 Box 上的 ~/ROBOT/cue_events.log（run_session.py 每次提示追加一行：时间、提示音、文字），
收到一行就显示文字，并用笔记本的喇叭响：准备 = 一声短的，进画面 = 三声高的，离开 = 两声低的。
先开着它，再在 Box 上运行 `python3 run_session.py cue-test` 试一遍。Ctrl+C 退出。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import winsound

PATTERNS = {"ready": [(880, 200)], "go": [(1320, 250)] * 3, "leave": [(660, 400)] * 2}


def play(kind: str) -> None:
    for freq, ms in PATTERNS.get(kind, []):
        winsound.Beep(freq, ms)
        time.sleep(0.05)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="user@10.255.250.132")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    proc = subprocess.Popen(["ssh", "-o", "ServerAliveInterval=5", args.host, "tail -n0 -F ~/ROBOT/cue_events.log"],
                            stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    print(f"在听 {args.host} 的提示（Ctrl+C 退出）……", flush=True)
    try:
        for line in proc.stdout:
            parts = line.rstrip("\n").split(" ", 2)
            if len(parts) < 3:
                continue
            threading.Thread(target=play, args=(parts[1],), daemon=True).start()
            now = time.time()
            print(f"{time.strftime('%H:%M:%S', time.localtime(now))}.{int(now % 1 * 1000):03d}  >>> {parts[2]}",
                  flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
