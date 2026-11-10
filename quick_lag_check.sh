#!/usr/bin/env bash
# 录完马上查图像是否晚一帧：只从 Box 拷同步摆动段（前后各放宽 0.4 s）的左目图和 csv，约 200 张、1 分钟一条，
# 在笔记本跑 scripts/frame_lag_check.py。结论和拷全部左目图后跑的一样，但报告写在 data/quick_lag/ 下，
# 不是交付用的 frame_lag_report.json（交付前按运行手册第 8 节在完整数据上再跑一次）。
# 用法（笔记本 Git Bash，仓库根目录）：  ./quick_lag_check.sh <录制名>...
# 环境变量：BOX（默认 user@10.255.250.132）、PY（默认 calibration 的 venv）。
set -u
BOX="${BOX:-user@10.255.250.132}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/data/quick_lag"
PY="${PY:-$ROOT/../calibration/.venv/Scripts/python.exe}"
[ $# -gt 0 ] || { echo "用法：$0 <录制名>..."; exit 2; }
status=0
for N in "$@"; do
  mkdir -p "$OUT/$N/save_data"
  # 文件清单直接管道给 tar（不落临时文件，同时跑几个也不会互相覆盖）
  ssh "$BOX" "cd ~/ROBOT/sessions_verify/$N/save_data && python3 - <<'PY' | tar cf - -T -
import csv, json
segs = [s for s in json.load(open('motion_log.json', encoding='utf-8'))['segments']
        if s.get('kind') == 'move' and s.get('name', '').startswith('wiggle') and s.get('start_box_ns')]
win = [(s['start_box_ns'] / 1e6 - 400, s['end_box_ns'] / 1e6 + 400) for s in segs]
print('img_timeline.csv'); print('zed_imu.csv'); print('motion_log.json')
for r in csv.DictReader(open('img_timeline.csv')):
    if any(a <= float(r['zed_timestamp_ms']) <= b for a, b in win):
        print(r['filename_left'])
PY
" | tar xf - -C "$OUT/$N/save_data" || { echo "$N：拷贝失败"; status=1; continue; }
  echo "$N：拷了 $(ls "$OUT/$N/save_data/left" | wc -l) 张"
  (cd "$ROOT/scripts" && PYTHONUTF8=1 "$PY" -B frame_lag_check.py "$OUT/$N/save_data") || status=1
done
exit $status
