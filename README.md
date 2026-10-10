# FR5 + ZED X 数据录制（SIO_SLAM 机械臂真值）

> 最后同步：2026-10-10。**正本在 Box 的 `~/ROBOT/fr5_record/`**，本文件夹是笔记本上的备份。
> Box 上的脚本改动后，在本文件夹运行 `.\sync_from_box.ps1` 拉回最新版本。

## 1. 目标

用 FR5 机械臂带着 ZED X 做已知运动，同时录制双目图像、ZED 机身 IMU 和机械臂状态，离线算出相机位姿真值，供 SIO_SLAM 评估。

第一批只录 6 条：相机始终**水平朝前**，每条只沿一个方向做纯平移：前进、后退、右移、左移、上升、下降（默认 0.05 m/s，行程 300 mm）。

相机真值：`base_T_camera(t) = base_T_flange(t) · flange_T_left_camera`。其中手眼外参来自标定仓库 s003 的结果（`scripts/handeye_s003_result.json`），前提是相机安装之后没有动过。

## 2. 硬件、网络与软件版本

| 项目 | 值 |
|---|---|
| 机械臂 | FAIRINO FR5，型号 `FR5-V1-002(V6.0)`；WebApp `v3.9.1`；控制器软件 `V3.9.5-QX` |
| 相机 | ZED X（S/N 49953548），ZED SDK 5.1.2，HD1080 / 15 fps，校正后左目，`FLIP_MODE.OFF`，关闭自标定 |
| 计算单元 | ZED Box Mini（Jetson，L4T R36.4.4，Ubuntu 22.04，Python 3.10） |
| Box WiFi | `10.255.250.132`（DHCP，可能会变），笔记本通过它 SSH：`ssh user@10.255.250.132` |
| Box 有线 | `192.168.58.11/24`（NetworkManager "Wired connection 1"，固定 IP，无网关），网线直连 FR5 控制器 `192.168.58.2` |
| 控制器端口 | 20003 XML-RPC 指令；20004 实时状态推送（**同一时间只服务一个连接**）；80 / 9999 WebApp |
| FAIRINO Python SDK | V2.2.1（对应机器人 V3.9.1），官方仓库 `FAIR-INNOVATION/fairino-python-sdk` 提交 `060c68bb85f05d0097cc5efa360ddc61d6bb72e0` 的 `linux/fairino/Robot.py`，装在 Box `~/.local/lib/python3.10/site-packages/fairino/`，SHA256 `b0dbb2e4…103e`。只依赖标准库 |
| WebApp 浏览器 | Box 上用 Falkon：`falkon http://192.168.58.2/ &` |

SDK V2.2.5（对应 V3.9.5）试过，不能用：它改用 20005 端口的 CNDE 协议，这台控制器拒绝连接 20005。

## 3. 文件说明

| 文件 | 说明 |
|---|---|
| `scripts/run_session.py` | **总控脚本，所有可改参数都在文件开头**。子命令：`set-start`、`plan`、`goto-start`、`speed-test`、`sync-test`、`record 方向` |
| `scripts/fr5_motion.py` | 运动规划与执行：相机保持水平的平移和同步摆动，用控制器逆解做检查；`MoveL` 用**百分比速度**非阻塞下发，由状态判断到位，随时可以 `StopMotion`；运动中实时测法兰线速度，超过预期 1.5 倍 + 5 mm/s 就中止 |
| `scripts/fr5_record.py` | 机械臂状态录制：独占 20004 端口，每包打 Box 时间戳，停止时一次写盘；`open_rpc()` 给 SDK 运动脚本共用同一个连接 |
| `scripts/fr5_groundtruth.py` | 离线真值：控制器时钟拟合 → 陀螺仪互相关估 Δt → 曝光中心修正 → 插值出相机真值（只依赖 numpy） |
| `scripts/test_fr5_groundtruth.py`、`scripts/test_fr5_motion.py` | 离线测试（合成数据，不连机械臂）：`python3 -B -m unittest test_fr5_groundtruth test_fr5_motion` |
| `scripts/movel_probe.py` | 诊断：发零行程 MoveL/MoveJ（目标 = 当前位姿），看哪种被控制器拒绝；`--set physical` 试物理速度模式的取值 |
| `scripts/plain_sdk_test.py` | 诊断：完全照官方例子，用 `Robot.RPC` 发一条零行程 `MoveJ` |
| `scripts/handeye_s003_result.json` | 标定仓库 `results/s003/result.json` 的副本（手眼外参） |
| `scripts/speed_calibration.json` | `speed-test` 的结果：速度百分比 1% 对应多少 mm/s（平移）和 °/s（同步摆动）。还没跑过 speed-test 时没有这个文件 |
| `scripts/start_pose.json` | `set-start` 保存的水平起点（当前：相机光心约 (597, −476, 562) mm，关节 [119.51, 14.41, −74.33, −119.99, −44.77, −0.02]） |
| `zed_record/zed_record.cpp` | Box 上 `~/ROBOT/zed_imu/zed_record.cpp` 的当前版本 |
| `zed_record/zed_record.patch` | 相对 10-08 版本的改动：加了 `--exposure-us`、`--analog-gain-mdb` 参数，第 30 帧读回实际曝光并写进 `time_meta.txt` |
| `sync_from_box.ps1` | 从 Box 拉取以上文件到本文件夹 |

## 4. 在 Box 上安装

1. **SDK**：把上面那个提交的 `linux/fairino/Robot.py` 放到 `~/.local/lib/python3.10/site-packages/fairino/`，再建一个空的 `__init__.py`，用 `sha256sum` 核对。
2. **zed_record**：源码在 `~/ROBOT/zed_imu/`，改动见 `zed_record/zed_record.patch`。改之前的版本备份为 `zed_record.cpp.bak-20261010` 和 `build/zed_record.bak-20261010`。重新编译：
   ```bash
   cd ~/ROBOT/zed_imu/build && make -j4 zed_record
   ```
3. **脚本**：`scripts/` 里的文件放到 Box 的 `~/ROBOT/fr5_record/`。

## 5. 使用流程

运动和录制最好在 **Box 自己的终端**里运行，人站在机械臂旁边，手边有急停。每次运动前脚本都会列出计划，等你输入 `yes` 才会动。

```bash
cd ~/ROBOT/fr5_record
python3 run_session.py set-start               # 读当前位姿，算出水平起点并保存（不动）
python3 run_session.py plan                    # 空跑检查 6 条录制和 sync-test 的全部目标点（不动）
python3 run_session.py goto-start --scale 0.25 # 低速移到水平起点
python3 run_session.py speed-test              # 速度标定（不开相机）；改了 WebApp 速度百分比要重跑
python3 run_session.py plan                    # 再空跑一次，这次会显示每段的速度百分比
python3 run_session.py sync-test               # 时间同步验证（只做同步摆动），重复 3 次
python3 run_session.py record forward          # 正式录制，依次 backward / right / left / up / down
```

每条录制的结构：静止 5 s → 同步摆动（绕相机竖直轴 ±10°，20°/s，来回 2 次）→ 静止 3 s → 单向平移 → 静止 3 s → 同步摆动 → 静止 5 s。每条约 31 s，约 0.4 GB。

- **速度百分比**：控制器拒绝物理速度模式（见 §9），所以 `MoveL` 用百分比速度。百分比和实际速度的关系受 WebApp 速度百分比影响，SDK 又读不到它，因此先用 `speed-test` 实测，结果存在 `speed_calibration.json`。`sync-test` 和 `record` 必须有这个文件，并且其中的 `WEBAPP_SPEED_PCT`、`ACCEL_PCT`、`WIGGLE_DEG` 和脚本开头一致。`goto-start` 和 `speed-test` 在没标定时按保守假设换算（1% = 默认速度 1000 mm/s 的 1%），实际只会更慢。
- **speed-test 做什么**：先移到起点；竖直上移 `SPEED_TEST_MM` 用固定的 `SPEED_TEST_PCT`，按实测换算后以录制速度下移；再绕相机光心竖直轴转 `WIGGLE_DEG`（固定百分比），按实测换算后以摆动角速度转回。去程和回程算出的“速度 / 百分比”相差超过 `MAX_SPEED_ERROR` 会警告（不成正比）。
- **每段运动后**终端会打印实测的法兰线速度、角速度和用时，可以和计划里的“预计”对照。
- **首次运行**：用 `--scale 0.25` 低速演练，但这时同步摆动只有 5°/s，算不出 Δt，这条数据不用。
- **按 Ctrl+C**：先发 `StopMotion`，再正常停止相机，保存已录的数据。
- **SSH 断开**：处理方式和 Ctrl+C 相同，之后的输出写进录制目录里的 `run_session.log`。

## 6. 可调参数（`run_session.py` 开头）

超出范围脚本会拒绝运行。另外，速度和曝光的组合必须满足"最近物体处的模糊 ≤ `MAX_BLUR_PX`"。

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `SPEED_MPS` | 0.05 | 0.02–0.10 | 平移速度 m/s；下限待平稳性实测确认 |
| `STROKE_M` | 0.30 | 0.05–0.30 | 行程，以起点为中心两边各一半 |
| `EXPOSURE_US` | 10000 | 1000–15000 | 曝光，传给 zed_record |
| `ANALOG_GAIN_MDB` | 1000 | 1000–16000 | 模拟增益，传给 zed_record |
| `MAX_BLUR_PX` / `NEAREST_DEPTH_M` | 1.0 / 1.0 | 0.3–2.0 / 0.3–20 | 允许的运动模糊（1080p 像素）和画面最近物体距离 |
| `HOLD_START_S` / `HOLD_MID_S` / `HOLD_END_S` | 5 / 3 / 5 | 3–30 / 2–30 / 3–30 | 静止时长 |
| `WIGGLE_DEG` / `WIGGLE_DEG_S` / `WIGGLE_CYCLES` | 10 / 20 / 2 | 5–15 / 12–40 / 1–4 | 同步摆动 |
| `REPOSITION_MPS` | 0.05 | 0.01–0.10 | 录制前移到起点的速度 |
| `WEBAPP_SPEED_PCT` | 10 | 1–100 | **WebApp 里的速度百分比，照抄过来**（SDK 读不到）。两边必须一致，改了要重跑 speed-test |
| `ACCEL_PCT` | 100 | 1–100 | MoveL 加速度百分比（oacc），实际加减速时间看 speed-test 的输出；改了要重跑 speed-test |
| `SPEED_TEST_PCT` / `SPEED_TEST_MM` | 5 / 100 | 1–10 / 50–150 | speed-test 去程的速度百分比和竖直行程 |
| `MAX_SPEED_ERROR` | 0.10 | 0.02–0.3 | 平移实测速度和预计相差超过它就警告 |
| `MIN_CAMERA_Z_M` | 0.35 | 0–1.5 | 相机光心最低高度（基坐标系 z） |
| `JOINT_MIN_DEG` / `JOINT_MAX_DEG` | 见文件 | | 允许的关节范围，按现场确认 |
| `MAX_SPEED_RIPPLE` | 0.05 | 0.01–0.2 | 平移巡航段速度波动超过它就警告"不平稳" |

默认参数下，1 m 处的运动模糊为 0.37 px（前后平移时画面边缘约 0.48 px）。

## 7. 输出数据

```text
~/ROBOT/sessions/YYYYmmdd_HHMMSS_<名称>/
├─ zed_record.log            zed_record 的输出（结束时一次写入）
├─ run_session.log           仅在 SSH 断开 / 被终止时才有
└─ save_data/
   ├─ left/  right/          <ZED 时间戳 ms>.jpg（校正后，JPEG）
   ├─ img_timeline.csv       每帧：图像时间戳、对应 IMU 时间戳、文件名
   ├─ zed_imu.csv            机身 IMU 全部样本（约 200 Hz），含 image_sync_trigger
   ├─ time_meta.txt          相机设置，含第 30 帧读回的实际曝光
   ├─ fr5_state.csv          机械臂状态（约 100 Hz）：Box 时间、控制器时间、关节、法兰位姿、错误码等
   ├─ fr5_meta.json          丢包、断线、时钟跳变等检查结果
   ├─ motion_log.json        每段指令的时间、速度百分比、预计速度、实测速度，平移段实际速度和波动，所用速度标定
   ├─ groundtruth_tum.txt    每张图一行：timestamp tx ty tz qx qy qz qw（Unix 秒、米），左目在 FR5 基坐标系下
   ├─ groundtruth_arm_rate_tum.txt   机械臂每个状态包一行，格式同上
   ├─ groundtruth_images.csv         每张图的真值和标记（ok / outside / gap / robot_error）
   └─ groundtruth_report.json        时钟拟合、Δt、曝光中心偏差、警告
```

真值可以单独重算：

```bash
python3 fr5_groundtruth.py <save_data 目录> --handeye handeye_s003_result.json
```

## 8. 时间同步与真值方法

- **统一时钟**：所有时间戳都统一到 Box 系统时间（Unix）。ZED 的图像和 IMU 时间戳本来就是 Box 系统时间；机械臂状态包在 Box 收到时打时间戳，同时记下包里的控制器时间。
- **控制器时钟换算**：控制器时钟没有同步，比 Box 快约 127 s，漂移约 200 ppm。每 2 s 取一次"到达时间 − 控制器时间"的最小值，直线拟合出偏差和漂移。
- **剩余时间差 Δt**：用 ZED 陀螺仪角速度大小和法兰姿态差分出的角速度大小做互相关。所以每条录制开头和结尾都要有同步摆动，录制满 60 s 时会把前后两段分开估计、互相核对。
- **曝光中心**：ZED 图像时间戳不在曝光中间，实测比曝光中心晚约 3 ms。用 `image_sync_trigger=1` 的 IMU 样本，每条录制自动估一次，在曝光中心取真值；输出里仍写图像原来的时间戳。由于 IMU 采样只有 5 ms 一个，这个估计最多有约 ±0.8 ms 的偏差。
- **合成数据测试结果**：Δt 误差小于 1 ms，真值位置误差 RMS 0.01 mm、转角误差 RMS 0.007°。

## 9. 当前进度（2026-10-10）

**已完成**
- Box 网络：有线直连 FR5；FAIRINO SDK V2.2.1 安装，读状态和查询都正常。
- `fr5_record`：实测 100 Hz、无丢包；和 WebApp 同时开着也不受影响（WebApp 用 80 / 9999 端口）。
- `fr5_groundtruth`：时钟拟合、Δt、曝光中心修正；合成数据测试通过。
- `zed_record`：曝光和增益参数化。实测设 5000 µs 读回 4999 µs，手动曝光确实生效。
- `run_session` / `fr5_motion`：参数检查；`set-start` 已保存水平起点；`plan` 空跑 6 条和 sync-test 全部通过控制器逆解检查。
- SSH 断开时的收尾逻辑已测试（没有运动的情况下）。
- 21 项离线测试在笔记本和 Box 上都通过（含假机械臂上的运动执行和超速中止测试）。
- 百分比速度版本已部署到 Box；在真机上做了只读检查：`plan` 和 speed-test 的全部目标点都通过控制器逆解检查。在笔记本上用假机械臂把 `speed-test → plan → goto-start` 完整模拟过一遍。

**已解决：SDK 运动指令全部返回 172**
- 现象：`MoveL`（各种参数）和 `MoveJ` 都返回 172，机械臂不动。当时处于自动模式、已使能、无报警，SDK 通讯正常。
- 原因：WebApp 里的**速度百分比是 0**。按 FAIRINO 错误码表，172 的意思是“运动速度不能为 0”。在 WebApp 里把速度百分比从 0 改成 10 后，百分比速度的 `MoveL`/`MoveJ` 返回 0。
- 这期间试过 SDK V2.2.5，不行（控制器没有 CNDE 20005 端口），已删除。

**已绕开：物理速度模式（mm/s）被拒绝，改用百分比速度**
- `velAccParamMode=1` 的零行程测试（`movel_probe.py --set physical`）：12.5–200 mm/s、加速度 100–2000 mm/s²、ovl 25/100 全部被拒，返回 182（给定物理速度超限）或 183（给定物理加速度超限）。有可能和 WebApp 速度百分比为 10% 有关，也可能是零行程测试本身的问题，没有再查。
- 改用百分比速度（`velAccParamMode=0`）。百分比和实际速度的换算用 `speed-test` 实测；运动中还会实时检查速度。
- 控制器的 `GetDefaultTransVel()` 返回 1000 mm/s。如果 WebApp 的 10% 也作用在 SDK 指令上，1% 约等于 1 mm/s，50 mm/s 需要 50%；speed-test 会给出实际值。

**接下来**
1. 在 Box 终端运行 `python3 run_session.py speed-test --scale 0.25`（首次低速），然后不加 `--scale` 再跑一次，作为正式标定。看输出的“1% = … mm/s”和“相当于默认速度 × …”。
2. `python3 run_session.py plan`，确认各段速度百分比都不超过 100%（同步摆动 20°/s 如果超过，需要调高 WebApp 速度百分比，并同时改 `WEBAPP_SPEED_PCT`，然后重跑 speed-test）。
3. `sync-test --scale 0.25` 低速演练，期间试一次运动中按 Ctrl+C 能否立即停下。
4. `sync-test` × 3，验证时间同步。
5. 写 `gt_board_check.py`（在笔记本上跑，调用标定仓库的标定板检测），把标定板竖着放在相机前方 0.8–1 m 处，验证真值。
6. 撤掉标定板，正式录 6 条。
7. 可选：加一个加速度计交叉检查；实测确定速度下限。

## 10. 注意事项

- **20004 只能有一个连接**：录制时不要另开 `Robot.RPC(...)`。运动脚本必须用 `StateRecorder.open_rpc()`。
- **不要调用 SDK 的 `LoggerInit()`**：会持续写日志到磁盘。
- **WebApp 速度百分比不能是 0**（否则所有运动指令返回 172），并且要和 `WEBAPP_SPEED_PCT` 一致。改了它就重跑 speed-test。万一忘了改，运动中的超速检查会在速度超过预期 1.5 倍 + 5 mm/s 时中止。
- **FAIRINO 错误码**：SDK 返回的正数错误码在官方“错误码对照表”里查，例如 172 运动速度不能为 0；182 给定物理速度超限；183 给定物理加速度超限。
- **相机同一时间只能被一个程序打开**：录制前关掉 ZED Explorer、Depth Viewer 和 `zed_snapshot.py`。
- **Box 的 WiFi 断过**：运动和录制尽量在 Box 本机终端跑。Box 没装 tmux 或 screen。
- **Box 上 Python 的 OpenCV 导入失败**（用户目录里的 numpy 2 和系统 OpenCV 冲突），所以标定板检测放在笔记本上做。
- **Box 开机后先预热**：标定时记录过 Box 重启后相机有冷热跳变，开机后先预热 30 分钟以上再录真值数据。
- **画面里不要有人走动**：静态场景实验要求场景完全静止。数据如果要公开，先征得画面里的人同意。

## 11. 变更记录

- 2026-10-09：Box 有线网口固定为 192.168.58.11 直连 FR5；安装 FAIRINO SDK V2.2.1；完成 `fr5_record.py`。
- 2026-10-10：完成 `fr5_groundtruth.py`（含曝光中心修正）、`fr5_motion.py`、`run_session.py` 和两个测试文件；`zed_record` 增加曝光和增益参数；加入断线收尾；排查 172（`movel_probe.py`、`plain_sdk_test.py`），确认 SDK V2.2.5 不兼容；在笔记本上建立本备份。
- 2026-10-10（下午）：查明 172 是 WebApp 速度百分比为 0（已改成 10）；物理速度模式返回 182/183，`fr5_motion.py` 改用百分比速度。新增 `speed-test` 子命令和 `speed_calibration.json`；每段运动实时测速，超速就中止；没到位又 5 s 不动就中止；每段运动后打印实测速度。删去 `ACCEL_MPS2`、`WIGGLE_ACCEL_MPS2`，新增 `WEBAPP_SPEED_PCT`、`ACCEL_PCT`、`SPEED_TEST_PCT`、`SPEED_TEST_MM`、`MAX_SPEED_ERROR`。测试增加到 21 项。
