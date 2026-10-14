# FR5 + ZED X 高速数据录制（SIO_SLAM 机械臂真值，high-speed 分支）

> 本分支录制高速运动的数据。低速数据（0.05 m/s 平移、5 °/s 旋转，2026-10-12/13）及其真值验证见 main 分支或标签 `data-v1`。
> 脚本的正本在 Box 的 `~/ROBOT/fr5_record/`，本仓库是笔记本上的备份，用 `.\sync_from_box.ps1` 从 Box 拉取。
> 实验过程、工程细节和问题处理见 [实验记录.md](实验记录.md)。

## 1. 目标

用 FAIRINO FR5 机械臂带着 ZED X 做已知的高速运动，同时录制双目图像、ZED 机身 IMU 和机械臂状态，离线算出逐帧的相机位姿真值，供 [SIO_SLAM](https://github.com/lihongliang0203/SIO_SLAM) 评估快速运动下的表现。

计划 12 条录制：

| 运动 | 内容 | 条数 |
|---|---|---|
| 平移 0.5 m/s | 相机水平朝前，沿一个方向平移：前后 440 mm、左右 530 mm、上下 440 mm，正反方向各一条 | 6 |
| 旋转 50 °/s | 绕左目光心单轴旋转：偏航 30°、俯仰 30°、横滚 90°，正反方向各一条 | 6 |

- **相机**：30 fps，曝光 2.5 ms。
- **场景**：画面里不能有人走动，不能拍到右侧的落地窗（左窗可以拍）。
- **相机真值**：方法和低速数据相同，`base_T_camera(t) = base_T_flange(t) · flange_T_left_camera`。手眼外参来自标定仓库 [fr5-zedx-handeye-calibration](https://github.com/SaxonShang/fr5-zedx-handeye-calibration) s003 的结果（`scripts/handeye_s003_result.json`）。高速路径和新曝光下的修正量要用标定板重新验证（3.3）。
- 1 m/s 平移不可行：在现有关节范围内，前后方向连匀速段都放不下。

## 2. 硬件、网络与软件版本

| 项目 | 值 |
|---|---|
| 机械臂 | FAIRINO FR5，型号 `FR5-V1-002(V6.0)`；WebApp `v3.9.1`；控制器软件 `V3.9.5-QX`；**WebApp 自动模式速度百分比 60%** |
| 相机 | ZED X（S/N 49953548），ZED SDK 5.1.2，HD1080，**30 fps**，全局快门；**手动曝光 2.5 ms，模拟增益 13 dB**；录校正后的左右目 JPEG，关闭自标定 |
| 相机录制程序 | `zed_record`：帧率可设（`--fps 15/30/60`），3 个线程并行存 JPEG |
| 计算单元 | ZED Box Mini（Jetson，L4T R36.4.4，Ubuntu 22.04，Python 3.10），自带显示器和鼠标 |
| 网络 | Box 有线 `192.168.58.11/24` 用网线直连 FR5 控制器 `192.168.58.2`（固定 IP，无网关）；Box WiFi `10.255.250.132`（DHCP，可能会变），笔记本通过它 SSH：`ssh user@10.255.250.132` |
| 控制器端口 | 20003 XML-RPC 指令；20004 实时状态推送（同一时间只服务一个连接）；80 / 9999 WebApp |
| FAIRINO Python SDK | V2.2.1（对应机器人 V3.9.1）：官方仓库 `FAIR-INNOVATION/fairino-python-sdk` 提交 `060c68bb85f05d0097cc5efa360ddc61d6bb72e0` 的 `linux/fairino/Robot.py`，只依赖标准库 |
| 笔记本 | 标定板验证和图像检查用标定仓库 `D:\天府绛溪实验室\calibration` 的 `.venv`（OpenCV 4.14，带 aruco）；Box 上的 Python 导入不了 OpenCV |

## 3. 算法

### 3.1 相机真值

和低速数据相同：每张图像在曝光中心时刻插值法兰位姿（位置线性插值，姿态 SLERP），再乘手眼外参，得到左目相机在 FR5 基坐标系下的位姿；输出仍用图像原来的时间戳。

修正量按“路径组”记录（脚本参数 `PATH_SET`，本分支为 `fast1`；低速数据为 `base`），写进每条录制的 `motion_log.json`，真值脚本据此查表：

| 修正 | 低速数据（`base`） | 高速数据（`fast1`） |
|---|---|---|
| 时间：取真值的时刻再晚 | 4.27 ms（10 ms 曝光下测得） | **待测**：2.5 ms 曝光下曝光中心从 −3.4 ms 变为 +0.8 ms，这项要重新测 |
| 行程放大（平移） | 左右、上下 1.0070，前后 1.0015 | **待测**：新路径，暂不放大，真值报告里有警告 |
| 转角放大（旋转） | 俯仰向上 1.0043、向下 1.0052 | 俯仰同一条路径，沿用；偏航 30°、横滚 90° 是新路径，**待测** |

同时输出一份不加这些修正的真值，用于对照。

### 3.2 时间同步

方法和低速数据相同：录制程序独占 20004 连接记下每个状态包的 Box 时间；控制器时钟按最小延迟的包直线拟合；剩余时间差 Δt 由 ZED 陀螺仪和法兰姿态的角速度大小互相关得到（每条录制开头和结尾各有一段同步摆动）；曝光中心由 `image_sync_trigger` 估计；`frame_lag_check.py` 检查图像内容和时间戳是否对齐。

### 3.3 真值验证（标定板）

方法和低速数据相同：标定板固定在相机前约 1.1 m，PnP 求出标定板位姿，乘真值后应始终不变；检查重复性、运动前后的差（行程 / 转角）和时间扫描。

**高速数据的验证待进行**：要测 2.5 ms 曝光下的时间修正、三条新平移路径的行程系数、偏航 30° 和横滚 90° 的转角系数，并确认俯仰系数在高速下不变。

### 3.4 运动与速度

- **平移路径**：每对方向一个行程中点（相对低速数据的起点偏移）和行程，由离线逆解在现有关节范围内找出：

  | 方向对 | 行程中点相对起点（前, 左, 上） | 中点光心（基坐标系） | 行程 | 匀速段（加速度按 2.5 m/s² 估） |
  |---|---|---|---|---|
  | 前 / 后 | (−20, +100, −50) mm | (607, −375, 512) mm | 440 mm | 约 340 mm，0.68 s |
  | 左 / 右 | (−50, +45, +50) mm | (562, −419, 612) mm | 530 mm | 约 430 mm，0.86 s |
  | 上 / 下 | (−50, +300, +130) mm | (636, −174, 692) mm | 440 mm | 约 340 mm，0.68 s |

  约束：关节在允许范围内（J2 ≤ 34°，大臂再往下倾肘部会伸到机柜边上）、光心不低于 0.35 m、两端都能做同步摆动、关节速度 ≤ 150 °/s、看不到右窗。匀速段长度取决于实测加速度（`speed-test` 会打印）。
- **旋转**：绕左目光心，MoveC。

  | 方向 | 转角 | 姿态范围（相对起点，正 = 向左 / 向上 / 顶部向右） | 50 °/s 时最快的关节 | 离右窗最近 |
  |---|---|---|---|---|
  | 偏航 | 30° | 朝向 −7.25～+22.75° | J5 61 °/s | 1.3° |
  | 俯仰 | 30° | 朝向 −5°，俯仰 −15～+15° | J4 96 °/s | 2.2° |
  | 横滚 | 90° | 朝向 −2°，横滚 −45～+45° | J6 51 °/s | 2.2° |

- **同步摆动**：平移录制只往左摆 10°（画面右边缘离右窗更远）；旋转录制绕本条的转轴往内侧摆；都是 20 °/s、来回 2 次。
- **不拍右窗**：玻璃区域（`GLASS_REGIONS`）只有右窗，`plan` 和 `record` 检查画面边缘离它至少 1°。平移时光心在动，按窗户离起点至少 7 m 算视差（偏保守）。平移录制离右窗 7.2–9.3°。

  ![实验室全景图](docs/scene_panorama.jpg)

  *全景图：横轴方位角、纵轴仰角（°），以起点光心为球心、相对起点水平朝向，向左为正。左侧 +57° 以左是左窗，右侧 −60° 以右是落地窗。*

- **关节速度**：录制前按预计速度检查每段路径上的关节速度，超过 150 °/s（FR5 额定 180 °/s）就拒绝；有实测加速度时按梯形速度曲线算（行程两端在加减速）。
- **运动模糊**：曝光 2.5 ms 时，0.5 m/s 平移在 1 m 处 0.92 px，50 °/s 旋转在画面中心 1.6 px。
- **图像**：2.5 ms、13 dB 的亮度和低速数据（10 ms、1 dB）相近，时间噪声约为它的 2 倍（2.1 对 1.05 个灰度级）。30 fps 下 60 s 空录不丢帧（帧间隔 31–35 ms）。
- **保护**：和低速数据相同（超速、光心偏离、状态中断、停顿时 StopMotion；Ctrl+C、SSH 断开时先停机械臂再停相机）。0.5 m/s 时停车距离约 0.1 m。

## 4. 脚本与参数

### 4.1 文件

| 文件 | 说明 |
|---|---|
| `scripts/run_session.py` | 总控脚本，**所有可改参数都在文件开头**。子命令：`set-start`、`plan`、`goto-start`、`speed-test`、`sync-test`、`movec-test`、`record <方向>`；选项 `--scale`、`--yes`、`--tag` |
| `scripts/fr5_motion.py` | 运动规划与执行：平移路径、绕光心的单轴旋转、同步摆动，逆解、关节速度和玻璃窗检查，速度换算，运动中的保护 |
| `scripts/fr5_record.py` | 机械臂状态录制：独占 20004，每包记 Box 时间，停止时一次写盘 |
| `scripts/zed_capture.py` | zed_record 子进程管理（启动、正常停止、检查是否正常退出） |
| `scripts/fr5_groundtruth.py` | 离线真值：时钟拟合 → Δt → 曝光中心 → 修正（按路径组查表）→ 插值出相机真值（只依赖 numpy） |
| `scripts/gt_board_check.py` | 标定板真值验证（笔记本上运行） |
| `scripts/frame_lag_check.py` | 检查图像内容和时间戳是否错位（笔记本上运行），有异常时退出码为 1 |
| `scripts/camera_s003_refit.yaml` | ZED 校正后左目 HD1080 内参，用 s003 图像重标 |
| `scripts/test_*.py` | 离线测试（合成数据，不连机械臂），共 68 项；要 OpenCV 的只在笔记本上运行，要 FAIRINO SDK 的只在 Box 上运行，缺了自动跳过 |
| `scripts/movel_probe.py` | 诊断：发零行程的 MoveL / MoveJ，看控制器拒绝哪种参数 |
| `scripts/handeye_s003_result.json` | 标定仓库 `results/s003/result.json` 的副本（手眼外参） |
| `scripts/start_pose.json` | `set-start` 保存的水平起点：相机光心约 (597, −476, 562) mm；平移路径的中点相对它偏移 |
| `scripts/speed_calibration.json` | `speed-test` 的结果（速度换算系数和加速度） |
| `zed_record/zed_record.cpp`、`zed_record.patch` | Box 上 `~/ROBOT/zed_imu/zed_record.cpp` 的当前版本，以及相对原版的改动（可设曝光、增益和帧率，多线程存图，第 30 帧读回实际曝光和增益） |
| `docs/` | README 用的图片 |
| `sync_from_box.ps1` | 从 Box 拉取 `scripts/` 和 `zed_record/` |
| `data/` | 从 Box 拷到笔记本的录制数据（不进 git） |

### 4.2 可调参数（`run_session.py` 开头）

超出范围脚本会拒绝运行。速度和曝光的组合还必须满足“最近物体处的模糊 ≤ `MAX_BLUR_PX`”。

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `PATH_SET` | fast1 | 字母、数字、`_`、`-` | 这一套路径几何的名字，写进 `motion_log.json`，真值按它查修正量；改了路径几何就换个名字 |
| `SPEED_MPS` | 0.5 | 0.02–0.6 | 平移速度 m/s |
| `TRANSLATION_PATHS` | 见 3.4 | 偏移每个分量 ≤ 300 mm，行程 0.05–0.6 m | 每对方向的行程中点（相对起点的前、左、上 mm）和行程 m |
| `TRANSLATION_WIGGLE_SIDE` | 1 | −1 / 0 / 1 | 平移录制和 sync-test 的同步摆动：1 只往左，0 两边，−1 只往右 |
| `YAW_DEG` / `PITCH_DEG` / `ROLL_DEG` | 30 / 30 / 90 | 5–35 / 5–30 / 5–90 | 三种旋转的总转角（以中心姿态为中心两边各一半） |
| `ROT_DEG_S` | 50 | 1–60 | 旋转角速度 °/s |
| `YAW_HEADING_DEG` / `PITCH_HEADING_DEG` / `ROLL_HEADING_DEG` | 7.75 / −5 / −2 | −10–20 / −10–0 / −10–10 | 三种旋转的中心朝向：起点绕竖直轴转的角度，正为向左 |
| `GLASS_REGIONS` | 右窗 | | 画面里不能出现的玻璃：每块是（方位角起、止、仰角下、上），从起点光心看、相对起点水平朝向 |
| `GLASS_MARGIN_DEG` | 1.0 | 0.5–5 | 画面离玻璃的最小角距离 |
| `WINDOW_DISTANCE_M` | 7 | 3–50 | 玻璃离起点光心的最近距离，平移时按它算视差 |
| `CAMERA_FPS` | 30 | 15 / 30 / 60 | 相机帧率，传给 zed_record |
| `EXPOSURE_US` | 2500 | 1000–15000 | 曝光，传给 zed_record |
| `ANALOG_GAIN_MDB` | 13000 | 1000–16000 | 模拟增益，传给 zed_record |
| `MAX_BLUR_PX` / `NEAREST_DEPTH_M` | 2.0 / 1.0 | 0.3–2.0 / 0.3–20 | 允许的运动模糊（1080p 像素），以及画面里最近物体的距离 |
| `HOLD_START_S` / `HOLD_MID_S` / `HOLD_END_S` | 5 / 3 / 5 | 3–30 / 2–30 / 3–30 | 开头、中间、结尾的静止时长 |
| `WIGGLE_DEG` / `WIGGLE_DEG_S` / `WIGGLE_CYCLES` | 10 / 20 / 2 | 5–15 / 12–40 / 1–4 | 同步摆动的幅度、角速度和来回次数 |
| `SYNC_TEST_HOLD_S` | 10 | 5–60 | sync-test 两次摆动之间的静止时长 |
| `REPOSITION_MPS` | 0.05 | 0.01–0.10 | 录制前移到起点的速度 |
| `WEBAPP_SPEED_PCT` | 60 | 1–100 | WebApp 里的速度百分比，照抄过来（SDK 读不到）；改了要重跑 speed-test |
| `ACCEL_PCT` | 100 | 1–100 | MoveL 加速度百分比；改了要重跑 speed-test |
| `SPEED_TEST_PCT` | 5 | 1–10 | speed-test 去程的速度百分比（speed-test 走上下那条路径的全程） |
| `MAX_SPEED_ERROR` / `MAX_SPEED_RIPPLE` | 0.10 / 0.05 | 0.02–0.3 / 0.01–0.2 | 平移实测速度偏差、匀速段波动超过它就警告 |
| `MIN_CAMERA_Z_M` | 0.35 | 0–1.5 | 相机光心最低高度（基坐标系 z） |
| `JOINT_MIN_DEG` / `JOINT_MAX_DEG` | 见文件 | | 允许的关节范围：J2 ≤ 34°，J4、J5 限制在已安全走过的范围，J6 为起点 ±50° |
| `MAX_JOINT_STEP_DEG` | 5 | 1–15 | 路径上每 10 mm / 2° 允许的最大关节变化，超过说明逆解可能跳支 |
| `MAX_JOINT_SPEED_DEG_S` | 150 | 30–170 | 录制段关节速度上限 °/s |
| `MIN_FREE_GB` | 5 | 1–100 | 开始录制前 Box 至少剩余的磁盘空间 |

`fr5_groundtruth.py` 开头的修正表：`TIME_CORRECTION_MS`、`TRAVEL_SCALE`、`ROTATION_SCALE`（后两个按路径组分开，3.1）。命令行选项 `--time-correction-ms`、`--scale-factor`、`--rotation-scale-factor` 可以手动指定。

## 5. 使用流程

### 5.1 安装（Box）

1. **SDK**：把第 2 节那个提交的 `linux/fairino/Robot.py` 放到 `~/.local/lib/python3.10/site-packages/fairino/`，再建一个空的 `__init__.py`，用 `sha256sum` 核对（`b0dbb2e4…103e`）。
2. **zed_record**：源码在 `~/ROBOT/zed_imu/`，打上 `zed_record/zed_record.patch` 后编译：
   ```bash
   cd ~/ROBOT/zed_imu/build && make -j4 zed_record
   ```
3. **脚本**：把 `scripts/` 里的文件放到 Box 的 `~/ROBOT/fr5_record/`。
4. **WebApp**：在 Box 上用 Falkon 打开（`falkon http://192.168.58.2/ &`），自动模式速度百分比设为 60%，和 `WEBAPP_SPEED_PCT` 一致。

### 5.2 录制步骤

在 Box 自己的终端里运行，人站在机械臂旁边，手边有急停。每次运动前脚本都会列出计划，输入 `yes` 才会动。

1. **速度标定**：改了 WebApp 速度百分比后必须先跑。先低速，再全速：
   ```bash
   cd ~/ROBOT/fr5_record
   python3 run_session.py speed-test --scale 0.2
   python3 run_session.py speed-test
   ```
   结果里有加速度和每条平移路径的匀速段长度。
2. **空跑检查**：`python3 run_session.py plan`（不动），检查全部路径的关节范围、关节速度和右窗。
3. **分档演练**：每条录制先降速跑，旁边有人盯着腕部、线缆和周围障碍物：
   ```bash
   python3 run_session.py record backward --scale 0.2
   python3 run_session.py record backward --scale 0.5
   ```
   演练的数据不用。移到起点那一段检查不过时，先录同一对的另一个方向（它的终点就是这一条的起点）。
4. **时间同步和标定板验证**：`sync-test`，以及 5.3 的标定板录制。
5. **正式录制**：
   ```bash
   python3 run_session.py record backward        # 平移：forward / backward / right / left / up / down
   python3 run_session.py record yaw_left        # 旋转：yaw_left / yaw_right / pitch_up / pitch_down / roll_left / roll_right
   ```
6. **录完的检查**（拷到笔记本后，在 `scripts/` 下用标定仓库的 venv 运行）：
   ```bash
   python -B frame_lag_check.py <save_data 目录> ...
   ```
   报“晚约 1 帧”的重录；还要确认画面里没有人走动、没有右窗。

- **每条录制的结构**：静止 5 s → 同步摆动 → 静止 3 s → 平移或旋转 → 静止 3 s → 同步摆动 → 静止 5 s，约 25 s；30 fps 时每条约 0.6 GB。结束后自动计算真值。
- **录制失败**：zed_record 没有正常结束时，最后打印“录制失败”，退出码为 1，这条要重录。
- `--tag board`：给录制目录名加后缀，用来区分验证录制和正式录制。

### 5.3 标定板真值验证

1. 标定板竖直放在起点相机正前方约 1.1 m 处，中心大致和相机同高，固定牢；短暂打开 ZED Explorer 确认它在画面中间附近，然后关掉。
2. 录验证录制（平移和偏航、横滚的反方向走同一条路径，录一个方向；俯仰正反都录）：
   ```bash
   python3 run_session.py record right --tag board
   python3 run_session.py record up --tag board
   python3 run_session.py record forward --tag board
   python3 run_session.py record yaw_left --tag board
   python3 run_session.py record pitch_up --tag board
   python3 run_session.py record pitch_down --tag board
   python3 run_session.py record roll_right --tag board
   ```
3. 拷到笔记本的 `data/board_check/<录制名>/`（`save_data/right/` 不用拷），在 `scripts/` 下运行：
   ```bash
   python -B frame_lag_check.py ..\data\board_check\<录制名>\save_data
   python -B gt_board_check.py ..\data\board_check\<录制名>\save_data
   ```

### 5.4 注意事项

- **高速运动的安全**：WebApp 60% 时所有运动都比低速数据快；0.5 m/s 时停车距离约 0.1 m。新起点附近先目测有没有障碍物；上下那条路径比原起点偏左 300 mm（光心 0.47–0.91 m 高），下端肘部比底座安装面低约 90 mm，第一次走时盯住肘部和机柜边缘。
- **20004 只能有一个连接**：录制时不要另开 `Robot.RPC(...)`，运动脚本必须用 `StateRecorder.open_rpc()`。
- **录制进程里不要做长时间的 Python 计算**；**不要调用 SDK 的 `LoggerInit()`**（会持续写日志）。
- **相机同一时间只能被一个程序打开**：录制前关掉 ZED Explorer、Depth Viewer 等。
- **Box 的 WiFi 会断**：运动和录制在 Box 本机终端运行。拷大量数据时先在 Box 上打 tar 包，再限速拷（如 `scp -l 24000`），拷完核对 sha256。
- **开机预热**：Box 重启后预热 30 分钟以上再录真值数据。
- **错误码**：SDK 返回的正数错误码来自控制器，查 FAIRINO 官方“错误码对照表”。

## 6. 输出数据

每条录制一个目录（Box 上）：

```text
~/ROBOT/sessions/YYYYmmdd_HHMMSS_<名称>[_<tag>]/
├─ zed_record.log            zed_record 的输出（结束时一次写入）
├─ run_session.log           仅在 SSH 断开 / 被终止时才有
└─ save_data/
   ├─ left/  right/          <ZED 时间戳 ms>.jpg（校正后，JPEG）
   ├─ img_timeline.csv       每帧：图像时间戳、对应 IMU 时间戳、文件名（按时间顺序）
   ├─ zed_imu.csv            机身 IMU 全部样本（约 200 Hz），含 image_sync_trigger
   ├─ time_meta.txt          相机设置（帧率、曝光、增益），含第 30 帧读回的实际曝光和增益
   ├─ fr5_state.csv          机械臂状态（约 100 Hz）：Box 时间、控制器时间、关节、法兰位姿、错误码等
   ├─ fr5_meta.json          丢包、断线、时钟跳变等
   ├─ motion_log.json        参数（含 PATH_SET、TRANSLATION_PATHS）、每段指令和实测速度、预计关节速度；
   │                         旋转录制还有转轴、转动中心和光心偏离；录制失败时 error 写明原因
   ├─ groundtruth_tum.txt    每张图一行：timestamp tx ty tz qx qy qz qw（Unix 秒、米），左目在 FR5 基坐标系下（已加修正）
   ├─ groundtruth_tum_uncorrected.txt  同上，不加修正（对照用）
   ├─ groundtruth_arm_rate_tum.txt   机械臂每个状态包一行，格式同上（已加修正）
   ├─ groundtruth_images.csv         每张图的真值和标记（ok / duplicate / outside / gap / robot_error）
   └─ groundtruth_report.json        时钟拟合、Δt、前后段差、曝光中心、各项修正（含路径组）、警告
```

笔记本上的副本放在 `data/` 下（不进 git）：高速正式录制计划放在 `data/recordings_fast/`，标定板验证录制在 `data/board_check/<录制名>/`。

真值可以单独重算：

```bash
python3 fr5_groundtruth.py <save_data 目录> --handeye handeye_s003_result.json
```

两个检查脚本在 `save_data` 目录里输出：`gt_board_check.py` 写 `board_check_report.json` 和 `board_check_frames.csv`；`frame_lag_check.py` 写 `frame_lag_report.json`。
