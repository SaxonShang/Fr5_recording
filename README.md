# FR5 + ZED X 数据录制（SIO_SLAM 机械臂真值）

> 脚本的正本在 Box 的 `~/ROBOT/fr5_record/`，本仓库是笔记本上的备份，用 `.\sync_from_box.ps1` 从 Box 拉取。
> 实验过程、工程细节、各次验证的原始结果和问题处理见 [实验记录.md](实验记录.md)。

## 1. 目标

用 FAIRINO FR5 机械臂带着 ZED X 做已知运动，同时录制双目图像、ZED 机身 IMU 和机械臂状态，离线算出逐帧的相机位姿真值，供 [SIO_SLAM](https://github.com/lihongliang0203/SIO_SLAM) 评估。

已完成两批正式录制（数据位置和使用说明见第 6 节）：

| 批次 | 运动 | 条数 |
|---|---|---|
| 平移（2026-10-12） | 相机水平朝前，沿一个方向平移 300 mm，0.05 m/s：前进、后退、右移、左移、上升、下降 | 6 |
| 旋转（2026-10-13） | 绕左目光心单轴旋转，5 °/s：偏航 10°、俯仰 30°、横滚 30°，正反方向各一条 | 6 |

- **相机真值**：`base_T_camera(t) = base_T_flange(t) · flange_T_left_camera`。手眼外参来自标定仓库[fr5-zedx-handeye-calibration](https://github.com/SaxonShang/fr5-zedx-handeye-calibration)**** s003 的结果（`scripts/handeye_s003_result.json`），前提是相机安装之后没有动过。
- 正式录制前验证了时间同步和真值（用固定的标定板，只在验证录制中出现）；画面里没有人走动，也没有拍到玻璃窗。

## 2. 硬件、网络与软件版本

| 项目 | 值 |
|---|---|
| 机械臂 | FAIRINO FR5，型号 `FR5-V1-002(V6.0)`；WebApp `v3.9.1`；控制器软件 `V3.9.5-QX`；WebApp 自动模式速度百分比 10% |
| 相机 | ZED X（S/N 49953548），ZED SDK 5.1.2，HD1080 / 15 fps，全局快门；录校正后的左右目 JPEG，关闭自标定；手动曝光 10 ms，模拟增益 1 dB |
| 计算单元 | ZED Box Mini（Jetson，L4T R36.4.4，Ubuntu 22.04，Python 3.10），自带显示器和鼠标 |
| 网络 | Box 有线 `192.168.58.11/24` 用网线直连 FR5 控制器 `192.168.58.2`（固定 IP，无网关）；Box WiFi `10.255.250.132`（DHCP，可能会变），笔记本通过它 SSH：`ssh user@10.255.250.132` |
| 控制器端口 | 20003 XML-RPC 指令；20004 实时状态推送（同一时间只服务一个连接）；80 / 9999 WebApp |
| FAIRINO Python SDK | V2.2.1（对应机器人 V3.9.1）：官方仓库 `FAIR-INNOVATION/fairino-python-sdk` 提交 `060c68bb85f05d0097cc5efa360ddc61d6bb72e0` 的 `linux/fairino/Robot.py`，只依赖标准库 |
| 笔记本 | 标定板验证和图像检查用标定仓库 `D:\天府绛溪实验室\calibration` 的 `.venv`（OpenCV 4.14，带 aruco）；Box 上的 Python 导入不了 OpenCV |

## 3. 算法

### 3.1 相机真值

机械臂以约 100 Hz 推送法兰位姿。每张图像在其曝光中心时刻插值法兰位姿（位置线性插值，姿态 SLERP），再乘手眼外参，得到左目相机在 FR5 基坐标系下的位姿。超出机械臂数据范围、或两侧状态包间隔超过 30 ms 的帧不给真值；带机械臂错误码的帧做标记；时间戳重复的帧只保留第一张。

真值里加了由标定板验证（3.3）得到的修正：

- **时间**：取真值的时刻再晚 4.27 ms（`TIME_CORRECTION_MS`，10 ms 曝光下测得）。
- **行程**（平移录制）：法兰平移沿运动方向、相对路径起点放大（`TRAVEL_SCALE`）：左右、上下 1.0070，前后 1.0015。
- **转角**（俯仰录制）：法兰姿态相对起点、绕转轴的转角放大，绕左目光心转（`ROTATION_SCALE`）：向上 1.0043，向下 1.0052。

同时输出一份不加这些修正的真值，用于对照。

### 3.2 时间同步

图像和 IMU 时间戳本来就是 Box 系统时间；机械臂状态包带的是控制器时间，没有同步，漂移约 200 ppm。所有数据统一换算到 Box 系统时间：

1. **状态包接收**：录制程序独占 20004 连接，每包到达时记下 Box 时间。
2. **控制器时钟换算**：每 2 s 取延迟最小的包，直线拟合出偏差和漂移，每条录制单独拟合。
3. **剩余时间差 Δt**：ZED 陀螺仪的角速度大小和法兰姿态差分出的角速度大小做互相关。每条录制开头和结尾各有一段同步摆动（绕光心转 10°，20 °/s，来回 2 次）提供转动；前 40% 和后 40% 分别估计、互相核对。
4. **曝光中心**：ZED 把曝光期间采到的 IMU 样本标成 `image_sync_trigger=1`，由此估出曝光中心在图像时间戳之前约 3.4 ms。真值在曝光中心取值，输出仍用图像原来的时间戳。
5. **图像和时间戳是否对齐**（`frame_lag_check.py`）：同步摆动时相机绕光心转，相邻两帧的特征点按纯旋转拟合出转角，得到图像看到的角速度，和陀螺仪比较，求出图像内容相对时间戳的延迟。不需要标定板和机械臂数据。

**结果**

- 同步自洽：全速 sync-test 4 次，每次前后段 Δt 相差 ≤ 1.2 ms，跨次的总换算残差 ≤ 0.24 ms；正式录制前后段相差 ≤ 0.8 ms。
- 标定板检验（3.3）：加时间修正后，平移录制残差 −0.30～+0.17 ms；旋转录制随转动方向不同，偏航 −1.3、俯仰 +1.6～+2.4、横滚 −0.2 ms。
- 图像对陀螺仪：按原始时间戳 −0.3～+0.5 ms（21 条正常录制），所以做 VIO 时图像和 IMU 直接用原始时间戳即可。ZED 偶尔会整条录制的图像比时间戳晚一帧（+66 ms，出现过 1 次），这项检查能发现。
- 时间误差的量级：1 ms 在 50 mm/s 平移时约 0.05 mm，在 20 °/s 转动时约 0.02°。

### 3.3 真值验证（标定板）

标定板（Kalibr Aprilgrid，6×6 个 tag，外框实测 412.5 mm）固定在相机前方约 1.1 m 处。每张左目图用 PnP 求出 `camera_T_board`（内参用 s003 图像重标的 `scripts/camera_s003_refit.yaml`），再乘真值得到 `base_T_board`。真值准确的话，`base_T_board` 应该始终不变。检查四项：

- **重复性**：同一位姿的不同静止段之间的差；
- **空间误差**：运动前后两个静止段之间的差，分解到起点相机系（x 右、y 下、z 前）；
- **转角对照**（旋转录制）：真值给出的转角，和标定板看到的转角比较；
- **时间扫描**：把真值时间整体平移 δ，找使同步摆动帧最一致的 δ（和陀螺仪互相关相互独立）。

尺度另用两种和标定板尺寸无关的方法复核：双目三角化（以出厂基线为尺度），以及用 s003 标定图像重标焦距。

**结果**（加修正后；修正前的数据和排查过程见实验记录）

| 项目 | 平移（300 mm） | 旋转 |
|---|---|---|
| 重复性（同一位姿） | 0.02–0.15 mm | 0.01–0.06 mm，0.002–0.007° |
| 时间残差 | −0.30～+0.17 ms | 偏航 −1.3，俯仰 +1.6～+2.4，横滚 −0.2 ms |
| 沿运动方向的行程 / 转角 | ≤ 0.1 mm | 偏航 −0.03%，横滚 −0.01%，俯仰 −0.05～+0.03% |
| 垂直方向 / 转轴方向 | 1.5–3.7 mm | 偏航 0.05°，横滚 0.10°，俯仰 0.13–0.14° |
| 运动前后姿态变化 | 0.05–0.11° | — |

- **时间**：真值比图像晚 4.27 ± 0.26 ms，是陀螺仪和图像之间的固定偏移，已修正。只在 10 ms 曝光下测过，换曝光要重新测。
- **行程**：机械臂实际走的比报告的长，左右、上下约 0.7%，前后约 0.15%。单目、双目、重标焦距三种方法一致，已修正。
- **转角**：俯仰时机械臂报的转角偏小，向上 0.43%、向下 0.52%（和方向有关），已修正；偏航、横滚不需要修正。手眼外参的误差不会改变转角大小，所以这项来自机械臂。
- **修不了的部分**：平移的垂直方向 1.5–3.7 mm、旋转的转轴方向 0.05–0.14°，可能来自手眼转角或机械臂，现有数据分不清，当作真值的不确定度。
- **相机内参**：用 s003 图像重标得 fx = fy = 736.23 px、cx 987.72、cy 555.14，焦距比出厂值大 0.23%。出厂双目校正有漂移：视差有约 +0.37 px 的固定偏差，左右目有约 0.4 px 的上下错位。

### 3.4 运动与速度

- **轨迹**：目标先按相机位姿规划，再除以手眼外参得到法兰目标。平移用 MoveL；旋转用 MoveC，法兰绕转轴走圆弧，光心不动。录制前用控制器逆解沿路径逐点检查：有解、关节在允许范围内、不跳到另一组逆解、相机不低于下限。
- **速度**：控制器不接受 MoveL 的物理速度（mm/s），所以用百分比速度，换算系数由 `speed-test` 实测。
- **同步摆动**：平移录制绕光心的竖直轴两边摆 ±10°；旋转录制绕本条的转轴、只往旋转范围内侧摆。
- **不拍到玻璃窗**：用旋转演练的静止帧按机械臂位姿拼出全景图，量出两侧玻璃窗的方位和仰角（`GLASS_REGIONS`）。`plan` 和 `record` 沿每条旋转路径检查画面四条边离玻璃的角距离，小于 `GLASS_MARGIN_DEG`（1°）就拒绝。

  ![实验室全景图](docs/scene_panorama.jpg)

  *全景图：横轴方位角、纵轴仰角（°），以起点光心为球心、相对起点水平朝向，向左为正。左侧 +57° 以左是左窗，右侧 −60° 以右是落地窗。*

- **关节范围**：腕部 J4、J5 限制在这套末端装置已经安全走过的范围内，J6 限制在起点 ±50°（`JOINT_MIN_DEG` / `JOINT_MAX_DEG`）。
- **运动模糊**：平移 `fx · v · T / Z`，旋转 `fx · ω · T`。0.05 m/s、曝光 10 ms 时 1 m 处 0.37 px；5 °/s 时 0.64 px。
- **保护**：运动中逐包监视法兰速度、角速度和 MoveC 的光心偏离，超过预期、状态中断或停顿时，用新的连接发 StopMotion。Ctrl+C、SSH 断开或被终止时，先停机械臂，再正常停相机、保存数据。

**结果**

- 速度换算（WebApp 10%）：1% 对应平移 1.001 mm/s、摆动 0.347 °/s。
- 正式录制实测：平移 49.9–50.0 mm/s（匀速段波动 2.4–4.2%）；旋转 5.18–5.19 °/s，光心偏离 ≤ 0.84 mm；同步摆动约 20 °/s。
- 按 Ctrl+C 后机械臂在 0.13 s 内停下。

## 4. 脚本与参数

### 4.1 文件

| 文件 | 说明 |
|---|---|
| `scripts/run_session.py` | 总控脚本，**所有可改参数都在文件开头**。子命令：`set-start`、`plan`、`goto-start`、`speed-test`、`sync-test`、`movec-test`、`record <方向>`；选项 `--scale`、`--yes`、`--tag` |
| `scripts/fr5_motion.py` | 运动规划与执行：平移、绕光心的单轴旋转、同步摆动，逆解检查和玻璃窗检查，速度换算，运动中的保护 |
| `scripts/fr5_record.py` | 机械臂状态录制：独占 20004，每包记 Box 时间，停止时一次写盘；`open_rpc()` 让 SDK 运动指令共用这个连接 |
| `scripts/zed_capture.py` | zed_record 子进程管理（启动、正常停止、检查是否正常退出） |
| `scripts/fr5_groundtruth.py` | 离线真值：时钟拟合 → Δt → 曝光中心 → 修正 → 插值出相机真值（只依赖 numpy） |
| `scripts/gt_board_check.py` | 标定板真值验证（笔记本上运行） |
| `scripts/frame_lag_check.py` | 检查图像内容和时间戳是否错位（笔记本上运行），有异常时退出码为 1 |
| `scripts/camera_s003_refit.yaml` | ZED 校正后左目 HD1080 内参，用 s003 图像重标（比出厂值准，也可给 SIO_SLAM 用） |
| `scripts/test_*.py` | 离线测试（合成数据，不连机械臂），共 63 项；要 OpenCV 的只在笔记本上运行，要 FAIRINO SDK 的只在 Box 上运行，缺了自动跳过 |
| `scripts/movel_probe.py` | 诊断：发零行程的 MoveL / MoveJ，看控制器拒绝哪种参数 |
| `scripts/handeye_s003_result.json` | 标定仓库 `results/s003/result.json` 的副本（手眼外参） |
| `scripts/start_pose.json` | `set-start` 保存的水平起点：相机光心约 (597, −476, 562) mm |
| `scripts/speed_calibration.json` | `speed-test` 的结果 |
| `zed_record/zed_record.cpp`、`zed_record.patch` | Box 上 `~/ROBOT/zed_imu/zed_record.cpp` 的当前版本，以及相对原版的改动（可设曝光和增益，并在第 30 帧读回实际值） |
| `docs/` | README 用的图片 |
| `sync_from_box.ps1` | 从 Box 拉取 `scripts/` 和 `zed_record/` |
| `data/` | 从 Box 拷到笔记本的录制数据（不进 git） |

### 4.2 可调参数（`run_session.py` 开头）

超出范围脚本会拒绝运行。速度和曝光的组合还必须满足“最近物体处的模糊 ≤ `MAX_BLUR_PX`”。

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `SPEED_MPS` | 0.05 | 0.02–0.10 | 平移速度 m/s |
| `STROKE_M` | 0.30 | 0.05–0.30 | 行程，以起点为中心两边各一半 |
| `YAW_DEG` / `PITCH_DEG` / `ROLL_DEG` | 10 / 30 / 30 | 5–30 / 5–30 / 5–90 | 偏航、俯仰、横滚录制的总转角（以中心姿态为中心两边各一半） |
| `ROT_DEG_S` | 5 | 1–10 | 旋转角速度 °/s |
| `YAW_HEADING_DEG` / `PITCH_HEADING_DEG` / `ROLL_HEADING_DEG` | −2.25 / −5 / −2 | −10–20 / −10–0 / −10–10 | 三种旋转录制的中心朝向：起点绕竖直轴转的角度，正为向左 |
| `GLASS_REGIONS` | 见文件 | | 画面里不能出现的玻璃：每块是（方位角起、止、仰角下、上），以起点光心为球心、相对起点水平朝向 |
| `GLASS_MARGIN_DEG` | 1.0 | 0.5–5 | 旋转录制全程画面离玻璃的最小角距离 |
| `EXPOSURE_US` | 10000 | 1000–15000 | 曝光，传给 zed_record |
| `ANALOG_GAIN_MDB` | 1000 | 1000–16000 | 模拟增益，传给 zed_record；缩短曝光后画面偏暗时调大 |
| `MAX_BLUR_PX` / `NEAREST_DEPTH_M` | 1.0 / 1.0 | 0.3–2.0 / 0.3–20 | 允许的运动模糊（1080p 像素），以及画面里最近物体的距离 |
| `HOLD_START_S` / `HOLD_MID_S` / `HOLD_END_S` | 5 / 3 / 5 | 3–30 / 2–30 / 3–30 | 开头、中间、结尾的静止时长 |
| `WIGGLE_DEG` / `WIGGLE_DEG_S` / `WIGGLE_CYCLES` | 10 / 20 / 2 | 5–15 / 12–40 / 1–4 | 同步摆动的幅度、角速度和来回次数 |
| `SYNC_TEST_HOLD_S` | 10 | 5–60 | sync-test 两次摆动之间的静止时长 |
| `REPOSITION_MPS` | 0.05 | 0.01–0.10 | 录制前移到起点的速度 |
| `WEBAPP_SPEED_PCT` | 10 | 1–100 | WebApp 里的速度百分比，照抄过来（SDK 读不到）；改了要重跑 speed-test |
| `ACCEL_PCT` | 100 | 1–100 | MoveL 加速度百分比；改了要重跑 speed-test |
| `SPEED_TEST_PCT` / `SPEED_TEST_MM` | 5 / 100 | 1–10 / 50–150 | speed-test 的速度百分比和竖直行程 |
| `MAX_SPEED_ERROR` / `MAX_SPEED_RIPPLE` | 0.10 / 0.05 | 0.02–0.3 / 0.01–0.2 | 平移实测速度偏差、匀速段波动超过它就警告 |
| `MIN_CAMERA_Z_M` | 0.35 | 0–1.5 | 相机光心最低高度（基坐标系 z） |
| `JOINT_MIN_DEG` / `JOINT_MAX_DEG` | 见文件 | | 允许的关节范围（3.4） |
| `MAX_JOINT_STEP_DEG` | 5 | 1–15 | 路径上每 10 mm / 2° 允许的最大关节变化，超过说明逆解可能跳支 |
| `MIN_FREE_GB` | 5 | 1–100 | 开始录制前 Box 至少剩余的磁盘空间 |

`fr5_groundtruth.py` 开头的修正参数：`TIME_CORRECTION_MS`、`TRAVEL_SCALE`、`ROTATION_SCALE`（3.1）。行程和转角系数只适用于当前起点的这些路径；时间修正只适用于 10 ms 曝光，曝光不同时报警。

命令行选项：

- `fr5_groundtruth.py`：`--max-lag-ms`（Δt 搜索范围，默认 300）；`--dt-ms`、`--exposure-offset-ms`（手动指定，不估计）；`--time-correction-ms`（0 表示不修正）、`--scale-factor`、`--rotation-scale-factor`（1 表示不放大）。
- `gt_board_check.py`：`--step`（每隔几张检测一张）、`--scan-ms`（时间扫描范围，默认 ±20 ms）、`--camera`、`--target`、`--calib-repo`。

## 5. 使用流程

### 5.1 安装（Box，一次性）

1. **SDK**：把第 2 节那个提交的 `linux/fairino/Robot.py` 放到 `~/.local/lib/python3.10/site-packages/fairino/`，再建一个空的 `__init__.py`，用 `sha256sum` 核对（`b0dbb2e4…103e`）。
2. **zed_record**：源码在 `~/ROBOT/zed_imu/`，打上 `zed_record/zed_record.patch` 后编译：
   ```bash
   cd ~/ROBOT/zed_imu/build && make -j4 zed_record
   ```
3. **脚本**：把 `scripts/` 里的文件放到 Box 的 `~/ROBOT/fr5_record/`。
4. **WebApp**：在 Box 上用 Falkon 打开（`falkon http://192.168.58.2/ &`）。自动模式的速度百分比不能是 0，并且要和 `WEBAPP_SPEED_PCT` 一致（当前为 10）。

### 5.2 录制

在 Box 自己的终端里运行，人站在机械臂旁边，手边有急停。每次运动前脚本都会列出计划，输入 `yes` 才会动。

```bash
cd ~/ROBOT/fr5_record
python3 run_session.py set-start               # 读当前位姿，算出相机水平的起点并保存（不动）
python3 run_session.py plan                    # 空跑：检查全部录制的路径（不动）；也可只列几条，如 plan yaw_left pitch_up
python3 run_session.py goto-start --scale 0.25 # 低速移到起点
python3 run_session.py speed-test              # 速度标定（不开相机）；改了 WebApp 速度百分比或 ACCEL_PCT 要重跑
python3 run_session.py sync-test               # 时间同步验证（只做同步摆动）
python3 run_session.py movec-test              # MoveC 探测（不开相机）
python3 run_session.py record forward          # 平移录制，另有 backward / right / left / up / down
python3 run_session.py record yaw_left         # 旋转录制，另有 yaw_right / pitch_up / pitch_down / roll_left / roll_right
```

- **每条录制的结构**：静止 5 s → 同步摆动 → 静止 3 s → 平移或旋转 → 静止 3 s → 同步摆动 → 静止 5 s。平移约 33 s、0.4 GB；旋转 24–28 s、0.3 GB。结束后自动计算真值。
- **录制失败**：zed_record 没有正常结束时，最后打印“录制失败”，退出码为 1，`motion_log.json` 的 `error` 写明原因，这条要重录。
- `--scale 0.25`：按比例降速，用于低速演练（这时算不出 Δt，数据不用）。第一次跑新的运动先这样演练，旁边有人盯着腕部和线缆。
- `--tag board`：给录制目录名加后缀（如 `…_right_board/`），用来区分验证录制和正式录制。
- 按 Ctrl+C 或 SSH 断开：先停机械臂，再正常停相机，保存已录的数据。
- **录完的检查**（拷到笔记本后，在 `scripts/` 下用标定仓库的 venv 运行）：
  ```bash
  python -B frame_lag_check.py <save_data 目录> ...
  ```
  报“晚约 1 帧”的重录。正式录制还要确认画面里没有人走动。

### 5.3 标定板真值验证

1. 用标定时的那块 Aprilgrid，竖直放在起点相机正前方约 1.1 m 处，中心大致和相机同高，固定牢。
2. 短暂打开 ZED Explorer，确认标定板完整地在画面中间附近，然后关掉。
3. 在 Box 上录验证录制。平移、偏航、横滚的反方向走同一段路径，只录一个方向；俯仰的误差和方向有关，正反都录：
   ```bash
   python3 run_session.py record right --tag board
   python3 run_session.py record up --tag board
   python3 run_session.py record forward --tag board
   python3 run_session.py record yaw_left --tag board
   python3 run_session.py record pitch_up --tag board
   python3 run_session.py record pitch_down --tag board
   python3 run_session.py record roll_right --tag board
   ```
4. 把每条录制拷到笔记本的 `data/board_check/<录制名>/`（`save_data/right/` 不用拷），然后在 `scripts/` 下用标定仓库的 venv 运行：
   ```bash
   python -B frame_lag_check.py ..\data\board_check\<录制名>\save_data
   python -B gt_board_check.py ..\data\board_check\<录制名>\save_data
   ```

### 5.4 注意事项

- **20004 只能有一个连接**：录制时不要另开 `Robot.RPC(...)`，运动脚本必须用 `StateRecorder.open_rpc()`。
- **录制进程里不要做长时间的 Python 计算**：真值计算等重活放在录制停止之后。
- **不要调用 SDK 的 `LoggerInit()`**：它会持续往磁盘写日志。
- **相机同一时间只能被一个程序打开**：录制前关掉 ZED Explorer、Depth Viewer 等。
- **Box 的 WiFi 会断**：运动和录制在 Box 本机终端运行（Box 没有 tmux / screen）。拷大量数据时先在 Box 上打 tar 包，再限速拷（如 `scp -l 24000`），拷完核对 sha256。
- **开机预热**：Box 重启后预热 30 分钟以上再录真值数据。
- **场景**：画面里不要有人走动，也不要拍到玻璃窗；数据如果要公开，先征得画面里的人同意。
- **错误码**：SDK 返回的正数错误码来自控制器，查 FAIRINO 官方“错误码对照表”。例如 172 表示运动速度不能为 0，182 / 183 表示物理速度 / 加速度超限。

## 6. 输出数据

每条录制一个目录（Box 上）：

```text
~/ROBOT/sessions/YYYYmmdd_HHMMSS_<名称>[_<tag>]/
├─ zed_record.log            zed_record 的输出（结束时一次写入）
├─ run_session.log           仅在 SSH 断开 / 被终止时才有
└─ save_data/
   ├─ left/  right/          <ZED 时间戳 ms>.jpg（校正后，JPEG）
   ├─ img_timeline.csv       每帧：图像时间戳、对应 IMU 时间戳、文件名
   ├─ zed_imu.csv            机身 IMU 全部样本（约 200 Hz），含 image_sync_trigger
   ├─ time_meta.txt          相机设置，含第 30 帧读回的实际曝光和增益
   ├─ fr5_state.csv          机械臂状态（约 100 Hz）：Box 时间、控制器时间、关节、法兰位姿、错误码等
   ├─ fr5_meta.json          丢包、断线、时钟跳变等
   ├─ motion_log.json        每段指令和实测速度；旋转录制还有转轴、转动中心和光心偏离；录制失败时 error 写明原因
   ├─ groundtruth_tum.txt    每张图一行：timestamp tx ty tz qx qy qz qw（Unix 秒、米），左目在 FR5 基坐标系下（已加修正）
   ├─ groundtruth_tum_uncorrected.txt  同上，不加修正（对照用）
   ├─ groundtruth_arm_rate_tum.txt   机械臂每个状态包一行，格式同上（已加修正）
   ├─ groundtruth_images.csv         每张图的真值和标记（ok / duplicate / outside / gap / robot_error）
   └─ groundtruth_report.json        时钟拟合、Δt、前后段差、曝光中心、各项修正、警告
```

笔记本上的副本放在 `data/` 下（不进 git）：

- 正式录制：`data/recordings/`（6 条平移，含左右目，2.4 GB）和 `data/recordings_rotation/`（6 条旋转，含左右目，2.0 GB），各附给数据使用者的 `使用说明.md` 和标定文件 `calib/`；
- 标定板验证录制：`data/board_check/<录制名>/`。

真值可以单独重算：

```bash
python3 fr5_groundtruth.py <save_data 目录> --handeye handeye_s003_result.json
```

两个检查脚本在 `save_data` 目录里输出：

- `gt_board_check.py`：`board_check_report.json`（重复性、静止组之间的差、转角对照、时间扫描）和 `board_check_frames.csv`（每帧的检测结果和偏差）；
- `frame_lag_check.py`：`frame_lag_report.json`（图像相对时间戳的延迟、是否正常、扫描曲线）。
