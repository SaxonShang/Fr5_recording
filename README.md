# FR5 + ZED X 验证数据集录制（holdout1）

> 用 FAIRINO FR5 机械臂带着 ZED X 走事先定好的多段路径，录双目图像、机身 IMU 和机械臂状态，离线算出逐帧的相机位姿真值，作为 [kalman-fusion-lab](https://github.com/SaxonShang/kalman-fusion-lab) 的留出评测数据。录制方案是该仓库 `实验记录.md` 的 13.11 节。
>
> 现状（2026-11-09）：脚本完成，离线检查通过，**还没有上机，没有录制数据**。
> 实验过程、工程细节和**全部操作命令**见 [实验记录.md](实验记录.md)。

## 1. 目标

用同一批录像回答三个问题：横滚能否跨方向复现；平移运动上的位置与旋转代价是否复现；退化是否随场景几何和纹理改变。评测端在模型冻结后才运行这批数据。

**要录的内容**

| 用途 | 优先版 | 完整版 |
|---|---:|---:|
| 加计辨识，两种动作各 2 条 | 4 | 4 |
| 带板核验，每种正式轨迹 2 条 | 6 | 8 |
| 正式录像，每个“轨迹 × 场景”条件 3 条 | 15 | 18 |
| **合计** | **25** | **30** |

另有开工核对 2 条，不进入评测，并预留 3–5 条重录。

**路径**（时长是按速度标定的估算，含首尾静止和同步摆动）

| 录制名 | 内容 | 段数 | 时长 |
|---|---|---:|---:|
| `roll_bidir` | 绕左目光心横滚：中立 → +50° → −50° → 中立，50 °/s，两端各停 4 s | 3 | 37 s |
| `trans_level_first`、`trans_tilt_first` | 同一段平移走两种姿态：上移、返回、前移、返回，每段 0.40 m、0.5 m/s；水平一组，俯仰 +25° 一组，两种先后顺序 | 10 | 63 s |
| `s_path` | S 形折线，法兰 0.30 m/s，每段转动不超过 50 °/s，每点停 1 s | 12 | 50 s |
| `eight` | 空间 8 字 3 圈，法兰 0.25 m/s，每段转动不超过 50 °/s，每点停 1 s（仅完整版） | 24 | 70 s |
| `accel_static_a`、`accel_static_b` | 10 个姿态每个停 7 s，20 °/s，b 的访问顺序相反 | 9 | 117 s |
| `accel_rate_slow_first`、`accel_rate_fast_first` | 俯仰、横滚、斜向的同一序列，15 °/s 和 35 °/s 各走一遍，交换快慢顺序 | 18 | 87 s |

**正式录像的条件**

| 条件名 | 路径 | 场景 |
|---|---|---|
| `roll_bidir` | `roll_bidir` | A |
| `trans_two_attitude` | 依次 `trans_level_first`、`trans_tilt_first`、`trans_level_first` | A |
| `s_path_A`、`s_path_B`、`s_path_C` | `s_path`（`--tag sceneA/B/C`） | A / B / C |
| `eight` | `eight`（仅完整版） | A |

场景 A 是实验室的近远层次场景；B 是约 3 m × 2 m 的非重复高纹理硬板，距 S 点光心约 1.0 m；C 位置和尺寸同 B，换成周期约 3 cm 的二维重复图案。

## 2. 硬件、网络与软件版本

| 项目 | 值 |
|---|---|
| 机械臂 | FAIRINO FR5，型号 `FR5-V1-002(V6.0)`；WebApp `v3.9.1`；控制器软件 `V3.9.5-QX`；WebApp 自动模式速度百分比 60%。底座在 2026-10-16 之后搬到了实验室的另一位置 |
| 相机 | ZED X（S/N 49953548），ZED SDK 5.1.2，HD1080，30 fps，全局快门；手动曝光 2.5 ms，模拟增益 13 dB；录校正后的左右目 JPEG，关闭自标定 |
| 相机录制程序 | `zed_record`（本仓库的补丁）：可设曝光、增益和帧率，3 个线程并行存图 |
| 手眼外参 | `handeye_s003_result.json`，法兰到左目，sha256 以 `a8212d48` 开头；相机没有拆过，由开工核对确认仍然有效 |
| 计算单元 | ZED Box Mini（Jetson，L4T R36.4.4，Ubuntu 22.04，Python 3.10），自带显示器和鼠标 |
| 网络 | Box 有线 `192.168.58.11/24` 直连 FR5 控制器 `192.168.58.2`；Box WiFi `10.255.250.132`（DHCP，可能会变），笔记本通过它 SSH |
| 控制器端口 | 20003 XML-RPC 指令；20004 实时状态推送（同一时间只服务一个连接）；80 / 9999 WebApp |
| FAIRINO Python SDK | V2.2.1：官方仓库 `FAIR-INNOVATION/fairino-python-sdk` 提交 `060c68bb85f05d0097cc5efa360ddc61d6bb72e0` 的 `linux/fairino/Robot.py` |
| 笔记本 | 带板检查和图像检查用标定仓库 `D:\天府绛溪实验室\calibration` 的 `.venv`（OpenCV 带 aruco）；Box 上的 Python 导入不了 OpenCV |

## 3. 算法

### 3.1 相机真值

每张图像在曝光中心时刻插值法兰位姿，再乘手眼外参：`base_T_camera(t) = base_T_flange(t) · flange_T_left_camera`，得到左目相机在 FR5 基坐标系下的位姿。2.5 ms 曝光下取真值的时刻再晚 3.75 ms，带板验收的时间残差检验这个值。

### 3.2 路径点修正

机械臂报告的位姿在不同路径点上有几毫米、零点几度的偏差，而且和路径有关。标定板固定时，不加修正的真值乘 PnP 得到的 `B_k` 在每个路径点上应该相同，所以路径点 k 的修正是 `ΔT_k = B_0 B_k⁻¹`（基坐标系左乘，`ΔT_0 = I`）。第 k 段内按进度 s 插值：

```text
exp(s · log(ΔT_{k+1} ΔT_k⁻¹)) · ΔT_k
```

MoveL 和组合段的 s 取法兰在弦上的投影，MoveC 取绕转轴转过的比例。每次到访单独一个 ΔT，往返经过同一位置也按来向分开。修正由第一条带板录像拟合，第二条独立验收，写在 `waypoint_corrections.json`。加计路径不做这项修正。

### 3.3 时间同步

- **控制器时钟到 Box 时间**：按延迟最小的状态包拟合直线。
- **Δt**：陀螺仪和法兰角速度互相关，只用开头和结尾两段同步摆动，主体转动会把结果带偏。两段各估一次，核对漂移。
- **曝光中心**：由 `image_sync_trigger` 估计。
- **图像错位检查**：`frame_lag_check.py` 对比图像和陀螺仪的时序，整帧错位的录像要重录。默认用光流；重复纹理的场景 C 用帧差法（`--method framediff`），它的正常结果约 −7～−2 ms，晚一帧约 +29 ms。

录制期间关闭 Box 的自动对时。1 ms 的时间误差在 0.5 m/s 时约 0.5 mm，在 50 °/s 时约 0.05°。

### 3.4 运动

每条录制的结构：移到起点（不录）→ 静止 5 s → 同步摆动 → 静止 3 s → 主体路径 → 静止 3 s → 同步摆动 → 静止 5 s。多姿态停驻的主体前后各静止 7 s。

- **同步摆动**：10°、20 °/s、来回 2 次，只往一侧摆。绕相机竖直轴，横滚绕光轴。哪一侧由 `plan` 选。
- **段**：每段是一条 MoveL 或 MoveC，`blendR=0`，到位后停住，最短停 1 s。
- **组合段速度**：同时平移和转动的段，用时 `T = max(L/v, θ/ω)`，ω = 50 °/s，速度百分比 `p = max(L/(5.96·T), θ/(2.152·T))`。L 是法兰行程（mm），θ 是转角（°）。短段不追目标线速度。
- **路径检查**：`plan` 用控制器逆解沿路径采样，只报通过或不通过：有解、关节在范围内、关节速度 ≤ 150 °/s、离奇异 ≥ 5°、构型不变、不拍右窗。多段路径不自动收缩，不通过就人工改路径表。
- **运动中保护**：超速（组合段线速度 1.5 倍 + 5 mm/s，角速度 2 倍 + 2 °/s）、MoveC 光心偏离、状态中断、停顿时发 StopMotion；Ctrl+C 或 SSH 断开时先停机械臂再停相机。

### 3.5 精度

**数据录完后在这里写实测结果。** 每种正式轨迹的第二条带板录像必须满足下面的门槛，否则不进入正式录制：

| 项目 | 门槛 |
|---|---|
| 路径点静止帧的标定板位置差（相对第 0 组） | ≤ 1.0 mm、≤ 0.10° |
| 运动帧的标定板偏离 | RMS ≤ 2.5 mm，最大 ≤ 5.0 mm |
| 运动帧的姿态偏离 RMS | ≤ 0.20° |
| 时间残差 | ≤ 1.5 ms |

离线检查（解析逆解，实际值以上机 `plan` 为准）：

| 路径 | 关节最快 °/s |
|---|---:|
| `roll_bidir` | 51 |
| `trans_level_first`、`trans_tilt_first` | 116 |
| `s_path` | 63 |
| `eight` | 82 |
| `accel_static_a`、`accel_static_b` | 44 |
| `accel_rate_slow_first`、`accel_rate_fast_first` | 77 |

## 4. 脚本与参数

### 4.1 文件

| 文件 | 说明 |
|---|---|
| `scripts/holdout1_paths.py` | 路径表：参考位置、路径点、段的速度和停留。**改路径只改这里** |
| `scripts/run_session.py` | 总控，其他可改参数都在文件开头。子命令 `plan`、`speed-test`、`record`、`warmup`；选项 `--scale`、`--yes`、`--tag` |
| `scripts/fr5_motion.py` | 运动规划、逆解检查、速度换算和逐段执行 |
| `scripts/fr5_limits.py` | 开工核对用：由拖动标记定 `right`、`roll_right` 的路径 |
| `scripts/fr5_record.py`、`scripts/zed_capture.py` | 机械臂状态录制（独占 20004）、`zed_record` 子进程管理 |
| `scripts/fr5_groundtruth.py` | 离线真值，包括路径点修正 |
| `scripts/end_pose_fit.py` | 由带板录像拟合路径点修正，留一检验，`--write` 写修正表（笔记本） |
| `scripts/gt_board_check.py` | 带板验证，多段路径打印验收结果（笔记本） |
| `scripts/frame_lag_check.py` | 图像和时间戳是否错位，光流法或帧差法（笔记本） |
| `scripts/startup_checks.py` | 开工清单的数值检查：相机连续录制、重力方向、到位稳定、每段实测 |
| `scripts/make_conditions.py` | 生成交付用的 `conditions.yaml` |
| `scripts/handeye_poses.py`、`scripts/handeye_capture.py` | 开工核对不通过、紧固支架后重标手眼时用 |
| `scripts/zed_imu_extrinsic.py` | 读 ZED 出厂的左目–IMU 外参和 IMU 噪声参数 |
| `scripts/fr5_kin.py` | FR5 解析运动学，只用于离线测试 |
| `scripts/test_*.py` | 离线测试（不连机械臂）；要 OpenCV 的只在笔记本运行，要 FAIRINO SDK 的只在 Box 运行，缺了自动跳过 |
| `scripts/handeye_s003_result.json`、`camera_s003_refit.yaml`、`camera_imu.yaml` | 手眼外参、校正后左目 HD1080 内参、左目–IMU 外参 |
| `scripts/start_pose.json` | 开工核对路径用的参考起点 |
| `scripts/waypoint_corrections.json` | 路径点修正表，`end_pose_fit.py --write` 生成 |
| `zed_record/` | Box 上 `zed_record.cpp` 的当前版本和补丁 |
| `deploy_to_box.ps1`、`sync_from_box.ps1` | 把脚本部署到 Box 的 `~/ROBOT/fr5_verify`；把现场改过的脚本拉回来 |
| `data/` | 录制数据（不进 git） |

### 4.2 可调参数（`run_session.py` 开头）

| 参数 | 值 | 说明 |
|---|---|---|
| `HOLDOUT_WINDOW` | `None` | 底座搬动后的右窗。`None` 时 `plan` 和 `record` 拒绝 holdout1 路径；看不到窗或已遮住时 `regions` 写空表 |
| `CAMERA_FPS`、`EXPOSURE_US`、`ANALOG_GAIN_MDB` | 30、2500、13000 | 相机设置 |
| `WIGGLE_DEG`、`WIGGLE_DEG_S`、`WIGGLE_CYCLES` | 10、20、2 | 同步摆动 |
| `HOLD_START_S`、`HOLD_MID_S`、`HOLD_END_S` | 5、3、5 | 首尾和中间的静止 |
| `WEBAPP_SPEED_PCT` | 60 | WebApp 速度百分比，改了要重跑 `speed-test` |
| `SINGULAR_MARGIN_DEG`、`MAX_JOINT_SPEED_DEG_S` | 5、150 | 离奇异位形、关节速度的限制 |
| `DEV_JOINT_MIN_DEG`、`DEV_JOINT_MAX_DEG` | 见文件 | 开发录制走过的关节范围，超出时只提示 |
| `PATH_SET`、`SPEED_MPS`、`ROT_DEG_S`、`GLASS_REGIONS`、`LIMIT_*` | fast2 的值 | 开工核对两条的路径参数，不要改 |

## 5. 使用流程

每一步的完整命令和通过标准在 [实验记录.md 的“运行手册”](实验记录.md#运行手册)。

1. **部署**：`deploy_to_box.ps1` 把脚本放到 Box 的 `~/ROBOT/fr5_verify`，录制数据写在 `~/ROBOT/sessions_verify`。
2. **开工核对**：确认时钟、原有路径、周边和线缆、手眼外参、重力方向、到位稳定时间和相机连续录制。手眼核对把标定板放在基坐标 (1.65, −0.02, 0.56) m，朝 −X。
3. **新路径准备**：量右窗，`plan` 空跑，0.2、0.5、1.0 倍速演练，试录场景 B、C。
4. **带板核验**：每种轨迹录 2 条，第一条拟合修正，第二条验收。
5. **正式录制**：移走标定板，按路径录。S 形按轮次录，轮内场景顺序依次为 ABC、BCA、CAB。
6. **录完检查和交付**：每条都查整帧错位，生成 `conditions.yaml`。

**同一天的规则**

- 开录前机械臂运转 30–60 分钟（`warmup`）。
- 同一路径的带板和正式录像在同一天完成，带板到正式录制不超过约 2 小时。
- 当天结束再录一条带板看漂移；换天先补 2 条带板并重新验收。
- 从带板到正式录完，标记、速度标定、WebApp 百分比和 `run_session.py` 参数都不能动。

**注意事项**

- 在 Box 自己的终端里运行，手边有急停。每次运动前脚本列出计划，输入 `yes` 才会动。
- 录制期间关闭 Box 自动对时。关着对时的时候重启，系统时间会回到 1970，要先打开对时，等 synchronized 后再关。
- 20004 只能有一个连接，相机同一时间只能被一个程序打开。
- 大批数据用 USB SSD 拷，WiFi 拷 40 GB 要 6–7 小时。

## 6. 输出数据

每条录制一个目录（Box 上）：

```text
~/ROBOT/sessions_verify/YYYYmmdd_HHMMSS_<名称>[_<tag>]/
├─ zed_record.log            zed_record 的输出
├─ run_session.log           仅在 SSH 断开或被终止时才有
└─ save_data/
   ├─ left/  right/          <ZED 时间戳 ms>.jpg（校正后）
   ├─ img_timeline.csv       每帧的图像时间戳、IMU 时间戳、文件名
   ├─ zed_imu.csv            机身 IMU 全部样本（约 200 Hz），含 image_sync_trigger
   ├─ time_meta.txt          相机设置和读回的实际曝光、增益
   ├─ fr5_state.csv          机械臂状态（约 100 Hz）
   ├─ fr5_meta.json          丢包、断线、时钟跳变
   ├─ motion_log.json        参数、路径表记录、每段指令和实测
   ├─ groundtruth_tum.txt    每张图一行：timestamp tx ty tz qx qy qz qw（Unix 秒、米），左目在基坐标系下，已修正
   ├─ groundtruth_tum_uncorrected.txt  同上，不加修正
   ├─ groundtruth_arm_rate_tum.txt     机械臂每个状态包一行，已修正
   ├─ groundtruth_images.csv 每张图的真值和标记
   ├─ groundtruth_report.json 时钟拟合、Δt、曝光中心、路径点修正、警告
   └─ frame_lag_report.json  图像错位检查（笔记本上生成）
```

**交付给评测端**

| 项目 | 位置 |
|---|---|
| 正式录像 | 笔记本 `data/recordings_holdout1/`，评测机 `data/recordings/recordings_holdout1/`，含 `calib/` 和 `conditions.yaml` |
| 带板和加计录像 | `data/board_check/`，不放进批次目录 |
| `calib/` | `camera_s003_refit.yaml`、`camera_imu.yaml`、`SN49953548.conf` |
| 硬性检查 | `frame_lag_report.json` 中 `ok=true`、`frames_late=0`、`at_scan_edge=false`；场景 C 多一个 `method: framediff` |
