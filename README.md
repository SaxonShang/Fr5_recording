# FR5 + ZED X 高速数据录制（SIO_SLAM 机械臂真值，high-speed 分支）

> 本分支在 main 的基础上把运动提高到 0.5 m/s、50 °/s。录制方式、真值算法和标定板验证的思路与 main 相同，本文着重写不同之处和结果。低速数据（2026-10-12/13）见 main 分支或标签 `data-v1`。
> 脚本的正本在 Box 的 `~/ROBOT/fr5_record/`，本仓库是笔记本上的备份，用 `.\sync_from_box.ps1` 从 Box 拉取。
> 实验过程、工程细节、各次验证的原始结果和问题处理见 [实验记录.md](实验记录.md)。

## 1. 目标

用 FAIRINO FR5 机械臂带着 ZED X 做已知的高速运动，同时录制双目图像、ZED 机身 IMU 和机械臂状态，离线算出逐帧的相机位姿真值，供 [SIO_SLAM](https://github.com/lihongliang0203/SIO_SLAM) 评估快速运动下的表现。

2026-10-15 完成 6 条正式录制（数据位置见第 6 节）：

| 录制 | 运动 | 范围 | 实测速度 | 图像 |
|---|---|---|---|---|
| `up` | 上升（最下到最上），相机水平朝前 | 737.6 mm | 503.9 mm/s | 697 对 |
| `right` | 右移（最左到最右） | 758.1 mm | 503.8 mm/s | 697 对 |
| `forward` | 前进（最后到最前） | 515.6 mm | 502.8 mm/s | 683 对 |
| `yaw_left` | 绕左目光心偏航（右到左） | 54.5° | 50.1 °/s | 680 对 |
| `pitch_up` | 绕左目光心俯仰（下到上） | 68.0° | 50.1 °/s | 687 对 |
| `roll_right` | 绕左目光心横滚（左到右） | 162.1° | 50.1 °/s | 742 对 |

**和 main（低速数据）的不同**

| | main | 本分支 |
|---|---|---|
| 速度 | 平移 0.05 m/s，旋转 5 °/s | 平移 0.5 m/s，旋转 50 °/s |
| 相机 | 15 fps，曝光 10 ms，增益 1 dB | 30 fps，曝光 2.5 ms，增益 13 dB |
| 运动范围 | 固定起点；平移 300 mm，旋转 10–30°；正反方向各一条 | 每个方向一条，走满能走的范围：由人工拖出的一对极限位姿（拖动标记）定，不回起点 |
| 场景 | 不拍任何玻璃窗 | 只避开右侧落地窗，左窗可以拍 |
| 真值修正 | 低速标定板测得 | 高速标定板重新测得（3.1） |

- **相机真值**：`base_T_camera(t) = base_T_flange(t) · flange_T_left_camera`，和 main 相同。手眼外参来自标定仓库 [fr5-zedx-handeye-calibration](https://github.com/SaxonShang/fr5-zedx-handeye-calibration) s003 的结果（`scripts/handeye_s003_result.json`），前提是相机安装之后没有动过。
- 用固定的标定板验证了时间和真值（13 条验证录制，路径和正式录制完全相同，8 条在正式录制之前、5 条在之后）；正式录制的画面里没有人走动，没有拍到右窗。

## 2. 硬件、网络与软件版本

和 main 相同，不同的是加粗的几项：

| 项目 | 值 |
|---|---|
| 机械臂 | FAIRINO FR5，型号 `FR5-V1-002(V6.0)`；WebApp `v3.9.1`；控制器软件 `V3.9.5-QX`；**WebApp 自动模式速度百分比 60%** |
| 相机 | ZED X（S/N 49953548），ZED SDK 5.1.2，HD1080，**30 fps**，全局快门；**手动曝光 2.5 ms，模拟增益 13 dB**；录校正后的左右目 JPEG，关闭自标定 |
| 相机录制程序 | `zed_record`（打了本仓库的补丁）：**可设帧率，3 个线程并行存图** |
| 计算单元 | ZED Box Mini（Jetson，L4T R36.4.4，Ubuntu 22.04，Python 3.10），自带显示器和鼠标 |
| 网络 | Box 有线 `192.168.58.11/24` 用网线直连 FR5 控制器 `192.168.58.2`（固定 IP，无网关）；Box WiFi `10.255.250.132`（DHCP，可能会变），笔记本通过它 SSH：`ssh user@10.255.250.132` |
| 控制器端口 | 20003 XML-RPC 指令；20004 实时状态推送（同一时间只服务一个连接）；80 / 9999 WebApp |
| FAIRINO Python SDK | V2.2.1（对应机器人 V3.9.1）：官方仓库 `FAIR-INNOVATION/fairino-python-sdk` 提交 `060c68bb85f05d0097cc5efa360ddc61d6bb72e0` 的 `linux/fairino/Robot.py`，只依赖标准库 |
| 笔记本 | 标定板验证和图像检查用标定仓库 `D:\天府绛溪实验室\calibration` 的 `.venv`（OpenCV 4.14，带 aruco）；Box 上的 Python 导入不了 OpenCV |

## 3. 算法

### 3.1 相机真值

和 main 相同：每张图像在曝光中心时刻插值法兰位姿，再乘手眼外参，得到左目相机在 FR5 基坐标系下的位姿。修正量用高速的标定板录制重新测过（3.3）。时间修正按曝光查表；行程、转角系数按路径组查表（本分支的路径组是 `fast2`，写在每条录制的 `motion_log.json` 里）：

| 修正 | 低速数据（main） | 高速数据（本分支） |
|---|---|---|
| 时间：取真值的时刻再晚 | 4.27 ms（10 ms 曝光） | 3.75 ms（2.5 ms 曝光） |
| 行程放大（平移） | 左右、上下 1.0070，前后 1.0015 | 上下 1.0007，左右 1.0058，前后 1.0043 |
| 转角放大（旋转） | 俯仰向上 1.0043、向下 1.0052 | 俯仰向上 0.9964；偏航、横滚 1.0 |
| 端点位姿修正 | — | 6 个方向各一个刚体变换（`END_POSE_CORRECTION`） |

**端点位姿修正**是本分支新加的：机械臂报告的终点除了行程、转角，位置和其他方向的姿态也偏几毫米、零点几度（横滚转到头约 10 mm、0.7°）。由标定板测出终点还差的刚体变换 ΔT，运动中按进度 s（走过的距离或转过的角度占全程的比例）补上 exp(s·log ΔT)。它和行程、转角系数一样只适用于测过的那条路径。

没有录过的方向（例如 `down`）不修正，真值报告里会警告。同时输出一份不加这些修正的真值，用于对照。

### 3.2 时间同步

方法和 main 相同（控制器时钟按最小延迟的状态包直线拟合，剩余时间差 Δt 由陀螺仪和法兰角速度互相关，曝光中心由 `image_sync_trigger` 估计，`frame_lag_check.py` 检查图像和时间戳是否错位）。录制期间关掉了 Box 的自动对时：WiFi 上的 NTP 会持续调整系统时钟，让控制器时钟的拟合变差。

**结果**

- 时钟拟合残差 0.4–1.1 ms（8 条标定板录制和 6 条正式录制）。前后两段 Δt 之差：平移、偏航、横滚 0.04–1.9 ms；俯仰 2.6–3.7 ms，是那个位姿下互相关方法本身的偏差，标定板的时间扫描在两端是一致的。
- 标定板检验（3.3）：加时间修正后残差 −0.5～+1.0 ms。
- **图像对陀螺仪**：全速录制都在 −3.9～−6.1 ms（10 ms 曝光时约 0），即图像内容对应的 IMU 时刻比图像时间戳晚约 4.6 ms。**做 VIO 时要在线估计图像和 IMU 的时间差，或先把图像时间戳加约 4.6 ms**，不能像低速数据那样直接用原始时间戳。
- ZED 偶尔整条录制的图像比时间戳晚一帧（30 fps 下表现为 +29 ms）：10-15 检查的 26 条里出现 1 次（一条标定板录制，已重录），正式录制都没有。
- 时间误差的量级：1 ms 在 0.5 m/s 时约 0.5 mm，在 50 °/s 时约 0.05°。

### 3.3 真值验证（标定板）

方法和 main 相同（PnP 求标定板位姿，乘真值后应始终不变；检查重复性、运动前后的差、转角对照和时间扫描）。本分支的标定板只放了一处，约在基坐标 (1.65, −0.02, 0.56) m，13 条验证录制都用它：每个方向上午、下午各一条（上下上午录了两条、下午没录，俯仰上午两条、下午一条）。摆放前用演练录制的真值逐帧核对过，每条录制两端都能看到标定板（俯仰两端只看到 12–24 个 tag，其余都是整块）。

**结果**（加全部修正后，13 条验证录制）

| 项目 | 平移（516–758 mm） | 旋转（54–162°） |
|---|---|---|
| 重复性（同一位姿） | 0.01–0.07 mm | ≤ 0.02° |
| 两端标定板位置差 | 0.06–0.44 mm | 0.05–0.28 mm；转角差 ≤ 0.03%，转轴方向 ≤ 0.02° |
| 运动中标定板偏离（RMS / 最大） | 0.6–0.9 / 1.6–2.7 mm | 偏航 0.5 / 1.4，俯仰 0.6–0.8 / 1.7–2.0，横滚 1.5–1.6 / 3.4 mm |
| 运动中姿态偏离（RMS） | 0.07–0.09° | 0.06–0.16° |
| 时间残差 | −0.47～+0.46 ms | +0.02～+0.95 ms |

- **时间**：2.5 ms 曝光下真值比图像晚 3.75 ms（各方向 3.2–4.4 ms），和 10 ms 曝光的 4.27 ms 接近，已修正。两档按曝光分开存，低速数据重算结果不变。
- **行程**：机械臂实际走得比报告的远，上下 0.07%，左右 0.58%，前后 0.43%，已修正。和低速（约 0.7% / 0.15%）不同：高速路径在另一种构型、另一片区域，系数不能通用。
- **转角**：俯仰时机械臂实际少转 0.36%（低速那条路径是多转 0.43%），已修正；偏航、横滚不需要修正。
- **端点位姿**：只修行程、转角时，终点还剩几毫米、零点几度（平移垂直方向最大 6.5 mm；旋转转轴偏 0.14–0.69°，转动中心偏离使标定板位置差 3–10 mm，横滚最大）。来源是机械臂在这片区域的运动学误差和手眼外参，相机内参和时间不是主要来源。端点位姿修正以后两端闭合。同一方向上午、下午两次测的修正量相差 ≤ 0.04°、≤ 2 mm；用一条录制拟合、另一条检验（留一法），运动中最大偏离从 2–10 mm 降到 1.1–3.6 mm。
- **剩下的误差**：在运动中途，左右、横滚最大 2.5–3.4 mm（误差不和进度成正比），前后约 1.5 mm。运动中的数还包含时间误差、运动模糊和 PnP 噪声。
- **精度结论**：真值在两端和静止段的精度约 0.1–0.4 mm、0.02–0.08°，运动中约 0.5–1.6 mm RMS（最大约 3.4 mm）、0.06–0.16°。足够评测高速下 SLAM 的毫米到厘米级误差，不适合评测亚毫米精度。修正量只适用于测过的这 6 条路径。

### 3.4 运动与速度

- **拖动标记定路径**（`mark`、`limits`、`fr5_limits.py`）：人在手动模式下把机械臂拖到每个方向能接受的极限，记下 12 个标记，每对标记定一条录制。平移：相机水平、姿态不变走直线，三条都朝基坐标 +X。旋转：绕两个标记光心的中点单轴转（MoveC）。标记只是大概的极限，脚本用控制器逆解把两端往里收：关节不超过 FR5 范围和控制器软限位、构型不变、离奇异位形至少 5°、关节速度不超过 150 °/s、相机不低于下限、不拍右窗、两端能做同步摆动。`plan` 打印每条最后的范围，以及两端停在那里的原因。
- **不回起点**：每条录制从机械臂当前位姿直线移到这条的起点（不录），录完停在终点。
- **同步摆动**：平移录制只往一侧摆 10°（哪一侧由 `plan` 按行程选），旋转录制往转动范围内侧摆；都是 20 °/s、来回 2 次。
- **不拍右窗**：沿路径检查画面边缘离右侧落地窗至少 1°。

  ![实验室全景图](docs/scene_panorama.jpg)

  *全景图：横轴方位角、纵轴仰角（°），以参考起点的光心为球心、相对其水平朝向，向左为正。左侧 +57° 以左是左窗（可以拍），右侧 −60° 以右是落地窗（不能拍）。*

- **运动模糊**：曝光 2.5 ms 时，0.5 m/s 平移在 1 m 处 0.92 px，50 °/s 旋转在画面中心 1.6 px。
- **保护**：和 main 相同（超速、光心偏离、状态中断、停顿时 StopMotion；Ctrl+C、SSH 断开时先停机械臂再停相机）。0.5 m/s 时停车距离约 0.1 m。

**结果**

- 速度换算（WebApp 60%）：1% 对应平移 5.960 mm/s、摆动 2.152 °/s；全速时加速度约 1.08 m/s²。
- 正式录制实测：平移 502.8–503.9 mm/s（匀速段波动 0.6–1.0%）；旋转 50.1 °/s，光心偏离 ≤ 2.8 mm；同步摆动约 20 °/s。演练中关节最快实测 122 °/s。
- 30 fps 不丢帧（帧间隔 32–35 ms），机械臂状态无丢包。

## 4. 脚本与参数

### 4.1 文件

“改”“新”是相对 main 而言。

| 文件 | 说明 |
|---|---|
| `scripts/run_session.py` | 改：总控脚本，**所有可改参数都在文件开头**。子命令：`set-start`、`mark <方向>`、`limits`、`plan`、`goto-start`、`speed-test`、`sync-test`、`movec-test`、`record <方向>`；选项 `--scale`、`--yes`、`--tag` |
| `scripts/fr5_limits.py` | 新：拖动标记 → 录制路径（方向、姿态、转动中心，用控制器逆解收两端） |
| `scripts/fr5_motion.py` | 改：运动规划与执行，加了关节速度检查和平移录制的右窗检查 |
| `scripts/fr5_record.py` | 机械臂状态录制：独占 20004，每包记 Box 时间，停止时一次写盘 |
| `scripts/zed_capture.py` | zed_record 子进程管理 |
| `scripts/fr5_groundtruth.py` | 改：离线真值；时间修正按曝光、行程和转角系数按路径组查表 |
| `scripts/gt_board_check.py` | 标定板真值验证（笔记本上运行） |
| `scripts/end_pose_fit.py` | 新：由同一路径的标定板录制算端点位姿修正，并用运动段每帧检验（笔记本上运行） |
| `scripts/frame_lag_check.py` | 检查图像内容和时间戳是否错位（笔记本上运行），有异常时退出码为 1 |
| `scripts/camera_s003_refit.yaml` | ZED 校正后左目 HD1080 内参，用 s003 图像重标 |
| `scripts/test_*.py` | 离线测试（合成数据、假逆解，不连机械臂），共 94 项；要 OpenCV 的只在笔记本上运行，要 FAIRINO SDK 的只在 Box 上运行，缺了自动跳过 |
| `scripts/movel_probe.py` | 诊断：发零行程的 MoveL / MoveJ，看控制器拒绝哪种参数 |
| `scripts/handeye_s003_result.json` | 手眼外参（标定仓库 `results/s003/result.json` 的副本） |
| `scripts/start_pose.json` | 参考起点：只用来定右窗的方向和做 sync-test，录制不回这里 |
| `scripts/limit_poses.json` | 新：`mark` 记下的 12 个拖动标记 |
| `scripts/speed_calibration.json` | `speed-test` 的结果（速度换算系数和加速度） |
| `zed_record/zed_record.cpp`、`zed_record.patch` | 改：Box 上 `~/ROBOT/zed_imu/zed_record.cpp` 的当前版本和相对原版的补丁（可设曝光、增益和帧率，多线程存图，停止时先等抓图线程结束再关相机） |
| `docs/` | README 用的图片 |
| `sync_from_box.ps1` | 从 Box 拉取 `scripts/` 和 `zed_record/` |
| `data/` | 从 Box 拷到笔记本的录制数据（不进 git） |

### 4.2 可调参数（`run_session.py` 开头）

超出范围脚本会拒绝运行。下表只列和 main 不同或新增的参数，其余同 main。

| 参数 | 本分支 | main | 说明 |
|---|---|---|---|
| `PATH_SET` | fast2 | — | 路径组名，写进 `motion_log.json`，真值按它查行程、转角系数；重新拖动标记、路径变了就要换个名字 |
| `SPEED_MPS` / `ROT_DEG_S` | 0.5 / 50 | 0.05 / 5 | 平移、旋转速度（范围 0.02–0.6 m/s、1–60 °/s） |
| `CAMERA_FPS` | 30 | （15） | 相机帧率，15 / 30 / 60 |
| `EXPOSURE_US` / `ANALOG_GAIN_MDB` | 2500 / 13000 | 10000 / 1000 | 曝光、增益 |
| `MAX_BLUR_PX` | 2.0 | 1.0 | 允许的运动模糊（1080p 像素） |
| `WEBAPP_SPEED_PCT` | 60 | 10 | WebApp 里的速度百分比；改了要重跑 speed-test |
| `MIN_CAMERA_Z_M` | 0.18 | 0.35 | 相机光心最低高度，只是兜底，上下范围由标记定 |
| `LIMIT_MARGIN_MM` / `LIMIT_MARGIN_DEG` | 30 / 3 | — | 录制两端离拖动标记至少收进这么多 |
| `MIN_STROKE_M` / `MIN_ROTATION_DEG` | 0.15 / 15 | — | 收完后行程、转角的下限，不够就不录 |
| `JOINT_MIN_DEG` / `JOINT_MAX_DEG` | FR5 关节范围 | 保守范围 | 运行时再和控制器的关节软限位取交集；腕部、线缆、机柜由拖动标记保证 |
| `SOFT_LIMIT_MARGIN_DEG` / `SINGULAR_MARGIN_DEG` | 2 / 5 | — | 离软限位、离奇异位形（J3、J5 = 0）至少这么多 ° |
| `MAX_JOINT_SPEED_DEG_S` | 150 | — | 录制段关节速度上限（FR5 额定 180 °/s） |
| `TRANSLATION_WIGGLE_AUTO` | True | — | 平移录制的同步摆动往哪一侧按行程自动选 |
| `UP_DOWN_HEADING_BASE_DEG` | 0 | — | 上下录制的相机朝向（基坐标系，0 = 朝 +X，和左右、前后一致） |
| `GLASS_REGIONS` | 右窗 | 两侧窗 | 画面里不能出现的玻璃 |

去掉的参数：`STROKE_M`、`YAW_DEG` / `PITCH_DEG` / `ROLL_DEG`、三个 `*_HEADING_DEG`、`SPEED_TEST_MM`：范围和朝向改由拖动标记定，speed-test 走上下那条路径的全程。

`fr5_groundtruth.py` 开头的修正表：`TIME_CORRECTION_MS_BY_EXPOSURE_US`（按曝光）、`TRAVEL_SCALE`、`ROTATION_SCALE`、`END_POSE_CORRECTION`（后三个按路径组），见 3.1。命令行选项同 main，另加 `--no-end-correction`（不加端点位姿修正）。

## 5. 使用流程

### 5.1 安装（Box）

同 main：装 SDK、给 `zed_record` 打上本分支的 `zed_record/zed_record.patch` 后重新编译、把 `scripts/` 放到 `~/ROBOT/fr5_record/`。WebApp 自动模式速度百分比设为 60%，和 `WEBAPP_SPEED_PCT` 一致。

### 5.2 录制

在 Box 自己的终端里运行，人站在机械臂旁边，手边有急停。每次运动前脚本都会列出计划，输入 `yes` 才会动。

1. **拖动标记**（手动模式）：把机械臂拖到某个方向能接受的极限，松手停稳后记下；12 个逐个做，顺序随意：
   ```bash
   cd ~/ROBOT/fr5_record
   python3 run_session.py mark down        # 平移：down / up、left / right、backward / forward
   python3 run_session.py mark yaw_right   # 旋转：yaw_right / yaw_left、pitch_down / pitch_up、roll_left / roll_right
   python3 run_session.py limits           # 列出全部标记和每条录制的路径（不连机械臂）
   ```
2. **速度标定**（自动模式，改了 WebApp 速度百分比后必须跑）：`speed-test --scale 0.2`，再全速 `speed-test`。
3. **空跑检查**：`python3 run_session.py plan`（不动），看 6 条录制的范围和两端停在哪里、为什么。
4. **分档演练**：每条先 `--scale 0.2`、再 `--scale 0.5`、最后全速走一遍（路径和正式录制相同，数据不用），盯住腕部、线缆和周围障碍物：
   ```bash
   python3 run_session.py record up --scale 0.2
   ```
5. **标定板验证**（5.3）。
6. **正式录制**，共 6 条：
   ```bash
   python3 run_session.py record up        # 平移：up、right、forward
   python3 run_session.py record yaw_left  # 旋转：yaw_left、pitch_up、roll_right
   ```
7. **录完的检查**：拷到笔记本后，在 `scripts/` 下用标定仓库的 venv 运行 `python -B frame_lag_check.py <save_data 目录>`，报“晚约 1 帧”的重录；确认画面里没有人走动、没有右窗。

- **每条录制的结构**：（移到起点，不录）→ 静止 5 s → 同步摆动 → 静止 3 s → 平移或旋转 → 静止 3 s → 同步摆动 → 静止 5 s，约 25 s、0.75 GB。结束后自动计算真值。
- **录制失败**：zed_record 没有正常结束时，最后打印“录制失败”，退出码为 1，这条要重录。
- `--tag board`：给录制目录名加后缀，用来区分验证录制和正式录制。

### 5.3 标定板真值验证

1. **摆板**：标定板要在每条录制的两端都在画面里。可以用演练录制的真值逐帧算标定板投影来核对。本次一处（约 (1.65, −0.02, 0.56) m，朝 −X）就覆盖了 6 条。
2. **录**：6 条各一条 `--tag board`，每个方向最好录两条（端点位姿修正可以用一条拟合、另一条检验）。**从这里到正式录完，标记、速度标定、WebApp 速度百分比和 `run_session.py` 的参数都不能动**，否则路径会变，测出的系数就不适用于正式录制。
3. **检查**：拷到笔记本的 `data/board_check/<录制名>/`（`save_data/right/` 不用拷），在 `scripts/` 下运行：
   ```bash
   python -B frame_lag_check.py ..\data\board_check\<录制名>\save_data
   python -B gt_board_check.py ..\data\board_check\<录制名>\save_data
   ```
4. **写回和闭合**：把测得的时间修正（= 当时加的修正 + 时间扫描的 δ）和行程、转角系数写进 `fr5_groundtruth.py`；再对每个方向运行 `python -B end_pose_fit.py <同一方向的几条 save_data>`，把它打印的那一行填进 `END_POSE_CORRECTION`。同步到 Box，重算标定板录制和正式录制的真值，再跑 `gt_board_check.py` 确认两端标定板位置一致。

### 5.4 注意事项

- **高速运动的安全**：0.5 m/s 时停车距离约 0.1 m。拖标记时就要确认那个位姿周围没有障碍物；第一次走新路径时盯住肘部和机柜边缘。
- **构型**：一对标记最好在同一种构型里拖；构型不同时一条录制只能用其中一种，路径会短一些。录制之间构型不同时，脚本会要求先手动拖到下一条的起点附近。
- **录制期间关掉 Box 自动对时**（`sudo timedatectl set-ntp false`），全部录完再打开（`set-ntp true`）。
- **每条录制都要查晚一帧**（5.2 第 7 步），偶尔会出现。
- **20004 只能有一个连接**：录制时不要另开 `Robot.RPC(...)`，运动脚本必须用 `StateRecorder.open_rpc()`。
- **相机同一时间只能被一个程序打开**：录制前关掉 ZED Explorer、Depth Viewer 等。
- **Box 的 WiFi 会断**：运动和录制在 Box 本机终端运行。
- **开机预热**：Box 重启后预热 30 分钟以上再录真值数据；正常关机，直接断电会丢掉刚写的数据。
- 其余同 main（不要调用 SDK 的 `LoggerInit()`、录制进程里不做重计算、错误码查官方对照表）。

## 6. 输出数据

每条录制一个目录（Box 上），格式和 main 相同：

```text
~/ROBOT/sessions/YYYYmmdd_HHMMSS_<名称>[_<tag>]/
├─ zed_record.log            zed_record 的输出（结束时一次写入）
├─ run_session.log           仅在 SSH 断开 / 被终止时才有
└─ save_data/
   ├─ left/  right/          <ZED 时间戳 ms>.jpg（校正后，JPEG）
   ├─ img_timeline.csv       每帧：图像时间戳、对应 IMU 时间戳、文件名
   ├─ zed_imu.csv            机身 IMU 全部样本（约 200 Hz），含 image_sync_trigger
   ├─ time_meta.txt          相机设置（帧率、曝光、增益），含第 30 帧读回的实际曝光和增益
   ├─ fr5_state.csv          机械臂状态（约 100 Hz）：Box 时间、控制器时间、关节、法兰位姿、错误码等
   ├─ fr5_meta.json          丢包、断线、时钟跳变等
   ├─ motion_log.json        参数（含路径组）、拖动标记和由它定的路径、每段指令和实测速度；
   │                         旋转录制还有转轴、转动中心和光心偏离；录制失败时 error 写明原因
   ├─ groundtruth_tum.txt    每张图一行：timestamp tx ty tz qx qy qz qw（Unix 秒、米），左目在 FR5 基坐标系下（已加修正）
   ├─ groundtruth_tum_uncorrected.txt  同上，不加修正（对照用）
   ├─ groundtruth_arm_rate_tum.txt   机械臂每个状态包一行，格式同上（已加修正）
   ├─ groundtruth_images.csv         每张图的真值和标记（ok / duplicate / outside / gap / robot_error）
   └─ groundtruth_report.json        时钟拟合、Δt、前后段差、曝光中心、各项修正、警告
```

笔记本上的副本放在 `data/` 下（不进 git）：

- 正式录制：`data/recordings_highspeed/`（6 条，含左右目，约 4.5 GB）；
- 标定板验证录制：`data/board_check/<录制名>/`（只有左目）。

真值可以单独重算：

```bash
python3 fr5_groundtruth.py <save_data 目录> --handeye handeye_s003_result.json
```
