/**
 * zed_record.cpp
 *
 * 只录 ZED：从 SDK 读双目图像 + 机身 IMU。自由运行（软件 FPS），
 * 不拉 GPIO、不等 FPGA 触发，也不做复位锁定第一帧 / 标定轴硬加。
 *
 * 输出（可执行文件旁 save_data/）：
 *   - left/<zed_timestamp_ms>.jpg、right/<zed_timestamp_ms>.jpg
 *   - zed_imu.csv（getSensorsDataBatch）
 *   - img_timeline.csv
 *   - time_meta.txt
 *
 * 时间戳直接用 SDK 设备时：zed_timestamp_ms / zed_imu_timestamp_ns。
 * Ctrl+C 结束。有 DISPLAY 时左目预览，ESC 关窗口采集继续。
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <limits.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

namespace fs = std::filesystem;

// ============================================================
// 全局
// ============================================================
static std::atomic<bool> g_running{true};
// 相机帧率；运行时可用 --fps 15|30|60 覆盖（见 main）
static constexpr int kDefaultCameraFps = 15;
static int g_camera_fps = kDefaultCameraFps;
static std::mutex g_log_mtx;

using SteadyClock = std::chrono::steady_clock;

static void on_signal(int) { g_running.store(false); }

static void log_line(const std::string& s) {
  std::lock_guard<std::mutex> lock(g_log_mtx);
  std::cout << s << std::endl;
}

static void try_avoid_last_cpu(const char* name) {
  const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu < 2) {
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (long i = 0; i < ncpu - 1; ++i) {
    CPU_SET(static_cast<int>(i), &set);
  }
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    log_line(std::string(name) + " 避开末核失败（" + std::strerror(errno) +
             "）");
  }
}

/** 存图/预览让出 CPU，避免压住 IMU 采集 */
static void try_lower_nice(int inc, const char* name) {
  errno = 0;
  const int n = ::nice(inc);
  if (n == -1 && errno != 0) {
    log_line(std::string(name) + " 降低 nice 失败（" + std::strerror(errno) +
             "）");
  } else {
    log_line(std::string(name) + " nice=" + std::to_string(n));
  }
}

/** 可执行文件所在目录（build/），用于落盘 */
static fs::path exe_dir() {
  char buf[PATH_MAX];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) {
    buf[n] = '\0';
    return fs::path(buf).parent_path();
  }
  return fs::current_path();
}

/** sudo 跑时文件是 root 的；结束时交还给 SUDO_UID，方便普通用户删除 */
static bool sudo_invoker(uid_t* uid, gid_t* gid) {
  if (::geteuid() != 0) {
    return false;
  }
  const char* u = std::getenv("SUDO_UID");
  const char* g = std::getenv("SUDO_GID");
  if (!u || !g || !*u || !*g) {
    return false;
  }
  char* end = nullptr;
  const unsigned long uid_ul = std::strtoul(u, &end, 10);
  if (!end || *end != '\0') {
    return false;
  }
  end = nullptr;
  const unsigned long gid_ul = std::strtoul(g, &end, 10);
  if (!end || *end != '\0') {
    return false;
  }
  *uid = static_cast<uid_t>(uid_ul);
  *gid = static_cast<gid_t>(gid_ul);
  return true;
}

static void give_save_data_back_to_user(const fs::path& dir) {
  uid_t uid = 0;
  gid_t gid = 0;
  if (!sudo_invoker(&uid, &gid)) {
    return;
  }
  std::error_code ec;
  if (!fs::exists(dir, ec)) {
    return;
  }
  auto chown_one = [&](const fs::path& p) {
    if (::chown(p.c_str(), uid, gid) != 0) {
      log_line(std::string("chown 失败: ") + p.string() + " (" +
               std::strerror(errno) + ")");
    }
  };
  chown_one(dir);
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end && !ec;
       ++it) {
    chown_one(it->path());
  }
  log_line("save_data 已交还给 uid=" + std::to_string(uid) +
           "（sudo 写出的文件可直接删除）");
}

// ============================================================
// 有界队列（图像可丢旧帧；IMU 用 push_wait 阻塞等待，保证不丢）
// ============================================================
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}

  /** 队满时丢最旧（适合图像） */
  bool push(T item) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!running_) {
      return false;
    }
    if (q_.size() >= max_size_) {
      q_.pop_front();
      ++dropped_;
    }
    q_.push_back(std::move(item));
    cv_.notify_one();
    return true;
  }

  /** 队满时阻塞等待，直到有空位（适合 IMU 完整保存） */
  bool push_wait(T item) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_space_.wait(lock, [this] {
      return !running_ || q_.size() < max_size_;
    });
    if (!running_) {
      return false;
    }
    q_.push_back(std::move(item));
    cv_.notify_one();
    return true;
  }

  bool pop(T& out) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return !q_.empty() || !running_; });
    if (q_.empty()) {
      return false;
    }
    out = std::move(q_.front());
    q_.pop_front();
    cv_space_.notify_one();
    return true;
  }

  /** 非阻塞取一帧；空则返回 false（主线程搬运用，避免卡住无法 Ctrl+C） */
  bool try_pop(T& out) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (q_.empty()) {
      return false;
    }
    out = std::move(q_.front());
    q_.pop_front();
    cv_space_.notify_one();
    return true;
  }

  /** 已 stop 且空，消费者可以退出 */
  bool finished() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return !running_ && q_.empty();
  }

  /** 等到有数据或 stop，最多 timeout（双队列合写时避免空转） */
  void wait_for_data(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait_for(lock, timeout, [this] { return !q_.empty() || !running_; });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      running_ = false;
    }
    cv_.notify_all();
    cv_space_.notify_all();
  }

  size_t dropped() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return dropped_;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.size();
  }

 private:
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::condition_variable cv_space_;
  std::deque<T> q_;
  size_t max_size_;
  size_t dropped_ = 0;
  bool running_ = true;
};

struct ZedFrameItem {
  SteadyClock::time_point recv{};  // grab 成功时刻（只用于诊断，不写盘）
  uint64_t timestamp_ms = 0;       // ZED IMAGE 设备时间戳
  uint64_t zed_imu_ns = 0;
  sl::Mat image;        // 左目
  sl::Mat image_right;  // 右目
};

/** ZED 机身 IMU 全量样本（getSensorsDataBatch，一行一条） */
struct ZedImuSample {
  uint64_t zed_imu_timestamp_ns = 0;
  uint64_t zed_timestamp_ms = 0;
  int image_sync_trigger = 0;
  float accel[3]{};  // m/s^2
  float gyro[3]{};   // deg/s
  float quat[4]{};   // qw,qx,qy,qz
};

/** 预览用：只保留最新一张小图，给窗口看画面 */
struct PreviewSlot {
  static constexpr int kWidth = 960;
  static constexpr int kHeight = 540;

  std::mutex mtx;
  cv::Mat bgr_small;
  uint64_t timestamp_ms = 0;
  bool fresh = false;

  void try_publish(const sl::Mat& image, uint64_t ts_ms) {
    std::unique_lock<std::mutex> lock(mtx, std::try_to_lock);
    if (!lock.owns_lock()) {
      return;
    }
    if (image.getWidth() <= 0 || image.getHeight() <= 0) {
      return;
    }
    cv::Mat bgra(static_cast<int>(image.getHeight()),
                 static_cast<int>(image.getWidth()), CV_8UC4,
                 const_cast<sl::uchar1*>(image.getPtr<sl::uchar1>(sl::MEM::CPU)));
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    cv::resize(bgr, bgr_small, cv::Size(kWidth, kHeight));
    timestamp_ms = ts_ms;
    fresh = true;
  }

  bool try_consume(cv::Mat& out, uint64_t& ts_ms) {
    std::unique_lock<std::mutex> lock(mtx, std::try_to_lock);
    if (!lock.owns_lock() || !fresh || bgr_small.empty()) {
      return false;
    }
    bgr_small.copyTo(out);
    ts_ms = timestamp_ms;
    fresh = false;
    return true;
  }
};

/** 图像时间线：直接写 SDK 设备戳，不做标定轴 */
class ImageTimeline {
 public:
  explicit ImageTimeline(fs::path path) : path_(std::move(path)) {
    ofs_.open(path_);
    if (ofs_) {
      ofs_ << "zed_timestamp_ms,zed_imu_timestamp_ns,"
              "filename_left,filename_right\n";
    } else {
      log_line("无法创建图像时间线 CSV: " + path_.string());
    }
  }

  // seq：取帧顺序编号（从 0 起，每帧一个）。多个存图线程完成顺序不定，先按编号排好再按顺序写；
  // row 为空表示这一帧没存成（跳过）。
  void write_row(uint64_t seq, std::optional<std::string> row) {
    std::lock_guard<std::mutex> lock(mtx_);
    pending_[seq] = std::move(row);
    while (!pending_.empty() && pending_.begin()->first == next_seq_) {
      if (pending_.begin()->second && ofs_) {
        ofs_ << *pending_.begin()->second << '\n';
        ++rows_;
        if ((rows_ % g_camera_fps) == 0) {
          ofs_.flush();
        }
      }
      pending_.erase(pending_.begin());
      ++next_seq_;
    }
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (ofs_) {
      ofs_.flush();
      ofs_.close();
    }
  }

  uint64_t rows() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return rows_;
  }

 private:
  mutable std::mutex mtx_;
  fs::path path_;
  std::ofstream ofs_;
  uint64_t rows_ = 0;
  uint64_t next_seq_ = 0;
  std::map<uint64_t, std::optional<std::string>> pending_;
};

// ============================================================
// ZED 初始化（软件 FPS 自由运行，默认 15，--fps 可改；不依赖外部触发）
// ============================================================
// 全手动曝光（ZED X）。单位：EXPOSURE_TIME=μs，ANALOG_GAIN=mdB。
// 曝光和模拟增益的默认值；运行时可用 --exposure-us / --analog-gain-mdb 覆盖（见 main）
static constexpr int kDefaultExposureUs = 10000;     // 10ms
static constexpr int kDefaultAnalogGainMdB = 1000;  // 约 1000–16000，偏暗再调大
static constexpr int kManualDigitalGain = 1;       // 约 1–256
static int g_exposure_us = kDefaultExposureUs;
static int g_analog_gain_mdb = kDefaultAnalogGainMdB;
// 第 30 帧时读回的实际值（设置刚下发时立刻读回的值不可靠）
static std::atomic<int> g_exposure_readback{-1};
static std::atomic<int> g_gain_readback{-1};

static bool apply_manual_exposure(sl::Camera& zed) {
  if (zed.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 0) !=
      sl::ERROR_CODE::SUCCESS) {
    log_line("设置 AEC_AGC=0 失败");
    return false;
  }
  if (zed.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE_TIME,
                            g_exposure_us) != sl::ERROR_CODE::SUCCESS) {
    log_line("设置 EXPOSURE_TIME 失败");
    return false;
  }
  if (zed.setCameraSettings(sl::VIDEO_SETTINGS::ANALOG_GAIN,
                            g_analog_gain_mdb) != sl::ERROR_CODE::SUCCESS) {
    log_line("设置 ANALOG_GAIN 失败");
    return false;
  }
  if (zed.setCameraSettings(sl::VIDEO_SETTINGS::DIGITAL_GAIN,
                            kManualDigitalGain) != sl::ERROR_CODE::SUCCESS) {
    log_line("设置 DIGITAL_GAIN 失败");
    return false;
  }

  int exposure_us = -1;
  int analog_gain = -1;
  int digital_gain = -1;
  int aec_agc = -1;
  zed.getCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, aec_agc);
  zed.getCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE_TIME, exposure_us);
  zed.getCameraSettings(sl::VIDEO_SETTINGS::ANALOG_GAIN, analog_gain);
  zed.getCameraSettings(sl::VIDEO_SETTINGS::DIGITAL_GAIN, digital_gain);
  log_line("手动曝光已锁定: AEC_AGC=" + std::to_string(aec_agc) +
           ", EXPOSURE_TIME=" + std::to_string(exposure_us) + " us (" +
           std::to_string(exposure_us / 1000.0) +
           " ms), ANALOG_GAIN=" + std::to_string(analog_gain) +
           " mdB, DIGITAL_GAIN=" + std::to_string(digital_gain));
  return true;
}

std::unique_ptr<sl::Camera> init_zedx() {
  auto zed = std::make_unique<sl::Camera>();
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::HD1080;
  init_params.camera_fps = g_camera_fps;
  init_params.depth_mode = sl::DEPTH_MODE::NONE;
  init_params.camera_image_flip = sl::FLIP_MODE::OFF;
  init_params.camera_disable_self_calib = true;

  log_line("正在打开 ZED X 相机... (软件 FPS=" + std::to_string(g_camera_fps) +
           ", 自由运行，不使用硬件触发)");
  const sl::ERROR_CODE err = zed->open(init_params);
  if (err != sl::ERROR_CODE::SUCCESS) {
    log_line(std::string("打开失败: ") + sl::toString(err).c_str());
    return nullptr;
  }

  if (!apply_manual_exposure(*zed)) {
    log_line("手动曝光设置失败，关闭相机。");
    zed->close();
    return nullptr;
  }
  return zed;
}

// ============================================================
// 抓图线程：grab 图像 + getSensorsDataBatch IMU，原样入队
// ============================================================
void grab_thread_func(sl::Camera* zed, BoundedQueue<ZedFrameItem>& image_q,
                      BoundedQueue<ZedImuSample>& zed_imu_q,
                      std::atomic<bool>& stop_event) {
  try_avoid_last_cpu("抓图线程");
  sl::RuntimeParameters runtime_params;
  std::vector<sl::SensorsData> zed_imu_batch;
  uint64_t last_ts = 0;
  uint64_t last_zed_imu_ns = 0;
  bool have_zed_imu = false;
  int corrupted_count = 0;
  uint64_t frame_count = 0;
  size_t last_batch_n = 0;

  while (!stop_event.load() && g_running.load()) {
    const sl::ERROR_CODE status = zed->grab(runtime_params);

    if (status == sl::ERROR_CODE::SUCCESS) {
      corrupted_count = 0;
      const auto recv = SteadyClock::now();
      const uint64_t current_ts =
          zed->getTimestamp(sl::TIME_REFERENCE::IMAGE).getMilliseconds();
      uint64_t zed_imu_ns = 0;
      sl::SensorsData sensors;
      if (zed->getSensorsData(sensors, sl::TIME_REFERENCE::IMAGE) ==
              sl::ERROR_CODE::SUCCESS &&
          sensors.imu.is_available) {
        zed_imu_ns = sensors.imu.timestamp.getNanoseconds();
      }
      zed_imu_batch.clear();
      last_batch_n = 0;
      if (zed->getSensorsDataBatch(zed_imu_batch) == sl::ERROR_CODE::SUCCESS) {
        for (const sl::SensorsData& s : zed_imu_batch) {
          if (!s.imu.is_available) {
            continue;
          }
          const uint64_t sample_ns = s.imu.timestamp.getNanoseconds();
          if (have_zed_imu && sample_ns == last_zed_imu_ns) {
            continue;
          }
          last_zed_imu_ns = sample_ns;
          have_zed_imu = true;

          const sl::Orientation q = s.imu.pose.getOrientation();
          ZedImuSample rec;
          rec.zed_imu_timestamp_ns = sample_ns;
          rec.zed_timestamp_ms = current_ts;
          rec.image_sync_trigger = s.image_sync_trigger;
          rec.accel[0] = s.imu.linear_acceleration.x;
          rec.accel[1] = s.imu.linear_acceleration.y;
          rec.accel[2] = s.imu.linear_acceleration.z;
          rec.gyro[0] = s.imu.angular_velocity.x;
          rec.gyro[1] = s.imu.angular_velocity.y;
          rec.gyro[2] = s.imu.angular_velocity.z;
          rec.quat[0] = q.w;
          rec.quat[1] = q.x;
          rec.quat[2] = q.y;
          rec.quat[3] = q.z;
          zed_imu_q.push_wait(std::move(rec));
          ++last_batch_n;
        }
      }
      sl::Mat image;
      sl::Mat image_right;
      if (zed->retrieveImage(image, sl::VIEW::LEFT, sl::MEM::CPU) ==
          sl::ERROR_CODE::SUCCESS) {
        ZedFrameItem item;
        item.recv = recv;
        item.timestamp_ms = current_ts;
        item.zed_imu_ns = zed_imu_ns;
        image.copyTo(item.image);
        if (zed->retrieveImage(image_right, sl::VIEW::RIGHT, sl::MEM::CPU) ==
            sl::ERROR_CODE::SUCCESS) {
          image_right.copyTo(item.image_right);
        }
        image_q.push(std::move(item));
      }

      ++frame_count;
      if (frame_count == 30) {
        int exposure_us = -1;
        int analog_gain = -1;
        zed->getCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE_TIME, exposure_us);
        zed->getCameraSettings(sl::VIDEO_SETTINGS::ANALOG_GAIN, analog_gain);
        g_exposure_readback.store(exposure_us);
        g_gain_readback.store(analog_gain);
        log_line("第 30 帧读回: EXPOSURE_TIME=" + std::to_string(exposure_us) +
                 " us, ANALOG_GAIN=" + std::to_string(analog_gain) + " mdB");
      }
      if (last_ts != 0) {
        const double diff = static_cast<double>(current_ts - last_ts);
        if (frame_count <= 5 || (frame_count % g_camera_fps) == 0) {
          std::ostringstream oss;
          oss << "ZED 捕获 #" << frame_count << " ts=" << current_ts
              << " ms, 间隔=" << std::fixed << std::setprecision(2) << diff
              << " ms, zed_imu_batch=" << last_batch_n;
          if (diff > 63.0 && diff < 71.0) {
            oss << " [OK 15Hz]";
          } else {
            oss << " [警告: 频率异常]";
          }
          log_line(oss.str());
        }
      }
      last_ts = current_ts;
    } else if (status == sl::ERROR_CODE::CORRUPTED_FRAME) {
      ++corrupted_count;
      if (corrupted_count <= 3 || (corrupted_count % 30) == 0) {
        log_line("grab 警告: CORRUPTED FRAME #" +
                 std::to_string(corrupted_count) + "，跳过");
      }
      continue;
    } else if (static_cast<int>(status) < 0) {
      log_line(std::string("grab 警告: ") + sl::toString(status).c_str());
      continue;
    } else if (status == sl::ERROR_CODE::END_OF_SVOFILE_REACHED ||
               status == sl::ERROR_CODE::CAMERA_NOT_DETECTED) {
      log_line(std::string("grab 结束: ") + sl::toString(status).c_str());
      break;
    } else {
      log_line(std::string("grab 错误: ") + sl::toString(status).c_str() +
               "，重试");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  image_q.stop();
  zed_imu_q.stop();
}

// ============================================================
// 主线程搬运：每帧原样转给存图队列；每隔几帧更新预览小图
// ============================================================
void forward_frame_to_save(ZedFrameItem item, BoundedQueue<ZedFrameItem>& save_q,
                           PreviewSlot* preview, uint64_t& n) {
  ++n;
  if (preview && (n % 3) == 0) {
    preview->try_publish(item.image, item.timestamp_ms);
  }
  save_q.push(std::move(item));
}

// ============================================================
// ZED 预览：挂在主线程（waitKey(2) 兼作空转休眠），不影响存盘
// ============================================================
static constexpr const char* kPreviewWinName = "ZED Preview";

static bool start_preview_window() {
  try {
    cv::namedWindow(kPreviewWinName, cv::WINDOW_NORMAL);
    cv::resizeWindow(kPreviewWinName, PreviewSlot::kWidth, PreviewSlot::kHeight);
  } catch (const cv::Exception& e) {
    log_line(std::string("预览窗口创建失败（跳过显示）: ") + e.what());
    return false;
  }
  log_line("预览已开。ESC 关窗口，采集继续。");
  return true;
}

/** 显示最新小图；ESC 返回 false，调用方应关窗。无新帧时 waitKey(2) 让出 CPU */
static bool pump_preview_window(PreviewSlot& preview) {
  cv::Mat show;
  uint64_t ts_ms = 0;
  if (preview.try_consume(show, ts_ms)) {
    try {
      std::ostringstream title;
      title << "ts=" << ts_ms << " ms";
      cv::putText(show, title.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
                  0.7, cv::Scalar(0, 255, 0), 2);
      cv::imshow(kPreviewWinName, show);
    } catch (const cv::Exception& e) {
      log_line(std::string("预览显示异常（跳过本帧）: ") + e.what());
    }
  }
  const int key = cv::waitKey(2);
  if (key == 27) {
    log_line("预览窗口已关闭（ESC）。");
    return false;
  }
  return true;
}

static void stop_preview_window() {
  try {
    cv::destroyWindow(kPreviewWinName);
  } catch (...) {
  }
}

static fs::path unique_jpg_path(const fs::path& image_dir, uint64_t ts_ms) {
  fs::path path = image_dir / (std::to_string(ts_ms) + ".jpg");
  if (!fs::exists(path)) {
    return path;
  }
  for (int k = 2; k < 10000; ++k) {
    path = image_dir / (std::to_string(ts_ms) + "_" + std::to_string(k) + ".jpg");
    if (!fs::exists(path)) {
      return path;
    }
  }
  return path;
}

// ============================================================
// ZED 存图线程（kSaveThreads 个并行）：左目 left/、右目 right/；文件名用 zed_timestamp_ms
// ============================================================
static constexpr int kSaveThreads = 3;   // 30 fps 时单线程 JPEG 编码只到约 20 fps（10-14 实测）

struct SaveShared {
  std::mutex take_mtx;                  // 取帧 + 编号，保证编号就是取帧顺序
  uint64_t next_seq = 0;
  std::mutex path_mtx;                  // unique_jpg_path 查重名 + 占位，避免两个线程取到同一个文件名
  std::atomic<uint64_t> saved{0};
};

static fs::path claim_jpg_path(SaveShared& shared, const fs::path& image_dir, uint64_t ts_ms) {
  std::lock_guard<std::mutex> lock(shared.path_mtx);
  const fs::path path = unique_jpg_path(image_dir, ts_ms);
  std::ofstream(path).close();          // 先占住文件名，随后由 SDK 覆盖写入
  return path;
}

void zed_save_thread_func(BoundedQueue<ZedFrameItem>& image_q, SaveShared& shared,
                          ImageTimeline& timeline, const fs::path& left_dir,
                          const fs::path& right_dir) {
  try_avoid_last_cpu("存图线程");
  try_lower_nice(10, "存图线程");
  while (true) {
    ZedFrameItem item;
    uint64_t seq = 0;
    {
      std::lock_guard<std::mutex> lock(shared.take_mtx);
      if (!image_q.pop(item)) {
        break;
      }
      seq = shared.next_seq++;
    }
    const fs::path path_l = claim_jpg_path(shared, left_dir, item.timestamp_ms);
    const sl::ERROR_CODE wr_l = item.image.write(path_l.string().c_str());
    if (wr_l != sl::ERROR_CODE::SUCCESS) {
      log_line(std::string("存左目失败: ") + path_l.string() + " -> " +
               sl::toString(wr_l).c_str());
      std::error_code ec;
      fs::remove(path_l, ec);
      timeline.write_row(seq, std::nullopt);
      continue;
    }
    std::string name_r;
    if (item.image_right.getWidth() > 0) {
      const fs::path path_r = claim_jpg_path(shared, right_dir, item.timestamp_ms);
      const sl::ERROR_CODE wr_r =
          item.image_right.write(path_r.string().c_str());
      if (wr_r != sl::ERROR_CODE::SUCCESS) {
        log_line(std::string("存右目失败: ") + path_r.string() + " -> " +
                 sl::toString(wr_r).c_str());
        std::error_code ec;
        fs::remove(path_r, ec);
      } else {
        name_r = (fs::path("right") / path_r.filename()).string();
      }
    }
    std::ostringstream row;
    row << item.timestamp_ms << ',' << item.zed_imu_ns << ','
        << (fs::path("left") / path_l.filename()).string() << ',' << name_r;
    timeline.write_row(seq, row.str());
    const uint64_t saved = ++shared.saved;
    if (saved <= 3 || (saved % 30) == 0) {
      log_line("已保存 ZED 双目 #" + std::to_string(saved) + ": left/" +
               path_l.filename().string() +
               (name_r.empty() ? "" : (" + " + name_r)));
    }
  }
}

void csv_save_thread_func(BoundedQueue<ZedImuSample>& zed_imu_q,
                          const fs::path& zed_imu_csv_path) {
  try_avoid_last_cpu("CSV 存盘");
  try_lower_nice(5, "CSV 存盘");

  std::ofstream ofs_zed(zed_imu_csv_path);
  if (!ofs_zed) {
    log_line("无法创建 ZED IMU CSV: " + zed_imu_csv_path.string());
  } else {
    ofs_zed << "zed_imu_timestamp_ns,zed_timestamp_ms,"
               "image_sync_trigger,ax,ay,az,gx,gy,gz,qw,qx,qy,qz\n";
    ofs_zed << std::fixed << std::setprecision(8);
  }

  uint64_t saved_zed = 0;
  while (true) {
    bool progress = false;
    ZedImuSample zs;
    while (zed_imu_q.try_pop(zs)) {
      progress = true;
      if (!ofs_zed) {
        continue;
      }
      ofs_zed << zs.zed_imu_timestamp_ns << ',' << zs.zed_timestamp_ms << ','
              << zs.image_sync_trigger << ','
              << zs.accel[0] << ',' << zs.accel[1] << ',' << zs.accel[2] << ','
              << zs.gyro[0] << ',' << zs.gyro[1] << ',' << zs.gyro[2] << ','
              << zs.quat[0] << ',' << zs.quat[1] << ',' << zs.quat[2] << ','
              << zs.quat[3] << '\n';
      ++saved_zed;
      if ((saved_zed % 400) == 0) {
        ofs_zed.flush();
      }
    }
    if (progress) {
      continue;
    }
    if (zed_imu_q.finished()) {
      break;
    }
    zed_imu_q.wait_for_data(std::chrono::milliseconds(5));
  }

  if (ofs_zed) {
    ofs_zed.flush();
    log_line("ZED IMU CSV 保存结束，共 " + std::to_string(saved_zed) +
             " 行 -> " + zed_imu_csv_path.string() + "，丢弃队列 " +
             std::to_string(zed_imu_q.dropped()) + " 条");
  }
}

static bool write_time_meta(const fs::path& path, uint64_t image_rows) {
  std::ofstream ofs(path);
  if (!ofs) {
    return false;
  }
  ofs << "mode=zed_sdk_free_run\n";
  ofs << "description=grab_stereo_image_and_body_imu_from_sdk; "
         "no_fpga_trigger; no_calib_lock_first_frame\n";
  ofs << "camera_fps=" << g_camera_fps << '\n';
  ofs << "resolution=HD1080\n";
  ofs << "exposure_us=" << g_exposure_us << '\n';
  ofs << "analog_gain_mdb=" << g_analog_gain_mdb << '\n';
  ofs << "exposure_us_readback_frame30=" << g_exposure_readback.load() << '\n';
  ofs << "analog_gain_mdb_readback_frame30=" << g_gain_readback.load() << '\n';
  ofs << "digital_gain=" << kManualDigitalGain << '\n';
  ofs << "image_rows=" << image_rows << '\n';
  ofs << "note=filename_is_zed_timestamp_ms; "
         "time_is_sdk_zed_timestamp_ms_and_zed_imu_timestamp_ns\n";
  return true;
}

static bool parse_int_arg(const char* text, int lo, int hi, int* out) {
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || v < lo || v > hi) {
    return false;
  }
  *out = static_cast<int>(v);
  return true;
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--exposure-us" && i + 1 < argc &&
        parse_int_arg(argv[i + 1], 100, 60000, &g_exposure_us)) {
      ++i;
    } else if (arg == "--analog-gain-mdb" && i + 1 < argc &&
               parse_int_arg(argv[i + 1], 1000, 16000, &g_analog_gain_mdb)) {
      ++i;
    } else if (arg == "--fps" && i + 1 < argc &&
               parse_int_arg(argv[i + 1], 15, 60, &g_camera_fps) &&
               (g_camera_fps == 15 || g_camera_fps == 30 || g_camera_fps == 60)) {
      ++i;
    } else {
      std::cerr << "用法: " << argv[0]
                << " [--exposure-us 100..60000] [--analog-gain-mdb 1000..16000] [--fps 15|30|60]\n"
                << "默认: --exposure-us " << kDefaultExposureUs
                << " --analog-gain-mdb " << kDefaultAnalogGainMdB
                << " --fps " << kDefaultCameraFps << std::endl;
      return 2;
    }
  }
  try_avoid_last_cpu("主线程");
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  log_line("===== zed_record：只录 ZED 双目 + 机身 IMU（SDK 自由运行） =====");

  const fs::path save_root = exe_dir();
  const fs::path left_dir = save_root / "save_data/left";
  const fs::path right_dir = save_root / "save_data/right";
  const fs::path zed_imu_csv_path = save_root / "save_data/zed_imu.csv";
  const fs::path timeline_path = save_root / "save_data/img_timeline.csv";
  const fs::path meta_path = save_root / "save_data/time_meta.txt";
  std::error_code ec;
  fs::create_directories(left_dir, ec);
  if (ec) {
    log_line("创建目录失败: " + left_dir.string() + " : " + ec.message());
    return 1;
  }
  fs::create_directories(right_dir, ec);
  if (ec) {
    log_line("创建目录失败: " + right_dir.string() + " : " + ec.message());
    return 1;
  }
  give_save_data_back_to_user(save_root / "save_data");
  log_line("数据保存目录: " + save_root.string());
  log_line("  左目: " + left_dir.string() + "/<zed_ts_ms>.jpg");
  log_line("  右目: " + right_dir.string() + "/<zed_ts_ms>.jpg");
  log_line("  ZED IMU: " + zed_imu_csv_path.string());
  log_line("  图像时间线: " + timeline_path.string());
  log_line("时间戳直接用 SDK：zed_timestamp_ms / zed_imu_timestamp_ns。");
  log_line("曝光设定: " + std::to_string(g_exposure_us) + " us, 模拟增益: " +
           std::to_string(g_analog_gain_mdb) + " mdB, 帧率: " + std::to_string(g_camera_fps) + " fps");

  auto zed = init_zedx();
  if (!zed) {
    log_line("ZED X 初始化失败，退出。");
    return 1;
  }

  ImageTimeline timeline(timeline_path);
  BoundedQueue<ZedFrameItem> image_q(60);
  BoundedQueue<ZedFrameItem> save_q(60);
  PreviewSlot preview_slot;
  BoundedQueue<ZedImuSample> zed_imu_q(60000);

  std::atomic<bool> stop_event{false};

  const char* display_env = std::getenv("DISPLAY");
  bool enable_preview = (display_env != nullptr && display_env[0] != '\0');
  if (!enable_preview) {
    log_line("未检测到 DISPLAY，跳过实时预览（采集/存盘不受影响）。");
  } else if (!start_preview_window()) {
    enable_preview = false;
  }

  std::thread grab_th(grab_thread_func, zed.get(), std::ref(image_q),
                      std::ref(zed_imu_q), std::ref(stop_event));
  SaveShared save_shared;
  std::vector<std::thread> zed_save_ths;
  for (int k = 0; k < kSaveThreads; ++k) {
    zed_save_ths.emplace_back(zed_save_thread_func, std::ref(save_q), std::ref(save_shared),
                              std::ref(timeline), left_dir, right_dir);
  }
  std::thread csv_save_th(csv_save_thread_func, std::ref(zed_imu_q),
                          zed_imu_csv_path);

  log_line("运行中：抓图 / 存双目 / ZED IMU CSV / 预览。Ctrl+C 退出。");

  PreviewSlot* preview_ptr = enable_preview ? &preview_slot : nullptr;
  uint64_t forwarded = 0;
  while (g_running.load()) {
    ZedFrameItem item;
    const bool got = image_q.try_pop(item);
    if (got) {
      forward_frame_to_save(std::move(item), save_q, preview_ptr, forwarded);
    }
    if (enable_preview) {
      if (!pump_preview_window(preview_slot)) {
        stop_preview_window();
        enable_preview = false;
        preview_ptr = nullptr;
      }
    } else if (!got) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  log_line("用户中断，正在清理...");
  if (preview_ptr != nullptr || enable_preview) {
    stop_preview_window();
    enable_preview = false;
    preview_ptr = nullptr;
  }
  stop_event.store(true);

  if (zed) {
    zed->close();
    log_line("ZED 已关闭。");
  }

  if (grab_th.joinable()) {
    grab_th.join();
  }
  ZedFrameItem leftover;
  while (image_q.try_pop(leftover)) {
    forward_frame_to_save(std::move(leftover), save_q, nullptr, forwarded);
  }
  save_q.stop();
  image_q.stop();
  zed_imu_q.stop();
  for (auto& th : zed_save_ths) {
    if (th.joinable()) {
      th.join();
    }
  }
  log_line("ZED 存图线程结束（" + std::to_string(kSaveThreads) + " 个），共保存 " +
           std::to_string(save_shared.saved.load()) + " 对，丢弃队列 " +
           std::to_string(save_q.dropped()) + " 帧");
  if (csv_save_th.joinable()) {
    csv_save_th.join();
  }
  timeline.stop();

  if (write_time_meta(meta_path, timeline.rows())) {
    log_line("已写 time_meta.txt");
  } else {
    log_line("写入 time_meta.txt 失败: " + meta_path.string());
  }

  give_save_data_back_to_user(save_root / "save_data");
  log_line("程序结束。");
  return 0;
}
