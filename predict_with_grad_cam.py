"""
predict_with_grad_cam.py

CARLA 多任务在线测试脚本 (Grad-CAM 版本)，模型输出为：
[steer, throttle, brake]

该脚本基于 predict_new.py 修改，使用 Grad-CAM 替代反卷积方案
生成可解释的模型决策热力图。可直接配合
train_new.py 与 model_multitask.py 生成的 checkpoint 使用。
"""

import argparse
import math
import os
import time
from collections import deque
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


from model_multitask import MultiTaskNvidiaModel
from gradcam_visualization import GradCAMMaskVisualizer


class _SteerOnlyModelAdapter(torch.nn.Module):
    """将多任务模型适配为仅输出 steer 的前向接口，供 mask 可视化复用。"""

    def __init__(self, multitask_model):
        super().__init__()
        self.multitask_model = multitask_model
        self._speed_kmh = 0.0

    @property
    def conv_layers(self):
        return getattr(self.multitask_model, "conv_layers", None)

    def set_speed_kmh(self, speed_kmh):
        self._speed_kmh = float(speed_kmh)

    def forward(self, x):
        if getattr(self.multitask_model, "use_speed_input", False):
            speed_tensor = torch.full(
                (x.shape[0],),
                self._speed_kmh,
                dtype=torch.float32,
                device=x.device,
            )
            outputs = self.multitask_model(x, prev_speed_kmh=speed_tensor)
        else:
            outputs = self.multitask_model(x)
        return outputs["steer"]


# 统一仪表盘组件：将相机画面、控制量、时序曲线和状态灯汇总到单窗口。
class IntegratedDashboard:
    """单窗口统一仪表盘：展示相机画面、控制量、趋势与状态。"""

    def __init__(self, max_speed, history_size=120):
        # 1) 基础配置与滚动历史缓存。
        self.max_speed = float(max_speed)
        self.history_size = int(history_size)

        self.steer_hist = deque(maxlen=self.history_size)
        self.throttle_hist = deque(maxlen=self.history_size)
        self.brake_hist = deque(maxlen=self.history_size)
        self.speed_hist = deque(maxlen=self.history_size)
        self.fps_hist = deque(maxlen=self.history_size)
        self.latency_hist = deque(maxlen=self.history_size)
        self.conflict_hist = deque(maxlen=self.history_size)
        self.overspeed_hist = deque(maxlen=self.history_size)

    def update(self, steer, throttle, brake, speed, fps, latency_ms, conflict, overspeed):
        # 2) 每帧更新一次时序数据，供趋势图绘制。
        self.steer_hist.append(float(steer))
        self.throttle_hist.append(float(throttle))
        self.brake_hist.append(float(brake))
        self.speed_hist.append(float(speed))
        self.fps_hist.append(float(fps))
        self.latency_hist.append(float(latency_ms))
        self.conflict_hist.append(1.0 if conflict else 0.0)
        self.overspeed_hist.append(1.0 if overspeed else 0.0)

    def _draw_line_chart(self, canvas, origin, size, series, color, y_min, y_max, title):
        # 3) 通用折线图绘制函数：用于控制量、速度、FPS、时延等趋势展示。
        x0, y0 = origin
        w, h = size
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
        cv2.putText(canvas, title, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        if len(series) < 2:
            return

        vals = np.array(series, dtype=np.float32)
        vals = np.clip(vals, y_min, y_max)
        if y_max - y_min < 1e-6:
            return

        xs = np.linspace(x0 + 4, x0 + w - 4, num=len(vals)).astype(np.int32)
        ys_norm = (vals - y_min) / (y_max - y_min)
        ys = (y0 + h - 4 - ys_norm * (h - 24)).astype(np.int32)

        points = np.stack([xs, ys], axis=1).reshape((-1, 1, 2))
        cv2.polylines(canvas, [points], False, color, 2)

    def _draw_status_lamp(self, canvas, center, is_on, label, on_color):
        # 4) 圆形状态灯：用于冲突/超速等离散事件指示。
        color = on_color if is_on else (60, 60, 60)
        cv2.circle(canvas, center, 10, color, -1)
        cv2.circle(canvas, center, 10, (220, 220, 220), 1)
        cv2.putText(canvas, label, (center[0] + 15, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    def _prepare_sub_view(self, image_bgr, size, empty_text):
        # 将任意输入图像适配到子窗口尺寸；缺失时渲染占位画面。
        w, h = size
        if image_bgr is None:
            view = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(view, empty_text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1)
            return view
        return cv2.resize(image_bgr, (w, h))

    def _blit_sub_view(self, canvas, image_bgr, origin, title):
        # 在总画布上放置子窗口并绘制标题。
        x0, y0 = origin
        h, w = image_bgr.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = image_bgr
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (85, 85, 85), 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + 24), (20, 20, 20), -1)
        cv2.putText(canvas, title, (x0 + 8, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)

    def render(self, front_camera_bgr, spectator_camera_bgr, trajectory_bgr, attention_heatmap_bgr, stats, run_info):
        # 5) 渲染统一仪表盘：左侧相机，右侧指标与曲线。
        panel_w = 540
        panel_h = 720
        left_w = 960
        sub_w = left_w // 2
        sub_h = panel_h // 2

        canvas = np.zeros((panel_h, left_w + panel_w, 3), dtype=np.uint8)

        top_left = self._prepare_sub_view(spectator_camera_bgr, (sub_w, sub_h), "No spectator view")
        top_right = self._prepare_sub_view(front_camera_bgr, (sub_w, sub_h), "No front camera")
        bottom_left = self._prepare_sub_view(trajectory_bgr, (sub_w, sub_h), "No trajectory view")
        bottom_right = self._prepare_sub_view(attention_heatmap_bgr, (sub_w, sub_h), "No attention heatmap")

        self._blit_sub_view(canvas, top_left, (0, 0), "Spectator View")
        self._blit_sub_view(canvas, top_right, (sub_w, 0), "Front Camera")
        self._blit_sub_view(canvas, bottom_left, (0, sub_h), "Live Trajectory")
        self._blit_sub_view(canvas, bottom_right, (sub_w, sub_h), "Attention Heatmap")

        # 标题区域
        x_off = left_w
        cv2.rectangle(canvas, (x_off, 0), (x_off + panel_w, 56), (25, 25, 25), -1)
        cv2.putText(canvas, "E2E Driving Dashboard", (x_off + 14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
        cv2.putText(
            canvas,
            f"Town={run_info['town']}  SpeedInput={run_info['use_speed_input']}  t={stats['elapsed_s']:.1f}s",
            (x_off + 14, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 190, 190),
            1,
        )

        # 当前控制与状态数值
        cv2.putText(canvas, f"Steer:    {stats['steer']:+.3f}", (x_off + 18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 240), 2)
        cv2.putText(canvas, f"Throttle: {stats['throttle']:.3f}", (x_off + 18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 220, 40), 2)
        cv2.putText(canvas, f"Brake:    {stats['brake']:.3f}", (x_off + 18, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 90, 255), 2)

        speed_color = (40, 220, 40) if stats["speed_kmh"] <= self.max_speed else (0, 140, 255)
        cv2.putText(
            canvas,
            f"Speed:    {stats['speed_kmh']:.1f} / {self.max_speed:.1f} km/h",
            (x_off + 18, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            speed_color,
            2,
        )
        cv2.putText(canvas, f"FPS:      {stats['fps']:.1f}", (x_off + 18, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)
        cv2.putText(canvas, f"Latency:  {stats['latency_ms']:.2f} ms", (x_off + 18, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)

        # 显示分类头预测（如果提供）
        tl_label = stats.get("tl_label", None)
        tl_prob = stats.get("tl_prob", None)
        stop_prob = stats.get("stop_prob", None)
        if tl_label is not None and tl_prob is not None:
            tl_names = ["RED", "GREEN", "YELLOW", "UNKNOWN"]
            try:
                tl_text = tl_names[int(tl_label)]
            except Exception:
                tl_text = str(tl_label)
            cv2.putText(canvas, f"TrafficLight: {tl_text} ({tl_prob:.2f})", (x_off + 250, 248 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 40), 1)

        if stop_prob is not None:
            stop_str = "STOP" if float(stop_prob) >= 0.5 else "GO"
            cv2.putText(canvas, f"IsStopped: {stop_str} ({stop_prob:.2f})", (x_off + 250, 276 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 80, 80), 1)

        # 状态指示灯
        self._draw_status_lamp(canvas, (x_off + 36, 252), stats["conflict"], "Throttle-Brake Conflict", (0, 165, 255))
        self._draw_status_lamp(canvas, (x_off + 36, 282), stats["overspeed"], "Overspeed", (0, 0, 255))
        self._draw_status_lamp(canvas, (x_off + 36, 312), stats.get("red_stop_active", False), "Red-Light Stop Safety", (0, 0, 255))

        # 油门/刹车柱状条
        bar_x = x_off + 360
        cv2.putText(canvas, "T", (bar_x + 4, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1)
        cv2.rectangle(canvas, (bar_x, 94), (bar_x + 26, 214), (100, 100, 100), 1)
        th = int(120 * np.clip(stats["throttle"], 0.0, 1.0))
        cv2.rectangle(canvas, (bar_x + 1, 214 - th), (bar_x + 25, 214), (40, 220, 40), -1)

        cv2.putText(canvas, "B", (bar_x + 44, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 90, 255), 1)
        cv2.rectangle(canvas, (bar_x + 40, 94), (bar_x + 66, 214), (100, 100, 100), 1)
        bh = int(120 * np.clip(stats["brake"], 0.0, 1.0))
        cv2.rectangle(canvas, (bar_x + 41, 214 - bh), (bar_x + 65, 214), (40, 90, 255), -1)

        # 趋势图区域
        self._draw_line_chart(canvas, (x_off + 12, 320), (250, 110), self.steer_hist, (0, 240, 240), -1.0, 1.0, "Steer Trend")
        self._draw_line_chart(canvas, (x_off + 276, 320), (250, 110), self.throttle_hist, (40, 220, 40), 0.0, 1.0, "Throttle Trend")
        self._draw_line_chart(canvas, (x_off + 12, 442), (250, 110), self.brake_hist, (40, 90, 255), 0.0, 1.0, "Brake Trend")
        self._draw_line_chart(canvas, (x_off + 276, 442), (250, 110), self.speed_hist, (255, 255, 255), 0.0, max(self.max_speed * 1.3, 1.0), "Speed Trend")
        self._draw_line_chart(canvas, (x_off + 12, 564), (250, 110), self.fps_hist, (180, 180, 255), 0.0, 60.0, "FPS Trend")
        self._draw_line_chart(canvas, (x_off + 276, 564), (250, 110), self.latency_hist, (255, 180, 120), 0.0, 80.0, "Latency (ms)")

        cv2.putText(
            canvas,
            "Keys: Q Quit  S Save Dashboard Screenshot",
            (x_off + 12, 705),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (190, 190, 190),
            1,
        )

        return canvas


# CARLA 在线测试器：负责连接仿真、模型推理、控制下发与结果汇总。
class CarlaMultiTaskModelTester:
    def __init__(
        self,
        model_path,
        host="localhost",
        port=2000,
        use_speed_input=False,
        enable_mask_visualization=True,
        mask_alpha=0.35,
        mask_update_interval=1,
        save_mask_video=False,
        mask_video_path="analysis_output/mask_overlay.mp4",
        mask_video_fps=20.0,
        red_stop_tl_enter_prob=0.80,
        red_stop_tl_exit_prob=0.55,
        red_stop_release_speed_kmh=1.0,
        red_stop_min_brake=0.35,
        red_stop_max_throttle=0.0,
        red_stop_release_frames=8,
    ):
        # 1) 运行配置与 CARLA 运行时对象。
        self.host = host
        self.port = port
        self.use_speed_input = use_speed_input

        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.camera = None
        self.overview_camera = None
        self.spectator = None
        self.actors = []
        self.original_settings = None

        self.latest_image = None
        self.latest_overview_image = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

        # 实时注意力热力图（Grad-CAM）
        self.enable_mask_visualization = bool(enable_mask_visualization)
        self.mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
        self.mask_update_interval = max(1, int(mask_update_interval))
        self.save_mask_video = bool(save_mask_video)
        self.mask_video_path = Path(mask_video_path)
        self.mask_video_fps = float(mask_video_fps)
        self.mask_video_writer = None
        self.last_mask_overlay = None
        self._mask_model_adapter = None
        self.mask_visualizer = None
        self._mask_error_logged = False
        if self.enable_mask_visualization:
            try:
                self._mask_model_adapter = _SteerOnlyModelAdapter(self.model)
                self.mask_visualizer = GradCAMMaskVisualizer(
                    model=self._mask_model_adapter,
                    device=self.device,
                    overlay_alpha=self.mask_alpha,
                )
                # Apply recommended Grad-CAM defaults for sharper, focused heatmaps
                try:
                    mg = self.mask_visualizer.mask_generator
                    mg.min_spatial = 8
                    mg.postprocess_enabled = True
                    mg.post_percentile = 98.0
                    mg.post_gamma = 1.6
                    mg.post_gaussian_sigma = 1.0
                    mg.post_threshold = 0.04
                except Exception:
                    pass
                print(
                    "Grad-CAM heatmap enabled "
                    f"(alpha={self.mask_alpha:.2f}, interval={self.mask_update_interval})"
                )
            except Exception as e:
                print(f"Failed to initialize Grad-CAM heatmap: {e}")
                print("Continue without Grad-CAM visualization")
                self.enable_mask_visualization = False

        # 红灯强制停车安全门：仅由 traffic-light 分类头触发（is_stop 仅用于观测）。
        self.red_stop_tl_enter_prob = float(np.clip(red_stop_tl_enter_prob, 0.0, 1.0))
        self.red_stop_tl_exit_prob = float(np.clip(red_stop_tl_exit_prob, 0.0, 1.0))
        self.red_stop_release_speed_kmh = float(max(0.0, red_stop_release_speed_kmh))
        self.red_stop_min_brake = float(np.clip(red_stop_min_brake, 0.0, 1.0))
        self.red_stop_max_throttle = float(np.clip(red_stop_max_throttle, 0.0, 1.0))
        self.red_stop_release_frames = int(max(1, red_stop_release_frames))
        self.red_stop_active = False
        self.red_stop_release_counter = 0

        # 2) 控制平滑状态（指数平滑的历史值）。
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_long_cmd = 0.0
        self.steer_in_deadzone = True

        # 3) 会话级历史记录（用于结束时统计摘要）。
        self.history = []

    def _init_mask_video_writer(self, frame_bgr):
        """按当前仪表盘帧尺寸初始化视频写入器。"""
        try:
            self.mask_video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.mask_video_writer = cv2.VideoWriter(
                str(self.mask_video_path),
                fourcc,
                max(1.0, self.mask_video_fps),
                (width, height),
            )
            if not self.mask_video_writer.isOpened():
                print(f"Failed to open mask video writer: {self.mask_video_path}")
                self.mask_video_writer = None
            else:
                print(f"Recording dashboard video to: {self.mask_video_path}")
        except Exception as e:
            print(f"Failed to initialize mask video writer: {e}")
            self.mask_video_writer = None

    def load_model(self, model_path):
        # 加载多任务模型与 checkpoint，兼容完整 checkpoint 或纯 state_dict。
        print(f"Loading multi-task model from: {model_path}")

        model = MultiTaskNvidiaModel(
            pretrained=False,
            freeze_features=False,
            use_speed_input=self.use_speed_input,
        )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_file, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "epoch" in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if "val_total_loss" in checkpoint:
                print(f"Checkpoint val_total_loss: {checkpoint['val_total_loss']:.6f}")
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        print("Model loaded successfully")
        return model

    def _get_weather_preset(self, weather_type):
        """根据字符串类型返回对应的 CARLA 天气预设。"""
        if weather_type == "sunny":
            return carla.WeatherParameters.ClearNoon

        if weather_type == "foggy":
            return carla.WeatherParameters(
                cloudiness=10.0,
                precipitation=0.0,
                sun_altitude_angle=45.0,
                fog_density=75.0,
                fog_distance=0.0,
                wetness=0.0,
                wind_intensity=0.0,
            )

        if weather_type == "rainy":
            return carla.WeatherParameters.MidRainyNoon

        if weather_type == "night":
            return carla.WeatherParameters.ClearNight

        return carla.WeatherParameters.ClearNoon

    def _apply_weather(self, weather_type):
        try:
            self.world.set_weather(self._get_weather_preset(weather_type))
        except Exception as e:
            print(f"Warning: Failed to apply weather '{weather_type}': {e}. Reverting to sunny.")
            self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def connect_to_carla(self, town="Town01", weather_type="sunny"):
        # 连接 CARLA 并切换到同步模式，保证控制与渲染时序稳定。
        try:
            print(f"Connecting to CARLA at {self.host}:{self.port}, town={town}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)

            self.world = self.client.load_world(town)
            self.map = self.world.get_map()
            time.sleep(2)

            # 设置天气
            self._apply_weather(weather_type)
            time.sleep(2)

            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / 20.0
            self.world.apply_settings(settings)
            print("Connected to CARLA")
            return True
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False

    def spawn_vehicle(self, spawn_index=0):
        # 生成测试车辆，若目标点失败则尝试备用生成点。
        try:
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.find("vehicle.tesla.model3")
            if not vehicle_bp:
                vehicle_bp = bp_lib.filter("vehicle.*")[0]

            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points in current map")

            spawn_index = max(0, min(spawn_index, len(spawn_points) - 1))
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_index])

            if not self.vehicle:
                # 备用出生点重试
                for i in range(min(10, len(spawn_points))):
                    idx = (spawn_index + i) % len(spawn_points)
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[idx])
                    if self.vehicle:
                        spawn_index = idx
                        break

            if not self.vehicle:
                raise RuntimeError("Failed to spawn vehicle")

            self.actors.append(self.vehicle)
            print(f"Vehicle spawned at spawn point {spawn_index}")
            return True
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return False

    def spawn_camera(self):
        # 生成前视 RGB 相机，并注册回调更新最新图像帧。
        try:
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "640")
            camera_bp.set_attribute("image_size_y", "480")
            camera_bp.set_attribute("fov", "90")

            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0.0),
            )

            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.actors.append(self.camera)
            self.camera.listen(self.camera_callback)

            print("Camera spawned")
            return True
        except Exception as e:
            print(f"Failed to spawn camera: {e}")
            return False

    def spawn_overview_camera(self):
        # 生成概览相机：近似 spectator 视角，用于左上子窗口。
        try:
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "640")
            camera_bp.set_attribute("image_size_y", "480")
            camera_bp.set_attribute("fov", "100")

            overview_transform = carla.Transform(
                carla.Location(x=-8.0, z=6.0),
                carla.Rotation(pitch=-20.0),
            )

            self.overview_camera = self.world.spawn_actor(camera_bp, overview_transform, attach_to=self.vehicle)
            self.actors.append(self.overview_camera)
            self.overview_camera.listen(self.overview_camera_callback)

            print("Overview camera spawned")
            return True
        except Exception as e:
            print(f"Failed to spawn overview camera: {e}")
            return False

    def camera_callback(self, image):
        # CARLA 原始帧转 numpy，去除 alpha 通道。
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        self.latest_image = arr[:, :, :3]

    def overview_camera_callback(self, image):
        # 概览相机原始帧转 numpy，去除 alpha 通道。
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        self.latest_overview_image = arr[:, :, :3]

    def follow_vehicle_spectator(self, distance=6.0, height=2.5, pitch=-15.0):
        # 将 spectator 固定在车辆后上方，便于观察行驶表现。
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()

        cam_location = transform.location - forward * distance
        cam_location.z += height

        cam_rotation = carla.Rotation(
            pitch=pitch,
            yaw=transform.rotation.yaw,
            roll=0.0,
        )

        self.spectator.set_transform(carla.Transform(cam_location, cam_rotation))

    def _preprocess_image(self, image_bgr):
        # 推理前处理：BGR->RGB、缩放、归一化、转 Tensor。
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (200, 66))

        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std

        image_tensor = torch.from_numpy(image_normalized).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        return image_tensor, image_resized

    def predict_controls(self, image_bgr, current_speed_kmh, return_inference_bundle=False):
        # 模型推理：输出并裁剪为有效控制范围。
        image_tensor, image_resized = self._preprocess_image(image_bgr)
        
        # 如果需要 inference_bundle（用于 Grad-CAM），则不禁用梯度
        if return_inference_bundle:
            # 保存启用梯度的张量用于 Grad-CAM 计算
            image_tensor_for_gradcam = image_tensor.clone().detach().requires_grad_(True)
            
            # 推理时仍然禁用梯度以加快速度，但保留 image_tensor_for_gradcam 用于后续 Grad-CAM
            with torch.no_grad():
                if self.use_speed_input:
                    speed_tensor = torch.tensor([current_speed_kmh], dtype=torch.float32, device=self.device)
                    outputs = self.model(image_tensor, prev_speed_kmh=speed_tensor)
                else:
                    outputs = self.model(image_tensor)
        else:
            with torch.no_grad():
                if self.use_speed_input:
                    speed_tensor = torch.tensor([current_speed_kmh], dtype=torch.float32, device=self.device)
                    outputs = self.model(image_tensor, prev_speed_kmh=speed_tensor)
                else:
                    outputs = self.model(image_tensor)
            image_tensor_for_gradcam = None

        # model returns dict with tensors shaped (B,) or (B,num_classes)
        steer = float(outputs["steer"][0].item())
        throttle = float(outputs["throttle"][0].item())
        brake = float(outputs["brake"][0].item())

        # traffic-light and stop predictions (probabilities)
        tl_label = None
        tl_prob = None
        if "tl_logits" in outputs and outputs["tl_logits"] is not None:
            tl_logits = outputs["tl_logits"]
            tl_probs = torch.softmax(tl_logits, dim=1)
            topv, topi = torch.max(tl_probs, dim=1)
            tl_label = int(topi[0].item())
            tl_prob = float(topv[0].item())

        stop_prob = None
        if "stop_logit" in outputs and outputs["stop_logit"] is not None:
            stop_logit = outputs["stop_logit"]
            stop_prob = float(torch.sigmoid(stop_logit)[0].item())

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        inference_bundle = None
        if return_inference_bundle:
            inference_bundle = {
                "image_tensor": image_tensor_for_gradcam,
                "resized_rgb": image_resized,
            }

        return steer, throttle, brake, tl_label, tl_prob, stop_prob, inference_bundle

    def _render_attention_heatmap_view(self, inference_bundle, current_speed_kmh, frame_count, width=480, height=360):
        # 在主循环中生成/复用实时注意力热力图，失败时平滑退化为占位图。
        if not self.enable_mask_visualization or self.mask_visualizer is None:
            view = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(view, "Grad-CAM disabled", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1)
            return view

        should_update = (frame_count % self.mask_update_interval == 0) or (self.last_mask_overlay is None)
        if should_update and inference_bundle is not None:
            try:
                if self._mask_model_adapter is not None:
                    self._mask_model_adapter.set_speed_kmh(current_speed_kmh)

                # 验证张量数据
                tensor_data = inference_bundle.get("image_tensor")
                resized_rgb = inference_bundle.get("resized_rgb")
                
                if tensor_data is None:
                    raise ValueError("image_tensor missing from inference_bundle")
                if resized_rgb is None:
                    raise ValueError("resized_rgb missing from inference_bundle")
                
                # 验证张量形状和设备
                if not isinstance(tensor_data, torch.Tensor):
                    raise TypeError(f"image_tensor must be torch.Tensor, got {type(tensor_data)}")
                
                if tensor_data.dim() != 4:
                    raise ValueError(f"image_tensor must be 4D (batch, channel, height, width), got shape {tensor_data.shape}")

                mask_output = self.mask_visualizer.build_overlay_from_model_input(
                    model_input_tensor=tensor_data,
                    resized_rgb=resized_rgb,
                )
                self.last_mask_overlay = mask_output["overlay_bgr"]
            except Exception as e:
                if not self._mask_error_logged:
                    import traceback
                    print(f"[ERROR] Grad-CAM generation failed: {e}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    self._mask_error_logged = True

        if self.last_mask_overlay is None:
            view = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(view, "Waiting for Grad-CAM heatmap...", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1)
            return view

        return cv2.resize(self.last_mask_overlay, (width, height), interpolation=cv2.INTER_LINEAR)

    def _post_process_control(
        self,
        steer,
        throttle,
        brake,
        current_speed,
        max_speed,
        steer_smooth_alpha,
        long_smooth_alpha,
        steer_deadzone_enter,
        steer_deadzone_exit,
        steer_small_angle_th,
        steer_small_rate_limit,
        steer_large_rate_limit,
    ):
        # 控制后处理：转向使用动态低通+滞回死区+分段限速，纵向继续统一滤波。
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        # 1) 转向控制：动态低通 + 滞回死区 + 分段变化率限制。
        steer_alpha_base = float(np.clip(steer_smooth_alpha, 0.0, 0.95))    # 把 steer_smooth_alpha 限制到区间 [0.0, 0.95]
        steer_small_angle_th = float(np.clip(steer_small_angle_th, 0.05, 0.35)) # 小角度阈值限制在 [0.05, 0.35]
        if abs(steer) > steer_small_angle_th:   # 如果转向角大于小角度阈值，则认为是大角度转弯
            # 大角度转弯阶段降低平滑强度，减少入弯迟滞。
            steer_alpha = max(0.15, steer_alpha_base - 0.25)
        else:
            # 否则认为是小角度转弯，平滑强度不变
            steer_alpha = steer_alpha_base

        """
        一阶低通滤波：当前输出 = α * 上一输出 + (1-α) * 当前原始输入，限制到[-1,1]范围。
        α (即steer_alpha)越大越平滑但响应越慢。
        """
        steer_lp = float(np.clip(steer_alpha * self.prev_steer + (1.0 - steer_alpha) * steer, -1.0, 1.0))

        # 限制死区阈值，确保合理范围且 exit 大于 enter。
        steer_deadzone_enter = float(np.clip(steer_deadzone_enter, 0.0, 0.25))
        steer_deadzone_exit = float(np.clip(steer_deadzone_exit, steer_deadzone_enter + 1e-3, 0.35))

        """
        # Schmitt 触发器（带滞回）实现说明：
        死区是一个steer被认定为噪声的区域，当steer_lp< enter时，认为太小是噪声，进入死区，
        直到steer升高到比exit还大时才认为是有效转向信号，离开死区。
        # - 标志含义：`self.steer_in_deadzone` 为状态位，True 表示当前被判定为处于死区，
        #   False 表示处于正常跟随状态（按滤波后信号跟随）。
        # - 滞回阈值：使用两个阈值构成滞回回路，`steer_deadzone_enter`（进入阈值，较小）
        #   与 `steer_deadzone_exit`（退出阈值，较大），能够避免阈值附近的频繁震荡（抖动）。
        # - 行为要点：
        #   1) 若当前已在死区（标志为 True）：
        #      - 当 `abs(steer_lp) < steer_deadzone_exit` 时，继续认为处于死区。
        #        为避免突然从小值跳回完整值，采用“软死区”策略：将 `steer_lp`
        #        按比例缩放到更接近 0 的值作为 `steer_target`（factor = abs(steer_lp)/steer_deadzone_exit）。
        #      - 若 `abs(steer_lp)` 超过退出阈值，则清除死区标志（设为 False），
        #        并将 `steer_target` 设为当前的 `steer_lp`（恢复正常跟随）。
        #   2) 若当前不在死区（标志为 False）：
        #      - 当 `abs(steer_lp) < steer_deadzone_enter` 时，判定为进入死区，
        #        将标志设为 True 并（当前帧）把 `steer_target` 同样按照软死区处理，以抑制微小噪声；
        #      - 否则保持正常跟随（`steer_target = steer_lp`）。
        # - 额外说明：缩放时使用 `eps` 防止除零；最终的 `steer_target` 会经过
        #   后续的速率限制（`steer_delta` 基于 `self.prev_steer`）进行限制和平滑，
        #   因此不会立即导致舵角的瞬时突变。
        """
        eps = 1e-6
        if self.steer_in_deadzone:  # 已处于死区
            # 若仍低于退出阈值：继续保持死区态势，按比例缩小 steer_lp（软死区过渡）
            if abs(steer_lp) < steer_deadzone_exit:
                factor = abs(steer_lp) / (steer_deadzone_exit + eps)
                steer_target = steer_lp * factor
            # 否则退出死区，恢复正常跟随
            else:
                self.steer_in_deadzone = False
                steer_target = steer_lp
        else:  # 当前不在死区
            # 若小于进入阈值：判定为进入死区（抑制噪声），标志设 True，当前目标置 0
            if abs(steer_lp) < steer_deadzone_enter:
                self.steer_in_deadzone = True
                factor = abs(steer_lp) / (steer_deadzone_exit + eps)
                steer_target = steer_lp * factor
            # 否则保持正常跟随（使用滤波后的 steer_lp）
            else:
                steer_target = steer_lp

        # 把小角度时的转向变化率限制在 steer_small_rate_limit，大角度时限制在 steer_large_rate_limit，确保小幅调整更平滑，大幅转向更敏捷。
        steer_small_rate_limit = float(np.clip(steer_small_rate_limit, 0.0, 0.20))
        steer_large_rate_limit = float(np.clip(steer_large_rate_limit, steer_small_rate_limit + 1e-3, 0.50))

        if abs(steer_target) < steer_small_angle_th:
            steer_max_delta = steer_small_rate_limit
        else:
            steer_max_delta = steer_large_rate_limit

        steer_delta = float(np.clip(steer_target - self.prev_steer, -steer_max_delta, steer_max_delta))
        steer = float(np.clip(self.prev_steer + steer_delta, -1.0, 1.0))

        # 2) 纵向控制：将 throttle/brake 合成为单一加速度指令，统一平滑后再拆分。
        raw_long_cmd = float(np.clip(throttle - brake, -1.0, 1.0))
        long_activity = float(np.clip(max(abs(raw_long_cmd), brake), 0.0, 1.0))
        long_alpha_base = float(np.clip(long_smooth_alpha + 0.10, 0.0, 0.98))
        long_alpha = float(np.clip(long_alpha_base - 0.50 * long_activity, 0.20, 0.98))

        # 制动事件优先响应（如红灯前减速）：额外降低滤波强度。
        if brake > 0.22 and raw_long_cmd < self.prev_long_cmd:
            long_alpha = min(long_alpha, 0.32)

        long_cmd = long_alpha * self.prev_long_cmd + (1.0 - long_alpha) * raw_long_cmd

        # 小幅噪声直接忽略，避免油门/刹车轻微抖动导致走走停停。
        long_deadzone = 0.03
        if abs(long_cmd) < long_deadzone:
            long_cmd = 0.0

        # 超速保护：接近上限就禁止继续加速，超速时施加最小制动倾向。
        if current_speed > max_speed * 0.99:
            long_cmd = min(long_cmd, 0.0)
        if current_speed > max_speed:
            overspeed = min((current_speed - max_speed) / max(max_speed, 1e-3), 1.0)
            long_cmd = min(long_cmd, -(0.12 + 0.28 * overspeed))

        # 限制纵向指令变化率，平稳时更稳，急加减速时允许更快跟随。
        long_max_delta = 0.08 + 0.22 * long_activity
        if raw_long_cmd < self.prev_long_cmd:
            long_max_delta += 0.06
        long_delta = float(np.clip(long_cmd - self.prev_long_cmd, -long_max_delta, long_max_delta))
        long_cmd = float(np.clip(self.prev_long_cmd + long_delta, -1.0, 1.0))

        throttle = max(0.0, long_cmd)
        brake = max(0.0, -long_cmd)

        # 3) 踏板死区 + 互斥抑制：进一步清除微小抖动和冲突。
        pedal_deadzone = 0.04
        if throttle < pedal_deadzone:
            throttle = 0.0
        if brake < pedal_deadzone:
            brake = 0.0

        if throttle > 0.0 and brake > 0.0:
            if throttle >= brake:
                brake = 0.0
            else:
                throttle = 0.0

        self.prev_steer = steer
        self.prev_throttle = throttle
        self.prev_brake = brake
        self.prev_long_cmd = long_cmd

        return steer, throttle, brake

    def _apply_traffic_light_stop_safety(self, throttle, brake, tl_label, tl_prob, current_speed):
        # 红灯停车安全逻辑：
        # - 仅使用 traffic-light 分支：高阈值进入、低阈值保持（滞回）
        # - 仅在“低速+连续多帧”时释放，防止短暂识别抖动导致误放行
        is_red = (tl_label is not None) and (int(tl_label) == 0)
        tl_conf = float(tl_prob) if tl_prob is not None else 0.0

        enter_signal = is_red and tl_conf >= self.red_stop_tl_enter_prob
        hold_signal = is_red and tl_conf >= self.red_stop_tl_exit_prob

        if not self.red_stop_active:
            if enter_signal:
                self.red_stop_active = True
                self.red_stop_release_counter = 0
        else:
            can_release = (not hold_signal) and (current_speed <= self.red_stop_release_speed_kmh)
            if can_release:
                self.red_stop_release_counter += 1
                if self.red_stop_release_counter >= self.red_stop_release_frames:
                    self.red_stop_active = False
                    self.red_stop_release_counter = 0
            else:
                self.red_stop_release_counter = 0

        if self.red_stop_active:
            # 强制停车态：禁止（或强抑制）油门，并设置速度相关最小制动下限。
            throttle = min(float(throttle), self.red_stop_max_throttle)
            speed_ratio = float(np.clip(current_speed / 30.0, 0.0, 1.0))
            target_brake = self.red_stop_min_brake + 0.35 * speed_ratio
            brake = max(float(brake), target_brake)

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        return throttle, brake, self.red_stop_active

    def create_control_visualization(self, steer, throttle, brake, speed_kmh, max_speed):
        # 独立控制面板（保留方法，当前主流程使用统一仪表盘渲染）。
        viz = np.zeros((320, 480, 3), dtype=np.uint8)

        center = (110, 160)
        radius = 80
        cv2.circle(viz, center, radius, (120, 120, 120), 2)

        angle = steer * 90.0
        end_x = int(center[0] + radius * 0.8 * np.sin(np.radians(angle)))
        end_y = int(center[1] - radius * 0.8 * np.cos(np.radians(angle)))
        cv2.line(viz, center, (end_x, end_y), (0, 255, 255), 4)

        # 油门柱状条
        cv2.putText(viz, "Throttle", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(viz, (220, 70), (250, 250), (100, 100, 100), 1)
        t_h = int(180 * np.clip(throttle, 0.0, 1.0))
        cv2.rectangle(viz, (220, 250 - t_h), (250, 250), (0, 200, 0), -1)

        # 刹车柱状条
        cv2.putText(viz, "Brake", (290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(viz, (290, 70), (320, 250), (100, 100, 100), 1)
        b_h = int(180 * np.clip(brake, 0.0, 1.0))
        cv2.rectangle(viz, (290, 250 - b_h), (320, 250), (0, 0, 220), -1)

        cv2.putText(viz, f"Steer: {steer:+.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        cv2.putText(viz, f"Throttle: {throttle:.3f}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        cv2.putText(viz, f"Brake: {brake:.3f}", (20, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

        color = (0, 255, 0) if speed_kmh <= max_speed else (0, 100, 255)
        cv2.putText(viz, f"Speed: {speed_kmh:.1f} / {max_speed:.1f} km/h", (220, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return viz

    def _get_map_waypoints_xy(self, step=2.0):
        # 尝试提取地图路网采样点，用于轨迹图背景。
        if self.map is None:
            return np.array([]), np.array([])

        try:
            waypoints = self.map.generate_waypoints(float(step))
            if not waypoints:
                return np.array([]), np.array([])

            road_x = np.array([wp.transform.location.x for wp in waypoints], dtype=np.float32)
            road_y = np.array([wp.transform.location.y for wp in waypoints], dtype=np.float32)
            return road_x, road_y
        except Exception as e:
            print(f"Failed to build road waypoint background: {e}")
            return np.array([]), np.array([])

    def _render_live_trajectory_view(self, road_x, road_y, width=480, height=360):
        # 实时轨迹视图：蓝色车辆轨迹叠加黄色路网。
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        margin = 20

        if not self.history:
            cv2.putText(canvas, "Waiting for trajectory...", (14, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            return canvas

        traj_x = np.array([item.get("x", np.nan) for item in self.history], dtype=np.float32)
        traj_y = np.array([item.get("y", np.nan) for item in self.history], dtype=np.float32)
        # CARLA/UE 使用左手坐标系，显示时将 y 取反以匹配直观转向方向。
        traj_y = -traj_y

        valid_traj = np.isfinite(traj_x) & np.isfinite(traj_y)
        traj_x = traj_x[valid_traj]
        traj_y = traj_y[valid_traj]
        if traj_x.size < 1:
            cv2.putText(canvas, "Trajectory unavailable", (14, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            return canvas

        x_min = float(np.min(traj_x))
        x_max = float(np.max(traj_x))
        y_min = float(np.min(traj_y))
        y_max = float(np.max(traj_y))

        span_x = max(x_max - x_min, 20.0)
        span_y = max(y_max - y_min, 20.0)
        x_pad = 0.30 * span_x + 8.0
        y_pad = 0.30 * span_y + 8.0

        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # 调整边界比例，防止坐标轴缩放畸变。
        plot_w = max(width - 2 * margin, 2)
        plot_h = max(height - 2 * margin, 2)
        target_ratio = float(plot_w) / float(plot_h)
        span_x = max(x_max - x_min, 1e-3)
        span_y = max(y_max - y_min, 1e-3)

        current_ratio = span_x / span_y
        if current_ratio > target_ratio:
            desired_y = span_x / target_ratio
            delta = 0.5 * max(desired_y - span_y, 0.0)
            y_min -= delta
            y_max += delta
        else:
            desired_x = span_y * target_ratio
            delta = 0.5 * max(desired_x - span_x, 0.0)
            x_min -= delta
            x_max += delta

        def _map_xy(x_vals, y_vals):
            u = margin + (x_vals - x_min) / max(x_max - x_min, 1e-6) * float(plot_w)
            v = height - margin - (y_vals - y_min) / max(y_max - y_min, 1e-6) * float(plot_h)
            return u, v

        # 绘制局部路网背景（黄色）。
        if road_x.size > 0 and road_y.size > 0:
            valid_road = np.isfinite(road_x) & np.isfinite(road_y)
            rx = road_x[valid_road]
            ry = -road_y[valid_road]
            if rx.size > 0:
                road_mask = (rx >= x_min) & (rx <= x_max) & (ry >= y_min) & (ry <= y_max)
                rx = rx[road_mask]
                ry = ry[road_mask]
                if rx.size > 0:
                    u, v = _map_xy(rx, ry)
                    ui = np.clip(np.round(u).astype(np.int32), 0, width - 1)
                    vi = np.clip(np.round(v).astype(np.int32), 0, height - 1)
                    canvas[vi, ui] = (0, 255, 255)

        # 绘制轨迹（蓝色）与起止点。
        tu, tv = _map_xy(traj_x, traj_y)
        tpts = np.stack([np.round(tu).astype(np.int32), np.round(tv).astype(np.int32)], axis=1).reshape((-1, 1, 2))
        if len(tpts) >= 2:
            cv2.polylines(canvas, [tpts], isClosed=False, color=(255, 0, 0), thickness=2)
        if len(tpts) >= 1:
            cv2.circle(canvas, tuple(tpts[0, 0]), 4, (0, 255, 0), -1)
            cv2.circle(canvas, tuple(tpts[-1, 0]), 4, (0, 0, 255), -1)

        cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (70, 70, 70), 1)
        return canvas

    def _render_live_steer_history_view(self, width=480, height=360):
        # 实时完整 steer 历史图：x 轴随仿真推进自动缩放，保证全历史可见。
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        margin = 30
        plot_w = max(width - 2 * margin, 2)
        plot_h = max(height - 2 * margin, 2)

        cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (70, 70, 70), 1)
        cv2.line(canvas, (margin, height // 2), (width - margin, height // 2), (55, 55, 55), 1)

        if not self.history:
            cv2.putText(canvas, "Waiting for steer history...", (15, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            return canvas

        time_vals = np.array([item.get("elapsed_s", item.get("sim_time", 0.0)) for item in self.history], dtype=np.float32)
        steer_vals = np.array([item.get("steer", 0.0) for item in self.history], dtype=np.float32)

        valid = np.isfinite(time_vals) & np.isfinite(steer_vals)
        time_vals = time_vals[valid]
        steer_vals = steer_vals[valid]

        if time_vals.size < 1:
            cv2.putText(canvas, "Invalid steer history", (15, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            return canvas

        # 降采样到接近像素宽度，避免历史很长时绘制过慢。
        if len(time_vals) > plot_w:
            idx = np.linspace(0, len(time_vals) - 1, num=plot_w).astype(np.int32)
            time_draw = time_vals[idx]
            steer_draw = steer_vals[idx]
        else:
            time_draw = time_vals
            steer_draw = steer_vals

        t0 = float(time_draw[0])
        t1 = float(max(time_draw[-1], t0 + 1e-3))
        y0 = -1.0
        y1 = 1.0

        u = margin + (time_draw - t0) / (t1 - t0) * float(plot_w)
        v = margin + (y1 - steer_draw) / (y1 - y0) * float(plot_h)

        pts = np.stack([np.round(u).astype(np.int32), np.round(v).astype(np.int32)], axis=1).reshape((-1, 1, 2))
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], isClosed=False, color=(255, 220, 0), thickness=2)

        cv2.putText(canvas, f"t=[{t0:.1f}, {t1:.1f}] s", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
        cv2.putText(canvas, f"steer_now={steer_vals[-1]:+.3f}", (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
        return canvas

    def _format_result_name_value(self, value):
        # 将参数值转换为紧凑字符串，便于写入文件名。
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    def show_post_run_plots(
        self,
        spawn_index,
        max_speed,
        steer_smooth_alpha,
        steer_deadzone_enter,
        steer_deadzone_exit,
        steer_small_angle_th,
        save_result_image,
    ):
        # 仿真结束后展示轨迹、速度与控制量的汇总图窗。
        if plt is None:
            print("matplotlib is not available, skip post-run plots")
            return
        if not self.history:
            print("No history recorded, skip post-run plots")
            return

        # 统一使用墙钟经过时间（与 run_test 的 duration 和 dashboard elapsed_s 一致）。
        time_axis = np.array(
            [item.get("elapsed_s", item.get("sim_time", 0.0)) for item in self.history],
            dtype=np.float32,
        )
        x_vals = np.array([item.get("x", np.nan) for item in self.history], dtype=np.float32)
        y_vals = np.array([item.get("y", np.nan) for item in self.history], dtype=np.float32)
        # 与实时轨迹视图保持一致：绘图时使用 (x, -y)。
        y_vals_plot = -y_vals
        speed_vals = np.array([item.get("speed", 0.0) for item in self.history], dtype=np.float32)
        steer_vals = np.array([item.get("steer", 0.0) for item in self.history], dtype=np.float32)
        throttle_vals = np.array([item.get("throttle", 0.0) for item in self.history], dtype=np.float32)
        brake_vals = np.array([item.get("brake", 0.0) for item in self.history], dtype=np.float32)

        road_x, road_y = self._get_map_waypoints_xy(step=2.0)
        road_y_plot = -road_y

        fig = plt.figure(figsize=(18, 9))
        gs = fig.add_gridspec(2, 3)

        ax_traj = fig.add_subplot(gs[0, :2])
        ax_speed = fig.add_subplot(gs[0, 2])
        ax_steer = fig.add_subplot(gs[1, 0])
        ax_throttle = fig.add_subplot(gs[1, 1])
        ax_brake = fig.add_subplot(gs[1, 2])

        # 第一行 1：车辆轨迹（可选叠加地图路网采样点）。
        if road_x.size > 0 and road_y_plot.size > 0:
            ax_traj.scatter(road_x, road_y_plot, s=1.0, c="yellow", alpha=0.45, label="Map roads")
        ax_traj.plot(x_vals, y_vals_plot, color="blue", linewidth=2.0, label="Vehicle trajectory")
        if x_vals.size > 0 and y_vals_plot.size > 0:
            ax_traj.scatter([x_vals[0]], [y_vals_plot[0]], c="lime", s=40, marker="o", label="Start")
            ax_traj.scatter([x_vals[-1]], [y_vals_plot[-1]], c="red", s=40, marker="x", label="End")
        ax_traj.set_title("Trajectory (x-y)")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.axis("equal")
        ax_traj.legend(loc="best")

        # 第一行 2：速度变化。
        ax_speed.plot(time_axis, speed_vals, color="tab:orange", linewidth=1.8)
        ax_speed.set_title("Speed vs Elapsed Time")
        ax_speed.set_xlabel("Elapsed Time (s)")
        ax_speed.set_ylabel("Speed (km/h)")
        ax_speed.grid(True, alpha=0.3)

        # 第二行：控制量变化。
        ax_steer.plot(time_axis, steer_vals, color="tab:cyan", linewidth=1.6)
        ax_steer.set_title("Steer vs Elapsed Time")
        ax_steer.set_xlabel("Elapsed Time (s)")
        ax_steer.set_ylabel("Steer")
        ax_steer.set_ylim(-1.05, 1.05)
        ax_steer.grid(True, alpha=0.3)

        ax_throttle.plot(time_axis, throttle_vals, color="tab:green", linewidth=1.6)
        ax_throttle.set_title("Throttle vs Elapsed Time")
        ax_throttle.set_xlabel("Elapsed Time (s)")
        ax_throttle.set_ylabel("Throttle")
        ax_throttle.set_ylim(-0.02, 1.02)
        ax_throttle.grid(True, alpha=0.3)

        ax_brake.plot(time_axis, brake_vals, color="tab:red", linewidth=1.6)
        ax_brake.set_title("Brake vs Elapsed Time")
        ax_brake.set_xlabel("Elapsed Time (s)")
        ax_brake.set_ylabel("Brake")
        ax_brake.set_ylim(-0.02, 1.02)
        ax_brake.grid(True, alpha=0.3)

        fig.suptitle("CARLA Run Summary")
        fig.tight_layout()
        if save_result_image:
            # 仅在自然结束（到达仿真时长）时保存结果图。
            timestamp_str = time.strftime("%Y%m%d%H%M")
            save_dir = Path("analysis_output") / "MultitaskModel2_analysis"
            save_dir.mkdir(parents=True, exist_ok=True)

            fname = (
                f"SimulationResult_{timestamp_str}"
                f"_spawnpoint[{self._format_result_name_value(spawn_index)}]"
                f"_max_speed[{self._format_result_name_value(max_speed)}]"
                f"_steer_smooth_alpha[{self._format_result_name_value(steer_smooth_alpha)}]"
                f"_steer_deadzone_enter[{self._format_result_name_value(steer_deadzone_enter)}]"
                f"_steer_deadzone_exit[{self._format_result_name_value(steer_deadzone_exit)}]"
                f"_steer_small_angle_th[{self._format_result_name_value(steer_small_angle_th)}].png"
            )
            save_path = save_dir / fname
            save_path_str = str(save_path.resolve())
            # Windows 默认路径长度上限约 260，超长时使用 \\?\ 前缀以保留完整命名格式。
            if os.name == "nt" and len(save_path_str) >= 240 and not save_path_str.startswith("\\\\?\\"):
                save_path_str = "\\\\?\\" + save_path_str

            fig.savefig(save_path_str, dpi=180)
            print(f"Saved simulation result figure: {save_path}")
        else:
            print("Run interrupted before duration reached, skip saving result figure")

        print("Showing post-run summary plots, close the figure window to continue cleanup")
        plt.show()

    def run_test(
        self,
        duration=120,
        spawn_index=0,
        town="Town01",
        weather_type="sunny",
        max_speed=40.0,
        steer_smooth_alpha=0.55,
        long_smooth_alpha=0.55,
        steer_deadzone_enter=0.05,
        steer_deadzone_exit=0.08,
        steer_small_angle_th=0.12,
        steer_small_rate_limit=0.02,
        steer_large_rate_limit=0.10,
    ):
        # 主测试循环：仿真步进 -> 推理 -> 后处理 -> 下发控制 -> 仪表盘渲染。
        if not self.connect_to_carla(town, weather_type):
            return False
        if not self.spawn_vehicle(spawn_index):
            return False
        if not self.spawn_camera():
            return False
        if not self.spawn_overview_camera():
            print("Warning: overview camera unavailable, spectator subview will be empty")

        road_x, road_y = self._get_map_waypoints_xy(step=2.0)

        print("Starting multi-task autonomous test")
        print(
            f"Duration: {duration}s, Max speed: {max_speed} km/h, Use speed input: {self.use_speed_input}"
        )
        print(f"Weather type: {weather_type}")
        print(
            f"Steer smooth alpha: {steer_smooth_alpha:.2f}, "
            f"Long smooth alpha: {long_smooth_alpha:.2f}"
        )
        print(
            f"Steer deadzone enter/exit: {steer_deadzone_enter:.3f}/{steer_deadzone_exit:.3f}, "
            f"small-angle threshold: {steer_small_angle_th:.3f}"
        )
        print(
            f"Steer rate limit small/large: {steer_small_rate_limit:.3f}/{steer_large_rate_limit:.3f} per tick"
        )
        print(
            f"Attention heatmap: {self.enable_mask_visualization} "
            f"(interval={self.mask_update_interval}, alpha={self.mask_alpha:.2f})"
        )
        print(
            f"Save mask video: {self.save_mask_video} "
            f"(path={self.mask_video_path}, fps={self.mask_video_fps:.1f})"
        )
        print(
            "Red-stop safety: "
            f"tl_enter/exit={self.red_stop_tl_enter_prob:.2f}/{self.red_stop_tl_exit_prob:.2f}, "
            f"release_speed={self.red_stop_release_speed_kmh:.2f} km/h, "
            f"release_frames={self.red_stop_release_frames}"
        )
        print("Press Q to quit, S to save screenshot")

        dashboard = IntegratedDashboard(max_speed=max_speed, history_size=120)
        cv2.namedWindow("E2E Driving Dashboard", cv2.WINDOW_AUTOSIZE)

        start_time = time.time()
        frame_count = 0
        screenshot_count = 0
        user_requested_quit = False
        save_result_image = False
        try:
            while time.time() - start_time < duration:
                # 1) 同步步进与跟车视角更新。
                self.world.tick()
                self.spectator = self.world.get_spectator()
                self.follow_vehicle_spectator()

                if self.latest_image is None:
                    continue

                velocity = self.vehicle.get_velocity()
                current_speed = 3.6 * math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
                elapsed = time.time() - start_time

                veh_transform = self.vehicle.get_transform()
                veh_x = float(veh_transform.location.x)
                veh_y = float(veh_transform.location.y)

                # 2) 推理并记录单帧推理时延。
                pred_t0 = time.perf_counter()
                (
                    steer,
                    throttle,
                    brake,
                    tl_label,
                    tl_prob,
                    stop_prob,
                    inference_bundle,
                ) = self.predict_controls(
                    self.latest_image,
                    current_speed,
                    return_inference_bundle=self.enable_mask_visualization,
                )
                latency_ms = (time.perf_counter() - pred_t0) * 1000.0

                # 首帧初始化历史状态，避免刚启动时 prev_* 为 0 导致过强滤波和响应滞后
                if frame_count == 0:
                    self.prev_steer = float(np.clip(steer, -1.0, 1.0))
                    self.prev_throttle = float(np.clip(throttle, 0.0, 1.0))
                    self.prev_brake = float(np.clip(brake, 0.0, 1.0))
                    self.prev_long_cmd = float(np.clip(throttle - brake, -1.0, 1.0))
                    self.steer_in_deadzone = abs(self.prev_steer) < steer_deadzone_enter

                # 3) 后处理并生成可执行控制。
                conflict_before = throttle > 0.2 and brake > 0.2
                steer, throttle, brake = self._post_process_control(
                    steer,
                    throttle,
                    brake,
                    current_speed,
                    max_speed,
                    steer_smooth_alpha,
                    long_smooth_alpha,
                    steer_deadzone_enter,
                    steer_deadzone_exit,
                    steer_small_angle_th,
                    steer_small_rate_limit,
                    steer_large_rate_limit,
                )

                # 3.1) 红灯强制停车安全门：在平滑后、下发控制前执行。
                throttle, brake, red_stop_active = self._apply_traffic_light_stop_safety(
                    throttle=throttle,
                    brake=brake,
                    tl_label=tl_label,
                    tl_prob=tl_prob,
                    current_speed=current_speed,
                )
                overspeed = current_speed > max_speed

                # 4) 下发车辆控制。
                control = carla.VehicleControl(
                    throttle=float(np.clip(throttle, 0.0, 1.0)),
                    steer=float(np.clip(steer, -1.0, 1.0)),
                    brake=float(np.clip(brake, 0.0, 1.0)),
                )
                self.vehicle.apply_control(control)

                # 5) 更新统计与仪表盘数据源。
                fps = frame_count / max(elapsed, 1e-6) if frame_count > 0 else 0.0

                dashboard.update(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    speed=current_speed,
                    fps=fps,
                    latency_ms=latency_ms,
                    conflict=conflict_before,
                    overspeed=overspeed,
                )

                self.history.append(
                    {
                        "timestamp": time.time(),
                        "elapsed_s": elapsed,
                        "x": veh_x,
                        "y": veh_y,
                        "speed": current_speed,
                        "steer": steer,
                        "throttle": throttle,
                        "brake": brake,
                        "latency_ms": latency_ms,
                        "conflict": 1.0 if conflict_before else 0.0,
                        "overspeed": 1.0 if overspeed else 0.0,
                        "red_stop_active": 1.0 if red_stop_active else 0.0,
                    }
                )

                live_trajectory_view = self._render_live_trajectory_view(road_x, road_y, width=480, height=360)
                live_attention_heatmap_view = self._render_attention_heatmap_view(
                    inference_bundle=inference_bundle,
                    current_speed_kmh=current_speed,
                    frame_count=frame_count,
                    width=480,
                    height=360,
                )

                # 6) 渲染统一仪表盘并显示。
                dashboard_frame = dashboard.render(
                    front_camera_bgr=self.latest_image,
                    spectator_camera_bgr=self.latest_overview_image,
                    trajectory_bgr=live_trajectory_view,
                    attention_heatmap_bgr=live_attention_heatmap_view,
                    stats={
                        "elapsed_s": elapsed,
                        "steer": steer,
                        "throttle": throttle,
                        "brake": brake,
                        "speed_kmh": current_speed,
                        "fps": fps,
                        "latency_ms": latency_ms,
                        "conflict": conflict_before,
                        "overspeed": overspeed,
                        "red_stop_active": red_stop_active,
                        "tl_label": tl_label,
                        "tl_prob": tl_prob,
                        "stop_prob": stop_prob,
                    },
                    run_info={
                        "town": town,
                        "use_speed_input": self.use_speed_input,
                    },
                )

                cv2.imshow("E2E Driving Dashboard", dashboard_frame)

                if self.save_mask_video and dashboard_frame is not None:
                    if self.mask_video_writer is None:
                        self._init_mask_video_writer(dashboard_frame)
                    if self.mask_video_writer is not None:
                        self.mask_video_writer.write(dashboard_frame)

                # 7) 键盘交互：退出或保存当前仪表盘截图。
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("User requested quit")
                    user_requested_quit = True
                    break
                if key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"carla_multitask_dashboard_{ts}_{screenshot_count}.png"
                    cv2.imwrite(fname, dashboard_frame)
                    print(f"Saved screenshot: {fname}")
                    screenshot_count += 1

                frame_count += 1
                if frame_count % 50 == 0:
                    fps = frame_count / max(elapsed, 1e-6)
                    print(
                        f"t={elapsed:6.1f}s | fps={fps:5.1f} | speed={current_speed:5.1f} | "
                        f"steer={steer:+.3f} | throttle={throttle:.3f} | brake={brake:.3f} | "
                        f"red_stop={'Y' if red_stop_active else 'N'} | latency={latency_ms:.2f}ms"
                    )

            if not user_requested_quit:
                save_result_image = True

        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error during run_test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.print_summary(start_time)
            self.show_post_run_plots(
                spawn_index=spawn_index,
                max_speed=max_speed,
                steer_smooth_alpha=steer_smooth_alpha,
                steer_deadzone_enter=steer_deadzone_enter,
                steer_deadzone_exit=steer_deadzone_exit,
                steer_small_angle_th=steer_small_angle_th,
                save_result_image=save_result_image,
            )
            self.cleanup()

        return True

    def print_summary(self, start_time):
        # 会话结束后的聚合统计输出。
        elapsed = time.time() - start_time
        print("\nTEST SUMMARY")
        print("=" * 60)
        print(f"Elapsed time: {elapsed:.2f}s")
        print(f"Frames: {len(self.history)}")
        if self.history:
            speeds = [x["speed"] for x in self.history]
            steers = [x["steer"] for x in self.history]
            throttles = [x["throttle"] for x in self.history]
            brakes = [x["brake"] for x in self.history]
            latencies = [x.get("latency_ms", 0.0) for x in self.history]
            conflict_vals = [x.get("conflict", 0.0) for x in self.history]
            overspeed_vals = [x.get("overspeed", 0.0) for x in self.history]
            red_stop_vals = [x.get("red_stop_active", 0.0) for x in self.history]
            print(f"Avg FPS: {len(self.history) / max(elapsed, 1e-6):.2f}")
            print(f"Speed avg/max: {np.mean(speeds):.2f} / {np.max(speeds):.2f} km/h")
            print(f"Steer avg abs: {np.mean(np.abs(steers)):.4f}")
            print(f"Throttle avg: {np.mean(throttles):.4f}")
            print(f"Brake avg: {np.mean(brakes):.4f}")
            print(f"Inference latency avg/p95: {np.mean(latencies):.2f} / {np.percentile(latencies, 95):.2f} ms")
            print(f"Throttle-Brake conflict ratio: {100.0 * np.mean(conflict_vals):.2f}%")
            print(f"Overspeed ratio: {100.0 * np.mean(overspeed_vals):.2f}%")
            print(f"Red-stop safety active ratio: {100.0 * np.mean(red_stop_vals):.2f}%")
        print("=" * 60)

    def cleanup(self):
        # 资源清理：销毁 actor、恢复世界设置、关闭窗口。
        print("Cleaning up actors and windows")

        if self.mask_video_writer is not None:
            try:
                self.mask_video_writer.release()
                print(f"Saved mask video: {self.mask_video_path}")
            except Exception:
                pass
            self.mask_video_writer = None

        if self.client:
            for actor in self.actors:
                try:
                    actor.destroy()
                except Exception:
                    pass

        if self.world and self.original_settings:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass

        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    # 命令行参数：模型路径、仿真参数、控制平滑与速度分支开关。
    parser = argparse.ArgumentParser(description="Test multi-task driving model in CARLA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained multi-task checkpoint")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--duration", type=int, default=120, help="Test duration in seconds")
    parser.add_argument("--spawn_point", type=int, default=0, help="Spawn point index")
    parser.add_argument(
        "--town",
        type=str,
        default="Town01",
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"],
        help="CARLA town",
    )
    parser.add_argument("--max_speed", type=float, default=20.0, help="Speed guard in km/h")
    parser.add_argument(
        "--weather_type",
        type=str,
        default="sunny",
        choices=["sunny", "foggy", "rainy", "night"],
        help="CARLA weather preset",
    )
    parser.add_argument("--steer_smooth_alpha", type=float, default=0.48, help="Steering smoothing factor in [0,1)")
    parser.add_argument("--long_smooth_alpha", type=float, default=0.40, help="Longitudinal smoothing factor in [0,1)")
    parser.add_argument("--steer_deadzone_enter", type=float, default=0.05, help="Steer deadzone enter threshold")
    parser.add_argument("--steer_deadzone_exit", type=float, default=0.07, help="Steer deadzone exit threshold")
    parser.add_argument("--steer_small_angle_th", type=float, default=0.1, help="Small-angle steering threshold")
    parser.add_argument("--steer_small_rate_limit", type=float, default=0.02, help="Small-angle steer max delta per tick")
    parser.add_argument("--steer_large_rate_limit", type=float, default=0.30, help="Large-angle steer max delta per tick")
    parser.add_argument("--red_stop_tl_enter_prob", type=float, default=0.80, help="Enable red-stop when red-light prob >= this value")
    parser.add_argument("--red_stop_tl_exit_prob", type=float, default=0.60, help="Keep red-stop while red-light prob >= this value")
    parser.add_argument("--red_stop_release_speed_kmh", type=float, default=1.0, help="Release red-stop only below this speed")
    parser.add_argument("--red_stop_min_brake", type=float, default=0.35, help="Minimum brake when red-stop is active")
    parser.add_argument("--red_stop_max_throttle", type=float, default=0.0, help="Maximum throttle when red-stop is active")
    parser.add_argument("--red_stop_release_frames", type=int, default=8, help="Consecutive release-ready frames required to exit red-stop")
    parser.add_argument("--use_speed_input", action="store_true", help="Enable speed branch in model forward")
    parser.add_argument("--mask_alpha", type=float, default=0.45, help="Alpha blend value for attention heatmap overlay")
    parser.add_argument("--mask_update_interval", type=int, default=1, help="Update attention heatmap every N frames")
    parser.add_argument("--save_mask_video", action="store_true", help="Save dashboard with attention heatmap as mp4 video")
    parser.add_argument("--mask_video_path", type=str, default="analysis_output/mask_overlay.mp4", help="Output path for recorded mask video")
    parser.add_argument("--mask_video_fps", type=float, default=20.0, help="FPS used for recorded mask video")
    parser.add_argument("--enable_mask_viz", dest="enable_mask_viz", action="store_true", help="Enable attention heatmap visualization")
    parser.add_argument("--disable_mask_viz", dest="enable_mask_viz", action="store_false", help="Disable attention heatmap visualization")
    parser.set_defaults(enable_mask_viz=True)

    args = parser.parse_args()

    # 将平滑系数限制在合理范围，避免过强滞后。
    steer_smooth_alpha = float(np.clip(args.steer_smooth_alpha, 0.0, 0.95))
    long_smooth_alpha = float(np.clip(args.long_smooth_alpha, 0.0, 0.95))
    steer_deadzone_enter = float(np.clip(args.steer_deadzone_enter, 0.0, 0.25))
    steer_deadzone_exit = float(np.clip(args.steer_deadzone_exit, steer_deadzone_enter + 0.01, 0.35))
    steer_small_angle_th = float(np.clip(args.steer_small_angle_th, 0.05, 0.35))
    steer_small_rate_limit = float(np.clip(args.steer_small_rate_limit, 0.0, 0.20))
    steer_large_rate_limit = float(np.clip(args.steer_large_rate_limit, steer_small_rate_limit + 0.01, 0.50))

    # 初始化测试器并启动测试。
    tester = CarlaMultiTaskModelTester(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        use_speed_input=args.use_speed_input,
        enable_mask_visualization=args.enable_mask_viz,
        mask_alpha=args.mask_alpha,
        mask_update_interval=args.mask_update_interval,
        save_mask_video=args.save_mask_video,
        mask_video_path=args.mask_video_path,
        mask_video_fps=args.mask_video_fps,
        red_stop_tl_enter_prob=args.red_stop_tl_enter_prob,
        red_stop_tl_exit_prob=args.red_stop_tl_exit_prob,
        red_stop_release_speed_kmh=args.red_stop_release_speed_kmh,
        red_stop_min_brake=args.red_stop_min_brake,
        red_stop_max_throttle=args.red_stop_max_throttle,
        red_stop_release_frames=args.red_stop_release_frames,
    )

    ok = tester.run_test(
        duration=args.duration,
        spawn_index=args.spawn_point,
        town=args.town,
        max_speed=args.max_speed,
        weather_type=args.weather_type,
        steer_smooth_alpha=steer_smooth_alpha,
        long_smooth_alpha=long_smooth_alpha,
        steer_deadzone_enter=steer_deadzone_enter,
        steer_deadzone_exit=steer_deadzone_exit,
        steer_small_angle_th=steer_small_angle_th,
        steer_small_rate_limit=steer_small_rate_limit,
        steer_large_rate_limit=steer_large_rate_limit,
    )

    if ok:
        print("Test completed")
    else:
        print("Test failed")


if __name__ == "__main__":
    main()
