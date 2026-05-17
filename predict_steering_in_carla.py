"""
predict_steering_in_carla.py

功能：在 CARLA 仿真环境中加载训练好的转向回归模型，对车辆摄像头图像进行推理，生成转向控制并通过简单 PID 控制器实现速度控制。

主要组件：
- `PIDController`：简单的 PID 控制器，用于将速度误差映射到 throttle/brake 输出。
- `CarlaSteeringModelTester`：主类，负责连接 CARLA、生成车辆与传感器、接收相机帧、调用模型预测、生成控制、采集并在运行结束时导出分析报告。

实现要点：
- 使用 PyTorch 加载模型并在 GPU（若可用）上运行推理。
- 将摄像头帧按训练时的预处理（Resize, Normalize）转换为张量并输入模型，模型输出单个转向值。
- 用 PID 控制器平滑速度控制，避免突变。
- 支持集成的高级分析模块（若可用），否则退回基础可视化与日志记录。
"""

import carla
import cv2
import numpy as np
import time
import argparse
import torch
import math
from pathlib import Path
from model import NvidiaModelTransferLearning, NvidiaModel
from mask_visualization import MaskVisualizer

# 添加 PID 控制器类
class PIDController:
    """简单 PID 控制器，用于将速度误差转换为 throttle/brake 输出。

    参数：kp/ki/kd 为比例/积分/微分系数，output_limits 为输出取值范围。
    用法：在每个仿真步调用 `update(error, dt)`，返回受限的控制输出。
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0)):
        # PID 参数
        self.kp = kp
        self.ki = ki  
        self.kd = kd
        # 转角输出限制
        self.output_limits = output_limits
        
        # 误差
        self.last_error = 0.0
        # 积分项
        self.integral = 0.0
        self.last_time = None
        
    def update(self, error, dt=None):
        """传入误差，返回PID控制output。"""
        # 自动计算 时间间隔dt（如果没有显式提供），以便基于真实仿真时间得出积分/微分项
        if dt is None:
            current_time = time.time()
            if self.last_time is None:
                dt = 0.01  # 首次调用时采用一个小的默认 dt
            else:
                dt = current_time - self.last_time
            self.last_time = current_time

        # 计算 PID 三项
        proportional = self.kp * error
        # 积分项累积（存在漂移风险，若长期不清零可加入积分限幅）
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0

        # 由error组合pid输出
        output = proportional + self.ki * self.integral + self.kd * derivative
        # 把输出限制到（-1, 1）范围内，适合 throttle/brake 控制
        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        self.last_error = error
        return output
    
    def reset(self):
        # 重置 PID 状态
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

# 尝试导入高级分析模块，用于分析置信度等信息
ANALYSIS_AVAILABLE = True
try:
    from analysis.integrated_analyzer import IntegratedAutonomousDrivingAnalyzer
    ANALYSIS_AVAILABLE = True
    print('[Tester] Advanced analysis modules loaded successfully')
except ImportError as e:
    print(f'[Tester] Analysis modules not available: {e}')
    print('[Tester] Running in basic mode without advanced analysis.')
    ANALYSIS_AVAILABLE = False


class CarlaSteeringModelTester:
    """主控类：负责与 CARLA 环境交互并运行模型推理/控制循环。

    主要职责：
    - 连接并配置 CARLA 世界（同步模式、天气、仿真设置）。
    - 生成车辆与摄像头、碰撞与车道入侵传感器并绑定回调。
    - 加载训练好的 PyTorch 模型并在摄像头帧到达时执行推理。
    - 根据模型输出生成转向控制，并结合 PID 控制器生成速度控制（throttle/brake）。
    - 可选：将运行数据传入高级分析模块以生成详尽的评估报告。
    """

    def __init__(self, model_path, host='localhost', port=2000,
                 use_speed_input=False, enable_analysis=True,
                 enable_mask_visualization=True, mask_alpha=0.45,
                 mask_update_interval=1, save_mask_video=False,
                 mask_video_path='analysis_output/mask_overlay.mp4',
                 mask_video_fps=20.0):
        self.host = host
        self.port = port
        self.use_speed_input = use_speed_input
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.spectator_camera = None
        self.actors = []
        self.original_settings = None
        self.map = None
        
        # 读入训练好的模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)

        # 可解释性可视化（Deconvolution Mask）
        self.enable_mask_visualization = bool(enable_mask_visualization)
        self.mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
        self.mask_update_interval = max(1, int(mask_update_interval))
        self.save_mask_video = bool(save_mask_video)
        self.mask_video_path = Path(mask_video_path)
        self.mask_video_fps = float(mask_video_fps)
        self.mask_video_writer = None
        self.last_mask_overlay = None
        self.mask_visualizer = None
        if self.enable_mask_visualization:
            try:
                self.mask_visualizer = MaskVisualizer(
                    model=self.model,
                    device=self.device,
                    overlay_alpha=self.mask_alpha,
                )
                print('[Tester] Deconvolution mask visualization enabled')
            except Exception as e:
                print(f'[Tester] Failed to initialize mask visualization: {e}')
                print('[Tester] Continue without deconvolution mask visualization.')
                self.enable_mask_visualization = False
        
        # 上一帧图像数据
        self.latest_image = None
        self.latest_spectator_image = None
        
        # 表现历史记录，用于分析和报告生成
        self.predictions_history = []
        self.control_history = []
        self.spectator = None

        # Advanced Analysis System
        if enable_analysis and not ANALYSIS_AVAILABLE:
            raise RuntimeError('Advanced analysis modules are required. Install the analysis package or run with enable_analysis=False.')

        self.enable_analysis = enable_analysis and ANALYSIS_AVAILABLE
        if self.enable_analysis:
            self.analyzer = IntegratedAutonomousDrivingAnalyzer(
                max_history=2000,  # 利用大量内存以保存更多历史记录（例如 64GB 可用时）
                enable_advanced_analysis=True
            )
            print('[Tester] Advanced analysis system initialized')
        else:
            self.analyzer = None

        # 碰撞与车道入侵状态检测参数初始化
        self.collision_sensor = None    # 碰撞传感器实例
        self.collision_detected = False # 当前是否检测到碰撞的标志位
        self.lane_invasion_sensor = None    # 车道入侵传感器实例
        self.lane_departure_flag = False    # 当前是否发生车道入侵的标志位
        self.lane_invasion_events = []  # 记录车道入侵事件的列表，包含时间戳和入侵类型
        self.lane_offset_history = []   # 记录车辆相对于车道中心的偏移历史，用于分析车辆在车道内的表现
        
        # 实例化pid控制器
        self.speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, output_limits=(-0.8, 0.8))
        
    def load_model(self, model_path):
        """加载训练好的模型并返回可用于推理的模型实例。

        支持的 checkpoint 格式（.pt文件）：
        - 包含 `model_state_dict` 字段的完整检查点（包含 epoch/val_loss 等信息）。
        - 直接保存的 state_dict（纯权重字典）。

        方法会将模型移动到 `self.device` 并调用 `eval()`。
        """
        print(f"🤖 Loading steering model from {model_path}")
        
        # 实例化模型架构（不加载预训练权重，因为我们会加载训练好的权重）
        model = NvidiaModel(pretrained=False,freeze_features=False,)
        
        # 如果模型路径存在，则导入模型参数
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查checkpoint是否包含完整的训练状态
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 打印模型的训练信息
                if 'val_loss' in checkpoint:
                    print(f"✅ Model loaded with validation loss: {checkpoint['val_loss']:.6f}")
                
                if 'epoch' in checkpoint:
                    print(f"   📈 Trained for {checkpoint['epoch']} epochs")
                    
                    
            else:
                model.load_state_dict(checkpoint)
                
            # 移动到目标设备
            model.to(self.device)
            # 进入评估模式，关闭模型各层的dropout 和 batchnorm 的训练行为，用于推理
            model.eval()
            print("✅ model loaded successfully!")
            return model
        else:
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    def connect_to_carla(self, town='Town01'):
        """连接到 CARLA 服务器并设置仿真世界"""
        try:
            print(f"🌍 Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            
            # 加载指定城镇
            self.world = self.client.load_world(town)
            self.map = self.world.get_map()
            time.sleep(3)
            
            # 设置天气为晴朗正午
            weather = carla.WeatherParameters.ClearNoon  # pyright: ignore[reportAttributeAccessIssue] #  carla.WeatherParameters.MidRainyNoon  # carla.WeatherParameters.ClearNight

            self.world.set_weather(weather)
            time.sleep(2)

            self.original_settings = self.world.get_settings()
            
            # 设置同步模式和固定时间步长，以便控制仿真节奏并获得一致的帧率（约20 FPS）
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / 20  # 目标约 20 FPS（用于测试）
            self.world.apply_settings(settings)
            
            print(f"✅ Connected to CARLA successfully! Town: {town}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self, spawn_index=0):
        """在指定生成点生成车辆"""
        try:
            blueprint_library = self.world.get_blueprint_library() # pyright: ignore[reportOptionalMemberAccess]
            
            # 获取 Tesla Model 3（与训练数据中的车辆一致）
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]
                print("⚠️ Tesla Model 3 not found, using alternative vehicle")

            time.sleep(3)
            
            # 获取生成点
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available")
            
            # 使用指定的生成点
            spawn_point = spawn_points[0]
            time.sleep(2)
            
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if not self.vehicle:
                # 如果第一个生成点失败，则尝试其它生成点
                for i in range(min(5, len(spawn_points))):
                    alt_spawn = spawn_points[(spawn_index + i) % len(spawn_points)]
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, alt_spawn)
                    if self.vehicle:
                        spawn_index = (spawn_index + i) % len(spawn_points)
                        break
                
                if not self.vehicle:
                    raise Exception("Failed to spawn vehicle at any spawn point")
            
            self.actors.append(self.vehicle)
            print(f"🚗 Vehicle spawned at spawn point {spawn_index}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to spawn vehicle: {e}")
            return False
    
    def follow_vehicle_spectator(self, distance=6.0, height=2.5, pitch=-15):
        """
        重新定位观众视角，使其位于车辆后方 `distance` 米、地面上方 `height` 米，
        并以 `pitch` 度向下略微俯视。每个仿真步调用一次。
        """
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()

        # 将摄像头放置在车辆后方
        cam_location = transform.location - forward * distance
        cam_location.z += height

        # 复制车辆偏航角以保持相同朝向，然后根据 pitch 向下倾斜
        cam_rotation = carla.Rotation(
            pitch=pitch,
            yaw=transform.rotation.yaw,
            roll=0.0
        )

        self.spectator.set_transform(carla.Transform(cam_location, cam_rotation))

    def spawn_camera(self):
        """在车辆上生成摄像头传感器"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # 与训练数据使用的摄像头参数保持一致
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')
            
            # 摄像头位置（采用中心视角以匹配训练设置）
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0.0)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            self.actors.append(self.camera)
            
            # 设置回调函数
            self.camera.listen(self.camera_callback)

            # 额外创建一个跟车第三人称视角，用于 dashboard 右下角显示
            self.spawn_spectator_camera()

            self.spawn_collision_sensor()
            self.spawn_lane_invasion_sensor()

            print('[Tester] Camera spawned and listening')
            return True

        except Exception as e:
            print(f"[Tester] Failed to spawn camera: {e}")
            return False

    def spawn_spectator_camera(self):
        """生成跟车第三人称摄像头，用于 dashboard 右下角观察车辆姿态。"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            spectator_bp = blueprint_library.find('sensor.camera.rgb')

            spectator_bp.set_attribute('image_size_x', '640')
            spectator_bp.set_attribute('image_size_y', '480')
            spectator_bp.set_attribute('fov', '90')

            spectator_transform = carla.Transform(
                carla.Location(x=-6.0, z=2.5),
                carla.Rotation(pitch=-15.0)
            )

            self.spectator_camera = self.world.spawn_actor(
                spectator_bp,
                spectator_transform,
                attach_to=self.vehicle
            )
            self.actors.append(self.spectator_camera)
            self.spectator_camera.listen(self.spectator_camera_callback)
            print('[Tester] Spectator dashboard camera spawned')
        except Exception as e:
            self.spectator_camera = None
            print(f"[Tester] Failed to spawn spectator dashboard camera: {e}")
    
    def spawn_collision_sensor(self):
        """生成碰撞检测传感器"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            collision_bp = blueprint_library.find('sensor.other.collision')
            
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.actors.append(self.collision_sensor)
            
            # 设置碰撞回调
            self.collision_sensor.listen(self.collision_callback)
            print("🛡️ Collision sensor spawned")
            
        except Exception as e:
            print(f"⚠️ Failed to spawn collision sensor: {e}")
    
    def spawn_lane_invasion_sensor(self):
        """生成车道入侵传感器以检测偏离车道的情况"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            lane_bp = blueprint_library.find('sensor.other.lane_invasion')

            self.lane_invasion_sensor = self.world.spawn_actor(
                lane_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.actors.append(self.lane_invasion_sensor)

            self.lane_invasion_sensor.listen(self.lane_invasion_callback)
            print('[Tester] 车道入侵传感器已生成')
        except Exception as e:
            print(f'[Tester] Failed to spawn lane invasion sensor: {e}')

    def lane_invasion_callback(self, event):
        """处理车道入侵（越过车道线）事件"""
        self.lane_departure_flag = True
        try:
            markings = [str(mark.type) for mark in event.crossed_lane_markings]
        except Exception:
            markings = []

        self.lane_invasion_events.append(
            {"timestamp": time.time(), "markings": markings}
        )
        if len(self.lane_invasion_events) > 500:
            self.lane_invasion_events.pop(0)

        if self.enable_analysis:
            print(f"[Tester] 检测到车道入侵: {markings}")

    def _compute_lane_alignment(self, transform):
        """返回车辆相对于车道中心的带符号横向偏移和车道宽度。"""
        if not self.map:
            return None, None

        try:
            waypoint = self.map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        except Exception:
            waypoint = None

        if waypoint is None:
            return None, None

        waypoint_transform = waypoint.transform
        lane_origin = waypoint_transform.location
        vehicle_loc = transform.location

        dx = vehicle_loc.x - lane_origin.x
        dy = vehicle_loc.y - lane_origin.y
        dz = vehicle_loc.z - lane_origin.z

        right_vector = waypoint_transform.get_right_vector()
        lateral_offset = dx * right_vector.x + dy * right_vector.y + dz * right_vector.z

        return lateral_offset, waypoint.lane_width

    def camera_callback(self, image):
        """处理接收到的摄像头图像"""
        # 将 CARLA 图像数据转换为 NumPy 数组
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3]  # 去掉 alpha 通道，保留 RGB

    def spectator_camera_callback(self, image):
        """处理跟车第三人称摄像头图像。"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_spectator_image = array[:, :, :3]

    def prepare_model_input(self, image):
        """将原始 BGR 图像转换为模型推理输入，并返回 resize 后 RGB 图像。"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (200, 66))

        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std

        image_tensor = torch.from_numpy(image_normalized).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image_tensor, image_resized
    
    def predict_controls(self, image, current_speed_kmh=None, return_inference_bundle=False):
        """使用模型对图像进行推理，预测转向角度"""
        try:
            image_tensor, image_resized = self.prepare_model_input(image)
            
            # 推理（在 no_grad 上下文中以节省显存并避免记录梯度）
            with torch.no_grad():
                # 使用模型进行前向预测，得到张量后用 item() 获取标量转向角
                prediction = self.model(image_tensor)
                steering_angle = float(prediction.reshape(-1)[0].item())
            # 输出为模型预测的转向值（通常在训练时用了弧度或[-1,1] 标准化），
            
            # 这里直接裁剪到 [-1,1] 作为最终控制信号范围。
            steering_angle = float(np.clip(steering_angle, -1.0, 1.0))

            if return_inference_bundle:
                return steering_angle, {
                    'model_input_tensor': image_tensor.detach(),
                    'resized_rgb': image_resized,
                }

            return steering_angle
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            if return_inference_bundle:
                return 0.0, None
            return 0.0

    def run_test(self, duration=60, spawn_index=0, town='Town01', 
                 safety_mode=True, max_speed=10):
        """运行自动驾驶测试并收集分析数据"""
        
        # 设置
        if not self.connect_to_carla(town):
            return False
        
        if not self.spawn_vehicle(spawn_index):
            return False
        
        if not self.spawn_camera():
            return False
        
        # 初始化车辆状态
        print('[Tester] Initializing vehicle physics...')
        self._initialize_vehicle()

        # 重置单次运行状态
        self.lane_departure_flag = False
        self.lane_invasion_events.clear()
        self.lane_offset_history.clear()

        # 为新运行重置 PID 控制器
        self.speed_pid.reset()
        print('[Tester] PID controller reset for new run')
        
        print(f"\n🏁 Starting Autonomous Steering Test")
        print(f"⏱️  Duration: {duration} seconds")
        print(f"🌍 Town: {town}")
        print(f"🛡️  Safety mode: {safety_mode}")
        print(f"🚀 Max speed: {max_speed} km/h")
        print(f"📊 Speed input: {self.use_speed_input}")
        print(f"🧠 Mask visualization: {self.enable_mask_visualization}")
        if self.enable_mask_visualization:
            print(f"🧠 Mask update interval: every {self.mask_update_interval} frame(s)")
            print(f"🎥 Save mask video: {self.save_mask_video}")
            if self.save_mask_video:
                print(f"🎥 Mask video path: {self.mask_video_path}")
            print("🧠 Mask algorithm: paper-style layerwise deconvolution masking")
        print("Press 'Q' to quit early, 'S' to save screenshot")
        if self.enable_analysis:
            print("📊 Analysis reports will be automatically generated at simulation end")
        
        # 单窗口显示：real-time + mask + control
        cv2.namedWindow("Unified Visualization", cv2.WINDOW_AUTOSIZE)
        
        start_time = time.time()
        frame_count = 0
        screenshot_count = 0
        
        try:
            while time.time() - start_time < duration:
                self.spectator = self.world.get_spectator()
                # 推动仿真步进
                self.world.tick()
                self.follow_vehicle_spectator()

                # 检查是否接收到图像
                if self.latest_image is None:
                    time.sleep(0.01)
                    continue
                
                if self.enable_mask_visualization:
                    predicted_steering, inference_bundle = self.predict_controls(
                        self.latest_image,
                        return_inference_bundle=True,
                    )
                else:
                    predicted_steering = self.predict_controls(self.latest_image)
                    inference_bundle = None
                # 由carla api获取当前速度
                velocity = self.vehicle.get_velocity()
                # 由carla api获取车辆当前的位置信息和朝向信息，以便计算相对于车道中心的偏移量
                transform = self.vehicle.get_transform()
                time.sleep(0.01)  # Allow time for physics update
                # 将速度从 m/s 转换为 km/h
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                # 计算车辆相对于车道中心的偏移量和车道宽度，并记录历史数据以供分析
                lane_offset, lane_width = self._compute_lane_alignment(transform)
                # 记录带符号的偏移量（左负右正）和绝对偏移量，以及车道宽度（如果可用）
                signed_lane_offset = lane_offset if lane_offset is not None else float('nan')
                abs_lane_offset = abs(lane_offset) if lane_offset is not None else float('nan')
                if lane_width is None:
                    lane_width = float('nan')

                self.lane_offset_history.append({
                    'timestamp': time.time(),
                    'offset': signed_lane_offset,
                    'lane_width': lane_width
                })
                if len(self.lane_offset_history) > 5000:
                    self.lane_offset_history.pop(0)
                lane_departed_this_step = self.lane_departure_flag

                
                # PID-based speed control (much smoother than manual thresholds)
                speed_error = max_speed - current_speed
                pid_output = self.speed_pid.update(speed_error)
                
                # Convert PID output to throttle/brake (output为正值表示油门，负值表示刹车)
                if pid_output > 0:
                    # Positive output = need to speed up
                    throttle = pid_output  # Already limited by PID output_limits
                    brake = 0.0
                else:
                    # Negative output = need to slow down
                    throttle = 0.0
                    brake = -pid_output  # Convert negative to positive brake value
                
                # 应用控制
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=predicted_steering,
                    brake=brake
                )
                self.vehicle.apply_control(control)
                
                # 存储预测历史以便汇总
                self.predictions_history.append({
                    'steering': predicted_steering,
                    'speed': current_speed,
                    'speed_error': speed_error,
                    'pid_output': pid_output,
                    'throttle': throttle,
                    'brake': brake,
                    'lane_offset': signed_lane_offset,
                    'lane_offset_abs': abs_lane_offset,
                    'lane_width': lane_width,
                    'lane_invasion': lane_departed_this_step,
                    'timestamp': time.time()
                })
                
                
                # 为分析准备数据
                if self.enable_analysis:
                    # 为分析准备模型输出与车辆状态
                    model_output = {'steering': predicted_steering}
                    vehicle_state = {
                        'speed': current_speed,
                        'actual_position': [transform.location.x, transform.location.y],
                        'distance_to_center': abs_lane_offset,
                        'lane_offset': signed_lane_offset,
                        'lane_width': lane_width,
                        'lane_invasion': lane_departed_this_step,
                        'collision_occurred': self.collision_detected
                    }
                    
                    # 为置信度分析准备图像张量
                    image_tensor = None
                    if self.latest_image is not None:
                        try:
                            # 将图像转换为张量格式（与 predict_controls 一致）
                            image_tensor, _ = self.prepare_model_input(self.latest_image)
                        except Exception as e:
                            print(f"Image tensor preparation error: {e}")
                    
                    # 运行综合分析
                    display_image = self.analyzer.analyze_step(
                        model_output, vehicle_state, self.latest_image.copy(), 
                        self.model, image_tensor
                    )
                else:
                    # 无高级分析时不做额外绘制，保留原始图像供统一窗口使用
                    display_image = self.latest_image.copy()
                
                if lane_departed_this_step:
                    self.lane_departure_flag = False

                # 创建控制可视化
                control_viz = self.create_control_visualization({'steering': predicted_steering}, current_speed, max_speed)

                # 左上固定显示原图（相机输入）
                original_display = self.latest_image.copy()
                mask_display = original_display.copy()

                # 使用与 real-time 相同的相机帧生成并覆盖 mask
                if self.enable_mask_visualization and self.mask_visualizer and inference_bundle is not None:
                    try:
                        need_update_mask = (frame_count % self.mask_update_interval == 0) or (self.last_mask_overlay is None)
                        if need_update_mask:
                            mask_output = self.mask_visualizer.build_overlay_from_model_input(
                                model_input_tensor=inference_bundle['model_input_tensor'],
                                resized_rgb=inference_bundle['resized_rgb'],
                            )
                            # 将模型输入尺度(66x200)的 mask 拉伸到 real-time 画面尺寸，确保视角一致
                            h, w = original_display.shape[:2]
                            mask_resized = cv2.resize(mask_output['mask'], (w, h), interpolation=cv2.INTER_LINEAR)
                            heatmap_bgr = self.mask_visualizer.mask_generator.create_heatmap(mask_resized)
                            mask_display = cv2.addWeighted(
                                original_display,
                                1.0 - self.mask_alpha,
                                heatmap_bgr,
                                self.mask_alpha,
                                0.0,
                            )
                            self.last_mask_overlay = mask_display
                        else:
                            cached = self.last_mask_overlay
                            if cached is not None and cached.shape[:2] == original_display.shape[:2]:
                                mask_display = cached
                            else:
                                self.last_mask_overlay = None
                                mask_display = original_display.copy()
                    except Exception as e:
                        print(f"[Tester] Mask visualization error: {e}")
                        mask_display = original_display.copy()
                else:
                    cv2.putText(mask_display, "Mask Visualization Disabled", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                unified_view = self.create_unified_visualization(
                    original_bgr=original_display,
                    mask_bgr=mask_display,
                    control_viz=control_viz,
                    spectator_bgr=self.latest_spectator_image,
                    steering_value=predicted_steering,
                    speed_kmh=current_speed,
                )

                cv2.imshow("Unified Visualization", unified_view)

                if self.save_mask_video and unified_view is not None:
                    if self.mask_video_writer is None:
                        self._init_mask_video_writer(unified_view)
                    if self.mask_video_writer is not None:
                        self.mask_video_writer.write(unified_view)
                
                # 检查用户输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 User requested quit")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    shot_path = f"carla_test_{timestamp}_{screenshot_count}.png"
                    cv2.imwrite(shot_path, unified_view)
                    print(f"📸 Screenshot saved: {shot_path}")
                    screenshot_count += 1
                
                
                frame_count += 1
                
                # 每5秒打印一次统计信息
                if frame_count % 50 == 0:  # 约每5秒（目标 10 FPS）
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    if self.enable_analysis:
                        # 获取分析统计
                        dashboard_stats = self.analyzer.dashboard.get_current_stats()
                        safety_score = self.analyzer.safety_analyzer.calculate_safety_score()
                        confidence_summary = self.analyzer.confidence_analyzer.get_confidence_summary()
                        
                        confidence_val = 0
                        if isinstance(confidence_summary, dict):
                            confidence_val = confidence_summary.get('recent_confidence', 0)
                        
                        print(f"📊 Time: {elapsed:5.1f}s | FPS: {fps:4.1f} | Speed: {current_speed:5.1f} km/h | "
                              f"Steering: {predicted_steering:5.3f} | PID: {pid_output:5.3f} | Safety: {safety_score:.1f} | Confidence: {confidence_val:.3f}")
                    else:
                        print(f"📊 Time: {elapsed:5.1f}s | FPS: {fps:4.1f} | Speed: {current_speed:5.1f} km/h | "
                              f"S: {predicted_steering:5.3f} | PID: {pid_output:5.3f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                self.print_test_summary(start_time)
            except Exception as e:
                print(f"❌ Error generating test summary: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.cleanup()
        
        
        return True
    
    def _initialize_vehicle(self):
        """🔧 正确初始化车辆物理和仿真设置"""
        try:
            # 等待车辆状态稳定
            time.sleep(1)
            
            # 应用初始控制以激活物理模拟
            initial_control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )
            self.vehicle.apply_control(initial_control)
            
            # 推动仿真若干步以稳定
            for _ in range(5):
                self.world.tick()
                time.sleep(0.1)
            
            # 施加短促油门使车辆启动
            startup_control = carla.VehicleControl(
                throttle=0.3,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )
            self.vehicle.apply_control(startup_control)
            
            # 让物理模拟继续稳定
            for _ in range(3):
                self.world.tick()
                time.sleep(0.1)
            
            print("✅ Vehicle physics initialized")
            
        except Exception as e:
            print(f"⚠️ Vehicle initialization warning: {e}")
    
    def create_control_visualization(self, controls, current_speed, max_speed):
        """创建控制输出的可视化表示"""
        viz = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 方向盘可视化
        center = (100, 150)
        radius = 80
        
        # 绘制方向盘圆环
        cv2.circle(viz, center, radius, (100, 100, 100), 2)
        
        # 绘制方向指示线
        angle = controls['steering'] * 90  # 转换为度
        end_x = int(center[0] + radius * 0.8 * np.sin(np.radians(angle)))
        end_y = int(center[1] - radius * 0.8 * np.cos(np.radians(angle)))
        cv2.line(viz, center, (end_x, end_y), (0, 255, 255), 4)
        
        
        # 添加文本标签
        cv2.putText(viz, f"Steer: {controls['steering']:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(viz, f"Speed: {current_speed:.3f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return viz

    def create_unified_visualization(self, original_bgr, mask_bgr, control_viz, spectator_bgr, steering_value, speed_kmh):
        """四分屏布局：左上原图，右上mask，左下速度控制，右下第三人称视角。"""
        if original_bgr is None:
            original_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        if mask_bgr is None:
            mask_bgr = original_bgr.copy()

        h, w = original_bgr.shape[:2]
        mask_resized = cv2.resize(mask_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        control_panel = cv2.resize(control_viz, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(control_panel, f"Steering: {steering_value:+.3f} | Speed: {speed_kmh:5.1f} km/h", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(control_panel, "Speed Control", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        if spectator_bgr is not None:
            spectator_panel = cv2.resize(spectator_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            spectator_panel = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(spectator_panel, "Spectator View Unavailable", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        cv2.putText(spectator_panel, "Spectator View", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        top_row = np.hstack([original_bgr, mask_resized])
        bottom_row = np.hstack([control_panel, spectator_panel])

        return np.vstack([top_row, bottom_row])

    def _init_mask_video_writer(self, mask_panel):
        """按 mask 面板尺寸初始化视频写入器。"""
        try:
            self.mask_video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = mask_panel.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.mask_video_writer = cv2.VideoWriter(
                str(self.mask_video_path),
                fourcc,
                max(1.0, self.mask_video_fps),
                (width, height)
            )

            if not self.mask_video_writer.isOpened():
                print(f"[Tester] Failed to open video writer: {self.mask_video_path}")
                self.mask_video_writer = None
            else:
                print(f"[Tester] Recording mask video to: {self.mask_video_path}")
        except Exception as e:
            print(f"[Tester] Failed to initialize mask video writer: {e}")
            self.mask_video_writer = None
    
    def print_test_summary(self, start_time):
        """打印测试汇总统计信息"""
        print(f"\n📊 测试汇总")
        print("🔍 开始生成测试汇总...")
        
        elapsed = time.time() - start_time
        
        # 计算统计信息（移除阻止汇总生成的早期返回）
        if self.predictions_history:
            steering_values = [p['steering'] for p in self.predictions_history]
        else:
            print("⚠️ No predictions history available")
            steering_values = []


        if self.lane_offset_history:
            lane_offsets = [entry['offset'] for entry in self.lane_offset_history if entry['offset'] == entry['offset']]
            if lane_offsets:
                mean_abs_offset = float(np.mean(np.abs(lane_offsets)))
                max_abs_offset = float(np.max(np.abs(lane_offsets)))
                print(f'Lane offset | mean abs: {mean_abs_offset:.3f} m, max abs: {max_abs_offset:.3f} m')

        lane_event_count = len(self.lane_invasion_events)
        if lane_event_count:
            print(f'Lane invasions detected: {lane_event_count}')
        else:
            print('Lane invasions detected: 0')

        if self.enable_analysis and self.analyzer:
            try:
                # 生成最终的综合分析报告
                print("🔍 正在生成最终分析报告...")
                final_report = self.analyzer.generate_comprehensive_report(save_plots=True)
                
                # 性能汇总
                perf_summary = final_report.get('performance_summary', {})
                print(f"📦 Total frames: {perf_summary.get('total_frames', 0)}")
                print(f"🎯 Average FPS: {perf_summary.get('average_fps', 0):.1f}")
                
                # 整体评分
                overall_score = final_report.get('overall_score', {})
                print(f"\n🏆 整体表现")
                print(f"   最终得分: {overall_score.get('overall_score', 0):.1f}/100")
                print(f"   等级: {overall_score.get('grade', 'N/A')}")
                
                # 各项得分
                component_scores = overall_score.get('component_scores', {})
                print(f"\n📈 COMPONENT SCORES:")
                for component, score in component_scores.items():
                    print(f"   {component.capitalize()}: {score:.1f}/100")
                
                # 安全性分析
                safety_report = final_report.get('safety_analysis', {})
                if safety_report and 'message' not in safety_report:
                    print(f"\n🛡️  安全性分析:")
                    print(f"   安全得分: {safety_report.get('overall_safety_score', 100):.1f}/100")
                    print(f"   安全事件总数: {safety_report.get('total_safety_events', 0)}")
                    print(f"   高严重性事件: {safety_report.get('high_severity_events', 0)}")
                    
                    risk_factors = safety_report.get('risk_factors', {})
                    print(f"   Risk Factors:")
                    print(f"     - Collision Risk: {risk_factors.get('collision_risk', 0):.1f}%")
                    print(f"     - Lane Keeping Risk: {risk_factors.get('lane_keeping_risk', 0):.1f}%")
                    print(f"     - Speed Risk: {risk_factors.get('speed_risk', 0):.1f}%")
                
                # 置信度分析
                confidence_report = final_report.get('confidence_analysis', {})
                if confidence_report and 'message' not in confidence_report:
                    print(f"\n🎯 置信度分析:")
                    print(f"   平均置信度: {confidence_report.get('mean_confidence', 0):.3f}")
                    print(f"   置信度趋势: {confidence_report.get('confidence_trend', 'unknown').title()}")
                    print(f"   低置信度比例: {confidence_report.get('low_confidence_ratio', 0)*100:.1f}%")
                
                # 轨迹分析
                trajectory_metrics = final_report.get('trajectory_analysis', {})
                if trajectory_metrics and 'message' not in trajectory_metrics:
                    print(f"\n🛣️  轨迹分析:")
                    print(f"   总行驶距离: {trajectory_metrics.get('total_distance', 0):.1f}m")
                    print(f"   路径效率: {trajectory_metrics.get('path_efficiency', 0)*100:.1f}%")
                    print(f"   平均速度: {trajectory_metrics.get('average_speed', 0):.1f} km/h")
                    print(f"   转向平滑度: {1.0 - trajectory_metrics.get('steering_smoothness', 0):.3f}")
                
                # 导出数据
                print(f"\n💾 正在导出分析数据...")
                
                # 为本次仿真创建带时间戳的输出目录
                simulation_timestamp = time.strftime("%Y%m%d_%H%M%S")
                timestamped_output_dir = f"analysis_output/simulation_{simulation_timestamp}"
                
                import os
                if not os.path.exists("analysis_output"):
                    os.makedirs("analysis_output")
                if not os.path.exists(timestamped_output_dir):
                    os.makedirs(timestamped_output_dir)
                
                # 导出综合分析到时间戳目录
                self.analyzer.export_all_data(timestamped_output_dir)
                
                # 同时更新主输出目录以保存最新结果
                self.analyzer.export_all_data("analysis_output")
                
                print(f"📁 Simulation results automatically saved to: {timestamped_output_dir}")
                print(f"📁 Latest results also available in: analysis_output/")
                
            except Exception as e:
                print(f"❌ Error during analysis report generation: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️ Continuing with basic summary...")
            
        else:
            # 基本汇总（无高级分析）
            if self.predictions_history:
                steering_values = [p['steering'] for p in self.predictions_history]
                print(f"📦 Frames processed: {len(self.predictions_history)}")
                print(f"🎯 Average FPS: {len(self.predictions_history) / elapsed:.1f}")
                print(f"\n🎛️  BASIC STATISTICS:")
                print(f"   Steering - Avg: {np.mean(steering_values):6.3f}, Std: {np.std(steering_values):6.3f}")
            else:
                print("📦 No prediction data recorded")
        
        print(f"\n💡 CONTROLS:")
        print(f"   Q: Quit | S: Save Screenshot")
        if self.enable_analysis:
            print(f"   📊 Reports automatically generated at simulation end")
        print("=" * 70)
        
    
    def cleanup(self):
        """清理所有生成的 Actor 并恢复仿真设置"""
        print("🧹 开始清理...")

        # 释放视频写入器
        if self.mask_video_writer is not None:
            try:
                self.mask_video_writer.release()
                print(f"🎥 Mask video saved: {self.mask_video_path}")
            except Exception:
                pass
            self.mask_video_writer = None
        
        # 销毁所有 actor
        if self.client:
            for actor in self.actors:
                try:
                    actor.destroy()
                except:
                    pass
        
        # 恢复原始仿真设置
        if self.world and self.original_settings:
            try:
                self.world.apply_settings(self.original_settings)
            except:
                pass
        
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        
        print("✅ 清理完成")


def main():
    parser = argparse.ArgumentParser(description="Test Steering Model in CARLA")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained steering model file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    parser.add_argument('--duration', type=int, default=180,
                       help='Test duration in seconds')
    parser.add_argument('--spawn_point', type=int, default=0,
                       help='Spawn point index')
    parser.add_argument('--town', type=str, default='Town01',
                       choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt'],
                       help='CARLA town to test in')
    parser.add_argument('--use_speed_input', action='store_true',
                       help='Use current speed as model input (if model supports it)')
    parser.add_argument('--safety_mode', action='store_true', default=True,
                       help='Enable safety mode (speed limiting, collision avoidance)')
    parser.add_argument('--max_speed', type=float, default=20,
                       help='Maximum allowed speed in km/h (safety mode)')
    parser.add_argument('--mask_alpha', type=float, default=0.35,
                       help='Alpha blend value for deconvolution mask overlay')
    parser.add_argument('--mask_update_interval', type=int, default=1,
                       help='Update deconvolution mask every N frames (default: 1)')
    parser.add_argument('--save_mask_video', action='store_true',
                       help='Save deconvolution mask visualization window into a video file')
    parser.add_argument('--mask_video_path', type=str, default='analysis_output/mask_overlay.mp4',
                       help='Output path for recorded mask visualization video')
    parser.add_argument('--mask_video_fps', type=float, default=20.0,
                       help='FPS used for recorded mask visualization video')
    parser.add_argument('--enable_mask_viz', dest='enable_mask_viz', action='store_true',
                       help='Enable deconvolution mask visualization window')
    parser.add_argument('--disable_mask_viz', dest='enable_mask_viz', action='store_false',
                       help='Disable deconvolution mask visualization window')
    parser.set_defaults(enable_mask_viz=True)
    
    args = parser.parse_args()
    
    print("🚗 CARLA STEERING MODEL TESTER")
    print("=" * 50)
    
    # Create tester
    tester = CarlaSteeringModelTester(
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
    )
    
    # Run test
    success = tester.run_test(
        duration=args.duration,
        spawn_index=args.spawn_point,
        town=args.town,
        safety_mode=args.safety_mode,
        max_speed=args.max_speed
    )
    
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed!")


if __name__ == '__main__':
    main()
    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town01 --duration 180 --max_speed 10
    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town02 --duration 180
    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town04 --duration 380 --max_speed 5
    #  python predict_steering_in_carla.py --model_path "carla_steering_best.pt" --town Town02 --duration 380 --max_speed 25
    #  python predict_steering_in_carla.py --model_path "carla_steering_best_restnet_weathers.pt" --town Town10HD_Opt --duration 380 --max_speed 25
    # python predict_steering_in_carla.py --model_path "models/carla_multi_control_best.pt" --town Town10HD_Opt --duration 380 --max_speed 10
