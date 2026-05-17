# 导入依赖库：CARLA 接口、数值与图像处理、并发与文件操作
import carla
import numpy as np
import collections
import cv2
import argparse
import os
import csv
import threading
import time
import queue
from datetime import datetime
import concurrent.futures
from pathlib import Path

class CarlaDataCollector:
    # CARLA 数据采集器：封装世界设置、车辆与摄像头的创建、帧处理与数据保存逻辑
    def __init__(self, host='localhost', port=2000, max_frames=10000, map='Town01', output_dir='data_weathers'):
        # 网络与采集配置
        self.host = host
        self.port = port
        self.max_frames = max_frames
        self.frame_counter = 0
        self.data_queue = queue.Queue()
        self.lock = threading.Lock()

        # 天气调度顺序（可按需修改）
        """
        最后通过setweather方法设置到world中,具体参数参考方法_get_weather_preset()
        天气也可以通过API/util中的environment.py中的预设进行设置，
        或者直接使用carla.WeatherParameters类构造自定义天气参数
        """
        self.weather_order = ['sunny', 'foggy', 'rainy', 'night']   # ***这里考虑做拓展***

        # 连接 CARLA 服务器并设置超时
        self.client = carla.Client(self.host, self.port) # type: ignore
        self.client.set_timeout(20.0)
        
        # 地图初始化
        self.map = map  # 可选城镇示例: 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt'
        # 数据集保存根目录（相对路径或绝对路径）
        self.output_dir = output_dir
        
        # 摄像头参数（分辨率与视场角）
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90
        
    def setup_world(self, town_name):
        """设置并返回指定城镇的 world 对象：加载地图、应用初始天气、同步模式设置"""
        try:
            world = self.client.load_world(town_name)   # 创建地图
            time.sleep(3)

            # 在地图加载后统一设置所有交通灯的红/绿/黄持续时间
            self._set_all_traffic_lights_duration(world, red_time=2.0, green_time=10.0, yellow_time=3.0)

            # 初始天气：晴朗
            weather = carla.WeatherParameters(
                sun_altitude_angle=70.0,
            )
            world.set_weather(weather)  # 通过 API 中的set_weather方法设置天气参数

            time.sleep(3)

            # 切换到同步模式并设置固定帧率（10 FPS）
            settings = world.get_settings()
            time.sleep(3)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # 固定时间步长，约 10 FPS
            world.apply_settings(settings)
            time.sleep(4)
            return world    # 返回值是world对象
        except Exception as e:
            print(f"Error setting up world {town_name}: {e}")
            return None

    def _set_all_traffic_lights_duration(self, world, red_time=2.0, green_time=10.0, yellow_time=3.0):
        """设置当前地图中所有交通灯的红绿黄灯持续时间（单位：秒）。"""
        try:
            traffic_lights = world.get_actors().filter('traffic.traffic_light')
            for traffic_light in traffic_lights:
                traffic_light.set_red_time(red_time)
                traffic_light.set_green_time(green_time)
                traffic_light.set_yellow_time(yellow_time)

            print(
                f"已设置交通灯时长：红灯={red_time}s，绿灯={green_time}s，黄灯={yellow_time}s；"
                f"共 {len(traffic_lights)} 个交通灯"
            )
        except Exception as e:
            print(f"Warning: 设置交通灯持续时间失败: {e}")
    
    def setup_vehicle_and_cameras(self, world):
        """创建并配置车辆与 3 个摄像头（中/左/右），启用自动驾驶与交通管理器设置"""
        try:
            # 获取地图中的这些预设点
            spawn_points = world.get_map().get_spawn_points()
            time.sleep(1.0)
            # 在预设点中随机选一个作为车辆生成点
            spawn_point = np.random.choice(spawn_points)
            time.sleep(1.0)

            # 获取车辆蓝图库
            blueprint_library = world.get_blueprint_library()
            time.sleep(1.0)
            # 选择特定车型（Tesla Model 3）作为主驾驶车辆
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            time.sleep(4.0)
            # 在生成点生成车辆
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            time.sleep(4.0)

            # 实例化traffic_manager
            traffic_manager = self.client.get_trafficmanager()
            time.sleep(5.0)
            # 忽略百分之n的红绿灯以提升数据多样性（可调整）
            """
            ***********************************************************************
            这块要考虑如果不忽略红绿灯采集数据是否训练出来的模型会在红灯停下，需要验证一下。
            ***********************************************************************
            """
            traffic_manager.ignore_lights_percentage(vehicle, 0)
            # 设置交通管理器为同步模式
            traffic_manager.set_synchronous_mode(True)
            time.sleep(4.0)
            
            # 设置跟车距离为2米
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            #traffic_manager.set_hybrid_physics_mode(True)
            

            time.sleep(2.0)
            # 全局速度调整：比默认速度快 10%
            traffic_manager.global_percentage_speed_difference(10.0)
            time.sleep(2.0)

            # 通过Traffic Manager让生成的车辆自动驾驶
            vehicle.set_autopilot(True, traffic_manager.get_port())
            time.sleep(2.0)

            # 摄像头蓝图与属性设置（RGB 摄像头）
            # 获取rgb相机蓝图
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            # 设置摄像头属性：分辨率和视场角
            camera_bp.set_attribute('image_size_x', str(self.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.camera_height))
            camera_bp.set_attribute('fov', str(self.camera_fov))
            
            # 摄像头相对车辆的位置变换（中心/左/右）
            camera_transforms = {
                'center': carla.Transform(carla.Location(x=2.0, z=1.4)), # type: ignore
                'left': carla.Transform(carla.Location(x=2.0, y=-1.5, z=1.4)), # type: ignore
                'right': carla.Transform(carla.Location(x=2.0, y=1.2, z=1.4)) # type: ignore
            }
            
            # 生成并附加摄像头到车辆
            cameras = {}
            for position, transform in camera_transforms.items():
                '''根据rgb_camera蓝图、位置变换创建摄像头，并附加到车辆上'''
                camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
                cameras[position] = camera
            
            time.sleep(2.0)
            return vehicle, cameras # 返回车辆和3个摄像头对象
            
        except Exception as e:
            print(f"Error setting up vehicle and cameras: {e}")
            return None, None
    
    def process_image(self, image):
        """将 CARLA 图像转换为 NumPy 数组并转换为 RGB 格式"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)   # 创建数组
        array = array.reshape((image.height, image.width, 4))   # 重塑为 (H, W, 4) 的 RGBA 图像
        array = array[:, :, :3]       # 去除 alpha 通道，变为 (H, W, 3) 的 RGB 图像
        array = array[:, :, ::-1]  # 翻转通道顺序，BGR -> RGB
        return array    # array是三维数组，形状是H*W个像素，每个像素又包含rgb三个通道的值，
    
    def save_frame_data(self, vehicle, images, dataset_path, frame_num, timestamp, traffic_light_state=None, is_stopped=None):
        """保存单帧数据：保存三路图像并返回要写入 CSV 的行数据列表"""
        try:
            # 获取车辆控制量
            """
            方法的返回值是carla.VehicleControl对象，这是个结构体包含当前车辆的控制状态，如转向、油门、刹车等信息。
             - steer: 转向角度，范围 [-1.0, 1.0]，负值表示向左转，正值表示向右转。
             - throttle: 油门值，范围 [0.0, 1.0]，0.0   表示不踩油门，1.0 表示全踩油门。
             - brake: 刹车值，范围 [0.0, 1.0]，0.0 表示不踩刹车，1.0 表示全踩刹车。
             - hand_brake: 手刹状态，布尔值，True 表示手刹拉起，False 表示手刹放下。
             - reverse: 倒车状态，布尔值，True 表示车辆处于倒车模式，False 表示车辆处于前进模式。
             - manual_gear_shift: 手动换挡状态，布尔值，True 表示启用手动换挡，False 表示自动换挡。
                …………略
            """
            control = vehicle.get_control()
            time.sleep(0.1)
            
            # 获取车辆速度
            """
            返回值是一个 carla.Vector3D 对象，包含车辆在 x、y、z 方向上的速度分量（单位为 m/s）。
            可以通过计算这个向量的模长来得到车辆的总速度，然后转换为 km/h。
            """
            velocity = vehicle.get_velocity()
            time.sleep(0.1)
            speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # 确保图像保存目录存在
            for position in ['center', 'left', 'right']:
                os.makedirs(dataset_path / f'images_{position}', exist_ok=True)

            # 保存图像并收集要写入 CSV 的数据
            csv_data = []   # 创建一个列表来存储当前帧的 CSV 数据行
            for position, image_array in images.items():
                filename = f'{position}_{frame_num}.png'    # 生成图像文件名，格式为 "摄像头位置_帧编号.png"
                image_path = dataset_path / f'images_{position}' / filename # 生成图像的文件路径
                # 将 RGB 转为 OpenCV 的 BGR 并写入磁盘（即保存命好名图片到路径）
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), image_bgr)

                # 根据左右摄像头添加方向偏移（用于训练时的数据增强/校正）
                base_steering = control.steer   # 获取方向盘转角的基础值
                if position == 'left':
                    steering_angle = base_steering + 0.15  # 左摄像头偏移
                elif position == 'right':
                    steering_angle = base_steering - 0.15  # 右摄像头偏移
                else:
                    steering_angle = base_steering

                # 将一行数据添加到 CSV 数据列表中，包含图像文件名、控制量、速度、摄像头位置、帧编号和时间戳
                csv_data.append({
                    'frame_filename': filename,
                    'steering_angle': f'{steering_angle:.6f}',
                    'throttle': f'{control.throttle:.6f}',
                    'brake': f'{control.brake:.6f}',
                    'speed_kmh': f'{speed_kmh:.2f}',
                    'camera_position': position,
                    'frame_number': frame_num,
                    'timestamp': f'{timestamp:.6f}',
                    'traffic_light_state': traffic_light_state if traffic_light_state is not None else 'unknown',
                    'is_stopped': '1' if is_stopped else '0'
                })
            
            return csv_data # 返回值是列表，每个元素是一个字典，包含当前帧的图像文件名、控制量、速度、摄像头位置、帧编号和时间戳等信息，用于后续写入 CSV 文件
            
        except Exception as e:
            print(f"Error saving frame data: {e}")
            return []

    def _get_traffic_light_state(self, vehicle):
        """尝试从车辆/附近交通灯获取红绿灯状态，返回字符串：'Red'/'Green'/'Yellow'/'unknown'"""
        try:
            # 尝试直接从 vehicle 获取（部分 CARLA 版本支持）
            if hasattr(vehicle, 'get_traffic_light_state'):
                state = vehicle.get_traffic_light_state()   # 没有灯时返回green
                return str(state)

            # 备选：获取车辆最近的 traffic light actor
            if hasattr(vehicle, 'get_traffic_light'):
                tl = vehicle.get_traffic_light()
                if tl is not None and hasattr(tl, 'get_state'):
                    return str(tl.get_state())

        except Exception:
            pass

        return 'unknown'
    
    def collect_data_for_town(self, town_name, thread_id):
        """对指定城镇执行数据采集主循环：负责世界/车辆初始化、逐帧保存和天气切换"""
        print(f"线程编号 {thread_id}: 开始采集地图 {town_name}的数据")
        
        # 创建采集数据集的保存路径为 "data_weathers/dataset_carla_人工编号_{town_name}"，
        dataset_path = Path(self.output_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # 创建 CSV 文件路径并初始化数据缓冲区，后续将批量写入 CSV 文件以提升性能
        csv_file_path = dataset_path / 'steering_data.csv'
        csv_data_buffer = []
        
        try:
            
            # 配置世界地图并获取 world 对象
            world = self.setup_world(town_name)
            time.sleep(3.0)
            if world is None:
                return
            
            # 在world对象创建车辆和摄像头
            vehicle, cameras = self.setup_vehicle_and_cameras(world)
            if vehicle is None or cameras is None:
                return
            
            time.sleep(3.0)
            # 用字典缓存最新三路图像以便同步保存
            latest_images = {pos: None for pos in ['center', 'left', 'right']}
            
            # 定义摄像头回调函数：每当摄像头捕获新图像时更新对应位置的缓存图像
            def camera_callback(image, position):
                latest_images[position] = self.process_image(image) # type: ignore
            
            # 注册摄像头的回调函数以获取图像
            for position, camera in cameras.items():
                # 开启摄像头监听，每当有新图像时调用 camera_callback 更新 latest_images 中对应位置的图像数据
                """
                camera.listen方法是carla中sensor的一个方法，用于注册一个回调函数，
                当摄像头捕获到新图像时会自动调用这个函数，并将图像数据作为参数传入。
                """
                camera.listen(lambda image, pos=position: camera_callback(image, pos))
            
            frame_count = 0 # 采集帧计数器初始化
            start_time = time.time()    # 记录采集开始时间，用于计算每帧的时间戳
            
            # ------------------------------------------------------------------
            # 动态天气调度：根据采集进度分阶段切换天气
            # ------------------------------------------------------------------
            # 根据天气类型平均划分每种天气多少桢
            phase_size = max(1, self.max_frames // len(self.weather_order))
            current_phase = 0   # 当前天气阶段索引
            # 设置天气，方法定义在下面
            self._apply_weather(world, self.weather_order[current_phase])

            print(f"线程编号 {thread_id}: 开始采集地图 {town_name}的数据")
            
            # 事件化保存参数：停车检测阈值、预保存帧、恢复后保存帧、在停驶时最大保存帧数
            SPEED_STOP_THRESH = 1.0  # km/h 以下视为静止
            SPEED_RESUME_THRESH = 2.0  # km/h 认为恢复行驶
            PRE_SAVE_FRAMES = 10
            POST_SAVE_FRAMES = 5
            MAX_SAVE_STOP_FRAMES = 3

            last_frames = collections.deque(maxlen=PRE_SAVE_FRAMES)
            stopped = False
            stop_saved_count = 0
            post_resume_left = 0

            while frame_count < self.max_frames:
                try:
                    # 世界更新一帧
                    world.tick()

                    # 等待所有传感器都获取到数据后，进行事件化保存
                    if all(img is not None for img in latest_images.values()):
                        current_time = time.time() - start_time

                        # 获取车辆状态用于事件判定
                        try:
                            control = vehicle.get_control()
                            velocity = vehicle.get_velocity()
                            speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                            traffic_state = self._get_traffic_light_state(vehicle)
                        except Exception:
                            control = None
                            speed_kmh = 0.0
                            traffic_state = 'unknown'

                        # 缓存当前帧到预保存队列
                        frame_entry = {
                            'images': latest_images.copy(),
                            'frame_num': frame_count,
                            'timestamp': current_time,
                            'control': control,
                            'speed_kmh': speed_kmh,
                            'traffic_light_state': traffic_state
                        }
                        last_frames.append(frame_entry)

                        # 如果正在做恢复后保存阶段，优先保存并计数
                        if post_resume_left > 0:
                            frame_data = self.save_frame_data(vehicle, frame_entry['images'], dataset_path, frame_entry['frame_num'], frame_entry['timestamp'], traffic_light_state=traffic_state, is_stopped=False)
                            csv_data_buffer.extend(frame_data)
                            post_resume_left -= 1
                            frame_count += 1

                        else:
                            # 非恢复阶段：根据速度判断进入/保持/退出停驶事件
                            if not stopped:
                                # 未停：检测是否进入停驶
                                if speed_kmh < SPEED_STOP_THRESH:
                                    # 进入停驶：保存预保存的若干帧 + 当前帧作为停靠帧
                                    for item in list(last_frames):
                                        fd = self.save_frame_data(vehicle, item['images'], dataset_path, item['frame_num'], item['timestamp'], traffic_light_state=item.get('traffic_light_state', 'unknown'), is_stopped=False)
                                        csv_data_buffer.extend(fd)

                                    fd = self.save_frame_data(vehicle, frame_entry['images'], dataset_path, frame_entry['frame_num'], frame_entry['timestamp'], traffic_light_state=traffic_state, is_stopped=True)
                                    csv_data_buffer.extend(fd)

                                    stopped = True
                                    stop_saved_count = 1
                                    last_frames.clear()
                                    frame_count += 1

                                else:
                                    # 正常行驶：按原方式保存当前帧
                                    fd = self.save_frame_data(vehicle, frame_entry['images'], dataset_path, frame_entry['frame_num'], frame_entry['timestamp'], traffic_light_state=traffic_state, is_stopped=False)
                                    csv_data_buffer.extend(fd)
                                    frame_count += 1

                            else:
                                # 当前处于停驶状态
                                if speed_kmh < SPEED_STOP_THRESH:
                                    # 仍在停驶：限制保存的停驶帧数以避免冗余
                                    if stop_saved_count < MAX_SAVE_STOP_FRAMES:
                                        fd = self.save_frame_data(vehicle, frame_entry['images'], dataset_path, frame_entry['frame_num'], frame_entry['timestamp'], traffic_light_state=traffic_state, is_stopped=True)
                                        csv_data_buffer.extend(fd)
                                        stop_saved_count += 1
                                    # 即使不保存，也推进帧计数以避免无限循环
                                    frame_count += 1
                                else:
                                    # 恢复行驶：保存若干恢复帧
                                    fd = self.save_frame_data(vehicle, frame_entry['images'], dataset_path, frame_entry['frame_num'], frame_entry['timestamp'], traffic_light_state=traffic_state, is_stopped=False)
                                    csv_data_buffer.extend(fd)
                                    post_resume_left = POST_SAVE_FRAMES
                                    stopped = False
                                    stop_saved_count = 0
                                    frame_count += 1

                        # 每隔若干数据批量写入 CSV（此处按 300 条记录写入，约 100 帧）
                        if len(csv_data_buffer) >= 300:  # 100 frames * 3 cameras
                            self.write_csv_data(csv_file_path, csv_data_buffer)
                            csv_data_buffer = []

                        if frame_count % 500 == 0 and frame_count > 0:
                            print(f"线程编号：{thread_id} | 世界地图：({town_name}) | 已经采集了 {frame_count} 桢数据")

                        # 判断是否达到切换到下一天气阶段的阈值
                        if (current_phase < len(self.weather_order) - 1 and 
                            frame_count >= (current_phase + 1) * phase_size):
                            current_phase += 1
                            next_weather = self.weather_order[current_phase]
                            self._apply_weather(world, next_weather)
                            print(f"线程编号：{thread_id} 世界地图：({town_name}): 天气已切换到 {next_weather}，在帧 {frame_count} 处")

                        # 重置图片缓存以等待下一帧的图像数据更新
                        latest_images = {pos: None for pos in ['center', 'left', 'right']}

                except Exception as e:
                    print(f"Thread {thread_id}: Error in collection loop: {e}")
                    break
            
            # 写入剩余未写入的 CSV 数据
            if csv_data_buffer:
                self.write_csv_data(csv_file_path, csv_data_buffer)

            print(f"线程编号 {thread_id}: 完成对于地图 {town_name}的数据采集，一共采集了 {frame_count} 桢数据")
            
        except Exception as e:
            print(f"Thread {thread_id}: Error in {town_name}: {e}")
        
        finally:
            # 清理已创建的演员（摄像头、车辆）并恢复 world 设置
            try:
                if 'cameras' in locals():
                    for camera in cameras.values(): # type: ignore
                        camera.destroy()
                if 'vehicle' in locals():
                    vehicle.destroy() # type: ignore
                
                # 恢复 world 的同步设置
                if 'world' in locals():
                    settings = world.get_settings() # type: ignore
                    settings.synchronous_mode = False
                    world.apply_settings(settings) # type: ignore
                    
            except Exception as e:
                print(f"Thread {thread_id}: Cleanup error: {e}")
    
    def write_csv_data(self, csv_file_path, csv_data_buffer):
        """将缓冲区中的字典数据写入到 CSV 文件，首次写入时添加表头"""
        try:
            file_exists = csv_file_path.exists()
            
            with open(csv_file_path, 'a', newline='') as csvfile:
                fieldnames = ['frame_filename', 'steering_angle', 'throttle', 'brake', 
                             'speed_kmh', 'camera_position', 'frame_number', 'timestamp',
                             'traffic_light_state', 'is_stopped']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(csv_data_buffer)
                
        except Exception as e:
            print(f"Error writing CSV data: {e}")
    
    def run_collection(self):
        """使用线程池并发运行多城镇的数据采集任务（当前按单个 map 列表运行）"""
        # map 列表中包含所有需要采集的地图的字符串，当然这里只用一个地图
        print(f"开始线程编号为 {len([self.map])}的线程")
        print(f"每个地图最大采集帧数为: {self.max_frames}")
        
        # 使用 ThreadPoolExecutor 管理线程，这里选择16线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for i, town in enumerate([self.map]):
                future = executor.submit(self.collect_data_for_town, town, i)
                futures.append(future)
            
            # 等待所有线程完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Thread completed with error: {e}")
        
        print("所有城镇的数据采集都完成了!")

    # ------------------------------------------------------------------
    # Weather helpers
    # ------------------------------------------------------------------

    def _get_weather_preset(self, weather_type):
        """根据字符串类型返回对应的 CARLA 天气预设（包含若干参数的结构体）；若无内置预设则构建自定义参数"""

        # 晴朗：使用内置的 ClearNoon
        if weather_type == 'sunny':
            # carla自带的晴朗天气预设，结构体参数已经设置好了，直接返回即可
            return carla.WeatherParameters.ClearNoon

        # 有雾：CARLA 没有纯雾预设，构造自定义参数
        if weather_type == 'foggy':
            return carla.WeatherParameters(
                cloudiness=10.0,
                precipitation=0.0,
                sun_altitude_angle=45.0,
                fog_density=75.0,
                fog_distance=0.0,
                wetness=0.0,
                wind_intensity=0.0
            )

        # 雨天：使用内置的 MidRainyNoon
        if weather_type == 'rainy':
            return carla.WeatherParameters.MidRainyNoon

        # 夜间：使用内置的 ClearNight
        if weather_type == 'night':
            return carla.WeatherParameters.ClearNight

        # 默认回退到晴朗
        return carla.WeatherParameters.ClearNoon

    def _apply_weather(self, world, weather_type):
        """将指定天气应用到 world；发生错误时回退为晴天并打印警告"""
        try:
            # 通过get_weather_preset方法获取对应天气的参数结构体，并通过set_weather方法应用到world中
            world.set_weather(self._get_weather_preset(weather_type))
        except Exception as e:
            # 出错时回退到晴朗天气
            print(f"Warning: Failed to apply weather '{weather_type}': {e}. Reverting to sunny.")
            world.set_weather(carla.WeatherParameters.ClearNoon)

def main():
    # 采集参数配置（支持命令行覆盖）
    parser = argparse.ArgumentParser(description='CARLA 数据采集脚本')
    parser.add_argument('--dataset_subname', type=str, default='default', help='将数据保存到 data_weathers/{dataset_subname} 目录')
    parser.add_argument('--max-frames', type=int, default=20000, help='每个地图采集的最大帧数')
    parser.add_argument('--map', type=str, default='Town01', help='要采集的地图名称')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA 主机')
    parser.add_argument('--port', type=int, default=2000, help='CARLA 端口')
    args = parser.parse_args()

    MAX_FRAMES = args.max_frames
    HOST = args.host
    BASE_PORT = args.port

    print("开始数据采集，配置如下：")
    print(f"每个地图最大采集帧数: {MAX_FRAMES}")
    print(f"数据保存目录: data_weathers/{args.dataset_subname}")
    print("核查Carla服务器端已经打开!")

    # 需要采集的地图
    maps = [args.map]
    """
    maps  =  
    [
        'Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt'
    ]
    """
   
    # 实例化数据采集器
    # 构造最终保存根目录为 data_weathers/{dataset-name}
    output_dir = str(Path('data_weathers') / args.dataset_subname)
    collector = CarlaDataCollector(host=HOST, port=BASE_PORT, max_frames=MAX_FRAMES, map=maps[0], output_dir=output_dir)

    time.sleep(5.0)


    # 开始采集
    try:
        collector.run_collection()
    except KeyboardInterrupt:
        print("\n采集进程被用户键盘中断")
    except Exception as e:
        print(f"Error during data collection: {e}")


if __name__ == "__main__":
    main()