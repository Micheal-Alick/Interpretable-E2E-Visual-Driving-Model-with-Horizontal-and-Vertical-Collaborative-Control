import concurrent.futures
import os
import time
from pathlib import Path
import argparse
import ast
from collections import deque

import carla
import cv2
import numpy as np

from dataset_collect_in_carla import CarlaDataCollector


class IntegratedCollectionDashboard:
    """Single-window dashboard for data collection visualization."""

    def __init__(self, history_size=2000):
        self.window_name = 'CARLA Collection Dashboard'
        self.panel_h = 720
        self.left_w = 960
        self.panel_w = 460
        self.sub_w = self.left_w // 2
        self.sub_h = self.panel_h // 2
        self.traj_history = deque(maxlen=max(50, int(history_size)))
        self.steer_history = deque(maxlen=max(50, int(history_size)))

    def update_history(self, elapsed_s, x, y, steer):
        self.traj_history.append((float(x), float(y)))
        self.steer_history.append((float(elapsed_s), float(steer)))

    def _prepare_sub_view(self, image_bgr, size, empty_text):
        width, height = size
        if image_bgr is None:
            view = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                view,
                empty_text,
                (12, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            return view
        return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    def _blit_sub_view(self, canvas, image_bgr, origin, title):
        x0, y0 = origin
        height, width = image_bgr.shape[:2]
        canvas[y0:y0 + height, x0:x0 + width] = image_bgr
        cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (85, 85, 85), 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + 24), (20, 20, 20), -1)
        cv2.putText(canvas, title, (x0 + 8, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)

    def _render_trajectory_view(self, road_x, road_y, width, height):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        margin = 20

        if len(self.traj_history) < 1:
            cv2.putText(canvas, 'Waiting for trajectory...', (14, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            return canvas

        traj = np.array(self.traj_history, dtype=np.float32)
        traj_x = traj[:, 0]
        traj_y = -traj[:, 1]  # Keep display orientation consistent with driving intuition.

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

        if road_x.size > 0 and road_y.size > 0:
            valid_road = np.isfinite(road_x) & np.isfinite(road_y)
            rx = road_x[valid_road]
            ry = -road_y[valid_road]
            road_mask = (rx >= x_min) & (rx <= x_max) & (ry >= y_min) & (ry <= y_max)
            rx = rx[road_mask]
            ry = ry[road_mask]
            if rx.size > 0:
                u, v = _map_xy(rx, ry)
                ui = np.clip(np.round(u).astype(np.int32), 0, width - 1)
                vi = np.clip(np.round(v).astype(np.int32), 0, height - 1)
                canvas[vi, ui] = (0, 255, 255)

        tu, tv = _map_xy(traj_x, traj_y)
        tpts = np.stack([np.round(tu).astype(np.int32), np.round(tv).astype(np.int32)], axis=1).reshape((-1, 1, 2))
        if len(tpts) >= 2:
            cv2.polylines(canvas, [tpts], isClosed=False, color=(255, 0, 0), thickness=2)
        if len(tpts) >= 1:
            cv2.circle(canvas, tuple(tpts[0, 0]), 4, (0, 255, 0), -1)
            cv2.circle(canvas, tuple(tpts[-1, 0]), 4, (0, 0, 255), -1)

        cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (70, 70, 70), 1)
        return canvas

    def _render_steer_history_view(self, width, height):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        margin = 30
        plot_w = max(width - 2 * margin, 2)
        plot_h = max(height - 2 * margin, 2)

        cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (70, 70, 70), 1)
        cv2.line(canvas, (margin, height // 2), (width - margin, height // 2), (55, 55, 55), 1)

        if len(self.steer_history) < 1:
            cv2.putText(canvas, 'Waiting for steering history...', (15, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
            return canvas

        arr = np.array(self.steer_history, dtype=np.float32)
        time_vals = arr[:, 0]
        steer_vals = arr[:, 1]

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

        cv2.putText(canvas, f't=[{t0:.1f}, {t1:.1f}] s', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
        cv2.putText(canvas, f'steer_now={steer_vals[-1]:+.3f}', (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
        return canvas

    def render(self, spectator_bgr, camera_bgr, road_x, road_y, stats):
        canvas = np.zeros((self.panel_h, self.left_w + self.panel_w, 3), dtype=np.uint8)

        top_left = self._prepare_sub_view(spectator_bgr, (self.sub_w, self.sub_h), 'No spectator camera')
        top_right = self._prepare_sub_view(camera_bgr, (self.sub_w, self.sub_h), 'No camera frame')
        bottom_left = self._render_trajectory_view(road_x, road_y, self.sub_w, self.sub_h)
        bottom_right = self._render_steer_history_view(self.sub_w, self.sub_h)

        self._blit_sub_view(canvas, top_left, (0, 0), 'Spectator View')
        self._blit_sub_view(canvas, top_right, (self.sub_w, 0), 'Camera View')
        self._blit_sub_view(canvas, bottom_left, (0, self.sub_h), 'Live Trajectory on Road')
        self._blit_sub_view(canvas, bottom_right, (self.sub_w, self.sub_h), 'Steering History')

        x_off = self.left_w
        cv2.rectangle(canvas, (x_off, 0), (x_off + self.panel_w, self.panel_h), (18, 18, 18), -1)
        cv2.rectangle(canvas, (x_off, 0), (x_off + self.panel_w, 56), (25, 25, 25), -1)
        cv2.putText(canvas, 'Collection Dashboard', (x_off + 14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
        cv2.putText(canvas, 'Real-time Status', (x_off + 14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (185, 185, 185), 1)

        y = 92
        line_gap = 34
        cv2.putText(canvas, f'Elapsed Time: {stats["elapsed_s"]:.1f} s', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (220, 220, 220), 2)
        y += line_gap
        cv2.putText(canvas, f'Frame: {stats["frame"]}', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (220, 220, 220), 2)

        y += line_gap + 8
        cv2.putText(canvas, f'Map: {stats["map_name"]}', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
        y += line_gap
        cv2.putText(canvas, f'Weather: {stats["weather_name"]}', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)

        y += line_gap + 8
        cv2.putText(canvas, f'Throttle: {stats["throttle"]:.3f}', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (50, 220, 50), 2)
        y += line_gap
        cv2.putText(canvas, f'Brake: {stats["brake"]:.3f}', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (60, 110, 255), 2)

        y += line_gap + 8
        cv2.putText(canvas, f'Speed: {stats["speed_kmh"]:.1f} km/h', (x_off + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 215, 255), 2)

        cv2.putText(canvas, 'Keys: Q Quit  S Save Screenshot', (x_off + 18, self.panel_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
        return canvas


class CarlaDataCollectorVisualization(CarlaDataCollector):
    """在原始采集器基础上增加可视化能力：
    1) 让 CARLA 的 spectator 跟随车辆，便于在 CarlaUE4 窗口观察行驶过程
    2) 在 OpenCV 中显示统一 dashboard（多视图 + 状态信息）
    """

    def __init__(
        self,
        host='localhost',
        port=2000,
        max_frames=3000,
        map='Town01',
        output_dir='data_weathers',
        enable_spectator_follow=True,
        enable_cv_preview=True,
        simulation_fps=20,
        capture_interval_ticks=2,
        spectator_smooth_alpha=0.35,
        weathers_choices=None,
    ):
        super().__init__(host=host, port=port, max_frames=max_frames, map=map, output_dir=output_dir)
        self.enable_spectator_follow = enable_spectator_follow
        self.enable_cv_preview = enable_cv_preview
        self.dashboard = IntegratedCollectionDashboard(history_size=max_frames)
        self.latest_spectator_image = None
        self.simulation_fps = max(5, int(simulation_fps))
        self.capture_interval_ticks = max(1, int(capture_interval_ticks))
        self.spectator_smooth_alpha = max(0.0, min(1.0, float(spectator_smooth_alpha)))
        self._last_spectator_transform = None

        if weathers_choices is not None:
            cleaned = [str(x).strip() for x in weathers_choices if str(x).strip()]
            if cleaned:
                self.weather_order = cleaned

    def setup_world(self, town_name):
        """复用父类建图逻辑，并将同步步长调整为可视化友好的帧率。"""
        world = super().setup_world(town_name)
        if world is None:
            return None

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.simulation_fps)
        world.apply_settings(settings)
        return world

    def _lerp_angle(self, current, target, alpha):
        """角度线性插值（处理 180/-180 跨界）。"""
        delta = (target - current + 180.0) % 360.0 - 180.0
        return current + delta * alpha

    def _update_spectator_view(self, world, vehicle):
        """将 spectator 相机放到车辆后上方，实现跟车视角。"""
        if not self.enable_spectator_follow:
            return

        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        forward = transform.get_forward_vector()

        # 车辆后方 8 米、上方 3.5 米的第三人称视角
        follow_location = carla.Location(
            x=location.x - 8.0 * forward.x,
            y=location.y - 8.0 * forward.y,
            z=location.z + 3.5,
        )
        target_transform = carla.Transform(
            follow_location,
            carla.Rotation(pitch=-15.0, yaw=rotation.yaw, roll=0.0),
        )

        # 对 spectator 视角做轻量平滑，减少“跳帧感”。
        if self._last_spectator_transform is None or self.spectator_smooth_alpha <= 0.0:
            new_transform = target_transform
        else:
            alpha = self.spectator_smooth_alpha
            last = self._last_spectator_transform
            new_loc = carla.Location(
                x=last.location.x + (target_transform.location.x - last.location.x) * alpha,
                y=last.location.y + (target_transform.location.y - last.location.y) * alpha,
                z=last.location.z + (target_transform.location.z - last.location.z) * alpha,
            )
            new_rot = carla.Rotation(
                pitch=last.rotation.pitch + (target_transform.rotation.pitch - last.rotation.pitch) * alpha,
                yaw=self._lerp_angle(last.rotation.yaw, target_transform.rotation.yaw, alpha),
                roll=last.rotation.roll + (target_transform.rotation.roll - last.rotation.roll) * alpha,
            )
            new_transform = carla.Transform(new_loc, new_rot)

        spectator.set_transform(new_transform)
        self._last_spectator_transform = new_transform

    def _spawn_dashboard_spectator_camera(self, world, vehicle):
        """创建用于 dashboard 左上角显示的第三人称相机。"""
        try:
            bp_lib = world.get_blueprint_library()
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '100')

            camera_transform = carla.Transform(
                carla.Location(x=-8.0, z=3.5),
                carla.Rotation(pitch=-15.0),
            )
            return world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        except Exception as e:
            print(f'Failed to spawn dashboard spectator camera: {e}')
            return None

    def _dashboard_spectator_callback(self, image):
        """缓存 dashboard 专用 spectator 视角图像。"""
        self.latest_spectator_image = self.process_image(image)

    def _get_map_waypoints_xy(self, world, step=2.0):
        """提取道路采样点，用于轨迹与路网叠加显示。"""
        try:
            waypoints = world.get_map().generate_waypoints(float(step))
            if not waypoints:
                return np.array([]), np.array([])

            road_x = np.array([wp.transform.location.x for wp in waypoints], dtype=np.float32)
            road_y = np.array([wp.transform.location.y for wp in waypoints], dtype=np.float32)
            return road_x, road_y
        except Exception as e:
            print(f'Failed to build road waypoint background: {e}')
            return np.array([]), np.array([])

    def _save_frame_data_fast(self, vehicle, images, dataset_path, frame_num, timestamp):
        """轻量保存单帧数据：避免在采集主线程中引入额外 sleep。"""
        try:
            control = vehicle.get_control()
            velocity = vehicle.get_velocity()
            speed_kmh = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            for position in ['center', 'left', 'right']:
                os.makedirs(dataset_path / f'images_{position}', exist_ok=True)

            csv_data = []
            for position, image_array in images.items():
                filename = f'{position}_{frame_num}.png'
                image_path = dataset_path / f'images_{position}' / filename

                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), image_bgr)

                base_steering = control.steer
                if position == 'left':
                    steering_angle = base_steering + 0.15
                elif position == 'right':
                    steering_angle = base_steering - 0.15
                else:
                    steering_angle = base_steering
                # 补充 traffic_light_state 与 is_stopped 字段
                try:
                    traffic_state = self._get_traffic_light_state(vehicle)
                except Exception:
                    traffic_state = 'unknown'

                is_stopped = speed_kmh < 1.0

                csv_data.append(
                    {
                        'frame_filename': filename,
                        'steering_angle': f'{steering_angle:.6f}',
                        'throttle': f'{control.throttle:.6f}',
                        'brake': f'{control.brake:.6f}',
                        'speed_kmh': f'{speed_kmh:.2f}',
                        'camera_position': position,
                        'frame_number': frame_num,
                        'timestamp': f'{timestamp:.6f}',
                        'traffic_light_state': traffic_state,
                        'is_stopped': '1' if is_stopped else '0',
                    }
                )

            return csv_data
        except Exception as e:
            print(f"Error saving frame data (fast): {e}")
            return []

    def collect_data_for_town(self, town_name, thread_id=0):
        """带可视化的数据采集主循环。"""
        print(f"线程编号 {thread_id}: 开始采集地图 {town_name} 的数据（可视化版本）")

        dataset_path = Path(self.output_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        csv_file_path = dataset_path / 'steering_data.csv'
        csv_data_buffer = []

        try:
            world = self.setup_world(town_name)
            time.sleep(2.0)
            if world is None:
                return

            vehicle, cameras = self.setup_vehicle_and_cameras(world)
            if vehicle is None or cameras is None:
                return

            road_x, road_y = self._get_map_waypoints_xy(world, step=2.0)
            dashboard_camera = None
            self.latest_spectator_image = None
            self.dashboard = IntegratedCollectionDashboard(history_size=self.max_frames)
            if self.enable_cv_preview:
                dashboard_camera = self._spawn_dashboard_spectator_camera(world, vehicle)
                if dashboard_camera is not None:
                    dashboard_camera.listen(self._dashboard_spectator_callback)
                cv2.namedWindow(self.dashboard.window_name, cv2.WINDOW_AUTOSIZE)

            latest_images = {pos: None for pos in ['center', 'left', 'right']}

            def camera_callback(image, position):
                latest_images[position] = self.process_image(image)  # type: ignore

            for position, camera in cameras.items():
                camera.listen(lambda image, pos=position: camera_callback(image, pos))

            frame_count = 0
            sim_tick_count = 0
            start_time = time.time()
            pending_futures = []
            io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

            phase_size = max(1, self.max_frames // len(self.weather_order))
            current_phase = 0
            self._apply_weather(world, self.weather_order[current_phase])

            while frame_count < self.max_frames:
                world.tick()
                sim_tick_count += 1

                # 每个 tick 都刷新 spectator 跟车视角
                self._update_spectator_view(world, vehicle)

                control = vehicle.get_control()
                velocity = vehicle.get_velocity()
                speed_kmh = 3.6 * np.sqrt(
                    velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                )
                weather_name = self.weather_order[current_phase]
                elapsed_s = time.time() - start_time

                vehicle_transform = vehicle.get_transform()
                self.dashboard.update_history(
                    elapsed_s=elapsed_s,
                    x=vehicle_transform.location.x,
                    y=vehicle_transform.location.y,
                    steer=control.steer,
                )

                if self.enable_cv_preview:
                    camera_bgr = None
                    if latest_images['center'] is not None:
                        camera_bgr = cv2.cvtColor(latest_images['center'], cv2.COLOR_RGB2BGR)

                    spectator_bgr = None
                    if self.latest_spectator_image is not None:
                        spectator_bgr = cv2.cvtColor(self.latest_spectator_image, cv2.COLOR_RGB2BGR)

                    dashboard_frame = self.dashboard.render(
                        spectator_bgr=spectator_bgr,
                        camera_bgr=camera_bgr,
                        road_x=road_x,
                        road_y=road_y,
                        stats={
                            'elapsed_s': elapsed_s,
                            'frame': sim_tick_count,
                            'map_name': town_name,
                            'weather_name': weather_name,
                            'throttle': control.throttle,
                            'brake': control.brake,
                            'speed_kmh': speed_kmh,
                        },
                    )
                    cv2.imshow(self.dashboard.window_name, dashboard_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print('Detected q key, stop collection early.')
                        break
                    if key == ord('s'):
                        ts = time.strftime('%Y%m%d_%H%M%S')
                        fname = f'collection_dashboard_{ts}_{sim_tick_count}.png'
                        cv2.imwrite(fname, dashboard_frame)
                        print(f'Saved screenshot: {fname}')

                # 每 N 个仿真 tick 执行一次数据落盘，减少 I/O 对可视化流畅度的影响
                if (
                    sim_tick_count % self.capture_interval_ticks == 0
                    and all(img is not None for img in latest_images.values())
                ):
                    current_time = time.time() - start_time

                    # 将图像写盘放到后台线程，减少主循环阻塞
                    future = io_executor.submit(
                        self._save_frame_data_fast,
                        vehicle,
                        latest_images.copy(),
                        dataset_path,
                        frame_count,
                        current_time,
                    )
                    pending_futures.append(future)

                    # 非阻塞回收已完成任务
                    for task in pending_futures[:]:
                        if task.done():
                            try:
                                frame_data = task.result()
                                if frame_data:
                                    csv_data_buffer.extend(frame_data)
                            except Exception as e:
                                print(f"后台保存任务异常: {e}")
                            pending_futures.remove(task)

                    if len(csv_data_buffer) >= 300:
                        self.write_csv_data(csv_file_path, csv_data_buffer)
                        csv_data_buffer = []

                    frame_count += 1
                    if frame_count % 500 == 0:
                        print(
                            f"线程编号：{thread_id} | 地图：{town_name} | 已采集 {frame_count} 帧"
                        )

                    if (
                        current_phase < len(self.weather_order) - 1
                        and frame_count >= (current_phase + 1) * phase_size
                    ):
                        current_phase += 1
                        next_weather = self.weather_order[current_phase]
                        self._apply_weather(world, next_weather)
                        print(
                            f"线程编号：{thread_id} | 地图：{town_name} | "
                            f"天气切换到 {next_weather}（帧 {frame_count}）"
                        )

                    latest_images = {pos: None for pos in ['center', 'left', 'right']}

            # 回收剩余后台任务并写出剩余 CSV
            for task in pending_futures:
                try:
                    frame_data = task.result()
                    if frame_data:
                        csv_data_buffer.extend(frame_data)
                except Exception as e:
                    print(f"后台保存任务异常: {e}")

            io_executor.shutdown(wait=True)

            if csv_data_buffer:
                self.write_csv_data(csv_file_path, csv_data_buffer)

            print(
                f"线程编号 {thread_id}: 地图 {town_name} 采集完成，共采集 {frame_count} 帧（可视化版本）"
            )
            print(f"数据保存目录: {dataset_path.resolve()}")
            print(f"标注CSV路径: {csv_file_path.resolve()}")

        except Exception as e:
            print(f"线程编号 {thread_id}: 地图 {town_name} 采集异常: {e}")

        finally:
            try:
                if self.enable_cv_preview:
                    cv2.destroyAllWindows()

                if 'dashboard_camera' in locals() and dashboard_camera is not None:
                    dashboard_camera.destroy()

                if 'cameras' in locals():
                    for camera in cameras.values():  # type: ignore
                        camera.destroy()
                if 'vehicle' in locals():
                    vehicle.destroy()  # type: ignore

                if 'world' in locals():
                    settings = world.get_settings()  # type: ignore
                    settings.synchronous_mode = False
                    world.apply_settings(settings)  # type: ignore

            except Exception as e:
                print(f"线程编号 {thread_id}: 清理资源异常: {e}")

    def run_collection(self):
        """可视化模式下采用单线程执行，避免 GUI 预览在多线程中的兼容问题。"""
        print('开始可视化采集流程...')
        print(f'目标地图: {self.map}')
        print(f'最大采集帧数: {self.max_frames}')
        print(f'仿真帧率(FPS): {self.simulation_fps}')
        print(f'每 {self.capture_interval_ticks} 个 tick 保存 1 帧')
        print(f'CarlaUE4 跟车视角: {self.enable_spectator_follow}')
        print(f'OpenCV Dashboard: {self.enable_cv_preview}')

        self.collect_data_for_town(self.map, thread_id=0)
        print('可视化采集流程结束。')


def main():
    def parse_weathers_choices(value):
        """Parse weather list from either Python-list string or single weather token."""
        text = str(value).strip()
        if not text:
            return []

        # Support: "['sunny','foggy']" or "[\"sunny\",\"foggy\"]"
        if text.startswith('[') and text.endswith(']'):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception as e:
                raise argparse.ArgumentTypeError(f'Invalid weathers_choices list: {value}') from e

        # Support: "sunny,foggy,rainy" or single token
        if ',' in text:
            return [item.strip() for item in text.split(',') if item.strip()]
        return [text]

    parser = argparse.ArgumentParser(description='CARLA 可视化数据采集')
    parser.add_argument('--dataset_subname', type=str, default='default', help='将数据保存到 data_weathers/{dataset_subname} 目录')
    parser.add_argument('--max-frames', type=int, default=30000, help='每个地图采集的最大帧数')
    parser.add_argument('--map', type=str, default='Town01', help='要采集的地图名称')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA 主机')
    parser.add_argument('--port', type=int, default=2000, help='CARLA 端口')
    parser.add_argument('--no-spectator', dest='enable_spectator_follow', action='store_false', help='禁用 spectator 跟随')
    parser.add_argument('--no-preview', dest='enable_cv_preview', action='store_false', help='禁用 OpenCV Dashboard')
    parser.add_argument('--simulation-fps', type=int, default=20, help='可视化模式下仿真帧率')
    parser.add_argument('--capture-interval-ticks', type=int, default=2, help='每 N tick 保存一帧图片')
    parser.add_argument(
        '--weathers_choices',
        type=parse_weathers_choices,
        nargs='*',
        default=['sunny', 'foggy', 'rainy', 'night'],
        help="天气列表，支持多种写法：--weathers_choices sunny foggy 或 --weathers_choices \"['sunny','foggy']\"",
    )
    args = parser.parse_args()

    if args.weathers_choices and isinstance(args.weathers_choices[0], list):
        weathers_choices = args.weathers_choices[0]
    else:
        weathers_choices = args.weathers_choices

    weathers_choices = [str(x).strip() for x in weathers_choices if str(x).strip()]
    if not weathers_choices:
        weathers_choices = ['sunny', 'foggy', 'rainy', 'night']

    MAX_FRAMES = args.max_frames
    HOST = args.host
    BASE_PORT = args.port
    MAP_NAME = args.map

    print('开始可视化数据采集，配置如下：')
    print(f'地图: {MAP_NAME}')
    print(f'每个地图最大采集帧数: {MAX_FRAMES}')
    print(f'数据保存目录: data_weathers/{args.dataset_subname}')
    print(f'天气序列: {weathers_choices}')
    print('请确认 CarlaUE4.exe 已启动并保持渲染窗口可见。')

    # 构造最终保存根目录为 data_weathers/{dataset-name}
    output_dir = str(Path('data_weathers') / args.dataset_subname)

    collector = CarlaDataCollectorVisualization(
        host=HOST,
        port=BASE_PORT,
        max_frames=MAX_FRAMES,
        map=MAP_NAME,
        output_dir=output_dir,
        enable_spectator_follow=args.enable_spectator_follow,
        enable_cv_preview=args.enable_cv_preview,
        simulation_fps=args.simulation_fps,
        capture_interval_ticks=args.capture_interval_ticks,
        weathers_choices=weathers_choices,
    )

    time.sleep(2.0)

    try:
        collector.run_collection()
    except KeyboardInterrupt:
        print('\n采集进程被用户中断。')
    except Exception as e:
        print(f'采集过程中发生错误: {e}')


if __name__ == '__main__':
    main()
