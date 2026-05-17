"""
ability_test.py

Collect and report simulation ability metrics, then append one row to ability_test.csv.
"""

import csv
import time
from pathlib import Path

import carla
import numpy as np


class AbilityMetricsTracker:
    """Track required driving-ability metrics for one simulation run."""

    CSV_HEADER = [
        "run_start_datetime",
        "weather_type",
        "spawn_location",
        "avg_speed_kmh",
        "avg_impact_jerk_mps3",
        "avg_steer_rate_per_s",
        "ignored_red_light_count",
        "collision_count",
        "lane_invasion_count",
        "task_completed",
    ]

    # 初始化能力评测器并配置 CSV 输出路径。
    def __init__(self, csv_path=None):
        self.csv_path = Path(csv_path) if csv_path is not None else Path(__file__).with_name("ability_test.csv")
        self._world_map = None
        self._warning_printed = {
            "traffic_light": False,
            "lane_check": False,
        }
        self.reset()

    # 重置单次仿真过程中的全部统计状态。
    def reset(self):
        self.weather_type = ""
        self.spawn_location = ""
        self._run_start_wall_time = None
        self._run_end_wall_time = None

        self._elapsed_s = []
        self._speed_kmh = []
        self._speed_kmh_without_red_wait = []
        self._steer = []

        self.ignored_red_light_count = 0
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.wrong_lane_detected = False
        self.offroad_stuck_detected = False
        self.roadside_collision_stuck_detected = False
        self.abnormal_stop_detected = False

        self._red_violation_latched = False
        self._offroad_start_elapsed = None
        self._abnormal_stop_start_elapsed = None
        self._is_offroad_current = False
        self._is_in_junction_current = False

        self._last_collision_stamp = -1e9
        self._last_lane_invasion_stamp = -1e9
        self._collision_cooldown_s = 1.0
        self._lane_invasion_cooldown_s = 0.5

    # 在仿真启动时初始化本次评测上下文信息。
    def start_run(self, weather_type, spawn_location, world_map, run_start_wall_time=None):
        self.reset()
        self.weather_type = str(weather_type)
        self.spawn_location = str(spawn_location)
        self._world_map = world_map
        self._run_start_wall_time = float(run_start_wall_time if run_start_wall_time is not None else time.time())

    # 记录一次碰撞事件（过滤车辆/行人碰撞并做冷却去重）。
    def record_collision(self, other_actor_type=None):
        if other_actor_type is not None:
            other_actor_type = str(other_actor_type)
            if other_actor_type.startswith("vehicle.") or other_actor_type.startswith("walker."):
                return

        now = time.time()
        if now - self._last_collision_stamp >= self._collision_cooldown_s:
            self.collision_count += 1
            self._last_collision_stamp = now

    # 记录一次压线事件并做时间冷却去重。
    def record_lane_invasion(self):
        now = time.time()
        if now - self._last_lane_invasion_stamp >= self._lane_invasion_cooldown_s:
            self.lane_invasion_count += 1
            self._last_lane_invasion_stamp = now

    # 每帧更新速度、转角和行为事件相关统计。
    def update_step(self, elapsed_s, speed_kmh, steer, throttle, brake, vehicle):
        elapsed_s = float(elapsed_s)
        speed_kmh = float(speed_kmh)
        steer = float(steer)
        throttle = float(throttle)
        brake = float(brake)

        self._elapsed_s.append(elapsed_s)
        self._speed_kmh.append(speed_kmh)
        self._steer.append(steer)

        red_light_ahead = self._is_red_light_ahead(vehicle=vehicle)
        is_red_waiting = red_light_ahead and speed_kmh <= 2.0 and (brake >= 0.10 or throttle <= 0.15)
        if not is_red_waiting:
            self._speed_kmh_without_red_wait.append(speed_kmh)

        self._update_red_light_violation(
            speed_kmh=speed_kmh,
            throttle=throttle,
            brake=brake,
            red_light_ahead=red_light_ahead,
        )
        self._update_lane_and_road_status(elapsed_s=elapsed_s, speed_kmh=speed_kmh, vehicle=vehicle)
        self._update_abnormal_stop_status(
            elapsed_s=elapsed_s,
            speed_kmh=speed_kmh,
            red_light_ahead=red_light_ahead,
            is_red_waiting=is_red_waiting,
        )

    # 判断车辆当前前方是否为红灯。
    def _is_red_light_ahead(self, vehicle):
        red_light_ahead = False
        try:
            if vehicle is not None and vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                red_light_ahead = traffic_light is not None and traffic_light.get_state() == carla.TrafficLightState.Red
        except Exception as e:
            if not self._warning_printed["traffic_light"]:
                print(f"Warning: failed to read traffic light state for metrics: {e}")
                self._warning_printed["traffic_light"] = True
        return red_light_ahead

    # 判断并统计闯红灯行为次数。
    def _update_red_light_violation(self, speed_kmh, throttle, brake, red_light_ahead):
        violates_red = red_light_ahead and speed_kmh > 3.0 and throttle > 0.10 and brake < 0.10
        if violates_red and not self._red_violation_latched:
            self.ignored_red_light_count += 1
            self._red_violation_latched = True

        should_release_latch = (not red_light_ahead) or speed_kmh < 0.8 or brake > 0.2
        if should_release_latch:
            self._red_violation_latched = False

    # 判断逆行与离路卡死等任务失败状态。
    def _update_lane_and_road_status(self, elapsed_s, speed_kmh, vehicle):
        if self._world_map is None or vehicle is None:
            return

        self._is_offroad_current = False
        self._is_in_junction_current = False

        try:
            location = vehicle.get_location()
            waypoint = self._world_map.get_waypoint(
                location,
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                self._is_offroad_current = True
                if self._offroad_start_elapsed is None:
                    self._offroad_start_elapsed = elapsed_s
                offroad_duration = elapsed_s - self._offroad_start_elapsed
                if offroad_duration >= 2.0 and speed_kmh < 1.0:
                    self.offroad_stuck_detected = True
                    if self.collision_count > 0:
                        self.roadside_collision_stuck_detected = True
                return

            self._offroad_start_elapsed = None
            self._is_in_junction_current = bool(waypoint.is_junction)

            if waypoint.is_junction or speed_kmh < 5.0:
                return

            vehicle_forward = vehicle.get_transform().get_forward_vector()
            lane_forward = waypoint.transform.get_forward_vector()
            direction_dot = (
                vehicle_forward.x * lane_forward.x
                + vehicle_forward.y * lane_forward.y
                + vehicle_forward.z * lane_forward.z
            )
            if direction_dot < -0.1:
                self.wrong_lane_detected = True
        except Exception as e:
            if not self._warning_printed["lane_check"]:
                print(f"Warning: failed to read lane state for metrics: {e}")
                self._warning_printed["lane_check"] = True

    # 判断是否出现非红灯等待导致的异常停车。
    def _update_abnormal_stop_status(self, elapsed_s, speed_kmh, red_light_ahead, is_red_waiting):
        if elapsed_s < 5.0:
            self._abnormal_stop_start_elapsed = None
            return

        should_track_abnormal_stop = (
            (not red_light_ahead)
            and (not is_red_waiting)
            and (not self._is_offroad_current)
            and (not self._is_in_junction_current)
            and speed_kmh <= 0.8
        )

        if should_track_abnormal_stop:
            if self._abnormal_stop_start_elapsed is None:
                self._abnormal_stop_start_elapsed = elapsed_s
            elif (elapsed_s - self._abnormal_stop_start_elapsed) >= 8.0:
                self.abnormal_stop_detected = True
        else:
            self._abnormal_stop_start_elapsed = None

    # 计算纵向冲击度（加速度变化率）平均值。
    @staticmethod
    def _compute_avg_impact_jerk(elapsed_s, speed_kmh):
        if len(elapsed_s) < 3:
            return 0.0

        t = np.array(elapsed_s, dtype=np.float64)
        v = np.array(speed_kmh, dtype=np.float64) / 3.6

        jerk_vals = []
        for idx in range(2, len(t)):
            dt_prev = t[idx - 1] - t[idx - 2]
            dt_now = t[idx] - t[idx - 1]
            if dt_prev <= 1e-6 or dt_now <= 1e-6:
                continue
            a_prev = (v[idx - 1] - v[idx - 2]) / dt_prev
            a_now = (v[idx] - v[idx - 1]) / dt_now
            jerk_vals.append(abs(a_now - a_prev) / dt_now)

        if not jerk_vals:
            return 0.0
        return float(np.mean(jerk_vals))

    # 在转向事件区间内计算平均转角变化率。
    @staticmethod
    def _compute_avg_steer_rate_in_events(elapsed_s, steer):
        if len(elapsed_s) < 2:
            return 0.0

        t = np.array(elapsed_s, dtype=np.float64)
        s = np.array(steer, dtype=np.float64)

        trigger_th = 0.10
        zero_th = 0.02
        rates = []
        in_event = False
        seen_near_zero = True

        for idx in range(1, len(t)):
            dt = t[idx] - t[idx - 1]
            if dt <= 1e-6:
                continue

            prev_abs = abs(s[idx - 1])
            curr_abs = abs(s[idx])

            if prev_abs <= zero_th:
                seen_near_zero = True

            if (not in_event) and seen_near_zero and curr_abs > trigger_th:
                in_event = True
                seen_near_zero = False

            if in_event:
                rates.append(abs(s[idx] - s[idx - 1]) / dt)
                if curr_abs <= zero_th:
                    in_event = False
                    seen_near_zero = True

        if not rates:
            return 0.0
        return float(np.mean(rates))

    # 汇总并生成七项能力评测指标结果。
    def _build_metrics(self):
        avg_speed = (
            float(np.mean(self._speed_kmh_without_red_wait))
            if self._speed_kmh_without_red_wait
            else 0.0
        )
        avg_impact_jerk = self._compute_avg_impact_jerk(self._elapsed_s, self._speed_kmh)
        avg_steer_rate = self._compute_avg_steer_rate_in_events(self._elapsed_s, self._steer)

        has_valid_run = len(self._elapsed_s) > 0
        task_completed = int(
            has_valid_run
            and (not self.roadside_collision_stuck_detected)
            and (not self.abnormal_stop_detected)
        )

        return {
            "avg_speed_kmh": avg_speed,
            "avg_impact_jerk_mps3": avg_impact_jerk,
            "avg_steer_rate_per_s": avg_steer_rate,
            "ignored_red_light_count": int(self.ignored_red_light_count),
            "collision_count": int(self.collision_count),
            "lane_invasion_count": int(self.lane_invasion_count),
            "task_completed": int(task_completed),
        }

    # 将一次仿真结果按固定列顺序追加写入 CSV。
    def _append_csv_row(self, run_start_datetime, metrics):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        need_header = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(self.CSV_HEADER)
            writer.writerow(
                [
                    str(run_start_datetime),
                    self.weather_type,
                    self.spawn_location,
                    f"{metrics['avg_speed_kmh']:.6f}",
                    f"{metrics['avg_impact_jerk_mps3']:.6f}",
                    f"{metrics['avg_steer_rate_per_s']:.6f}",
                    int(metrics["ignored_red_light_count"]),
                    int(metrics["collision_count"]),
                    int(metrics["lane_invasion_count"]),
                    int(metrics["task_completed"]),
                ]
            )

    # 结束评测、打印指标，并落盘到 CSV。
    def finalize_and_report(self, run_end_wall_time=None):
        if self._run_start_wall_time is None:
            self._run_start_wall_time = time.time()
        self._run_end_wall_time = float(run_end_wall_time if run_end_wall_time is not None else time.time())
        run_start_datetime = time.strftime("%Y%m%d%H%M", time.localtime(self._run_start_wall_time))

        metrics = self._build_metrics()
        self._append_csv_row(run_start_datetime, metrics)

        print("\nABILITY TEST METRICS")
        print("=" * 60)
        print(f"Average speed (km/h): {metrics['avg_speed_kmh']:.4f}")
        print(f"Average impact jerk (m/s^3): {metrics['avg_impact_jerk_mps3']:.6f}")
        print(f"Average steer rate in events (1/s): {metrics['avg_steer_rate_per_s']:.6f}")
        print(f"Ignored red-light count: {metrics['ignored_red_light_count']}")
        print(f"Collision count: {metrics['collision_count']}")
        print(f"Lane invasion count: {metrics['lane_invasion_count']}")
        print(f"Task completed (0/1): {metrics['task_completed']}")
        print(f"Metrics CSV updated: {self.csv_path}")
        print("=" * 60)

        return metrics
