"""摩托车聚集检测策略模块 - 用于检测夜间红外飙车场景"""
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from enum import Enum


class TrackingState(Enum):
    """追踪状态枚举"""
    IDLE = "idle"  # 空闲状态，未进入追踪模式
    TRACKING = "tracking"  # 追踪中
    LOST = "lost"  # 目标丢失，但未超时
    FAILED = "failed"  # 追踪失败


class MotorcycleGatheringStrategy(ABC):
    """摩托车聚集检测策略基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def detect_gathering(self, motorcycle_boxes: List[Dict]) -> List[int]:
        """
        检测聚集的摩托车组

        Args:
            motorcycle_boxes: 摩托车目标框列表

        Returns:
            List[int]: 参与聚集的目标框索引列表
        """
        pass

    @staticmethod
    def box_center(box: Dict) -> Tuple[float, float]:
        """获取目标框中心点"""
        return (box['x'], box['y'])


class DistanceGatheringStrategy(MotorcycleGatheringStrategy):
    """基于距离的聚集检测策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.distance_threshold = config.get('distance_threshold', 150)  # 默认150像素
        self.min_gathering_count = config.get('min_gathering_count', 3)  # 最少聚集数量
        self.use_center_distance = config.get('use_center_distance', True)

    def detect_gathering(self, motorcycle_boxes: List[Dict]) -> List[int]:
        """
        基于距离检测聚集的摩托车

        算法思路：
        1. 计算所有目标框之间的距离矩阵
        2. 找出每个目标框的邻居（距离小于阈值）
        3. 找出满足最小聚集数量的目标组
        """
        from .logger import log_task_debug

        if len(motorcycle_boxes) < self.min_gathering_count:
            log_task_debug(
                f"[模型8验证] 聚集检测失败 - 目标数量不足: {len(motorcycle_boxes)} < {self.min_gathering_count}"
            )
            return []

        n = len(motorcycle_boxes)
        # 构建邻接矩阵（距离小于阈值则为邻居）
        adjacency = [[False] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                distance = self.calculate_distance(motorcycle_boxes[i], motorcycle_boxes[j])
                if distance <= self.distance_threshold:
                    adjacency[i][j] = True
                    adjacency[j][i] = True

        # 使用连通分量算法找出聚集组
        visited = [False] * n
        gathering_indices = []

        for i in range(n):
            if not visited[i]:
                # BFS找出连通分量
                component = []
                queue = [i]
                visited[i] = True

                while queue:
                    node = queue.pop(0)
                    component.append(node)

                    for neighbor in range(n):
                        if adjacency[node][neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)

                # 如果连通分量大小满足最小聚集数量，则加入结果
                if len(component) >= self.min_gathering_count:
                    gathering_indices.extend(component)

        if gathering_indices:
            log_task_debug(
                f"[模型8验证] 聚集检测通过 - 检测到{len(gathering_indices)}个聚集目标"
            )
        else:
            log_task_debug(
                f"[模型8验证] 聚集检测失败 - 没有满足距离阈值的聚集组 (距离阈值:{self.distance_threshold}px, 最小数量:{self.min_gathering_count})"
            )

        return gathering_indices

    def calculate_distance(self, box1: Dict, box2: Dict) -> float:
        """计算两个边界框之间的距离"""
        if self.use_center_distance:
            # 使用中心点距离
            center1 = self.box_center(box1)
            center2 = self.box_center(box2)
            return math.sqrt(
                (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
            )
        else:
            # 使用边界框最小距离
            # 计算两个矩形的最近距离
            x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
            x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']

            # 矩形1的范围
            left1, right1 = x1 - w1/2, x1 + w1/2
            top1, bottom1 = y1 - h1/2, y1 + h1/2

            # 矩形2的范围
            left2, right2 = x2 - w2/2, x2 + w2/2
            top2, bottom2 = y2 - h2/2, y2 + h2/2

            # 计算x和y方向的距离
            dx = max(left1 - right2, left2 - right1, 0)
            dy = max(top1 - bottom2, top2 - bottom1, 0)

            return math.sqrt(dx**2 + dy**2)


class NoGatheringStrategy(MotorcycleGatheringStrategy):
    """无聚集检测策略 - 所有检测到的摩托车都认为是聚集的"""

    def detect_gathering(self, motorcycle_boxes: List[Dict]) -> List[int]:
        """不进行检测，返回所有摩托车索引"""
        return list(range(len(motorcycle_boxes)))


class MotorcycleGatheringManager:
    """摩托车聚集检测管理器 - 负责完整的聚集检测和追踪逻辑"""

    def __init__(self, strategy: MotorcycleGatheringStrategy, config: Dict[str, Any]):
        self.strategy = strategy
        self.config = config

        # 目标追踪历史记录 {track_id: [{'timestamp': float, 'center': (x, y), 'box': dict}]}
        self.tracking_history = defaultdict(list)

        # 上报时间记录 {track_id: last_report_time}
        self.report_time_map = {}

        # 最后检测时间记录 {track_id: last_detection_time}
        self.last_detection_time_map = {}

        # 速度计算相关
        self.speed_window_size = config.get('speed_window_size', 2)  # 计算速度使用的帧数窗口

        # ========== 追踪模式相关属性 ==========
        self.tracking_mode_duration = config.get('tracking_mode_duration', 5.0)  # 追踪模式超时时间（秒）
        self.max_tracking_inferences = config.get('max_tracking_inferences', 3)  # ⭐ 最大推理次数（可配置）
        self.tracking_state = TrackingState.IDLE  # 当前追踪状态
        self.tracking_target_id: Optional[str] = None  # 当前追踪的target_id
        self.tracking_start_time: Optional[float] = None  # 追踪开始时间
        self.tracking_last_known_info: Optional[Dict] = None  # 最后已知的追踪目标信息（用于丢失时上报）
        self.tracking_lost_start_time: Optional[float] = None  # 目标丢失开始时间

        # ========== 轨迹跟踪相关属性 ==========
        self.tracking_trajectory_points: List[Dict] = []  # 轨迹点记录列表
        self.tracking_inference_count = 0  # 推理计数器

        # ========== 上报冷却期相关属性 ==========
        self.report_cooldown = config.get('report_cooldown', 60.0)  # ⭐ 上报冷却期（秒）
        self.last_report_timestamp: Optional[float] = None  # 最后上报时间戳

    def detect_gathering_motorcycles(self, motorcycle_boxes: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """
        检测聚集的摩托车

        Args:
            motorcycle_boxes: 摩托车目标框列表

        Returns:
            Tuple[List[Dict], List[int]]: (聚集的目标框列表, 聚集的索引列表)
        """
        if not motorcycle_boxes:
            return [], []

        gathering_indices = self.strategy.detect_gathering(motorcycle_boxes)
        gathering_boxes = [motorcycle_boxes[i] for i in gathering_indices]

        return gathering_boxes, gathering_indices

    def update_tracking_history(self, boxes: List[Dict], current_time: float = None):
        """
        更新目标追踪历史

        Args:
            boxes: 目标框列表
            current_time: 当前时间戳
        """
        if current_time is None:
            current_time = datetime.now().timestamp()

        for box in boxes:
            track_id = box.get('track_id')
            if track_id and track_id != 'unknown':
                center = self.box_center(box)
                self.tracking_history[track_id].append({
                    'timestamp': current_time,
                    'center': center,
                    'box': box
                })

                # 更新最后检测时间
                self.last_detection_time_map[track_id] = current_time

                # 限制历史记录大小
                if len(self.tracking_history[track_id]) > self.speed_window_size + 5:
                    self.tracking_history[track_id] = self.tracking_history[track_id][-self.speed_window_size-5:]

    @staticmethod
    def box_center(box: Dict) -> Tuple[float, float]:
        """获取目标框中心点"""
        return (box['x'], box['y'])

    def is_in_cooldown_period(self, current_time: float) -> bool:
        """
        检查是否处于上报冷却期

        Args:
            current_time: 当前时间戳

        Returns:
            bool: True表示在冷却期内，False表示可以上报
        """
        if self.last_report_timestamp is None:
            return False  # 从未上报过，不在冷却期

        time_since_last_report = current_time - self.last_report_timestamp
        return time_since_last_report < self.report_cooldown

    def update_last_report_time(self, current_time: float):
        """
        更新最后上报时间戳

        Args:
            current_time: 当前时间戳
        """
        self.last_report_timestamp = current_time

    def calculate_speed_and_direction(self, track_id: str) -> Tuple[float, float, str]:
        """
        计算目标的速度和方向

        Args:
            track_id: 跟踪ID

        Returns:
            Tuple[float, float, str]: (速度(像素/秒), 方向角度(度), 方向描述)
        """
        if track_id not in self.tracking_history:
            return 0.0, 0.0, "unknown"

        history = self.tracking_history[track_id]
        if len(history) < 2:
            return 0.0, 0.0, "stationary"

        # 使用最近的两个点计算速度
        latest = history[-1]
        previous = history[-2]

        dt = latest['timestamp'] - previous['timestamp']
        if dt <= 0:
            return 0.0, 0.0, "unknown"

        # 计算位移
        dx = latest['center'][0] - previous['center'][0]
        dy = latest['center'][1] - previous['center'][1]
        distance = math.sqrt(dx**2 + dy**2)

        # 计算速度（像素/秒）
        speed = distance / dt

        # 计算方向角度（度），0度表示向右，90度表示向下
        angle = math.degrees(math.atan2(dy, dx))

        # 将角度转换为方向描述
        direction = self.angle_to_direction(angle)

        return speed, angle, direction

    @staticmethod
    def angle_to_direction(angle: float) -> str:
        """将角度转换为方向描述"""
        # 将角度归一化到0-360
        angle = angle % 360

        if 337.5 <= angle or angle < 22.5:
            return "east"  # 东
        elif 22.5 <= angle < 67.5:
            return "southeast"  # 东南
        elif 67.5 <= angle < 112.5:
            return "south"  # 南
        elif 112.5 <= angle < 157.5:
            return "southwest"  # 西南
        elif 157.5 <= angle < 202.5:
            return "west"  # 西
        elif 202.5 <= angle < 247.5:
            return "northwest"  # 西北
        elif 247.5 <= angle < 292.5:
            return "north"  # 北
        else:
            return "northeast"  # 东北

    # ========== 新增：追踪模式相关方法 ==========

    def enter_tracking_mode(self, motorcycle_boxes: List[Dict], current_time: float) -> bool:
        """
        进入追踪模式

        选择置信度最高的track_id作为追踪目标，并初始化轨迹记录

        Args:
            motorcycle_boxes: 聚集的摩托车目标框列表
            current_time: 当前时间戳

        Returns:
            bool: 是否成功进入追踪模式
        """
        if not motorcycle_boxes:
            return False

        # 选择置信度最高的track_id
        best_box = max(motorcycle_boxes, key=lambda b: b.get('score', 0))
        track_id = best_box.get('track_id')

        if not track_id or track_id == 'unknown':
            return False

        # 进入追踪模式
        self.tracking_state = TrackingState.TRACKING
        self.tracking_target_id = track_id
        self.tracking_start_time = current_time

        # ✅ 初始化轨迹记录
        self.tracking_trajectory_points = []  # 清空之前的轨迹记录
        self.tracking_inference_count = 0  # 重置推理计数器

        # 记录当前追踪目标信息
        speed, angle, direction = self.calculate_speed_and_direction(track_id)
        self.tracking_last_known_info = {
            'track_id': track_id,
            'box': best_box,
            'speed': speed,
            'angle': angle,
            'direction': direction,
            'timestamp': current_time
        }

        from .logger import log_task
        log_task(f"[模型8轨迹检测] 进入轨迹追踪模式 - track_id:{track_id}, 置信度:{best_box.get('score', 0):.2f}, 将进行{self.max_tracking_inferences}次推理")

        return True

    def update_tracking_mode(self, motorcycle_boxes: List[Dict], current_time: float) -> Dict[str, Any]:
        """
        更新轨迹追踪模式状态

        处理轨迹追踪的所有逻辑：
        1. 检查追踪目标是否在当前帧中
        2. 如果找到目标，记录轨迹点
        3. 在第N次推理时上报
        4. 上报后重置轨迹追踪模式

        Args:
            motorcycle_boxes: 当前帧的所有摩托车目标框
            current_time: 当前时间戳

        Returns:
            Dict: 追踪报告
            {
                'state': TrackingState,
                'should_report': bool,
                'tracking_info': dict or None,
                'is_timeout': bool,
                'is_final_report': bool  # ✅ 新增：是否为最终报告（上报后应重置）
            }
        """
        from .logger import log_task, log_task_debug

        report = {
            'state': self.tracking_state,
            'should_report': False,
            'tracking_info': None,
            'is_timeout': False,
            'is_final_report': False  # ✅ 新增：是否为最终报告
        }

        if self.tracking_state == TrackingState.IDLE:
            return report

        # 计算追踪总持续时间（从追踪开始到现在）
        elapsed_time = current_time - self.tracking_start_time

        # 在当前帧中查找追踪目标
        target_box = None
        for box in motorcycle_boxes:
            if box.get('track_id') == self.tracking_target_id:
                target_box = box
                break

        if target_box is not None:
            # ✅ 找到目标，记录轨迹点
            center = self.box_center(target_box)
            trajectory_point = {
                'timestamp': current_time,
                'center': center,
                'box': target_box,
                'inference_count': self.tracking_inference_count
            }
            self.tracking_trajectory_points.append(trajectory_point)

            # 更新最后已知信息
            speed, angle, direction = self.calculate_speed_and_direction(self.tracking_target_id)
            self.tracking_last_known_info = {
                'track_id': self.tracking_target_id,
                'box': target_box,
                'speed': speed,
                'angle': angle,
                'direction': direction,
                'timestamp': current_time
            }

            # 增加推理计数
            self.tracking_inference_count += 1

            log_task_debug(f"[模型8轨迹检测] 记录轨迹点 ({self.tracking_inference_count}/{self.max_tracking_inferences}) - track_id:{self.tracking_target_id}, 位置:({center[0]:.1f}, {center[1]:.1f})")

            # ✅ 检查是否达到第3次推理
            if self.tracking_inference_count >= self.max_tracking_inferences:
                # 达到3次推理，准备上报
                report['should_report'] = True
                report['is_final_report'] = True  # 标记为最终报告

                # 构建追踪信息（使用最后已知信息）
                if self.tracking_last_known_info:
                    report['tracking_info'] = {
                        'track_id': self.tracking_last_known_info['track_id'],
                        'timestamp': current_time,
                        'box': {
                            'x': self.tracking_last_known_info['box']['x'],
                            'y': self.tracking_last_known_info['box']['y'],
                            'width': self.tracking_last_known_info['box']['width'],
                            'height': self.tracking_last_known_info['box']['height'],
                            'score': self.tracking_last_known_info['box']['score'],
                            'class': self.tracking_last_known_info['box'].get('className', 'motorcycle')
                        },
                        'speed': {
                            'pixels_per_second': round(self.tracking_last_known_info['speed'], 2),
                            'angle_degrees': round(self.tracking_last_known_info['angle'], 2),
                            'direction': self.tracking_last_known_info['direction']
                        },
                        'tracking_state': self.tracking_state.value,
                        'elapsed_time': round(elapsed_time, 2),
                        'trajectory_points': self.tracking_trajectory_points,  # ✅ 新增：轨迹点列表
                        'inference_count': self.tracking_inference_count  # ✅ 新增：推理次数
                    }

                log_task(f"[模型8轨迹检测] 完成{self.max_tracking_inferences}次推理，准备上报 - track_id:{self.tracking_target_id}, 总时长:{elapsed_time:.2f}秒")
                return report

        else:
            # ✅ 未找到目标，标记为丢失
            if self.tracking_state == TrackingState.TRACKING:
                # 从追踪状态转为丢失状态
                log_task_debug(f"[模型8轨迹检测] 目标丢失 - track_id:{self.tracking_target_id}")
                self.tracking_state = TrackingState.LOST
                self.tracking_lost_start_time = current_time  # 记录丢失开始时间

            # ✅ 检查丢失时长是否超过限制
            lost_duration = current_time - self.tracking_lost_start_time
            if lost_duration >= self.tracking_mode_duration:
                # 丢失超过5秒，标记为失败
                self.tracking_state = TrackingState.FAILED
                report['state'] = TrackingState.FAILED
                report['is_timeout'] = True
                report['is_final_report'] = True  # 失败也是最终报告
                report['tracking_info'] = {
                    'track_id': self.tracking_target_id,
                    'message': '追踪失败（目标丢失超过5秒）',
                    'elapsed_time': round(elapsed_time, 2),
                    'lost_duration': round(lost_duration, 2)
                }

                log_task(f"[模型8轨迹检测] 追踪失败 - track_id:{self.tracking_target_id}, 总时长:{elapsed_time:.2f}秒, 丢失时长:{lost_duration:.2f}秒")
                return report

            log_task_debug(f"[模型8轨迹检测] 目标丢失中 - track_id:{self.tracking_target_id}, 丢失时长:{lost_duration:.2f}秒")

        return report

    def reset_tracking_mode(self):
        """重置轨迹追踪模式（包括轨迹记录）"""
        from .logger import log_task_debug
        log_task_debug(f"[模型8轨迹检测] 重置轨迹追踪模式 - 之前的track_id:{self.tracking_target_id}, 状态:{self.tracking_state.value}")

        self.tracking_state = TrackingState.IDLE
        self.tracking_target_id = None
        self.tracking_start_time = None
        self.tracking_last_known_info = None
        self.tracking_lost_start_time = None  # ✅ 清除丢失开始时间

        # ✅ 清除轨迹记录
        self.tracking_trajectory_points = []
        self.tracking_inference_count = 0

    def is_in_tracking_mode(self) -> bool:
        """判断是否处于追踪模式"""
        return self.tracking_state != TrackingState.IDLE

    def get_tracking_target_id(self) -> Optional[str]:
        """获取当前追踪的track_id"""
        return self.tracking_target_id

    @staticmethod
    def draw_direction_arrow(image, trajectory_points: List[Dict], box: Dict, arrow_config: Dict[str, Any] = None):
        """
        在目标框上绘制行进方向箭头

        Args:
            image: numpy数组格式的图片
            trajectory_points: 轨迹点列表（至少2个点）
            box: 目标框信息
            arrow_config: 箭头配置字典
                {
                    'enabled': bool,  # 是否启用箭头绘制
                    'length': int,  # 箭头长度（像素）
                    'color': list,  # 箭头颜色 [B, G, R]
                    'thickness': int,  # 线条粗细
                    'tip_length': float  # 箭头头部比例（0-1）
                }

        Returns:
            image: 绘制了箭头的图片
        """
        import cv2
        import numpy as np

        # ✅ 默认箭头配置
        if arrow_config is None:
            arrow_config = {
                'enabled': True,
                'length': 60,
                'color': [0, 0, 255],  # 红色
                'thickness': 5,
                'tip_length': 0.3
            }

        # 检查是否启用箭头绘制
        if not arrow_config.get('enabled', True):
            return image

        if len(trajectory_points) < 2:
            return image

        # ✅ 使用第一个点和最后一个点计算总体方向
        first_point = trajectory_points[0]['center']
        last_point = trajectory_points[-1]['center']

        # 计算方向向量
        dx = last_point[0] - first_point[0]
        dy = last_point[1] - first_point[1]

        # 计算距离和角度
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 10:  # 如果移动距离太小，不绘制箭头
            return image

        # 归一化方向向量
        dx_norm = dx / distance
        dy_norm = dy / distance

        # 计算箭头起点（目标框中心）
        box_center_x = int(box['x'])
        box_center_y = int(box['y'])

        # 读取箭头配置
        arrow_length = arrow_config.get('length', 60)
        arrow_color = tuple(arrow_config.get('color', [0, 0, 255]))
        arrow_thickness = arrow_config.get('thickness', 5)
        arrow_tip_length = arrow_config.get('tip_length', 0.3)

        # 计算箭头终点
        arrow_end_x = int(box_center_x + dx_norm * arrow_length)
        arrow_end_y = int(box_center_y + dy_norm * arrow_length)

        # 绘制箭头主线
        cv2.arrowedLine(
            image,
            (box_center_x, box_center_y),
            (arrow_end_x, arrow_end_y),
            arrow_color,
            arrow_thickness,
            tipLength=arrow_tip_length
        )

        return image


class MotorcycleGatheringStrategyFactory:
    """摩托车聚集检测策略工厂"""

    _strategies = {
        'distance': DistanceGatheringStrategy,
        'none': NoGatheringStrategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> MotorcycleGatheringStrategy:
        """创建检测策略实例"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"未知的聚集检测策略: {strategy_name}. 可用策略: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)

    @classmethod
    def create_manager(cls, strategy_name: str, config: Dict[str, Any]) -> MotorcycleGatheringManager:
        """创建检测管理器实例"""
        strategy = cls.create_strategy(strategy_name, config)
        return MotorcycleGatheringManager(strategy, config)

    @classmethod
    def create_complete_gathering_system(cls, model_index: int, config, task_id: str = None) -> MotorcycleGatheringManager:
        """
        创建完整的摩托车聚集检测系统

        Args:
            model_index: 模型索引
            config: 配置对象
            task_id: 任务ID

        Returns:
            MotorcycleGatheringManager: 完整的聚集检测管理器
        """
        try:
            # 获取模型配置
            model_config = config.model_list[model_index]

            # 获取检测策略配置
            strategy_name = model_config.get('gathering_strategy', 'distance')
            gathering_config = model_config.get('gathering_config', {})

            # ✅ 添加默认配置（支持新的参数）
            complete_config = {
                'distance_threshold': gathering_config.get('distance_threshold', 150),
                'min_gathering_count': gathering_config.get('min_gathering_count', 3),
                'use_center_distance': gathering_config.get('use_center_distance', True),
                'speed_window_size': gathering_config.get('speed_window_size', 2),
                'max_tracking_inferences': gathering_config.get('max_tracking_inferences', 3),  # ⭐ 可配置推理次数
                'report_cooldown': gathering_config.get('report_cooldown', 60.0),  # ⭐ 上报冷却期
                'tracking_mode_duration': model_config.get('tracking_mode_duration', 5.0)  # 超时时间
            }

            # 创建管理器
            manager = cls.create_manager(strategy_name, complete_config)

            if task_id:
                from .logger import log_task, log_task_debug
                log_task(f"摩托车聚集检测系统初始化完成 - 任务ID:{task_id}, 策略:{strategy_name}")
                log_task_debug(f"策略配置 - 任务ID:{task_id}, 配置:{complete_config}")

            return manager

        except Exception as e:
            if task_id:
                from .logger import log_task_error
                log_task_error(f"摩托车聚集检测系统初始化失败 - 任务ID:{task_id}, 错误:{str(e)}")
            # 使用默认配置
            return cls.create_manager('distance', {
                'distance_threshold': 150,
                'min_gathering_count': 3,
                'max_tracking_inferences': 3,
                'report_cooldown': 60.0,
                'tracking_mode_duration': 5.0
            })

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取所有可用的策略名称"""
        return list(cls._strategies.keys())
