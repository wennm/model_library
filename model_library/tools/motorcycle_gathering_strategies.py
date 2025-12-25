"""摩托车聚集检测策略模块 - 用于检测夜间红外飙车场景"""
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict


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
        self.report_interval = config.get('report_interval', 1.0)  # 上报间隔（秒）

        # 飙车判断相关
        self.racing_speed_threshold = config.get('racing_speed_threshold', 20.0)  # 飙车速度阈值（像素/秒）

        # 目标丢失超时时间（秒）
        self.track_timeout = config.get('track_timeout', 5.0)

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

    def detect_racing_motorcycles(self, motorcycle_boxes: List[Dict], current_time: float) -> Tuple[List[Dict], List[int]]:
        """
        检测聚集且速度达到飙车阈值的摩托车

        Args:
            motorcycle_boxes: 摩托车目标框列表
            current_time: 当前时间戳

        Returns:
            Tuple[List[Dict], List[int]]: (飙车的目标框列表, 飙车的索引列表)
        """
        from .logger import log_task_debug

        if not motorcycle_boxes:
            return [], []

        # 第一步：检测聚集
        gathering_boxes, gathering_indices = self.detect_gathering_motorcycles(motorcycle_boxes)

        if not gathering_boxes:
            return [], []

        # 第二步：计算聚集目标的速度，过滤出达到飙车阈值的目标
        racing_boxes = []
        racing_indices = []

        for idx, box in enumerate(gathering_boxes):
            track_id = box.get('track_id')
            if not track_id or track_id == 'unknown':
                log_task_debug(
                    f"[模型8验证] 飙车检测跳过 - track_id未知"
                )
                continue

            # 计算速度
            speed, angle, direction = self.calculate_speed_and_direction(track_id)

            # 判断是否达到飙车阈值
            if speed >= self.racing_speed_threshold:
                racing_boxes.append(box)
                racing_indices.append(gathering_indices[idx])
                log_task_debug(
                    f"[模型8验证] 飙车检测通过 - track_id:{track_id}, 速度:{speed:.2f}px/s >= {self.racing_speed_threshold}px/s, 方向:{direction}"
                )
            else:
                log_task_debug(
                    f"[模型8验证] 飙车检测失败 - track_id:{track_id}, 速度:{speed:.2f}px/s < {self.racing_speed_threshold}px/s, 方向:{direction}"
                )

        return racing_boxes, racing_indices

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

    def should_report(self, track_id: str, current_time: float) -> bool:
        """
        判断是否应该上报该目标

        Args:
            track_id: 跟踪ID
            current_time: 当前时间戳

        Returns:
            bool: 是否应该上报
        """
        # 检查目标是否超时（丢失超过5秒）
        if track_id in self.last_detection_time_map:
            time_since_last_detection = current_time - self.last_detection_time_map[track_id]
            if time_since_last_detection > self.track_timeout:
                # 目标超时，不再上报
                return False

        if track_id not in self.report_time_map:
            # 首次上报
            self.report_time_map[track_id] = current_time
            return True

        # 检查是否超过上报间隔
        time_since_last_report = current_time - self.report_time_map[track_id]
        if time_since_last_report >= self.report_interval:
            self.report_time_map[track_id] = current_time
            return True

        return False

    def get_tracking_info(self, track_id: str, box: Dict, current_time: float) -> Dict[str, Any]:
        """
        获取目标的追踪信息，用于上报（包含速度信息）

        Args:
            track_id: 跟踪ID
            box: 目标框
            current_time: 当前时间戳

        Returns:
            Dict: 追踪信息字典
        """
        # 计算速度和方向
        speed, angle, direction = self.calculate_speed_and_direction(track_id)

        tracking_info = {
            'track_id': track_id,
            'timestamp': current_time,
            'box': {
                'x': box['x'],
                'y': box['y'],
                'width': box['width'],
                'height': box['height'],
                'rotation': box.get('rotation', 0),
                'score': box['score'],
                'class': box.get('className', 'motorcycle')
            },
            'speed': {
                'pixels_per_second': round(speed, 2),
                'angle_degrees': round(angle, 2),
                'direction': direction
            },
            'is_racing': speed >= self.racing_speed_threshold  # 标记是否在飙车
        }

        return tracking_info

    def cleanup_old_history(self, current_time: float, max_age: float = 10.0):
        """
        清理过期的追踪历史

        Args:
            current_time: 当前时间戳
            max_age: 历史记录最大保留时间（秒）
        """
        for track_id in list(self.tracking_history.keys()):
            # 检查是否超时
            if track_id in self.last_detection_time_map:
                time_since_last_detection = current_time - self.last_detection_time_map[track_id]
                if time_since_last_detection > self.track_timeout:
                    # 目标超时，删除所有相关数据
                    del self.tracking_history[track_id]
                    if track_id in self.last_detection_time_map:
                        del self.last_detection_time_map[track_id]
                    if track_id in self.report_time_map:
                        del self.report_time_map[track_id]
                    continue

            # 过滤过期的记录
            self.tracking_history[track_id] = [
                record for record in self.tracking_history[track_id]
                if current_time - record['timestamp'] <= max_age
            ]

            # 如果该track_id没有有效记录，删除
            if not self.tracking_history[track_id]:
                del self.tracking_history[track_id]

                # 同时清理上报时间记录
                if track_id in self.report_time_map:
                    del self.report_time_map[track_id]


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

            # 添加默认配置
            complete_config = {
                'distance_threshold': gathering_config.get('distance_threshold', 150),
                'min_gathering_count': gathering_config.get('min_gathering_count', 3),
                'use_center_distance': gathering_config.get('use_center_distance', True),
                'speed_window_size': gathering_config.get('speed_window_size', 2),
                'report_interval': model_config.get('report_interval', 1.0),
                'racing_speed_threshold': gathering_config.get('racing_speed_threshold', 20.0),  # 飙车速度阈值
                'track_timeout': gathering_config.get('track_timeout', 5.0)  # 目标丢失超时时间
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
            return cls.create_manager('distance', {'distance_threshold': 150, 'min_gathering_count': 3})

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取所有可用的策略名称"""
        return list(cls._strategies.keys())
