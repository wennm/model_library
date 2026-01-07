"""事故验证策略模块 - 提供多种事故验证策略"""
import math
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from shapely.geometry import Polygon


class GeometryUtils:
    """几何计算工具类 - 统一管理几何相关的计算功能"""

    @staticmethod
    def intersection_judgment(box1, box_list, threshold=0.2):
        """
        输入一个yolo的xywhr格式边界框以及一个8点格式边界框列表，
        判断后者有哪些与前者相交，相交面积占box2面积的比例超过阈值则判断为事故车辆
        返回列表中的索引。

        Args:
            box1: 单个边界框坐标 [x, y, width, height, rotation] (xywhr格式，事故区域)
            box_list: 边界框列表，每个元素为 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (8点格式，车辆列表)
            threshold: 相交面积占box2面积的比例阈值，默认0.2

        Returns:
            list: 相交的边界框在列表中的索引（事故车辆索引）
        """
        # 将box1从xywhr转换为多边形
        x, y, w, h, angle = box1
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
        box1_vertices = [(cx * cos_a - cy * sin_a + x, cx * sin_a + cy * cos_a + y)
                        for cx, cy in corners]
        poly1 = Polygon(box1_vertices)

        intersecting_indices = []
        for i, box2 in enumerate(box_list):
            poly2 = Polygon(box2)
            intersection = poly1.intersection(poly2)

            if intersection.area > 0:
                # 计算相交面积占box2面积的比例
                overlap_ratio = intersection.area / poly2.area
                if overlap_ratio >= threshold:
                    intersecting_indices.append(i)

        return intersecting_indices

    @staticmethod
    def convert_xywhr_to_polygon(x, y, w, h, angle):
        """
        将xywhr格式的边界框转换为Shapely多边形

        Args:
            x, y: 中心点坐标
            w, h: 宽度和高度
            angle: 旋转角度（弧度）

        Returns:
            Polygon: Shapely多边形对象
        """
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
        vertices = [(cx * cos_a - cy * sin_a + x, cx * sin_a + cy * cos_a + y)
                   for cx, cy in corners]
        return Polygon(vertices)

    @staticmethod
    def calculate_overlap_ratio(poly1, poly2):
        """
        计算两个多边形的重叠比例（交集面积占较小多边形面积的比例）

        Args:
            poly1: 第一个多边形
            poly2: 第二个多边形

        Returns:
            float: 重叠比例 (0-1)
        """
        intersection = poly1.intersection(poly2)
        if intersection.area > 0:
            min_area = min(poly1.area, poly2.area)
            return intersection.area / min_area
        return 0.0

    @staticmethod
    def calculate_distance(box1, box2, use_center_distance=True):
        """
        计算两个边界框之间的距离

        Args:
            box1: 第一个边界框 {'x', 'y', 'width', 'height', 'rotation'}
            box2: 第二个边界框 {'x', 'y', 'width', 'height', 'rotation'}
            use_center_distance: 是否使用中心点距离，False则使用多边形最小距离

        Returns:
            float: 距离值
        """
        if use_center_distance:
            # 使用中心点距离
            center1 = (box1['x'], box1['y'])
            center2 = (box2['x'], box2['y'])
            distance = math.sqrt(
                (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
            )
            return distance
        else:
            # 使用多边形之间的最小距离
            poly1 = GeometryUtils.convert_xywhr_to_polygon(
                box1['x'], box1['y'], box1['width'], box1['height'], box1['rotation']
            )
            poly2 = GeometryUtils.convert_xywhr_to_polygon(
                box2['x'], box2['y'], box2['width'], box2['height'], box2['rotation']
            )
            return poly1.distance(poly2)


class AccidentVerificationStrategy(ABC):
    """事故验证策略基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """
        验证事故是否为真实事故

        Args:
            accident_boxes: 事故目标框列表
            pedestrian_boxes: 行人目标框列表

        Returns:
            List[int]: 被验证为真实事故的事故目标框索引列表
        """
        pass

    @abstractmethod
    def should_process_without_pedestrians(self) -> bool:
        """
        当没有行人时是否处理事故

        Returns:
            bool: True表示处理所有事故，False表示忽略所有事故
        """
        pass

    @staticmethod
    def box_to_polygon(box: Dict) -> Polygon:
        """将目标框转换为多边形 - 使用GeometryUtils统一处理"""
        return GeometryUtils.convert_xywhr_to_polygon(
            box['x'], box['y'], box['width'], box['height'], box['rotation']
        )

    @staticmethod
    def box_center(box: Dict) -> tuple:
        """获取目标框中心点"""
        return (box['x'], box['y'])


class OverlapStrategy(AccidentVerificationStrategy):
    """面积重叠验证策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.overlap_threshold = config.get('overlap_threshold', 0.3)

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """基于面积重叠验证事故"""
        verified_indices = []

        for i, accident_box in enumerate(accident_boxes):
            accident_poly = self.box_to_polygon(accident_box)

            # 检查与每个行人框的重叠
            overlap_count = 0
            for pedestrian_box in pedestrian_boxes:
                pedestrian_poly = self.box_to_polygon(pedestrian_box)

                # 使用GeometryUtils计算重叠比例
                overlap_ratio = GeometryUtils.calculate_overlap_ratio(accident_poly, pedestrian_poly)
                if overlap_ratio >= self.overlap_threshold:
                    overlap_count += 1

            # 需要两个及以上行人才通过验证
            if overlap_count >= 2:
                verified_indices.append(i)

        return verified_indices

    def should_process_without_pedestrians(self) -> bool:
        """重叠策略需要行人验证，无行人时忽略事故"""
        return False


class DistanceStrategy(AccidentVerificationStrategy):
    """距离验证策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.distance_threshold = config.get('distance_threshold', 150)  # 默认150像素
        self.use_center_distance = config.get('use_center_distance', True)

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """基于距离验证事故"""
        verified_indices = []

        for i, accident_box in enumerate(accident_boxes):
            # 检查与每个行人框的距离
            nearby_pedestrian_count = 0
            for pedestrian_box in pedestrian_boxes:
                # 使用GeometryUtils计算距离
                distance = GeometryUtils.calculate_distance(
                    accident_box, pedestrian_box, self.use_center_distance
                )

                if distance <= self.distance_threshold:
                    nearby_pedestrian_count += 1

            # 需要两个及以上行人才通过验证
            if nearby_pedestrian_count >= 2:
                verified_indices.append(i)

        return verified_indices

    def should_process_without_pedestrians(self) -> bool:
        """距离策略需要行人验证，无行人时忽略事故"""
        return False


class CombinedStrategy(AccidentVerificationStrategy):
    """联合验证策略 - 满足任一条件即可"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 初始化子策略
        self.overlap_strategy = OverlapStrategy(config)
        self.distance_strategy = DistanceStrategy(config)
        self.require_both = config.get('require_both', False)  # 是否需要同时满足两个条件

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """联合验证事故"""
        overlap_indices = set(self.overlap_strategy.verify(accident_boxes, pedestrian_boxes))
        distance_indices = set(self.distance_strategy.verify(accident_boxes, pedestrian_boxes))

        if self.require_both:
            # 需要同时满足两个条件
            verified_indices = list(overlap_indices & distance_indices)
        else:
            # 满足任一条件即可
            verified_indices = list(overlap_indices | distance_indices)

        return verified_indices

    def should_process_without_pedestrians(self) -> bool:
        """联合策略需要行人验证，无行人时忽略事故"""
        return False


class NoVerificationStrategy(AccidentVerificationStrategy):
    """无验证策略 - 所有检测到的事故都认为是真实的"""

    def verify(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """不进行验证，返回所有事故索引"""
        return list(range(len(accident_boxes)))

    def should_process_without_pedestrians(self) -> bool:
        """无验证策略处理所有事故，无论是否有行人"""
        return True


class AccidentVerificationManager:
    """事故验证管理器 - 负责完整的事故验证和处理逻辑"""

    def __init__(self, strategy: AccidentVerificationStrategy, config: Dict[str, Any]):
        self.strategy = strategy
        self.config = config
        self.class_confidence = config.get('class_confidence', {})

    def get_verified_accidents(self, accident_boxes: List[Dict], pedestrian_boxes: List[Dict]) -> List[int]:
        """
        获取通过验证的事故索引

        Args:
            accident_boxes: 事故目标框列表
            pedestrian_boxes: 行人目标框列表

        Returns:
            List[int]: 通过验证的事故索引列表
        """
        if not accident_boxes:
            return []

        # 如果没有行人，根据策略决定是否处理
        if not pedestrian_boxes:
            if self.strategy.should_process_without_pedestrians():
                return self.strategy.verify(accident_boxes, pedestrian_boxes)
            else:
                return []

        # 有行人时，使用策略进行验证
        return self.strategy.verify(accident_boxes, pedestrian_boxes)

    def apply_class_confidence_thresholds(self, model) -> bool:
        """
        应用分类别置信度阈值到模型（支持8个类别）

        Args:
            model: 要设置阈值的模型

        Returns:
            bool: 是否成功设置
        """
        try:
            if hasattr(model, 'set_class_thresholds') and self.class_confidence:
                # 构建阈值字典（使用class_id作为键）
                thresholds_dict = {}

                # 核心类别
                if 'accident' in self.class_confidence:
                    thresholds_dict[0] = self.class_confidence['accident']
                if 'pedestrian' in self.class_confidence:
                    thresholds_dict[1] = self.class_confidence['pedestrian']

                # 车辆相关类别
                if 'motorcycle' in self.class_confidence:
                    thresholds_dict[2] = self.class_confidence['motorcycle']
                if 'car' in self.class_confidence:
                    thresholds_dict[3] = self.class_confidence['car']

                # 事故相关类别
                if 'motorcycle_accident' in self.class_confidence:
                    thresholds_dict[4] = self.class_confidence['motorcycle_accident']
                if 'large_vehicle' in self.class_confidence:
                    thresholds_dict[5] = self.class_confidence['large_vehicle']

                # 执法相关类别
                if 'traffic_police' in self.class_confidence:
                    thresholds_dict[6] = self.class_confidence['traffic_police']
                if 'police_motorcycle' in self.class_confidence:
                    thresholds_dict[7] = self.class_confidence['police_motorcycle']

                # 使用字典形式批量设置阈值
                if thresholds_dict:
                    model.set_class_thresholds(thresholds_dict=thresholds_dict)
                    return True
            return False
        except Exception as e:
            # 记录错误但不中断程序
            import logging
            logging.error(f"应用分类阈值失败: {str(e)}")
            return False

    def plot_verified_accidents_only(self, result, verified_accident_items):
        """
        只绘制验证后的真实事故框，不绘制行人框

        Args:
            result: YOLO检测结果
            verified_accident_items: 验证后的事故检测项列表

        Returns:
            numpy.ndarray: 绘制后的图像
        """
        try:
            from ultralytics.utils.plotting import colors

            # 复制原始图像
            plot_img = result.orig_img.copy()

            # 只绘制验证后的事故框
            for accident_item in verified_accident_items:
                # 获取事故框参数
                x, y, w, h, angle = accident_item['x'], accident_item['y'], accident_item['width'], accident_item['height'], accident_item['rotation']
                track_id = accident_item.get('track_id', 'unknown')
                confidence = accident_item['score']

                # 计算旋转矩形的四个角点
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                rotated_corners = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T + np.array([x, y])

                # 转换为整数坐标
                points = rotated_corners.astype(int)

                # 选择颜色 - 使用红色表示事故 (BGR格式: 0, 0, 255)
                color = (0, 0, 255)  # 必须使用红色，因为VLM提示词中指定了"红色框选区域"

                # 绘制旋转矩形
                cv2.polylines(plot_img, [points], True, color, 2)

                # 保存调试图片
                debug = self.config.get('debug', False)
                if debug:
                    import time
                    debug_path = f"output/debug_accident_{track_id}_{int(time.time())}.jpg"
                    cv2.imwrite(debug_path, plot_img)
                    h, w = plot_img.shape[:2]
                    print(f"调试图片已保存: {debug_path}, 尺寸: {w}x{h}")

            return plot_img
        except Exception as e:
            # 如果绘制失败，返回原图
            return result.orig_img.copy()

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取当前策略信息"""
        return {
            'strategy_name': self.strategy.__class__.__name__,
            'config': self.strategy.config,
            'class_confidence': self.class_confidence,
            'should_process_without_pedestrians': self.strategy.should_process_without_pedestrians()
        }


class AccidentStrategyFactory:
    """事故验证策略工厂"""

    _strategies = {
        'overlap': OverlapStrategy,
        'distance': DistanceStrategy,
        'combined': CombinedStrategy,
        'none': NoVerificationStrategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> AccidentVerificationStrategy:
        """创建验证策略实例"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"未知的事故验证策略: {strategy_name}. 可用策略: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)

    @classmethod
    def create_manager(cls, strategy_name: str, config: Dict[str, Any]) -> AccidentVerificationManager:
        """创建验证管理器实例"""
        strategy = cls.create_strategy(strategy_name, config)
        return AccidentVerificationManager(strategy, config)

    @classmethod
    def create_complete_accident_system(cls, model_index: int, config, model, task_id: str = None) -> AccidentVerificationManager:
        """
        创建完整的事故识别系统（包括分类阈值设置）

        Args:
            model_index: 模型索引
            config: 配置对象
            model: 模型实例
            task_id: 任务ID

        Returns:
            AccidentVerificationManager: 完整的事故验证管理器
        """
        try:
            # 获取模型配置
            model_config = config.model_list[model_index]

            # 获取验证策略配置
            strategy_name = model_config.get('verification_strategy', 'overlap')
            verification_config = model_config.get('verification_config', {})

            # 获取分类阈值配置
            class_confidence = model_config.get('class_confidence', {})

            # 合并配置
            complete_config = {
                **verification_config,
                'class_confidence': class_confidence,
                'debug': config.config.get('debug', False)  # 从全局配置中获取debug选项
            }

            # 创建管理器
            manager = cls.create_manager(strategy_name, complete_config)

            # 应用分类阈值
            success = manager.apply_class_confidence_thresholds(model)

            if task_id:
                from .logger import log_task, log_task_debug, log_task_error
                strategy_info = manager.get_strategy_info()
                log_task(f"事故识别系统初始化完成 - 任务ID:{task_id}, 策略:{strategy_name}")
                log_task(f"分类阈值应用结果 - 任务ID:{task_id}, 成功:{success}")
                log_task_debug(f"策略详情 - 任务ID:{task_id}, 详情:{strategy_info}")

            return manager

        except Exception as e:
            if task_id:
                from .logger import log_task_error
                log_task_error(f"事故识别系统初始化失败 - 任务ID:{task_id}, 错误:{str(e)}")
            # 使用默认配置
            return cls.create_manager('overlap', {'overlap_threshold': 0.3})

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取所有可用的策略名称"""
        return list(cls._strategies.keys())