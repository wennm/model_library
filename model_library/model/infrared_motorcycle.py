"""夜间红外摩托车检测模型 - 用于检测飙车聚集场景"""
from .base_model import BaseModel
from ultralytics.engine.results import Results
from datetime import datetime
from typing import List, Dict, Any


class InfraredMotorcycleModel(BaseModel):
    """夜间红外摩托车检测模型"""

    def __init__(self, model_path, model_index: int = None, estimated_memory: int = 1000, device_override: str = None):
        super().__init__(model_path, model_index, estimated_memory, device_override)

        # 配置参数
        self.confidence_threshold = 0.5  # 默认置信度阈值
        self.enable_sahi = False  # 是否启用SAHI
        self.sahi_config = None

        # 聚集检测管理器（稍后通过外部方法设置）
        self.gathering_manager = None

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = threshold

    def enable_sahi_inference(self, sahi_config: dict = None):
        """启用SAHI切片推理"""
        self.enable_sahi = True
        self.sahi_config = sahi_config

    def set_gathering_manager(self, gathering_manager):
        """设置聚集检测管理器"""
        self.gathering_manager = gathering_manager

    def post_process(self, results: Results, current_timestamp: float = None) -> List[Dict]:
        """
        提取检测结果并应用聚集检测

        Args:
            results: YOLO检测结果
            current_timestamp: 当前时间戳

        Returns:
            List[Dict]: 处理后的检测结果列表
        """
        # 基础的后处理
        motorcycle_boxes = self._extract_boxes(results)

        # 如果有聚集检测管理器，进行聚集检测
        if self.gathering_manager and motorcycle_boxes:
            # 更新追踪历史
            if current_timestamp is None:
                current_timestamp = datetime.now().timestamp()
            self.gathering_manager.update_tracking_history(motorcycle_boxes, current_timestamp)

        return motorcycle_boxes

    def _extract_boxes(self, results: Results) -> List[Dict]:
        """
        从Results中提取标准目标框信息（不使用OBB）

        Args:
            results: YOLO检测结果

        Returns:
            List[Dict]: 目标框列表
        """
        from ..tools.logger import log_task_debug

        results_dict = []
        filtered_count = 0  # 统计被过滤的目标数
        for result in results:
            if len(result) == 0:
                continue

            # 只使用标准边界框（不使用OBB）
            boxes = result.boxes
            if boxes is None:
                continue

            names = result.names
            xywh = boxes.xywh.tolist()
            cls = boxes.cls.tolist()
            conf = boxes.conf.tolist()

            # 获取track_id
            try:
                track_id = boxes.id.tolist() if hasattr(boxes, 'id') and boxes.id is not None else "unknown"
            except:
                track_id = "unknown"

            # 提取每个目标框
            for i, box in enumerate(xywh):
                class_id = int(cls[i])
                confidence = conf[i]

                # 应用置信度阈值过滤
                if confidence < self.confidence_threshold:
                    filtered_count += 1
                    continue

                box_params = {
                    "x": box[0],
                    "y": box[1],
                    "width": box[2],
                    "height": box[3],
                    "score": confidence,
                    "track_id": track_id[i] if isinstance(track_id, list) else track_id,
                    "classed": class_id,
                    "className": names[class_id],
                    "text": ""
                }
                results_dict.append(box_params)

        # 输出置信度过滤汇总日志
        if filtered_count > 0:
            log_task_debug(
                f"[模型8验证] 置信度过滤 - 通过:{len(results_dict)}, 过滤:{filtered_count}, 阈值:{self.confidence_threshold:.3f}"
            )

        return results_dict

    def detect_gathering_and_get_reports(self, results: Results, current_timestamp: float = None) -> List[Dict[str, Any]]:
        """
        检测聚集并生成需要上报的追踪信息

        Args:
            results: YOLO检测结果
            current_timestamp: 当前时间戳

        Returns:
            List[Dict]: 需要上报的追踪信息列表
        """
        if current_timestamp is None:
            current_timestamp = datetime.now().timestamp()

        # 提取所有摩托车框
        all_motorcycles = self.post_process(results, current_timestamp)

        if not all_motorcycles or not self.gathering_manager:
            return []

        # 检测聚集的摩托车
        gathering_boxes, gathering_indices = self.gathering_manager.detect_gathering_motorcycles(all_motorcycles)

        if not gathering_boxes:
            return []

        # 为每个聚集的目标生成上报信息
        reports = []
        for box in gathering_boxes:
            track_id = box.get('track_id')
            if track_id and track_id != 'unknown':
                # 判断是否是首次上报
                is_first_report = track_id not in self.gathering_manager.report_time_map

                # 判断是否应该上报
                if self.gathering_manager.should_report(track_id, current_timestamp):
                    tracking_info = self.gathering_manager.get_tracking_info(
                        track_id, box, current_timestamp, is_first_report
                    )
                    reports.append(tracking_info)

        # 清理过期历史
        self.gathering_manager.cleanup_old_history(current_timestamp)

        return reports

    def detect_gathering_and_get_frame_report(self, results: Results, current_timestamp: float = None) -> Dict[str, Any]:
        """
        检测飙车并生成帧级别的上报信息（同一帧的多个目标在一条消息中）

        检测逻辑：
        1. 检测聚集的摩托车（3个以上目标）
        2. 计算聚集目标的速度
        3. 过滤出速度达到飙车阈值的目标
        4. 判断是否需要上报

        Args:
            results: YOLO检测结果
            current_timestamp: 当前时间戳

        Returns:
            Dict: 帧级别的飙车报告，包含所有需要上报的目标信息
            {
                'timestamp': float,
                'gathering_count': int,  # 聚集数量
                'racing_count': int,  # 飙车数量（速度达到阈值）
                'tracking_infos': List[Dict],  # 该帧所有飙车目标的追踪信息
                'has_new_reports': bool  # 是否有新的上报目标
            }
        """
        if current_timestamp is None:
            current_timestamp = datetime.now().timestamp()

        # 提取所有摩托车框
        all_motorcycles = self.post_process(results, current_timestamp)

        if not all_motorcycles or not self.gathering_manager:
            return {
                'timestamp': current_timestamp,
                'gathering_count': 0,
                'racing_count': 0,
                'tracking_infos': [],
                'has_new_reports': False
            }

        # 检测聚集的摩托车（先聚集）
        gathering_boxes, _ = self.gathering_manager.detect_gathering_motorcycles(all_motorcycles)

        if not gathering_boxes:
            return {
                'timestamp': current_timestamp,
                'gathering_count': 0,
                'racing_count': 0,
                'tracking_infos': [],
                'has_new_reports': False
            }

        # 检测飙车的摩托车（聚集 + 速度阈值）
        racing_boxes, _ = self.gathering_manager.detect_racing_motorcycles(all_motorcycles, current_timestamp)

        frame_report = {
            'timestamp': current_timestamp,
            'gathering_count': len(gathering_boxes),
            'racing_count': len(racing_boxes),
            'tracking_infos': [],
            'has_new_reports': False
        }

        if not racing_boxes:
            # 有聚集但没有飙车，不上报
            return frame_report

        # 为每个飙车的目标生成追踪信息
        for box in racing_boxes:
            track_id = box.get('track_id')
            if track_id and track_id != 'unknown':
                # 判断是否应该上报
                should_report = self.gathering_manager.should_report(track_id, current_timestamp)

                if should_report:
                    frame_report['has_new_reports'] = True

                # 获取追踪信息（包含速度）
                tracking_info = self.gathering_manager.get_tracking_info(
                    track_id, box, current_timestamp
                )

                # 标记这个目标是否需要新上报
                tracking_info['should_report'] = should_report
                frame_report['tracking_infos'].append(tracking_info)

        # 清理过期历史
        self.gathering_manager.cleanup_old_history(current_timestamp)

        return frame_report

    def detect_with_sahi(self, source, conf=0.5, **kwargs):
        """
        使用SAHI进行切片推理（如果启用）

        Args:
            source: 图像源
            conf: 置信度阈值
            **kwargs: 其他参数

        Returns:
            检测结果
        """
        if self.enable_sahi and self.sahi_config:
            # 导入SAHI
            try:
                from sahi import AutoDetectionModel
                from sahi.predict import get_prediction

                # SAHI配置
                sahi_conf = self.sahi_config
                initial_confidence = sahi_conf.get('initial_confidence', conf)

                # 使用SAHI进行推理
                # 注意：这里需要根据实际的SAHI版本进行适配
                results = self.model.predict(
                    source,
                    conf=initial_confidence,
                    **kwargs
                )
                return results
            except ImportError:
                print("SAHI未安装，回退到常规推理")
                self.enable_sahi = False
                return self.detect_image(source, conf=conf, **kwargs)
            except Exception as e:
                print(f"SAHI推理失败: {str(e)}，回退到常规推理")
                self.enable_sahi = False
                return self.detect_image(source, conf=conf, **kwargs)
        else:
            # 常规推理
            return self.detect_image(source, conf=conf, **kwargs)
