"""交通拥堵检测模型 - 用于检测交通拥堵（支持智能上报策略）"""
from .base_model import BaseModel
from ultralytics.engine.results import Results
from typing import List, Dict, Any
from model_library.tools.logger import log_task_debug, log_task
from datetime import datetime


class CongestionModel(BaseModel):
    """交通拥堵检测模型 - 检测交通拥堵并报警（支持状态机模式）"""

    def __init__(self, model_path, model_index: int = None, estimated_memory: int = 900,
                 device_override: str = None, config: float = 0.5, congestion_threshold: int = 15,
                 reporting_strategy: str = "state_change", reporting_cooldown: int = 60):
        """
        初始化交通拥堵检测模型

        Args:
            model_path: 模型文件路径
            model_index: 模型索引，用于GPU分配
            estimated_memory: 预估显存需求(MB)
            device_override: 强制指定设备，覆盖自动分配
            config: 检测置信度阈值
            congestion_threshold: 交通拥堵阈值（车辆总数）
            reporting_strategy: 上报策略
                - "state_change": 状态变化时上报（推荐，默认）
                - "cooldown": 冷却期模式（首次上报后冷却期内不再上报）
                - "every_frame": 每帧都上报（原始模式，不推荐）
            reporting_cooldown: 上报冷却时间（秒），仅在state_change模式下的持续更新使用
        """
        # 调用父类构造函数
        super().__init__(model_path, model_index, estimated_memory, device_override)

        # 保存配置参数
        self.confidence_threshold = config
        self.congestion_threshold = congestion_threshold
        self.reporting_strategy = reporting_strategy
        self.reporting_cooldown = reporting_cooldown

        # 拥堵状态管理
        self.congestion_state = {
            "is_congestion": False,          # 当前是否拥堵
            "start_time": None,              # 拥堵开始时间
            "last_report_time": None,        # 上次上报时间
            "peak_count": 0,                 # 峰值车辆数
            "total_updates": 0,              # 上报次数
            "consecutive_miss_frames": 0,    # 连续未检测到拥堵的帧数
            "congestion_end_threshold": 30,  # 判定拥堵结束的连续帧数（约5秒@6fps）
        }

        # 车辆类别映射（VisDrone数据集）
        self.vehicle_class_names = {
            3: "car",           # 汽车
            4: "van",           # 货车
            5: "truck",         # 卡车
            6: "tricycle",       # 三轮车
            7: "awning-tricycle",         # 篷车
            8: "bus"        # 公交车
        }


        print(f"交通拥堵检测模型初始化完成 - 拥堵阈值:{congestion_threshold}, 置信度:{config}, 上报策略:{reporting_strategy}")

    def post_process(self, results: Results, **kwargs) -> List[Dict]:
        """
        提取检测结果并更新拥堵状态

        Args:
            results: YOLO检测结果
            **kwargs: 其他参数

        Returns:
            List[Dict]: 处理后的检测结果列表
        """
        # 基础的后处理
        vehicle_boxes = self._extract_boxes(results)

        if not vehicle_boxes:
            # 没有检测到任何目标
            return self._handle_empty_detection()

        # 统计各类型车辆数量
        vehicle_counts = {
            "car": 0,
            "van": 0,
            "truck": 0,
            "bus": 0,
            "motor": 0,
            "bicycle": 0
        }

        for det in vehicle_boxes:
            class_id = det.get('classed', -1)
            class_name = self.vehicle_class_names.get(class_id)

            if class_name:
                vehicle_counts[class_name] += 1

        total_count = sum(vehicle_counts.values())

        # 计算整体包围盒（所有车辆的最小外接矩形）
        bounding_box = self._calculate_bounding_box(vehicle_boxes)

        # 更新拥堵状态
        congestion_info = self._update_congestion_state(vehicle_counts, total_count)

        # 添加统计信息和整体包围盒到第一个检测框（用于返回给detector）
        if vehicle_boxes:
            vehicle_boxes[0]['congestion_info'] = congestion_info
            vehicle_boxes[0]['bounding_box'] = bounding_box  # 添加整体包围盒

            # 输出调试日志
            log_task_debug(
                f"[拥堵检测] 汽车:{vehicle_counts['car']}, 货车:{vehicle_counts['van']}, "
                f"卡车:{vehicle_counts['truck']}, 巴士:{vehicle_counts['bus']}, "
                f"摩托:{vehicle_counts['motor']}, 自行:{vehicle_counts['bicycle']}, "
                f"总计:{total_count}, 阈值:{self.congestion_threshold}, "
                f"是否拥堵:{congestion_info['is_congestion']}, "
                f"状态:{congestion_info['state']}, 事件类型:{congestion_info.get('event_type', None)}"
            )

        return vehicle_boxes

    def _extract_boxes(self, results: Results) -> List[Dict]:
        """从Results中提取标准目标框信息"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue

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

        return results_dict

    def _calculate_bounding_box(self, vehicle_boxes: List[Dict]) -> Dict:
        """
        计算所有车辆的整体包围盒（最小外接矩形）

        Args:
            vehicle_boxes: 车辆检测框列表

        Returns:
            Dict: 包含整体包围盒信息的字典，格式为：
                {
                    "x": 最小x坐标,
                    "y": 最小y坐标,
                    "width": 宽度,
                    "height": 高度,
                    "className": "congestion_area"
                }
        """
        if not vehicle_boxes:
            return {}

        # 初始化边界值
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0

        for box in vehicle_boxes:
            x = box.get('x', 0)
            y = box.get('y', 0)
            width = box.get('width', 0)
            height = box.get('height', 0)

            # 更新最小值和最大值
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)

        # 计算整体包围盒
        bounding_box = {
            "x": min_x,
            "y": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
            "className": "congestion_area"
        }

        return bounding_box

    def _update_congestion_state(self, vehicle_counts: Dict[str, int], total_count: int) -> Dict:
        """
        更新拥堵状态并判断是否应该上报

        Args:
            vehicle_counts: 各类车辆数量字典
            total_count: 总数量

        Returns:
            Dict: 拥堵信息字典
        """
        current_time = datetime.now().timestamp()
        is_congestion = total_count >= self.congestion_threshold

        state = self.congestion_state

        congestion_info = {
            'vehicle_counts': vehicle_counts,
            'total_count': total_count,
            'threshold': self.congestion_threshold,
            'is_congestion': is_congestion,
            'state': state['is_congestion'],  # 之前的拥堵状态
            'should_report': False,
            'event_type': None,
            'elapsed_time': 0,
            'peak_count': state['peak_count'],
            'total_updates': state['total_updates']
        }

        if self.reporting_strategy == "every_frame":
            # 原始模式：每帧都上报
            congestion_info['should_report'] = is_congestion
            congestion_info['event_type'] = 'congestion_detected' if is_congestion else None
            return congestion_info

        # 状态机模式
        if not state['is_congestion'] and is_congestion:
            # 状态转换：未拥堵 → 拥堵
            log_task(f"[状态转换] 未拥堵 → 拥堵，数量:{total_count}")

            state['is_congestion'] = True
            state['start_time'] = current_time
            state['last_report_time'] = current_time
            state['peak_count'] = total_count
            state['total_updates'] = 1
            state['consecutive_miss_frames'] = 0

            congestion_info['should_report'] = True
            congestion_info['event_type'] = 'congestion_start'
            congestion_info['state'] = True
            congestion_info['peak_count'] = total_count
            congestion_info['total_updates'] = 1

        elif state['is_congestion'] and is_congestion:
            # 状态：持续拥堵中
            state['consecutive_miss_frames'] = 0

            # 更新峰值
            if total_count > state['peak_count']:
                state['peak_count'] = total_count

            elapsed_time = current_time - state['last_report_time']

            # 检查是否需要更新上报
            should_update = elapsed_time >= self.reporting_cooldown

            if should_update:
                state['last_report_time'] = current_time
                state['total_updates'] += 1

                congestion_info['should_report'] = True
                congestion_info['event_type'] = 'congestion_update'
                congestion_info['elapsed_time'] = int(elapsed_time)
                congestion_info['peak_count'] = state['peak_count']
                congestion_info['total_updates'] = state['total_updates']

                log_task(f"[拥堵更新] 已持续{int(elapsed_time)}秒，当前数量:{total_count}，峰值:{state['peak_count']}")
            else:
                log_task_debug(f"[拥堵持续] 未到上报时间，已{int(elapsed_time)}秒/{self.reporting_cooldown}秒")

        elif state['is_congestion'] and not is_congestion:
            # 状态：拥堵中，当前帧未检测到拥堵
            state['consecutive_miss_frames'] += 1

            log_task_debug(f"[拥堵检测] 连续{state['consecutive_miss_frames']}帧未检测到拥堵")

            # 检查是否应该判定拥堵结束
            if state['consecutive_miss_frames'] >= state['congestion_end_threshold']:
                # 判定拥堵结束
                duration = int(current_time - state['start_time'])

                log_task(f"[状态转换] 拥堵 → 未拥堵，持续时长:{duration}秒，峰值:{state['peak_count']}")

                # 重置状态
                state['is_congestion'] = False
                state['start_time'] = None
                state['last_report_time'] = current_time
                peak_count = state['peak_count']
                total_updates = state['total_updates']

                state['peak_count'] = 0
                state['total_updates'] = 0
                state['consecutive_miss_frames'] = 0

                # 准备上报信息
                congestion_info['should_report'] = True
                congestion_info['event_type'] = 'congestion_end'
                congestion_info['state'] = False
                congestion_info['duration'] = duration
                congestion_info['peak_count'] = peak_count
                congestion_info['total_updates'] = total_updates

        # 之前未拥堵，当前也未拥堵 - 不做任何事
        elif not state['is_congestion'] and not is_congestion:
            state['consecutive_miss_frames'] = 0

        # 更新返回信息中的当前状态
        congestion_info['state'] = state['is_congestion']

        return congestion_info

    def _handle_empty_detection(self) -> List[Dict]:
        """处理未检测到任何目标的情况"""
        state = self.congestion_state

        if state['is_congestion']:
            # 当前处于拥堵状态，但未检测到目标
            state['consecutive_miss_frames'] += 1

            log_task_debug(f"[空检测] 连续{state['consecutive_miss_frames']}帧未检测到目标")

            # 检查是否应该判定拥堵结束
            if state['consecutive_miss_frames'] >= state['congestion_end_threshold']:
                current_time = datetime.now().timestamp()
                duration = int(current_time - state['start_time'])

                log_task(f"[状态转换] 拥堵 → 未拥堵（空检测），持续时长:{duration}秒")

                # 重置状态
                state['is_congestion'] = False
                peak_count = state['peak_count']
                total_updates = state['total_updates']

                state['start_time'] = None
                state['last_report_time'] = current_time
                state['peak_count'] = 0
                state['total_updates'] = 0
                state['consecutive_miss_frames'] = 0

                # 返回拥堵结束信息（需要上报）
                congestion_info = {
                    'vehicle_counts': {
                        "car": 0,
                        "van": 0,
                        "truck": 0,
                        "bus": 0,
                        "motor": 0,
                        "bicycle": 0
                    },
                    'total_count': 0,
                    'threshold': self.congestion_threshold,
                    'is_congestion': False,
                    'state': False,
                    'should_report': True,
                    'event_type': 'congestion_end',
                    'duration': duration,
                    'peak_count': peak_count,
                    'total_updates': total_updates
                }

                return [{'congestion_info': congestion_info}]

        return []

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None,
                     imgsz: tuple = (640, 640), verbose: bool = True, half=True):
        """检测单张图片中的车辆"""
        actual_conf = conf if conf != 0.5 else self.confidence_threshold

        log_task_debug(f"[拥堵检测] 开始检测 - conf:{actual_conf}, classes:{classes}, imgsz:{imgsz}")

        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=actual_conf, classes=classes,
                                         imgsz=imgsz, verbose=verbose, half=half, device=self.device)
        else:
            results = self.model.predict(source, stream=stream, conf=actual_conf, imgsz=imgsz,
                                         verbose=verbose, half=half, device=self.device)
        return results

    def reset_congestion_state(self):
        """重置拥堵状态（用于测试或手动重置）"""
        self.congestion_state = {
            "is_congestion": False,
            "start_time": None,
            "last_report_time": None,
            "peak_count": 0,
            "total_updates": 0,
            "consecutive_miss_frames": 0,
            "congestion_end_threshold": 30,
        }
        log_task("[状态重置] 拥堵状态已重置")
