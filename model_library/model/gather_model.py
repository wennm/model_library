"""行人检测模型 - 用于检测人群聚集（支持智能上报策略）"""
from .base_model import BaseModel
from ultralytics.engine.results import Results
from typing import List, Dict, Any
from model_library.tools.logger import log_task_debug, log_task
from datetime import datetime


class GatherModel(BaseModel):
    """行人检测模型 - 检测人群聚集并报警（支持状态机模式）"""

    def __init__(self, model_path, model_index: int = None, estimated_memory: int = 900,
                 device_override: str = None, config: float = 0.5, gathering_threshold: int = 10,
                 reporting_strategy: str = "state_change", reporting_cooldown: int = 60):
        """
        初始化行人检测模型

        Args:
            model_path: 模型文件路径
            model_index: 模型索引，用于GPU分配
            estimated_memory: 预估显存需求(MB)
            device_override: 强制指定设备，覆盖自动分配
            config: 检测置信度阈值
            gathering_threshold: 人群聚集阈值（pedestrian + people总数）
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
        self.gathering_threshold = gathering_threshold
        self.reporting_strategy = reporting_strategy
        self.reporting_cooldown = reporting_cooldown

        # 聚集状态管理
        self.gathering_state = {
            "is_gathering": False,          # 当前是否聚集
            "start_time": None,             # 聚集开始时间
            "last_report_time": None,       # 上次上报时间
            "peak_count": 0,                # 峰值人数
            "total_updates": 0,             # 上报次数
            "consecutive_miss_frames": 0,   # 连续未检测到聚集的帧数
            "gathering_end_threshold": 30,  # 判定聚集结束的连续帧数（约5秒@6fps）
        }

        print(f"行人检测模型初始化完成 - 聚集阈值:{gathering_threshold}, 置信度:{config}, 上报策略:{reporting_strategy}")

    def post_process(self, results: Results, **kwargs) -> List[Dict]:
        """
        提取检测结果并更新聚集状态

        Args:
            results: YOLO检测结果
            **kwargs: 其他参数

        Returns:
            List[Dict]: 处理后的检测结果列表
        """
        # 基础的后处理
        pedestrian_boxes = self._extract_boxes(results)

        if not pedestrian_boxes:
            # 没有检测到任何目标
            return self._handle_empty_detection()

        # 统计pedestrian(0)和people(1)的数量
        pedestrian_count = 0
        people_count = 0

        for det in pedestrian_boxes:
            class_id = det.get('classed', -1)

            if class_id == 0:  # pedestrian
                pedestrian_count += 1
            elif class_id == 1:  # people
                people_count += 1

        total_count = pedestrian_count + people_count

        # 更新聚集状态
        gathering_info = self._update_gathering_state(
            pedestrian_count, people_count, total_count
        )

        # 添加统计信息到第一个检测框（用于返回给detector）
        if pedestrian_boxes:
            pedestrian_boxes[0]['gathering_info'] = gathering_info

            # 输出调试日志
            log_task_debug(
                f"[行人检测] 行人:{pedestrian_count}, 人群:{people_count}, "
                f"总计:{total_count}, 阈值:{self.gathering_threshold}, "
                f"是否聚集:{gathering_info['is_gathering']}, "
                f"状态:{gathering_info['state']}, 事件类型:{gathering_info.get('event_type', None)}"
            )

        return pedestrian_boxes

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

    def _update_gathering_state(self, pedestrian_count: int, people_count: int, total_count: int) -> Dict:
        """
        更新聚集状态并判断是否应该上报

        Args:
            pedestrian_count: 行人数量
            people_count: 人群数量
            total_count: 总数量

        Returns:
            Dict: 聚集信息字典
        """
        current_time = datetime.now().timestamp()
        is_gathering = total_count >= self.gathering_threshold

        state = self.gathering_state

        gathering_info = {
            'pedestrian_count': pedestrian_count,
            'people_count': people_count,
            'total_count': total_count,
            'threshold': self.gathering_threshold,
            'is_gathering': is_gathering,
            'state': state['is_gathering'],  # 之前的聚集状态
            'should_report': False,
            'event_type': None,
            'elapsed_time': 0,
            'peak_count': state['peak_count'],
            'total_updates': state['total_updates']
        }

        if self.reporting_strategy == "every_frame":
            # 原始模式：每帧都上报
            gathering_info['should_report'] = is_gathering
            gathering_info['event_type'] = 'gathering_detected' if is_gathering else None
            return gathering_info

        # 状态机模式
        if not state['is_gathering'] and is_gathering:
            # 状态转换：未聚集 → 聚集
            log_task(f"[状态转换] 未聚集 → 聚集，数量:{total_count}")

            state['is_gathering'] = True
            state['start_time'] = current_time
            state['last_report_time'] = current_time
            state['peak_count'] = total_count
            state['total_updates'] = 1
            state['consecutive_miss_frames'] = 0

            gathering_info['should_report'] = True
            gathering_info['event_type'] = 'gathering_start'
            gathering_info['state'] = True
            gathering_info['peak_count'] = total_count
            gathering_info['total_updates'] = 1

        elif state['is_gathering'] and is_gathering:
            # 状态：持续聚集中
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

                gathering_info['should_report'] = True
                gathering_info['event_type'] = 'gathering_update'
                gathering_info['elapsed_time'] = int(elapsed_time)
                gathering_info['peak_count'] = state['peak_count']
                gathering_info['total_updates'] = state['total_updates']

                log_task(f"[聚集更新] 已持续{int(elapsed_time)}秒，当前数量:{total_count}，峰值:{state['peak_count']}")
            else:
                log_task_debug(f"[聚集持续] 未到上报时间，已{int(elapsed_time)}秒/{self.reporting_cooldown}秒")

        elif state['is_gathering'] and not is_gathering:
            # 状态：聚集中，当前帧未检测到聚集
            state['consecutive_miss_frames'] += 1

            log_task_debug(f"[聚集检测] 连续{state['consecutive_miss_frames']}帧未检测到聚集")

            # 检查是否应该判定聚集结束
            if state['consecutive_miss_frames'] >= state['gathering_end_threshold']:
                # 判定聚集结束
                duration = int(current_time - state['start_time'])

                log_task(f"[状态转换] 聚集 → 未聚集，持续时长:{duration}秒，峰值:{state['peak_count']}")

                # 重置状态
                state['is_gathering'] = False
                state['start_time'] = None
                state['last_report_time'] = current_time
                peak_count = state['peak_count']
                total_updates = state['total_updates']

                state['peak_count'] = 0
                state['total_updates'] = 0
                state['consecutive_miss_frames'] = 0

                # 准备上报信息
                gathering_info['should_report'] = True
                gathering_info['event_type'] = 'gathering_end'
                gathering_info['state'] = False
                gathering_info['duration'] = duration
                gathering_info['peak_count'] = peak_count
                gathering_info['total_updates'] = total_updates

        # 之前未聚集，当前也未聚集 - 不做任何事
        elif not state['is_gathering'] and not is_gathering:
            state['consecutive_miss_frames'] = 0

        # 更新返回信息中的当前状态
        gathering_info['state'] = state['is_gathering']

        return gathering_info

    def _handle_empty_detection(self) -> List[Dict]:
        """处理未检测到任何目标的情况"""
        state = self.gathering_state

        if state['is_gathering']:
            # 当前处于聚集状态，但未检测到目标
            state['consecutive_miss_frames'] += 1

            log_task_debug(f"[空检测] 连续{state['consecutive_miss_frames']}帧未检测到目标")

            # 检查是否应该判定聚集结束
            if state['consecutive_miss_frames'] >= state['gathering_end_threshold']:
                current_time = datetime.now().timestamp()
                duration = int(current_time - state['start_time'])

                log_task(f"[状态转换] 聚集 → 未聚集（空检测），持续时长:{duration}秒")

                # 重置状态
                state['is_gathering'] = False
                peak_count = state['peak_count']
                total_updates = state['total_updates']

                state['start_time'] = None
                state['last_report_time'] = current_time
                state['peak_count'] = 0
                state['total_updates'] = 0
                state['consecutive_miss_frames'] = 0

                # 返回聚集结束信息（需要上报）
                gathering_info = {
                    'pedestrian_count': 0,
                    'people_count': 0,
                    'total_count': 0,
                    'threshold': self.gathering_threshold,
                    'is_gathering': False,
                    'state': False,
                    'should_report': True,
                    'event_type': 'gathering_end',
                    'duration': duration,
                    'peak_count': peak_count,
                    'total_updates': total_updates
                }

                return [{'gathering_info': gathering_info}]

        return []

    def detect_image(self, source, conf=0.5, stream=False, classes: list = None,
                     imgsz: tuple = (640, 640), verbose: bool = True, half=True):
        """检测单张图片中的行人"""
        actual_conf = conf if conf != 0.5 else self.confidence_threshold

        log_task_debug(f"[行人检测] 开始检测 - conf:{actual_conf}, classes:{classes}, imgsz:{imgsz}")

        if classes is not None:
            results = self.model.predict(source, stream=stream, conf=actual_conf, classes=classes,
                                         imgsz=imgsz, verbose=verbose, half=half, device=self.device)
        else:
            results = self.model.predict(source, stream=stream, conf=actual_conf, imgsz=imgsz,
                                         verbose=verbose, half=half, device=self.device)
        return results

    def reset_gathering_state(self):
        """重置聚集状态（用于测试或手动重置）"""
        self.gathering_state = {
            "is_gathering": False,
            "start_time": None,
            "last_report_time": None,
            "peak_count": 0,
            "total_updates": 0,
            "consecutive_miss_frames": 0,
            "gathering_end_threshold": 30,
        }
        log_task("[状态重置] 聚集状态已重置")
