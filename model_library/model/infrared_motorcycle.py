"""夜间红外摩托车检测模型 - 用于检测飙车聚集场景（轨迹追踪模式）"""
from .base_model import BaseModel, device
from ultralytics.engine.results import Results
from datetime import datetime
from typing import List, Dict, Any, Optional
from model_library.tools.logger import log_task_error, log_task_debug
from model_library.utils.sahi_detector import SAHIPlateDetector


class InfraredMotorcycleModel(BaseModel):
    """夜间红外摩托车检测模型 - 支持SAHI切片推理"""

    def __init__(self, model_path, enable_sahi=False, sahi_config=None, enable_tracking=True,
                 model_index=None, estimated_memory=2500, device_override=None, config=0.5):
        """
        初始化夜间红外摩托车检测模型，支持SAHI切片推理和跟踪

        Args:
            model_path: 模型文件路径
            enable_sahi: 是否启用SAHI切片推理
            sahi_config: SAHI配置参数
            enable_tracking: 是否启用跟踪（单张图片建议关闭）
            model_index: 模型索引，用于GPU分配
            estimated_memory: 预估显存需求(MB)
            device_override: 强制指定设备，覆盖自动分配
            config: 通用置信度阈值（当禁用SAHI时使用）
        """
        # 调用父类构造函数，传递GPU分配参数
        super().__init__(model_path, model_index, estimated_memory, device_override)

        # SAHI和跟踪配置
        self.enable_sahi = enable_sahi
        self.enable_tracking = enable_tracking

        # 置信度阈值配置
        self.confidence_threshold = config  # 使用config作为默认置信度阈值

        # SAHI配置 - 确保是字典类型
        if sahi_config is None:
            self.sahi_config = {}
        elif isinstance(sahi_config, dict):
            self.sahi_config = sahi_config
        else:
            print(f"警告: sahi_config不是字典类型，收到: {type(sahi_config)}，使用默认配置")
            self.sahi_config = {}

        # 初始化SAHI检测器
        if self.enable_sahi:
            try:
                self.sahi_detector = SAHIPlateDetector(
                    model_path=model_path,
                    confidence_threshold=self.sahi_config.get('initial_confidence', 0.7),
                    device=self.sahi_config.get('device', None)
                )
                print("SAHI夜间红外摩托车检测器已启用")
            except ImportError as e:
                print(f"SAHI初始化失败，回退到标准YOLO: {e}")
                self.enable_sahi = False
            except Exception as e:
                print(f"SAHI初始化失败，回退到标准YOLO: {e}")
                self.enable_sahi = False
        else:
            print("使用标准YOLO夜间红外摩托车检测")

        print(f"跟踪模式: {'启用' if self.enable_tracking else '禁用（单张图片模式）'}")

        # 聚集检测管理器（稍后通过外部方法设置）
        self.gathering_manager = None

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = threshold

    def set_gathering_manager(self, gathering_manager):
        """设置聚集检测管理器"""
        self.gathering_manager = gathering_manager

    def post_process(self, results: Results, current_timestamp: float = None) -> List[Dict]:
        """
        提取检测结果并更新追踪历史

        轨迹追踪模式：只负责提取目标框和更新历史，不进行上报判断

        Args:
            results: YOLO检测结果
            current_timestamp: 当前时间戳

        Returns:
            List[Dict]: 处理后的检测结果列表
        """
        # 基础的后处理
        motorcycle_boxes = self._extract_boxes(results)

        # 如果有聚集检测管理器，更新追踪历史
        if self.gathering_manager and motorcycle_boxes:
            # 更新追踪历史（用于速度计算）
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

        # 确定使用的置信度阈值：如果启用SAHI，使用initial_confidence，否则使用confidence_threshold
        threshold = self.confidence_threshold
        if self.enable_sahi and self.sahi_config:
            threshold = self.sahi_config.get('initial_confidence', self.confidence_threshold)

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
                if confidence < threshold:
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
                f"[模型8验证] 置信度过滤 - 通过:{len(results_dict)}, 过滤:{filtered_count}, 阈值:{threshold:.3f}"
            )

        return results_dict

    def track_video(self, source, conf=0.5, stream=False, vid_stride=1, classes: list = None,
                    imgsz: tuple = (640, 640), verbose: bool = True, iou=0.3, half=True):
        """
        重写视频跟踪方法，支持SAHI+跟踪和标准YOLO跟踪

        注意：对于RTMP实时流，SAHI切片推理性能较差，因此使用标准YOLO跟踪，
        但会使用SAHI配置的initial_confidence作为置信度阈值
        """
        # ✅ 修复：使用self.device而不是全局device变量
        actual_device = self.device

        # 如果启用了SAHI，使用SAHI配置的initial_confidence作为置信度
        actual_conf = conf
        if self.enable_sahi:
            actual_conf = self.sahi_config.get('initial_confidence', conf)
            if verbose:
                print(f"SAHI模式已启用，使用initial_confidence={actual_conf}作为置信度阈值")

        # ✅ 添加调试日志
        log_task_debug(f"[模型8] 开始track_video - device:{actual_device}, conf:{actual_conf}, imgsz:{imgsz}")

        try:
            # 使用标准YOLO视频跟踪（适用于RTMP实时流）
            # ✅ 修复：使用self.device而不是全局device变量
            if classes is not None:
                results = self.model.track(source, stream=stream, conf=actual_conf, vid_stride=vid_stride, classes=classes,
                                           imgsz=imgsz, iou=iou, verbose=verbose, half=half, device=actual_device, tracker="botsort_cus.yaml")
            else:
                results = self.model.track(source, stream=stream, conf=actual_conf, vid_stride=vid_stride, verbose=verbose,
                                           iou=iou, half=half, device=actual_device, tracker="botsort_cus.yaml")

            log_task_debug(f"[模型8] track_video调用成功")
            return results
        except Exception as e:
            log_task_error(f"[模型8] track_video调用失败: {str(e)}")
            raise
