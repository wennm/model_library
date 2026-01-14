"""
流适配器 - 将StreamManager的帧流适配为YOLO track_video兼容的迭代器

核心功能：
- 提供与YOLO track_video相同的迭代器接口
- 从FrameBuffer获取帧
- 调用模型的单帧推理方法
- 保持追踪状态
"""

import time
import numpy as np
from typing import Optional, Iterator, Any, List
from dataclasses import dataclass

from .frame_buffer import FrameBuffer
from .logger import log_task, log_task_debug


@dataclass
class StreamResultAdapter:
    """
    YOLO结果适配器

    模拟YOLO的Results对象，提供相同的接口
    """
    orig_img: np.ndarray  # 原始图像
    orig_shape: tuple  # 原始图像形状 (height, width)
    _detections: Any  # 检测结果（YOLO Results对象）
    _frame_id: int = 0  # 帧ID
    _timestamp: float = 0.0  # 时间戳

    def __len__(self):
        """返回检测到的目标数量"""
        if self._detections is None:
            return 0
        return len(self._detections)

    def plot(self, **kwargs):
        """绘制检测结果（调用YOLO的plot方法）"""
        if self._detections is not None:
            return self._detections.plot(**kwargs)
        else:
            return self.orig_img

    @property
    def boxes(self):
        """获取检测框"""
        if self._detections is not None:
            return self._detections.boxes
        return None

    @property
    def obb(self):
        """获取OBB检测框"""
        if self._detections is not None:
            return self._detections.obb
        return None

    @property
    def names(self):
        """获取类别名称字典"""
        if self._detections is not None:
            return self._detections.names
        return None


class StreamIterator:
    """
    流迭代器 - 模拟YOLO的track_video返回的迭代器

    提供与track_video相同的迭代接口，但从FrameBuffer获取帧
    """

    def __init__(
        self,
        model: Any,
        frame_buffer: FrameBuffer,
        task_id: str,
        vid_stride: int = 2,
        classes: Optional[List[int]] = None,
        conf: float = 0.5,
        imgsz: Optional[tuple] = None,
        verbose: bool = False,
        stop_check_callback: Optional[callable] = None
    ):
        self.model = model
        self.frame_buffer = frame_buffer
        self.task_id = task_id
        self.vid_stride = vid_stride
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.verbose = verbose
        self.stop_check_callback = stop_check_callback

        # 统计信息
        self._frames_received = 0  # 接收到的帧数
        self._frame_count = 0  # 应该处理的帧序号
        self._inference_count = 0  # 实际推理的帧数
        self._start_time = time.time()

        log_task_debug(f"StreamIterator初始化 - task_id:{task_id}, vid_stride:{vid_stride}, "
                       f"conf:{conf}, imgsz:{imgsz}")

    def __iter__(self):
        """返回迭代器对象"""
        return self

    def __next__(self) -> StreamResultAdapter:
        """
        获取下一帧的推理结果

        Returns:
            StreamResultAdapter对象

        Raises:
            StopIteration: 当流结束时
        """
        # 自旋等待有效帧（最多等待5秒）
        max_wait_time = 5.0  # 最大等待时间（秒）
        wait_start = time.time()
        check_interval = 0.01  # 检查间隔（10ms）

        while True:
            # 检查停止信号
            if self.stop_check_callback and self.stop_check_callback():
                log_task(f"收到停止信号，结束迭代 - task_id:{self.task_id}")
                raise StopIteration

            # 检查是否超时
            elapsed = time.time() - wait_start
            if elapsed > max_wait_time:
                # 超时，检查流是否已结束
                if self.frame_buffer.is_empty() and self.frame_buffer._stop_event.is_set():
                    log_task(f"流已结束，停止迭代 - task_id:{self.task_id}")
                    raise StopIteration
                else:
                    # 流还在运行，只是暂时没帧，重置等待时间
                    wait_start = time.time()
                    log_task_debug(f"等待帧超时，重置等待 - task_id:{self.task_id}")

            # 从缓冲区获取帧
            buffered_frame = self.frame_buffer.get_frame()

            if buffered_frame is not None:
                # 获取到有效帧
                self._frames_received += 1

                # 检查是否应该跳过此帧（vid_stride逻辑）
                if self._frame_count % self.vid_stride != 0:
                    # 跳过此帧，继续循环
                    self._frame_count += 1
                    if self.verbose and self._frame_count % 100 == 0:
                        log_task_debug(f"跳过帧 - task_id:{self.task_id}, "
                                      f"received:{self._frames_received}, frame_count:{self._frame_count}, "
                                      f"processed:{self._inference_count}")
                    continue  # 继续等待下一帧

                # 应该处理此帧，退出等待循环
                self._frame_count += 1  # 只在这里增加一次
                break

            # 短暂休眠后重试
            time.sleep(check_interval)

        # 推理单帧
        try:
            inference_start = time.time()

            # 直接调用底层YOLO模型的track方法，并传递persist=True保持追踪状态
            # self.model.model 是BaseModel中的YOLO实例
            results = self.model.model.track(
                source=buffered_frame.frame,
                persist=True,  # 关键：保持追踪器状态
                classes=self.classes,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=self.verbose,
                stream=False  # 单帧推理，返回列表
            )

            inference_time = time.time() - inference_start
            self._inference_count += 1

            if self.verbose and self._inference_count % 30 == 0:
                log_task_debug(f"推理进度 - task_id:{self.task_id}, "
                              f"frames:{self._inference_count}, avg_time:{inference_time:.3f}s")

            # 返回适配后的结果
            return StreamResultAdapter(
                orig_img=buffered_frame.frame,
                orig_shape=buffered_frame.frame.shape[:2],
                _detections=results[0] if results else None,
                _frame_id=buffered_frame.frame_id,
                _timestamp=buffered_frame.timestamp
            )

        except Exception as e:
            log_task_error(f"推理异常 - task_id:{self.task_id}, frame_id:{buffered_frame.frame_id}, 错误:{str(e)}")
            # 返回空结果，继续下一帧
            return StreamResultAdapter(
                orig_img=buffered_frame.frame,
                orig_shape=buffered_frame.frame.shape[:2],
                _detections=None,
                _frame_id=buffered_frame.frame_id,
                _timestamp=buffered_frame.timestamp
            )

    def get_stats(self) -> dict:
        """获取统计信息"""
        elapsed = time.time() - self._start_time
        return {
            "task_id": self.task_id,
            "frames_received": self._frames_received,
            "frames_processed": self._inference_count,
            "frame_count": self._frame_count,
            "elapsed_time": elapsed,
            "fps": self._inference_count / elapsed if elapsed > 0 else 0,
            "vid_stride": self.vid_stride
        }


def create_stream_iterator(
    model: Any,
    video_path: str,
    task_id: str,
    vid_stride: int = 2,
    classes: Optional[List[int]] = None,
    conf: float = 0.5,
    imgsz: Optional[tuple] = None,
    verbose: bool = False,
    stop_check_callback: Optional[callable] = None,
    use_stream_manager: bool = True
) -> Iterator:
    """
    创建流迭代器

    Args:
        model: YOLO模型实例
        video_path: 视频流URL
        task_id: 任务ID
        vid_stride: 帧跳过间隔
        classes: 类别过滤
        conf: 置信度阈值
        imgsz: 图像尺寸
        verbose: 是否输出详细信息
        stop_check_callback: 停止检查回调函数
        use_stream_manager: 是否使用StreamManager（默认True）

    Returns:
        迭代器对象（与YOLO track_video兼容）

    Note:
        如果use_stream_manager=False，直接调用YOLO的track_video（原有逻辑）
    """
    if not use_stream_manager:
        # 使用原有逻辑
        log_task(f"使用原有track_video - task_id:{task_id}, url:{video_path}")
        return model.track_video(
            video_path,
            stream=True,
            vid_stride=vid_stride,
            classes=classes,
            imgsz=imgsz,
            verbose=verbose,
            conf=conf
        )
    else:
        # 使用StreamManager
        from .stream_manager import stream_manager
        from .frame_buffer import FrameBuffer

        log_task(f"使用StreamManager - task_id:{task_id}, url:{video_path}")

        # 创建帧缓冲区
        frame_buffer = FrameBuffer(
            task_id=task_id,
            buffer_size=30,
            vid_stride=vid_stride,
            blocking=False
        )

        # 订阅流
        success = stream_manager.subscribe_stream(
            stream_url=video_path,
            subscriber_id=task_id,
            frame_callback=frame_buffer.put_frame,
            model_name=task_id  # 使用task_id作为model_name
        )

        if not success:
            raise Exception(f"订阅流失败 - task_id:{task_id}, url:{video_path}")

        # 创建并返回迭代器
        return StreamIterator(
            model=model,
            frame_buffer=frame_buffer,
            task_id=task_id,
            vid_stride=vid_stride,
            classes=classes,
            conf=conf,
            imgsz=imgsz,
            verbose=verbose,
            stop_check_callback=stop_check_callback
        )
