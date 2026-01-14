"""
帧缓冲区 - 用于在StreamManager和Detector之间传递帧

核心功能：
- 线程安全的帧队列
- 支持帧跳过（vid_stride）
- 支持阻塞/非阻塞模式
- 自动管理内存
"""

import threading
import queue
import time
from typing import Optional, Any
from dataclasses import dataclass

from .stream_manager import StreamFrame
from .logger import log_task, log_task_debug


@dataclass
class BufferedFrame:
    """缓冲帧数据"""
    frame: Any  # OpenCV图像（numpy数组）
    timestamp: float  # 时间戳
    frame_id: int  # 全局帧序号

    def __repr__(self):
        return f"BufferedFrame(id={self.frame_id}, ts={self.timestamp:.3f})"


class FrameBuffer:
    """
    帧缓冲区

    职责：
    - 从StreamManager接收帧
    - 提供线程安全的帧访问接口
    - 支持帧跳过逻辑
    - 防止内存溢出
    """

    def __init__(
        self,
        task_id: str,
        buffer_size: int = 30,  # 缓冲区大小（帧数）
        vid_stride: int = 2,  # 帧跳过间隔
        blocking: bool = False,  # 是否阻塞等待
        timeout: float = 5.0  # 阻塞超时时间
    ):
        self.task_id = task_id
        self.buffer_size = buffer_size
        self.vid_stride = vid_stride
        self.blocking = blocking
        self.timeout = timeout

        # 帧队列
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)

        # 帧计数
        self._received_count = 0  # 接收到的帧总数
        self._consumed_count = 0  # 消费的帧总数
        self._last_frame_id = -1  # 最后一帧的ID

        # 控制
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        log_task_debug(f"FrameBuffer初始化 - task_id:{task_id}, buffer_size:{buffer_size}, "
                       f"vid_stride:{vid_stride}, blocking:{blocking}")

    def put_frame(self, stream_frame: StreamFrame):
        """
        将帧放入缓冲区

        Args:
            stream_frame: StreamManager提供的帧
        """
        if self._stop_event.is_set():
            return

        try:
            # 非阻塞方式放入队列
            self._frame_queue.put_nowait(
                BufferedFrame(
                    frame=stream_frame.frame,
                    timestamp=stream_frame.timestamp,
                    frame_id=stream_frame.frame_id
                )
            )
            self._received_count += 1
            self._last_frame_id = stream_frame.frame_id

        except queue.Full:
            # 缓冲区满，丢弃最老的帧
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(
                    BufferedFrame(
                        frame=stream_frame.frame,
                        timestamp=stream_frame.timestamp,
                        frame_id=stream_frame.frame_id
                    )
                )
                # 静默丢弃，不输出日志
            except:
                pass

    def get_frame(self) -> Optional[BufferedFrame]:
        """
        从缓冲区获取帧

        Returns:
            BufferedFrame或None（如果无帧可用）

        Note:
            vid_stride逻辑已移到StreamIterator中处理
        """
        if self._stop_event.is_set():
            return None

        # 获取帧（不做帧跳过，由StreamIterator处理）
        if self.blocking:
            try:
                buffered_frame = self._frame_queue.get(timeout=self.timeout)
                self._consumed_count += 1
                return buffered_frame
            except queue.Empty:
                log_task_debug(f"获取帧超时 - task_id:{self.task_id}, timeout:{self.timeout}")
                return None
        else:
            try:
                buffered_frame = self._frame_queue.get_nowait()
                self._consumed_count += 1
                return buffered_frame
            except queue.Empty:
                return None

    def stop(self):
        """停止缓冲区"""
        self._stop_event.set()

        # 清空队列
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        log_task_debug(f"FrameBuffer已停止 - task_id:{self.task_id}, "
                       f"received:{self._received_count}, consumed:{self._consumed_count}")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "task_id": self.task_id,
            "buffer_size": self.buffer_size,
            "vid_stride": self.vid_stride,
            "received_count": self._received_count,
            "consumed_count": self._consumed_count,
            "current_queue_size": self._frame_queue.qsize(),
            "last_frame_id": self._last_frame_id,
            "is_stopped": self._stop_event.is_set()
        }

    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self._frame_queue.empty()

    def has_frame(self) -> bool:
        """是否有可用的帧"""
        return not self._frame_queue.empty()
