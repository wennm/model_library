"""
视频流管理器 - 实现视频帧共享机制，解决RTMP连接冲突

核心功能：
- 单例模式管理所有活跃视频流
- 发布-订阅模式实现帧共享
- 自动资源管理和清理
- 线程安全设计

使用场景：
- 同一RTMP流需要支持多个模型并发推理
- 避免重复连接造成的资源浪费和连接冲突
"""

import threading
import time
import asyncio
import queue
import cv2
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .video_backend import create_video_capture, VideoBackend
from .rtmp_config import auto_rtmp_config
from .logger import log_task, log_task_error, log_task_debug


class StreamStatus(Enum):
    """流状态枚举"""
    STARTING = "starting"      # 启动中
    RUNNING = "running"        # 运行中
    STOPPING = "stopping"      # 停止中
    STOPPED = "stopped"        # 已停止
    ERROR = "error"            # 错误状态


@dataclass
class StreamFrame:
    """视频帧数据类"""
    frame: Any  # OpenCV图像帧（numpy数组）
    timestamp: float  # 时间戳
    frame_id: int  # 帧序号

    def __repr__(self):
        return f"StreamFrame(frame_id={self.frame_id}, timestamp={self.timestamp})"


@dataclass
class StreamSubscription:
    """流订阅者信息"""
    subscriber_id: str  # 订阅者唯一标识（如task_id）
    callback: Callable[[StreamFrame], None]  # 帧回调函数
    model_name: str = ""  # 模型名称（用于日志）
    created_at: float = field(default_factory=time.time)  # 订阅创建时间

    def __repr__(self):
        return f"StreamSubscription(id={self.subscriber_id}, model={self.model_name})"


class StreamReader:
    """
    单个视频流的读取器

    职责：
    - 连接并读取视频流
    - 管理该流的所有订阅者
    - 分发帧到所有订阅者
    - 自动处理连接失败和重连
    """

    def __init__(self, stream_url: str, stream_id: str):
        self.stream_url = stream_url
        self.stream_id = stream_id

        # 订阅者管理
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._subscriptions_lock = threading.RLock()

        # 流控制
        self._status = StreamStatus.STOPPED
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        # 帧计数
        self._frame_count = 0
        self._last_frame_time = 0

        # 健康监控
        self._health_check_interval = 10  # 健康检查间隔（秒）
        self._max_idle_time = 180  # 最大空闲时间（秒）

        log_task_debug(f"StreamReader初始化 - stream_id:{stream_id}, url:{stream_url}")

    def subscribe(self, subscription: StreamSubscription) -> bool:
        """
        订阅此流

        Args:
            subscription: 订阅信息

        Returns:
            是否订阅成功
        """
        with self._subscriptions_lock:
            if subscription.subscriber_id in self._subscriptions:
                log_task_debug(f"订阅者已存在 - stream_id:{self.stream_id}, subscriber:{subscription.subscriber_id}")
                return False

            self._subscriptions[subscription.subscriber_id] = subscription
            log_task(f"新增订阅者 - stream_id:{self.stream_id}, subscriber:{subscription.subscriber_id}, "
                    f"model:{subscription.model_name}, 总订阅数:{len(self._subscriptions)}")

            # 如果流未运行，启动流
            if self._status in [StreamStatus.STOPPED, StreamStatus.ERROR]:
                self._start_stream()

            return True

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        取消订阅

        Args:
            subscriber_id: 订阅者ID

        Returns:
            是否取消成功
        """
        with self._subscriptions_lock:
            if subscriber_id not in self._subscriptions:
                log_task_debug(f"订阅者不存在 - stream_id:{self.stream_id}, subscriber:{subscriber_id}")
                return False

            subscription = self._subscriptions.pop(subscriber_id)
            log_task(f"移除订阅者 - stream_id:{self.stream_id}, subscriber:{subscriber_id}, "
                    f"剩余订阅数:{len(self._subscriptions)}")

            # 如果没有订阅者了，停止流
            if len(self._subscriptions) == 0 and self._status == StreamStatus.RUNNING:
                log_task(f"无订阅者，停止流读取 - stream_id:{self.stream_id}")
                self._stop_stream()

            return True

    def get_subscription_count(self) -> int:
        """获取当前订阅者数量"""
        with self._subscriptions_lock:
            return len(self._subscriptions)

    def _start_stream(self):
        """启动流读取线程"""
        if self._status == StreamStatus.RUNNING:
            return

        self._status = StreamStatus.STARTING
        self._stop_event.clear()

        # 启动读取线程
        self._reader_thread = threading.Thread(
            target=self._read_stream_loop,
            name=f"StreamReader-{self.stream_id}",
            daemon=True
        )
        self._reader_thread.start()

        log_task(f"启动流读取线程 - stream_id:{self.stream_id}, url:{self.stream_url}")

    def _stop_stream(self):
        """停止流读取"""
        if self._status in [StreamStatus.STOPPED, StreamStatus.STOPPING]:
            return

        log_task(f"停止流读取 - stream_id:{self.stream_id}")
        self._status = StreamStatus.STOPPING
        self._stop_event.set()

        # 等待线程结束
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=5.0)
            if self._reader_thread.is_alive():
                log_task_error(f"流读取线程退出超时 - stream_id:{self.stream_id}")

        self._status = StreamStatus.STOPPED

    def _read_stream_loop(self):
        """流读取主循环"""
        cap = None

        try:
            # 自动检测RTMP配置
            config = auto_rtmp_config(self.stream_url)

            # 连接视频流
            log_task_debug(f"连接视频流 - stream_id:{self.stream_id}, backend:{config.backend.value}")
            cap = create_video_capture(self.stream_url, config.backend)

            if not cap.isOpened:
                log_task_error(f"无法连接视频流 - stream_id:{self.stream_id}, url:{self.stream_url}")
                self._status = StreamStatus.ERROR
                return

            # 获取流信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            log_task(f"视频流连接成功 - stream_id:{self.stream_id}, FPS:{fps}, 分辨率:{width}x{height}")

            self._status = StreamStatus.RUNNING
            self._last_frame_time = time.time()

            # 读取帧循环
            while not self._stop_event.is_set():
                ret, frame = cap.read()

                if not ret:
                    log_task_error(f"读取帧失败 - stream_id:{self.stream_id}")
                    break

                # 更新时间戳
                self._last_frame_time = time.time()
                self._frame_count += 1

                # 创建帧对象
                stream_frame = StreamFrame(
                    frame=frame,
                    timestamp=self._last_frame_time,
                    frame_id=self._frame_count
                )

                # 分发帧到所有订阅者
                self._dispatch_frame(stream_frame)

            # 正常退出
            log_task(f"流读取循环正常退出 - stream_id:{self.stream_id}, 总帧数:{self._frame_count}")

        except Exception as e:
            log_task_error(f"流读取异常 - stream_id:{self.stream_id}, 错误:{str(e)}")
            self._status = StreamStatus.ERROR
        finally:
            # 释放资源
            if cap:
                cap.release()
                log_task_debug(f"释放视频捕获 - stream_id:{self.stream_id}")

            self._status = StreamStatus.STOPPED

    def _dispatch_frame(self, stream_frame: StreamFrame):
        """
        分发帧到所有订阅者

        Args:
            stream_frame: 视频帧对象
        """
        with self._subscriptions_lock:
            if not self._subscriptions:
                return

            # 复制订阅者列表（避免在回调中持有锁）
            subscriptions = list(self._subscriptions.values())

        # 分发帧（不在锁内执行回调，避免死锁）
        for subscription in subscriptions:
            try:
                subscription.callback(stream_frame)
            except Exception as e:
                log_task_error(f"帧回调异常 - subscriber:{subscription.subscriber_id}, "
                              f"stream_id:{self.stream_id}, 错误:{str(e)}")

    def get_status(self) -> StreamStatus:
        """获取流状态"""
        return self._status

    def get_info(self) -> Dict[str, Any]:
        """获取流信息"""
        with self._subscriptions_lock:
            subscribers = [
                {
                    "id": sub.subscriber_id,
                    "model": sub.model_name,
                    "created_at": datetime.fromtimestamp(sub.created_at).isoformat()
                }
                for sub in self._subscriptions.values()
            ]

        return {
            "stream_id": self.stream_id,
            "stream_url": self.stream_url,
            "status": self._status.value,
            "subscriber_count": len(self._subscriptions),
            "subscribers": subscribers,
            "frame_count": self._frame_count,
            "last_frame_time": datetime.fromtimestamp(self._last_frame_time).isoformat() if self._last_frame_time > 0 else None
        }


class StreamManager:
    """
    视频流管理器（单例模式）

    职责：
    - 管理所有活跃的视频流
    - 提供订阅/取消订阅接口
    - 自动清理无订阅的流
    - 线程安全
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 流管理
        self._streams: Dict[str, StreamReader] = {}  # stream_url -> StreamReader
        self._streams_lock = threading.RLock()

        self._initialized = True
        log_task("StreamManager初始化完成（单例模式）")

    def subscribe_stream(
        self,
        stream_url: str,
        subscriber_id: str,
        frame_callback: Callable[[StreamFrame], None],
        model_name: str = ""
    ) -> bool:
        """
        订阅视频流

        Args:
            stream_url: 视频流URL
            subscriber_id: 订阅者唯一标识（如task_id）
            frame_callback: 帧回调函数
            model_name: 模型名称（用于日志）

        Returns:
            是否订阅成功
        """
        with self._streams_lock:
            # 获取或创建StreamReader
            if stream_url not in self._streams:
                stream_id = f"stream_{hash(stream_url) & 0x7FFFFFFFFFFFFFFF}"  # 正数hash
                reader = StreamReader(stream_url, stream_id)
                self._streams[stream_url] = reader
                log_task(f"创建新流读取器 - stream_id:{stream_id}, url:{stream_url}")
            else:
                reader = self._streams[stream_url]

            # 创建订阅
            subscription = StreamSubscription(
                subscriber_id=subscriber_id,
                callback=frame_callback,
                model_name=model_name
            )

            success = reader.subscribe(subscription)

            if success:
                log_task(f"订阅成功 - subscriber:{subscriber_id}, stream:{stream_url}, "
                        f"model:{model_name}, 总订阅数:{reader.get_subscription_count()}")
            else:
                log_task_error(f"订阅失败 - subscriber:{subscriber_id}, stream:{stream_url}")

            return success

    def unsubscribe_stream(self, stream_url: str, subscriber_id: str) -> bool:
        """
        取消订阅视频流

        Args:
            stream_url: 视频流URL
            subscriber_id: 订阅者ID

        Returns:
            是否取消成功
        """
        with self._streams_lock:
            if stream_url not in self._streams:
                log_task_debug(f"流不存在 - url:{stream_url}, subscriber:{subscriber_id}")
                return False

            reader = self._streams[stream_url]
            success = reader.unsubscribe(subscriber_id)

            # 如果流没有订阅者且已停止，从管理器中移除
            if reader.get_subscription_count() == 0 and reader.get_status() == StreamStatus.STOPPED:
                log_task(f"移除无订阅的流 - url:{stream_url}")
                del self._streams[stream_url]

            return success

    def get_stream_info(self, stream_url: str) -> Optional[Dict[str, Any]]:
        """
        获取流信息

        Args:
            stream_url: 视频流URL

        Returns:
            流信息字典，如果流不存在则返回None
        """
        with self._streams_lock:
            if stream_url not in self._streams:
                return None

            return self._streams[stream_url].get_info()

    def get_all_streams(self) -> List[Dict[str, Any]]:
        """获取所有流的信息"""
        with self._streams_lock:
            return [reader.get_info() for reader in self._streams.values()]

    def cleanup_stream(self, stream_url: str):
        """
        强制清理流

        Args:
            stream_url: 视频流URL
        """
        with self._streams_lock:
            if stream_url not in self._streams:
                return

            reader = self._streams[stream_url]
            reader._stop_stream()
            del self._streams[stream_url]

            log_task(f"强制清理流 - url:{stream_url}")

    def cleanup_all(self):
        """停止所有流"""
        with self._streams_lock:
            for stream_url in list(self._streams.keys()):
                self.cleanup_stream(stream_url)

        log_task("清理所有流完成")


# 全局单例实例
stream_manager = StreamManager()
