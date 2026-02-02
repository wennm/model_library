"""检测器，视频流后台推理任务。适用于对接开发部的工作流程"""
import time
import threading
import math
import cv2
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict

from ..model.model_manager import model_manager
from ..tools.utils import Config
from ..client.mqtt_client import MQTTClient
from ..client.minio_client import MinioClient
from .reasoner import reasoner_single
from .logger import log_task, log_task_error, log_task_debug
from .accident_strategies import AccidentStrategyFactory
from .vlm_verifier import VLMVerifier
from .video_backend import create_video_capture, VideoBackend, VideoBackendConfig
from .rtmp_config import auto_rtmp_config
from .mqtt_formatter import MQTTMessageFormatter
from .accident_strategies import GeometryUtils
from .resource_cleanup import resource_cleanup_manager
from .motorcycle_gathering_strategies import MotorcycleGatheringStrategyFactory
from .stream_adapter import create_stream_iterator
from .stream_manager import stream_manager
from .drawing_utils import plot_gathering_bounding_box, plot_congestion_bounding_box

BeiJingTime = ZoneInfo("Asia/Shanghai")

# 全局配置：是否使用StreamManager进行视频帧共享
# True: 多个模型共享同一RTMP连接（推荐）
# False: 每个模型独立连接RTMP（原有方式）
USE_STREAM_MANAGER = True


class Detector:
    _instance_count = 0  # 类级别的实例计数器

    def __init__(self, model_index: int, video_path: str, pixel_position: list = None, task_id: str = None):
        Detector._instance_count += 1
        self.model_index = model_index
        self.video_path = video_path
        self.task_id = task_id
        self.pixel_position = pixel_position

        log_task(
            f"Detector创建 - 任务ID:{task_id}, 当前总数:{Detector._instance_count}, 模型索引:{model_index}, 视频:{video_path}")

        self.config = Config()
        self.mqtt_client = MQTTClient()
        self.mqtt_client.connect()
        self.minio_client = MinioClient()
        self.model_name = self.config.model_list[self.model_index]['model_name']
        self.model_conf = self.config.model_list[self.model_index].get("config", 0.5)
        self.classes = self.config.model_list[self.model_index].get('classes', [0])
        self.time_step = self.config.model_list[self.model_index].get('time_step', 60)  # 推送间隔
        self.topic = self.get_topic()

        # 初始化事故验证管理器（仅用于模型3）
        self.verification_manager = None
        self.model_index_3 = self.model_index == 3

        # 初始化 VLM 验证器 (仅用于模型3)
        self.vlm_verifier = None
        if self.model_index_3:
            model_config = self.config.model_list[self.model_index]
            vlm_config = model_config.get('vlm_verification', {})
            global_ms_conf = self.config.config.get('modelscope', {})
            self.vlm_verifier = VLMVerifier(vlm_config, global_ms_conf)
            # 读取MQTT冷却时间配置
            self.mqtt_cooldown = model_config.get('mqtt_cooldown', 60)
            self.last_mqtt_report_time = 0  # 上次MQTT上报时间（时间戳）

        # 初始化摩托车聚集检测管理器（仅用于模型8）
        self.gathering_manager = None
        self.model_index_8 = self.model_index == 8

        # 初始化行人检测模型标志（仅用于模型9）
        self.model_index_9 = self.model_index == 9

        # 初始化交通拥堵检测模型标志（仅用于模型10）
        self.model_index_10 = self.model_index == 10

        log_task_debug(f"获取模型实例 - 任务ID:{task_id}, 模型:{self.model_name}")
        self.model = model_manager.get_model(self.model_index, task_id)

        # ⭐ 模型3特殊初始化：强制模型5使用与模型3相同的GPU（测试环境单GPU共享）
        if self.model_index_3:
            model3_device = getattr(self.model, 'device', None)
            if model3_device and model3_device.startswith('cuda'):
                # 临时修改模型5配置，添加device_override（使模型5共享GPU）
                self.config.model_list[5]['device_override'] = model3_device

        # 在模型加载后初始化完整的事故识别系统（仅用于模型3）
        if self.model_index_3:
            self.verification_manager = AccidentStrategyFactory.create_complete_accident_system(
                self.model_index, self.config, self.model, task_id
            )

        # 在模型加载后初始化完整的摩托车聚集检测系统（仅用于模型8）
        if self.model_index_8:
            self.gathering_manager = MotorcycleGatheringStrategyFactory.create_complete_gathering_system(
                self.model_index, self.config, task_id
            )
            # 将聚集检测管理器设置到模型中
            if hasattr(self.model, 'set_gathering_manager'):
                self.model.set_gathering_manager(self.gathering_manager)

            # ✅ 读取箭头配置
            model_config = self.config.model_list[self.model_index]
            self.arrow_config = model_config.get('arrow_config', {
                'enabled': True,
                'min_distance': 3.0,
                'length': 60,
                'color': [0, 0, 255],
                'thickness': 5,
                'tip_length': 0.3
            })
            log_task_debug(f"模型8箭头配置 - 任务ID:{task_id}, 配置:{self.arrow_config}")

        log_task(f"检测器初始化完成 - 任务ID:{task_id}, 模型:{self.model_name}, MQTT主题:{self.topic}")

        # 注册到资源清理管理器
        resource_cleanup_manager.register_task(task_id, self, self.model_index)

        # 添加停止控制机制
        self._should_stop = False  # 检查任务执行状态。包括自动轮询以及手动停止
        self.stream_timeout = 180  # 3分钟超时
        self._stop_event = asyncio.Event()

        # 监控线程管理
        self._monitor_thread = None  # 保存监控线程引用
        self._monitor_shutdown_event = threading.Event()  # 监控线程关闭信号
        self._monitor_started = False  # 新增：跟踪监控线程是否已启动

        # 推理日志控制（每1秒记录一次）
        self._last_log_time = 0  # 上次记录推理日志的时间
        self._log_interval = 2.0  # 推理日志记录间隔（秒）

    
    def __del__(self):
        """析构函数，用于跟踪对象何时被真正销毁"""
        try:
            # 检查Python是否正在关闭
            import sys
            if sys.meta_path is None:
                # Python正在关闭，跳过复杂的清理操作
                Detector._instance_count = max(0, Detector._instance_count - 1)
                return

            # 安全检查：确保对象已正确初始化
            if hasattr(self, '_monitor_thread') and self._monitor_thread is not None:
                # 确保监控线程被正确停止
                self._stop_monitor_thread()

            # 使用资源清理管理器清理
            if hasattr(self, 'task_id'):
                resource_cleanup_manager.cleanup_task(self.task_id, force=True)

            Detector._instance_count = max(0, Detector._instance_count - 1)
            log_task(f"Detector销毁 - 任务ID:{getattr(self, 'task_id', 'unknown')}, 剩余:{Detector._instance_count}")
        except Exception as e:
            # 在Python关闭时，某些操作可能会失败，这是正常的
            if "sys.meta_path is None" in str(e) or "Python is likely shutting down" in str(e):
                # Python正在关闭，减少计数器但不记录日志
                Detector._instance_count = max(0, Detector._instance_count - 1)
            else:
                # 其他异常仍然记录
                print(f"析构函数异常: {str(e)}")

    def enhanced_cleanup(self) -> bool:
        """
        增强的资源清理方法

        Returns:
            清理是否成功
        """
        try:
            # 使用资源清理管理器进行完整清理
            return resource_cleanup_manager.cleanup_task(
                getattr(self, 'task_id', 'unknown'),
                force=True
            )
        except Exception as e:
            log_task_error(f"增强清理失败 - 任务ID:{getattr(self, 'task_id', 'unknown')}, 错误:{str(e)}")
            return False

    @classmethod
    def get_instance_count(cls):
        """获取当前存活的实例数量"""
        return cls._instance_count

  
    async def _save_and_publish_accident(self, result, accident_item, object_name, ori_img_shape, timestamp_str, message=None, accident_type=None, verification_info=None):
        """
        保存事故图像并发布MQTT消息（支持事故类型和验证详情）

        Args:
            result: YOLO检测结果
            accident_item: 事故检测项
            object_name: 存储对象名
            ori_img_shape: 原始图像尺寸
            timestamp_str: 时间戳字符串
            message: 可选的事故类型描述消息
            accident_type: 事故类型 (normal/motorcycle/large_vehicle)
            verification_info: 验证详情字典
        """
        try:
            # 检查MQTT冷却时间（仅用于模型3）
            if self.model_index_3:
                current_time = time.time()
                time_since_last_report = current_time - self.last_mqtt_report_time

                if time_since_last_report < self.mqtt_cooldown:
                    remaining_time = self.mqtt_cooldown - time_since_last_report
                    log_task_debug(f"MQTT冷却中，跳过上报和保存 - 任务ID:{self.task_id}, "
                                 f"剩余冷却时间:{remaining_time:.1f}秒")
                    return  # 直接返回，不保存图片也不上报MQTT

            # 使用策略工厂的绘制方法，只绘制验证后的真实事故框，不绘制行人框
            infer_image = self.verification_manager.plot_verified_accidents_only(result, [accident_item])
            _, _ = self.minio_client.upload_image_array(
                image_array=infer_image,
                object_name=object_name,
                image_format='jpg',
                quality=85
            )

            # 使用MQTT格式化器构建消息（传递事故类型和验证详情）
            mqtt_message = MQTTMessageFormatter.format_accident_message(
                object_name=object_name,
                accident_item=accident_item,
                ori_img_shape=ori_img_shape,
                task_id=self.task_id,
                timestamp_str=timestamp_str,
                message=message,
                accident_type=accident_type,
                verification_info=verification_info
            )
            # ⭐ objNum由MQTT格式化器自动计算：1 + accident_car_count

            # 发送MQTT消息
            log_task_debug(f"发送事故MQTT消息 - 任务ID:{self.task_id}, 主题:{self.topic}")
            mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)
            log_task_debug(f"MQTT发送结果 - 任务ID:{self.task_id}, 成功:{mqtt_success}")

            # 更新最后上报时间（仅用于模型3）
            if self.model_index_3 and mqtt_success:
                self.last_mqtt_report_time = time.time()
                log_task_debug(f"更新MQTT上报时间 - 任务ID:{self.task_id}, "
                             f"冷却时间:{self.mqtt_cooldown}秒")

        except Exception as e:
            log_task_error(f"事故保存和发布失败 - 任务ID:{self.task_id}, 错误:{str(e)}")


    def get_model_info(self):
        """获取当前使用的模型信息"""
        loaded_models = model_manager.get_loaded_models()
        return {
            "current_model_index": self.model_index,
            "current_model_name": self.model_name,
            "loaded_models": loaded_models,
            "total_loaded_models": model_manager.get_model_count()
        }

    def get_topic(self):
        current_timestamp = datetime.now()
        datetime_str = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")  # 精确到秒
        topic_name = f"{datetime_str}-{self.model_name}"
        return topic_name

    def _stop_monitor_thread(self):
        """安全停止监控线程"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            log_task_debug(f"停止监控线程 - 任务ID:{self.task_id}")

            # 发送停止信号
            self._should_stop = True
            self._monitor_shutdown_event.set()

            # 等待线程退出
            try:
                self._monitor_thread.join(timeout=6.0)  # 增加等待时间到5秒
                if self._monitor_thread.is_alive():
                    log_task_error(f"监控线程退出超时 - 任务ID:{self.task_id}")
                else:
                    log_task_debug(f"监控线程已安全退出 - 任务ID:{self.task_id}")
            except Exception as e:
                log_task_error(f"停止监控线程时异常 - 任务ID:{self.task_id}, 错误:{str(e)}")

        self._monitor_started = False

    def request_stop(self):
        """请求停止workflow"""
        log_task(f"任务停止请求 - 任务ID:{self.task_id}")

        # 设置停止标志
        self._should_stop = True
        self._stop_event.set()

        # 停止监控线程
        self._stop_monitor_thread()

        try:
            self.mqtt_client.disconnect()
            log_task_debug(f"MQTT连接已断开 - 任务ID:{self.task_id}")
        except Exception as e:
            log_task_error(f"断开MQTT连接失败 - 任务ID:{self.task_id}, 错误:{str(e)}")

    def is_stop_requested(self):
        """检查是否收到停止请求"""
        return self._should_stop

    async def check_stop(self):
        """异步检查停止请求"""
        if self._should_stop:
            log_task(f"检测到停止请求，正在停止工作流 - 任务ID:{self.task_id}")
            self.mqtt_client.disconnect()
            return True
        return False

    def check_stream_alive(self):
        """监控主推理循环的健康状态（通过最后帧时间而非频繁创建连接）"""
        log_task_debug(f"流健康监控启动 - 任务ID:{self.task_id}")
        check_interval = 10  # 每10秒检查一次
        max_idle_time = 180  # 3分钟无活动则认为流断开

        # 异常计数机制：允许一定次数的异常，避免误停止
        consecutive_failures = 0  # 连续异常次数
        max_failures = 3  # 连续3次异常（每次间隔10秒）就停止任务
        init_wait_time = 30  # 修复：减少初始化等待时间到30秒

        # 使用两个事件控制退出：主停止信号和监控线程专用关闭信号
        while not self._should_stop and not self._monitor_shutdown_event.is_set():
            # 等待检查间隔，但同时响应关闭信号
            if self._monitor_shutdown_event.wait(timeout=check_interval):
                log_task_debug(f"监控线程收到关闭信号，退出循环 - 任务ID:{self.task_id}")
                break

            try:
                # 检查主推理循环的最后活动时间，而不是创建新的VideoCapture
                # 这样可以避免与YOLO竞争RTMP连接
                if hasattr(self, '_last_frame_time'):
                    idle_time = time.time() - self._last_frame_time
                    
                    if idle_time < max_idle_time:
                        # 流状态正常，重置失败计数
                        log_task_debug(f"流状态正常 - 任务ID:{self.task_id}, 闲置时间:{idle_time:.1f}秒")
                        consecutive_failures = 0  # ✅ 恢复正常时重置计数
                    else:
                        # 流长时间无响应，累积失败次数
                        consecutive_failures += 1
                        log_task_error(
                            f"流长时间无响应 - 任务ID:{self.task_id}, 闲置时间:{idle_time:.1f}秒, "
                            f"连续异常:{consecutive_failures}/{max_failures}")
                        
                        # 达到失败上限，停止任务
                        if consecutive_failures >= max_failures:
                            log_task_error(
                                f"流连续{consecutive_failures}次无响应（{consecutive_failures * check_interval}秒），停止任务 - 任务ID:{self.task_id}")
                            self._should_stop = True
                            break
                else:
                    # 初始化阶段，还没有开始接收帧
                    # 检查是否超过初始化等待时间
                    if hasattr(self, '_monitor_start_time'):
                        wait_time = time.time() - self._monitor_start_time
                        if wait_time > init_wait_time:
                            consecutive_failures += 1
                            log_task_error(
                                f"推理循环启动超时 - 任务ID:{self.task_id}, "
                                f"已等待:{wait_time:.1f}秒, 连续异常:{consecutive_failures}/{max_failures}")
                            
                            if consecutive_failures >= max_failures:
                                log_task_error(
                                    f"推理循环启动失败，停止任务 - 任务ID:{self.task_id}")
                                self._should_stop = True
                                break
                        else:
                            log_task_debug(f"等待推理循环启动 - 任务ID:{self.task_id}, 已等待:{wait_time:.1f}秒")
                    else:
                        # 记录监控启动时间
                        self._monitor_start_time = time.time()
                        log_task_debug(f"等待推理循环启动 - 任务ID:{self.task_id}")

            except Exception as e:
                consecutive_failures += 1
                log_task_error(
                    f"流健康监控异常 - 任务ID:{self.task_id}, 错误:{str(e)}, "
                    f"连续异常:{consecutive_failures}/{max_failures}")
                
                # 达到失败上限，停止任务
                if consecutive_failures >= max_failures:
                    log_task_error(
                        f"流健康监控连续{consecutive_failures}次异常，停止任务 - 任务ID:{self.task_id}")
                    self._should_stop = True
                    break

        log_task_debug(f"流健康监控线程退出 - 任务ID:{self.task_id}")

    async def run_video(self):
        log_task(f"开始视频推理任务 - 任务ID:{self.task_id}")

        # 初始化变量，确保在finally块中可用
        fps = None
        width = None
        height = None
        vid_stride = 2

        try:
            # 使用增强的视频后端配置连接RTMP流 - 先连接，再启动监控
            log_task_debug(f"开始连接视频流 - 任务ID:{self.task_id}, 地址:{self.video_path}")

            # 自动检测并获取适合的RTMP配置
            config = auto_rtmp_config(self.video_path)

            # 尝试连接视频流并获取流信息
            cap = create_video_capture(self.video_path, config.backend)

            if not cap.isOpened:
                log_task_error(f"无法连接到视频流 - 任务ID:{self.task_id}, 地址:{self.video_path}")
                raise Exception(f"无法连接到视频流: {self.video_path}")

            # 获取视频流信息
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 使用配置管理器计算合适的帧间隔
            from .rtmp_config import RTMPConfigManager
            vid_stride = 2
            vid_stride = vid_stride if vid_stride > 0 else 1

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 释放测试连接
            cap.release()

            # 获取配置等待时间
            wait_time = RTMPConfigManager.get_wait_time_for_rtmp()

            log_task(f"视频流连接成功 - 任务ID:{self.task_id}, FPS:{fps}, 分辨率:{width}x{height}, 推理间隔:{vid_stride}帧")
            log_task_debug(f"连接参数 - 任务ID:{self.task_id}, 后端:{config.backend.value}, 最大重试:{config.max_retries}, 重试延迟:{config.retry_delay}s")
            log_task_debug(f"等待{wait_time}秒让RTMP服务器准备好...")
            await asyncio.sleep(wait_time)

            # RTMP连接成功后再启动监控线程
            log_task_debug(f"RTMP连接成功，启动监控线程 - 任务ID:{self.task_id}")
            self._monitor_thread = threading.Thread(target=self.check_stream_alive, daemon=True)
            self._monitor_thread.start()
            self._monitor_started = True
            log_task_debug(f"流健康监控线程启动 - 任务ID:{self.task_id}")

            # 设置初始帧时间，避免健康监控误判
            self._last_frame_time = time.time()
            log_task_debug(f"设置初始帧时间 - 任务ID:{self.task_id}")

            # ========== 开始视频推理逻辑 ==========
            log_task_debug(f"开始视频推理 - 任务ID:{self.task_id}, 模型索引:{self.model_index}")

            # 根据模型索引开始推理
            # 使用StreamManager进行帧共享，避免RTMP连接冲突
            if self.model_index == 1:
                results = create_stream_iterator(
                    model=self.model,
                    video_path=self.video_path,
                    task_id=self.task_id,
                    vid_stride=vid_stride,
                    classes=self.classes,
                    imgsz=(int(height), int(width)),
                    verbose=False,
                    conf=self.model_conf,
                    stop_check_callback=lambda: self._should_stop,
                    use_stream_manager=USE_STREAM_MANAGER
                )
            elif self.model_index_3:
                results = create_stream_iterator(
                    model=self.model,
                    video_path=self.video_path,
                    task_id=self.task_id,
                    vid_stride=vid_stride,
                    classes=self.classes,
                    imgsz=(int(height), int(width)),
                    verbose=False,
                    conf=self.model_conf,
                    stop_check_callback=lambda: self._should_stop,
                    use_stream_manager=USE_STREAM_MANAGER
                )
            else:
                results = create_stream_iterator(
                    model=self.model,
                    video_path=self.video_path,
                    task_id=self.task_id,
                    vid_stride=vid_stride,
                    classes=self.classes,
                    imgsz=(int(height), int(width)),
                    verbose=False,
                    conf=self.model_conf,
                    stop_check_callback=lambda: self._should_stop,
                    use_stream_manager=USE_STREAM_MANAGER
                )
            # 根据模型类型执行不同的推理逻辑
            if self.model_index == 1:
                # 消防通道占用，需要跟踪占用时间
                track_records = defaultdict(lambda: {'first_seen': None, 'last_seen': None, 'violation': False})
                time_threshold = self.config.model_list[self.model_index].get('time_threshold', 30)  # 默认30秒
                current_frame_ids = set()
                # 添加连续未出现帧数跟踪
                consecutive_missing_frames = defaultdict(int)
                max_missing_frames = 5  # 连续5帧未出现则移除

                frame_count = 0

                for result in results:
                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    current_timestamp = datetime.now(BeiJingTime)
                    date_str = current_timestamp.strftime("%Y-%m-%d")
                    # 检查停止请求
                    if await self.check_stop():
                        log_task(f"模型1收到停止请求，退出推理循环 - 任务ID:{self.task_id}")
                        self._should_stop = True
                        break

                    ori_img_shape = result.orig_shape
                    frame_count += 1
                    current_time = frame_count * vid_stride / fps  # 当前视频时间（秒）

                    results_dict = self.model.post_process([result], pixel_position=self.pixel_position)

                    # 获取当前帧检测到的所有track_id并处理
                    current_frame_ids = set()
                    for result_item in results_dict:
                        track_id = result_item.get('track_id', None)
                        if track_id is not None:
                            current_frame_ids.add(track_id)

                            # 更新追踪记录
                            if track_records[track_id]['first_seen'] is None:
                                track_records[track_id]['first_seen'] = current_time
                                log_task(f"检测到新目标进入消防通道 - 任务ID:{self.task_id}, 目标ID:{track_id}")

                            track_records[track_id]['last_seen'] = current_time

                            # 检查是否违规
                            duration = current_time - track_records[track_id]['first_seen']
                            if duration >= time_threshold:
                                track_records[track_id]['violation'] = True
                                object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{track_id}.jpg"
                                log_task(
                                    f"消防通道占用违规 - 任务ID:{self.task_id}, 目标ID:{track_id}, 占用时长:{duration:.1f}秒, 图片:{object_name}")
                                infer_image = result.plot()
                                _, _ = self.minio_client.upload_image_array(
                                    image_array=infer_image,
                                    object_name=object_name,
                                    image_format='jpg',
                                    quality=85
                                )

                                # 使用MQTT格式化器构建消防通道占用消息
                                mqtt_message = MQTTMessageFormatter.format_fire_lane_violation_message(
                                    object_name=object_name,
                                    result_item=result_item,
                                    obj_num=len(result),
                                    ori_img_shape=ori_img_shape
                                )
                                # 发送到MQTT主题: {类别名}
                                log_task_debug(f"发送MQTT消息 - 任务ID:{self.task_id}, 主题:{self.topic}")
                                mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

                    # 更新所有track_id的连续未出现帧数
                    for track_id in list(track_records.keys()):
                        if track_id not in current_frame_ids:
                            consecutive_missing_frames[track_id] += 1
                            # 如果连续5帧未出现，则从track_records中移除
                            if consecutive_missing_frames[track_id] >= max_missing_frames:
                                log_task_debug(
                                    f"目标移除追踪 - 任务ID:{self.task_id}, 目标ID:{track_id}, 连续{max_missing_frames}帧未检测")
                                del track_records[track_id]
                                del consecutive_missing_frames[track_id]
                        else:
                            # 重置连续未出现帧数
                            consecutive_missing_frames[track_id] = 0

            elif self.model_index_3:
                # 事故检测模型，要补充车辆识别
                accident_id = []

                for result in results:
                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    # 检查停止请求
                    accident_time_start = time.time()
                    if await self.check_stop():
                        log_task(f"模型3收到停止请求，退出推理循环 - 任务ID:{self.task_id} \n")
                        self._should_stop = True
                        break

                    if len(result) == 0:
                        continue

                    # 推理日志（每1秒记录一次）
                    current_time = time.time()
                    if current_time - self._last_log_time >= self._log_interval:
                        log_task(f"模型{self.model_index}推理中")
                        self._last_log_time = current_time

                    # 后处理检测结果
                    results_list = self.model.post_process([result])

                    ori_img_shape = result.orig_shape
                    if not results_list:
                        continue
                    # 分离不同类别的检测结果（支持7类目标）
                    accident_boxes = []  # class=0 (accident)
                    pedestrian_boxes = []  # class=1 (pedestrian)
                    motorcycle_boxes = []  # class=2 (motorcycle)
                    car_boxes = []  # class=3 (car)
                    large_vehicle_boxes = []  # class=4 (large vehicle)
                    traffic_police_boxes = []  # class=5 (traffic police)
                    police_motorcycle_boxes = []  # class=6 (police motorcycle)

                    for result_item in results_list:
                        class_id = result_item.get('classed', result_item.get('class_id', -1))
                        class_name = result_item.get('className', '')

                        if class_id == 0 or class_name == 'accident':  # class=0
                            accident_boxes.append(result_item)
                        elif class_id == 1 or class_name == 'pedestrian':  # class=1  ⚠️ 修正拼写
                            pedestrian_boxes.append(result_item)
                        elif class_id == 2 or class_name == 'motorcycle':  # class=2
                            motorcycle_boxes.append(result_item)
                        elif class_id == 3 or class_name == 'car':  # class=3
                            car_boxes.append(result_item)
                        elif class_id == 4 or class_name == 'large_vehicle':  # class=4
                            large_vehicle_boxes.append(result_item)
                        elif class_id == 5 or class_name == 'traffic_police':  # class=5
                            traffic_police_boxes.append(result_item)
                        elif class_id == 6 or class_name == 'police_motorcycle':  # class=6
                            police_motorcycle_boxes.append(result_item)

                    # 如果没有检测到事故，跳过
                    if not accident_boxes:
                        continue

                    # 使用验证管理器获取通过验证的事故
                    verified_indices = self.verification_manager.get_verified_accidents(accident_boxes, pedestrian_boxes)

                    # 构建详细的验证日志
                    pedestrian_count = len(pedestrian_boxes)
                    traffic_police_count = len(traffic_police_boxes)
                    total_pedestrian_police = pedestrian_count + traffic_police_count

                    # 获取验证策略信息
                    strategy_name = self.verification_manager.get_strategy_info()['strategy_name']

                    if len(verified_indices) == 0:
                        # 验证未通过，输出详细信息
                        if total_pedestrian_police < 2:
                            # 数量不足
                            log_task(
                                f"事故验证完成 - 任务ID:{self.task_id}, 总事故数:{len(accident_boxes)}, "
                                f"验证通过数:0 (原因:{strategy_name}策略-行人+交警数量不足, "
                                f"检测到行人:{pedestrian_count}, 交警:{traffic_police_count}, 总数:{total_pedestrian_police} < 2)"
                            )
                        else:
                            # 数量足够但不满足策略条件（如重叠面积不足或距离太远）
                            log_task(
                                f"事故验证完成 - 任务ID:{self.task_id}, 总事故数:{len(accident_boxes)}, "
                                f"验证通过数:0 (原因:{strategy_name}策略-数量满足但条件不符, "
                                f"行人:{pedestrian_count}, 交警:{traffic_police_count}, 总数:{total_pedestrian_police} ≥ 2, "
                                f"但未满足{strategy_name}策略的具体条件)"
                            )
                    else:
                        # 验证通过
                        log_task_debug(
                            f"事故验证完成 - 任务ID:{self.task_id}, 总事故数:{len(accident_boxes)}, "
                            f"验证通过数:{len(verified_indices)} ({strategy_name}策略, 行人:{pedestrian_count}, 交警:{traffic_police_count})"
                        )

                    # 处理所有通过验证的事故
                    for idx in verified_indices:
                        result_item = accident_boxes[idx]
                        current_timestamp = datetime.now(BeiJingTime)
                        date_str = current_timestamp.strftime("%Y-%m-%d")
                        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

                        id = result_item.get('track_id', None)
                        if id in accident_id:
                            log_task_debug(f"重复事故事件，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
                            continue

                        log_task(f"检测到验证后的真实事故 - 任务ID:{self.task_id}, 事件ID:{id}")
                        object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{id}.jpg"
                        log_task_debug(f"事故图片保存路径 - 任务ID:{self.task_id}, 路径:{object_name}")
                        accident_id.append(id)

                        # ========== 事故车辆数量识别（使用模型5专用车辆检测） ==========
                        car_time_start = time.time()

                        # 提取事故框OBB坐标
                        accident_obb = [
                            result_item['x'],
                            result_item['y'],
                            result_item['width'],
                            result_item['height'],
                            result_item['rotation']
                        ]

                        # ⭐ 调用模型5（专用车辆检测模型）检测车辆
                        car_time_start = time.time()

                        try:
                            car_result = await reasoner_single.infer_image(
                                result.orig_img,
                                5,  # 模型5
                                post_msg=False
                            )

                            # 处理模型5的检测结果
                            motorcycle_boxes_m5 = []  # 摩托车/电动车
                            car_boxes_m5 = []  # 普通车辆（car, van）
                            large_vehicle_boxes_m5 = []  # 大型车辆（truck, bus）

                            if car_result and len(car_result) > 0:
                                result_m5 = car_result[0]

                                # 获取OBB检测框信息
                                if hasattr(result_m5, 'obb') and result_m5.obb is not None:
                                    obb_m5 = result_m5.obb
                                    cls_m5 = obb_m5.cls.tolist()
                                    conf_m5 = obb_m5.conf.tolist()
                                    xywhr_m5 = obb_m5.xywhr.tolist()

                                    # 分类模型5的检测结果
                                    for i, (class_id_m5, conf_m5, box_m5) in enumerate(zip(cls_m5, conf_m5, xywhr_m5)):
                                        class_id_m5 = int(class_id_m5)

                                        # 模型5类别映射：
                                        # 0: pedestrian (忽略)
                                        # 1: people (忽略)
                                        # 2: bicycle → 摩托车
                                        # 3: car → 普通车辆
                                        # 4: van → 普通车辆
                                        # 5: truck → 大型车辆
                                        # 6: tricycle → 摩托车
                                        # 7: awning-tricycle → 摩托车
                                        # 8: bus → 大型车辆
                                        # 9: motor → 摩托车

                                        if class_id_m5 in [0, 1]:
                                            # 忽略行人和人群
                                            continue

                                        # 转换为统一格式
                                        vehicle_box = {
                                            'x': box_m5[0],
                                            'y': box_m5[1],
                                            'width': box_m5[2],
                                            'height': box_m5[3],
                                            'rotation': box_m5[4],  # OBB旋转角度
                                            'score': conf_m5,
                                            'track_id': f"m5_{i}",
                                            'classed': class_id_m5,
                                            'className': result_m5.names[class_id_m5]
                                        }

                                        # 分类到对应的列表
                                        if class_id_m5 in [3, 4]:  # car, van
                                            car_boxes_m5.append(vehicle_box)
                                        elif class_id_m5 in [5, 8]:  # truck, bus
                                            large_vehicle_boxes_m5.append(vehicle_box)
                                        elif class_id_m5 in [2, 6, 7, 9]:  # bicycle, tricycle, awning-tricycle, motor
                                            motorcycle_boxes_m5.append(vehicle_box)

                            # 合并所有车辆
                            all_vehicle_boxes_m5 = car_boxes_m5 + motorcycle_boxes_m5 + large_vehicle_boxes_m5

                            log_task_debug(
                                f"模型5车辆检测完成 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                f"摩托车:{len(motorcycle_boxes_m5)}, 汽车:{len(car_boxes_m5)}, "
                                f"大型车辆:{len(large_vehicle_boxes_m5)}, 总计:{len(all_vehicle_boxes_m5)}"
                            )

                        except Exception as e:
                            log_task_error(f"模型5检测失败 - 任务ID:{self.task_id}, 事件ID:{id}, 错误:{str(e)}")
                            import traceback
                            log_task_error(f"错误详情: {traceback.format_exc()}")
                            all_vehicle_boxes_m5 = []

                        if len(all_vehicle_boxes_m5) == 0:  # 没有识别到车辆
                            log_task_debug(f"模型5未检测到车辆，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
                            continue

                        # 将所有车辆框转换为OBB格式
                        all_vehicle_obb = []
                        for vehicle_box in all_vehicle_boxes_m5:
                            vehicle_obb = [
                                vehicle_box['x'],
                                vehicle_box['y'],
                                vehicle_box['width'],
                                vehicle_box['height'],
                                vehicle_box['rotation']
                            ]
                            # 将xywhr转换为8点格式
                            import math
                            x, y, w, h, angle = vehicle_obb
                            cos_a, sin_a = math.cos(angle), math.sin(angle)
                            corners = [[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]
                            box_vertices = [(cx * cos_a - cy * sin_a + x, cx * sin_a + cy * cos_a + y)
                                          for cx, cy in corners]
                            all_vehicle_obb.append(box_vertices)

                        # ⭐ 获取车辆重叠阈值配置（与一步验证保持一致）
                        from model_library.tools.utils import Config
                        config_obj = Config()
                        model_config = config_obj.model_list[3]
                        verification_config = model_config.get('verification_config', {})
                        vehicle_overlap_threshold = verification_config.get('vehicle_overlap_threshold', 0.5)

                        # 计算车辆与事故框的交集
                        # ⭐ 使用config中配置的阈值，默认0.5（与一步验证保持一致）
                        inter_index = GeometryUtils.intersection_judgment(
                            accident_obb,
                            all_vehicle_obb,
                            threshold=vehicle_overlap_threshold
                        )
                        accident_car = len(inter_index)

                        # ⭐ 增强日志：显示重叠判断详情
                        log_task_debug(
                            f"车辆重叠判断 - 任务ID:{self.task_id}, 事件ID:{id}, "
                            f"检测到车辆数:{len(all_vehicle_obb)}, "
                            f"事故框内车辆数:{accident_car}, "
                            f"重叠阈值:{vehicle_overlap_threshold}(config配置)"
                        )

                        if accident_car == 0:
                            log_task_debug(f"事故框内无车辆，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
                            continue

                        # 提取涉事车辆的OBB坐标
                        accident_obb_list = [all_vehicle_obb[i] for i in inter_index]
                        # 转换格式：[[x,y],[x,y],[x,y],[x,y]] -> [x,y,x,y,x,y,x,y]
                        accident_obb_list = [[coord for point in shape for coord in point] for shape in accident_obb_list]

                        car_time_end = time.time()

                        # 将车辆统计信息添加到结果项
                        result_item['accident_car_count'] = accident_car
                        result_item['accident_car_xyxy'] = accident_obb_list

                        # ⭐ 增强日志：记录车辆类别统计详情（使用模型5检测结果）
                        log_task_debug(
                            f"事故车辆识别完成 - 任务ID:{self.task_id}, 事件ID:{id}, "
                            f"车辆总数:{accident_car}, "
                            f"摩托车:{len(motorcycle_boxes_m5)}, "
                            f"汽车:{len(car_boxes_m5)}, "
                            f"大型车辆:{len(large_vehicle_boxes_m5)}, "
                            f"耗时:{car_time_end - car_time_start:.3f}秒"
                        )

                        # 事故类型初步分类（摩托车/大型车辆/普通）
                        # ⭐ 使用模型5检测到的车辆进行分类
                        from model_library.tools.accident_strategies import classify_accident_type

                        # 合并模型3检测的目标（事故、行人、交警）和模型5检测的车辆
                        all_boxes = (accident_boxes + pedestrian_boxes + motorcycle_boxes_m5 +
                                   car_boxes_m5 + large_vehicle_boxes_m5 + traffic_police_boxes +
                                   police_motorcycle_boxes)
                        accident_type_info = classify_accident_type(result_item, all_boxes)
                        accident_type = accident_type_info['accident_type']
                        accident_message = accident_type_info['message']
                        log_task_debug(f"事故类型分类完成 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                     f"类型:{accident_type}, "
                                     f"有交警:{accident_type_info['has_police']}, "
                                     f"行人:{accident_type_info['pedestrian_count']}, "
                                     f"交警:{accident_type_info['police_count']}, "
                                     f"消息:{accident_message}")

                        # ========== 新的协同验证流程 ==========
                        # 步骤1: 一步验证（人+车辆验证）
                        from model_library.tools.accident_strategies import verify_accident_with_vehicles

                        # ⭐ 复用之前获取的配置对象（避免重复读取）
                        vehicle_verification_config = model_config.get('verification_config', {}).get('vehicle_verification', {})
                        vlm_config = model_config.get('vlm_verification', {})

                        # 执行一步验证（使用模型5检测到的车辆）
                        step1_result = verify_accident_with_vehicles(
                            result_item,
                            pedestrian_boxes,
                            traffic_police_boxes,
                            car_boxes_m5,  # ⭐ 使用模型5检测的汽车
                            motorcycle_boxes_m5,  # ⭐ 使用模型5检测的摩托车
                            large_vehicle_boxes_m5,  # ⭐ 使用模型5检测的大型车辆
                            vehicle_verification_config
                        )

                        # ⭐ 增强日志：显示一步验证的详细信息
                        original_score = step1_result.get('original_score', 0.0)
                        yolo_boost = step1_result.get('yolo_score_boost', 0.0)
                        boosted_score = step1_result['boosted_score']
                        boost_reason = step1_result['boost_reason']

                        log_task(f"一步验证完成 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                f"原始YOLO分数:{original_score:.3f}, "
                                f"分数提升:{yolo_boost:.3f}({boost_reason}), "
                                f"提升后分数:{boosted_score:.3f}, "
                                f"结果:{step1_result['message']}")

                        if not step1_result['passed']:
                            log_task(f"一步验证失败，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}, 原因:{step1_result['message']}")
                            continue

                        # 使用一步验证确定的事故类型（优先使用一步验证的结果）
                        accident_type = step1_result['accident_type']
                        boosted_yolo_score = boosted_score

                        log_task(f"一步验证通过 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                f"事故类型:{accident_type}, "
                                f"YOLO分数:{original_score:.3f} → {boosted_yolo_score:.3f} (+{yolo_boost:.3f})")

                        # 步骤2: VLM多模态验证（获取置信度分数）
                        vlm_confidence = 0.0
                        vlm_enabled = self.vlm_verifier and self.vlm_verifier.enabled

                        if vlm_enabled:
                            vlm_start_time = time.time()
                            # 根据事故类型选择对应的prompt
                            prompt_type = "普通" if accident_type == "normal" else ("摩托车" if accident_type == "motorcycle" else "大型车辆")

                            log_task_debug(f"VLM验证开始 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                         f"事故类型:{accident_type}, Prompt类型:{prompt_type}, "
                                         f"流式输出:{self.vlm_verifier.stream}, "
                                         f"超时配置:{self.vlm_verifier.total_timeout}秒")

                            # 使用绘制了事故框的图片进行验证，帮助大模型聚焦
                            vlm_image = self.verification_manager.plot_verified_accidents_only(result, [result_item])
                            # 根据事故类型使用对应的prompt进行验证，获取置信度分数
                            vlm_confidence = self.vlm_verifier.verify_accident_with_type(vlm_image, accident_type)

                            vlm_elapsed = time.time() - vlm_start_time
                            # 检查是否使用了默认分数（超时或失败）
                            used_default = "使用默认分数" if vlm_confidence == self.vlm_verifier.default_confidence else "正常返回"

                            log_task(f"VLM验证完成 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                    f"Prompt类型:{prompt_type}, "
                                    f"VLM置信度:{vlm_confidence:.3f}({used_default}), "
                                    f"耗时:{vlm_elapsed:.3f}秒")
                        else:
                            log_task_debug(f"VLM验证未启用 - 任务ID:{self.task_id}, 事件ID:{id}, 使用默认分数:{vlm_confidence:.3f}")

                        # 步骤3: YOLO与VLM协同验证
                        yolo_weight = vlm_config.get('yolo_weight', 0.5)
                        vlm_weight = vlm_config.get('vlm_weight', 0.5)
                        final_threshold = vlm_config.get('final_threshold', 0.6)

                        # 计算加权总分
                        final_score = yolo_weight * boosted_yolo_score + vlm_weight * vlm_confidence

                        # ⭐ 增强日志：显示协同验证的详细计算过程
                        log_task(f"协同验证计算 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                f"计算: {yolo_weight:.2f}×{boosted_yolo_score:.3f} + {vlm_weight:.2f}×{vlm_confidence:.3f} = {final_score:.3f}, "
                                f"阈值:{final_threshold}")

                        if final_score < final_threshold:
                            log_task(f"❌ 协同验证失败 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                    f"总分:{final_score:.3f} < 阈值:{final_threshold}, 跳过上报")
                            continue

                        log_task(f"✅ 协同验证通过 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                f"总分:{final_score:.3f} >= 阈值:{final_threshold}, "
                                f"事故类型:{accident_type}, 准备上报")

                        # 更新事故消息（根据一步验证的结果）
                        if accident_type == "motorcycle":
                            accident_message = f"摩托车事故（{step1_result['message']}）"
                        elif accident_type == "large_vehicle":
                            accident_message = f"大型车辆事故（{step1_result['message']}）"
                        else:  # normal
                            accident_message = f"普通事故（{step1_result['message']}）"

                        # ⭐ 增强日志：上报前汇总所有验证信息
                        log_task(f"🚨 事故上报汇总 - 任务ID:{self.task_id}, 事件ID:{id}, "
                                f"类型:{accident_type}, "
                                f"YOLO:{original_score:.3f}→{boosted_yolo_score:.3f}, "
                                f"VLM:{vlm_confidence:.3f}, "
                                f"最终总分:{final_score:.3f}, "
                                f"消息:{accident_message}")

                        # ⭐ 构建验证详情信息（用于MQTT消息）
                        verification_info = {
                            "pedestrian_count": step1_result.get('pedestrian_count', 0),
                            "police_count": step1_result.get('police_count', 0),
                            "car_count": step1_result.get('car_count', 0),
                            "motorcycle_count": step1_result.get('motorcycle_count', 0),
                            "large_vehicle_count": step1_result.get('large_vehicle_count', 0),
                            "has_police": step1_result.get('police_count', 0) > 0,
                            "original_yolo_score": round(original_score, 3),
                            "boosted_yolo_score": round(boosted_yolo_score, 3),
                            "yolo_score_boost": round(yolo_boost, 3),
                            "vlm_confidence": round(vlm_confidence, 3),
                            "final_score": round(final_score, 3)
                        }

                        # 保存和上报事故信息（传递事故类型和验证详情）
                        await self._save_and_publish_accident(
                            result, result_item, object_name, ori_img_shape, timestamp_str,
                            accident_message, accident_type, verification_info
                        )
                    accident_time_end = time.time()
                    log_task_debug(
                        f"事故检测处理完成 - 任务ID:{self.task_id}, 总耗时:{accident_time_end - accident_time_start:.3f}秒")

            elif self.model_index_8:
                # 夜间红外摩托车飙车检测模型 - 轨迹追踪模式（检测聚集→记录轨迹→上报一次）
                for result in results:
                    # ✅ 修复：在循环开始就定义current_timestamp，确保所有代码路径都能访问
                    current_timestamp = datetime.now().timestamp()

                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    # 检查停止请求
                    if await self.check_stop():
                        log_task(f"模型8收到停止请求，退出推理循环 - 任务ID:{self.task_id} \n")
                        self._should_stop = True
                        break

                    # 推理日志（每1秒记录一次）
                    current_time = time.time()
                    if current_time - self._last_log_time >= self._log_interval:
                        log_task(f"模型{self.model_index}推理中")
                        self._last_log_time = current_time

                    # 如果没有检测到目标，也要处理轨迹追踪模式更新
                    if len(result) == 0:
                        log_task_debug(f"[模型8轨迹检测] 当前帧未检测到任何目标")
                        all_motorcycles = []
                    else:
                        # 提取所有摩托车目标
                        all_motorcycles = self.model.post_process([result])
                        log_task_debug(f"[模型8轨迹检测] 检测到{len(all_motorcycles)}个摩托车")

                    # ========== 轨迹追踪模式逻辑 ==========
                    tracking_mode_report = None

                    # 检查是否处于轨迹追踪模式
                    if self.gathering_manager.is_in_tracking_mode():
                        # 更新轨迹追踪模式状态
                        tracking_mode_report = self.gathering_manager.update_tracking_mode(
                            all_motorcycles, current_timestamp
                        )

                        # 检查是否追踪失败
                        if tracking_mode_report['is_timeout']:
                            # 追踪失败，只记录日志，不再发送MQTT消息
                            tracking_failure_info = tracking_mode_report['tracking_info']

                            log_task(f"[模型8轨迹检测] 追踪失败 - track_id:{tracking_failure_info['track_id']}, 重置轨迹追踪模式")

                            # 重置轨迹追踪模式
                            self.gathering_manager.reset_tracking_mode()
                            continue

                        # 检查是否应该上报
                        if tracking_mode_report['should_report']:
                            tracking_info = tracking_mode_report['tracking_info']
                            is_final_report = tracking_mode_report.get('is_final_report', False)

                            log_task(f"[模型8轨迹检测] 上报追踪信息 - track_id:{tracking_info['track_id']}, 状态:{tracking_info['tracking_state']}, 已用时间:{tracking_info['elapsed_time']}秒, 推理次数:{tracking_info.get('inference_count', 'N/A')}")

                            # 构建时间戳
                            current_dt = datetime.fromtimestamp(current_timestamp, BeiJingTime)
                            date_str = current_dt.strftime("%Y-%m-%d")
                            timestamp_str = current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")

                            # 生成文件名
                            object_name = f"ai/{date_str}/{self.model_name}/{current_dt}.jpg"

                            # ✅ 新绘图逻辑：绘制所有摩托车的目标框，并为置信度最高的摩托车绘制箭头
                            # cv2 和 numpy 已在文件顶部导入，无需重复导入

                            # 获取原始图片
                            infer_image = result.orig_img.copy() if len(result) > 0 else result.orig_img

                            # 绘制所有摩托车的目标框
                            for motorcycle in all_motorcycles:
                                x, y = int(motorcycle['x']), int(motorcycle['y'])
                                w, h = int(motorcycle['width'] / 2), int(motorcycle['height'] / 2)

                                # 绘制矩形框（绿色）
                                cv2.rectangle(infer_image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)

                                # 添加标签
                                label = f"{motorcycle.get('className', 'motorcycle')} {motorcycle['score']:.2f}"
                                cv2.putText(infer_image, label, (x - w, y - h - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # ✅ 为置信度最高的摩托车绘制箭头
                            if 'trajectory_points' in tracking_info and len(tracking_info['trajectory_points']) >= 2:
                                trajectory_points = tracking_info['trajectory_points']
                                target_box = tracking_info['box']

                                log_task_debug(f"[模型8轨迹检测] 准备绘制箭头 - 轨迹点数:{len(trajectory_points)}, 推理次数:{tracking_info.get('inference_count', 'N/A')}")
                                log_task_debug(f"[模型8轨迹检测] 目标框信息 - x:{target_box['x']}, y:{target_box['y']}, 箭头配置:{self.arrow_config}")

                                # 使用管理器的静态方法绘制箭头（传递箭头配置）
                                from .motorcycle_gathering_strategies import MotorcycleGatheringManager
                                infer_image = MotorcycleGatheringManager.draw_direction_arrow(
                                    infer_image,
                                    trajectory_points,
                                    target_box,
                                    arrow_config=self.arrow_config  # ✅ 使用配置中的箭头参数
                                )

                                log_task_debug(f"[模型8轨迹检测] 已绘制行进方向箭头 - 轨迹点数:{len(trajectory_points)}, 推理次数:{tracking_info.get('inference_count', 'N/A')}, 箭头长度:{self.arrow_config.get('length', 60)}")
                            else:
                                log_task_debug(f"[模型8轨迹检测] 跳过箭头绘制 - trajectory_points存在:{'trajectory_points' in tracking_info}, 数量:{len(tracking_info.get('trajectory_points', []))}")

                            # 上传图片
                            _, _ = self.minio_client.upload_image_array(
                                image_array=infer_image,
                                object_name=object_name,
                                image_format='jpg',
                                quality=85
                            )

                            # 构建追踪模式MQTT消息
                            frame_report = {
                                'timestamp': current_timestamp,
                                'gathering_count': 1 if tracking_info['tracking_state'] == 'tracking' else 0,
                                'racing_count': 1,
                                'tracking_infos': [tracking_info],
                                'has_new_reports': True,
                                'is_tracking_mode': True,  # 标记为追踪模式
                                'tracking_state': tracking_info['tracking_state']
                            }

                            mqtt_message = MQTTMessageFormatter.format_motorcycle_tracking_mode_message(
                                object_name=object_name,
                                tracking_info=tracking_info,
                                ori_img_shape=result.orig_shape if len(result) > 0 else (720, 1280),
                                task_id=self.task_id,
                                timestamp_str=timestamp_str
                            )

                            # 发送到MQTT主题
                            log_task_debug(f"发送轨迹追踪模式MQTT消息 - track_id:{tracking_info['track_id']}, 主题:{self.topic}")
                            print(mqtt_message)
                            mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

                            # ✅ 如果是最终报告，重置轨迹追踪模式并更新冷却时间
                            if is_final_report:
                                log_task(f"[模型8轨迹检测] 上报完成，重置轨迹追踪模式并进入冷却期 - track_id:{tracking_info['track_id']}, 冷却期:{self.gathering_manager.report_cooldown}秒")
                                self.gathering_manager.update_last_report_time(current_timestamp)  # ✅ 更新冷却时间
                                self.gathering_manager.reset_tracking_mode()

                    else:
                        # 不在轨迹追踪模式，检测是否满足聚集条件
                        min_count = self.gathering_manager.strategy.min_gathering_count if self.gathering_manager else 3

                        # ✅ 检查是否在冷却期内
                        if self.gathering_manager.is_in_cooldown_period(current_timestamp):
                            cooldown_remaining = self.gathering_manager.report_cooldown - (current_timestamp - self.gathering_manager.last_report_timestamp)
                            log_task_debug(f"[模型8轨迹检测] 处于上报冷却期，剩余 {cooldown_remaining:.1f} 秒，跳过聚集检测")
                        elif len(all_motorcycles) >= min_count:
                            # 检测聚集
                            gathering_boxes, _ = self.gathering_manager.detect_gathering_motorcycles(all_motorcycles)

                            if gathering_boxes:
                                # 进入轨迹追踪模式
                                success = self.gathering_manager.enter_tracking_mode(gathering_boxes, current_timestamp)

                                if success:
                                    tracking_target_id = self.gathering_manager.get_tracking_target_id()
                                    log_task(f"[模型8轨迹检测] 检测到{len(gathering_boxes)}车聚集（min_count={min_count}），进入轨迹追踪模式 - track_id:{tracking_target_id}")
                                else:
                                    log_task_debug(f"[模型8轨迹检测] 无法进入轨迹追踪模式（无有效track_id）")
                            else:
                                log_task_debug(f"[模型8轨迹检测] 检测到{len(all_motorcycles)}个摩托车，但未满足聚集条件")
                        else:
                            log_task_debug(f"[模型8轨迹检测] 未满足聚集条件（<{min_count}辆摩托车），当前数量:{len(all_motorcycles)}")

            elif self.model_index_9:
                # 行人检测模型 - 检测人群聚集并报警
                for result in results:
                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    # 检查停止请求
                    if await self.check_stop():
                        log_task(f"模型9收到停止请求，退出推理循环 - 任务ID:{self.task_id}")
                        self._should_stop = True
                        break

                    # 推理日志（每1秒记录一次）
                    current_time = time.time()
                    if current_time - self._last_log_time >= self._log_interval:
                        log_task(f"模型{self.model_index}推理中")
                        self._last_log_time = current_time


                    # 提取检测结果
                    detections = self.model.post_process([result])

                    if not detections:
                        continue

                    # 获取人群聚集信息（在第一个检测框中）
                    gathering_info = detections[0].get('gathering_info', {})

                    # 检查是否应该上报（基于状态机逻辑）
                    if not gathering_info.get('should_report', False):
                        log_task_debug(f"[模型9] 不满足上报条件 - 当前数量:{gathering_info.get('total_count', 0)}, 状态:{gathering_info.get('state', False)}, 事件类型:{gathering_info.get('event_type', None)}")
                        continue

                    # 满足上报条件，发送MQTT消息
                    event_type = gathering_info.get('event_type', 'gathering_detected')
                    total_count = gathering_info.get('total_count', 0)
                    pedestrian_count = gathering_info.get('pedestrian_count', 0)
                    people_count = gathering_info.get('people_count', 0)

                    log_task(f"[模型9] {event_type} - 总数:{total_count} (行人:{pedestrian_count}, 人群:{people_count})")

                    # 构建时间戳
                    current_timestamp = datetime.now(BeiJingTime)
                    date_str = current_timestamp.strftime("%Y-%m-%d")
                    timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

                    # 生成文件名
                    object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}.jpg"

                    # 绘制检测框并上传图片（只绘制整体大框，不绘制所有行人小框）
                    ori_img_shape = result.orig_shape
                    infer_image = plot_gathering_bounding_box(result, detections)
                    _, _ = self.minio_client.upload_image_array(
                        image_array=infer_image,
                        object_name=object_name,
                        image_format='jpg',
                        quality=85
                    )

                    # 使用MQTT格式化器构建消息
                    mqtt_message = MQTTMessageFormatter.format_gathering_message(
                        object_name=object_name,
                        detections=detections,
                        ori_img_shape=ori_img_shape,
                        task_id=self.task_id,
                        timestamp_str=timestamp_str,
                        gathering_info=gathering_info
                    )

                    # 发送MQTT消息
                    log_task_debug(f"发送人群聚集MQTT消息 - 数量:{total_count}, 主题:{self.topic}")
                    print(mqtt_message)
                    mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

            elif self.model_index_10:
                # 交通拥堵检测模型 - 检测交通拥堵并报警
                for result in results:
                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    # 检查停止请求
                    if await self.check_stop():
                        log_task(f"模型10收到停止请求，退出推理循环 - 任务ID:{self.task_id}")
                        self._should_stop = True
                        break

                    # 推理日志（每1秒记录一次）
                    current_time = time.time()
                    if current_time - self._last_log_time >= self._log_interval:
                        log_task(f"模型{self.model_index}推理中")
                        self._last_log_time = current_time

                    # 提取检测结果
                    detections = self.model.post_process([result])

                    if not detections:
                        continue

                    # 获取交通拥堵信息（在第一个检测框中）
                    congestion_info = detections[0].get('congestion_info', {})

                    # 检查是否应该上报（基于状态机逻辑）
                    if not congestion_info.get('should_report', False):
                        log_task_debug(f"[模型10] 不满足上报条件 - 当前数量:{congestion_info.get('total_count', 0)}, 状态:{congestion_info.get('state', False)}, 事件类型:{congestion_info.get('event_type', None)}")
                        continue

                    # 满足上报条件，发送MQTT消息
                    event_type = congestion_info.get('event_type', 'congestion_detected')
                    total_count = congestion_info.get('total_count', 0)
                    vehicle_counts = congestion_info.get('vehicle_counts', {})

                    log_task(f"[模型10] {event_type} - 总数:{total_count} (汽车:{vehicle_counts.get('car', 0)}, 货车:{vehicle_counts.get('van', 0)}, 卡车:{vehicle_counts.get('truck', 0)}, 巴士:{vehicle_counts.get('bus', 0)}, 摩托:{vehicle_counts.get('motor', 0)}, 自行:{vehicle_counts.get('bicycle', 0)})")

                    # 构建时间戳
                    current_timestamp = datetime.now(BeiJingTime)
                    date_str = current_timestamp.strftime("%Y-%m-%d")
                    timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

                    # 生成文件名
                    object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}.jpg"

                    # 绘制检测框并上传图片（只绘制整体大框，不绘制所有车辆小框）
                    ori_img_shape = result.orig_shape
                    infer_image = plot_congestion_bounding_box(result, detections)
                    _, _ = self.minio_client.upload_image_array(
                        image_array=infer_image,
                        object_name=object_name,
                        image_format='jpg',
                        quality=85
                    )

                    # 使用MQTT格式化器构建消息
                    mqtt_message = MQTTMessageFormatter.format_congestion_message(
                        object_name=object_name,
                        detections=detections,
                        ori_img_shape=ori_img_shape,
                        task_id=self.task_id,
                        timestamp_str=timestamp_str,
                        congestion_info=congestion_info
                    )

                    # 发送MQTT消息
                    log_task_debug(f"发送交通拥堵MQTT消息 - 数量:{total_count}, 主题:{self.topic}")
                    print(mqtt_message)
                    mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

            else:
                # 其他模型，简单逻辑识别即告警
                type_id = []
                for result in results:
                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    # 推理日志（每1秒记录一次）
                    current_time = time.time()
                    if current_time - self._last_log_time >= self._log_interval:
                        log_task(f"模型{self.model_index}推理中")
                        self._last_log_time = current_time

                    current_timestamp = datetime.now(BeiJingTime)
                    date_str = current_timestamp.strftime("%Y-%m-%d")
                    timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                    # 检查停止请求
                    if await self.check_stop():
                        log_task(f"其他模型收到停止请求，退出推理循环 - 任务ID:{self.task_id}")
                        self._should_stop = True
                        break

                    ori_img_shape = result.orig_shape
                    results_dict = self.model.post_process([result])
                    for result_item in results_dict:
                        id = result_item.get('track_id', None)
                        # 过滤掉 track_id 为 "unknown" 的消息（追踪器未初始化）
                        if id == "unknown":
                            log_task_debug(f"跳过未初始化的track_id - 任务ID:{self.task_id}, 目标ID:{id}")
                            continue
                        # 唯一性判别，模型2，6不需要进行唯一性判别
                        print("--------视频推理中------")
                        if id in type_id:
                        # if id in type_id and self.model_index not in [2,6]:
                            log_task_debug(f"重复事故事件，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
                            continue
                        object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{id}.jpg"
                        log_task(f"检测到目标 - 任务ID:{self.task_id}, 目标ID:{id}, 图片:{object_name}")
                        type_id.append(id)
                        infer_image = result.plot()
                        _, _ = self.minio_client.upload_image_array(
                            image_array=infer_image,
                            object_name=object_name,
                            image_format='jpg',
                            quality=85
                        )
                        # 使用MQTT格式化器构建通用检测消息
                        mqtt_message = MQTTMessageFormatter.format_general_detection_message(
                            object_name=object_name,
                            result_item=result_item,
                            obj_num=len(result),
                            ori_img_shape=ori_img_shape,
                            task_id=self.task_id,
                            timestamp_str=timestamp_str
                        )

                        # 发送到MQTT主题: {类别名}
                        log_task_debug(f"发送MQTT消息 - 任务ID:{self.task_id}, 目标ID:{id}, 主题:{self.topic}")
                        print(mqtt_message)
                        mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

            # 视频推理正常完成，停止健康监控
            log_task(f"视频推理正常完成 - 任务ID:{self.task_id}")
            self._should_stop = True

        except Exception as e:
            log_task_error(f"视频推理任务失败 - 任务ID:{self.task_id}, 错误:{str(e)}")
            self._should_stop = True
            raise e
        finally:
            # 确保资源清理在所有情况下都会执行
            log_task_debug(f"开始清理推理任务资源 - 任务ID:{self.task_id}")

            # 清理StreamManager订阅（如果使用了StreamManager）
            if USE_STREAM_MANAGER:
                try:
                    success = stream_manager.unsubscribe_stream(self.video_path, self.task_id)
                    if success:
                        log_task(f"取消订阅流成功 - 任务ID:{self.task_id}, url:{self.video_path}")
                    else:
                        log_task_debug(f"取消订阅流失败（可能不存在） - 任务ID:{self.task_id}, url:{self.video_path}")
                except Exception as e:
                    log_task_error(f"取消订阅流异常 - 任务ID:{self.task_id}, 错误:{str(e)}")

            # 确保监控线程被正确清理
            if self._monitor_started:
                log_task_debug(f"清理监控线程 - 任务ID:{self.task_id}")
                self._stop_monitor_thread()

            # 断开MQTT连接
            try:
                self.mqtt_client.disconnect()
                log_task_debug(f"MQTT连接已断开 - 任务ID:{self.task_id}")
            except Exception as e:
                log_task_error(f"断开MQTT连接失败 - 任务ID:{self.task_id}, 错误:{str(e)}")

            log_task_debug(f"推理任务资源清理完成 - 任务ID:{self.task_id}")

