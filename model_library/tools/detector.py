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

BeiJingTime = ZoneInfo("Asia/Shanghai")


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
        self.accident_id = []  # 普通事故ID集合
        self.motorcycle_accident_id = []  # 摩托车事故ID集合

        # 初始化 VLM 验证器 (仅用于模型3)
        self.vlm_verifier = None
        if self.model_index_3:
            model_config = self.config.model_list[self.model_index]
            vlm_config = model_config.get('vlm_verification', {})
            global_ms_conf = self.config.config.get('modelscope', {})
            self.vlm_verifier = VLMVerifier(vlm_config, global_ms_conf)

        # 初始化摩托车聚集检测管理器（仅用于模型8）
        self.gathering_manager = None
        self.model_index_8 = self.model_index == 8

        log_task_debug(f"获取模型实例 - 任务ID:{task_id}, 模型:{self.model_name}")
        self.model = model_manager.get_model(self.model_index, task_id)

        # 在模型加载后初始化完整的事故识别系统（仅用于模型3）
        if self.model_index_3:
            self.verification_manager = AccidentStrategyFactory.create_complete_accident_system(
                self.model_index, self.config, self.model, task_id
            )

            # 读取事故框内人员统计的重叠阈值配置
            model_config = self.config.model_list[self.model_index]
            verification_config = model_config.get('verification_config', {})
            self.person_overlap_threshold = verification_config.get('overlap_threshold', 0.3)
            log_task_debug(f"事故框内人员统计阈值 - 任务ID:{task_id}, 阈值:{self.person_overlap_threshold}")


        # 在模型加载后初始化完整的摩托车聚集检测系统（仅用于模型8）
        if self.model_index_8:
            self.gathering_manager = MotorcycleGatheringStrategyFactory.create_complete_gathering_system(
                self.model_index, self.config, task_id
            )
            # 将聚集检测管理器设置到模型中
            if hasattr(self.model, 'set_gathering_manager'):
                self.model.set_gathering_manager(self.gathering_manager)

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

    async def _process_accident_event(self, result, accident_item, person_boxes, ori_img_shape, accident_type='accident'):
        """
        处理单个事故事件（普通事故或摩托车事故）

        Args:
            result: YOLO检测结果
            accident_item: 事故检测项
            person_boxes: 人员框列表（行人+交警）
            ori_img_shape: 原始图像尺寸
            accident_type: 事故类型 ('accident' 或 'motorcycle_accident')
        """
        current_timestamp = datetime.now(BeiJingTime)
        date_str = current_timestamp.strftime("%Y-%m-%d")
        timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

        id = accident_item.get('track_id', None)

        # 使用不同的ID集合来避免普通事故和摩托车事故的ID冲突
        id_set = self.motorcycle_accident_id if accident_type == 'motorcycle_accident' else self.accident_id
        accident_type_name = "摩托车事故" if accident_type == 'motorcycle_accident' else "普通事故"

        if id in id_set:
            log_task_debug(f"重复{accident_type_name}事件，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
            return

        log_task(f"检测到验证后的真实{accident_type_name} - 任务ID:{self.task_id}, 事件ID:{id}")
        object_name = f"ai/{date_str}/{self.model_name}/{current_timestamp}_{id}.jpg"
        log_task_debug(f"{accident_type_name}图片保存路径 - 任务ID:{self.task_id}, 路径:{object_name}")
        id_set.append(id)

        # 事故车辆数量识别
        car_time_start = time.time()
        car_result = await reasoner_single.infer_image(result.orig_img, 5, post_msg=False)
        accident_obb = [accident_item['x'], accident_item['y'], accident_item['width'],
                              accident_item['height'], accident_item['rotation']]

        if len(car_result[0]) == 0:  # 没有识别到车辆
            log_task_debug(f"{accident_type_name}现场无车辆，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
            return

        car_result_obb = car_result[0].obb.xyxyxyxy.tolist()
        inter_index = GeometryUtils.intersection_judgment(accident_obb, car_result_obb)
        accident_car = len(inter_index)

        if accident_car == 0:
            log_task_debug(f"{accident_type_name}现场无车辆，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
            return

        accident_obb_list = [car_result_obb[i] for i in inter_index]
        # 转换格式：[[x,y],[x,y],[x,y],[x,y]] -> [x,y,x,y,x,y,x,y]
        accident_obb_list = [[coord for point in shape for coord in point] for shape in accident_obb_list]

        car_time_end = time.time()
        accident_item['accident_car_count'] = accident_car
        accident_item['accident_car_xyxy'] = accident_obb_list
        log_task_debug(f"{accident_type_name}车辆识别完成 - 任务ID:{self.task_id}, 事件ID:{id}, 车辆数:{accident_car}, 耗时:{car_time_end - car_time_start:.3f}秒")

        # VLM 多模态验证 - 根据事故类型选择不同的prompt
        if self.vlm_verifier and self.vlm_verifier.enabled:
            vlm_start_time = time.time()
            # 使用绘制了事故框的图片进行验证，帮助大模型聚焦
            vlm_image = self.verification_manager.plot_verified_accidents_only(result, [accident_item])

            # 根据事故类型选择prompt
            if accident_type == 'motorcycle_accident':
                is_confirmed = self.vlm_verifier.verify_accident(vlm_image, prompt_type='prompt2')
            else:
                is_confirmed = self.vlm_verifier.verify_accident(vlm_image, prompt_type='prompt1')

            log_task_debug(f"VLM验证结果: {is_confirmed}, 耗时:{time.time() - vlm_start_time:.3f}秒")

            if not is_confirmed:
                log_task(f"VLM未确认{accident_type_name}，跳过上报 - 任务ID:{self.task_id}, 事件ID:{id}")
                return

        # 计算该事故框内的人员类型（用于生成消息）
        # 检查哪些person_box与该事故框有重叠
        accident_poly = GeometryUtils.convert_xywhr_to_polygon(
            accident_item['x'], accident_item['y'],
            accident_item['width'], accident_item['height'],
            accident_item['rotation']
        )

        accident_pedestrian_count = 0
        accident_police_count = 0

        for person_box in person_boxes:
            person_poly = GeometryUtils.convert_xywhr_to_polygon(
                person_box['x'], person_box['y'],
                person_box['width'], person_box['height'],
                person_box['rotation']
            )
            # 计算IoU，使用配置的overlap_threshold判断是否在事故框内
            intersection = accident_poly.intersection(person_poly)
            if intersection.area > 0:
                iou = intersection.area / min(accident_poly.area, person_poly.area)
                if iou > self.person_overlap_threshold:  # 使用配置的阈值
                    if person_box.get('className') == 'pedestrian':
                        accident_pedestrian_count += 1
                    elif person_box.get('className') == 'Traffic Police':
                        accident_police_count += 1

        accident_has_police = accident_police_count > 0

        log_task_debug(f"{accident_type_name}框内人员统计 - 任务ID:{self.task_id}, 事件ID:{id}, 行人数:{accident_pedestrian_count}, 交警数:{accident_police_count}")

        # 保存和上报事故信息
        await self._save_and_publish_accident(
            result, accident_item, object_name, ori_img_shape, timestamp_str,
            accident_has_police, accident_pedestrian_count, accident_police_count,
            accident_type
        )



    async def _save_and_publish_accident(self, result, accident_item, object_name, ori_img_shape, timestamp_str,
                                        has_police=False, pedestrian_count=0, police_count=0,
                                        accident_type='accident'):
        """
        保存事故图像并发布MQTT消息

        Args:
            result: YOLO检测结果
            accident_item: 事故检测项
            object_name: 存储对象名
            ori_img_shape: 原始图像尺寸
            timestamp_str: 时间戳字符串
            has_police: 事故框内是否有交警
            pedestrian_count: 事故框内行人数
            police_count: 事故框内交警数
            accident_type: 事故类型 ('accident' 或 'motorcycle_accident')
        """
        try:
            # 使用策略工厂的绘制方法，只绘制验证后的真实事故框，不绘制行人框
            infer_image = self.verification_manager.plot_verified_accidents_only(result, [accident_item])
            _, _ = self.minio_client.upload_image_array(
                image_array=infer_image,
                object_name=object_name,
                image_format='jpg',
                quality=85
            )

            # 生成message内容（根据事故类型）
            if accident_type == 'motorcycle_accident':
                if has_police:
                    message = f"检测到摩托车交通事故,有交警,行人数:{pedestrian_count},交警数:{police_count}"
                else:
                    message = f"检测到摩托车交通事故,无交警,行人数:{pedestrian_count}"
            else:  # 普通事故
                if has_police:
                    message = f"检测到交通事故,有交警,行人数:{pedestrian_count},交警数:{police_count}"
                else:
                    message = f"检测到交通事故,无交警,行人数:{pedestrian_count}"

            # 使用MQTT格式化器构建消息
            mqtt_message = MQTTMessageFormatter.format_accident_message(
                object_name=object_name,
                accident_item=accident_item,
                ori_img_shape=ori_img_shape,
                task_id=self.task_id,
                timestamp_str=timestamp_str,
                message=message
            )
            # 确保objNum使用len(result)以保持原有逻辑
            mqtt_message["imageInfo"]["objNum"] = len(result)

            # 发送MQTT消息
            log_task_debug(f"发送事故MQTT消息 - 任务ID:{self.task_id}, 主题:{self.topic}, 消息:{message}")
            mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)
            log_task_debug(f"MQTT发送结果 - 任务ID:{self.task_id}, 成功:{mqtt_success}")

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
            if self.model_index == 1:
                results = self.model.track_video(self.video_path, stream=True, vid_stride=vid_stride, classes=self.classes,
                                                 imgsz=(int(height), int(width)), verbose=False, conf=self.model_conf)
            elif self.model_index_3:
                results = self.model.track_video(self.video_path, stream=True, vid_stride=vid_stride, classes=self.classes,
                                                 imgsz=(int(height), int(width)), verbose=False, conf=self.model_conf)
            else:
                results = self.model.track_video(self.video_path, stream=True, vid_stride=vid_stride, classes=self.classes,
                                                 imgsz=(int(height), int(width)), verbose=False, conf=self.model_conf)
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
                self.accident_id = []
                self.motorcycle_accident_id = []  # 摩托车事故ID集合

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
                    log_task(f"模型模型推理中")

                    # 后处理检测结果
                    results_list = self.model.post_process([result])

                    ori_img_shape = result.orig_shape
                    if not results_list:
                        continue
                    # 分离不同类别的检测结果
                    accident_boxes = []  # class=0 (accident)
                    motorcycle_accident_boxes = []  # class=4 (motorcycle accident)
                    person_boxes = []  # class=1 (pedestrian) + class=6 (Traffic Police)

                    for result_item in results_list:
                        class_name = result_item.get('className', '')
                        if class_name == 'accident':  # class=0
                            accident_boxes.append(result_item)
                        elif class_name == 'motorcycle accident':  # class=4 (摩托车事故)
                            motorcycle_accident_boxes.append(result_item)
                        elif class_name == 'pedestrian':  # class=1 (行人) - 修复拼写错误
                            person_boxes.append(result_item)
                        elif class_name == 'Traffic Police':  # class=6 (交警)
                            person_boxes.append(result_item)

                    # 统计人员类型（用于后续生成消息）
                    pedestrian_count = sum(1 for box in person_boxes if box.get('className') == 'pedestrian')
                    police_count = sum(1 for box in person_boxes if box.get('className') == 'Traffic Police')
                    has_police = police_count > 0

                    # 处理普通事故 (class=0: accident)
                    if accident_boxes:
                        # 使用验证管理器获取通过验证的事故
                        verified_indices = self.verification_manager.get_verified_accidents(accident_boxes, person_boxes)
                        log_task_debug(f"普通事故验证完成 - 任务ID:{self.task_id}, 总事故数:{len(accident_boxes)}, 验证通过数:{len(verified_indices)}, 行人数:{pedestrian_count}, 交警数:{police_count}")

                        # 处理所有通过验证的普通事故
                        for idx in verified_indices:
                            await self._process_accident_event(
                                result, accident_boxes[idx], person_boxes,
                                ori_img_shape, accident_type='accident'
                            )

                    # 处理摩托车事故 (class=4: motorcycle accident)
                    if motorcycle_accident_boxes:
                        # 使用验证管理器获取通过验证的摩托车事故
                        verified_indices = self.verification_manager.get_verified_accidents(motorcycle_accident_boxes, person_boxes)
                        log_task_debug(f"摩托车事故验证完成 - 任务ID:{self.task_id}, 总事故数:{len(motorcycle_accident_boxes)}, 验证通过数:{len(verified_indices)}, 行人数:{pedestrian_count}, 交警数:{police_count}")

                        # 处理所有通过验证的摩托车事故
                        for idx in verified_indices:
                            await self._process_accident_event(
                                result, motorcycle_accident_boxes[idx], person_boxes,
                                ori_img_shape, accident_type='motorcycle_accident'
                            )

                    accident_time_end = time.time()
                    log_task_debug(
                        f"事故检测处理完成 - 任务ID:{self.task_id}, 总耗时:{accident_time_end - accident_time_start:.3f}秒")

            elif self.model_index_8:
                # 夜间红外摩托车飙车检测模型 - 新追踪模式（3车聚集→追踪5秒）
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

                    # 每次推理都输出日志（与Model 3保持一致）
                    log_task(f"模型8推理中")

                    # 如果没有检测到目标，也要处理追踪模式更新
                    if len(result) == 0:
                        log_task_debug(f"[模型8追踪] 当前帧未检测到任何目标")
                        all_motorcycles = []
                    else:
                        # 提取所有摩托车目标
                        all_motorcycles = self.model.post_process([result])
                        log_task_debug(f"[模型8追踪] 检测到{len(all_motorcycles)}个摩托车")

                    # ========== 新追踪模式逻辑 ==========
                    tracking_mode_report = None

                    # 检查是否处于追踪模式
                    if self.gathering_manager.is_in_tracking_mode():
                        # 更新追踪模式状态
                        tracking_mode_report = self.gathering_manager.update_tracking_mode(
                            all_motorcycles, current_timestamp
                        )

                        # 检查是否追踪失败
                        if tracking_mode_report['is_timeout']:
                            # 发送追踪失败消息
                            tracking_failure_info = tracking_mode_report['tracking_info']

                            log_task(f"[模型8追踪] 追踪失败 - track_id:{tracking_failure_info['track_id']}, 发送失败消息")

                            # 构建追踪失败MQTT消息
                            current_dt = datetime.fromtimestamp(current_timestamp, BeiJingTime)
                            date_str = current_dt.strftime("%Y-%m-%d")
                            timestamp_str = current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")

                            mqtt_message = MQTTMessageFormatter.format_tracking_failure_message(
                                track_id=tracking_failure_info['track_id'],
                                elapsed_time=tracking_failure_info['elapsed_time'],
                                task_id=self.task_id,
                                timestamp_str=timestamp_str
                            )

                            # 发送追踪失败消息
                            log_task_debug(f"发送追踪失败MQTT消息 - track_id:{tracking_failure_info['track_id']}, 主题:{self.topic}")
                            print(mqtt_message)
                            mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

                            # 重置追踪模式
                            self.gathering_manager.reset_tracking_mode()
                            continue

                        # 检查是否应该上报
                        if tracking_mode_report['should_report']:
                            tracking_info = tracking_mode_report['tracking_info']

                            log_task(f"[模型8追踪] 上报追踪信息 - track_id:{tracking_info['track_id']}, 状态:{tracking_info['tracking_state']}, 已用时间:{tracking_info['elapsed_time']}秒")

                            # 构建时间戳
                            current_dt = datetime.fromtimestamp(current_timestamp, BeiJingTime)
                            date_str = current_dt.strftime("%Y-%m-%d")
                            timestamp_str = current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")

                            # 生成文件名
                            object_name = f"ai/{date_str}/{self.model_name}/{current_dt}.jpg"

                            # 绘制检测框并上传图片（即使丢失也上传最后已知位置的图片）
                            infer_image = result.plot() if len(result) > 0 else result.orig_img
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
                            log_task_debug(f"发送追踪模式MQTT消息 - track_id:{tracking_info['track_id']}, 主题:{self.topic}")
                            print(mqtt_message)
                            mqtt_success = self.mqtt_client.publish_message(self.topic, mqtt_message)

                    else:
                        # 不在追踪模式，检测是否满足聚集条件
                        min_count = self.gathering_manager.strategy.min_gathering_count if self.gathering_manager else 3
                        if len(all_motorcycles) >= min_count:
                            # 检测聚集
                            gathering_boxes, _ = self.gathering_manager.detect_gathering_motorcycles(all_motorcycles)

                            if gathering_boxes:
                                # 进入追踪模式
                                success = self.gathering_manager.enter_tracking_mode(gathering_boxes, current_timestamp)

                                if success:
                                    tracking_target_id = self.gathering_manager.get_tracking_target_id()
                                    log_task(f"[模型8追踪] 检测到{len(gathering_boxes)}车聚集（min_count={min_count}），进入追踪模式 - track_id:{tracking_target_id}")
                                else:
                                    log_task_debug(f"[模型8追踪] 无法进入追踪模式（无有效track_id）")
                            else:
                                log_task_debug(f"[模型8追踪] 检测到{len(all_motorcycles)}个摩托车，但未满足聚集条件")
                        else:
                            log_task_debug(f"[模型8追踪] 未满足聚集条件（<{min_count}辆摩托车），当前数量:{len(all_motorcycles)}")


            else:
                # 其他模型，简单逻辑识别即告警
                type_id = []
                for result in results:
                    # 更新最后帧时间（用于健康监控）
                    self._last_frame_time = time.time()

                    print("视频正常推理")
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

