"""MQTT消息格式化工具模块"""
from typing import Dict, Any, List


class MQTTMessageFormatter:
    """MQTT消息格式化器，负责生成各种类型的MQTT消息格式，保持与现有代码完全一致"""

    @staticmethod
    def format_accident_message(
        object_name: str,
        accident_item: Dict[str, Any],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str,
        message: str = None
    ) -> Dict[str, Any]:
        """
        格式化事故检测MQTT消息 - 支持事故类型message字段

        Args:
            object_name: 事故图片存储对象名
            accident_item: 事故检测项
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串
            message: 可选的消息字段，描述事故类型和详细信息

        Returns:
            dict: 事故检测MQTT消息
        """
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = accident_item.get("objNum", len(accident_item) if isinstance(accident_item, list) else 1)
        mqtt_message["imageInfo"]["boxs"] = accident_item
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 添加message字段（如果提供）
        if message:
            mqtt_message["imageInfo"]["message"] = message

        return mqtt_message

    @staticmethod
    def format_fire_lane_violation_message(
        object_name: str,
        result_item: Dict[str, Any],
        obj_num: int,
        ori_img_shape: tuple
    ) -> Dict[str, Any]:
        """
        格式化消防通道占用违规MQTT消息 - 保持与detector.py:427-437完全一致

        Args:
            object_name: 违规图片存储对象名
            result_item: 检测结果项
            obj_num: 检测到的对象数量 (len(result))
            ori_img_shape: 原始图像尺寸

        Returns:
            dict: 消防通道占用MQTT消息
        """
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = obj_num
        mqtt_message["imageInfo"]["boxs"] = result_item
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["message"] = "检测到消防通道被占用"

        return mqtt_message

    @staticmethod
    def format_general_detection_message(
        object_name: str,
        result_item: Dict[str, Any],
        obj_num: int,
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化通用检测MQTT消息 - 保持与detector.py:604-616完全一致

        Args:
            object_name: 检测图片存储对象名
            result_item: 检测结果项
            obj_num: 检测到的对象数量 (len(result))
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 通用检测MQTT消息
        """
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = obj_num
        mqtt_message["imageInfo"]["boxs"] = result_item
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["message"] = "检测到目标"
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        return mqtt_message

    @staticmethod
    def format_motorcycle_frame_message(
        object_name: str,
        frame_report: Dict[str, Any],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化夜间红外摩托车飙车帧级别追踪MQTT消息（同一帧的多个飙车目标在一条消息中）

        Args:
            object_name: 检测图片存储对象名
            frame_report: 帧级别的飙车追踪报告，包含:
                - timestamp: 时间戳
                - gathering_count: 聚集数量
                - racing_count: 飙车数量（速度达到阈值）
                - tracking_infos: 追踪信息列表（多个飙车目标）
                - has_new_reports: 是否有新的上报目标
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 摩托车飙车帧级别追踪MQTT消息
        """
        tracking_infos = frame_report['tracking_infos']
        gathering_count = frame_report['gathering_count']
        racing_count = frame_report['racing_count']

        # 构建所有目标框信息
        boxes_data = []
        for tracking_info in tracking_infos:
            track_id = tracking_info['track_id']
            box_info = tracking_info['box']
            speed_info = tracking_info.get('speed')

            # 构建单个目标框信息
            box_data = {
                "x": box_info['x'],
                "y": box_info['y'],
                "width": box_info['width'],
                "height": box_info['height'],
                "score": box_info['score'],
                "track_id": track_id,
                "className": box_info.get('class', 'motorcycle')
            }

            # 所有飙车目标都包含速度信息
            if speed_info:
                box_data["speed_pixels_per_second"] = speed_info['pixels_per_second']
                box_data["direction_angle"] = speed_info['angle_degrees']
                box_data["direction"] = speed_info['direction']

            boxes_data.append(box_data)

        # 构建MQTT消息
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = len(boxes_data)
        mqtt_message["imageInfo"]["boxs"] = boxes_data  # 多个飙车目标框
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 构建消息内容（飙车统计信息）
        message_parts = []
        message_parts.append(f"检测到夜间摩托车飙车")
        message_parts.append(f"聚集数量:{gathering_count}")
        message_parts.append(f"飙车数量:{racing_count}")

        # 计算平均速度
        if tracking_infos:
            speeds = [info['speed']['pixels_per_second'] for info in tracking_infos if info.get('speed')]
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                message_parts.append(f"平均速度:{avg_speed:.1f}像素/秒")

        mqtt_message["imageInfo"]["message"] = ", ".join(message_parts)

        return mqtt_message

    @staticmethod
    def format_motorcycle_tracking_mode_message(
        object_name: str,
        tracking_info: Dict[str, Any],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化摩托车轨迹追踪模式MQTT消息（单目标轨迹追踪上报）

        Args:
            object_name: 检测图片存储对象名
            tracking_info: 追踪信息，包含:
                - track_id: 跟踪ID
                - box: 目标框信息
                - speed: 速度信息
                - tracking_state: 追踪状态 (tracking/lost)
                - elapsed_time: 已追踪时间
                - trajectory_points: 轨迹点列表
                - inference_count: 推理次数
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 摩托车轨迹追踪模式MQTT消息
        """
        track_id = tracking_info['track_id']
        box_info = tracking_info['box']
        speed_info = tracking_info.get('speed', {})
        tracking_state = tracking_info.get('tracking_state', 'unknown')
        elapsed_time = tracking_info.get('elapsed_time', 0.0)

        # 构建单个目标框信息
        box_data = {
            "x": box_info['x'],
            "y": box_info['y'],
            "width": box_info['width'],
            "height": box_info['height'],
            "score": box_info['score'],
            "track_id": track_id,
            "className": box_info.get('class', 'motorcycle')
        }

        # 添加速度信息
        if speed_info:
            box_data["speed_pixels_per_second"] = speed_info.get('pixels_per_second', 0)
            box_data["direction_angle"] = speed_info.get('angle_degrees', 0)
            box_data["direction"] = speed_info.get('direction', 'unknown')

        # 构建MQTT消息
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = 1
        mqtt_message["imageInfo"]["boxs"] = [box_data]
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 构建消息内容
        state_text = "追踪中" if tracking_state == "tracking" else "目标丢失"
        speed_text = f"{speed_info.get('pixels_per_second', 0):.1f}像素/秒" if speed_info else "未知"
        direction_text = speed_info.get('direction', '未知') if speed_info else "未知"

        message_parts = [
            f"摩托车轨迹追踪模式",
            f"追踪状态:{state_text}",
            f"track_id:{track_id}",
            f"已追踪时间:{elapsed_time:.1f}秒",
            f"速度:{speed_text}",
            f"方向:{direction_text}"
        ]

        mqtt_message["imageInfo"]["message"] = ", ".join(message_parts)

        return mqtt_message

    @staticmethod
    def format_tracking_failure_message(
        track_id: str,
        elapsed_time: float,
        task_id: str,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        格式化追踪失败MQTT消息

        Args:
            track_id: 跟踪ID
            elapsed_time: 追踪持续时间（秒）
            task_id: 任务ID
            timestamp_str: 时间戳字符串

        Returns:
            dict: 追踪失败MQTT消息
        """
        # 构建MQTT消息
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = ""  # 追踪失败时没有图片
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = 0
        mqtt_message["imageInfo"]["boxs"] = []
        mqtt_message["imageInfo"]["imageWidth"] = 0
        mqtt_message["imageInfo"]["imageHeight"] = 0
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 添加追踪失败信息
        mqtt_message["imageInfo"]["track_id"] = track_id
        mqtt_message["imageInfo"]["tracking_status"] = "failed"
        mqtt_message["imageInfo"]["elapsed_time"] = round(elapsed_time, 2)
        mqtt_message["imageInfo"]["message"] = f"追踪失败, track_id:{track_id}, 追踪持续时间:{elapsed_time:.2f}秒"

        return mqtt_message

    @staticmethod
    def format_gathering_message(
        object_name: str,
        detections: List[Dict[str, Any]],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str,
        gathering_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        格式化人群聚集检测MQTT消息 - 模型9专用（支持状态机模式）

        Args:
            object_name: 检测图片存储对象名
            detections: 检测结果列表（所有行人和人群）
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串
            gathering_info: 人群聚集统计信息，包含:
                - pedestrian_count: 行人数量
                - people_count: 人群数量
                - total_count: 总数量
                - threshold: 聚集阈值
                - is_gathering: 是否聚集
                - event_type: 事件类型 (gathering_start/gathering_update/gathering_end)
                - elapsed_time: 已持续时长（秒）
                - duration: 总持续时长（秒，仅gathering_end）
                - peak_count: 峰值人数
                - total_updates: 更新次数

        Returns:
            dict: 人群聚集检测MQTT消息
        """
        # 从第一个检测框中获取整体包围盒
        bounding_box = detections[0].get('bounding_box', {}) if detections else {}

        # 构建目标框信息（只返回整体大框）
        boxes_data = []
        if bounding_box:
            box_info = {
                "x": bounding_box.get('x', 0),
                "y": bounding_box.get('y', 0),
                "width": bounding_box.get('width', 0),
                "height": bounding_box.get('height', 0),
                "className": bounding_box.get('className', 'gathering_area'),
                "score": 1.0  # 整体包围盒没有置信度，设置为1.0
            }
            boxes_data.append(box_info)

        # 构建MQTT消息（保持与其他模型一致的格式）
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = len(boxes_data)
        mqtt_message["imageInfo"]["boxs"] = boxes_data
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 获取事件类型和统计信息
        event_type = gathering_info.get('event_type', 'gathering_detected')
        total_count = gathering_info.get('total_count', 0)
        pedestrian_count = gathering_info.get('pedestrian_count', 0)
        people_count = gathering_info.get('people_count', 0)
        threshold = gathering_info.get('threshold', 10)
        peak_count = gathering_info.get('peak_count', total_count)
        total_updates = gathering_info.get('total_updates', 1)

        # 根据事件类型构建消息
        if event_type == 'gathering_start':
            message = f"检测到人群聚集，数量为：{total_count}"
        elif event_type == 'gathering_update':
            elapsed_time = gathering_info.get('elapsed_time', 0)
            message = f"人群聚集持续中，数量为：{total_count}，已持续：{elapsed_time}秒"
        elif event_type == 'gathering_end':
            duration = gathering_info.get('duration', 0)
            message = f"人群聚集已解散，持续时长：{duration}秒，峰值人数：{peak_count}"
        else:
            message = f"检测到人群聚集，数量为：{total_count}"

        mqtt_message["imageInfo"]["message"] = message

        # 添加人群聚集统计信息（作为扩展字段）
        mqtt_message["imageInfo"]["statistics"] = {
            "pedestrian_count": pedestrian_count,
            "people_count": people_count,
            "total_count": total_count,
            "threshold": threshold,
            "event_type": event_type,
            "peak_count": peak_count,
            "total_updates": total_updates
        }

        # 添加事件特定字段
        if event_type == 'gathering_update':
            mqtt_message["imageInfo"]["elapsed_time"] = gathering_info.get('elapsed_time', 0)
        elif event_type == 'gathering_end':
            mqtt_message["imageInfo"]["duration"] = gathering_info.get('duration', 0)

        return mqtt_message

    @staticmethod
    def format_congestion_message(
        object_name: str,
        detections: List[Dict[str, Any]],
        ori_img_shape: tuple,
        task_id: str,
        timestamp_str: str,
        congestion_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        格式化交通拥堵检测MQTT消息 - 模型10专用（支持状态机模式）

        Args:
            object_name: 检测图片存储对象名
            detections: 检测结果列表（所有车辆）
            ori_img_shape: 原始图像尺寸
            task_id: 任务ID
            timestamp_str: 时间戳字符串
            congestion_info: 交通拥堵统计信息，包含:
                - vehicle_counts: 各类车辆数量字典 {car, van, truck, bus, motor, bicycle}
                - total_count: 总数量
                - threshold: 拥堵阈值
                - is_congestion: 是否拥堵
                - event_type: 事件类型 (congestion_start/congestion_update/congestion_end)
                - elapsed_time: 已持续时长（秒）
                - duration: 总持续时长（秒，仅congestion_end）
                - peak_count: 峰值车辆数
                - total_updates: 更新次数

        Returns:
            dict: 交通拥堵检测MQTT消息
        """
        # 从第一个检测框中获取整体包围盒
        bounding_box = detections[0].get('bounding_box', {}) if detections else {}

        # 构建目标框信息（只返回整体大框）
        boxes_data = []
        if bounding_box:
            box_info = {
                "x": bounding_box.get('x', 0),
                "y": bounding_box.get('y', 0),
                "width": bounding_box.get('width', 0),
                "height": bounding_box.get('height', 0),
                "className": bounding_box.get('className', 'congestion_area'),
                "score": 1.0  # 整体包围盒没有置信度，设置为1.0
            }
            boxes_data.append(box_info)

        # 构建MQTT消息（保持与其他模型一致的格式）
        mqtt_message = {"imageInfo": {}}
        mqtt_message["imageInfo"]["imageId"] = ""
        mqtt_message["imageInfo"]["dataType"] = "url"
        mqtt_message["imageInfo"]["imageUrl"] = object_name
        mqtt_message["imageInfo"]["data"] = ""
        mqtt_message["imageInfo"]["objNum"] = len(boxes_data)
        mqtt_message["imageInfo"]["boxs"] = boxes_data
        mqtt_message["imageInfo"]["imageWidth"] = ori_img_shape[1]
        mqtt_message["imageInfo"]["imageHeight"] = ori_img_shape[0]
        mqtt_message["imageInfo"]["imageSize"] = ""
        mqtt_message["imageInfo"]["task_id"] = task_id
        mqtt_message["imageInfo"]["timestamp"] = timestamp_str

        # 获取事件类型和统计信息
        event_type = congestion_info.get('event_type', 'congestion_detected')
        total_count = congestion_info.get('total_count', 0)
        vehicle_counts = congestion_info.get('vehicle_counts', {})
        threshold = congestion_info.get('threshold', 10)
        peak_count = congestion_info.get('peak_count', total_count)
        total_updates = congestion_info.get('total_updates', 1)

        # 根据事件类型构建消息
        if event_type == 'congestion_start':
            message = f"检测到交通拥堵，车辆数为：{total_count}"
        elif event_type == 'congestion_update':
            elapsed_time = congestion_info.get('elapsed_time', 0)
            message = f"交通拥堵持续中，车辆数为：{total_count}，已持续：{elapsed_time}秒"
        elif event_type == 'congestion_end':
            duration = congestion_info.get('duration', 0)
            message = f"交通拥堵已解除，持续时长：{duration}秒，峰值车辆数：{peak_count}"
        else:
            message = f"检测到交通拥堵，车辆数为：{total_count}"

        mqtt_message["imageInfo"]["message"] = message

        # 添加交通拥堵统计信息（作为扩展字段）
        mqtt_message["imageInfo"]["statistics"] = {
            "vehicle_counts": vehicle_counts,
            "total_count": total_count,
            "threshold": threshold,
            "event_type": event_type,
            "peak_count": peak_count,
            "total_updates": total_updates
        }

        # 添加事件特定字段
        if event_type == 'congestion_update':
            mqtt_message["imageInfo"]["elapsed_time"] = congestion_info.get('elapsed_time', 0)
        elif event_type == 'congestion_end':
            mqtt_message["imageInfo"]["duration"] = congestion_info.get('duration', 0)

        return mqtt_message
