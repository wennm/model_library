"""
模型推理路由组
提供视频流模型推理服务，支持异步执行和MQTT结果推送
"""

import asyncio
import uuid
import json
import threading
import ast

from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Form, Path
import logging


from ..tools.detector import Detector
from ..model.model_manager import model_manager
from ..tools.reasoner import reasoner_single as reasoner
from ..tools.gpu_manager import gpu_manager  # 添加GPU管理器导入
from ..tools.logger import log_task, log_task_error, log_task_debug  # 添加日志工具

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/infer",
    tags=["模型推理"],
    responses={
        400: {"description": "请求参数错误"},
        404: {"description": "任务或资源未找到"},
        500: {"description": "服务器内部错误"},
        503: {"description": "服务暂时不可用"}
    }
)

# 存储运行中的任务
running_tasks: Dict[str, Dict[str, Any]] = {}


def run_workflow_in_thread(workflow: Detector, task_id: str):
    """在线程中运行异步workflow"""
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 运行异步workflow
        loop.run_until_complete(workflow.run_video())

        print("任务已完成")
        # 更新状态为完成
        if task_id in running_tasks:
            running_tasks[task_id]["status"] = "completed"
            running_tasks[task_id]["end_time"] = datetime.now().isoformat()
            # 保留workflow引用但标记为可清理，让垃圾回收器处理
            # running_tasks[task_id]["workflow"] = None

    except Exception as e:
        # 更新状态为失败
        print(f"{e}")
        if task_id in running_tasks:
            if workflow.is_stop_requested():
                running_tasks[task_id]["status"] = "stopped"
            else:
                running_tasks[task_id]["status"] = "failed"
                running_tasks[task_id]["error_message"] = str(e)
            running_tasks[task_id]["end_time"] = datetime.now().isoformat()
            # 清理workflow引用，释放内存
            running_tasks[task_id]["workflow"] = None
    finally:
        # 修复：确保所有资源都被正确清理
        try:
            # 1. 停止监控线程
            if hasattr(workflow, '_stop_monitor_thread'):
                workflow._stop_monitor_thread()

            # 2. 请求停止workflow
            if hasattr(workflow, 'request_stop'):
                workflow.request_stop()

            # 3. 清理GPU资源引用
            if hasattr(workflow, 'model_index') and workflow.model_index is not None:
                try:
                    gpu_manager.release_model(workflow.model_index)
                    log_task(f"已释放模型 {workflow.model_index} 的GPU引用")
                except Exception as gpu_error:
                    log_task_error(f"GPU资源清理失败: {str(gpu_error)}")

            # 4. 清理MQTT连接
            if hasattr(workflow, 'mqtt_client'):
                try:
                    workflow.mqtt_client.disconnect()
                    log_task_debug(f"MQTT连接已断开")
                except Exception as mqtt_error:
                    log_task_error(f"MQTT断开失败: {str(mqtt_error)}")

        except Exception as cleanup_error:
            log_task_error(f"资源清理异常: {str(cleanup_error)}")
        finally:
            # 删除workflow对象引用
            del workflow
            loop.close()
            log_task("任务线程已完全清理")


@router.post(
    "/video",
    summary="启动视频流推理任务",
    description="""
    启动一个异步的视频流推理任务，支持实时AI检测和结果推送。

    ## 功能特点
    - **异步处理**: 任务在后台异步执行，立即返回任务ID
    - **实时推送**: 检测结果通过MQTT实时推送到指定主题
    - **状态监控**: 支持任务状态查询和管理
    - **资源管理**: 自动清理已完成任务，防止内存泄漏

    ## 支持的视频格式
    - **RTMP流**: rtmp://server/live/stream
    - **RTSP流**: rtsp://ip:port/path
    - **本地文件**: /path/to/video.mp4
    - **HTTP流**: http://server/video.mp4

    ## 使用流程
    1. 调用此接口创建推理任务
    2. 获取返回的task_id和mqtt_topic
    3. 订阅MQTT主题接收实时结果
    4. 使用task_id查询任务状态
    5. 任务完成后调用清理接口

    ## 注意事项
    - 任务创建后会立即开始执行
    - 每个任务会占用相应的GPU/CPU资源
    - 建议定期清理已完成任务
    - 消防通道占用检测必须提供pixel_position参数
    """,
    response_description="推理任务创建成功，返回任务ID、MQTT主题等任务信息"
)
async def start_inference(
        video_path: str = Form(
            ...,
            description="视频流地址，支持多种格式：<br>"
                     "- RTMP: rtmp://server/live/stream<br>"
                     "- RTSP: rtsp://ip:port/path<br>"
                     "- 本地文件: /path/to/video.mp4<br>"
                     "- HTTP流: http://server/video.mp4",
            examples=[
                {"value": "rtmp://live.example.com/stream1", "description": "RTMP直播流"},
                {"value": "rtsp://192.168.1.100:554/stream", "description": "RTSP摄像头流"},
                {"value": "/videos/test.mp4", "description": "本地视频文件"},
                {"value": "http://example.com/video.mp4", "description": "HTTP视频流"}
            ]
        ),
        model_index: int = Form(
            ...,
            description="模型类型索引，对应不同的检测功能：<br>"
                     "**0**: 电梯摩托车检测<br>"
                     "**1**: 消防通道占用检测 *(需要pixel_position)*<br>"
                     "**2**: 火点检测<br>"
                     "**3**: 事故检测 *(自动触发车辆计数)*<br>"
                     "**4**: 车牌识别检测<br>"
                     "**5**: 车辆检测<br>"
                     "**6**: 红外行人检测<br>"
                     "**7**: 人脸提取<br>"
                     "**8**: 夜间红外摩托车飙车检测 *(聚集+速度阈值)*",
            ge=0, le=8,
            examples=[
                {"value": 0, "description": "电梯摩托车检测"},
                {"value": 1, "description": "消防通道占用检测"},
                {"value": 2, "description": "火点检测"},
                {"value": 3, "description": "事故检测"},
                {"value": 4, "description": "车牌识别检测"},
                {"value": 5, "description": "车辆检测"},
                {"value": 6, "description": "红外行人检测"},
                {"value": 7, "description": "人脸提取"}
            ]
        ),
        pixel_position: Optional[str] = Form(
            None,
            description="""
            像素位置坐标，仅消防通道占用检测(model_index=1)需要。

            格式：JSON字符串，表示多边形的顶点坐标列表。

            示例：[[0,941],[0,1342],[2152,1338],[2173,586],[1110,460]]

            说明：
            - 坐标系：左上角为原点(0,0)
            - 格式：[[x1,y1],[x2,y2],...]
            - 最少3个点构成多边形
            - 点的顺序按多边形边缘排列
            """,
            examples=[
                {
                    "value": "[[0,941],[0,1342],[2152,1338],[2173,586],[1110,460]]",
                    "description": "消防通道多边形区域"
                }
            ]
        )
):
    try:
        # 输入验证
        if not video_path or not video_path.strip():
            return {
                "status": "error",
                "code": 400,
                "msg": "视频路径不能为空",
                "data": {}
            }

        if model_index not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            return {
                "status": "error",
                "code": 400,
                "msg": "模型索引必须是 0, 1, 2, 3, 4, 5, 6, 7, 8 中的一个",
                "data": {}
            }

        # 解析pixel_position
        parsed_pixel_position = None
        if pixel_position:
            try:
                parsed_pixel_position = ast.literal_eval(pixel_position)
                if not isinstance(parsed_pixel_position, list):
                    raise ValueError("pixel_position必须是坐标列表")
            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "status": "error",
                    "code": 400,
                    "msg": f"pixel_position格式错误: {str(e)}",
                    "data": {}
                }

        # 生成唯一任务ID
        task_id = str(uuid.uuid4()).replace('-', '_')

        # 创建workflow实例
        workflow = Detector(
            model_index=model_index,
            video_path=video_path.strip(),
            pixel_position=parsed_pixel_position,
            task_id = task_id
        )

        # 获取MQTT主题和模型名称
        mqtt_topic = workflow.topic
        model_name = workflow.model_name

        # 在后台线程中启动任务
        thread = threading.Thread(target=run_workflow_in_thread, args=(workflow, task_id), daemon=True)

        # 记录任务信息（包括线程引用）
        running_tasks[task_id] = {
            "workflow": workflow,
            "thread": thread,
            "status": "running",
            "mqtt_topic": mqtt_topic,
            "model_name": model_name,
            "start_time": None,
            "error_message": None
        }

        thread.start()

        logger.info(f"创建推理任务 {task_id}, 模型: {model_name}, MQTT主题: {mqtt_topic}")

        resp = {
            "status": "succeed",
            "code": 200,
            "msg": "推理任务已创建，正在后台执行",
            "data": {
                "task_id": task_id,
                "mqtt_topic": mqtt_topic,
                "model_name": model_name,
                "task_status": "running",
                "start_time": datetime.now().isoformat()
            }
        }
        return resp
    except Exception as e:
        logger.error(f"创建推理任务失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"创建推理任务失败: {str(e)}",
            "data": {}
        }


@router.post(
    "/image",
    summary="图像推理检测",
    description="""
    对单张图片进行AI推理检测，立即返回检测结果。

    ## 功能特点
    - **同步处理**: 立即返回检测结果
    - **多格式支持**: 支持本地文件和在线图片
    - **高精度**: 基于YOLO等先进检测算法
    - **结构化输出**: 返回标准化的检测结果

    ## 支持的图片格式
    - **本地文件**: /path/to/image.jpg
    - **在线图片**: http://example.com/image.jpg
    - **常见格式**: JPG, PNG, BMP, TIFF等

    ## 返回结果格式
    - 检测框坐标 (x1, y1, x2, y2)
    - 置信度分数
    - 类别标签
    - 车牌OCR结果（车牌检测模型）

    ## 使用场景
    - 图片批量检测
    - 实时图片分析
    - 移动端上传检测
    - 第三方系统集成
    """,
    response_description="图像检测结果，包含检测到的目标列表和置信度信息"
)
async def start_inference_image(
        image_path: str = Form(
            ...,
            description="图片地址，支持多种来源：<br>"
                     "- 本地文件: /path/to/image.jpg<br>"
                     "- HTTP图片: http://example.com/image.jpg<br>"
                     "- HTTPS图片: https://example.com/image.jpg<br>"
                     "- 支持格式: JPG, PNG, BMP, TIFF等",
            examples=[
                {"value": "/images/test.jpg", "description": "本地图片文件"},
                {"value": "http://example.com/image.jpg", "description": "HTTP图片"},
                {"value": "https://example.com/image.png", "description": "HTTPS图片"}
            ]
        ),
        model_index: int = Form(
            ...,
            description="模型类型索引：<br>"
                     "**0**: 电梯摩托车检测<br>"
                     "**1**: 消防通道占用检测<br>"
                     "**2**: 火点检测<br>"
                     "**3**: 事故检测<br>"
                     "**4**: 车牌识别检测 *(含OCR)*<br>"
                     "**5**: 车辆检测<br>"
                     "**6**: 红外行人检测<br>"
                     "**7**: 人脸提取<br>"
                     "**8**: 夜间红外摩托车飙车检测 *(聚集+速度阈值)*",
            ge=0, le=8,
            examples=[
                {"value": 0, "description": "电梯摩托车检测"},
                {"value": 1, "description": "消防通道占用检测"},
                {"value": 2, "description": "火点检测"},
                {"value": 3, "description": "事故检测"},
                {"value": 4, "description": "车牌识别检测"},
                {"value": 5, "description": "车辆检测"},
                {"value": 6, "description": "红外行人检测"},
                {"value": 7, "description": "人脸提取"},
                {"value": 8, "description": "夜间红外摩托车飙车检测"}
            ]
        ),
):
    resp = {
        "status": "succeed",
        "code": 200,
        "msg": "推理成功",
        "data": {}
    }
    try:
        # 输入验证
        if not image_path or not image_path.strip():
            resp.update({
                "status": "error",
                "code": 400,
                "msg": "图片路径不能为空"
            })
            return resp

        if model_index not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            resp.update({
                "status": "error",
                "code": 400,
                "msg": "模型索引必须是 0, 1, 2, 3, 4, 5, 6, 7, 8 中的一个"
            })
            return resp

        # 使用异步推理
        result = await reasoner.infer_image(image_path.strip(), model_index)
        
        resp["data"]["item"] = result
        # resp["data"]["count"] = len(result) if result else 0
        
        if not result:
            resp.update({
                "code": 200,  # 没有检测到目标也是成功的
                "msg": "未检测到目标对象"
            })
        
        return resp
        
    except Exception as e:
        logger.error(f"图像推理失败: {str(e)}")
        resp.update({
            "msg": f"图片推理失败：{str(e)}",
            "status": "error", 
            "code": 500
        })
        return resp


@router.get(
    "/status/{task_id}",
    summary="查询推理任务状态",
    description="""
    查询指定推理任务的详细状态信息。

    ## 状态说明
    - **running**: 任务正在运行
    - **stopping**: 任务正在停止中
    - **completed**: 任务已完成
    - **failed**: 任务执行失败
    - **stopped**: 任务被手动停止

    ## 返回信息
    - 任务ID和当前状态
    - MQTT主题名称
    - 使用的模型名称
    - 开始时间
    - 错误信息（如果有）
    - 停止请求状态

    ## 使用场景
    - 监控任务执行进度
    - 检查任务是否完成
    - 调试任务失败原因
    - 确认任务停止状态
    """,
    response_description="任务状态详细信息，包括执行状态、时间信息等"
)
async def get_task_status(
    task_id: str = Path(
        ...,
        description="任务ID，创建推理任务时返回的唯一标识符",
        examples=[
            {"value": "123e4567-e89b-12d3-a456-426614174000", "description": "示例任务ID"}
        ]
    )
):
    if task_id not in running_tasks:
        return {
            "status": "error",
            "code": 404,
            "msg": "任务不存在",
            "data": {}
        }

    task_info = running_tasks[task_id]

    # 判断当前任务状态
    current_status = task_info["status"]
    workflow = task_info["workflow"]

    # 检查workflow是否为None（任务完成后会被设置为None）
    stop_requested = False
    if workflow is not None:
        stop_requested = workflow.is_stop_requested()
        if stop_requested and current_status == "running":
            current_status = "stopping"

    return {
        "status": "succeed",
        "code": 200,
        "msg": "查询成功",
        "data": {
            "task_id": task_id,
            "task_status": current_status,
            "mqtt_topic": task_info["mqtt_topic"],
            "model_name": task_info["model_name"],
            "start_time": task_info.get("start_time", ""),
            "error_message": task_info.get("error_message"),
            "stop_requested": stop_requested
        }
    }


@router.delete(
    "/stop/{task_id}",
    summary="停止推理任务",
    description="""
    停止正在运行的推理任务，释放系统资源。

    ## 功能说明
    - 发送停止信号到正在运行的任务
    - 任务会在当前处理完成后安全停止
    - 异步操作，立即返回响应
    - 自动清理相关资源

    ## 停止流程
    1. 接收停止请求
    2. 向任务发送停止信号
    3. 任务完成当前帧处理后退出
    4. 自动清理内存和线程资源

    ## 注意事项
    - 停止操作是异步的，不会立即生效
    - 已完成的任务无需停止
    - 停止后的任务状态变为"stopped"
    - 建议定期清理已停止的任务
    """,
    response_description="停止请求确认信息，包含任务ID和状态"
)
async def stop_task(
    task_id: str = Path(
        ...,
        description="要停止的任务ID",
        examples=[
            {"value": "123e4567-e89b-12d3-a456-426614174000", "description": "示例任务ID"}
        ]
    )
):
    if task_id not in running_tasks:
        return {
            "status": "error",
            "code": 404,
            "msg": "任务不存在",
            "data": {}
        }

    task_info = running_tasks[task_id]

    if task_info["status"] in ["completed", "failed", "stopped"]:
        return {
            "status": "succeed",
            "code": 200,
            "msg": f"任务已结束，状态: {task_info['status']}",
            "data": {
                "task_id": task_id,
                "task_status": task_info["status"]
            }
        }

    # 设置停止请求标志
    workflow = task_info["workflow"]
    workflow.request_stop()

    logger.info(f"发送停止请求到推理任务 {task_id}")

    return {
        "status": "succeed",
        "code": 200,
        "msg": "停止请求已发送，任务将在短时间内停止",
        "data": {
            "task_id": task_id,
            "task_status": "stopping"
        }
    }


@router.get(
    "/tasks",
    summary="获取所有任务列表",
    description="""
    获取系统中当前所有推理任务的状态信息。

    ## 返回信息
    - 每个任务的详细状态
    - 任务总数统计
    - 按状态分组的信息

    ## 使用场景
    - 系统监控和状态检查
    - 任务管理和资源监控
    - 性能分析和优化
    - 故障排查和调试

    ## 信息详情
    每个任务包含：
    - 任务ID和状态
    - MQTT主题
    - 模型名称
    - 开始时间
    - 错误信息（如有）
    """,
    response_description="所有任务的完整列表，包含详细状态信息"
)
async def list_tasks():
    tasks = []
    for task_id, task_info in running_tasks.items():
        # 判断当前任务状态
        current_status = task_info["status"]
        workflow = task_info["workflow"]
        if workflow.is_stop_requested() and current_status == "running":
            current_status = "stopping"

        tasks.append({
            "task_id": task_id,
            "task_status": current_status,
            "mqtt_topic": task_info["mqtt_topic"],
            "model_name": task_info["model_name"],
            "start_time": task_info.get("start_time", ""),
            "error_message": task_info.get("error_message"),
            "stop_requested": workflow.is_stop_requested()
        })

    return {
        "status": "succeed",
        "code": 200,
        "msg": "查询成功",
        "data": {
            "tasks": tasks,
            "total": len(tasks)
        }
    }


@router.get(
    "/models",
    summary="获取支持的模型列表",
    description="""
    获取系统支持的所有AI模型信息，包括模型说明和使用要求。

    ## 模型类型说明
    1. **电梯摩托车检测** (0) - 检测电梯内的摩托车违规停放
    2. **消防通道占用检测** (1) - 检测消防通道是否被占用，需要定义检测区域
    3. **火点检测** (2) - 检测图像中的火点，用于火灾预警
    4. **事故检测** (3) - 检测交通事故，自动统计涉事车辆数量
    5. **车牌识别检测** (4) - 检测车辆并识别车牌号码
    6. **车辆检测** (5) - 通用车辆检测
    7. **红外行人检测** (6) - 基于红外图像的行人检测

    ## 特殊要求
    - 消防通道占用检测需要提供像素位置参数
    - 事故检测会自动触发车辆计数功能
    - 车牌识别检测包含OCR文字识别

    ## 返回信息
    - 模型名称和描述
    - 是否需要特殊参数
    - 模型功能说明
    """,
    response_description="所有支持的模型详细信息列表"
)
async def get_models():
    models = {
        0: {
            "name": "elevator_motor",
            "description": "电梯摩托车检测",
            "requires_pixel_position": False
        },
        1: {
            "name": "fire_lane_blockage",
            "description": "消防通道占用检测",
            "requires_pixel_position": True
        },
        2: {
            "name": "fire_detect",
            "description": "火点检测",
            "requires_pixel_position": False
        },
        3: {
            "name": "accident",
            "description": "事故检测",
            "requires_pixel_position": False
        }
    }

    return {
        "status": "succeed",
        "code": 200,
        "msg": "查询成功",
        "data": {
            "models": models
        }
    }


@router.delete(
    "/cleanup",
    summary="清理已完成的任务",
    description="""
    清理所有已完成的推理任务，释放系统资源。

    ## 清理范围
    - **completed**: 已成功完成的任务
    - **failed**: 执行失败的任务
    - **stopped**: 被手动停止的任务

    ## 功能说明
    - 删除任务记录和相关引用
    - 释放内存和线程资源
    - 返回清理统计信息
    - 保留正在运行的任务

    ## 使用建议
    - 定期调用此接口清理资源
    - 避免内存泄漏和资源浪费
    - 监控系统运行状态
    - 适合自动化运维脚本调用

    ## 注意事项
    - 清理操作不可逆
    - 建议在系统负载较低时执行
    - 清理后任务记录将无法查询
    """,
    response_description="清理操作结果，包含清理的任务数量和剩余任务信息"
)
async def cleanup_completed_tasks():
    """
    清理已完成的推理任务，释放内存资源
    生产环境建议定期调用此接口（如每30分钟）
    """
    try:
        completed_tasks = []
        force_cleanup_tasks = []  # 强制清理的任务
        memory_freed_tasks = []  # 内存释放的任务

        # 清理配置
        cleanup_completed_age = 300  # 5分钟前完成的任务
        cleanup_failed_age = 1800    # 30分钟前失败的任务
        cleanup_running_age = 7200   # 2小时前还在运行的任务（可能是僵尸任务）

        current_time = datetime.now()

        for task_id, task_info in list(running_tasks.items()):
            task_status = task_info["status"]
            workflow = task_info.get("workflow")

            # 计算任务年龄
            task_age = None
            if task_info.get("end_time"):
                try:
                    end_time = datetime.fromisoformat(task_info["end_time"])
                    task_age = (current_time - end_time).total_seconds()
                except:
                    pass
            elif task_info.get("start_time"):
                try:
                    start_time = datetime.fromisoformat(task_info["start_time"])
                    task_age = (current_time - start_time).total_seconds()
                except:
                    pass

            # 根据状态和年龄决定是否清理
            should_cleanup = False
            cleanup_reason = ""

            if task_status in ["completed"] and task_age and task_age > cleanup_completed_age:
                should_cleanup = True
                cleanup_reason = "completed_task"
                completed_tasks.append(task_id)

            elif task_status in ["failed", "stopped"] and task_age and task_age > cleanup_failed_age:
                should_cleanup = True
                cleanup_reason = "failed_task"
                completed_tasks.append(task_id)

            elif task_status == "running" and task_age and task_age > cleanup_running_age:
                # 可能是僵尸任务，强制清理
                should_cleanup = True
                cleanup_reason = "zombie_task"
                force_cleanup_tasks.append(task_id)

                # 强制停止工作流
                if workflow:
                    try:
                        if hasattr(workflow, 'request_stop'):
                            workflow.request_stop()
                        if hasattr(workflow, '_stop_monitor_thread'):
                            workflow._stop_monitor_thread()
                    except Exception as stop_error:
                        logger.warning(f"强制停止任务失败 {task_id}: {stop_error}")

            if should_cleanup:
                # 清理workflow引用，触发垃圾回收
                if workflow:
                    try:
                        # 使用资源清理管理器进行深度清理
                        resource_cleanup_manager.cleanup_task(task_id, force=True)
                        memory_freed_tasks.append(task_id)
                    except Exception as cleanup_error:
                        logger.warning(f"资源清理失败 {task_id}: {cleanup_error}")

                # 从运行任务列表中移除
                if task_id in running_tasks:
                    del running_tasks[task_id]

        total_cleaned = len(completed_tasks) + len(force_cleanup_tasks)

        # 强制垃圾回收
        import gc
        gc.collect()

        logger.info(f"清理完成 - 总计:{total_cleaned}, 正常:{len(completed_tasks)}, 强制:{len(force_cleanup_tasks)}, 内存释放:{len(memory_freed_tasks)}")
        log_task(f"清理了 {total_cleaned} 个任务 (正常:{len(completed_tasks)}, 强制:{len(force_cleanup_tasks)})")

        return {
            "status": "succeed",
            "code": 200,
            "msg": f"成功清理 {total_cleaned} 个任务",
            "data": {
                "completed_tasks": completed_tasks,
                "force_cleanup_tasks": force_cleanup_tasks,
                "memory_freed_tasks": memory_freed_tasks,
                "total_cleaned": total_cleaned,
                "remaining_tasks": len(running_tasks),
                "cleanup_policy": {
                    "completed_after_seconds": cleanup_completed_age,
                    "failed_after_seconds": cleanup_failed_age,
                    "running_force_after_seconds": cleanup_running_age
                }
            }
        }

    except Exception as e:
        logger.error(f"清理任务失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"清理任务失败: {str(e)}",
            "data": {}
        }


@router.get(
    "/models/status",
    summary="获取模型加载状态",
    description="""
    获取当前系统中所有模型的加载状态和缓存信息。

    ## 返回信息
    - 已加载的模型列表
    - 模型总数统计
    - 模型管理器状态

    ## 使用场景
    - 系统状态监控
    - 资源使用情况检查
    - 性能分析和优化
    - 模型缓存管理

    ## 注意事项
    - 此接口提供模型缓存状态信息
    - 可以用于监控系统资源使用
    - 帮助优化模型加载策略
    """,
    response_description="模型加载状态信息"
)
async def get_models_status():
    try:
        loaded_models = model_manager.get_loaded_models()
        total_models = model_manager.get_model_count()

        return {
            "status": "succeed",
            "code": 200,
            "msg": "查询成功",
            "data": {
                "loaded_models": loaded_models,
                "total_loaded_models": total_models,
                "model_manager_status": "active"
            }
        }

    except Exception as e:
        logger.error(f"获取模型状态失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"获取模型状态失败: {str(e)}",
            "data": {}
        }


@router.delete(
    "/models/clear/{model_index}",
    summary="清除指定模型缓存",
    description="""
    清除指定模型的缓存，释放内存资源。

    ## 功能说明
    - 从内存中卸载指定模型
    - 释放模型占用的GPU/CPU资源
    - 下次使用时重新加载

    ## 参数说明
    - model_index: 模型索引（0-7）

    ## 注意事项
    - 清除后下次使用会重新加载
    - 正在使用的模型不会被立即卸载
    - 建议在系统负载低时使用
    """,
    response_description="模型缓存清除结果"
)
async def clear_model_cache(
    model_index: int = Path(
        ...,
        description="要清除缓存的模型索引",
        ge=0, le=7
    )
):
    try:
        model_manager.clear_model(model_index)

        logger.info(f"已清除模型 {model_index} 的缓存")

        return {
            "status": "succeed",
            "code": 200,
            "msg": f"模型 {model_index} 缓存已清除",
            "data": {
                "cleared_model_index": model_index,
                "remaining_models": model_manager.get_model_count()
            }
        }

    except Exception as e:
        logger.error(f"清除模型缓存失败: {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "msg": f"清除模型缓存失败: {str(e)}",
            "data": {}
        }
