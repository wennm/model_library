"""
绘制工具模块 - 提供各种检测框绘制功能
"""
import cv2
import numpy as np
from typing import Dict, List, Any
from ultralytics.engine.results import Results


def plot_gathering_bounding_box(result: Results, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    绘制人群聚集的整体大框（不绘制所有行人小框）

    Args:
        result: YOLO推理结果
        detections: 检测结果列表

    Returns:
        numpy.ndarray: 绘制了整体大框的图片
    """
    # 从第一个检测框中获取整体包围盒
    bounding_box = detections[0].get('bounding_box', {}) if detections else {}

    # 获取原始图片
    orig_img = result.orig_img

    # 如果有整体包围盒，绘制它
    if bounding_box:
        x = int(bounding_box.get('x', 0))
        y = int(bounding_box.get('y', 0))
        width = int(bounding_box.get('width', 0))
        height = int(bounding_box.get('height', 0))

        # 绘制绿色矩形框 (BGR格式)
        color = (0, 255, 0)  # 绿色
        thickness = 3
        cv2.rectangle(orig_img, (x, y), (x + width, y + height), color, thickness)

        # 添加标签文本
        label = f"Gathering Area (Total: {len(detections)})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # 绘制标签背景
        label_bg_x1 = x
        label_bg_y1 = y - text_size[1] - 10
        label_bg_x2 = x + text_size[0] + 10
        label_bg_y2 = y
        cv2.rectangle(orig_img, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)

        # 绘制标签文本
        text_x = x + 5
        text_y = y - 5
        cv2.putText(orig_img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return orig_img


def plot_congestion_bounding_box(result: Results, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    绘制交通拥堵的整体大框（不绘制所有车辆小框）

    Args:
        result: YOLO推理结果
        detections: 检测结果列表

    Returns:
        numpy.ndarray: 绘制了整体大框的图片
    """
    # 从第一个检测框中获取整体包围盒
    bounding_box = detections[0].get('bounding_box', {}) if detections else {}

    # 获取原始图片
    orig_img = result.orig_img

    # 如果有整体包围盒，绘制它
    if bounding_box:
        x = int(bounding_box.get('x', 0))
        y = int(bounding_box.get('y', 0))
        width = int(bounding_box.get('width', 0))
        height = int(bounding_box.get('height', 0))

        # 绘制红色矩形框 (BGR格式)
        color = (0, 0, 255)  # 红色
        thickness = 3
        cv2.rectangle(orig_img, (x, y), (x + width, y + height), color, thickness)

        # 添加标签文本
        label = f"Congestion Area (Total: {len(detections)})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # 绘制标签背景
        label_bg_x1 = x
        label_bg_y1 = y - text_size[1] - 10
        label_bg_x2 = x + text_size[0] + 10
        label_bg_y2 = y
        cv2.rectangle(orig_img, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)

        # 绘制标签文本
        text_x = x + 5
        text_y = y - 5
        cv2.putText(orig_img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return orig_img


def plot_bounding_box_with_label(
    image: np.ndarray,
    bounding_box: Dict[str, Any],
    label: str,
    color: tuple = (0, 255, 0),
    thickness: int = 3,
    font_scale: float = 1.0
) -> np.ndarray:
    """
    通用函数：在图片上绘制矩形框和标签

    Args:
        image: 输入图片（numpy数组）
        bounding_box: 包含x, y, width, height的字典
        label: 标签文本
        color: 矩形框颜色（BGR格式）
        thickness: 线条粗细
        font_scale: 字体大小

    Returns:
        numpy.ndarray: 绘制后的图片
    """
    if not bounding_box:
        return image

    x = int(bounding_box.get('x', 0))
    y = int(bounding_box.get('y', 0))
    width = int(bounding_box.get('width', 0))
    height = int(bounding_box.get('height', 0))

    # 绘制矩形框
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

    # 添加标签文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

    # 绘制标签背景
    label_bg_x1 = x
    label_bg_y1 = y - text_size[1] - 10
    label_bg_x2 = x + text_size[0] + 10
    label_bg_y2 = y
    cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)

    # 绘制标签文本
    text_x = x + 5
    text_y = y - 5
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return image
