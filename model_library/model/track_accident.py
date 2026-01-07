from .base_model import BaseModel
from ultralytics.engine.results import Results
from datetime import datetime



class TrackAccident(BaseModel):
    def __init__(self, model_path, model_index: int = None, estimated_memory: int = 1000, device_override: str = None):
        super().__init__(model_path, model_index, estimated_memory, device_override)
        self.track_id = 0
        self.track_dict = {}

        # 分类阈值配置（8个类别）
        self.class_thresholds = {
            0: 0.5,  # 0: accident - 交通事故
            1: 0.5,  # 1: pedestrian - 行人
            2: 0.4,  # 2: motorcycle - 摩托车
            3: 0.5,  # 3: car - 汽车
            4: 0.6,  # 4: motorcycle accident - 摩托车事故
            5: 0.5,  # 5: large vehicle - 大型车辆
            6: 0.4,  # 6: Traffic Police - 交警
            7: 0.5,  # 7: police motorcycle - 警用摩托
        }

    def set_class_thresholds(self, accident_threshold=None, pedestrian_threshold=None,
                            motorcycle_threshold=None, car_threshold=None,
                            motorcycle_accident_threshold=None, large_vehicle_threshold=None,
                            traffic_police_threshold=None, police_motorcycle_threshold=None,
                            thresholds_dict=None):
        """
        设置分类别的检测阈值（支持8个类别）

        Args:
            accident_threshold: 0: 交通事故检测阈值
            pedestrian_threshold: 1: 行人检测阈值
            motorcycle_threshold: 2: 摩托车检测阈值
            car_threshold: 3: 汽车检测阈值
            motorcycle_accident_threshold: 4: 摩托车事故检测阈值
            large_vehicle_threshold: 5: 大型车辆检测阈值
            traffic_police_threshold: 6: 交警检测阈值
            police_motorcycle_threshold: 7: 警用摩托检测阈值
            thresholds_dict: 字典形式批量设置，会覆盖单独参数
        """
        # 如果提供了字典，优先使用字典
        if thresholds_dict:
            for class_id, threshold in thresholds_dict.items():
                if class_id in self.class_thresholds:
                    self.class_thresholds[class_id] = threshold
        else:
            # 使用单独参数设置
            if accident_threshold is not None:
                self.class_thresholds[0] = accident_threshold
            if pedestrian_threshold is not None:
                self.class_thresholds[1] = pedestrian_threshold
            if motorcycle_threshold is not None:
                self.class_thresholds[2] = motorcycle_threshold
            if car_threshold is not None:
                self.class_thresholds[3] = car_threshold
            if motorcycle_accident_threshold is not None:
                self.class_thresholds[4] = motorcycle_accident_threshold
            if large_vehicle_threshold is not None:
                self.class_thresholds[5] = large_vehicle_threshold
            if traffic_police_threshold is not None:
                self.class_thresholds[6] = traffic_police_threshold
            if police_motorcycle_threshold is not None:
                self.class_thresholds[7] = police_motorcycle_threshold

    
    def post_process(self,results:Results)->list:
        """OBB目标追踪来定位事故，支持分类别阈值过滤"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue
            obb = result.obb
            if not obb:
                continue
            names = result.names
            xywhr = obb.xywhr.tolist()
            cls = obb.cls.tolist()
            conf = obb.conf.tolist()
            try:
                track_id = obb.id.tolist()
            except:
                # track_id =  f"{datetime.now().timestamp()}"
                track_id =  f"unknown"

            for i,box in enumerate(xywhr):
                class_id = int(cls[i])
                confidence = conf[i]

                # 应用分类别阈值过滤
                threshold = self.class_thresholds.get(class_id, 0.5)
                if confidence < threshold:
                    continue  # 跳过低于阈值的结果

                box_params = {
                    "x":box[0],
                    "y":box[1],
                    "width":box[2],
                    "height":box[3],
                    "rotation":box[4],
                    "score":confidence,
                    "track_id":track_id[i] if isinstance(track_id, list) else track_id,
                    "classed":class_id,
                    "className":names[class_id],
                    "text":""
                }
                results_dict.append(box_params)
        return results_dict

    def post_process_accidents_only(self,results:Results)->list:
        """只返回事故类别的检测结果，用于绘制验证后的真实事故框"""
        results_dict = []
        for result in results:
            if len(result) == 0:
                continue
            obb = result.obb
            if not obb:
                continue
            names = result.names
            xywhr = obb.xywhr.tolist()
            cls = obb.cls.tolist()
            conf = obb.conf.tolist()
            try:
                track_id = obb.id.tolist()
            except:
                track_id =  f"unknown"

            for i,box in enumerate(xywhr):
                class_id = int(cls[i])
                confidence = conf[i]

                # 只处理事故类别 (class_id=0)
                if class_id != 0:
                    continue  # 跳过非事故类别

                # 应用事故类别阈值过滤
                threshold = self.class_thresholds.get(class_id, 0.5)
                if confidence < threshold:
                    continue  # 跳过低于阈值的结果

                box_params = {
                    "x":box[0],
                    "y":box[1],
                    "width":box[2],
                    "height":box[3],
                    "rotation":box[4],
                    "score":confidence,
                    "track_id":track_id[i] if isinstance(track_id, list) else track_id,
                    "classed":class_id,
                    "className":names[class_id],
                    "text":""
                }
                results_dict.append(box_params)
        return results_dict