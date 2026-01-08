from .base_model import BaseModel
from ultralytics.engine.results import Results
from datetime import datetime



class TrackAccident(BaseModel):
    def __init__(self, model_path, model_index: int = None, estimated_memory: int = 1000, device_override: str = None):
        super().__init__(model_path, model_index, estimated_memory, device_override)
        self.track_id = 0
        self.track_dict = {}

        # 分类阈值配置（支持7类目标）
        self.class_thresholds = {
            0: 0.5,  # 0-事故类别默认阈值
            1: 0.5,  # 1-行人类别默认阈值
            2: 0.5,  # 2-摩托类别默认阈值
            3: 0.5,  # 3-汽车类别默认阈值
            4: 0.5,  # 4-大型车辆类别默认阈值
            5: 0.5,  # 5-交警类别默认阈值
            6: 0.5   # 6-警用摩托类别默认阈值
        }

    def set_class_thresholds(self, accident_threshold=None, pedestrian_threshold=None,
                           motorcycle_threshold=None, car_threshold=None,
                           large_vehicle_threshold=None, traffic_police_threshold=None,
                           police_motorcycle_threshold=None):
        """
        设置分类别的检测阈值（支持7类目标）
        Args:
            accident_threshold: 0-事故检测阈值
            pedestrian_threshold: 1-行人检测阈值
            motorcycle_threshold: 2-摩托车检测阈值
            car_threshold: 3-汽车检测阈值
            large_vehicle_threshold: 4-大型车辆检测阈值
            traffic_police_threshold: 5-交警检测阈值
            police_motorcycle_threshold: 6-警用摩托检测阈值
        """
        if accident_threshold is not None:
            self.class_thresholds[0] = accident_threshold
        if pedestrian_threshold is not None:
            self.class_thresholds[1] = pedestrian_threshold
        if motorcycle_threshold is not None:
            self.class_thresholds[2] = motorcycle_threshold
        if car_threshold is not None:
            self.class_thresholds[3] = car_threshold
        if large_vehicle_threshold is not None:
            self.class_thresholds[4] = large_vehicle_threshold
        if traffic_police_threshold is not None:
            self.class_thresholds[5] = traffic_police_threshold
        if police_motorcycle_threshold is not None:
            self.class_thresholds[6] = police_motorcycle_threshold

    
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