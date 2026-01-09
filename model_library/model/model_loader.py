
from importlib import import_module
import inspect
from model_library.tools.logger import log_task_debug, log_task_error
from model_library.tools.utils import Config

class ModelLoader:

    def __init__(self):
        """初始化配置"""
        self.config = Config()

    def load_model(self, model_index: int, task_id: str | None = None):
        try:
            # 读取配置并通过 class_path 反射式加载模型类
            model_cfg = self.config.model_list[model_index]
            model_path = model_cfg['model_path']
            class_path = model_cfg.get('class_path', 'model_library.model.base_model.BaseModel')
            log_task_debug(f"加载模型文件 - 任务ID:{task_id}, 路径:{model_path}, 类:{class_path}")

            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = import_module(module_name)
                model_class = getattr(module, class_name)
            except Exception as import_err:
                log_task_error(f"模型类导入失败 - 任务ID:{task_id}, 类:{class_path}, 错误:{str(import_err)}")
                raise

            # 根据构造函数参数名自动匹配传参（优先使用配置中的同名键）
            try:
                signature = inspect.signature(model_class.__init__)
                init_kwargs = {}
                for param_name, param in signature.parameters.items():
                    if param_name == 'self':
                        continue
                    if param_name in model_cfg:
                        init_kwargs[param_name] = model_cfg[param_name]

                # 确保 model_path 总是可用
                if 'model_path' not in init_kwargs:
                    init_kwargs['model_path'] = model_path

                # 添加模型索引（用于GPU分配）
                if 'model_index' not in init_kwargs:
                    init_kwargs['model_index'] = model_index

                # 添加预估显存（从配置中读取）
                if 'estimated_memory' in model_cfg:
                    init_kwargs['estimated_memory'] = model_cfg['estimated_memory']
                elif 'estimated_memory' not in init_kwargs:
                    # 根据模型类型设置默认预估显存
                    init_kwargs['estimated_memory'] = self._get_default_memory_for_model(model_index)

                log_task_debug(f"模型加载参数 - 任务ID:{task_id}, 模型索引:{model_index}, 参数:{init_kwargs}")
                model = model_class(**init_kwargs)
            except Exception as init_err:
                log_task_error(f"模型实例化失败 - 任务ID:{task_id}, 类:{class_path}, 错误:{str(init_err)}")
                raise

            log_task_debug(f"模型加载成功 - 任务ID:{task_id}")
            return model
        except Exception as e:
            log_task_error(f"模型加载失败 - 任务ID:{task_id}, 错误:{str(e)}")
            raise

    def _get_default_memory_for_model(self, model_index: int) -> int:
        """根据模型类型返回默认的预估显存(MB)"""
        # 模型显存需求预估（MB）
        model_memory_map = {
            0: 800,   # elevator_motor - 小模型
            1: 1200,  # fire_lane_blockage - 中等模型
            2: 2000,  # fire_detect - 大模型
            3: 1500,  # accident - 中大模型
            4: 1000,  # license plate - 中等模型+OCR
            5: 900,   # car - 小模型
            6: 2500,  # infrared - 大模型+SAHI
            7: 700,   # face_detect - 小模型
            8: 2500,  # infrared_motorcycle - 夜间红外摩托车检测（大模型+SAHI+追踪）
            9: 900,   # gathering - 行人检测模型（小模型）
        }
        return model_memory_map.get(model_index, 1000)  # 默认1GB


model_loader = ModelLoader()