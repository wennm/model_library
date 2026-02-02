import base64
import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, Any
from model_library.tools.logger import log_task_debug, log_task_error

class VLMVerifier:
    """VLM 多模态验证器 - 返回置信度分数"""

    def __init__(self, config: Dict[str, Any], global_modelscope_conf: Dict[str, Any] = None):
        self.enabled = config.get('enabled', False)

        # 支持三种事故类型的prompt（输出置信度分数）
        self.prompt_normal = config.get('prompt_normal',
            "仅识别红色框选区域，区域内是否发生了车辆碰撞或交通事故？红框内一定要有两个以上的人在面对面交流，一定要有两辆车。必须只回答一个(0,1)区间的置信度分数。")
        self.prompt_motorcycle = config.get('prompt_motorcycle', self.prompt_normal)
        self.prompt_large_vehicle = config.get('prompt_large_vehicle', self.prompt_normal)
        self.current_prompt = self.prompt_normal  # 当前使用的prompt
        self.timeout = config.get('timeout', 10.0)
        self.stream = config.get('stream', True)  # 是否使用流式输出，默认开启
        self.stream_timeout = config.get('stream_timeout', 10.0)  # 流式接收超时时间
        self.total_timeout = config.get('total_timeout', 15.0)  # 硬性总超时时间
        self.default_confidence = config.get('default_confidence', 0.5)  # 超时或失败时的默认分数

        if not self.enabled:
            return

        # 初始化配置列表（优先级从高到低）
        self.configs = []
        self.current_config_index = 0
        self.client = None

        # 优先使用局部配置，否则使用全局配置
        local_ms_conf = config.get('modelscope', {})
        if global_modelscope_conf is None:
            global_modelscope_conf = {}

        # 构建配置列表（按优先级排序）
        configs = []

        # 1. 局部配置优先
        if local_ms_conf.get("api_key"):
            configs.append({
                "api_key": local_ms_conf.get("api_key"),
                "base_url": local_ms_conf.get("base_url", "http://10.1.38.201:8000/v1/"),
                "model": local_ms_conf.get("model", "traffic_accident_qwen2_5vl_32b_detail"),
                "name": "local_config"
            })

        # 2. 全局配置（modelscope1）作为备用
        if global_modelscope_conf.get("api_key"):
            configs.append({
                "api_key": global_modelscope_conf.get("api_key"),
                "base_url": global_modelscope_conf.get("base_url", "http://10.1.38.201:8000/v1/"),
                "model": global_modelscope_conf.get("model", "traffic_accident_qwen2_5vl_32b_detail"),
                "name": "global_config"
            })

        # 3. 默认配置（modelscope2）作为最后备用
        configs.append({
            "api_key": "ms-12f2520f-7ec4-4a83-b40b-6bcd2ebae367",
            "base_url": "https://api-inference.modelscope.cn/v1/",
            "model": "Qwen/Qwen2.5-VL-32B-Instruct",
            "name": "fallback_config"
        })

        self.configs = configs

        if not self.configs:
            log_task_error("VLM验证已启用但未配置任何可用的API配置")
            self.enabled = False
            return

        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
            # 初始化客户端
            self._init_client()
        except ImportError:
            log_task_error("缺少 openai 依赖，VLM验证将不可用")
            self.enabled = False

    def _check_keywords_optimized(self, text: str) -> str:
        """
        优化的关键词检查方法，解决"没有"包含"有"等冲突问题
        Returns: "positive", "negative", "unknown"
        """
        text_lower = text.lower().strip()

        # 1. 优先检查完整的冲突词汇（主要问题解决）
        full_negative_words = ["没有", "不是", "并无", "并未", "并不存在", "并没有", "并无发生", "并无事故"]
        for word in full_negative_words:
            if word in text_lower:
                return "negative"

        # 2. 再检查单字否定词（但要排除肯定词上下文）
        single_negative = ["否", "no", "未", "false"]
        for neg in single_negative:
            if neg in text_lower:
                # 确保不与肯定词形成冲突
                if not any(pos in text_lower for pos in ["有", "是", "yes"]):
                    return "negative"

        # 3. 最后检查肯定词
        positive_words = ["是", "yes", "有", "true", "correct", "发生", "确实", "的确", "确实存在"]
        for word in positive_words:
            if word in text_lower:
                return "positive"

        return "unknown"

    def _extract_confidence_score(self, text: str) -> Optional[float]:
        """
        从VLM输出中提取置信度分数
        ⭐ 严格限制：只提取0.01-0.99范围的数字，避免提取列表编号

        Args:
            text: VLM输出的文本

        Returns:
            float: 提取到的置信度分数（0.01-0.99），如果无法提取则返回None
        """
        import re

        # ⭐ 策略1：优先匹配带"置信度"关键词的数字
        keyword_patterns = [
            r'置信度[：:]\s*([0-9.]+)',  # "置信度: 0.85"
            r'置信度分数[：:]\s*([0-9.]+)',  # "置信度分数: 0.85"
            r'分数[：:]\s*([0-9.]+)',  # "分数: 0.85"
            r'置信度为\s*([0-9.]+)',  # "置信度为0.85"
        ]

        for pattern in keyword_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = float(match.group(1))
                    # 归一化到[0, 1]
                    if score > 1:
                        score = score / 100.0
                    # ⭐ 严格限制：只接受0.01-0.99范围
                    if 0.01 <= score <= 0.99:
                        return score
                except (ValueError, IndexError):
                    continue

        # ⭐ 策略2：匹配纯小数格式（0.xxxx），避免匹配列表编号
        # 只匹配 "0.xxx" 格式，不匹配 "1.0" 或 "100"
        decimal_pattern = r'\b0\.[0-9]+\b'
        numbers = re.findall(decimal_pattern, text)
        if numbers:
            try:
                score = float(numbers[0])
                # ⭐ 严格限制：只接受0.01-0.99范围
                if 0.01 <= score <= 0.99:
                    return score
            except (ValueError, IndexError):
                pass

        # 无法提取有效分数
        return None

    def _init_client(self):
        """初始化OpenAI客户端，支持配置切换"""
        if self.current_config_index >= len(self.configs):
            log_task_error("所有VLM配置都已尝试失败")
            self.client = None
            return False

        config = self.configs[self.current_config_index]
        try:
            self.client = self.OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"].rstrip("/") + "/"
            )
            log_task_debug(f"VLM初始化客户端成功，使用配置: {config['name']}")
            return True
        except Exception as e:
            log_task_error(f"VLM初始化客户端失败，配置: {config['name']}, 错误: {e}")
            return False

    def _try_next_config(self):
        """尝试下一个配置"""
        self.current_config_index += 1
        if self.current_config_index >= len(self.configs):
            log_task_error("所有VLM配置都已尝试失败")
            return False

        log_task_debug(f"VLM尝试切换到配置 #{self.current_config_index + 1}")
        return self._init_client()

    @property
    def current_config(self):
        """获取当前配置"""
        if self.current_config_index < len(self.configs):
            return self.configs[self.current_config_index]
        return None

    def _call_api_with_timeout(self, api_call_func, timeout_seconds):
        """
        强制超时的API调用包装器
        Args:
            api_call_func: API调用函数
            timeout_seconds: 超时时间（秒）
        Returns:
            response或None（如果超时）
        """
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = api_call_func()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # 线程仍在运行，说明超时了
            log_task_error(f"API调用强制超时({timeout_seconds}秒)")
            return None, TimeoutError(f"API调用超时({timeout_seconds}秒)")

        if exception[0]:
            return None, exception[0]

        return result[0], None

    def verify_accident(self, image: np.ndarray) -> float:
        """
        使用 VLM 验证事故，返回置信度分数（支持流式/非流式，带早期退出和超时降级）

        Returns:
            float: 置信度分数（0-1），1.0表示确认为事故，0.0表示不是事故
        """
        if not self.enabled:
            return 1.0  # 如果未启用，默认返回最高置信度

        if image is None or image.size == 0:
            return 0.0

        # 使用配置中的总超时时间
        total_timeout = self.total_timeout
        overall_start_time = time.time()

        # 重置配置索引，每次验证都从最优配置开始尝试
        self.current_config_index = 0
        # 确保初始化至少一个可用的客户端
        if not self._init_client():
            # 如果第一个配置初始化失败，尝试其他配置
            while self.current_config_index < len(self.configs):
                if self._try_next_config():
                    break

        # 根据配置选择流式或非流式
        if self.stream:
            return self._verify_with_stream(image, overall_start_time, total_timeout)
        else:
            return self._verify_without_stream(image, overall_start_time, total_timeout)

    def set_prompt_by_accident_type(self, accident_type: str):
        """
        根据事故类型设置使用的prompt

        Args:
            accident_type: 事故类型 ("motorcycle", "large_vehicle", "normal")
        """
        if accident_type == "motorcycle":
            self.current_prompt = self.prompt_motorcycle
        elif accident_type == "large_vehicle":
            self.current_prompt = self.prompt_large_vehicle
        else:  # normal
            self.current_prompt = self.prompt_normal

    def verify_accident_with_type(self, image: np.ndarray, accident_type: str) -> float:
        """
        使用指定的事故类型验证事故（自动选择prompt）

        Args:
            image: 图像数组
            accident_type: 事故类型 ("motorcycle", "large_vehicle", "normal")

        Returns:
            float: 置信度分数（0-1），1.0表示确认为事故，0.0表示不是事故
        """
        # 设置对应的prompt
        self.set_prompt_by_accident_type(accident_type)
        # 执行验证
        return self.verify_accident(image)

    def _verify_with_stream(self, image: np.ndarray, overall_start_time: float, total_timeout: float) -> float:
        """流式验证（支持早期退出和配置切换），返回置信度分数"""
        start_time = time.time()

        # 图像编码（只需要编码一次）
        # ⭐ 直接使用BGR格式编码，不需要转换为RGB（cv2.imencode期望BGR格式）
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            log_task_error("VLM验证: 图像编码失败")
            return 0.0

        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64_image}"

        # 尝试所有可用配置
        while self.current_config_index < len(self.configs):
            # 检查总超时时间
            overall_elapsed = time.time() - overall_start_time
            if overall_elapsed > total_timeout:
                log_task_error(f"VLM验证总超时({overall_elapsed:.1f}秒)，返回默认分数: {self.default_confidence}")
                return self.default_confidence
            config = self.current_config
            if not config or not self.client:
                if not self._try_next_config():
                    break
                continue

            collected_content = []
            config_start_time = time.time()

            try:
                # 使用强制超时的API调用
                def make_api_call():
                    return self.client.chat.completions.create(
                        model=config["model"],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.current_prompt},
                                    {"type": "image_url", "image_url": {"url": image_url}}
                                ]
                            }
                        ],
                        stream=True,  # 开启流式输出
                        timeout=self.timeout,
                        temperature=0.1
                    )

                log_task_debug(f"正在调用VLM模型流式验证: {config['model']}, 配置: {config['name']}")
                response, api_error = self._call_api_with_timeout(make_api_call, self.timeout)

                if api_error:
                    raise api_error

                # API连接成功，开始流式接收
                log_task_debug(f"VLM API连接成功，开始流式接收: {config['name']}")

                # 逐块接收响应
                for chunk in response:
                    # 检查当前配置的超时（使用每个配置的独立超时时间）
                    config_elapsed = time.time() - config_start_time
                    if config_elapsed > self.stream_timeout:
                        log_task_error(f"VLM流式接收超时({config_elapsed:.1f}秒)，尝试下一个配置")
                        break  # 跳出循环，尝试下一个配置

                    # 提取内容
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        delta = chunk.choices[0].delta.content
                        if delta:
                            collected_content.append(delta)

                            # 拼接已收到的内容
                            partial = ''.join(collected_content).strip()

                            # 早期退出：尝试提取置信度分数
                            if len(partial) > 3:
                                confidence = self._extract_confidence_score(partial)
                                if confidence is not None:
                                    elapsed = time.time() - start_time
                                    log_task_debug(f"VLM提前提取到置信度分数(耗时{elapsed:.2f}秒, 配置: {config['name']}): {confidence} (原始文本: {partial})")
                                    return confidence

                # 流式接收完成，使用完整内容提取分数
                content = ''.join(collected_content).strip()
                elapsed = time.time() - start_time
                log_task_debug(f"VLM验证响应成功(耗时{elapsed:.2f}秒, 配置: {config['name']}): {content}")

                # 尝试提取置信度分数
                confidence = self._extract_confidence_score(content)
                if confidence is not None:
                    log_task_debug(f"VLM成功提取置信度分数: {confidence}")
                    return confidence
                else:
                    # ⭐ 如果无法提取分数，直接返回默认分数（不使用关键词检查）
                    log_task_debug(f"VLM无法提取置信度分数，返回默认分数: {self.default_confidence}")
                    return self.default_confidence

            except Exception as e:
                elapsed = time.time() - config_start_time
                error_type = type(e).__name__

                # 区分API连接失败和流式接收失败
                if "Timeout" in error_type or "Connection" in error_type or "502" in str(e) or "503" in str(e) or "504" in str(e):
                    log_task_error(f"VLM API连接失败(耗时{elapsed:.2f}秒, 配置: {config['name']}, {error_type}): {e}")
                    # API连接失败，立即切换配置
                    log_task_debug(f"API连接失败，立即切换到下一个配置")
                    if not self._try_next_config():
                        break
                    continue
                else:
                    log_task_error(f"VLM流式接收失败(耗时{elapsed:.2f}秒, 配置: {config['name']}, {error_type}): {e}")
                    # 尝试下一个配置
                    if not self._try_next_config():
                        break
                    continue

        # 所有配置都尝试失败，返回默认分数
        log_task_error(f"所有VLM配置都失败，返回默认分数: {self.default_confidence}")
        return self.default_confidence
    
    def _verify_without_stream(self, image: np.ndarray, overall_start_time: float, total_timeout: float) -> float:
        """非流式验证（支持配置切换），返回置信度分数"""
        start_time = time.time()

        # 图像编码（只需要编码一次）
        # ⭐ 直接使用BGR格式编码，不需要转换为RGB（cv2.imencode期望BGR格式）
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            log_task_error("VLM验证: 图像编码失败")
            return 0.0

        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64_image}"

        # 尝试所有可用配置
        while self.current_config_index < len(self.configs):
            # 检查总超时时间
            overall_elapsed = time.time() - overall_start_time
            if overall_elapsed > total_timeout:
                log_task_error(f"VLM验证总超时({overall_elapsed:.1f}秒)，返回默认分数: {self.default_confidence}")
                return self.default_confidence
            config = self.current_config
            if not config or not self.client:
                if not self._try_next_config():
                    break
                continue

            config_start_time = time.time()

            try:
                # 使用强制超时的API调用
                def make_api_call():
                    return self.client.chat.completions.create(
                        model=config["model"],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.current_prompt},
                                    {"type": "image_url", "image_url": {"url": image_url}}
                                ]
                            }
                        ],
                        stream=False,  # 关闭流式输出
                        timeout=self.timeout,
                        temperature=0.1
                    )

                log_task_debug(f"正在调用VLM模型非流式验证: {config['model']}, 配置: {config['name']}")
                response, api_error = self._call_api_with_timeout(make_api_call, self.timeout)

                if api_error:
                    raise api_error

                # API连接成功，继续处理响应

                if not response.choices or not response.choices[0].message:
                    log_task_error("VLM响应为空")
                    raise Exception("VLM响应为空")

                content = response.choices[0].message.content.strip()
                elapsed = time.time() - start_time
                log_task_debug(f"VLM验证响应成功(耗时{elapsed:.2f}秒, 配置: {config['name']}): {content}")

                # 尝试提取置信度分数
                confidence = self._extract_confidence_score(content)
                if confidence is not None:
                    log_task_debug(f"VLM成功提取置信度分数: {confidence}")
                    return confidence
                else:
                    # ⭐ 如果无法提取分数，直接返回默认分数（不使用关键词检查）
                    log_task_debug(f"VLM无法提取置信度分数，返回默认分数: {self.default_confidence}")
                    return self.default_confidence

            except Exception as e:
                elapsed = time.time() - config_start_time
                error_type = type(e).__name__

                # 区分API连接失败和响应处理失败
                if "Timeout" in error_type or "Connection" in error_type or "502" in str(e) or "503" in str(e) or "504" in str(e):
                    log_task_error(f"VLM API连接失败(耗时{elapsed:.2f}秒, 配置: {config['name']}, {error_type}): {e}")
                    # API连接失败，立即切换配置
                    log_task_debug(f"API连接失败，立即切换到下一个配置")
                    if not self._try_next_config():
                        break
                    continue
                else:
                    log_task_error(f"VLM响应处理失败(耗时{elapsed:.2f}秒, 配置: {config['name']}, {error_type}): {e}")
                    # 尝试下一个配置
                    if not self._try_next_config():
                        break
                    continue

        # 所有配置都尝试失败，返回默认分数
        log_task_error(f"所有VLM配置都失败，返回默认分数: {self.default_confidence}")
        return self.default_confidence

