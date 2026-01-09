from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import json
import asyncio
import threading
from urllib.parse import parse_qs
from model_library.router.config import config_router
from model_library.router.infer import router as infer_router, cleanup_completed_tasks
from model_library.router.gpu_router import router as gpu_router
from model_library.tools import log_api_complete

# API标签元数据配置
tags_metadata = [
    {
        "name": "配置管理",
        "description": "系统配置相关接口，包括模型配置查询等",
    },
    {
        "name": "模型推理",
        "description": "AI模型推理服务，支持视频流和图像推理",
        "externalDocs": {
            "description": "模型使用说明",
            "url": "https://example.com/models-docs",
        },
    },
]

# FastAPI应用配置
app = FastAPI(
    title="南山消防智能检测系统 API",
    description="""
## 南山消防智能检测系统

这是一个基于FastAPI构建的现代化AI模型推理系统，专门为消防安全智能检测而设计。

### 主要功能

- **多模型支持**: 支持9种不同类型的消防相关AI检测模型
- **实时推理**: 支持视频流实时处理和单张图像推理
- **异步处理**: 采用异步架构，支持高并发请求
- **消息推送**: 集成MQTT，实时推送推理结果
- **对象存储**: 集成MinIO，自动存储推理结果
- **完整监控**: 提供任务状态监控和管理功能

### 支持的模型类型

1. **电梯摩托车检测** (model_index=0)
2. **消防通道占用检测** (model_index=1) - 需要像素位置参数
3. **火点检测** (model_index=2)
4. **事故检测** (model_index=3) - 包含车辆计数功能
5. **车牌识别检测** (model_index=4) - 结合OCR识别
6. **车辆检测** (model_index=5)
7. **红外行人检测** (model_index=6)
8. **夜间红外摩托车飙车检测** (model_index=8) - 聚集+速度阈值检测
9. **行人聚集检测** (model_index=9) - 检测人群聚集并报警

### 技术特点

- 🚀 **高性能**: 基于FastAPI和UVicorn，支持异步处理
- 🔧 **易部署**: 支持Docker容器化部署
- 📊 **可监控**: 完整的API日志和任务状态监控
- 🔌 **易集成**: 标准RESTful API，支持跨域访问
    """,
    version="1.0.0",
    summary="南山消防智能检测系统 - AI模型推理服务",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "API支持团队",
        "url": "https://example.com/contact",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "identifier": "MIT",
    },
    openapi_tags=tags_metadata,
    openapi_url="/openapi.json",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
)

# 自定义Request类，支持重复读取body
class RequestWithCachedBody:
    def __init__(self, request: Request, body: bytes):
        self._request = request
        self._body = body
    
    def __getattr__(self, name):
        return getattr(self._request, name)
    
    async def body(self):
        return self._body

# API日志中间件
@app.middleware("http")
async def api_log_middleware(request: Request, call_next):
    """自动记录所有API请求和响应"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = str(request.url.path)
    
    # 获取请求参数
    request_params = {}
    original_body = b""
    
    try:
        # 1. 获取查询参数
        if request.query_params:
            request_params["query"] = dict(request.query_params)
        
        # 2. 获取请求体（POST请求）
        if method in ["POST", "PUT", "PATCH"]:
            # 读取原始请求体
            original_body = await request.body()
            
            if original_body:
                content_type = request.headers.get("content-type", "")
                
                if "application/json" in content_type:
                    # JSON数据
                    try:
                        request_params["body"] = json.loads(original_body.decode())
                    except:
                        request_params["body"] = {"raw_size": len(original_body)}
                
                elif "application/x-www-form-urlencoded" in content_type:
                    # Form数据
                    try:
                        form_data = parse_qs(original_body.decode())
                        # 转换为简单字典格式
                        request_params["form"] = {k: v[0] if len(v) == 1 else v for k, v in form_data.items()}
                    except:
                        request_params["form"] = {"parse_error": True}
                
                elif "multipart/form-data" in content_type:
                    # 使用FastAPI的form()方法正确解析multipart
                    try:
                        # 重新读取请求体（因为可能已经被消费）
                        await request.body()  # 重置body读取器

                        # 使用FastAPI的form()方法解析
                        form_data = await request.form()

                        # 提取字段值
                        form_fields = {}
                        for field_name, field_value in form_data.items():
                            if hasattr(field_value, 'filename'):
                                # 文件字段
                                form_fields[field_name] = f"file:{field_value.filename}"
                            else:
                                # 普通字段
                                form_fields[field_name] = str(field_value)

                        request_params["multipart"] = {
                            "fields": form_fields,
                            "size": len(original_body)
                        }
                    except Exception as e:
                        request_params["multipart"] = {
                            "error": str(e),
                            "size": len(original_body)
                        }
                else:
                    # 其他类型，只记录大小
                    request_params["body"] = {"size": len(original_body), "content_type": content_type}
        
        # 重新构造request对象，因为body只能读取一次
        if original_body:
            request = RequestWithCachedBody(request, original_body)
    
    except Exception as e:
        request_params["parse_error"] = str(e)
    
    try:
        # 调用下一个处理器
        response = await call_next(request)
        duration = time.time() - start_time
        
        # 读取并记录响应内容
        response_data = {"status_code": response.status_code}
        
        try:
            # 读取响应体
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # 尝试解析响应内容
            if response_body:
                try:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        # JSON响应，解析并记录
                        response_json = json.loads(response_body.decode('utf-8'))
                        response_data["body"] = response_json
                    else:
                        # 非JSON响应，只记录大小
                        response_data["body_size"] = len(response_body)
                        response_data["content_type"] = content_type
                except json.JSONDecodeError:
                    # JSON解析失败，记录原始内容（截断）
                    body_text = response_body.decode('utf-8', errors='ignore')
                    response_data["body_text"] = body_text[:500] + "..." if len(body_text) > 500 else body_text
                except Exception as e:
                    response_data["body_parse_error"] = str(e)
            
            # 重新构造响应对象
            from fastapi.responses import Response as FastAPIResponse
            new_response = FastAPIResponse(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )
            
            # 记录成功的API调用
            log_api_complete(
                method=method,
                path=path,
                client_ip=client_ip,
                request_params=request_params,
                response_data=response_data,
                status_code=response.status_code,
                duration=duration
            )
            
            return new_response
            
        except Exception as response_error:
            # 响应处理失败，记录错误但仍然返回原响应
            response_data["response_read_error"] = str(response_error)
            
            log_api_complete(
                method=method,
                path=path,
                client_ip=client_ip,
                request_params=request_params,
                response_data=response_data,
                status_code=response.status_code,
                duration=duration
            )
            
            return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # 记录失败的API调用
        log_api_complete(
            method=method,
            path=path,
            client_ip=client_ip,
            request_params=request_params,
            status_code=500,
            duration=duration,
            error=str(e)
        )
        raise

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

app.include_router(config_router, prefix="/ai_model")
app.include_router(infer_router,prefix="/ai_model")
app.include_router(gpu_router,prefix="/ai_model")


# 后台自动清理任务
async def periodic_cleanup():
    """定期清理已完成的推理任务"""
    cleanup_interval = 1800  # 30分钟清理一次

    while True:
        try:
            await asyncio.sleep(cleanup_interval)

            # 执行清理
            result = await cleanup_completed_tasks()

            if result.get("status") == "succeed":
                total_cleaned = result["data"]["total_cleaned"]
                remaining = result["data"]["remaining_tasks"]
                if total_cleaned > 0:
                    print(f"[自动清理] 清理了 {total_cleaned} 个任务，剩余 {remaining} 个任务")
            else:
                print(f"[自动清理] 清理失败: {result.get('msg')}")

        except Exception as e:
            print(f"[自动清理] 异常: {str(e)}")
            # 出错时等待较短时间后重试
            await asyncio.sleep(300)  # 5分钟后重试


def start_background_cleanup():
    """启动后台清理任务"""
    async def run_cleanup():
        # 等待应用启动完成后开始清理任务
        await asyncio.sleep(60)  # 启动1分钟后开始第一次清理
        await periodic_cleanup()

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_cleanup())
        except Exception as e:
            print(f"后台清理线程异常: {e}")
        finally:
            loop.close()

    cleanup_thread = threading.Thread(target=run_in_thread, daemon=True)
    cleanup_thread.start()
    print("后台自动清理任务已启动 (每30分钟清理一次)")


# 在应用启动时启动后台清理任务
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    start_background_cleanup()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5122, access_log=False)  # 关闭默认访问日志，使用我们的中间件