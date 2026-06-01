"""
多代理智能客服系统 - FastAPI 后端
基于 LangGraph 的意图识别、多专业 Agent 协作、质量检查与人工升级
"""

import logging
import time
import threading
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入您的客服系统核心类
from multi_agent import CustomerServiceSystem

# ==================== 配置加载 ====================
dotenv.load_dotenv()

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 全局状态与并发保护 ====================
service_lock = threading.Lock()
service_system: Optional[CustomerServiceSystem] = None

stats: Dict[str, Any] = {
    "total_queries": 0,
    "total_escalations": 0,
    "last_query_time": None,
    "confidences": [],          # 最近 100 次意图置信度
    "quality_scores": []        # 最近 100 次质量评分
}

# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用启动与关闭"""
    global service_system
    logger.info("智能客服系统 API 启动中...")
    try:
        service_system = CustomerServiceSystem()
        logger.info("客服系统实例已创建，所有组件初始化完成")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise
    yield
    logger.info(f"智能客服系统 API 关闭，共处理 {stats['total_queries']} 次查询")

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="多代理智能客服系统 API",
    description="基于 LangGraph 的意图分类、多专业 Agent 协作与质量保障的智能客服",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境请替换为具体域名
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response

# ==================== Pydantic 模型 ====================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    use_history: bool = Field(True, description="是否参考对话历史")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="客服回复")
    intent: str = Field(..., description="识别出的用户意图")
    confidence: float = Field(..., ge=0.0, le=1.0, description="意图置信度")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="质检评分")
    escalated: bool = Field(..., description="是否已升级到人工")
    timestamp: str = Field(..., description="响应时间戳")

class HistoryResponse(BaseModel):
    chat_history: List[Dict[str, str]]
    count: int

class HealthResponse(BaseModel):
    status: str
    service_initialized: bool
    total_queries: int
    timestamp: str

class StatsResponse(BaseModel):
    total_queries: int
    total_escalations: int
    average_confidence: float
    average_quality_score: float
    last_query_time: Optional[str]

# ==================== 工具函数 ====================
def get_service() -> CustomerServiceSystem:
    """安全获取客服系统实例，未初始化则返回 503"""
    if service_system is None:
        raise HTTPException(status_code=503, detail="客服系统尚未初始化")
    return service_system

def update_stats(confidence: float, quality_score: float, escalated: bool):
    """线程安全地更新统计信息（调用前需持有 service_lock）"""
    stats["total_queries"] += 1
    if escalated:
        stats["total_escalations"] += 1
    stats["last_query_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    stats["confidences"].append(confidence)
    if len(stats["confidences"]) > 100:
        stats["confidences"].pop(0)
    stats["quality_scores"].append(quality_score)
    if len(stats["quality_scores"]) > 100:
        stats["quality_scores"].pop(0)

# ==================== API 端点 ====================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """系统健康检查"""
    init = service_system is not None
    queries = stats["total_queries"] if init else 0
    return HealthResponse(
        status="healthy" if init else "not initialized",
        service_initialized=init,
        total_queries=queries,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取运行统计信息"""
    with service_lock:
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0.0
        avg_qual = sum(stats["quality_scores"]) / len(stats["quality_scores"]) if stats["quality_scores"] else 0.0
        return StatsResponse(
            total_queries=stats["total_queries"],
            total_escalations=stats["total_escalations"],
            average_confidence=round(avg_conf, 2),
            average_quality_score=round(avg_qual, 2),
            last_query_time=stats["last_query_time"]
        )

@app.post("/query", response_model=QueryResponse, tags=["对话"])
async def query_service(request: QueryRequest):
    """向客服系统提问"""
    with service_lock:
        srv = get_service()
        original_history = None
        if not request.use_history:
            # 备份并清空历史（如果不需要参考历史）
            original_history = srv.current_history  # 假设系统内部维护了当前对话历史
            srv.current_history = []                # 需要您的 CustomerServiceSystem 支持此属性
        try:
            # 准备对话历史（从系统当前状态获取）
            chat_history = srv.current_history if hasattr(srv, 'current_history') else []
            result = srv.handle_message(request.question, chat_history)

            # 更新统计
            update_stats(result["confidence"], result["quality_score"], result["escalated"])

            return QueryResponse(
                answer=result["response"],
                intent=result["intent"],
                confidence=result["confidence"],
                quality_score=result["quality_score"],
                escalated=result["escalated"],
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
            )
        except Exception as e:
            logger.error(f"查询失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if original_history is not None:
                srv.current_history = original_history

@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """获取当前对话历史"""
    with service_lock:
        srv = get_service()
        if hasattr(srv, 'current_history'):
            history = srv.current_history
        else:
            # 如果没有内置历史管理，返回空
            history = []
        return HistoryResponse(
            chat_history=history,
            count=len(history)
        )

@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """清除对话历史"""
    with service_lock:
        srv = get_service()
        if hasattr(srv, 'current_history'):
            srv.current_history = []
        return {"message": "对话历史已清除", "success": True}

@app.post("/reset", tags=["系统"])
async def reset_system():
    """重置整个系统（重新初始化客服系统并清空统计）"""
    global service_system, stats
    with service_lock:
        service_system = CustomerServiceSystem()
        stats = {
            "total_queries": 0,
            "total_escalations": 0,
            "last_query_time": None,
            "confidences": [],
            "quality_scores": []
        }
        return {"message": "系统已重置", "success": True}

# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )