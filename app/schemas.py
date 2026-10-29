"""Pydantic v2 数据模型。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    reset: bool = False


class IngestResult(BaseModel):
    chunks: int
    files: int
    elapsed_sec: float
    reset: bool


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []
    turn_id: str | None = None   # 前端每轮 uuid，供刷新后 /chat/resume 重连


class SourceChunk(BaseModel):
    id: str
    text: str
    source: str          # 来源文件路径（相对语料根）
    section: str         # 章节路径，例：01-Qt基础 > 信号槽 > 注意事项
    score: float


class SearchResult(BaseModel):
    query: str
    k: int
    hits: list[SourceChunk]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    chat_model: str
    embed_model: str
    chunks: int
    rss_mb: float
    # 服务配置（前端状态栏展示）
    n_ctx: int = 0
    n_threads: int = 0
    n_batch: int = 0
    temperature: float = 0.0
    max_tokens: int = 0
    top_k: int = 0
    # 资源消耗（系统级，随 /health 轮询刷新）
    load1: float = 0.0
    mem_total_gb: float | None = None
    mem_avail_gb: float | None = None
    uptime_sec: float = 0.0
    # CPU 信息（启动时采一次，静态值）
    cpu_logical: int = 0       # 逻辑核（线程）数
    cpu_physical: int = 0      # 物理核数
    cpu_freq_ghz: float | None = None  # 标称主频 GHz
