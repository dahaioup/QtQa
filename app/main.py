"""FastAPI 入口：4 个接口 + 静态目录。"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import config, generate, retrieve
from .ingest import build_index
from .schemas import ChatRequest, Health, IngestRequest, IngestResult, SearchResult

app = FastAPI(title="Qt 知识库 RAG 问答系统")


@app.on_event("startup")
def _startup_warmup() -> None:
    """后台预热模型：Metal shader 编译约 150s，不预热的话用户第一问要等 3 分钟。"""
    threading.Thread(target=generate.warmup, daemon=True).start()


_START = time.time()


def _rss_mb() -> float:
    """当前进程常驻内存（MB）。"""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(ru / 1024 / 1024, 1)          # macOS: bytes
    except Exception:
        return 0.0


def _sys_mem() -> tuple[float | None, float | None]:
    """整机内存 (总, 可用) GB。macOS sysconf 拿不到可用量，用 vm_stat 兜底。"""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return None, None
    avail = None
    try:
        avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 2**30
    except Exception:
        try:
            import re
            import subprocess
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            free = sum(int(re.search(rf"Pages {k}:\s+(\d+)", out).group(1))
                       for k in ("free", "inactive", "speculative")
                       if re.search(rf"Pages {k}:\s+(\d+)", out))
            avail = free * page / 2**30
        except Exception:
            avail = None
    return round(total, 1), (round(avail, 1) if avail is not None else None)


def _cpu_info() -> tuple[int, int, float | None]:
    """CPU (逻辑核, 物理核, 标称主频 GHz)。静态值，进程内缓存一次。"""
    global _CPU_INFO
    if _CPU_INFO is not None:
        return _CPU_INFO
    logical, physical, freq = os.cpu_count() or 0, 0, None
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.logicalcpu", "hw.physicalcpu",
                              "hw.cpufrequency"], capture_output=True, text=True, timeout=2)
        vals = out.stdout.split()
        if len(vals) == 3:
            logical, physical = int(vals[0]), int(vals[1])
            freq = round(int(vals[2]) / 1e9, 2) if int(vals[2]) > 0 else None
    except Exception:
        physical = logical
    if not physical:
        physical = logical
    _CPU_INFO = (logical, physical, freq)
    return _CPU_INFO


_CPU_INFO: tuple[int, int, float | None] | None = None


@app.get("/health", response_model=Health)
def health() -> Health:
    mem_total, mem_avail = _sys_mem()
    cpu_logical, cpu_physical, cpu_freq = _cpu_info()
    try:
        load1 = round(os.getloadavg()[0], 2)
    except Exception:
        load1 = 0.0
    return Health(
        status="ok",
        chat_model=config.CHAT_MODEL_PATH.name,
        embed_model=config.EMBED_MODEL,
        chunks=retrieve.chunk_count(),
        rss_mb=_rss_mb(),
        n_ctx=config.CHAT_N_CTX,
        n_threads=config.CHAT_N_THREADS,
        n_batch=config.CHAT_N_BATCH,
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=config.CHAT_MAX_TOKENS,
        top_k=config.TOP_K,
        load1=load1,
        mem_total_gb=mem_total,
        mem_avail_gb=mem_avail,
        uptime_sec=round(time.time() - _START, 0),
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        cpu_freq_ghz=cpu_freq,
    )


@app.post("/ingest", response_model=IngestResult)
def ingest(req: IngestRequest) -> IngestResult:
    return IngestResult(**build_index(reset=req.reset))


@app.get("/search", response_model=SearchResult)
def search(q: str, k: int = config.TOP_K, source: str | None = None) -> SearchResult:
    return SearchResult(query=q, k=k, hits=retrieve.search(q, k=k, source=source))


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse_events(turn_id: str):
    """跟随会话槽：先回放已产生的事件，再跟随到 done 为止。

    生成在后台线程写槽，这里只读槽——客户端断开不影响生成。
    跟随途中槽被新一轮覆盖 → 发 interrupted 事件后结束。
    """
    sent = 0
    while True:
        state, evs, fin = generate.get_turn_state(turn_id)
        if state == "miss":
            yield 'event: interrupted\ndata: {"reason": "overwritten"}\n\n'
            return
        for ev in evs[sent:]:
            yield (f"event: {ev['event']}\n"
                   f"data: {json.dumps(ev['data'], ensure_ascii=False)}\n\n")
        sent = len(evs)
        if fin:
            return
        time.sleep(0.2)    # 轮询槽；0.2s 对 token 流（~25ms/token）足够


@app.post("/chat")
def chat(req: ChatRequest):
    """SSE 流式：事件类型 token / sources / done。

    生成由 chat_start 放进后台线程并写入会话槽，本响应只负责跟随槽：
    客户端刷新/断开不会中断生成，重连走 /chat/resume 拿完整结果。
    """
    history = [t.model_dump() for t in req.history]
    tid = generate.chat_start(req.question, history=history,
                              turn_id=req.turn_id)
    if tid is None:
        # 忙碌：不排队，直接拒绝（0.25s 内返回，绝不占用生成锁等待）
        def busy_stream():
            yield 'event: token\ndata: "服务器正在努力检索其他任务，请稍后再试。"\n\n'
            yield ('event: done\n'
                   'data: {"elapsed_sec": 0, "first_token_sec": null, "busy": true}\n\n')
        return StreamingResponse(busy_stream(), media_type="text/event-stream",
                                 headers=_SSE_HEADERS)
    return StreamingResponse(_sse_events(tid), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/cancel")
def cancel():
    """停止当前正在生成的请求（token 循环下一个 token 处停下并释放锁）。"""
    running = generate.request_cancel()
    return {"ok": True, "running": running}


@app.get("/chat/resume")
def resume(turn_id: str):
    """刷新后按 turn_id 重连：回放槽内已产生的事件，并继续跟随直到结束。

    - 槽空（服务器重启）或 id 不匹配（槽被新请求覆盖）→ 409，前端显示
      「该会话异常中断」；
    - 槽内已有完整结果（服务端生成早已结束）→ 直接回放全部事件 + done；
    - 仍在生成 → 回放后继续跟随，新事件到达即推送。
    """
    state, _, _ = generate.get_turn_state(turn_id)
    if state == "miss":
        return JSONResponse(status_code=409,
                            content={"detail": "turn not found or overwritten",
                                     "turn_id": turn_id})
    return StreamingResponse(_sse_events(turn_id),
                             media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.get("/")
def index():
    # no-cache：前端迭代频繁，强制浏览器每次回源校验，避免旧 JS 卡住新功能
    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html",
                        headers={"Cache-Control": "no-cache"})
