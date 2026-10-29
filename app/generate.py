"""拼 prompt、调 llama.cpp（Qwen2.5-0.5B Instruct Q4_K_M，纯 CPU）、流式产出。

生成约束（防幻觉）：
- 只能依据提供的资料回答，资料里没有就回答「资料中未提及」
- 每个结论后面带引用编号 [1] [2]，对应 sources
- 温度 0.2，max_tokens 160（0.5B 输出短而准）
"""
from __future__ import annotations

import threading
from typing import Generator, Iterable

from . import config, retrieve
from .schemas import SourceChunk

# 0.5B 只听得懂摘录式 prompt；1.5B 能执行多段复合指令，故升级为「组织式 + 禁编造」。
# 实测 1.5B 若放任组织语言，会编造示例代码（thread.start(new QTimer) 并非合法 Qt）
# 并输出 ```cpp 栅栏、被限长拦腰截断；摘录式又会照贴资料里的截断尾巴。
# 三条防线：只用资料信息（可归纳、禁编造代码/事实）、禁 Markdown 栅栏与元信息行、
# 未覆盖就拒答。
SYSTEM_PROMPT = (
    "你是Qt开发问答助手。请严格根据下面「资料」中的内容回答问题。\n"
    "1. 只使用资料里出现过的信息作答；可以归纳、重组、分点，但不得编造事实、"
    "数字、API 或代码示例——资料里没写过的代码一律不要写。\n"
    "2. 如果资料中没有与问题相关的内容，只回答一句：资料中未提及。\n"
    "3. 不要输出 Markdown 代码块（```）；不要复述「来源：」「第 N 页」这类标注行。\n"
    "4. 回答末尾标注引用编号，如[1][2]。先给结论，再简洁展开；用中文。"
)

_model = None
_lock = threading.Lock()
# 忙碌互斥：整个请求生命周期（检索+生成）只允许一个在跑，后来的直接拒绝
# （不排队——排队会让超时遗弃的请求继续占锁，后面全部连锁超时，实测 6 连超时）。
_busy = threading.Lock()
# 取消令牌：停止按钮或页面刷新（sendBeacon /cancel）置位，token 循环检查后
# 立即结束流。解决「刷新后无法停止」——前端刷新只断 SSE 连接，服务端生成
# 会持锁跑满全场；置位后生成在下一个 token 处停下并释放锁。
_cancel = threading.Event()


def request_cancel() -> bool:
    """置位取消标志；有请求在跑返回 True（没有则只清标志）。"""
    running = _busy.locked()
    _cancel.set()
    return running


# 服务端单槽会话缓冲：只保留最近一轮的全部事件（sources/token/done），
# 支撑「刷新后凭 turn_id 重连续流」。新请求覆盖槽 → 旧 id resume 返回 409
# = 前端显示「该会话异常中断」；服务器重启槽空 → 同样 409。
_turn_lock = threading.Lock()
_last_turn: dict = {"id": None, "events": [], "done": False}


def get_turn_state(turn_id: str) -> tuple[str, list[dict], bool]:
    """返回 (状态, 事件拷贝, done)。状态: ok | miss。"""
    with _turn_lock:
        if _last_turn["id"] != turn_id or _last_turn["id"] is None:
            return "miss", [], False
        return "ok", list(_last_turn["events"]), _last_turn["done"]


def load_model():
    """懒加载 GGUF 模型（首次调用约 1–2 秒，之后常驻内存）。"""
    global _model
    if _model is None:
        from llama_cpp import Llama
        _model = Llama(
            model_path=str(config.CHAT_MODEL_PATH),
            n_ctx=config.CHAT_N_CTX,
            n_threads=config.CHAT_N_THREADS,
            n_batch=config.CHAT_N_BATCH,       # 512 会在 Metal 上 OOM（-3），见 config 注释
            n_gpu_layers=config.CHAT_N_GPU_LAYERS,   # Metal：AMD GPU 上 4 tok/s vs CPU 1 tok/s
            temperature=config.CHAT_TEMPERATURE,
            verbose=False,
        )
    return _model


def warmup() -> None:
    """服务启动时后台调用：加载模型 + 用真实大小的假 prompt 生成几个 token。

    必须用大 prompt：Metal 只有在实际跑到大 batch 时才编译对应 shader
    （实测约 150s）。之前用「你好」这类 2-token 小 prompt 预热，用户第一问
    还是会卡在 shader 编译上。"""
    if not _busy.acquire(blocking=False):
        return                      # 已有请求在跑，跳过预热（首个请求会自己加载）
    try:
        import time as _t
        print(f"[{_t.time():.0f}] warmup: acquiring lock", flush=True)
        with _lock:                 # 必须拿锁：与 chat_stream 用的是同一个 model 实例
            print(f"[{_t.time():.0f}] warmup: lock ok, loading model", flush=True)
            model = load_model()
            # ~1000 字 ≈ 1100 token，贴近真实 RAG prompt 的 batch 形状；
            # 注意不能太大：曾用 5009 token 的假 prompt 超出 n_ctx=4096，
            # 预热每次都静默失败，用户第一问还是要等 150s shader 编译。
            big_text = "Qt 开发经验示例。" * 130
            model.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"资料：\n[1] {big_text}\n\n问题：测试"},
                ],
                max_tokens=8, temperature=0.0,
            )
            # 清掉预热留下的 KV 残留，否则首个真实请求接着用会顶爆上下文
            model._ctx.kv_cache_clear()
            model.reset()
            print(f"[{_t.time():.0f}] warmup: done", flush=True)
    except Exception as e:
        # 预热失败不阻塞启动，首个请求会自己重试加载
        print(f"[warmup failed: {type(e).__name__}: {e}]", flush=True)
    finally:
        _busy.release()


def _best_window(text: str, question: str, size: int = 120) -> str:
    """在 chunk 内选与问题 bigram 重合度最高、句界对齐的 size 字窗口。

    只给前 120 字时，答案在块中部的题（如 qss 换肤）模型看不到就直接答
    「资料中未提及」；滑窗取最相关段落，同样的 token 预算命中率更高。
    字符步长滑窗会在关键内容前切断预算：信号槽题里引导段（"type：用于指明
    信号和槽的关联方式…"）bigram 重合最高，256 字预算全花在引导段上，
    恰好在「下表为其取值及意义」处截断，枚举表（DirectConnection 等）一个字
    都没进窗口，模型只能编。改为句界对齐：从每个句子起累积到 size 生成
    候选窗，问句重合最高者胜——以引导句开头的窗口会沿句滚进表格。
    """
    import re
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return text
    qs = {q[i:i + 2] for q in re.findall(r"[一-龥]{2,}", question)
          for i in range(len(q) - 1)}
    # 按句读切分（保留句读），从每个句子起累积填充 size 字预算
    sents = re.split(r"(?<=[。；！？])", text)
    windows = []
    for i in range(len(sents)):
        win = ""
        for s in sents[i:]:
            if win and len(win) + len(s) > size:
                break
            win += s
            if len(win) >= size:
                break
        if win:
            windows.append(win)
    if not windows:  # 理论不可达（至少一个句段）
        windows = [text[:size]]
    # 指针锚定：枚举/列表类问题（哪几种、有哪些、取值…）的答案常在
    # 「下表为其取值」这类指针之后的表格里，而表格用词（关联类型）与问题
    # 用词（连接方式）不同，纯 bigram 重合选窗永远选不到它（信号槽题实测：
    # 引导段 3 分、枚举表 1-2 分）。对这类问题，把指针所在句起的窗口
    # 作为额外候选并加分，使其能胜过引导段。
    n_plain = len(windows)
    stitch_bonus: set[int] = set()
    if bool(re.search(r"哪几|几种|有哪些|取值|类型|列表|分别", question)):
        for m in re.finditer(r"(下表|如下|如下表|分别列出|列表)", text):
            anchor = text.rfind("。", 0, m.start()) + 1  # 回退到句首
            # 枚举拼接：从指针句起，把所有含 `Qt::` 标识的表行按行累加成
            # 紧凑候选，累计超 2× 预算就停（行间赘述 prose 对「有哪几种」
            # 类问题可舍）。实测真实源 BlockingQueued 在锚点后 422 字、
            # Unique 545 字，放大窗口全收要 3× 预算；表行累加只要 246 字
            # 就凑齐 5 种取值。注意 PDF 抽取文本含换行空格
            # （"BlockingQueuedCon nection"），匹配用 Qt::\s*[A-Za-z]。
            rows, acc = [], 0
            for s in re.split(r"(?<=[。；！？])", text[anchor:]):
                if re.search(r"Qt::\s*[A-Za-z]", s):
                    if rows and acc + len(s) > size * 2:
                        break
                    rows.append(s)
                    acc += len(s)
            if len(rows) >= 3:
                stitch = "".join(rows)
                if len(stitch) > 60:
                    windows.append(stitch)
                    stitch_bonus.add(len(windows) - 1)
            # 普通指针窗：预算 1.5×，从指针句开始的连续窗口
            ptr_size = min(len(text) - anchor, int(size * 1.5))
            win = ""
            for s in re.split(r"(?<=[。；！？])", text[anchor:]):
                if win and len(win) + len(s) > ptr_size:
                    break
                win += s
                if len(win) >= ptr_size:
                    break
            if win:
                windows.append(win)
    # 枚举拼接窗直接置顶：bigram 分无法衡量它——表行词汇（关联类型）与
    # 问题（连接方式）错配正是它存在的原因，实测 246 字含全 5 种取值的
    # 拼接窗 raw 只有 1 分（1+13=14）仍输给指针窗（5+11=16）。≥3 个
    # Qt:: 行紧随指针是构造性枚举证据，在多个拼接窗里选 raw 最高者，
    # 以压倒分胜出；其余拼接窗丢弃。
    chosen_stitch = None
    if stitch_bonus:
        chosen_stitch = max(
            stitch_bonus,
            key=lambda i: sum(
                1 for g in (windows[i][j:j + 2] for j in range(len(windows[i]) - 1))
                if g in qs))
    best, bs = windows[0], -1
    for idx, win in enumerate(windows):
        if idx in stitch_bonus:
            if idx != chosen_stitch:
                continue
            sc = 10 ** 6
        else:
            sc = sum(1 for g in (win[i:i + 2] for i in range(len(win) - 1)) if g in qs)
            if idx >= n_plain and sc >= 1:
                # 指针锚定窗保证性加分：引入段（"把信号连接到槽"）天然包含
                # 问题高频 bigram，实测得 8 分，枚举表本身只有 3 分，普通
                # 加分追不上。枚举意图 + 文中有「下表/如下」指针时，答案
                # 定义在指针之后，直接置顶（原始重合 ≥1 才加，避免误命中
                # 完全无关的下表）。
                sc += len(qs) + 1
        if sc > bs:
            bs, best = sc, win
    # 不加「……」尾巴：摘录式模型会把截断痕迹原样抄进答案。
    # 句累窗口仍可能在半句处停（末句超长被跳过），末尾按句读收口。
    if len(best) > size // 2:
        cut = max(best.rfind(c) for c in "。；！？")
        if cut >= size // 2:
            best = best[:cut + 1]
    return best.strip()


def build_prompt(question: str, sources: list[SourceChunk],
                 history: list[dict] | None = None) -> list[dict]:
    lines = ["资料："]
    for i, s in enumerate(sources, 1):
        # 只给前 BRIEF_CHARS 字：1.5B 在本机 prompt 评估 ~20 tok/s，
        # 全文（5×600 字）会让首 token 等 1 分钟以上。
        # 前端 sources 卡片仍展示全文，模型只看最相关窗口。
        brief = _best_window(s.text, question, config.BRIEF_CHARS)
        lines.append(f"[{i}] 来源：{s.source}（{s.section}）\n{brief}")
    lines.append(f"\n问题：{question}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(lines)},
    ]
    # 历史只带最近 3 轮，避免吃掉上下文窗口
    if history:
        trimmed = [{"role": h["role"], "content": h["content"]}
                   for h in history[-4:] if h.get("content")]
        messages = trimmed + messages
    return messages


def _grounding(answer: str, chunks: list[SourceChunk], n: int = 8) -> float:
    """答案与检索资料的 n-gram 重合度：摘录式回答会很高（~1.0），编造的为 0。

    n=8 是实测区分点；比对前先去掉所有空白——PDF 提取的文本带大量空格
    换行（如 "2.1 元 对 象 系 统"），空白会打断 n-gram 造成误拒
    （实测去空白后：幻觉 0.00 / 正常摘录 1.00）。
    """
    import re
    a = re.sub(r"\[\d+\]", "", answer)
    a = re.sub(r"\s+", "", a).strip()
    texts = [re.sub(r"\s+", "", c.text) for c in chunks]
    grams = [a[i:i + n] for i in range(len(a) - n + 1)]
    if not grams:
        return 1.0
    blob = " ".join(texts)
    return sum(1 for g in grams if g in blob) / len(grams)


# 问题关键词守卫：0.5B 常把摘录改写成概括（g≈0），此时不能仅凭 grounding 拒答，
# 要再看问题的核心词是否在资料里出现过——出现过说明资料确实沾边，放行；
# 完全不沾边（微信支付/PyQt6 之于 Qt 语料）才是真「资料中未提及」。
# 停用词过滤掉「应用/功能/机制」这类到处都出现的泛词，否则守卫失效。
_Q_STOP = {
    "怎么", "如何", "什么", "哪些", "哪个", "是", "的", "用", "了", "和", "在", "有",
    "请问", "实现", "使用", "方法", "步骤", "相关", "问题", "教程", "开发", "程序",
    "应用", "功能", "机制", "适合", "更好", "商业", "项目", "正确", "做法", "Qt",
}


def _question_terms_grounded(question: str, chunks: list[SourceChunk]) -> bool:
    """问题核心词（中文拆 2-gram 部件 + 英文词）是否至少一个出现在资料里。

    bigram 部件同样要过滤停用词：否则「应用如何接入微信支付」连续串拆出的
    「应用」会命中资料（到处都有「应用」），让微信支付题绕过拒答。
    """
    import re
    blob = re.sub(r"\s+", "", " ".join(c.text for c in chunks)).lower()
    terms = re.findall(r"[A-Za-z][A-Za-z0-9.+]*|[一-龥]{2,}", question)
    for t in terms:
        if t in _Q_STOP:
            continue
        parts = ([t[i:i + 2] for i in range(len(t) - 1)]
                 if re.match(r"^[一-龥]+$", t) else [t])
        parts = [p for p in parts if p not in _Q_STOP]
        if any(p.lower() in blob for p in parts):
            return True
    return False


def chat_start(question: str, history: list[dict] | None = None,
               k: int = config.TOP_K,
               turn_id: str | None = None) -> str | None:
    """启动一轮问答：后台线程生成，事件写入单槽 _last_turn。

    返回槽 id（turn_id 为空时自动生成，供 SSE 跟随用）；忙碌时返回 None，
    由调用方直接给客户端拒绝提示——绝不排队（排队会让超时遗弃的请求继续
    占锁，后续全部连锁超时，实测 6 连超时）；warmup 也占 _busy，启动预热
    期间的请求同样收到友好提示而不是干等。

    生成与 HTTP 响应解耦：客户端断开（刷新/关页）只断响应流，服务端
    继续把整轮结果写进槽，前端刷新后凭 turn_id 走 /chat/resume 重连
    拿完整结果。停止走 /cancel 置位 _cancel，token 循环下一 token 停。
    """
    import time as _t
    if not _busy.acquire(blocking=False):
        return None
    _cancel.clear()   # 新请求开始：清掉上一轮遗留的取消标志
    tid = turn_id or f"auto-{_t.time():.0f}-{id(question) & 0xffff:x}"
    with _turn_lock:  # 新一轮覆盖槽 → 旧 id resume 返回 miss
        _last_turn.update(id=tid, events=[], done=False)

    def _run():
        try:
            for ev in _chat_stream_impl(question, history, k):
                with _turn_lock:
                    _last_turn["events"].append(ev)
        finally:
            with _turn_lock:
                _last_turn["done"] = True
            _busy.release()

    threading.Thread(target=_run, daemon=True).start()
    return tid


def _chat_stream_impl(question: str, history: list[dict] | None = None,
                      k: int = config.TOP_K) -> Generator[dict, None, None]:
    """检索 + 流式生成。产出事件：
    {"event": "sources", "data": [SourceChunk...]}
    {"event": "token", "data": "..."}
    {"event": "done", "data": {"elapsed_sec": ..., "first_token_sec": ...}}
    """
    import time
    t0 = time.time()

    sources = retrieve.search(question, k=k)
    yield {"event": "sources", "data": [s.model_dump() for s in sources]}

    if not sources:
        yield {"event": "token", "data": "资料中未提及。"}
        yield {"event": "done", "data": {"elapsed_sec": round(time.time() - t0, 2),
                                         "first_token_sec": None}}
        return

    # 前置拒答：问题核心词在检索资料里一个都不出现（如 PyQt6/PySide6 之于
    # Qt C++ 语料），就直接拒答，不进生成。原先只在生成后 grounding 低时才查，
    # 实测 0.5B 会摘沾边内容（QList 对比）把 grounding 刷高，绕过拒答。
    # 只要问题有任何一个核心词出现在资料里就放行，误拒风险低。
    if not _question_terms_grounded(question, sources):
        yield {"event": "token", "data": "资料中未提及。"}
        yield {"event": "done", "data": {"elapsed_sec": round(time.time() - t0, 2),
                                         "first_token_sec": None, "refused": True}}
        return

    messages = build_prompt(question, sources, history)
    # 偶发自拒重试：核心词已命中资料（pre-check 通过），但 1.5B 偶尔仍自己
    # 输出「资料中未提及」（采样抖动，同题时对时拒）。首拒时一个字都还没
    # 放行，追加一句「资料是有的」重试一次；再拒就如实放行拒答。
    nudge = ("\n\n（补充说明：检索结果已命中与该问题相关的资料，资料是存在的。"
             "请直接根据资料作答，不要回答「资料中未提及」。）")

    def _self_refusal(text: str) -> bool:
        return "资料中未提及" in text.lstrip(" \n「\"'“")[:15]

    for attempt in (0, 1):
        msgs = messages if attempt == 0 else \
            messages[:-1] + [{**messages[-1],
                              "content": messages[-1]["content"] + nudge}]
        print(f"[{time.time():.0f}] chat: attempt{attempt} acquiring lock "
              f"(prompt chars={sum(len(m['content']) for m in msgs)})", flush=True)
        with _lock:
            print(f"[{time.time():.0f}] chat: lock ok, clearing KV", flush=True)
            model = load_model()
            # llm_cpp 的 create_chat_completion 不会自动清 KV cache，跨请求累积
            # 到 n_ctx 上限后 llama_decode 返回 -3（实测崩点随请求次数递增）。
            model._ctx.kv_cache_clear()
            model.reset()
            first_token_t = None
            # 流式幻觉拦截：攒够 40 字先校验 n-gram 重合度，低于阈值且问题
            # 核心词不在资料里 → 真幻觉（如把 Qt6 CMake 说成 5.4），拒答；
            # 核心词在资料里 → 改写式回答，放行（仍可经 sources 卡片溯源）。
            pending = ""       # 尚未确认放行的前缀
            checked = False
            self_refused = False
            stream: Iterable[dict] = model.create_chat_completion(
                messages=msgs,
                temperature=config.CHAT_TEMPERATURE,
                max_tokens=config.CHAT_MAX_TOKENS,
                repeat_penalty=config.CHAT_REPEAT_PENALTY,
                stream=True,
            )
            for chunk in stream:
                if _cancel.is_set():
                    # 前端停止按钮 / 页面刷新（sendBeacon）触发：立刻停下，
                    # 不把僵尸生成跑满全场（用户反馈：刷新后无法停止）
                    _cancel.clear()
                    print(f"[{time.time():.0f}] chat: cancelled at client "
                          f"request", flush=True)
                    yield {"event": "done", "data": {
                        "elapsed_sec": round(time.time() - t0, 2),
                        "first_token_sec": round(first_token_t, 2) if first_token_t else None,
                        "cancelled": True}}
                    return
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    if first_token_t is None:
                        first_token_t = time.time() - t0
                        print(f"[{time.time():.0f}] chat: first token at "
                              f"{first_token_t:.1f}s", flush=True)
                    if not checked:
                        pending += delta
                        if attempt == 0 and _self_refusal(pending):
                            self_refused = True
                            break          # 尚未放行任何字，可安全重试
                        if len(pending) >= 40:
                            if _grounding(pending, sources) < config.GROUNDING_MIN:
                                if not _question_terms_grounded(question, sources):
                                    # 编造（g≈0）且问题核心词不在资料里 → 真拒答
                                    yield {"event": "token", "data": "资料中未提及。"}
                                    yield {"event": "done", "data": {
                                        "elapsed_sec": round(time.time() - t0, 2),
                                        "first_token_sec": round(first_token_t, 2),
                                        "refused": True}}
                                    return
                            checked = True
                            yield {"event": "token", "data": pending}
                        # 未攒够前不放行，避免幻觉前缀先流出去
                    else:
                        yield {"event": "token", "data": delta}
            if self_refused:
                continue        # 进入重试；attempt 1 不再判自拒，走正常流程
            if not checked and pending:
                # 答案不足 40 字就生成完了：同样校验后放行/拒答
                if _grounding(pending, sources) < config.GROUNDING_MIN \
                        and not _question_terms_grounded(question, sources):
                    yield {"event": "token", "data": "资料中未提及。"}
                else:
                    yield {"event": "token", "data": pending}
            break               # 正常完成，跳出重试循环
    yield {"event": "done", "data": {
        "elapsed_sec": round(time.time() - t0, 2),
        "first_token_sec": round(first_token_t, 2) if first_token_t else None,
    }}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "信号槽连接方式有哪些"
    for ev in chat_stream(q):
        print(ev["event"], ev["data"] if ev["event"] != "token" else ev["data"],
              end="", flush=True)
    print()
