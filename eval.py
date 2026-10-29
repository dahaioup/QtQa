"""评估脚本：20 条 Qt 问题，统计检索命中率 / 首 token 延迟 / 总耗时 / 拒答率。

命中判定：top-1 检索相关度 ≥0.7 即算命中。语料加入 PDF 电子书后，正确答案
可能来自任意来源（README 或某本 PDF 的专章），不再限定文件名。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

# (问题, 期望命中的来源文件关键字[任一命中即算命中], 期望是否拒答)
CASES: list[tuple[str, list[str], bool]] = [
    ("信号槽的第五个参数有什么作用", ["README.md"], False),
    ("Qt 信号槽连接方式有哪几种", ["README.md"], False),
    ("Qt 跨线程更新 UI 的正确做法", ["README.md"], False),
    ("Qt 中 QString 和 std::string 怎么转换", ["README.md"], False),
    ("Qt 如何实现界面换肤 qss", ["README.md"], False),
    ("Qt 定时器 QTimer 怎么使用", ["README.md"], False),
    ("Qt 多线程用 QThread 还是 QtConcurrent", ["README.md"], False),
    ("Qt 内存泄漏怎么排查", ["README.md"], False),
    ("Qt 日志输出用什么", ["README.md"], False),
    ("Qt 编译报错大量错误从哪里看起", ["README.md"], False),
    ("Qt 信号槽和回调函数的区别", ["README.md"], False),
    ("Qt 如何读写配置文件", ["README.md"], False),
    ("Qt 控件自适应布局怎么做", ["README.md"], False),
    ("Qt 数据库 sqlite 怎么用", ["README.md"], False),
    ("Qt 网络通信 QNetworkAccessManager 用法", ["README.md"], False),
    ("Qt 如何打包发布 exe", ["README.md"], False),
    ("Qt Creator 好用的插件有哪些", ["README.md"], False),
    ("什么是 Qt 的元对象系统", ["README.md"], False),
    # 下面两条预期资料中没有，用于测拒答
    # 注意：拒答题要选资料完全真空的话题。若资料有「沾边」内容（如 Qt6+CMake
    # 同时出现过），0.5B 摘录式模型会照实摘录沾边片段而不拒答——这是摘录
    # 风格的固有边界，README「已知局限」有记录。
    ("Qt 应用如何接入微信支付", ["README.md"], True),
    ("PyQt6 和 PySide6 哪个更适合商业项目", ["README.md"], True),
]


def run_chat(question: str) -> tuple[list[dict], list[str], float, float]:
    """返回 (sources, tokens, first_token_sec, total_sec)。"""
    body = json.dumps({"question": question, "history": []}).encode()
    req = urllib.request.Request(BASE + "/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    sources: list[dict] = []
    tokens: list[str] = []
    first_token_sec: float | None = None
    with urllib.request.urlopen(req, timeout=300) as resp:  # 1.5B 负载波动下首 token 可达数百秒
        event = None
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "sources":
                    sources = data
                elif event == "token":
                    if first_token_sec is None:
                        first_token_sec = time.time() - t0
                    tokens.append(data)
                elif event == "done":
                    first_token_sec = data.get("first_token_sec") or first_token_sec
                event = None
    return sources, tokens, first_token_sec or 0.0, time.time() - t0


def main() -> None:
    hits = 0
    refusals = 0
    latencies_first: list[float] = []
    latencies_total: list[float] = []
    rows: list[tuple[str, str, float, float, str]] = []

    for q, want_files, expect_refuse in CASES:
        try:
            sources, tokens, first, total = run_chat(q)
        except (urllib.error.URLError, TimeoutError) as exc:
            rows.append((q, "ERROR", 0, 0, str(exc)))
            continue

        answer = "".join(tokens).strip()
        refused = "资料中未提及" in answer
        best_score = max((s["score"] for s in sources), default=0.0)
        hit = best_score >= 0.70

        if expect_refuse:
            refusals += 1 if refused else 0
        elif hit:
            hits += 1
        latencies_first.append(first)
        latencies_total.append(total)
        mark = "拒答✓" if expect_refuse and refused else (
            "命中✓" if hit and not expect_refuse else "✗")
        rows.append((q, mark, first, total, answer[:40]))

    answerable = [c for c in CASES if not c[2]]
    hit_rate = hits / len(answerable) * 100 if answerable else 0.0
    refuse_rate = refusals / max(1, len(CASES) - len(answerable)) * 100
    latencies_total.sort()
    p95 = latencies_total[int(len(latencies_total) * 0.95) - 1] if latencies_total else 0

    print(f"\n{'问题':<40}{'结果':<8}{'首token(s)':<12}{'总耗时(s)':<12}")
    print("-" * 80)
    for q, mark, first, total, _ in rows:
        print(f"{q[:38]:<40}{mark:<8}{first:<12.2f}{total:<12.2f}")
    print("-" * 80)
    print(f"检索命中率:      {hit_rate:.0f}%  ({hits}/{len(answerable)}, 目标 ≥70%)")
    print(f"拒答率(无资料题): {refuse_rate:.0f}%  ({refusals}/{len(CASES)-len(answerable)})")
    print(f"首 token 延迟:   平均 {sum(latencies_first)/max(1,len(latencies_first)):.2f}s")
    print(f"总耗时 P95:       {p95:.2f}s  (目标 <12s)")


if __name__ == "__main__":
    main()
