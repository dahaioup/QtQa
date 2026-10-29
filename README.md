# Qt 开发经验知识库 · RAG 问答系统（本机 CPU 版）

一个可内部使用的 Qt 领域 RAG 问答系统：浏览器问一个 Qt 问题 → 逐字流式回答 → 每个结论带引用编号 → 点开引用卡片看原文 → 资料里没有的会回答「资料中未提及」。

## 架构

```
┌────────────┐  POST /chat (SSE)   ┌──────────────────────────────────────┐
│ 浏览器      │ ──────────────────► │ FastAPI (uvicorn)                    │
│ index.html │ ◄────────────────── │  retrieve.py → 向量+BM25 混合检索      │
│ 单文件      │  token/sources/done │  generate.py → 防幻觉 prompt + 流式    │
└────────────┘                     │  + 答案溯源校验（n-gram 拦截幻觉）      │
                                   └───────┬──────────────────┬───────────┘
                                           │                  │
                              ┌────────────▼─────┐   ┌────────▼─────────┐
                              │ Chroma           │   │ llama.cpp        │
                              │ PersistentClient │   │ Qwen2.5-1.5B     │
                              │ ./data/chroma    │   │ Instruct Q4_K_M  │
                              │ 8000 块 / 768 维  │   │ 纯 CPU / 6 线程   │
                              └────────────▲─────┘   └──────────────────┘
                                           │
                              ┌────────────┴─────────────┐
                              │ bge-base-zh-v1.5（768维） │
                              │ 本地加载，查询 ~66ms       │
                              └──────────────────────────┘
```

数据流：`corpus/` 语料 → `ingest.sh` 切块（400–600 字，15% 重叠）→ bge 编码 → Chroma 持久化。
问答：问题 → 混合检索 top-5 → 拼装防幻觉 prompt → llama.cpp 流式生成 → n-gram 溯源校验 → SSE 推给浏览器。

## 功能

- **流式问答**：SSE 逐字输出，首字/总耗时实时展示
- **引用溯源**：每个结论带 `[1] [2]` 编号，前端可展开引用卡片（来源路径/章节/相关度/原文）
- **混合检索**：bge 向量余弦（0.65）+ BM25 关键词（0.35），词面召回通道补专有名词（QThread 等）
- **防幻觉**：摘录式 prompt + 生成后 n-gram 溯源校验 + 问题核心词两级判定，资料外问题一律答「资料中未提及」
- **会话续传**：turn-id 会话槽，刷新/断线自动重连续流；服务端可取消生成
- **历史持久化**：最近 20 轮本地保存，刷新恢复；一键清空
- **服务状态面板**：模型/参数/索引块数/内存/负载/CPU 实时展示
- **语料处理**：Markdown + PDF 文本层提取（页码入元数据），近重复块双向去重，轮询配额防大文件挤占
- **评测脚本**：20 题（18 正确 + 2 拒答），报告检索命中率/拒答率/延迟分位

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.12、FastAPI、uvicorn、Pydantic v2 |
| 检索 | Chroma（HNSW 余弦）、bge-base-zh-v1.5、jieba 分词 + BM25 |
| 生成 | llama-cpp-python、Qwen2.5-1.5B-Instruct Q4_K_M（GGUF，纯 CPU） |
| 前端 | 原生 HTML/JS（单文件，fetch 流式解析 SSE） |
| 语料 | pypdf（PDF 文本层）、自研标题栈切块器 |

## 快速开始

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 模型文件放置：models/qwen2.5-1.5b-instruct-q4_k_m.gguf、models/bge-base-zh-v1.5/
./ingest.sh     # 建索引（首次约 100 分钟）
./run.sh        # 启动服务，打开 http://127.0.0.1:8000
python3 eval.py # 跑评测
```
