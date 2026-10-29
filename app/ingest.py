"""克隆 / 解析 / 切分 / 入库。

切分规则（按文档要求）：
- 优先按 Markdown 标题层级切，章节路径写进 metadata（例：01-Qt基础 > 信号槽 > 注意事项）
- 单块 400–600 个中文字符，重叠 15%
- 表格和代码块不被切断，单独成块
- 入库总量上限 3000 块，超过时按文件大小降序取
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from . import config


# ---------- 语料获取 ----------

def ensure_corpus() -> Path:
    """浅克隆语料仓库；已存在则跳过。"""
    if (config.CORPUS_DIR / "README.md").exists():
        return config.CORPUS_DIR
    config.CORPUS_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://gitee.com/feiyangqingyun/qtkaifajingyan",
         str(config.CORPUS_DIR)],
        check=True,
    )
    return config.CORPUS_DIR


# ---------- 文本解析 ----------

class _HTMLTextExtractor(HTMLParser):
    """把 .html 当纯文本读：去标签，保段落换行。"""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def _read_pdf(path: Path) -> str:
    """提取 PDF 文本层。扫描版（无文字层）返回近空串，会被上游过滤掉。

    页码转成「第 N 页」标题行，让 PDF 块也能带上章节定位信息。
    """
    from pypdf import PdfReader
    parts: list[str] = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if len(text) < 20:      # 封面/空白页/扫描页
                continue
            parts.append(f"# 第 {i} 页\n{text}")
    except Exception as exc:        # 损坏的 PDF 不让整个入库失败
        print(f"[ingest] PDF 解析失败 {path.name}: {exc}")
    return "\n\n".join(parts)


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    raw = path.read_bytes()
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return ""
    if path.suffix.lower() in (".html", ".htm"):
        extractor = _HTMLTextExtractor()
        extractor.feed(text)
        text = "".join(extractor.parts)
    return text


# ---------- 切分 ----------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^```")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Chunk:
    text: str
    source: str      # 相对语料根的路径
    section: str     # 章节路径
    order: int


def _split_code_blocks(section_lines: list[str]) -> list[tuple[str, list[str]]]:
    """把一段内容切成 (kind, lines)，kind ∈ {code, text}；代码块/表格不与其他内容合并。"""
    pieces: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    in_code = False
    code_buf: list[str] = []
    table_buf: list[str] = []

    def flush_text():
        nonlocal buf
        if any(l.strip() for l in buf):
            pieces.append(("text", buf))
        buf = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            pieces.append(("table", table_buf))
            table_buf = []

    for line in section_lines:
        if in_code:
            code_buf.append(line)
            if CODE_FENCE_RE.match(line.strip()):
                pieces.append(("code", code_buf))
                code_buf, in_code = [], False
            continue
        if CODE_FENCE_RE.match(line.strip()):
            flush_text()
            flush_table()
            code_buf, in_code = [line], True
            continue
        if TABLE_ROW_RE.match(line):
            flush_text()
            table_buf.append(line)
            continue
        flush_table()
        buf.append(line)
    if in_code and code_buf:
        pieces.append(("code", code_buf))
    flush_text()
    flush_table()
    return pieces


def _pack(piece_lines: list[str], min_chars: int, max_chars: int,
          overlap: int) -> list[str]:
    """把连续文本行打包成 400–600 字的块；超长块按句子边界滑窗，尾部 15% 重叠。"""
    text = "\n".join(piece_lines).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars // 4 else [text]

    # 按中文句号/换行切句子再聚包，尽量不切断语义
    sentences = re.split(r"(?<=[。！？；\n])", text)
    chunks: list[str] = []
    buf = ""
    for sen in sentences:
        if len(buf) + len(sen) > max_chars and len(buf) >= min_chars:
            chunks.append(buf.strip())
            tail = buf[-int(len(buf) * overlap):]
            buf = tail + sen
        else:
            buf += sen
    if buf.strip():
        if chunks and len(buf) < min_chars:
            chunks[-1] += buf   # 尾巴并回上一块，避免碎块
        else:
            chunks.append(buf.strip())
    return chunks


def parse_file(path: Path, rel: str) -> list[Chunk]:
    """按 Markdown 标题层级切；非 md 文件整体按滑窗切。"""
    text = _read_text(path)
    if not text.strip():
        return []

    lines = text.splitlines()
    chunks: list[Chunk] = []
    order = 0

    # PDF 里注入了「# 第 N 页」标题行，走同一套按标题切分的逻辑，
    # 块的 section 就能带页码定位
    if path.suffix.lower() in (".md", ".pdf"):
        # 按标题切段，维护章节栈：遇到同级或上级标题先弹出再压入
        stack: list[tuple[int, str]] = []   # (标题级别, 标题文字)
        cur_lines: list[str] = []

        def emit():
            nonlocal order, cur_lines
            section = " > ".join(t for _, t in stack) or rel
            for piece_kind, piece_lines in _split_code_blocks(cur_lines):
                for c in _pack(piece_lines, config.CHUNK_MIN_CHARS,
                               config.CHUNK_MAX_CHARS,
                               int(config.CHUNK_MAX_CHARS * config.CHUNK_OVERLAP)):
                    # 代码/表格单独成块时把种类标进前缀，检索时更明确
                    body = c if piece_kind == "text" else f"[{piece_kind}] {c}"
                    chunks.append(Chunk(body, rel, section, order))
                    order += 1
            cur_lines = []

        for line in lines:
            m = HEADING_RE.match(line)
            if m:
                emit()
                level = len(m.group(1))
                title = m.group(2).strip()
                stack = [s for s in stack if s[0] < level] + [(level, title)]
            else:
                cur_lines.append(line)
        emit()
    else:
        for c in _pack(lines, config.CHUNK_MIN_CHARS, config.CHUNK_MAX_CHARS,
                       int(config.CHUNK_MAX_CHARS * config.CHUNK_OVERLAP)):
            chunks.append(Chunk(c, rel, rel, order))
            order += 1
    return chunks


def collect_chunks(root: Path) -> list[Chunk]:
    """收集所有文本文件并切块；总量超限时按文件大小降序保留。"""
    files: list[tuple[int, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix.lower() not in config.TEXT_SUFFIXES:
            continue
        files.append((p.stat().st_size, p))
    files.sort(key=lambda t: -t[0])   # 大文件优先

    # 轮转分配：一个 871 页的 PDF 单文件就能产出上万块，若按「大文件优先、
    # 满额即停」会被它独占额度，其他来源（含 README）全被挤掉。改为每个
    # 文件都解析、逐文件轮转取块，直到总量到上限——每个来源都有份额。
    per_file: list[list[Chunk]] = []
    seen: set[str] = set()
    for size, p in files:
        kept: list[Chunk] = []
        for c in parse_file(p, str(p.relative_to(root))):
            # 同文块去重（README 里存在大段重复内容）
            if c.text in seen:
                continue
            seen.add(c.text)
            kept.append(c)
        if kept:
            per_file.append(kept)

    chunks: list[Chunk] = []
    idx = 0
    while len(chunks) < config.MAX_CHUNKS:
        added = False
        for kept in per_file:
            if idx < len(kept):
                chunks.append(kept[idx])
                added = True
                if len(chunks) >= config.MAX_CHUNKS:
                    break
        if not added:
            break
        idx += 1
    return chunks[: config.MAX_CHUNKS]


# ---------- 向量化入库 ----------

def build_index(reset: bool = False) -> dict:
    t0 = time.time()
    root = ensure_corpus()

    import chromadb
    from sentence_transformers import SentenceTransformer

    if reset and config.CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(config.CHROMA_DIR)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="qt_kb", metadata={"hnsw:space": "cosine"})

    chunks = collect_chunks(root)
    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True).tolist()

    ids = [f"chunk-{i:05d}" for i in range(len(chunks))]
    metadatas = [{"source": c.source, "section": c.section, "order": c.order}
                 for c in chunks]
    # 分批写入，单批 500 条，避开 chroma 的批量上限
    for start in range(0, len(ids), 500):
        end = start + 500
        collection.upsert(ids=ids[start:end], documents=texts[start:end],
                          metadatas=metadatas[start:end],
                          embeddings=vectors[start:end])

    return {
        "chunks": len(ids),
        "files": len({c.source for c in chunks}),
        "elapsed_sec": round(time.time() - t0, 2),
        "reset": reset,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_index(reset=True), ensure_ascii=False, indent=2))
