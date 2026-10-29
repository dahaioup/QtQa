"""向量检索 + 元数据过滤 + BM25 混合。

混合分数 = (1 - BM25_WEIGHT) * 向量余弦 + BM25_WEIGHT * 归一化 BM25
"""
from __future__ import annotations

import re
import threading

import chromadb

from . import config
from .schemas import SourceChunk

# bge-zh 系列的查询侧指令前缀（官方约定）：检索质量提升明显
# bge-zh-v1.5 官方建议查询侧带这条指令前缀，但实测对本语料是负优化：
# 18 个可答题离线 A/B，带前缀 9/18、裸查询 17/18（top-1 相似度 ≥0.70 计命中）。
# 原因推测：指令把查询向量推入「通用检索」语义锥，反而更贴合 PDF 的通用介绍页；
# 裸查询保留问题本身的 Qt 领域语义。bge-base 下留空，bge-small 时代是需要的。
QUERY_PREFIX = ""

_client: chromadb.PersistentClient | None = None
_collection = None
_embedder = None
_lock = threading.Lock()

# 查询侧 BM25 用的小型停用词（中文单字 + 英文虚词），避免把「的/了」当特征
_STOP = set("的了是在和有与及或这那个你我他她它们不我")


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL,
                                        device=config.EMBED_DEVICE)
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            _collection = _client.get_collection("qt_kb")
        except Exception:
            _collection = None
    return _collection


def chunk_count() -> int:
    col = _get_collection()
    return col.count() if col else 0


# 词的全库 df（含该词的块数）缓存：词面召回和拒答判定都会问，量不大但同一词
# 会被反复查。df 用 limit 封顶计数即可——我们只需要「≥10 常见 / 罕见」的判别。
_df_cache: dict[str, int] = {}


def _corpus_df(col, term: str, cap: int = 20) -> int:
    if term not in _df_cache:
        got = col.get(where_document={"$contains": term},
                      limit=cap, include=[])
        _df_cache[term] = len(got["ids"])
    return _df_cache[term]


def _tokenize(text: str) -> list[str]:
    """中文按双字切、英文按词切，够 BM25 用又不引依赖。"""
    tokens: list[str] = []
    word = ""
    for ch in text.lower():
        if ch.isascii() and ch.isalnum():
            word += ch
            continue
        if word:
            tokens.append(word)
            word = ""
        if ch.isalnum() and ch not in _STOP:
            tokens.append(ch)
    if word:
        tokens.append(word)
    # 再补一层双字组合，缓解中文切词粒度问题
    bigrams = [a + b for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    import math
    tokenized = [_tokenize(d) for d in docs]
    q_tokens = set(_tokenize(query))
    n = len(tokenized)
    if n == 0:
        return [0.0] * len(docs)
    df: dict[str, int] = {}
    for toks in tokenized:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(t) for t in tokenized) / n
    scores: list[float] = []
    for toks in tokenized:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            denom = tf[t] + k1 * (1 - b + b * len(toks) / avgdl)
            s += idf * tf[t] * (k1 + 1) / denom
        scores.append(s)
    return scores


def search(query: str, k: int = config.TOP_K,
           source: str | None = None, section: str | None = None,
           hybrid: bool = True) -> list[SourceChunk]:
    """检索 top-k 片段。source/section 是元数据过滤（前缀匹配）。"""
    # 领域锚点：语料全是 Qt，查询带不带「Qt」对 bge-base 影响巨大——
    # 实测「信号槽连接方式有哪几种」不带 Qt 时正确答案页（GUI手册第66页）
    # 连 top-5 都进不了（营销页登顶 0.581），带 Qt 则第66页 0.770 第一。
    # 用户输入与 hint 都可能不带 Qt，入库前统一补锚。
    if "qt" not in query.lower():
        query = "Qt " + query
    with _lock:
        col = _get_collection()
        if col is None or col.count() == 0:
            return []
        where = None
        clauses = []
        if source:
            clauses.append({"source": {"$eq": source}})
        if section:
            clauses.append({"section": {"$eq": section}})
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}

        embedder = _get_embedder()
        qvec = embedder.encode([QUERY_PREFIX + query],
                               normalize_embeddings=True).tolist()
        # 向量召回多一些候选，再与 BM25 混合重排。
        # k*8：同一页被切成几十个渐进重叠块时（bge-base 对这类块打分极高），
        # top-20 会被同页重复块挤满，塌缩去重后没有新内容可填，只能拿垃圾页凑数。
        candidate_n = max(k * 8, 48)
        res = col.query(query_embeddings=qvec, n_results=candidate_n,
                        where=where, include=["documents", "metadatas", "distances"])

        docs: list[str] = res["documents"][0] or []
        metas = res["metadatas"][0] or []
        dists = res["distances"][0] or []
        ids = res["ids"][0] or []
        vec_scores = [1.0 - d for d in dists]          # 余弦相似度

        # 词面召回通道：bge-small 对 ASCII 标识符不敏感（实测含 QThread 的块
        # 连 top-40 向量候选都进不了，混合重排的候选池里没有就永远召不回）。
        # 2-gram 碎片方案已废弃：「何接/信支」这种跨词边界碎片会把拒答题
        # （微信支付）的沾边块拉进候选池，让前置拒答检查误判为「资料沾边」。
        # 改用 jieba 分词 + 全库 df 门槛：
        #   - ASCII 标识符（QThread/QPushButton）无条件补召回——本身就是特异词；
        #   - 中文词只补「常见词」（df≥10，如 布局/控件/网络）；
        #   - 罕见中文词（微信 df=6、支付 df=0）不补——拒答题拉不进沾边块。
        if hybrid and docs:
            seen_ids = set(ids)
            from .generate import _Q_STOP  # 惰性导入避免环依赖
            import jieba
            lex_terms = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", query)
            words = {w for w in jieba.cut(query)
                     if len(w) >= 2 and w not in _Q_STOP
                     and re.fullmatch(r"[一-龥]+", w)}
            lex_terms += [w for w in words if _corpus_df(col, w) >= 10]
            lex_terms = lex_terms[:10]
            for term in lex_terms[:6]:
                got = col.get(where=where, where_document={"$contains": term},
                              limit=5,
                              include=["documents", "metadatas", "embeddings"])
                embs = got["embeddings"]
                for j, gid in enumerate(got["ids"]):
                    if gid in seen_ids:
                        continue
                    seen_ids.add(gid)
                    ids.append(gid)
                    docs.append(got["documents"][j])
                    metas.append(got["metadatas"][j])
                    # 词面命中块用真实余弦（向量都存在 Chroma 里），不能灌
                    # max 分——实测营销 PDF 被词面词带进来后顶着作弊分排到第一
                    emb = embs[j]
                    vec_scores.append(
                        float(sum(a * b for a, b in zip(qvec[0], emb))))

        if hybrid and docs:
            bm = _bm25_scores(query, docs)
            max_bm = max(bm) or 1.0
            bm = [x / max_bm for x in bm]
            scores = [(1 - config.BM25_WEIGHT) * v + config.BM25_WEIGHT * b_
                      for v, b_ in zip(vec_scores, bm)]
        else:
            scores = vec_scores

        ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
        hits: list[SourceChunk] = []
        seen_texts: set[str] = set()
        # 近重复过滤：打包器 15% 尾部重叠 + 滑窗会产生边缘差几个字的重复块
        # （实测同一 PDF 页能召回 5 个几乎相同的块，引用卡片全是重复的）。
        # 按去空白字符 8-gram 双向包含判定（单向只查「候选被已有块覆盖」不够：
        # 渐进超集长块只覆盖短块的 60-77%，会全部放行）：
        #   - 候选几乎被已有块覆盖（≥85%）→ 冗余，跳过；
        #   - 已有短块几乎被候选覆盖 → 候选是信息更全的超集，原地升级替换。
        kept_grams: list[set[str]] = []
        section_count: dict[tuple[str, str], int] = {}

        def _grams(text: str) -> set[str]:
            norm = re.sub(r"\s+", "", text).lower()
            return {norm[j:j + 8] for j in range(max(len(norm) - 7, 1))}

        for i in ranked:
            if len(hits) >= k:
                break
            if docs[i] in seen_texts:
                continue
            # 同一 (来源, 章节) 限额 2 块：保证引用来源多样，不被单页挤满
            sec_key = (metas[i].get("source", ""), metas[i].get("section", ""))
            if section_count.get(sec_key, 0) >= 2:
                continue
            g = _grams(docs[i])
            if not g:
                continue
            cover_in = max((len(g & kg) / len(g) for kg in kept_grams),
                           default=0.0)
            if cover_in >= 0.85:
                continue  # 候选冗余
            chunk = SourceChunk(id=ids[i], text=docs[i],
                                source=metas[i].get("source", ""),
                                section=metas[i].get("section", ""),
                                score=round(scores[i], 4))
            replaced = False
            for pos, kg in enumerate(kept_grams):
                inter = len(g & kg)
                if inter and inter / len(kg) >= 0.85:
                    hits[pos] = chunk
                    kept_grams[pos] = g
                    seen_texts.add(docs[i])
                    replaced = True
                    break
            if replaced:
                continue
            kept_grams.append(g)
            seen_texts.add(docs[i])
            section_count[sec_key] = section_count.get(sec_key, 0) + 1
            hits.append(chunk)
        return hits


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "信号槽连接方式"
    for hit in search(q):
        print(f"[{hit.score}] {hit.source} :: {hit.section}")
        print("   " + hit.text[:80].replace("\n", " ") + " ...")
