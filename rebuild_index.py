"""切换 embedding 模型后重建向量索引（不重新解析语料）。

从现有 collection 读回全部 chunks 的文本与元数据 → 用 config.EMBED_MODEL
的新模型重编码（维度变了必须先删库重建，Chroma 不支持改维）→ 写回。

用法：source .venv/bin/activate && python rebuild_index.py
"""
import json
import time

from app import config


def main() -> None:
    t0 = time.time()
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    backup = config.DATA_DIR / "qt_kb_text_backup.json"
    old = client.get_collection("qt_kb")
    data = old.get(include=["documents", "metadatas"])
    ids = data["ids"]
    texts = data["documents"]
    metas = data["metadatas"]
    if ids:
        print(f"读回 {len(ids)} 条（旧索引，即将用 {config.EMBED_MODEL} 重编码）")
        # 先备份文本，重建失败也不用重新跑 17 分钟的语料解析
        backup.write_text(json.dumps(
            {"ids": ids, "documents": texts, "metadatas": metas},
            ensure_ascii=False), encoding="utf-8")
        print(f"文本已备份 -> {backup}")
    elif backup.exists():
        # 上次运行中途失败已删空 collection：从备份恢复，且绝不覆盖备份
        blob = json.loads(backup.read_text(encoding="utf-8"))
        ids, texts, metas = blob["ids"], blob["documents"], blob["metadatas"]
        print(f"collection 已空，从备份恢复 {len(ids)} 条并继续重建")
    else:
        raise SystemExit("collection 为空且没有备份，请重跑 ./ingest.sh")

    client.delete_collection("qt_kb")
    col = client.get_or_create_collection(name="qt_kb",
                                          metadata={"hnsw:space": "cosine"})

    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True).tolist()
    dim = len(vectors[0])
    for start in range(0, len(ids), 500):
        end = start + 500
        col.upsert(ids=ids[start:end], documents=texts[start:end],
                   metadatas=metas[start:end], embeddings=vectors[start:end])
    print(f"重建完成：{col.count()} 条 / {dim} 维，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
