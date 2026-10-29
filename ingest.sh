#!/bin/bash
# 一键入库：克隆语料（如缺失）+ 解析切分 + 向量化入 Chroma
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python -m app.ingest
