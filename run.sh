#!/bin/bash
# 一键启动：uvicorn 服务，打开 http://127.0.0.1:8000 即聊天界面
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
