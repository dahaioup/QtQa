"""集中管理模型名、路径、参数。所有可调项都在这里，方便调参与排查。"""
from __future__ import annotations

from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus" / "qtkaifajingyan"
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MODELS_DIR = ROOT / "models"

# ---- 模型 ----
# bge-small→bge-base（768 维、409MB）：small 对 QThread 这类 ASCII 标识符不敏感、
# 语义分辨率不足，用户定「可用性优先」后升级。注意切换后必须重建向量索引
# （维度 512→768），见 rebuild_index.py。
EMBED_MODEL = str(MODELS_DIR / "bge-base-zh-v1.5")
EMBED_DEVICE = "cpu"
# 0.5B→1.5B（用户定：可用性优先、延迟其次）：1.5B 实测 320 token prompt 首 token
# 16s、生成 ~10 tok/s、加载 63s；0.5B 快 4 倍但只会摘录、偶发编造，效果差。
# 1.5B 能执行「归纳但不编造」的复合指令，prompt/限长按 1.5B 重调。
CHAT_MODEL_PATH = MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
CHAT_N_CTX = 4096                                # 上下文窗口：检索块 + prompt + 512 输出
CHAT_N_THREADS = 6                               # i7-9750H 物理 6 核（12 线程）
CHAT_N_GPU_LAYERS = 0                            # 0 = 纯 CPU（0.5B 时实测快于 Metal）
CHAT_N_BATCH = 256
CHAT_TEMPERATURE = 0.2                           # 防幻觉：低温度
CHAT_REPEAT_PENALTY = 1.1                        # 1.5B 复读弱，1.2 会过度惩罚合理重复
CHAT_MAX_TOKENS = 300                            # 1.5B 写完整分点回答；160 截断过半段代码块
BRIEF_CHARS = 256                                # 每块喂模型的窗口字数（0.5B 时 120）
GROUNDING_MIN = 0.3                              # 答案与资料的 8-gram 重合度阈值：
                                                 # 实测幻觉编造 ~0.00、正常摘录 ~0.56
                                                 # （n=15 会误杀轻微改写的摘录）

# ---- 切分 ----
CHUNK_MIN_CHARS = 400
CHUNK_MAX_CHARS = 600
CHUNK_OVERLAP = 0.15
MAX_CHUNKS = 8000                                # 入库总量上限。文档建议 3000，
                                                 # 但语料加入 PDF 电子书（4 本有文本层，
                                                 # 约 130 万字）后 3000 会把 README.md
                                                 # 的块挤掉（大文件优先），故上调
TEXT_SUFFIXES = {".md", ".txt", ".html", ".cpp", ".h", ".pdf"}

# ---- 检索 ----
TOP_K = 5
BM25_WEIGHT = 0.35                               # 混合权重：0.65*向量 + 0.35*BM25
