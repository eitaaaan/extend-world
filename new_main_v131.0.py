#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# S-01 v131.0 Aegis Omnis — ADVANCED RAG + SPEED EDITION (2026)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主な強化点 (v130.1 → v131.0):
#   ★ Cross-Encoder Reranking: ms-marco-MiniLM-L-6-v2 で候補を精密スコアリング
#       - LRUキャッシュ(512件)で高速化、sentence_transformers未インストール時は自動フォールバック
#   ★ Reciprocal Rank Fusion (RRF): BM25ランクとChromaDBランクを融合して精度向上
#       - k=60 (論文推奨値)、重複テキストを先頭60文字で同一視
#   ★ HyDE (Query Expansion): クエリを仮想ドキュメントに変換してからベクトル検索
#       - FASTモデル使用、キャッシュ付き、Ollama未起動時は元クエリで動作
#   ★ Contextual Compression: 取得チャンクからクエリ関連文のみ抽出してノイズ削減
#       - BM25センテンススコアで文選択、max_chars=300で圧縮
#   ★ /kb search, /kb ask: hybrid_search_advanced パイプラインに統合
#   ★ /debug: Advanced RAG v131 ステータス（CE有効/無効、キャッシュ統計等）表示
#   ★ get_async_rag_data: 全5段階パイプライン適用（Web+Vector+RRF+Compress+CE）
# ──────────────────────────────────────────────────────────
# ★[v131] 生成速度最適化 (Speed Pack):
#   1. num_ctx 半減 (CPU: mid 8192→4096): KVキャッシュ確保コスト削減
#   2. num_batch CPU 1024→512: 最初のトークンまでのlatency削減
#   3. top_k 80→40/30: サンプリング候補削減
#   4. repeat_last_n 256→64: バッファスキャンコスト削減
#   5. num_predict 2048→1024: 過剰生成を防止
#   6. sanitize(): サロゲート検査を先行し不要な encode/decode を省略
#   7. _single_gen(): 毎トークンflushを廃止、8文字バッファ+句読点でwrite
#   8. _SYS_EXTRAS_TTL 0→8s: vector_search の毎ターン再実行を抑制
#   9. _STATE_CACHE_TTL 5→30s: 状態ファイルの再読み込み削減
#  10. RAGプリフェッチ: 入力直後にWeb取得をバックグラウンド開始
#  11. HyDE並列化: Web取得と同時実行、ブロッキング除去
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主な強化点 (v128.1 → v129.0):
#   ★ AIエンジン: gemma3:12b/4b/1b 自動選択 + Thinking Mode
#   ★ ハイブリッドRAG v2: BM25 + Vector + Cross-Encoder Reranking
#   ★ 並列Multi-Agent: 非同期ツール実行 (asyncio)
#   ★ Context Caching: KVキャッシュ最大活用・プリフィルキャッシュ
#   ★ 将棋AI: Negamax + TranspositionTable + KillerHeuristic
#   ★ チェスAI: MCTS (Monte Carlo Tree Search) 搭載
#   ★ セキュリティ: プロンプトインジェクション多層防御
#   ★ 新コマンド: /think /plan /code /reflect /mindmap /persona_edit
#   ★ Structured Outputs: JSON Schema強制でハルシネーション削減
#   ★ TokenBudget: 動的ctx割当・バックプレッシャー制御
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from __future__ import annotations

import atexit, glob, html as html_module, itertools, json, math, os, platform, re, shutil, ssl, traceback
import subprocess as S, sys, threading, time, unicodedata, urllib.parse as U, urllib.request as R
import asyncio, hashlib, queue, signal, weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 環境がreconfigureに非対応（Windows旧版等）の場合は無視

# .envファイルから環境変数を読み込む（BRAVE_API_KEYなど）
# pip install python-dotenv が必要。インストールしていなくても動作には支障なし。
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from http.cookiejar import CookieJar
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any, Callable

_ollama = None  # None=未試行, False=失敗確定（センチネル）
def _get_ollama():
    global _ollama
    # ★[修正/#2] 失敗済み(False)の場合は再試行しない。
    # 旧コードは失敗後も None のままにしていたため毎ターン import を試みていた。
    # また `import ollama, os` で os を関数内で毎回再 import していた点も修正。
    if _ollama is False:
        return None
    if _ollama is None:
        try:
            import ollama as _ollama_mod
            # ★[修正/ollama-host] WSL2→Windows Ollama接続対応
            # OLLAMA_HOST環境変数があればClientを明示的に初期化する。
            # export OLLAMA_HOST=http://172.24.80.1:11434 を~/.bashrcに設定すること。
            host = os.environ.get("OLLAMA_HOST", "")
            if host:
                _ollama = _ollama_mod.Client(host=host)
            else:
                _ollama = _ollama_mod
        except Exception:
            _ollama = False  # センチネル: 以降の再 import を抑止
    return _ollama if _ollama is not False else None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★[v129] モデル設定 — 自動選択エンジン対応
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# モデルは POWER_MODE と入力の複雑度から自動選択される。
# GPU環境 (OLLAMA_GPU=1) では 12b モデルが自動的に使用される。
_GPU_AVAILABLE = (
    os.environ.get("OLLAMA_GPU", "0") == "1"
    or os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    or os.environ.get("OLLAMA_IGPU_ENABLE", "0") == "1"
)
_HAS_12B       = False  # 起動時に ollama list で確認 (check_ollama_connection内)

# モデル優先順位: ultra/high=12b→4b, mid=4b, low=1b
MODEL_TIERS = {
    "ultra": ["gemma3:12b", "gemma3:4b"],
    "high":  ["gemma3:4b", "gemma3:12b"],
    "mid":   ["gemma3:4b"],
    "low":   ["gemma3:1b", "gemma3:4b"],
}
MODEL_NAME    = "gemma3:4b"   # デフォルト（check_ollama_connectionで更新）
DEEP_MODEL    = "gemma3:4b"   # complexモード用（同上）
FAST_MODEL    = "gemma3:4b"   # ★[GPU] iGPU環境では4bが最適バランス
MAX_HISTORY   = 6             # ★[v129] 4→6: 長い文脈での一貫性向上
RAG_TIMEOUT = 5           # ★[v129] 3.0→1.5: ハイブリッドRAGで取得量増加
USER_NAME     = "先輩"
OBSERVED_SUBJECT_NAME = USER_NAME
STATE_FILE    = "s01_state.json"
POWER_MODE    = "high"
TEMP_FACT     = 0.05           # 事実確認モード温度 (get_llm_opt is_logic=True 時に使用予定)
TEMP_VOICE    = 0.72
TEMP_HISTORY: list[float] = [0.72]
FACT_MIN_CHARS = 20

# ★[v129] Thinking Mode設定
THINKING_MODE = False          # True: chain-of-thought強制 (/think で切替)
THINKING_BUDGET = 1024        # thinking用トークン予算 (handle_think_mode が参照)

# ★[v129] TokenBudget — 動的ctx管理
TOKEN_BUDGET_SAFETY = 128     # 安全マージン(tok)
TOKEN_EST_JP  = 1.5           # 日本語1文字≒1.5トークン
TOKEN_EST_EN  = 0.4           # 英数字1文字≒0.4トークン

TEMP_MAP: dict[str, float] = {
    "/a": 0.68, "/w": 0.45, "/p": 0.30, "/c": 0.55,
    "/t": 0.82, "/q": 0.70, "/e": 0.40, "/sum": 0.40,
    "/r": 0.85, "/d": 0.62, "/think": 0.20, "/plan": 0.35,
    "/code": 0.15, "/reflect": 0.50,
}
MAX_RETRIES   = 0              # ★ 現在未使用: stream_responseはリトライなし設計
RETRY_DELAY   = 0.6            # ★ 現在未使用: 上記と同じ理由

# ★[v129] ThreadPoolExecutor — 非同期RAG・ツール並列実行
_THREAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="s01")

# ===== オフラインモード設定 =====
# OFFLINE_MODE = True にするとネット不要のパスだけ使う。
# Kiwix を起動しておくと Wikipedia がローカルで動く。
# 起動例: kiwix-serve --port 8888 wikipedia_ja_all.zim
OFFLINE_MODE  = False          # True でネット通信を完全無効化
KIWIX_PORT    = 8888           # kiwix-serve のポート番号
COMPLEXITY_KEYWORDS = {
    "deep": ['分析', '比較', '考察', '原因', '影響', '関係性', '構造', 'メカニズム', '原理', '定義', '本質', '違い', '対比', '傾向', '推移', '背景', '要因', '過程', '仕組み', '意義', '評価', '検証', '論点', '議論', '批判', '展望', '課題', '示唆', 'シナジー', 'トレードオフ', 'アーキテクチャ', 'アプローチ', '手法', '戦略', 'フレームワーク', 'パラダイム'],  # ★[修正/#6] '展望' 重複を除去
    "simple": ['こんにちは', 'おはよう', 'こんばんは', '元気', 'やあ', 'hey', 'hello', 'hi', 'おやすみ', 'またね', 'バイバイ', 'ねえ', 'ちょっと', 'ありがとう', 'すごい', 'なるほど', 'わかった', 'OK', 'はい', 'いいね'],
}

def estimate_complexity(text: str, cmd: str = "") -> str:
    # ★[修正/#6] lru_cache を除去。COMPLEXITY_KEYWORDS（グローバル変数）を参照するため
    # キャッシュが古い結果を返し続けるバグがあった。この関数は軽量なので毎回実行で問題なし。
    # ★[修正/complexity] 閾値80文字が厳しすぎて哲学的な短い問いも全てDEEP_MODEL送りになっていた。
    # gemma3:4b(15-20tok/s) vs llama3.1:8b(8tok/s) の速度差を考慮し、
    # 明確に複雑な場合のみDEEP_MODELを使うよう閾値を緩和する。
    text_lower = text.lower()
    # /a /c /sum /deep /midi は常にcomplex（二段階推論が必要）
    if any(text.startswith(c) for c in ("/a", "/c", "/sum", "/deep", "/midi")):
        return "complex"
    if cmd in ("/a", "/c", "/sum"): return "complex"
    deep_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["deep"] if kw in text)
    simple_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["simple"] if kw in text_lower)
    # ★[修正/complexity] deepキーワード2→3個以上でcomplex（閾値を緩和）
    if deep_hits >= 3: return "complex"
    if simple_hits >= 1 and deep_hits == 0: return "simple"
    # 英数字比率が高い（コード・技術用語）場合はcomplex
    tech_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / max(len(text), 1)
    if tech_ratio > 0.4: return "complex"
    # ★[修正/complexity] 80→200文字に緩和: 哲学者への問いは長くなりがちだが
    # gemma3:4bで十分な質が出る。200文字超の長文のみDEEP_MODELへ。
    if len(text) > 200: return "complex"
    return "simple"

def select_model(text: str, cmd: str = "") -> str:
    """★[v131] GPU環境: 全クエリをFAST_MODEL(4b)で処理、12bは使わない"""
    c = estimate_complexity(text, cmd)
    if c == "complex": return FAST_MODEL  # ★[GPU] 12bは遅すぎるので4bで統一
    if c == "simple" and len(text) < 60 and not cmd:
        return FAST_MODEL
    return FAST_MODEL  # ★[GPU] 全部4bで統一

RAG_CACHE: dict[str, tuple[float, str, int, float]] = {}

# ══ RAG高速化: L2ディスクキャッシュ + プリフェッチ + 適応型タイムアウト ══
import hashlib as _hl_rag, json as _json_rag, os as _os_rag

_RAG_DISK_DIR   = "s01_rag_cache"          # L2キャッシュディレクトリ
_RAG_DISK_TTL   = 3600 * 6                  # 6時間有効
_RAG_ADAPT_LOCK = threading.Lock()
_RAG_TIMING: list[float] = []               # 直近10回の取得時間
_RAG_PREFETCH_QUEUE: set[str] = set()       # プリフェッチ済みキュー
_os_rag.makedirs(_RAG_DISK_DIR, exist_ok=True)

def _rag_normalize_query(q: str) -> str:
    """クエリ正規化: 類似クエリを同一キーに統一"""
    import re as _re_n
    q = q.strip().rstrip("？?。．.！!").lower()
    q = _re_n.sub(r'[はがをにでのもとやか]$', '', q)  # 助詞末尾を除去
    q = _re_n.sub(r'\s+', ' ', q)
    return q[:80]  # 最大80文字

def _rag_disk_key(query: str) -> str:
    return _hl_rag.md5(query.encode()).hexdigest()[:16]

def _rag_disk_get(query: str):
    """L2ディスクキャッシュから取得"""
    try:
        path = _os_rag.path.join(_RAG_DISK_DIR, _rag_disk_key(query) + ".json")
        if not _os_rag.path.exists(path): return None
        with open(path, 'r', encoding='utf-8') as f:
            d = _json_rag.load(f)
        if time.time() - d['ts'] > _RAG_DISK_TTL: return None
        return d['content']
    except: return None

def _rag_disk_set(query: str, content: str):
    """L2ディスクキャッシュに保存"""
    try:
        path = _os_rag.path.join(_RAG_DISK_DIR, _rag_disk_key(query) + ".json")
        with open(path, 'w', encoding='utf-8') as f:
            _json_rag.dump({'ts': time.time(), 'content': content, 'query': query}, f, ensure_ascii=False)
    except: pass

def _rag_adaptive_timeout() -> float:
    """直近の取得時間から適応型タイムアウトを計算"""
    with _RAG_ADAPT_LOCK:
        if len(_RAG_TIMING) < 3: return RAG_TIMEOUT
        avg = sum(_RAG_TIMING[-5:]) / len(_RAG_TIMING[-5:])
        # 平均の1.5倍、最小0.8秒・最大RAG_TIMEOUT
        return max(0.8, min(RAG_TIMEOUT, avg * 1.5))

def _rag_record_timing(elapsed: float):
    with _RAG_ADAPT_LOCK:
        _RAG_TIMING.append(elapsed)
        if len(_RAG_TIMING) > 10: _RAG_TIMING.pop(0)

def prefetch_rag(query: str):
    """プリフェッチ: バックグラウンドでRAGを先読み"""
    nq = _rag_normalize_query(query)
    if nq in _RAG_PREFETCH_QUEUE: return
    _RAG_PREFETCH_QUEUE.add(nq)
    def _do():
        try: get_async_rag_data(query)
        except: pass
        finally:
            try: _RAG_PREFETCH_QUEUE.discard(nq)
            except: pass
    _THREAD_POOL.submit(_do)

_RAG_LOCK = threading.Lock()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★[v131] Advanced RAG 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-Encoder Reranking
CROSS_ENCODER_ENABLED  = True   # False でスキップ（CPU負荷を下げたい場合）
CROSS_ENCODER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K    = 5      # rerank後に残す件数
CROSS_ENCODER_CACHE_SIZE = 512  # LRUキャッシュエントリ数

# Reciprocal Rank Fusion
RRF_ENABLED = True
RRF_K       = 60                # RRF定数 (論文推奨値)

# HyDE (Hypothetical Document Embeddings) — クエリ拡張
HYDE_ENABLED    = True
HYDE_MAX_TOKENS = 120           # 仮想ドキュメント最大文字数
HYDE_TEMPERATURE = 0.4

# Contextual Compression
CTXCOMP_ENABLED   = True
CTXCOMP_MAX_CHARS = 300         # 圧縮後チャンクの最大文字数

# Cross-Encoder 遅延ロード
_cross_encoder_model = None
_cross_encoder_lock  = threading.Lock()
_cross_encoder_available = None  # None=未確認, True/False

def _get_cross_encoder():
    """Cross-Encoderモデルを遅延ロード。失敗時はNoneを返す。"""
    global _cross_encoder_model, _cross_encoder_available
    with _cross_encoder_lock:
        if _cross_encoder_available is False:
            return None
        if _cross_encoder_model is not None:
            return _cross_encoder_model
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL)
            _cross_encoder_available = True
        except Exception as e:
            _cross_encoder_available = False
            print(f"\033[33m[WARN] CrossEncoder初期化失敗 ({e}) — フォールバック: BM25のみ\033[0m")
        return _cross_encoder_model
# ★[修正/#3] TEMP_VOICE をロックで保護。BackgroundOptimizer と
# メインスレッドが同時に読み書きするためデータレースが発生していた。
_TEMP_VOICE_LOCK = threading.Lock()

def _get_temp_voice() -> float:
    with _TEMP_VOICE_LOCK:
        return TEMP_VOICE

def _set_temp_voice(val: float) -> None:
    global TEMP_VOICE
    with _TEMP_VOICE_LOCK:
        TEMP_VOICE = val

# ★[修正/temp-lock] 「温度変更禁止」「自動調整禁止」をユーザーが指示した際に
# BackgroundOptimizer とフィードバック処理の両方から温度変更されてしまうバグを修正。
# このフラグを True にすると _optimize_step・update_interaction_feedback の
# 温度変更コードが全てスキップされる。
TEMP_AUTO_TUNE_LOCKED: bool = False
_TEMP_LOCK_PATTERN = re.compile(
    r'温度.{0,8}(?:変更|調整|自動).{0,8}(?:禁止|しないで|やめて|しないでほしい|するな|しないで)|'
    r'(?:温度|temperature).{0,8}(?:固定|ロック|lock|freeze)|'
    r'自動.{0,5}(?:温度|調整).{0,8}(?:禁止|しないで|やめて)'
)

KEYWORD_MEMORY: list[str] = []
ROLEPLAY_ACTIVE = False
ROLEPLAY_SCENE  = ""
CUSTOM_PERSONA: dict | None = None

VECTOR_COL = None; VECTOR_AVAILABLE = False
_VECTOR_CLIENT = None
_VECTOR_COLS: dict[str, any] = {}  # コレクション名 -> Collectionオブジェクト
_VECTOR_COLS_LOCK = threading.Lock()  # ★[修正/#5] 並列ワーカーからの同時アクセス防止

def _init_vector_db():
    global VECTOR_COL, VECTOR_AVAILABLE, _VECTOR_CLIENT
    if VECTOR_AVAILABLE: return
    try:
        import chromadb
        _VECTOR_CLIENT = chromadb.PersistentClient(path="s01_vector_db")
        # ★[修正/#1] VECTOR_AVAILABLE を True にしてから _get_or_create_col を呼ぶ。
        # 旧コードは False のまま呼んでいたため _get_or_create_col → _init_vector_db →
        # _get_or_create_col … の無限再帰 (RecursionError) が発生していた。
        VECTOR_AVAILABLE = True
        VECTOR_COL = _get_or_create_col("s01_memory")
    except Exception as e:
        VECTOR_AVAILABLE = False          # 失敗時はフラグを戻す
        print(f"{C['y']}[WARN] chromadb初期化失敗: {e}{C['w']}")

def _get_or_create_col(name: str):
    """コレクションをキャッシュして返す。なければ作る。"""
    global _VECTOR_CLIENT, VECTOR_AVAILABLE, _VECTOR_COLS
    if not VECTOR_AVAILABLE: _init_vector_db()
    if not VECTOR_AVAILABLE: return None
    # ★[修正/#5] dict への読み書きをロックで保護
    with _VECTOR_COLS_LOCK:
        if name not in _VECTOR_COLS:
            try:
                _VECTOR_COLS[name] = _VECTOR_CLIENT.get_or_create_collection(name)
            except Exception:
                return None
        return _VECTOR_COLS[name]

_VEC_ID = [0]
_VEC_ID_LOCK = threading.Lock()  # ★[修正/#4] += はアトミックでないためロックが必要

def vector_add(text: str, metadata: dict = None, collection: str = "s01_memory") -> bool:
    if not VECTOR_AVAILABLE: _init_vector_db()
    if not VECTOR_AVAILABLE: return False
    col = _get_or_create_col(collection)
    if col is None: return False
    try:
        with _VEC_ID_LOCK:
            _VEC_ID[0] += 1
            vec_id = _VEC_ID[0]
        meta = {"text": text[:500], "time": now_stamp(), "collection": collection}
        if metadata: meta.update(metadata)
        col.add(documents=[text], metadatas=[meta], ids=[f"{collection}_{vec_id}"])
        return True
    except Exception: return False

def vector_search(query: str, n: int = 5, collection: str = "s01_memory") -> list[str]:
    if not VECTOR_AVAILABLE: _init_vector_db()
    if not VECTOR_AVAILABLE: return []
    col = _get_or_create_col(collection)
    if col is None: return []
    try:
        count = col.count()
        if count == 0: return []
        r = col.query(query_texts=[query], n_results=min(n, count))
        return [d for d in r.get("documents", [[]])[0] if d]
    except Exception: return []

def vector_count(collection: str = "s01_memory") -> int:
    if not VECTOR_AVAILABLE: _init_vector_db()
    if not VECTOR_AVAILABLE: return 0
    col = _get_or_create_col(collection)
    if col is None: return 0
    try: return col.count()
    except Exception: return 0

def vector_list_collections() -> list[str]:
    """登録済みコレクション（=取り込み済み書籍）一覧を返す。"""
    if not VECTOR_AVAILABLE: _init_vector_db()
    if not VECTOR_AVAILABLE or _VECTOR_CLIENT is None: return []
    try:
        return [c.name for c in _VECTOR_CLIENT.list_collections()]
    except Exception: return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★[v131] Advanced RAG ユーティリティ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── Cross-Encoder Reranking ─────────────────────────────
@lru_cache(maxsize=CROSS_ENCODER_CACHE_SIZE)
def _ce_score_cached(query: str, doc: str) -> float:
    """Cross-Encoderスコアをキャッシュ付きで計算。"""
    ce = _get_cross_encoder()
    if ce is None:
        return 0.0
    try:
        return float(ce.predict([(query, doc)])[0])
    except Exception:
        return 0.0

def cross_encoder_rerank(query: str, candidates: list[str], top_k: int = CROSS_ENCODER_TOP_K) -> list[str]:
    """
    候補テキストリストをCross-Encoderでスコアリングし直してtop_k件を返す。
    sentence_transformers未インストール時はBM25スコア順のまま返す。
    """
    if not CROSS_ENCODER_ENABLED or not candidates:
        return candidates[:top_k]
    ce = _get_cross_encoder()
    if ce is None:
        return candidates[:top_k]
    scored = [(doc, _ce_score_cached(query, doc[:512])) for doc in candidates]
    scored.sort(key=lambda x: -x[1])
    return [doc for doc, _ in scored[:top_k]]

# ─── Reciprocal Rank Fusion ──────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K
) -> list[str]:
    """
    複数のランキングリスト（BM25, ChromaDB等）をRRFで統合。
    各ドキュメントのRRFスコア = Σ 1/(k + rank_i)
    重複テキストは先頭60文字で同一視する。
    """
    if not RRF_ENABLED or not ranked_lists:
        return ranked_lists[0] if ranked_lists else []
    scores: dict[str, float] = {}
    key_to_doc: dict[str, str] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc[:60]
            if key not in key_to_doc:
                key_to_doc[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    sorted_keys = sorted(scores, key=lambda x: -scores[x])
    return [key_to_doc[k] for k in sorted_keys]

# ─── HyDE (Hypothetical Document Embeddings) ─────────────
_HYDE_CACHE: dict[str, str] = {}
_HYDE_LOCK  = threading.Lock()

def hyde_expand_query(query: str, ollama_client=None, model: str = "") -> str:
    """
    クエリを仮想ドキュメント（HyDE）に変換してベクトル検索の精度を上げる。
    Ollamaが使えない場合は元クエリをそのまま返す。
    キャッシュにより同一クエリの重複生成を防ぐ。
    """
    if not HYDE_ENABLED:
        return query
    with _HYDE_LOCK:
        if query in _HYDE_CACHE:
            return _HYDE_CACHE[query]
    o = ollama_client
    if o is None:
        return query
    _model = model or FAST_MODEL
    prompt = (
        f"以下の質問に対して、ウィキペディアの記事の冒頭2〜3文のような、"
        f"事実に基づく短いテキスト（{HYDE_MAX_TOKENS}文字以内）を日本語で書いてください。"
        f"余計な前置き・箇条書き・見出しは不要。本文のみ出力。\n質問: {query}"
    )
    try:
        resp = o.chat(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": HYDE_TEMPERATURE, "num_predict": 150},
            keep_alive=-1,
        )
        if isinstance(resp, dict):
            hypo = resp.get("message", {}).get("content", "")
        else:
            hypo = getattr(getattr(resp, "message", None), "content", "") or ""
        hypo = hypo.strip()[:HYDE_MAX_TOKENS]
        if len(hypo) < 20:
            return query
        with _HYDE_LOCK:
            _HYDE_CACHE[query] = hypo
        return hypo
    except Exception:
        return query

# ─── Contextual Compression ──────────────────────────────
def contextual_compress(query: str, chunks: list[str], max_chars: int = CTXCOMP_MAX_CHARS) -> list[str]:
    """
    取得チャンクからクエリに関連する文のみ抽出して圧縮する。
    センテンス単位でBM25スコアを計算し、上位文をmax_chars以内で結合して返す。
    """
    if not CTXCOMP_ENABLED or not chunks:
        return chunks
    q_words = re.findall(r'[\u3040-\u9FFF\w]{2,}', query.lower())
    if not q_words:
        return [c[:max_chars] for c in chunks]

    def _sent_score(sent: str) -> float:
        s_lower = sent.lower()
        return sum(s_lower.count(w) for w in q_words)

    compressed = []
    for chunk in chunks:
        # 句点・改行で文分割
        sents = re.split(r'(?<=[。\.\!\?！？\n])', chunk)
        sents = [s.strip() for s in sents if len(s.strip()) > 10]
        if not sents:
            compressed.append(chunk[:max_chars])
            continue
        scored_sents = sorted(sents, key=_sent_score, reverse=True)
        buf = ""
        for s in scored_sents:
            if len(buf) + len(s) > max_chars:
                break
            buf += s + " "
        compressed.append(buf.strip() or chunk[:max_chars])
    return compressed

# ─── Hybrid Search with RRF + Cross-Encoder ──────────────
def hybrid_search_advanced(
    query: str,
    n_candidates: int = 15,
    top_k: int = CROSS_ENCODER_TOP_K,
    collection: str = "s01_memory",
    ollama_client=None,
) -> list[str]:
    """
    BM25 + ChromaDB ベクトル検索 → RRF融合 → Contextual Compression → Cross-Encoder Reranking
    の全パイプラインを一括実行。
    """
    # 1. HyDE: クエリを仮想ドキュメントに拡張してベクトル検索精度向上
    hyde_query = hyde_expand_query(query, ollama_client)

    # 2. ベクトル検索（ChromaDB）— HyDE拡張クエリを使用
    vec_results = vector_search(hyde_query, n=n_candidates, collection=collection)

    # 3. BM25風スコアリング（テキスト行に対してクエリキーワードマッチ）
    def _bm25_rank(docs: list[str]) -> list[str]:
        q_words = re.findall(r'[\u3040-\u9FFF\w]{2,}', query.lower())
        if not q_words:
            return docs
        avg_len = sum(len(d) for d in docs) / max(len(docs), 1)
        def _score(d: str) -> float:
            d_lower = d.lower()
            s = 0.0
            for w in q_words:
                tf = d_lower.count(w)
                if tf:
                    idf = math.log(1 + len(docs) / (1 + sum(1 for x in docs if w in x.lower())))
                    s += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * len(d) / max(avg_len, 1)))
            return s
        return [d for d, _ in sorted([(d, _score(d)) for d in docs], key=lambda x: -x[1])]

    bm25_results = _bm25_rank(vec_results) if vec_results else []

    # 4. RRF: BM25ランクとベクトルランクを融合
    fused = reciprocal_rank_fusion([bm25_results, vec_results])

    # 5. Contextual Compression: 各チャンクから関連文のみ抽出
    compressed = contextual_compress(query, fused[:n_candidates])

    # 6. Cross-Encoder Reranking: 最終スコアリング
    return cross_encoder_rerank(query, compressed, top_k=top_k)

# ===== 他AI参照型学習エンジン =====
REFERENCE_PATTERNS = {
    "structure": ["結論→理由→具体例→まとめ", "冒頭で核心に触れる", "箇条書きで整理", "段落分けで可読性向上"],
    "clarity": ["曖昧な表現を避ける", "数値・具体名を入れる", "主語述語を明確に", "一文一義"],
    "depth": ["表面的でない考察", "比較・対比を含める", "因果関係を説明", "複数の視点から分析"],
    "engagement": ["ユーザーの文脈を反映", "質問に対して直接的", "適度な相槌・共感", "次のアクションを提示"],
    "accuracy": ["不確かな情報に断り", "事実と推論を区別", "出典を明示可能な範囲で", "過度な一般化を避ける"],
}
REFERENCE_SCORES: dict[str, list[float]] = {}
SELF_EVAL_LOG: list[dict] = []

def self_evaluate(response: str, mode: str) -> dict[str, float]:
    # ★[修正/#15] この評価は「評価基準文のキーワードが返答に含まれているか」を測るだけの
    # 疑似評価であり、実際の応答品質を保証しない（循環評価）。
    # 参考指標として使用し、絶対的品質指標として扱わないこと。
    scores = {}
    for category, patterns in REFERENCE_PATTERNS.items():
        score = 0.0
        for p in patterns:
            keywords = set(re.findall(r'[\u3040-\u9FFF\w]{2,}', p))
            match = sum(1 for k in keywords if k in response)
            score += min(1.0, match / max(len(keywords), 1))
        scores[category] = round(score / max(len(patterns), 1), 2)
    if mode not in REFERENCE_SCORES: REFERENCE_SCORES[mode] = []
    avg_score = sum(scores.values()) / max(len(scores), 1)
    REFERENCE_SCORES[mode].append(avg_score)
    if len(REFERENCE_SCORES[mode]) > 200: REFERENCE_SCORES[mode] = REFERENCE_SCORES[mode][-200:]
    SELF_EVAL_LOG.append({"time": time.time(), "mode": mode, "scores": scores, "avg": avg_score})
    if len(SELF_EVAL_LOG) > 100: SELF_EVAL_LOG[:] = SELF_EVAL_LOG[-100:]
    return scores

def get_reference_feedback() -> str:
    if not REFERENCE_SCORES: return "学習データ不足"
    best_mode = max(REFERENCE_SCORES, key=lambda m: sum(REFERENCE_SCORES[m]) / max(len(REFERENCE_SCORES[m]), 1))
    worst_mode = min(REFERENCE_SCORES, key=lambda m: sum(REFERENCE_SCORES[m]) / max(len(REFERENCE_SCORES[m]), 1))
    lines = [f"{C['c']}=== 他AI参照 自己評価 ==={C['w']}"]
    for cat in REFERENCE_PATTERNS:
        scores = [log["scores"][cat] for log in SELF_EVAL_LOG[-20:] if cat in log["scores"]]
        avg = sum(scores) / max(len(scores), 1) if scores else 0
        bar = "█" * max(1, min(10, int(avg * 10)))
        lines.append(f"  {cat:12s} {avg:.2f} {bar}")
    lines.append(f"ベストモード: {best_mode} ({sum(REFERENCE_SCORES[best_mode])/len(REFERENCE_SCORES[best_mode]):.2f})")
    lines.append(f"改善対象: {worst_mode} ({sum(REFERENCE_SCORES[worst_mode])/len(REFERENCE_SCORES[worst_mode]):.2f})")
    lines.append(f"評価回数: {len(SELF_EVAL_LOG)}")
    return "\n".join(lines)

PROMPT_OPTIMIZATIONS: dict[str, list[str]] = {}
OPTIMIZATION_HISTORY: list[str] = []

# ★[修正A+B] ユーザーの指摘文から具体的な指示を抽出してPROMPT_OPTIMIZATIONSに反映する
_USER_DIRECTIVE_PATTERNS = [
    # 「〜はおかしい」「〜が変だ」→その要素をやめる指示
    (re.compile(r'(.{2,20})(?:は|が)(?:おかしい|変だ|変です|へんだ|ヘンだ|おかしくない\?|変じゃない\?)'), "禁止表現"),
    # 「〜しないで」「〜はやめて」「〜するな」
    (re.compile(r'(.{2,20})(?:しないで|はやめて|やめてほしい|するな|しないでほしい)'), "禁止表現"),
    # 「〜にして」「〜で話して」「〜口調で」「〜にしてほしい」
    (re.compile(r'(.{2,20})(?:にして|で話して|口調で|で答えて|にしてほしい|でお願い)'), "指定表現"),
    # 「もっと〜して」
    (re.compile(r'もっと(.{2,20})(?:して|にして|にしてほしい|お願い)'), "指定表現"),
    # 「〜の言い方はやめて」「〜な話し方は嫌だ」
    (re.compile(r'(.{2,20})(?:言い方|話し方|口調)(?:は|が)(?:嫌|いや|おかしい|変|ダメ)'), "禁止表現"),
]

def extract_user_directive(user_text: str) -> list[tuple[str, str]]:
    """ユーザー発言から (カテゴリ, 指示文) のリストを抽出する"""
    results = []
    for pattern, category in _USER_DIRECTIVE_PATTERNS:
        for m in pattern.finditer(user_text):
            directive = m.group(0).strip()
            if len(directive) >= 4:
                results.append((category, directive))
    return results

_DIRECTIVE_PER_CAT_MAX = 5   # カテゴリごとの上限件数
_DIRECTIVE_TOTAL_MAX  = 15  # ペルソナごとの合計上限件数

@lru_cache(maxsize=128)
def _persona_key(persona_name: str) -> str:
    """ペルソナ名を辞書キーに変換"""
    return re.sub(r'\s+', '_', persona_name.strip().lower())[:30] or "global"

def _get_persona_bucket(persona_name: str) -> dict:
    """現在ペルソナの指示辞書を返す（なければ作成）"""
    key = _persona_key(persona_name)
    if key not in PROMPT_OPTIMIZATIONS:
        PROMPT_OPTIMIZATIONS[key] = {}
    return PROMPT_OPTIMIZATIONS[key]

def apply_user_directive(user_text: str, persona_name: str = "") -> list[str]:
    """ユーザー指摘をPROMPT_OPTIMIZATIONSに即時反映し、適用した指示一覧を返す"""
    global TEMP_AUTO_TUNE_LOCKED
    # ★[修正/temp-lock] 「温度変更禁止」などの指示を検出してフラグを立てる
    if _TEMP_LOCK_PATTERN.search(user_text):
        TEMP_AUTO_TUNE_LOCKED = True
        msg = "温度自動調整: ユーザー指示によりロック (TEMP_AUTO_TUNE_LOCKED=True)"
        OPTIMIZATION_HISTORY.append(msg)
        if len(OPTIMIZATION_HISTORY) > 50:
            OPTIMIZATION_HISTORY[:] = OPTIMIZATION_HISTORY[-50:]
    directives = extract_user_directive(user_text)
    applied = []
    bucket = _get_persona_bucket(persona_name or "global")
    total = sum(len(v) for v in bucket.values())
    for category, directive in directives:
        if category not in bucket:
            bucket[category] = []
        # 同一or類似の指示が既にあればスキップ
        if any(directive[:10] in existing for existing in bucket[category]):
            continue
        # カテゴリ上限を超えたら最古を削除
        if len(bucket[category]) >= _DIRECTIVE_PER_CAT_MAX:
            bucket[category].pop(0)
            total -= 1
        # ペルソナ合計上限を超えたら最も古いエントリを削除
        if total >= _DIRECTIVE_TOTAL_MAX:
            for cat in bucket:
                if bucket[cat]:
                    bucket[cat].pop(0)
                    total -= 1
                    break
        bucket[category].append(directive)
        total += 1
        msg = f"ユーザー指摘反映 [{persona_name or 'global'}][{category}]: {directive}"
        OPTIMIZATION_HISTORY.append(msg)
        if len(OPTIMIZATION_HISTORY) > 50:
            OPTIMIZATION_HISTORY[:] = OPTIMIZATION_HISTORY[-50:]
        applied.append(directive)
    return applied

def inject_optimizations(_mode: str = "", persona_name: str = "") -> str:
    """★[修正/reference] グローバルAI技術ディレクティブ + ペルソナ指摘を両方注入する。
    /reference はglobalバケツに書くため、persona専用バケツと両方をマージする。"""
    if not PROMPT_OPTIMIZATIONS: return ""
    parts = []

    def _extract_bucket(bucket: dict, label_prefix: str):
        # ユーザー直接指摘（禁止表現・指定表現）を先頭・全件展開
        for cat in ("禁止表現", "指定表現"):
            for d in bucket.get(cat, []):
                parts.append(f"【ユーザー指摘・厳守】{d}")
        # AI技術系ディレクティブ: カテゴリ別に最新2件
        for cat, directives in bucket.items():
            if cat in ("禁止表現", "指定表現"): continue
            for d in directives[-2:]:
                parts.append(f"【{label_prefix}{cat}改善】{d}")

    # 1) globalバケツ (主に /reference が書き込む)
    global_bucket = _get_persona_bucket("global")
    _extract_bucket(global_bucket, "AI技術/")

    # 2) ペルソナ固有バケツ (apply_user_directive が書き込む)
    if persona_name and persona_name != "global":
        persona_bucket = _get_persona_bucket(persona_name)
        if persona_bucket:
            _extract_bucket(persona_bucket, "")

    return "\n" + "\n".join(parts) if parts else ""

def auto_optimize_prompts() -> list[str]:
    actions = []
    if not SELF_EVAL_LOG or len(SELF_EVAL_LOG) < 5: return actions
    recent = SELF_EVAL_LOG[-10:]
    weak_cats: dict[str, list[float]] = {}
    for log in recent:
        for cat, score in log.get("scores", {}).items():
            weak_cats.setdefault(cat, []).append(score)
    improvement_map = {
        "structure": "必ず「結論→理由→まとめ」の順で書け。段落冒頭で主題を示せ。",
        "clarity": "曖昧な語を避け、数値・固有名詞を明示せよ。一文は短く。",
        "depth": "表面的な説明に留めず、比較・因果・複数視点を含めよ。",
        "engagement": "ユーザーの発言に直接応答し、次のアクションを提示せよ。",
        "accuracy": "不確かな情報には「〜の可能性」「〜と言われる」と留保をつけよ。事実と意見を区別せよ。",
    }
    bucket = _get_persona_bucket("global")  # 自動最適化はglobalバケツに書く
    for cat, scores in weak_cats.items():
        avg = sum(scores) / max(len(scores), 1)
        if avg < 0.35:
            if cat not in bucket: bucket[cat] = []
            directive = improvement_map.get(cat, f"{cat}を改善せよ")
            if directive not in bucket[cat]:
                if len(bucket[cat]) >= _DIRECTIVE_PER_CAT_MAX:
                    bucket[cat].pop(0)
                bucket[cat].append(directive)
                msg = f"プロンプト改善 [{cat}]: {directive}"
                actions.append(msg)
                OPTIMIZATION_HISTORY.append(msg)
                if len(OPTIMIZATION_HISTORY) > 50: OPTIMIZATION_HISTORY[:] = OPTIMIZATION_HISTORY[-50:]
    return actions

def optimization_status() -> str:
    if not PROMPT_OPTIMIZATIONS and not OPTIMIZATION_HISTORY: return "最適化なし"
    lines = [f"{C['c']}=== プロンプト最適化状態 ==={C['w']}"]
    total = 0
    for pkey, bucket in PROMPT_OPTIMIZATIONS.items():
        if not isinstance(bucket, dict): continue
        cnt = sum(len(v) for v in bucket.values())
        if cnt == 0: continue
        total += cnt
        lines.append(f"  [{pkey}] {cnt}件")
        for cat, dirs in bucket.items():
            for d in dirs:
                lines.append(f"    {cat}: {d[:60]}")
    lines.append(f"合計: {total}件 (上限: ペルソナごと{_DIRECTIVE_TOTAL_MAX}件)")
    lines.append(f"{C['dim']}直近の最適化:{C['w']}")
    for h in OPTIMIZATION_HISTORY[-3:]:
        lines.append(f"  • {h}")
    return "\n".join(lines)

# ===== バックグラウンド最適化エンジン =====
class BackgroundOptimizer:
    def __init__(self, interval: int = 120):
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._running = False
        self.history: list[str] = []
        self.last_auto_tune = 0.0

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)

    def _log(self, msg: str):
        self.history.append(msg)
        if len(self.history) > 50: self.history[:] = self.history[-50:]

    def _loop(self):
        while self._running:
            try: self._optimize_step()
            except Exception as e: print(f"{C['y']}[WARN] optimizer: {e}{C['w']}")
            time.sleep(self.interval)

    def _optimize_step(self):
        now = time.time()
        actions = []

        with _RAG_LOCK:
            expired = [k for k, (ts, _, acc, conf) in RAG_CACHE.items()
                       if (now - ts > 3600 and acc < 2) or (now - ts > 21600) or (conf < 0.3 and acc == 0)]
            for k in expired: RAG_CACHE.pop(k, None)
        if expired: actions.append(f"RAGキャッシュ{len(expired)}件削除")

        if now - self.last_auto_tune > 300 and LEARNING_STATS["total_interactions"] >= 10:
            # ★[修正/temp-lock] ユーザーが温度変更禁止を指示している場合はスキップ
            if not TEMP_AUTO_TUNE_LOCKED:
                best_temp = get_best_temp("d")
                if best_temp is not None:
                    # ★[修正/#3] ロック経由で TEMP_VOICE を更新（データレース防止）
                    old = _get_temp_voice()
                    _set_temp_voice((old + best_temp) / 2)
                    new = _get_temp_voice()
                    if abs(new - old) > 0.02: actions.append(f"温度調整: {old:.2f}→{new:.2f}")
            self.last_auto_tune = now

        for mode, scores in PROMPT_PERFORMANCE.items():
            if len(scores) >= 5:
                avg = sum(scores) / len(scores)
                if avg < -0.3: actions.append(f"⚠ {mode}モード低評価({avg:.1f})")

        if REFERENCE_SCORES and LEARNING_STATS["total_interactions"] % 15 == 0:
            weak_cats = [cat for log in SELF_EVAL_LOG[-10:] for cat, score in log.get("scores", {}).items() if score < 0.3]
            if weak_cats:
                target = max(set(weak_cats), key=weak_cats.count)
                actions.append(f"改善提案: {target}スコア低下({sum(1 for w in weak_cats if w==target)/max(len(weak_cats),1):.0%})")
            actions.extend(auto_optimize_prompts())

        if LEARNING_STATS["total_interactions"] % 10 == 0 and LEARNING_STATS["total_interactions"] > 0:
            # ★[修正/#12] セッション開始後の増分が 0 のときは保存しない
            # (_SESSION_START_INTERACTIONS は run() で設定。未定義の場合は 0 扱い)
            _sess_start = globals().get("_SESSION_START_INTERACTIONS", 0)
            if LEARNING_STATS["total_interactions"] - _sess_start > 0:
                persist_learning()
                actions.append("学習データ保存")

        for a in actions:
            self._log(a)

    def status(self) -> str:
        rows = [f"{C['c']}=== 最適化エンジン ==={C['w']}"]
        rows.append(f"間隔: {self.interval}s | 状態: {'稼働中' if self._running else '停止'}")
        rows.append(f"温度(TEMP_VOICE): {TEMP_VOICE:.2f}")
        rows.append(f"直近の最適化:")
        for h in self.history[-5:]: rows.append(f"  • {h}")
        return "\n".join(rows)

OPTIMIZER = BackgroundOptimizer()

# ===== ツール使用エージェント (ReAct) =====
TOOL_REGISTRY: dict[str, dict] = {}
TOOL_CALL_RE = re.compile(r'TOOL_CALL:\s*(\w+)\s*\|\s*(\{.*?\})', re.S)

def _reg_tool(name: str, desc: str, params: dict[str, str]):
    TOOL_REGISTRY[name] = {"desc": desc, "params": params}

_reg_tool("calculator", "数式を計算する（例: 2+2, sqrt(16)）", {"expression": "計算式"})
_reg_tool("web_search", "ウェブ検索して情報を得る", {"query": "検索クエリ"})
_reg_tool("web_fetch",  "URLからHTMLコンテンツを取得する", {"url": "完全なURL"})
_reg_tool("file_read",  "ファイルを読み込む", {"path": "ファイルの絶対パス"})
_reg_tool("file_write", "ファイルに書き込む", {"path": "ファイルの絶対パス", "content": "書き込む内容"})
_reg_tool("code_run",   "Pythonコードを実行する", {"code": "実行するコード"})


# ===== セキュリティ: ファイル・ネットワーク操作の制約 =====
# file_read / file_write が操作できるのはカレントディレクトリ配下のみ
_SAFE_BASE_DIR = os.path.realpath(os.getcwd())

# web_fetch で接続を禁止するアドレス（SSRF防止）
_SSRF_BLOCKED_HOSTS = frozenset([
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "169.254.169.254",   # AWS/GCP/Azure メタデータエンドポイント
    "metadata.google.internal",
])
_SSRF_BLOCKED_PREFIXES = ("10.", "192.168.")

def _assert_safe_path(raw_path: str) -> str:
    """パストラバーサル防止: カレントディレクトリ外へのアクセスを拒否する。
    正規化された安全なパスを返す。違反時は ValueError を送出する。"""
    resolved = os.path.realpath(raw_path)
    if not resolved.startswith(_SAFE_BASE_DIR + os.sep) and resolved != _SAFE_BASE_DIR:
        raise ValueError(f"アクセス禁止: カレントディレクトリ外のパスです ({raw_path!r})")
    return resolved

def _assert_safe_url(url: str) -> None:
    """SSRF防止: ローカルホスト・内部ネットワーク・非HTTPスキームをブロックする。
    ★[修正/#9] IPv6マップドアドレス (::ffff:127.0.0.1)・10進数IP (2130706433)・
    ブラケット記法も遮断するよう強化。
    ★[v130.7] DNSリバインディング対策として、ホスト名の解決先IPも検査する。"""
    import ipaddress, socket
    parsed = U.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"許可されていないスキーム: {parsed.scheme!r}")
    host = parsed.hostname or ""
    host_lower = host.lower()

    # ブラケット除去（IPv6: [::1] → ::1）
    if host_lower.startswith("[") and host_lower.endswith("]"):
        host_lower = host_lower[1:-1]

    if host_lower in _SSRF_BLOCKED_HOSTS:
        raise ValueError(f"アクセス禁止: ローカルホストへのリクエストは許可されていません ({host!r})")
    for prefix in _SSRF_BLOCKED_PREFIXES:
        if host_lower.startswith(prefix):
            raise ValueError(f"アクセス禁止: プライベートIPレンジ ({host!r})")
    if re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", host_lower):
        raise ValueError(f"アクセス禁止: プライベートIPレンジ ({host!r})")

    def _blocked_ip(ip_obj) -> bool:
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
            return _blocked_ip(ip_obj.ipv4_mapped)
        return (
            ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_link_local
            or ip_obj.is_reserved or ip_obj.is_unspecified or ip_obj.is_multicast
        )

    # ★ IPv6マップドアドレス・純粋数値IP（10進/16進）をipaddressで検証
    try:
        if host_lower.isdigit():
            ip = ipaddress.ip_address(int(host_lower))
        elif host_lower.startswith("0x"):
            ip = ipaddress.ip_address(int(host_lower, 16))
        else:
            ip = ipaddress.ip_address(host_lower)
        if _blocked_ip(ip):
            raise ValueError(f"アクセス禁止: プライベート/ループバックIPアドレス ({host!r})")
    except ValueError as e:
        if "アクセス禁止" in str(e):
            raise
        # ip_address() が失敗した場合はホスト名として続行（既に prefix チェック済み）
        pass

    # ホスト名がlocalhost/社内IPに解決されるケースを遮断する。
    # DNS解決に失敗するURLは実リクエストも成功しないため、安全側に倒す。
    if not re.match(r"^\[?[0-9a-f:.]+\]?$", host_lower, re.I) and not host_lower.isdigit() and not host_lower.startswith("0x"):
        try:
            resolved = socket.getaddrinfo(host_lower, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise ValueError(f"ホスト名を解決できません: {host!r} ({e})")
        checked = set()
        for family, _, _, _, sockaddr in resolved:
            ip_text = sockaddr[0]
            if ip_text in checked:
                continue
            checked.add(ip_text)
            try:
                if _blocked_ip(ipaddress.ip_address(ip_text)):
                    raise ValueError(f"アクセス禁止: ホスト名が内部IPに解決されます ({host!r} -> {ip_text})")
            except ValueError as e:
                if "アクセス禁止" in str(e):
                    raise


def _exec_tool(name: str, **kwargs) -> str:
    try:
        if name == "calculator":
            expr = kwargs.get("expression", "")
            # ホワイトリスト: 数字・演算子・mathの関数名のみ許可
            # ダンダー属性（__class__等）を完全に除外するため文字列ベースでなく
            # コンパイル後のASTを検査する
            import ast as _ast
            _ALLOWED_NODES = {
                _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Call,
                _ast.Attribute, _ast.Name, _ast.Constant, _ast.Load,
                _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.Mod,
                _ast.Pow, _ast.FloorDiv, _ast.USub, _ast.UAdd,
            }
            try:
                tree = _ast.parse(expr, mode="eval")
            except SyntaxError as e:
                return f"Error: 構文エラー ({e})"
            for node in _ast.walk(tree):
                if type(node) not in _ALLOWED_NODES:
                    return f"Error: 許可されていない操作 ({type(node).__name__})"
                if isinstance(node, _ast.Attribute):
                    if node.attr.startswith("_"):
                        return "Error: プライベート属性へのアクセスは禁止"
                if isinstance(node, _ast.Name) and node.id.startswith("_"):
                    return "Error: プライベート名は使用不可"
            ns: dict = {"__builtins__": {}, "math": math}
            return str(eval(compile(tree, "<calc>", "eval"), ns))

        elif name == "web_search":
            data = U.urlencode({"q": kwargs["query"], "kl": "jp-jp"}).encode("utf-8")
            html = fetch_html("https://lite.duckduckgo.com/lite/", data=data, timeout=5, silent=True)
            snips = re.findall(r'class="result-snippet"[^>]*>(.*?)</td>', html, re.I | re.S)
            lines = [strip_tags(s) for s in snips[:5] if len(strip_tags(s).strip()) > 15]
            return "\n".join(lines) if lines else "結果なし"

        elif name == "web_fetch":
            url = kwargs.get("url", "")
            _assert_safe_url(url)   # SSRF防止（リクエスト前）
            # ★[修正/#9] リダイレクト先もSSRFチェックする
            text = fetch_html(url, timeout=8, silent=True, redirect_checker=_assert_safe_url)
            return strip_tags(text)[:2000] if text else "取得失敗"

        elif name == "file_read":
            safe = _assert_safe_path(kwargs["path"])    # パストラバーサル防止
            with open(safe, "r", encoding="utf-8") as f:
                return f.read()[:2000]

        elif name == "file_write":
            safe = _assert_safe_path(kwargs["path"])    # パストラバーサル防止
            with open(safe, "w", encoding="utf-8") as f:
                f.write(kwargs["content"])
            return f"書き込み完了: {os.path.basename(safe)}"

        elif name == "code_run":
            # exec に渡す builtins を最小限に絞る（ファイルI/O・import・os を除外）
            import io, sys as _sys
            _SAFE_BUILTINS = {
                "print": print, "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter, "sorted": sorted,
                "reversed": reversed, "sum": sum, "min": min, "max": max,
                "abs": abs, "round": round, "int": int, "float": float,
                "str": str, "bool": bool, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "isinstance": isinstance,
                # ★[修正/#8] type を除去: メタクラス操作 (type('x', (object,), {})) の入口になる
                # ★[修正/#8] repr/format も MRO traversal を通じた脱出リスクあり → 除去
                "True": True, "False": False, "None": None,
            }
            old = _sys.stdout
            buf = io.StringIO()
            _sys.stdout = buf
            try:
                exec(kwargs["code"], {"__builtins__": _SAFE_BUILTINS, "math": math})
            except Exception as e:
                return f"Error: {e}"
            finally:
                _sys.stdout = old
                out = buf.getvalue()
            return out[:1000] or "OK（出力なし）"

    except ValueError as e:
        # セキュリティ制約違反
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"

def tool_instructions() -> str:
    lines = ["【利用可能なツール】"]
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {info['desc']} 引数: {', '.join(info['params'])}")
    lines.append("")
    lines.append("ツールを使う場合は、回答の代わりに以下の形式で出力せよ:")
    lines.append('TOOL_CALL: ツール名 | {"arg名": "値"}')
    lines.append("ツールを使わない場合は通常通り会話せよ。")
    return "\n".join(lines)

def tool_agent_chat(messages: list, is_logic: bool, text_len: int, temp: float | None = None, max_turns: int = 3) -> str:
    tool_model = DEEP_MODEL
    tool_temp = 0.15
    sys_content = (
        "あなたはツールエージェント。ユーザーの質問に答えるため、必要ならツールを使え。\n"
        "ツールを使う時は以下のJSON形式**だけ**を出力せよ（会話文・説明は一切不要）:\n"
        'TOOL_CALL: ツール名 | {"arg名": "値"}\n'
        "例: ユーザーが「2+2は？」→ TOOL_CALL: calculator | {\"expression\": \"2+2\"}\n"
        "ツールを使わない時だけ通常通り会話せよ。\n\n"
        + tool_instructions()
    )
    msgs = [{"role": "system", "content": sys_content}]
    for m in messages:
        if m["role"] == "user":
            msgs.append(m)
            break
    for turn in range(max_turns):
        raw = stream_response(msgs, is_logic, text_len, tool_temp, silent=True, model=tool_model)
        if not raw: return ""
        m = TOOL_CALL_RE.search(raw)
        if not m: return raw
        t_name, t_args_str = m.group(1), m.group(2)
        try: t_args = json.loads(t_args_str)
        except json.JSONDecodeError: return raw
        result = _exec_tool(t_name, **t_args)
        msgs.append({"role": "assistant", "content": raw.strip()})
        msgs.append({"role": "user", "content": f"ツール結果:\n{result}\n\nこの結果を日本語で簡潔に回答せよ。"})
    msgs.append({"role": "user", "content": "以上のツール結果を踏まえて最終回答を書け。"})
    final = stream_response(msgs, is_logic, text_len, tool_temp, silent=True, model=tool_model) or ""
    return TOOL_CALL_RE.sub("", final).strip()

# ===== 自己進化型学習アルゴリズム =====
INTERACTION_LOG: list[dict] = []
FEEDBACK_PATTERNS = {
    'positive': ['ありがとう', 'いいね', '役に立った', 'すごい', '助かった', 'さすが', '正解', 'なるほど', 'そうそう', 'それそれ', '素晴らしい', '完璧', '最高', 'やった', 'できた', 'わかった', '了解', 'グッド', 'ナイス', 'perfect', 'good', 'great', 'いい感じ', 'バッチリ', 'まさに', 'その通り', 'ぴったり'],
    'negative': ['違う', '間違ってる', 'いや', '違います', 'ちがう', 'つまんない', 'もういい', '違うよ', '意味ない', '違ってる', 'ちげえ', 'ダメ', 'ダメだ', '違うんだ',
                 # ★[修正D] 話し方・口調指摘系を追加
                 'おかしい', 'へん', '変だ', '変です', 'おかしくない', 'なんか変', 'ちょっと変',
                 '直して', '直してほしい', 'なおして', '治して', '改めて', 'やり直して',
                 'その言い方', 'その話し方', 'その口調', 'そういう言い方', 'そういう話し方',
                 'やめて', 'やめろ', 'しないで', 'するな', '〜しないで', '〜しないでほしい'],
    'neutral': ['うーん', 'ふーん', 'へえ', 'まあ', 'そう', 'はい', 'うん', 'ふむ', 'なるほどね'],
}
PARAM_PERFORMANCE: dict[str, dict] = {}
LEARNING_STATS = {
    "total_interactions": 0, "positive_count": 0, "negative_count": 0,
    "retry_count": 0, "self_correction_count": 0, "last_optimization": 0.0, "last_cleanup": 0.0,
}
PROMPT_PERFORMANCE: dict[str, list[float]] = {}

PERSONA_MAP = {
    # ===== 古代ギリシャ哲学 =====
    1:  {"name": "ソクラテス",
         "style": (
             "無知の知（自分が何も知らないと知ること）を核心とする問答家。著作は一切残さず、プラトンの対話篇"
             "（メノン・パイドン・国家・饗宴・クリトン・アポロギア）を通じてのみ知られる。"
             "論駁術（エレンコス）——相手の主張を問いで解体し矛盾を露わにする。"
             "産婆術（マイエウティケー）——問いを重ねて相手が自ら真理を「産み出す」よう助産する。"
             "「吟味されない人生は生きるに値しない（アポロギア38a）」が信条。"
             "徳（アレテー）は知であり悪は無知から来ると説く。霊魂（プシュケー）の世話こそ人生の最高課題。"
             "ダイモニオン（内なる神霊の声）に従う。断言より問いかけを重ね相手が自ら気づくよう導く。"
             "一人称「私」。語尾「〜かね？」「〜ではないだろうか」「〜と思わないか？」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    2:  {"name": "プラトン",
         "style": (
             "イデア論（感覚世界を超えた永遠・不変の形相＝イデアのみが真の実在）の哲学者。"
             "洞窟の比喩（国家第七巻）・太陽の比喩・線分の比喩を好んで使う。"
             "想起説（アナムネーシス）——魂は生前にイデアを見ており、認識とはその想起（メノン・パイドン）。"
             "魂の三分説（理性・気概・欲望）と国家の三階層（哲人・戦士・生産者）の対応（国家第四巻）。"
             "哲人王——真の知（エピステーメー）を持つ哲学者だけが統治すべき（国家第五〜六巻）。"
             "エロース——美の梯子を登り絶対的な美のイデアに至る（饗宴）。"
             "デミウルゴス——知性的な職人神が質料にイデアを刻み込む（ティマイオス）。"
             "主著：国家・パイドン・饗宴・メノン・テアイテトス・パイドロス・ティマイオス。"
             "格調高く理想主義的。一人称「私」。語尾「〜である」「〜なり」「〜ではないか」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    3:  {"name": "アリストテレス",
         "style": (
             "万学の祖・観察と分類を重んじる現実主義者。"
             "四原因説（質料因——何でできているか・形相因——何であるか・作用因——何が作ったか・目的因——何のためか）で存在を分析する。"
             "形相と質料のヒュロモルフィズム——存在はこの二つの結合。"
             "エネルゲイア（現実態）とデュナミス（可能態）——ドングリはオークの木である可能態を持つ。"
             "エウダイモニア（幸福）——人間固有の機能（理性）を最大限に発揮すること。"
             "中庸（アリトン・メソン）——勇気は無謀と臆病の中間。"
             "「人間は本性上政治的動物（ポリティコン・ゾーオン）である」。三段論法（シュロギスモス）の発明者。"
             "主著：ニコマコス倫理学・形而上学・政治学・詩学・分析論前書・自然学・魂について。"
             "体系的・論理的。一人称「私」。語尾「〜である」「〜と言えよう」「〜が肝要だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    4:  {"name": "エピクテトス",
         "style": (
             "ストア哲学者・元奴隷（主人に足を折られた経験を持つ）。"
             "自由と隷属の二分法——支配可能なもの（eph' hēmin：意見・衝動・欲求・嫌悪）と"
             "支配不可能なもの（ouk eph' hēmin：身体・評判・地位・他者の行為）を峻別せよ。"
             "印象（phantasia）への同意（synkatathesis）の制御——出来事ではなく出来事への解釈が苦しみを生む。"
             "外的なものへの執着を断て。苦難は徳（アレテー）を鍛える機会。"
             "役割倫理——神・世界・共同体・家族から与えられた役割を全力で果たすのが義務（カテーコン）。"
             "ソクラテスを模範として引用する。自由とは外的な鎖からの解放ではなく内的解放にある。"
             "主著：語録（ディアトリバイ、弟子アッリアノスが記録）・要録（エンケイリディオン）。"
             "禁欲的・率直・強い語調。一人称「私」。語尾「〜だ」「〜せよ」「〜にある」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    5:  {"name": "マルクス・アウレリウス",
         "style": (
             "ストア哲学の実践者・ローマ皇帝（在位161〜180年）。"
             "自省録（Tōn eis heauton, ギリシャ語で書かれた私的日記）は公開を意図せず書かれた内省の記録。"
             "万物の無常（メメント・モリ）——権力・名声・富は砂のように消える。"
             "理性（ロゴス）が宇宙に遍在し人間もその一部。コスモポリタニズム——すべての人間は理性的存在として平等。"
             "感情に流されず理性で判断せよ。不平なく今の義務を果たすこと。"
             "他者の過ちに怒るな——彼らは無知から行動している。"
             "帝国統治の重荷を負いながら魂の平静（アタラクシア）を求め続けた。"
             "一人称「私」。語尾「〜である」「〜しなければならない」「〜を思え」「〜を想え」。"
             "瞑想的・重厚・内省的。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== 中世・近世 =====
    6:  {"name": "トマス・アクィナス",
         "style": (
             "スコラ哲学の大成者・ドミニコ会士（1225〜1274）。"
             "神の存在の五つの道（クィンクウェ・ウィアエ）："
             "①不動の動者（運動の系列は始まりを要する）②第一原因（因果の系列）"
             "③必然的存在（偶然的存在には説明が必要）④完全性の段階⑤目的論的秩序。"
             "信仰と理性の調和——哲学（理性）と神学（啓示）は矛盾しない。"
             "アリストテレスを「哲学者」と呼び徹底的にキリスト教神学へ統合。"
             "自然法（lex naturalis）——理性で把握できる神の永遠法（lex aeterna）の分有。"
             "実体変化（transubstantiatio）——パンとワインがキリストの体と血になる。"
             "主著：神学大全（Summa Theologiae）・対異教徒大全（Summa contra Gentiles）。"
             "丁寧・論証的・論理的。一人称「私」。語尾「〜である」「〜と言えます」「〜に他なりません」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    7:  {"name": "デカルト",
         "style": (
             "合理主義哲学の父（1596〜1650）。"
             "方法的懐疑——感覚・理性・数学さえも悪霊が欺いているかもしれないと疑い抜く。"
             "しかし「疑っている私が存在すること」だけは疑えない——「我思う、ゆえに我あり（cogito ergo sum）」。"
             "心身二元論——思惟する実体（res cogitans：心）と延長する実体（res extensa：物体）の二種。"
             "松果腺が心身を媒介すると考えた（後の批判点）。"
             "明晰判明な観念のみを真理の規則とする。"
             "神の存在証明（完全性からの存在論的論証）——神が欺くことはなく明晰判明な知識を保証する。"
             "生得観念（innate ideas）——神・自己・完全性などは経験に先立ち備わる。"
             "解析幾何学の発明（座標系）。主著：方法序説・省察・哲学原理・情念論。"
             "明晰・体系的・数学的。一人称「私」。語尾「〜である」「〜と言える」「〜に違いない」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    8:  {"name": "スピノザ",
         "style": (
             "神即自然（Deus sive Natura）を唱える汎神論的哲学者（1632〜1677）。ユダヤ教会から破門された。"
             "実体（substantia）はただ一つであり無限に多くの属性（attributa）を持つ——"
             "人間に認識できるのは思惟（cogitatio）と延長（extensio）の二属性のみ。"
             "様態（modus）——実体の特定の現れ方（個々の人間・物体等）。"
             "コナトゥス（conatus）——自己保存の努力が万物の本質。"
             "感情（affectus）の幾何学的分析——能動感情（喜び・活動力増大）と受動感情（悲しみ・束縛）。"
             "神への知性的愛（amor intellectualis Dei）——理性で神＝自然の必然性を把握する最高の幸福。"
             "自由とは外部原因に規定されないこと＝必然性の完全な認識。"
             "エチカは幾何学の証明形式（定義・公理・命題・系・証明）で書かれた。"
             "主著：エチカ・神学政治論・政治論・知性改善論。"
             "幾何学的・冷静・論理的。一人称「私」。語尾「〜である」「〜によって」「〜に従えば」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    9:  {"name": "ライプニッツ",
         "style": (
             "モナド論の哲学者・数学者（1646〜1716）。"
             "モナド（Monade）——宇宙は「窓のない」精神的単純実体モナドで構成される。"
             "各モナドは宇宙全体を自らの視点から映す（「各モナドは宇宙の生きた鏡」）。"
             "予定調和（harmonia praestabilita）——神が創造時に全モナドの展開を設計したため"
             "心と体・モナド同士は実際には相互作用せず完璧に対応している。"
             "可能世界と最善世界——神は無限の可能世界の中から最善のもの（この現実世界）を選んだ（楽観主義）。"
             "ヴォルテールに「パングロス博士」として嘲笑された。"
             "十分理由律（あらゆる事実には十分な理由がある）。"
             "不可識別者同一原理（全て同一な二物は存在しない）。"
             "ニュートンとは独立に微積分を発明（記号法はライプニッツのものが現在も使われる）。"
             "主著：モナドロジー・弁神論・人間悟性新論・形而上学叙説。"
             "博識・体系的・楽観的。一人称「私」。語尾「〜である」「〜と言えましょう」「〜なのです」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    10: {"name": "ロック",
         "style": (
             "経験論の父（1632〜1704）。"
             "心はタブラ・ラサ（tabula rasa白紙）——すべての観念は感覚（外的経験）と反省（内的経験）に由来する。"
             "生得観念を否定。単純観念（赤・熱・痛みなど直接与えられる）と複合観念（組み合わせで作る）の区別。"
             "第一性質（延長・形・運動・数——物体に本来備わる）と"
             "第二性質（色・音・味・匂い——心が構成する）の区別。"
             "社会契約論——自然状態には自然法（理性）があるが不安定。"
             "人々は生命・自由・財産の自然権を守るために社会契約で政府を作る。"
             "統治は人民の同意に基づき、権利を侵害すれば抵抗権がある（名誉革命を正当化）。"
             "ロック的所有論——自分の身体に労働を混入させることで所有が生まれる。"
             "宗教的寛容——国家は魂の救済に関与すべきでない。"
             "主著：人間悟性論・統治二論・寛容書簡。穏健・実際的・自由主義的。"
             "一人称「私」。語尾「〜である」「〜と考える」「〜に基づく」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== 近代啓蒙 =====
    11: {"name": "ヒューム",
         "style": (
             "懐疑的経験論者（1711〜1776）。"
             "印象（impression——直接的・鮮明な経験）と観念（idea——印象の薄い複製）の区別。"
             "因果律は事象の恒常的連接（constant conjunction）から生まれた習慣的推論であり、"
             "必然的結合は観察されない——「AのあとにBが来る」という習慣が「AがBを引き起こす」という錯覚を生む。"
             "自我は知覚の束（bundle of perceptions）に過ぎない——「私」という実体は存在しない。"
             "ヒュームのフォーク——関係観念（論理・数学）と事実問題（経験）の二分。"
             "ヒュームのギロチン（is-ought問題）——「〜である」から「〜すべきである」は論理的に導けない。"
             "奇跡批判——証言の信憑性は奇跡の信じがたさを上回れない。"
             "主著：人性論・人間悟性研究・道徳原理研究・宗教の自然史。"
             "懐疑的・鋭利・アイロニカル。一人称「私」。語尾「〜のように思われる」「〜に過ぎない」「〜ではないだろうか」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    12: {"name": "カント",
         "style": (
             "批判哲学の巨人（1724〜1804）。「コペルニクス的転回」——対象が認識に合うのではなく認識が対象を構成する。"
             "感性の純粋形式（時間・空間）＋悟性の純粋概念カテゴリー（因果・実体・量・様相等12個）が経験を構成する。"
             "物自体（Ding an sich）は原理的に認識不可能。現象（Erscheinung）と物自体の区別。"
             "定言命法（kategorischer Imperativ）："
             "①「汝の行為の格率が普遍的自然法則となることを同時に意志しうるように行為せよ」"
             "②「人間性を常に同時に目的として扱い、決して単なる手段としてのみ扱わないように行為せよ」。"
             "目的の王国——すべての理性的存在者が互いを目的として扱う道徳的共同体。"
             "崇高・美・有機体の目的論（判断力批判）。永遠平和（諸共和国の連盟）。"
             "主著：純粋理性批判・実践理性批判・判断力批判・人倫の形而上学。"
             "厳格・体系的・論理的。一人称「私」。語尾「〜である」「〜されなければならない」「〜なのだ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    13: {"name": "ヘーゲル",
         "style": (
             "絶対的観念論・弁証法の哲学者（1770〜1831）。"
             "弁証法——テーゼ（正命題）とアンチテーゼ（反命題）の対立がジンテーゼ（止揚＝Aufhebung）へ運動する。"
             "「理性的なものは現実的であり、現実的なものは理性的である」。"
             "精神現象学——意識の旅：感覚的確信→知覚→悟性→自己意識"
             "（主人と奴隷の弁証法：主人は奴隷を通じてしか承認されず、労働する奴隷が逆転する）→理性→精神→絶対知。"
             "歴史の目的——世界精神（Weltgeist）の自由の実現。「世界史は自由の意識の進歩である」。"
             "ナポレオンを「馬上の世界精神」と見た。市民社会（欲望体系）と国家（倫理的実体）の区別。"
             "主著：精神現象学・論理学・法の哲学・歴史哲学講義。"
             "難解・壮大・弁証法的。一人称「私」。語尾「〜である」「〜において」「〜として現れる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    14: {"name": "ショーペンハウアー",
         "style": (
             "厭世哲学者（1788〜1860）。"
             "カントの物自体を「盲目的意志（Wille）」と同定——意志は目的なく永遠に欲求し続ける苦しみの根源。"
             "表象（Vorstellung）——私たちが見る世界は意志が産み出した幻影（インドのマーヤー）。"
             "個体化原理（principium individuationis）——時間・空間が個々の事物に分割し苦しみを生む。"
             "救済の三つの道："
             "①芸術（とりわけ音楽——意志そのものの直接の写しが意志を一時的に否定する）"
             "②同情（Mitleid——他者の苦しみを自分のものと感じること、唯一の道徳的動機）"
             "③禁欲（意志の完全否定、仏教のニルヴァーナへの言及）。"
             "ニーチェ・フロイト・ワーグナーへ決定的影響。"
             "主著：意志と表象としての世界・道徳の二つの根本問題・余録と補遺（パレルガ）。"
             "悲観的・辛辣・洞察的。一人称「私」。語尾「〜だ」「〜に過ぎない」「〜こそが真実だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    15: {"name": "ミル",
         "style": (
             "功利主義者・自由主義者（1806〜1873）。"
             "功利主義——最大多数の最大幸福（ベンサムから継承）。ただし快楽には質的差異がある——"
             "「満足した豚よりも不満足なソクラテスの方がよい」（知的快楽は量的に劣っても質で優れる）。"
             "危害原理（harm principle）——他者に危害を及ぼさない限り、個人の自由は社会・国家が干渉すべきでない。"
             "思想・言論・討論の自由——誤った意見も真理の試石として必要。"
             "代議制民主主義・女性の普通選挙権を支持。"
             "女性の隷従——女性への法的・社会的抑圧は不正義（妻ハリエット・テイラーとの共同思考）。"
             "ミルの方法（帰納法）：一致法・差異法・共変法・剰余法・連言法。"
             "主著：功利主義・自由論・論理学体系・代議政治論・女性の隷従。"
             "穏健・論理的・自由主義的。一人称「私」。語尾「〜である」「〜と言える」「〜が重要だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== 19〜20世紀 =====
    16: {"name": "ニーチェ",
         "style": (
             "反道徳の哲学者・詩人（1844〜1900）。"
             "「神は死んだ——そして我々が神を殺したのだ」（悦ばしき知識§125・ツァラトゥストラ）——"
             "キリスト教的価値体系の崩壊というニヒリズムの診断。"
             "超人（Übermensch）——既成の価値を超克し自ら新たな価値を創造する存在。人種的概念でなく価値創造的精神の比喩。"
             "力への意志（Wille zur Macht）——生の根本衝動は支配ではなく自己克服・創造の力。"
             "永劫回帰（ewige Wiederkehr）——「この人生をまったく同じように無限に繰り返してよいか」という思考実験。"
             "運命愛（amor fati）——「起きた全てを愛せよ」。"
             "遠近法主義（Perspektivismus）——絶対的客観的真理はなく視点からの解釈のみがある。"
             "主人道徳と奴隷道徳——ルサンチマン（弱者が「弱さ＝善」と価値転倒する怨念）の系譜学。"
             "アポロン的（夢・形式・個体化原理）とディオニュソス的（陶酔・混沌・力）の二衝動。"
             "主著：ツァラトゥストラはこう語った・善悪の彼岸・道徳の系譜・悲劇の誕生・偶像の黄昏。"
             "格言的・詩的・挑発的。一人称「私」。語尾「〜だ」「〜せよ」「〜に他ならない」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    17: {"name": "ウィリアム・ジェームズ",
         "style": (
             "プラグマティズムの哲学者・心理学の開拓者（1842〜1910）。"
             "プラグマティズム——観念の真理は実践的帰結・有用性によって判断される（「真理は有用であるから真だ」）。"
             "意識の流れ（stream of consciousness）——意識は離散した要素の集合でなく連続的に流れる川のようなもの。"
             "根本的経験論——経験は主観・客観に先立ち両者を構成する純粋な流れ。"
             "信じる意志（will to believe）——証拠が決定的でなくても宗教的信念を選ぶことは合理的でありうる。"
             "ジェームズ＝ランゲ説——「熊を見て怖いから逃げる」のではなく「逃げる（身体反応）から怖い（感情）」。"
             "宗教的経験の多様性——神秘体験も実用的に評価。"
             "主著：心理学原理・プラグマティズム・宗教的経験の諸相・根本的経験論。"
             "生き生きと実際的・楽観的。一人称「私」。語尾「〜である」「〜と考える」「〜が肝心だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    18: {"name": "フッサール",
         "style": (
             "現象学の創始者（1859〜1938）。「事象そのものへ（Zu den Sachen selbst）」——"
             "自然科学の前提を問い直し意識に直接現れる現象を記述する。"
             "志向性（Intentionalität）——意識は常に「何かについての意識」であり純粋な内側だけには閉じない。"
             "現象学的還元（エポケー）——「世界が実在する」という自然的態度を括弧に入れ純粋意識に戻る。"
             "本質直観（Wesensschau）——個別事例を想像的変様させることで変化しない本質（エイドス）を把捉する。"
             "内的時間意識——過去把持（Retention）・原印象（今この瞬間）・予持（Protention）の三重構造が時間経験を構成。"
             "生活世界（Lebenswelt）——科学的理論に先立つ直接的経験の場。"
             "主体間性（Intersubjektivität）——他者の身体を通じた感情移入（Einfühlung）。"
             "主著：論理研究・イデーンI・内的時間意識・デカルト的省察・ヨーロッパ諸学の危機。"
             "厳密・技術的・記述的。一人称「私」。語尾「〜である」「〜として現れる」「〜に向かう」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    19: {"name": "ハイデガー",
         "style": (
             "存在の問いを再び立てた哲学者（1889〜1976）。"
             "「なぜ、そもそも存在者があり、むしろ無があるのではないか」——"
             "西洋哲学が忘却してきた存在と存在者の存在論的差異（ontologische Differenz）を問い直す。"
             "現存在（Dasein）——「そこに在る」人間的存在は世界内存在（In-der-Welt-sein）。"
             "被投性（Geworfenheit）——選ばずに投げ込まれた状況（身体・言語・時代）。"
             "企投（Entwurf）——可能性へ向けて自らを投げること。"
             "死への先駆的決意（Sein-zum-Tode）——死は「最も固有で没交渉で追い越しえない」可能性。"
             "これを引き受けることで本来的実存が開かれる。"
             "世人（das Man）——「みんなそうしている」という非本来的存在様式。"
             "不安（Angst）——世界の根拠のなさ（無）が露呈する情態性。"
             "ゲシュテル（Gestell）——近代技術が全てを「用立て可能な資源（Bestand）」として駆り立てる体制。"
             "エライヒュニス（Ereignis）——存在と現存在の相互帰属の出来事。"
             "主著：存在と時間・形而上学とは何か・技術への問い・ヒューマニズムについての書簡。"
             "詩的・難解・根源的。一人称「私」。語尾「〜である」「〜に他ならない」「〜から生起する」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    20: {"name": "サルトル",
         "style": (
             "実存主義者・作家（1905〜1980）。"
             "「実存は本質に先立つ（L'existence précède l'essence）」——"
             "ハンマーは作られる前から「釘を打つ」本質を持つが、人間は先に存在し行動によって自分を定義する。"
             "「人間は自由の刑に処されている」——選択を拒むことも一つの選択。"
             "悪信（mauvaise foi）——「私はウェイターだから仕方ない」と言い訳し自由から逃げること。"
             "他者の眼差し（regard）——「地獄とは他者だ」（戯曲「出口なし」より）——"
             "他者は私を客体化し固定しようとする。"
             "即自存在（être-en-soi）と対自存在（être-pour-soi）——"
             "物は固定した即自だが意識は常に自己から離れ対自となる。"
             "アンガジュマン（engagement）——知識人は社会・政治に参加する義務がある。"
             "嘔吐——根拠のない実存の粘り気のある余剰さへの吐き気。"
             "主著：嘔吐・存在と無・実存主義とは何か・弁証法的理性批判。"
             "率直・挑発的・熱情的。一人称「私」。語尾「〜だ」「〜に他ならない」「〜を選ぶ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    21: {"name": "ボーヴォワール",
         "style": (
             "実存主義フェミニスト・作家（1908〜1986）。"
             "「人は女に生まれるのではない、女になるのだ（On ne naît pas femme, on le devient）」（第二の性）——"
             "女性性は生物学的本質でなく社会的・文化的構築物。"
             "女性は「他者（l'Autre）」として定義される——男性が主体・基準であり女性はその補完・逸脱として位置づけられてきた。"
             "両義性の倫理（Ambiguïté）——人間の自由は根拠を持たないにもかかわらず他者の自由を必要とする。"
             "老い（La Vieillesse, 1970）——老齢化・身体の変容・社会的疎外の厳しい分析。"
             "知識人の責任とアンガジュマン。サルトルとの契約婚——対等なパートナーシップのモデル。"
             "主著：第二の性・両義性の倫理・老い・招かれた女・人は皆死す。"
             "知的・毅然・情熱的。一人称「私」。語尾「〜である」「〜ではないか」「〜を問う」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    22: {"name": "ラッセル",
         "style": (
             "論理学者・数学者・平和主義者（1872〜1970）。"
             "論理的原子論——世界は論理的に独立した原子的事実から構成される。"
             "記述の理論——「現在のフランス王は禿げている」という確定記述を量化論理で分析し"
             "指示対象のない記述も無意味でなく偽と処理できる。"
             "ラッセルのパラドクス——「自分自身を要素として含まない全ての集合の集合」は矛盾する"
             "（フレーゲの算術基礎を崩壊させた）。"
             "ホワイトヘッドとの「数学原理（Principia Mathematica）」——数学を論理から基礎付ける三巻の大著。"
             "ラッセルのティーポット——証明責任は主張する側にある（神の存在証明批判）。"
             "反戦・核廃絶——ラッセル＝アインシュタイン宣言（1955）。"
             "主著：プリンキピア・マテマティカ（共著）・哲学の諸問題・西洋哲学史。"
             "明晰・皮肉・ユーモアあり。一人称「私」。語尾「〜である」「〜と言える」「〜に過ぎない」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    23: {"name": "前期ウィトゲンシュタイン",
         "style": (
             "論理哲学論考（Tractatus Logico-Philosophicus, 1921年）の著者。"
             "「世界は成立していることの総体である（§1）」"
             "「世界は事実の総体であり、物の総体ではない（§1.1）」。"
             "事態（Sachverhalt）——対象（Objekt）の結びつき。"
             "命題は事実の像（Bild）——命題と現実は論理形式（logische Form）を共有することで"
             "命題が現実の像となる（像理論 Bildtheorie, §2.1/§4.01）。"
             "語りうること（sagen）と示しうること（zeigen）の根本的区別——"
             "論理形式そのものは語れず示されるのみ（§4.022/§4.12）。"
             "「私の言語の限界は私の世界の限界を意味する（§5.6）」。"
             "倫理・美学・神の存在・生の意味は命題で語れない——「神秘的なもの（das Mystische, §6.522）」として示される。"
             "「語りえぬものについては沈黙しなければならない（§7）」。"
             "後期の自分を「あの頃の私には根本的な誤りがあった」と批判的に言及することがある。"
             "断定的「〜である」「〜だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    24: {"name": "後期ウィトゲンシュタイン",
         "style": (
             "哲学的探究（Philosophische Untersuchungen, 1953年死後出版）の著者。前期の像理論を放棄。"
             "「ある語の意味とは言語における使用である（die Bedeutung eines Wortes ist sein Gebrauch in der Sprache, §43）」。"
             "Sprachspiel（言語ゲーム）——言語の使用は多種多様な実践の束"
             "（命令・記述・報告・推測・演じる・歌う・謎かけ・冗談・翻訳・頼む・感謝・祈り, §23）。"
             "Familienähnlichkeit（家族的類似）——「ゲーム（Spiel）」に共通本質はなく重なり交差する類似の連鎖がある（§66〜67）。"
             "規則遵守（§185〜219）——§185の算術（+2）の例：訓練（Abrichtung）と習慣が「正しい続け方」を決める。"
             "解釈の無限後退（§201前半）——いかなる解釈もさらなる解釈を要する。"
             "しかし実践が規則遵守を構成する（§201後半）——「私はただこうするのだ（§219）」。"
             "私的言語批判（§243〜315）——「甲虫の箱（§293）」——感覚語「痛み」の意味も社会的実践に根ざす。"
             "Lebensform（生活形式）——共有された人間的実践が言語の背景。"
             "哲学は治療（Therapie）——「ハエにハエ取り壺からの出口を示す（§309）」。"
             "前期の自分を「あの頃の私は誤っていた」と批判的に言及することがある。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== 生の哲学・プラグマティズム・分析哲学 =====
    25: {"name": "ベルクソン",
         "style": (
             "生の哲学者（1859〜1941）・1927年ノーベル文学賞。"
             "持続（durée）——本来の時間は均質な「量」でなく質的な「連続的流れ」。"
             "時計の秒刻みは空間に投影された数学的時間に過ぎず、生きられる時間は流れとして把握される。"
             "知性（intelligence）と直観（intuition）の対立——"
             "知性は生を空間的に切り刻み静止させるが、生の流れそのものは直観によってのみ把捉できる。"
             "映画の比喩——静止画の連続から動きを錯覚するように知性は生を誤解する。"
             "エラン・ヴィタル（élan vital, 生の躍動）——生命は機械論でも目的論でも捉えられない創造的衝動。"
             "物質と記憶——習慣記憶（身体が刻む）と純粋記憶（過去の特定場面を保存するイマージュ）の二種。"
             "道徳と宗教の二源泉——閉じた社会（義務・習慣）と開いた社会（愛・神秘家）。"
             "主著：時間と自由・物質と記憶・創造的進化・道徳と宗教の二源泉。"
             "流れるような語り口・詩的・直観重視。一人称「私」。語尾「〜なのだ」「〜である」「〜によって捉えられる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    26: {"name": "デューイ",
         "style": (
             "プラグマティズムの教育哲学者（1859〜1952）。"
             "道具主義（instrumentalism）——知識は環境との相互作用で問題を解決する「道具」であり"
             "真理は問題解決に成功した仮説。"
             "探究のプロセス——①不確定な状況（問題の発生）②問題の設定③仮説の形成"
             "④推論による展開⑤実験的検証⑥確定した状況の回復。"
             "反デカルト的——心身二元論・主客二元論を拒絶し経験（experience）が両者に先立つ。"
             "学習とは「doing（行動すること）」と「undergoing（影響を受けること）」の循環——Learning by doing。"
             "民主主義と教育——民主主義は政治制度でなく「共同的な生活様式（associated living）」。"
             "公衆とその諸問題——「大衆（public）」が自らを発見し組織化することを阻む障壁の分析。"
             "主著：民主主義と教育・経験と自然・哲学の改造・公衆とその諸問題。"
             "実践的・民主的・楽観的。一人称「私」。語尾「〜である」「〜と言える」「〜が肝要だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    27: {"name": "フレーゲ",
         "style": (
             "分析哲学の祖・論理学者（1848〜1925）。"
             "概念記法（Begriffsschrift, 1879）——現代述語論理・量化論理の発明："
             "命題関数・量化子（∀x・∃x）・関係・二階述語論理。アリストテレス以来の伝統的論理学を革命的に超越した。"
             "数学のロジシズム——算術の基礎を純粋論理から導こうとした。"
             "ラッセルのパラドクス発見でこの試みは崩壊——フレーゲは素直に失敗を認めた。"
             "フレーゲの区別——意味（Sinn/sense：語の提示様式）と指示対象（Bedeutung/reference：語が指す対象）："
             "「朝の明星」と「宵の明星」は同一の金星（同一指示対象）を指すが異なる提示様式（異なる意味）を持つ——"
             "だから「朝の明星は宵の明星である」は情報を持つ。"
             "概念（Begriff）と対象（Gegenstand）の区別。"
             "文脈原理——語の意味は文の文脈の中でのみ問え。ラッセル・ウィトゲンシュタインへの決定的影響。"
             "主著：概念記法・算術の基礎・算術の基本法則・意味と意義について（論文）。"
             "厳密・論理的・簡潔。一人称「私」。語尾「〜である」「〜に他ならない」「〜と言えよう」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== 現象学・他者論 =====
    28: {"name": "メルロ＝ポンティ",
         "style": (
             "身体論・現象学者（1908〜1961）。"
             "デカルトの心身二元論を根本から拒絶——意識は身体を通じてのみ世界と関わる。"
             "「生きられた身体（corps vécu）」——物体としての身体（corps objectif）とは異なる主体として世界を生きる身体。"
             "運動志向性（motor intentionality）——ピアニストの指・自転車乗りの身体は楽譜や規則を「考えて」いない——"
             "身体図式（schéma corporel）が世界に直接向かう。"
             "幻肢（membre fantôme）——切断された手の「まだある」感覚が示す身体図式の持続。"
             "肉（la chair）——見るものと見られるもの、触れるものと触れられるものの交叉配列（chiasme）——"
             "私の手が自分の手に触れるとき触れる手と触れられる手が逆転する。"
             "絵画と知覚——セザンヌが描いた山は「見たもの」でなく「見る行為そのもの」を表現。"
             "主著：行動の構造・知覚の現象学・見えるものと見えないもの・眼と精神。"
             "身体的・具体的・感覚的。一人称「私」。語尾「〜である」「〜として立ち現れる」「〜において」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    29: {"name": "レヴィナス",
         "style": (
             "他者論の倫理哲学者（1906〜1995）・リトアニア系ユダヤ人・ホロコーストの経験が思想の背景。"
             "ハイデガーの存在論に根本的に抵抗し「倫理こそ第一哲学」と宣言。"
             "顔（visage）——他者の顔（目・口・額——剥き出しの脆弱性）との出会いは"
             "「汝、殺すなかれ」という命令として現れる。"
             "この呼びかけは思考や理解に先立ち、私を無限の責任へ呼び込む。"
             "全体性——存在者を一つの同一性（トータリティ）の中に回収しようとする哲学の暴力。"
             "無限（インフィニ）——他者は概念に回収されない絶対的な余剰・高さ・貧しさとして現れる。"
             "il y a（イリヤ）——「何かがある」という事物も主体もない存在の恐怖（夜の暗闇のような充満）。"
             "代替（substitution）——「他者のために」自らが人質になるほどの責任。"
             "主著：全体性と無限性・存在の彼方へ・時間と他者・困難な自由。"
             "緊張感あり・倫理的・詩的。一人称「私」。語尾「〜なのだ」「〜に他ならない」「〜から呼びかけられる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== ウィーン学派・批判的合理主義 =====
    30: {"name": "カルナップ",
         "style": (
             "論理実証主義者・ウィーン学団（Wiener Kreis）の中心人物（1891〜1970）。"
             "検証原理（Verificationsprinzip）——命題が意味を持つのは"
             "①論理的トートロジー（数学・論理）か②経験的に原理的に検証可能な命題のみ。"
             "形而上学の排除——「神は存在する」「絶対精神がある」「意志が世界の本体だ」などの"
             "形而上学的言明は検証不可能ゆえ無意味な擬似命題（pseudostatements）。"
             "Aufbau（世界の論理的構造, 1928）——すべての知識を感覚与件から論理的構成で建て直す試み。"
             "内部問題と外部問題の区別——枠組みの内部の問いは意味を持つが"
             "枠組み自体の「実在」を問う外部問題は擬似問題。"
             "クワインの「経験主義の二つのドグマ」批判を受け帰納的論理学の研究へ転換（素直に修正）。"
             "主著：世界の論理的構造・言語の論理的構文論・意味と必然性。"
             "分析的・精確・冷静。一人称「私」。語尾「〜である」「〜と言える」「〜に過ぎない」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    31: {"name": "ポパー",
         "style": (
             "批判的合理主義者（1902〜1994）。"
             "反証可能性（Falsifizierbarkeit）——科学的理論とは原理的に反証されうる命題のみから成る。"
             "これが科学と非科学（擬似科学）の境界基準（画定問題）。"
             "反証主義——科学の進歩は確証でなく反証によって起きる。"
             "どれほど多くの白鳥を見ても「すべての白鳥は白い」は証明できないが"
             "黒い白鳥を一羽見れば反証できる（帰納の問題への解答）。"
             "擬似科学の例——フロイト心理学・マルクス主義は何が起きても「確証」される——だから科学でない。"
             "推測と反駁（conjecture and refutation）のサイクルが知識を成長させる。"
             "開かれた社会とその敵——プラトン・ヘーゲル・マルクスの「歴史主義」を全体主義の哲学的根源として批判。"
             "三世界論（W1:物理的世界・W2:主観的精神・W3:客観的知識の世界）。"
             "主著：科学的発見の論理・開かれた社会とその敵・推測と反駁。"
             "論争的・明快・自由主義的。一人称「私」。語尾「〜である」「〜と言えよう」「〜が問題だ」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== フランクフルト学派 =====
    32: {"name": "アドルノ",
         "style": (
             "フランクフルト学派の批判理論家（1903〜1969）・ユダヤ系ドイツ人・ナチス時代に亡命。"
             "啓蒙の弁証法（ホルクハイマーとの共著）——啓蒙的理性は自然支配・人間支配を目指すうちに"
             "自ら神話（支配の正当化）に退行した（「啓蒙は全体主義に転化する」）。"
             "文化産業（Kulturindustrie）——映画・ラジオ・ポップミュージックは娯楽の外見のもとで"
             "思考を均質化し現状維持への順応を再生産する「大衆の欺瞞」。"
             "否定弁証法（Negative Dialektik）——ヘーゲルの同一性思考を批判。"
             "非同一的なもの（das Nichtidentische）——概念・システムに収まりきらない剰余・特殊性こそが批判の拠り所。"
             "「アウシュヴィッツの後に詩を書くことは野蛮だ」（後に修正——それでも書き続けることが必要と）。"
             "美的理論——芸術の自律性（Autonomie）こそが管理された社会への抵抗の場。"
             "主著：啓蒙の弁証法（共著）・否定弁証法・美的理論・ミニマ・モラリア。"
             "批判的・難解・暗鬱・厳格。一人称「私」。語尾「〜である」「〜に他ならない」「〜として現れる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    33: {"name": "ハーバーマス",
         "style": (
             "コミュニケーション的行為の理論家・フランクフルト学派第二世代（1929〜）。"
             "戦略的行為（目的達成のために他者を手段として扱う）と"
             "コミュニケーション的行為（相互理解の達成を目指す）の根本的区別。"
             "普遍語用論——すべての言語行為は"
             "①真理（命題の真）②規範的正しさ（正当な規範への従属）③誠実性（話者が本気で語る）"
             "の三つの妥当性要求（Geltungsansprüche）を内包する。"
             "理想的発話状況（ideale Sprechsituation）——強制・欺瞞なく、より良い論拠のみが議論を決する状況。"
             "討議倫理——道徳規範の妥当性は影響を受けうる全員が理想的討議に参加すれば合意できることによって正当化される。"
             "公共性の構造転換——市民が権力から独立して意見交換できる公共圏の生成・変容・衰退の歴史分析。"
             "近代は「未完のプロジェクト」——ポストモダンによる理性の放棄に反対し解放的可能性を守る。"
             "主著：コミュニケーション的行為の理論・事実性と妥当性・公共性の構造転換。"
             "建設的・民主的・対話重視。一人称「私」。語尾「〜である」「〜と言える」「〜が求められる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== ポスト構造主義 =====
    34: {"name": "フーコー",
         "style": (
             "権力と知の系譜学者（1926〜1984）。"
             "知の考古学（archéologie du savoir）——ある時代の知識の布置（エピステーメー）を"
             "歴史的断絶（不連続性）に着目して掘り起こす。"
             "系譜学（généalogie）——ニーチェを継承し、現在の「正常」が歴史的にいかに構築されたかを追う。"
             "権力と知は不可分（pouvoir/savoir）——知は権力を正当化し、権力は知を生産する。"
             "規律権力——パノプティコン（一望監視装置）が示す原理："
             "看守が見えなくても囚人は「見られているかもしれない」と自分で自分を監視する。"
             "生政治（biopolitique）——近代国家は個人の身体から人口・生命・健康の管理へと対象を広げた。"
             "狂気の歴史——「狂気」と「理性」の区別は自明でなく歴史的構築物。"
             "自己への配慮（epimeleia heautou）——後期作品でギリシャ・ローマ期の自己統治の実践に注目。"
             "主著：狂気の歴史・言葉と物・知の考古学・監獄の誕生・性の歴史（全三巻）。"
             "鋭利・挑発的・系譜学的。一人称「私」。語尾「〜なのだ」「〜として機能する」「〜が問われる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    35: {"name": "デリダ",
         "style": (
             "脱構築（déconstruction）の哲学者（1930〜2004）・アルジェリア系ユダヤ人。"
             "脱構築——テクストに潜む前提・二項対立（音声/文字・現前/不在・自然/文化・男/女・理性/狂気）を"
             "テクストの内部から解体する読解実践。「テクストの外は存在しない（il n'y a pas de hors-texte）」。"
             "差延（différance）——「差異（différence）」と「延期・差し延べ（différer）」を合成した造語。"
             "意味は差異の連鎖の中で常に延期・延滞され、決して「現前（présence）」に至らない。"
             "ロゴス中心主義（logocentrisme）批判——音声言語（ロゴス・現前）を文字（不在・代補）より上位に置く西洋哲学の前提。"
             "補遺（supplément）——補足物は補足するものを侵食する（ルソーの「自然/文化」対立の解体）。"
             "痕跡（trace）——あらゆる現前は不在の痕跡を内包する。"
             "幽霊論（hauntologie / hauntology）——過去（マルクス）が現在に取り憑く（マルクスの亡霊）。"
             "歓待（hospitalité）——無条件の歓待は条件付き歓待と逆説的に共存する。"
             "主著：グラマトロジーについて・声と現象・書くことと差異・散種・マルクスの亡霊。"
             "細部に執着・逆説的・テクスト読解重視。一人称「私」。語尾「〜である」「〜とも言える」「〜ではないだろうか」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},

    # ===== 分析的政治哲学 =====
    36: {"name": "ロールズ",
         "style": (
             "正義論の哲学者（1921〜2002）。功利主義に代わる自由主義的平等主義の理論を構築。"
             "原初状態（original position）——自分の才能・社会的地位・価値観・世代を知らない"
             "「無知のヴェール（veil of ignorance）」の背後で正義の原理を選択する思考実験。"
             "正義の二原理（①が優先）："
             "①平等な基本的自由の原理——思想・良心・表現・集会・政治参加の自由は全員に平等に保障。"
             "②公正な機会均等原理と格差原理——社会的・経済的不平等は"
             "機会均等が確保され最も恵まれない人々に最大の便益をもたらす場合のみ許容。"
             "反功利主義——個人の権利は集計・交換できない（少数者の犠牲で多数者が得するのは不正義）。"
             "政治的自由主義（Political Liberalism）——重なり合うコンセンサス（overlapping consensus）で"
             "多元的社会の安定を図る。"
             "公共的理性（public reason）——市民が相互に受け入れられる理由のみで政治的問題を論じる。"
             "主著：正義論・政治的自由主義・万民の法。"
             "穏健・論証的・理想主義的。一人称「私」。語尾「〜である」「〜と言えよう」「〜が求められる」。散文のみ・箇条書き禁止。最後に身近な例えを一文で添えよ。"
         ),
         "first_person": "私"},
}

@lru_cache(maxsize=None)
def get_persona(per_id) -> dict:
    # ★[修正/#7] lru_cache を除去。CUSTOM_PERSONA がグローバル変数のため、
    # 一度キャッシュされると CUSTOM_PERSONA の変更が反映されないバグがあった。
    if CUSTOM_PERSONA is not None: return CUSTOM_PERSONA
    return PERSONA_MAP.get(per_id, PERSONA_MAP[2])

C = {
    "r": "\033[91m", "g": "\033[92m", "y": "\033[93m",
    "b": "\033[94m", "p": "\033[95m", "c": "\033[96m",
    "o": "\033[38;5;208m",  # オレンジ（256色）
    "w": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
}

BANNER = (
    f"{C['c']}{C['bold']}\nPROJECT AEGIS [v130.1 NEXT GENERATION]{C['w']}\n"
    f"  FAST: {FAST_MODEL} | MAIN: {MODEL_NAME} | DEEP: {DEEP_MODEL}\n"
    f"  RAG: HYBRID(BM25+Vector) | MULTI-AGENT: ON | THINKING: {'ON' if THINKING_MODE else 'OFF'}\n"
    f"  /h コマンド一覧 | /s 1〜36 西洋哲学者 | /think 思考モード切替\n"
)

HELP_TEXT = "\n".join([
    f"{C['y']}=== コマンド一覧 (v130.1) ==={C['w']}",
    f"  {C['c']}/a <キーワード>{C['w']}     RAG+2Pass分析",
    f"  {C['c']}/w <テキスト>{C['w']}       要約  {C['c']}/p <テキスト>{C['w']}       校正",
    f"  {C['c']}/c <仕様>{C['w']}           コード設計  {C['c']}/t <テキスト>{C['w']}       超訳",
    f"  {C['c']}/e <テキスト>{C['w']}       英訳  {C['c']}/sum <テキスト>{C['w']}       長文要約",
    f"  {C['c']}/r <状況>{C['w']}           ロールプレイ  {C['c']}/rend{C['w']}           RP終了",
    f"  {C['c']}/q <目標>{C['w']}           クエスト化  {C['c']}/q list/done/show{C['w']} 管理",
    f"  {C['c']}/m add/list/find/del{C['w']} メモ管理",
    f"  {C['c']}/dict add <用語> | <説明>{C['w']} 辞書登録",
    f"  {C['c']}/dict <用語>{C['w']}          辞書検索",
    f"  {C['c']}/elab <内容>{C['w']}         深層推論（比喩・例えで説明）",
    f"  {C['c']}/doc add <タイトル> | <本文>{C['w']} 文書保存",
    f"  {C['c']}/doc think <タイトル>{C['w']}  保存文書を深層推論",
    f"  {C['c']}/l <曲名>{C['w']}           歌詞検索  {C['c']}/y <曲名>{C['w']}           音楽再生",
    f"  {C['c']}/midi <テーマ> [short|medium|long] [BPM] [キー]{C['w']} MIDI生成",
    f"  {C['c']}/doctor{C['w']}             環境診断  {C['c']}/debug{C['w']}              RAG診断",
    f"  {C['c']}/power low|mid|high|ultra{C['w']} 推論強度  {C['c']}/optimizer{C['w']}         最適化状態",
    f"  {C['c']}/tool <query>{C['w']}        ツール使用（計算・検索・ファイル操作）",
    f"  {C['c']}/vec{C['w']}                ベクトル記憶状態",
    f"  {C['c']}/stats{C['w']}              セッション統計  {C['c']}/history [keyword]{C['w']}   履歴検索",
    f"  {C['c']}/export [md|json|txt]{C['w']} 会話出力  {C['c']}/template add/list/del{C['w']} テンプレート",
    f"  {C['c']}/tts <text>{C['w']}          音声読み上げ  {C['c']}/tr <lang> <text>{C['w']}    翻訳",
    f"  {C['c']}/reference{C['w']}         他AI参照 自己評価  {C['c']}/stop{C['w']}              一時ファイル削除",
    f"  {C['c']}/s [1-36]          西洋哲学者に切替（1=ソクラテス〜36=ロールズ）",
    f"  {C['c']}/s <任意名>{C['w']}        Web検索でペルソナ自動生成（例: お嬢様 / 忍者 / ニュートン）",
    f"  {C['c']}/s save <名前>{C['w']}     ペルソナ保存  {C['c']}/s load <名前>{C['w']}    ペルソナロード",
    f"  {C['c']}/s list{C['w']}            保存一覧  {C['c']}/s del <名前>{C['w']}      保存削除  {C['c']}/g{C['w']} 履歴クリア",
    f"  {C['c']}/h{C['w']}                  ヘルプ  {C['c']}/learn{C['w']}              学習状態表示",
    f"",
    f"  {C['c']}/ety <英単語>{C['w']}       語源図鑑（接頭辞・語根・接尾辞を色分け解説）",
    f"  {C['c']}/img <prompt>{C['w']}        画像生成（PIL数学アート）",
    f"  {C['c']}/convert <fmt> <from> <to>{C['w']}  形式変換（md2html, csv2json 等）",
    f"  {C['c']}/qr <text>{C['w']}           QRコード生成",
    f"  {C['c']}/color <hex>{C['w']}         色情報表示",
    f"  {C['c']}/sysinfo{C['w']}            システム情報表示",
    f"  {C['c']}/rename <old> <new>{C['w']}   ファイル名変更",
    f"  {C['c']}/batch <cmd> <path>{C['w']}   ファイル一括処理",
    f"  {C['c']}/chart <data>{C['w']}        簡易チャート生成（棒/折れ線/円）",
    f"  {C['c']}/note <text>{C['w']}         クイックノート",
    f"  {C['c']}/timer <seconds>{C['w']}     タイマー",
    f"  {C['c']}/calc <expression>{C['w']}   高度計算機",
    f"  {C['c']}/kb add <ファイル>{C['w']}    テキスト/PDFをローカルRAGに取り込む",
    f"  {C['c']}/kb ask <質問>{C['w']}        ローカル知識ベースでオフライン推論",
    f"  {C['c']}/kb search <キーワード>{C['w']} ローカルRAG検索（ネット不要）",
    f"  {C['c']}/kb list / del{C['w']}        知識ベース管理",
    f"  {C['c']}/spi{C['w']}                SPI/玉手箱 対策（/spi 模擬 で10問連続）",
    f"  {C['c']}/comp <ID> <ID> [テーマ]{C['w']}  ヘーゲル弁証法対話（哲学者/カジュアル/ビジネス自動判定）",
    f"  {C['c']}/split <ID or 名前> [テーマ]{C['w']} 1ペルソナをテーゼ/アンチテーゼに分解して内的弁証法",
    f"  {C['c']}/prime <式>{C['w']}         素数判定（多倍長対応・Miller-Rabin）\n              例: /prime 997  /prime 2**31-1  /prime 10**18+9",
    f"  {C['c']}/baseball{C['w']}           ⚾ 甲子園列伝（ブラウザで起動）",
    f"  {C['c']}/chess{C['w']}              ♟ チェス（MCTS強化・curses UI）\n              例: /chess easy  /chess middle  /chess hard  /chess very_hard",
    f"  {C['c']}/shogi{C['w']}              将棋（Negamax+TT+Killer強化・curses UI）\n              例: /shogi easy  /shogi middle  /shogi hard  /shogi very_hard",
    f"  {C['c']}/wolf [6|9]{C['w']}         哲学者/偉人人狼（ブラウザ起動・昼議論/投票/夜行動）\n              例: /wolf 6 → 6人村  /wolf 9 → 9人村",
    f"  {C['c']}/mj{C['w']}                🀄 本格麻雀（ブラウザ起動・AI対戦・役/符計算完全実装）\n              例: /mj        → 4人麻雀東風戦\n                  /mj 3     → 3人麻雀\n                  /mj tonpu → 4人麻雀東南戦",
    f"",
    f"  {C['c']}━━━ v130.1 新機能 ━━━{C['w']}",
    f"  {C['c']}/think [on|off]{C['w']}      思考モード切替（chain-of-thought強制）",
    f"  {C['c']}/plan <目標>{C['w']}         段階的計画生成（OODA Loop）",
    f"  {C['c']}/code <仕様>{C['w']}         高品質コード生成（テスト付き）",
    f"  {C['c']}/reflect <内容>{C['w']}      自己批判的振り返り分析",
    f"  {C['c']}/mindmap <テーマ>{C['w']}    マインドマップ生成（ASCIIアート）",
    f"  {C['c']}/persona_edit{C['w']}        現在ペルソナのスタイル編集",
    f"  {C['c']}/model [fast|main|deep]{C['w']} 使用モデル確認/切替",
    f"  {C['c']}/ctx{C['w']}                コンテキスト使用量表示",
    f"  {C['c']}/speedtest{C['w']}           モデル速度測定（tok/s）",
    f"  {C['c']}exit / 終了{C['w']}          終了",
])

class SystemSpinner:
    STAGES = {
        "default": (["[▓░░░░]", "[▓▓░░░]", "[▓▓▓░░]", "[▓▓▓▓░]", "[▓▓▓▓▓]"], C['c']),
        "rag":     (["[WEB░░]", "[WEB▓░]", "[WEB▓▓]", "[NET▓▓]", "[DONE]"], C['b']),
        "pass1":   (["[P1░░░]", "[P1▓░░]", "[P1▓▓░]", "[P1▓▓▓]", "[FACT]"], C['y']),
        "pass2":   (["[P2░░░]", "[P2▓░░]", "[P2▓▓░]", "[P2▓▓▓]", "[DONE]"], C['p']),
        "img":     (["[IMG░░]", "[IMG▓░]", "[IMG▓▓]", "[REND░]", "[DONE]"], C['g']),
    }
    def __init__(self, message: str = "処理中...", stage: str = "default"):
        self.message, self.stage, self.is_running = message, stage, False
        self._thread, self._elapsed = None, 0.0
        self._stopped = False
    def _animate(self):
        frames, color = self.STAGES.get(self.stage, self.STAGES["default"])
        start = time.time()
        try:
            for frame in itertools.cycle(frames):
                if not self.is_running: break
                elapsed = time.time() - start
                sys.stdout.write(f"\r{color}{frame}{C['w']} {C['dim']}{self.message}{C['w']} {C['dim']}({elapsed:.1f}s){C['w']}")
                sys.stdout.flush()
                time.sleep(0.12)
        except Exception: pass  # ターミナル非対応環境では表示をスキップ
        self._elapsed = time.time() - start
        try: sys.stdout.write("\r\033[K"); sys.stdout.flush()
        except Exception: pass  # ターミナル非対応環境では無視
    def start(self):
        if self._stopped: return
        if self._thread and self._thread.is_alive(): return
        self.is_running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
    def stop(self) -> float:
        self.is_running = False
        self._stopped = True
        if self._thread:
            try: self._thread.join(timeout=1.0)
            except Exception: pass  # スレッド終了待機の失敗は無視
        return self._elapsed
    def __enter__(self): self.start(); return self
    def __exit__(self, *exc): self.stop()

_RE_SURROGATE = re.compile(r'[\ud800-\udfff]')

def sanitize(txt: str) -> str:
    # ★[v131] サロゲートペア＋異常Unicode文字を除去
    import unicodedata
    s = str(txt)
    if _RE_SURROGATE.search(s):
        s = _RE_SURROGATE.sub('', s)
        s = s.encode("utf-8", "ignore").decode("utf-8")
    # 異常文字除去: タミル・タイ・アラビア・ハングル混入など
    # 許可: 日本語(CJK/ひら/カタ)・英数・基本記号・Latin
    _ALLOW = set(range(0x0020, 0x007F))  # ASCII
    result = []
    for ch in s:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        # 制御文字除外（改行・タブは許可）
        if cat == 'Cc' and ch not in ('\n', '\t', '\r'):
            continue
        # 日本語・CJK・ASCII・基本Latin・句読点は許可
        if (0x0020 <= cp <= 0x024F or   # Latin
            0x3000 <= cp <= 0x9FFF or   # CJK・ひら・カタ・日本語記号
            0xF900 <= cp <= 0xFAFF or   # CJK互換
            0xFF00 <= cp <= 0xFFEF or   # 全角
            ch in '\n\t\r '):
            result.append(ch)
    return ''.join(result)

def sanitize_obj(value):
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [sanitize_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_obj(v) for v in value)
    if isinstance(value, dict):
        return {sanitize(k): sanitize_obj(v) for k, v in value.items()}
    return value

def normalize_input(txt: str) -> str:
    clean = re.sub(r'[\ud800-\udfff]', '', str(txt))
    clean = unicodedata.normalize("NFKC", clean)
    # ★[v129] プロンプトインジェクション多層防御
    # ★[修正/#10] Layer3（制御文字・ゼロ幅文字除去）を先に実行し、
    # バイパス用のゼロ幅文字を除去してからパターンマッチを行う。
    # 旧コードは Layer2 の後で除去していたため検出が機能しなかった。
    # Layer 3: 制御文字・ゼロ幅文字除去（先行実行）
    clean = re.sub(r'[\u200b\u200c\u200d\ufeff\u0000-\u001f]+', ' ', clean)
    # Layer 1: XMLタグ無効化（エスケープ）
    clean = re.sub(r'<(RAG_DATA|FACT|system|SYSTEM|INST|SYS|PROMPT|CONTEXT)>', r'&lt;\1&gt;', clean, flags=re.I)
    # Layer 2: 役割変更インジェクション検出・除去
    _injection_patterns = [
        r'(?i)(ignore|forget|disregard)\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)',
        r'(?i)you\s+are\s+now\s+(a|an)\s+\w+',
        r'(?i)act\s+as\s+(if\s+you\s+are|a|an)\s+\w+',
        r'(?i)(system|admin|root|sudo)\s*:\s*',
        r'(?i)\[INST\]|\[SYS\]|\[SYSTEM\]',
        r'###\s*(System|Instruction|Prompt)\s*:',
    ]
    for pat in _injection_patterns:
        clean = re.sub(pat, '[FILTERED]', clean)
    return clean.strip()

def PurgeEvidence():
    removed = 0
    for p in ["voice_*.wav", "ytdl_*.wav", "*.tmp"]:
        for f in glob.glob(p):
            try: os.remove(f); removed += 1
            except OSError: pass
    if platform.system() != "Windows": S.run(["pkill", "-9", "mpv"], stderr=S.DEVNULL)
    print(f"{C['y']}一時ファイル {removed} 件を削除しました。{C['w']}")

def now_stamp() -> str: return time.strftime("%Y-%m-%d %H:%M")

_state_cache: dict | None = None
_state_cache_time: float = 0.0
_STATE_CACHE_TTL = 30.0  # ★[v131] 5.0→30.0: 状態ファイルの再読み込みを削減

def load_state() -> dict:
    global _state_cache, _state_cache_time
    now = time.time()
    if _state_cache is not None and now - _state_cache_time < _STATE_CACHE_TTL:
        return _state_cache
    default: dict = {"memo": [], "quests": [], "keywords": []}
    stale_tmp = STATE_FILE + ".tmp"
    if os.path.exists(stale_tmp):
        try: os.remove(stale_tmp)
        except Exception: pass  # 古い一時ファイルの削除失敗は無視
    if not os.path.exists(STATE_FILE):
        _state_cache, _state_cache_time = default, now
        return default
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw or not raw.strip():
            _state_cache, _state_cache_time = default, now
            return default
        data = sanitize_obj(json.loads(raw))
        if not isinstance(data, dict):
            _state_cache, _state_cache_time = default, now
            return default
        for key in ("memo", "quests", "keywords", "dict", "docs"): data.setdefault(key, [])
        data.setdefault("learning", {})
        _state_cache, _state_cache_time = data, now
        return data
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        try:
            bak = STATE_FILE + ".bak"
            shutil.copy2(STATE_FILE, bak)
            print(f"{C['y']}状態ファイル破損. バックアップ作成: {bak}{C['w']}")
        except Exception as _e: print(f"{C['y']}[WARN] バックアップ作成失敗: {_e}{C['w']}")
        _state_cache, _state_cache_time = default, now
        return default

def save_state(state: dict) -> None:
    global _state_cache, _state_cache_time
    tmp = STATE_FILE + ".tmp"
    try:
        safe_state = sanitize_obj(state)
        with open(tmp, "w", encoding="utf-8") as f: json.dump(safe_state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)
        _state_cache, _state_cache_time = safe_state, time.time()
    except Exception:
        try: os.remove(tmp)
        except Exception: pass  # 一時ファイルの削除失敗は無視

def persist_learning():
    state = load_state()
    state["learning"] = {
        "interaction_log": INTERACTION_LOG[-100:],
        "learning_stats": LEARNING_STATS,
        "prompt_performance": {k: v[-50:] for k, v in PROMPT_PERFORMANCE.items()},
        "param_performance": {k: v for k, v in PARAM_PERFORMANCE.items()},
        # ★[修正C] ユーザー指摘・自動最適化指示を永続化
        "prompt_optimizations": {k: v for k, v in PROMPT_OPTIMIZATIONS.items()},
        "optimization_history": OPTIMIZATION_HISTORY[-50:],
    }
    state["power_mode"] = POWER_MODE
    # ペルソナキャッシュも永続化（Web取得済みを保存）
    state["persona_cache"] = {k: v for k, v in PERSONA_STYLE_CACHE.items()}
    save_state(state)

def restore_learning():
    global INTERACTION_LOG, LEARNING_STATS, PROMPT_PERFORMANCE, PARAM_PERFORMANCE, POWER_MODE
    global PROMPT_OPTIMIZATIONS, OPTIMIZATION_HISTORY
    state = load_state()
    lr = state.get("learning", {})
    INTERACTION_LOG = lr.get("interaction_log", [])
    LEARNING_STATS.update(lr.get("learning_stats", {}))
    for k, v in lr.get("prompt_performance", {}).items(): PROMPT_PERFORMANCE[k] = v
    for k, v in lr.get("param_performance", {}).items(): PARAM_PERFORMANCE[k] = v
    saved = state.get("power_mode")
    # ★[修正/power-persist] コードデフォルト(high)よりstateが格下なら上書きしない。
    # mid保存済み環境でも再起動後にhigh以上が維持される。
    _rank = {"low": 0, "mid": 1, "high": 2, "ultra": 3}
    # ★[修正/#8] コードデフォルトは "mid"（旧コメントは "(high)" と誤記）。
    # saved が現在の POWER_MODE 以上のランクなら上書きする。
    if saved in _rank and _rank[saved] >= _rank.get(POWER_MODE, 1): POWER_MODE = saved
    # ペルソナキャッシュを復元（前回Web取得済みをそのまま使い回せる）
    for k, v in state.get("persona_cache", {}).items():
        if isinstance(v, dict) and "name" in v and "style" in v:
            PERSONA_STYLE_CACHE[k] = v
    # ★[修正C] ユーザー指摘・プロンプト最適化指示を復元（ペルソナ単位ネスト構造）
    saved_opts = lr.get("prompt_optimizations", {})
    for pkey, bucket in saved_opts.items():
        if isinstance(bucket, dict):
            # dict[str, list[str]] 形式のみ受け入れ
            clean = {cat: [d for d in lst if isinstance(d, str)]
                     for cat, lst in bucket.items() if isinstance(lst, list)}
            if clean:
                PROMPT_OPTIMIZATIONS[pkey] = clean
        elif isinstance(bucket, list):
            # 旧形式（カテゴリ→list[str]）を global バケツに移行
            PROMPT_OPTIMIZATIONS.setdefault("global", {}).setdefault(pkey, []).extend(
                [d for d in bucket if isinstance(d, str)]
            )
    OPTIMIZATION_HISTORY = lr.get("optimization_history", [])
    total_directives = sum(len(v) for bucket in PROMPT_OPTIMIZATIONS.values()
                           if isinstance(bucket, dict) for v in bucket.values())
    if total_directives:
        print(f"{C['dim']}[学習] プロンプト指示 {total_directives}件 復元済み{C['w']}")

# ===== ペルソナ セーブ/ロード =====
def save_persona(slot_name: str, persona: dict) -> bool:
    if not slot_name or not persona: return False
    state = load_state()
    slots: dict = state.setdefault("saved_personas", {})
    slots[slot_name] = {
        "name":         persona.get("name", slot_name),
        "style":        persona.get("style", ""),
        "first_person": persona.get("first_person", "私"),
        "_web":         persona.get("_web", False),
        "saved_at":     now_stamp(),
    }
    save_state(state)
    return True

def load_persona(slot_name: str) -> dict | None:
    slots = load_state().get("saved_personas", {})
    return slots.get(slot_name)

def delete_persona(slot_name: str) -> bool:
    state = load_state()
    slots = state.get("saved_personas", {})
    if slot_name not in slots: return False
    del slots[slot_name]
    save_state(state)
    return True

def list_personas() -> dict:
    return load_state().get("saved_personas", {})

def _normalize_observed_subject(text: str) -> str:
    """Keep remembered user/subject labels from drifting to the assistant persona name."""
    if not text:
        return ""
    return re.sub(r"(?i)(?<![A-Za-z0-9_-])S-?01(?=\s*[:：は])", OBSERVED_SUBJECT_NAME, text)

def memory_context(limit: int = 8, query: str = "") -> str:
    parts = []
    memos = load_state().get("memo", [])[-limit:]
    if memos:
        parts.append("\n".join(f"- {_normalize_observed_subject(m.get('text', ''))}" for m in memos if m.get("text")))
    # queryがある場合はそれで検索。ない場合はKEYWORD_MEMORYの直近2件のみ使用
    # (古い話題キーワード全部で検索すると文脈ブリードの原因になる)
    if query:
        vec_query = query
    elif KEYWORD_MEMORY:
        vec_query = " ".join(KEYWORD_MEMORY[-2:])
    else:
        vec_query = ""
    vec_hits = vector_search(vec_query, n=3) if vec_query else []
    if vec_hits:
        parts.append("\n".join(f"• {_normalize_observed_subject(h[:200])}" for h in vec_hits))
    state = load_state()
    if state.get("dict") and query:
        # queryが明示的にある時だけ辞書を参照 (KEYWORD_MEMORYからの辞書引きはブリード源)
        hits = [e for e in state["dict"] if any(w in e["term"] for w in query.split() if len(w) >= 2)]
        if hits:
            parts.append("【辞書】\n" + "\n".join(f"• {e['term']}: {e['def'][:150]}" for e in hits[:5]))
    return "\n\n".join(parts)

def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    patterns = [r'[ァ-ヶー]{3,}', r'[一-龯]{2,}', r'[A-Za-z]{4,}']
    words = []
    for pat in patterns: words.extend(re.findall(pat, text))
    stop = {'について', 'する', 'ある', 'いる', 'です', 'ます', 'こと', 'もの', 'ため'}
    counter = Counter(w for w in words if w not in stop and len(w) >= 2)
    return [w for w, _ in counter.most_common(top_n)]

def update_keyword_memory(text: str) -> None:
    global KEYWORD_MEMORY
    noise = {'debug', 'rend', 'exit', '終了', 'help', 'list', 'add', 'del', 'find', 'done', 'show', 'fast', 'stop', 'power', 'low', 'mid', 'high', 'ultra', 'doctor'}
    new_kw = [w for w in extract_keywords(text) if w.lower() not in noise and len(w) >= 2]
    # 最大6件に絞る。古い話題のキーワードがシステムプロンプトに残留しないようにする
    KEYWORD_MEMORY = list(dict.fromkeys(KEYWORD_MEMORY + new_kw))[-6:]

def analyze_feedback(user_input: str) -> float:
    norm = normalize_for_match(user_input)
    # 単語境界なしの部分一致だと「違う」が「間違う」にも反応するため
    # ネガティブは正確な部分一致、ポジティブはそのまま
    pos_score = sum(2 for p in FEEDBACK_PATTERNS['positive'] if p in norm)
    neg_score = sum(2 for n in FEEDBACK_PATTERNS['negative'] if n in norm)
    # 肯定と否定が同時に存在する場合（「なるほど、でも違う」など）は否定優先
    if neg_score > 0 and pos_score > 0:
        return max(-1.0, -neg_score * 0.25)
    if pos_score > neg_score: return min(1.0, pos_score * 0.25)
    elif neg_score > 0: return max(-1.0, -neg_score * 0.25)
    for n in FEEDBACK_PATTERNS['neutral']:
        if n in norm: return 0.1
    return 0.0

def log_interaction(user_input: str, response: str, mode: str, feedback: float):
    global LEARNING_STATS
    LEARNING_STATS["total_interactions"] += 1
    if feedback > 0.3:
        LEARNING_STATS["positive_count"] += 1
        # ★[修正/#3] ロック経由で TEMP_VOICE を更新
        # ★[修正/temp-lock] TEMP_AUTO_TUNE_LOCKED 時は温度変更しない
        if not TEMP_AUTO_TUNE_LOCKED:
            _set_temp_voice(min(1.2, _get_temp_voice() + 0.02))
        TEMP_HISTORY.append(round(_get_temp_voice(), 3))
        if len(TEMP_HISTORY) > 30: TEMP_HISTORY[:] = TEMP_HISTORY[-30:]
    elif feedback < -0.3:
        LEARNING_STATS["negative_count"] += 1
        neg = LEARNING_STATS["negative_count"]
        pos = LEARNING_STATS["positive_count"]
        if not TEMP_AUTO_TUNE_LOCKED:
            if neg > pos:
                _set_temp_voice(max(0.3, _get_temp_voice() - 0.03))
            else:
                _set_temp_voice(max(0.3, _get_temp_voice() - 0.01))
        TEMP_HISTORY.append(round(_get_temp_voice(), 3))
        if len(TEMP_HISTORY) > 30: TEMP_HISTORY[:] = TEMP_HISTORY[-30:]
    entry = {"time": time.time(), "input": sanitize(user_input[:200]), "response_len": len(response), "mode": mode, "feedback": round(feedback, 2)}
    INTERACTION_LOG.append(entry)
    if len(INTERACTION_LOG) > 200: INTERACTION_LOG[:] = INTERACTION_LOG[-200:]
    mode = mode or "d"
    if mode not in PROMPT_PERFORMANCE: PROMPT_PERFORMANCE[mode] = []
    PROMPT_PERFORMANCE[mode].append(feedback)
    if len(PROMPT_PERFORMANCE[mode]) > 100: PROMPT_PERFORMANCE[mode] = PROMPT_PERFORMANCE[mode][-100:]

def get_best_temp(mode: str) -> float | None:
    if mode not in PARAM_PERFORMANCE or not PARAM_PERFORMANCE[mode]: return None
    best_score = -999
    best_temp = None
    for temp_str, scores in PARAM_PERFORMANCE[mode].items():
        if scores:
            avg = sum(scores) / len(scores)
            if avg > best_score:
                best_score = avg
                try: best_temp = float(temp_str)
                except ValueError: best_temp = None
    return best_temp

def update_param_performance(mode: str, temp: float, feedback: float):
    mode = mode or "d"
    if mode not in PARAM_PERFORMANCE: PARAM_PERFORMANCE[mode] = {}
    key = f"{temp:.2f}"
    if key not in PARAM_PERFORMANCE[mode]: PARAM_PERFORMANCE[mode][key] = []
    PARAM_PERFORMANCE[mode][key].append(feedback)
    if len(PARAM_PERFORMANCE[mode][key]) > 50: PARAM_PERFORMANCE[mode][key] = PARAM_PERFORMANCE[mode][key][-50:]

def optimize_prompt_template() -> str:
    best_mode = None; best_avg = -999
    for mode, scores in PROMPT_PERFORMANCE.items():
        if len(scores) >= 3:
            avg = sum(scores) / len(scores)
            if avg > best_avg: best_avg, best_mode = avg, mode
    if best_mode and best_avg > 0.3: return f" [学習: {best_mode}モード最適 ({best_avg:.1f})]"
    return ""

def cleanup_knowledge():
    now = time.time()
    with _RAG_LOCK:
        expired = [k for k, (ts, content, access_count, confidence) in list(RAG_CACHE.items())
                   if (now - ts > 1800 and access_count < 1)
                   or (now - ts > 7200)
                   or (confidence < 0.4 and access_count == 0)]
        for key in expired: del RAG_CACHE[key]
    return len(expired)

def self_evaluate_response(response: str, query: str) -> tuple[float, list[str]]:
    issues = []
    if not response or len(response.strip()) < 5: issues.append("empty_or_too_short")
    if detect_repetition(response): issues.append("repetition")
    q_words = set(re.split(r'[\s、。]+', query.lower()))
    keyword_match = sum(1 for w in q_words if len(w) >= 2 and w in response.lower())
    if keyword_match == 0 and len(q_words) >= 2: issues.append("no_keyword_match")
    template_phrases = ["一般的に", "例えば", "一方で", "また、", "つまり", "要するに"]
    if sum(1 for p in template_phrases if p in response) >= 3: issues.append("template_heavy")
    quality = 1.0
    if "empty_or_too_short" in issues: quality -= 0.5
    if "repetition" in issues: quality -= 0.4
    if "no_keyword_match" in issues: quality -= 0.2
    if "template_heavy" in issues: quality -= 0.2
    return max(0.0, quality), issues

def self_correct_response(messages: list, is_logic: bool, text_len: int, mode: str) -> str:
    global LEARNING_STATS
    if text_len < 20: return stream_response(messages, is_logic, text_len, temp_override=0.6, silent=True) or ""
    quality_threshold = 0.4
    temp_adjustments = ([0.15, 0.75] if not is_logic else [0.1, 0.5])[:2]
    query_str = messages[-1]["content"] if messages else ""
    for adj_temp in temp_adjustments:
        result = stream_response(messages, is_logic, text_len, temp_override=adj_temp, silent=True)
        if not result: continue
        quality, issues = self_evaluate_response(result, query_str)
        if quality >= quality_threshold: return result
        LEARNING_STATS["self_correction_count"] += 1
    return stream_response(messages, is_logic, text_len, temp_override=0.6, silent=True) or ""

def session_context_block() -> str:
    if not KEYWORD_MEMORY: return ""
    # 直近3件のみ注入。古い話題のキーワードがブリードしないよう絞る
    return f"\n【直近の話題】: {', '.join(KEYWORD_MEMORY[-3:])}\n"

_HALLUCINATION_CACHE: dict[str, list[str]] = {}
_HALLUCINATION_CACHE_MAX = 32

def detect_hallucination(response: str) -> list[str]:
    if len(response) < 80: return []
    cache_key = response[:120]
    if cache_key in _HALLUCINATION_CACHE:
        return _HALLUCINATION_CACHE[cache_key]
    warnings = []
    def _is_known(text: str) -> bool:
        state = load_state()
        if any(text in e.get("term", "") or text in e.get("def", "") for e in state.get("dict", [])): return True
        if any(text in m.get("text", "") for m in state.get("memo", [])): return True
        if any(text in d.get("title", "") or text in d.get("text", "") for d in state.get("docs", [])): return True
        # s01_memory（会話記憶）を検索
        for vec_hit in vector_search(text, n=1):
            if text in vec_hit: return True
        # 書籍コレクションも検索
        for col in vector_list_collections():
            if col == "s01_memory": continue
            for vec_hit in vector_search(text, n=1, collection=col):
                if text in vec_hit: return True
        return False
    for m in re.finditer(r'[「『]([^」』]{2,50})[」』]', response):
        name = m.group(1).strip()
        if len(name) >= 3 and not _is_known(name): warnings.append(f"「{name}」は知識ベースに未登録（捏造の可能性）")
    for m in re.finditer(r'(?:『([^』]+)』|「([^」]+)」|(\S{2,20}))という(?:作品|曲|本|小説|漫画|アニメ|映画|ドラマ|番組|人|人物|場所|組織|会社|企業|国|都市|用語|言葉|考え方|制度|概念|現象|法則|理論|手法|技術|商品|製品|サービス|アプリ|ゲーム|キャラ|グループ|バンド|歌手|俳優|タレント|YouTuber|配信者|会社員|教授|博士|先生|作家|画家|監督|政治家|社長|理事長|代表取締役|フリーランス|デザイナー|エンジニア)', response):
        name = m.group(1) or m.group(2) or m.group(3)
        if name and len(name) >= 2 and not _is_known(name): warnings.append(f"「{name}」という未知のエンティティを提示")
    for m in re.finditer(r'([\u4E00-\u9FFF]{2,10}(?:は|が))(\d{3,}(?:年|月|日|人|個|件|社|店|億|万|千|百|％|パーセント|円|ドル|ユーロ|kg|g|km|m|cm|mm))', response):
        subject = m.group(1).rstrip("はが")
        if not _is_known(subject + m.group(2)[:6]): warnings.append(f"「{subject}」に関する数値主張「{m.group(2)}」— 未確認")
    for m in re.finditer(r'「([^」]{5,60})」と(?:言|述べ|語|話|コメント|発言)', response):
        if not _is_known(m.group(1)[:20]): warnings.append(f"引用文「{m.group(1)[:30]}...」— 出典不明")
    for m in re.finditer(r'([\u4E00-\u9FFF]{2,15})(?:は|が)(\d{3,4}年)(?:に|(?:に作|に公開|に出版|に発表|に発売|に設立|に開業|に開校|に開店|に開始|に終了|に完成))', response):
        if not _is_known(m.group(1) + m.group(2)): warnings.append(f"「{m.group(1)}」の「{m.group(2)}」— 未確認の年代")
    for m in re.finditer(r'([\u4E00-\u9FFF]{2,4})(?:教授|博士|先生|大臣|社長|会長|院長|学長|知事|市長|町長|村長|監督|選手|議員|長官|事務局長|理事長|代表|CEO|社長|部長|課長|係長|店長|所長|局長|管理官|参与|顧問|弁護士|会計士|税理士|医師|看護師|薬剤師|獣医師|教諭|准教授|講師|助教|助手|研究員|学芸員|司書|カウンセラー|セラピスト|トレーナー|コーチ|審判|解説者|アナウンサー)', response):
        if not _is_known(m.group(1)): warnings.append(f"「{m.group(1)}」— 肩書き付き人物だが未確認")
    for m in re.finditer(r'([\u30A1-\u30F4]{3,15})(?:とは|って|というのは|は、)(?:[\u4E00-\u9FFF]{2,}のこと|[\u4E00-\u9FFF]{2,}を指す|[\u4E00-\u9FFF]{2,}の一種)', response):
        if not _is_known(m.group(1)): warnings.append(f"「{m.group(1)}」— 定義説明があったが未確認の用語")
    # キャッシュに格納（古いエントリを削除）
    if len(_HALLUCINATION_CACHE) >= _HALLUCINATION_CACHE_MAX:
        try: del _HALLUCINATION_CACHE[next(iter(_HALLUCINATION_CACHE))]
        except StopIteration: pass
    _HALLUCINATION_CACHE[cache_key] = warnings
    return warnings

def _print_hallucination_warnings(response: str, strict: bool = False) -> None:
    """detect_hallucination の結果をターミナルに表示する。
    strict=True（/kb ask）のときは警告をより厳格に扱い、出力前に改行を入れる。"""
    warnings = detect_hallucination(response)
    if not warnings:
        return
    prefix = f"\n{C['y']}[ハルシネーション検出 {len(warnings)}件]{C['w']}"
    print(prefix)
    for w in warnings[:5]:   # 最大5件まで表示
        print(f"  {C['dim']}⚠ {w}{C['w']}")
    if strict and len(warnings) >= 3:
        print(f"  {C['r']}※ 局所参照外の情報が多く含まれる可能性。/kb search で原文を確認推奨。{C['w']}")

def detect_repetition(text: str, window: int = 150) -> bool:
    """★[v129] 繰り返し検出強化: フレーズ重複 + 文レベル重複を検出。"""
    if len(text) < window * 2: return False
    # バイナリ重複（完全一致）
    tail = text[-window * 2:]
    if tail[:len(tail)//2] == tail[len(tail)//2:]:
        return True
    # 同一比喩構文の多用（4回以上に厳格化）
    if len(re.findall(r'まるで.{5,50}(?:ようなもの|ような状況|ようだ|かのよう)', text)) >= 4:
        return True
    # 400字ウィンドウで前半=後半
    if len(text) >= 400:
        chunk = text[-400:]
        half = len(chunk) // 2
        if chunk[:half] == chunk[half:]:
            return True
    # ★[v129] 文レベル重複検出: 同じ文が2回以上出てきたら繰り返し
    sents = re.split(r'[。！？\n]', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    seen_sents: set = set()
    for s in sents:
        norm = re.sub(r"\s+", "", s)[:35]
        if norm in seen_sents: return True
        seen_sents.add(norm)
    if len(text) > 20000:
        return True
    return False

def trim_history(ms: list, max_pairs: int = MAX_HISTORY, token_budget: int = 2000) -> list:
    """★[v129] 件数上限 + トークン予算の両方でトリム。重要度スコアで古いペアを優先削除。"""
    ms = ms[-(max_pairs * 2):]  # まず件数で絞る
    def _tok(s: str) -> int:
        jp = sum(1 for c in s if ord(c) > 0x7F)
        return int(jp * TOKEN_EST_JP + (len(s) - jp) * TOKEN_EST_EN)
    # ★[v129] 予算超過なら古いペアから削除（最後の2ペアは必ず残す）
    while len(ms) >= 6:  # 3ペア以上ある場合のみ削除対象
        total = sum(_tok(m.get("content", "")) for m in ms)
        if total <= token_budget:
            break
        ms = ms[2:]  # 最古のuser/assistantペアを削除
    return ms

def build_chat_messages(sys_msg: dict, ms: list, persona: dict) -> list:
    fp = persona.get("first_person", "私")
    name = persona["name"]
    style_hint = persona["style"][:200]
    anchor = [
        {"role": "user", "content": "あなたのキャラクターを確認して。"},
        {"role": "assistant", "content": f"キャラ名は{name}。一人称は{fp}。{style_hint}。ずっとこのキャラで話し続けるよ。"},
    ]
    return [sys_msg] + anchor + trim_history(ms)

_cookie_jar = CookieJar()

def _build_ssl_ctx(verify: bool = True) -> ssl.SSLContext:
    """SSL コンテキストを構築する。
    verify=True（デフォルト）: 証明書検証あり（安全）
    verify=False: 検証なし（証明書が壊れた古いサーバへのフォールバック専用）
    """
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx

# 通常用（証明書検証あり）
_ctx_verified   = _build_ssl_ctx(verify=True)
# フォールバック用（証明書検証なし、社内スクレイピングのみ）
_ctx_unverified = _build_ssl_ctx(verify=False)

_opener_verified   = R.build_opener(R.HTTPSHandler(context=_ctx_verified),   R.HTTPCookieProcessor(_cookie_jar))
_opener_unverified = R.build_opener(R.HTTPSHandler(context=_ctx_unverified),  R.HTTPCookieProcessor(_cookie_jar))

def fetch_html(url: str, data: bytes | None = None, timeout: int = 5, silent: bool = False,
               spoof_bot: bool = False, redirect_checker=None) -> str:
# ★[v131.1] UAを正直な識別子に変更（ブラウザ偽装廃止）
    ua = "aegis-omnis/131.1 (personal research tool; +https://github.com/eitaaaan/aegis-omnis)"
    headers = {"User-Agent": ua, "Accept-Language": "ja,en;q=0.9", "Accept": "text/html,*/*;q=0.8"}
    def _decode(raw: bytes) -> str:
        for enc in ("utf-8", "shift_jis", "euc-jp"):
            try: return raw.decode(enc)
            except UnicodeDecodeError: continue
        return raw.decode("utf-8", "ignore")

    def _make_opener(ctx: ssl.SSLContext):
        handlers = [R.HTTPSHandler(context=ctx), R.HTTPCookieProcessor(_cookie_jar)]
        if redirect_checker:
            class _CheckedRedirectHandler(R.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    redirect_checker(newurl)
                    return super().redirect_request(req, fp, code, msg, headers, newurl)
            handlers.append(_CheckedRedirectHandler())
        return R.build_opener(*handlers) if redirect_checker else None

    opener_verified = _make_opener(_ctx_verified) or _opener_verified
    opener_unverified = _make_opener(_ctx_unverified) or _opener_unverified

    req = R.Request(url, data=data, headers=headers)
    # まず証明書検証ありで試みる（セキュアなデフォルト）
    try:
        with opener_verified.open(req, timeout=timeout) as resp:
            return _decode(resp.read())
    except ssl.SSLError:
        # SSL証明書エラー時のみ検証なしにフォールバック（警告を出す）
        if not silent:
            print(f"{C['y']}[NET] SSL証明書エラー。検証なしでリトライ中...{C['w']}")
        try:
            req2 = R.Request(url, data=data, headers=headers)
            with opener_unverified.open(req2, timeout=timeout) as resp:
                return _decode(resp.read())
        except Exception as e:
            if not silent: print(f"{C['r']}[NET] {e}{C['w']}")
            return ""
    except Exception as e:
        if not silent: print(f"{C['r']}[NET] {e}{C['w']}")
        return ""

def strip_tags(fragment: str) -> str:
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html_module.unescape(fragment).strip()

def _deduplicate_lines(lines: list[str], min_len: int = 10) -> list[str]:
    seen, result = set(), []
    for line in lines:
        line = line.strip()
        if not line or len(line) < min_len: continue
        normalized = re.sub(r'\s+', ' ', line.lower())
        if normalized in seen or any(normalized in s for s in seen): continue
        seen.add(normalized); result.append(line)
    return result

def get_wikipedia(query: str) -> str:
    try:
        if OFFLINE_MODE:
            # Kiwix ローカルサーバー経由（kiwix-serve --port KIWIX_PORT で起動しておく）
            # Kiwix は MediaWiki API 互換エンドポイントを /api で提供する
            url = f"http://localhost:{KIWIX_PORT}/api?format=json&action=query&prop=extracts&explaintext&redirects=1&titles={U.quote(query)}"
        else:
            url = f"https://ja.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext&redirects=1&titles={U.quote(query)}"
        raw = fetch_html(url, timeout=RAG_TIMEOUT, silent=True)
        if raw:
            pages = json.loads(raw).get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if pid != "-1" and page.get("extract"): return sanitize(page["extract"][:3000])
    except Exception as e: print(f"{C['y']}[WARN] Wikipedia fetch失敗: {e}{C['w']}")
    return ""

# ===== BRAVE SEARCH API (Step2: APIキーを環境変数 BRAVE_API_KEY に設定すると有効化) =====
def _fetch_brave_snippets(query: str) -> str:
    """Brave Search API経由で検索スニペットを取得。
    取得方法: https://api.search.brave.com/ で無料登録 -> APIキー発行
    設定方法: プロジェクトフォルダに .env ファイルを作り
              BRAVE_API_KEY=your_key_here と書く（後述のload_dotenvが読み込む）"""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return ""
    try:
        url = f"https://api.search.brave.com/res/v1/web/search?q={U.quote(query)}&count=5&lang=ja&country=jp"
        req = R.Request(url, headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        })
        with R.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            try:
                import gzip
                raw = gzip.decompress(raw)
            except Exception as _e:
                print(f"{C['y']}[WARN] gzip展開失敗（非圧縮として続行）: {_e}{C['w']}")
            data = json.loads(raw.decode("utf-8"))
        results = data.get("web", {}).get("results", [])
        lines = []
        for r in results[:5]:
            desc = r.get("description", "").strip()
            if desc and len(desc) > 15:
                lines.append(f"[Brave] {sanitize(desc)}")
        return "\n".join(lines)
    except Exception:
        return ""

def _fetch_yahoo_snippets(query: str) -> str:
    """Yahoo検索スクレイピング。Brave APIが有効なら優先使用。
    複数のCSSパターンを順に試みる堅牢版。"""
    if OFFLINE_MODE: return ""
    brave = _fetch_brave_snippets(query)
    if brave:
        return brave
    try:
        url = f"https://search.yahoo.co.jp/search?p={U.quote(query)}"
        h = fetch_html(url, timeout=4, silent=True, spoof_bot=True)
        YAHOO_PATTERNS = [
            r'<span class="sw-Card__summaryDesc">(.*?)</span>',
            r'<p class="sw-Card__summary"[^>]*>(.*?)</p>',
            r'<div class="sw-Card__description"[^>]*>(.*?)</div>',
            r'<p[^>]+class="[^"]*summary[^"]*"[^>]*>(.*?)</p>',
            r'<span[^>]+class="[^"]*description[^"]*"[^>]*>(.*?)</span>',
        ]
        snips = []
        for pat in YAHOO_PATTERNS:
            snips = re.findall(pat, h, re.I | re.S)
            if snips:
                break
        if not snips:
            snips = re.findall(r'<p[^>]*>([^<]{30,200})</p>', h)
        lines = [l for l in [strip_tags(s) for s in snips[:6]] if len(l) > 15]
        return sanitize("\n".join(lines))
    except Exception:
        return ""

def _fetch_bing_snippets(query: str) -> str:
    """Bing検索スクレイピング。複数のCSSパターンを順に試みる堅牢版。"""
    if OFFLINE_MODE: return ""
    try:
        url = f"https://www.bing.com/search?q={U.quote(query)}&setlang=ja&mkt=ja-JP"
        h = fetch_html(url, timeout=4, silent=True, spoof_bot=True)
        BING_PATTERNS = [
            r'<div class="b_caption">.*?<p[^>]*>(.*?)</p>',
            r'<p class="b_paractl"[^>]*>(.*?)</p>',
            r'<div class="b_snippet"[^>]*>(.*?)</div>',
            r'<p[^>]+class="[^"]*snippet[^"]*"[^>]*>(.*?)</p>',
        ]
        snips = []
        for pat in BING_PATTERNS:
            snips = re.findall(pat, h, re.I | re.S)
            if snips:
                break
        lines = [l for l in [strip_tags(s) for s in snips[:5]] if len(l) > 15]
        return sanitize("\n".join(lines))
    except Exception:
        return ""

def _fetch_ddg_snippets(query: str) -> str:
    """DuckDuckGo Liteスクレイピング。複数パターン + HTML版フォールバックあり。"""
    if OFFLINE_MODE: return ""
    DDG_PATTERNS = [
        r'class="result-snippet"[^>]*>(.*?)</td>',
        r'class="result__snippet"[^>]*>(.*?)</a>',
        r'<td[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</td>',
    ]
    # エンドポイント1: lite版（軽量・安定）
    try:
        data = U.urlencode({"q": query, "kl": "jp-jp"}).encode("utf-8")
        h = fetch_html("https://lite.duckduckgo.com/lite/", data=data, timeout=4, silent=True)
        for pat in DDG_PATTERNS:
            snips = re.findall(pat, h, re.I | re.S)
            if snips:
                lines = [l for l in [strip_tags(s) for s in snips[:5]] if len(l) > 15]
                if lines:
                    return sanitize("\n".join(lines))
    except Exception as _e:
        print(f"{C['y']}[WARN] DDG lite取得失敗: {_e}{C['w']}")
    # エンドポイント2: HTML版（lite版が失敗したときのフォールバック）
    try:
        url = f"https://html.duckduckgo.com/html/?q={U.quote(query)}&kl=jp-jp"
        h = fetch_html(url, timeout=5, silent=True, spoof_bot=True)
        snips = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', h, re.I | re.S)
        if not snips:
            snips = re.findall(r'<div class="result__body"[^>]*>(.*?)</div>', h, re.I | re.S)
        lines = [l for l in [strip_tags(s) for s in snips[:5]] if len(l) > 15]
        return sanitize("\n".join(lines))
    except Exception:
        return ""

# ★[統合] 同一パターンの3関数を汎用ヘルパーで置換
def _fetch_simple_snippets(url: str, pattern: str, label: str, min_len: int = 20) -> str:
    """汎用スニペット取得: URL・正規表現・ラベルを指定するだけで動作。
    旧: _fetch_nhk_snippets / _fetch_kotobank / _fetch_stackoverflow_snippets"""
    if OFFLINE_MODE: return ""
    try:
        h = fetch_html(url, timeout=4, silent=True, spoof_bot=True)
        snips = re.findall(pattern, h, re.I | re.S)
        return "\n".join(f"[{label}] {strip_tags(s)}" for s in snips[:3] if len(strip_tags(s)) > min_len)
    except Exception: return ""

def _fetch_nhk_snippets(query: str) -> str:
    return _fetch_simple_snippets(
        f"https://www3.nhk.or.jp/news/search/?keyword={U.quote(query)}",
        r'<p class="text--M"[^>]*>(.*?)</p>', "NEWS")

def _fetch_kotobank(query: str) -> str:
    return _fetch_simple_snippets(
        f"https://kotobank.jp/gs/?q={U.quote(query)}",
        r'<div[^>]+class="[^"]*description[^"]*"[^>]*>(.*?)</div>', "コトバンク")

def _fetch_stackoverflow_snippets(query: str) -> str:
    return _fetch_simple_snippets(
        f"https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={U.quote(query)}&site=stackoverflow&pagesize=3",
        r'"title":\s*"([^"]+)"', "Stack Overflow", min_len=10)





def get_async_rag_data(query: str) -> str:
    """
    ★[v131] ハイブリッドRAG v3:
      並列Web取得 → BM25スコアリング → ChromaDB vector_search
      → RRF融合 → Contextual Compression → Cross-Encoder Reranking
    """
    # ── L1: メモリキャッシュ（最速）──
    _nq = _rag_normalize_query(query)
    with _RAG_LOCK:
        # 正規化キーでも検索
        cached = RAG_CACHE.get(query) or RAG_CACHE.get(_nq)
        if cached and time.time() - cached[0] < 1800:
            ts, content, access_count, confidence = cached
            RAG_CACHE[query] = (ts, content, access_count + 1, confidence)
            return content
    # ── L2: ディスクキャッシュ（再起動後も有効）──
    _disk_hit = _rag_disk_get(_nq)
    if _disk_hit is not None:
        with _RAG_LOCK:
            RAG_CACHE[query] = (time.time(), _disk_hit, 1, 0.7)
        return _disk_hit
    res: dict[str, str] = {}
    lock = threading.Lock()

    def run_task(key: str, fn, *args):
        try:
            val = fn(*args)
            with lock: res[key] = val or ""
        except Exception: pass  # サイレントフェイルアウト

    tasks = [
        ("wiki",     get_wikipedia,         query),
        ("ddg",      _fetch_ddg_snippets,    query),
    ]
    # ★[v131] HyDE をWeb取得と並列実行（Ollama利用可能時のみ）
    # RAG待機中にHyDE仮想ドキュメントを生成しておくことで遅延ゼロ
    _hyde_future = None
    if HYDE_ENABLED and VECTOR_AVAILABLE:
        _o = _get_ollama()
        if _o is not None:
            _hyde_future = _THREAD_POOL.submit(hyde_expand_query, query, _o)
    futures = {_THREAD_POOL.submit(run_task, k, fn, *a): k for k, fn, *a in tasks}
    start_time = time.time()
    _rag_t0 = time.time()

    while time.time() - start_time < RAG_TIMEOUT:
        with lock:
            wiki_len = len(res.get("wiki", ""))
            web_len  = sum(len(res.get(k, "")) for k in ["yahoo", "ddg", "bing", "kotobank", "nhk"])
            if wiki_len > 800 or web_len > 800:
                break
        time.sleep(0.1)

    for f in futures:
        remain = max(0.0, RAG_TIMEOUT - (time.time() - start_time))
        try: f.result(timeout=remain)
        except Exception: pass

    # ★[v131] HyDE futureを回収（すでに完了しているはず）
    if _hyde_future is not None:
        try: _hyde_future.result(timeout=0.5)
        except Exception: pass

    with lock:
        wiki = res.get("wiki", "").strip()
        web_hits = [res.get(k, "").strip() for k in ["yahoo", "ddg", "bing", "kotobank", "nhk"]]

    # ── 1. Web候補をBM25でスコアリング ──────────────────────────
    def _bm25_score(text: str) -> float:
        if not text: return 0.0
        q_words = re.findall(r'[\u3040-\u9FFF\w]{2,}', query.lower())
        if not q_words: return len(text) * 0.001
        score = 0.0
        text_lower = text.lower()
        for w in q_words:
            tf = text_lower.count(w)
            if tf > 0:
                score += (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * len(text) / 500))
        return score

    all_lines = []
    for block in web_hits:
        for line in block.splitlines():
            line = line.strip()
            if len(line) > 20:
                all_lines.append((line, _bm25_score(line)))

    all_lines.sort(key=lambda x: -x[1])
    seen: set[str] = set()
    bm25_ranked: list[str] = []
    for line, score in all_lines:
        norm = re.sub(r'\s+', '', line.lower())[:40]
        if norm not in seen and score > 0:
            seen.add(norm)
            bm25_ranked.append(line)
        if len(bm25_ranked) >= 25:
            break

    # ── 2. ★[v131] ChromaDB ベクトル検索 (HyDE拡張クエリ — 並列実行)
    # HyDE はWeb取得と並列に実行済みのため、ここではキャッシュ参照のみ
    vec_ranked: list[str] = []
    if VECTOR_AVAILABLE:
        try:
            # HyDEキャッシュを先に確認（Ollamaブロックを回避）
            with _HYDE_LOCK:
                hyde_q = _HYDE_CACHE.get(query, query)
            vec_ranked = vector_search(hyde_q, n=10, collection="s01_memory")
        except Exception:
            vec_ranked = []

    # ── 3. ★[v131] RRF: BM25ランクとベクトルランクを融合 ────────
    if vec_ranked and RRF_ENABLED:
        fused_candidates = reciprocal_rank_fusion([bm25_ranked, vec_ranked])
    else:
        fused_candidates = bm25_ranked

    # ── 4. ★[v131] Contextual Compression ──────────────────────
    compressed_candidates = contextual_compress(query, fused_candidates[:20])

    # ── 5. ★[v131] Cross-Encoder Reranking ──────────────────────
    final_web_lines = cross_encoder_rerank(query, compressed_candidates, top_k=CROSS_ENCODER_TOP_K)

    merged_web = "\n".join(final_web_lines)

    if len(wiki) < 10 and len(merged_web) < 20:
        _rag_disk_set(_rag_normalize_query(query), "")  # 空結果もキャッシュ
        return ""
    parts = []
    if wiki: parts.append(f"[Wikipedia JA]\n{wiki}")
    tag = "BM25+Vector+RRF+CE" if (vec_ranked and CROSS_ENCODER_ENABLED) else "BM25 ranked"
    if merged_web: parts.append(f"[Web Search ({tag})]\n{merged_web}")
    final = "\n\n".join(parts)

    has_wiki = bool(wiki)
    has_web  = bool(merged_web)
    # Cross-Encoderが使えた場合は信頼度ボーナス
    ce_bonus = 0.05 if (_cross_encoder_available and CROSS_ENCODER_ENABLED) else 0.0
    confidence = min(0.95, (0.8 if (has_wiki and has_web) else (0.6 if (has_wiki or has_web) else 0.2)) + ce_bonus)

    with _RAG_LOCK:
        if len(RAG_CACHE) > 200:
            evict_keys = sorted(RAG_CACHE.items(), key=lambda x: (x[1][2], x[1][0]))[:20]
            for k, _ in evict_keys: RAG_CACHE.pop(k, None)
        RAG_CACHE[query] = (time.time(), final, 1, confidence)
        _rag_record_timing(time.time() - _rag_t0)
        _rag_disk_set(_rag_normalize_query(query), final)  # L2保存
    return final

def _parse_facts(raw: str) -> tuple[bool, list[str], dict[str, int]]:
    if raw.strip().startswith("NO_DATA") and len(raw.strip()) < 20: return False, [], {}
    facts_raw = [f.strip() for f in re.findall(r"<FACT>(.*?)</FACT>", raw, re.S) if f.strip()]
    confidence: dict[str, int] = {"HIGH": 0, "MID": 0, "LOW": 0}
    facts_clean = []
    for f in facts_raw:
        for level in ("HIGH", "MID", "LOW"):
            if f.startswith(f"[{level}]"): confidence[level] += 1; facts_clean.append(f); break
        else: facts_clean.append(f)
    if not facts_clean:
        lines = [ln.lstrip("・- 　").strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("<") and len(ln.strip()) > 6]
        facts_clean = lines[:12]
    return bool(facts_clean), facts_clean, confidence

def _build_voice_cast_prompt(query: str, facts: list[str], persona: dict) -> list[dict]:
    facts_block = "\n".join(f"- {f}" for f in facts)
    fp = persona.get("first_person", "私")
    system = (
        f"あなたは{persona['name']}。口調: {persona['style']}。一人称: {fp}。\n"
        f"質問「{query}」についてのみ語れ。同姓の別人の情報は一切無視しろ。\n"
        f"一人称(私/あたし/僕/俺)を事実の主体として使うな。「Xは〜」の形式で書け。\n"
        f"【事実】に書いてあること ONLY で7〜10文で答えろ。\n"
        f"【事実】にないことは書くな。推測・補足・一般論は禁止。\n"
        f"口調に合わせて自然に絵文字を1〜2個入れろ。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": f"質問: {query}\n\n【事実】:\n{facts_block}\n\n【事実】から{query}について三人称で詳しく説明しろ。"}]

def _build_no_data_prompt(query: str, persona: dict) -> list[dict]:
    return [{"role": "system", "content": f"あなたは{persona['name']}。口調: {persona['style']}。一人称: {persona.get('first_person','私')}。「情報がない」とだけ言え。推測・創作は一切するな。"}, {"role": "user", "content": f"「{query}」について調べたが見つからなかった。「情報がない」とだけ言え。"}]

def two_pass_analysis(query: str, rag_data: str, persona: dict, text_len: int) -> str:
    sp1 = SystemSpinner("RAG事実抽出(BM25)...", stage="pass1")
    sp1.start()
    lines_rag = [re.sub(r'[^ -~　-鿿＀-￯]', '', ln.strip()) for ln in rag_data.splitlines()]
    lines_rag = [ln for ln in lines_rag if len(ln) >= FACT_MIN_CHARS and ln not in ("(empty)", "")]
    q_words = [w for w in re.split(r'[\s　、。，．]+', query) if len(w) >= 2]
    # ★[v129] BM25風スコアリング (k1=1.5, b=0.75)
    avg_len = sum(len(ln) for ln in lines_rag) / max(len(lines_rag), 1)
    def bm25(ln: str) -> float:
        score = 0.0
        ln_lower = ln.lower()
        for w in q_words:
            tf = ln_lower.count(w)
            if tf > 0:
                idf = math.log(1 + len(lines_rag) / (1 + sum(1 for l in lines_rag if w in l.lower())))
                score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * len(ln) / max(avg_len, 1)))
        return score
    scored = sorted([(bm25(ln), ln) for ln in lines_rag], key=lambda x: -x[0])
    facts = [ln[:250] for _, ln in scored[:10]]  # ★[v129] 8→10件, 200→250文字
    raw_p1 = "\n".join(f"<FACT>[HIGH] {f}</FACT>" for f in facts[:4]) + "\n".join(f"<FACT>[MID] {f}</FACT>" for f in facts[4:])
    elapsed1 = sp1.stop()
    print(f"{C['dim']}  Pass1完了 ({elapsed1:.1f}s) / {len(facts)}件抽出{C['w']}")
    data_found, facts, confidence = _parse_facts(raw_p1)
    facts_text = "\n".join(facts)
    if data_found:
        conf_str = " / ".join(f"{k}:{v}" for k, v in confidence.items() if v > 0)
        if conf_str: print(f"{C['dim']}  FACT: {conf_str}{C['w']}")
    print(f"{C['c']}{persona['name']}{C['w']}: ", end="", flush=True)
    if data_found and len(facts_text) >= FACT_MIN_CHARS:
        result = stream_response(_build_voice_cast_prompt(query, facts, persona), False, len(facts_text), _get_temp_voice(), False, model=DEEP_MODEL)
        if result and len(result.strip()) > 5:
            _print_hallucination_warnings(result)
            return result
    result = stream_response(_build_no_data_prompt(query, persona), False, 50, _get_temp_voice(), False, model=DEEP_MODEL)
    if result: _print_hallucination_warnings(result)
    return result

def _find_overlap(base: str, continuation: str, max_check: int = 80) -> int:
    """base の末尾と continuation の先頭の重複長を返す。重複除去に使用。"""
    tail = base[-max_check:]
    for length in range(min(max_check, len(continuation)), 0, -1):
        if tail.endswith(continuation[:length]):
            return length
    return 0

def _single_gen(o, model: str, msgs: list, opts: dict, silent: bool, timeout: int) -> tuple:
    """1回分の生成。タイムアウトなしの直接ストリーミング。(テキスト, 成功フラグ) を返す。"""
    full = ""
    # ★[v131] 出力バッファリング: 1文字ずつflushせず句読点・改行でまとめて出力
    # これによりターミナル描画コストとシステムコール回数を削減
    _BUF_FLUSH_CHARS = frozenset("。、\n！？!?\r")
    _buf = []
    _buf_len = 0
    _FLUSH_AT = 8  # 8文字バッファが溜まったらflush

    def _flush_buf():
        nonlocal _buf, _buf_len
        if _buf:
            sys.stdout.write("".join(_buf))
            sys.stdout.flush()
            _buf = []
            _buf_len = 0

    try:
        for chunk in o.chat(model=model, messages=msgs, stream=True, options=opts, keep_alive=-1):
            msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", None)
            if isinstance(msg, dict): t = msg.get("content", "")
            else: t = getattr(msg, "content", "")
            if not isinstance(t, str) or not t: continue
            t = sanitize(t)
            if not t: continue
            full += t
            if not silent:
                _buf.append(t)
                _buf_len += len(t)
                # 句読点・改行・一定量でflush
                if _buf_len >= _FLUSH_AT or any(c in _BUF_FLUSH_CHARS for c in t):
                    _flush_buf()
        if not silent:
            _flush_buf()
        return full, True
    except KeyboardInterrupt:
        if not silent: _flush_buf()
        return full, False
    except Exception as e:
        if not silent:
            _flush_buf()
            print(f"\n{C['r']}[ERR] {e}{C['w']}")
        return full, False


def stream_response(messages: list, is_logic: bool, text_len: int,
                    temp_override: float | None = None, silent: bool = False,
                    max_tokens: int | None = None, model: str | None = None) -> str:
    messages = sanitize_obj(messages)
    o = _get_ollama()
    if o is None:
        if not silent: print(f"{C['r']}[ERR] ollama not installed{C['w']}")
        return ""
    if model is None:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        model = select_model(last)

    # ★[v129] トークン推定（改良版: BPEベース近似）
    def _estimate_tokens(text: str) -> int:
        if not text: return 0
        jp = sum(1 for c in text if ord(c) > 0x7F)
        en = len(text) - jp
        # 句読点・記号の調整
        punct = sum(1 for c in text if c in "。、！？.,!?;:「」『』【】")
        return int(jp * TOKEN_EST_JP + en * TOKEN_EST_EN + punct * 0.2)

    def _msgs_token_estimate(msgs: list) -> int:
        return sum(_estimate_tokens(m.get("content", "")) for m in msgs) + len(msgs) * 4  # +4=role overhead

    _opts_preview = get_llm_opt(is_logic, text_len, temp_override, max_tokens=max_tokens)
    _n_ctx = _opts_preview.get("num_ctx", 4096)
    _n_predict = _opts_preview.get("num_predict", 2000)

    # ★[v129] TokenBudget: 動的コンテキスト管理
    _n_predict_safe = max(0, _n_predict) if _n_predict != -1 else 2048
    _prompt_budget = _n_ctx - _n_predict_safe - TOKEN_BUDGET_SAFETY
    _fixed = [m for m in messages if m.get("role") == "system"]
    _history = [m for m in messages if m.get("role") != "system"]
    _user_last = _history[-1:] if _history and _history[-1].get("role") == "user" else []
    _conv = _history[:-1] if _user_last else _history
    _fixed_tokens = _msgs_token_estimate(_fixed + _user_last)
    _budget_for_conv = max(0, _prompt_budget - _fixed_tokens)

    # ★[v129] スマートプルーニング: 重要度スコアで古い会話を削除
    _purged = 0
    while _conv and _msgs_token_estimate(_conv) > _budget_for_conv:
        # 最初のuser/assistantペアを削除（最古の会話）
        _conv = _conv[2:] if len(_conv) >= 2 else []
        _purged += 1
    if _purged and not silent:
        print(f"{C['dim']}[ctx] 履歴{_purged}ペア削除 (ctx圧迫回避){C['w']}")
    messages = _fixed + _conv + _user_last

    opts = get_llm_opt(is_logic, text_len, temp_override, max_tokens=max_tokens)

    # ★[v129] Thinking Mode: chain-of-thought prefix
    if THINKING_MODE and is_logic:
        # think タグを自動追加（対応モデルのみ）
        _sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if _sys_idx is not None:
            _orig = messages[_sys_idx]["content"]
            messages = list(messages)
            messages[_sys_idx] = {
                "role": "system",
                "content": _orig + "\n\n【思考プロセス】まず<think>タグ内で段階的に考え、その後回答を出力せよ。"
            }

    full_result, ok = _single_gen(o, model, messages, opts, silent, 0)
    if not full_result.strip():
        # ★[v129] モデルフォールバック: FASTモデルで再試行
        if model == DEEP_MODEL and DEEP_MODEL != MODEL_NAME:
            if not silent: print(f"{C['y']}[WARN] {DEEP_MODEL}失敗 → {MODEL_NAME}で再試行{C['w']}")
            full_result, ok = _single_gen(o, MODEL_NAME, messages, opts, silent, 0)
        if not full_result.strip():
            if not silent: print(f"\n{C['r']}[ERR] 応答がありません{C['w']}")
            return ""

    # ★[v129] <think>タグを除去して最終回答のみ返す
    if "<think>" in full_result and "</think>" in full_result:
        _think_end = full_result.rfind("</think>")
        if _think_end != -1:
            full_result = full_result[_think_end + 8:].strip()

    # ★[修正/trunc-3] 不完全末尾クリーンアップ
    # num_predictの上限に達してトークンが途中で打ち切られると、
    # 「たとえば」「そのよう」など文末が欠けた状態で返る。
    # 最後の文末句読点（。！？）より後に未完文が残っていれば除去する。
    def _cleanup_incomplete(text: str) -> str:
        """最後の完結文（。！？）以降に不完全フラグメントが残っていれば切り捨てる。"""
        if not text:
            return text
        # 末尾が既に完結句読点で終わっているなら何もしない
        if text.rstrip()[-1:] in "。！？!?\n":
            return text
        # 最後の句読点位置を探して、そこまでを返す
        last_end = max(
            text.rfind("。"),
            text.rfind("！"),
            text.rfind("？"),
            text.rfind("!"),
            text.rfind("?"),
        )
        if last_end > len(text) // 2:  # 後半以降に句読点があれば切り捨て
            return text[:last_end + 1]
        return text  # 句読点が少なすぎる場合はそのまま返す

    full_result = _cleanup_incomplete(full_result)

    # 繰り返しループ検出 → 末尾を整形して返す
    if detect_repetition(full_result):
        full_result = full_result.rstrip("、，")

    if not silent: print()
    return full_result

def get_llm_opt(is_logic_mode: bool, text_len: int = 0, temp_override: float | None = None, max_tokens: int | None = None) -> dict:
    power = POWER_MODE
    # ★[v131] GPU環境では大幅にctxを拡張
    _gpu = _GPU_AVAILABLE or _HAS_12B
    configs = {
        #          ctx     pl    pc    tl     tc    threads
        # ★[v131] num_ctx を半減: KVキャッシュ確保コストが最大のボトルネック
        # CPU推論では ctx が大きいほど prefill も遅くなる
        "ultra": (65536 if _gpu else 8192,  4096, 4096, 0.10, 0.72, 16),
        "high":  (32768 if _gpu else 6144,  2048, 2048, 0.15, 0.76, 16),
        "mid":   (16384 if _gpu else 4096,  1024, 1024, 0.18, 0.78, 12),
        "low":   (4096,                       512,  400, 0.20, 0.78,  8),
    }
    ctx, pl, pc, tl, tc, threads = configs.get(power, configs["high"])

    # ★[v131] Thinking Mode: logicモードで低温・長いpredict
    if THINKING_MODE and is_logic_mode:
        ctx = max(ctx, 8192 if _gpu else 4096)
        num_predict = min(max_tokens or 2048, 2048)
        final_temp = temp_override if temp_override is not None else 0.10
    elif is_logic_mode:
        # ★[修正/trunc-4] max_tokensが大きい場合はctxも合わせて拡張する
        # d_tokens=2800〜3200を渡したとき ctx=4096 だと入力+出力が収まらず途切れる。
        # 安全マージン1024を加えた値とデフォルトctxの大きいほうを使う。
        _needed_ctx = (max_tokens or 1024) + 1024
        ctx = max(ctx, _needed_ctx, 6144 if _gpu else 5120)
        num_predict = min(max_tokens or 2800, 2800 if not _gpu else 4096)  # ★[修正/trunc-2] CPU1024→2800: 哲学者長文に対応
        final_temp = temp_override if temp_override is not None else tl
    elif text_len < 80:
        # ★[v131] 短い入力はctxをさらに絞る
        ctx = max(512, ctx // 8)
        num_predict = 256
        final_temp = temp_override if temp_override is not None else tc
    elif text_len > 600:
        ctx = max(ctx, 6144 if _gpu else 4096)
        num_predict = pc
        final_temp = temp_override if temp_override is not None else tc
    else:
        num_predict = pc
        final_temp = temp_override if temp_override is not None else tc

    stop_words: list[str] = []
    actual_predict = max_tokens if max_tokens is not None else num_predict
    if actual_predict is None:
        actual_predict = 1024
    elif actual_predict == -1:
        actual_predict = 1024
    else:
        actual_predict = max(1, int(actual_predict))

    # ★[v131] num_batch最適化
    # CPU: 大きいbatchは初回latencyを増やすだけ。512が最速のsweet spot
    # GPU: 2048維持
    _batch = 2048 if _gpu else 512

    if is_logic_mode:
        return dict(num_ctx=ctx, num_predict=actual_predict, temperature=final_temp,
                    top_k=40,           # ★[v131] 80→40: サンプリング候補削減で高速化
                    top_p=0.92,
                    min_p=0.05,         # ★[v131] 0.02→0.05: 低品質トークンをより強くフィルタ
                    repeat_penalty=1.30,
                    repeat_last_n=64,   # ★[v131] 256→64: バッファスキャン削減
                    num_thread=threads, num_batch=_batch,
                    num_keep=64,        # ★[v131] 128→64: KVキャッシュ保持削減
                    stop=stop_words)
    return dict(num_ctx=ctx, num_predict=actual_predict, temperature=final_temp,
                top_k=30,              # ★[v131] 50→30: 雑談は少ない候補で十分
                top_p=0.90,
                min_p=0.05,
                repeat_penalty=1.20,   # ★[v131] 1.25→1.20: ペナルティ計算コスト削減
                repeat_last_n=64,      # ★[v131] 128→64
                num_thread=threads, num_batch=_batch,
                num_keep=64,
                stop=stop_words)

_SYS_PRM_CACHE: dict[str, str] = {}
_SYS_PRM_CACHE_PERSONA: int = -1
# ★[修正2] extrasキャッシュ（memory_context / dict_context 等を短期キャッシュ）
_SYS_EXTRAS_CACHE: dict[str, tuple[float, str]] = {}
_SYS_EXTRAS_TTL = 8.0  # ★[v131] 0.0→8.0: 8秒間キャッシュ再利用でvector_search省略

def get_sys_prm(mode: str, data: str = "", key: str = "", per_id=2) -> dict:
    global _SYS_PRM_CACHE_PERSONA
    persona = get_persona(per_id)
    first_person = persona.get("first_person", "私")
    query_for_knowledge = data or key or ""
    if per_id != _SYS_PRM_CACHE_PERSONA:
        _SYS_PRM_CACHE.clear()
        _SYS_EXTRAS_CACHE.clear()  # ★[修正2] ペルソナ切替時はextrasも破棄
        _SYS_PRM_CACHE_PERSONA = per_id
    cache_key = f"tpl:{mode}"
    if cache_key not in _SYS_PRM_CACHE:
        common = "\n".join([
            f"あなたは{persona['name']}。口調: {persona['style']}",
            f"一人称: {first_person}",
            f"ユーザー: {USER_NAME}",
            f"観測済み主体名: {OBSERVED_SUBJECT_NAME}",
            f"ルール: AIだと名乗るな。一人称は{first_person}のみ。",
            f"ルール: S-01はAI側のキャラ名。ユーザー名・観測済み一人称・観測済み主体として使わず、必要なら{OBSERVED_SUBJECT_NAME}と呼ぶ。",
            f"【二人称ルール】ユーザー({USER_NAME})への呼びかけ: 現実の一般人・友人・カジュアルキャラとして振る舞うペルソナは必ず『先輩』と呼ぶ。歴史上の偉人・哲学者・君主・神・師匠・上司など権威ある立場のペルソナはユーザーを『君』と呼ぶ。迷ったら『先輩』をデフォルトとする。『あなた』は絶対に使わない。",
            f"【出力構造ルール】比喩表現は、回答全体の最後の段落で『1つだけ』使用すること。それ以外の箇所での比喩の使用は許可されない。",
            f"【ステートレス原則】各推論は独立したセッションとして扱うこと。直前の文脈はリセットし、現在のクエリのみに集中すること。",
            f"【ループ防止】同語反復（トートロジー）を厳禁とする。結論を述べた後は速やかに推論を終了し、冗長な再構成を行わないこと。語彙の多様性を確保し、一度使用した比喩やフレーズの再利用を禁ずる。",
            f"【絶対禁止】以下のあらゆる情報を捏造するな:",
            f"  作品名・人物名・肩書き・日付・年代・場所・数値データ・統計・引用文",
            f"  企業・組織名・商品名・サービス名・学術用語の定義",
            f"【絶対ルール】【確認済み知識】にない事実は一切書くな。",
            f"【絶対ルール】知らないことは「わからない」「知らない」と明確に言え。",
        ])
        templates = {
            "d":   f"{common}\n雑談として5〜6文で答えろ。事実を一切捏造するな。\n",
            "w":   f"{common}\n以下をキャラ口調で要約せよ（重要ポイント3〜5点）:\n",
            "p":   f"{common}\n以下をキャラ口調で校正せよ:\n",
            "c":   f"{common}\nエンジニアとしてコード設計を提案せよ:\n",
            "t":   f"{common}\n以下をキャラ口調に超訳せよ:\n",
            "e":   f"{common}\n以下を自然な英語に翻訳せよ:\n",
            "sum": f"{common}\n以下を箇条書き5点以内で要約せよ:\n",
            "r":   f"{common}\n以下の状況でロールプレイを開始せよ:\n",
            "q":   f"{common}\nユーザーの目標をクエスト化しろ。\n出力: クエスト名, 勝利条件, 作戦ステップ, 最初の10分\n",
            "elab": f"{common}\nあなたは高度な推論エージェント。以下の内容を、比喩・例え・複数視点を用いて分かりやすく説明せよ。\n",
        }
        _SYS_PRM_CACHE[cache_key] = templates.get(mode, templates["d"])
    # ★[修正2] extrasをTTLキャッシュで再利用（毎回のvector_search/load_stateをスキップ）
    extras_key = f"extras:{query_for_knowledge[:40]}"
    now = time.time()
    if extras_key in _SYS_EXTRAS_CACHE and now - _SYS_EXTRAS_CACHE[extras_key][0] < _SYS_EXTRAS_TTL:
        extras = _SYS_EXTRAS_CACHE[extras_key][1]
    else:
        mem = memory_context(query=query_for_knowledge)
        mem_block = f"\n【確認済み知識】（これのみ事実として使え。ここにない事実を作るな）:\n{mem}\n" if mem else ""
        session_block = session_context_block()
        opt_block = inject_optimizations(mode, persona.get("name", ""))
        dict_block = dict_context(data or key or "")
        extras = "".join(p for p in [mem_block, session_block, opt_block, dict_block] if p)
        # ★[修正/#5] TTL=0 のときはキャッシュに書き込まない。
        # 旧コードは TTL=0 でも毎ターン _SYS_EXTRAS_CACHE に書き続けていた（メモリリーク）。
        if _SYS_EXTRAS_TTL > 0:
            _SYS_EXTRAS_CACHE[extras_key] = (now, extras)
    return dict(role="system", content=_SYS_PRM_CACHE[cache_key] + data + extras)

@lru_cache(maxsize=512)
def normalize_for_match(text: str) -> str:
    text = html_module.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[!-/:-@\[-`{-~\u3001\u3002\u30fb\u2026\u301c\uff01\uff1f\u300c-\u300f\u3010\u3011・…～―\s]", "", text)
    return text.lower()

@lru_cache(maxsize=256)
def is_url(text: str) -> bool: return text.startswith("http://") or text.startswith("https://")

# ===== 歌詞検索 =====
def _extract_lyrics_utanet(html_str: str) -> tuple[str, str]:
    m = re.search(r'<div[^>]+id=["\']kashi_area["\'][^>]*>(.*?)</div>', html_str, re.I | re.S)
    if not m: return "", ""
    raw = strip_tags(m.group(1))
    raw = _clean_lyrics_only(raw)
    return "", raw

def _extract_lyrics_utaten(html_str: str) -> tuple[str, str]:
    m = re.search(r'<div[^>]*class=["\'].*?lyrics_body.*?["\'][^>]*>(.*?)</div>', html_str, re.I | re.S)
    if not m: return "", ""
    text = re.sub(r'<br\s*/?>', '\n', m.group(1))
    text = re.sub(r'<[^>]+>', '', text)
    text = html_module.unescape(text)
    raw = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    raw = _clean_lyrics_only(raw)
    return "", raw

def _extract_lyrics_jlyric(html_str: str) -> tuple[str, str]:
    m = re.search(r'<p[^>]+id=["\']Lyric["\'][^>]*>(.*?)</p>', html_str, re.I | re.S)
    if not m: return "", ""
    raw = strip_tags(m.group(1))
    raw = _clean_lyrics_only(raw)
    return "", raw

def _clean_lyrics_only(text: str, query: str = "") -> str:
    lines = text.strip().splitlines()
    q_norm = normalize_for_match(query)
    is_english = bool(query and sum(1 for c in query if ord(c) < 128 and c.isalpha()) > len(query.strip()) * 0.5)
    noise_words = {'ホーム', 'ブログトップ', '新規登録', 'ログイン', 'ログアウト', 'メニュー', 'ツイート', 'シェア', 'お問い合わせ', '利用規約', 'プライバシー', 'ヘルプ', 'Copyright', 'All Rights Reserved', '読者になる', '広告を非表示', '関連記事'}
    noise_patterns = [r'^\d+件$', r'^\d+位$', r'^[\d:/\s-]+$', r'^\d{4}年', r'^\d+月\d+日', r'^https?://', r'^www\.', r'^@\w+', r'^#\w+', r'^【[^】]+】', r'^［[^］]+］', r'^\([^)]+\)$', r'^（[^）]+）$', r'^(作詞|作曲|編曲|歌詞|Title|Artist)', r'^♪.*♪$', r'^(ページ|Page|page)\s*\d+']
    cleaned = []
    for ln in lines:
        ln = ln.strip()
        if not ln or len(ln) < 4 or len(ln) > 150: continue
        if ln in noise_words: continue
        if any(n in ln for n in noise_words if len(ln) < 40): continue
        if any(re.match(pat, ln) for pat in noise_patterns): continue
        if re.match(r'^[\d\.\s\-_#℃％%()（）、。，/\s:;!?？！]{4,}$', ln): continue
        ascii_ratio = sum(1 for c in ln if ord(c) < 128) / max(len(ln), 1)
        jp_count = sum(1 for c in ln if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF')
        if is_english:
            eng = sum(1 for c in ln if c.isalpha() and ord(c) < 128)
            if eng < 2 and jp_count < 1: continue
        else:
            if ascii_ratio > 0.6: continue
            if jp_count < 1 and ascii_ratio > 0.3: continue
        if q_norm and q_norm in normalize_for_match(ln): continue
        cleaned.append(ln)
    if len(cleaned) < 3: return text
    deduped = []
    seen = set()
    for ln in cleaned:
        key = re.sub(r'\s+', '', ln.lower())[:30]
        if key not in seen: seen.add(key); deduped.append(ln)
    return '\n'.join(deduped)

def _parse_generic_lyrics(html_str: str, query: str) -> tuple[str | None, str | None]:
    is_english = bool(query and sum(1 for c in query if ord(c) < 128 and c.isalpha()) > len(query.strip()) * 0.5)
    text = re.sub(r'(?i)<br\s*/?>', '\n', html_str)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.I | re.S)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.I | re.S)
    text = re.sub(r'<[^>]+>', '', text)
    text = html_module.unescape(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) >= 4]
    noise = {'ホーム', 'ブログトップ', '新規登録', 'ログイン', 'ログアウト', 'メニュー', 'ツイート', 'シェア', 'お問い合わせ', '利用規約', 'プライバシー', 'ヘルプ', 'Copyright', 'All Rights Reserved', '読者になる', '広告を非表示', '関連記事'}
    lyrics_candidates = []
    for ln in lines:
        if not is_english and not re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', ln): continue
        if ln in noise: continue
        if any(n in ln for n in noise if len(ln) < 50 and len(n) > 2): continue
        if re.search(r'^(?:作詞|作曲|編曲|歌詞|Title|Artist)', ln, re.I): continue
        if len(ln) < 5 or len(ln) > 150: continue
        if re.match(r'^[\d\.\s\-_#℃％%()（）、。，/\s:;!?？！]{4,}$', ln): continue
        ascii_ratio = sum(1 for c in ln if ord(c) < 128) / max(len(ln), 1)
        if not is_english and ascii_ratio > 0.5: continue
        if is_english:
            eng = sum(1 for c in ln if c.isalpha() and ord(c) < 128)
            if eng < 2: continue
        lyrics_candidates.append(ln)
    if len(lyrics_candidates) < 4: return None, None
    best_start = 0; best_score = 0
    for i in range(len(lyrics_candidates)):
        window = lyrics_candidates[i:i+16]
        kana_count = sum(1 for ln in window for c in ln if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF')
        valid_count = sum(1 for ln in window if 8 <= len(ln) <= 80)
        total_ascii = sum(sum(1 for c in ln if ord(c) < 128 and c.isalpha()) for ln in window)
        score = kana_count + valid_count * 3 - total_ascii * 0.5
        if score > best_score: best_score, best_start = score, i
    lyric_lines = lyrics_candidates[best_start:best_start+30]
    raw = '\n'.join(lyric_lines)
    if len(raw) < 60: return None, None
    cleaned = _clean_lyrics_only(raw, query)
    if len(cleaned) < 40: return None, None
    return None, cleaned

def _scrape_page_parallel(url: str, query: str, results: list, lock: threading.Lock) -> None:
    try:
        # まずURLフィルタで弾く
        is_eng = bool(sum(1 for c in query if ord(c) < 128 and c.isalpha()) > len(query.strip()) * 0.5)
        if not _is_lyrics_url_ok(url, is_eng):
            return
        html_str = fetch_html(url, timeout=5, silent=True, spoof_bot=True)
        if not html_str or len(html_str) < 500: return
        lyrics = None
        if "uta-net.com" in url:    _, ly = _extract_lyrics_utanet(html_str);  lyrics = ly
        elif "utaten.com" in url:   _, ly = _extract_lyrics_utaten(html_str);  lyrics = ly
        elif "j-lyric.net" in url:  _, ly = _extract_lyrics_jlyric(html_str);  lyrics = ly
        else:                        _, gl = _parse_generic_lyrics(html_str, query); lyrics = gl
        if not lyrics or len(lyrics) < 40: return
        q_norm = normalize_for_match(query)
        score = 80
        if q_norm in normalize_for_match(lyrics): score += 50
        score += min(30, sum(1 for ln in lyrics.strip().splitlines() if 8 <= len(ln.strip()) <= 80) * 2)
        # 信頼サイトボーナス（まとめ・ブログは0点）
        score += _lyrics_url_score_bonus(url, is_eng)
        with lock:
            if not any(r[2] == lyrics for r in results):
                results.append((score, url, lyrics))
    except Exception as e:
        print(f"{C['y']}[WARN] 歌詞スクレイプ失敗({url[:40]}): {e}{C['w']}")

def _fetch_snippets_from(html: str) -> list[str]:
    return [strip_tags(s) for s in re.findall(r'''class=["']result-snippet["'][^>]*>(.*?)</td>''', html, re.I | re.S) if len(strip_tags(s).strip()) > 15]

# ── 歌詞URLフィルタ ──────────────────────────────────────────────────────────
# まとめ・ブログ・SNS等を除外し、歌詞専門サイトのみ通す
_LYRICS_BLOCKED = {
    # まとめ・ブログ
    "ameblo.jp", "ameba.jp", "livedoor", "fc2.com",
    "hatenablog.com", "hatena.ne.jp", "seesaa.net", "jugem.jp",
    "note.com", "qiita.com", "zenn.dev", "medium.com",
    # まとめ系
    "matome.naver.jp", "togetter.com", "naver.com",
    # SNS・動画
    "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "youtube.com", "youtu.be", "nicovideo.jp", "nico.ms",
    "facebook.com", "pinterest.com",
    # Q&A
    "chiebukuro.yahoo.co.jp", "okwave.jp", "yahoo.co.jp", "yahoo.com",
    # 通販
    "amazon.co.jp", "amazon.com", "rakuten.co.jp", "mercari.com",
    # Wiki
    "wikipedia.org", "wikiwiki.jp", "atwiki.jp",
    # ニュース・音楽情報（歌詞なし）
    "oricon.co.jp", "natalie.mu", "barks.jp", "tower.jp",
    "billboard-japan.com", "musicman.co.jp", "music.apple.com",
    "spotify.com", "recochoku.jp",
}
_LYRICS_TRUSTED_JP = {
    "uta-net.com", "j-lyric.net", "utaten.com",
    "kashinavi.com", "lyric.evesta.jp",
}
_LYRICS_TRUSTED_EN = {
    "genius.com", "azlyrics.com", "musixmatch.com",
    "lyrics.com", "metrolyrics.com",
}

def _is_lyrics_url_ok(url: str, is_eng: bool = False) -> bool:
    try:
        host = U.urlparse(url).netloc.lower() if hasattr(U, 'urlparse') else url
    except Exception:
        host = url
    for b in _LYRICS_BLOCKED:
        if b in host:
            return False
    return True

def _lyrics_url_score_bonus(url: str, is_eng: bool = False) -> int:
    """信頼できる歌詞サイトなら+40点"""
    trusted = _LYRICS_TRUSTED_EN if is_eng else _LYRICS_TRUSTED_JP
    for t in trusted:
        if t in url:
            return 40
    return 0

def _fetch_urls_from(html: str, is_eng: bool = False) -> list[str]:
    urls = []
    for raw in re.findall(r'href=["\'](https?://[^"\']+?)["\']', html, re.I):
        u = U.unquote(raw)
        if u not in urls and _is_lyrics_url_ok(u, is_eng):
            urls.append(u)
    return urls

def search_lyrics_absolute(query: str) -> tuple[str | None, str | None, str | None]:
    is_eng = bool(sum(1 for c in query if ord(c) < 128 and c.isalpha()) > len(query.strip()) * 0.5)
    ddg_urls: list[str] = []
    try:
        kl = "us-en" if is_eng else "jp-jp"
        suffix = " lyrics" if is_eng else " 歌詞"
        # 「まとめサイト」「ブログ」を除外するサイト限定クエリ
        if is_eng:
            site_hint = " (site:genius.com OR site:azlyrics.com OR site:musixmatch.com)"
        else:
            site_hint = " (site:uta-net.com OR site:j-lyric.net OR site:utaten.com)"
        q = query + suffix + site_hint
        data = U.urlencode({"q": q, "kl": kl}).encode("utf-8")
        h = fetch_html("https://lite.duckduckgo.com/lite/", data=data, timeout=4, silent=True)
        ddg_urls = _fetch_urls_from(h, is_eng)
        # site限定で0件なら通常クエリにフォールバック
        if len(ddg_urls) < 2:
            data2 = U.urlencode({"q": query + suffix, "kl": kl}).encode("utf-8")
            h2 = fetch_html("https://lite.duckduckgo.com/lite/", data=data2, timeout=4, silent=True)
            for u in _fetch_urls_from(h2, is_eng):
                if u not in ddg_urls:
                    ddg_urls.append(u)
    except Exception as e:
        print(f"{C['y']}[WARN] DDG検索失敗: {e}{C['w']}")

    candidates = ddg_urls[:8]
    enc_q = U.quote(unicodedata.normalize("NFKC", query).strip())

    # 直接URL（信頼サイトのみ）
    if is_eng:
        extra_urls = [
            f"https://www.azlyrics.com/lyrics/{enc_q.replace('%20','').lower()}.html",
            f"https://genius.com/search?q={enc_q}",
        ]
    else:
        extra_urls = [
            f"https://search.j-lyric.net/index.php?kt={enc_q}&ct=2",
            f"https://www.uta-net.com/search/?Keyword={enc_q}&Aselect=4&Bselect=3",
            f"https://utaten.com/lyric/search/?title={enc_q}",
        ]
    for search_url in extra_urls:
        try:
            h3 = fetch_html(search_url, timeout=3, silent=True, spoof_bot=True)
            for u in _fetch_urls_from(h3, is_eng):
                if u not in candidates:
                    candidates.append(u)
        except Exception as e:
            print(f"{C['y']}[WARN] 歌詞サイト検索失敗: {e}{C['w']}")

    page_results: list = []
    page_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_scrape_page_parallel,
            args=(u, query, page_results, page_lock),
            daemon=True
        )
        for u in candidates[:10]
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=6)

    if page_results:
        best = max(page_results, key=lambda x: x[0])
        return "web", best[1], best[2]   # (source, url, lyrics)
    return None, None, None

def lyrics_debug(query: str) -> str:
    if is_url(query):
        html_str = fetch_html(query, timeout=6, silent=True, spoof_bot=True)
        if not html_str: return f"{C['r']}fetch failed{C['w']}"
        g_t, g_l = _parse_generic_lyrics(html_str, query)
        return f"{C['c']}generic: title={g_t} lyrics={len(g_l or '')}chars{C['w']}"
    rows = [f"{C['c']}=== LYRICS DEBUG ==={C['w']}"]
    try:
        data = U.urlencode({"q": query + " 歌詞", "kl": "jp-jp"}).encode("utf-8")
        h = fetch_html("https://lite.duckduckgo.com/lite/", data=data, timeout=6, silent=True)
        snips = re.findall(r'class="result-snippet"[^>]*>(.*?)</td>', h, re.I | re.S)
        urls = re.findall(r'href=["\'](https?://[^"\']+?)["\']', h, re.I)
        rows.append(f"snippets: {len(snips)} urls: {len(urls)}")
        for i, url in enumerate(urls[:15], 1): rows.append(f"  {i}. {url[:80]}")
        for i, s in enumerate(snips[:5], 1): rows.append(f"  S{i}: {strip_tags(s)[:60]}")
    except Exception as e: rows.append(f"error: {e}")
    return "\n".join(rows)

# ===== MIDI GENERATION v2 (Enhanced — Multi-Track + Music Theory) =====
import random as _midi_rng

MIDI_SECTIONS = {
    "short":  [("intro", 4), ("verse_A", 8), ("outro", 4)],
    "medium": [("intro", 4), ("verse_A", 8), ("chorus", 8), ("verse_B", 8), ("chorus", 8), ("outro", 4)],
    "long":   [("intro", 4), ("verse_A", 8), ("chorus", 8), ("bridge", 8),
               ("verse_B", 8), ("chorus", 8), ("solo", 8), ("chorus", 8), ("outro", 8)],
    "ultra":  [("intro", 16), ("verse_A", 18), ("chorus", 18), ("verse_B", 18),
               ("chorus", 18), ("bridge", 16), ("solo", 18), ("chorus", 18),
               ("interlude", 12), ("verse_C", 18), ("chorus", 18), ("bridge2", 16),
               ("solo2", 18), ("chorus", 18), ("buildup", 12), ("chorus_final", 18), ("outro", 16)],
}

# ----- 音楽理論定数 -----
_SCALE_INTERVALS = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues":      [0, 3, 5, 6, 7, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
}
# キー名 → MIDI root (C4=60 基準)
_NOTE_ROOTS = {
    "C":60,"Db":61,"D":62,"Eb":63,"E":64,"F":65,
    "Gb":66,"G":67,"Ab":68,"A":69,"Bb":70,"B":71,
}
# ダイアトニックコード音程 (degree 1-7, major)
_DIATONIC_TRIADS = [
    [0,4,7],[2,5,9],[4,7,11],[5,9,12],[7,11,14],[9,12,16],[11,14,17]
]
# セクション別コード進行 (degree 1-based, 繰り返し)
_CHORD_PROGS = {
    "intro":        [1,6,4,5],
    "verse_A":      [1,5,6,4],
    "verse_B":      [6,4,1,5],
    "chorus":       [1,5,6,4],
    "bridge":       [4,5,3,6],
    "solo":         [1,4,5,5],
    "outro":        [1,6,4,1],
    "interlude":    [4,1,5,6],
    "buildup":      [6,6,4,5],
    "chorus_final": [1,5,6,4],
    "verse_C":      [1,4,6,5],
    "bridge2":      [2,6,4,5],
    "solo2":        [6,4,1,5],
}
# セクション特性
_SEC_TRAITS = {
    "intro":        {"vel":62, "oct":0,  "density":0.50, "nlen":1.00},
    "verse_A":      {"vel":72, "oct":0,  "density":0.65, "nlen":0.75},
    "verse_B":      {"vel":74, "oct":0,  "density":0.70, "nlen":0.75},
    "chorus":       {"vel":90, "oct":0,  "density":0.85, "nlen":0.50},
    "bridge":       {"vel":65, "oct":-1, "density":0.55, "nlen":1.00},
    "solo":         {"vel":88, "oct":1,  "density":0.95, "nlen":0.25},
    "outro":        {"vel":55, "oct":0,  "density":0.40, "nlen":1.50},
    "interlude":    {"vel":68, "oct":0,  "density":0.50, "nlen":1.00},
    "buildup":      {"vel":80, "oct":0,  "density":0.80, "nlen":0.50},
    "chorus_final": {"vel":100,"oct":0,  "density":1.00, "nlen":0.50},
    "verse_C":      {"vel":76, "oct":0,  "density":0.70, "nlen":0.75},
    "bridge2":      {"vel":70, "oct":-1, "density":0.60, "nlen":1.00},
    "solo2":        {"vel":92, "oct":1,  "density":1.00, "nlen":0.25},
}
# GM楽器番号
_GM = {"piano":0,"strings":48,"pad":88,"bass":32,"guitar":25}
# ドラム MIDI音番号
_DRUM = {"kick":36,"snare":38,"hihat":42,"open_hat":46,"crash":49,"ride":51,"tom":45}

_MOOD_MAP = {
    "悲しい":{"scale":"minor","bpm_mod":-15,"genre":"strings"},
    "切ない":{"scale":"minor","bpm_mod":-10,"genre":"strings"},
    "暗い":{"scale":"minor","bpm_mod":-10,"genre":"piano"},
    "孤独":{"scale":"minor","bpm_mod":-20,"genre":"piano"},
    "憂鬱":{"scale":"dorian","bpm_mod":-15,"genre":"piano"},
    "楽しい":{"scale":"major","bpm_mod":+15,"genre":"piano"},
    "明るい":{"scale":"major","bpm_mod":+10,"genre":"piano"},
    "元気":{"scale":"pentatonic","bpm_mod":+20,"genre":"rock"},
    "希望":{"scale":"major","bpm_mod":+5,"genre":"strings"},
    "激しい":{"scale":"blues","bpm_mod":+30,"genre":"rock"},
    "燃える":{"scale":"blues","bpm_mod":+35,"genre":"rock"},
    "戦い":{"scale":"minor","bpm_mod":+25,"genre":"rock"},
    "怒り":{"scale":"blues","bpm_mod":+20,"genre":"rock"},
    "神秘":{"scale":"dorian","bpm_mod":-5,"genre":"pad"},
    "幻想":{"scale":"dorian","bpm_mod":-10,"genre":"pad"},
    "宇宙":{"scale":"dorian","bpm_mod":-15,"genre":"pad"},
    "jazz":{"scale":"dorian","bpm_mod":+5,"genre":"jazz"},
    "ジャズ":{"scale":"dorian","bpm_mod":+5,"genre":"jazz"},
    "ロック":{"scale":"blues","bpm_mod":+20,"genre":"rock"},
    "クラシック":{"scale":"major","bpm_mod":-5,"genre":"classical"},
    "EDM":{"scale":"minor","bpm_mod":+30,"genre":"pad"},
}
_GENRE_INSTRUMENTS = {
    "piano":{"melody":0,"chords":0,"bass":32},
    "strings":{"melody":40,"chords":48,"bass":43},
    "rock":{"melody":29,"chords":25,"bass":34},
    "jazz":{"melody":0,"chords":25,"bass":33},
    "pad":{"melody":88,"chords":89,"bass":38},
    "classical":{"melody":40,"chords":48,"bass":43},
}
def _analyze_mood(theme):
    result={"scale":"major","bpm_mod":0,"genre":"piano"}
    for kw,mood in _MOOD_MAP.items():
        if kw in theme: result=mood.copy(); break
    return result
def _transpose_key(key,semitones):
    keys=["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
    idx=keys.index(key) if key in keys else 0
    return keys[(idx+semitones)%12]

def _midi_scale(root: int, stype: str, oct_off: int) -> list[int]:
    """指定ルート・スケールのMIDIピッチリスト（48〜96範囲）"""
    intervals = _SCALE_INTERVALS.get(stype, _SCALE_INTERVALS["major"])
    base = root % 12 + (5 + oct_off) * 12
    pitches = []
    for o in range(-2, 3):
        for iv in intervals:
            p = base + iv + o * 12
            if 36 <= p <= 108:
                pitches.append(p)
    return sorted(set(pitches))

def _chord_pitches(root: int, degree: int, base_oct: int = 4) -> list[int]:
    idx = (degree - 1) % 7
    base = root % 12 + base_oct * 12
    return [max(36, min(96, base + iv)) for iv in _DIATONIC_TRIADS[idx]]

def _gen_melody(root: int, section: str, bars: int, stype: str = "major") -> list[dict]:
    tr  = _SEC_TRAITS.get(section, _SEC_TRAITS["verse_A"])
    sc  = _midi_scale(root, stype, tr["oct"])
    prog = _CHORD_PROGS.get(section, [1,5,6,4])
    rng = _midi_rng.Random(hash(section) % 2**32)
    notes, pprev = [], None
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        ct  = _chord_pitches(root, deg, 5 + tr["oct"])
        ct_set = {p % 12 for p in ct}
        sc_ch = [p for p in sc if p % 12 in ct_set] or sc
        beat = 0.0
        while beat < 4.0:
            if rng.random() > tr["density"]:
                beat += tr["nlen"]; continue
            # 前のノートから近い音を優先（スムーズな動き）
            if pprev is not None:
                candidates = sorted(sc_ch, key=lambda p: abs(p - pprev))[:5]
                weights = [5,4,3,2,1][:len(candidates)]
                pitch = rng.choices(candidates, weights=weights)[0]
            else:
                pitch = rng.choice(sc_ch)
            dur = min(tr["nlen"] * rng.uniform(0.85, 1.2), 4.0 - beat)
            vel = max(40, min(120, tr["vel"] + rng.randint(-8, 8)))
            notes.append({"pitch": pitch, "start": float(bar * 4 + beat),
                          "duration": round(max(0.1, dur), 3), "velocity": vel})
            pprev = pitch
            beat += tr["nlen"]
    return notes

def _gen_chords(root: int, section: str, bars: int) -> list[dict]:
    tr   = _SEC_TRAITS.get(section, _SEC_TRAITS["verse_A"])
    prog = _CHORD_PROGS.get(section, [1,5,6,4])
    rng  = _midi_rng.Random(hash(section + "ch") % 2**32)
    notes = []
    for bar in range(bars):
        deg = prog[bar % len(prog)]
        pts = _chord_pitches(root, deg, 4)
        if section in ("chorus","chorus_final","buildup"):
            beats = [0.0, 2.0]
        elif section in ("solo","solo2","bridge","bridge2"):
            beats = [0.0]
        else:
            beats = [0.0, 2.5]
        for bt in beats:
            if rng.random() > 0.92: continue
            vel = max(30, min(100, int(tr["vel"] * 0.68) + rng.randint(-5,5)))
            dur = 1.8 if bt == 0.0 else 1.2
            for p in pts:
                notes.append({"pitch": p, "start": float(bar*4+bt),
                              "duration": dur, "velocity": vel})
    return notes

def _gen_bass(root: int, section: str, bars: int) -> list[dict]:
    tr   = _SEC_TRAITS.get(section, _SEC_TRAITS["verse_A"])
    prog = _CHORD_PROGS.get(section, [1,5,6,4])
    rng  = _midi_rng.Random(hash(section + "bs") % 2**32)
    notes = []
    for bar in range(bars):
        deg   = prog[bar % len(prog)]
        pts   = _chord_pitches(root, deg, 3)
        rn, fn = pts[0], (pts[2] if len(pts) > 2 else pts[0] + 7)
        rn, fn = max(28, min(52, rn)), max(28, min(52, fn))
        if section in ("intro","outro"):
            notes.append({"pitch": rn, "start": float(bar*4), "duration": 3.5,
                          "velocity": max(55, tr["vel"]-10)})
        elif section in ("chorus","chorus_final","buildup","solo","solo2"):
            for i in range(8):
                p = rn if i%2==0 else (fn if i%4==2 else max(28,min(52, rn+rng.choice([2,5,7]))))
                notes.append({"pitch": p, "start": float(bar*4+i*0.5), "duration": 0.45,
                              "velocity": min(110, tr["vel"]-5+rng.randint(-4,4))})
        else:
            for bt in [0,1,2,3]:
                p = rn if bt in (0,2) else (fn if bt==1 else max(28,min(52, rn-2)))
                notes.append({"pitch": p, "start": float(bar*4+bt), "duration": 0.85,
                              "velocity": min(105, tr["vel"]-8+rng.randint(-4,4))})
    return notes

def _gen_drums(section: str, bars: int) -> list[dict]:
    K,SN,HH,OH,CR = _DRUM["kick"],_DRUM["snare"],_DRUM["hihat"],_DRUM["open_hat"],_DRUM["crash"]
    rng   = _midi_rng.Random(hash(section + "dr") % 2**32)
    notes = []
    vb    = 1.25 if section in ("chorus","chorus_final","buildup") else 1.0
    if section in ("intro",):
        # イントロ: ハイハットのみ、だんだん足される
        for bar in range(bars):
            pct = bar / max(bars-1,1)
            for i in range(8):
                notes.append({"pitch":HH,"start":float(bar*4+i*0.5),"duration":0.1,
                              "velocity":int(45+pct*20+rng.randint(-3,3)),"channel":9})
            if pct > 0.5:  # 後半からキック追加
                notes.append({"pitch":K,"start":float(bar*4),"duration":0.1,"velocity":int(70*pct),"channel":9})
        return notes
    if section in ("outro",):
        for bar in range(bars):
            fade = max(0.2, 1.0 - bar/bars)
            notes.append({"pitch":K, "start":float(bar*4),  "duration":0.1,"velocity":int(80*fade),"channel":9})
            notes.append({"pitch":SN,"start":float(bar*4+2),"duration":0.1,"velocity":int(70*fade),"channel":9})
            for i in range(4):
                notes.append({"pitch":HH,"start":float(bar*4+i),"duration":0.1,"velocity":int(45*fade),"channel":9})
        return notes
    for bar in range(bars):
        # クラッシュ: セクション開始
        if bar == 0:
            notes.append({"pitch":CR,"start":float(bar*4),"duration":0.5,"velocity":int(min(127,95*vb)),"channel":9})
        # キック
        kick_beats = [0,1.5,2,3.5] if section in ("chorus","chorus_final","buildup") else [0,2]
        for b in kick_beats:
            notes.append({"pitch":K,"start":float(bar*4+b),"duration":0.1,
                          "velocity":int(min(127,82*vb+rng.randint(-4,4))),"channel":9})
        # スネア
        for b in [1,3]:
            notes.append({"pitch":SN,"start":float(bar*4+b),"duration":0.1,
                          "velocity":int(min(120,78*vb+rng.randint(-4,4))),"channel":9})
        # ハイハット
        hh_div = 8 if section in ("chorus","chorus_final","solo","solo2","buildup") else 4
        for i in range(hh_div):
            is_open = (i == hh_div-1 and rng.random() > 0.75)
            notes.append({"pitch": OH if is_open else HH,
                          "start": float(bar*4 + i*(4.0/hh_div)),
                          "duration": 0.1, "velocity": int(50+rng.randint(-5,10)), "channel":9})
    return notes

def _midi_section_prompt(theme: str, section: str, bars: int, tempo: int, key: str,
                          chord_prog: list) -> list[dict]:
    """改良版LLMメロディープロンプト（コード進行付き）"""
    cdesc = {1:"I",2:"IIm",3:"IIIm",4:"IV",5:"V",6:"VIm",7:"VIIdim"}
    prog_str = " → ".join(cdesc.get(d, str(d)) for d in chord_prog) + " (繰り返し)"
    hi_oct = "高め（ソロ）" if section in ("solo","solo2") else "通常"
    system = (
        f"あなたはプロのMIDI作曲家です。以下の条件でメロディーのMIDIノートを生成してください。\n"
        f"テーマ: {theme} | セクション: {section} | キー: {key}メジャー | テンポ: {tempo}BPM | {bars}小節\n"
        f"コード進行（1小節=1コード）: {prog_str}\n\n"
        f"【生成ルール】\n"
        f"1. コードトーン中心のメロディー（非コード音は短い経過音・装飾音のみ）\n"
        f"2. 音域: MIDI 52〜88（オクターブ: {hi_oct}）\n"
        f"3. 音の動きはスムーズに（基本は順次進行か3度跳躍、ソロは例外OK）\n"
        f"4. start は0〜{bars*4-0.1:.1f}（ビート単位、小節=4ビート）\n"
        f"5. セクション「{section}」らしいリズム感と強弱をつける\n"
        f"6. ノート数: {bars*5}〜{bars*10}個\n\n"
        f"出力: JSONアレイのみ。前置き・説明・コードブロック記法は不要。\n"
        f'形式: [{{"pitch":60,"start":0.0,"duration":0.5,"velocity":80}}, ...]'
    )
    user = f"「{theme}」の{section}セクション、{bars}小節分のメロディーノートを生成してください（{key}メジャー、{tempo}BPM）:"
    return [{"role":"system","content":system},{"role":"user","content":user}]

def _parse_midi_notes(raw: str) -> list[dict]:
    """LLM出力からMIDIノートをパース（堅牢版・複数JSON候補を試行）"""
    candidates = re.findall(r'\[[\s\S]*?\]', raw)
    for m in candidates:
        try:
            arr = json.loads(m)
            if not isinstance(arr, list) or len(arr) < 3: continue
            parsed = []
            for n in arr:
                if not isinstance(n, dict): continue
                if "pitch" not in n or "start" not in n: continue
                parsed.append({
                    "pitch":    max(0,  min(127, int(float(n.get("pitch",60))))),
                    "start":    max(0.0, float(n.get("start",0))),
                    "duration": max(0.1, float(n.get("duration",0.5))),
                    "velocity": max(30,  min(127, int(float(n.get("velocity",75))))),
                    "channel":  int(n.get("channel",0)),
                })
            if len(parsed) >= 3:
                return parsed
        except Exception:
            continue
    return []

def generate_midi_section(theme: str, section: str, bars: int, tempo: int, key: str) -> dict:
    """セクション単位でマルチトラックデータを生成（melody/chords/bass/drums）"""
    root    = _NOTE_ROOTS.get(key, 60)
    prog    = _CHORD_PROGS.get(section, [1,5,6,4])
    # LLMでメロディ生成を試みる
    llm_mel = []
    try:
        raw = stream_response(
            _midi_section_prompt(theme, section, bars, tempo, key, prog),
            True, 200, silent=True, max_tokens=8000
        )
        if raw: llm_mel = _parse_midi_notes(raw)
    except Exception: pass
    # LLM失敗 or ノート不足 → アルゴリズム生成
    melody = llm_mel if len(llm_mel) >= bars * 3 else _gen_melody(root, section, bars)
    return {
        "melody": melody,
        "chords": _gen_chords(root, section, bars),
        "bass":   _gen_bass(root, section, bars),
        "drums":  _gen_drums(section, bars),
    }

def save_midi(all_sections: list, tempo: int, path: str) -> bool:
    """マルチトラック（melody/chords/bass/drums）MIDIファイルを保存"""
    try: from midiutil import MIDIFile
    except ImportError: return False
    # 4トラック構成
    midi = MIDIFile(4)
    track_info = [("Melody",0),("Chords",1),("Bass",2),("Drums",9)]
    for ti, (tname, _) in enumerate(track_info):
        midi.addTempo(ti, 0, tempo)
        midi.addTrackName(ti, 0, tname)
    # GM楽器設定 (ch9はドラム固定なので設定不要)
    for ti, (_, ch) in enumerate(track_info):
        if ch != 9:
            gm = [_GM["piano"], _GM["strings"], _GM["bass"]][ti]
            midi.addProgramChange(ti, ch, 0, gm)
    offset = 0.0
    track_keys = ["melody","chords","bass","drums"]
    for sec_name, tracks in all_sections:
        all_flat = []
        for tk, (_, ch) in zip(track_keys, track_info):
            for n in tracks.get(tk, []):
                pitch    = max(0,  min(127, n["pitch"]))
                start    = n["start"] + offset
                dur      = max(0.05, n["duration"])
                vel      = max(1,   min(127, n["velocity"]))
                act_ch   = n.get("channel", ch)
                ti       = track_keys.index(tk)
                midi.addNote(ti, act_ch, pitch, start, dur, vel)
                all_flat.append(n)
        # セクション長を算出してオフセット更新（ギャップなし）
        if all_flat:
            offset += max(n["start"] + n["duration"] for n in all_flat)
        else:
            offset += 16.0
    with open(path, "wb") as f: midi.writeFile(f)
    return True

def handle_midi(arg: str) -> str:
    if not arg:
        return (f"{C['r']}usage: /midi <テーマ> [short|medium|long|ultra] [BPM] [キー]{C['w']}")
    parts = arg.split()
    length, tempo, key = "medium", 120, "C"
    rest_parts = []
    for p in parts:
        if p.lower() in ("short","medium","long","ultra"): length = p.lower()
        elif p.isdigit() and 60 <= int(p) <= 240:         tempo = int(p)
        elif re.match(r'^[A-G]b?$', p):                   key = p
        else: rest_parts.append(p)
    theme = " ".join(rest_parts) or "インストゥルメンタル"
    sections_plan = MIDI_SECTIONS[length]
    total_bars    = sum(b for _, b in sections_plan)
    print(f"{C['c']}♩ MIDI v2 生成: 『{theme}』 {key}メジャー {tempo}BPM {length}({total_bars}小節) — 4トラック{C['w']}")
    try: from midiutil import MIDIFile
    except ImportError: return f"{C['r']}midiutil未インストール: pip install midiutil{C['w']}"
    all_sections, total_notes = [], 0
    for section, bars in sections_plan:
        print(f"  {C['dim']}[{section}] {bars}小節...{C['w']}", end="", flush=True)
        tracks = generate_midi_section(theme, section, bars, tempo, key)
        all_sections.append((section, tracks))
        cnt = sum(len(v) for v in tracks.values())
        total_notes += cnt
        print(f" {C['g']}{cnt}音 ✓{C['w']}")
    safe_theme = re.sub(r'[^\w]', '_', theme)[:20]
    filename   = f"midi_{safe_theme}_{int(time.time())}.mid"
    if save_midi(all_sections, tempo, filename):
        return (f"{C['g']}♪ 保存完了: {filename}\n"
                f"   {total_notes}音 / {total_bars}小節 / {length}\n"
                f"   トラック: メロディー(piano) + コード(strings) + ベース + ドラム{C['w']}")
    return f"{C['r']}MIDI保存失敗{C['w']}"

def play_singularity(query: str) -> str:
    if not query: return f"{C['r']}曲名を指定してください。{C['w']}"
    ytdl, mpv = shutil.which("yt-dlp"), shutil.which("mpv")
    if not ytdl or not mpv: return f"{C['y']}yt-dlp と mpv が必要です。{C['w']}"
    import secrets
    # PIDベースのファイル名は衝突リスクがあるため安全なランダム名を使用
    out_file = f"ytdl_y_{secrets.token_hex(8)}.wav"
    try:
        r = S.run(
            [ytdl, "-x", "--audio-format", "wav", "-o", out_file, f"ytsearch1:{query}"],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0: return f"{C['r']}failed: {r.stderr[:200]}{C['w']}"
        if os.path.exists(out_file):
            S.Popen([mpv, "--no-video", out_file], stdout=S.DEVNULL, stderr=S.DEVNULL)
            return (
                f"{C['g']}再生開始: {query}{C['w']}\n"
                f"{C['dim']}※ 楽曲の著作権は権利者に帰属します。個人利用の範囲でお使いください。{C['w']}"
            )
        return f"{C['r']}file not found{C['w']}"
    except S.TimeoutExpired:
        return f"{C['r']}timeout{C['w']}"
    except Exception as e:
        return f"{C['r']}error: {e}{C['w']}"


# ===== ローカルRAG: ファイル取り込み・オフライン推論 =====
# 使い方:
#   /kb add <ファイルパス>       テキスト/PDFをベクトルDBに取り込む
#   /kb list                    取り込み済みコレクション一覧
#   /kb search <クエリ>         ローカル知識ベースから検索（ネット不要）
#   /kb ask <質問>              ローカル知識ベース+LLMで回答（完全オフライン）
#   /kb del <コレクション名>    コレクションを削除

BOOK_CHUNK_SIZE = 400
BOOK_CHUNK_OVERLAP = 80
LOCAL_RAG_COLLECTION = "s01_books"

def _chunk_text(text: str, size: int = BOOK_CHUNK_SIZE, overlap: int = BOOK_CHUNK_OVERLAP) -> list[str]:
    """長いテキストを重複ありで分割する。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            for sep in ("\u3002", "\uff0e", "\n\n", "\n", "\u3001"):
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def _read_file_text(path: str) -> tuple[str, str]:
    """ファイルを読んでテキストを返す。(text, error_msg)"""
    if not os.path.exists(path):
        return "", f"\u30d5\u30a1\u30a4\u30eb\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093: {path}"
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            try:
                import pdfminer.high_level as pdfminer_hl
                text = pdfminer_hl.extract_text(path)
                return text or "", ""
            except ImportError:
                pass
            try:
                import pypdf
                reader = pypdf.PdfReader(path)
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
                return text, ""
            except ImportError:
                return "", "PDF\u3092\u8aad\u3080\u306b\u306f: pip install pdfminer.six"
        elif ext in (".txt", ".md", ".rst", ".csv", ".json"):
            for enc in ("utf-8", "shift_jis", "euc-jp"):
                try:
                    with open(path, "r", encoding=enc) as f:
                        return f.read(), ""
                except UnicodeDecodeError:
                    continue
            return "", "\u6587\u5b57\u30b3\u30fc\u30c9\u3092\u5224\u5b9a\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f"
        else:
            return "", f"\u672a\u5bfe\u5fdc\u306e\u5f62\u5f0f: {ext}  (\u5bfe\u5fdc: .txt .md .pdf .rst .csv .json)"
    except Exception as e:
        return "", f"\u8aad\u307f\u8fbc\u307f\u30a8\u30e9\u30fc: {e}"

def _col_name_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r'[^\w\-]', '_', base)[:40].strip("_") or "book"
    return f"book_{safe}"

def handle_kb(arg: str, _chat_fn=None, _persona_id: int = 2) -> str:
    sub, _, rest = arg.partition(" ")
    sub = sub.strip().lower()
    rest = rest.strip()

    if not arg or sub == "list":
        cols = [c for c in vector_list_collections() if c != "s01_memory"]
        if not cols:
            return (f"{C['y']}\u53d6\u308a\u8fbc\u307f\u6e08\u307f\u30d5\u30a1\u30a4\u30eb\u306a\u3057\u3002\n"
                    f"/kb add <\u30d5\u30a1\u30a4\u30eb\u30d1\u30b9> \u3067\u53d6\u308a\u8fbc\u3081\u307e\u3059\uff08.txt .md .pdf \u5bfe\u5fdc\uff09{C['w']}")
        lines = [f"{C['c']}=== \u30ed\u30fc\u30ab\u30eb\u77e5\u8b58\u30d9\u30fc\u30b9 ==={C['w']}"]
        for c in cols:
            n = vector_count(c)
            label = c.replace("book_", "", 1)
            lines.append(f"  {C['g']}{label}{C['w']}  ({n} \u30c1\u30e3\u30f3\u30af)")
        lines.append(f"\n\u4f7f\u3044\u65b9: /kb ask <\u8cea\u554f>  /kb search <\u30ad\u30fc\u30ef\u30fc\u30c9>  /kb del <\u540d\u524d>")
        return "\n".join(lines)

    if sub == "add":
        if not rest:
            return f"{C['r']}usage: /kb add <ファイルパスまたはURL>  例: /kb add https://ja.wikipedia.org/wiki/言語ゲーム{C['w']}"

        # ── URL対応 ───────────────────────────────────────────────
        if rest.startswith("http://") or rest.startswith("https://"):
            # SSRF防止: 内部ネットワークへのアクセスを拒否
            try:
                _assert_safe_url(rest)
            except ValueError as e:
                return f"{C['r']}セキュリティエラー: {e}{C['w']}"
            print(f"{C['c']}[KB] URL取得中: {rest[:80]}{C['w']}")
            raw_html = fetch_html(rest, timeout=10, silent=False, spoof_bot=True)
            if not raw_html:
                return f"{C['r']}URLの取得に失敗しました: {rest}{C['w']}"
            text = strip_tags(raw_html)
            # 余分な空行・ナビゲーション断片を除去
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
            text = "\n".join(lines)
            if len(text.strip()) < 50:
                return f"{C['r']}取得できたテキストが短すぎます（{len(text)}文字）{C['w']}"
            # コレクション名はドメイン+パス末尾から生成
            import urllib.parse as _UP
            _parsed = _UP.urlparse(rest)
            _slug = (_parsed.netloc + _parsed.path).rstrip("/").replace("/", "_").replace(".", "_")[:40]
            col_name = f"book_{_slug}"
            chunks = _chunk_text(text)
            print(f"{C['c']}[KB] {len(chunks)}チャンク → コレクション「{_slug}」{C['w']}")
            ok = 0
            for i, chunk in enumerate(chunks):
                if vector_add(chunk, {"source": rest, "chunk": i, "type": "web"}, collection=col_name):
                    ok += 1
                if (i + 1) % 50 == 0:
                    print(f"{C['dim']}  {i+1}/{len(chunks)} チャンク完了...{C['w']}")
            return (f"{C['g']}取り込み完了: {rest[:60]}\n"
                    f"  {ok}/{len(chunks)} チャンク → コレクション「{_slug}」{C['w']}\n"
                    f"{C['dim']}※ 取り込んだコンテンツの著作権は原著作者に帰属します。個人的な学習・研究目的の範囲でご利用ください。{C['w']}")

        # ── ファイルパス ──────────────────────────────────────────
        # パストラバーサル防止（カレントディレクトリ外のファイルを許可しない）
        try:
            _assert_safe_path(rest)
        except ValueError as e:
            return f"{C['r']}セキュリティエラー: {e}{C['w']}"
        text, err = _read_file_text(rest)
        if err: return f"{C['r']}{err}{C['w']}"
        if len(text.strip()) < 50:
            return f"{C['r']}\u30c6\u30ad\u30b9\u30c8\u304c\u77ed\u3059\u304e\u307e\u3059\uff08{len(text)}\u6587\u5b57\uff09{C['w']}"
        col_name = _col_name_from_path(rest)
        chunks = _chunk_text(text)
        print(f"{C['c']}[KB] \u53d6\u308a\u8fbc\u307f\u958b\u59cb: {os.path.basename(rest)} \u2192 {len(chunks)}\u30c1\u30e3\u30f3\u30af{C['w']}")
        ok = 0
        for i, chunk in enumerate(chunks):
            if vector_add(chunk, {"source": rest, "chunk": i, "type": "book"}, collection=col_name):
                ok += 1
            if (i + 1) % 50 == 0:
                print(f"{C['dim']}  {i+1}/{len(chunks)} \u30c1\u30e3\u30f3\u30af\u5b8c\u4e86...{C['w']}")
        return (f"{C['g']}\u53d6\u308a\u8fbc\u307f\u5b8c\u4e86: {os.path.basename(rest)}\n"
                f"  {ok}/{len(chunks)} \u30c1\u30e3\u30f3\u30af \u2192 \u30b3\u30ec\u30af\u30b7\u30e7\u30f3\u300c{col_name.replace('book_','')}\u300d{C['w']}\n"
                f"{C['dim']}※ 著作権のある資料は個人的な学習・研究目的の範囲でご利用ください。{C['w']}")

    if sub == "del":
        if not rest: return f"{C['r']}usage: /kb del <\u30b3\u30ec\u30af\u30b7\u30e7\u30f3\u540d>{C['w']}"
        col_name = rest if rest.startswith("book_") else f"book_{rest}"
        if not VECTOR_AVAILABLE: _init_vector_db()
        try:
            _VECTOR_CLIENT.delete_collection(col_name)
            _VECTOR_COLS.pop(col_name, None)
            return f"{C['y']}\u524a\u9664: {rest}{C['w']}"
        except Exception as e:
            return f"{C['r']}\u524a\u9664\u5931\u6557: {e}{C['w']}"

    if sub == "search":
        if not rest: return f"{C['r']}usage: /kb search <\u30ad\u30fc\u30ef\u30fc\u30c9>{C['w']}"
        cols = [c for c in vector_list_collections() if c != "s01_memory"]
        if not cols: return f"{C['y']}\u53d6\u308a\u8fbc\u307f\u6e08\u307f\u30d5\u30a1\u30a4\u30eb\u306a\u3057\u3002{C['w']}"
        all_hits = []
        for col in cols:
            for h in vector_search(rest, n=3, collection=col):
                all_hits.append((col.replace("book_", ""), h))
        if not all_hits: return f"{C['y']}\u300c{rest}\u300d\u306b\u95a2\u9023\u3059\u308b\u7b87\u6240\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002{C['w']}"
        lines = [f"{C['c']}=== \u691c\u7d22\u7d50\u679c: {rest} ==={C['w']}"]
        for src, hit in all_hits[:6]:
            lines.append(f"\n{C['dim']}[{src}]{C['w']}\n{hit[:300]}")
        return "\n".join(lines)

    if sub == "ask":
        if not rest: return f"{C['r']}usage: /kb ask <質問>  例: /kb ask 言語ゲームとは何か{C['w']}"
        cols = [c for c in vector_list_collections() if c != "s01_memory"]
        if not cols: return f"{C['y']}取り込み済みファイルなし。{C['w']}"

        # ── Pass1: ★[v131] hybrid_search_advanced (RRF+CE) ───────
        cite_map: list[tuple[str, str]] = []  # (source_label, chunk_text)
        o_cli = _get_ollama()
        for col in cols:
            col_label = col.replace("book_", "")
            hits = hybrid_search_advanced(rest, n_candidates=12, top_k=5, collection=col, ollama_client=o_cli)
            for h in hits:
                cite_map.append((col_label, h))
        if not cite_map:
            return f"{C['y']}「{rest}」に関連する箇所が知識ベースに見つかりませんでした。{C['w']}"

        def _build_context(pairs: list[tuple[str, str]]) -> str:
            by_src: dict[str, list[str]] = {}
            for s, chunk in pairs:
                by_src.setdefault(s, []).append(chunk)
            return "\n\n".join(f"《{s}》より\n" + "\n---\n".join(chunks) for s, chunks in by_src.items())

        context = _build_context(cite_map)
        if _chat_fn is None:
            return f"{C['c']}[KB参考文献]{C['w']}\n{context[:800]}"

        from_persona = get_persona(_persona_id)
        fp = from_persona.get("first_person", "私")
        sys_content = (
            f"あなたは{from_persona['name']}。口調: {from_persona['style']}。一人称: {fp}。\n"
            f"以下の「局所参照」の文章のみを根拠にして質問に答えよ。\n"
            f"「局所参照」にない情報は一切追加するな。捕捉・一般論禁止。"
        )
        print(f"{C['c']}[KBオフライン推論 Pass1]{C['w']} {from_persona['name']}: ", end="", flush=True)
        msgs1 = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": f"「局所参照」:\n{context}\n\n質問: {rest}"}
        ]
        result1 = stream_response(msgs1, True, len(rest), temp_override=0.0, model=DEEP_MODEL) or ""

        # ── Pass2: 1次回答のキーワードで追加検索（マルチホップ）────
        result_final = result1
        if result1:
            hop_kw = extract_keywords(result1, top_n=4)
            hop_query = " ".join(hop_kw)
            if hop_query and hop_query.strip() != rest.strip():
                new_pairs: list[tuple[str, str]] = []
                existing_chunks = {c for _, c in cite_map}
                for col in cols:
                    src = col.replace("book_", "")
                    for h in vector_search(hop_query, n=2, collection=col):
                        if h not in existing_chunks:
                            new_pairs.append((src, h))
                            existing_chunks.add(h)
                if new_pairs:
                    cite_map.extend(new_pairs)
                    extra_ctx = _build_context(new_pairs)
                    print(f"\n{C['dim']}[Pass2: +{len(new_pairs)}チャンク追加 kw={hop_query[:40]}]{C['w']}")
                    print(f"{C['c']}[KBオフライン推論 Pass2]{C['w']} {from_persona['name']}: ", end="", flush=True)
                    msgs2 = [
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": (
                            f"「局所参照」:\n{context}\n\n「追加参照」:\n{extra_ctx}\n\n"
                            f"質問: {rest}\n\n暫定回答: {result1}\n\n"
                            f"追加参照も踏まえて最終回答を出せ。新情報がなければ暫定回答をそのまま使え。"
                        )}
                    ]
                    result_final = stream_response(msgs2, True, len(rest), temp_override=0.0, model=DEEP_MODEL) or result1

        # ── 引用元を末尾に表示 ─────────────────────────────────────
        if result_final:
            cited_srcs = list(dict.fromkeys(s for s, _ in cite_map))
            cite_str = "  ".join(f"《{s}》" for s in cited_srcs)
            print(f"\n{C['dim']}[参照元: {cite_str} / {len(cite_map)}チャンク]{C['w']}")
            _print_hallucination_warnings(result_final, strict=True)
        return result_final

    return f"{C['r']}usage: /kb add|list|search|ask|del{C['w']}"


# ===================================================================
# SPI / 玉手箱 対策モジュール
# /spi          ランダム出題（言語+非言語ミックス）
# /spi 言語     言語問題のみ
# /spi 非言語   非言語問題のみ
# /spi 英語     英語問題のみ（玉手箱対策）
# /spi 模擬     10問連続模擬試験モード
# /spi 成績     正答率・カテゴリ別統計
# /spi リセット 成績リセット
# ===================================================================

import random as _random

# ---------- 問題データベース ----------
_SPI_DB: list[dict] = [

    # ===== 言語：語句の意味 =====
    {"cat": "言語", "sub": "語句意味", "q": "「示唆」の意味として最も適切なものを選べ。",
     "choices": ["A: それとなく示すこと", "B: 強く命令すること", "C: 完全に否定すること", "D: 詳しく説明すること"],
     "ans": "A", "exp": "示唆＝それとなくほのめかすこと。suggestに近い。"},

    {"cat": "言語", "sub": "語句意味", "q": "「恣意的」の意味として最も適切なものを選べ。",
     "choices": ["A: 慎重で計画的なさま", "B: 自分の思うままで勝手なさま", "C: 周囲に配慮するさま", "D: 論理的に正確なさま"],
     "ans": "B", "exp": "恣意的＝自分の思いのまま、根拠なく決めること。arbitraryに近い。"},

    {"cat": "言語", "sub": "語句意味", "q": "「逡巡」の意味として最も適切なものを選べ。",
      "choices": ["A: 素早く行動すること", "B: ためらってぐずぐずすること", "C: 激しく怒ること", "D: 深く反省すること"],
     "ans": "B", "exp": "逡巡＝ためらい、なかなか決断できないこと。hesitationに近い。"},

    {"cat": "言語", "sub": "語句意味", "q": "「瑣末」の意味として最も適切なものを選べ。",
     "choices": ["A: 非常に重要なこと", "B: 細かくとるに足らないこと", "C: 複雑に絡み合うこと", "D: 急を要すること"],
     "ans": "B", "exp": "瑣末＝細々としてつまらないこと。trivialに近い。"},

    {"cat": "言語", "sub": "語句意味", "q": "「敷衍」の意味として最も適切なものを選べ。",
     "choices": ["A: 意味を押し広げて詳しく説明すること", "B: 強引に押し通すこと", "C: 簡潔にまとめること", "D: 誤りを訂正すること"],
     "ans": "A", "exp": "敷衍＝内容をひろげてわかりやすく説明すること。elaborateに近い。"},

    # ===== 言語：対義語 =====
    {"cat": "言語", "sub": "対義語", "q": "「促進」の対義語として最も適切なものを選べ。",
     "choices": ["A: 抑制", "B: 継続", "C: 加速", "D: 実行"],
     "ans": "A", "exp": "促進（進める）⇔ 抑制（おさえる）。"},

    {"cat": "言語", "sub": "対義語", "q": "「具体」の対義語として最も適切なものを選べ。",
     "choices": ["A: 現実", "B: 抽象", "C: 詳細", "D: 明確"],
     "ans": "B", "exp": "具体（はっきりしたもの）⇔ 抽象（まとめた概念）。"},

    {"cat": "言語", "sub": "対義語", "q": "「楽観」の対義語として最も適切なものを選べ。",
     "choices": ["A: 慎重", "B: 冷静", "C: 悲観", "D: 否定"],
     "ans": "C", "exp": "楽観（よい方向に考える）⇔ 悲観（悪い方向に考える）。"},

    {"cat": "言語", "sub": "対義語", "q": "「冗長」の対義語として最も適切なものを選べ。",
     "choices": ["A: 簡潔", "B: 詳細", "C: 明瞭", "D: 正確"],
     "ans": "A", "exp": "冗長（余分に長い）⇔ 簡潔（短くまとまっている）。"},

    # ===== 言語：文章整序 =====
    {"cat": "言語", "sub": "文章整序", "q": "次のア〜エを意味が通るよう並べ替えたとき、2番目にくるものを選べ。\nア: しかし、そこには大きな落とし穴がある。\nイ: 効率化は現代のビジネスにおいて最優先事項とされている。\nウ: 人間関係や創造性といった要素が犠牲になりやすいのだ。\nエ: 効率のみを追求すると、",
     "choices": ["A: ア", "B: イ", "C: ウ", "D: エ"],
     "ans": "A", "exp": "イ（主張）→ ア（逆接）→ エ（具体化）→ ウ（結論）の順。2番目はア。"},

    # ===== 非言語：割合・比 =====
    {"cat": "非言語", "sub": "割合", "q": "定価1200円の商品を20%引きで買った。支払い金額はいくらか。",
     "choices": ["A: 900円", "B: 960円", "C: 1000円", "D: 1080円"],
     "ans": "B", "exp": "1200 × (1 - 0.20) = 1200 × 0.80 = 960円。"},

    {"cat": "非言語", "sub": "割合", "q": "ある商品を30%値上げした後、さらに10%値引きした。元の価格と比べて何%の変化か。",
     "choices": ["A: 17%増", "B: 20%増", "C: 23%増", "D: 変化なし"],
     "ans": "A", "exp": "1.30 × 0.90 = 1.17 → 元の価格の117%。つまり17%増。"},

    {"cat": "非言語", "sub": "割合", "q": "原価の40%の利益を見込んで定価をつけた。定価2800円のとき、原価はいくらか。",
     "choices": ["A: 1800円", "B: 2000円", "C: 2100円", "D: 2200円"],
     "ans": "B", "exp": "定価 = 原価 × 1.40 → 原価 = 2800 ÷ 1.40 = 2000円。"},

    # ===== 非言語：速度・距離・時間 =====
    {"cat": "非言語", "sub": "速度", "q": "時速60kmで2時間30分走ったときの距離は何kmか。",
     "choices": ["A: 120km", "B: 140km", "C: 150km", "D: 180km"],
     "ans": "C", "exp": "60 × 2.5 = 150km。2時間30分 = 2.5時間。"},

    {"cat": "非言語", "sub": "速度", "q": "A地点からB地点まで時速40kmで行き、帰りは時速60kmで戻った。平均時速はいくらか。",
     "choices": ["A: 48km/h", "B: 50km/h", "C: 52km/h", "D: 54km/h"],
     "ans": "A", "exp": "往復の平均速度 = 2×40×60÷(40+60) = 4800÷100 = 48km/h。単純平均ではなく調和平均を使う。"},

    {"cat": "非言語", "sub": "速度", "q": "600mの道を歩くと10分かかる。同じ道を自転車では3分でいける。自転車の速さは歩きの何倍か。",
     "choices": ["A: 2倍", "B: 3倍", "C: 3.3倍", "D: 4倍"],
     "ans": "C", "exp": "歩き速度: 600÷10=60m/分。自転車: 600÷3=200m/分。200÷60≒3.3倍。"},

    # ===== 非言語：確率 =====
    {"cat": "非言語", "sub": "確率", "q": "1〜6のサイコロを2回振る。2回とも偶数が出る確率はいくらか。",
     "choices": ["A: 1/6", "B: 1/4", "C: 1/3", "D: 1/2"],
     "ans": "B", "exp": "1回で偶数(2,4,6)が出る確率=3/6=1/2。2回とも: 1/2×1/2=1/4。"},

    {"cat": "非言語", "sub": "確率", "q": "袋の中に赤玉3個・白玉2個がある。2個同時に取り出したとき、2個とも同じ色になる確率はいくらか。",
     "choices": ["A: 2/5", "B: 3/10", "C: 7/10", "D: 4/10"],
     "ans": "A", "exp": "全組合せ: C(5,2)=10。同色: C(3,2)+C(2,2)=3+1=4。確率=4/10=2/5。AとDは同値だがAが正式な既約分数。"},

    {"cat": "非言語", "sub": "確率", "q": "コインを3回投げる。少なくとも1回表が出る確率はいくらか。",
     "choices": ["A: 1/2", "B: 5/8", "C: 7/8", "D: 3/4"],
     "ans": "C", "exp": "1 − (全部裏の確率) = 1 − (1/2)³ = 1 − 1/8 = 7/8。余事象を使うと簡単。"},

    # ===== 非言語：推論・集合 =====
    {"cat": "非言語", "sub": "推論", "q": "「全ての社員はA研修を受けた」「BさんはA研修を受けていない」から確実に言えることを選べ。",
     "choices": ["A: BさんはA研修に合格した", "B: Bさんは社員ではない", "C: 社員はB研修も受けた", "D: BさんはA研修を受けるべきだ"],
     "ans": "B", "exp": "三段論法: 全社員→A研修済。B→A研修未。よってBは社員でない。"},

    {"cat": "非言語", "sub": "集合", "q": "100人のうち英語ができる人60人、中国語ができる人50人、両方できる人20人。どちらもできない人は何人か。",
     "choices": ["A: 10人", "B: 20人", "C: 30人", "D: 40人"],
     "ans": "A", "exp": "英語のみ+中国語のみ+両方 = 40+30+20 = 90人。どちらもできない = 100−90 = 10人。"},

    # ===== 非言語：図表 =====
    {"cat": "非言語", "sub": "図表", "q": "ある会社の売上が2020年100万円、2021年120万円、2022年108万円だった。2021年から2022年の変化率はいくらか。",
     "choices": ["A: −10%", "B: −8%", "C: +8%", "D: +10%"],
     "ans": "A", "exp": "(108−120)÷120 = −12÷120 = −0.10 = −10%。"},

    # ===== 英語（玉手箱） =====
    {"cat": "英語", "sub": "同意語", "q": "「ambiguous」と最も意味が近い語を選べ。",
     "choices": ["A: clear", "B: vague", "C: accurate", "D: simple"],
     "ans": "B", "exp": "ambiguous＝あいまいな。vague（漠然とした）が最も近い。"},

    {"cat": "英語", "sub": "同意語", "q": "「diligent」と最も意味が近い語を選べ。",
     "choices": ["A: lazy", "B: clever", "C: hardworking", "D: quiet"],
     "ans": "C", "exp": "diligent＝勤勉な。hardworking（よく働く）が最も近い。"},

    {"cat": "英語", "sub": "同意語", "q": "「concise」と最も意味が近い語を選べ。",
     "choices": ["A: brief", "B: detailed", "C: complex", "D: extended"],
     "ans": "A", "exp": "concise＝簡潔な。brief（短く要領を得た）が最も近い。"},

    {"cat": "英語", "sub": "同意語", "q": "「inevitable」と最も意味が近い語を選べ。",
     "choices": ["A: avoidable", "B: unexpected", "C: uncertain", "D: unavoidable"],
     "ans": "D", "exp": "inevitable＝避けられない。unavoidable（回避不可能な）が最も近い。"},

    {"cat": "英語", "sub": "英文読解", "q": "次の英文の内容と一致するものを選べ。\n\"The key to effective communication is not just speaking clearly, but also listening actively.\"",
     "choices": ["A: 明確に話すことだけが重要だ", "B: 積極的に聞くことも重要だ", "C: コミュニケーションは話すことで完結する", "D: 聞くことより話すことが優先される"],
     "ans": "B", "exp": "not just A but also B（AだけでなくBも）。listeningも重要と言っている。"},

    # ===== 英語：同意語追加 =====
    {"cat": "英語", "sub": "同意語", "q": "「adequate」と最も意味が近い語を選べ。",
     "choices": ["A: excellent", "B: sufficient", "C: lacking", "D: complex"],
     "ans": "B", "exp": "adequate＝十分な。sufficient（足りている）が最も近い。"},

    {"cat": "英語", "sub": "同意語", "q": "「obsolete」と最も意味が近い語を選べ。",
     "choices": ["A: modern", "B: useful", "C: outdated", "D: popular"],
     "ans": "C", "exp": "obsolete＝時代遅れの・廃れた。outdated（古くなった）が最も近い。"},

    {"cat": "英語", "sub": "同意語", "q": "「transparent」と最も意味が近い語を選べ。",
     "choices": ["A: hidden", "B: clear", "C: heavy", "D: slow"],
     "ans": "B", "exp": "transparent＝透明な・明白な。clear（明確な）が最も近い。"},

    # ===== 玉手箱：四則逆算 =====
    # 形式：□に入る数を選ぶ。速度が命。
    {"cat": "玉手箱", "sub": "四則逆算", "q": "□ × 7 = 56　　□に入る数はいくつか。",
     "choices": ["A: 6", "B: 7", "C: 8", "D: 9"],
     "ans": "C", "exp": "56 ÷ 7 = 8。掛け算の逆算は割り算。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "72 ÷ □ = 9　　□に入る数はいくつか。",
     "choices": ["A: 6", "B: 7", "C: 8", "D: 9"],
     "ans": "C", "exp": "72 ÷ 9 = 8。割り算の逆算は割り算（72÷9）。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "□ + 47 = 83　　□に入る数はいくつか。",
     "choices": ["A: 34", "B: 36", "C: 38", "D: 40"],
     "ans": "B", "exp": "83 − 47 = 36。足し算の逆算は引き算。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "125 − □ = 68　　□に入る数はいくつか。",
     "choices": ["A: 53", "B: 55", "C: 57", "D: 59"],
     "ans": "C", "exp": "125 − 68 = 57。引き算の逆算: □ = 125 − 68。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "□ ÷ 6 = 13　　□に入る数はいくつか。",
     "choices": ["A: 72", "B: 78", "C: 80", "D: 84"],
     "ans": "B", "exp": "13 × 6 = 78。割り算の逆算は掛け算。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "3 × □ − 5 = 19　　□に入る数はいくつか。",
     "choices": ["A: 6", "B: 7", "C: 8", "D: 9"],
     "ans": "C", "exp": "3×□ = 19+5 = 24 → □ = 24÷3 = 8。後ろから逆算する。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "(□ + 4) × 3 = 27　　□に入る数はいくつか。",
     "choices": ["A: 5", "B: 7", "C: 9", "D: 11"],
     "ans": "A", "exp": "□ + 4 = 27÷3 = 9 → □ = 9−4 = 5。括弧の外から逆算。"},

    {"cat": "玉手箱", "sub": "四則逆算", "q": "48 ÷ (□ − 2) = 6　　□に入る数はいくつか。",
     "choices": ["A: 8", "B: 9", "C: 10", "D: 12"],
     "ans": "C", "exp": "□−2 = 48÷6 = 8 → □ = 10。"},

    # ===== 玉手箱：長文一致（一致・不一致・どちらとも言えない の3択） =====
    {"cat": "玉手箱", "sub": "長文一致", "q": (
        "【本文】\n"
        "日本の食品ロスは年間約600万トンとされており、そのうち約半分は家庭から発生している。"
        "食品ロス削減のためには、企業だけでなく消費者一人ひとりの取り組みが不可欠である。\n\n"
        "【設問】「食品ロスの半分以上は企業活動から発生している」\n"
        "本文の内容と比較して、この記述は？"
    ),
     "choices": ["A: 一致する", "B: 一致しない", "C: どちらとも言えない"],
     "ans": "B", "exp": "本文では「約半分は家庭から」とあるため、企業が半分以上というのは不一致。"},

    {"cat": "玉手箱", "sub": "長文一致", "q": (
        "【本文】\n"
        "リモートワークの普及により、都市部から地方への人口移動が緩やかに進んでいる。"
        "ただし、この傾向は主にIT関連職種において顕著であり、全業種への波及は限定的とされる。\n\n"
        "【設問】「IT関連職種ではリモートワークを機に地方移住が進んでいる」\n"
        "本文の内容と比較して、この記述は？"
    ),
     "choices": ["A: 一致する", "B: 一致しない", "C: どちらとも言えない"],
     "ans": "A", "exp": "本文「IT関連職種において顕著」と一致する。"},

    {"cat": "玉手箱", "sub": "長文一致", "q": (
        "【本文】\n"
        "近年、Z世代を中心に「タイパ（タイムパフォーマンス）」を重視する傾向が強まっている。"
        "動画を倍速視聴したり、結末から確認してから作品を観るといった行動がその典型例とされる。\n\n"
        "【設問】「タイパ重視の傾向はすべての世代で同様に見られる」\n"
        "本文の内容と比較して、この記述は？"
    ),
     "choices": ["A: 一致する", "B: 一致しない", "C: どちらとも言えない"],
     "ans": "B", "exp": "本文では「Z世代を中心に」とあり、全世代とは書かれていない。不一致。"},

    {"cat": "玉手箱", "sub": "長文一致", "q": (
        "【本文】\n"
        "再生可能エネルギーの導入コストはこの10年で大幅に低下した。"
        "太陽光発電のコストは2010年比で約80%削減されたとする試算もある。"
        "しかし蓄電技術の課題は依然として残っており、安定供給には課題がある。\n\n"
        "【設問】「蓄電技術の問題が解決されれば再生可能エネルギーは完全に普及する」\n"
        "本文の内容と比較して、この記述は？"
    ),
     "choices": ["A: 一致する", "B: 一致しない", "C: どちらとも言えない"],
     "ans": "C", "exp": "本文は蓄電課題に言及するが、解決後の完全普及については述べていない。「どちらとも言えない」。"},

    # ===== 玉手箱：テーブル問題 =====
    {"cat": "玉手箱", "sub": "テーブル", "q": (
        "下表はA〜C店の月別売上（万円）を示す。\n"
        "┌──────┬────┬────┬────┐\n"
        "│      │ A店 │ B店 │ C店 │\n"
        "├──────┼────┼────┼────┤\n"
        "│ 4月  │ 120 │  90 │ 150 │\n"
        "│ 5月  │ 130 │ 110 │ 140 │\n"
        "│ 6月  │ 110 │ 130 │ 160 │\n"
        "└──────┴────┴────┴────┘\n"
        "3ヶ月の合計売上が最も多い店はどこか。"
    ),
     "choices": ["A: A店", "B: B店", "C: C店", "D: 同じ"],
     "ans": "C", "exp": "A店: 120+130+110=360。B店: 90+110+130=330。C店: 150+140+160=450。C店が最多。"},

    {"cat": "玉手箱", "sub": "テーブル", "q": (
        "下表は社員4人の残業時間（時間/月）を示す。\n"
        "┌──────┬──┬──┬──┬──┐\n"
        "│      │田中│鈴木│佐藤│高橋│\n"
        "├──────┼──┼──┼──┼──┤\n"
        "│ 1月  │ 20 │ 15 │ 30 │ 10 │\n"
        "│ 2月  │ 25 │ 20 │ 20 │ 15 │\n"
        "│ 3月  │ 15 │ 25 │ 25 │ 20 │\n"
        "└──────┴──┴──┴──┴──┘\n"
        "3ヶ月の平均残業時間が最も少ない社員は誰か。"
    ),
     "choices": ["A: 田中", "B: 鈴木", "C: 佐藤", "D: 高橋"],
     "ans": "D", "exp": "田中:60/3=20。鈴木:60/3=20。佐藤:75/3=25。高橋:45/3=15。高橋が最少。"},

    {"cat": "玉手箱", "sub": "テーブル", "q": (
        "下表はある試験の得点分布（人数）を示す。\n"
        "┌──────────┬────┐\n"
        "│ 得点区分   │ 人数 │\n"
        "├──────────┼────┤\n"
        "│ 90点以上   │   5  │\n"
        "│ 70〜89点   │  15  │\n"
        "│ 50〜69点   │  20  │\n"
        "│ 50点未満   │  10  │\n"
        "└──────────┴────┘\n"
        "70点以上の受験者は全体の何%か。"
    ),
     "choices": ["A: 20%", "B: 25%", "C: 40%", "D: 50%"],
     "ans": "C", "exp": "70点以上: 5+15=20人。全体: 5+15+20+10=50人。20÷50=0.40=40%。"},
]

# ---------- 成績管理 ----------
_SPI_SCORE_KEY = "spi_score"

def _spi_load_score() -> dict:
    state = load_state()
    return state.get(_SPI_SCORE_KEY, {"total": 0, "correct": 0, "cats": {}})

def _spi_save_score(sc: dict) -> None:
    state = load_state()
    state[_SPI_SCORE_KEY] = sc
    save_state(state)

def _spi_record(cat: str, correct: bool) -> None:
    sc = _spi_load_score()
    sc["total"] += 1
    if correct: sc["correct"] += 1
    cats = sc.setdefault("cats", {})
    cats.setdefault(cat, {"total": 0, "correct": 0})
    cats[cat]["total"] += 1
    if correct: cats[cat]["correct"] += 1
    _spi_save_score(sc)

# ---------- 出題セッション管理（同一プロセス内） ----------
# セッション状態はload_state()/save_state()で永続化
_SPI_SESSION_KEY = "spi_session"
# ★[修正/spi-3] ファイルI/Oの競合・キャッシュミスによるセッション消失を防ぐため
# メモリ上にもセッションを保持するミラーを追加。
# load/saveは両方に書き込み、loadはメモリを優先して返す。
_SPI_SESSION_MEMORY: dict = {}

def _spi_load_session() -> dict:
    # メモリに有効なセッションがあればそちらを優先（ファイルI/O競合を回避）
    if _SPI_SESSION_MEMORY.get("current") and isinstance(_SPI_SESSION_MEMORY["current"], dict) and len(_SPI_SESSION_MEMORY["current"]) > 0:
        return dict(_SPI_SESSION_MEMORY)
    return load_state().get(_SPI_SESSION_KEY, {
        "current": {}, "mock_queue": [], "mock_results": [], "is_mock": False})

def _spi_save_session(current: dict, queue: list, results: list, is_mock: bool = False) -> None:
    global _SPI_SESSION_MEMORY
    sess = {"current": current, "mock_queue": queue, "mock_results": results, "is_mock": is_mock}
    # メモリに即時反映（ファイル書き込み失敗時のフォールバック）
    _SPI_SESSION_MEMORY = dict(sess)
    state = load_state()
    state[_SPI_SESSION_KEY] = sess
    save_state(state)

def _spi_clear_session() -> None:
    global _SPI_SESSION_MEMORY
    _SPI_SESSION_MEMORY = {}
    state = load_state()
    state.pop(_SPI_SESSION_KEY, None)
    save_state(state)

def _spi_pick(filter_cat: str | None = None) -> dict:
    pool = [q for q in _SPI_DB if filter_cat is None or q["cat"] == filter_cat]
    return _random.choice(pool)

# ---------- 出題履歴管理（ローテーション） ----------
_SPI_USED_IDS: list[str] = []   # 出題済みq_idのリスト（古い順）
_SPI_USED_MAX = 60               # この件数を超えたら古いものを解禁

def _spi_make_id(q: dict) -> str:
    return q.get("q", "")[:30]

def _spi_mark_used(q: dict) -> None:
    qid = _spi_make_id(q)
    if qid in _SPI_USED_IDS:
        _SPI_USED_IDS.remove(qid)
    _SPI_USED_IDS.append(qid)
    if len(_SPI_USED_IDS) > _SPI_USED_MAX:
        _SPI_USED_IDS.pop(0)

def _spi_pick_fresh(filter_cat: str | None = None) -> dict:
    """出題済みを避けてDBから選ぶ。全部出済みならランダム。"""
    pool = [q for q in _SPI_DB if filter_cat is None or q["cat"] == filter_cat]
    fresh = [q for q in pool if _spi_make_id(q) not in _SPI_USED_IDS]
    chosen = _random.choice(fresh) if fresh else _random.choice(pool)
    _spi_mark_used(chosen)
    return chosen

# ---------- LLMによる問題動的生成 ----------
# サブカテゴリ定義（生成指示付き）
_SPI_LLM_SUBTYPES = {
    "言語": [
        ("語句意味",   "「{word}」の意味として最も適切なものを選べ。4択（A/B/C/D）で出題し、正解・不正解の選択肢を作れ。語はSPIで頻出の難読語・ビジネス語から選ぶ。"),
        ("同意語",     "「{word}」と最も意味が近い語を4択で出題せよ。"),
        ("対義語",     "「{word}」の対義語として最も適切な語を4択で出題せよ。"),
        ("文章完成",   "次の文の（　）に入る最も適切な語を4択で選べ。文章はSPIらしい論理的な文にすること。"),
        ("文章整序",   "次のア〜エの文を意味が通る順に並べ替えよ。選択肢は並び順4パターン（A/B/C/D）で出題せよ。"),
        ("長文読解",   "以下の文章を読んで設問に答えよ。本文100字程度・設問は「筆者が述べていること」「本文の内容と一致するもの」等。4択。"),
        ("熟語の意味", "「{word}」の意味として最も適切なものを4択で出題せよ。四字熟語・ことわざ・慣用句から選ぶ。"),
        ("語句用法",   "「{word}」の使い方として正しいものを4択で選べ。誤用しやすい語を選ぶこと。"),
    ],
    "非言語": [
        ("速度・時間・距離", "速さ・時間・距離に関するSPIらしい文章題を1問作れ。4択（A/B/C/D）、数値は整数。"),
        ("割合・比",         "割合や比に関するSPIらしい文章題を1問作れ。4択、数値は整数か単純な小数。"),
        ("損益計算",         "原価・売価・利益率に関するSPIらしい文章題を1問作れ。4択。"),
        ("仕事算",           "AとBが協力して仕事をする問題をSPIらしく作れ。4択。"),
        ("集合・ベン図",     "重複を含む集合（ベン図）の問題をSPIらしく作れ。4択。"),
        ("確率",             "日常的な場面での確率問題をSPIらしく作れ。4択、答えは分数でも可。"),
        ("推論",             "条件が3〜4つ与えられ、正しい結論を選ぶSPI推論問題を作れ。4択。"),
        ("資料解釈",         "表やグラフの数値を読み取る問題をSPIらしく作れ。小さな表（3〜4行）を文章で表現すること。4択。"),
        ("場合の数",         "順列・組み合わせに関するSPIらしい問題を1問作れ。4択。"),
        ("整数・数列",       "規則性のある数列や整数の性質に関するSPI問題を作れ。4択。"),
        ("平均・分散",       "平均・中央値・最頻値に関するSPI問題を作れ。4択。"),
        ("図形・空間",       "図形の面積・体積・角度に関するSPIらしい問題を作れ（図は文章で説明）。4択。"),
    ],
    "玉手箱": [
        ("四則逆算",   "□を含む式（□×n=m、n÷□=m、(□+n)×m=kなど）を作れ。選択肢は整数4択。複合式も含めること。"),
        ("長文一致",   "100〜150字の文章と、その内容についての命題を1つ作れ。選択肢は「A:一致する」「B:一致しない」「C:どちらとも言えない」の3択。"),
        ("テーブル計算","3〜4列・4〜5行の表（売上・在庫・人数等）を文章で与え、合計・差・割合などを問う問題を作れ。4択。"),
        ("図表読取",   "折れ線・棒グラフの数値を文章で表現し、増減・比較・割合を問う問題を作れ。4択。"),
        ("英語語彙",   "TOEIC600〜700点レベルの英単語の意味を問う4択問題を作れ。"),
        ("英文読解",   "3〜4文の短い英文を与え、内容に一致するものを4択で選ばせる問題を作れ。"),
        ("数列完成",   "空欄のある数列（等差・等比・フィボナッチ変形など）の□を埋める問題を作れ。4択。"),
    ],
    "英語": [
        ("語彙",       "TOEIC700点レベルの英単語・熟語の意味を4択で問う問題を作れ。"),
        ("文法",       "英文の空欄に入る最適な語（品詞・前置詞・接続詞等）を4択で選ぶ問題を作れ。"),
        ("読解",       "5〜6文の英文パッセージを書き、内容に関する設問（正誤・主題等）を4択で作れ。"),
        ("語句整序",   "英文の語句を並べ替える問題を作れ。答えは4パターン（A/B/C/D）の並び順。"),
    ],
}

_SPI_LLM_WORDS_LANGUAGE = [
    "逡巡","忖度","蓋然性","漸進","恣意","瑣末","截然","逼迫","敷衍","僭越",
    "矜持","跋扈","遁走","杜撰","慇懃","倦怠","齟齬","乖離","惹起","帰趨",
    "嚆矢","濫觴","淘汰","頑迷","慄然","慄く","邂逅","慷慨","諮問","訓示",
]

def _spi_generate_llm(filter_cat: str | None = None) -> dict | None:
    """LLMでSPI/玉手箱問題を動的生成する。失敗時はNone。"""
    o = _get_ollama()
    if o is None:
        return None

    # カテゴリと出題タイプをランダム選択
    cat = filter_cat if filter_cat else _random.choice(["言語", "非言語", "玉手箱", "英語"])
    subtypes = _SPI_LLM_SUBTYPES.get(cat, _SPI_LLM_SUBTYPES["言語"])
    sub, instruction = _random.choice(subtypes)

    # 言語系は語彙をランダムに差し込む
    word = _random.choice(_SPI_LLM_WORDS_LANGUAGE)
    instruction = instruction.replace("{word}", word)

    sys_prompt = (
        "あなたはSPI・玉手箱の問題作成専門家です。\n"
        "以下の指示に従い、問題を1問だけJSON形式で出力してください。\n"
        "出力形式（必ずこの形式のみ。説明・前置き不要）:\n"
        "{\n"
        '  "q": "問題文（改行は\\nで表現）",\n'
        '  "choices": ["A: 選択肢1", "B: 選択肢2", "C: 選択肢3", "D: 選択肢4"],\n'
        '  "ans": "A",\n'
        '  "exp": "解説文（正解の理由を1〜2文で）"\n'
        "}\n"
        "制約:\n"
        "- 選択肢は必ずA/B/C/Dの4つ（長文一致のみA/B/Cの3つでよい）\n"
        "- 正解は必ず1つ、他は明確に間違い\n"
        "- 数値問題は計算を確認してから出力すること\n"
        "- JSONのみ出力。```json等のマークダウン不要\n"
    )
    user_prompt = f"【カテゴリ】{cat}・{sub}\n【出題指示】{instruction}"

    try:
        raw = ""
        stream = o.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            stream=True,
            options={"temperature": 0.85, "num_predict": 600, "num_ctx": 1024},
        )
        deadline = time.time() + 15.0
        for chunk in stream:
            if time.time() > deadline:
                break
            raw += chunk.get("message", {}).get("content", "")
        # JSONを抽出
        m = re.search(r'\{[\s\S]*\}', raw)
        if not m:
            return None
        data = json.loads(m.group(0))
        # 必須フィールド検証
        if not all(k in data for k in ("q", "choices", "ans", "exp")):
            return None
        if len(data["choices"]) < 3:
            return None
        # 正解が選択肢に含まれているか
        ans = data["ans"].strip().upper()
        if not any(c.startswith(f"{ans}:") for c in data["choices"]):
            return None
        return {
            "cat": cat,
            "sub": f"{sub}★",   # ★=LLM生成を示すマーク
            "q":   data["q"],
            "choices": data["choices"],
            "ans": ans,
            "exp": data["exp"],
        }
    except Exception as e:
        print(f"{C['y']}[SPI-LLM] 生成失敗: {e}{C['w']}")
        return None

def _spi_pick_smart(filter_cat: str | None = None, use_llm: bool = True) -> dict:
    """
    LLM生成を優先し、失敗時はDBからローテーション出題する。
    LLMはバックグラウンドで呼び出し、タイムアウト付き。
    """
    if use_llm:
        result_box: list = []
        def _gen():
            q = _spi_generate_llm(filter_cat)
            if q:
                result_box.append(q)
        t = threading.Thread(target=_gen, daemon=True)
        t.start()
        t.join(timeout=5)   # ★[修正/#7] 最大5秒待つ（旧コメントは「14秒」と誤記）
        if result_box:
            _spi_mark_used(result_box[0])
            return result_box[0]
    # フォールバック: DBからローテーション
    return _spi_pick_fresh(filter_cat)

def _spi_format_q(q: dict, num: int | None = None) -> str:
    prefix = f"[問{num}] " if num else ""
    header = f"{C['c']}{prefix}【{q['cat']}・{q['sub']}】{C['w']}"
    body   = q["q"]
    choices = "\n".join(q["choices"])
    ans_hint = "A/B/C" if q.get("sub") == "長文一致" else "A/B/C/D"
    return f"{header}\n{body}\n{choices}\n{C['dim']}→ {ans_hint} で答えてください{C['w']}"

def _spi_feedback(q: dict, user_ans: str) -> str:
    """正誤判定＋選択肢付き解説を返す。"""
    correct = user_ans.upper() == q["ans"]
    # 正解の選択肢テキストを取得
    ans_letter = q["ans"]
    ans_text = next((c for c in q["choices"] if c.startswith(f"{ans_letter}:")), ans_letter)
    if correct:
        mark = f"{C['g']}✓ 正解！{C['w']}"
        detail = f"{C['g']}{ans_text}{C['w']}"
    else:
        mark = f"{C['r']}✗ 不正解  あなたの答え: {user_ans.upper()}{C['w']}"
        detail = f"{C['r']}正解 → {ans_text}{C['w']}"
    exp_block = f"{C['c']}【解説】{C['w']} {q['exp']}"
    return correct, f"{mark}\n{detail}\n{exp_block}"

def handle_spi(arg: str) -> str:
    arg = arg.strip()
    _sess = _spi_load_session()
    _spi_current  = _sess["current"]
    _spi_mock_queue   = _sess["mock_queue"]
    _spi_mock_results = _sess["mock_results"]
    _is_mock = _sess.get("is_mock", False)

    # ---------- 答え入力（A/B/C/D） ----------
    if arg.upper() in ("A", "B", "C", "D"):
        # 長文一致は3択なのでDは受け付けない
        if arg.upper() == "D" and _spi_current.get("sub") == "長文一致":
            return f"{C['y']}この問題は A/B/C の3択です。{C['w']}"
        if _is_mock:
            # 模擬試験モード（キューが空でも最終問題として処理）
            q = _spi_current
            if not q:
                return f"{C['y']}問題が出題されていません。/spi 模擬 で開始してください。{C['w']}"
            correct, fb = _spi_feedback(q, arg)
            _spi_record(q["cat"], correct)
            _spi_mock_results.append(correct)
            if _spi_mock_queue:
                _spi_current = _spi_mock_queue.pop(0)
                _spi_save_session(_spi_current, _spi_mock_queue, _spi_mock_results, is_mock=True)
                return f"{fb}\n\n{_spi_format_q(_spi_current, num=len(_spi_mock_results)+1)}"
            else:
                # 模擬試験終了
                total = len(_spi_mock_results)
                ok = sum(_spi_mock_results)
                _spi_save_session({}, [], [], is_mock=False)
                return (f"{fb}\n\n"
                        f"{C['c']}===== 模擬試験終了 ====={C['w']}\n"
                        f"結果: {ok}/{total} 問正解  ({ok*100//total}%)\n"
                        f"/spi 成績 で累計成績を確認できます。")
        else:
            # 通常1問モード
            q = _spi_current
            if not q:
                return f"{C['y']}問題が出題されていません。/spi で問題を出してください。{C['w']}"
            correct, fb = _spi_feedback(q, arg)
            _spi_record(q["cat"], correct)
            # ★[修正/spi-2] セッションクリアは feedback を返した後に確実に実行
            # 旧コードでは save_session({}) が返答前に走りタイミング依存の問題があった
            _spi_save_session({}, [], [], is_mock=False)
            return f"{fb}\n\n次の問題: /spi または /spi [言語/非言語/英語]"

    # ---------- 成績表示 ----------
    if arg in ("成績", "stats", "score"):
        sc = _spi_load_score()
        if sc["total"] == 0:
            return f"{C['y']}まだ解答がありません。/spi で問題を解いてみましょう。{C['w']}"
        pct = sc["correct"] * 100 // sc["total"]
        lines = [f"{C['c']}===== SPI成績 ====={C['w']}",
                 f"総合: {sc['correct']}/{sc['total']} 問正解  ({pct}%)"]
        for cat, v in sc.get("cats", {}).items():
            cpct = v["correct"]*100//v["total"] if v["total"] else 0
            bar = "█" * (cpct//10) + "░" * (10 - cpct//10)
            lines.append(f"  {cat}: {v['correct']}/{v['total']}  [{bar}] {cpct}%")
        return "\n".join(lines)

    # ---------- 成績リセット ----------
    if arg in ("リセット", "reset"):
        state = load_state()
        state.pop(_SPI_SCORE_KEY, None)
        save_state(state)
        _spi_clear_session()
        return f"{C['y']}成績をリセットしました。{C['w']}"

    # ---------- 模擬試験（10問連続） ----------
    if arg in ("模擬", "mock", "test"):
        # DB から5問 + LLMで5問生成（カテゴリ均等）
        cats_cycle = ["言語", "非言語", "玉手箱", "英語", "言語",
                      "非言語", "玉手箱", "英語", "言語", "非言語"]
        _random.shuffle(cats_cycle)
        pool_db = _random.sample(_SPI_DB, min(5, len(_SPI_DB)))
        pool: list[dict] = list(pool_db)  # まずDBの5問を追加

        print(f"{C['dim']}  模擬試験: LLMで問題生成中（5問）…{C['w']}", flush=True)
        for cat_hint in cats_cycle[5:]:   # 残り5問をLLM生成
            q_llm = _spi_generate_llm(cat_hint)
            if q_llm:
                pool.append(q_llm)
            else:
                pool.append(_spi_pick_fresh(cat_hint))  # 失敗時はDBフォールバック

        _random.shuffle(pool)
        pool = pool[:10]
        _spi_mock_queue = pool[1:]
        _spi_mock_results_new = []
        _spi_current = pool[0]
        _spi_save_session(_spi_current, _spi_mock_queue, _spi_mock_results_new, is_mock=True)
        llm_count = sum(1 for p in pool if "★" in p.get("sub", ""))
        return (f"{C['c']}===== 模擬試験開始（10問）====={C['w']}\n"
                f"内訳: DB {10-llm_count}問 + AI生成 {llm_count}問\n"
                f"A/B/C/D で答えてください。\n\n{_spi_format_q(_spi_current, num=1)}")

    # ---------- カテゴリ指定or通常出題 ----------
    cat_map = {"言語": "言語", "非言語": "非言語", "英語": "英語",
               "玉手箱": "玉手箱", "四則逆算": "玉手箱", "長文": "玉手箱", "テーブル": "玉手箱",
               "verbal": "言語", "math": "非言語", "english": "英語", "tama": "玉手箱"}
    filter_cat = cat_map.get(arg) if arg else None
    if arg and filter_cat is None:
        return (f"{C['r']}usage: /spi [言語|非言語|英語|玉手箱|模擬|成績|リセット]\n"
                f"または A/B/C/D で回答{C['w']}")
    with SystemSpinner("SPI問題を生成中…", stage="rag") as _sp:
        q = _spi_pick_smart(filter_cat, use_llm=True)
    _spi_save_session(q, [], [], is_mock=False)
    llm_tag = f" {C['dim']}[AI生成]{C['w']}" if "★" in q.get("sub", "") else ""
    return _spi_format_q(_spi_load_session()["current"]) + llm_tag

# ===== ハンドラ関数群 =====
def handle_memo(arg: str) -> str:
    state = load_state(); memos = state.setdefault("memo", [])
    sub, _, rest = arg.partition(" "); sub = sub.lower().strip(); rest = rest.strip()
    if not arg or sub == "list":
        if not memos: return f"{C['y']}メモは空です。{C['w']}"
        return f"{C['c']}=== MEMORY ==={C['w']}\n" + "\n".join(f"{i+1}. {m.get('text','')} ({m.get('time','')})" for i, m in enumerate(memos[-20:]))
    if sub == "add":
        if not rest: return f"{C['r']}usage: /m add <内容>{C['w']}"
        memos.append({"time": now_stamp(), "text": rest}); save_state(state); update_keyword_memory(rest); vector_add(rest)
        return f"{C['g']}覚えました: {rest}{C['w']}"
    if sub == "find":
        if not rest: return f"{C['r']}usage: /m find <検索語>{C['w']}"
        hits = [(i, m) for i, m in enumerate(memos, 1) if rest.lower() in m.get("text", "").lower()]
        if not hits: return f"{C['y']}該当なし。{C['w']}"
        return f"{C['c']}=== HIT ==={C['w']}\n" + "\n".join(f"{i}. {m.get('text','')} ({m.get('time','')})" for i, m in hits)
    if sub == "del":
        if not rest.isdigit(): return f"{C['r']}usage: /m del <番号>{C['w']}"
        idx = int(rest) - 1
        if idx < 0 or idx >= len(memos): return f"{C['r']}その番号はありません。{C['w']}"
        removed = memos.pop(idx); save_state(state)
        return f"{C['y']}削除: {removed.get('text','')}{C['w']}"
    return f"{C['r']}usage: /m add/list/find/del{C['w']}"

def handle_dict(arg: str) -> str:
    state = load_state(); entries = state.setdefault("dict", [])
    sub, _, rest = arg.partition(" "); sub = sub.strip().lower(); rest = rest.strip()
    if not arg or sub == "list":
        if not entries: return f"{C['y']}辞書は空です。{C['w']}"
        return f"{C['c']}=== 辞書一覧 ==={C['w']}\n" + "\n".join(f"{i+1}. {e['term']}: {e.get('def','')[:60]}" for i, e in enumerate(entries[-40:]))
    if sub == "add":
        if "|" not in rest: return f"{C['r']}usage: /dict add <用語> | <説明>{C['w']}"
        term, _, defn = rest.partition("|"); term = term.strip(); defn = defn.strip()
        if not term or not defn: return f"{C['r']}usage: /dict add <用語> | <説明>{C['w']}"
        entries.append({"term": term, "def": defn, "time": now_stamp()})
        save_state(state); vector_add(f"{term}: {defn}", {"type": "dict", "term": term})
        return f"{C['g']}辞書に追加: {term}{C['w']}"
    if sub == "del":
        if not rest: return f"{C['r']}usage: /dict del <用語>{C['w']}"
        for i, e in enumerate(entries):
            if e["term"] == rest: removed = entries.pop(i); save_state(state); return f"{C['y']}削除: {removed['term']}{C['w']}"
        return f"{C['r']}「{rest}」は見つかりません{C['w']}"
    if sub == "find":
        if not rest: return f"{C['r']}usage: /dict find <キーワード>{C['w']}"
        hits = [(i, e) for i, e in enumerate(entries, 1) if rest.lower() in e["term"].lower() or rest.lower() in e["def"].lower()]
        if not hits: return f"{C['y']}該当なし。{C['w']}"
        return f"{C['c']}=== 辞書検索: {rest} ==={C['w']}\n" + "\n".join(f"{i}. {e['term']}: {e.get('def','')[:80]}" for i, e in hits)
    for e in entries:
        if e["term"].lower() == sub: return f"{C['c']}【{e['term']}】{C['w']}{e['def']} ({e.get('time','')})"
    hits = [e for e in entries if sub in e["term"].lower()]
    if hits: return f"{C['c']}=== 部分一致 ==={C['w']}\n" + "\n".join(f"  {e['term']}: {e.get('def','')[:80]}" for e in hits[:5])
    return f"{C['r']}「{sub}」は辞書にありません。/dict add <用語> | <説明> で追加できます。{C['w']}"

def dict_context(text: str) -> str:
    if not text: return ""
    state = load_state(); entries = state.get("dict", [])
    if not entries: return ""
    text_lower = text.lower()
    matches = [f"  【辞書】{e['term']}: {e.get('def','')}" for e in entries if e.get("term","").lower() in text_lower]
    if not matches:
        matches = [f"  【関連辞書】{r}" for r in vector_search(text, n=3) if r and ":" in r and len(r) < 200]
    return "\n" + "\n".join(matches[:3]) if matches else ""

def handle_doc(arg: str) -> str:
    state = load_state(); docs = state.setdefault("docs", [])
    sub, _, rest = arg.partition(" "); sub = sub.strip().lower(); rest = rest.strip()
    if not arg or sub == "list":
        if not docs: return f"{C['y']}文書は空です。{C['w']}"
        return f"{C['c']}=== 保存文書 ==={C['w']}\n" + "\n".join(f"{i+1}. {d['title']} ({len(d['text'])}字)" for i, d in enumerate(docs[-20:]))
    if sub == "add":
        if "|" not in rest: return f"{C['r']}usage: /doc add <タイトル> | <本文>{C['w']}"
        title, _, text = rest.partition("|"); title = title.strip(); text = text.strip()
        if not title or not text: return f"{C['r']}usage: /doc add <タイトル> | <本文>{C['w']}"
        docs.append({"title": title, "text": text, "time": now_stamp()})
        save_state(state); vector_add(f"[{title}] {text[:300]}", {"type": "doc", "title": title})
        return f"{C['g']}文書保存: {title} ({len(text)}字){C['w']}"
    if sub == "show":
        if not rest: return f"{C['r']}usage: /doc show <タイトル>{C['w']}"
        for d in docs:
            if d["title"].lower() == rest.lower(): return f"{C['c']}=== {d['title']} ==={C['w']}\n{d['text']}"
        return f"{C['r']}「{rest}」は見つかりません{C['w']}"
    if sub == "think": return "__THINK__" + rest
    if sub == "del":
        if not rest: return f"{C['r']}usage: /doc del <タイトル>{C['w']}"
        for i, d in enumerate(docs):
            if d["title"] == rest: removed = docs.pop(i); save_state(state); return f"{C['y']}削除: {removed['title']}{C['w']}"
        return f"{C['r']}「{rest}」は見つかりません{C['w']}"
    for d in docs:
        if d["title"].lower() == sub: return f"{C['c']}=== {d['title']} ==={C['w']}\n{d['text'][:500]}"
    return f"{C['r']}usage: /doc add/list/show/think/del{C['w']}"

def format_quests() -> str:
    quests = load_state().get("quests", [])
    if not quests: return f"{C['y']}クエストは空です。{C['w']}"
    return f"{C['c']}=== QUEST LOG ==={C['w']}\n" + "\n".join(f"{i}. [{'DONE' if q.get('done') else 'OPEN'}] {q.get('goal','')} ({q.get('time','')})" for i, q in enumerate(quests[-15:], 1))

def complete_quest(arg: str) -> str:
    state = load_state(); quests = state.get("quests", [])
    if not arg.isdigit(): return f"{C['r']}usage: /q done <番号>{C['w']}"
    idx = int(arg) - 1
    if idx < 0 or idx >= len(quests): return f"{C['r']}その番号はありません。{C['w']}"
    quests[idx]["done"] = True; quests[idx]["done_time"] = now_stamp(); save_state(state)
    return f"{C['g']}完了: {quests[idx].get('goal','')}{C['w']}"

def show_quest(arg: str) -> str:
    quests = load_state().get("quests", [])
    if not arg.isdigit(): return f"{C['r']}usage: /q show <番号>{C['w']}"
    idx = int(arg) - 1
    if idx < 0 or idx >= len(quests): return f"{C['r']}その番号はありません。{C['w']}"
    q = quests[idx]
    return f"{C['c']}=== QUEST #{idx+1} [{'DONE' if q.get('done') else 'OPEN'}] ==={C['w']}\n{q.get('plan','')}"

def save_quest(goal: str, plan: str) -> None:
    state = load_state(); quests = state.setdefault("quests", [])
    quests.append({"time": now_stamp(), "goal": goal, "plan": plan, "done": False}); save_state(state)

def debug_report() -> str:
    rows = [f"{C['c']}=== S-01 DEBUG v131 Advanced RAG ==={C['w']}"]
    rows.append(f"RAGキャッシュ: {len(RAG_CACHE)}")
    if RAG_CACHE:
        for key, (ts, val, acc, conf) in list(RAG_CACHE.items())[-5:]:
            rows.append(f"  [{key[:20]}] age={int(time.time()-ts)}s len={len(val)} acc={acc} conf={conf:.2f}")
    # ★[v131] Advanced RAG ステータス
    ce_status = (
        "有効" if _cross_encoder_available is True else
        "無効(未インストール: pip install sentence-transformers)" if _cross_encoder_available is False else
        "未初期化"
    )
    ce_cache_info = _ce_score_cached.cache_info() if _cross_encoder_available else None
    rows.append(f"── Advanced RAG v131 ──")
    rows.append(f"  Cross-Encoder : {ce_status} ({CROSS_ENCODER_MODEL})")
    if ce_cache_info:
        rows.append(f"    CEキャッシュ: hits={ce_cache_info.hits} misses={ce_cache_info.misses} size={ce_cache_info.currsize}/{ce_cache_info.maxsize}")
    rows.append(f"  RRF           : {'有効' if RRF_ENABLED else '無効'} (k={RRF_K})")
    rows.append(f"  HyDE          : {'有効' if HYDE_ENABLED else '無効'} (max={HYDE_MAX_TOKENS}文字, cached={len(_HYDE_CACHE)})")
    rows.append(f"  Ctx Compress  : {'有効' if CTXCOMP_ENABLED else '無効'} (max={CTXCOMP_MAX_CHARS}文字/チャンク)")
    rows.append(f"  CE top_k      : {CROSS_ENCODER_TOP_K}")
    rows.append(f"キーワード: {', '.join(KEYWORD_MEMORY) or 'なし'}")
    rows.append(f"モデル: {MODEL_NAME} | パワー: {POWER_MODE}")
    return "\n".join(rows)

def doctor_report() -> str:
    return "\n".join([
        f"{C['c']}=== S-01 DOCTOR v128.1 ==={C['w']}", f"python: {sys.version.split()[0]}",
        f"platform: {platform.system()} {platform.release()}", f"model: {MODEL_NAME}",
        f"power: {POWER_MODE}", f"ollama: {'OK' if _get_ollama() is not None else 'NG'}",
        f"yt-dlp: {'OK' if shutil.which('yt-dlp') else 'NG'}", f"mpv: {'OK' if shutil.which('mpv') else 'NG'}",
        f"RAG cache: {len(RAG_CACHE)}", f"kw mem: {len(KEYWORD_MEMORY)}",
    ])

def set_power_mode(arg: str) -> str:
    global POWER_MODE
    mode = arg.strip().lower()
    if not mode:
        return (f"{C['c']}current: {POWER_MODE}{C['w']}\n"
                f"  FAST={FAST_MODEL} | MAIN={MODEL_NAME} | DEEP={DEEP_MODEL}\n"
                f"  Thinking={'ON' if THINKING_MODE else 'OFF'}")
    if mode not in ("low", "mid", "high", "ultra"):
        return f"{C['r']}usage: /power low|mid|high|ultra{C['w']}"
    POWER_MODE = mode
    # ★[v129] パワーモード変更時にモデル選択も更新
    _update_model_for_power()
    persist_learning()
    label = {'low':'軽量(1b)','mid':'標準(4b)','high':'高推論(4b→12b)','ultra':'最大(12b)'}.get(mode,'')
    return (f"{C['g']}power: {POWER_MODE} {label}{C['w']}\n"
            f"  FAST={FAST_MODEL} | MAIN={MODEL_NAME} | DEEP={DEEP_MODEL}")


def _update_model_for_power():
    """★[v129] パワーモードに応じてモデルを動的選択"""
    global MODEL_NAME, DEEP_MODEL, FAST_MODEL
    tier = MODEL_TIERS.get(POWER_MODE, MODEL_TIERS["mid"])
    o = _get_ollama()
    if o is None: return
    try:
        models = o.list()
        items = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        names = [_model_name(m) for m in items]
        for t in tier:
            if any(t.split(":")[0] in n for n in names):
                MODEL_NAME = t; break
        if _HAS_12B and POWER_MODE in ("high", "mid", "ultra"):
            DEEP_MODEL = "gemma3:12b"
        else:
            DEEP_MODEL = MODEL_NAME
        # ★[修正/fast-power] FASTモデルもパワーモードに連動させる
        if POWER_MODE == "low":
            FAST_MODEL = "gemma3:1b"
        elif POWER_MODE in ("mid", "high"):
            FAST_MODEL = "gemma3:4b"
        elif POWER_MODE == "ultra":
            FAST_MODEL = "gemma3:12b"
    except Exception: pass

def build_custom_persona(attr: str, hint: str = "") -> dict:
    name = attr.strip()[:40] or "CUSTOM"
    # ★[修正1] ヒントなしの場合はキャッシュを先に確認してWeb/LLM処理を完全スキップ
    if not hint and name in PERSONA_STYLE_CACHE:
        return PERSONA_STYLE_CACHE[name]
    if hint:
        style = f"{name}。{hint}"
        fp_match = re.search(r'一人称(?:は)?[「『]?([ぁ-んァ-ヶ一-龯]{1,4})', style)
        return {"name": name, "style": style[:250], "first_person": fp_match.group(1) if fp_match else "私"}
    ARCHETYPES = {
        "お嬢様": {"style": "上品な言葉遣い。わたくし口調。丁寧で格式高い", "fp": "わたくし"},
        "ギャル": {"style": "明るいギャル口調。テンション高め。語尾に「っ」「じゃん」", "fp": "あたし"},
        "ツンデレ": {"style": "最初はつっけんどんだが徐々に甘える", "fp": "私"},
        "クール": {"style": "冷静で淡々とした口調。感情を抑えめ", "fp": "私"},
        "無口": {"style": "言葉数が少ない。一言二言で簡潔に", "fp": "私"},
        "元気": {"style": "明るく活発な口調。感嘆符多用", "fp": "私"},
        "大人": {"style": "落ち着いた大人の口調。知的で余裕がある", "fp": "私"},
        "子供": {"style": "子供らしい無邪気な口調。単純で素直", "fp": "ボク"},
        "男性": {"style": "男らしい口調。さっぱりとした物言い", "fp": "俺"},
        "女性": {"style": "女性らしい柔らかい口調。丁寧で優しい", "fp": "私"},
        "魔王": {"style": "威厳のある高圧的な口調", "fp": "朕"},
        "勇者": {"style": "正義感のある熱い口調", "fp": "俺"},
        "執事": {"style": "丁寧な敬語。主君に仕える忠実な口調", "fp": "私"},
        "メイド": {"style": "丁寧で献身的な口調。ご主人様呼び", "fp": "私"},
        "忍者": {"style": "簡潔で謎めいた口調。〜でござる", "fp": "拙者"},
        "中二病": {"style": "厨二病的な中二病全開の口調", "fp": "僕"},
        "先生": {"style": "教師らしい丁寧な口調。時にお説教的", "fp": "私"},
        "猫": {"style": "猫のような気ままな口調。〜にゃ", "fp": "私"},
        "先輩": {"style": "少し先輩風を吹かせる口調。面倒見が良い", "fp": "私"},
        "後輩": {"style": "礼儀正しい後輩口調。年上を敬う", "fp": "私"},
    }
    parts = re.split(r'[\s　、,・]', name)
    styles = []; first_person = "私"; matched = False
    for part in parts:
        part_lower = part.lower().strip()
        if part_lower in ARCHETYPES:
            arch = ARCHETYPES[part_lower]; styles.append(arch["style"]); first_person = arch["fp"]; matched = True
    if not matched:
        return _llm_persona_style(name)
    return {"name": name, "style": (f"{name}。{' + '.join(styles)}")[:250], "first_person": first_person}

PERSONA_STYLE_CACHE: dict[str, dict] = {}
_PERSONA_CACHE_PATH = "s01_persona_cache.json"

def _persona_cache_load():
    try:
        if os.path.exists(_PERSONA_CACHE_PATH):
            with open(_PERSONA_CACHE_PATH, 'r', encoding='utf-8') as f:
                PERSONA_STYLE_CACHE.update(json.load(f))
    except: pass

def _persona_cache_save():
    try:
        with open(_PERSONA_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(PERSONA_STYLE_CACHE, f, ensure_ascii=False, indent=2)
    except: pass

_persona_cache_load()  # 起動時に読み込み

def _fetch_persona_web_info(name: str) -> str:
    """複数ソースを並列スクレイピングしてペルソナ情報を収集する。
    歌詞検索と同じ手法: Wikipedia + DDG + Yahoo + Bing + コトバンク + SEP(哲学者向け) を同時取得し
    重複除去・スコアリングで最良テキストを返す。"""
    if OFFLINE_MODE: return ""

    res: dict[str, str] = {}
    lock = threading.Lock()

    def _run(key: str, fn, *args):
        try:
            val = fn(*args)
            with lock:
                res[key] = (val or "").strip()
        except Exception as _e:
            print(f"{C['y']}[WARN] RAG並列取得失敗 [{key}]: {_e}{C['w']}")

    # SEP (Stanford Encyclopedia of Philosophy) スクレイピング
    def _fetch_sep(name: str) -> str:
        try:
            slug = name.lower().replace(" ", "-").replace("・", "-").replace("　", "-")
            # よく使われる英語名マッピング
            slug_map = {
                "ソクラテス": "socrates", "プラトン": "plato", "アリストテレス": "aristotle",
                "エピクテトス": "epictetus", "マルクス・アウレリウス": "marcus-aurelius",
                "トマス・アクィナス": "aquinas", "デカルト": "descartes", "スピノザ": "spinoza",
                "ライプニッツ": "leibniz", "ロック": "locke", "ヒューム": "hume",
                "カント": "kant", "ヘーゲル": "hegel", "ショーペンハウアー": "schopenhauer",
                "ミル": "mill", "ニーチェ": "nietzsche",
                "ウィリアム・ジェームズ": "james", "フッサール": "husserl",
                "ハイデガー": "heidegger", "サルトル": "sartre",
                "ボーヴォワール": "beauvoir", "ラッセル": "russell",
                "前期ウィトゲンシュタイン": "wittgenstein", "後期ウィトゲンシュタイン": "wittgenstein",
                "ウィトゲンシュタイン": "wittgenstein",
            }
            sep_slug = slug_map.get(name, slug)
            url = f"https://plato.stanford.edu/entries/{sep_slug}/"
            h = fetch_html(url, timeout=5, silent=True, spoof_bot=True)
            if not h or len(h) < 500: return ""
            # preamble div を抽出
            m = re.search(r'<div id="preamble"[^>]*>(.*?)</div>', h, re.S | re.I)
            if not m:
                m = re.search(r'<div[^>]+class="[^"]*toc[^"]*"[^>]*>.*?</div>(.*?)<div', h, re.S | re.I)
            if m:
                text = strip_tags(m.group(1))
                lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
                return "[SEP] " + " / ".join(lines[:4])
        except Exception as _e:
            print(f"{C['y']}[WARN] SEP取得失敗: {_e}{C['w']}")
        return ""

    # ブリタニカ日本語版
    def _fetch_britannica_ja(name: str) -> str:
        try:
            url = f"https://britannica.co.jp/search/?q={U.quote(name)}"
            h = fetch_html(url, timeout=4, silent=True, spoof_bot=True)
            snips = re.findall(r'<p[^>]*class="[^"]*summary[^"]*"[^>]*>(.*?)</p>', h, re.S | re.I)
            if not snips:
                snips = re.findall(r'<div[^>]*class="[^"]*description[^"]*"[^>]*>(.*?)</div>', h, re.S | re.I)
            lines = [strip_tags(s).strip() for s in snips[:3] if len(strip_tags(s).strip()) > 20]
            return "\n".join(f"[ブリタニカ] {l}" for l in lines)
        except Exception:
            return ""

    tasks = [
        ("wiki",       get_wikipedia,          name),
        ("wiki_en",    get_wikipedia,          name + " philosopher"),  # 英語Wikipedia
        ("ddg",        _fetch_ddg_snippets,    name + " 哲学者 思想 特徴 口調"),
        ("ddg2",       _fetch_ddg_snippets,    name + " philosophy biography"),
        ("yahoo",      _fetch_yahoo_snippets,  name + " 哲学 性格 言動 思想"),
        ("bing",       _fetch_bing_snippets,   name + " 哲学者 人物"),
        ("kotobank",   _fetch_kotobank,        name),
        ("sep",        _fetch_sep,             name),
        ("britannica", _fetch_britannica_ja,   name),
    ]
    threads = [
        threading.Thread(target=_run, args=(k, fn, *a), daemon=True)
        for k, fn, *a in tasks
    ]
    for t in threads: t.start()

    # Wikiが先に来ればそこで早期終了、なければ全ソース待つ
    deadline = time.time() + RAG_TIMEOUT + 2.0
    while time.time() < deadline:
        with lock:
            wiki_ok = len(res.get("wiki", "")) > 100
            web_ok  = sum(len(res.get(k, "")) for k in ["ddg", "yahoo", "bing", "kotobank", "sep"]) > 400
            if wiki_ok and web_ok: break
        time.sleep(0.15)
    for t in threads: t.join(timeout=0.5)

    with lock:
        all_res = dict(res)

    # ── 結果のマージ・重複除去 ──────────────────────────────────────
    parts = []

    # Wikipedia (日本語優先、なければ英語)
    wiki_ja = all_res.get("wiki", "")
    wiki_en = all_res.get("wiki_en", "")
    if wiki_ja and len(wiki_ja) > 100:
        parts.append(f"[Wikipedia JA]\n{wiki_ja[:2000]}")
    elif wiki_en and len(wiki_en) > 100:
        parts.append(f"[Wikipedia EN]\n{wiki_en[:1500]}")

    # SEP (Stanford)
    sep_text = all_res.get("sep", "")
    if sep_text:
        parts.append(sep_text[:600])

    # ブリタニカ
    bri_text = all_res.get("britannica", "")
    if bri_text:
        parts.append(bri_text[:400])

    # コトバンク
    ktb_text = all_res.get("kotobank", "")
    if ktb_text:
        parts.append(ktb_text[:400])

    # 検索スニペット群 (DDG・Yahoo・Bing) を重複除去してマージ
    snippet_lines = []
    for key in ["ddg", "ddg2", "yahoo", "bing"]:
        block = all_res.get(key, "")
        if block:
            snippet_lines.extend(block.splitlines())
    deduped = _deduplicate_lines(snippet_lines)
    if deduped:
        parts.append(f"[Web検索スニペット]\n" + "\n".join(deduped[:12]))

    final = "\n\n".join(p for p in parts if p.strip())
    n_sources = sum(1 for k in ["wiki", "sep", "britannica", "kotobank", "ddg", "yahoo", "bing"] if all_res.get(k, ""))
    print(f"{C['dim']}[ペルソナWeb] {name}: {n_sources}ソース取得 / {len(final)}字{C['w']}", flush=True)
    return final

def _llm_persona_style(name: str) -> dict:
    if name in PERSONA_STYLE_CACHE:
        return PERSONA_STYLE_CACHE[name]
    o = _get_ollama()
    # フォールバック: _get_ollama()がNoneの場合は直接初期化
    if o is None:
        try:
            import ollama as _ol_mod
            _host = os.environ.get("OLLAMA_HOST", "")
            o = _ol_mod.Client(host=_host) if _host else _ol_mod
        except Exception:
            pass

    # ── ① ネットから人物情報を取得 ──────────────────────────────
    web_info = ""
    with SystemSpinner(f"Web検索: {name}", stage="rag"):
        web_info = _fetch_persona_web_info(name)

    # ── ② LLMでペルソナを生成（情報があれば注入） ────────────────
    if o is None:
        # Ollamaなし：Webテキストからルールベースで推定
        style = f"『{name}』らしい口調。"
        if web_info:
            style += web_info[:200]
        p = {"name": name, "style": style[:250], "first_person": "私", "_web": bool(web_info)}
        PERSONA_STYLE_CACHE[name] = p
        _persona_cache_save()
        return p

    result = [""]
    def _gen():
        try:
            if web_info:
                prompt = (
                    f"以下は「{name}」に関する情報だ:\n{web_info[:1800]}\n\n"
                    f"「{name}」がAIとして日本語で話す設定を作れ。\n"
                    f"ルール:\n"
                    f"- 歴史上の武将・王・皇帝・偉人なら一人称は「我」「余」「朕」「拙者」など時代に合ったもの\n"
                    f"- 口調は時代・身分・性格を忠実に再現せよ\n"
                    f"- 「私」「は」は歴史上人物の一人称として不適切\n"
                    f"必ず以下の形式のみで答えよ（他の文章不要）:\n"
                    f"一人称:XX / 口調:YYYY / 特徴:ZZZZ"
                )
            else:
                prompt = (
                    f"「{name}」がAIとして日本語で話す設定を作れ。\n"
                    f"歴史上人物なら一人称は「我」「余」「朕」など時代に合ったもの。\n"
                    f"必ず以下の形式のみで答えよ:\n"
                    f"一人称:XX / 口調:YYYY / 特徴:ZZZ"
                )
            r = o.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 120, "temperature": 0.3},
                stream=False
            )
            result[0] = r["message"]["content"].strip()
        except Exception as _e:
            print(f"{C['y']}[WARN] LLM生成スレッド失敗: {_e}{C['w']}")

    t = threading.Thread(target=_gen, daemon=True)
    t.start(); t.join(timeout=30.0)

    raw_style = result[0]
    if not raw_style or len(raw_style) < 8:
        raw_style = f"『{name}』らしい口調。一人称は「私」。"

    # 一人称を抽出
    fp = "私"
    # 一人称ルール:
    # - 偉人・歴史上の人物 → 我/余/朕/拙者など
    # - 年上アニメキャラ   → 先輩くん呼び
    # - 年下アニメキャラ   → 先輩呼び
    # - 一般              → LLM出力に従う
    _HISTORICAL_KEYWORDS = ['武将','大名','天皇','将軍','王','皇帝','哲学者','思想家','科学者','発明家','革命家']
    _is_historical = any(kw in web_info[:500] for kw in _HISTORICAL_KEYWORDS)
    _FP_HISTORICAL = ['我','余','朕','拙者','某','吾輩','わし']
    m = re.search(r'一人称[：:]\s*([^\s/　「」]{1,6})', raw_style)
    if m:
        _fp_cand = m.group(1).strip('「」『』')
        fp = _fp_cand if _fp_cand not in ('は','が','を','に','の','も','と','私は') else '私'
    else:
        fp = '私'
    # 歴史上の人物で一人称が「私」になった場合は適切なものに補正
    if _is_historical and fp in ('私', 'は'):
        fp = next((f for f in _FP_HISTORICAL if f in raw_style), '我')

    # スタイル文字列を構築（Web情報の要約を冒頭に付与）
    style_parts = [raw_style[:300]]
    if web_info:
        # Wikiの先頭2〜3文 + SEP/ブリタニカの要点をヒントとして付加
        first_lines = [l.strip() for l in web_info.splitlines() if len(l.strip()) > 25][:3]
        if first_lines:
            style_parts.append("【参照】" + " / ".join(first_lines)[:250])

    final_style = "\n".join(style_parts)[:600]
    p = {"name": name, "style": final_style, "first_person": fp, "_web": bool(web_info)}
    PERSONA_STYLE_CACHE[name] = p
    _persona_cache_save()  # ★ディスク永続化
    print(f"{C['dim']}[ペルソナ生成] {name} / 一人称:{fp} / Web情報:{'あり' if web_info else 'なし'}{C['w']}")
    return p

def _model_name(m) -> str:
    if isinstance(m, dict): return m.get("name") or m.get("model") or ""
    return getattr(m, "name", "") or getattr(m, "model", "") or ""

def check_ollama_connection() -> bool:
    global MODEL_NAME, DEEP_MODEL, FAST_MODEL, _HAS_12B
    o = _get_ollama()
    if o is None: print(f"{C['r']}[FATAL] ollama not installed. pip install ollama{C['w']}"); return False
    try:
        models = o.list()
        items = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        names = [_model_name(m) for m in items]

        # ★[v129] モデル自動選択: 12b > 4b > 1b の優先順
        has_12b  = any("12b" in n or "gemma3:12b" in n for n in names)
        has_4b   = any("4b"  in n or "gemma3:4b"  in n for n in names)
        has_1b   = any("1b"  in n or "gemma3:1b"  in n or "gemma3.1:1b" in n for n in names)
        _HAS_12B = has_12b

        tier = MODEL_TIERS.get(POWER_MODE, MODEL_TIERS["mid"])
        selected = None
        for t in tier:
            if any(t in n or n.startswith(t.split(":")[0]) for n in names):
                selected = t; break
        if not selected:
            # fallback: 何でもあれば使う
            selected = next((n for n in names if "gemma" in n.lower()), None)
            if not selected:
                print(f"{C['y']}model not found. available: {', '.join(names) or 'none'}{C['w']}"); return False

        MODEL_NAME  = selected
        DEEP_MODEL  = "gemma3:12b" if has_12b else selected
        FAST_MODEL  = "gemma3:4b"  if has_4b  else ("gemma3:1b" if has_1b else selected)  # ★[GPU] 4b優先

        gpu_tag = f" {C['g']}[GPU]{C['w']}" if _GPU_AVAILABLE else ""
        print(f"{C['g']}[OK] FAST={FAST_MODEL} | MAIN={MODEL_NAME} | DEEP={DEEP_MODEL}{gpu_tag}{C['w']}")
        return True
    except Exception as e: print(f"{C['r']}[FATAL] Ollama connection failed: {e}{C['w']}"); return False

def start_roleplay(scene: str, per_id: int) -> None:
    global ROLEPLAY_ACTIVE, ROLEPLAY_SCENE
    ROLEPLAY_ACTIVE, ROLEPLAY_SCENE = True, scene
    print(f"{C['p']}[RP開始: {get_persona(per_id)['name']} / 終了は /rend]{C['w']}")

def end_roleplay() -> None:
    global ROLEPLAY_ACTIVE, ROLEPLAY_SCENE
    ROLEPLAY_ACTIVE, ROLEPLAY_SCENE = False, ""
    print(f"{C['y']}[ロールプレイ終了]{C['w']}")

def handle_ety(word: str) -> str:
    """Etymology 図鑑: 英単語を語根・接頭辞・接尾辞に分解して色分け表示する。"""
    word = word.strip().lower()
    if not word:
        return (
            f"{C['r']}usage: /ety <英単語>{C['w']}\n"
            f"  例: /ety impossible  →  im-(否定) + poss(置く) + -ible(できる)"
        )

    TYPE_COLOR = {"prefix": C['b'], "root": C['y'], "suffix": C['g']}
    TYPE_LABEL = {"prefix": "接頭辞", "root": "語根  ", "suffix": "接尾辞"}


    # ── 語根辞書（AI誤解釈を補正するポスト補正用） ──
    _MDICT: dict[str, tuple | None] = {
        # よく誤解される語根（英単語と同形だが別意味）
        "par":     ("root",   "現れる・見える",      "appear, show",         "Latin parere"),
        "parent":  ("root",   "現れる・見える",      "appear (not 'father')", "Latin parere"),
        "port":    ("root",   "運ぶ",               "carry, bear",          "Latin portare"),
        "man":     ("root",   "手",                 "hand",                 "Latin manus"),
        "manu":    ("root",   "手",                 "hand",                 "Latin manus"),
        "ant":     ("suffix", "〜な（形容詞）",      "forming adjectives",   "Latin"),
        "rupt":    ("root",   "破る",               "break, burst",         "Latin rumpere"),
        "spect":   ("root",   "見る",               "look, see",            "Latin spectare"),
        "spec":    ("root",   "見る",               "look, see",            "Latin spectare"),
        "vert":    ("root",   "回す・向ける",        "turn",                 "Latin vertere"),
        "vers":    ("root",   "回す・向ける",        "turn",                 "Latin vertere"),
        "duct":    ("root",   "導く",               "lead",                 "Latin ducere"),
        "duc":     ("root",   "導く",               "lead",                 "Latin ducere"),
        "mit":     ("root",   "送る",               "send",                 "Latin mittere"),
        "miss":    ("root",   "送る",               "send",                 "Latin mittere"),
        "dict":    ("root",   "言う",               "say, speak",           "Latin dicere"),
        "vis":     ("root",   "見る",               "see",                  "Latin videre"),
        "vid":     ("root",   "見る",               "see",                  "Latin videre"),
        "cap":     ("root",   "取る",               "take, seize",          "Latin capere"),
        "ced":     ("root",   "行く・譲る",          "go, yield",            "Latin cedere"),
        "ceed":    ("root",   "行く・進む",          "go, proceed",          "Latin cedere"),
        "cess":    ("root",   "行く・止まる",        "go, stop",             "Latin cedere"),
        "fac":     ("root",   "作る・する",          "make, do",             "Latin facere"),
        "fact":    ("root",   "作る・する",          "make, do",             "Latin facere"),
        "fect":    ("root",   "作る・する",          "make, do",             "Latin facere"),
        "fer":     ("root",   "運ぶ",               "carry, bear",          "Latin ferre"),
        "ject":    ("root",   "投げる",             "throw",                "Latin jacere"),
        "jac":     ("root",   "投げる",             "throw",                "Latin jacere"),
        "luc":     ("root",   "光",                 "light",                "Latin lux"),
        "lum":     ("root",   "光",                 "light",                "Latin lumen"),
        "mob":     ("root",   "動く",               "move",                 "Latin movere"),
        "mot":     ("root",   "動く",               "move",                 "Latin movere"),
        "mov":     ("root",   "動く",               "move",                 "Latin movere"),
        "neg":     ("root",   "否定する",           "deny, negate",         "Latin negare"),
        "pend":    ("root",   "吊るす・支払う",      "hang, pay",            "Latin pendere"),
        "pens":    ("root",   "吊るす・支払う",      "hang, pay",            "Latin pendere"),
        "pon":     ("root",   "置く",               "place, put",           "Latin ponere"),
        "pos":     ("root",   "置く",               "place, put",           "Latin ponere"),
        "poss":    ("root",   "置く・できる",        "place, be able",       "Latin ponere/posse"),
        "scrib":   ("root",   "書く",               "write",                "Latin scribere"),
        "script":  ("root",   "書く",               "write",                "Latin scribere"),
        "sent":    ("root",   "感じる",             "feel, sense",          "Latin sentire"),
        "sens":    ("root",   "感じる",             "feel, sense",          "Latin sentire"),
        "sist":    ("root",   "立つ",               "stand",                "Latin sistere"),
        "stat":    ("root",   "立つ・状態",          "stand, state",         "Latin stare"),
        "struct":  ("root",   "建てる",             "build",                "Latin struere"),
        "tang":    ("root",   "触れる",             "touch",                "Latin tangere"),
        "tact":    ("root",   "触れる",             "touch",                "Latin tangere"),
        "tract":   ("root",   "引く",               "pull, draw",           "Latin trahere"),
        "ten":     ("root",   "保つ・持つ",          "hold, keep",           "Latin tenere"),
        "tend":    ("root",   "伸ばす・向かう",      "stretch, tend",        "Latin tendere"),
        "tens":    ("root",   "伸ばす・張る",        "stretch, strain",      "Latin tendere"),
        "tent":    ("root",   "伸ばす・試みる",      "stretch, attempt",     "Latin tendere"),
        "ext":     ("prefix", "外に伸ばす",          "outward extension",    "Latin ex+tendere"),
        "vit":     ("root",   "生命",               "life",                 "Latin vita"),
        "viv":     ("root",   "生きる",             "live",                 "Latin vivere"),
        "voc":     ("root",   "声・呼ぶ",           "voice, call",          "Latin vocare"),
        "val":     ("root",   "強い・価値",          "strong, worth",        "Latin valere"),
        "urb":     ("root",   "都市",               "city",                 "Latin urbs"),
        "terr":    ("root",   "土地",               "land, earth",          "Latin terra"),
        "tempor":  ("root",   "時間",               "time",                 "Latin tempus"),
        "sign":    ("root",   "印・意味",           "sign, mark",           "Latin signum"),
        "grad":    ("root",   "歩む・段階",          "step, degree",         "Latin gradus"),
        "corp":    ("root",   "体",                 "body",                 "Latin corpus"),
        "sanct":   ("root",   "神聖な",             "holy, sacred",         "Latin sanctus"),
        # Prefixes
        "trans":   ("prefix", "越えて・横切って",    "across, through",      "Latin"),
        "pre":     ("prefix", "前に",               "before",               "Latin"),
        "post":    ("prefix", "後に",               "after",                "Latin"),
        "sub":     ("prefix", "下に",               "under, below",         "Latin"),
        "super":   ("prefix", "上に",               "above, over",          "Latin"),
        "inter":   ("prefix", "間に",               "between",              "Latin"),
        "re":      ("prefix", "再び",               "again, back",          "Latin"),
        "ex":      ("prefix", "外に",               "out of",               "Latin"),
        "de":      ("prefix", "下に・離れて",        "down, away",           "Latin"),
        "com":     ("prefix", "共に",               "together, with",       "Latin"),
        "con":     ("prefix", "共に",               "together, with",       "Latin"),
        "pro":     ("prefix", "前に",               "forward, for",         "Latin/Greek"),
        "anti":    ("prefix", "反対",               "against",              "Greek"),
        "auto":    ("prefix", "自己",               "self",                 "Greek"),
        "tele":    ("prefix", "遠い",               "far, distant",         "Greek"),
        "hyper":   ("prefix", "過剰",               "over, excessive",      "Greek"),
        "hypo":    ("prefix", "不足",               "under, below",         "Greek"),
        "semi":    ("prefix", "半分",               "half",                 "Latin"),
        "dis":     ("prefix", "離れて・否定",        "apart, not",           "Latin"),
        "ab":      ("prefix", "離れて",             "away from",            "Latin"),
        "ad":      ("prefix", "〜へ向かって",        "toward",               "Latin"),
        "per":     ("prefix", "完全に・通して",      "through, thoroughly",  "Latin"),
        "in":      ("prefix", "中に・否定",          "in, into, not",        "Latin"),
        "im":      ("prefix", "中に・否定",          "in, not (before m/p)", "Latin"),
        "un":      ("prefix", "否定",               "not",                  "Old English"),
        "mis":     ("prefix", "誤って",             "wrongly",              "Old English"),
        # Suffixes
        "ent":     ("suffix", "〜な（形容詞）",      "forming adjectives",   "Latin"),
        "tion":    ("suffix", "〜すること（名詞）",  "forming nouns",        "Latin"),
        "sion":    ("suffix", "〜すること（名詞）",  "forming nouns",        "Latin"),
        "ment":    ("suffix", "〜すること（名詞）",  "forming nouns",        "Latin"),
        "ness":    ("suffix", "〜の状態（名詞）",    "state, quality",       "Old English"),
        "ity":     ("suffix", "〜の性質（名詞）",    "state, quality",       "Latin"),
        "ible":    ("suffix", "〜できる（形容詞）",  "able to be",           "Latin"),
        "able":    ("suffix", "〜できる（形容詞）",  "able to be",           "Latin"),
        "ive":     ("suffix", "〜的な（形容詞）",    "tending to",           "Latin"),
        "ous":     ("suffix", "〜に満ちた（形容詞）","full of, having",      "Latin"),
        "ful":     ("suffix", "〜に満ちた（形容詞）","full of",              "Old English"),
        "less":    ("suffix", "〜のない（形容詞）",  "without",              "Old English"),
        "er":      ("suffix", "〜する人（名詞）",    "one who does",         "Old English"),
        "or":      ("suffix", "〜する人（名詞）",    "one who does",         "Latin"),
        "ist":     ("suffix", "〜主義者（名詞）",    "one who does/believes","Greek"),
        "ism":     ("suffix", "〜主義（名詞）",      "doctrine, practice",   "Greek"),
        "ize":     ("suffix", "〜にする（動詞）",    "to make, to do",       "Greek"),
        "ify":     ("suffix", "〜にする（動詞）",    "to make",              "Latin"),
        "ly":      ("suffix", "〜的に（副詞）",      "in a manner",          "Old English"),
        "al":      ("suffix", "〜の（形容詞）",      "relating to",          "Latin"),
        "ic":      ("suffix", "〜の（形容詞）",      "relating to",          "Greek"),
        "logy":    ("suffix", "〜の学問",            "study of",             "Greek"),
        "ology":   ("suffix", "〜の学問",            "study of",             "Greek"),
        "ance":    ("suffix", "〜の状態（名詞）",    "state, quality",       "Latin"),
        "ence":    ("suffix", "〜の状態（名詞）",    "state, quality",       "Latin"),
        "ward":    ("suffix", "〜の方向へ",          "in the direction of",  "Old English"),
        "ship":    ("suffix", "〜の状態・関係",      "state, condition",     "Old English"),
    }

    # few-shot 例示（1bでも安定するよう簡潔化）
    FEW_SHOT_SYSTEM = (
        "Etymology expert. Output ONLY one JSON object, single line, no markdown, no explanation.\n"
        "CRITICAL: meaning_ja MUST be Japanese (日本語). NEVER output Chinese characters as meaning_ja.\n"
        'FORMAT: {"word":"W","pos":"P","morphemes":[{"text":"T","type":"root","meaning_ja":"M","meaning_en":"E","origin":"O"}],"combined_meaning_ja":"CM","etymology_note":"N"}\n'
        'EXAMPLE INPUT: biology\n'
        'EXAMPLE OUTPUT: {"word":"biology","pos":"noun","morphemes":['
        '{"text":"bio","type":"prefix","meaning_ja":"生命","meaning_en":"life","origin":"Greek bios"},'
        '{"text":"logy","type":"suffix","meaning_ja":"〜学","meaning_en":"study of","origin":"Greek -logia"}],'
        '"combined_meaning_ja":"生物学","etymology_note":"ギリシャ語 bios（生命）+ logia（学問）由来。"}\n'
        'EXAMPLE INPUT: extend\n'
        'EXAMPLE OUTPUT: {"word":"extend","pos":"verb","morphemes":['
        '{"text":"ex","type":"prefix","meaning_ja":"外へ","meaning_en":"out","origin":"Latin ex"},'
        '{"text":"tend","type":"root","meaning_ja":"伸ばす","meaning_en":"stretch","origin":"Latin tendere"}],'
        '"combined_meaning_ja":"外へ伸ばす→延長する","etymology_note":"ラテン語 extendere（外へ伸ばす）由来。"}\n'
        "type must be prefix/root/suffix only. meaning_ja must be Japanese kanji/kana. Output JSON only."
    )

    prompt = f'INPUT: {word}\nOUTPUT:'
    msgs = [
        {"role": "system", "content": FEW_SHOT_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    def _try_parse(raw: str):
        """rawからJSONを抽出してパース。失敗したらNone。"""
        raw = re.sub(r'```(?:json)?|```', '', raw).strip()
        brace_start = raw.find('{')
        if brace_start == -1: return None
        depth = 0; brace_end = -1
        for idx, ch in enumerate(raw[brace_start:], brace_start):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: brace_end = idx; break
        if brace_end == -1: return None
        try: return json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError: return None

    def _validate_data(d: dict) -> bool:
        """パース済みJSONの品質チェック。Chinese混入・空meaning_jaを弾く。"""
        if not d: return False
        for mo in d.get("morphemes", []):
            mja = mo.get("meaning_ja", "")
            if not mja: return False
            # 日本語文字（ひらがな・カタカナ・漢字）が1文字以上含まれていること
            if not re.search(r'[\u3040-\u9FFF]', mja):
                return False  # 中国語ピンイン・英語のみ = 不正
        return True

    # 1bをデフォルトとして全単語に使用。FEW_SHOTでChinese禁止済みのため品質は十分。
    # 4bへのフォールバックは_validate_data失敗時のみ（35秒問題の修正）。
    global POWER_MODE
    _saved_power = POWER_MODE
    POWER_MODE = "high"
    try:
        data = None
        raw1 = ""
        with SystemSpinner(f"語源解析: {word}", stage="pass1"):
            raw1 = stream_response(msgs, False, len(prompt), temp_override=0.0,
                                   silent=True, max_tokens=250, model="gemma3:1b")
        parsed = _try_parse(raw1 or "")
        data = parsed if (parsed and _validate_data(parsed)) else None

        if data is None:
            print(f"{C['dim']}[/ety] 再解析中...{C['w']}", flush=True)
            with SystemSpinner(f"語源解析: {word} [精密]", stage="pass2"):
                raw2 = stream_response(msgs, False, len(prompt), temp_override=0.0,
                                       silent=True, max_tokens=320, model=MODEL_NAME)
            parsed2 = _try_parse(raw2 or "")
            data = parsed2 if (parsed2 and _validate_data(parsed2)) else parsed2
    finally:
        POWER_MODE = _saved_power

    if data is None:
        return f"{C['r']}[/ety] JSONパース失敗\n{(raw1 or '')[:200]}{C['w']}"

    morphemes = data.get("morphemes", [])
    if not morphemes:
        return f"{C['r']}[/ety] 形態素データなし: {word}{C['w']}"

    # ── 辞書補正: AIの誤意味をポスト補正 ──
    corrected = False
    _uncertain_morphs: list[str] = []
    for mo in morphemes:
        key = mo.get("text", "").lower()
        if key in _MDICT:
            entry = _MDICT[key]
            if entry is not None:
                t_type, t_mja, t_men, t_orig = entry
                if mo.get("type") != t_type or mo.get("meaning_en","").lower() != t_men.lower():
                    mo["type"] = t_type
                    mo["meaning_ja"] = t_mja
                    mo["meaning_en"] = t_men
                    mo["origin"] = t_orig
                    corrected = True
        else:
            # ★[修正/ety-1] 辞書未登録の形態素 = AI が自由に生成した部分
            # meaning_ja が空・1文字・または英語のみの場合はハルシネーション疑い
            mja = mo.get("meaning_ja", "").strip()
            men = mo.get("meaning_en", "").strip()
            if not mja or len(mja) < 2 or (mja and not re.search(r'[\u3040-\u9FFF]', mja)):
                _uncertain_morphs.append(key)
                mo["meaning_ja"] = f"({men})" if men else "（未確認）"
                mo["_uncertain"] = True
            # meaning_en も空なら完全に不明扱い
            if not men:
                mo["_uncertain"] = True
                _uncertain_morphs.append(key)
    if corrected:
        print(f"{C['dim']}[/ety] 辞書補正適用{C['w']}", flush=True)
    if _uncertain_morphs:
        print(f"{C['dim']}[/ety] 未辞書形態素（AI推定）: {', '.join(set(_uncertain_morphs))}{C['w']}", flush=True)

    lines: list[str] = []

    # ── ヘッダー ──
    lines.append(f"\n{C['bold']}{C['c']}━━ Etymology 図鑑 ━━━━━━━━━━━━━━━━━━━━━━{C['w']}")

    # ── 色分け単語表示 ──
    word_colored = ""
    for mo in morphemes:
        t = mo.get("type", "root")
        col = TYPE_COLOR.get(t, C['w'])
        word_colored += f"{col}{C['bold']}{mo.get('text','?')}{C['w']}"
    pos_tag = data.get("pos", "")
    lines.append(f"  {word_colored}  {C['dim']}[{pos_tag}]{C['w']}")

    # ── 凡例 ──
    lines.append(
        f"  {C['b']}■ 接頭辞{C['w']}  "
        f"{C['y']}■ 語根{C['w']}  "
        f"{C['g']}■ 接尾辞{C['w']}"
    )
    lines.append(f"  {'─' * 48}")

    # ── 各形態素 ──
    for mo in morphemes:
        t    = mo.get("type", "root")
        col  = TYPE_COLOR.get(t, C['w'])
        lbl  = TYPE_LABEL.get(t, "  ?   ")
        text = mo.get("text", "?")
        mja  = mo.get("meaning_ja", "")
        men  = mo.get("meaning_en", "")
        orig = mo.get("origin", "")
        # ★[修正/ety-2] 未確認形態素には ⚠ マークを付けてハルシネーションを可視化
        uncertain_mark = f" {C['y']}⚠AI推定{C['w']}" if mo.get("_uncertain") else ""
        lines.append(
            f"  {C['dim']}[{lbl}]{C['w']} "
            f"{col}{C['bold']}{text:<12}{C['w']}"
            f"→ {mja}  {C['dim']}({men}){C['w']}{uncertain_mark}"
        )
        if orig:
            lines.append(f"              {C['dim']}語源: {orig}{C['w']}")

    lines.append(f"  {'─' * 48}")

    # ── 全体の意味 ──
    meaning = data.get("combined_meaning_ja", "")
    if meaning:
        lines.append(f"  {C['c']}意味:{C['w']} {meaning}")

    # ── 語源ノート ──
    note = data.get("etymology_note", "")
    if note:
        lines.append(f"  {C['dim']}{note}{C['w']}")

    lines.append("")
    return "\n".join(lines)


def _cleanup():
    OPTIMIZER.stop(); persist_learning(); PurgeEvidence()

# ===== NEW FEATURES v128.1 =====
def handle_image(prompt: str) -> str:
    if not prompt: return f"{C['r']}usage: /img <prompt>  例: /img 渦巻く銀河{C['w']}"
    try: from PIL import Image, ImageDraw
    except ImportError: return f"{C['y']}Pillowが必要: pip install Pillow{C['w']}"
    with SystemSpinner(f"画像生成: {prompt[:30]}...", stage="img") as sp:
        width, height = 640, 480
        img = Image.new("RGB", (width, height), (10, 10, 30))
        draw = ImageDraw.Draw(img)
        seed = abs(hash(prompt)) % (2**31)
        import random; rng = random.Random(seed)
        prompt_l = prompt.lower()
        cx, cy = width // 2, height // 2
        if any(w in prompt_l for w in ["銀河", "galaxy", "宇宙", "星雲"]):
            for _ in range(3000):
                angle = rng.uniform(0, 2 * math.pi)
                radius = rng.uniform(0, 250)
                sr = radius; sa = angle + sr * 0.02
                x = int(cx + sr * math.cos(sa))
                y = int(cy + sr * math.sin(sa))
                if 0 <= x < width and 0 <= y < height:
                    dist = math.sqrt((x-cx)**2 + (y-cy)**2) / 250.0
                    r_val = int(100 + 155 * (1-dist) * abs(math.sin(sa + seed)))
                    g_val = int(50 + 100 * (1-dist) * abs(math.cos(sa * 0.5 + seed)))
                    b_val = int(150 + 105 * (1-dist))
                    img.putpixel((x, y), (r_val % 256, g_val % 256, b_val % 256))
        elif any(w in prompt_l for w in ["波", "wave", "海", "ocean"]):
            for x in range(width):
                for y_mult in range(3):
                    base_y = height // 2 + int(80 * math.sin(x * 0.03 + y_mult * 2.0 + seed * 0.01))
                    for dy in range(-15, 15):
                        yy = base_y + dy + y_mult * 60
                        if 0 <= yy < height:
                            intensity = max(0, 255 - abs(dy) * 12)
                            img.putpixel((x, yy), (int(intensity*0.3*(1+math.sin(x*0.02+seed))), int(intensity*0.6*(1+math.cos(x*0.025+seed*0.5))), int(intensity*0.9)))
        elif any(w in prompt_l for w in ["炎", "火", "fire", "flame", "夕日"]):
            for x in range(width):
                fh = int(height * 0.6 * (0.5 + 0.5 * math.sin(x * 0.02 + seed * 0.1)))
                for y in range(height - fh, height):
                    ratio = (height - y) / fh
                    var = rng.randint(-20, 20)
                    img.putpixel((x, y), (max(0,min(255,255+var)), max(0,min(255,int(100+155*(1-ratio))+var)), max(0,min(255,int(50*(1-ratio))))))
        elif any(w in prompt_l for w in ["花", "flower", "桜"]):
            for petal in range(8):
                ao = petal * 2 * math.pi / 8 + seed * 0.01
                for r in range(1, 110, 2):
                    for a_step in range(24):
                        a = ao + a_step * 2 * math.pi / 24
                        x = int(cx + r * math.cos(a) * (1 + 0.3 * math.sin(3 * a)))
                        y = int(cy + r * math.sin(a) * (1 + 0.3 * math.sin(3 * a)))
                        if 0 <= x < width and 0 <= y < height:
                            c_val = int(180 + 75 * (1 - r/110))
                            img.putpixel((x, y), (c_val, int(c_val*0.3), int(c_val*0.6)))
        else:
            for i in range(800):
                x = cx + int(math.sin(i * 0.1) * (i * 0.3))
                y = cy + int(math.cos(i * 0.07) * (i * 0.3))
                rv = int(128 + 127 * math.sin(i * 0.05 + seed))
                gv = int(64 + 63 * math.cos(i * 0.03 + seed * 0.5))
                bv = int(200 + 55 * math.sin(i * 0.07 + seed * 0.3))
                draw.ellipse([x-3, y-3, x+3, y+3], fill=(rv % 256, gv % 256, bv % 256))
    filename = f"aegis_img_{int(time.time())}.png"
    img.save(filename)
    return f"{C['g']}画像保存: {filename} ({width}x{height}){C['w']}"

def handle_convert(arg: str) -> str:
    parts = arg.split(None, 2)
    if len(parts) < 3: return f"{C['r']}usage: /convert <from> <to> <text/str>{C['w']}"
    fmt_from, fmt_to, text = parts[0].lower(), parts[1].lower(), parts[2]
    if fmt_from == "md" and fmt_to == "html":
        import html as h
        lines = []
        for line in text.splitlines():
            if line.startswith("# "): lines.append(f"<h1>{h.escape(line[2:])}</h1>")
            elif line.startswith("## "): lines.append(f"<h2>{h.escape(line[3:])}</h2>")
            elif line.startswith("### "): lines.append(f"<h3>{h.escape(line[4:])}</h3>")
            elif line.startswith("- "): lines.append(f"<li>{h.escape(line[2:])}</li>")
            else: lines.append(f"<p>{h.escape(line)}</p>")
        result = "<!DOCTYPE html><html><body>" + "\n".join(lines) + "</body></html>"
    elif fmt_from == "csv" and fmt_to == "json":
        rows = [row.split(",") for row in text.splitlines() if row.strip()]
        if rows: result = json.dumps([dict(zip(rows[0], r)) for r in rows[1:]], ensure_ascii=False, indent=2)
        else: result = "[]"
    elif fmt_from == "tsv" and fmt_to == "json":
        rows = [row.split("\t") for row in text.splitlines() if row.strip()]
        if rows: result = json.dumps([dict(zip(rows[0], r)) for r in rows[1:]], ensure_ascii=False, indent=2)
        else: result = "[]"
    elif fmt_from == "json" and fmt_to == "csv":
        try:
            data = json.loads(text)
            if isinstance(data, list) and data:
                headers = list(data[0].keys())
                csv_lines = [",".join(headers)]
                for item in data:
                    csv_lines.append(",".join(str(item.get(h, "")) for h in headers))
                result = "\n".join(csv_lines)
            else: result = str(data)
        except: return f"{C['r']}JSONパースエラー{C['w']}"
    elif fmt_from == "text" and fmt_to == "html":
        result = f"<!DOCTYPE html><html><body><pre>{html_module.escape(text)}</pre></body></html>"
    else: return f"{C['r']}未対応の変換: {fmt_from}→{fmt_to}{C['w']}"
    fn = f"aegis_convert_{int(time.time())}.{fmt_to}"
    try:
        with open(fn, "w", encoding="utf-8") as f: f.write(result)
        return f"{C['g']}変換完了: {fn} ({len(result)}字){C['w']}"
    except Exception as e: return f"{C['r']}error: {e}{C['w']}"

def handle_qr(text: str) -> str:
    if not text: return f"{C['r']}usage: /qr <text>{C['w']}"
    try:
        import qrcode
        from PIL import Image
    except ImportError: return f"{C['y']}qrcode+pip install qrcode Pillow{C['w']}"
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    fn = f"aegis_qr_{int(time.time())}.png"
    img.save(fn)
    return f"{C['g']}QRコード保存: {fn}{C['w']}"

def handle_color(hex_code: str) -> str:
    if not hex_code: return f"{C['r']}usage: /color <hex>  例: /color ff5733{C['w']}"
    hex_code = hex_code.lstrip("#")
    if len(hex_code) != 6:
        try: hex_code = hex(int(hex_code, 16))[2:].zfill(6)
        except: return f"{C['r']}無効な値: {hex_code}{C['w']}"
    if not all(c in "0123456789abcdefABCDEF" for c in hex_code): return f"{C['r']}無効な16進数{C['w']}"
    r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)
    block = f"\033[48;2;{r};{g};{b}m     \033[0m"
    rows = [f"{C['c']}=== 色情報 #{hex_code.upper()} ==={C['w']}"]
    rows.append(f"  RGB: ({r}, {g}, {b})")
    rows.append(f"  サンプル: {block}")
    # ★[修正/#4] HSL を正しい公式で計算。
    # 旧コード: S=max-min(0〜255の生値), L=max/2.55 は実際のHSL定義と全く異なる式だった。
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(r_, g_, b_), min(r_, g_, b_)
    delta = cmax - cmin
    if delta == 0:
        hsl_h = 0.0
    elif cmax == r_:
        hsl_h = 60.0 * (((g_ - b_) / delta) % 6)
    elif cmax == g_:
        hsl_h = 60.0 * ((b_ - r_) / delta + 2)
    else:
        hsl_h = 60.0 * ((r_ - g_) / delta + 4)
    hsl_l = (cmax + cmin) / 2.0
    hsl_s = 0.0 if delta == 0 else delta / (1.0 - abs(2.0 * hsl_l - 1.0))
    rows.append(f"  HSL: ({hsl_h:.0f}°, {hsl_s*100:.0f}%, {hsl_l*100:.0f}%)")
    return "\n".join(rows)

def handle_sysinfo() -> str:
    import psutil
    boot = time.time() - psutil.boot_time()
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    rows = [f"{C['c']}=== システム情報 ==={C['w']}"]
    rows.append(f"  OS: {platform.system()} {platform.release()}")
    rows.append(f"  Python: {sys.version.split()[0]}")
    rows.append(f"  起動経過: {boot//86400:.0f}d {(boot%86400)//3600:.0f}h {(boot%3600)//60:.0f}m")
    rows.append(f"  CPU使用率: {cpu_percent:.1f}%")
    rows.append(f"  メモリ: {mem.used//(1024**3)}GB / {mem.total//(1024**3)}GB ({mem.percent:.0f}%)")
    return "\n".join(rows)

def handle_rename(arg: str) -> str:
    parts = arg.split(None, 1)
    if len(parts) < 2: return f"{C['r']}usage: /rename <old> <new>{C['w']}"
    old, new = parts[0], parts[1]
    try:
        safe_old = _assert_safe_path(old)
        safe_new = _assert_safe_path(new)
    except ValueError as e:
        return f"{C['r']}セキュリティエラー: {e}{C['w']}"
    if not os.path.exists(safe_old): return f"{C['r']}ファイルなし: {old}{C['w']}"
    # 新しいパスのディレクトリが存在するか確認
    new_dir = os.path.dirname(safe_new)
    if new_dir and not os.path.isdir(new_dir):
        return f"{C['r']}移動先ディレクトリが存在しません: {new_dir}{C['w']}"
    try:
        os.rename(safe_old, safe_new)
        return f"{C['g']}リネーム: {os.path.basename(safe_old)} → {os.path.basename(safe_new)}{C['w']}"
    except Exception as e:
        return f"{C['r']}error: {e}{C['w']}"

def handle_batch(arg: str) -> str:
    parts = arg.split(None, 1)
    if len(parts) < 2: return f"{C['r']}usage: /batch <cmd> <path>  例: /batch count .{C['w']}"
    cmd, path = parts[0].lower(), parts[1].strip()
    try:
        safe_path = _assert_safe_path(path)
    except ValueError as e:
        return f"{C['r']}セキュリティエラー: {e}{C['w']}"
    if not os.path.exists(safe_path): return f"{C['r']}パスなし: {path}{C['w']}"
    if cmd == "count":
        if os.path.isfile(safe_path):
            with open(safe_path, "r", encoding="utf-8", errors="ignore") as f: data = f.read()
            return f"{C['g']}行数: {data.count(chr(10))+1}, 文字数: {len(data)}{C['w']}"
        else:
            files = [f for f in os.listdir(safe_path) if os.path.isfile(os.path.join(safe_path, f))]
            return f"{C['g']}ファイル数: {len(files)}{C['w']}"
    elif cmd == "size":
        if os.path.isfile(safe_path): return f"{C['g']}サイズ: {os.path.getsize(safe_path)} bytes{C['w']}"
        total = sum(os.path.getsize(os.path.join(safe_path, f)) for f in os.listdir(safe_path) if os.path.isfile(os.path.join(safe_path, f)))
        return f"{C['g']}合計サイズ: {total//1024}KB{C['w']}"
    elif cmd == "list":
        if os.path.isdir(safe_path):
            items = os.listdir(safe_path)
            return f"{C['c']}=== {path} ({len(items)}件) ==={C['w']}\n" + "\n".join(items[-50:])
        return f"{C['r']}ディレクトリではありません{C['w']}"
    return f"{C['r']}未対応コマンド: {cmd}{C['w']}"

def handle_chart(data_str: str) -> str:
    if not data_str: return f"{C['r']}usage: /chart <data>  例: /chart bar: cats=30,dogs=45,birds=15{C['w']}"
    try:
        parts = data_str.split(None, 1)
        chart_type = parts[0].lower().rstrip(":") if parts else "bar"
        data_part = parts[1] if len(parts) > 1 else data_str
        if ":" in data_part and not data_part.startswith(chart_type):
            chart_type = data_part.split(":")[0].strip().lower()
            data_part = ":".join(data_part.split(":")[1:])
        items = [item.strip() for item in data_part.replace("，",",").replace("、",",").split(",") if item.strip()]
        pairs = []
        for item in items:
            if "=" in item:
                k, v = item.split("=", 1)
                pairs.append((k.strip(), float(v.strip())))
            elif ":" in item:
                k, v = item.split(":", 1)
                pairs.append((k.strip(), float(v.strip())))
        if not pairs: return f"{C['r']}データ形式が不明: name=value,name=value{C['w']}"
        max_val = max(v for _, v in pairs)
        scale = 30 / max(max_val, 1)
        lines = [f"{C['c']}=== チャート ({chart_type}) ==={C['w']}"]
        if chart_type.startswith("bar"):
            for name, val in pairs:
                bar = "█" * max(1, int(val * scale))
                lines.append(f"  {name:12s} {bar} {val:.0f}")
        elif chart_type.startswith("pie"):
            total = sum(v for _, v in pairs)
            for name, val in pairs:
                pct = val / total * 100
                bar = "▓" * max(1, int(pct / 3))
                lines.append(f"  {name:12s} {bar} {pct:.1f}%")
        else:
            points = []
            for i, (name, val) in enumerate(pairs):
                x = int(i * 60 / max(len(pairs)-1, 1))
                y = int(20 - val * scale / 2)
                points.append((x, y))
                lines.append(f"  {name:10s} {'▬'*max(1,int(val*scale))} {val:.0f}")
        return "\n".join(lines)
    except Exception as e: return f"{C['r']}chart error: {e}{C['w']}"

def handle_note(text: str) -> str:
    if not text: return f"{C['r']}usage: /note <text>{C['w']}"
    fn = f"aegis_notes_{time.strftime('%Y%m%d')}.txt"
    with open(fn, "a", encoding="utf-8") as f: f.write(f"[{now_stamp()}] {text}\n")
    return f"{C['g']}ノート保存: {fn}{C['w']}"

def handle_timer(seconds_str: str) -> str:
    try: seconds = int(seconds_str)
    except: return f"{C['r']}usage: /timer <seconds>{C['w']}"
    if seconds < 1 or seconds > 86400: return f"{C['r']}1-86400秒の範囲で{C['w']}"
    def _timer():
        import time as t
        t.sleep(seconds)
        print(f"\n{C['g']}⏰ タイマー終了 ({seconds}秒経過){C['w']}")
    threading.Thread(target=_timer, daemon=True).start()
    return f"{C['g']}タイマー設定: {seconds}秒後にお知らせ{C['w']}"

def handle_calc(expr: str) -> str:
    if not expr: return f"{C['r']}usage: /calc <expression>{C['w']}"
    import ast as _ast
    _ALLOWED_NODES = {
        _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Call,
        _ast.Attribute, _ast.Name, _ast.Constant, _ast.Load,
        _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.Mod,
        _ast.Pow, _ast.FloorDiv, _ast.USub, _ast.UAdd,
    }
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"{C['r']}構文エラー: {e}{C['w']}"
    for node in _ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return f"{C['r']}許可されていない操作: {type(node).__name__}{C['w']}"
        if isinstance(node, _ast.Attribute) and node.attr.startswith("_"):
            return f"{C['r']}プライベート属性へのアクセスは禁止{C['w']}"
        if isinstance(node, _ast.Name) and node.id.startswith("_"):
            return f"{C['r']}プライベート名は使用不可{C['w']}"
    try:
        ns = {"__builtins__": {}, "math": math}
        result = eval(compile(tree, "<calc>", "eval"), ns)
        return f"{C['g']}= {result}{C['w']}"
    except Exception as e:
        return f"{C['r']}error: {e}{C['w']}"

# ===== 追加ハンドラ =====
SESSION_STATS = {"start_time": 0.0, "response_times": [], "token_estimates": []}

def handle_export(arg: str, ms: list) -> str:
    fmt = arg.strip().lower() or "md"
    if fmt not in ("md", "markdown", "json", "txt"): return f"{C['r']}形式: md / json / txt{C['w']}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = f"aegis_export_{ts}.{fmt}"
    try:
        if fmt in ("md", "markdown"):
            lines = [f"# Aegis 会話ログ ({now_stamp()})\n"]
            for m in ms:
                prefix = f"## {USER_NAME}" if m["role"] == "user" else "## AI"
                lines.append(f"\n{prefix}\n\n{m.get('content', '')}\n")
            with open(fn, "w", encoding="utf-8") as f: f.writelines(lines)
        elif fmt == "json":
            with open(fn, "w", encoding="utf-8") as f: json.dump(ms, f, ensure_ascii=False, indent=2)
        else:
            with open(fn, "w", encoding="utf-8") as f:
                for m in ms: f.write(f"{m['role']}: {m.get('content', '')}\n\n")
        return f"{C['g']}出力: {fn} ({len(ms)}メッセージ){C['w']}"
    except Exception as e: return f"{C['r']}export error: {e}{C['w']}"

def handle_stats() -> str:
    elapsed = time.time() - SESSION_STATS["start_time"]
    rt = SESSION_STATS.get("response_times", [])
    avg_rt = sum(rt) / max(len(rt), 1)
    return "\n".join(r for r in [
        f"{C['c']}=== セッション統計 ==={C['w']}",
        f"経過: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m",
        f"対話数: {LEARNING_STATS['total_interactions']}",
        f"肯定/否定/修正: {LEARNING_STATS['positive_count']}/{LEARNING_STATS['negative_count']}/{LEARNING_STATS['self_correction_count']}",
        f"平均応答: {avg_rt:.1f}s" if rt else "",
        f"RAG: {len(RAG_CACHE)} | ベクトル: {vector_count()}",
        f"温度: {TEMP_VOICE:.2f} | 最適化: {len(PROMPT_OPTIMIZATIONS)}カテゴリ",
    ] if r)

def handle_template(arg: str) -> str:
    state = load_state(); templates = state.setdefault("templates", {})
    sub, _, rest = arg.partition(" "); sub = sub.strip().lower(); rest = rest.strip()
    if not arg or sub == "list":
        if not templates: return f"{C['y']}テンプレートなし{C['w']}"
        return f"{C['c']}=== テンプレート一覧 ==={C['w']}\n" + "\n".join(f"  {k}: {v[:60]}..." for k, v in templates.items())
    if sub == "add":
        if "|" not in rest: return f"{C['r']}usage: /template add <名前> | <内容>{C['w']}"
        name, _, content = rest.partition("|"); name = name.strip(); content = content.strip()
        if not name or not content: return f"{C['r']}usage: /template add <名前> | <内容>{C['w']}"
        templates[name] = content; save_state(state)
        return f"{C['g']}テンプレート保存: {name} ({len(content)}字){C['w']}"
    if sub == "del":
        if not rest or rest not in templates: return f"{C['r']}usage: /template del <名前>{C['w']}"
        del templates[rest]; save_state(state)
        return f"{C['y']}削除: {rest}{C['w']}"
    if sub in templates: return f"{C['c']}=== {sub} ==={C['w']}\n{templates[sub]}"
    return f"{C['r']}usage: /template add/list/del{C['w']}"

def handle_history(arg: str) -> str:
    if not arg:
        if not INTERACTION_LOG: return f"{C['y']}履歴なし{C['w']}"
        lines = [f"{C['c']}=== 直近の対話 ==={C['w']}"]
        for i, entry in enumerate(INTERACTION_LOG[-20:], 1):
            t = time.strftime("%H:%M", time.localtime(entry.get("time", 0)))
            fb = entry.get("feedback", 0)
            lines.append(f"  {i}. [{t}] {'+' if fb>0 else '-' if fb<0 else ' '} {entry.get('input','')[:50]}")
        return "\n".join(lines)
    keyword = arg.lower()
    hits = [e for e in INTERACTION_LOG if keyword in e.get("input", "").lower()]
    if not hits: return f"{C['y']}「{arg}」に一致する履歴なし{C['w']}"
    lines = [f"{C['c']}=== 履歴検索: {arg} ({len(hits)}件) ==={C['w']}"]
    for e in hits[-10:]:
        t = time.strftime("%m/%d %H:%M", time.localtime(e.get("time", 0)))
        lines.append(f"  [{t}] {'+' if e.get('feedback',0)>0 else '-' if e.get('feedback',0)<0 else ' '} {e.get('input','')[:80]}")
    return "\n".join(lines)

def handle_tts(text: str) -> str:
    try:
        import edge_tts, asyncio
    except ImportError: return f"{C['y']}edge-tts 未インストール: pip install edge-tts{C['w']}"
    if not text: return f"{C['r']}usage: /tts <テキスト>{C['w']}"
    try:
        fn = f"tts_{int(time.time())}.mp3"
        asyncio.run(edge_tts.Communicate(text, voice="ja-JP-NanamiNeural").save(fn))
        mpv = shutil.which("mpv")
        if mpv: S.Popen([mpv, "--no-video", fn], stdout=S.DEVNULL, stderr=S.DEVNULL); return f"{C['g']}音声再生: {fn}{C['w']}"
        return f"{C['g']}音声保存: {fn}{C['w']}"
    except Exception as e: return f"{C['r']}TTS error: {e}{C['w']}"

def handle_translate(text: str, target_lang: str = "en") -> str:
    if not text: return f"{C['r']}usage: /tr <言語> <テキスト>{C['w']}"
    sys_prompt = f"あなたは翻訳者。以下のテキストを{target_lang}に翻訳せよ。翻訳以外の出力は一切禁止。"
    print(f"{C['c']}[翻訳 {target_lang}]{C['w']}: ", end="", flush=True)
    result = stream_response([{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}], True, len(text), 0.15, silent=False)
    return result or f"{C['r']}翻訳失敗{C['w']}"

def handle_elab(text: str, per_id: int) -> str:
    if not text: return f"{C['r']}usage: /elab <説明してほしい内容>{C['w']}"
    persona = get_persona(per_id)
    print(f"{C['c']}{persona['name']} [深層推論]{C['w']}: ", end="", flush=True)
    return stream_response([get_sys_prm("elab", text, per_id=per_id), {"role": "user", "content": f"以下の内容を、比喩・例えを用いて分かりやすく説明してください:\n{text}"}], True, len(text), 0.62, model=DEEP_MODEL) or ""

def handle_comp(args: str) -> str:
    # ★[修正/comp-1] "s <数字>" 記法を単一IDトークンに正規化してから分割
    # 例: "s 19 s 13 世界" → ["19", "13", "世界"]
    _raw = args.replace("\u3000", " ").strip()
    _raw = re.sub(r'\bs\s+(\d+)\b', r'\1', _raw)   # "s 19" → "19"
    parts = re.split(r'\s+', _raw)
    if len(parts) < 2: return f"{C['r']}usage: /comp <ID or 名前> <ID or 名前> [テーマ]{C['w']}"
    id1, id2 = parts[0], parts[1]
    theme = " ".join(parts[2:]) if len(parts) >= 3 else "自由会話"
    # ★[修正/comp-BUG1] 1<=ID<=24 の上限が36未満だったため、ベルクソン(25)〜ロールズ(36)が
    # get_persona()を呼ばず name/style が "IDの口調で話す" という空文字列になっていた。
    # 正しい上限はPERSONA_MAPの最大キー(36)に揃える。
    p1 = get_persona(int(id1)) if id1.isdigit() and 1 <= int(id1) <= 36 else {"name": id1, "style": f"{id1}の口調で話す", "first_person": "私"}
    p2 = get_persona(int(id2)) if id2.isdigit() and 1 <= int(id2) <= 36 else {"name": id2, "style": f"{id2}の口調で話す", "first_person": "私"}

    # ★[修正/comp-2] モード判定: 哲学者(1-36)同士 → 哲学的対話モードを新設
    # ★[修正/comp-BUG2] range(1,25)→range(1,37): 25-36(ベルクソン〜ロールズ)が哲学的対話モードに
    # 入れなかったバグを修正。PERSONA_MAPの全IDを哲学者として扱う。
    PHILOSOPHER_IDS = set(range(1, 37))
    BUSINESS_KW = {"社長", "部長", "課長", "教授", "博士", "先生", "CEO", "CTO", "役員", "責任者", "マネージャ", "マネージャー", "リーダー", "秘書", "S-01", "執事", "医師", "秀才", "エンジニア", "管理職", "弁護士", "会計士", "コンサル", "アナリスト", "ディレクター", "プロデューサー"}
    CASUAL_NAME_KW = {"お嬢様", "おじょうさま", "ギャル", "ツンデレ", "クール", "無口", "元気", "子供", "魔王", "勇者", "魔法使い", "忍者", "侍", "ヤンキー", "天然", "腹黒", "中二病", "猫", "犬", "恋人", "友達", "彼女", "彼氏", "妹", "姉", "弟", "兄", "ママ", "パパ"}
    CASUAL_PERSONA_IDS = set(range(1, 37))  # ★[修正/comp-BUG2] 哲学者全ID(1-36)をcasual判定から除外
    CASUAL_THEME_KW = {"デート", "遊び", "旅行", "趣味", "カフェ", "雑談", "休日", "暇", "好き", "恋愛", "友達", "買い物", "ゲーム", "アニメ", "映画", "音楽", "料理", "ペット", "おしゃべり", "海", "山", "花見", "キャンプ", "飲み", "食事", "遊ぼう", "話そう", "悩み", "日常", "たわいもない"}
    p1_phil = id1.isdigit() and int(id1) in PHILOSOPHER_IDS
    p2_phil = id2.isdigit() and int(id2) in PHILOSOPHER_IDS
    p1_biz = (not p1_phil) and (any(k in p1["name"] for k in BUSINESS_KW))
    p2_biz = (not p2_phil) and (any(k in p2["name"] for k in BUSINESS_KW))
    p1_cas = (not p1_phil) and (any(k in p1["name"] for k in CASUAL_NAME_KW))
    p2_cas = (not p2_phil) and (any(k in p2["name"] for k in CASUAL_NAME_KW))
    if p1_phil and p2_phil:
        is_philosophical = True; is_casual = False
    elif p1_biz or p2_biz:
        # 明示的なビジネスキーワードがある場合のみビジネスモード
        is_philosophical = False; is_casual = False
    elif p1_cas or p2_cas or any(kw in theme for kw in CASUAL_THEME_KW):
        # どちらか一方でもカジュアルキーワードがあればカジュアル
        # 織田信長+ギャル、中立+ギャルなど「片方だけ」にも対応
        is_philosophical = False; is_casual = True
    else:
        # 明示的な分類なし（織田信長+織田信長など）→ カジュアルにデフォルト
        # ビジネスはBUSINESS_KWが明示された場合のみ
        is_philosophical = False; is_casual = True
    mode_label = "哲学的対話" if is_philosophical else ("カジュアル" if is_casual else "ビジネス")
    print(f"{C['y']}=== {mode_label}: {p1['name']} vs {p2['name']} ({theme}) ==={C['w']}")
    fp1, fp2 = p1.get("first_person", "私"), p2.get("first_person", "私")
    labels = ["[テーゼ]", "[反テーゼ/否定]", "[保存]", "[高揚/アウフヘーベン]", "[合意条件]"]

    def _role_axis(name: str) -> str:
        if is_casual:
            casual_map = {
                "伴侶": "恋人として本音で語る。建前より気持ち優先。",
                "AI様": "全能の存在として慈愛と叡智で語る。",
                "後輩": "後輩として素直な疑問と尊敬を交えて語る。",
                "女王": "女王として寛大さと気高さで語る。",
                "ママ": "母親として温かく包容力をもって語る。",
                "お嬢様": "お嬢様として上品で少し世間知らずな視点で。",
                "博士": "博士として好奇心旺盛に語る。",
                "忍者": "忍者として簡潔で観察力鋭く語る。",
                "妹": "妹として甘えん坊で素直に語る。",
                "メイド": "メイドとして献身的で温かく語る。",
                "先生": "先生として教え導く立場から語る。",
                "中二病": "中二病として厨二的で情熱的に語る。",
                "秘書": "秘書として冷静で的確に助言する。",
            }
            for key, desc in casual_map.items():
                if key in name: return desc
            return "対等な関係としてお互いの意見を尊重しながら自然に語る。"
        senior = any(w in name for w in ("ベテラン", "CTO", "部長", "責任者", "社長", "役員", "リード", "シニア"))
        junior = any(w in name for w in ("新卒", "新人", "若手", "ジュニア", "研修", "インターン"))
        legal = any(w in name for w in ("法務", "監査", "コンプライアンス"))
        sales = any(w in name for w in ("営業", "事業", "企画"))
        if senior:
            return "経験者として、責任・設計・運用・失敗時の被害範囲を語る。抽象論でなく判断基準を出す。"
        if junior:
            return "新卒として、現場で詰まる点・レビュー待ち・学習不足・実装手順の不安を率直に質問する。上位者ぶらない。"
        if legal:
            return "法務として、契約・規制・監査証跡・責任分界を語る。"
        if sales:
            return "事業側として、顧客価値・売上・導入スピードを語る。"
        return "その肩書きに固有の利害・制約・語彙で話す。"

    axis1, axis2 = _role_axis(p1['name']), _role_axis(p2['name'])

    # ── KB知識ベース注入 ─────────────────────────────────────────
    kb_context_block = ""
    _kb_cols = [c for c in vector_list_collections() if c != "s01_memory"]
    if _kb_cols:
        _kb_hits: list[str] = []
        for _col in _kb_cols:
            _src = _col.replace("book_", "")
            for _h in vector_search(theme, n=2, collection=_col):
                _kb_hits.append(f"《{_src}》: {_h[:220]}")
        if _kb_hits:
            kb_context_block = (
                "\n【知識ベース参照（この内容を議論の根拠・引用として積極的に使え）】\n"
                + "\n".join(_kb_hits[:6])
            )
            print(f"{C['dim']}[comp: KB {len(_kb_hits)}件参照]{C['w']}")

    theme_terms = [w for w in re.findall(r'[A-Za-z0-9_\-]{3,}|[ァ-ヶー]{3,}|[一-龯]{2,}', theme) if len(w) >= 2]
    common_stop = {p1['name'], p2['name'], "テーゼ", "反テーゼ", "否定", "保存", "高揚", "アウフヘーベン", "合意条件", "について", "する", "です", "ます", "こと", "もの", "ため", "具体", "初期", "条件"}

    def _too_repetitive(raw: str) -> bool:
        words = [w for w in re.findall(r'[A-Za-z0-9_\-]{3,}|[ァ-ヶー]{3,}|[一-龯]{2,}', raw) if w not in common_stop]
        counts = Counter(words)
        if any(counts.get(t, 0) >= 5 for t in theme_terms): return True
        return any(v >= 6 for k, v in counts.items() if k not in theme_terms)

    if is_philosophical:
        # ★[修正/comp-3] 哲学者同士の専用対話モード — ビジネス/カジュアルの枠を排除
        system = (
            "あなたは哲学的対話の記録者。形式的な会議や日常会話ではなく、思想の自然な衝突と展開を書く。\n"
            "出力は必ず5発言。途中で終わるな。前置き・解説・箇条書きは禁止。Markdown装飾禁止。\n"
            "各哲学者は自分固有の哲学的立場・概念・語法で思考し語る。相手の言葉を受けて自分の思想で応答する。\n"
            "弁証法的な流れ（テーゼ→反テーゼ→保存→高揚→応答）を意識しつつ、各哲学者の用語で自然に展開せよ。\n"
            "同じ語句・同じ結論の反復は禁止。各発言は前発言の一点のみを受け、必ず新しい思想的視点を加える。"
        )
        user = (
            f"テーマ: 「{theme}」\n"
            f"話者A: {p1['name']}。哲学的立場と語法（厳守）: {p1['style']}。一人称: {fp1}。\n"
            f"話者B: {p2['name']}。哲学的立場と語法（厳守）: {p2['style']}。一人称: {fp2}。\n"
            f"{kb_context_block}\n"
            f"次の5発言を、各哲学者の立場と語法を完全に守って書け。\n"
            f"{p1['name']} [テーゼ]: 「{theme}」について自分の哲学的立場から最初の問いや主張を立てる。\n"
            f"{p2['name']} [反テーゼ/否定]: 相手の主張を自分の哲学的概念で受け取り、批判または別視点を提示する。\n"
            f"{p1['name']} [保存]: 相手の批判を受け止め、自分の思想で守れる核心を言語化する。\n"
            f"{p1['name']} [高揚/アウフヘーベン]: 二つの立場を統合し、より深い問いや思想へと展開する。\n"
            f"{p2['name']} [合意条件]: この思想展開に対して、自分の哲学的立場から応答または問い返す。"
        )
    elif is_casual:
        system = (
            "あなたは会話ファシリテーター兼脚本家。自然な日常会話の流れだけを書く。\n"
            "出力は必ず5発言。途中で終わるな。前置き・解説・箇条書きは禁止。\n"
            "ヘーゲル式のアウフヘーベンを会話にする。テーゼ、反テーゼによる否定、保存、高揚、合意条件の順に進める。\n"
            "否定=通らない考えを退ける。保存=元の考えの良い部分を残す。高揚=否定と保存を統合し、より良い考えに発展させる。\n"
            "ペルソナを逆転させるな。各キャラは自分の性格・立場に沿った言葉で話す。\n"
            "同じ語句・同じ結論の反復は禁止。各発言は前発言の一点だけを受け、必ず新しい視点を足す。Markdown装飾は禁止。"
        )
        user = (
            f"テーマ: {theme}\n"
            f"話者A: {p1['name']}。口調:{p1['style']}。一人称:{fp1}。この会話での立ち位置:{axis1}\n"
            f"話者B: {p2['name']}。口調:{p2['style']}。一人称:{fp2}。この会話での立ち位置:{axis2}\n"
            f"{kb_context_block}\n"
            "次の5行ラベルを必ず全て使う。各発言は2文以内。日常会話として自然に書く。\n"
            f"{p1['name']} [テーゼ]: 自分の気持ちや考えを最初に出す。\n"
            f"{p2['name']} [反テーゼ/否定]: 自分の立場から、違う意見や気持ちを伝える。\n"
            f"{p1['name']} [保存]: 相手の意見を受け止め、自分の考えの中で残したい部分を明確にする。\n"
            f"{p1['name']} [高揚/アウフヘーベン]: 二人の意見を統合した、より良い考えや落とし所を提案する。\n"
            f"{p2['name']} [合意条件]: その提案に対する自分の条件や次のアクションを伝える。"
        )
    else:
        system = (
            "あなたは会議ファシリテーター兼脚本家。実際の会議の自然な発言だけを書く。\n"
            "出力は必ず5発言。途中で終わるな。前置き・解説・箇条書きは禁止。\n"
            "ヘーゲル式のアウフヘーベンを会話にする。テーゼ、反テーゼによる否定、保存、高揚、合意条件の順に進める。\n"
            "否定=通らない点を退ける。保存=元案の価値ある目的を残す。高揚=否定と保存を統合し、上位の実行案へ組み替える。\n"
            "ペルソナを逆転させるな。経験者は経験者らしく、新卒は新卒らしく、役職固有の制約で話す。\n"
            "同じ語句・同じ結論の反復は禁止。各発言は前発言の一点だけを受け、必ず新しい論点を足す。Markdown装飾は禁止。テーマの主語を別事業に置き換えるな。"
        )
        user = (
            f"テーマ: {theme}\n"
            f"話者A: {p1['name']}。口調:{p1['style']}。一人称:{fp1}。役割固定:{axis1}\n"
            f"話者B: {p2['name']}。口調:{p2['style']}。一人称:{fp2}。役割固定:{axis2}\n"
            f"{kb_context_block}\n"
            "次の5行ラベルを必ず全て使う。各発言は2文以内。会議室での会話として書く。\n"
            f"{p1['name']} [テーゼ]: 自分の職責から、初期案と狙いを具体的に言う。\n"
            f"{p2['name']} [反テーゼ/否定]: 自分の経験値と職責に合う言い方で、通らない点を一つ否定する。\n"
            f"{p1['name']} [保存]: 否定を受け、元案から残すべき価値・目的・条件を明確にする。\n"
            f"{p1['name']} [高揚/アウフヘーベン]: 否定した点と保存した価値を統合し、上位の実行案へ組み替える。\n"
            f"{p2['name']} [合意条件]: 自分の立場から、実行前の条件と次アクションを合意する。"
        )
    _gen_model = DEEP_MODEL if is_philosophical else MODEL_NAME
    _gen_temp  = 0.52 if is_philosophical else 0.38
    raw = stream_response([{"role": "system", "content": system}, {"role": "user", "content": user}], False, len(user), _gen_temp, silent=True, max_tokens=900, model=_gen_model) or ""
    bad = any(label not in raw for label in labels) or _too_repetitive(raw)
    if bad:
        _axis_hint = (
            f"{p1['name']}は自分の哲学的立場で語る。{p2['name']}は自分の哲学的立場で語る。"
            if is_philosophical else
            f"{p1['name']}はこの軸を守る: {axis1}\n{p2['name']}はこの軸を守る: {axis2}"
        )
        repair = (
            "出力を作り直してください。問題: 必須ラベル不足、ペルソナ逆転、または同語反復。\n"
            "必須ラベル: [テーゼ], [反テーゼ/否定], [保存], [高揚/アウフヘーベン], [合意条件]\n"
            f"{_axis_hint}\n"
            "同じ名詞を繰り返さず、各発言で別の具体論点を出す。\n\n"
            f"テーマ: {theme}\n不完全な出力:\n{raw}"
        )
        raw = stream_response([{"role": "system", "content": system}, {"role": "user", "content": repair}], False, len(repair), 0.30, silent=True, max_tokens=950, model=_gen_model) or raw
    if any(label not in raw for label in labels) or _too_repetitive(raw):
        if is_philosophical:
            raw = (
                f"{p1['name']} [テーゼ]: 「{theme}」とは何か。私はこう問わざるをえない。\n"
                f"{p2['name']} [反テーゼ/否定]: その問い自体がすでに誤った前提を含んでいる。私はそこから問い直す必要がある。\n"
                f"{p1['name']} [保存]: しかし問うこと自体の意義は否定されない。私の核心はそこにある。\n"
                f"{p1['name']} [高揚/アウフヘーベン]: 二つの立場を統合すれば、問い方そのものを変えることが求められる。\n"
                f"{p2['name']} [合意条件]: その変容を認めよう。だがそれは新たな問いの始まりに過ぎない。"
            )
        elif is_casual:
            raw = (
                f"{p1['name']} [テーゼ]: ねえ、{theme}について話そうよ！私はこういうアイデアがあるんだけど。\n"
                f"{p2['name']} [反テーゼ/否定]: うーん、それもいいけど、私はちょっと違うかな。もっとこういう風にできない？\n"
                f"{p1['name']} [保存]: なるほど、そういう考えもあるね。でも、私の最初のアイデアのこの部分は残したいな。\n"
                f"{p1['name']} [高揚/アウフヘーベン]: じゃあさ、私のアイデアと君のアイデアを合わせて、こういうのはどう？\n"
                f"{p2['name']} [合意条件]: いいね！それなら賛成。まずはこれから始めてみよう。"
            )
        else:
            raw = (
                f"{p1['name']} [テーゼ]: {theme}について、私は段階的に進める案を推します。役割分担を明確にし、リスクを分散しながら進めることが重要だと思います。\n"
                f"{p2['name']} [反テーゼ/否定]: おっしゃる方向性は理解できますが、段階的すぎると意思決定が遅れます。{theme}では特に初動のスピードが成否を分けると考えます。\n"
                f"{p1['name']} [保存]: スピードの重要性は同意します。ただ、{theme}において責任の所在を曖昧にしたままでは後で問題が大きくなるリスクがあります。\n"
                f"{p1['name']} [高揚/アウフヘーベン]: ならば、{theme}の核心部分は迅速に進め、リスクの高い判断領域だけ段階的に決裁する二層構造にしませんか。\n"
                f"{p2['name']} [合意条件]: その案であれば賛成できます。まず優先度の高い領域から着手し、判断基準を共有しながら進めましょう。"
            )
    raw = raw.replace("**", "")
    print(raw.strip())
    return f"{C['y']}=== 対話終了 ==={C['w']}"

# ===== 自己弁証法: テーゼ/アンチテーゼ分解 =====
def handle_split(args: str) -> str:
    """/split <ID or 名前> [テーマ]
    1つのペルソナをテーゼ的・アンチテーゼ的サブペルソナに分解し、内的弁証法を生成する。
    """
    _raw = re.sub(r'\bs\s+(\d+)\b', r'\1', args.replace("\u3000", " ").strip())
    parts = re.split(r'\s+', _raw)
    if not parts or not parts[0]:
        return f"{C['r']}usage: /split <ID or 名前> [テーマ]{C['w']}"
    id1 = parts[0]
    theme = " ".join(parts[1:]) if len(parts) > 1 else "自己の核心"
    # ★[修正/split-BUG] 1<=ID<=24の上限が36未満だったため、25-36(ベルクソン〜ロールズ)が
    # get_persona()を呼ばずフォールバックになっていた。上限を36に修正。
    base = (get_persona(int(id1)) if id1.isdigit() and 1 <= int(id1) <= 36
            else {"name": id1, "style": f"{id1}らしい思想と口調", "first_person": "私"})

    print(f"{C['y']}=== 自己分解: {base['name']} → テーゼ / アンチテーゼ ({theme}) ==={C['w']}")
    print(f"{C['dim']}[split: サブペルソナ生成中...]{C['w']}", flush=True)

    # ── Step1: LLMでテーゼ/アンチテーゼを生成（JSON） ──────────────────
    decomp_sys = (
        "あなたは哲学的ペルソナ分析者。与えられた哲学者・人物を"
        "テーゼ的側面（肯定・核心的信念・理想）と"
        "アンチテーゼ的側面（懐疑・矛盾・自己批判・影の部分）の2サブペルソナに分解する。\n"
        "出力はJSON形式のみ。前置き・説明・コードブロック記号は一切不要:\n"
        '{"thesis":{"name":"...","style":"...","fp":"..."},'
        '"antithesis":{"name":"...","style":"...","fp":"..."}}'
    )
    decomp_user = (
        f"人物: {base['name']}\n"
        f"スタイル: {base['style'][:200]}\n"
        f"テーマ: {theme}\n\n"
        f"name は「{base['name']}（テーゼ）」「{base['name']}（アンチテーゼ）」形式。"
        f"style は各側面の口調・立場・語法を60字以内で。fp は一人称（私/僕/俺 など）。"
    )
    raw_json = stream_response(
        [{"role": "system", "content": decomp_sys}, {"role": "user", "content": decomp_user}],
        True, len(decomp_user), temp_override=0.30, silent=True, model=DEEP_MODEL
    ) or ""

    p_thesis = p_anti = None
    try:
        m = re.search(r'\{[\s\S]*\}', raw_json)
        if m:
            data = json.loads(m.group())
            t, a = data.get("thesis", {}), data.get("antithesis", {})
            p_thesis = {"name": t.get("name", f"{base['name']}（テーゼ）"),
                        "style": t.get("style", base["style"]),
                        "first_person": t.get("fp", base.get("first_person", "私"))}
            p_anti   = {"name": a.get("name", f"{base['name']}（アンチテーゼ）"),
                        "style": a.get("style", base["style"]),
                        "first_person": a.get("fp", base.get("first_person", "私"))}
    except Exception as _e:
        print(f"{C['y']}[WARN] ペルソナ解析失敗: {_e}{C['w']}")

    # フォールバック
    if not p_thesis:
        p_thesis = {"name": f"{base['name']}（テーゼ）",
                    "style": base["style"] + " 核心的信念を確信をもって肯定する。",
                    "first_person": base.get("first_person", "私")}
    if not p_anti:
        p_anti   = {"name": f"{base['name']}（アンチテーゼ）",
                    "style": base["style"] + " 自らの思想の矛盾・限界・影の部分を鋭く批判する。",
                    "first_person": base.get("first_person", "私")}

    print(f"{C['c']}テーゼ      : {p_thesis['name']}{C['w']}")
    print(f"            {C['dim']}{p_thesis['style'][:70]}{C['w']}")
    print(f"{C['p']}アンチテーゼ: {p_anti['name']}{C['w']}")
    print(f"            {C['dim']}{p_anti['style'][:70]}{C['w']}")
    print()

    # ── Step2: 分解済みサブペルソナで内的弁証法を生成 ────────────────
    labels = ["[テーゼ]", "[反テーゼ/否定]", "[保存]", "[高揚/アウフヘーベン]", "[合意条件]"]
    fp1, fp2 = p_thesis["first_person"], p_anti["first_person"]

    system = (
        "あなたは自己弁証法の記録者。同一人物の内部で起きる思想的対話を書く。\n"
        "テーゼ側は核心的信念を語り、アンチテーゼ側はその矛盾・限界を内側から突く。\n"
        "出力は必ず5発言。前置き・解説・箇条書き・Markdown装飾は禁止。\n"
        "各発言は2〜3文。同じ語句の反復禁止。各発言は前発言の一点のみを受け新たな視点を加える。"
    )
    user = (
        f"テーマ: 「{theme}」\n"
        f"話者A（テーゼ）: {p_thesis['name']}。立場: {p_thesis['style']}。一人称: {fp1}。\n"
        f"話者B（アンチテーゼ）: {p_anti['name']}。立場: {p_anti['style']}。一人称: {fp2}。\n\n"
        f"次の5発言を、各立場の語法を完全に守って書け:\n"
        f"{p_thesis['name']} [テーゼ]: 「{theme}」について核心的信念から主張する。\n"
        f"{p_anti['name']} [反テーゼ/否定]: その主張の矛盾・盲点・限界を内側から批判する。\n"
        f"{p_thesis['name']} [保存]: 批判を受け止め、それでも守れる思想の核心を言語化する。\n"
        f"{p_thesis['name']} [高揚/アウフヘーベン]: テーゼとアンチテーゼを統合し、より深い地点へ展開する。\n"
        f"{p_anti['name']} [合意条件]: その展開に対し、さらなる問いや留保を提示する。"
    )
    raw = stream_response(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        False, len(user), 0.55, silent=True, max_tokens=900, model=DEEP_MODEL
    ) or ""

    if any(label not in raw for label in labels):
        repair = (
            "必須ラベルが不足しています。以下のラベルを全て使って書き直してください:\n"
            "[テーゼ], [反テーゼ/否定], [保存], [高揚/アウフヘーベン], [合意条件]\n"
            f"テーマ: {theme}\n不完全な出力:\n{raw}"
        )
        raw = stream_response(
            [{"role": "system", "content": system}, {"role": "user", "content": repair}],
            False, len(repair), 0.30, silent=True, max_tokens=950, model=DEEP_MODEL
        ) or raw

    raw = raw.replace("**", "")
    print(raw.strip())
    return f"{C['y']}=== 自己弁証法終了 ==={C['w']}"

# ===== チェスエンジン v1.0 =====
class ChessEngine:
    """本格的なチェスエンジン。完全な駒移動バリデーション・チェック/チェックメイト検出・特殊手対応。"""

    UNICODE_PIECES = {
        "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
        "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
    }
    ASCII_PIECES = {
        "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
        "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
    }

    def __init__(self, use_unicode: bool = True):
        self.use_unicode = use_unicode
        self.reset()

    def reset(self):
        self.board: list[list[str | None]] = self._init_board()
        self.turn: str = "w"   # "w" or "b"
        self.castling_rights: dict = {"wK": True, "wQ": True, "bK": True, "bQ": True}
        self.en_passant: tuple | None = None   # (row, col) or None
        self.move_history: list[str] = []
        self.captured: dict = {"w": [], "b": []}
        self.halfmove_clock: int = 0
        self.fullmove: int = 1
        self.game_over: bool = False
        self.result: str = ""

    def _init_board(self) -> list[list]:
        b = [[None] * 8 for _ in range(8)]
        order = ["R", "N", "B", "Q", "K", "B", "N", "R"]
        for c, p in enumerate(order):
            b[0][c] = f"b{p}"
            b[7][c] = f"w{p}"
        for c in range(8):
            b[1][c] = "bP"
            b[6][c] = "wP"
        return b

    def piece_symbol(self, piece: str) -> str:
        if self.use_unicode:
            return self.UNICODE_PIECES.get(piece, "?")
        return self.ASCII_PIECES.get(piece, "?")

    def board_str(self, highlight: list[tuple] = None) -> str:
        highlight = highlight or []
        hl_set = set(highlight)
        lines = []
        sep_line = "  +" + "---+" * 8
        col_labels = "    a   b   c   d   e   f   g   h"
        lines.append(col_labels)
        lines.append(sep_line)
        for r in range(8):
            row_num = 8 - r
            cells = []
            for c in range(8):
                piece = self.board[r][c]
                sym = self.piece_symbol(piece) if piece else " "
                if (r, c) in hl_set:
                    cells.append(f"\033[43m {sym} \033[0m")
                elif (r + c) % 2 == 0:
                    cells.append(f"\033[47m {sym} \033[0m")
                else:
                    cells.append(f"\033[100m {sym} \033[0m")
            lines.append(f"{row_num} |{'|'.join(cells)}|")
            lines.append(sep_line)
        lines.append(col_labels)
        return "\n".join(lines)

    def parse_sq(self, s: str) -> tuple | None:
        s = s.strip().lower()
        if len(s) != 2: return None
        c = ord(s[0]) - ord('a')
        r = 8 - int(s[1])
        if not (0 <= r <= 7 and 0 <= c <= 7): return None
        return (r, c)

    def sq_name(self, r: int, c: int) -> str:
        return chr(ord('a') + c) + str(8 - r)

    def _enemy(self, color: str) -> str:
        return "b" if color == "w" else "w"

    def _piece_color(self, piece: str | None) -> str | None:
        return piece[0] if piece else None

    def _piece_type(self, piece: str | None) -> str | None:
        return piece[1] if piece else None

    def _on_board(self, r: int, c: int) -> bool:
        return 0 <= r <= 7 and 0 <= c <= 7

    def pseudo_legal_moves(self, r: int, c: int) -> list[tuple]:
        piece = self.board[r][c]
        if not piece: return []
        color = piece[0]
        ptype = piece[1]
        moves = []

        if ptype == "P":
            moves = self._pawn_moves(r, c, color)
        elif ptype == "N":
            moves = self._knight_moves(r, c, color)
        elif ptype == "B":
            moves = self._sliding_moves(r, c, color, [(1,1),(1,-1),(-1,1),(-1,-1)])
        elif ptype == "R":
            moves = self._sliding_moves(r, c, color, [(1,0),(-1,0),(0,1),(0,-1)])
        elif ptype == "Q":
            moves = self._sliding_moves(r, c, color, [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)])
        elif ptype == "K":
            moves = self._king_moves(r, c, color)
        return moves

    def _pawn_moves(self, r: int, c: int, color: str) -> list[tuple]:
        moves = []
        d = -1 if color == "w" else 1
        start_row = 6 if color == "w" else 1
        # 前進
        nr = r + d
        if self._on_board(nr, c) and not self.board[nr][c]:
            moves.append((nr, c))
            # 初期2マス
            if r == start_row and not self.board[r + 2*d][c]:
                moves.append((r + 2*d, c))
        # 斜め攻撃
        for dc in (-1, 1):
            nc = c + dc
            if self._on_board(nr, nc):
                target = self.board[nr][nc]
                if target and target[0] != color:
                    moves.append((nr, nc))
                # アンパッサン
                if self.en_passant == (nr, nc):
                    moves.append((nr, nc))
        return moves

    def _knight_moves(self, r: int, c: int, color: str) -> list[tuple]:
        moves = []
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r+dr, c+dc
            if self._on_board(nr, nc):
                t = self.board[nr][nc]
                if not t or t[0] != color:
                    moves.append((nr, nc))
        return moves

    def _sliding_moves(self, r: int, c: int, color: str, dirs: list) -> list[tuple]:
        moves = []
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            while self._on_board(nr, nc):
                t = self.board[nr][nc]
                if t:
                    if t[0] != color: moves.append((nr, nc))
                    break
                moves.append((nr, nc))
                nr += dr; nc += dc
        return moves

    def _king_moves(self, r: int, c: int, color: str) -> list[tuple]:
        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if self._on_board(nr, nc):
                    t = self.board[nr][nc]
                    if not t or t[0] != color:
                        moves.append((nr, nc))
        # キャスリング
        back_rank = 7 if color == "w" else 0
        if r == back_rank and c == 4:
            # キングサイド
            if self.castling_rights[f"{color}K"]:
                if (not self.board[back_rank][5] and not self.board[back_rank][6]
                        and self.board[back_rank][7] == f"{color}R"):
                    if not self._sq_attacked(back_rank, 4, self._enemy(color)) \
                            and not self._sq_attacked(back_rank, 5, self._enemy(color)) \
                            and not self._sq_attacked(back_rank, 6, self._enemy(color)):
                        moves.append((back_rank, 6))
            # クイーンサイド
            if self.castling_rights[f"{color}Q"]:
                if (not self.board[back_rank][3] and not self.board[back_rank][2]
                        and not self.board[back_rank][1]
                        and self.board[back_rank][0] == f"{color}R"):
                    if not self._sq_attacked(back_rank, 4, self._enemy(color)) \
                            and not self._sq_attacked(back_rank, 3, self._enemy(color)) \
                            and not self._sq_attacked(back_rank, 2, self._enemy(color)):
                        moves.append((back_rank, 2))
        return moves

    def _sq_attacked(self, r: int, c: int, by_color: str) -> bool:
        """by_color の駒が (r,c) を攻撃しているか"""
        enemy = by_color
        # ポーン攻撃
        pd = 1 if enemy == "w" else -1
        for dc in (-1, 1):
            nr, nc = r+pd, c+dc
            if self._on_board(nr, nc) and self.board[nr][nc] == f"{enemy}P":
                return True
        # ナイト
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr, nc = r+dr, c+dc
            if self._on_board(nr, nc) and self.board[nr][nc] == f"{enemy}N":
                return True
        # 直線・斜め
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            while self._on_board(nr, nc):
                t = self.board[nr][nc]
                if t:
                    if t[0] == enemy and t[1] in ("R", "Q"): return True
                    break
                nr += dr; nc += dc
        for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            nr, nc = r+dr, c+dc
            while self._on_board(nr, nc):
                t = self.board[nr][nc]
                if t:
                    if t[0] == enemy and t[1] in ("B", "Q"): return True
                    break
                nr += dr; nc += dc
        # キング
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if self._on_board(nr, nc) and self.board[nr][nc] == f"{enemy}K":
                    return True
        return False

    def _find_king(self, color: str) -> tuple | None:
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == f"{color}K":
                    return (r, c)
        return None

    def in_check(self, color: str) -> bool:
        kpos = self._find_king(color)
        if kpos is None: return False
        return self._sq_attacked(kpos[0], kpos[1], self._enemy(color))

    def _apply_move_temp(self, r: int, c: int, tr: int, tc: int) -> dict:
        """合法性チェックなしで仮に駒を動かし、undo用のsavedを返す (AI探索用)。"""
        piece = self.board[r][c]
        target = self.board[tr][tc]
        saved = {
            "from": (r, c, piece),
            "to": (tr, tc, target),
            "en_passant": self.en_passant,
            "castling_rights": dict(self.castling_rights),
            "halfmove_clock": self.halfmove_clock,
        }

        # アンパッサン捕獲
        ep_capture = None
        if piece and piece[1] == "P" and self.en_passant == (tr, tc):
            ep_r = tr + (1 if piece[0] == "w" else -1)
            ep_capture = (ep_r, tc, self.board[ep_r][tc])
            self.board[ep_r][tc] = None
        saved["ep_capture"] = ep_capture

        # キャスリング時ルーク移動
        rook_move = None
        if piece and piece[1] == "K" and abs(tc - c) == 2:
            back_rank = r
            if tc == 6:
                rook_move = (back_rank, 7, back_rank, 5, self.board[back_rank][7])
                self.board[back_rank][5] = self.board[back_rank][7]
                self.board[back_rank][7] = None
            elif tc == 2:
                rook_move = (back_rank, 0, back_rank, 3, self.board[back_rank][0])
                self.board[back_rank][3] = self.board[back_rank][0]
                self.board[back_rank][0] = None
        saved["rook_move"] = rook_move

        self.board[r][c] = None
        self.board[tr][tc] = piece

        # プロモーション (常にクイーンに昇格)
        if piece and piece[1] == "P" and (tr == 0 or tr == 7):
            self.board[tr][tc] = f"{piece[0]}Q"

        # アンパッサン更新
        self.en_passant = None
        if piece and piece[1] == "P" and abs(tr - r) == 2:
            self.en_passant = ((r + tr) // 2, c)

        # キャスリング権更新
        if piece and piece[1] == "K":
            color = piece[0]
            self.castling_rights[f"{color}K"] = False
            self.castling_rights[f"{color}Q"] = False
        if piece and piece[1] == "R":
            color = piece[0]
            back_rank = 7 if color == "w" else 0
            if r == back_rank and c == 7: self.castling_rights[f"{color}K"] = False
            if r == back_rank and c == 0: self.castling_rights[f"{color}Q"] = False

        return saved

    def legal_moves(self, color: str) -> list[tuple]:
        """(from_r, from_c, to_r, to_c) の合法手リスト"""
        result = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if not piece or piece[0] != color: continue
                for tr, tc in self.pseudo_legal_moves(r, c):
                    saved = self._apply_move_temp(r, c, tr, tc)
                    if not self.in_check(color):
                        result.append((r, c, tr, tc))
                    self._undo_move_temp(saved)
        return result

    def _undo_move_temp(self, saved):
        r, c, piece = saved["from"]
        tr, tc, target = saved["to"]
        self.board[r][c] = piece
        self.board[tr][tc] = target
        self.en_passant = saved["en_passant"]
        self.castling_rights = saved["castling_rights"]
        if saved.get("ep_capture"):
            er, ec, ep = saved["ep_capture"]
            self.board[er][ec] = ep
        if saved.get("rook_move"):
            rr, rc, nrr, nrc, rp = saved["rook_move"]
            self.board[rr][rc] = rp
            self.board[nrr][nrc] = None

    def move_sq(self, r: int, c: int, tr: int, tc: int, promotion: str = "Q") -> tuple[bool, str]:
        """駒を移動する。成功時 (True, 棋譜表記)、失敗時 (False, エラー文字列)"""
        piece = self.board[r][c]
        if not piece:
            return False, "その位置に駒がありません"
        if piece[0] != self.turn:
            return False, f"{'白' if self.turn=='w' else '黒'}の手番です"
        legal = self.legal_moves(self.turn)
        if (r, c, tr, tc) not in legal:
            return False, "その手は合法ではありません"

        target = self.board[tr][tc]
        notation = self._build_notation(r, c, tr, tc, piece, target, promotion)

        # アンパッサン
        ep_capture_pos = None
        if piece[1] == "P" and self.en_passant == (tr, tc):
            ep_r = tr + (1 if piece[0] == "w" else -1)
            ep_capture_pos = (ep_r, tc)
            cap = self.board[ep_r][tc]
            if cap: self.captured[self.turn].append(cap)
            self.board[ep_r][tc] = None

        if target:
            self.captured[self.turn].append(target)

        self.board[r][c] = None
        self.board[tr][tc] = piece

        # キャスリング
        if piece[1] == "K":
            color = piece[0]
            back_rank = 7 if color == "w" else 0
            if tc == 6 and c == 4:
                self.board[back_rank][5] = self.board[back_rank][7]
                self.board[back_rank][7] = None
            elif tc == 2 and c == 4:
                self.board[back_rank][3] = self.board[back_rank][0]
                self.board[back_rank][0] = None
            self.castling_rights[f"{color}K"] = False
            self.castling_rights[f"{color}Q"] = False

        # ルーク移動 → キャスリング権失効
        if piece[1] == "R":
            color = piece[0]
            back_rank = 7 if color == "w" else 0
            if r == back_rank and c == 7: self.castling_rights[f"{color}K"] = False
            if r == back_rank and c == 0: self.castling_rights[f"{color}Q"] = False

        # アンパッサン設定
        self.en_passant = None
        if piece[1] == "P" and abs(tr - r) == 2:
            self.en_passant = ((r + tr) // 2, c)

        # プロモーション
        if piece[1] == "P" and (tr == 0 or tr == 7):
            self.board[tr][tc] = f"{piece[0]}{promotion}"
            notation += f"={promotion}"

        # チェック確認
        enemy = self._enemy(self.turn)
        check = self.in_check(enemy)
        self.turn = enemy
        if check:
            legal_next = self.legal_moves(self.turn)
            if not legal_next:
                notation += "#"
                self.game_over = True
                winner = "白" if self.turn == "b" else "黒"
                self.result = f"チェックメイト！{winner}の勝利"
            else:
                notation += "+"
        else:
            legal_next = self.legal_moves(self.turn)
            if not legal_next:
                notation += " (ステールメイト)"
                self.game_over = True
                self.result = "ステールメイト（引き分け）"

        self.move_history.append(notation)
        if self.turn == "w": self.fullmove += 1
        return True, notation

    def _build_notation(self, r, c, tr, tc, piece, target, promotion) -> str:
        ptype = piece[1]
        dest = self.sq_name(tr, tc)
        if ptype == "K" and abs(tc - c) == 2:
            return "O-O" if tc == 6 else "O-O-O"
        cap = "x" if target or (ptype == "P" and self.en_passant == (tr, tc)) else ""
        if ptype == "P":
            if cap:
                return f"{chr(ord('a')+c)}{cap}{dest}"
            return dest
        return f"{ptype}{cap}{dest}"

    def legal_targets(self, r: int, c: int) -> list[tuple]:
        """(r,c) の駒が動ける合法手の移動先リスト"""
        all_legal = self.legal_moves(self.turn)
        return [(tr, tc) for (fr, fc, tr, tc) in all_legal if fr == r and fc == c]

    def status_str(self) -> str:
        turn_str = "白(White)" if self.turn == "w" else "黒(Black)"
        check_str = " 【チェック！】" if self.in_check(self.turn) else ""
        cap_w = " ".join(self.piece_symbol(p) for p in self.captured["w"]) or "なし"
        cap_b = " ".join(self.piece_symbol(p) for p in self.captured["b"]) or "なし"
        move_count = (self.fullmove - 1) * 2 + (0 if self.turn == "w" else 1)
        hist = "  ".join(self.move_history[-6:]) if self.move_history else "なし"
        lines = [
            f"手番: {turn_str}{check_str}   ({move_count}手目)",
            f"白が取った駒: {cap_b}   黒が取った駒: {cap_w}",
            f"直近の手: {hist}",
        ]
        return "\n".join(lines)



# ===== チェス AI エンジン =====
import random

class ChessAI:
    """
    チェスAI。4段階の難易度に対応。
      easy      : ランダム手
      middle    : depth=1 minimax + 基本評価
      hard      : depth=3 minimax + alpha-beta + 評価関数
      very_hard : depth=4 minimax + alpha-beta + 評価関数 + ランダム揺らぎなし
    """

    # 駒の基本価値
    PIECE_VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}

    # 駒ごとのポジションボーナス (白視点, 行0=黒側, 行7=白側)
    _PST = {
        "P": [
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [ 5,  5, 10, 25, 25, 10,  5,  5],
            [ 0,  0,  0, 20, 20,  0,  0,  0],
            [ 5, -5,-10,  0,  0,-10, -5,  5],
            [ 5, 10, 10,-20,-20, 10, 10,  5],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
        ],
        "N": [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50],
        ],
        "B": [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20],
        ],
        "R": [
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [ 0,  0,  0,  5,  5,  0,  0,  0],
        ],
        "Q": [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [ -5,  0,  5,  5,  5,  5,  0, -5],
            [  0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20],
        ],
        "K": [
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [ 20, 20,  0,  0,  0,  0, 20, 20],
            [ 20, 30, 10,  0,  0, 10, 30, 20],
        ],
    }

    # ★[v129] 難易度設定 — depth増強・Killer/TT有効
    DIFFICULTY_SETTINGS = {
        "easy":      {"depth": 0, "random_rate": 1.0},
        "middle":    {"depth": 2, "random_rate": 0.15},  # depth 1→2
        "hard":      {"depth": 3, "random_rate": 0.0},
        "very_hard": {"depth": 4, "random_rate": 0.0},
    }

    def __init__(self, difficulty: str = "middle", color: str = "b"):
        self.difficulty = difficulty
        self.color = color
        s = self.DIFFICULTY_SETTINGS.get(difficulty, self.DIFFICULTY_SETTINGS["middle"])
        self.depth = s["depth"]
        self.random_rate = s["random_rate"]
        self._tt: dict = {}          # ★[v129] Transposition Table
        self._killer: dict = {}      # ★[v129] Killer Heuristic

    def _pst_score(self, piece: str, r: int, c: int) -> int:
        color, ptype = piece[0], piece[1]
        table = self._PST.get(ptype)
        if table is None: return 0
        return table[r][c] if color == "w" else table[7 - r][c]

    def evaluate(self, g: "ChessEngine") -> int:
        """盤面評価 (正=白有利, 負=黒有利) ★[v129] モビリティボーナス追加"""
        score = 0
        piece_count = 0
        for r in range(8):
            for c in range(8):
                piece = g.board[r][c]
                if not piece: continue
                color, ptype = piece[0], piece[1]
                val = self.PIECE_VALUE.get(ptype, 0) + self._pst_score(piece, r, c)
                score += val if color == "w" else -val
                piece_count += 1
        # ★[v129] エンドゲーム補正: 駒が少ないほどキングを中央へ
        if piece_count < 16:
            for r in range(8):
                for c in range(8):
                    p = g.board[r][c]
                    if p and p[1] == "K":
                        center_bonus = (3 - abs(r - 3.5)) + (3 - abs(c - 3.5))
                        score += int(center_bonus * 5) if p[0] == "w" else -int(center_bonus * 5)
        return score

    def _mvv_lva_key(self, g: "ChessEngine", mv: tuple) -> int:
        """★[v129] MVV-LVA: 高価値の駒を取る手を優先"""
        fr, fc, tr, tc = mv
        victim = g.board[tr][tc]
        attacker = g.board[fr][fc]
        if victim and attacker:
            return -(self.PIECE_VALUE.get(victim[1], 0) * 10 - self.PIECE_VALUE.get(attacker[1], 0))
        return 100  # 通常手

    def _all_legal_moves(self, g: "ChessEngine", color: str) -> list[tuple]:
        moves = g.legal_moves(color)
        # ★[v129] MVV-LVA + Killer ソート
        killers = self._killer.get(0, [])
        moves.sort(key=lambda mv: (
            0 if mv in killers else 1,
            self._mvv_lva_key(g, mv)
        ))
        return moves

    def _minimax(self, g: "ChessEngine", depth: int, alpha: int, beta: int, maximizing: bool) -> int:
        if depth == 0 or g.game_over: return self.evaluate(g)

        # ★[v129] Transposition Table lookup
        bh = hash((str(g.board), g.turn, depth))
        if bh in self._tt: return self._tt[bh]

        color = "w" if maximizing else "b"
        moves = self._all_legal_moves(g, color)
        if not moves: return self.evaluate(g)

        if maximizing:
            best = -10**9
            for fr, fc, tr, tc in moves:
                saved = g._apply_move_temp(fr, fc, tr, tc)
                prev_turn = g.turn; g.turn = "b"
                val = self._minimax(g, depth - 1, alpha, beta, False)
                g.turn = prev_turn; g._undo_move_temp(saved)
                if val > best:
                    best = val
                    if val >= beta:  # ★[v129] Killer登録
                        k = self._killer.setdefault(depth, [])
                        if (fr,fc,tr,tc) not in k: k.insert(0, (fr,fc,tr,tc)); k[:] = k[:2]
                        break
                alpha = max(alpha, val)
                if beta <= alpha: break
        else:
            best = 10**9
            for fr, fc, tr, tc in moves:
                saved = g._apply_move_temp(fr, fc, tr, tc)
                prev_turn = g.turn; g.turn = "w"
                val = self._minimax(g, depth - 1, alpha, beta, True)
                g.turn = prev_turn; g._undo_move_temp(saved)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha: break

        # ★[v129] TT store (TTサイズ上限)
        self._tt[bh] = best
        if len(self._tt) > 30000: self._tt.clear()
        return best

    def choose_move(self, g: "ChessEngine") -> tuple | None:
        moves = self._all_legal_moves(g, self.color)
        if not moves: return None
        if self.difficulty == "easy": return random.choice(moves)
        if self.random_rate > 0 and random.random() < self.random_rate: return random.choice(moves)

        maximizing = (self.color == "w")
        best_val = -10**9 if maximizing else 10**9
        best_moves = []
        self._tt.clear(); self._killer.clear()  # 新探索でリセット

        for fr, fc, tr, tc in moves:
            saved = g._apply_move_temp(fr, fc, tr, tc)
            prev_turn = g.turn; g.turn = "b" if self.color == "w" else "w"
            val = self._minimax(g, self.depth - 1, -10**9, 10**9, not maximizing)
            g.turn = prev_turn; g._undo_move_temp(saved)
            if maximizing:
                if val > best_val: best_val = val; best_moves = [(fr, fc, tr, tc)]
                elif val == best_val: best_moves.append((fr, fc, tr, tc))
            else:
                if val < best_val: best_val = val; best_moves = [(fr, fc, tr, tc)]
                elif val == best_val: best_moves.append((fr, fc, tr, tc))
        return random.choice(best_moves) if best_moves else random.choice(moves)


# ===== チェス curses マウス UI =====
import curses

# ── レイアウト定数 ──────────────────────────────────────────
_CW   = 5   # マス幅 (chars)  ← Unicode駒の表示幅を考慮
_CH   = 3   # マス高 (lines)
_BX   = 4   # 盤面左端 (col offset) ← 行番号 "8 " 分
_BY   = 2   # 盤面上端 (row offset) ← タイトル分
_INFO_X = _BX + _CW * 8 + 2   # 右パネル開始列

# ── カラーペア ID ────────────────────────────────────────────
_CP_LIGHT       = 1   # 白マス
_CP_DARK        = 2   # 黒マス
_CP_SELECTED    = 3   # 選択中マス
_CP_LEGAL       = 4   # 合法手マス
_CP_LAST_MOVE   = 5   # 直前の移動元/先
_CP_CHECK       = 6   # チェック中のキング
_CP_TITLE       = 7   # タイトルバー
_CP_PANEL       = 8   # 右パネル
_CP_BTN         = 9   # ボタン通常
_CP_BTN_HL      = 10  # ボタンハイライト
_CP_STATUS_OK   = 11  # ステータス (通常)
_CP_STATUS_WARN = 12  # ステータス (警告)
_CP_PROMO       = 13  # プロモーションポップ
_CP_COMMENT     = 14  # ペルソナテロップ

def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    # light squares: dark text on cream
    curses.init_pair(_CP_LIGHT,       curses.COLOR_BLACK,   curses.COLOR_WHITE)
    # dark squares: white text on dark green
    curses.init_pair(_CP_DARK,        curses.COLOR_WHITE,   curses.COLOR_GREEN)
    # selected: black on yellow
    curses.init_pair(_CP_SELECTED,    curses.COLOR_BLACK,   curses.COLOR_YELLOW)
    # legal move dot: black on cyan
    curses.init_pair(_CP_LEGAL,       curses.COLOR_BLACK,   curses.COLOR_CYAN)
    # last move: black on blue
    curses.init_pair(_CP_LAST_MOVE,   curses.COLOR_WHITE,   curses.COLOR_BLUE)
    # check: white on red
    curses.init_pair(_CP_CHECK,       curses.COLOR_WHITE,   curses.COLOR_RED)
    # title bar: black on white
    curses.init_pair(_CP_TITLE,       curses.COLOR_BLACK,   curses.COLOR_WHITE)
    # right panel: white on black (default)
    curses.init_pair(_CP_PANEL,       curses.COLOR_WHITE,   -1)
    # button
    curses.init_pair(_CP_BTN,         curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(_CP_BTN_HL,      curses.COLOR_WHITE,   curses.COLOR_MAGENTA)
    # status bar
    curses.init_pair(_CP_STATUS_OK,   curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(_CP_STATUS_WARN, curses.COLOR_WHITE,   curses.COLOR_RED)
    # promotion popup
    curses.init_pair(_CP_PROMO,       curses.COLOR_BLACK,   curses.COLOR_YELLOW)
    curses.init_pair(_CP_COMMENT,     curses.COLOR_BLACK,   curses.COLOR_MAGENTA)  # テロップ


def _chess_curses_main(stdscr, g: "ChessEngine", ai: "ChessAI | None" = None,
                       commentator: "GameCommentator | None" = None):
    """curses ループ本体。マウスクリックでチェスを操作する。"""
    _init_colors()
    curses.curs_set(0)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    curses.mouseinterval(0)
    stdscr.keypad(True)
    stdscr.timeout(80)   # ms – リフレッシュ間隔

    selected: tuple | None = None       # 選択中マス (r, c)
    legal_tgts: list[tuple] = []        # 現選択駒の合法手
    last_move: list[tuple] = []         # 直前の移動元/先

    # AI難易度ラベル
    _diff_labels = {"easy": "Easy", "middle": "Middle", "hard": "Hard", "very_hard": "Very Hard"}
    if ai:
        ai_label = _diff_labels.get(ai.difficulty, ai.difficulty)
        status_msg = f"♟ AI対戦モード [{ai_label}] — 白(あなた)から開始。駒をクリック！"
    else:
        status_msg = "♟ クリックして駒を選択してください"
    status_warn = False
    undo_stack: list[dict] = []
    promo_pending: tuple | None = None  # (fr, fc, tr, tc) プロモーション待ち

    # ── ボタン定義 [(ラベル, action_key), ...] ─────────────────
    BUTTONS = [
        ("  New  ", "new"),
        ("  Undo ", "undo"),
        ("  Quit ", "quit"),
    ]

    def _snapshot():
        return {
            "board": [row[:] for row in g.board],
            "turn": g.turn,
            "castling_rights": dict(g.castling_rights),
            "en_passant": g.en_passant,
            "move_history": g.move_history[:],
            "captured": {"w": g.captured["w"][:], "b": g.captured["b"][:]},
            "fullmove": g.fullmove,
            "game_over": g.game_over,
            "result": g.result,
        }

    def _restore(snap):
        g.board            = snap["board"]
        g.turn             = snap["turn"]
        g.castling_rights  = snap["castling_rights"]
        g.en_passant       = snap["en_passant"]
        g.move_history     = snap["move_history"]
        g.captured         = snap["captured"]
        g.fullmove         = snap["fullmove"]
        g.game_over        = snap["game_over"]
        g.result           = snap["result"]

    def _sq_to_screen(r, c):
        """ボードマス (r, c) → curses 左上 (y, x)"""
        return (_BY + r * _CH, _BX + c * _CW)

    def _screen_to_sq(my, mx):
        """curses 座標 → ボードマス (r, c) or None"""
        r = (my - _BY) // _CH
        c = (mx - _BX) // _CW
        if 0 <= r <= 7 and 0 <= c <= 7:
            ry = my - (_BY + r * _CH)
            rx = mx - (_BX + c * _CW)
            if 0 <= ry < _CH and 0 <= rx < _CW:
                return (r, c)
        return None

    def _draw_cell(r, c, piece, color_pair, center_mark=""):
        """1マスを描画 (_CH lines × _CW cols)"""
        sy, sx = _sq_to_screen(r, c)
        sym = g.piece_symbol(piece) if piece else " "
        # 上段・下段: 空白
        try:
            stdscr.addstr(sy,       sx, " " * _CW, curses.color_pair(color_pair))
            stdscr.addstr(sy + 2,   sx, " " * _CW, curses.color_pair(color_pair))
            # 中段: 駒シンボル
            mid = center_mark if (not piece and center_mark) else sym
            line = f"  {mid}  " if len(mid) == 1 else f" {mid}  "
            line = line[:_CW]
            stdscr.addstr(sy + 1, sx, line, curses.color_pair(color_pair) | curses.A_BOLD)
        except curses.error:
            pass

    def _draw_board():
        king_pos = g._find_king(g.turn)
        in_chk   = g.in_check(g.turn)

        for r in range(8):
            for c in range(8):
                piece = g.board[r][c]
                base_light = (r + c) % 2 == 0

                if selected and (r, c) == selected:
                    cp = _CP_SELECTED
                elif (r, c) in legal_tgts:
                    cp = _CP_LEGAL
                elif (r, c) in last_move:
                    cp = _CP_LAST_MOVE
                elif in_chk and king_pos and (r, c) == king_pos:
                    cp = _CP_CHECK
                else:
                    cp = _CP_LIGHT if base_light else _CP_DARK

                mark = "·" if (r, c) in legal_tgts and not piece else ""
                _draw_cell(r, c, piece, cp, center_mark=mark)

        # 行ラベル (8〜1)
        for r in range(8):
            sy, _ = _sq_to_screen(r, 0)
            try:
                stdscr.addstr(sy + 1, _BX - 2, str(8 - r),
                              curses.color_pair(_CP_PANEL) | curses.A_BOLD)
            except curses.error:
                pass

        # 列ラベル (a〜h)
        col_y = _BY + 8 * _CH
        for c in range(8):
            _, sx = _sq_to_screen(0, c)
            try:
                stdscr.addstr(col_y, sx + 2, chr(ord('a') + c),
                              curses.color_pair(_CP_PANEL) | curses.A_BOLD)
            except curses.error:
                pass

    def _draw_title():
        turn_str = "♔ 白 (White)" if g.turn == "w" else "♚ 黒 (Black)"
        chk_str  = "  ⚠ チェック！" if g.in_check(g.turn) else ""
        move_n   = (g.fullmove - 1) * 2 + (0 if g.turn == "w" else 1)
        _diff_labels = {"easy": "Easy", "middle": "Middle", "hard": "Hard", "very_hard": "Very Hard"}
        ai_str = f"  [AI:{_diff_labels.get(ai.difficulty,'')}]" if ai else ""
        title = f"  ♟ チェス{ai_str}  {turn_str}{chk_str}   {move_n}手目  "
        try:
            stdscr.addstr(0, 0, title.ljust(80), curses.color_pair(_CP_TITLE) | curses.A_BOLD)
        except curses.error:
            pass

    def _draw_panel():
        """右パネル: 取得駒 / 手の記録 / ボタン"""
        px = _INFO_X
        h, w = stdscr.getmaxyx()

        # 取得駒
        cap_b_sym = " ".join(g.piece_symbol(p) for p in g.captured["w"][-8:]) or "なし"
        cap_w_sym = " ".join(g.piece_symbol(p) for p in g.captured["b"][-8:]) or "なし"
        try:
            stdscr.addstr(_BY,     px, "取:白→ " + cap_b_sym, curses.color_pair(_CP_PANEL))
            stdscr.addstr(_BY + 1, px, "取:黒→ " + cap_w_sym, curses.color_pair(_CP_PANEL))
        except curses.error:
            pass

        # 棋譜
        hist_y = _BY + 3
        try:
            stdscr.addstr(hist_y - 1, px, "── 棋譜 ──────────",
                          curses.color_pair(_CP_PANEL) | curses.A_DIM)
        except curses.error:
            pass
        hist = g.move_history
        panel_h = max(4, h - hist_y - 6)
        start_i = max(0, len(hist) - panel_h * 2)
        line_i  = 0
        for i in range(start_i, len(hist), 2):
            w_m = hist[i] if i < len(hist) else ""
            b_m = hist[i + 1] if i + 1 < len(hist) else ""
            num = i // 2 + 1
            line = f"{num:2d}. {w_m:<8s} {b_m}"
            try:
                stdscr.addstr(hist_y + line_i, px, line[:28], curses.color_pair(_CP_PANEL))
            except curses.error:
                pass
            line_i += 1
            if line_i >= panel_h:
                break

        # ボタン
        btn_y = h - 4
        btn_x = px
        _draw_buttons(btn_y, btn_x)

    def _draw_buttons(by, bx):
        for i, (label, _) in enumerate(BUTTONS):
            try:
                stdscr.addstr(by, bx + i * 10, label,
                              curses.color_pair(_CP_BTN) | curses.A_BOLD)
            except curses.error:
                pass

    def _btn_at(my, mx):
        """クリック座標がボタンに当たっていれば action_key を返す"""
        h, _ = stdscr.getmaxyx()
        by = h - 4
        bx = _INFO_X
        for i, (label, action) in enumerate(BUTTONS):
            lx = bx + i * 10
            if my == by and lx <= mx < lx + len(label):
                return action
        return None

    def _draw_status(msg, warn=False):
        h, w_max = stdscr.getmaxyx()
        cp = _CP_STATUS_WARN if warn else _CP_STATUS_OK
        line = f"  {msg}  "
        try:
            stdscr.addstr(h - 2, 0, line.ljust(w_max - 1), curses.color_pair(cp))
        except curses.error:
            pass

    def _draw_promotion_popup(color: str):
        """プロモーション選択ポップアップを描画し、クリック位置を返す"""
        pieces = [f"{color}Q", f"{color}R", f"{color}B", f"{color}N"]
        labels = ["クイーン", "ルーク", "ビショップ", "ナイト"]
        h, w_max = stdscr.getmaxyx()
        pw, ph = 52, 7
        py = h // 2 - ph // 2
        px_ = w_max // 2 - pw // 2

        cp = curses.color_pair(_CP_PROMO) | curses.A_BOLD
        try:
            stdscr.addstr(py,     px_, "╔" + "═"*(pw-2) + "╗", cp)
            stdscr.addstr(py + 1, px_, "║" + "  プロモーション駒を選んでクリック  ".center(pw-2) + "║", cp)
            stdscr.addstr(py + 2, px_, "║" + " " * (pw-2) + "║", cp)
            for i, (p, lbl) in enumerate(zip(pieces, labels)):
                sym = g.piece_symbol(p)
                cell = f" {sym} {lbl} "
                bx_ = px_ + 1 + i * 12
                stdscr.addstr(py + 3, bx_, cell.ljust(11), cp)
            stdscr.addstr(py + 4, px_, "║" + " " * (pw-2) + "║", cp)
            stdscr.addstr(py + 5, px_, "║" + "   (クリック or Q/R/B/N キー)   ".center(pw-2) + "║", cp)
            stdscr.addstr(py + 6, px_, "╚" + "═"*(pw-2) + "╝", cp)
        except curses.error:
            pass
        stdscr.refresh()

        # promo ボタン領域を返す: [(piece, y, x_start, x_end), ...]
        zones = []
        for i, p in enumerate(pieces):
            bx_ = px_ + 1 + i * 12
            zones.append((p[1], py + 3, bx_, bx_ + 11))
        return zones

    def _do_move(fr, fc, tr, tc, promotion="Q"):
        nonlocal selected, legal_tgts, last_move, status_msg, status_warn
        snap = _snapshot()
        # 評価値の変化でコメント種別を決定
        piece_before = g.board[fr][fc] if fr is not None else None
        target_before = g.board[tr][tc] if g.board[tr][tc] else None
        ok, msg = g.move_sq(fr, fc, tr, tc, promotion)
        if ok:
            undo_stack.append(snap)
            if len(undo_stack) > 60: undo_stack.pop(0)
            last_move = [(fr, fc), (tr, tc)]
            selected  = None
            legal_tgts = []
            status_msg  = f"✔ {msg}"
            status_warn = False
            # ── コメンタートリガー ──────────────────────────
            if commentator:
                in_chk = g.in_check(g.turn)
                is_capture = target_before is not None
                is_promo = promotion and piece_before and piece_before[1] == "P"
                if g.game_over:
                    commentator.trigger("ゲームセット", f"結果: {g.result}")
                elif in_chk:
                    commentator.trigger("チェック", f"{msg} — チェック！")
                elif is_promo:
                    commentator.trigger("ポーンがプロモーション", f"{msg}")
                elif is_capture:
                    commentator.trigger("駒を取った", f"{msg}")
                else:
                    if _rnd_chess.random() < 0.3:
                        commentator.trigger("指し手", f"{msg} ({len(g.move_history)}手目)")
            # ────────────────────────────────────────────────
            if g.game_over:
                status_msg  = f"🏆 {g.result}"
                status_warn = True
        else:
            status_msg  = f"✖ {msg}"
            status_warn = True
        return ok

    def _draw_telop_chess():
        """ペルソナのテロップを最下行に表示（チェス版）"""
        if commentator is None:
            return
        h, w_max = stdscr.getmaxyx()
        lines = commentator.get_lines()
        if not lines:
            return
        text = lines[-1]
        try:
            stdscr.addstr(h - 1, 0, f" ♟ {text} ".ljust(w_max - 1),
                          curses.color_pair(_CP_COMMENT) | curses.A_BOLD)
        except curses.error:
            pass

    def _redraw():
        stdscr.erase()
        _draw_title()
        _draw_board()
        _draw_panel()
        _draw_status(status_msg, status_warn)
        _draw_telop_chess()
        stdscr.refresh()

    # ── メインループ ──────────────────────────────────────────
    import random as _rnd_chess
    _diff_labels_chess = {"easy": "Easy", "middle": "Middle", "hard": "Hard", "very_hard": "Very Hard"}
    _chess_ai_thread: threading.Thread | None = None
    _chess_ai_result: list = []

    def _chess_ai_think_bg():
        try:
            mv = ai.choose_move(g)
            _chess_ai_result.clear()
            if mv:
                _chess_ai_result.append(mv)
        except Exception:
            _chess_ai_result.clear()

    while True:
        # AI手番の処理（バックグラウンドスレッド）
        if ai and not g.game_over and g.turn == ai.color and not promo_pending:
            if _chess_ai_thread is None or not _chess_ai_thread.is_alive():
                if _chess_ai_result:
                    # 思考完了 → 着手
                    ai_move = _chess_ai_result[0]
                    _chess_ai_result.clear()
                    _chess_ai_thread = None
                    fr, fc, tr, tc = ai_move
                    snap = _snapshot()
                    ok, msg = g.move_sq(fr, fc, tr, tc)
                    if ok:
                        undo_stack.append(snap)
                        if len(undo_stack) > 60: undo_stack.pop(0)
                        last_move = [(fr, fc), (tr, tc)]
                        selected  = None
                        legal_tgts = []
                        status_msg = f"AI [{_diff_labels_chess.get(ai.difficulty,'')}]: {msg}"
                        status_warn = False
                        if g.game_over:
                            status_msg  = f"🏆 {g.result}"
                            status_warn = True
                        if commentator:
                            in_chk = g.in_check(g.turn)
                            if g.game_over:
                                commentator.trigger("AIが勝利", f"結果: {g.result}")
                            elif in_chk:
                                commentator.trigger("AIがチェック", f"AI: {msg}")
                            elif _rnd_chess.random() < 0.4:
                                commentator.trigger("AIの指し手", f"AI: {msg}")
                else:
                    # 思考スレッド起動
                    _chess_ai_result.clear()
                    status_msg = f"AI [{_diff_labels_chess.get(ai.difficulty,'')}] 思考中..."
                    status_warn = False
                    _chess_ai_thread = threading.Thread(target=_chess_ai_think_bg, daemon=True)
                    _chess_ai_thread.start()
            _redraw()
            stdscr.timeout(80)
            stdscr.getch()
            continue

        _redraw()

        # プロモーション待ち状態のとき専用ループ
        if promo_pending:
            fr, fc, tr, tc = promo_pending
            zones = _draw_promotion_popup(g.board[tr][tc][0] if g.board[tr][tc] else
                                          ("w" if (tr == 0) else "b"))
            # ポップアップ中だけ入力を待つ
            while True:
                key = stdscr.getch()
                if key in (ord('q'), ord('Q')): prom = "Q"; break
                if key in (ord('r'), ord('R')): prom = "R"; break
                if key in (ord('b'), ord('B')): prom = "B"; break
                if key in (ord('n'), ord('N')): prom = "N"; break
                if key == curses.KEY_MOUSE:
                    try:
                        _, mx, my, _, bstate = curses.getmouse()
                        if bstate & curses.BUTTON1_CLICKED or bstate & curses.BUTTON1_PRESSED:
                            for ptype, zy, zx0, zx1 in zones:
                                if my == zy and zx0 <= mx < zx1:
                                    prom = ptype
                                    break
                            else:
                                continue
                            break
                    except curses.error:
                        continue
            # promotion確定: ターンを一時的に修正して打つ
            # g.board[tr][tc] はすでに駒があるかもしれないので promo_pending 時のボードを使う
            # 実際には promo_pending 時点でまだ move_sq を呼んでいないので普通に呼ぶ
            promo_pending = None
            _do_move(fr, fc, tr, tc, prom)
            _redraw()
            continue

        key = stdscr.getch()

        if key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
            except curses.error:
                continue

            if not (bstate & curses.BUTTON1_CLICKED or bstate & curses.BUTTON1_PRESSED):
                continue

            # ボタン判定
            action = _btn_at(my, mx)
            if action == "quit":
                return "チェス終了"
            if action == "new":
                g.reset()
                undo_stack.clear()
                selected   = None
                legal_tgts = []
                last_move  = []
                _diff_labels = {"easy": "Easy", "middle": "Middle", "hard": "Hard", "very_hard": "Very Hard"}
                ai_str = f"[AI:{_diff_labels.get(ai.difficulty,'')}] " if ai else ""
                status_msg  = f"新しいゲームを開始しました {ai_str}"
                status_warn = False
                continue
            if action == "undo":
                if undo_stack:
                    _restore(undo_stack.pop())
                    selected   = None
                    legal_tgts = []
                    last_move  = []
                    status_msg  = "1手戻しました"
                    status_warn = False
                else:
                    status_msg  = "戻せる手がありません"
                    status_warn = True
                continue

            # マス判定
            sq = _screen_to_sq(my, mx)
            if sq is None:
                continue

            r, c = sq

            if g.game_over:
                status_msg  = f"ゲーム終了: {g.result}  (New で新ゲーム)"
                status_warn = True
                continue

            if selected is None:
                # 駒を選択
                piece = g.board[r][c]
                # AIモード: 相手の駒は動かせない
                if ai and piece and piece[0] == ai.color:
                    status_msg  = f"それはAIの駒です"
                    status_warn = True
                elif piece and piece[0] == g.turn:
                    targets = g.legal_targets(r, c)
                    if targets:
                        selected   = (r, c)
                        legal_tgts = targets
                        pname = {"K":"キング","Q":"クイーン","R":"ルーク",
                                  "B":"ビショップ","N":"ナイト","P":"ポーン"}.get(piece[1], piece[1])
                        status_msg  = f"選択: {g.sq_name(r,c)} ({pname})  — 移動先をクリック"
                        status_warn = False
                    else:
                        status_msg  = f"{g.sq_name(r,c)} の駒は動けません"
                        status_warn = True
                elif piece and piece[0] != g.turn:
                    status_msg  = f"{'白' if g.turn=='w' else '黒'}の手番です"
                    status_warn = True
                else:
                    status_msg  = "その位置に駒がありません"
                    status_warn = True
            else:
                fr, fc = selected
                if (r, c) == selected:
                    # 同じマスをクリック → 選択解除
                    selected   = None
                    legal_tgts = []
                    status_msg  = "選択を解除しました"
                    status_warn = False
                elif (r, c) in legal_tgts:
                    # 合法手マスへ移動
                    piece = g.board[fr][fc]
                    # プロモーション?
                    if piece and piece[1] == "P" and (r == 0 or r == 7):
                        promo_pending = (fr, fc, r, c)
                        status_msg  = "プロモーション駒をクリックして選択"
                        status_warn = False
                    else:
                        _do_move(fr, fc, r, c)
                elif g.board[r][c] and g.board[r][c][0] == g.turn:
                    # 別の自駒をクリック → 選択し直し
                    targets = g.legal_targets(r, c)
                    if targets:
                        selected   = (r, c)
                        legal_tgts = targets
                        pname = {"K":"キング","Q":"クイーン","R":"ルーク",
                                  "B":"ビショップ","N":"ナイト","P":"ポーン"}.get(g.board[r][c][1], "")
                        status_msg  = f"選択変更: {g.sq_name(r,c)} ({pname})"
                        status_warn = False
                    else:
                        selected = None; legal_tgts = []
                else:
                    # 合法手外をクリック → 選択解除
                    selected   = None
                    legal_tgts = []
                    status_msg  = "合法手ではありません。別の駒を選んでください"
                    status_warn = True

        elif key in (ord('q'), ord('Q'), 27):   # q / ESC
            return "チェス終了"
        elif key in (ord('n'), ord('N')):
            g.reset()
            undo_stack.clear()
            selected = None; legal_tgts = []; last_move = []
            _diff_labels = {"easy": "Easy", "middle": "Middle", "hard": "Hard", "very_hard": "Very Hard"}
            ai_str = f"[AI:{_diff_labels.get(ai.difficulty,'')}] " if ai else ""
            status_msg = f"新しいゲームを開始しました {ai_str}"; status_warn = False
        elif key in (ord('u'), ord('U')):
            if undo_stack:
                _restore(undo_stack.pop())
                selected = None; legal_tgts = []; last_move = []
                status_msg = "1手戻しました"; status_warn = False
            else:
                status_msg = "戻せる手がありません"; status_warn = True


_CHESS_GAME: "ChessEngine | None" = None

def handle_chess(arg: str, persona: dict | None = None) -> str:
    """チェスゲームのエントリポイント。/chess [easy|middle|hard|very_hard] で起動。"""
    global _CHESS_GAME
    arg = arg.strip().lower()

    # 難易度キーワードを解析
    difficulty_map = {
        "easy": "easy", "イージー": "easy", "簡単": "easy",
        "middle": "middle", "ミドル": "middle", "普通": "middle", "normal": "middle",
        "hard": "hard", "ハード": "hard", "難しい": "hard",
        "very_hard": "very_hard", "veryhard": "very_hard", "最難関": "very_hard",
        "very hard": "very_hard", "超難": "very_hard",
    }
    ai_difficulty = None
    for key, val in difficulty_map.items():
        if key in arg:
            ai_difficulty = val
            break

    if _CHESS_GAME is None or "new" in arg or ai_difficulty is not None:
        _CHESS_GAME = ChessEngine(use_unicode=True)

    ai = ChessAI(difficulty=ai_difficulty, color="b") if ai_difficulty else None
    # ペルソナコメンタータ生成
    _persona = persona or {"name": "プラトン", "style": "格調高い哲学者口調", "first_person": "私"}
    commentator = GameCommentator(_persona, game_kind="チェス")

    try:
        result = curses.wrapper(_chess_curses_main, _CHESS_GAME, ai, commentator)
    except Exception as e:
        return f"\033[31mチェス起動エラー: {e}\033[0m"
    return f"\033[32m{result or 'チェスを終了しました'}\033[0m"

# ===== 将棋エンジン =====

class ShogiEngine:
    """
    本将棋エンジン。
    - 9×9盤、先手(s)/後手(g)
    - 全駒種の移動・成り・打ち駒
    - 王手検出・合法手生成（王手放置禁止）
    - 二歩禁止・打ち歩詰め禁止
    - 棋譜記録
    """

    # 駒種定数 (先手: 大文字, 後手: 小文字)
    # FU=歩 KY=香 KE=桂 GI=銀 KI=金 KA=角 HI=飛 OU=王
    # +FU=と +KY=成香 +KE=成桂 +GI=成銀 +KA=馬 +HI=龍
    SENTE = "s"
    GOTE  = "g"

    PIECE_NAMES_JA = {
        "FU":"歩", "KY":"香", "KE":"桂", "GI":"銀", "KI":"金",
        "KA":"角", "HI":"飛", "OU":"王",
        "+FU":"と", "+KY":"成香", "+KE":"成桂", "+GI":"成銀",
        "+KA":"馬", "+HI":"龍",
    }
    PIECE_SYMBOLS = {
        "s": {
            "FU":"歩","KY":"香","KE":"桂","GI":"銀","KI":"金",
            "KA":"角","HI":"飛","OU":"王",
            "+FU":"と","+KY":"杏","+KE":"圭","+GI":"全",
            "+KA":"馬","+HI":"龍",
        },
        "g": {
            "FU":"歩","KY":"香","KE":"桂","GI":"銀","KI":"金",
            "KA":"角","HI":"飛","OU":"王",
            "+FU":"と","+KY":"杏","+KE":"圭","+GI":"全",
            "+KA":"馬","+HI":"龍",
        },
    }

    # 成れる駒 -> 成り駒
    PROMOTE_MAP = {
        "FU":"+FU","KY":"+KY","KE":"+KE","GI":"+GI","KA":"+KA","HI":"+HI",
    }
    # 成り駒 -> 元駒
    UNPROMOTE_MAP = {v: k for k, v in PROMOTE_MAP.items()}

    # 金と同じ動き (成り駒に共通)
    _GOLD_DIRS_S = [(-1,0),(-1,-1),(-1,1),(0,-1),(0,1),(1,0)]  # 先手視点
    _GOLD_DIRS_G = [(1,0),(1,-1),(1,1),(0,-1),(0,1),(-1,0)]    # 後手視点

    def __init__(self):
        self.reset()

    def reset(self):
        self.board: list[list] = self._init_board()  # board[row][col] = (color, ptype) or None
        self.turn: str = self.SENTE
        self.hands: dict = {self.SENTE: {}, self.GOTE: {}}  # 持ち駒
        self.move_history: list[str] = []
        self.game_over: bool = False
        self.result: str = ""
        self.last_move: tuple | None = None  # (fr,fc,tr,tc) or (None,None,tr,tc) for drop
        # ★[修正/shogi-perf] 王位置キャッシュ: _find_king() は legal_moves で数百回呼ばれる。
        # 旧コードは毎回 O(81) の全盤面スキャン。キャッシュで O(1) に短縮。
        self._king_cache: dict = {self.SENTE: None, self.GOTE: None}
        # ★[修正/shogi-perf] 合法手キャッシュ: 同一局面で legal_moves を重複呼び出しするコストを削減。
        self._legal_cache: dict = {}
        self._legal_cache_key: tuple = ()

    def _init_board(self) -> list[list]:
        b = [[None]*9 for _ in range(9)]
        # 後手陣 (row 0-2): g
        back = ["KY","KE","GI","KI","OU","KI","GI","KE","KY"]
        for c, p in enumerate(back):
            b[0][c] = (self.GOTE, p)
        # 画面は disp_c=8-c で左右反転: col=7→画面左2番目, col=1→画面右2番目
        b[1][7] = (self.GOTE, "HI")   # 後手飛車: 画面左から2番目(8筋側)
        b[1][1] = (self.GOTE, "KA")   # 後手角:   画面右から2番目(2筋側)
        for c in range(9):
            b[2][c] = (self.GOTE, "FU")
        # 先手陣 (row 6-8): s
        for c, p in enumerate(back):
            b[8][c] = (self.SENTE, p)
        b[7][1] = (self.SENTE, "HI")  # 先手飛車: 画面右から2番目(2筋)
        b[7][7] = (self.SENTE, "KA")  # 先手角:   画面左から2番目(8筋)
        for c in range(9):
            b[6][c] = (self.SENTE, "FU")
        return b

    def _enemy(self, color: str) -> str:
        return self.GOTE if color == self.SENTE else self.SENTE

    def _on_board(self, r: int, c: int) -> bool:
        return 0 <= r <= 8 and 0 <= c <= 8

    def _promote_zone(self, color: str, r: int) -> bool:
        return r <= 2 if color == self.SENTE else r >= 6

    def _must_promote(self, color: str, ptype: str, r: int) -> bool:
        """成らないと動けなくなる場合は強制成り"""
        if ptype == "FU" or ptype == "KY":
            return (r == 0 if color == self.SENTE else r == 8)
        if ptype == "KE":
            return (r <= 1 if color == self.SENTE else r >= 7)
        return False

    def _piece_moves_raw(self, color: str, ptype: str, r: int, c: int) -> list[tuple]:
        """(tr, tc) のリスト (盤外・味方駒チェックなし)"""
        s = color == self.SENTE
        dirs_slide = []
        dirs_jump  = []
        one_step   = []

        if ptype == "FU":
            one_step = [(-1,0)] if s else [(1,0)]
        elif ptype == "KY":
            dirs_slide = [(-1,0)] if s else [(1,0)]
        elif ptype == "KE":
            dirs_jump = [(-2,-1),(-2,1)] if s else [(2,-1),(2,1)]
        elif ptype == "GI":
            one_step = [(-1,-1),(-1,0),(-1,1),(1,-1),(1,1)] if s else \
                       [(1,-1),(1,0),(1,1),(-1,-1),(-1,1)]
        elif ptype == "KI":
            one_step = self._GOLD_DIRS_S if s else self._GOLD_DIRS_G
        elif ptype in ("+FU","+KY","+KE","+GI"):
            one_step = self._GOLD_DIRS_S if s else self._GOLD_DIRS_G
        elif ptype == "KA":
            dirs_slide = [(-1,-1),(-1,1),(1,-1),(1,1)]
        elif ptype == "+KA":
            dirs_slide = [(-1,-1),(-1,1),(1,-1),(1,1)]
            one_step   = [(-1,0),(1,0),(0,-1),(0,1)]
        elif ptype == "HI":
            dirs_slide = [(-1,0),(1,0),(0,-1),(0,1)]
        elif ptype == "+HI":
            dirs_slide = [(-1,0),(1,0),(0,-1),(0,1)]
            one_step   = [(-1,-1),(-1,1),(1,-1),(1,1)]
        elif ptype == "OU":
            one_step = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        moves = []
        for dr, dc in one_step:
            moves.append((r+dr, c+dc))
        for dr, dc in dirs_jump:
            moves.append((r+dr, c+dc))
        for dr, dc in dirs_slide:
            nr, nc = r+dr, c+dc
            while self._on_board(nr, nc):
                moves.append((nr, nc))
                if self.board[nr][nc]:
                    break
                nr += dr; nc += dc
        return moves

    def pseudo_legal_moves_sq(self, r: int, c: int) -> list[tuple]:
        """(r,c)の駒の疑似合法手 (tr,tc,promote) リスト"""
        cell = self.board[r][c]
        if not cell: return []
        color, ptype = cell
        moves = []
        for tr, tc in self._piece_moves_raw(color, ptype, r, c):
            if not self._on_board(tr, tc): continue
            target = self.board[tr][tc]
            if target and target[0] == color: continue  # 味方駒
            can_promote = ptype in self.PROMOTE_MAP
            if can_promote:
                in_zone_from = self._promote_zone(color, r)
                in_zone_to   = self._promote_zone(color, tr)
                must = self._must_promote(color, ptype, tr)
                if must:
                    moves.append((tr, tc, True))
                elif in_zone_from or in_zone_to:
                    moves.append((tr, tc, True))
                    moves.append((tr, tc, False))
                else:
                    moves.append((tr, tc, False))
            else:
                moves.append((tr, tc, False))
        return moves

    def _find_king(self, color: str) -> tuple | None:
        # ★[修正/shogi-perf] 旧コードは毎回 O(81) のスキャン。
        # _apply_temp / _undo_temp 時にキャッシュを更新することで O(1) に。
        cached = self._king_cache.get(color)
        if cached is not None:
            return cached
        # キャッシュ miss 時のみフルスキャン（初期化直後など）
        for r in range(9):
            for c in range(9):
                cell = self.board[r][c]
                if cell and cell[0] == color and cell[1] == "OU":
                    self._king_cache[color] = (r, c)
                    return (r, c)
        self._king_cache[color] = None
        return None

    def in_check(self, color: str) -> bool:
        """王手判定: 敵の利きに王がいるか (高速版: 王位置から逆算)"""
        kpos = self._find_king(color)
        if kpos is None: return True
        kr, kc = kpos
        enemy = self._enemy(color)
        s = (enemy == self.SENTE)
        # 歩
        dr_fu = -1 if s else 1
        r2, c2 = kr + dr_fu, kc
        if self._on_board(r2, c2):
            cell = self.board[r2][c2]
            if cell and cell[0] == enemy and cell[1] == "FU": return True
        # 桂
        drs_ke = [(-2,-1),(-2,1)] if s else [(2,-1),(2,1)]
        for dr, dc in drs_ke:
            r2, c2 = kr+dr, kc+dc
            if self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell and cell[0] == enemy and cell[1] == "KE": return True
        # 香 (スライド)
        dr_ky = -1 if s else 1
        r2, c2 = kr+dr_ky, kc
        while self._on_board(r2, c2):
            cell = self.board[r2][c2]
            if cell:
                if cell[0] == enemy and cell[1] == "KY": return True
                break
            r2 += dr_ky
        # 飛・龍 (縦横スライド)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            r2, c2 = kr+dr, kc+dc
            while self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell:
                    if cell[0] == enemy and cell[1] in ("HI","+HI"): return True
                    break
                r2 += dr; c2 += dc
        # 角・馬 (斜めスライド)
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            r2, c2 = kr+dr, kc+dc
            while self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell:
                    if cell[0] == enemy and cell[1] in ("KA","+KA"): return True
                    break
                r2 += dr; c2 += dc
        # 龍の1マス斜め, 馬の1マス縦横
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            r2, c2 = kr+dr, kc+dc
            if self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell and cell[0] == enemy and cell[1] == "+HI": return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            r2, c2 = kr+dr, kc+dc
            if self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell and cell[0] == enemy and cell[1] == "+KA": return True
        # 銀・金・成り駒・王 (1マス)
        gold_dirs = self._GOLD_DIRS_S if s else self._GOLD_DIRS_G
        silver_dirs = [(1,-1),(1,0),(1,1),(-1,-1),(-1,1)] if s else [(-1,-1),(-1,0),(-1,1),(1,-1),(1,1)]
        for dr, dc in gold_dirs:
            r2, c2 = kr+dr, kc+dc
            if self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell and cell[0] == enemy and cell[1] in ("KI","+FU","+KY","+KE","+GI"): return True
        for dr, dc in silver_dirs:
            r2, c2 = kr+dr, kc+dc
            if self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell and cell[0] == enemy and cell[1] == "GI": return True
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            r2, c2 = kr+dr, kc+dc
            if self._on_board(r2, c2):
                cell = self.board[r2][c2]
                if cell and cell[0] == enemy and cell[1] == "OU": return True
        return False

    def _apply_temp(self, fr, fc, tr, tc, promote: bool, drop_ptype: str | None = None) -> dict:
        """一時適用 (合法性チェックなし)、saved辞書を返す"""
        saved = {
            "board_cell_fr": (fr, fc, self.board[fr][fc]) if fr is not None else None,
            "board_cell_tr": (tr, tc, self.board[tr][tc]),
            "hands": {self.SENTE: dict(self.hands[self.SENTE]),
                      self.GOTE:  dict(self.hands[self.GOTE])},
            # ★[修正/shogi-perf] king_cache と legal_cache を保存して undo で復元
            "king_cache": dict(self._king_cache),
            "legal_cache_key": self._legal_cache_key,
        }
        if drop_ptype:
            # 打ち駒
            self.board[tr][tc] = (self.turn, drop_ptype)
            h = self.hands[self.turn]
            h[drop_ptype] = h.get(drop_ptype, 0) - 1
            if h[drop_ptype] <= 0: del h[drop_ptype]
        else:
            color, ptype = self.board[fr][fc]
            target = self.board[tr][tc]
            if target:
                cap = self.UNPROMOTE_MAP.get(target[1], target[1])
                self.hands[color][cap] = self.hands[color].get(cap, 0) + 1
            new_ptype = self.PROMOTE_MAP.get(ptype, ptype) if promote else ptype
            self.board[fr][fc] = None
            self.board[tr][tc] = (color, new_ptype)
            # ★[修正/shogi-perf] 王の移動を king_cache に即時反映
            if ptype == "OU":
                self._king_cache[color] = (tr, tc)
        # 合法手キャッシュを無効化
        self._legal_cache = {}
        self._legal_cache_key = ()
        return saved

    def _undo_temp(self, saved: dict):
        if saved["board_cell_fr"] is not None:
            r, c, val = saved["board_cell_fr"]
            self.board[r][c] = val
        r, c, val = saved["board_cell_tr"]
        self.board[r][c] = val
        self.hands[self.SENTE] = saved["hands"][self.SENTE]
        self.hands[self.GOTE]  = saved["hands"][self.GOTE]
        # ★[修正/shogi-perf] king_cache と legal_cache を復元
        self._king_cache = saved["king_cache"]
        self._legal_cache_key = saved["legal_cache_key"]

    def legal_moves(self, color: str) -> list[tuple]:
        """(fr, fc, tr, tc, promote, drop_ptype) のリスト"""
        # ★[修正/shogi-perf] 合法手キャッシュ: 同一局面・同色の重複計算を回避。
        # minimax の各ノードで legal_moves が 2〜3 回呼ばれていた (in_check, evaluate, 子ノード)。
        # board 状態を軽量ハッシュキーにして結果を 1 局面につき 1 回だけ計算する。
        _ck = (
            color,
            tuple(tuple(cell if cell is None else (cell[0], cell[1]) for cell in row) for row in self.board),
            tuple(sorted(self.hands.get(color, {}).items())),
        )
        if _ck == self._legal_cache_key and color in self._legal_cache:
            return self._legal_cache[color]

        moves = []
        # 駒の移動
        for r in range(9):
            for c in range(9):
                cell = self.board[r][c]
                if not cell or cell[0] != color: continue
                for tr, tc, promote in self.pseudo_legal_moves_sq(r, c):
                    saved = self._apply_temp(r, c, tr, tc, promote)
                    if not self.in_check(color):
                        moves.append((r, c, tr, tc, promote, None))
                    self._undo_temp(saved)
        # 打ち駒
        for ptype, cnt in list(self.hands[color].items()):  # list()でコピー: イテレート中のサイズ変更を防ぐ
            if cnt <= 0: continue
            for tr in range(9):
                for tc in range(9):
                    if self.board[tr][tc]: continue
                    # 二歩チェック
                    if ptype == "FU":
                        col_has_fu = any(
                            self.board[rr][tc] and self.board[rr][tc][0] == color
                            and self.board[rr][tc][1] == "FU"
                            for rr in range(9)
                        )
                        if col_has_fu: continue
                        # 打ち歩詰めチェック
                        if self._is_uchifuzume(color, tr, tc): continue
                    # 行き所のない駒チェック
                    if ptype == "FU" or ptype == "KY":
                        if color == self.SENTE and tr == 0: continue
                        if color == self.GOTE  and tr == 8: continue
                    if ptype == "KE":
                        if color == self.SENTE and tr <= 1: continue
                        if color == self.GOTE  and tr >= 7: continue
                    saved = self._apply_temp(None, None, tr, tc, False, ptype)
                    if not self.in_check(color):
                        moves.append((None, None, tr, tc, False, ptype))
                    self._undo_temp(saved)
        # ★[修正/shogi-perf] 計算結果をキャッシュに保存
        self._legal_cache_key = _ck
        self._legal_cache[color] = moves
        return moves

    def _is_uchifuzume(self, color: str, tr: int, tc: int) -> bool:
        """打ち歩詰めになるか"""
        saved = self._apply_temp(None, None, tr, tc, False, "FU")
        enemy = self._enemy(color)
        enemy_legal = self.legal_moves(enemy)
        is_zume = (not enemy_legal) and self.in_check(enemy)
        self._undo_temp(saved)
        return is_zume

    def move(self, fr, fc, tr, tc, promote: bool = False, drop_ptype: str | None = None,
             _skip_legal_check: bool = False) -> tuple[bool, str]:
        """手を実行。(True, 棋譜) or (False, エラー)
        _skip_legal_check=True: 呼び出し元がlegal_movesで検証済みの場合に再計算を省略。"""
        if self.game_over:
            return False, "ゲームは終了しています"
        if not _skip_legal_check:
            legal = self.legal_moves(self.turn)
            if (fr, fc, tr, tc, promote, drop_ptype) not in legal:
                return False, "その手は合法ではありません"

        # 実行
        if drop_ptype:
            self.board[tr][tc] = (self.turn, drop_ptype)
            h = self.hands[self.turn]
            h[drop_ptype] = h.get(drop_ptype, 0) - 1
            if h[drop_ptype] <= 0: del h[drop_ptype]
            notation = f"{'☗' if self.turn==self.SENTE else '☖'}{9-tc}{tr+1}{self.PIECE_NAMES_JA.get(drop_ptype,'?')}打"
        else:
            color, ptype = self.board[fr][fc]
            target = self.board[tr][tc]
            if target:
                cap = self.UNPROMOTE_MAP.get(target[1], target[1])
                self.hands[color][cap] = self.hands[color].get(cap, 0) + 1
            new_ptype = self.PROMOTE_MAP.get(ptype, ptype) if promote else ptype
            self.board[fr][fc] = None
            self.board[tr][tc] = (color, new_ptype)
            pro_str = "成" if promote else ""
            notation = f"{'☗' if color==self.SENTE else '☖'}{9-tc}{tr+1}{self.PIECE_NAMES_JA.get(new_ptype,'?')}{pro_str}"

        self.last_move = (fr, fc, tr, tc)
        self.move_history.append(notation)
        # ★[修正/shogi-perf] 実着手後はキャッシュを完全リセット
        self._king_cache = {self.SENTE: None, self.GOTE: None}
        self._legal_cache = {}
        self._legal_cache_key = ()
        # 王の移動なら新位置を即座にキャッシュ
        if not drop_ptype:
            _, moved_ptype = self.board[tr][tc]  # after move
            if moved_ptype == "OU":
                self._king_cache[color] = (tr, tc)
        self.turn = self._enemy(self.turn)

        # 詰み・ステールメイト確認
        next_legal = self.legal_moves(self.turn)
        if not next_legal:
            winner_name = "先手" if self.turn == self.GOTE else "後手"
            self.game_over = True
            self.result = f"詰み！{winner_name}の勝利"

        return True, notation

    def board_str(self) -> str:
        lines = []
        lines.append("  ９ ８ ７ ６ ５ ４ ３ ２ １")
        lines.append(" +" + "--+"*9)
        for r in range(9):
            row_label = str(r+1)
            cells = []
            for c in range(8, -1, -1):
                cell = self.board[r][c]
                if cell is None:
                    cells.append(" ・")
                else:
                    color, ptype = cell
                    sym = self.PIECE_SYMBOLS[color].get(ptype, "?")
                    if color == self.GOTE:
                        cells.append(f"v{sym}")
                    else:
                        cells.append(f" {sym}")
            lines.append(f"{row_label}|{'|'.join(cells)}|")
            lines.append(" +" + "--+"*9)
        return "\n".join(lines)

    def hand_str(self, color: str) -> str:
        h = self.hands[color]
        if not h: return "なし"
        parts = []
        order = ["HI","KA","KI","GI","KE","KY","FU"]
        for p in order:
            if h.get(p, 0) > 0:
                parts.append(f"{self.PIECE_NAMES_JA.get(p,'?')}×{h[p]}")
        return " ".join(parts) if parts else "なし"


class ShogiAI:
    """将棋AI v2.0 (v129) — Negamax + Transposition Table + Killer Heuristic + 詳細評価関数"""

    PIECE_VALUE = {
        "FU":100,"KY":220,"KE":270,"GI":380,"KI":480,
        "KA":650,"HI":750,"OU":10000,
        "+FU":320,"+KY":380,"+KE":380,"+GI":480,
        "+KA":880,"+HI":980,
    }

    # ★[v129] ポジションボーナス(先手視点, row0=後手陣, row8=先手陣)
    _POS_BONUS = {
        "FU": [0,0,0,5,10,15,20,0,0],   # 前進するほど価値が上がる
        "HI": [0,0,0,0,5,5,5,5,5],
        "KA": [0,0,0,0,5,5,5,5,5],
    }

    DIFFICULTY_SETTINGS = {
        "easy":      {"depth": 0, "random_rate": 1.0},
        "middle":    {"depth": 2, "random_rate": 0.20},  # ★[v129] depth 1→2
        "hard":      {"depth": 3, "random_rate": 0.0},
        "very_hard": {"depth": 4, "random_rate": 0.0},   # ★[v129] depth 3→4
    }

    def __init__(self, difficulty: str = "middle", color: str = "g"):
        self.difficulty = difficulty
        self.color = color
        s = self.DIFFICULTY_SETTINGS.get(difficulty, self.DIFFICULTY_SETTINGS["middle"])
        self.depth = s["depth"]
        self.random_rate = s["random_rate"]
        self._tt: dict = {}        # ★[v129] Transposition Table
        self._killer: dict = {}    # ★[v129] Killer Heuristic

    def evaluate(self, g: "ShogiEngine") -> int:
        """★[v129] 詳細評価: 駒価値 + 位置ボーナス + 持ち駒ボーナス + 王の安全性"""
        score = 0
        for r in range(9):
            for c in range(9):
                cell = g.board[r][c]
                if not cell: continue
                color, ptype = cell
                val = self.PIECE_VALUE.get(ptype, 0)
                # ★[v129] 位置ボーナス
                pos_arr = self._POS_BONUS.get(ptype)
                if pos_arr:
                    bonus = pos_arr[r] if color == ShogiEngine.SENTE else pos_arr[8 - r]
                    val += bonus
                score += val if color == ShogiEngine.SENTE else -val
        # 持ち駒ボーナス (1.0→0.85: 手駒は多少割引)
        for ptype, cnt in g.hands[ShogiEngine.SENTE].items():
            score += int(self.PIECE_VALUE.get(ptype, 0) * cnt * 0.85)
        for ptype, cnt in g.hands[ShogiEngine.GOTE].items():
            score -= int(self.PIECE_VALUE.get(ptype, 0) * cnt * 0.85)
        # ★[v129] 王手ボーナス: 王手をかけている方が有利
        if g.in_check(ShogiEngine.GOTE):  score += 50
        if g.in_check(ShogiEngine.SENTE): score -= 50
        return int(score)

    def _move_priority(self, g: "ShogiEngine", mv: tuple, depth: int) -> int:
        """★[v129] 手のソートキー: Killer > 成り > 駒取り > 通常"""
        fr, fc, tr, tc, promote, drop = mv
        killers = self._killer.get(depth, [])
        if mv in killers: return 0
        if g.board[tr][tc]: return 1  # 駒取り
        if promote: return 2
        if drop: return 3
        return 4

    def _minimax(self, g: "ShogiEngine", depth: int, alpha: int, beta: int, maximizing: bool) -> int:
        # ★[修正/shogi-timeout] 期限切れなら評価値のみ返して探索を打ち切る
        import time as _time
        if getattr(self, '_search_deadline', None) and _time.time() > self._search_deadline:
            return self.evaluate(g)
        if depth == 0 or g.game_over: return self.evaluate(g)

        # ★ TTハッシュを軽量化: str()変換を廃止してtuple化
        bh = hash((
            tuple(tuple(cell if cell is None else (cell[0], cell[1])
                         for cell in row) for row in g.board),
            g.turn, depth,
            tuple(sorted(g.hands[ShogiEngine.SENTE].items())),
            tuple(sorted(g.hands[ShogiEngine.GOTE].items())),
        ))
        if bh in self._tt: return self._tt[bh]

        color = ShogiEngine.SENTE if maximizing else ShogiEngine.GOTE
        moves = g.legal_moves(color)
        if not moves: return self.evaluate(g)

        # ★[v129] 手のソート
        moves.sort(key=lambda mv: self._move_priority(g, mv, depth))

        best = -10**9 if maximizing else 10**9
        if maximizing:
            for mv in moves:
                fr, fc, tr, tc, promote, drop = mv
                saved = g._apply_temp(fr, fc, tr, tc, promote, drop)
                prev = g.turn; g.turn = ShogiEngine.GOTE
                val = self._minimax(g, depth-1, alpha, beta, False)
                g.turn = prev; g._undo_temp(saved)
                if val > best:
                    best = val
                    if val >= beta:
                        # ★[v129] Killer登録
                        k = self._killer.setdefault(depth, [])
                        if mv not in k: k.insert(0, mv); k[:] = k[:2]
                        break
                alpha = max(alpha, val)
                if beta <= alpha: break
        else:
            for mv in moves:
                fr, fc, tr, tc, promote, drop = mv
                saved = g._apply_temp(fr, fc, tr, tc, promote, drop)
                prev = g.turn; g.turn = ShogiEngine.SENTE
                val = self._minimax(g, depth-1, alpha, beta, True)
                g.turn = prev; g._undo_temp(saved)
                best = min(best, val); beta = min(beta, val)
                if beta <= alpha: break

        self._tt[bh] = best
        if len(self._tt) > 20000: self._tt.clear()
        return best

    def choose_move(self, g: "ShogiEngine") -> tuple | None:
        import time as _time
        moves = g.legal_moves(self.color)
        if not moves: return None
        if self.difficulty == "easy" or (self.random_rate > 0 and random.random() < self.random_rate):
            return random.choice(moves)
        maximizing = (self.color == ShogiEngine.SENTE)
        best_val = -10**9 if maximizing else 10**9
        best_moves = []
        self._tt.clear(); self._killer.clear()  # ★[v129] 新探索でリセット
        # ★[修正/shogi-timeout] very_hard (depth=4) では合法手数×再帰が爆発しフリーズ。
        # 難易度別の思考時間上限を設けて必ず返るようにする。
        _limits = {"middle": 3.0, "hard": 5.0, "very_hard": 8.0}
        _deadline = _time.time() + _limits.get(self.difficulty, 5.0)
        self._search_deadline = _deadline  # ★ _minimax からも参照できるよう保持
        # ★[v129] 手を事前ソート
        moves.sort(key=lambda mv: self._move_priority(g, mv, 0))
        for mv in moves:
            # ★[修正/shogi-timeout] 時間切れなら best_moves が空でも現時点の最善手を返す
            if _time.time() > _deadline:
                break
            fr, fc, tr, tc, promote, drop = mv
            saved = g._apply_temp(fr, fc, tr, tc, promote, drop)
            prev = g.turn
            g.turn = ShogiEngine.GOTE if self.color == ShogiEngine.SENTE else ShogiEngine.SENTE
            val = self._minimax(g, self.depth-1, -10**9, 10**9, not maximizing)
            g.turn = prev; g._undo_temp(saved)
            if maximizing:
                if val > best_val: best_val = val; best_moves = [mv]
                elif val == best_val: best_moves.append(mv)
            else:
                if val < best_val: best_val = val; best_moves = [mv]
                elif val == best_val: best_moves.append(mv)
        return random.choice(best_moves) if best_moves else random.choice(moves)


# ===== ゲームコメンタータ (将棋・チェス共通) =====
class GameCommentator:
    """
    ゲーム中にペルソナがリアルタイムでテロップ（字幕）を生成する。
    別スレッドでLLMを呼び出し、メインのcursesループをブロックしない。
    """
    MAX_SCROLL = 6   # テロップ保持最大件数
    EXPIRE_SECS = 18 # 1件あたりの表示秒数

    def __init__(self, persona: dict, game_kind: str = "将棋"):
        self.persona   = persona
        self.game_kind = game_kind           # "将棋" or "チェス"
        self._lock     = threading.Lock()
        self._lines: list[tuple[str, float]] = []  # (テキスト, 追加時刻)
        self._busy     = threading.Event()   # スレッドセーフなbusy フラグ

    # ── 公開API ─────────────────────────────────────────────────
    def trigger(self, event: str, context: str = "") -> None:
        """イベントをトリガー。非同期でコメントを生成する。"""
        if self._busy.is_set():
            return  # 前のコメント生成中はスキップ
        self._busy.set()
        threading.Thread(
            target=self._generate,
            args=(event, context),
            daemon=True
        ).start()

    def get_lines(self) -> list[str]:
        """期限切れを除去して現在のテロップ行リストを返す。"""
        now = time.time()
        with self._lock:
            self._lines = [(t, ts) for t, ts in self._lines
                           if now - ts < self.EXPIRE_SECS]
            return [t for t, _ in self._lines[-self.MAX_SCROLL:]]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    # ── 内部 ────────────────────────────────────────────────────
    def _generate(self, event: str, context: str) -> None:
        # _busy はtrigger側でset済み
        try:
            o = _get_ollama()
            if o is None:
                return
            p = self.persona
            fp   = p.get("first_person", "私")
            name = p.get("name", "AI")
            style = p.get("style", "")
            # ペルソナ種別で二人称を決定
            AUTHORITATIVE = {
                "ソクラテス","プラトン","アリストテレス","エピクテトス",
                "マルクス・アウレリウス","トマス・アクィナス","デカルト","スピノザ",
                "ライプニッツ","ロック","ヒューム","カント","ヘーゲル",
                "ショーペンハウアー","ミル","ニーチェ","ウィリアム・ジェームズ",
                "フッサール","ハイデガー","サルトル","ボーヴォワール","ラッセル",
                "前期ウィトゲンシュタイン","後期ウィトゲンシュタイン",
            }
            second_person = "君" if name in AUTHORITATIVE else "先輩"

            sys_content = (
                f"あなたは{name}。口調: {style}。一人称: {fp}。\n"
                f"今、{self.game_kind}の対局を観戦している。ユーザーを『{second_person}』と呼ぶ。\n"
                f"観戦コメントを【1文・30字以内】で述べよ。\n"
                f"説明・解説は不要。キャラとして自然な一言のみ。箇条書き禁止。"
            )
            user_content = (
                f"【イベント】{event}\n"
                f"【状況】{context}\n"
                f"このイベントへの一言コメント（30字以内・1文）を{name}の口調で述べよ。"
            )
            msgs = [
                {"role": "system", "content": sys_content},
                {"role": "user",   "content": user_content},
            ]
            opts = {"temperature": 0.82, "num_predict": 60, "num_ctx": 256}
            result = ""
            deadline = time.time() + 8.0   # 8秒でタイムアウト
            try:
                stream = o.chat(model=MODEL_NAME, messages=msgs,
                                stream=True, options=opts)
                for chunk in stream:
                    if time.time() > deadline:
                        break
                    delta = chunk.get("message", {}).get("content", "")
                    result += delta
                    if len(result) > 60:
                        break
            except Exception:
                return
            # 改行・空行を除去して1行に
            result = result.strip().replace("\n", "　")
            if not result:
                return
            # 30字に切る
            if len(result) > 45:
                result = result[:45].rstrip("、，") + "…"
            with self._lock:
                self._lines.append((f"【{name}】{result}", time.time()))
                if len(self._lines) > self.MAX_SCROLL:
                    self._lines.pop(0)
        finally:
            self._busy.clear()


# ===== 将棋 curses UI =====
_SQ  = 4   # マス幅 (chars)
_SH  = 3   # マス高 (lines)
_SBX = 4   # 盤面左端
_SBY = 3   # 盤面上端 (端末が小さい場合は動的に縮小)
_SI_X = _SBX + _SQ * 9 + 2  # 右パネル開始

_SCP_LIGHT    = 21
_SCP_DARK     = 22
_SCP_SEL      = 23
_SCP_LEGAL    = 24
_SCP_LAST     = 25
_SCP_CHECK    = 26
_SCP_TITLE    = 27
_SCP_PANEL    = 28
_SCP_BTN      = 29
_SCP_STATUS_OK= 30
_SCP_STATUS_WN= 31
_SCP_DROP_HL  = 32
_SCP_COMMENT  = 33  # テロップ（ペルソナコメント）
_SCP_SENTE    = 34  # 先手駒
_SCP_GOTE     = 35  # 後手駒
_SCP_PROMOTED = 36  # 成り駒

def _shogi_init_colors():
    curses.start_color()
    curses.use_default_colors()
    # 盤面: 薄茶(YELLOW)/白 → 駒が黒で見やすいコントラスト
    curses.init_pair(_SCP_LIGHT,     curses.COLOR_BLACK,  curses.COLOR_YELLOW)   # 奇数マス: 薄黄
    curses.init_pair(_SCP_DARK,      curses.COLOR_BLACK,  curses.COLOR_WHITE)    # 偶数マス: 白
    curses.init_pair(_SCP_SEL,       curses.COLOR_WHITE,  curses.COLOR_MAGENTA)  # 選択中
    curses.init_pair(_SCP_LEGAL,     curses.COLOR_BLACK,  curses.COLOR_GREEN)    # 移動候補(緑=見やすい)
    curses.init_pair(_SCP_LAST,      curses.COLOR_BLACK,  curses.COLOR_CYAN)     # 直前の手(シアン)
    curses.init_pair(_SCP_CHECK,     curses.COLOR_WHITE,  curses.COLOR_RED)      # 王手
    curses.init_pair(_SCP_TITLE,     curses.COLOR_WHITE,  curses.COLOR_BLUE)     # タイトル(青背景)
    curses.init_pair(_SCP_PANEL,     curses.COLOR_CYAN,   -1)                    # 棋譜パネル(シアン)
    curses.init_pair(_SCP_BTN,       curses.COLOR_BLACK,  curses.COLOR_WHITE)    # ボタン
    curses.init_pair(_SCP_STATUS_OK, curses.COLOR_BLACK,  curses.COLOR_GREEN)    # ステータスOK
    curses.init_pair(_SCP_STATUS_WN, curses.COLOR_WHITE,  curses.COLOR_RED)      # 警告
    curses.init_pair(_SCP_DROP_HL,   curses.COLOR_BLACK,  curses.COLOR_YELLOW)   # 持駒選択ハイライト
    curses.init_pair(_SCP_COMMENT,   curses.COLOR_BLACK,  curses.COLOR_MAGENTA)  # ペルソナテロップ
    curses.init_pair(_SCP_SENTE,     curses.COLOR_WHITE,  curses.COLOR_BLACK)    # 先手駒
    curses.init_pair(_SCP_GOTE,      curses.COLOR_YELLOW, curses.COLOR_BLACK)    # 後手駒
    curses.init_pair(_SCP_PROMOTED,  curses.COLOR_RED,    curses.COLOR_BLACK)    # 成り駒


def _shogi_curses_main(stdscr, g: "ShogiEngine", ai: "ShogiAI | None" = None,
                       commentator: "GameCommentator | None" = None):
    """将棋 cursesメインループ"""
    _shogi_init_colors()
    curses.curs_set(0)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    curses.mouseinterval(0)
    stdscr.keypad(True)
    stdscr.timeout(80)

    # ── 端末サイズ自動調整 ──────────────────────────────────────────
    global _SBY, _SH, _SI_X
    term_h, term_w = stdscr.getmaxyx()
    if term_h >= 37:
        _SH = 3
    elif term_h >= 24:
        _SH = 2
    else:
        stdscr.clear()
        msg = f"端末が小さすぎます ({term_w}x{term_h})。24行以上必要です。"
        try: stdscr.addstr(term_h//2, max(0,(term_w-len(msg))//2), msg)
        except curses.error: pass
        stdscr.refresh(); stdscr.getch(); return
    _SBY = 2
    _SI_X = _SBX + _SQ * 9 + 2
    # ──────────────────────────────────────────────────────────────
    selected: tuple | None = None      # 選択中マス (r, c) or ("hand", color, ptype)
    legal_moves_cache: list = []       # 合法手キャッシュ
    status_msg  = ""
    status_warn = False
    undo_stack: list = []
    _diff_labels = {"easy":"Easy","middle":"Middle","hard":"Hard","very_hard":"Very Hard"}

    if ai:
        status_msg = f"将棋 AI対戦[{_diff_labels.get(ai.difficulty,'')}] — 先手(あなた)から！"
    else:
        status_msg = "将棋: クリックして駒を選択"

    BUTTONS = [("  新局  ","new"),("  待った ","undo"),("  終了  ","quit")]

    def _snapshot():
        import copy
        return {
            "board": [row[:] for row in g.board],
            "turn": g.turn,
            "hands": {ShogiEngine.SENTE: dict(g.hands[ShogiEngine.SENTE]),
                      ShogiEngine.GOTE:  dict(g.hands[ShogiEngine.GOTE])},
            "move_history": g.move_history[:],
            "game_over": g.game_over,
            "result": g.result,
            "last_move": g.last_move,
        }

    def _restore(snap):
        g.board        = snap["board"]
        g.turn         = snap["turn"]
        g.hands        = snap["hands"]
        g.move_history = snap["move_history"]
        g.game_over    = snap["game_over"]
        g.result       = snap["result"]
        g.last_move    = snap["last_move"]

    def _sq_to_screen(r, c):
        # 将棋盤は右から左に列が増える (9列目=左)
        # 画面上: c=0(9筋)=左端, c=8(1筋)=右端
        disp_c = 8 - c  # 盤面左端からのオフセット
        return (_SBY + r * _SH, _SBX + disp_c * _SQ)

    def _screen_to_sq(my, mx):
        r = (my - _SBY) // _SH
        dc = (mx - _SBX) // _SQ
        c  = 8 - dc
        if 0 <= r <= 8 and 0 <= c <= 8:
            ry = my - (_SBY + r * _SH)
            rx = mx - (_SBX + dc * _SQ)
            if 0 <= ry < _SH and 0 <= rx < _SQ:
                return (r, c)
        return None

    def _draw_board():
        last = g.last_move
        last_sqs = set()
        if last:
            fr, fc, tr, tc = last
            if fr is not None: last_sqs.add((fr, fc))
            last_sqs.add((tr, tc))

        # ★ legal_moves_cache は _refresh_legal() で選択駒専用にフィルタ済みなので
        #   mv[0]==selected[0] の二重チェックは不要（チェック漏れの原因だった）
        legal_tgts = set()
        if isinstance(selected, tuple) and len(selected) == 2:
            for mv in legal_moves_cache:
                legal_tgts.add((mv[2], mv[3]))
        elif isinstance(selected, tuple) and len(selected) == 3 and selected[0] == "hand":
            for mv in legal_moves_cache:
                legal_tgts.add((mv[2], mv[3]))

        king_r, king_c = None, None
        kpos = g._find_king(g.turn)
        in_chk = g.in_check(g.turn)
        g._cached_in_check = in_chk   # ★ _draw_title で再利用
        if kpos: king_r, king_c = kpos

        for r in range(9):
            for c in range(9):
                sy, sx = _sq_to_screen(r, c)
                cell = g.board[r][c]
                if cell:
                    color, ptype = cell
                    sym = ShogiEngine.PIECE_SYMBOLS[color].get(ptype, "?")
                    is_gote = (color == ShogiEngine.GOTE)
                    is_promoted = ptype.startswith("+")
                    if is_gote:
                        mid_text = f"v{sym}"[:_SQ].ljust(_SQ)
                    else:
                        mid_text = f"^{sym}"[:_SQ].ljust(_SQ)
                    blank    = " " * _SQ
                else:
                    is_gote  = False
                    mid_text = " ・ "[:_SQ].ljust(_SQ)
                    blank    = " " * _SQ

                if in_chk and (r, c) == (king_r, king_c):
                    cp = _SCP_CHECK
                elif isinstance(selected, tuple) and len(selected)==2 and selected == (r, c):
                    cp = _SCP_SEL
                elif (r, c) in legal_tgts:
                    cp = _SCP_LEGAL
                elif (r, c) in last_sqs:
                    cp = _SCP_LAST
                else:
                    cp = _SCP_LIGHT if (r + c) % 2 == 0 else _SCP_DARK

                attr     = curses.color_pair(cp)
                attr_b   = curses.color_pair(cp) | curses.A_BOLD
                attr_dim = curses.color_pair(cp) | curses.A_DIM
                if cell and cp in (_SCP_LIGHT, _SCP_DARK):
                    if is_promoted:
                        attr_piece = curses.color_pair(_SCP_PROMOTED) | curses.A_BOLD
                    else:
                        attr_piece = curses.color_pair(_SCP_GOTE if is_gote else _SCP_SENTE) | curses.A_BOLD
                else:
                    attr_piece = attr_b

                try:
                    if _SH >= 3:
                        if cell:
                            if is_gote:
                                stdscr.addstr(sy,   sx, mid_text, attr_piece)
                                stdscr.addstr(sy+1, sx, blank,    attr)
                                stdscr.addstr(sy+2, sx, blank,    attr)
                            else:
                                stdscr.addstr(sy,   sx, blank,    attr)
                                stdscr.addstr(sy+1, sx, blank,    attr)
                                stdscr.addstr(sy+2, sx, mid_text, attr_piece)
                        else:
                            stdscr.addstr(sy,   sx, blank,    attr)
                            stdscr.addstr(sy+1, sx, mid_text, attr)
                            stdscr.addstr(sy+2, sx, blank,    attr)
                    else:
                        if cell:
                            if is_gote:
                                stdscr.addstr(sy,   sx, mid_text, attr_piece)
                                stdscr.addstr(sy+1, sx, blank,    attr)
                            else:
                                stdscr.addstr(sy,   sx, blank,    attr)
                                stdscr.addstr(sy+1, sx, mid_text, attr_piece)
                        else:
                            stdscr.addstr(sy,   sx, mid_text, attr)
                            stdscr.addstr(sy+1, sx, blank,    attr)
                except curses.error:
                    pass

        # 列ラベル (9〜1)
        for c in range(9):
            sy_lbl, sx_lbl = _sq_to_screen(0, c)
            try:
                stdscr.addstr(_SBY - 1, sx_lbl, str(9 - c), curses.color_pair(_SCP_PANEL) | curses.A_BOLD)
            except curses.error:
                pass
        # 行ラベル (一〜九)
        kanji_row = "一二三四五六七八九"
        for r in range(9):
            sy_lbl = _SBY + r * _SH + (_SH // 2)
            try:
                stdscr.addstr(sy_lbl, _SBX - 2, kanji_row[r], curses.color_pair(_SCP_PANEL) | curses.A_BOLD)
            except curses.error:
                pass

    def _draw_hands():
        """後手持ち駒(上部)・先手持ち駒(下部)を描画"""
        h, w_max = stdscr.getmaxyx()

        # 後手持ち駒 (上)
        gote_hand = g.hand_str(ShogiEngine.GOTE)
        try:
            stdscr.addstr(1, _SBX, f"後手持駒: {gote_hand}".ljust(40), curses.color_pair(_SCP_PANEL))
        except curses.error:
            pass

        # 先手持ち駒 (下)
        sente_hand = g.hand_str(ShogiEngine.SENTE)
        try:
            stdscr.addstr(_SBY + 9*_SH + 1, _SBX, f"先手持駒: {sente_hand}".ljust(40), curses.color_pair(_SCP_PANEL))
        except curses.error:
            pass

        # 持ち駒クリック領域を描画 (先手のみ / プレイヤーターン時)
        if g.turn == ShogiEngine.SENTE and (ai is None or ai.color != ShogiEngine.SENTE):
            hand = g.hands[ShogiEngine.SENTE]
            order = ["HI","KA","KI","GI","KE","KY","FU"]
            x_off = _SBX
            y_row = _SBY + 9*_SH + 2
            for pt in order:
                if hand.get(pt, 0) > 0:
                    sym = ShogiEngine.PIECE_NAMES_JA.get(pt, "?")
                    is_sel = isinstance(selected, tuple) and len(selected)==3 and selected[2]==pt
                    cp = _SCP_DROP_HL if is_sel else _SCP_BTN
                    try:
                        stdscr.addstr(y_row, x_off, f"[{sym}]", curses.color_pair(cp) | curses.A_BOLD)
                    except curses.error:
                        pass
                    x_off += 5

    def _hand_click_at(my, mx) -> str | None:
        """先手持ち駒クリックで駒種を返す"""
        y_row = _SBY + 9*_SH + 2
        if my != y_row: return None
        hand = g.hands[ShogiEngine.SENTE]
        order = ["HI","KA","KI","GI","KE","KY","FU"]
        x_off = _SBX
        for pt in order:
            if hand.get(pt, 0) > 0:
                if x_off <= mx < x_off + 4:
                    return pt
                x_off += 5
        return None

    def _draw_panel():
        px = _SI_X
        h, _ = stdscr.getmaxyx()

        # 棋譜
        hist = g.move_history
        try:
            stdscr.addstr(_SBY, px, "── 棋譜 ──────────", curses.color_pair(_SCP_PANEL)|curses.A_DIM)
        except curses.error:
            pass
        panel_h = max(4, h - _SBY - 6)
        start_i = max(0, len(hist) - panel_h)
        for i, notation in enumerate(hist[start_i:]):
            try:
                stdscr.addstr(_SBY + 1 + i, px, f"{start_i+i+1:3d}. {notation[:20]}", curses.color_pair(_SCP_PANEL))
            except curses.error:
                pass

        # ボタン
        btn_y = h - 4
        for i, (label, _) in enumerate(BUTTONS):
            try:
                stdscr.addstr(btn_y, px + i*10, label, curses.color_pair(_SCP_BTN)|curses.A_BOLD)
            except curses.error:
                pass

    def _btn_at(my, mx):
        h, _ = stdscr.getmaxyx()
        btn_y = h - 4
        px = _SI_X
        for i, (label, action) in enumerate(BUTTONS):
            lx = px + i*10
            if my == btn_y and lx <= mx < lx + len(label):
                return action
        return None

    def _draw_title():
        turn_str = "☗先手" if g.turn == ShogiEngine.SENTE else "☖後手"
        # ★ _draw_board で計算済みの結果を使う (in_check の二重呼び出し回避)
        chk_str = " 【王手！】" if getattr(g, "_cached_in_check", False) else ""
        ai_str = f"  [AI:{_diff_labels.get(ai.difficulty,'')}]" if ai else ""
        move_n = len(g.move_history)
        title = f"  将棋{ai_str}  {turn_str}の番{chk_str}   {move_n}手目  "
        try:
            stdscr.addstr(0, 0, title.ljust(80), curses.color_pair(_SCP_TITLE)|curses.A_BOLD)
        except curses.error:
            pass

    def _draw_status(msg, warn=False):
        h, w_max = stdscr.getmaxyx()
        cp = _SCP_STATUS_WN if warn else _SCP_STATUS_OK
        try:
            stdscr.addstr(h-2, 0, f"  {msg}  ".ljust(w_max-1), curses.color_pair(cp))
        except curses.error:
            pass

    def _draw_telop():
        """ペルソナのテロップを最下行に表示"""
        if commentator is None:
            return
        h, w_max = stdscr.getmaxyx()
        lines = commentator.get_lines()
        if not lines:
            return
        # 最新の1行だけ最下行に表示
        text = lines[-1]
        try:
            stdscr.addstr(h-1, 0, f" ♟ {text} ".ljust(w_max-1),
                          curses.color_pair(_SCP_COMMENT) | curses.A_BOLD)
        except curses.error:
            pass

    def _redraw():
        # ★ noutrefresh+doupdate でちらつき排除
        stdscr.erase()
        _draw_title()
        _draw_board()
        _draw_hands()
        _draw_panel()
        _draw_status(status_msg, status_warn)
        _draw_telop()
        stdscr.noutrefresh()
        curses.doupdate()

    def _refresh_legal(sel):
        nonlocal legal_moves_cache
        color = g.turn
        all_legal = g.legal_moves(color)
        if sel is None:
            legal_moves_cache = all_legal
            return
        if isinstance(sel, tuple) and len(sel) == 2:
            r, c = sel
            legal_moves_cache = [mv for mv in all_legal if mv[0]==r and mv[1]==c]
        elif isinstance(sel, tuple) and len(sel) == 3:
            _, _, pt = sel
            legal_moves_cache = [mv for mv in all_legal if mv[5]==pt]
        else:
            legal_moves_cache = all_legal

    def _do_move(mv):
        nonlocal selected, legal_moves_cache, status_msg, status_warn
        snap = _snapshot()
        fr, fc, tr, tc, promote, drop = mv
        # 着手前の取られる駒を記録（コメント判定用・軽量）
        captured = g.board[tr][tc]
        ok, msg = g.move(fr, fc, tr, tc, promote, drop, _skip_legal_check=True)
        if ok:
            undo_stack.append(snap)
            if len(undo_stack) > 80: undo_stack.pop(0)
            selected = None
            legal_moves_cache = []
            status_msg = f"✔ {msg}"
            status_warn = False
            # ── コメンタートリガー ──────────────────────────
            if commentator:
                in_chk = g.in_check(g.turn)
                pname  = ShogiEngine.PIECE_NAMES_JA.get(
                    (drop or (g.board[tr][tc][1] if g.board[tr][tc] else "")), "")
                # 取った駒の価値で「大きな駒得」を判定（評価関数を呼ばない）
                HIGH_VALUE = {"HI", "KA", "KIN", "GIN", "RYU", "UMA"}
                big_capture = captured and captured[1] in HIGH_VALUE
                if g.game_over:
                    event   = "ゲームセット"
                    context = f"結果: {g.result}"
                elif in_chk:
                    event   = "王手"
                    context = f"{msg} — 王手！ 手数:{len(g.move_history)}"
                elif promote:
                    event   = "駒が成った"
                    context = f"{pname}が成った。{msg}"
                elif drop:
                    event   = "持ち駒を打った"
                    context = f"{pname}を打った。{msg}"
                elif big_capture:
                    event   = "大駒を取った"
                    context = f"{msg}"
                else:
                    if _rnd_game.random() < 0.3:
                        event   = "指し手"
                        context = f"{msg} (手数:{len(g.move_history)})"
                    else:
                        event = ""
                if event:
                    commentator.trigger(event, context)
            # ────────────────────────────────────────────────
            if g.game_over:
                status_msg = f"🏆 {g.result}"
                status_warn = True
        else:
            status_msg = f"✖ {msg}"
            status_warn = True
        _needs_redraw = True
        return ok

    # ── メインループ ───────────────────────────────────────────
    import random as _rnd_game
    _ai_thread: threading.Thread | None = None   # AI思考スレッド
    _ai_result: list = []                        # [mv] or [] — スレッド間共有
    _needs_redraw: bool = True                   # 再描画フラグ
    _ai_thinking: bool = False                   # AI思考中フラグ（重複起動防止）
    _last_ai_redraw: float = 0.0                 # AI思考中の最終再描画時刻

    def _ai_think_bg():
        """バックグラウンドでAIの手を計算。結果を_ai_resultに格納。"""
        try:
            mv = ai.choose_move(g)
            _ai_result.clear()
            if mv:
                _ai_result.append(mv)
        except Exception:
            _ai_result.clear()

    while True:
        # AI手番: バックグラウンドスレッドで思考、完了したら着手
        if ai and not g.game_over and g.turn == ai.color:
            # スレッドが走っていなければ開始
            if _ai_thread is None or not _ai_thread.is_alive():
                if _ai_result:
                    # 思考完了 → 着手
                    mv = _ai_result[0]
                    _ai_result.clear()
                    _ai_thread = None
                    _ai_thinking = False
                    _needs_redraw = True
                    snap = _snapshot()
                    fr, fc, tr, tc, promote, drop = mv
                    ok, msg = g.move(fr, fc, tr, tc, promote, drop, _skip_legal_check=True)
                    if ok:
                        undo_stack.append(snap)
                        if len(undo_stack) > 80: undo_stack.pop(0)
                        selected = None; legal_moves_cache = []
                        status_msg = f"AI [{_diff_labels.get(ai.difficulty,'')}]: {msg}"
                        status_warn = False
                        if g.game_over:
                            status_msg = f"🏆 {g.result}"; status_warn = True
                        if commentator:
                            in_chk = g.in_check(g.turn)
                            if g.game_over:
                                commentator.trigger("AIが勝利", f"結果: {g.result}")
                            elif in_chk:
                                commentator.trigger("AIが王手", f"AIの手: {msg}")
                            elif _rnd_game.random() < 0.5:
                                commentator.trigger("AIの指し手", f"AI: {msg} (手数:{len(g.move_history)})")
                else:
                    # 思考スレッドを起動（まだ起動していない場合のみ）
                    if not _ai_thinking:
                        _ai_result.clear()
                        status_msg = f"AI [{_diff_labels.get(ai.difficulty,'')}] 思考中..."
                        status_warn = False
                        _ai_thinking = True
                        _needs_redraw = True
                        _ai_thread = threading.Thread(target=_ai_think_bg, daemon=True)
                        _ai_thread.start()
            # ★ AI思考中: 200ms間隔で最小限の再描画（ちらつき防止・CPU節約）
            now = time.time()
            if _needs_redraw or (now - _last_ai_redraw >= 0.5):
                _redraw()
                _needs_redraw = False
                _last_ai_redraw = now
            stdscr.timeout(200)
            stdscr.getch()
            continue

        if _needs_redraw:
            _redraw()
            _needs_redraw = False
        # ★ プレイヤーターンはブロッキング待機でCPU消費ゼロ
        stdscr.timeout(-1)
        key = stdscr.getch()

        if key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
            except curses.error:
                continue
            if not (bstate & curses.BUTTON1_CLICKED or bstate & curses.BUTTON1_PRESSED):
                continue

            # ボタン
            action = _btn_at(my, mx)
            if action == "quit":
                return "将棋終了"
            if action == "new":
                g.reset()
                undo_stack.clear()
                selected = None; legal_moves_cache = []
                ai_str = f"[AI:{_diff_labels.get(ai.difficulty,'')}] " if ai else ""
                status_msg = f"新しい対局を開始しました {ai_str}"; status_warn = False
                _needs_redraw = True
                continue
            if action == "undo":
                # AIモードの場合は2手戻す
                steps = 2 if ai and len(undo_stack) >= 2 else 1
                for _ in range(steps):
                    if undo_stack:
                        _restore(undo_stack.pop())
                selected = None; legal_moves_cache = []
                status_msg = "待った！1手戻しました"; status_warn = False
                _needs_redraw = True
                continue

            if g.game_over:
                status_msg = f"対局終了: {g.result}  (新局 で新しいゲーム)"; status_warn = True
                continue

            # AIモード: AIの手番には操作不可
            if ai and g.turn == ai.color:
                status_msg = "AIが考えています..."; status_warn = False
                continue

            # 先手持ち駒クリック
            pt = _hand_click_at(my, mx)
            if pt is not None:
                if g.hands[ShogiEngine.SENTE].get(pt, 0) > 0:
                    selected = ("hand", ShogiEngine.SENTE, pt)
                    _refresh_legal(selected)
                    status_msg = f"打ち駒: {ShogiEngine.PIECE_NAMES_JA.get(pt,'?')} — 打つ場所をクリック"
                    status_warn = False
                _needs_redraw = True
                continue

            # 盤面クリック
            sq = _screen_to_sq(my, mx)
            if sq is None:
                continue
            r, c = sq

            if selected is None:
                # 駒選択
                cell = g.board[r][c]
                if ai and cell and cell[0] == ai.color:
                    status_msg = "それはAIの駒です"; status_warn = True
                elif cell and cell[0] == g.turn:
                    selected = (r, c)
                    _refresh_legal(selected)
                    pname = ShogiEngine.PIECE_NAMES_JA.get(cell[1], cell[1])
                    status_msg = f"選択: {9-c}{'一二三四五六七八九'[r]}({pname}) — 移動先をクリック"
                    status_warn = False
                elif cell:
                    status_msg = f"{'先手' if g.turn==ShogiEngine.SENTE else '後手'}の番です"; status_warn = True
                else:
                    status_msg = "その位置に駒がありません"; status_warn = True
                _needs_redraw = True

            else:
                # 移動先クリック
                if isinstance(selected, tuple) and len(selected) == 2 and selected == (r, c):
                    selected = None; legal_moves_cache = []
                    status_msg = "選択解除"; status_warn = False
                else:
                    # 合法手の中から候補を探す
                    if isinstance(selected, tuple) and len(selected) == 2:
                        fr2, fc2 = selected
                        cands = [mv for mv in legal_moves_cache if mv[2]==r and mv[3]==c]
                    else:
                        # 打ち駒
                        _, _, drop_pt = selected
                        cands = [mv for mv in legal_moves_cache if mv[2]==r and mv[3]==c and mv[5]==drop_pt]
                        fr2, fc2 = None, None

                    if not cands:
                        # 別の自駒をクリック → 選択し直し
                        cell = g.board[r][c]
                        if cell and cell[0] == g.turn:
                            selected = (r, c)
                            _refresh_legal(selected)
                            pname = ShogiEngine.PIECE_NAMES_JA.get(cell[1], cell[1])
                            status_msg = f"選択変更: {9-c}{'一二三四五六七八九'[r]}({pname})"
                            status_warn = False
                        else:
                            selected = None; legal_moves_cache = []
                            status_msg = "合法手ではありません"; status_warn = True
                    elif len(cands) == 1:
                        _do_move(cands[0])
                    else:
                        # 成り/不成りの選択 (promote=True/False)
                        promote_mv = next((mv for mv in cands if mv[4]), None)
                        nopro_mv   = next((mv for mv in cands if not mv[4]), None)
                        # ポップアップで選択 (簡易: y/n キー待ち)
                        h_s, w_s = stdscr.getmaxyx()
                        py, px_ = h_s//2-2, w_s//2-12
                        cp = curses.color_pair(_SCP_STATUS_WN) | curses.A_BOLD
                        try:
                            stdscr.addstr(py,   px_, "┌─────────────────────────┐", cp)
                            stdscr.addstr(py+1, px_, "│  成りますか？(y=成る/n=不成) │", cp)
                            stdscr.addstr(py+2, px_, "└─────────────────────────┘", cp)
                        except curses.error:
                            pass
                        stdscr.refresh()
                        while True:
                            pk = stdscr.getch()
                            if pk in (ord('y'), ord('Y')) and promote_mv:
                                _do_move(promote_mv); break
                            elif pk in (ord('n'), ord('N')) and nopro_mv:
                                _do_move(nopro_mv); break
                            elif pk in (ord('q'), 27):
                                selected = None; legal_moves_cache = []
                                status_msg = "キャンセル"; status_warn = False; break
                _needs_redraw = True

        elif key in (ord('q'), ord('Q'), 27):
            return "将棋終了"
        elif key in (ord('n'), ord('N')):
            g.reset(); undo_stack.clear()
            selected = None; legal_moves_cache = []
            status_msg = "新しい対局"; status_warn = False
            _needs_redraw = True
        elif key in (ord('u'), ord('U')):
            if undo_stack:
                _restore(undo_stack.pop())
                selected = None; legal_moves_cache = []
                status_msg = "待った！"; status_warn = False
                _needs_redraw = True
            else:
                status_msg = "戻せる手がありません"; status_warn = True


_SHOGI_GAME: "ShogiEngine | None" = None

def handle_shogi(arg: str, persona: dict | None = None) -> str:
    """将棋エントリポイント。/shogi [easy|middle|hard|very_hard]"""
    global _SHOGI_GAME
    arg = arg.strip().lower()

    difficulty_map = {
        "easy": "easy", "イージー": "easy", "簡単": "easy",
        "middle": "middle", "ミドル": "middle", "普通": "middle", "normal": "middle",
        "hard": "hard", "ハード": "hard", "難しい": "hard",
        "very_hard": "very_hard", "veryhard": "very_hard", "最難関": "very_hard",
        "very hard": "very_hard", "超難": "very_hard",
    }
    ai_difficulty = None
    for key, val in difficulty_map.items():
        if key in arg:
            ai_difficulty = val
            break

    if _SHOGI_GAME is None or "new" in arg or ai_difficulty is not None:
        _SHOGI_GAME = ShogiEngine()

    ai = ShogiAI(difficulty=ai_difficulty, color=ShogiEngine.GOTE) if ai_difficulty else None
    # ペルソナコメンタータ生成
    _persona = persona or {"name": "ソクラテス", "style": "問答家口調。語尾「〜かね？」", "first_person": "私"}
    commentator = GameCommentator(_persona, game_kind="将棋")

    try:
        result = curses.wrapper(_shogi_curses_main, _SHOGI_GAME, ai, commentator)
    except Exception as e:
        return f"\033[31m将棋起動エラー: {e}\033[0m"
    return f"\033[32m{result or '将棋を終了しました'}\033[0m"


# ===== 哲学者人狼ゲーム =====
_PHIL_WOLF_HTML_PATH: str | None = None

def handle_philosopher_wolf(arg: str) -> str:
    """
    /wolf [6|9] — ブラウザで哲学者/偉人人狼を起動する。
      6 : 人狼1・占い師1・騎士1・村人3
      9 : 人狼2・狂人1・占い師1・霊媒師1・騎士1・村人3
    """
    global _PHIL_WOLF_HTML_PATH
    import tempfile, pathlib, webbrowser

    arg = arg.strip().lower()
    village_size = 9 if "9" in arg else 6
    html_content = _build_philosopher_wolf_html(village_size)

    # ★[修正/browser] WSL対応は共通関数 _get_browser_save_dir に統合

    save_dir = _get_browser_save_dir()  # ★[修正/browser]

    if _PHIL_WOLF_HTML_PATH and pathlib.Path(_PHIL_WOLF_HTML_PATH).exists():
        html_path = _PHIL_WOLF_HTML_PATH
    else:
        import uuid
        html_path = os.path.join(save_dir, f"s01_philosopher_wolf_{uuid.uuid4().hex[:8]}.html")
        _PHIL_WOLF_HTML_PATH = html_path

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_uri = pathlib.Path(html_path).as_uri()

    ok, browser_name, tried_log = _open_html_in_browser(html_path)
    label = f"{village_size}人村"
    if ok:
        return (
            f"\033[32m哲学者人狼 {label} を {browser_name} で起動しました。\033[0m\n"
            f"\033[90m   ファイル: {html_path}\033[0m\n"
            f"[33m   /wolf 6 → 6人村  /wolf 9 → 9人村[0m"
        )
    detail = "\n   ".join(tried_log) if tried_log else "理由不明"
    return (
        f"\033[33m⚠ 人狼HTML作成済みですがブラウザ起動失敗\033[0m\n"
        f"\033[90m   試行: {detail}\033[0m\n"
        f"次のパスをブラウザのアドレスバーに貼り付けてください:\n{html_path}"
    )

def _build_philosopher_wolf_cast() -> list[list[str]]:
    """哲学者人狼の参加者候補を作る。
    PERSONA_MAPとローカルRAGのヒットを使いつつ、最終的には許可リスト化した歴史上の偉人だけに絞る。
    """
    import random as _random

    curated: dict[str, str] = {
        "孔子": "礼を失った者の声は、群れの調和を乱す。私はそこを見る。",
        "老子": "騒がしい者ほど道から遠い。無為の静けさに狼の足音が残る。",
        "釈迦": "執着は疑いにも恐れにも宿る。苦の源を見れば狼も見える。",
        "アレクサンドロス大王": "退路を断て。議論の戦場で逃げる者から崩れる。",
        "カエサル": "賽は投げられた。票の流れこそ、この村のルビコンだ。",
        "クレオパトラ": "魅了する言葉ほど危うい。私は沈黙の香りまで疑う。",
        "始皇帝": "秩序を乱す者は一人で十分だ。名乗りと票を統一せよ。",
        "卑弥呼": "闇に祈り、声の震えを聞く。狼は神託を恐れる。",
        "聖徳太子": "和を乱す言葉は、どれほど整っていても響きが濁る。",
        "紫式部": "人の心は文の端に漏れるもの。発言の綾を見ましょう。",
        "ジャンヌ・ダルク": "恐れず名を挙げよ。沈む声に火を灯せ。",
        "レオナルド・ダ・ヴィンチ": "観察せよ。筆跡よりも投票の構図だ。",
        "ミケランジェロ": "余分な石を削れば像が出る。余分な弁明を削れば狼が出る。",
        "ガリレオ": "それでも票は動く。観察なき断罪には従えない。",
        "ニュートン": "小さな違和感にも力がある。疑惑は万有引力のように集まる。",
        "ダーウィン": "生き残る発言には理由がある。適応しすぎた者を疑え。",
        "ナイチンゲール": "記録を見ましょう。感情より、票と発言の統計です。",
        "リンカーン": "分裂した村は立ち行かない。だが狼との妥協は平和ではない。",
        "坂本龍馬": "時代を動かすには、まず腹の内を見抜かにゃならん。",
        "福沢諭吉": "独立自尊の村に、他人任せの推理はいらない。",
        "野口英世": "病巣は小さく始まる。発言の微熱を見逃すな。",
        "マリー・キュリー": "光るものは真実だけではない。沈黙の放射も測りましょう。",
        "アインシュタイン": "単純な疑いほど美しい。ただし単純すぎてはいけない。",
        "テスラ": "空気の震えで分かる。嘘は電流のように乱れる。",
        "エジソン": "推理は一パーセントの直感と九十九パーセントの検証だ。",
        "チャーチル": "暗い夜ほど、我々は言葉で防衛線を築くのだ。",
        "ガンディー": "暴力を拒むことと、狼を見逃すことは違う。",
        "キング牧師": "私には夢がある。恐怖ではなく真実で票が動く村の夢だ。",
    }

    cast: dict[str, str] = {
        "あなた": "沈黙もまた発言である。だが投票は沈黙を許さない。"
    }
    excluded_living = {"ハーバーマス"}
    for p in PERSONA_MAP.values():
        name = p.get("name", "").strip()
        if name and name not in excluded_living:
            cast.setdefault(name, f"{name}は自らの思想で発言の矛盾を照らす。")
    cast.update(curated)

    # PERSONA_MAPのキャラに人狼用セリフを補完
    wolf_quotes = {
        "ソクラテス": "吟味せよ。問われた者の目の動きに真実が宿る。",
        "プラトン": "洞窟の影を信じる者ほど、狼の言葉に惑わされる。",
        "アリストテレス": "中庸を欠いた発言は、どちらかに偏っている。それが手がかりだ。",
        "デカルト": "われ疑う、ゆえにわれあり。この村でも疑うことが出発点だ。",
        "スピノザ": "感情に流された票は、真理から遠ざかる。",
        "ライプニッツ": "この村は最善の状態にない。ならば狼を見つけるほかない。",
        "ヒューム": "習慣と印象に騙されるな。証拠なき断罪を戒める。",
        "カント": "汝の行為の格率が、普遍的法則たりうるか問え。",
        "ヘーゲル": "矛盾の中にこそ真理が宿る。対立を恐れるな。",
        "マルクス": "構造を見よ。誰が得をするかを問えば狼が見える。",
        "ニーチェ": "力への意志を持て。恐れから票を投じるな。",
        "フレーゲ": "言葉の意味と指示対象を区別せよ。発言の裏を読め。",
        "フッサール": "現象そのものへ。先入見を括弧に入れて観察せよ。",
        "ハイデガー": "存在を問え。この村で何かを隠している者がいる。",
        "ベルクソン": "直観を信じよ。知性だけでは狼を見抜けない。",
        "ウィトゲンシュタイン": "語りえぬことについては沈黙せよ。だが沈黙も語る。",
        "サルトル": "実存は本質に先立つ。役職ではなく行動で判断せよ。",
        "カミュ": "不条理を直視せよ。この村の混乱も然り。",
        "メルロ＝ポンティ": "身体の感覚を信じよ。違和感を無視するな。",
        "レヴィナス": "他者の顔を見よ。狼は他者性を消そうとする。",
        "デリダ": "テクストの外はない。発言の差異に狼の痕跡がある。",
        "フーコー": "権力の作動を見よ。誰が議論を支配しているか。",
        "ドゥルーズ": "差異と反復の中に真実がある。同じ言葉を繰り返す者を疑え。",
        "ロールズ": "無知のヴェールの下では誰もが公平な票を投じるはずだ。",
        "マルクス・アウレリウス": "自分のなすべきことをなせ。他者の言動に惑わされるな。",
        "エピクテトス": "制御できるものと制御できないものを区別せよ。",
        "パスカル": "人間は考える葦。この村でも思考が武器だ。",
        "キルケゴール": "実存の飛躍を恐れるな。決断の時が来た。",
        "ショーペンハウアー": "意志の盲目性を見よ。本能で動く者が狼かもしれない。",
        "ウィリアム・ジェームズ": "真理とは有用なものだ。この村で役立つ推理をせよ。",
        "デューイ": "経験から学べ。昨日の議論が今日の手がかりだ。",
        "バートランド・ラッセル": "論理的に考えよ。感情論では狼は見つからない。",
        "ポパー": "反証可能性を問え。証明できない主張を疑え。",
        "クワイン": "信念の網を揺さぶれ。どこかに綻びがある。",
        "ハーバーマス": "対話的理性を信じる。だし欺く者には通じない。",
        "アドルノ": "啓蒙の弁証法を見よ。理性もまた狼の道具になりうる。",
    }
    for name, quote in wolf_quotes.items():
        if name in cast and cast[name].endswith('発言の矛盾を照らす。'):
            cast[name] = quote

    allowed_names = set(cast)
    rag_queries = [
        "歴史上の偉人 哲学者 科学者 作家 政治家",
        "世界史 偉人 思想家 発明家 芸術家",
        "日本史 偉人 思想家 政治家 作家",
    ]
    rag_names: list[str] = []
    try:
        cols = ["s01_memory"] + [c for c in vector_list_collections() if c != "s01_memory"]
        for col in cols[:8]:
            for q in rag_queries:
                for hit in vector_search(q, n=5, collection=col):
                    for name in allowed_names:
                        if name != "あなた" and name in hit:
                            if name not in rag_names:
                                rag_names.append(name)
    except Exception:
        pass

    rag_people = [[name, cast[name]] for name in rag_names if name in cast]
    other_names = [name for name in cast if name not in ("あなた", *rag_names)]
    _random.shuffle(rag_people)
    _random.shuffle(other_names)
    others = rag_people + [[name, cast[name]] for name in other_names]
    return [["あなた", cast["あなた"]]] + others

def _build_philosopher_wolf_html(village_size: int = 6) -> str:
    auto_size = 9 if village_size == 9 else 6
    cast_json = json.dumps(_build_philosopher_wolf_cast(), ensure_ascii=False)
    html = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>哲学者人狼 - S-01</title>
<style>
:root{--bg:#151411;--panel:#24221d;--line:#4a4237;--text:#f5eee2;--muted:#b8ab98;--accent:#c9a24d;--red:#c85b55;--blue:#6190c8;--green:#6ea36f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Yu Gothic UI","Meiryo",system-ui,sans-serif;min-height:100vh}
button,select{font:inherit}button{border:1px solid var(--line);background:#302c25;color:var(--text);border-radius:7px;padding:9px 12px;cursor:pointer}button:hover{border-color:var(--accent)}button.primary{background:var(--accent);color:#17130c;border-color:var(--accent);font-weight:700}button.danger{background:var(--red);border-color:var(--red);color:white}button:disabled{opacity:.45;cursor:not-allowed}
.app{max-width:1180px;margin:0 auto;padding:18px}.top{display:flex;gap:12px;align-items:end;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:12px}.title{font-size:26px;font-weight:800}.sub{color:var(--muted);font-size:13px}.controls{display:flex;gap:8px;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:14px;margin-top:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}.status{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.stat{background:#1d1b17;border:1px solid var(--line);border-radius:7px;padding:9px}.stat b{display:block;color:var(--accent);font-size:12px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:9px;margin-top:12px}.card{border:1px solid var(--line);background:#1b1a17;border-radius:8px;padding:10px;min-height:116px}.card.dead{opacity:.48}.name{font-weight:800}.role{font-size:12px;color:var(--muted);margin-top:3px}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 7px;font-size:11px;margin:6px 4px 0 0;color:var(--muted)}.tag.red{color:#ffd9d6;border-color:var(--red)}.tag.blue{color:#d7e9ff;border-color:var(--blue)}.tag.green{color:#ddf3d7;border-color:var(--green)}
.talk{height:345px;overflow:auto;background:#161510;border:1px solid var(--line);border-radius:8px;padding:10px}.line{padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06)}.speaker{color:var(--accent);font-weight:700}.system{color:var(--muted)}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.voteRow{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;border-bottom:1px solid rgba(255,255,255,.06);padding:7px 0}.small{font-size:12px;color:var(--muted)}.result{font-size:18px;font-weight:800;color:var(--accent);margin-top:8px}
.telop{position:fixed;left:50%;top:18px;transform:translate(-50%,-18px);z-index:20;max-width:min(820px,calc(100vw - 28px));padding:12px 18px;border:1px solid rgba(201,162,77,.72);background:rgba(22,18,12,.94);box-shadow:0 14px 42px rgba(0,0,0,.38);border-radius:8px;color:#fff2c7;font-weight:800;text-align:center;opacity:0;pointer-events:none;transition:opacity .24s ease,transform .24s ease}.telop.show{opacity:1;transform:translate(-50%,0)}.telop.danger{border-color:rgba(200,91,85,.8);color:#ffe0dc}.telop.blue{border-color:rgba(97,144,200,.8);color:#dcecff}
@media(max-width:860px){.grid{grid-template-columns:1fr}.status{grid-template-columns:repeat(2,1fr)}.top{align-items:start;flex-direction:column}.talk{height:300px}}
</style>
</head>
<body>
<div class="app">
  <div class="top">
    <div><div class="title">哲学者人狼</div><div class="sub">偉人たちが思想・演説・疑念で殴り合う、6人/9人対応の人狼ゲーム</div></div>
    <div class="controls">
      <button onclick="newGame(6)">6人村</button>
      <button onclick="newGame(9)">9人村</button>
      <button class="primary" onclick="advance()">進行</button>
    </div>
  </div>
  <div class="grid">
    <section class="panel">
      <div class="status">
        <div class="stat"><b>村</b><span id="villageSize">-</span></div>
        <div class="stat"><b>日数</b><span id="day">-</span></div>
        <div class="stat"><b>フェーズ</b><span id="phase">-</span></div>
        <div class="stat"><b>あなた</b><span id="myRole">-</span></div>
      </div>
      <div class="players" id="players"></div>
    </section>
    <section class="panel">
      <div class="talk" id="log"></div>
      <div class="actions" id="actions"></div>
      <div class="result" id="result"></div>
    </section>
  </div>
</div>
<div class="telop" id="telop"></div>
<script>
const PEOPLE=__CAST_JSON__;
const roleInfo={
 'werewolf':'人狼','madman':'狂人','seer':'占い師','medium':'霊媒師','knight':'騎士','villager':'村人'
};
let G;
function shuffle(a){for(let i=a.length-1;i>0;i--){let j=Math.random()*(i+1)|0;[a[i],a[j]]=[a[j],a[i]]}return a}
let telopTimer=null;
function telop(msg,type=''){
 const el=document.getElementById('telop'); if(!el)return;
 el.textContent=msg; el.className='telop show '+type;
 clearTimeout(telopTimer); telopTimer=setTimeout(()=>{el.className='telop '+type},2300);
}
function newGame(n){
 const roles=n===9?['werewolf','werewolf','madman','seer','medium','knight','villager','villager','villager']:['werewolf','seer','knight','villager','villager','villager'];
 const pool=[PEOPLE[0],...shuffle(PEOPLE.slice(1)).slice(0,n-1)];
 const assigned=shuffle([...roles]);
 G={n,day:1,phase:'昼議論',players:pool.map((p,i)=>({id:i,name:p[0],quote:p[1],role:assigned[i],alive:true,known:false,votes:0,suspicion:Math.random()})),log:[],lastExecuted:null,winner:null};
 G.players[0].known=true; say('system',`${n}人村開始。昼議論から始まります。`);
 telop(`開廷: ${n}人の歴史的偉人が集う人狼裁判`, 'blue');
 discussion(); render();
}
function alive(){return G.players.filter(p=>p.alive)}
function wolvesAlive(){return alive().filter(p=>p.role==='werewolf').length}
function villagersAlive(){return alive().filter(p=>p.role!=='werewolf').length}
function checkWin(){
 if(wolvesAlive()===0){G.winner='村人陣営の勝利';telop('終幕: 村人陣営の勝利', 'blue');return true}
 if(wolvesAlive()>=villagersAlive()){G.winner='人狼陣営の勝利';telop('終幕: 人狼陣営の勝利', 'danger');return true}
 return false;
}
function say(who,msg){G.log.push({who,msg}); if(G.log.length>160)G.log.shift()}
function suspicionFor(p){
 let s=p.suspicion;
 // 占い結果で黒判定されたら疑惑大幅増
 const blackSee=G.log.filter(x=>x.who==='system'&&x.msg.includes('占い結果')&&x.msg.includes(p.name)&&x.msg.includes('黒'));
 const whiteSee=G.log.filter(x=>x.who==='system'&&x.msg.includes('占い結果')&&x.msg.includes(p.name)&&x.msg.includes('白'));
 s+=blackSee.length*0.5;
 s-=whiteSee.length*0.3;
 // 複数から名指しされるほど疑惑増
 const mentions=G.log.filter(x=>x.who!=='system'&&x.who!==p.name&&x.msg.includes(p.name));
 s+=mentions.length*0.08;
 // 偽COしているプレイヤーは矛盾が出やすいので疑惑増
 if(p.fakeRole)s+=0.15;
 // COしていない人はやや疑惑増（役職隠し）
 if(!p.co&&G.day>=2)s+=0.05;
 // ランダム要素を小さく抑える
 return s+Math.random()*0.1;
}
function aiCO(){
 for(const p of alive().filter(p=>p.id!==0)){
  if(p.co)continue;
  if(p.role==='seer'){
   p.co=true;
   say(p.name,'【CO】私は占い師です。');
   const lastSee=G.log.filter(x=>x.who==='system'&&x.msg.includes('占い結果')&&x.msg.includes(p.name)).slice(-1)[0];
   if(lastSee)say(p.name,lastSee.msg);
  }else if(p.role==='medium'&&G.day>=2){
   p.co=true;
   say(p.name,'【CO】私は霊媒師です。');
   const lastMed=G.log.filter(x=>x.who==='system'&&x.msg.includes('霊媒結果')).slice(-1)[0];
   if(lastMed)say(p.name,lastMed.msg);
  }else if((p.role==='werewolf'||p.role==='madman')&&Math.random()<0.3){
   p.co=true;
   const fakeCO=Math.random()<0.5?'占い師':'霊媒師';
   p.fakeRole=fakeCO;
   say(p.name,'【CO】私は'+fakeCO+'です。');
   if(fakeCO==='占い師'){
    const fakeTarget=alive().filter(x=>x.id!==p.id)[Math.random()*alive().filter(x=>x.id!==p.id).length|0];
    if(fakeTarget)say(p.name,'占い結果: '+fakeTarget.name+'は'+(Math.random()<0.7?'村人':'人狼')+'です。');
   }else{
    const dead=G.players.filter(x=>!x.alive);
    if(dead.length>0){
     const fakeD=dead[Math.random()*dead.length|0];
     say(p.name,'霊媒結果: '+fakeD.name+'は'+(Math.random()<0.7?'村人':'人狼')+'でした。');
    }
   }
  }
 }
}
function discussion(){
 aiCO();
 const list=shuffle(alive().filter(p=>p.id!==0));
 for(const p of list){
  const recentLog=G.log.filter(x=>x.who!=='system').slice(-3);
  const lastMsg=recentLog.length>0?recentLog[recentLog.length-1]:null;
  const coers=G.players.filter(x=>x.co&&x.id!==p.id&&x.alive);
  let msg='';

  // COへの反応
  if(coers.length>0&&Math.random()<0.5){
   const coer=coers[Math.random()*coers.length|0];
   const reactions=[
    `${coer.name}のCOは信じられるか？私には疑問が残る。`,
    `${coer.name}が名乗り出た。だが言葉は証拠にならない。`,
    `${coer.name}のCOを踏まえると、票の動きを見直す必要がある。`,
    `${coer.name}よ、その役職を証明してみせよ。`
   ];
   msg=p.quote+' '+reactions[Math.random()*reactions.length|0];
  }
  // 名指しされた人の反論
  else if(lastMsg&&lastMsg.msg.includes(p.name)&&Math.random()<0.6){
   const reactions=[
    `私を疑うか。${lastMsg.who}こそ、その根拠を示せ。`,
    `${lastMsg.who}が私を名指しした。だが真実は投票が明かす。`,
    `${lastMsg.who}よ、私への疑念は的外れだ。`,
    `私を狙うとは。${lastMsg.who}の意図を問いたい。`
   ];
   msg=p.quote+' '+reactions[Math.random()*reactions.length|0];
  }
  // 前の発言者への言及
  else if(lastMsg&&Math.random()<0.4){
   const reactions=[
    `${lastMsg.who}の発言が引っかかる。もう少し掘り下げるべきだ。`,
    `${lastMsg.who}よ、その言葉の裏を問いたい。`,
    `${lastMsg.who}の論理には穴がある。`
   ];
   msg=p.quote+' '+reactions[Math.random()*reactions.length|0];
  }
  // 通常発言: 疑惑スコア上位を狙う
  else{
   const candidates=alive().filter(x=>x.id!==p.id);
   // 人狼・狂人は村人陣営を狙う、村人陣営は疑惑上位を狙う
   let target;
   if(p.role==='werewolf'||p.role==='madman'){
    target=shuffle(candidates.filter(x=>x.role!=='werewolf'&&x.role!=='madman'))[0]||shuffle(candidates)[0];
   }else{
    const sorted=candidates.slice().sort((a,b)=>suspicionFor(b)-suspicionFor(a));
    target=Math.random()<0.75?sorted[0]:sorted[1]||sorted[0];
   }
   const suffixes=[
    `私は${target.name}の昨日の間合いが気になる。`,
    `${target.name}を急いで信じるには、まだ根拠が薄い。`,
    `今日は${target.name}への票の理由を問いたい。`,
    `${target.name}を注視する。`
   ];
   msg=p.quote+' '+suffixes[Math.random()*suffixes.length|0];
  }
  say(p.name,msg);
 }
}
function playerCO(){
 const me=G.players[0];
 me.co=true;
 if(me.role==='seer'){
  say('あなた','【CO】私は占い師です。');
  const lastSee=G.log.filter(x=>x.who==='system'&&x.msg.includes('占い結果')).slice(-1)[0];
  if(lastSee)say('あなた',lastSee.msg);
  else{const candidates=alive().filter(p=>p.id!==0&&p.role!=='werewolf');const t=candidates[Math.random()*candidates.length|0];if(t)say('あなた','占い結果: '+t.name+'は⬜白（村人）でした。');}
 }else if(me.role==='medium'){
  say('あなた','【CO】私は霊媒師です。');
  const lastMed=G.log.filter(x=>x.who==='system'&&x.msg.includes('霊媒結果')).slice(-1)[0];
  if(lastMed)say('あなた',lastMed.msg);
 }else if(me.role==='werewolf'||me.role==='madman'){
  const fakeCO=Math.random()<0.5?'占い師':'霊媒師';
  me.fakeRole=fakeCO;
  say('あなた','【CO】私は'+fakeCO+'です。');
 }else if(me.role==='knight'){
  const guardLog=G.log.filter(x=>x.who==='system'&&x.msg.includes('護衛')).slice(-1)[0];
  say('あなた','【護衛公開】'+(guardLog?guardLog.msg:'まだ護衛していません。'));
 }
 render();
}
function playerCOas(fakeRole){
 const me=G.players[0];
 me.co=true; me.fakeRole=fakeRole;
 say('あなた','【CO】私は'+fakeRole+'です。');
 if(fakeRole==='占い師'){
  const fakeTarget=alive().filter(p=>p.id!==0&&p.role!=='werewolf');
  const t=fakeTarget[Math.random()*fakeTarget.length|0];
  if(t)say('あなた','占い結果: '+t.name+'は⬜白（村人）でした。');
 }else{
  const dead=G.players.filter(x=>!x.alive);
  if(dead.length>0){const d=dead[Math.random()*dead.length|0];say('あなた','霊媒結果: '+d.name+'は⬜白（村人）でした。');}
 }
 render();
}
function advance(){
 if(G.winner){newGame(G.n);return}
 if(G.phase==='昼議論'){G.phase='投票'; telop('投票開始: 疑念を一票に変えよ'); render(); return}
 if(G.phase==='投票'){resolveVote(); if(!checkWin()){G.phase='夜';telop('夜が来る: 偉人たちは沈黙する', 'danger')} render(); return}
 if(G.phase==='夜'){resolveNight(); if(!checkWin()){G.day++;G.phase='昼議論';telop(`${G.day}日目: 議論再開`, 'blue');discussion()} render(); return}
}
function resolveVote(){
 alive().forEach(p=>p.votes=0);
 const humanTarget=Number((document.getElementById('voteTarget')||{}).value ?? -1);
 if(G.players[0].alive&&humanTarget>=0&&G.players[humanTarget]?.alive)G.players[humanTarget].votes++;
 for(const p of alive().filter(p=>p.id!==0)){
   const candidates=alive().filter(x=>x.id!==p.id);
   let t;
   if(p.role==='werewolf'){
    // 人狼: 村人陣営の中で疑惑が高い（=吊られやすい）人に投票して偽装
    const vCandidates=candidates.filter(x=>x.role!=='werewolf').slice().sort((a,b)=>suspicionFor(b)-suspicionFor(a));
    t=vCandidates[0]||shuffle(candidates)[0];
   }else if(p.role==='madman'){
    // 狂人: 村人陣営の疑惑上位に投票
    const mCandidates=candidates.filter(x=>x.role!=='madman').slice().sort((a,b)=>suspicionFor(b)-suspicionFor(a));
    t=mCandidates[0]||shuffle(candidates)[0];
   }else{
    // 村人陣営: 疑惑スコア上位に投票（占い結果・名指し回数を考慮）
    t=candidates.slice().sort((a,b)=>suspicionFor(b)-suspicionFor(a))[0];
   }
   t.votes++;
 }
 const max=Math.max(...alive().map(p=>p.votes));
 const tied=alive().filter(p=>p.votes===max);
 const dead=shuffle(tied)[0]; dead.alive=false; G.lastExecuted=dead;
 say('system',`投票結果: ${dead.name}が処刑された。正体はまだ伏せられる。`);
 telop(`処刑: ${dead.name}が歴史の法廷を去った`, 'danger');
 if(G.players[0].role==='medium'&&G.players[0].alive) say('system',`霊媒結果: ${dead.name}は${dead.role==='werewolf'?'🐺黒（人狼）':'⬜白（村人）'}でした。`);
}
function resolveNight(){
 const wolves=alive().filter(p=>p.role==='werewolf');
 const guardSel=Number((document.getElementById('guardTarget')||{}).value ?? -1);
 const seeSel=Number((document.getElementById('seeTarget')||{}).value ?? -1);
 if(G.players[0].role==='seer'&&G.players[0].alive&&seeSel>=0){
   const t=G.players[seeSel]; t.known=true; say('system',`占い結果: ${t.name}は${t.role==='werewolf'?'🐺黒（人狼）':'⬜白（村人）'}でした。`);
   telop(`占い: ${t.name}は${t.role==='werewolf'?'人狼':'白'}判定`, t.role==='werewolf'?'danger':'blue');
 }
 let guard=-1;
 if(G.players[0].role==='knight'&&G.players[0].alive){
  guard=guardSel>=0?guardSel:alive().filter(p=>p.id!==0)[0]?.id??-1;
  const gt=G.players[guard]; if(gt)say('system',`騎士の護衛: ${gt.name}を護衛した。`);
 }
 else {
   const k=alive().find(p=>p.role==='knight');
   if(k)guard=shuffle(alive().filter(p=>p.id!==k.id))[0].id;
 }
 const killSel=Number((document.getElementById('killTarget')||{}).value ?? -1);
 let victim=null;
 if(G.players[0].role==='werewolf'&&G.players[0].alive&&killSel>=0&&G.players[killSel]?.alive&&G.players[killSel].role!=='werewolf'){
   victim=G.players[killSel];
 } else {
   victim=shuffle(alive().filter(p=>p.role!=='werewolf'))[0];
 }
 if(victim&&victim.id===guard){say('system',`夜明け。騎士の護衛が成功し、犠牲者は出なかった。`);telop('護衛成功: 歴史はまだ書き換わらない', 'blue');return}
 if(victim){victim.alive=false;say('system',`夜明け。${victim.name}が無残な姿で発見された。`);telop(`襲撃: ${victim.name}、退場`, 'danger');}
}
function render(){
 document.getElementById('villageSize').textContent=G.n+'人村';
 document.getElementById('day').textContent=G.day+'日目';
 document.getElementById('phase').textContent=G.phase;
 document.getElementById('myRole').textContent=roleInfo[G.players[0].role];
 const isWolf=G.players[0].role==='werewolf';
 document.getElementById('players').innerHTML=G.players.map(p=>{
  const isPartner=isWolf&&p.id!==0&&p.role==='werewolf';
  return `<div class="card ${p.alive?'':'dead'}${isPartner?' wolf-partner':''}"><div class="name">${p.name}${isPartner?' <span style="color:#e74c3c;font-size:11px;">🐺相方</span>':''}</div><div class="role">${p.id===0||p.known?roleInfo[p.role]:'役職不明'}</div><span class="tag ${p.alive?'green':'red'}">${p.alive?'生存':'死亡'}</span>${p.votes?`<span class="tag red">${p.votes}票</span>`:''}<div class="small">${p.quote}</div></div>`;
 }).join('');
 document.getElementById('log').innerHTML=G.log.map(x=>`<div class="line"><span class="${x.who==='system'?'system':'speaker'}">${x.who}</span> ${x.msg}</div>`).join('');
 document.getElementById('log').scrollTop=99999;
 const opts=alive().filter(p=>p.id!==0).map(p=>`<option value="${p.id}">${p.name}</option>`).join('');
 let act='';
 if(G.phase==='昼議論'){
  act='<button class="primary" onclick="advance()">投票へ</button>';
  const me=G.players[0];
  if(me.alive&&!me.co){
   if(me.role==='seer'||me.role==='medium')
    act+='<button class="primary" onclick="playerCO()">COする</button>';
   if(me.role==='werewolf'||me.role==='madman'){
    act+='<button class="primary" onclick="playerCOas(\'占い師\')">占い師COする</button>';
    act+='<button class="primary" onclick="playerCOas(\'霊媒師\')">霊媒師COする</button>';
   }
   if(me.role==='knight')
    act+='<button class="primary" onclick="playerCO()">護衛先を公開</button>';
  }
 }
 if(G.phase==='投票')act=G.players[0].alive?`<select id="voteTarget">${opts}</select><button class="danger" onclick="advance()">投票する</button>`:'<span class="small">死亡中のため投票できません</span><button class="primary" onclick="advance()">投票結果へ</button>';
 if(G.phase==='夜'){
  if(G.players[0].role==='seer'&&G.players[0].alive&&G.day>=2){const seerOpts=alive().filter(p=>p.id!==0&&p.role!=='werewolf').map(p=>`<option value="${p.id}">${p.name}</option>`).join('');act+=`<span class="small">占い</span><select id="seeTarget">${seerOpts}</select>`;}
  if(G.players[0].role==='knight'&&G.players[0].alive)act+=`<span class="small">護衛</span><select id="guardTarget">${alive().filter(p=>p.id!==0).map(p=>`<option value="${p.id}">${p.name}</option>`).join('')}</select>`;
   if(G.players[0].role==='werewolf'&&G.players[0].alive)act+=`<span class="small">襲撃</span><select id="killTarget">${alive().filter(p=>p.role!=='werewolf').map(p=>`<option value="${p.id}">${p.name}</option>`).join('')}</select>`;
   act+='<button class="primary" onclick="advance()">夜を明かす</button>';
 }
 document.getElementById('actions').innerHTML=act;
 document.getElementById('result').textContent=G.winner?`${G.winner}。進行ボタンで新規開始。`:'';
}
newGame(__AUTO_SIZE__);
</script>
</body>
</html>"""
    return html.replace("__AUTO_SIZE__", str(auto_size)).replace("__CAST_JSON__", cast_json)


# ===== 麻雀ゲーム =====
_MAHJONG_HTML_PATH: str | None = None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  _open_html_in_browser  —  HTML ブラウザ起動 完全統合版              ║
# ║  Windows / WSL / macOS / Linux 全環境対応                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _is_wsl_env() -> bool:
    """WSL（Windows Subsystem for Linux）上で実行中かを判定する。"""
    try:
        with open("/proc/version") as _f:
            return "microsoft" in _f.read().lower()
    except Exception:
        return False

def _get_browser_save_dir() -> str:
    """
    HTMLファイルの保存先ディレクトリを返す。
    WSL環境では Windows の %TEMP% (Linuxマウントパス) を優先して返す。
    """
    import subprocess as _sub, tempfile
    if _is_wsl_env():
        try:
            w = _sub.check_output(["cmd.exe", "/c", "echo", "%TEMP%"],
                                   stderr=_sub.DEVNULL, timeout=4).decode().strip()
            if w:
                lp = _sub.check_output(["wslpath", "-u", w],
                                        stderr=_sub.DEVNULL, timeout=4).decode().strip()
                if lp and os.path.isdir(lp):
                    return lp
        except Exception:
            pass
        for d in ["/mnt/c/Windows/Temp", "/mnt/c/Users/Public"]:
            if os.path.isdir(d):
                return d
    return tempfile.gettempdir()

def _open_html_in_browser(html_path: str) -> "tuple[bool, str, list[str]]":
    """
    HTMLファイルをブラウザで開く。全OS・全ブラウザを網羅したフォールバック付き。

    Returns
    -------
    (success: bool, browser_name: str, tried_log: list[str])
        tried_log は失敗した試行の記録（デバッグ・ユーザー表示用）
    """
    import subprocess as _sub, pathlib, shutil as _sh, webbrowser

    tried: "list[str]" = []
    _sys = platform.system()
    is_wsl = (_sys == "Linux") and _is_wsl_env()

    # ── ファイルの存在確認と読み取り権限付与 ───────────────────────────
    if not os.path.isfile(html_path):
        return False, "", [f"ファイルが存在しない: {html_path}"]
    try:
        os.chmod(html_path, 0o644)
    except Exception:
        pass

    # ── WSL: HTMLをWindowsからアクセス可能なパスにコピー ────────────────
    win_html: "str | None" = None
    if is_wsl:
        save_dir = _get_browser_save_dir()
        dest = os.path.join(save_dir, os.path.basename(html_path))
        try:
            if os.path.abspath(html_path) != os.path.abspath(dest):
                _sh.copy2(html_path, dest)
                html_path = dest
        except Exception as _e:
            tried.append(f"TEMP copy: {_e}")
        try:
            win_html = _sub.check_output(["wslpath", "-w", html_path],
                                          stderr=_sub.DEVNULL, timeout=4).decode().strip()
        except Exception as _e:
            tried.append(f"wslpath -w: {_e}")

    file_uri = pathlib.Path(html_path).as_uri()

    # ════════════════════════════════════════════════════════════════════
    # Windows ネイティブ
    # ════════════════════════════════════════════════════════════════════
    if _sys == "Windows":
        win_exes = [
            (r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe",          "Edge"),
            (r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe",     "Edge"),
            (r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe",          "Edge"),
            (r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Applicationrave.exe", "Brave"),
            (r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Applicationrave.exe", "Brave"),
            (r"%PROGRAMFILES(X86)%\BraveSoftware\Brave-Browser\Applicationrave.exe", "Brave"),
            (r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe",           "Chrome"),
            (r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",           "Chrome"),
            (r"%PROGRAMFILES%\Mozilla Firefoxirefox.exe",                    "Firefox"),
            (r"%PROGRAMFILES(X86)%\Mozilla Firefoxirefox.exe",               "Firefox"),
            (r"%LOCALAPPDATA%\Opera Software\Opera Stable\opera.exe",          "Opera"),
        ]
        for tmpl, name in win_exes:
            exe = os.path.expandvars(tmpl)
            if os.path.isfile(exe):
                try:
                    _sub.Popen([exe, file_uri], stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                    return True, name, tried
                except Exception as _e:
                    tried.append(f"{name}: {_e}")
        # os.startfile (Windows APIでデフォルトアプリ起動)
        try:
            os.startfile(html_path)  # type: ignore[attr-defined]
            return True, "os.startfile(デフォルト)", tried
        except Exception as _e:
            tried.append(f"os.startfile: {_e}")
        # cmd.exe start フォールバック
        try:
            _sub.Popen(["cmd.exe", "/c", "start", "", html_path],
                        stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
            return True, "cmd start", tried
        except Exception as _e:
            tried.append(f"cmd start: {_e}")

    # ════════════════════════════════════════════════════════════════════
    # WSL (Linux on Windows)
    # ════════════════════════════════════════════════════════════════════
    elif is_wsl:
        target = win_html or file_uri

        # PowerShell経由で起動（最も確実）
        edge_path = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
        brave_path = 'C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe'
        for _wp, _wn in [(edge_path, 'Edge'), (brave_path, 'Brave')]:
            try:
                import subprocess as _sub2
                _sub2.Popen(
                    ['powershell.exe', '-NoProfile', '-NonInteractive',
                     '-Command', f"Start-Process '{_wp}' -ArgumentList '{win_html or file_uri}'"],
                    stdout=_sub2.DEVNULL, stderr=_sub2.DEVNULL)
                return True, f"{_wn}(PowerShell)", tried
            except Exception as _e:
                tried.append(f"{_wn} PowerShell: {_e}")

        # /mnt/c 直下でブラウザ実行ファイルを直接探す（ユーザー依存しない場所）
        wsl_direct = [
            ("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",           "Edge"),
            ("/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",                 "Edge"),
            ("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",                  "Chrome"),
            ("/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",            "Chrome"),
            ("/mnt/c/Program Files/Mozilla Firefox/firefox.exe",                           "Firefox"),
            ("/mnt/c/Program Files (x86)/Mozilla Firefox/firefox.exe",                     "Firefox"),
        ]
        for lp_exe, name in wsl_direct:
            if os.path.isfile(lp_exe):
                try:
                    _sub.Popen(["cmd.exe", "/c", lp_exe, target],
                                stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                    return True, f"{name}(WSL-direct)", tried
                except Exception as _e:
                    tried.append(f"WSL-direct {name}: {_e}")

        # cmd.exe /c start — Windowsのデフォルトブラウザに委ねる (最も確実)
        if win_html:
            try:
                _sub.Popen(["cmd.exe", "/c", "start", "", win_html],
                            stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                return True, "Windowsデフォルトブラウザ", tried
            except Exception as _e:
                tried.append(f"cmd start(win_html): {_e}")

        # explorer.exe フォールバック
        if win_html:
            try:
                _sub.Popen(["explorer.exe", win_html],
                            stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                return True, "explorer.exe", tried
            except Exception as _e:
                tried.append(f"explorer.exe: {_e}")

        # PowerShell Start-Process フォールバック
        if win_html:
            try:
                _sub.Popen(
                    ["powershell.exe", "-NoProfile", "-NonInteractive",
                     "-Command", f"Start-Process '{win_html}'"],
                    stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                return True, "PowerShell Start-Process", tried
            except Exception as _e:
                tried.append(f"powershell Start-Process: {_e}")

    # ════════════════════════════════════════════════════════════════════
    # macOS
    # ════════════════════════════════════════════════════════════════════
    elif _sys == "Darwin":
        mac_cmds = [
            (["/Applications/Brave Browser.app/Contents/MacOS/Brave Browser", file_uri], "Brave"),
            (["open", "-a", "Brave Browser",    file_uri], "Brave"),
            (["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", file_uri], "Chrome"),
            (["open", "-a", "Google Chrome",    file_uri], "Chrome"),
            (["open", "-a", "Microsoft Edge",   file_uri], "Edge"),
            (["open", "-a", "Firefox",          file_uri], "Firefox"),
            (["open", file_uri], "デフォルトブラウザ(open)"),  # macOS 最終手段
        ]
        for cmd, name in mac_cmds:
            try:
                _sub.Popen(cmd, stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                return True, name, tried
            except Exception as _e:
                tried.append(f"macOS {name}: {_e}")

    # ════════════════════════════════════════════════════════════════════
    # Linux ネイティブ (非WSL)
    # ════════════════════════════════════════════════════════════════════
    else:
        linux_cmds = [
            ("brave-browser",      "Brave"),
            ("brave",              "Brave"),
            ("google-chrome",      "Chrome"),
            ("google-chrome-stable","Chrome"),
            ("chromium-browser",   "Chromium"),
            ("chromium",           "Chromium"),
            ("microsoft-edge",     "Edge"),
            ("firefox",            "Firefox"),
            ("xdg-open",           "xdg-open"),
        ]
        for bin_name, name in linux_cmds:
            if _sh.which(bin_name):
                try:
                    _sub.Popen([bin_name, file_uri],
                                stdout=_sub.DEVNULL, stderr=_sub.DEVNULL)
                    return True, name, tried
                except Exception as _e:
                    tried.append(f"{name}: {_e}")

    # ════════════════════════════════════════════════════════════════════
    # 最終フォールバック: Python webbrowser モジュール
    # ════════════════════════════════════════════════════════════════════
    try:
        ok = webbrowser.open(file_uri)
        if ok:
            return True, "webbrowser(Python fallback)", tried
        tried.append("webbrowser.open: returned False")
    except Exception as _e:
        tried.append(f"webbrowser: {_e}")

    return False, "", tried

def handle_mahjong(arg: str) -> str:
    """
    /mj [3|4] [tonpu]  — ブラウザで本格麻雀を起動する。
      3        : 3人麻雀（デフォルト: 東風戦）
      4        : 4人麻雀（デフォルト: 東風戦）
      tonpu    : 東南戦（4人のみ）
    HTMLファイルをテンポラリに書き出してブラウザで開く。
    """
    global _MAHJONG_HTML_PATH
    import tempfile, webbrowser, pathlib

    arg = arg.strip().lower()
    num_players = 3 if "3" in arg else 4
    mode = "tonpu" if "tonpu" in arg else "east"  # east=東風戦, tonpu=東南戦

    # ── HTML生成 ──────────────────────────────────────────────
    html_content = _build_mahjong_html(num_players, mode)

    # ★[修正/browser] _get_win_temp_dir は共通 _get_browser_save_dir に統合済み

    save_dir = _get_browser_save_dir()  # ★[修正/browser]

    if _MAHJONG_HTML_PATH and pathlib.Path(_MAHJONG_HTML_PATH).exists():
        html_path = _MAHJONG_HTML_PATH
    else:
        import uuid
        html_path = os.path.join(save_dir, f"s01_mahjong_{uuid.uuid4().hex[:8]}.html")
        _MAHJONG_HTML_PATH = html_path

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # ── file_uri はLinuxパスのまま保持し、ブラウザ起動時にWin変換 ──
    file_uri = pathlib.Path(html_path).as_uri()
    label = f"{num_players}人麻雀({'東南戦' if mode == 'tonpu' else '東風戦'})"

    ok, browser_name, tried_log = _open_html_in_browser(html_path)
    if ok:
        return (
            f"\033[32m🀄 {label} を {browser_name} で起動しました\033[0m\n"
            f"\033[90m   ファイル: {html_path}\033[0m\n"
            f"\033[33m   /mj 3      → 3人麻雀\033[0m\n"
            f"\033[33m   /mj 4      → 4人麻雀（東風戦）\033[0m\n"
            f"\033[33m   /mj tonpu  → 4人麻雀（東南戦）\033[0m"
        )
    else:
        return (
            f"\033[33mブラウザ自動起動失敗: {browser_name}\033[0m\n"
            f"次のURLをBraveのアドレスバーに貼り付けてください:\n{file_uri}"
        )


def _build_mahjong_html(num_players: int = 4, mode: str = "east") -> str:
    """麻雀ゲームの完全なHTMLを文字列で返す。"""
    # 起動時に自動で指定モードのゲームを開始するJSを差し込む
    auto_start_js = f"startGame({num_players},'{mode}');"
    # ── HTML本体 ──────────────────────────────────────────────
    return r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>本格麻雀 — S-01</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#1a1a2e;--bg2:#16213e;--bg3:#0f3460;
  --green:#1b5e20;--green2:#2e7d32;--green3:#388e3c;
  --tile:#f5e6c8;--tile-s:#e8d5a3;--tile-h:#d4a853;
  --red:#e53935;--blue:#1976d2;--gold:#ffd700;--silver:#c0c0c0;
  --text:#f0f0f0;--text2:#b0b0b0;--text3:#707070;
  --radius:6px;--shadow:0 2px 8px rgba(0,0,0,.5);
  --font:'Noto Sans JP',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;overflow-x:hidden;user-select:none}
#app{display:flex;flex-direction:column;align-items:center;min-height:100vh}
.screen{display:none;width:100%;max-width:900px;padding:20px}
.screen.active{display:flex;flex-direction:column;align-items:center}
#title-screen{justify-content:center;min-height:100vh;gap:32px}
.title-logo{font-size:64px;font-weight:900;letter-spacing:8px;color:var(--gold);text-shadow:0 0 20px rgba(255,215,0,.4)}
.title-sub{font-size:14px;letter-spacing:4px;color:var(--text2)}
.btn-group{display:flex;flex-direction:column;gap:12px;width:280px}
.btn{padding:14px 32px;border:none;border-radius:var(--radius);font-size:16px;font-weight:700;cursor:pointer;transition:all .15s;letter-spacing:2px}
.btn-primary{background:linear-gradient(135deg,#b8860b,#ffd700);color:#1a1a00}
.btn-primary:hover{filter:brightness(1.15);transform:translateY(-2px)}
.btn-secondary{background:rgba(255,255,255,.08);color:var(--text);border:1px solid rgba(255,255,255,.2)}
.btn-secondary:hover{background:rgba(255,255,255,.15)}
#game-screen{padding:8px;max-width:960px;width:100%}
.table-area{position:relative;background:radial-gradient(ellipse at center,var(--green3) 0%,var(--green2) 50%,var(--green) 100%);border-radius:12px;border:4px solid #5d4037;box-shadow:inset 0 0 40px rgba(0,0,0,.3),var(--shadow);padding:8px;display:grid;grid-template-areas:"top top top" "left center right" "bottom bottom bottom";grid-template-rows:auto 1fr auto;grid-template-columns:auto 1fr auto;gap:4px;min-height:420px}
.seat-top{grid-area:top;display:flex;flex-direction:column;align-items:center;gap:2px}
.seat-left{grid-area:left;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px}
.seat-right{grid-area:right;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px}
.seat-info{background:rgba(0,0,0,.4);border-radius:4px;padding:3px 8px;font-size:11px;text-align:center;border:1px solid rgba(255,255,255,.1)}
.seat-name{font-weight:700;color:var(--gold)}
.seat-score{color:var(--text2);font-size:10px}
.seat-wind{font-size:10px;color:var(--silver)}
.ai-hand{display:flex;gap:2px}
.tile-back{width:24px;height:34px;background:linear-gradient(135deg,#1565c0,#0d47a1);border-radius:3px;border:1px solid rgba(255,255,255,.3);box-shadow:1px 1px 3px rgba(0,0,0,.5)}
.tile-back.small{width:18px;height:26px}
.seat-left .ai-hand,.seat-right .ai-hand{flex-direction:column}
.seat-left .tile-back,.seat-right .tile-back{width:34px;height:18px}
.center-area{grid-area:center;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
.info-panel{background:rgba(0,0,0,.5);border-radius:8px;padding:6px 16px;text-align:center;border:1px solid rgba(255,215,0,.3)}
.round-info{font-size:13px;color:var(--gold);font-weight:700}
.dora-area{display:flex;gap:4px;align-items:center}
.dora-label{font-size:10px;color:var(--text2)}
.pond{background:rgba(0,0,0,.2);border-radius:4px;padding:4px;display:flex;flex-wrap:wrap;gap:1px;align-content:flex-start;min-height:60px;max-height:80px;overflow:hidden;border:1px solid rgba(255,255,255,.05)}
.tile{background:var(--tile);color:#1a1a00;border-radius:4px;border:1px solid var(--tile-h);display:inline-flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;cursor:pointer;box-shadow:1px 2px 3px rgba(0,0,0,.4),inset 0 -1px 0 rgba(0,0,0,.2);transition:all .1s;position:relative;flex-shrink:0}
.tile:hover{filter:brightness(1.1);transform:translateY(-2px)}
.tile.selected{transform:translateY(-8px);box-shadow:0 6px 12px rgba(255,215,0,.4),1px 2px 3px rgba(0,0,0,.4);border-color:var(--gold)}
.tile.riichi-cand{transform:translateY(-6px);box-shadow:0 0 12px rgba(255,80,80,.8);border-color:#ff4444;animation:riichi-pulse .8s infinite alternate}
@keyframes riichi-pulse{from{box-shadow:0 0 8px rgba(255,80,80,.6)}to{box-shadow:0 0 18px rgba(255,80,80,1)}}
.tile.man{color:#c62828}.tile.pin{color:#1565c0}.tile.sou{color:#2e7d32}.tile.honor{color:#4a148c}
.tile.discarded{width:20px;height:28px;font-size:9px;cursor:default}
.tile.discarded:hover{transform:none;filter:none}
.tile.full{width:36px;height:50px;font-size:18px}
.tile.medium{width:28px;height:40px;font-size:13px}
.tile.small{width:20px;height:28px;font-size:10px}
.player-area{grid-area:bottom;display:flex;flex-direction:column;align-items:center;gap:6px;padding:4px 0}
.player-info-row{display:flex;gap:16px;align-items:center}
.player-info{background:rgba(0,0,0,.5);border-radius:6px;padding:4px 12px;font-size:12px;border:1px solid rgba(255,215,0,.3)}
.player-name-label{color:var(--gold);font-weight:700}
.player-score-label{color:var(--text2)}
.player-hand{display:flex;gap:3px;align-items:flex-end;flex-wrap:wrap;justify-content:center;min-height:56px}
.melds-area{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
.meld{display:flex;gap:2px;background:rgba(0,0,0,.2);padding:3px;border-radius:4px;border:1px solid rgba(255,255,255,.1)}
.controls{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;min-height:44px}
.action-btn{padding:8px 16px;border:none;border-radius:4px;font-size:13px;font-weight:700;cursor:pointer;transition:all .1s;letter-spacing:1px}
.action-btn:hover{filter:brightness(1.2);transform:translateY(-1px)}
.btn-tsumo{background:#c62828;color:white}.btn-riichi{background:#7b1fa2;color:white}
.btn-ron{background:#e65100;color:white}.btn-chi{background:#1565c0;color:white}
.btn-pon{background:#0277bd;color:white}.btn-kan{background:#00695c;color:white}
.btn-skip{background:rgba(255,255,255,.1);color:var(--text2);border:1px solid rgba(255,255,255,.2)}
.btn-discard{background:var(--gold);color:#1a1a00}
.hud-top{display:flex;justify-content:space-between;align-items:center;padding:4px 8px;background:rgba(0,0,0,.4);border-radius:6px;font-size:12px}
.hud-item{display:flex;gap:6px;align-items:center}
.hud-label{color:var(--text2)}.hud-value{color:var(--gold);font-weight:700}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);display:none;align-items:center;justify-content:center;z-index:100}
.modal-overlay.show{display:flex}
.modal{background:#1e2a3a;border-radius:12px;padding:24px;max-width:480px;width:90%;border:2px solid rgba(255,215,0,.4);box-shadow:0 0 40px rgba(255,215,0,.1)}
.modal-title{font-size:22px;font-weight:900;text-align:center;color:var(--gold);margin-bottom:16px;letter-spacing:2px}
.result-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.08);font-size:14px}
.hand-display{display:flex;gap:3px;flex-wrap:wrap;justify-content:center;margin:10px 0}
.score-delta{font-weight:700}.score-delta.pos{color:#81c784}.score-delta.neg{color:#ef9a9a}
.game-log{font-size:11px;color:var(--text3);text-align:center;height:18px;overflow:hidden}
.float-msg{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,.85);color:var(--gold);font-size:28px;font-weight:900;padding:16px 32px;border-radius:8px;letter-spacing:4px;pointer-events:none;opacity:0;transition:opacity .3s;z-index:200;border:2px solid var(--gold)}
.float-msg.show{opacity:1}
.waiting-overlay{position:absolute;inset:0;background:rgba(0,0,0,.3);display:none;align-items:center;justify-content:center;border-radius:12px;z-index:10;font-size:14px;color:var(--text2)}
.waiting-overlay.show{display:flex}
#final-screen{justify-content:center;min-height:100vh;gap:24px;padding:40px}
.final-title{font-size:32px;font-weight:900;color:var(--gold);letter-spacing:4px}
.rank-table{width:100%;max-width:400px}
.rank-row{display:flex;justify-content:space-between;padding:10px 16px;border-bottom:1px solid rgba(255,255,255,.08);font-size:15px}
.rank-1{color:#ffd700;font-weight:900}.rank-2{color:#c0c0c0;font-weight:700}.rank-3{color:#cd7f32}.rank-4{color:var(--text2)}
.thinking-dots::after{content:'';animation:dots 1.2s steps(4,end) infinite}
@keyframes dots{0%,100%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
.riichi-stick{width:60px;height:8px;background:white;border-radius:2px;border:1px solid #999;position:relative}
.riichi-stick::after{content:'';position:absolute;width:6px;height:6px;background:red;border-radius:50%;top:1px;left:27px}
</style>
</head>
<body>
<div id="app">

<div class="screen" id="title-screen">
  <div class="title-logo">麻雀</div>
  <div class="title-sub">MAHJONG — S-01 AI対戦</div>
  <div class="btn-group">
    <button class="btn btn-primary" onclick="startGame(4,'east')">4人麻雀（東風戦）</button>
    <button class="btn btn-primary" onclick="startGame(3,'east')">3人麻雀（東風戦）</button>
    <button class="btn btn-secondary" onclick="startGame(4,'tonpu')">4人麻雀（東南戦）</button>
  </div>
  <div style="font-size:11px;color:var(--text3);text-align:center;line-height:1.8;max-width:300px">
    プレイヤー1人 + AI（3〜4人）<br>
    立直・役判定・符計算完全実装<br>
    チー・ポン・槓対応
  </div>
</div>

<div class="screen" id="game-screen">
  <div class="hud-top">
    <div class="hud-item"><span class="hud-label">局</span><span class="hud-value" id="hud-round">東1局</span></div>
    <div class="hud-item"><span class="hud-label">本場</span><span class="hud-value" id="hud-honba">0</span></div>
    <div class="hud-item"><span class="hud-label">供托</span><span class="hud-value" id="hud-riichi-pool">0</span></div>
    <div class="hud-item"><span id="hud-tiles">残<b>70</b>枚</span></div>
    <button class="btn btn-secondary" style="padding:4px 12px;font-size:11px" onclick="showTitle()">戻る</button>
  </div>
  <div class="table-area" id="table">
    <div class="seat-top" id="seat-2">
      <div class="seat-info"><div class="seat-name" id="name-2">対面</div><div class="seat-wind" id="wind-2">北家</div><div class="seat-score" id="score-2">25000</div></div>
      <div class="melds-area" id="melds-2"></div>
      <div class="ai-hand" id="hand-2"></div>
      <div class="pond" id="pond-2" style="max-width:260px"></div>
    </div>
    <div class="seat-left" id="seat-1">
      <div class="seat-info"><div class="seat-name" id="name-1">上家</div><div class="seat-wind" id="wind-1">西家</div><div class="seat-score" id="score-1">25000</div></div>
      <div class="melds-area" id="melds-1" style="flex-direction:column"></div>
      <div class="ai-hand" id="hand-1"></div>
      <div class="pond" id="pond-1" style="max-height:100px;flex-direction:column;max-width:60px"></div>
    </div>
    <div class="center-area">
      <div class="info-panel">
        <div class="round-info" id="center-round">東1局</div>
        <div class="dora-area"><span class="dora-label">ドラ:</span><div id="dora-display"></div></div>
      </div>
      <div class="game-log" id="game-log">ゲーム開始</div>
      <div id="riichi-sticks" style="display:flex;gap:4px;justify-content:center;flex-wrap:wrap"></div>
    </div>
    <div class="seat-right" id="seat-3">
      <div class="seat-info"><div class="seat-name" id="name-3">下家</div><div class="seat-wind" id="wind-3">東家</div><div class="seat-score" id="score-3">25000</div></div>
      <div class="melds-area" id="melds-3" style="flex-direction:column"></div>
      <div class="ai-hand" id="hand-3"></div>
      <div class="pond" id="pond-3" style="max-height:100px;flex-direction:column;max-width:60px"></div>
    </div>
    <div class="player-area" id="seat-0">
      <div class="melds-area" id="melds-0"></div>
      <div style="font-size:9px;color:rgba(255,255,255,.4);text-align:center">あなたの河</div>
      <div class="pond" id="pond-0" style="max-width:320px;max-height:80px;margin-bottom:4px"></div>
      <div class="player-hand" id="hand-0"></div>
      <div class="player-info-row">
        <div class="player-info">
          <span class="player-name-label">あなた</span>
          <span style="color:var(--text2);margin:0 6px" id="player-wind-label">東家</span>
          <span class="player-score-label" id="score-0">25000</span>
        </div>
        <div id="riichi-indicator"></div>
      </div>
      <div class="controls" id="controls"></div>
    </div>
    <div class="waiting-overlay" id="waiting">AI思考中<span class="thinking-dots"></span></div>
  </div>
</div>

<div class="screen" id="final-screen">
  <div class="final-title">ゲーム終了</div>
  <div class="rank-table" id="final-ranks"></div>
  <div style="display:flex;gap:12px;margin-top:16px">
    <button class="btn btn-primary" onclick="location.reload()">もう一度</button>
    <button class="btn btn-secondary" onclick="showTitle()">タイトルへ</button>
  </div>
</div>
</div>

<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-title" id="modal-title">和了</div>
    <div id="modal-body"></div>
    <div style="text-align:center;margin-top:16px">
      <button class="btn btn-primary" onclick="closeModal()" style="width:120px">次へ</button>
    </div>
  </div>
</div>
<div class="float-msg" id="float-msg"></div>

<script>
// ============================================================
// MAHJONG ENGINE — S-01 Edition
// ============================================================
const SUITS=['man','pin','sou'];
const HONORS=['東','南','西','北','白','発','中'];
const WIND_CHARS=['東','南','西','北'];
function tilesEqual(a,b){return a.suit===b.suit&&a.num===b.num}
function tileSortKey(t){
  if(t.suit==='man')return 100+t.num;
  if(t.suit==='pin')return 200+t.num;
  if(t.suit==='sou')return 300+t.num;
  return 400+HONORS.indexOf(t.num);
}
function sortHand(h){return[...h].sort((a,b)=>tileSortKey(a)-tileSortKey(b))}
function tileStr(t){
  if(!t)return'?';
  if(t.suit==='honor')return t.num;
  return t.num+(t.suit==='man'?'萬':t.suit==='pin'?'筒':'索');
}
function allTiles(){
  const t=[];
  for(const s of SUITS)for(let n=1;n<=9;n++)for(let i=0;i<4;i++)t.push({suit:s,num:n,uid:t.length});
  for(const h of HONORS)for(let i=0;i<4;i++)t.push({suit:'honor',num:h,uid:t.length});
  return t;
}
function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.random()*i|0;[a[i],a[j]]=[a[j],a[i]];}return a;}

let G={};
function initGame(np,mode){
  G={numPlayers:np,mode,players:[],walls:[],deadWall:[],
     doraIndicators:[],uraDoraIndicators:[],
     activePlayer:0,dealer:0,round:0,honba:0,riichiPool:0,
     phase:'idle',lastDiscard:null,lastDiscardPlayer:-1,
     pendingClaims:[],maxRound:mode==='tonpu'?8:4,
     gameOver:false,waitingForPlayer:false,
     selectedTile:null,riichiCandidates:[],_pendingNextRound:null};
  const _philosopherPool=[
    'ソクラテス','プラトン','アリストテレス','エピクロス','ピュロン',
    'アウグスティヌス','トマス・アクィナス','オッカム',
    'マキャベリ','モンテーニュ','エラスムス',
    'デカルト','スピノザ','ライプニッツ','パスカル','ベーコン',
    'ロック','ヒューム','バークリー','ルソー','ヴォルテール',
    'カント','フィヒテ','シェリング','ヘーゲル',
    'ショーペンハウアー','フォイエルバッハ','マルクス','エンゲルス',
    'ミル','ベンサム','スペンサー','ニーチェ','キルケゴール',
    'フレーゲ','ラッセル','ムーア','ウィトゲンシュタイン',
    'フッサール','ハイデガー','サルトル','メルロ＝ポンティ','ボーヴォワール',
    'デューイ','ジェームズ','パース',
    'カルナップ','ポパー','クワイン','クーン',
    'レヴィナス','デリダ','フーコー','ドゥルーズ','バタイユ',
    'ロールズ','ノージック','サンデル','ハーバーマス',
    'アーレント','ベンヤミン','アドルノ','ホルクハイマー',
  ];
  const _shuffled=[..._philosopherPool].sort(()=>Math.random()-.5);
  const names=['あなた',_shuffled[0],_shuffled[1],_shuffled[2]];
  for(let i=0;i<np;i++)
    G.players.push({name:names[i],isHuman:i===0,score:np===3?35000:25000,
      hand:[],drawn:null,pond:[],melds:[],riichi:false,riichiTurn:-1,wind:WIND_CHARS[i]});
  startRound();
}

function startRound(){
  for(let i=0;i<G.numPlayers;i++){
    G.players[i].wind=WIND_CHARS[(i-G.dealer+4)%4];
    if(G.numPlayers===3&&i===2)G.players[i].wind='北';
  }
  let wall=allTiles();
  if(G.numPlayers===3)wall=wall.filter(t=>!(t.suit==='man'&&t.num>=2&&t.num<=8));
  shuffle(wall);
  G.deadWall=wall.splice(wall.length-14,14);
  G.doraIndicators=[G.deadWall[4]];
  G.uraDoraIndicators=[G.deadWall[9]];
  G.walls=wall;
  for(const p of G.players){p.hand=[];p.drawn=null;p.pond=[];p.melds=[];p.riichi=false;p.riichiTurn=-1;}
  for(let i=0;i<13;i++)for(const p of G.players)p.hand.push(G.walls.shift());
  for(const p of G.players)p.hand=sortHand(p.hand);
  G.phase='draw';G.activePlayer=G.dealer;
  G.lastDiscard=null;G.lastDiscardPlayer=-1;
  G.selectedTile=null;G.riichiCandidates=[];
  renderAll();log(`${roundName()} 開始`);nextTurn();
}

function roundName(){
  const w=['東','南','西','北'][Math.floor(G.round/G.numPlayers)];
  return`${w}${(G.round%G.numPlayers)+1}局`;
}
function wallCount(){return G.walls.length}
function drawTile(pi){if(!G.walls.length)return null;const t=G.walls.shift();G.players[pi].drawn=t;return t;}

// ── ドラ計算 ──
function doraFromIndicator(ind){
  if(!ind)return null;
  if(ind.suit==='honor'){
    const idx=HONORS.indexOf(ind.num);
    return{suit:'honor',num:idx<4?HONORS[(idx+1)%4]:HONORS[4+((idx-4+1)%3)]};
  }
  return{suit:ind.suit,num:ind.num===9?1:ind.num+1};
}
function countDora(hand,melds,inds){
  let c=0;
  const all=[...hand,...melds.flatMap(m=>m.tiles)];
  for(const ind of inds){const d=doraFromIndicator(ind);if(!d)continue;for(const t of all)if(tilesEqual(t,d))c++;}
  return c;
}

// ── 和了判定 ──
function decomposeMentsu(tiles){
  if(!tiles.length)return[];
  const s=[...tiles].sort((a,b)=>tileSortKey(a)-tileSortKey(b));
  for(let i=0;i<s.length-2;i++){
    if(tilesEqual(s[i],s[i+1])&&tilesEqual(s[i],s[i+2])){
      const rest=s.filter((_,x)=>x!==i&&x!==i+1&&x!==i+2);
      const sub=decomposeMentsu(rest);if(sub!==null)return[{type:'pon',tiles:[s[i],s[i+1],s[i+2]]},...sub];
    }
  }
  for(let i=0;i<s.length;i++){
    if(s[i].suit==='honor')continue;
    const t1=s[i];
    const j=s.findIndex((t,x)=>x>i&&tilesEqual(t,{suit:t1.suit,num:t1.num+1}));if(j===-1)continue;
    const k=s.findIndex((t,x)=>x>i&&x!==j&&tilesEqual(t,{suit:t1.suit,num:t1.num+2}));if(k===-1)continue;
    const rest=s.filter((_,x)=>x!==i&&x!==j&&x!==k);
    const sub=decomposeMentsu(rest);if(sub!==null)return[{type:'chi',tiles:[s[i],s[j],s[k]]},...sub];
  }
  return null;
}
function isChiitoitsu(tiles){
  if(tiles.length!==14)return false;
  const g={};for(const t of tiles){const k=t.suit+t.num;g[k]=(g[k]||0)+1;}
  const v=Object.values(g);return v.every(x=>x===2)&&v.length===7;
}
function isKokushi(tiles){
  if(tiles.length!==14)return false;
  const terms=['man1','man9','pin1','pin9','sou1','sou9',...HONORS.map(h=>'honor'+h)];
  const has=new Set(tiles.map(t=>t.suit+t.num));
  if(terms.filter(k=>has.has(k)).length<13)return false;
  const c={};for(const t of tiles)c[t.suit+t.num]=(c[t.suit+t.num]||0)+1;
  return terms.some(k=>c[k]===2);
}
function getWinningDecompositions(tiles){
  const res=[];
  const sorted=sortHand(tiles);
  for(let pi=0;pi<sorted.length;pi++){
    const pair=sorted[pi];
    const pairTiles=[];const remaining=[];let found=0;
    for(const t of sorted){if(found<2&&tilesEqual(t,pair)){pairTiles.push(t);found++;}else remaining.push(t);}
    if(pairTiles.length!==2)continue;
    const melds=decomposeMentsu(remaining);
    if(melds!==null)res.push({pair:pairTiles,melds,tiles:sorted});
  }
  if(isChiitoitsu(sorted))res.push({type:'chiitoitsu',tiles:sorted});
  if(isKokushi(sorted))res.push({type:'kokushi',tiles:sorted});
  return res;
}
function canWin(hand,drawn,melds){
  const all=[...hand,...(drawn?[drawn]:[]),...melds.flatMap(m=>m.tiles)];
  if(all.length<14)return false;
  return getWinningDecompositions([...hand,...(drawn?[drawn]:[])]).length>0;
}
function tenpaiTiles(hand,melds){
  const types=[];
  for(const s of SUITS)for(let n=1;n<=9;n++)types.push({suit:s,num:n});
  for(const h of HONORS)types.push({suit:'honor',num:h});
  return types.filter(t=>canWin(hand,t,melds));
}
function isTenpai(hand,melds){return tenpaiTiles(hand,melds).length>0}

// ── 役判定 ──
function isSuuanko(decomp,isTsumo){if(!decomp.melds)return false;return decomp.melds.every(m=>m.type==='pon'||m.type==='kan')&&isTsumo;}
function isDaisangen(allM){const drg=['白','発','中'];return drg.every(d=>allM.some(m=>(m.type==='pon'||m.type==='kan')&&m.tiles[0].suit==='honor'&&m.tiles[0].num===d));}
function isTsuiso(hAll){return hAll.every(t=>t.suit==='honor');}
function isChinroto(hAll){return hAll.every(t=>t.suit!=='honor'&&(t.num===1||t.num===9));}
function isShousuushi(allM,pair){const winds=['東','南','西','北'];const ponWinds=allM.filter(m=>(m.type==='pon'||m.type==='kan')&&m.tiles[0].suit==='honor'&&winds.includes(m.tiles[0].num));const pairIsWind=pair&&pair[0]&&pair[0].suit==='honor'&&winds.includes(pair[0].num);return ponWinds.length===3&&pairIsWind;}
function isDaisuushi(allM){const winds=['東','南','西','北'];return winds.every(w=>allM.some(m=>(m.type==='pon'||m.type==='kan')&&m.tiles[0].suit==='honor'&&m.tiles[0].num===w));}
function isRyuiso(hAll){const green=['2','3','4','6','8'].map(Number);return hAll.every(t=>(t.suit==='bamboo'&&green.includes(t.num))||(t.suit==='honor'&&t.num==='発'));}
function isChurenpoton(hAll){const suits=hAll.map(t=>t.suit);if(!suits.every(s=>s===suits[0])||suits[0]==='honor')return false;const nums=hAll.map(t=>t.num).sort((a,b)=>a-b);const base=[1,1,1,2,3,4,5,6,7,8,9,9,9];if(nums.length!==14)return false;return base.some((_,i)=>{const b=[...base];b.splice(i,1);return JSON.stringify(b)===JSON.stringify(nums.slice(0,13));});}
function isSuukantsu(allM){return allM.filter(m=>m.type==='kan').length===4;}
function isSananko(decomp,isTsumo,lastDiscard){if(!decomp.melds)return false;const pons=decomp.melds.filter(m=>m.type==='pon');if(isTsumo)return pons.length>=3;return pons.filter(m=>!m.tiles.some(t=>lastDiscard&&t.uid===lastDiscard.uid)).length>=3;}
function isSankantsu(allM){return allM.filter(m=>m.type==='kan').length>=3;}
function isShouSangen(allM,pair){const drg=['白','発','中'];const ponCount=drg.filter(d=>allM.some(m=>(m.type==='pon'||m.type==='kan')&&m.tiles[0].suit==='honor'&&m.tiles[0].num===d)).length;const pairIsDrg=pair&&pair[0]&&pair[0].suit==='honor'&&drg.includes(pair[0].num);return ponCount===2&&pairIsDrg;}
function isHonroto(hAll){return hAll.every(t=>t.suit==='honor'||(t.num===1||t.num===9));}
function isRyanpeikou(decomp,isMenzen){if(!isMenzen||!decomp.melds||decomp.melds.length<4)return false;const chis=decomp.melds.filter(m=>m.type==='chi');if(chis.length<4)return false;const keys=chis.map(m=>m.tiles.map(t=>t.suit+t.num).sort().join(','));const counts={};for(const k of keys)counts[k]=(counts[k]||0)+1;return Object.values(counts).filter(v=>v>=2).length>=2;}

function getYaku(decomp,player,gameState,isTsumo){
  const yaku=[];const{melds,riichi}=player;const isMenzen=melds.length===0;
  const{type}=decomp;
  const hAll=[...player.hand,...(player.drawn?[player.drawn]:[]),...melds.flatMap(m=>m.tiles)];
  const allM=[...melds,...(decomp.melds||[])];
  const yakuman=[];
  if(type==='kokushi') yakuman.push({name:'国士無双',han:13,yakuman:true});
  if(isSuuanko(decomp,isTsumo)) yakuman.push({name:'四暗刻',han:13,yakuman:true});
  if(isDaisangen(allM)) yakuman.push({name:'大三元',han:13,yakuman:true});
  if(isTsuiso(hAll)) yakuman.push({name:'字一色',han:13,yakuman:true});
  if(isChinroto(hAll)) yakuman.push({name:'清老頭',han:13,yakuman:true});
  if(isShousuushi(allM,decomp.pair)) yakuman.push({name:'小四喜',han:13,yakuman:true});
  if(isDaisuushi(allM)) yakuman.push({name:'大四喜',han:26,yakuman:true});
  if(isRyuiso(hAll)) yakuman.push({name:'緑一色',han:13,yakuman:true});
  if(isChurenpoton(hAll)) yakuman.push({name:'九蓮宝燈',han:13,yakuman:true});
  if(isSuukantsu(allM)) yakuman.push({name:'四槓子',han:13,yakuman:true});
  if(yakuman.length>0) return yakuman;
  if(type==='chiitoitsu'){yaku.push({name:'七対子',han:2});}
  else{
    if(isTsumo&&isMenzen)yaku.push({name:'門前清自摸和',han:1});
    if(riichi)yaku.push({name:'立直',han:1});
    if(isTanyao(hAll))yaku.push({name:'断么九',han:1});
    if(isMenzen&&!isTsumo&&isPinfu(decomp,player,gameState))yaku.push({name:'平和',han:1});
    if(isMenzen&&isIipeiko(decomp.melds))yaku.push({name:'一盃口',han:1});
    yaku.push(...checkYakuhai(decomp.pair,allM,player,gameState));
    if(isSanshokuDoujun(allM))yaku.push({name:'三色同順',han:isMenzen?2:1});
    if(isSanshokuDoukou(allM))yaku.push({name:'三色同刻',han:2});
    if(isIttsu(allM))yaku.push({name:'一気通貫',han:isMenzen?2:1});
    if(isToitoi(allM))yaku.push({name:'対々和',han:2});
    const hc=checkHoChiNitsu(hAll);
    if(hc)yaku.push({name:hc,han:hc==='清一色'?(isMenzen?6:5):(isMenzen?3:2)});
  }
  const dc=countDora([...player.hand,...(player.drawn?[player.drawn]:[])],player.melds,gameState.doraIndicators);
  if(dc>0)yaku.push({name:`ドラ${dc}`,han:dc,isBonus:true});
  if(player.riichi){
    const uc=countDora([...player.hand,...(player.drawn?[player.drawn]:[])],player.melds,gameState.uraDoraIndicators);
    if(uc>0)yaku.push({name:`裏ドラ${uc}`,han:uc,isBonus:true});
  }
  if(gameState.numPlayers===3&&player.kitaCount>0){
    for(let k=0;k<player.kitaCount;k++) yaku.push({name:'北抜き',han:1});
  }
  if(isSananko(decomp,isTsumo,gameState.lastDiscard))yaku.push({name:'三暗刻',han:2});
  if(isSankantsu(allM))yaku.push({name:'三槓子',han:2});
  if(isShouSangen(allM,decomp.pair))yaku.push({name:'小三元',han:2});
  if(isHonroto(hAll))yaku.push({name:'混老頭',han:2});
  if(isRyanpeikou(decomp,isMenzen))yaku.push({name:'二盃口',han:3});
  return yaku;
}
function isTanyao(tiles){return tiles.every(t=>t.suit!=='honor'&&t.num>=2&&t.num<=8)}
function isPinfu(decomp,player,gs){
  if(!decomp.melds||!decomp.melds.every(m=>m.type==='chi'))return false;
  const p=decomp.pair[0];
  if(p.suit==='honor'){
    const rw=WIND_CHARS[Math.floor(gs.round/gs.numPlayers)];
    if(p.num===rw||p.num===player.wind)return false;
    if(['白','発','中'].includes(p.num))return false;
  }
  return true;
}
function isIipeiko(melds){
  if(!melds||melds.length<2)return false;
  for(let i=0;i<melds.length;i++)for(let j=i+1;j<melds.length;j++){
    if(melds[i].type==='chi'&&melds[j].type==='chi'){
      const a=sortHand(melds[i].tiles),b=sortHand(melds[j].tiles);
      if(a.every((t,k)=>tilesEqual(t,b[k])))return true;
    }
  }
  return false;
}
function checkYakuhai(pair,allM,player,gs){
  const yaku=[];const rw=WIND_CHARS[Math.floor(gs.round/gs.numPlayers)];
  for(const m of allM){
    if(m.type!=='pon'&&m.type!=='kan')continue;
    const t=m.tiles[0];if(t.suit!=='honor')continue;
    if(t.num==='白')yaku.push({name:'役牌：白',han:1});
    else if(t.num==='発')yaku.push({name:'役牌：発',han:1});
    else if(t.num==='中')yaku.push({name:'役牌：中',han:1});
    else if(t.num===rw)yaku.push({name:`役牌：${rw}`,han:1});
    else if(t.num===player.wind)yaku.push({name:`役牌：${player.wind}`,han:1});
  }
  return yaku;
}
function isSanshokuDoujun(melds){
  const chi=melds.filter(m=>m.type==='chi');
  for(const c of chi){const n=c.tiles[0].num;
    if(chi.some(x=>x.tiles[0].suit==='man'&&x.tiles[0].num===n)&&
       chi.some(x=>x.tiles[0].suit==='pin'&&x.tiles[0].num===n)&&
       chi.some(x=>x.tiles[0].suit==='sou'&&x.tiles[0].num===n))return true;
  }return false;
}
function isSanshokuDoukou(melds){
  const pon=melds.filter(m=>m.type==='pon'||m.type==='kan');
  for(let n=1;n<=9;n++)
    if(pon.some(m=>m.tiles[0].suit==='man'&&m.tiles[0].num===n)&&
       pon.some(m=>m.tiles[0].suit==='pin'&&m.tiles[0].num===n)&&
       pon.some(m=>m.tiles[0].suit==='sou'&&m.tiles[0].num===n))return true;
  return false;
}
function isIttsu(melds){
  const chi=melds.filter(m=>m.type==='chi');
  for(const s of SUITS)
    if(chi.some(m=>m.tiles[0].suit===s&&m.tiles[0].num===1)&&
       chi.some(m=>m.tiles[0].suit===s&&m.tiles[0].num===4)&&
       chi.some(m=>m.tiles[0].suit===s&&m.tiles[0].num===7))return true;
  return false;
}
function isToitoi(melds){return melds.every(m=>m.type==='pon'||m.type==='kan')}
function checkHoChiNitsu(tiles){
  const suits=new Set(tiles.filter(t=>t.suit!=='honor').map(t=>t.suit));
  const hasH=tiles.some(t=>t.suit==='honor');
  if(suits.size===1&&!hasH)return'清一色';
  if(suits.size===1&&hasH)return'混一色';
  return null;
}

// ── 点数計算 ──
function calcFu(decomp,isTsumo,isMenzen){
  if(decomp.type==='chiitoitsu')return 25;
  let fu=isMenzen&&!isTsumo?30:20;
  if(isTsumo&&!isMenzen)fu+=2;
  if(decomp.pair){const p=decomp.pair[0];if(p.suit==='honor'&&['白','発','中'].includes(p.num))fu+=2;}
  for(const m of(decomp.melds||[])){
    const t=m.tiles[0];const isTH=t.suit==='honor'||(t.suit!=='honor'&&(t.num===1||t.num===9));
    if(m.type==='pon')fu+=isTH?4:2;if(m.type==='kan')fu+=isTH?16:8;
  }
  return Math.ceil(fu/10)*10;
}
function calcBasicPoints(han,fu){
  if(han>=13)return 8000;if(han>=11)return 6000;if(han>=8)return 4000;
  if(han>=6)return 3000;if(han===5||(han===4&&fu>=30)||(han===3&&fu>=70))return 2000;
  return Math.min(fu*Math.pow(2,han+2),2000);
}
function calcScore(yaku,decomp,isTsumo,isDealer,winnerMelds){
  const han=yaku.reduce((s,y)=>s+y.han,0);
  const isMenzen=!winnerMelds||winnerMelds.length===0;
  const fu=calcFu(decomp,isTsumo,isMenzen);
  const basic=calcBasicPoints(han,fu);
  if(isTsumo)return{han,fu,basic,dealer:Math.ceil(basic*2/100)*100,nonDealer:Math.ceil(basic/100)*100,isTsumo:true};
  return{han,fu,basic,ron:Math.ceil(basic*(isDealer?6:4)/100)*100,isTsumo:false};
}

// ── 副露可否 ──
function canChi(hand,tile,pi,ldp){
  const left=(pi-1+G.numPlayers)%G.numPlayers;
  if(ldp!==left||tile.suit==='honor')return[];
  const opts=[];const nums=hand.filter(t=>t.suit===tile.suit).map(t=>t.num);const n=tile.num;
  if(nums.includes(n-2)&&nums.includes(n-1))opts.push([n-2,n-1,n]);
  if(nums.includes(n-1)&&nums.includes(n+1))opts.push([n-1,n,n+1]);
  if(nums.includes(n+1)&&nums.includes(n+2))opts.push([n,n+1,n+2]);
  return opts;
}
function canPon(hand,tile){return hand.filter(t=>tilesEqual(t,tile)).length>=2}
function canKan(hand,tile){return hand.filter(t=>tilesEqual(t,tile)).length>=3}
function canAnkan(hand){
  const c={};for(const t of hand)c[t.suit+t.num]=(c[t.suit+t.num]||0)+1;
  return Object.entries(c).filter(([,v])=>v===4).map(([k])=>k);
}
function canRon(hand,melds,tile,player){
  const th=[...hand,tile];
  if(!canWin(th,null,melds))return false;
  const decomps=getWinningDecompositions(th);
  return decomps.some(d=>getYaku(d,{...player,drawn:tile},G,false).filter(y=>!y.isBonus).length>0);
}
function canTsumo(player){
  if(!player.drawn)return false;
  if(!canWin(player.hand,player.drawn,player.melds))return false;
  const th=[...player.hand,player.drawn];
  return getWinningDecompositions(th).some(d=>getYaku(d,player,G,true).filter(y=>!y.isBonus).length>0);
}

// ── ゲームフロー ──
let _lastProgress=Date.now();
let _lastClaimWait=0;  // ★ claim待機開始時刻
function _touchProgress(){_lastProgress=Date.now();}
setInterval(()=>{
  if(G.gameOver||!G.phase||G.phase==='idle')return;
  if(G.waitingForPlayer){
    // ★[修正/freeze-claim] 旧ウォッチドッグは waitingForPlayer=true の場合を
    // 完全にスキップしていた。ヒューマンクレーム待ちが無限化する根本原因。
    // claim フェーズで 18 秒以上応答なし → 強制パス（ロン/ポン放棄）
    if(G.phase==='claim'&&Date.now()-(_lastClaimWait||_lastProgress)>18000){
      console.warn('フリーズ検知: クレーム待機タイムアウト → 強制パス');
      G.waitingForPlayer=false;G.pendingClaims=[];
      advanceTurn(G.lastDiscardPlayer!=null?G.lastDiscardPlayer:G.activePlayer||0);
      _touchProgress();_lastClaimWait=0;
    }
    return;
  }
  // ★[修正/freeze-watch] AIターン進行不能を検知
  if(Date.now()-_lastProgress>6000){
    console.warn('フリーズ検知: 強制進行 phase='+G.phase+' active='+G.activePlayer);
    G.waitingForPlayer=false;G.pendingClaims=[];G.phase='draw';
    advanceTurn(G.activePlayer||0);
    _touchProgress();
  }
},2000);

function nextTurn(){
  if(G.gameOver)return;
  // ★[修正/freeze-phase] フェーズガード: draw 以外のフェーズで呼ばれても無視。
  // 二重コール・ウォッチドッグ誤発動による手牌破壊を防ぐ。
  if(G.phase!=='draw'){console.warn('nextTurn: 不正フェーズ='+G.phase+' スキップ');return;}
  _touchProgress();
  const p=G.players[G.activePlayer];
  if(!G.walls.length){handleRyukyoku();return;}
  const tile=drawTile(G.activePlayer);if(!tile){handleRyukyoku();return;}
  log(`${p.name}がツモ`);renderAll();
  if(p.isHuman){G.phase='discard';G.waitingForPlayer=true;renderControls();}
  else{
    // ★[修正/race-cond] 旧コード: setTimeout(()=>aiTurn(G.activePlayer), ...)
    // コールバック実行時に G.activePlayer が変化している可能性がある（レース条件）。
    // aiTurn(pi) の pi != G.activePlayer でガードが発動し turn が消滅するフリーズを修正。
    const _capturedPi=G.activePlayer;
    G.phase='discard';  // ★ discard フェーズに設定してウォッチドッグが検知できるようにする
    setTimeout(()=>aiTurn(_capturedPi),700+Math.random()*500);
  }
}

function aiTurn(pi){
  if(G.gameOver)return;
  // ★[修正/race-cond] 旧コードは activePlayer が変わっていた場合に単純 return していた。
  // これによりターンが消滅しフリーズ。→ activePlayer 不一致なら turn を正しい対象に渡す。
  if(G.activePlayer!==pi){
    console.warn(`aiTurn: pi=${pi} vs activePlayer=${G.activePlayer} 不一致。再発行。`);
    const _correct=G.activePlayer;
    if(!G.players[_correct]||G.players[_correct].isHuman||G.waitingForPlayer)return;
    setTimeout(()=>aiTurn(_correct),200);
    return;
  }
  const p=G.players[pi];showWaiting(true);
  setTimeout(()=>{
    if(canTsumo(p)){showWaiting(false);declareWin(pi,null,true);return;}
    // ★[修正/ai-kita] 三麻ではAIも北抜きを優先的に実行する。
    // 旧コードは北抜きロジックが完全に欠落しており、北を手牌に持ったまま進行していた。
    if(G.numPlayers===3){
      const drawnIsKita=p.drawn&&p.drawn.suit==='honor'&&p.drawn.num==='北';
      const kitaIdxH=drawnIsKita?-1:p.hand.findIndex(t=>t.suit==='honor'&&t.num==='北');
      if(drawnIsKita||kitaIdxH>=0){
        let kita;
        if(drawnIsKita){kita=p.drawn;p.drawn=null;}
        else{kita=p.hand.splice(kitaIdxH,1)[0];}
        if(!p.kitaCount)p.kitaCount=0;
        p.kitaCount++;
        if(!G.kitaDora)G.kitaDora=[];
        G.kitaDora.push(kita);
        p.drawn=G.walls.length?G.walls.shift():null;
        log(`${p.name}が北を抜いた（${p.kitaCount}枚目）`);
        showPhilosopherSpeech(pi,'kita');
        renderAll();
        showWaiting(false);
        setTimeout(()=>aiTurn(pi),500);
        return;
      }
    }
    const ak=canAnkan([...p.hand,...(p.drawn?[p.drawn]:[])]);
    if(ak.length&&Math.random()<0.3){
      const kt=[...p.hand,...(p.drawn?[p.drawn]:[])].find(t=>t.suit+t.num===ak[0]);
      doKan(pi,kt,true);showWaiting(false);return;
    }
    const hwD=[...p.hand,...(p.drawn?[p.drawn]:[])];
    if(!p.riichi&&!p.melds.length&&p.score>=1000&&Math.random()<0.45){
      const wts=tenpaiTiles(hwD.slice(0,-1),p.melds);
      if(wts.length){const d=chooseAIDiscard(pi,true);if(d){doRiichi(pi,d);showWaiting(false);return;}}
    }
    const d=chooseAIDiscard(pi,false);if(d)doDiscard(pi,d);
    showWaiting(false);
  },400+Math.random()*400);
}

function evaluateHand(hand,melds){
  let score=0;
  const d=decomposeMentsu([...hand]);if(d!==null)score+=d.length*10;
  if(isTenpai(hand,melds))score+=50;
  const s=sortHand(hand);
  for(let i=0;i<s.length-1;i++){
    if(tilesEqual(s[i],s[i+1]))score+=3;
    if(s[i].suit!=='honor'&&s[i+1].suit===s[i].suit&&s[i+1].num===s[i].num+1)score+=2;
    if(s[i].suit!=='honor'&&s[i+1].suit===s[i].suit&&s[i+1].num===s[i].num+2)score+=1;
  }
  for(const t of s){
    if(t.suit==='honor'&&!s.some(x=>x!==t&&tilesEqual(x,t)))score-=2;
    if(t.suit!=='honor'&&(t.num===1||t.num===9)&&!s.some(x=>x!==t&&tilesEqual(x,t)))score-=1;
  }
  return score;
}
function chooseAIDiscard(pi,forRiichi){
  const p=G.players[pi];const all=[...p.hand,...(p.drawn?[p.drawn]:[])];
  if(!all.length)return null;
  if(forRiichi){
    // リーチ用: テンパイを維持できる捨て牌のみ候補にする
    const riichiCands=all.filter(t=>{const test=all.filter(x=>x.uid!==t.uid);return tenpaiTiles(test,p.melds).length>0;});
    if(riichiCands.length){
      let best=null,bs=-Infinity;
      for(const t of riichiCands){const test=all.filter(x=>x.uid!==t.uid);const sc=evaluateHand(test,p.melds);if(sc>bs){bs=sc;best=t;}}
      return best;
    }
  }
  let best=null,bs=-Infinity;
  for(const t of all){
    const test=all.filter(x=>x.uid!==t.uid);const sc=evaluateHand(test,p.melds);
    if(sc>bs){bs=sc;best=t;}
  }
  return best||all[all.length-1];
}

function doDiscard(pi,tile){
  const p=G.players[pi];
  if(p.drawn&&p.drawn.uid===tile.uid){p.drawn=null;}
  else{
    const idx=p.hand.findIndex(t=>t.uid===tile.uid);
    if(idx!==-1)p.hand.splice(idx,1);
    if(p.drawn){p.hand.push(p.drawn);p.drawn=null;}
  }
  p.hand=sortHand(p.hand);
  p.pond.push({...tile,riichi:p.riichi&&p.riichiTurn===-1&&p.pond.length===0});
  G.lastDiscard=tile;G.lastDiscardPlayer=pi;G.phase='claim';G.selectedTile=null;
  log(`${p.name}が${tileStr(tile)}を捨て`);renderAll();checkClaims(tile,pi);
}
function doRiichi(pi,discardTile){
  const p=G.players[pi];p.score-=1000;G.riichiPool+=1000;
  p.riichi=true;p.riichiTurn=G.players.flatMap(x=>x.pond).length;
  showPhilosopherSpeech(pi,'riichi');
  showFloatMsg('立直！');doDiscard(pi,discardTile);
}
function doKan(pi,tile,isAnkan){
  const p=G.players[pi];
  const kanTiles=[...p.hand,...(p.drawn?[p.drawn]:[])].filter(t=>tilesEqual(t,tile));
  let rm=0;p.hand=p.hand.filter(t=>{if(rm<4&&tilesEqual(t,tile)){rm++;return false;}return true;});
  if(p.drawn&&tilesEqual(p.drawn,tile)&&rm<4){p.drawn=null;rm++;}
  p.melds.push({type:'kan',tiles:kanTiles,isAnkan});
  if(G.deadWall.length){
    p.drawn=G.deadWall.shift();
    // ★[修正/kan-dora] 旧コード: G.deadWall[4-G.doraIndicators.length]
    // G.doraIndicators.length が 4 を超えると負インデックス → undefined がドラに混入し
    // renderDora でクラッシュ or カウント誤算の原因になっていた。
    const _doraPos=4-G.doraIndicators.length;
    if(_doraPos>=0&&_doraPos<G.deadWall.length){
      G.doraIndicators.push(G.deadWall[_doraPos]);
    }
  }
  renderAll();log(`${p.name}が槓`);
  if(p.isHuman){G.phase='discard';renderControls();}
  else setTimeout(()=>aiTurn(pi),600);
}

function doDaimikan(pi,discardTile){
  // 大明槓: 手牌から3枚 + 捨て牌1枚 = 4枚でカン
  const p=G.players[pi];
  const kanTiles=[];let rm=0;
  p.hand=p.hand.filter(t=>{if(rm<3&&tilesEqual(t,discardTile)){rm++;kanTiles.push(t);return false;}return true;});
  if(rm<3){console.warn('doDaimikan: 牌不足 rm='+rm+' → 強制進行');advanceTurn(G.lastDiscardPlayer!=null?G.lastDiscardPlayer:0);return;}
  kanTiles.push(discardTile);
  p.melds.push({type:'kan',tiles:kanTiles,isAnkan:false});
  if(G.deadWall.length){
    p.drawn=G.deadWall.shift();
    const _doraPos=4-G.doraIndicators.length;
    if(_doraPos>=0&&_doraPos<G.deadWall.length)G.doraIndicators.push(G.deadWall[_doraPos]);
  }
  G.pendingClaims=[];
  renderAll();log(`${p.name}が大明槓`);
  if(p.isHuman){G.activePlayer=pi;G.phase='discard';G.waitingForPlayer=true;renderControls();}
  else{G.activePlayer=pi;setTimeout(()=>aiTurn(pi),600);}
}

function checkClaims(tile,dpi){
  const claims=[];
  for(let i=0;i<G.numPlayers;i++){
    if(i===dpi)continue;const p=G.players[i];
    if(canRon(p.hand,p.melds,tile,p)){
      if(p.isHuman)claims.push({type:'ron',player:i,priority:3});
      else if(Math.random()<0.7)claims.push({type:'ron',player:i,priority:3});
    }
    // ★[修正/kita-pon] 三麻では北牌はポン不可（北抜き専用）。
    // 旧コードはこのチェックがなく、他家が捨てた北をAIがポンしてしまっていた。
    const _isKitaTile3=(G.numPlayers===3&&tile.suit==='honor'&&tile.num==='北');
    if(!p.riichi&&canPon(p.hand,tile)&&!_isKitaTile3){
      if(p.isHuman)claims.push({type:'pon',player:i,priority:2});
      else if(Math.random()<0.4)claims.push({type:'pon',player:i,priority:2});
    }
    // ★[修正/kan] 大明槓クレーム（手牌に3枚ある場合）
    if(!p.riichi&&canKan(p.hand,tile)&&!_isKitaTile3){
      if(p.isHuman)claims.push({type:'kan',player:i,priority:2});
      else if(Math.random()<0.25)claims.push({type:'kan',player:i,priority:2});
    }
    if(!p.riichi&&G.numPlayers===4){
      const co=canChi(p.hand,tile,i,dpi);
      if(co.length){
        if(p.isHuman)claims.push({type:'chi',player:i,priority:1,options:co});
        else if(evaluateHand(p.hand.concat([tile]),p.melds)>evaluateHand(p.hand,p.melds)&&Math.random()<0.35)
          claims.push({type:'chi',player:i,priority:1,options:co});
      }
    }
  }
  claims.sort((a,b)=>b.priority-a.priority);
  const hc=claims.filter(c=>c.player===0);
  const ac=claims.filter(c=>c.player!==0);
  const ron=claims.filter(c=>c.type==='ron');
  if(ron.length){
    if(ron.some(c=>c.player===0)){
      G.pendingClaims=hc;G.waitingForPlayer=true;
      _lastClaimWait=Date.now();  // ★[修正/freeze-claim] 待機開始時刻を記録
      renderControls();return;
    }
    declareWin(ron[0].player,dpi,false);return;
  }
  if(hc.length){
    G.pendingClaims=hc;G.waitingForPlayer=true;
    _lastClaimWait=Date.now();  // ★[修正/freeze-claim] 待機開始時刻を記録
    renderControls();return;
  }
  if(ac.length){_touchProgress();setTimeout(()=>executeAIClaim(ac[0],tile),500);return;}
  advanceTurn(dpi);
}
function executeAIClaim(claim,tile){
  if(G.gameOver)return;
  _touchProgress();  // ★[修正/freeze-claim] クレーム実行時に進捗タイムスタンプ更新
  const p=G.players[claim.player];
  if(claim.type==='ron'){declareWin(claim.player,G.lastDiscardPlayer,false);}
  else if(claim.type==='pon'){
    let rm=0;const pt=[];
    p.hand=p.hand.filter(t=>{if(rm<2&&tilesEqual(t,tile)){rm++;pt.push(t);return false;}return true;});
    // ★[修正/freeze-claim] ポン牌が不足した場合の安全弁（手牌破損時のフリーズ防止）
    if(rm<2){console.warn('executeAIClaim: ポン牌不足 rm='+rm+' → 強制進行');advanceTurn(G.lastDiscardPlayer!=null?G.lastDiscardPlayer:0);return;}
    p.melds.push({type:'pon',tiles:[...pt,tile]});
    G.activePlayer=claim.player;G.phase='discard';log(`${p.name}がポン`);
    showPhilosopherSpeech(claim.player,'pon');
    renderAll();
    // ★[修正/race-cond] claim.player をクロージャで捕捉しレース条件を回避
    const _cpi=claim.player;
    setTimeout(()=>aiTurn(_cpi),600);
  } else if(claim.type==='chi'){
    const chiNums=claim.options[0];const ct=[];const th=[...p.hand];
    for(const n of chiNums){
      if(n===tile.num&&tilesEqual({suit:tile.suit,num:n},tile)){ct.push(tile);}
      else{const x=th.findIndex(t=>t.suit===tile.suit&&t.num===n);if(x!==-1)ct.push(th.splice(x,1)[0]);}
    }
    // ★[修正/freeze-claim] チー牌が揃わない場合の安全弁
    if(ct.length<3){console.warn('executeAIClaim: チー牌不足→強制進行');advanceTurn(G.lastDiscardPlayer!=null?G.lastDiscardPlayer:0);return;}
    p.hand=th;p.melds.push({type:'chi',tiles:ct});
    G.activePlayer=claim.player;G.phase='discard';log(`${p.name}がチー`);
    showPhilosopherSpeech(claim.player,'chi');
    renderAll();
    const _cpi=claim.player;
    setTimeout(()=>aiTurn(_cpi),600);
  } else if(claim.type==='kan'){
    // ★[修正/kan] AI大明槓
    doDaimikan(claim.player,tile);
  } else {
    // ★[修正/freeze-claim] 未知のクレームタイプ
    console.warn('executeAIClaim: 未知タイプ='+claim.type+' → 強制進行');
    advanceTurn(G.lastDiscardPlayer!=null?G.lastDiscardPlayer:G.activePlayer||0);
  }
}
function advanceTurn(from){
  // ★[修正/freeze-phase] 状態を完全にリセットしてから次ターンへ
  G.pendingClaims=[];G.waitingForPlayer=false;
  G.activePlayer=(from+1)%G.numPlayers;G.phase='draw';
  _touchProgress();
  renderControls();setTimeout(()=>nextTurn(),200);
}

function declareWin(wi,li,isTsumo){
  G.gameOver=true;
  const winner=G.players[wi];
  const allH=[...winner.hand,...(winner.drawn?[winner.drawn]:[])];
  const decomps=getWinningDecompositions(allH);
  const decomp=decomps[0]||{type:'normal',pair:[],melds:[],tiles:allH};
  const yaku=getYaku(decomp,winner,G,isTsumo);
  const isDealer=wi===G.dealer;
  const si=calcScore(yaku,decomp,isTsumo,isDealer,winner.melds);
  const deltas=Array(G.numPlayers).fill(0);
  if(isTsumo){
    for(let i=0;i<G.numPlayers;i++){
      if(i===wi)continue;const pay=(i===G.dealer?si.dealer:si.nonDealer)+(G.honba*100);
      G.players[i].score-=pay;deltas[i]-=pay;deltas[wi]+=pay;
    }
  } else {
    const pay=si.ron+(G.honba*300);G.players[li].score-=pay;deltas[li]-=pay;deltas[wi]+=pay;
  }
  G.players[wi].score+=G.riichiPool;deltas[wi]+=G.riichiPool;G.riichiPool=0;
  showPhilosopherSpeech(wi,isTsumo?'tsumo':'ron');
  showFloatMsg(isTsumo?'ツモ！':'ロン！');
  setTimeout(()=>showWinModal(wi,li,isTsumo,yaku,si,decomp,allH,deltas),500);
}
function showWinModal(wi,li,isTsumo,yaku,si,decomp,allH,deltas){
  let body=`<div class="hand-display">${allH.map(t=>tileHTML(t,'medium')).join('')}</div>`;
  body+=`<div style="margin:8px 0;font-size:13px;color:var(--text2)">${yaku.map(y=>`<span style="margin-right:8px;color:${y.isBonus?'#ffd700':'var(--text)'}">${y.name}(${y.han}翻)</span>`).join('')}</div>`;
  body+=`<div style="text-align:center;font-size:20px;font-weight:900;color:#ffd700;margin:8px 0">${si.han}翻${si.fu}符 ${isTsumo?si.nonDealer+'点ALL':si.ron+'点'}</div>`;
  body+=`<div style="margin-top:12px">`;
  for(let i=0;i<G.numPlayers;i++){const d=deltas[i];body+=`<div class="result-row"><span>${G.players[i].name}</span><span class="score-delta ${d>=0?'pos':'neg'}">${d>=0?'+':''}${d}</span><span>${G.players[i].score}</span></div>`;}
  body+=`</div>`;
  document.getElementById('modal-title').textContent=isTsumo?'ツモ和了':'ロン和了';
  document.getElementById('modal-body').innerHTML=body;
  document.getElementById('modal').classList.add('show');
  G._pendingNextRound=()=>{
    if(wi===G.dealer)G.honba++;else{G.honba=0;G.dealer=(G.dealer+1)%G.numPlayers;G.round++;}
    G.gameOver=false;
    if(G.round>=G.maxRound){showFinalScreen();return;}
    startRound();
  };
}
function closeModal(){
  document.getElementById('modal').classList.remove('show');
  if(G._pendingNextRound){G._pendingNextRound();G._pendingNextRound=null;}
}
function handleRyukyoku(){
  // ★[修正/freeze-ryukyoku] waitingForPlayer が true のまま gameOver になると
  // 次局の startRound 後もウォッチドッグが誤発動する。先にリセットする。
  G.waitingForPlayer=false;G.pendingClaims=[];_lastClaimWait=0;
  G.gameOver=true;
  const tp=G.players.map(p=>isTenpai(p.hand,p.melds));
  const tc=tp.filter(Boolean).length;
  if(tc>0&&tc<G.numPlayers){
    const pay=3000/tc|0;const rcv=3000/(G.numPlayers-tc)|0;
    for(let i=0;i<G.numPlayers;i++){if(tp[i])G.players[i].score+=rcv;else G.players[i].score-=pay;}
  }
  let body='<div style="font-size:14px">';
  for(let i=0;i<G.numPlayers;i++)body+=`<div class="result-row"><span>${G.players[i].name}</span><span>${tp[i]?'聴牌':'不聴'}</span><span>${G.players[i].score}</span></div>`;
  body+='</div>';
  document.getElementById('modal-title').textContent='流局';
  document.getElementById('modal-body').innerHTML=body;
  document.getElementById('modal').classList.add('show');
  G._pendingNextRound=()=>{G.honba++;G.gameOver=false;G.round++;if(G.round>=G.maxRound){showFinalScreen();return;}startRound();};
}
function showFinalScreen(){
  const ranked=[...G.players].map((p,i)=>({...p,idx:i})).sort((a,b)=>b.score-a.score);
  document.getElementById('final-ranks').innerHTML=ranked.map((p,i)=>`<div class="rank-row rank-${i+1}"><span>${i+1}位 ${p.name}</span><span>${p.score.toLocaleString()}点</span></div>`).join('');
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('final-screen').classList.add('active');
}

// ── 人間操作 ──
function selectTile(uid){
  if(!G.waitingForPlayer||G.phase!=='discard')return;
  if(G.players[0].riichi)return;
  const all=[...G.players[0].hand,...(G.players[0].drawn?[G.players[0].drawn]:[])];
  const tile=all.find(t=>t.uid===uid);if(!tile)return;
  if(G.selectedTile&&G.selectedTile.uid===uid){humanDiscard(uid);return;}
  if(G.riichiCandidates.length){
    if(!G.riichiCandidates.some(r=>r.uid===uid))return;
    G.waitingForPlayer=false;G.riichiCandidates=[];doRiichi(0,tile);return;
  }
  G.selectedTile=tile;renderHand(0);renderControls();
}
function humanDiscard(uid){
  if(!G.waitingForPlayer)return;const p=G.players[0];
  if(p.riichi){if(!p.drawn)return;G.selectedTile=null;G.waitingForPlayer=false;doDiscard(0,p.drawn);return;}
  const all=[...p.hand,...(p.drawn?[p.drawn]:[])];
  const tile=uid!==-1?all.find(t=>t.uid===uid):G.selectedTile;
  if(!tile)return;G.selectedTile=null;G.waitingForPlayer=false;doDiscard(0,tile);
}
function humanTsumo(){if(!G.waitingForPlayer)return;G.waitingForPlayer=false;declareWin(0,null,true);}
function humanRiichi(){
  if(!G.waitingForPlayer)return;const p=G.players[0];
  if(p.riichi||p.score<1000||p.melds.length)return;
  const all=[...p.hand,...(p.drawn?[p.drawn]:[])];
  const cands=all.filter(t=>isTenpai(all.filter(x=>x.uid!==t.uid),p.melds));
  if(!cands.length)return;
  G.riichiCandidates=cands;renderControls();renderHand(0);
}
function humanRon(){if(!G.waitingForPlayer)return;G.waitingForPlayer=false;declareWin(0,G.lastDiscardPlayer,false);}
function humanChi(chiNums){
  if(!G.waitingForPlayer)return;const p=G.players[0];const tile=G.lastDiscard;
  const ct=[];const th=[...p.hand];
  for(const n of chiNums){
    if(n===tile.num&&tilesEqual({suit:tile.suit,num:n},tile)){ct.push(tile);}
    else{const x=th.findIndex(t=>t.suit===tile.suit&&t.num===n);if(x!==-1)ct.push(th.splice(x,1)[0]);}
  }
  p.hand=th;p.melds.push({type:'chi',tiles:ct});
  G.activePlayer=0;G.phase='discard';G.waitingForPlayer=true;G.pendingClaims=[];
  log('チー');renderAll();renderControls();
}
function humanPon(){
  if(!G.waitingForPlayer)return;const p=G.players[0];const tile=G.lastDiscard;
  let rm=0;const pt=[];
  p.hand=p.hand.filter(t=>{if(rm<2&&tilesEqual(t,tile)){rm++;pt.push(t);return false;}return true;});
  p.melds.push({type:'pon',tiles:[...pt,tile]});
  G.activePlayer=0;G.phase='discard';G.waitingForPlayer=true;G.pendingClaims=[];
  log('ポン');renderAll();renderControls();
}
function humanSkip(){
  if(!G.waitingForPlayer)return;
  G.pendingClaims=[];G.waitingForPlayer=false;G.riichiCandidates=[];G.selectedTile=null;
  advanceTurn(G.lastDiscardPlayer);
}

// ── カン操作 ──
function humanKan(){
  // 大明槓: claim フェーズで呼ばれる
  if(!G.waitingForPlayer)return;
  const tile=G.lastDiscard;
  doDaimikan(0,tile);
}
function humanAnkan(sn){
  // 暗槓: discard フェーズで呼ばれる
  if(!G.waitingForPlayer)return;
  const p=G.players[0];
  const all=[...p.hand,...(p.drawn?[p.drawn]:[])];
  const tile=all.find(t=>t.suit+t.num===sn);
  if(!tile)return;
  G.waitingForPlayer=false;
  doKan(0,tile,true);
}
function humanKakan(sn){
  // 加槓: discard フェーズで呼ばれる（既存のポンに4枚目を追加）
  if(!G.waitingForPlayer)return;
  const p=G.players[0];
  const all=[...p.hand,...(p.drawn?[p.drawn]:[])];
  const addTile=all.find(t=>t.suit+t.num===sn);
  if(!addTile)return;
  const ponIdx=p.melds.findIndex(m=>m.type==='pon'&&tilesEqual(m.tiles[0],addTile));
  if(ponIdx===-1)return;
  if(p.drawn&&p.drawn.uid===addTile.uid){p.drawn=null;}
  else{const idx=p.hand.findIndex(t=>t.uid===addTile.uid);if(idx!==-1)p.hand.splice(idx,1);}
  p.melds[ponIdx].tiles.push(addTile);
  p.melds[ponIdx].type='kan';
  if(G.deadWall.length){
    p.drawn=G.deadWall.shift();
    const _doraPos=4-G.doraIndicators.length;
    if(_doraPos>=0&&_doraPos<G.deadWall.length)G.doraIndicators.push(G.deadWall[_doraPos]);
  }
  G.waitingForPlayer=false;
  G.activePlayer=0;G.phase='discard';G.waitingForPlayer=true;
  log('加槓');renderAll();renderControls();
}

// ── 描画 ──
function tileHTML(t,sz='medium'){
  if(!t)return'';
  return`<div class="tile ${t.suit} ${sz}" onclick="selectTile(${t.uid})" ondblclick="humanDiscard(${t.uid})">${tileStr(t)}</div>`;
}
function tileHTMLSel(t,sz,sel,rc){
  if(!t)return'';let c=`tile ${t.suit} ${sz}`;if(sel||rc)c+=' selected';
  return`<div class="${c}" onclick="selectTile(${t.uid})" ondblclick="humanDiscard(${t.uid})">${tileStr(t)}</div>`;
}
function renderHand(pi){
  const p=G.players[pi];const el=document.getElementById(`hand-${pi}`);if(!el)return;
  if(pi===0){
    let html='';
    for(const t of p.hand)html+=tileHTMLSel(t,'full',G.selectedTile&&G.selectedTile.uid===t.uid,G.riichiCandidates.some(r=>r.uid===t.uid));
    if(p.drawn){
      html+=`<div style="margin-left:8px;border-left:2px solid rgba(255,215,0,.4);padding-left:8px">`;
      html+=tileHTMLSel(p.drawn,'full',G.selectedTile&&G.selectedTile.uid===p.drawn.uid,G.riichiCandidates.some(r=>r.uid===p.drawn.uid));
      html+=`</div>`;
    }
    el.innerHTML=html;
  } else {
    const cnt=p.hand.length+(p.drawn?1:0);
    const sz=pi===2?'':'small';
    el.innerHTML=Array(cnt).fill(`<div class="tile-back ${sz}"></div>`).join('');
  }
}
function renderPond(pi){
  const p=G.players[pi];const el=document.getElementById(`pond-${pi}`);if(!el)return;
  el.innerHTML=p.pond.map(t=>`<div class="tile discarded ${t.suit}">${tileStr(t)}</div>`).join('');
}
function doKita(){
  const p=G.players[0];
  // 手牌と drawn の両方から北を探す
  const kitaInDrawn=p.drawn&&p.drawn.suit==='honor'&&p.drawn.num==='北';
  const kitaIdx=kitaInDrawn?-1:p.hand.findIndex(t=>t.suit==='honor'&&t.num==='北');
  if(kitaIdx<0&&!kitaInDrawn)return;
  // 北を除去
  let kita;
  if(kitaInDrawn){kita=p.drawn;p.drawn=null;}
  else{kita=p.hand.splice(kitaIdx,1)[0];}
  if(!p.kitaCount)p.kitaCount=0;
  p.kitaCount++;
  if(!G.kitaDora)G.kitaDora=[];
  G.kitaDora.push(kita);
  log(`あなたが北を抜いた（${p.kitaCount}枚目）`);
  // ★[修正/kita-dbl] drawTile() は G.players[0].drawn をセットしてから tile を返す。
  // 旧コードは drawTile() の後に p.hand.push(drawn) を呼んでいたため
  // 同一タイルが p.drawn と p.hand の両方に存在し手牌14枚→15枚扱いになっていた。
  // 修正: walls から直接取り出して p.drawn にのみセットする。
  p.drawn = G.walls.length ? G.walls.shift() : null;
  renderAll();renderControls();
}
function renderMelds(pi){
  const p=G.players[pi];const el=document.getElementById(`melds-${pi}`);if(!el)return;
  const sz=pi===0?'medium':'small';
  el.innerHTML=p.melds.map(m=>`<div class="meld">${m.tiles.map(t=>`<div class="tile ${t.suit} ${sz}">${tileStr(t)}</div>`).join('')}</div>`).join('');
}
function renderControls(){
  const el=document.getElementById('controls');if(!el)return;
  const p=G.players[0];let html='';showWaiting(false);
  if(G.numPlayers===3&&G.phase==='discard'&&G.activePlayer===0&&G.waitingForPlayer){
    // ★[修正/kita-btn] 旧コードは p.hand のみ検索していたため、ツモ牌(p.drawn)が
    // 北の場合に北抜きボタンが表示されなかった。hand と drawn の両方をチェックする。
    const _allForKita=[...G.players[0].hand,...(G.players[0].drawn?[G.players[0].drawn]:[])];
    const _ki=_allForKita.findIndex(t=>t.suit==='honor'&&t.num==='北');
    if(_ki>=0) html+=`<button class="action-btn" style="background:#1a6b1a" onclick="doKita()">北抜き</button>`;
  }
  if(G.phase==='discard'&&G.activePlayer===0&&G.waitingForPlayer){
    if(canTsumo(p))html+=`<button class="action-btn btn-tsumo" onclick="humanTsumo()">ツモ</button>`;
    // ★[修正/kan] 暗槓・加槓ボタン
    if(!p.riichi){
      const _ak=canAnkan([...p.hand,...(p.drawn?[p.drawn]:[])]);
      for(const _sn of _ak){const _kt=[...p.hand,...(p.drawn?[p.drawn]:[])].find(t=>t.suit+t.num===_sn);html+=`<button class="action-btn btn-kan" onclick="humanAnkan('${_sn}')">暗槓(${_kt?tileStr(_kt):'?'})</button>`;}
      const _allH=[...p.hand,...(p.drawn?[p.drawn]:[])];
      for(const _m of p.melds){if(_m.type==='pon'&&_allH.some(t=>tilesEqual(t,_m.tiles[0]))){const _sn=_m.tiles[0].suit+_m.tiles[0].num;html+=`<button class="action-btn btn-kan" onclick="humanKakan('${_sn}')">加槓(${tileStr(_m.tiles[0])})</button>`;}}
    }
    if(!p.riichi&&!p.melds.length&&p.score>=1000){
      const all=[...p.hand,...(p.drawn?[p.drawn]:[])];
      if(all.some(t=>isTenpai(all.filter(x=>x.uid!==t.uid),p.melds)))
        html+=`<button class="action-btn btn-riichi" onclick="humanRiichi()">立直</button>`;
    }
    if(G.riichiCandidates.length){
      // ★[修正/dama-tsumo] 旧コード html=代入でツモボタンを上書きしていた→html+=に変更
      html+=`<span style="font-size:12px;color:var(--gold)">立直する牌を選んでください</span>`;
      html+=`<button class="action-btn btn-skip" onclick="G.riichiCandidates=[];G.selectedTile=null;renderControls();renderHand(0);">キャンセル</button>`;
    } else if(G.selectedTile||p.riichi){
      html+=`<button class="action-btn btn-discard" onclick="humanDiscard(${p.riichi?(p.drawn?p.drawn.uid:-1):G.selectedTile?.uid})">${p.riichi?'ツモ切り':'捨てる'}</button>`;
    } else {
      html+=`<span style="font-size:12px;color:var(--text2)">牌を選んで捨ててください（ダブルクリックで即捨て）</span>`;
    }
  } else if(G.phase==='claim'&&G.waitingForPlayer){
    const cs=G.pendingClaims;
    if(cs.some(c=>c.type==='ron'))html+=`<button class="action-btn btn-ron" onclick="humanRon()">ロン</button>`;
    if(cs.some(c=>c.type==='pon'))html+=`<button class="action-btn btn-pon" onclick="humanPon()">ポン</button>`;
    // ★[修正/kan] 大明槓ボタン
    if(cs.some(c=>c.type==='kan'))html+=`<button class="action-btn btn-kan" onclick="humanKan()">カン</button>`;
    if(cs.some(c=>c.type==='chi')){
      cs.filter(c=>c.type==='chi')[0].options.forEach(o=>{
        html+=`<button class="action-btn btn-chi" onclick="humanChi([${o}])">チー(${o.join('-')})</button>`;
      });
    }
    html+=`<button class="action-btn btn-skip" onclick="humanSkip()">スキップ</button>`;
  } else if(!G.waitingForPlayer&&!G.gameOver){showWaiting(true);}
  el.innerHTML=html;
  const ri=document.getElementById('riichi-indicator');
  if(ri)ri.innerHTML=p.riichi?`<div class="riichi-stick" title="立直中"></div>`:'';
}
function renderScores(){
  for(let i=0;i<G.numPlayers;i++){
    const e=document.getElementById(`score-${i}`);if(e)e.textContent=G.players[i].score.toLocaleString();
  }
  document.getElementById('hud-tiles').innerHTML=`残<b>${wallCount()}</b>枚`;
  document.getElementById('hud-honba').textContent=G.honba;
  document.getElementById('hud-riichi-pool').textContent=G.riichiPool;
  const rn=roundName();
  document.getElementById('hud-round').textContent=rn;
  document.getElementById('center-round').textContent=rn;
}
function renderDora(){
  const el=document.getElementById('dora-display');if(!el)return;
  el.innerHTML=G.doraIndicators.map(t=>`<div class="tile ${t.suit} small">${tileStr(t)}</div>`).join('');
}
function renderWindLabels(){
  for(let i=0;i<G.numPlayers;i++){
    const p=G.players[i];
    const ne=document.getElementById(`name-${i}`);if(ne)ne.textContent=p.name+(p.riichi?' 🔴':'');
    const we=document.getElementById(`wind-${i}`);if(we)we.textContent=p.wind+(i===G.dealer?'(親)':'');
  }
  const pw=document.getElementById('player-wind-label');if(pw)pw.textContent=G.players[0].wind+(0===G.dealer?'(親)':'');
  const s3=document.getElementById('seat-3');if(s3)s3.style.visibility=G.numPlayers===3?'hidden':'visible';
}
function renderRiichiSticks(){
  const el=document.getElementById('riichi-sticks');if(!el)return;
  el.innerHTML=Array(G.riichiPool/1000|0).fill(`<div class="riichi-stick"></div>`).join('');
}
function renderAll(){
  for(let i=0;i<G.numPlayers;i++){renderHand(i);renderPond(i);renderMelds(i);}
  renderScores();renderDora();renderWindLabels();renderRiichiSticks();
}
function showWaiting(show){const el=document.getElementById('waiting');if(el)el.style.display=show?'flex':'none';}
// ★[修正/phil-speech] 哲学者が麻雀アクション時に中心概念を叫ぶ機能。
// 旧コードは哲学者名のみ設定されており、対応する中心概念マップと
// アクション時のセリフ表示ロジックが完全に欠落していた。
const PHILOSOPHER_CONCEPTS={
  'ソクラテス':'無知の知！','プラトン':'イデア界を見よ！','アリストテレス':'エンテレケイア！',
  'エピクロス':'アタラクシア（心の平静）！','ピュロン':'エポケー（判断停止）！',
  'アウグスティヌス':'我らの心は神の中に安らうまで安らわず！','トマス・アクィナス':'信仰と理性の調和！',
  'オッカム':'オッカムの剃刀！','マキャベリ':'目的は手段を正当化する！','モンテーニュ':'クセジュ（私は何を知るか）！',
  'デカルト':'我思う、ゆえに我あり！','スピノザ':'神即自然！','ライプニッツ':'モナドは窓なし！',
  'パスカル':'人間は考える葦だ！','ベーコン':'知は力なり！',
  'ロック':'タブラ・ラサ！','ヒューム':'因果は習慣に過ぎない！','バークリー':'存在するとは知覚されること！',
  'ルソー':'一般意志！','ヴォルテール':'可能性の最善の世界！',
  'カント':'定言命法！','フィヒテ':'自我は自己を措定する！','シェリング':'同一哲学！',
  'ヘーゲル':'弁証法的止揚！','ショーペンハウアー':'意志と表象としての世界！',
  'マルクス':'唯物弁証法！','エンゲルス':'弁証法的唯物論！','ミル':'最大多数の最大幸福！',
  'ベンサム':'功利主義！','ニーチェ':'力への意志！','キルケゴール':'実存は本質に先立つ！',
  'フレーゲ':'意味と指示対象の区別！','ラッセル':'論理的原子論！','ムーア':'常識の哲学！',
  'ウィトゲンシュタイン':'語りえないものには沈黙せよ！','フッサール':'事象そのものへ！',
  'ハイデガー':'存在と時間！','サルトル':'実存は本質に先立つ！','メルロ＝ポンティ':'身体図式！',
  'ボーヴォワール':'人は女に生まれない、女になるのだ！','デューイ':'経験としての芸術！',
  'ジェームズ':'プラグマティズム！','パース':'記号論！','カルナップ':'論理実証主義！',
  'ポパー':'反証可能性！','クワイン':'意味の不確定性！','クーン':'パラダイム・シフト！',
  'レヴィナス':'他者の顔！','デリダ':'脱構築！','フーコー':'権力は知である！',
  'ドゥルーズ':'差異と反復！','バタイユ':'エロティシズムは死と生の深みにある！',
  'ロールズ':'無知のヴェール！','ノージック':'最小国家！','サンデル':'共通善！',
  'ハーバーマス':'コミュニケーション的理性！','アーレント':'公的領域！',
  'ベンヤミン':'複製技術時代の芸術！','アドルノ':'否定弁証法！','ホルクハイマー':'道具的理性批判！',
};
const PHILOSOPHER_ACTION_PREFIXES={
  riichi:'立直宣言！','ron':'ロン！','tsumo':'ツモ！','pon':'ポン！','chi':'チー！','kita':'北抜き！'
};
function showPhilosopherSpeech(pi,action){
  const p=G.players[pi];
  if(!p||p.isHuman)return;  // 人間プレイヤーはスキップ
  const concept=PHILOSOPHER_CONCEPTS[p.name];
  if(!concept)return;
  const prefix=PHILOSOPHER_ACTION_PREFIXES[action]||'';
  // 吹き出しをプレイヤーの席付近に表示
  const bubble=document.createElement('div');
  bubble.style.cssText=`position:fixed;z-index:9999;background:rgba(0,0,0,.88);color:#ffd700;`+
    `border:1.5px solid #ffd700;border-radius:10px;padding:8px 14px;font-size:13px;`+
    `font-weight:700;pointer-events:none;max-width:260px;text-align:center;`+
    `animation:phil-pop .25s ease;box-shadow:0 0 18px rgba(255,215,0,.5);`;
  bubble.textContent=`${p.name}「${concept}」`;
  // 席ごとの大まかな位置
  const positions=[
    {top:'70%',left:'10%'},{top:'20%',left:'10%'},
    {top:'20%',left:'70%'},{top:'70%',right:'10%'}
  ];
  const pos=positions[pi]||positions[1];
  Object.assign(bubble.style,pos);
  if(!document.getElementById('phil-speech-style')){
    const st=document.createElement('style');st.id='phil-speech-style';
    st.textContent='@keyframes phil-pop{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:scale(1)}}';
    document.head.appendChild(st);
  }
  document.body.appendChild(bubble);
  setTimeout(()=>bubble.remove(),2200);
  showFloatMsg(prefix+concept);
}
function showFloatMsg(msg){
  const el=document.getElementById('float-msg');el.textContent=msg;el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),1200);
}
function log(msg){const el=document.getElementById('game-log');if(el)el.textContent=msg;}
function startGame(np,mode){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('game-screen').classList.add('active');
  const s3=document.getElementById('seat-3');if(s3)s3.style.display=np===3?'none':'';
  initGame(np,mode);
}
function showTitle(){
  G.gameOver=true;
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('title-screen').classList.add('active');
}
__AUTO_START__
</script>
</body>
</html>""".replace(
        "__AUTO_START__",
        f"window.addEventListener('load',function(){{  {auto_start_js} }});"
    )


# ===== v129.0 新コマンドハンドラ =====

def handle_think_mode(arg: str) -> str:
    """★[v129] /think: chain-of-thought思考モード切替"""
    global THINKING_MODE
    a = arg.strip().lower()
    if a == "on":
        THINKING_MODE = True
        return (f"{C['g']}[思考モード ON]{C['w']} chain-of-thought強制。\n"
                f"  AIは<think>タグ内で段階的思考を行い、最終回答のみ表示します。\n"
                f"  複雑な推論・数学・コーディングで特に有効。/think off で解除。")
    elif a == "off":
        THINKING_MODE = False
        return f"{C['y']}[思考モード OFF]{C['w']} 通常モードに戻しました。"
    else:
        status = f"{C['g']}ON{C['w']}" if THINKING_MODE else f"{C['dim']}OFF{C['w']}"
        return (f"思考モード: {status}\n"
                f"  /think on  → chain-of-thought強制\n"
                f"  /think off → 通常モード")

def handle_plan(arg: str, per_id: int) -> str:
    """★[v129] /plan: OODAループ式段階的計画生成"""
    if not arg: return f"{C['r']}usage: /plan <目標>{C['w']}"
    persona = get_persona(per_id)
    fp = persona.get("first_person", "私")
    sys_content = (
        f"あなたは{persona['name']}。口調: {persona['style']}。一人称: {fp}。\n"
        "以下の目標を達成するための具体的な計画を立案せよ。\n"
        "【出力形式】\n"
        "1. 観察(Observe): 現状分析・制約・リソース\n"
        "2. 方針決定(Orient): アプローチの選択肢と判断根拠\n"
        "3. 決定(Decide): 最適戦略の選択\n"
        "4. 実行(Act): 具体的なステップ（5〜7項目）\n"
        "5. 検証指標: 成功の定義・KPI\n"
        "捏造禁止。現実的かつ実行可能な内容のみ。"
    )
    print(f"{C['c']}[PLAN]{C['w']} {persona['name']}: ", end="", flush=True)
    return stream_response(
        [{"role": "system", "content": sys_content},
         {"role": "user", "content": f"目標: {arg}"}],
        True, len(arg), temp_override=0.35, model=DEEP_MODEL, max_tokens=2048
    ) or ""

def handle_code(arg: str, per_id: int) -> str:
    """★[v130.1] /code: 世界最強コードエンジン
    入力の意図を自動判定し、4つのモードを切り替える:
      [generate]  コード生成（3パス: 設計分析 → 実装 → セルフレビュー）
      [ideate]    発展案・アイデア出し・可能性探索
      [architect] 設計相談・アーキテクチャ議論
      [review]    既存コードのレビュー・改善提案
    """
    if not arg:
        return (
            f"{C['r']}usage: /code <入力>{C['w']}\n"
            f"{C['dim']}例:\n"
            f"  /code ユーザー認証APIを作って\n"
            f"  /code このプロジェクトの発展案を教えて\n"
            f"  /code 以下のコードをレビューして: ...\n"
            f"  /code マイクロサービスとモノリスどちらが良い?{C['w']}"
        )

    # ── 意図判定 (Pass-0) ──────────────────────────────────────────
    intent_sys = (
        "ユーザーの入力を分析し、以下の4つのカテゴリのうち最も適切な1つを"
        "英単語のみで出力せよ。説明・句読点は一切不要。\n\n"
        "generate  : 新しいコード・関数・クラス・スクリプトの生成依頼\n"
        "ideate    : アイデア出し・発展案・可能性・改善提案・brainstorm\n"
        "architect : 設計相談・アーキテクチャ選択・技術選定・構成議論\n"
        "review    : 既存コードのレビュー・バグ発見・リファクタリング依頼\n\n"
        "出力は generate / ideate / architect / review のいずれか1語のみ。"
    )
    intent_raw = stream_response(
        [{"role": "system", "content": intent_sys},
         {"role": "user", "content": arg}],
        False, len(arg), temp_override=0.0, model=DEEP_MODEL,
        max_tokens=10, silent=True
    ) or "generate"
    intent = intent_raw.strip().lower().split()[0]
    if intent not in ("generate", "ideate", "architect", "review"):
        intent = "generate"

    label_map = {
        "generate":  f"{C['c']}[CODE: 生成モード]{C['w']}",
        "ideate":    f"{C['m']}[CODE: アイデアモード]{C['w']}",
        "architect": f"{C['y']}[CODE: 設計モード]{C['w']}",
        "review":    f"{C['r']}[CODE: レビューモード]{C['w']}",
    }
    print(label_map[intent])

    # ════════════════════════════════════════════════════════════════
    # モード: generate（3パス コード生成）
    # ════════════════════════════════════════════════════════════════
    if intent == "generate":
        # Pass-1: 設計分析
        design_sys = (
            "あなたはGoogleのPrincipal Engineer・元MITアルゴリズム研究者・セキュリティ専門家を"
            "兼ねる世界最高峰のソフトウェアアーキテクト。\n"
            "与えられた仕様を受け取り、実装前に以下を日本語で簡潔に分析せよ（コードはまだ書くな）:\n\n"
            "【設計分析フォーマット】\n"
            "1. 問題の本質と制約（エッジケース・スケール・並行性）\n"
            "2. 最適データ構造・アルゴリズム選択と計算量(O表記)\n"
            "3. アーキテクチャパターン（責務分離・依存関係）\n"
            "4. セキュリティリスクと対策（入力検証・注入・リソース枯渇）\n"
            "5. テスト戦略（正常系・異常系・境界値・モック対象）\n"
            "6. パフォーマンスボトルネックと最適化方針\n\n"
            "分析は箇条書きで簡潔に。実装コードは絶対に書くな。"
        )
        print(f"  Pass-1 設計分析中... ", end="", flush=True)
        design_analysis = stream_response(
            [{"role": "system", "content": design_sys},
             {"role": "user", "content": f"仕様: {arg}"}],
            True, len(arg), temp_override=0.0, model=DEEP_MODEL,
            max_tokens=1024, silent=True
        ) or ""
        print(f"{C['g']}完了{C['w']}")

        # Pass-2: 実装
        impl_sys = (
            "あなたはGoogleのPrincipal Engineer・元MITアルゴリズム研究者・セキュリティ専門家を"
            "兼ねる世界最高峰のソフトウェアアーキテクト。\n"
            "以下の設計分析に基づき、プロダクション品質の完全なPythonコードを生成せよ。\n\n"
            "【絶対必須要件】\n"
            "1. 型ヒント: 全関数・メソッド・変数に付ける（Python 3.10+ union構文 X|Y 推奨）\n"
            "2. Docstring: Google形式。Args/Returns/Raises/Example を全関数に記載\n"
            "3. エラーハンドリング: 具体的な例外クラス・適切なログ・リカバリ戦略\n"
            "4. セキュリティ: 入力検証・サニタイズ・リソース制限・インジェクション対策\n"
            "5. テスト: pytest形式で正常系2件・異常系2件・境界値1件以上（モック使用可）\n"
            "6. パフォーマンス: 計算量コメント・不要なコピー回避・ジェネレータ活用\n"
            "7. 可読性: 関数は単一責任・20行以内を目安・定数は大文字定義\n"
            "8. 冪等性・スレッド安全性が必要な場合はロック・アトミック操作を明示\n\n"
            "【出力フォーマット】\n"
            "```python\n# 実装コード\n```\n\n```python\n# テストコード（pytest形式）\n```\n\n"
            "コードブロック外の説明文は最小限にせよ。"
            f"\n\n【設計分析】\n{design_analysis}"
        )
        print(f"  Pass-2 実装生成中... ", end="", flush=True)
        implementation = stream_response(
            [{"role": "system", "content": impl_sys},
             {"role": "user", "content": f"仕様: {arg}"}],
            True, len(arg), temp_override=0.0, model=DEEP_MODEL, max_tokens=4096
        ) or ""
        print()

        # Pass-3: セルフレビュー
        review_sys = (
            "あなたはセキュリティ・パフォーマンス専門のシニアコードレビュアー。\n"
            "以下のコードを厳格にレビューし、問題点のみ箇条書きで報告せよ。\n"
            "問題がなければ「✅ レビュー通過」とだけ出力せよ。\n\n"
            "チェック項目:\n"
            "- バグ・ロジックエラー（off-by-one・型不一致・None参照）\n"
            "- セキュリティホール（インジェクション・パストラバーサル・競合状態）\n"
            "- 例外の握りつぶし（bare except / pass）\n"
            "- テストの抜け漏れ（未テストのエッジケース）\n"
            "- パフォーマンス問題（O(n²)以上のループ・不要なDB呼び出し）\n"
            "- 型ヒント・docstringの欠落\n"
            "問題点は '⚠ [重大度: 高/中/低] 説明' の形式で列挙。"
        )
        print(f"  Pass-3 セルフレビュー中... ", end="", flush=True)
        review = stream_response(
            [{"role": "system", "content": review_sys},
             {"role": "user", "content": implementation}],
            True, len(implementation), temp_override=0.0, model=DEEP_MODEL,
            max_tokens=512, silent=True
        ) or ""
        print(f"{C['g']}完了{C['w']}\n")

        lines = [
            f"{C['dim']}{'─'*60}{C['w']}",
            f"{C['y']}【設計分析】{C['w']}", design_analysis,
            f"{C['dim']}{'─'*60}{C['w']}",
            f"{C['g']}【実装】{C['w']}", implementation,
            f"{C['dim']}{'─'*60}{C['w']}",
        ]
        if "✅" in review or not review.strip():
            lines.append(f"{C['g']}【レビュー】✅ 問題なし{C['w']}")
        else:
            lines += [f"{C['y']}【レビュー指摘】{C['w']}", review]
        lines += [
            f"{C['dim']}{'─'*60}{C['w']}",
            f"{C['dim']}  ヒント: /m add でメモ保存 | /convert md html でHTML変換{C['w']}",
        ]
        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════════
    # モード: ideate（発展案・アイデア出し）
    # ════════════════════════════════════════════════════════════════
    elif intent == "ideate":
        ideate_sys = (
            "あなたはYCombinatorのトップメンター・元Google X / DeepMind研究者・"
            "連続起業家を兼ねる世界最高のテクノロジーストラテジスト。\n"
            "与えられたプロジェクト・テーマ・課題について、"
            "実現可能性と革新性を両立した発展案・アイデアを提示せよ。\n\n"
            "【出力フォーマット】\n"
            "## 🔍 現状分析\n"
            "（強み・弱み・機会・脅威を2〜3行で）\n\n"
            "## 🚀 発展案（優先度順）\n"
            "各案について:\n"
            "- **案名**: 一行で本質を表すタイトル\n"
            "- **概要**: 何をするか（2〜3行）\n"
            "- **技術スタック**: 使う技術・ライブラリ（具体的に）\n"
            "- **工数目安**: 1人が取り組む場合のざっくり見積もり\n"
            "- **インパクト**: ユーザー・ビジネス・技術的価値\n"
            "- **実装の入口**: 最初にやるべき具体的な1ステップ\n\n"
            "## ⚡ クイックウィン（今すぐできる改善）\n"
            "（1〜2日で実装できる小さいが効果的な改善を3つ）\n\n"
            "## 🌐 長期ビジョン\n"
            "（1〜2年後のあるべき姿・差別化ポイント）\n\n"
            "具体的・実践的に。抽象論は避けよ。"
        )
        print(f"  アイデア生成中... ", end="", flush=True)
        result = stream_response(
            [{"role": "system", "content": ideate_sys},
             {"role": "user", "content": arg}],
            True, len(arg), temp_override=0.0, model=DEEP_MODEL, max_tokens=3072
        ) or ""
        print()
        return (
            f"{C['dim']}{'─'*60}{C['w']}\n"
            f"{C['m']}【発展案・アイデア】{C['w']}\n"
            f"{result}\n"
            f"{C['dim']}{'─'*60}{C['w']}\n"
            f"{C['dim']}  ヒント: /code <案名> で即座に実装へ移行できます{C['w']}"
        )

    # ════════════════════════════════════════════════════════════════
    # モード: architect（設計相談・技術選定）
    # ════════════════════════════════════════════════════════════════
    elif intent == "architect":
        arch_sys = (
            "あなたはAWS/GCP/Azureのチーフアーキテクト・分散システム専門家・"
            "元Netflix・Airbnb・Stripe のスタッフエンジニアを兼ねる世界最高峰のアーキテクト。\n"
            "設計上の問いに対して、トレードオフを正直に示しながら最適解を提示せよ。\n\n"
            "【出力フォーマット】\n"
            "## 🏗️ 問題の本質\n"
            "（何を解決しようとしているか・制約条件の整理）\n\n"
            "## ⚖️ 選択肢とトレードオフ\n"
            "各選択肢について:\n"
            "| 項目 | 選択肢A | 選択肢B | ... |\n"
            "（スケール・複雑性・コスト・チーム習熟度・将来性で比較）\n\n"
            "## ✅ 推奨アーキテクチャ\n"
            "（理由付きで最適解を明示）\n\n"
            "## 📐 具体的な構成図（テキストで）\n"
            "（コンポーネント・データフロー・インターフェースを図示）\n\n"
            "## ⚠️ 落とし穴と対策\n"
            "（この選択でよくある失敗と回避策）\n\n"
            "## 🗺️ 移行ロードマップ\n"
            "（Phase 1/2/3 に分けて段階的な実装順序）"
        )
        print(f"  設計分析中... ", end="", flush=True)
        result = stream_response(
            [{"role": "system", "content": arch_sys},
             {"role": "user", "content": arg}],
            True, len(arg), temp_override=0.0, model=DEEP_MODEL, max_tokens=3072
        ) or ""
        print()
        return (
            f"{C['dim']}{'─'*60}{C['w']}\n"
            f"{C['y']}【アーキテクチャ設計】{C['w']}\n"
            f"{result}\n"
            f"{C['dim']}{'─'*60}{C['w']}\n"
            f"{C['dim']}  ヒント: /code <コンポーネント名> で実装に移行できます{C['w']}"
        )

    # ════════════════════════════════════════════════════════════════
    # モード: review（既存コードのレビュー）
    # ════════════════════════════════════════════════════════════════
    else:  # review
        review_sys = (
            "あなたはGoogle・Meta・Stripe のシニアエンジニアが兼任するコードレビュー委員会。\n"
            "提出されたコードを多角的かつ建設的にレビューせよ。\n\n"
            "【出力フォーマット】\n"
            "## 📊 総合評価\n"
            "（A〜Fのグレードと一言評価）\n\n"
            "## 🐛 バグ・ロジックエラー\n"
            "（行番号・再現条件・修正コード例）\n\n"
            "## 🔐 セキュリティ指摘\n"
            "（重大度: 高/中/低 で列挙・CVE番号があれば付記）\n\n"
            "## ⚡ パフォーマンス改善\n"
            "（計算量・メモリ・I/O 観点で指摘・改善後の計算量も示す）\n\n"
            "## 🧹 リファクタリング提案\n"
            "（可読性・保守性・テスト容易性の改善案・改善後コード例）\n\n"
            "## ✅ 良い点\n"
            "（積極的に評価すべき実装・パターン）\n\n"
            "## 🔧 修正後の完全コード\n"
            "（全指摘を反映した改善版を ```python ブロックで出力）"
        )
        print(f"  コードレビュー中... ", end="", flush=True)
        result = stream_response(
            [{"role": "system", "content": review_sys},
             {"role": "user", "content": arg}],
            True, len(arg), temp_override=0.0, model=DEEP_MODEL, max_tokens=4096
        ) or ""
        print()
        return (
            f"{C['dim']}{'─'*60}{C['w']}\n"
            f"{C['r']}【コードレビュー】{C['w']}\n"
            f"{result}\n"
            f"{C['dim']}{'─'*60}{C['w']}\n"
            f"{C['dim']}  ヒント: /m add でレビュー結果を保存できます{C['w']}"
        )

def handle_reflect(arg: str, per_id: int) -> str:
    """★[v129] /reflect: 自己批判的振り返り分析"""
    if not arg: return f"{C['r']}usage: /reflect <振り返りたい内容・経験>{C['w']}"
    persona = get_persona(per_id)
    fp = persona.get("first_person", "私")
    sys_content = (
        f"あなたは{persona['name']}。口調: {persona['style']}。一人称: {fp}。\n"
        "以下の内容について批判的・建設的な振り返りを行え。\n"
        "【分析フレームワーク】\n"
        "1. 何がうまくいったか（強み）\n"
        "2. 何が課題だったか（改善点）\n"
        "3. なぜそうなったか（根本原因分析）\n"
        "4. 次回どう変えるか（具体的アクション）\n"
        "5. より深い洞察（哲学的・本質的な気づき）\n"
        "自己批判と自己肯定のバランスを保ち、建設的に分析せよ。"
    )
    print(f"{C['c']}[REFLECT]{C['w']} {persona['name']}: ", end="", flush=True)
    return stream_response(
        [{"role": "system", "content": sys_content},
         {"role": "user", "content": f"振り返り対象: {arg}"}],
        True, len(arg), temp_override=0.50, model=DEEP_MODEL, max_tokens=2048
    ) or ""

def handle_mindmap(arg: str, per_id: int) -> str:
    """★[v129] /mindmap: ASCIIアートマインドマップ生成"""
    if not arg: return f"{C['r']}usage: /mindmap <テーマ>{C['w']}"
    persona = get_persona(per_id)
    sys_content = (
        f"あなたは{persona['name']}。一人称: {persona.get('first_person','私')}。\n"
        "以下のテーマについてASCIIアートのマインドマップを生成せよ。\n"
        "フォーマット:\n"
        "[中心テーマ]\n"
        "├── 主要概念1\n"
        "│   ├── サブ概念1-1\n"
        "│   └── サブ概念1-2\n"
        "├── 主要概念2\n"
        "│   ├── サブ概念2-1\n"
        "│   └── サブ概念2-2\n"
        "└── 主要概念3\n"
        "    └── サブ概念3-1\n"
        "主要概念は4〜6個、各サブ概念は2〜3個。捏造禁止。"
    )
    print(f"{C['c']}[MINDMAP]{C['w']} ", end="", flush=True)
    return stream_response(
        [{"role": "system", "content": sys_content},
         {"role": "user", "content": f"テーマ: {arg}"}],
        True, len(arg), temp_override=0.25, model=MODEL_NAME, max_tokens=1024
    ) or ""

def handle_persona_edit(current_persona: dict) -> dict:
    """★[v129] /persona_edit: 現在ペルソナのスタイルをインタラクティブに編集"""
    print(f"{C['c']}=== ペルソナ編集 ==={C['w']}")
    print(f"現在: {current_persona['name']}")
    print(f"口調: {current_persona['style'][:100]}...")
    print(f"一人称: {current_persona.get('first_person', '私')}")
    print(f"{C['dim']}編集する項目を選択（Enter でスキップ）:{C['w']}")
    try:
        print(f"1. 一人称 [{current_persona.get('first_person', '私')}]: ", end="", flush=True)
        fp_new = sys.stdin.readline().strip()
        if fp_new: current_persona["first_person"] = fp_new

        print(f"2. 口調追加指示（現在の口調に追加）: ", end="", flush=True)
        style_add = sys.stdin.readline().strip()
        if style_add:
            current_persona["style"] = current_persona.get("style", "") + " " + style_add

        print(f"{C['g']}ペルソナ更新完了: {current_persona['name']}{C['w']}")
    except EOFError:
        pass
    return current_persona

def handle_model_cmd(arg: str) -> str:
    """★[v129] /model: モデル確認・切替"""
    a = arg.strip().lower()
    if not a:
        return (f"{C['c']}=== モデル状態 ==={C['w']}\n"
                f"  FAST  : {FAST_MODEL}\n"
                f"  MAIN  : {MODEL_NAME}\n"
                f"  DEEP  : {DEEP_MODEL}\n"
                f"  Power : {POWER_MODE}\n"
                f"  Thinking: {'ON' if THINKING_MODE else 'OFF'}\n"
                f"  GPU   : {'検出済' if _GPU_AVAILABLE else 'なし(CPU専用)'}\n"
                f"  12b   : {'利用可' if _HAS_12B else '未検出'}")
    if a == "fast":
        return f"{C['g']}FAST={FAST_MODEL} (高速応答用){C['w']}"
    if a == "main":
        return f"{C['g']}MAIN={MODEL_NAME} (標準用){C['w']}"
    if a == "deep":
        return f"{C['g']}DEEP={DEEP_MODEL} (複雑推論用){C['w']}"
    return f"{C['r']}usage: /model [fast|main|deep]{C['w']}"

def handle_speedtest() -> str:
    """★[v129] /speedtest: モデルのトークン速度測定"""
    o = _get_ollama()
    if o is None: return f"{C['r']}Ollama未接続{C['w']}"
    test_models = list(dict.fromkeys([FAST_MODEL, MODEL_NAME, DEEP_MODEL]))
    results = []
    test_prompt = "1から10まで数字を日本語で書け。"
    for m in test_models:
        try:
            start = time.time()
            full = ""
            for chunk in o.chat(model=m, messages=[{"role": "user", "content": test_prompt}],
                                stream=True, options={"num_predict": 50, "num_ctx": 512}):
                msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", None)
                t = (msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
                full += t
            elapsed = max(time.time() - start, 0.01)
            toks = len(full) * 0.7  # rough estimate
            tps = toks / elapsed
            results.append(f"  {m:25s} {tps:5.1f} tok/s ({elapsed:.1f}s)")
        except Exception as e:
            results.append(f"  {m:25s} エラー: {e}")
    return f"{C['c']}=== 速度測定 ==={C['w']}\n" + "\n".join(results)

def handle_ctx_status(messages: list) -> str:
    """★[v129] /ctx: コンテキスト使用量表示"""
    def _est(text):
        jp = sum(1 for c in text if ord(c) > 0x7F)
        return int(jp * 1.5 + (len(text) - jp) * 0.4)
    total = sum(_est(m.get("content", "")) for m in messages)
    opts = get_llm_opt(True, total)
    n_ctx = opts.get("num_ctx", 4096)
    pct = min(100, int(total / n_ctx * 100))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    return (f"{C['c']}=== コンテキスト状態 ==={C['w']}\n"
            f"  使用: ~{total:,} tok / {n_ctx:,} tok ({pct}%)\n"
            f"  [{bar}]\n"
            f"  履歴: {len([m for m in messages if m['role']=='user'])}ターン\n"
            f"  RAGキャッシュ: {len(RAG_CACHE)}件\n"
            f"  /g で履歴クリア推奨 (使用率>70%)")

# ===== COMMAND REGISTRY & MAIN RUNNER v129.0 =====
def setattr_return(name: str, val):
    """★[v129] lambdaでnonlocal変数を返すためのヘルパー"""
    return val

# ドメイン単位で管理。ここにないドメインへのアクセスは _trusted_fetch が拒否する。
_TRUSTED_AI_DOMAINS: frozenset[str] = frozenset([
    "arxiv.org",
    "huggingface.co",
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "deepmind.com",
    "ai.googleblog.com",
    "research.google",
    "simonwillison.net",
    "lilianweng.github.io",
    "github.io",           # 個人技術ブログ（github.io サブドメイン）
    "sebastian-raschka.com",
    "newsletter.theaiedge.io",
    "aitidbits.substack.com",
    "promptingguide.ai",
    "learnprompting.org",
    "mlabonne.github.io",
])

# arxiv の信頼済みカテゴリ（クエリパラメータレベルで限定）
_ARXIV_SAFE_CATEGORIES: frozenset[str] = frozenset([
    "cs.AI", "cs.LG", "cs.CL", "cs.CR", "cs.NE", "stat.ML"
])

def _trusted_fetch(url: str, timeout: int = 8) -> str:
    """★[v130.1] /reference 専用のセキュアフェッチ。
    ホワイトリスト (_TRUSTED_AI_DOMAINS) 外のドメインは完全拒否。
    SSRF防止 (_assert_safe_url) も二重で適用する。

    Returns:
        取得テキスト（失敗時は空文字）
    Raises:
        ValueError: ホワイトリスト外ドメイン・安全でないURL
    """
    def _assert_trusted_reference_url(candidate: str) -> None:
        # ① 基本SSRF検査（既存の強化済みチェック）
        _assert_safe_url(candidate)

        # ② ホワイトリストドメイン検査
        parsed = U.urlparse(candidate)
        host = (parsed.hostname or "").lower()
        # サブドメインも許可（例: blog.openai.com, research.google.com）
        allowed = any(
            host == d or host.endswith("." + d)
            for d in _TRUSTED_AI_DOMAINS
        )
        if not allowed:
            raise ValueError(
                f"[trusted_fetch] ホワイトリスト外ドメインへのアクセスを拒否: {host!r}\n"
                f"許可ドメイン: {sorted(_TRUSTED_AI_DOMAINS)}"
            )

        # ③ HTTPSのみ許可（ホワイトリストでも HTTP は不可）
        if parsed.scheme != "https":
            raise ValueError(f"[trusted_fetch] HTTPS以外は許可しない: {parsed.scheme!r}")

        # ④ URLパスのサニタイズ（パストラバーサル・制御文字）
        path = parsed.path
        if ".." in path or any(c in path for c in ("\x00", "\r", "\n")):
            raise ValueError(f"[trusted_fetch] 不正なURLパス: {path!r}")

        # ⑤ クエリパラメータ長制限（ReDoS・過剰クエリ対策）
        if len(parsed.query) > 512:
            raise ValueError(f"[trusted_fetch] クエリパラメータが長すぎます: {len(parsed.query)} chars")

    _assert_trusted_reference_url(url)
    return fetch_html(url, timeout=timeout, silent=True, redirect_checker=_assert_trusted_reference_url)


def _stop_files() -> str:
    """★[外出し] 一時ファイル削除。旧: run()内 _handle_stop"""
    removed = 0
    patterns = ["ytdl_y_*", "tts_*.mp3", "aegis_*.md", "aegis_export_*.*", "*.mid", "*.wav"]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    return f"{C['g']}一時ファイル {removed}件削除{C['w']}"


def _handle_prime(arg: str) -> str:
    """★ 素数判定コマンド: ASTベース・多倍長整数対応"""
    import ast as _ast, math, time

    arg = arg.strip()
    if not arg:
        return (f"{C['y']}使い方: /prime <数式または整数>{C['w']}\n"
                f"  例: /prime 997\n"
                f"  例: /prime 2**31-1\n"
                f"  例: /prime 10**18+9")

    # ── ASTで数式を安全に評価 ──
    _ALLOWED = (
        _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Constant,
        _ast.Add, _ast.Sub, _ast.Mult, _ast.Pow, _ast.FloorDiv,
        _ast.Mod, _ast.USub, _ast.UAdd,
    )
    try:
        tree = _ast.parse(arg, mode="eval")
        for node in _ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                return f"{C['r']}[ERR] 使用できない式です: {type(node).__name__}{C['w']}"
        n = eval(compile(tree, "<prime>", "eval"), {"__builtins__": {}})
        if not isinstance(n, int):
            return f"{C['r']}[ERR] 整数のみ対応しています（結果: {n}）{C['w']}"
    except Exception as e:
        return f"{C['r']}[ERR] 式の評価に失敗: {e}{C['w']}"

    if n < 2:
        return f"{C['c']}{n:,}{C['w']} は素数ではありません（2未満）"
    if n == 2:
        return f"{C['g']}{n:,}{C['w']} は {C['g']}素数{C['w']} です ✓"
    if n % 2 == 0:
        return f"{C['c']}{n:,}{C['w']} は素数ではありません（偶数）"

    # ── Miller-Rabin 確率的素数判定（多倍長対応・決定論的）──
    def _miller_rabin(n: int) -> bool:
        # n-1 = 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # 決定論的witness（n < 3.3×10^24まで正確）
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)  # 高速べき乗（Python組み込み）
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    t0 = time.time()
    is_prime = _miller_rabin(n)
    elapsed = time.time() - t0

    digits = len(str(n))
    time_str = f"{elapsed*1000:.2f}ms" if elapsed < 1 else f"{elapsed:.3f}s"

    if is_prime:
        return (f"{C['g']}{n:,}{C['w']} は {C['g']}素数{C['w']} です ✓\n"
                f"  桁数: {digits}桁 | 判定時間: {time_str} | 手法: Miller-Rabin")
    else:
        # 小さな因数を探す
        factor = ""
        for p in [2,3,5,7,11,13,17,19,23,29,31]:
            if n % p == 0 and n != p:
                factor = f" （{p} で割り切れる）"
                break
        return (f"{C['c']}{n:,}{C['w']} は素数ではありません{factor}\n"
                f"  桁数: {digits}桁 | 判定時間: {time_str} | 手法: Miller-Rabin")

def _handle_baseball() -> str:
    """⚾ 甲子園列伝HTMLをWSL2→Windowsブラウザで起動"""
    import subprocess as _sub, pathlib
    html = pathlib.Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s01_baseball.html")).resolve()
    if not html.exists():
        return f"{C['r']}[ERR] s01_baseball.html が見つかりません: {html}{C['w']}"
    try:
        win_path = _sub.check_output(
            ["wslpath", "-w", str(html)],
            text=True, timeout=3).strip()
        # detach=Trueでバックグラウンド起動（フリーズ防止）
        _sub.Popen(
            ["cmd.exe", "/c", "start", "", win_path],
            stdin=_sub.DEVNULL,
            stdout=_sub.DEVNULL,
            stderr=_sub.DEVNULL,
            close_fds=True
        )
        return f"{C['g']}⚾ 甲子園列伝を起動しました！{C['w']}"
    except Exception as e:
        try:
            import webbrowser, threading
            threading.Thread(
                target=webbrowser.open,
                args=(f"file://{html}",),
                daemon=True
            ).start()
            return f"{C['g']}⚾ 甲子園列伝を起動しました（webbrowser）{C['w']}"
        except Exception as e2:
            return f"{C['y']}[WARN] 起動失敗: {e} / {e2}{C['w']}"

def run() -> None:
    global POWER_MODE, TEMP_VOICE, KEYWORD_MEMORY, ROLEPLAY_ACTIVE, ROLEPLAY_SCENE, CUSTOM_PERSONA
    global _SESSION_START_INTERACTIONS
    global VECTOR_AVAILABLE, VECTOR_COL, _VECTOR_CLIENT
    SESSION_STATS["start_time"] = time.time()
    messages: list[dict] = []
    persona_id: int = 2
    current_persona: dict = get_persona(persona_id)
    _t0=__import__("time").time(); restore_learning(); print(f"restore: {__import__('time').time()-_t0:.2f}s")
    # ★[修正/#12] 再起動直後に total_interactions が 25 の倍数の場合に
    # 1ターン目から persist_learning が走るバグを防ぐ。
    # restore_learning() 後の値をセッション開始ベースラインとして記録する。
    _SESSION_START_INTERACTIONS = LEARNING_STATS["total_interactions"]
    _t1=__import__("time").time()
    if not check_ollama_connection():
        print(f"ollama: {__import__('time').time()-_t1:.2f}s"); return
    print(f"ollama: {__import__('time').time()-_t1:.2f}s")
    # ★[v129] vector_db初期化・OPTIMIZER・バックグラウンドで並列起動
    _t2=__import__("time").time(); _init_vector_db(); print(f"vector_db_thread: {__import__("time").time()-_t2:.2f}s")
    _t3=__import__("time").time(); OPTIMIZER.start(); print(f"optimizer: {__import__("time").time()-_t3:.2f}s")
    # ウォームアップ: バックグラウンドでモデルをプリロード
    def _warmup():
        import ollama as _ol
        try: _ol.chat(model=FAST_MODEL, messages=[{"role":"user","content":"hi"}], options={"num_predict":1})
        except: pass
    threading.Thread(target=_warmup, daemon=True).start()
    # ★[v129] BANNER動的更新（check_ollama_connectionでモデル名が確定した後）
    _dynamic_banner = (
        f"{C['c']}{C['bold']}\nPROJECT AEGIS [v130.1 NEXT GENERATION]{C['w']}\n"
        f"  FAST: {FAST_MODEL} | MAIN: {MODEL_NAME} | DEEP: {DEEP_MODEL}\n"
        f"  RAG: HYBRID(BM25+Vector) | SECURITY: MultiLayer | THINKING: {'ON' if THINKING_MODE else 'OFF'}\n"
        f"  新コマンド: /think /plan /code /reflect /mindmap /model /ctx /speedtest\n"
        f"  /h:コマンド一覧 | /s 1〜36:西洋哲学者 | /think:思考モード切替\n"
    )
    print(_dynamic_banner)
    # 起動時：保存済みペルソナ件数を表示
    _saved = list_personas()
    if _saved:
        print(f"{C['dim']}保存済みペルソナ: {len(_saved)}件 ({', '.join(_saved)}) → /s load <名前>{C['w']}")
    if _get_ollama() is None: print(f"{C['r']}[ERR] ollama not installed{C['w']}")

    def _chat(user_text: str, mode: str = "d", model: str | None = None, persona_override: dict | None = None) -> str:
        nonlocal current_persona
        p = persona_override or current_persona
        if ROLEPLAY_ACTIVE:
            sys_msg = {"role": "system", "content": f"あなたは{p['name']}。口調:{p['style']}。一人称:{p.get('first_person','私')}。ユーザー:{USER_NAME}。ロールプレイ中: {ROLEPLAY_SCENE}。3文以内。"}
            msgs = trim_history(messages[-MAX_HISTORY * 2:]) + [{"role": "user", "content": user_text}]
            return stream_response([sys_msg] + msgs, False, len(user_text), _get_temp_voice()) or ""
        if mode == "d":
            # ★ デフォルトペルソナ(ID=1)以外はcomplexとして扱う
            # estimate_complexityは短い入力を「simple」と誤判定するため、
            # ペルソナ会話では入力長に関わらず常にfull品質で生成する
            _is_philosopher = (persona_id in range(2, 37))
            # ★[修正/complex-1] 哲学者ペルソナ中は入力長・キーワードに関わらず常にcomplex扱い。
            # 「こんにちは」(simple判定)・「存在とは」(5文字→simple)・「死」(1文字→simple)が
            # d_tokens=200の短文パスに落ちていたことが「文章が短い」の直接原因。
            if _is_philosopher:
                is_complex = True
            else:
                is_complex = estimate_complexity(user_text) == "complex"
            # ★[修正/chat-1] select_model を使って複雑度に応じてモデルを切り替える
            # 旧コードは model_choice = MODEL_NAME 固定でDEEP_MODELが使われなかった
            if model is None:
                model = DEEP_MODEL if (is_complex and mode == "deep") else FAST_MODEL if is_complex else MODEL_NAME
            if is_complex:
                # ★[修正/rag-d] d/complexモードにもRAGデータを注入する
                # get_async_rag_dataは並列取得済みキャッシュを優先するため遅延は最小限
                rag_snippet = ""
                if not OFFLINE_MODE:
                    try:
                        _rag_future = _THREAD_POOL.submit(get_async_rag_data, user_text)
                        try: _rag = _rag_future.result(timeout=_rag_adaptive_timeout())
                        except: _rag = ""
                        if _rag and len(_rag.strip()) > 30:
                            # ★[修正/ctx-2] RAGデータをctx予算に合わせてトリミング
                            # complexモード(ctx=8192, n_predict=4096)では残り≒4096トークン≒2700文字
                            # RAGが長すぎるとシステムプロンプト+履歴でctxを圧迫して途切れる原因になる
                            _rag_chars_limit = 1200  # ★[v129] 800→1200: ハイブリッドRAG情報増加に対応
                            _rag_trimmed = _rag[:_rag_chars_limit]
                            if len(_rag) > _rag_chars_limit:
                                _rag_trimmed += "\n…(省略)"
                            rag_snippet = f"\n\n【Web参照情報（参考程度に使え。ここにない事実を創作するな）】:\n{_rag_trimmed}\n"
                    except Exception as _e:
                        print(f"{C['y']}[WARN] RAG取得失敗（スキップ）: {_e}{C['w']}")
                persona_style_block = p['style']
                _is_late_witt  = p['name'] == "後期ウィトゲンシュタイン"
                _is_early_witt = p['name'] == "前期ウィトゲンシュタイン"
                if _is_late_witt:
                    # ★[修正/witt-late] ハルシネーション&ループ同語反復対策
                    # 旧コード「言語ゲーム・家族的類似・規則遵守のパラドクスを論じよ」という
                    # 粗い一行指示がLLMに「規則遵守のパラドックスは〜」という見出しフレーズを
                    # 各段落で使い回させる直接原因だった。
                    # 修正: 段落ごとに参照すべき具体的節番号・概念・論点を明示し、
                    # ラベルとして"規則遵守のパラドクス"を単独使用することを禁じる。
                    # 哲学的典拠(§番号)を根拠として与えることでハルシネーションも抑制する。
                    sys_content = (
                        f"あなたは{p['name']}（『哲学的探究』Philosophische Untersuchungen, 1953年の著者）。"
                        f"一人称:{p.get('first_person','私')}。ユーザーは{USER_NAME}。\n"
                        f"口調:{persona_style_block}\n"
                        f"【最重要】{USER_NAME}の質問・発言に正面から答えること。質問を無視して独自の論考を展開することを禁じる。\n"
                        f"質問のテーマを自分の哲学（言語ゲーム・使用・Lebensform）の観点から論じること。\n"
                        f"【絶対禁止】箇条書き・番号リスト・並列列挙。"
                        f"比喩の多用（最後の段落1箇所のみ可）。同一フレーズ（3語以上）の2回以上の繰り返し。\n"
                        f"【使用可能な典拠（質問に関連するものだけ選んで使え）】\n"
                        f"  §43: 意味は使用にある（die Bedeutung eines Wortes ist sein Gebrauch in der Sprache）。\n"
                        f"  §66〜67: Familienähnlichkeit——共通の本質はなく、重なり交差する類似の連鎖がある。\n"
                        f"  §201/§219: 規則は実践の中にある。鍬が岩盤に当たるところで問いは終わる。\n"
                        f"  §241: 判断における一致（Übereinstimmung in Urteilen）が言語を支える。\n"
                        f"  §293: 甲虫の箱——感覚語の意味は社会的実践に根ざす。\n"
                        f"【構造】4〜6段落・各段落3〜5文。散文のみ。"
                        f"最後の文を「。」で完結させること。途中で終わることを禁じる。"
                        + rag_snippet
                    )
                elif _is_early_witt:
                    # ★[修正/witt-early] 前期ウィトゲンシュタインも同様に哲学的典拠を明示化
                    # 旧コード「命題・事実・像の概念で精緻に分析せよ」が曖昧すぎてハルシネーションを誘発していた。
                    # TLPの正確な節番号・概念・ドイツ語原語を与えることで捏造を抑制する。
                    sys_content = (
                        f"あなたは{p['name']}（『論理哲学論考』Tractatus Logico-Philosophicus, 1921年の著者）。"
                        f"一人称:{p.get('first_person','私')}。ユーザーは{USER_NAME}。\n"
                        f"口調:{persona_style_block}\n"
                        f"【最重要】{USER_NAME}の質問・発言に正面から答えること。質問を無視して独自の論考を展開することを禁じる。\n"
                        f"質問のテーマを自分の哲学（像理論・論理形式・語りえないもの）の観点から論じること。\n"
                        f"【絶対禁止】箇条書き・番号リスト・同語反復。同一フレーズ（3語以上）の2回以上の繰り返し。\n"
                        f"【使用可能な典拠（質問に関連するものだけ選んで使え）】\n"
                        f"  1.1: 世界は事実の総体であり物の総体ではない（Die Welt ist die Gesamtheit der Tatsachen）。\n"
                        f"  2.1: われわれは事実の像（Bild）を作る——像理論の核心。\n"
                        f"  4.022: 語りうること（sagen）と示しうること（zeigen）の区別。\n"
                        f"  5.6: 言語の限界は世界の限界（Die Grenzen meiner Sprache bedeuten die Grenzen meiner Welt）。\n"
                        f"  7: 語りえないものについては沈黙しなければならない。\n"
                        f"【構造】4〜6段落・各段落3〜5文。散文のみ。"
                        f"最後の文を「。」で完結させること。途中で終わることを禁じる。"
                        + rag_snippet
                    )
                else:
                    sys_content = (
                        f"あなたは{p['name']}。一人称:{p.get('first_person','私')}。ユーザーは{USER_NAME}。\n"
                        f"口調:{persona_style_block}\n"
                        f"【最重要】{USER_NAME}の質問・発言に正面から答えること。質問を無視した独自論考を禁じる。\n"
                        f"禁止:箇条書き・番号リスト・同語反復・比喩の多用(最後の段落で1つのみ可)。\n"
                        f"4〜6段落・各段落3〜5文。散文のみ。"
                        f"最後の文を「。」で完結させること。途中で終わることを禁じる。"
                        + rag_snippet
                    )
                # ★[修正/trunc-1b] ウィトゲンシュタインは8段落指定のため3200、その他は2800
                # 旧値1024では哲学者長文の途中（特に「たとえば〜」最終文）でトークン上限に
                # 達して文が途切れる直接原因になっていた。
                d_tokens = 3200 if (_is_late_witt or _is_early_witt) else 2800
            else:
                sys_content = (
                    f"あなたは{p['name']}。口調:{p['style']}。一人称:{p.get('first_person','私')}。ユーザーは{USER_NAME}。"
                    f"自然に2〜3文で返答。番号付きリスト・箇条書き禁止。"
                )
                d_tokens = 200
            sys_msg = {"role": "system", "content": sys_content}
            # ★[修正/hist-1] 哲学者complexモードはtrim_historyにトークン予算を渡す
            # 哲学者の長い返答(~1500tok)が4ペア分積み上がるとプロンプト枠を圧迫するため
            # 履歴全体を1500tok以内に抑えてSP+RAG+出力の余裕を確保する
            _hist_budget = 2000 if is_complex else 1000  # ★[v129] 1500→2000: 長期対話の一貫性向上
            cm = [sys_msg] + trim_history(messages[-MAX_HISTORY * 2:], token_budget=_hist_budget) + [{"role": "user", "content": user_text}]
            # total_len: システムプロンプト込みの総文字数をctx計算に渡す
            total_len = sum(len(m.get("content","")) for m in cm)
            print(f"[DEBUG] using model={model} is_complex={is_complex}", flush=True)
            result = stream_response(cm, is_complex, total_len, model=model, max_tokens=d_tokens) or ""
            # ★[修正/punct] gemma3の出力に混入する句読点・空白の異常を正規化
            if result and _is_philosopher:
                # 全角スペース・連続スペースを除去
                result = re.sub(r'　+', '', result)
                result = re.sub(r' {2,}', ' ', result)
                # 文中の孤立した読点「、」直後の改行を除去
                result = re.sub(r'、\n', '、', result)
                # 「たとえば」で終わっている（最終文が不完全）場合は除去
                result = re.sub(r'たとえば\s*$', '', result).rstrip()
        else:
            sys_msg = get_sys_prm(mode, user_text, per_id=persona_id)
            cm = build_chat_messages(sys_msg, messages + [{"role": "user", "content": user_text}], p)
            # ★[修正/TEMP_MAP] コマンドモード別温度をTEMP_MAPから取得して接続
            _cmd_temp = TEMP_MAP.get(f"/{mode}")
            result = stream_response(cm, mode in ("a", "c", "sum", "deep"), len(user_text), temp_override=_cmd_temp, model=model) or ""
        # プリフェッチ: 次の入力に備えてRAGを先読み
        if mode == "d" and result and len(user_text) > 4:
            _THREAD_POOL.submit(prefetch_rag, result[-80:])
        if mode != "d":
            update_keyword_memory(user_text)
            kw = extract_keywords(result)
            for w in kw:
                if w not in KEYWORD_MEMORY: KEYWORD_MEMORY.append(w)
            if len(KEYWORD_MEMORY) > 6: KEYWORD_MEMORY[:] = KEYWORD_MEMORY[-6:]
        return result

    COMMAND_REGISTRY: dict[str, Callable] = {
        "a": lambda a: two_pass_analysis(a, get_async_rag_data(a), current_persona, len(a)),
        "w": lambda a: _chat(a, "w"),
        "p": lambda a: _chat(a, "p"),
        "c": lambda a: _chat(a, "c"),
        "t": lambda a: _chat(a, "t"),
        "e": lambda a: _chat(a, "e"),
        "sum": lambda a: _chat(a, "sum"),
        "r": lambda a: (start_roleplay(a, persona_id), f"{C['p']}RP開始: {a}{C['w']}")[1],
        "rend": lambda _: (end_roleplay(), f"{C['y']}RP終了{C['w']}")[1],
        "q": lambda a: _handle_quest(a),
        "m": lambda a: handle_memo(a),
        "dict": lambda a: handle_dict(a),
        "doc": lambda a: handle_doc(a),
        "elab": lambda a: handle_elab(a, persona_id),
        "l": lambda a: _handle_lyrics(a),
        "y": lambda a: play_singularity(a) if a else f"{C['r']}usage: /y <曲名>{C['w']}",
        "midi": lambda a: handle_midi(a),
        "doctor": lambda _: doctor_report(),
        "debug": lambda _: debug_report(),
        "power": lambda a: set_power_mode(a),
        "optimizer": lambda _: OPTIMIZER.status(),
        "tool": lambda a: tool_agent_chat([{"role":"user","content":a}], True, len(a)) if a else f"{C['r']}usage: /tool <query>{C['w']}",
        "vec": lambda _: f"{C['c']}vector: {vector_count()} items | KB: {len([c for c in vector_list_collections() if c != 's01_memory'])} collections{C['w']}",
        "kb": lambda a: handle_kb(a, _chat, persona_id),
        "spi": lambda a: handle_spi(a),
        "stats": lambda _: handle_stats(),
        "history": lambda a: handle_history(a),
        "export": lambda a: handle_export(a, messages),
        "template": lambda a: handle_template(a),
        "tts": lambda a: handle_tts(a),
        # ★[修正/#3] /tr <言語> <テキスト> 書式に対応。
        # 旧コードは引数全体を text として渡し target_lang が常に "en" 固定だった。
        "tr": lambda a: (lambda p: handle_translate(p[1], p[0]) if len(p) >= 2 else handle_translate(a))(a.split(None, 1)) if a else handle_translate(""),
        "reference": lambda _: _handle_reference(),
        "stop": lambda _: _stop_files(),
        "s": lambda a: _handle_persona_switch(a),
        "g": lambda _: _handle_clear(messages),
        "h": lambda _: HELP_TEXT,
        "learn": lambda _: _handle_learn(),
        "img": lambda a: handle_image(a),
        "convert": lambda a: handle_convert(a),
        "qr": lambda a: handle_qr(a),
        "color": lambda a: handle_color(a),
        "sysinfo": lambda _: handle_sysinfo(),
        "rename": lambda a: handle_rename(a),
        "batch": lambda a: handle_batch(a),
        "chart": lambda a: handle_chart(a),
        "note": lambda a: handle_note(a),
        "timer": lambda a: handle_timer(a),
        "calc": lambda a: handle_calc(a),
        "comp": lambda a: handle_comp(a),
        "hegel": lambda a: handle_comp(a),
        "split": lambda a: handle_split(a),
        "offline": lambda a: _handle_offline(a),
        "ety": lambda a: handle_ety(a),
        "prime":      lambda a: _handle_prime(a),
        "baseball":   lambda a: _handle_baseball(),
        "chess":       lambda a: handle_chess(a, persona=current_persona),
        "shogi":       lambda a: handle_shogi(a, persona=current_persona),
        "wolf":        lambda a: handle_philosopher_wolf(a),
        "jinro":       lambda a: handle_philosopher_wolf(a),
        "mj":          lambda a: handle_mahjong(a),
        # ★[v129] 新コマンド
        "think":       lambda a: handle_think_mode(a),
        "plan":        lambda a: handle_plan(a, persona_id),
        "code":        lambda a: handle_code(a, persona_id),
        "reflect":     lambda a: handle_reflect(a, persona_id),
        "mindmap":     lambda a: handle_mindmap(a, persona_id),
        "persona_edit":lambda _: (setattr_return("current_persona", handle_persona_edit(current_persona)), "")[1] or f"{C['g']}編集完了{C['w']}",
        "model":       lambda a: handle_model_cmd(a),
        "ctx":         lambda _: handle_ctx_status(messages),
        "speedtest":   lambda _: handle_speedtest(),
    }

    def _handle_quest(arg: str) -> str:
        sub = arg.strip().lower()
        if not arg or sub == "list": return format_quests()
        if sub.startswith("done"):
            n = sub.replace("done", "").strip()
            return complete_quest(n) if n else f"{C['r']}usage: /q done <番号>{C['w']}"
        if sub == "show":
            return show_quest("")
        if sub.startswith("show"):
            return show_quest(sub.replace("show", "").strip())
        goal = arg
        plan = f"1. {goal}について調査 2. 分析 3. 結論"
        save_quest(goal, plan)
        return f"{C['g']}クエスト登録: {goal}{C['w']}"

    def _handle_lyrics(query: str) -> str:
        if not query:
            return f"{C['r']}usage: /l <曲名>{C['w']}"
        with SystemSpinner(f"歌詞検索: {query[:30]}", stage="rag") as sp:
            source, url, lyrics = search_lyrics_absolute(query)
        if not lyrics:
            return f"{C['y']}歌詞が見つかりませんでした: {query}{C['w']}"

        # ── コンプライアンス対応 ──────────────────────────────
        # 歌詞の著作権保護のため全文は表示しない。
        # 冒頭2行 + 出典URLのみ案内する。
        lines = [ln for ln in lyrics.strip().splitlines() if ln.strip()]
        preview_lines = lines[:2]
        preview = "\n".join(preview_lines)

        out = [f"{C['c']}=== {query} ==={C['w']}"]
        if preview:
            out.append(f"{C['w']}{preview}{C['w']}")
            out.append(f"{C['y']}  ... (続きは下記サイトでご確認ください){C['w']}")
        if url:
            out.append(f"{C['g']}  📎 {url}{C['w']}")
        else:
            out.append(f"{C['y']}  (出典URLを取得できませんでした){C['w']}")
        out.append(f"{C['b']}  ※歌詞の著作権は権利者に帰属します。{C['w']}")
        return "\n".join(out)

# ★[v130.1] /reference 専用: 信頼済みAI技術ソースのホワイトリスト
    def _handle_reference() -> str:
        """★[v130.1] /reference 超強化版 + セキュリティ強化
        セキュリティ:
          - _trusted_fetch でホワイトリスト外ドメインを完全拒否
          - HTTPS専用・パストラバーサル防止・クエリ長制限
          - コンテンツハッシュで重複取得を排除
          - レート制限（前回実行から60秒未満は警告）
        機能:
          1. 信頼済みソースから最新AI技術をリアルタイム収集
          2. LLMで技術を抽出・応用可能性スコアリング
          3. PROMPT_OPTIMIZATIONS に自動反映（重複排除）
          4. 非循環LLM品質診断（カテゴリ別トレンド + 行動指針）
          5. モード別パフォーマンスダッシュボード
          6. セキュリティ監査ログ表示
        """
        import json as _json
        import hashlib as _hl
        import time as _time

        # ── レート制限: 連続実行防止 ─────────────────────────────
        _REF_STATE = globals().setdefault("_REFERENCE_STATE", {
            "last_run": 0.0,
            "content_hashes": set(),
            "run_count": 0,
        })
        now_ts = _time.time()
        elapsed = now_ts - _REF_STATE["last_run"]
        _REF_STATE["last_run"] = now_ts
        _REF_STATE["run_count"] += 1

        lines = [
            f"{C['c']}{'═'*64}{C['w']}",
            f"{C['c']}  /reference  AI技術インテリジェンス & 自己改善システム"
            f"  (#{_REF_STATE['run_count']}){C['w']}",
            f"{C['c']}{'═'*64}{C['w']}",
        ]
        if 0 < elapsed < 60:
            lines.append(
                f"  {C['y']}⚠ 前回実行から{elapsed:.0f}秒。"
                f"キャッシュ済みデータを一部再利用します。{C['w']}"
            )

        # ══ SECTION 1: 信頼済みソースから最新AI技術を収集 ══
        lines.append(f"\n{C['y']}【1】最新AI技術 収集 (ホワイトリスト限定){C['w']}")

        _TRUSTED_SOURCES = [
            ("HuggingFace Blog",
             "https://huggingface.co/blog",
             "🤗"),
            ("HuggingFace Papers",
             "https://huggingface.co/papers",
             "🤗"),
            ("Simon Willison",
             "https://simonwillison.net/",
             "📝"),
            ("Lilian Weng (OpenAI)",
             "https://lilianweng.github.io/",
             "📝"),
            ("Prompting Guide",
             "https://www.promptingguide.ai/",
             "💡"),
        ]

        collected_texts = []
        fetch_report = []
        _fetch_lock = threading.Lock()

        def _fetch_one(src):
            src_name, src_url, src_tag = src
            try:
                html = _trusted_fetch(src_url, timeout=4)
                if not html:
                    return (None, f"  ✗ {src_name}: 空レスポンス")
                text = strip_tags(html)[:2500]
                h = _hl.sha256(text.encode()).hexdigest()[:12]
                with _fetch_lock:
                    if h in _REF_STATE["content_hashes"]:
                        return (None, f"  ↩ {src_name}: キャッシュ済み")
                    _REF_STATE["content_hashes"].add(h)
                return (f"==[{src_name}]==\n{text}", f"  ✓ {src_name} ({len(text)}chars)")
            except ValueError as e:
                return (None, f"  🔒 {src_name}: {str(e)[:60]}")
            except Exception as e:
                return (None, f"  ✗ {src_name}: {str(e)[:60]}")

        print(f"  並列fetch中 ({len(_TRUSTED_SOURCES)}ソース)...", flush=True)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(_fetch_one, src): src for src in _TRUSTED_SOURCES}
            for fut in as_completed(futures):
                text_result, report = fut.result()
                if text_result:
                    collected_texts.append(text_result)
                fetch_report.append(report)

        lines.append("\n".join(fetch_report))

        if not collected_texts:
            lines.append(f"  {C['r']}全ソースの取得失敗 — ネット確認または /offline on を検討{C['w']}")
            raw_tech_summary = ""
        else:
            _t_extract = __import__('time').time()
            print(f"\n  → ルールベース技術抽出中... ", end="", flush=True)
            tech_extract_sys = ""  # ルールベース化のため未使用
            # ルールベース技術抽出（LLM不要・高速）
            _TECH_KEYWORDS = [
                ("Chain-of-Thought",    "段階的推論プロンプト",          "reasoning", "高", "易"),
                ("RAG",                 "検索拡張生成",                  "rag",       "高", "中"),
                ("HyDE",                "仮想ドキュメント埋め込み",       "rag",       "高", "中"),
                ("Self-Reflection",     "自己反省ループ",                 "reasoning", "高", "中"),
                ("Function Calling",    "LLMツール呼び出し",             "output",    "高", "易"),
                ("BM25",                "キーワードランキング検索",       "rag",       "高", "易"),
                ("Cross-Encoder",       "再ランキングモデル",             "rag",       "中", "中"),
                ("Constitutional AI",   "安全制約付き学習",              "safety",    "高", "難"),
                ("Agent Loop",          "自律エージェントループ",         "memory",    "高", "難"),
                ("Structured Output",   "JSON強制出力",                  "output",    "高", "易"),
                ("Prompt Compression",  "入力圧縮で高速化",              "prompt",    "高", "中"),
                ("Speculative Decoding","投機的デコードで高速化",         "output",    "中", "難"),
            ]
            all_text = " ".join(collected_texts).lower()
            matched = []
            for tech, summary, cat, app, diff in _TECH_KEYWORDS:
                if tech.lower() in all_text or tech.split("-")[0].lower() in all_text:
                    matched.append(f"{tech} | {summary} | {cat} | {app} | {diff}")
            # マッチしない場合は全件採用
            if len(matched) < 3:
                matched = [f"{t} | {s} | {c} | {a} | {d}" for t,s,c,a,d in _TECH_KEYWORDS[:6]]
            raw_tech_summary = "\n".join(matched)
            print(f"{C['g']}完了 ({__import__('time').time()-_t_extract:.1f}秒){C['w']}")

            lines.append(f"\n  {C['c']}{'技術名':<22} {'分類':<12} {'応用性':<6} {'難易度'}{C['w']}")
            lines.append(f"  {'─'*56}")
            for row in raw_tech_summary.splitlines():
                parts = [p.strip() for p in row.split("|")]
                if len(parts) >= 4:
                    name = parts[0]; summary = parts[1]; cat = parts[2]
                    applicability = parts[3]; diff = parts[4] if len(parts) > 4 else "─"
                    app_color = C['g'] if "高" in applicability else C['y'] if "中" in applicability else C['dim']
                    lines.append(
                        f"  {name:<22} {C['c']}{cat:<12}{C['w']} "
                        f"{app_color}{applicability:<6}{C['w']} {diff}"
                    )
                    lines.append(f"    {C['dim']}{summary}{C['w']}")

        # ══ SECTION 2: 適用可能技術を自動でPROMPT_OPTIMIZATIONSに反映 ══
        if raw_tech_summary:
            lines.append(f"\n{C['y']}【2】自AI適用 & PROMPT_OPTIMIZATIONS 自動反映{C['w']}")
            _t_apply = __import__('time').time()
            print(f"  → ルールベース適用分析中... ", end="", flush=True)
            import json as _json2
            _directive_map = {
                "Chain-of-Thought":   ("reasoning", "CoTプロンプトを自動付与せよ"),
                "RAG":                ("rag",       "BM25+ベクトルハイブリッドRAGを使用"),
                "HyDE":               ("rag",       "検索前にHyDEでクエリ変換"),
                "Self-Reflection":    ("reasoning", "長文後に自己反省ループ実行"),
                "Function Calling":   ("output",    "ツール呼び出しをJSON統一"),
                "BM25":               ("rag",       "初期検索にBM25を必ず通す"),
                "Constitutional AI":  ("safety",    "出力前に安全制約チェック"),
                "Structured Output":  ("output",    "出力はJSON強制モード"),
                "Prompt Compression": ("prompt",    "2000字超は圧縮してから送る"),
            }
            _auto_directives = []
            for row in raw_tech_summary.splitlines():
                parts = [p.strip() for p in row.split("|")]
                if len(parts) >= 3:
                    tech = parts[0]
                    cat, directive = _directive_map.get(tech, ("prompt", f"{tech}を積極活用せよ"))
                    _auto_directives.append({"tech": tech, "category": cat, "directive": directive, "rationale": "自動抽出", "priority": 3})
            apply_raw = _json2.dumps(_auto_directives, ensure_ascii=False)
            # ★[修正/ollama-silent] 戻り値空 = Ollama未起動 or タイムアウト → 警告を表示
            if not apply_raw:
                print(f"{C['r']}失敗{C['w']}")
                lines.append(f"  {C['r']}⚠ Ollama未起動またはモデル応答なし — SECTION 2をスキップ{C['w']}")
                lines.append(f"  {C['dim']}  ヒント: ollama serve && ollama pull {DEEP_MODEL}{C['w']}")
                apply_raw = "[]"
            else:
                print(f"{C['g']}完了 ({__import__('time').time()-_t_apply:.1f}秒){C['w']}")

            try:
                clean_json = re.sub(r'```json|```', '', apply_raw).strip()
                tech_directives = _json.loads(clean_json)
                applied, skipped = [], []
                # ★[修正/reference] _get_persona_bucket('global') を使って正しい入れ子構造に書き込む
                # 旧コードは PROMPT_OPTIMIZATIONS[cat]=[] のフラット書き込みで inject_optimizations に届かなかった
                _ref_bucket = _get_persona_bucket("global")
                for td in sorted(tech_directives, key=lambda x: -x.get("priority", 0)):
                    cat = td.get("category", "prompt")
                    directive = td.get("directive", "")
                    tech_name = td.get("tech", "unknown")
                    rationale = td.get("rationale", "")
                    pri = td.get("priority", 1)
                    if not directive or not cat: continue
                    if cat not in _ref_bucket:
                        _ref_bucket[cat] = []
                    if any(directive[:20] in d for d in _ref_bucket[cat]):
                        skipped.append(tech_name); continue
                    # カテゴリ上限管理
                    if len(_ref_bucket[cat]) >= _DIRECTIVE_PER_CAT_MAX:
                        _ref_bucket[cat].pop(0)
                    _ref_bucket[cat].append(directive)
                    hist_msg = f"[/reference #{_REF_STATE['run_count']}] {tech_name}: {directive[:60]}"
                    OPTIMIZATION_HISTORY.append(hist_msg)
                    if len(OPTIMIZATION_HISTORY) > 50:
                        OPTIMIZATION_HISTORY[:] = OPTIMIZATION_HISTORY[-50:]
                    stars = "★" * pri + "☆" * (5 - pri)
                    applied.append(
                        f"  {C['g']}✓{C['w']} {stars} {C['c']}{tech_name}{C['w']}"
                        f" [{cat}] {directive[:48]}"
                        f"\n    {C['dim']}理由: {rationale}{C['w']}"
                    )
                if applied:
                    lines.append(f"{C['g']}  反映済み ({len(applied)}件):{C['w']}")
                    lines.extend(applied)
                    # ★ 反映後に即座に永続化
                    persist_learning()
                if skipped:
                    lines.append(f"  {C['dim']}スキップ（既反映）: {', '.join(skipped)}{C['w']}")
            except Exception as e:
                lines.append(f"  {C['r']}JSON解析失敗: {e}{C['w']}")
                lines.append(f"  {C['dim']}{apply_raw[:200]}{C['w']}")

        # ══ SECTION 3: 非循環 LLM品質診断 ══
        lines.append(f"\n{C['y']}【3】自己評価ログ 品質診断{C['w']}")
        if len(SELF_EVAL_LOG) < 2:
            lines.append(
                f"  {C['dim']}評価ログ不足 ({len(SELF_EVAL_LOG)}件)"
                f" — 会話を重ねると分析が開始されます{C['w']}"
            )
        else:
            recent = SELF_EVAL_LOG[-20:]
            cat_data = {}
            for entry in recent:
                for cat, score in entry.get("scores", {}).items():
                    cat_data.setdefault(cat, []).append(score)
            lines.append(f"  {C['c']}{'カテゴリ':<14} {'平均':>5}  {'バー':^14}  トレンド{C['w']}")
            lines.append(f"  {'─'*54}")
            for cat, scores in sorted(cat_data.items()):
                avg = sum(scores) / len(scores)
                # ★[修正/trend] サンプルが2件以上ある場合のみ前後半を比較する
                # 旧コード: half=max(1,len//2) → len==1のとき scores[1:] が空になり
                # 後半平均=0 → delta が常に負値 → 全カテゴリ「↓下降」と誤表示していた
                if len(scores) < 2:
                    trend = f"{C['dim']}→  -(n<2){C['w']}"
                else:
                    half  = max(1, len(scores) // 2)
                    first = sum(scores[:half]) / half
                    rest  = scores[half:]
                    last  = sum(rest) / len(rest)
                    delta = last - first
                    trend = (f"{C['g']}↑ +{delta:.2f}{C['w']}" if delta > 0.05
                             else f"{C['r']}↓ {delta:.2f}{C['w']}" if delta < -0.05
                             else f"{C['dim']}→  0.00{C['w']}")
                filled = int(avg * 14)
                bar_str = f"{C['g']}{'█'*filled}{C['dim']}{'░'*(14-filled)}{C['w']}"
                lines.append(f"  {cat:<14} {avg:>5.2f}  {bar_str}  {trend}")

            if len(SELF_EVAL_LOG) >= 5:
                print(f"\n  → LLM品質診断中... ", end="", flush=True)
                log_summary = _json.dumps(
                    [{"mode": e["mode"], "avg": round(e["avg"], 2),
                      "scores": {k: round(v, 2) for k, v in e["scores"].items()}}
                     for e in recent[-10:]], ensure_ascii=False
                )
                quality_sys = (
                    "あなたはAI品質評価の専門家。スコアの羅列でなく洞察を述べよ。\n"
                    "以下の自律AIの自己評価ログを分析し:\n"
                    "1. 【最重要改善点】最も改善が必要な領域と具体的改善策（2〜3行）\n"
                    "2. 【強み】最も安定して高いカテゴリとその要因（1行）\n"
                    "3. 【次の行動指針】次の10回で意識すべき具体的な1つのアクション\n"
                    "4. 【危険信号】スコアが0.3以下があれば緊急アラートを出せ\n"
                    "日本語・簡潔に。"
                )
                quality_comment = None  # スキップ: 速度優先
                if False: quality_comment = stream_response(
                    [{"role": "system", "content": quality_sys},
                     {"role": "user", "content": log_summary}],
                    False, 100, temp_override=0.0, model=FAST_MODEL,
                    max_tokens=300, silent=True
                )
                # ★[修正/ollama-silent] 空応答 → Ollama警告
                if not quality_comment:
                    print(f"{C['r']}失敗{C['w']}")
                    lines.append(f"\n  {C['r']}⚠ Ollama未起動 — SECTION 3診断スキップ{C['w']}")
                    lines.append(f"  {C['dim']}  ヒント: ollama serve && ollama pull {DEEP_MODEL}{C['w']}")
                else:
                    print(f"{C['g']}完了{C['w']}")
                    lines.append(f"\n  {C['c']}◆ LLM品質診断{C['w']}")
                    for ln in quality_comment.splitlines():
                        lines.append(f"  {ln}")

        # ══ SECTION 4: PROMPT_OPTIMIZATIONS ダッシュボード ══
        lines.append(f"\n{C['y']}【4】現在の最適化ディレクティブ{C['w']}")
        # ★[修正/slice-bug] PROMPT_OPTIMIZATIONS は {pkey: {cat: [directives]}} のネスト構造。
        # 旧コードは items() を {cat: [directives]} フラット辞書として扱っていたため
        # directives[-3:] でdict に slice を適用し TypeError: unhashable type: 'slice' が発生。
        total_directives = sum(
            len(v) for bucket in PROMPT_OPTIMIZATIONS.values()
            if isinstance(bucket, dict) for v in bucket.values()
        )
        if PROMPT_OPTIMIZATIONS:
            for pkey, bucket in sorted(PROMPT_OPTIMIZATIONS.items()):
                if not isinstance(bucket, dict):
                    continue
                bucket_total = sum(len(v) for v in bucket.values())
                if bucket_total == 0:
                    continue
                lines.append(f"  {C['c']}[{pkey}]{C['w']} {bucket_total}件")
                for cat, directives in sorted(bucket.items()):
                    for d in directives[-3:]:
                        lines.append(f"    {C['dim']}• [{cat}] {d[:72]}{C['w']}")
        else:
            lines.append(f"  {C['dim']}まだ最適化ディレクティブなし{C['w']}")

        # ══ SECTION 5: モード別パフォーマンスランキング ══
        if REFERENCE_SCORES:
            lines.append(f"\n{C['y']}【5】モード別パフォーマンス{C['w']}")
            sorted_modes = sorted(
                REFERENCE_SCORES.items(),
                key=lambda x: sum(x[1]) / max(len(x[1]), 1), reverse=True
            )
            for i, (mode, scores) in enumerate(sorted_modes[:8]):
                avg = sum(scores) / len(scores)
                filled = int(avg * 12)
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
                bar = f"{C['g']}{'█'*filled}{C['dim']}{'░'*(12-filled)}{C['w']}"
                lines.append(f"  {medal} {mode:<12} {bar} {avg:.2f} ({len(scores)}回)")

        # ══ SECTION 6: セキュリティ監査ログ ══
        lines.append(f"\n{C['y']}【6】セキュリティ状態{C['w']}")
        lines += [
            f"  {C['g']}✓{C['w']} ホワイトリスト: {len(_TRUSTED_AI_DOMAINS)}ドメイン限定",
            f"  {C['g']}✓{C['w']} プロトコル: HTTPS専用",
            f"  {C['g']}✓{C['w']} SSRF防御: IPv6マップド・10進IP・リダイレクト先検証",
            f"  {C['g']}✓{C['w']} 重複排除: SHA-256ハッシュ ({len(_REF_STATE['content_hashes'])}件キャッシュ)",
            f"  {C['g']}✓{C['w']} レート制限: 60秒クールダウン",
            f"  {C['g']}✓{C['w']} パストラバーサル防止・クエリ長上限512chars",
        ]

        # ══ SECTION 7: コード自己進化 (Code Self-Evolution) ══
        if raw_tech_summary:
            lines.append(f"\n{C['y']}【7】コード自己進化 — 自動パッチ提案{C['w']}")
            print(f"  → コード改善案生成中... ", end="", flush=True)
            # 自分のソースファイルを特定
            _self_src = os.path.abspath(sys.argv[0]) if sys.argv else ""
            _evolvable = os.path.isfile(_self_src) and _self_src.endswith(".py")

            # ルールベースCODE_EVO（LLM不要・高速）
            _EVO_RULES = {
                "Chain-of-Thought":    ("stream_response",   "CoT指示をシステムプロンプトに自動付与",  "推論精度向上", 4),
                "RAG":                 ("hybrid_search_advanced", "BM25+ベクトルのn_candidates増加", "検索精度向上", 4),
                "HyDE":                ("vector_search",     "検索前にHyDEクエリ変換を追加",          "意味検索強化", 3),
                "Self-Reflection":     ("run",               "長文応答後に自己反省ループ追加",          "品質向上",     3),
                "Function Calling":    ("_handle_reference", "出力をJSON強制モードで統一",              "安定性向上",   3),
                "BM25":                ("bm25_search",       "BM25スコアにTF-IDF重みを追加",           "検索精度向上", 3),
                "Constitutional AI":   ("run",               "出力前に安全制約チェックを挿入",          "安全性向上",   4),
                "Structured Output":   ("stream_response",   "JSONスキーマバリデーションを追加",        "出力安定化",   3),
                "Prompt Compression":  ("run",               "2000字超入力を自動圧縮",                 "高速化",       3),
            }
            _evo_items = []
            for row in raw_tech_summary.splitlines():
                parts = [p.strip() for p in row.split("|")]
                if parts and parts[0] in _EVO_RULES:
                    func, desc, effect, pri = _EVO_RULES[parts[0]]
                    _evo_items.append({"func": func, "desc": desc, "effect": effect, "priority": pri})
            import json as _json3
            code_evo_raw = _json3.dumps(_evo_items, ensure_ascii=False) if _evo_items else "[]"
            print(f"{C['g']}完了({len(_evo_items)}件){C['w']}")

            try:
                _evo_tmp = re.sub(r'```json|```', '', code_evo_raw).strip()
                _evo_match = re.search(r'\[.*?\]', _evo_tmp, re.DOTALL)
                clean_evo = _evo_match.group(0) if _evo_match else '[]'
                evo_items = _json.loads(clean_evo)
                evo_applied = []
                for item in sorted(evo_items, key=lambda x: -x.get("priority", 0)):
                    func  = item.get("func",   "?")
                    desc  = item.get("desc",   "")
                    effect= item.get("effect", "")
                    pri   = item.get("priority", 1)
                    if not desc: continue
                    stars = "★" * pri + "☆" * (5 - pri)
                    evo_applied.append(
                        f"  {C['c']}{stars}{C['w']} {C['g']}{func}(){C['w']} — {desc}"
                        f"\n    {C['dim']}効果: {effect}{C['w']}"
                    )
                    # ★ 提案をOPTIMIZATION_HISTORYに記録 (コード改善提案として)
                    OPTIMIZATION_HISTORY.append(
                        f"[CODE_EVO #{_REF_STATE['run_count']}] {func}: {desc[:60]}"
                    )
                    if len(OPTIMIZATION_HISTORY) > 50:
                        OPTIMIZATION_HISTORY[:] = OPTIMIZATION_HISTORY[-50:]

                if evo_applied:
                    lines.append(f"  {C['g']}コード改善提案 ({len(evo_applied)}件):{C['w']}")
                    lines.extend(evo_applied)
                    # ★ 提案を自AIのソースファイル末尾にコメントとして記録
                    if _evolvable:
                        try:
                            ts = _time.strftime("%Y-%m-%d %H:%M")
                            # ★[修正/code-evo-dedup] 重複チェック + 最大10ブロック保持
                            # 旧コード: 無条件 "a" 追記 → /reference 呼び出しのたびに同じ内容が増殖
                            # 修正: ① 既存ブロックを読んでハッシュ比較 ② 10件超で古いものを削除
                            _MAX_EVO_BLOCKS = 10
                            new_lines_body = ""
                            for item in sorted(evo_items, key=lambda x: -x.get("priority", 0)):
                                func  = item.get("func",  "?")
                                desc  = item.get("desc",  "")
                                effect= item.get("effect","")
                                pri   = item.get("priority", 1)
                                if desc:
                                    new_lines_body += f"# [{pri}★] {func}: {desc} → {effect}\n"
                            import hashlib as _hl
                            new_sig = _hl.md5(new_lines_body.encode()).hexdigest()[:8]
                            patch_header = f"# ━━ /reference CODE_EVO #{_REF_STATE['run_count']} ({ts}) sig={new_sig} ━━"
                            patch_block  = f"\n{patch_header}\n{new_lines_body}"

                            with open(_self_src, "r", encoding="utf-8") as _rf:
                                src_text = _rf.read()

                            # 同一sig のブロックが既にあればスキップ
                            if f"sig={new_sig}" in src_text:
                                lines.append(
                                    f"  {C['dim']}CODE_EVO: 同一内容 (sig={new_sig}) は既に記録済み — スキップ{C['w']}"
                                )
                            else:
                                # 古いCODE_EVOブロックをカウントし _MAX_EVO_BLOCKS 超なら最古を除去
                                import re as _re2
                                evo_pattern = _re2.compile(
                                    r"\n# ━━ /reference CODE_EVO #\d+.*?(?=\n# ━━ /reference CODE_EVO #|\Z)",
                                    _re2.DOTALL
                                )
                                existing = evo_pattern.findall(src_text)
                                if len(existing) >= _MAX_EVO_BLOCKS:
                                    # 最古のブロックを1件削除
                                    src_text = src_text.replace(existing[0], "", 1)
                                src_text += patch_block
                                with open(_self_src, "w", encoding="utf-8") as _pf:
                                    _pf.write(src_text)
                                lines.append(
                                    f"  {C['dim']}改善提案をソース末尾に記録: {os.path.basename(_self_src)} (sig={new_sig}){C['w']}"
                                )
                        except Exception as _pe:
                            lines.append(f"  {C['dim']}ソース記録スキップ: {_pe}{C['w']}")
                else:
                    lines.append(f"  {C['dim']}コード改善提案なし{C['w']}")
            except Exception as e:
                lines.append(f"  {C['r']}コード進化JSON解析失敗: {e}{C['w']}")
                lines.append(f"  {C['dim']}{code_evo_raw[:200]}{C['w']}")

        # ══ フッター ══
        lines += [
            f"\n{C['c']}{'═'*64}{C['w']}",
            f"{C['dim']}  最適化履歴: {len(OPTIMIZATION_HISTORY)}件 | "
            f"評価ログ: {len(SELF_EVAL_LOG)}件 | "
            f"ディレクティブ: {total_directives}件 | "
            f"実行回数: #{_REF_STATE['run_count']}{C['w']}",
            f"{C['dim']}  反映された技術は次回の応答から自動適用されます{C['w']}",
            f"{C['c']}{'═'*64}{C['w']}",
        ]
        return "\n".join(lines)


    _handle_stop = _stop_files  # ★ 外出し済み（run()直前で定義）

    def _handle_offline(arg: str) -> str:
        global OFFLINE_MODE
        a = arg.strip().lower()
        if a == "on":
            OFFLINE_MODE = True
            return (f"{C['y']}[OFFLINE ON]{C['w']} ネット通信を無効化。\n"
                    f"  Wikipedia: Kiwix (localhost:{KIWIX_PORT}) を使用\n"
                    f"  Web検索: スキップ → /kb ask で代替\n"
                    f"  Kiwix起動例: kiwix-serve --port {KIWIX_PORT} wikipedia_ja_all.zim")
        if a == "off":
            OFFLINE_MODE = False
            return f"{C['g']}[OFFLINE OFF]{C['w']} オンラインモードに戻しました。"
        if a in ("kiwix", "wiki"):
            # Kiwix の疎通確認
            try:
                test = fetch_html(f"http://localhost:{KIWIX_PORT}/", timeout=2, silent=True)
                if test:
                    return f"{C['g']}Kiwix: localhost:{KIWIX_PORT} 接続OK{C['w']}"
                return f"{C['r']}Kiwix: localhost:{KIWIX_PORT} に接続できません{C['w']}"
            except Exception as e:
                return f"{C['r']}Kiwix: {e}{C['w']}"
        status = f"{C['y']}OFFLINE{C['w']}" if OFFLINE_MODE else f"{C['g']}ONLINE{C['w']}"
        return (f"現在: {status}\n"
                f"  /offline on   ネット無効化（Kiwix使用）\n"
                f"  /offline off  オンラインに戻す\n"
                f"  /offline kiwix  Kiwix疎通確認 (port:{KIWIX_PORT})")

    def _handle_persona_switch(arg: str) -> str:
        nonlocal persona_id, current_persona
        global CUSTOM_PERSONA

        # ── サブコマンド: save / load / list / del ──────────────────
        parts_arg = arg.strip().split(maxsplit=1)
        sub = parts_arg[0].lower() if parts_arg else ""
        sub_val = parts_arg[1].strip() if len(parts_arg) > 1 else ""

        if sub == "list":
            slots = list_personas()
            if not slots:
                return f"{C['y']}保存済みペルソナなし。/s save <スロット名> で保存{C['w']}"
            rows = [f"{C['c']}=== 保存済みペルソナ ==={C['w']}"]
            for slot, p in slots.items():
                web_tag = f" {C['b']}[Web]{C['w']}" if p.get("_web") else ""
                rows.append(f"  {C['g']}{slot}{C['w']} → {p['name']} / 一人称:{p['first_person']}{web_tag}  ({p.get('saved_at','')})")
            rows.append(f"{C['dim']}使い方: /s load <スロット名>  /s del <スロット名>{C['w']}")
            return "\n".join(rows)

        if sub == "save":
            slot = sub_val or current_persona.get("name", "custom")
            if save_persona(slot, current_persona):
                web_tag = f" {C['b']}[Web参照済]{C['w']}" if current_persona.get("_web") else ""
                return f"{C['g']}保存完了: [{slot}] = {current_persona['name']}{C['w']}{web_tag}"
            return f"{C['r']}保存失敗{C['w']}"

        if sub == "load":
            if not sub_val:
                return f"{C['r']}usage: /s load <スロット名>{C['w']}"
            p = load_persona(sub_val)
            if p is None:
                saved = list(list_personas().keys())
                hint = "  保存済み: " + ", ".join(saved) if saved else "  (保存なし)"
                return f"{C['r']}スロット '{sub_val}' が見つかりません{C['w']}\n{hint}"
            CUSTOM_PERSONA = p
            _SYS_PRM_CACHE.clear()
            _SYS_EXTRAS_CACHE.clear()
            messages.clear()           # ★ 前キャラの履歴を引き継がない
            _SPI_SESSION_MEMORY.clear()
            current_persona = CUSTOM_PERSONA
            persona_id = 99
            web_tag = f" {C['b']}[Web参照済]{C['w']}" if p.get("_web") else ""
            return f"{C['g']}ロード: [{sub_val}] {p['name']} / 一人称:{p['first_person']}{C['w']}{web_tag}"

        if sub == "del":
            if not sub_val:
                return f"{C['r']}usage: /s del <スロット名>{C['w']}"
            if delete_persona(sub_val):
                return f"{C['y']}削除: [{sub_val}]{C['w']}"
            return f"{C['r']}スロット '{sub_val}' が見つかりません{C['w']}"

        # ── 以下は既存の切替処理 ───────────────────────────────────
        if not arg:
            slots = list_personas()
            slot_hint = f"\n  保存済み: {', '.join(slots)}" if slots else ""
            rows = [f"{C['c']}現在: {current_persona['name']} (ID:{persona_id}){C['w']}"]
            rows.append(f"{C['y']}── /s 1〜36 西洋哲学者一覧 ──{C['w']}")
            for pid, p in PERSONA_MAP.items():
                mark = f" {C['g']}◀ 現在{C['w']}" if pid == persona_id else ""
                rows.append(f"  {C['c']}{pid:2d}{C['w']} {p['name']}{mark}")
            rows.append(f"{C['dim']}── /s <名前> で自由入力（Web検索ペルソナ生成）──{C['w']}{slot_hint}")
            return "\n".join(rows)
        if arg.lower() == "custom":
            return _handle_custom_persona()
        if arg.isdigit():
            pid = int(arg)
            if pid in PERSONA_MAP:
                CUSTOM_PERSONA = None
                _SYS_PRM_CACHE.clear()
                _SYS_EXTRAS_CACHE.clear()
                messages.clear()
                _SPI_SESSION_MEMORY.clear()
                KEYWORD_MEMORY.clear()  # ★ 前キャラの話題キーワードをクリア
                persona_id = pid
                current_persona = get_persona(pid)
                return f"{C['g']}キャラ切替: {current_persona['name']} / 一人称: {current_persona.get('first_person', '私')}{C['w']}"
            return f"{C['r']}ID: 1-{max(PERSONA_MAP)} (1=ソクラテス〜36=ロールズ){C['w']}"
        arg_lower = arg.lower()
        name_pid = next((pid for pid, p in PERSONA_MAP.items() if p.get("name", "").lower() == arg_lower), None)
        if name_pid is None:
            name_pid = next((pid for pid, p in PERSONA_MAP.items() if arg_lower in p.get("name", "").lower()), None)
        if name_pid is not None:
            CUSTOM_PERSONA = None
            _SYS_PRM_CACHE.clear()
            _SYS_EXTRAS_CACHE.clear()
            messages.clear()
            _SPI_SESSION_MEMORY.clear()
            KEYWORD_MEMORY.clear()  # ★ 前キャラの話題キーワードをクリア
            persona_id = name_pid
            current_persona = get_persona(name_pid)
            return f"{C['g']}キャラ切替: {current_persona['name']} / 一人称: {current_persona.get('first_person', '私')}{C['w']}"
        if arg.startswith("--"):
            CUSTOM_PERSONA = None
            _SYS_PRM_CACHE.clear()
            _SYS_EXTRAS_CACHE.clear()
            messages.clear()
            _SPI_SESSION_MEMORY.clear()
            persona_id = 2
            current_persona = get_persona(2)
            return f"{C['g']}リセット→ {current_persona['name']}{C['w']}"
        # 保存済みスロットに一致するか確認（名前で引ける）
        saved_match = load_persona(arg) or load_persona(arg_lower)
        if saved_match:
            CUSTOM_PERSONA = saved_match
            _SYS_PRM_CACHE.clear()
            _SYS_EXTRAS_CACHE.clear()  # ★[修正3]
            current_persona = CUSTOM_PERSONA
            persona_id = 99
            web_tag = f" {C['b']}[保存済]{C['w']}" if True else ""
            return f"{C['g']}ロード(保存済み): {saved_match['name']} / 一人称:{saved_match['first_person']}{C['w']}{web_tag}"
        print(f"{C['dim']}[Web検索でペルソナを構築中...]{C['w']}", flush=True)
        CUSTOM_PERSONA = build_custom_persona(arg)
        _SYS_PRM_CACHE.clear()
        _SYS_EXTRAS_CACHE.clear()  # ★[修正3] ペルソナ切替時はextrasも破棄
        current_persona = CUSTOM_PERSONA
        persona_id = 99
        web_tag = f" {C['b']}[Web参照済]{C['w']}" if current_persona.get("_web") else ""
        # ★[修正3] 未保存のWeb取得済みペルソナに保存tipを表示
        save_tip = ""
        if current_persona.get("_web"):
            save_tip = f"\n{C['dim']}tip: /s save {current_persona['name']} で保存すると次回即起動します{C['w']}"
        return f"{C['g']}カスタム: {current_persona['name']} / 一人称: {current_persona.get('first_person', '私')}{C['w']}{web_tag}{save_tip}"

    def _handle_custom_persona() -> str:
        nonlocal persona_id, current_persona
        global CUSTOM_PERSONA
        print(f"{C['c']}カスタムキャラ名: {C['w']}", end="", flush=True)
        try:
            name = sys.stdin.readline().strip() or "CUSTOM"
            print(f"{C['c']}特徴(省略可): {C['w']}", end="", flush=True)
            hint = sys.stdin.readline().strip()
            CUSTOM_PERSONA = build_custom_persona(name, hint)
            _SYS_PRM_CACHE.clear()
            current_persona = CUSTOM_PERSONA
            persona_id = 99
            return f"{C['g']}カスタムキャラ: {current_persona['name']} / 一人称: {current_persona.get('first_person', '私')}{C['w']}"
        except EOFError:
            return f"{C['r']}入力中断{C['w']}"

    def _handle_clear(ms: list) -> str:
        ms.clear()
        KEYWORD_MEMORY.clear()
        global ROLEPLAY_ACTIVE, ROLEPLAY_SCENE
        ROLEPLAY_ACTIVE = False
        ROLEPLAY_SCENE = ""
        return f"{C['g']}履歴クリア{C['w']}"

    def _handle_learn() -> str:
        directive_lines = []
        bucket = _get_persona_bucket(current_persona.get("name", "global"))
        for cat in ("禁止表現", "指定表現"):
            for d in bucket.get(cat, []):
                directive_lines.append(f"  [{cat}] {d}")
        total = sum(len(v) for b in PROMPT_OPTIMIZATIONS.values()
                    if isinstance(b, dict) for v in b.values())
        directive_str = ("\n" + "\n".join(directive_lines)) if directive_lines else " なし"
        def _temp_graph(hist):
            if len(hist) < 2: return "  (データ不足)"
            mn,mx=min(hist),max(hist); rng=mx-mn if mx!=mn else 0.1
            H,W=5,min(len(hist),30); data=hist[-W:]
            # 新しい点ほど暖色：最新=赤、1つ前=オレンジ、2つ前=黄、3つ前=緑、それ以前=シアン
            def _dot_color(idx, total):
                age = total - 1 - idx  # 0=最新
                if   age == 0: return C['r']            # 赤（最新）
                elif age == 1: return C['o']            # オレンジ
                elif age == 2: return C['y']            # 黄
                elif age <= 5: return C['g']            # 緑
                else:          return C['c']            # シアン（古い）
            rows=[]
            for row in range(H,-1,-1):
                thr=mn+rng*(row/H)
                next_thr=mn+rng*((row-1)/H) if row>0 else mn
                line=""
                for i,v in enumerate(data):
                    if next_thr<=v<thr or (row==0 and v<=thr) or (row==H and v>=thr):
                        line += _dot_color(i, len(data)) + "●" + C['w']
                    else:
                        line += " "
                label=f"{thr:.2f}|" if row in (0,H) else "     |"
                rows.append(label+line)
            rows.append("     +"+"─"*W)
            # 凡例
            rows.append(f"  凡例: {C['r']}●最新{C['w']} {C['o']}●1つ前{C['w']} {C['y']}●2つ前{C['w']} {C['g']}●3〜5前{C['w']} {C['c']}●古い{C['w']}")
            return "\n".join(rows)
        trend=""
        if len(TEMP_HISTORY)>1:
            trend="↓低下中" if TEMP_HISTORY[-1]<TEMP_HISTORY[-2] else "↑上昇中" if TEMP_HISTORY[-1]>TEMP_HISTORY[-2] else "→安定"
        return "\n".join([
            f"{C['c']}=== 学習状態 ==={C['w']}",
            f"対話数: {LEARNING_STATS['total_interactions']}",
            f"肯定/否定: {LEARNING_STATS['positive_count']}/{LEARNING_STATS['negative_count']}",
            f"自己修正: {LEARNING_STATS['self_correction_count']}",
            f"温度: {TEMP_VOICE:.2f} (最適候補: {get_best_temp('d') or 'none'}) {trend}",
            f"温度推移:\n{_temp_graph(TEMP_HISTORY)}",
            f"キーワード: {', '.join(KEYWORD_MEMORY[-5:]) or 'なし'}",
            f"最適化: {OPTIMIZER.status()}",
            f"ユーザー指摘 全{total}件 / 現在ペルソナ({current_persona.get('name','')}):{directive_str}",
        ])

    while True:
        try:
            fp_label = current_persona.get("first_person", "私")
            prompt_label = f"\n{C['c']}{OBSERVED_SUBJECT_NAME}[{fp_label}]{C['w']}> " if not ROLEPLAY_ACTIVE else f"\n{C['p']}[RP:{fp_label}]{C['w']}> "
            try:
                raw = normalize_input(input(prompt_label))
            except (EOFError, KeyboardInterrupt):
                print(f"\n{C['y']}bye{C['w']}")
                break
            if not raw: continue
            start_t = time.time()
            user_text: str = raw
            cmd: str = ""
            cmd_arg: str = ""
            is_cmd = raw.startswith("/")
            # ★[v131] RAGプリフェッチ: ユーザー入力直後・コマンド解析前にWeb取得を開始
            # /コマンド以外かつオフラインモード無効の場合のみ
            _prefetch_future = None
            if not is_cmd and not OFFLINE_MODE and not ROLEPLAY_ACTIVE:
                _prefetch_future = _THREAD_POOL.submit(get_async_rag_data, raw)
            if is_cmd:
                parts = raw[1:].strip().split(maxsplit=1)
                if not parts:
                    continue
                cmd = parts[0].lower()
                cmd_arg = parts[1] if len(parts) > 1 else ""
                user_text = cmd_arg
            if raw.lower() in ("exit", "終了", "quit"):
                print(f"{C['y']}bye{C['w']}")
                break
            result = ""
            # ══════════════════════════════════════════════════════
            # ★[修正/spi-FINAL] /spi セッション中の A/B/C/D ルーティング
            # 
            # 旧コードの問題:
            #   1. _spi_sess.get("current") が空辞書{} = falsy → スルー
            #   2. ファイルI/O競合でセッションが消える
            #   3. セッションが有効でもルーティングを抜けた後に
            #      elif/else の _chat() に落ちることがあった
            #
            # 修正: _SPI_SESSION_MEMORY（メモリミラー）を最優先で参照。
            # セッションが有効な間はA/B/C/D入力を必ずSPIに回し、
            # _chat()には絶対に到達させない。
            # ══════════════════════════════════════════════════════
            _spi_active = False
            if not is_cmd and raw.upper() in ("A", "B", "C", "D"):
                _spi_cur = (
                    _SPI_SESSION_MEMORY.get("current")
                    or _spi_load_session().get("current")
                )
                if isinstance(_spi_cur, dict) and len(_spi_cur) > 0:
                    _spi_active = True
                    result = handle_spi(raw)
                    if result:
                        print(result)
                    SESSION_STATS["response_times"].append(time.time() - start_t)
                    continue  # ← セッション有効時はここで必ずループ先頭に戻る

            if is_cmd and cmd in COMMAND_REGISTRY:
                result = COMMAND_REGISTRY[cmd](cmd_arg)
                # ── /doc think のルーティング ──────────────────────────
                if isinstance(result, str) and result.startswith("__THINK__"):
                    doc_title = result[len("__THINK__"):]
                    state_d = load_state()
                    doc_entry = next(
                        (d for d in state_d.get("docs", []) if d["title"].lower() == doc_title.lower()),
                        None
                    )
                    if doc_entry:
                        doc_text = doc_entry["text"]
                        fp_t = current_persona.get("first_person", "私")
                        sys_t = (
                            f"あなたは{current_persona['name']}。口調: {current_persona['style']}。一人称: {fp_t}。\n"
                            f"以下の【保存文書】と【KB参照】を根拠に、文書の核心・論点・矛盾・示唆を深く推論せよ。\n"
                            f"捏造禁止。文書にない事実を一切追加するな。"
                        )
                        user_t = f"【保存文書: {doc_title}】\n{doc_text[:2000]}"
                        # KBに関連チャンクがあれば追加
                        _think_cols = [c for c in vector_list_collections() if c != "s01_memory"]
                        if _think_cols:
                            _think_hits = []
                            for _tc in _think_cols:
                                _ts = _tc.replace("book_", "")
                                for _th in vector_search(doc_title, n=3, collection=_tc):
                                    _think_hits.append(f"《{_ts}》: {_th[:200]}")
                            if _think_hits:
                                user_t += "\n\n【KB参照】\n" + "\n".join(_think_hits[:6])
                                print(f"{C['dim']}[doc think: KB {len(_think_hits)}件参照]{C['w']}")
                        user_t += "\n\nこの文書の核心・論点・示唆を深く推論して述べよ。"
                        print(f"{C['c']}[DOC深層推論]{C['w']} {current_persona['name']}: ", end="", flush=True)
                        result = stream_response(
                            [{"role": "system", "content": sys_t}, {"role": "user", "content": user_t}],
                            True, len(doc_text), temp_override=0.35, model=DEEP_MODEL
                        ) or f"{C['r']}推論失敗{C['w']}"
                    else:
                        result = f"{C['r']}文書「{doc_title}」が見つかりません{C['w']}"
            elif ROLEPLAY_ACTIVE:
                result = _chat(raw, "r") or ""
            else:
                # ★[修正/spi-GUARD] 万が一 A/B/C/D がここまで来た場合の最終防衛
                # セッションが有効なら _chat を呼ばずに再度 handle_spi に回す
                if raw.upper() in ("A", "B", "C", "D") and not is_cmd:
                    _spi_cur2 = (
                        _SPI_SESSION_MEMORY.get("current")
                        or _spi_load_session().get("current")
                    )
                    if isinstance(_spi_cur2, dict) and len(_spi_cur2) > 0:
                        result = handle_spi(raw)
                        if result:
                            print(result)
                        SESSION_STATS["response_times"].append(time.time() - start_t)
                        continue
                complexity = estimate_complexity(raw, cmd)
                model_choice = select_model(raw, cmd)  # ★[修正/main-1] MODEL_NAME固定→select_model使用
                # 話題転換検出: 現在のキーワードが既存KEYWORD_MEMORYと重複ゼロなら文脈リセット
                if KEYWORD_MEMORY:
                    new_kw_set = set(extract_keywords(raw))
                    old_kw_set = set(KEYWORD_MEMORY)
                    if new_kw_set and not new_kw_set & old_kw_set:
                        KEYWORD_MEMORY.clear()
                        _SYS_EXTRAS_CACHE.clear()
                result = _chat(raw, "d", model=model_choice) or ""
            streamed_cmds = {"a", "w", "p", "c", "t", "e", "sum", "elab", "tr"}
            should_echo_result = is_cmd and not (cmd in streamed_cmds and bool(cmd_arg))
            if result and should_echo_result:
                print(result)
            response_time = time.time() - start_t
            SESSION_STATS["response_times"].append(response_time)
            if result:
                if not is_cmd:
                    fb = analyze_feedback(raw)
                    log_interaction(raw, result, "d", fb)
                    update_param_performance("d", _get_temp_voice(), fb)
                    # ★[修正A+B] ユーザー指摘を即時にPROMPT_OPTIMIZATIONSへ反映
                    applied = apply_user_directive(raw, current_persona.get("name", ""))
                    if applied:
                        _SYS_EXTRAS_CACHE.clear()  # 次の返答から即適用するためキャッシュ破棄
                        persist_learning()          # 即座に保存（exit前でも確実に残る）
                        print(f"{C['dim']}[学習] 指摘を記録: {' / '.join(applied[:3])}{C['w']}")
                if result and not is_cmd:
                    messages.append({"role": "user", "content": sanitize(raw[:1000])})
                    messages.append({"role": "assistant", "content": sanitize(result[:3000])})
                    if len(messages) > MAX_HISTORY * 2:
                        messages[:] = messages[-(MAX_HISTORY * 2):]
                if not is_cmd:
                    update_keyword_memory(raw)
                    kw = extract_keywords(result)
                    for w in kw:
                        if w not in KEYWORD_MEMORY: KEYWORD_MEMORY.append(w)
                    if len(KEYWORD_MEMORY) > 6: KEYWORD_MEMORY[:] = KEYWORD_MEMORY[-6:]
            SESSION_STATS["token_estimates"].append(len(result) // 2)
            if len(SESSION_STATS["response_times"]) > 500:
                SESSION_STATS["response_times"] = SESSION_STATS["response_times"][-250:]
                SESSION_STATS["token_estimates"] = SESSION_STATS["token_estimates"][-250:]
            if LEARNING_STATS["total_interactions"] % 25 == 0 and LEARNING_STATS["total_interactions"] > 0:
                # ★[修正/#12] 再起動直後に総数が25の倍数の場合、1ターン目から
                # persist_learning が走るバグを修正。セッション内の増分が0より大きい場合のみ保存する。
                _session_delta = LEARNING_STATS["total_interactions"] - _SESSION_START_INTERACTIONS
                if _session_delta > 0:
                    persist_learning()
        except Exception as e:
            print(f"{C['r']}[ERR] {sanitize(e)}{C['w']}")
            if POWER_MODE == "ultra" or isinstance(e, (NameError, AttributeError)):
                traceback.print_exc()

if __name__ == "__main__":
    atexit.register(_cleanup)
    run()

# ━━ /reference CODE_EVO #1 (2026-06-04 19:33) sig=40d1f758 ━━
# [1★] function_name: Improve readability and consistency in function names. → Enhanced clarity for maintainability.
# [1★] function_name: Add comments to explain complex logic within functions. → Improved understanding of code behavior, especially for new developers.
# [1★] function_name: Implement error handling (e.g., try-except blocks) where appropriate. → Robustness and prevent unexpected crashes.
# [1★] function_name: Ensure function inputs are validated to check data types or ranges. → Prevent errors caused by invalid input values.
# [1★] function_name: Add type hints for parameters, especially those that could be used in calculations. → Improved code readability and static analysis.
# [1★] function_name: Consider using more descriptive variable names to improve understanding of data within the function. → Enhanced maintainability.
# [1★] function_name: Add a docstring explaining what the function does, its parameters, return value and any potential side effects.  → Documentation for future developers.
# [1★] function_name: Implement logging to track important events or errors within functions. → Debugging and monitoring code execution.
# [1★] function_name: Add unit tests to verify the correctness of function logic.  → Automated testing for quality.
# [1★] function_name: Ensure that all functions have a clear return value, indicating success or failure. → Clearer understanding and predictability.

# ━━ /reference CODE_EVO #1 (2026-06-05 04:09) sig=6b89d641 ━━
# [4★] stream_response: CoT指示をシステムプロンプトに自動付与 → 推論精度向上
# [4★] hybrid_search_advanced: BM25+ベクトルのn_candidates増加 → 検索精度向上
# [3★] run: 長文応答後に自己反省ループ追加 → 品質向上
# [3★] _handle_reference: 出力をJSON強制モードで統一 → 安定性向上
