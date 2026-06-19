"""
dataset_builder.py — Unity 2022.3 LTS C# 코딩 어시스턴트 학습 데이터셋 빌더 (목표 100,000 samples)
환경: Ubuntu 22.04 / Python 3.10+  (datasets, transformers 필요. semantic dedup 시 sentence-transformers+faiss 권장)

────────────────────────────────────────────────────────────────────────────────
이 파일은 기존 `dataset.py`(같은 폴더) 위에 "강화 파이프라인"을 얹습니다.
dataset.py 의 소스 로더/헬퍼/유효성·해시 dedup/길이 필터/비율 믹스를 그대로 재사용하고,
요청 사양에 맞춰 다음을 추가합니다.

  [요청 사양 매핑]
  · 총 100,000 samples ........................ TARGET_TOTAL / build_unity_100k(target_total=...)
  · 우선순위 믹스 40/25/20/10/5 ............... MIX_UNITY_100K  (아래 매핑 참조)
  · 중복 제거(semantic similarity) ............ semantic_dedup()  (embedding+FAISS, 옵션)
  · deprecated / 2022.3 미동작 API 제거 ....... filter_deprecated_api() + DEPRECATED_*  패턴
  · 모호한 질문 제거 .......................... filter_ambiguous()
  · accepted answer 없는 Q&A 제거 ............. load_gamedev_curated()(score>=5 & accepted) /
                                                erocal 는 'resolved' 라 채택답변만 포함(소스 특성)
  · hallucination 감소 ........................ ① 공식문서(hypersniper) 우선비율 40% + inject_api_context
                                                ② source grounding 유지(API 컨텍스트를 system 에)
                                                ③ speculative 답변 필터 filter_speculative()
                                                ④ "정보 부족" 샘플 주입 make_abstain_dataset()
  · 출력 포맷 ................................. "messages" | "text" | "alpaca"(instruction/input/output)

  [데이터 우선순위 → 소스 매핑]  (build_dataset 의 mix 와 동일한 키 사용)
    40%  공식 Unity API 문서 기반   → hypersniper  (Hypersniper/unity_api_2022_3, 2022.3 API)
    25%  curated Unity instructions → vishnuoi     (vishnuOI/unity-dev-instructions)
    20%  resolved Q&A (accepted)    → erocal       (Erocal/Unity_ResolvedQuestions)
    10%  Unity code examples        → ibranze      (ibranze/codellama_unity3d_v2)
     5%  StackExchange (score>=5)   → gamedev      (mlfoundations-dev/stackexchange_gamedev)
                                                    (+ common_pile 는 기본 비활성: 노이즈 큼)

  [중요/주의]
  · 본 빌더는 "업샘플(복제) 금지" 를 지킵니다. 따라서 정제 후 실제 unique 가용량이 100k 미만이면
    비율을 유지한 채 가능한 최대치로 빌드하고 부족분을 리포트합니다(거짓 증량/환각 방지).
    100k 를 안전하게 채우는 정공법은 40% 슬롯(공식 API)을 공식 Scripting API 덤프로
    grounded 생성하는 것입니다 → build_grounded_api_samples() 훅 제공.
  · vishnuoi 는 ibranze 를 내포하므로 본 믹스에서는 ibranze(10%)와 동시 사용합니다.
    정확중복은 해시 dedup, 근사/의미중복은 semantic_dedup 으로 제거되어 이중카운트가 정리됩니다.
  · DEPRECATED_* 패턴/임계값은 "검증된 출발점"이며 완전하지 않습니다. 프로젝트에 맞게 확장하세요.

  [v3 버그수정 & 성능]
  · (🔴치명) abstain 규모 붕괴 수정: abstain 을 dedup 이전이 아니라 '비율 믹스 이후'에
    고유 생성하여 합침. make_abstain_dataset 은 템플릿/치환값을 대폭 확장하고 내부에서
    고유화하여, 요청 수만큼(가능 한도까지) 중복 없이 생성한다.
  · ibranze ⊂ vishnuOI 중복 정리: 해시 dedup 에 source_priority(ibranze>vishnuOI) 적용 →
    ibranze 슬롯이 비고, vishnuOI 는 고유분만 남아 25/10 비율이 보존된다.
  · semantic_dedup: 대규모(n≥2만)에서 HNSW(근사) 인덱스로 전환(누적 ~O(n log n)).
    기존 IndexFlatIP 는 누적 O(n^2)였음(주석도 정정).
  · load_gamedev_curated: 원본 1회 로드 + 메타 필터를 원본에 직접 적용(이중 로드 및
    base/raw 인덱스 정렬 붕괴 위험 제거).
  · 한국어 커버리지: 모호질문(_GENERIC_ONLY)·추측성(_SPECULATIVE)에 한국어 패턴 추가
    (한글에 불안정한 \b 의존 제거).
  · [성능] dedup·품질리포트 컬럼 일괄 fetch, 길이필터 batched 토크나이즈.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import json
import random
from collections import defaultdict
from typing import Optional

from datasets import Dataset, concatenate_datasets

# 기존 모듈 재사용 (같은 폴더에 dataset.py 가 있어야 함)
try:
    from dataset import (
        SYSTEM_PROMPT,
        _clean, _strip_html, _pair_to_messages, _ensure_system, _auto_messages,
        _normalized_fingerprint,
        _is_valid, _deduplicate, _filter_by_length, _cap_source, _apply_mix,
        load_unity_code, load_unity_qa, load_unity_api, load_unity_instructions,
        load_gamedev, load_common_pile,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "dataset_builder.py 는 dataset.py 와 같은 폴더에 있어야 합니다. "
        f"(원본 import 실패: {e})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 0. 우선순위 믹스 / 목표 규모
# ──────────────────────────────────────────────────────────────────────────────
TARGET_TOTAL = 100_000

# 요청 우선순위 그대로 (합 100). build_unity_100k 가 abstain 비율만큼 자동 스케일.
MIX_UNITY_100K = {
    "hypersniper": 40,   # 공식 Unity API 2022.3 문서 기반
    "vishnuoi":    25,   # curated instructions
    "erocal":      20,   # resolved Q&A (accepted only)
    "ibranze":     10,   # Unity code examples
    "gamedev":      5,   # StackExchange (score>=5 & accepted)
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Deprecated / Unity 2022.3 미동작 API 필터
#    REMOVED  = 2022.3 에서 제거/미동작(컴파일 불가 가능) → 항상 드롭
#    DEPRECATED = 동작은 하나 권장 안 함 → strict 모드에서 드롭, 기본은 드롭(요청: "최대한 제거")
#    ※ 검증된 출발점. 오탐을 줄이려고 경계(\b)와 대표 토큰만 사용. 필요 시 확장.
# ──────────────────────────────────────────────────────────────────────────────

# (A) 2022.3 에서 제거되었거나 동작하지 않는 API/시스템 → 하드 드롭
REMOVED_API_PATTERNS = [
    # 레거시 씬 로딩 (→ SceneManager)
    r"\bApplication\.LoadLevel(Async|Additive|AdditiveAsync)?\b",
    r"\bApplication\.loadedLevel(Name)?\b",
    r"\bApplication\.LoadLevelAdditive\b",
    r"\bOnLevelWasLoaded\b",
    # 레거시 WWW (→ UnityWebRequest)
    r"\bnew\s+WWW\b", r"\bWWWForm\b\s*(?=.*\bWWW\b)",
    # 레거시 네트워킹 (UNet HLAPI/LLAPI, RakNet) — 2018~2022 사이 제거
    r"\bNetworkView\b", r"\bNetworkServer\b", r"\bNetworkClient\b",
    r"\bNetworkTransport\b", r"\bMasterServer\b", r"\bNetworkMessageInfo\b",
    r"\bUnityEngine\.Networking\.NetworkManager\b",
    r"\[\s*RPC\s*\]",                       # 구 [RPC] 어트리뷰트
    # 레거시 파티클 시스템 (→ ParticleSystem)
    r"\bParticleEmitter\b", r"\bParticleAnimator\b", r"\bParticleRenderer\b",
    r"\bEllipsoidParticleEmitter\b", r"\bMeshParticleEmitter\b",
    # 레거시 GUI 컴포넌트
    r"\bGUIText\b", r"\bGUITexture\b", r"\bGUILayer\b",
    # 구 플랫폼 API
    r"\biPhoneSettings\b", r"\biPhoneInput\b", r"\biPhoneKeyboard\b", r"\biPhoneUtils\b",
    # 기타 제거/이동된 멤버
    r"\bApplication\.CaptureScreenshot\b",          # → ScreenCapture.CaptureScreenshot
    r"\bApplication\.RegisterLogCallback(Threaded)?\b",  # → logMessageReceived
    r"\bApplication\.ExternalEval\b",
    r"\bGameObject\.SetActiveRecursively\b",        # → SetActive
    r"\bDestroyObject\b",                           # → Destroy
    r"\bEditorGUIUtility\.LookLike(Controls|Inspector)\b",
    r"\bSecurity\.PrefetchSocketPolicy\b",
]

# (B) 동작은 하나 deprecated → 요청("최대한 제거")에 따라 기본 드롭
DEPRECATED_API_PATTERNS = [
    # Component 단축 접근자 (Unity 5에서 deprecated, 이후 제거) — 대표 타입만
    r"(?<![\w.])(?:this\.|gameObject\.)?(rigidbody|rigidbody2D|collider|collider2D|"
    r"renderer|audio|camera|light|animation|particleSystem|hingeJoint|networkView|"
    r"constantForce|guiText|guiTexture)\b\s*(?=[.\)\];,=])",
    # 에디터 콜백 구이름
    r"\bEditorApplication\.playmodeStateChanged\b",   # → playModeStateChanged
    r"\bComponent\.active\b",
    r"\bLight\.shadowConstantBias\b",
    r"\bTexture2D\.GetNativeTextureID\b",             # → GetNativeTexturePtr
]

_REMOVED_RE = re.compile("|".join(REMOVED_API_PATTERNS))
_DEPRECATED_RE = re.compile("|".join(DEPRECATED_API_PATTERNS))


def _assistant_text(msgs: list) -> str:
    return " ".join(m["content"] for m in msgs if m["role"] == "assistant")


def _full_text(msgs: list) -> str:
    return " ".join(m["content"] for m in msgs if m["role"] in ("user", "assistant"))


def filter_deprecated_api(ds: Dataset, *, strict: bool = True,
                          check_user: bool = False) -> Dataset:
    """
    deprecated/제거 API 가 '답변(assistant)'에 포함된 샘플 제거.
      strict=True  : REMOVED + DEPRECATED 모두 드롭 (요청 기본: "최대한 제거")
      strict=False : REMOVED 만 드롭 (deprecated 는 허용)
      check_user=True 면 질문(user)에도 같은 검사 적용(질문이 구 API 전제면 환각 유발 가능).
    """
    def _keep(ex):
        target = _full_text(ex["messages"]) if check_user else _assistant_text(ex["messages"])
        if _REMOVED_RE.search(target):
            return False
        if strict and _DEPRECATED_RE.search(target):
            return False
        return True

    before = len(ds)
    ds = ds.filter(_keep)
    print(f"[API  ] deprecated/removed 제거(strict={strict}): "
          f"{before:,} → {len(ds):,}행  (제거: {before - len(ds):,})")
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# 2. 모호한 질문 제거
# ──────────────────────────────────────────────────────────────────────────────
_GENERIC_ONLY = re.compile(
    r"^\W*(help|help me|fix(\s+this)?|please help|how|why|what|error|it("
    r"'s| is)?\s*not\s*work(ing)?|doesn'?t\s*work|bug|broken|plz|pls|"
    # 한국어 일반어 단독(맥락 없는 모호 질문)
    r"도와\s*줘|도와주세요|도와\s*주실래요|살려\s*줘|이거\s*왜\s*안\s*돼\??|"
    r"왜\s*안\s*되(나요|죠|지)?\??|안\s*돼요\??|안\s*됨|고쳐\s*줘|고쳐\s*주세요|"
    r"에러\s*(났|나요|뜸|떠요)|버그|모르겠어요?|어떻게\s*해요?\??)\W*$",
    re.IGNORECASE,
)
_HAS_ALNUM = re.compile(r"[A-Za-z0-9가-힣]")


def filter_ambiguous(ds: Dataset, *, min_user_chars: int = 12,
                     min_user_words: int = 3) -> Dataset:
    """
    학습에 쓰기엔 컨텍스트가 부족한 '모호한 질문' 제거.
      - user 메시지 길이 < min_user_chars
      - user 단어 수 < min_user_words (단, 코드블록이 있으면 통과)
      - "help / fix this / it doesn't work" 류 일반어 단독
      - 실질 내용(영문/숫자/한글) 없음
    """
    def _keep(ex):
        user = next((m["content"] for m in reversed(ex["messages"])
                     if m["role"] == "user"), "")
        u = _clean(user)
        if not _HAS_ALNUM.search(u):
            return False
        if _GENERIC_ONLY.match(u):
            return False
        has_code = "```" in u or re.search(r"[;{}]", u) is not None
        if len(u) < min_user_chars and not has_code:
            return False
        if len(u.split()) < min_user_words and not has_code:
            return False
        return True

    before = len(ds)
    ds = ds.filter(_keep)
    print(f"[AMBIG] 모호 질문 제거: {before:,} → {len(ds):,}행  (제거: {before - len(ds):,})")
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# 3. Speculative(추측성) 답변 제거 — hallucination 감소
#    근거 없는 추측을 보수적으로만 제거(코드/근거 없는 약한 답변 위주).
# ──────────────────────────────────────────────────────────────────────────────
_SPECULATIVE = re.compile(
    # 영어 추측성 마커 (단어 경계 사용)
    r"\b(i('| a)?m not sure|i'?m not certain|i think (it )?(might|may|could)|"
    r"not 100% sure|i guess|probably (try|it'?s)|maybe (try|you)|"
    r"i don'?t (really )?know but)\b"
    # 한국어 마커: \b 는 한글에 신뢰성이 낮아 경계 없이 매칭(공백 유무 무관)
    r"|(아마도|아마\s*맞|확실하지\s*않|잘\s*모르겠지만|모르겠지만|"
    r"추측(으로|컨대|건대)?|장담은\s*못|확신은\s*없|아닐\s*수도|일\s*수도\s*있)",
    re.IGNORECASE,
)


def filter_speculative(ds: Dataset, *, max_markers_without_code: int = 1) -> Dataset:
    """
    추측성 마커가 있고 코드블록도 근거(API 토큰)도 없는 '저신뢰' 답변 제거.
    (정당한 hedging 까지 죽이지 않도록 '코드/근거 없음' 조건을 함께 요구 — 보수적)
    """
    def _keep(ex):
        asst = _assistant_text(ex["messages"])
        markers = len(_SPECULATIVE.findall(asst))
        if markers == 0:
            return True
        has_code = "```" in asst
        # API/식별자 근거: PascalCase.method 패턴 존재 여부
        has_api = re.search(r"\b[A-Z][A-Za-z0-9]+\.[A-Za-z]", asst) is not None
        if not has_code and not has_api and markers >= max_markers_without_code:
            return False
        return True

    before = len(ds)
    ds = ds.filter(_keep)
    print(f"[SPEC ] 추측성 답변 제거: {before:,} → {len(ds):,}행  (제거: {before - len(ds):,})")
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# 4. Semantic dedup (의미 기반 근사중복 제거)
#    embedding(sentence-transformers) + FAISS 증분 인덱스로 O(n·k) 근사.
#    의존성 없으면 경고 후 skip(해시 dedup 으로 대체).
# ──────────────────────────────────────────────────────────────────────────────

def semantic_dedup(ds: Dataset, *, threshold: float = 0.92,
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   text_mode: str = "user", batch_size: int = 256,
                   keep: str = "first", use_hnsw: str = "auto",
                   hnsw_threshold_n: int = 20_000) -> Dataset:
    """
    의미 유사도 기반 중복 제거.
      threshold : 코사인 유사도 임계값(>= 이면 중복으로 간주). 0.90~0.95 권장.
      text_mode : 'user'(질문 기준, 권장) | 'pair'(질문+답변)
      keep      : 'first'(먼저 등장 유지) | 'longest'(답변 긴 것 유지)
      use_hnsw  : 'auto'(n>=hnsw_threshold_n 면 HNSW) | True | False

    [v3 성능 정정] 기존 IndexFlatIP 는 '정확' brute-force 라 search 가 O(현재 ntotal),
    누적 O(n^2) 였다(이전 주석의 'O(n^2) 회피'는 오류). 본 버전은 대규모(n>=2만)에서
    IndexHNSWFlat(근사)로 전환해 search 를 ~O(log n) 으로 낮춘다(누적 ~O(n log n)).
    소규모는 정확 IndexFlatIP 유지. 두 경로 모두 '증분 추가' 방식이라 keep 순서가 보존된다.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss  # faiss-cpu
    except Exception as e:  # pragma: no cover
        print(f"[SEMDUP] sentence-transformers/faiss 미설치 → semantic dedup 건너뜀 ({e}). "
              f"`pip install sentence-transformers faiss-cpu` 권장.")
        return ds

    def _text(ex):
        if text_mode == "pair":
            return _full_text(ex["messages"])
        return next((m["content"] for m in reversed(ex["messages"])
                     if m["role"] == "user"), "")

    msgs_col = ds["messages"]          # [성능] 컬럼 일괄 fetch
    n = len(msgs_col)
    if n == 0:
        return ds

    texts = [_text({"messages": m}) for m in msgs_col]

    # 'longest' 유지를 위해 답변 길이 내림차순 정렬(긴 답변이 먼저 등장 → keep)
    order = list(range(n))
    if keep == "longest":
        asst_len = [len(_assistant_text(m)) for m in msgs_col]
        order.sort(key=lambda i: asst_len[i], reverse=True)

    print(f"[SEMDUP] 임베딩 인코딩 중... (n={n:,}, model={model_name})")
    model = SentenceTransformer(model_name)
    emb = model.encode([texts[i] for i in order], batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]

    want_hnsw = (use_hnsw is True) or (use_hnsw == "auto" and n >= hnsw_threshold_n)
    if want_hnsw:
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        print(f"[SEMDUP] HNSW 인덱스 사용(근사, n={n:,}≥{hnsw_threshold_n:,}) → ~O(n log n)")
    else:
        index = faiss.IndexFlatIP(dim)
        print(f"[SEMDUP] Flat 인덱스 사용(정확, n={n:,})")

    keep_local = []
    for pos in range(n):
        v = emb[pos:pos + 1]
        if index.ntotal > 0:
            sims, _ = index.search(v, 1)
            if float(sims[0][0]) >= threshold:
                continue   # 의미 중복 → 드롭
        index.add(v)
        keep_local.append(pos)

    keep_global = sorted(order[p] for p in keep_local)
    removed = n - len(keep_global)
    print(f"[SEMDUP] 의미중복 제거(thr={threshold}, mode={text_mode}, "
          f"index={'HNSW' if want_hnsw else 'Flat'}): "
          f"{n:,} → {len(keep_global):,}행  (제거: {removed:,})")
    return ds.select(keep_global)


# ──────────────────────────────────────────────────────────────────────────────
# 5. accepted-answer-only / score 필터가 강화된 gamedev 로더
# ──────────────────────────────────────────────────────────────────────────────
_UNITY_KW = re.compile(
    r"\b(unity|monobehaviour|gameobject|transform|rigidbody|prefab|"
    r"scriptableobject|coroutine|unityengine|urp|hdrp|shaderlab|unityeditor)\b",
    re.IGNORECASE,
)


def load_gamedev_curated(*, min_score: int = 5, unity_only: bool = True) -> Dataset:
    """
    mlfoundations-dev/stackexchange_gamedev 를 score>=min_score & accepted answer 로만 필터.

    [v3] 원본을 1회만 로드하고, 메타 필터를 '원본'에 적용한 뒤 그 행들만 messages 로 변환한다.
         (이전: base=load_gamedev() 와 raw 를 각각 로드 → len 일치 가정으로 select. 로더에
          행 필터가 추가되면 인덱스 정렬이 조용히 깨졌음. 본 버전은 그 의존성을 제거.)
      ※ 가공본이라 컬럼명이 확정적이지 않음 → 가능한 필드명을 방어적으로 탐색.
        score/accepted 메타가 전혀 없으면 휴리스틱(답변 길이+코드블록)으로 대체.
      unity_only=True 면 Unity 키워드 없는 행 제거(엔진 혼재 방지).
    """
    from datasets import load_dataset
    raw = load_dataset("mlfoundations-dev/stackexchange_gamedev")
    raw = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
    cols = set(raw.column_names)

    score_keys = [k for k in ("score", "answer_score", "votes", "answer_votes") if k in cols]
    acc_keys = [k for k in ("accepted", "is_accepted", "answer_accepted",
                            "accepted_answer_id", "is_answer_accepted") if k in cols]

    def _to_msgs(ex):
        """원본 행 → {messages, source='gamedev'} (HTML strip 포함, dataset.load_gamedev 와 동일 규칙)"""
        msgs = _auto_messages(ex) or []
        for m in msgs:
            if m["role"] in ("user", "assistant"):
                m["content"] = _strip_html(m["content"])
        return {"messages": msgs, "source": "gamedev"}

    if not score_keys and not acc_keys:
        print(f"[GAMEDV] score/accepted 메타 없음(cols={sorted(cols)}). "
              f"휴리스틱(답변길이≥120 & 코드블록 우대)으로 대체.")
        ds = raw.map(_to_msgs, remove_columns=raw.column_names)

        def _heur(ex):
            asst = _assistant_text(ex["messages"])
            if unity_only and not _UNITY_KW.search(_full_text(ex["messages"])):
                return False
            return len(asst) >= 120

        before = len(ds)
        ds = ds.filter(_heur)
        print(f"[GAMEDV] 휴리스틱 필터: {before:,} → {len(ds):,}행")
        return ds

    def _meta_ok(ex):
        ok_score = True
        for k in score_keys:
            try:
                ok_score = ok_score and (int(ex[k]) >= min_score)
            except (TypeError, ValueError):
                pass
        ok_acc = True
        if acc_keys:
            ok_acc = False
            for k in acc_keys:
                v = ex[k]
                if isinstance(v, bool):
                    ok_acc = ok_acc or v
                elif isinstance(v, (int, float)):
                    ok_acc = ok_acc or (v not in (0, None))
                elif isinstance(v, str):
                    ok_acc = ok_acc or (v.strip() not in ("", "0", "false", "False"))
        return ok_score and ok_acc

    # 메타 필터를 '원본'에 직접 적용 → 행/메타 정렬이 항상 일치
    before = len(raw)
    raw = raw.filter(_meta_ok)
    print(f"[GAMEDV] score>={min_score} & accepted 필터: {before:,} → {len(raw):,}행 "
          f"(score_keys={score_keys}, acc_keys={acc_keys})")

    sel = raw.map(_to_msgs, remove_columns=raw.column_names)

    if unity_only:
        before = len(sel)
        sel = sel.filter(lambda ex: _UNITY_KW.search(_full_text(ex["messages"])) is not None)
        print(f"[GAMEDV] unity_only: {before:,} → {len(sel):,}행")
    return sel


# ──────────────────────────────────────────────────────────────────────────────
# 6. "정보 부족" (abstain) 샘플 — hallucination 감소용 거절/되묻기 학습
# ──────────────────────────────────────────────────────────────────────────────
_ABSTAIN_TEMPLATES = [
    # (user, assistant)  — 모두 grounded/안전. 환각 대신 '모름/되묻기/현대 대안 안내'를 학습.
    # 가능한 한 2개 이상 치환변수를 써서 고유 조합 수를 확보한다(업샘플 없이 비율 충족).
    ("Unity에서 {api} 어떻게 써?",
     "질문만으로는 정확히 답하기 어렵습니다. '{api}' 가 어떤 네임스페이스/클래스의 멤버인지, "
     "그리고 무엇을 하려는지(예: 입력 처리, 물리, UI)를 알려주시면 Unity 2022.3 기준으로 "
     "공식 API에 근거해 정확히 안내하겠습니다. 확인되지 않은 시그니처를 추측해서 답하지 않겠습니다."),
    ("{system} 만드는 중인데 {api} 에러 나요. 왜 안 돼?",
     "{system} 관련 코드와 에러 메시지가 없어서 '{api}' 사용 맥락을 단정할 수 없습니다. "
     "문제가 되는 코드 블록, 콘솔 에러 전문, Unity 버전(2022.3.x 권장)을 함께 주시면 "
     "재현 후 정확히 진단하겠습니다."),
    ("{api} 의 최신 시그니처가 뭐야?",
     "제가 확신할 수 없는 정확한 시그니처는 추측하지 않겠습니다. Unity 2022.3 Scripting API "
     "문서에서 '{api}' 항목을 확인하시는 것을 권장합니다. 확인되면 해당 시그니처에 맞춰 "
     "예제를 작성해 드리겠습니다."),
    ("{legacy} 써서 {system} 짜줘",
     "'{legacy}' 는 Unity 2022.3에서 더 이상 권장되지 않거나 제거되었습니다. 추측으로 동작하는 "
     "척하기보다는, 현행 권장 API로 대체해 작성하는 것이 안전합니다. {system} 에서 어떤 동작을 "
     "원하시는지 알려주시면 2022.3에서 동작하는 현대적 방식으로 구현해 드리겠습니다."),
    ("{genre} 게임에서 {system} 어떻게 구현해?",
     "'{genre}' 라는 장르만으로는 '{system}' 의 요구사항을 특정하기 어렵습니다. 핵심 메커닉, "
     "차원(2D/3D), 대상 플랫폼을 알려주시면 Unity 2022.3 기준으로 구조와 스크립트를 단계적으로 "
     "제안하겠습니다. 막연한 가정으로 코드를 지어내지는 않겠습니다."),
    ("{api} 와 {api2} 중에 뭐가 더 빨라?",
     "맥락 없이 '{api}' 와 '{api2}' 의 성능을 단정하긴 어렵습니다. 호출 빈도, 대상 플랫폼, 프로파일링 "
     "결과(예: Unity Profiler 의 CPU/GC 항목)를 알려주시면 측정 기반으로 비교해 드리겠습니다. "
     "근거 없는 성능 단정은 피하겠습니다."),
    ("{system} 에서 {api} 쓰면 에러 나는데 고쳐줘",
     "에러 메시지 전문과 해당 코드가 없으면 '{api}' 사용 맥락을 정확히 파악할 수 없습니다. "
     "스택트레이스, Unity 버전, 재현 단계를 주시면 {system} 맥락에 맞춰 진단하겠습니다."),
    ("{legacy} 가 deprecated 됐다는데 {system} 에선 뭘로 바꿔?",
     "'{legacy}' 는 Unity 2022.3 기준 권장되지 않습니다. 다만 '{system}' 에서의 정확한 대체 API는 "
     "원하시는 동작에 따라 달라지므로, 무엇을 하려는지 알려주시면 공식 문서에 근거한 현행 "
     "대체안을 제시하겠습니다."),
]
_ABSTAIN_APIS = [
    "DoThing()", "Foo.Bar", "GetComponentMagic", "Auto-everything", "MakeItWork",
    "SuperMove()", "FastUpdate", "Thing.Process", "GameManager.Run", "Helper.Fix",
    "Entity.Spawn", "Player.DoMagic", "World.Tick", "Net.Sync", "Save.All",
    "AI.Decide", "Pool.Grab", "UI.Refresh", "Audio.PlayMagic", "Cam.Track",
]
_ABSTAIN_LEGACY = [
    "Application.LoadLevel", "WWW", "NetworkView", "GUIText", "ParticleEmitter",
    "OnLevelWasLoaded", "rigidbody.velocity", "Application.CaptureScreenshot",
    "GUITexture", "DestroyObject", "NetworkServer", "iPhoneSettings",
    "GameObject.SetActiveRecursively", "Application.RegisterLogCallback", "GUILayer",
]
_ABSTAIN_SYSTEM = [
    "인벤토리 시스템", "이동 컨트롤러", "세이브/로드", "적 AI", "UI 매니저",
    "오디오 매니저", "오브젝트 풀링", "대화 시스템", "퀘스트 시스템", "카메라 추적",
    "스킬 시스템", "상점 시스템", "스폰 매니저", "씬 전환", "입력 시스템",
    "체력바 UI", "미니맵", "업적 시스템", "튜토리얼", "네트워크 동기화",
]
_ABSTAIN_GENRE = [
    "아무", "재밌는", "RPG", "플랫포머", "슈팅", "퍼즐", "오픈월드",
    "로그라이크", "타워디펜스", "리듬", "생존", "방치형", "레이싱", "디펜스",
]


def make_abstain_dataset(n: int, *, seed: int = 42) -> Dataset:
    """
    근거 없는 추측 대신 '정보 부족/되묻기/현대 대안' 을 학습시키는 합성 샘플 생성.

    [v3] 모든 (템플릿 × 사용 치환값) 조합을 '결정론적으로 전수 열거'하여 고유 집합을 만든
    뒤, seed 로 섞어 앞에서 n 개를 취한다(랜덤 재시도보다 정확/고속). 고유 한도(ceiling)를
    넘는 n 은 요청해도 한도까지만 반환한다(업샘플=복제 금지). 본 함수 산출물은 이미 고유라
    메인 파이프라인의 dedup 으로 소실되지 않으며, '비율 믹스 이후'에 합쳐진다.
    """
    from itertools import product
    valuemap = {"api": _ABSTAIN_APIS, "api2": _ABSTAIN_APIS,
                "legacy": _ABSTAIN_LEGACY, "system": _ABSTAIN_SYSTEM,
                "genre": _ABSTAIN_GENRE}
    all_fields = list(valuemap)

    uniq: dict = {}  # fingerprint -> {"messages": ...}
    for u_tmpl, a_tmpl in _ABSTAIN_TEMPLATES:
        used = [f for f in all_fields if ("{" + f + "}") in u_tmpl
                or ("{" + f + "}") in a_tmpl]
        combos = ([{}] if not used
                  else [dict(zip(used, vals))
                        for vals in product(*[valuemap[f] for f in used])])
        for fmt in combos:
            u = u_tmpl.format(**fmt)
            a = a_tmpl.format(**fmt)
            msgs = _pair_to_messages(u, a)
            uniq.setdefault(_normalized_fingerprint(msgs),
                            {"messages": msgs, "source": "abstain"})

    pool = list(uniq.values())
    ceiling = len(pool)
    random.Random(seed).shuffle(pool)
    target = max(int(n), 0)
    rows = pool[:target]

    if target > ceiling:
        print(f"[ABSTN] 고유 abstain 한도 도달: 요청 {target:,} → 생성 {len(rows):,}행 "
              f"(고유 한도={ceiling:,}. 템플릿/치환값을 늘리면 증량 가능, 업샘플은 안 함)")
    print(f"[ABSTN] 정보부족 샘플 생성: {len(rows):,}행 (고유, 한도 {ceiling:,})")
    return Dataset.from_list(rows) if rows else Dataset.from_dict(
        {"messages": [], "source": []})


# ──────────────────────────────────────────────────────────────────────────────
# 7. (옵션) 공식 Scripting API 덤프 → grounded API 샘플 (40% 슬롯 안전 증량용)
# ──────────────────────────────────────────────────────────────────────────────
def build_grounded_api_samples(api_jsonl: str, *, source: str = "hypersniper",
                               per_member: int = 1) -> Dataset:
    """
    공식 Unity Scripting API 를 크롤/덤프한 JSONL 로 grounded Q&A 를 생성.
      입력 JSONL 한 줄 예:
        {"member":"Rigidbody.AddForce", "summary":"Adds a force to the Rigidbody.",
         "signature":"public void AddForce(Vector3 force, ForceMode mode);",
         "url":"https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Rigidbody.AddForce.html",
         "example":"rb.AddForce(Vector3.up * 10f, ForceMode.Impulse);"}
    → answer 는 summary/signature/example 만 사용(speculation 금지, source grounding 유지).
      source="hypersniper" 로 두면 40% 슬롯에 그대로 합산됩니다.
    """
    rows = []
    with open(api_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            member = _clean(ex.get("member"))
            summary = _clean(ex.get("summary"))
            if not member or not summary:
                continue
            sig = _clean(ex.get("signature"))
            example = _clean(ex.get("example"))
            url = _clean(ex.get("url"))
            q = f"Unity 2022.3에서 {member} 는 무엇을 하나요? 사용법을 알려주세요."
            parts = [summary]
            if sig:
                parts.append(f"\n시그니처:\n```csharp\n{sig}\n```")
            if example:
                parts.append(f"\n예시:\n```csharp\n{example}\n```")
            if url:
                parts.append(f"\n근거(공식 문서): {url}")
            a = "".join(parts)
            rows.append({"messages": _pair_to_messages(q, a), "source": source})
    print(f"[GROUND] 공식 API grounded 샘플: {len(rows):,}행  (from {api_jsonl})")
    return Dataset.from_list(rows) if rows else Dataset.from_dict(
        {"messages": [], "source": []})


# ──────────────────────────────────────────────────────────────────────────────
# 8. 출력 포맷 변환 — messages / text / alpaca(instruction/input/output)
# ──────────────────────────────────────────────────────────────────────────────
def _messages_to_alpaca(msgs: list, *, keep_system: bool = True) -> dict:
    """
    messages → {instruction, input, output}.
      - 단일 turn: instruction=user, input="", output=assistant
      - 멀티 turn: 이전 대화를 instruction 에 역할 태그로 평탄화, output=마지막 assistant
      - keep_system=True 면 system 프롬프트를 instruction 앞에 [System] 으로 보존
    """
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    convo = [m for m in msgs if m["role"] in ("user", "assistant")]
    output = convo[-1]["content"] if convo and convo[-1]["role"] == "assistant" else ""
    history = convo[:-1] if output else convo

    if len([m for m in history if m["role"] == "user"]) <= 1 and len(history) <= 1:
        instruction = history[0]["content"] if history else ""
    else:
        lines = []
        for m in history:
            tag = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"[{tag}] {m['content']}")
        instruction = "\n".join(lines)

    if keep_system and system:
        instruction = f"[System] {system}\n{instruction}".strip()

    return {"instruction": _clean(instruction), "input": "", "output": _clean(output)}


def to_output_format(ds: Dataset, output_format: str, tokenizer=None, *,
                     keep_system: bool = True) -> Dataset:
    if output_format == "messages":
        return ds.remove_columns([c for c in ds.column_names if c != "messages"])
    if output_format == "alpaca":
        def _m(ex):
            return _messages_to_alpaca(ex["messages"], keep_system=keep_system)
        return ds.map(_m, remove_columns=ds.column_names)
    if output_format == "text":
        if tokenizer is None:
            raise ValueError("output_format='text' 는 tokenizer 가 필요합니다.")
        def _t(ex):
            return {"text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)}
        return ds.map(_t, remove_columns=ds.column_names)
    raise ValueError("output_format 은 'messages' | 'alpaca' | 'text' 중 하나여야 합니다.")


# ──────────────────────────────────────────────────────────────────────────────
# 9. 품질 평가 리포트
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_quality(ds: Dataset) -> dict:
    """소스별 품질지표: 건수/평균길이/코드블록비율/deprecated적발률/질문형비율."""
    stats = defaultdict(lambda: {"n": 0, "u_len": 0, "a_len": 0,
                                 "code": 0, "dep": 0, "q": 0})
    msgs_col = ds["messages"]                                   # [성능] 컬럼 일괄 fetch
    src_col = ds["source"] if "source" in ds.column_names else ["?"] * len(msgs_col)
    for msgs, src in zip(msgs_col, src_col):
        s = stats[src]
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        asst = _assistant_text(msgs)
        s["n"] += 1
        s["u_len"] += len(user)
        s["a_len"] += len(asst)
        s["code"] += 1 if "```" in asst else 0
        s["dep"] += 1 if (_REMOVED_RE.search(asst) or _DEPRECATED_RE.search(asst)) else 0
        s["q"] += 1 if "?" in user else 0

    print("\n[QUALITY] 소스별 품질지표")
    print(f"  {'source':<12}{'n':>8}{'avgU':>7}{'avgA':>7}{'code%':>7}{'dep%':>7}{'질문%':>7}")
    out = {}
    for src in sorted(stats):
        s = stats[src]
        n = max(s["n"], 1)
        row = {
            "n": s["n"],
            "avg_user_len": round(s["u_len"] / n, 1),
            "avg_asst_len": round(s["a_len"] / n, 1),
            "code_ratio": round(s["code"] / n, 3),
            "deprecated_ratio": round(s["dep"] / n, 3),
            "question_ratio": round(s["q"] / n, 3),
        }
        out[src] = row
        print(f"  {src:<12}{s['n']:>8,}{row['avg_user_len']:>7.0f}"
              f"{row['avg_asst_len']:>7.0f}{row['code_ratio']*100:>7.1f}"
              f"{row['deprecated_ratio']*100:>7.1f}{row['question_ratio']*100:>7.1f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 10. 메인 빌더 — 100k 강화 파이프라인
# ──────────────────────────────────────────────────────────────────────────────
def build_unity_100k(
    tokenizer,
    *,
    mix: Optional[dict] = None,
    target_total: int = TARGET_TOTAL,
    caps: Optional[dict] = None,
    extra_sources: Optional[dict] = None,   # 예: {"hypersniper": build_grounded_api_samples(...)}
    abstain_ratio: float = 0.02,            # "정보 부족" 샘플 비율(최종 대비)
    max_length: int = 2048,
    # 정제 토글
    strict_deprecated: bool = True,
    semantic: bool = True,
    semantic_threshold: float = 0.92,
    gamedev_min_score: int = 5,
    # 출력
    output_format: str = "messages",        # "messages" | "alpaca" | "text"
    val_ratio: float = 0.05,
    seed: int = 42,
    inject_api_context: bool = True,        # 공식 API 컨텍스트를 system 에 grounding
):
    """
    전체 파이프라인:
      load(소스별, gamedev=score>=N&accepted, hyper=grounding)  ← 지식소스만
      → (+grounded API extra)
      → 유효성 → 모호질문 → deprecated/2022.3미동작 → speculative
      → 해시 dedup(정확+근사, ibranze>vishnuOI 우선) → semantic dedup(의미)
      → 길이 필터 → 비율 믹스(40/25/20/10/5, knowledge_target)
      → [믹스 이후] abstain 합성 주입(최종 abstain_ratio 맞춤, 고유 생성)
      → 품질 리포트 → 출력 포맷 → 95:5 분리
    반환: (train_ds, eval_ds, quality_report)

    [v3 핵심 수정] abstain 을 dedup 이전에 합치면 템플릿 중복이 해시 dedup 에서 소실되어
    _apply_mix 의 feasible 이 abstain 기준으로 결정되며 전체 규모가 붕괴했다. 이제 abstain 은
    비율 믹스가 끝난 뒤 지식분 규모에 맞춰 '고유' 생성하여 합치므로 규모 붕괴가 없다.
    """
    mix = dict(mix or MIX_UNITY_100K)
    caps = caps or {}
    extra_sources = extra_sources or {}
    abstain_ratio = max(0.0, min(abstain_ratio, 0.5))   # 방어적 clamp

    # 지식소스 믹스는 100%로 정규화. abstain 은 파이프라인을 거치지 않고 '믹스 이후' 합친다.
    knowledge_mix = {k: v for k, v in mix.items() if k != "abstain"}
    knowledge_target = int(round(target_total * (1.0 - abstain_ratio)))

    print("=" * 74)
    print("[BUILD] Unity 2022.3 100k 빌더 시작")
    print(f"        목표 {target_total:,} / 믹스 {knowledge_mix} / abstain {abstain_ratio:.0%}")
    print("=" * 74)

    # ── 1) 소스 로드 ─────────────────────────────────────────────────────────
    loaded = []
    loaders = {
        "hypersniper": lambda: load_unity_api(inject_api_context=inject_api_context),
        "vishnuoi":    load_unity_instructions,
        "erocal":      load_unity_qa,          # 'resolved' = accepted answer only
        "ibranze":     load_unity_code,
        "gamedev":     lambda: load_gamedev_curated(min_score=gamedev_min_score),
        "common_pile": load_common_pile,
    }
    for name in knowledge_mix:
        if name not in loaders:
            print(f"[WARN ] 알 수 없는 소스 '{name}' 건너뜀")
            continue
        print(f"[DATA ] {name} 로드 중...")
        ds = loaders[name]()
        ds = _cap_source(ds, caps.get(name), seed)
        print(f"[DATA ] {name}: {len(ds):,}행")
        loaded.append(ds)

    # ── 2) 외부 보강(grounded API 등) 병합 ──────────────────────────────────
    for name, eds in extra_sources.items():
        if "source" not in eds.column_names:
            eds = eds.add_column("source", [name] * len(eds))
        eds = _cap_source(eds, caps.get(name), seed)
        print(f"[DATA ] (extra) {name}: {len(eds):,}행")
        loaded.append(eds)

    combined = concatenate_datasets(loaded)
    print(f"[DATA ] 병합(지식소스): {len(combined):,}행")

    # ── 3) 정제 단계 ─────────────────────────────────────────────────────────
    before = len(combined)
    combined = combined.filter(_is_valid)
    print(f"[VALID] 유효성: {before:,} → {len(combined):,}행")

    combined = filter_ambiguous(combined)
    combined = filter_deprecated_api(combined, strict=strict_deprecated)
    combined = filter_speculative(combined)

    # 정확+근사 해시 dedup. ibranze ⊂ vishnuoi 충돌 시 ibranze 판을 남겨 슬롯 비율 보존.
    present = set(combined["source"])
    dedup_priority = (["ibranze", "vishnuoi"]
                      if {"ibranze", "vishnuoi"} <= present else None)
    combined = _deduplicate(combined, source_priority=dedup_priority)
    # 의미 기반 dedup
    if semantic:
        combined = semantic_dedup(combined, threshold=semantic_threshold,
                                  text_mode="user", keep="longest")

    # 길이 필터(학습 seq_len 과 동일하게)
    combined = _filter_by_length(combined, tokenizer, max_length)

    # ── 4) 비율 믹스 (지식소스만, 정제 후라 비율이 정확) ────────────────────
    #     _apply_mix: 업샘플 없음 → 가용량 부족 시 비율 유지한 채 최대치로 빌드
    combined = _apply_mix(combined, knowledge_mix, knowledge_target, seed)
    n_knowledge = len(combined)

    # ── 5) abstain 합성 주입 (믹스 이후!) ───────────────────────────────────
    #     [핵심 수정] 이전 버전은 abstain 을 dedup 이전에 합쳐 해시 dedup 으로 ~수십 개까지
    #     소실 → _apply_mix 의 feasible 가 abstain 기준으로 결정되며 전체 규모가 붕괴했다.
    #     이제는 (1) 믹스로 지식분을 먼저 확정하고 (2) 그 규모에 맞춘 abstain 개수를
    #     '고유 생성'하여 합친다. abstain 은 별도 dedup 을 거치지 않으므로 소실되지 않는다.
    if abstain_ratio > 0 and n_knowledge > 0:
        # 최종 비율이 abstain_ratio 가 되도록: n_abstain = N*r/(1-r)
        n_abstain = int(round(n_knowledge * abstain_ratio / (1.0 - abstain_ratio)))
        abstain_ds = make_abstain_dataset(n_abstain, seed=seed)
        if len(abstain_ds) > 0:
            combined = concatenate_datasets([combined, abstain_ds]).shuffle(seed=seed)
        actual_r = len(abstain_ds) / max(len(combined), 1)
        print(f"[ABSTN] 주입 후 비율: {actual_r:.1%} (목표 {abstain_ratio:.0%}, "
              f"지식 {n_knowledge:,} + abstain {len(abstain_ds):,})")

    if len(combined) < target_total:
        print(f"[NOTE ] 실제 unique 빌드량 {len(combined):,} < 목표 {target_total:,}. "
              f"업샘플(복제) 금지 정책상 부족분은 채우지 않았습니다. "
              f"40% 슬롯을 build_grounded_api_samples()로 증량하면 안전하게 100k 도달 가능.")

    # ── 6) 품질 리포트 (출력 포맷 변환 전, source 컬럼 살아있을 때) ──────────
    quality = evaluate_quality(combined)

    # ── 7) 출력 포맷 + 분리 ──────────────────────────────────────────────────
    formatted = to_output_format(combined, output_format, tokenizer=tokenizer)
    split = formatted.train_test_split(test_size=val_ratio, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[SPLIT] 학습 {len(train_ds):,} / 검증 {len(eval_ds):,}  "
          f"(format={output_format}, {1-val_ratio:.0%}:{val_ratio:.0%})")
    return train_ds, eval_ds, quality


# ──────────────────────────────────────────────────────────────────────────────
# 사용 예시
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # from transformers import AutoTokenizer
    # tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    #
    # # (A) 기본: 40/25/20/10/5 + abstain 2%, 의미중복 제거, 100k 목표
    # train, val, q = build_unity_100k(tok, output_format="alpaca")
    #
    # # (B) 100k 안전 도달: 공식 Scripting API 덤프로 40% 슬롯 grounded 증량
    # api_extra = build_grounded_api_samples("unity_2022_3_scriptref.jsonl")
    # train, val, q = build_unity_100k(
    #     tok, extra_sources={"hypersniper": api_extra},
    #     semantic_threshold=0.93, output_format="messages")
    #
    # # (C) 저사양(빠른 실험): semantic dedup 끄고 strict 완화
    # train, val, q = build_unity_100k(tok, semantic=False, strict_deprecated=False,
    #                                  target_total=30000)
    #
    # # 저장
    # train.to_json("unity_100k_train.jsonl", force_ascii=False)
    # val.to_json("unity_100k_val.jsonl", force_ascii=False)
    pass
