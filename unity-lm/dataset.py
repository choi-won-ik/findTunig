"""
dataset.py — Unity 코딩 어시스턴트 학습 데이터 전처리 모듈
환경: Ubuntu 22.04 / Python 3.10+

[컬럼 사용/제외 규칙 — 표 기준]
  ibranze       question→user, answer→assistant                          (제외 없음)
  Hypersniper   question, answer                                          (topic/api는 단독 학습 제외, 중복행 제거)
  Erocal(messages)  messages 그대로                                       (system 통일 프롬프트로 정규화)
  vishnuOI      instruction(+input)→user, output→assistant               (id/메타 제외, ibranze 중복 dedup)
  gamedev       질문 본문→user, 채택/고득표 답변→assistant                (저득표·코드없는 잡음, HTML, 비-Unity 제거)
  common-pile   text(필터 후)                                            (metadata 제외, gamedev.SE/Unity 슬라이스만)

파이프라인:
  source별 load → ChatML(messages) 정규화 → (소스별 상한 샘플링) → 병합
  → 유효성 검사 → 중복 제거(정확+근사) → 길이 필터 → 95:5 분리

[중요 변경점 vs 이전 버전]
  1) 기본 출력은 "messages" 형식.  SFTTrainer에서 completion-only(assistant 토큰만 손실)를
     쓰려면 SFTConfig(assistant_only_loss=True) 로 학습하고 dataset_text_field 는 주지 않습니다.
     (구버전처럼 전체 text 손실이 필요하면 output_format="text" 로 호출)
  2) max_length 기본값 1024 → 2048.  ibranze 답변(최대 ~6.9k자)이 대량 탈락하던 문제 해결.
  3) 표의 6개 데이터셋 로더를 모두 구현. build_dataset(sources=[...]) 로 선택.
"""

import html
import re
import hashlib
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset

# ──────────────────────────────────────────────────────────────────────────────
# 공통 시스템 프롬프트 — 모든 소스에 동일한 system 역할 부여
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a Unity coding assistant. Answer with correct, idiomatic "
    "Unity C# (or HLSL/ShaderLab when relevant) and cite the relevant Unity API."
)


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _clean(s) -> str:
    """None 안전 strip"""
    return (s or "").strip()


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    """StackExchange 계열 본문의 HTML 태그/엔티티 제거"""
    s = _TAG_RE.sub(" ", s or "")
    s = html.unescape(s)
    return _WS_RE.sub(" ", s).strip()


def _pair_to_messages(user, assistant, system: str = SYSTEM_PROMPT) -> list:
    """단일 turn (system, user, assistant) messages 생성"""
    return [
        {"role": "system",    "content": _clean(system)},
        {"role": "user",      "content": _clean(user)},
        {"role": "assistant", "content": _clean(assistant)},
    ]


def _ensure_system(msgs: list) -> list:
    """system 메시지가 없으면 맨 앞에 통일 프롬프트 삽입"""
    msgs = [{"role": m["role"], "content": _clean(m["content"])} for m in msgs]
    if not msgs or msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return msgs


def _auto_messages(ex: dict) -> Optional[list]:
    """
    스키마를 자동 판별하여 messages 로 변환.
    instruction/output, prompt/completion, question/answer, messages 형식을 지원.
    (vishnuOI·gamedev 처럼 정확한 컬럼명이 확정되지 않은 소스를 방어적으로 처리)
    반환: messages 리스트 또는 None(판별 실패)
    """
    # 1) 이미 대화형
    if isinstance(ex.get("messages"), list) and ex["messages"]:
        return _ensure_system(ex["messages"])
    if isinstance(ex.get("conversations"), list) and ex["conversations"]:
        # {from, value} 스키마(ShareGPT) 호환
        role_map = {"human": "user", "user": "user", "gpt": "assistant",
                    "assistant": "assistant", "system": "system"}
        msgs = [{"role": role_map.get(m.get("from", ""), "user"),
                 "content": _clean(m.get("value", ""))} for m in ex["conversations"]]
        return _ensure_system(msgs)

    # 2) 단일 turn 쌍 — 후보 컬럼명들
    user_keys = ("instruction", "prompt", "question", "input", "query")
    asst_keys = ("output", "completion", "answer", "response")

    user = next((_clean(ex[k]) for k in user_keys if _clean(ex.get(k))), "")
    asst = next((_clean(ex[k]) for k in asst_keys if _clean(ex.get(k))), "")

    # instruction + input 동시 존재 시 합쳐서 user 구성
    if _clean(ex.get("instruction")) and _clean(ex.get("input")):
        user = f"{_clean(ex['instruction'])}\n\n{_clean(ex['input'])}"

    if user and asst:
        return _pair_to_messages(user, asst)
    return None


def _normalized_fingerprint(messages: list) -> str:
    """
    근사중복까지 잡기 위한 정규화 해시:
      - system 메시지는 모든 소스가 동일하므로 제외
      - 소문자화 + 공백 정규화 후 해시
    (동일 Q&A가 두 소스에 모두 있으면 하나만 유지)
    """
    parts = []
    for m in messages:
        if m["role"] == "system":
            continue
        norm = _WS_RE.sub(" ", m["content"].lower()).strip()
        parts.append(f"{m['role']}:{norm}")
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# 소스별 로더 — 각 함수는 {"messages": [...], "source": "<name>"} 스키마를 반환
# ──────────────────────────────────────────────────────────────────────────────

def load_unity_code() -> Dataset:
    """ibranze/codellama_unity3d_v2  (question→user, answer→assistant)"""
    ds = load_dataset("ibranze/codellama_unity3d_v2")["train"]

    def _m(ex):
        return {"messages": _pair_to_messages(ex.get("question", ""),
                                              ex.get("answer", "")),
                "source": "ibranze"}

    return ds.map(_m, remove_columns=ds.column_names)


def load_unity_qa() -> Dataset:
    """Erocal/Unity_ResolvedQuestions (messages 설정) — messages 그대로, system 정규화"""
    ds = load_dataset("Erocal/Unity_ResolvedQuestions", "messages")["train"]

    def _m(ex):
        return {"messages": _ensure_system(ex["messages"]), "source": "erocal"}

    return ds.map(_m, remove_columns=ds.column_names)


def load_unity_api(inject_api_context: bool = False) -> Dataset:
    """
    Hypersniper/unity_api_2022_3  (question→user, answer→assistant)
      - topic/api 는 단독 학습 제외(표 규칙). inject_api_context=True 면
        api 설명을 system 컨텍스트로만 주입(학습 turn 본문에는 미포함).
      - 합성 데이터라 중복이 많음 → 병합 후 전역 dedup 으로 정리.
    """
    ds = load_dataset("Hypersniper/unity_api_2022_3")["train"]

    def _m(ex):
        system = SYSTEM_PROMPT
        if inject_api_context and _clean(ex.get("api", "")):
            system = f"{SYSTEM_PROMPT}\n\n# Reference\n{_clean(ex['api'])}"
        return {"messages": _pair_to_messages(ex.get("question", ""),
                                              ex.get("answer", ""), system),
                "source": "hypersniper"}

    return ds.map(_m, remove_columns=ds.column_names)


def load_unity_instructions() -> Dataset:
    """
    vishnuOI/unity-dev-instructions  (instruction(+input)→user, output→assistant)
      - 여러 소스를 통합한 메타셋이며 ibranze 를 내포함 → ibranze 와 동시 사용 시
        build_dataset 가 경고. id/메타 컬럼은 _auto_messages 가 무시.
      - train 스플릿만 사용(test 는 평가 누수 방지를 위해 제외).
      ※ 정확한 컬럼명이 확정되지 않아 _auto_messages 로 방어적 판별.
    """
    ds = load_dataset("vishnuOI/unity-dev-instructions")
    ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    cols = ds.column_names

    def _m(ex):
        return {"messages": _auto_messages(ex) or [], "source": "vishnuoi"}

    return ds.map(_m, remove_columns=cols)


def load_gamedev() -> Dataset:
    """
    mlfoundations-dev/stackexchange_gamedev  (질문→user, 채택/고득표 답변→assistant)
      - HTML 제거. 비어있거나 비정상 행은 유효성 단계에서 탈락.
      ※ gamedev.SE 덤프 가공본으로 정확한 컬럼명 미확정 → _auto_messages 방어 판별 후
        본문 HTML strip. (Unity 외 엔진 혼재 가능 → 필요 시 키워드 후처리 권장)
    """
    ds = load_dataset("mlfoundations-dev/stackexchange_gamedev")
    ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    cols = ds.column_names

    def _m(ex):
        msgs = _auto_messages(ex) or []
        # 본문 HTML 정리
        for m in msgs:
            if m["role"] in ("user", "assistant"):
                m["content"] = _strip_html(m["content"])
        return {"messages": msgs, "source": "gamedev"}

    return ds.map(_m, remove_columns=cols)


def load_common_pile(keyword: str = "unity", max_docs: int = 5000) -> Dataset:
    """
    common-pile/stackexchange  (text 필터 후 슬라이스만)
      ※ 실험적/선택적.  10M~100M 문서 규모라 streaming 으로 키워드 필터링하여
        소량만 추출.  원본은 질문+답변+댓글이 하나의 text 로 합쳐진 '문서' 형태라
        Q/A 쌍이 아님 → 여기서는 first paragraph 를 user, 나머지를 assistant 로
        근사 매핑하는 임시 처리.  품질이 낮으므로 기본 비활성.
        제대로 쓰려면 metadata 의 사이트(gamedev.SE/SO unity 태그) 필드로 필터해야 함.
    """
    stream = load_dataset("common-pile/stackexchange", split="train",
                          streaming=True)
    rows = []
    for ex in stream:
        text = ex.get("text", "") or ""
        if keyword.lower() not in text.lower():
            continue
        parts = [p for p in text.split("\n\n") if _clean(p)]
        if len(parts) < 2:
            continue
        user = parts[0]
        asst = "\n\n".join(parts[1:])
        rows.append({"messages": _pair_to_messages(user, asst), "source": "common_pile"})
        if len(rows) >= max_docs:
            break
    print(f"[DATA] common_pile: keyword='{keyword}' → {len(rows):,}행 추출(실험적)")
    return Dataset.from_list(rows) if rows else Dataset.from_dict(
        {"messages": [], "source": []})


# 소스명 → 로더 매핑 (build_dataset 의 sources 인자에서 참조)
SOURCE_LOADERS = {
    "ibranze":     load_unity_code,
    "erocal":      load_unity_qa,
    "hypersniper": load_unity_api,
    "vishnuoi":    load_unity_instructions,
    "gamedev":     load_gamedev,
    "common_pile": load_common_pile,
}


# ──────────────────────────────────────────────────────────────────────────────
# 필터 함수들
# ──────────────────────────────────────────────────────────────────────────────

def _is_valid(ex: dict) -> bool:
    """
    유효 샘플 조건:
      - messages 비어있지 않음
      - 마지막 메시지 role == "assistant"
      - user 메시지 1개 이상 존재
      - 마지막 user/assistant 내용이 모두 비어있지 않음
    """
    msgs = ex["messages"]
    if not msgs:
        return False
    if msgs[-1]["role"] != "assistant":
        return False
    roles = {m["role"] for m in msgs}
    if "user" not in roles:
        return False
    last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    last_asst = msgs[-1]["content"]
    return bool(_clean(last_user)) and bool(_clean(last_asst))


def _deduplicate(ds: Dataset) -> Dataset:
    """정규화 해시 기반 정확+근사 중복 제거 (system 제외, 소문자/공백 정규화)"""
    seen: set = set()
    keep_indices: list = []
    for i, ex in enumerate(ds):
        key = _normalized_fingerprint(ex["messages"])
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)
    removed = len(ds) - len(keep_indices)
    print(f"[DEDUP] {len(ds):,}행 → {len(keep_indices):,}행  (제거: {removed:,})")
    return ds.select(keep_indices)


def _filter_by_length(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    """
    chat template 적용 후 실제 토큰 수 > max_length 인 샘플 제거.
    (자르지 않고 통째로 제외 → 잘린 코드 학습 방지.  ibranze 의 긴 답변을 살리려면
     max_length 를 학습 seq_len 과 동일하게(예: 2048) 두는 것이 중요)
    """
    def _within_limit(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)
        n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
        return n_tokens <= max_length

    before = len(ds)
    ds = ds.filter(_within_limit)
    removed = before - len(ds)
    print(f"[LEN  ] max_length={max_length}: {before:,}행 → {len(ds):,}행  (제거: {removed:,})")
    return ds


def _cap_source(ds: Dataset, n: int, seed: int) -> Dataset:
    """소스별 상한 샘플링 (혼합 비율 근사용). n 이 행 수보다 크면 그대로 반환."""
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


# ──────────────────────────────────────────────────────────────────────────────
# 메인 빌드 함수
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    tokenizer,
    sources: Optional[list] = None,
    caps: Optional[dict] = None,
    max_length:  int   = 2048,
    val_ratio:   float = 0.05,
    seed:        int   = 42,
    max_samples: Optional[int] = None,
    output_format: str = "messages",   # "messages"(완료-구간 손실용) | "text"(전체 손실)
    inject_api_context: bool = False,
):
    """
    전체 전처리 파이프라인.

    단계:
      1) sources 의 각 소스 로드 → ChatML(messages) 정규화
      2) (caps 가 있으면) 소스별 상한 샘플링
      3) 병합
      4) 유효성 검사
      5) 중복 제거(정확+근사)
      6) 길이 필터(max_length 초과 제외)
      7) output_format 에 맞춰 출력 컬럼 구성
      8) 95:5 학습/검증 분리

    Args:
        tokenizer:    HF 토크나이저 (chat template + 토큰 수 계산용)
        sources:      사용할 소스 리스트. None → ["erocal", "ibranze"] (기존 MVP 동작 유지)
                      가능값: ibranze / erocal / hypersniper / vishnuoi / gamedev / common_pile
        caps:         {소스명: 최대행수} — 혼합 비율 근사용. 예: {"hypersniper": 4000}
        max_length:   최대 토큰 수(초과 샘플 제거). 학습 seq_len 과 동일하게 둘 것.
        val_ratio:    검증셋 비율(0.05 = 5%)
        seed:         분리/샘플링 시드
        max_samples:  디버그용 전체 상한
        output_format:
            "messages" → 컬럼 ["messages"].  SFTConfig(assistant_only_loss=True) 와 함께 사용
                         (system/user 토큰 손실 제외, assistant 응답만 학습).
            "text"     → 컬럼 ["text"].  SFTConfig(dataset_text_field="text") 로 전체 손실 학습.
        inject_api_context: Hypersniper 의 api 설명을 system 컨텍스트로만 주입할지 여부.

    Returns:
        (train_dataset, eval_dataset)
    """
    if sources is None:
        sources = ["erocal", "ibranze"]
    caps = caps or {}

    unknown = [s for s in sources if s not in SOURCE_LOADERS]
    if unknown:
        raise ValueError(f"알 수 없는 source: {unknown}. 가능값: {list(SOURCE_LOADERS)}")

    # ibranze ↔ vishnuOI 중복 경고 (vishnuOI 가 ibranze 를 내포)
    if "ibranze" in sources and "vishnuoi" in sources:
        print("[WARN ] ibranze 와 vishnuoi 를 동시 사용 중입니다. vishnuOI 는 ibranze 를 "
              "내포하므로 이중 카운트 위험이 있습니다(dedup 으로 일부만 제거됨). "
              "둘 중 하나만 사용하는 것을 권장합니다.")

    # ── 로드 + 소스별 상한 ────────────────────────────────────────────────────
    loaded = []
    for name in sources:
        print(f"[DATA] {name} 로드 중...")
        loader = SOURCE_LOADERS[name]
        ds = loader(inject_api_context=inject_api_context) if name == "hypersniper" else loader()
        ds = _cap_source(ds, caps.get(name), seed)
        print(f"[DATA] {name}: {len(ds):,}행")
        loaded.append(ds)

    # ── 병합 ────────────────────────────────────────────────────────────────
    combined = concatenate_datasets(loaded)
    print(f"[DATA] 병합 후: {len(combined):,}행")

    # ── 유효성 검사 ─────────────────────────────────────────────────────────
    before_valid = len(combined)
    combined = combined.filter(_is_valid)
    print(f"[VALID] 유효성 필터: {before_valid:,} → {len(combined):,}행")

    # ── 중복 제거 ───────────────────────────────────────────────────────────
    combined = _deduplicate(combined)

    # ── 길이 필터 ───────────────────────────────────────────────────────────
    combined = _filter_by_length(combined, tokenizer, max_length)

    # ── 샘플 수 제한(디버그) ─────────────────────────────────────────────────
    if max_samples is not None:
        n = min(max_samples, len(combined))
        combined = combined.shuffle(seed=seed).select(range(n))
        print(f"[DATA] max_samples={max_samples} 적용 → {len(combined):,}행")

    # ── 출력 컬럼 구성 ──────────────────────────────────────────────────────
    if output_format == "messages":
        # source 컬럼만 제거, messages 유지 (completion-only 학습용)
        combined = combined.remove_columns(
            [c for c in combined.column_names if c != "messages"])
    elif output_format == "text":
        def _to_text(ex):
            return {"text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)}
        combined = combined.map(
            _to_text, remove_columns=combined.column_names)
    else:
        raise ValueError("output_format 은 'messages' 또는 'text' 여야 합니다.")

    # ── 95:5 분리 ───────────────────────────────────────────────────────────
    split = combined.train_test_split(test_size=val_ratio, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[SPLIT] 학습: {len(train_ds):,}행 / 검증: {len(eval_ds):,}행  "
          f"(비율 {1 - val_ratio:.0%}:{val_ratio:.0%}, format={output_format})")

    return train_ds, eval_ds


# ──────────────────────────────────────────────────────────────────────────────
# 사용 예시 (실행하려면 토크나이저 필요)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # from transformers import AutoTokenizer
    # tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    #
    # # (A) 기존 MVP 동작 (Erocal + ibranze, completion-only)
    # train_ds, eval_ds = build_dataset(tok, max_length=2048)
    #
    # # (B) 권장 혼합 — 비율은 caps 로 근사 (ibranze 40 / erocal 30 / hyper 20 / gamedev 10)
    # train_ds, eval_ds = build_dataset(
    #     tok,
    #     sources=["ibranze", "erocal", "hypersniper", "gamedev"],
    #     caps={"hypersniper": 2000, "gamedev": 1000},  # ibranze·erocal 은 전량
    #     max_length=2048,
    #     output_format="messages",   # SFTConfig(assistant_only_loss=True) 와 함께
    # )
    pass
