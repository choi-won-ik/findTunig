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

[v2 추가 — 데이터셋 분석 반영]
  4) 비율(%) 혼합:  build_dataset(mix={"ibranze":40,"erocal":30,...}) 지원.
     정제(유효성·dedup·길이) 후 'source' 기준으로 비율 다운샘플(업샘플=복제 없음).
     레시피 상수 제공:  MIX_RECOMMENDED / MIX_VISHNUOI_BACKBONE / MIX_SINGLE_IBRANZE / MIX_RECOMMENDED_PERF
  5) 2단계 SFT 커리큘럼:  build_curriculum() → {"phase1":(tr,va), "phase2":(tr,va)}.
     Phase1=API/코드 각인(ibranze+hypersniper), Phase2=추론/문제해결(erocal+gamedev+perf).
  6) 성능 최적화 보강셋:  6개 셋 공통 최대 약점.  load_local_jsonl(path) 로 직접 큐레이션 JSONL을
     로드 → extra_sources={"perf": ...} 로 주입하여 mix 슬롯(예: 8%)에 끼워넣음.
  7) 5대 능력 커버리지 리포트:  빌드 시 선택 소스의 api/bugfix/codegen/design/perf 커버를 출력(약점 식별).
  8) target_total:  "양보다 질"(정제 15k~40k) 운용을 위해 비율 유지하며 전체 규모 상한 지정.

[v3 — 버그 수정 & 성능]
  9) _deduplicate(source_priority=...):  ibranze↔vishnuOI 충돌 시 우선순위 소스를 유지.
     build_dataset 는 둘이 함께 있으면 자동으로 ibranze 우선(슬롯 비율 보존).
 10) [성능] _deduplicate 컬럼 일괄 fetch, _filter_by_length batched 토크나이즈.
"""

import html
import re
import random
import hashlib
from collections import defaultdict
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


def load_local_jsonl(path: str, source: str = "perf") -> Dataset:
    """
    로컬 JSONL 보강셋 로더 (주로 '성능 최적화' 직접 큐레이션용 — 6개 셋 공통 약점).
      예시 한 줄(아무 스키마나 자동 판별):
        {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
        {"instruction":"Draw call 배칭으로 ...","output":"..."}
        {"question":"GC Alloc 줄이려면?","answer":"..."}
      사용:
        perf = load_local_jsonl("perf_optimization.jsonl")   # Job System/Burst, Profiler 등
        build_dataset(..., extra_sources={"perf": perf}, mix=MIX_RECOMMENDED_PERF)
    """
    ds = load_dataset("json", data_files=path, split="train")
    cols = ds.column_names

    def _m(ex):
        return {"messages": _auto_messages(ex) or [], "source": source}

    return ds.map(_m, remove_columns=cols)


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
# 능력 커버리지 프로파일 & 혼합 레시피 (데이터셋 분석 표 기준)
#   - SOURCE_PROFILE: 각 소스의 순위/강점/약점 → _coverage_report 에서 약점 식별
#   - MIX_*: build_dataset(mix=...) 에 바로 넣는 비율(%) 레시피
# 5대 능력 코드: api(API 암기) / bugfix(버그 해결) / codegen(C# 코드 생성)
#               / design(게임 시스템 설계) / perf(성능 최적화)
# ──────────────────────────────────────────────────────────────────────────────
SOURCE_PROFILE = {
    "ibranze":     {"rank": 1, "strong": ["codegen", "api"],   "weak": ["bugfix", "design", "perf"]},
    "erocal":      {"rank": 2, "strong": ["bugfix", "design"], "weak": ["api"],  "note": "URP/HDRP/XR 실전 해결"},
    "vishnuoi":    {"rank": 3, "strong": ["codegen", "api"],    "weak": ["perf"], "note": "ibranze 내포(이중카운트 주의)"},
    "hypersniper": {"rank": 4, "strong": ["api"],               "weak": ["bugfix", "design", "perf"], "note": "합성·중복 → 샘플링 필요"},
    "gamedev":     {"rank": 5, "strong": ["design", "bugfix"],  "weak": ["perf"], "note": "고득표 필터 권장"},
    "common_pile": {"rank": 6, "strong": [],                    "weak": ["perf"], "note": "범용 코퍼스 → 무거운 필터 전 제외"},
    # 직접 큐레이션 보강셋(로컬). load_local_jsonl → extra_sources 로 주입.
    "perf":        {"rank": 0, "strong": ["perf"],              "weak": [],       "note": "직접 큐레이션(배칭/GC Alloc/Job·Burst/Profiler)"},
}

# (1) 권장 혼합 — vishnuOI 제외 직접 조합 (중복 회피)
MIX_RECOMMENDED = {"ibranze": 40, "erocal": 30, "hypersniper": 20, "gamedev": 10}

# (1-perf) 권장 혼합 + 성능 최적화 보강 슬롯(8%) — perf 는 extra_sources 로 공급
MIX_RECOMMENDED_PERF = {"ibranze": 37, "erocal": 28, "hypersniper": 17, "gamedev": 10, "perf": 8}

# (2) vishnuOI 백본 — ibranze 단독 투입 금지(이미 내포)
MIX_VISHNUOI_BACKBONE = {"vishnuoi": 55, "erocal": 25, "hypersniper": 10, "gamedev": 10}

# (3) 단일 데이터셋 추천
MIX_SINGLE_IBRANZE = {"ibranze": 100}

# 2단계 SFT 커리큘럼 기본 구성 (build_curriculum 기본값)
#   Phase 1 — API/코드 각인 (1~2 epoch):  ibranze + hypersniper(sample)
#   Phase 2 — 추론/문제해결 (1 epoch):    erocal + gamedev (+ perf 보강)
CURRICULUM_PHASE1 = {"ibranze": 70, "hypersniper": 30}
CURRICULUM_PHASE2 = {"erocal": 55, "gamedev": 25, "perf": 20}


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


def _deduplicate(ds: Dataset, source_priority: Optional[list] = None) -> Dataset:
    """
    정규화 해시 기반 정확+근사 중복 제거 (system 제외, 소문자/공백 정규화).

    source_priority:
        None  → 먼저 등장한 행을 유지(기존 동작).
        list  → 충돌 시 우선순위가 높은(리스트 앞쪽) source 의 행을 유지.
                예: ["ibranze", "vishnuoi"] → 같은 Q&A 가 두 소스에 있으면 ibranze 판을 남김.
                (vishnuoi 가 ibranze 를 내포하는 경우, ibranze 슬롯이 비율을 채울 수 있게 함)

    [성능] ds[i] 행 단위 접근(역직렬화 비용 큼) 대신 컬럼 일괄 fetch 후 파이썬 루프.
    """
    msgs_col = ds["messages"]
    n = len(msgs_col)

    if source_priority:
        rank = {s: i for i, s in enumerate(source_priority)}
        src_col = ds["source"] if "source" in ds.column_names else ["?"] * n
        default_rank = len(rank)
        best: dict = {}  # fp -> (rank, index)
        for i in range(n):
            key = _normalized_fingerprint(msgs_col[i])
            r = rank.get(src_col[i], default_rank)
            prev = best.get(key)
            if prev is None or r < prev[0]:
                best[key] = (r, i)
        keep_indices = sorted(v[1] for v in best.values())
    else:
        seen: set = set()
        keep_indices = []
        for i in range(n):
            key = _normalized_fingerprint(msgs_col[i])
            if key not in seen:
                seen.add(key)
                keep_indices.append(i)

    removed = n - len(keep_indices)
    print(f"[DEDUP] {n:,}행 → {len(keep_indices):,}행  (제거: {removed:,}"
          f"{', priority=' + '>'.join(source_priority) if source_priority else ''})")
    return ds.select(keep_indices)


def _filter_by_length(ds: Dataset, tokenizer, max_length: int,
                      batch_size: int = 512) -> Dataset:
    """
    chat template 적용 후 실제 토큰 수 > max_length 인 샘플 제거.
    (자르지 않고 통째로 제외 → 잘린 코드 학습 방지.  ibranze 의 긴 답변을 살리려면
     max_length 를 학습 seq_len 과 동일하게(예: 2048) 두는 것이 중요)

    [성능] 행 단위 tokenizer 호출 대신 batched filter 로 토크나이저를 일괄 호출.
    """
    def _batch_within(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        enc = tokenizer(texts, truncation=False)
        return [len(ids) <= max_length for ids in enc["input_ids"]]

    before = len(ds)
    ds = ds.filter(_batch_within, batched=True, batch_size=batch_size)
    removed = before - len(ds)
    print(f"[LEN  ] max_length={max_length}: {before:,}행 → {len(ds):,}행  (제거: {removed:,})")
    return ds


def _cap_source(ds: Dataset, n: int, seed: int) -> Dataset:
    """소스별 상한 샘플링 (혼합 비율 근사용). n 이 행 수보다 크면 그대로 반환."""
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _coverage_report(all_sources: list) -> None:
    """선택된 소스들의 5대 능력 커버리지를 출력(약점 식별용)."""
    abilities = ["api", "bugfix", "codegen", "design", "perf"]
    label = {"api": "API암기", "bugfix": "버그해결", "codegen": "코드생성",
             "design": "설계", "perf": "최적화"}
    cover = {a: [] for a in abilities}
    for s in all_sources:
        for a in SOURCE_PROFILE.get(s, {}).get("strong", []):
            cover[a].append(s)
    print("[COVER] 5대 능력 커버리지:")
    for a in abilities:
        srcs = ", ".join(cover[a]) if cover[a] else "⚠ 없음 → 직접 보강 필요"
        print(f"         - {label[a]:<7}: {srcs}")


def _apply_mix(ds: Dataset, mix: dict, target_total: Optional[int], seed: int) -> Dataset:
    """
    정제 끝난 데이터셋을 'source' 컬럼 기준으로 비율(%) 다운샘플링.
      - 업샘플(중복 복제) 없음:  가장 빡빡한 소스에 맞춰 전체 규모 결정
        (total = min_s  available_s / frac_s).
      - target_total 이 주어지면 그 값으로 추가 상한("양보다 질" 운용).
      - mix 에 있으나 데이터가 없는 소스는 경고 후 제외(남은 소스 상대비율 유지).
      - mix 에 없는(로드된) 소스는 0%로 간주하여 드롭.
    dedup 이후에 호출되므로 ibranze↔vishnuOI 정확중복은 이미 제거된 상태에서 비율이 맞춰짐.
    """
    by_src = defaultdict(list)
    for i, s in enumerate(ds["source"]):
        by_src[s].append(i)
    present = set(by_src)

    requested = {k: v for k, v in mix.items() if v and v > 0}
    missing = [k for k in requested if k not in present or not by_src[k]]
    for k in missing:
        print(f"[MIX  ] '{k}' 데이터 없음 → 비율 {requested[k]}% 무시")
    requested = {k: v for k, v in requested.items() if k in present and by_src[k]}
    if not requested:
        raise ValueError("[MIX] 유효한 소스가 없습니다. mix/sources/extra_sources 확인.")

    for k in sorted(present - set(requested)):
        print(f"[MIX  ] '{k}' 는 mix 에 없음 → {len(by_src[k]):,}행 드롭")

    pct_sum = sum(requested.values())
    if abs(pct_sum - 100) > 1e-6:
        print(f"[MIX  ] 비율 합 {pct_sum}%(≠100) → 상대비율로 정규화 처리")
    fracs = {k: v / pct_sum for k, v in requested.items()}

    feasible = min(len(by_src[k]) / fracs[k] for k in requested)
    total = feasible if target_total is None else min(feasible, float(target_total))

    keep, report = [], {}
    for off, k in enumerate(sorted(fracs)):
        n = min(int(round(total * fracs[k])), len(by_src[k]))
        idx = by_src[k][:]
        random.Random(seed + off).shuffle(idx)
        keep.extend(idx[:n])
        report[k] = n

    keep.sort()
    out = ds.select(keep)
    kept = len(keep)
    print("[MIX  ] 비율 적용 결과:")
    for k in sorted(report):
        share = report[k] / kept * 100 if kept else 0
        print(f"         - {k:<12} {report[k]:>7,}행  ({share:4.1f}%)")
    print(f"[MIX  ] 합계 {kept:,}행  (target_total={target_total})")
    return out.shuffle(seed=seed)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 빌드 함수
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    tokenizer,
    sources: Optional[list] = None,
    caps: Optional[dict] = None,
    mix: Optional[dict] = None,
    target_total: Optional[int] = None,
    extra_sources: Optional[dict] = None,
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
      1) sources 의 각 소스 로드 → ChatML(messages) 정규화  (+ extra_sources 병합)
      2) (caps 가 있으면) 소스별 상한 샘플링  (로드 직후, 절대값)
      3) 병합
      4) 유효성 검사
      5) 중복 제거(정확+근사)
      6) 길이 필터(max_length 초과 제외)
      7) (mix 가 있으면) 'source' 기준 비율(%) 다운샘플  ← 정제 후라 비율이 정확
      8) output_format 에 맞춰 출력 컬럼 구성
      9) 95:5 학습/검증 분리

    Args:
        tokenizer:    HF 토크나이저 (chat template + 토큰 수 계산용)
        sources:      사용할 소스 리스트. None → ["erocal", "ibranze"] (기존 MVP 동작 유지)
                      가능값: ibranze / erocal / hypersniper / vishnuoi / gamedev / common_pile
        caps:         {소스명: 최대행수} — 로드 직후 절대 상한. 예: {"hypersniper": 4000}
        mix:          {소스명: 비율%} — 정제 후 비율 다운샘플. 예: MIX_RECOMMENDED.
                      업샘플(복제) 없이 가장 빡빡한 소스 기준으로 규모 결정.
                      caps 와 함께 쓰면: caps 로 먼저 절대 상한 → mix 로 최종 비율.
        target_total: mix 사용 시 비율 유지하며 전체 규모 상한(예: 30000). "양보다 질" 운용.
        extra_sources:{소스명: Dataset} — 외부에서 만든 보강셋 주입. 각 Dataset 은 'messages'
                      컬럼 필수(없으면 'source'=키 로 자동 부여). 예: {"perf": load_local_jsonl(path)}.
                      mix 의 키로 사용 가능(예: MIX_RECOMMENDED_PERF 의 "perf" 8%).
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
    extra_sources = extra_sources or {}

    unknown = [s for s in sources if s not in SOURCE_LOADERS]
    if unknown:
        raise ValueError(f"알 수 없는 source: {unknown}. 가능값: {list(SOURCE_LOADERS)}")

    all_source_names = list(sources) + list(extra_sources)

    # 5대 능력 커버리지 리포트 (약점 식별 — 특히 'perf' 는 6개 셋 공통 약점)
    _coverage_report(all_source_names)

    # ibranze ↔ vishnuOI 중복 경고 (vishnuOI 가 ibranze 를 내포)
    both_full = ("ibranze" in sources and "vishnuoi" in sources)
    if mix:
        both_full = (mix.get("ibranze", 0) > 0 and mix.get("vishnuoi", 0) > 0)
    if both_full:
        print("[WARN ] ibranze 와 vishnuoi 를 동시 사용 중입니다. vishnuOI 는 ibranze 를 "
              "내포하므로 이중 카운트 위험이 있습니다(정확중복만 dedup 제거됨). "
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

    # ── 외부 보강셋(extra_sources) 정규화 후 병합 대상에 추가 ──────────────────
    for name, eds in extra_sources.items():
        if "messages" not in eds.column_names:
            raise ValueError(f"extra_sources['{name}'] 에 'messages' 컬럼이 없습니다.")
        eds = eds.remove_columns([c for c in eds.column_names if c != "messages"])
        eds = eds.add_column("source", [name] * len(eds))
        eds = _cap_source(eds, caps.get(name), seed)
        print(f"[DATA] (extra) {name}: {len(eds):,}행")
        loaded.append(eds)

    # ── 병합 ────────────────────────────────────────────────────────────────
    combined = concatenate_datasets(loaded)
    print(f"[DATA] 병합 후: {len(combined):,}행")

    # ── 유효성 검사 ─────────────────────────────────────────────────────────
    before_valid = len(combined)
    combined = combined.filter(_is_valid)
    print(f"[VALID] 유효성 필터: {before_valid:,} → {len(combined):,}행")

    # ── 중복 제거 ───────────────────────────────────────────────────────────
    #   ibranze ⊂ vishnuoi 인 경우, 충돌 시 ibranze 판을 남겨야 ibranze 슬롯이 비율을
    #   채울 수 있고 vishnuoi 는 '고유분'만 남아 이중카운트가 정리됨.
    present_sources = set(combined["source"])
    dedup_priority = (["ibranze", "vishnuoi"]
                      if {"ibranze", "vishnuoi"} <= present_sources else None)
    combined = _deduplicate(combined, source_priority=dedup_priority)

    # ── 길이 필터 ───────────────────────────────────────────────────────────
    combined = _filter_by_length(combined, tokenizer, max_length)

    # ── 비율(%) 혼합 — 정제 후 'source' 기준 다운샘플 ─────────────────────────
    if mix:
        combined = _apply_mix(combined, mix, target_total, seed)

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
# 2단계 SFT 커리큘럼 빌더
# ──────────────────────────────────────────────────────────────────────────────

def build_curriculum(
    tokenizer,
    *,
    phase1_mix: Optional[dict] = None,
    phase2_mix: Optional[dict] = None,
    extra_sources: Optional[dict] = None,
    max_length: int = 2048,
    target_total: Optional[int] = None,
    val_ratio: float = 0.05,
    seed: int = 42,
    inject_api_context: bool = True,
):
    """
    2단계 SFT 커리큘럼용 데이터 빌더.
      Phase 1 — API/코드 각인 (권장 1~2 epoch):  ibranze + hypersniper(sample)
                → Unity API·C# 패턴을 먼저 주입
      Phase 2 — 추론/문제해결 (권장 1 epoch):    erocal + gamedev (+ perf 보강)
                → 버그해결·설계·최적화 같은 '사고형' 응답 강화

    epoch 수는 학습 루프에서 지정합니다(여기서는 단계별 데이터만 구성).
    perf 보강셋을 쓰려면 extra_sources={"perf": load_local_jsonl(path)} 로 넘기세요
    (없으면 Phase2 에서 perf 슬롯은 자동 제외되고 나머지 비율이 유지됩니다).

    Returns:
        {"phase1": (train, eval), "phase2": (train, eval)}
    """
    phase1_mix = phase1_mix or CURRICULUM_PHASE1
    phase2_mix = phase2_mix or CURRICULUM_PHASE2
    extra_sources = extra_sources or {}

    def _run(tag, mix):
        print("=" * 70)
        print(f"[CURRICULUM] {tag}")
        print("=" * 70)
        return build_dataset(
            tokenizer,
            sources=[s for s in mix if s in SOURCE_LOADERS],
            extra_sources={k: v for k, v in extra_sources.items() if k in mix},
            mix=mix,
            target_total=target_total,
            max_length=max_length,
            val_ratio=val_ratio,
            seed=seed,
            output_format="messages",
            inject_api_context=inject_api_context,
        )

    return {
        "phase1": _run("Phase 1 — API/코드 각인", phase1_mix),
        "phase2": _run("Phase 2 — 추론/문제해결", phase2_mix),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 사용 예시 (실행하려면 토크나이저 필요)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # from transformers import AutoTokenizer
    # tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    #
    # # (A) 기존 MVP 동작 (Erocal + ibranze, completion-only)
    # train_ds, eval_ds = build_dataset(tok, max_length=2048)
    #
    # # (B) 권장 혼합 — 비율(%) 로 지정 (ibranze 40 / erocal 30 / hyper 20 / gamedev 10)
    # train_ds, eval_ds = build_dataset(
    #     tok,
    #     sources=["ibranze", "erocal", "hypersniper", "gamedev"],
    #     mix=MIX_RECOMMENDED,
    #     target_total=30000,          # "양보다 질": 정제 15k~40k 권장
    #     max_length=2048,
    #     inject_api_context=True,
    #     output_format="messages",    # SFTConfig(assistant_only_loss=True) 와 함께
    # )
    #
    # # (C) vishnuOI 백본 (ibranze 단독 투입 금지 — 이미 내포)
    # train_ds, eval_ds = build_dataset(
    #     tok, sources=["vishnuoi", "erocal", "hypersniper", "gamedev"],
    #     mix=MIX_VISHNUOI_BACKBONE, max_length=2048)
    #
    # # (D) 성능 최적화 보강셋(직접 큐레이션) 8% 끼워넣기
    # perf = load_local_jsonl("perf_optimization.jsonl")   # 배칭/GC/Job·Burst/Profiler Q&A
    # train_ds, eval_ds = build_dataset(
    #     tok, sources=["ibranze", "erocal", "hypersniper", "gamedev"],
    #     extra_sources={"perf": perf},
    #     mix=MIX_RECOMMENDED_PERF, max_length=2048)
    #
    # # (E) 2단계 커리큘럼 (Phase1 각인 → Phase2 추론)
    # bundle = build_curriculum(tok, extra_sources={"perf": perf}, target_total=25000)
    # p1_tr, p1_va = bundle["phase1"]   # 먼저 1~2 epoch 학습
    # p2_tr, p2_va = bundle["phase2"]   # 이어서 1 epoch 학습
    pass
