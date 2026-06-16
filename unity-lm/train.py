"""
train.py — Qwen2.5-Coder-1.5B-Instruct QLoRA 파인튜닝 (Unity 코딩 어시스턴트)
환경  : Ubuntu 22.04 / Python 3.10+ / CUDA 12.x
GPU   : RTX 5060 (Blackwell) 최적화 — max_seq_len 4096, batch 4, Flash Attention 2

출력 구조 (--base_dir 기준):
  output/
  ├── training_logs.csv
  ├── loss_curve.png
  ├── accuracy_curve.png
  ├── lora_adapter/
  └── merged_model/

OOM 자동 복구 순서:
  batch_size 감소 → sequence_length 감소 → gradient_accumulation 증가

실행 예:
  python train.py                                 # RTX 5060 최적값으로 학습
  python train.py --base_dir ./runs/exp01         # 출력 경로 변경
  python train.py --max_samples 200 --epochs 1   # 빠른 테스트
  python train.py --skip_merge                    # 학습만 (merge 생략)
  python train.py --skip_train                    # merge만 (기존 어댑터 사용)
  python train.py --no_oom_fallback               # OOM 자동 복구 비활성화
"""

import argparse
import gc
import glob
import os
import traceback

# pyarrow(datasets) 를 torch보다 먼저 import해야 Windows DLL 충돌 방지
from dataset import build_dataset
from monitor import MonitoredSFTTrainer, TrainingMonitorCallback, DriveBackupCallback, plot_curves

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig


# ──────────────────────────────────────────────────────────────────────────────
# GPU 환경 감지
# ──────────────────────────────────────────────────────────────────────────────

def detect_gpu():
    """
    GPU 정보 및 최적 설정 감지

    Returns dict:
      name        — GPU 이름 (예: "NVIDIA GeForce RTX 5060")
      vram_gb     — VRAM 크기 (GB)
      bf16        — BF16 지원 여부 (Ampere 이상 True)
      is_rtx50xx  — RTX 50xx Blackwell 시리즈 여부
      cc_major    — CUDA Compute Capability major 버전
    """
    if not torch.cuda.is_available():
        return {"name": "CPU", "vram_gb": 0, "bf16": False,
                "is_rtx50xx": False, "cc_major": 0}

    props    = torch.cuda.get_device_properties(0)
    name     = torch.cuda.get_device_name(0)
    vram_gb  = props.total_memory / 1e9
    bf16     = torch.cuda.is_bf16_supported()
    cc_major = props.major

    # RTX 50xx (Blackwell GB20x) 시리즈: GPU 이름에 "50" 포함
    is_rtx50xx = any(tag in name for tag in ["5060", "5070", "5080", "5090", "50 "])

    return {
        "name": name, "vram_gb": vram_gb, "bf16": bf16,
        "is_rtx50xx": is_rtx50xx, "cc_major": cc_major,
    }


def detect_flash_attention(cc_major: int) -> str:
    """
    Flash Attention 2 지원 여부 자동 감지

    우선순위:
      flash_attention_2 — flash-attn 패키지 + sm_80 이상 (Ampere/Hopper/Blackwell)
      sdpa              — PyTorch 2.0+ 내장 SDPA (Flash Attention 미설치 시 fallback)
      eager             — GPU 없을 때

    RTX 5060 (Blackwell cc≥8.0) 기준 flash_attention_2 정상 작동

    Returns:
        "flash_attention_2" | "sdpa" | "eager"
    """
    if not torch.cuda.is_available():
        return "eager"

    if cc_major < 8:
        # Volta/Turing/Pascal 등 구형 GPU — Flash Attention 2 미지원
        print(f"[ATTN] GPU cc={cc_major}.x < 8.0 → SDPA 사용")
        return "sdpa"

    try:
        import flash_attn  # noqa: F401
        print(f"[ATTN] Flash Attention 2 활성화 (cc={cc_major}.x, flash-attn 감지)")
        return "flash_attention_2"
    except ImportError:
        print("[ATTN] flash-attn 패키지 없음 → SDPA 사용")
        print("       Flash Attention 2 설치: pip install flash-attn --no-build-isolation")
        return "sdpa"


# ──────────────────────────────────────────────────────────────────────────────
# OOM 자동 복구 스케줄
# ──────────────────────────────────────────────────────────────────────────────

def get_fallback_schedule(
    initial_bs: int,
    initial_sl: int,
    initial_ga: int,
) -> list[tuple[int, int, int]]:
    """
    CUDA OOM 발생 시 시도할 (batch_size, max_length, grad_accum) 순서 생성

    조정 우선순위:
      ① batch_size 절반 (VRAM 직결 효과)
      ② sequence_length 절반 (메모리 제곱 감소)
      ③ gradient_accumulation 증가 (유효 배치 유지)

    RTX 5060 8GB 기준 초기값 (4, 4096, 8) → 최소 (1, 1024, 32) 까지 자동 시도
    """
    candidates = [
        (initial_bs,              initial_sl,              initial_ga),        # 기본
        (max(1, initial_bs // 2), initial_sl,              initial_ga),        # ① batch 절반
        (max(1, initial_bs // 2), max(512, initial_sl // 2), initial_ga),      # ② seq 절반
        (1,                       max(512, initial_sl // 2), initial_ga * 2),  # ③ grad 2배
        (1,                       max(512, initial_sl // 4), initial_ga * 4),  # 최소
    ]
    # 중복 제거 (순서 유지)
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen2.5-Coder-1.5B QLoRA 파인튜닝 — RTX 5060 최적화"
    )

    # 경로
    p.add_argument("--model_id",   default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--base_dir",   default="./output",
                   help="모든 출력의 루트 디렉토리")
    p.add_argument("--output_dir", default=None,
                   help="LoRA 어댑터 경로 (기본: base_dir/lora_adapter)")
    p.add_argument("--merged_dir", default=None,
                   help="Merged 모델 경로 (기본: base_dir/merged_model)")

    # 데이터
    p.add_argument("--val_ratio",   type=float, default=0.05)
    # RTX 5060 권장값 4096 (이전 GTX 기준 1024에서 상향)
    p.add_argument("--max_length",  type=int,   default=4096,
                   help="최대 시퀀스 길이 (RTX 5060 권장: 4096)")
    p.add_argument("--max_samples", type=int,   default=None)

    # 학습 하이퍼파라미터 — RTX 5060 8GB GDDR7 기준 권장값
    p.add_argument("--epochs",       type=float, default=3.0)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="디바이스당 배치 크기 (RTX 5060 권장: 4)")
    p.add_argument("--grad_accum",   type=int,   default=8,
                   help="Gradient accumulation (RTX 5060 권장: 8, 유효 배치=32)")
    p.add_argument("--save_steps",   type=int,   default=50)
    p.add_argument("--logging_steps", type=int,  default=1)
    p.add_argument("--seed",         type=int,   default=42)

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # 제어 플래그
    # Optimizer
    # paged_adamw_8bit: bitsandbytes 8-bit paged optimizer — T4/16GB 권장
    #   optimizer 상태(momentum+variance)를 8-bit NF로 저장 → VRAM ~450MB 절약
    # adamw_torch_fused: PyTorch 내장 fused kernel — H100/A100 고성능 환경 권장
    p.add_argument("--optim", default="paged_adamw_8bit",
                   help="optimizer 종류 (기본: paged_adamw_8bit / 고사양: adamw_torch_fused)")

    p.add_argument("--drive_backup_dir", default=None,
                   help="Google Drive 체크포인트 실시간 백업 경로 (None=비활성). "
                        "예: /content/drive/MyDrive/.../output_colab")

    p.add_argument("--skip_train",      action="store_true")
    p.add_argument("--skip_merge",      action="store_true")
    p.add_argument("--no_oom_fallback", action="store_true",
                   help="OOM 자동 복구 비활성화 (즉시 중단)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 모델 빌드 — OOM 재시도를 위해 파라미터로 분리
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_id: str, compute_dtype, attn_impl: str, has_cuda: bool):
    """
    QLoRA 4bit NF4 모델 로드

    별도 함수로 분리한 이유: OOM 발생 시 기존 모델을 삭제하고
    더 작은 설정으로 이 함수를 재호출하기 위함

    Args:
        model_id:       HuggingFace 모델 ID
        compute_dtype:  torch.bfloat16 | torch.float16 | torch.float32
        attn_impl:      "flash_attention_2" | "sdpa" | "eager"
        has_cuda:       CUDA 사용 가능 여부
    """
    if not has_cuda:
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다.\n"
            "  - GPU 드라이버 확인: nvidia-smi\n"
            "  - PyTorch CUDA 확인: python -c \"import torch; print(torch.version.cuda)\"\n"
            "  - RTX 50xx(Blackwell)의 경우 PyTorch 2.7+ 필요"
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},  # 모든 레이어를 GPU 0에 강제 배치
        torch_dtype=compute_dtype,  # dtype= 은 잘못된 kwarg — torch_dtype= 이 정확한 파라미터
        attn_implementation=attn_impl,
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # KV 캐시 비활성화 (gradient checkpointing과 충돌 방지)
    model.config.use_cache = False
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Merge
# ──────────────────────────────────────────────────────────────────────────────

def merge_and_save(model_id: str, adapter_dir: str, merged_dir: str):
    """
    Base Model + LoRA Adapter → HuggingFace fp16 모델로 저장
    llama.cpp convert_hf_to_gguf.py 호환성을 위해 fp16으로 저장
    """
    print(f"\n[MERGE] ─────────────────────────────────────────────────────")
    print(f"[MERGE] Base model 로드: {model_id}")

    # 4bit 가중치는 merge 불가 → fp16 full precision으로 다시 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
    )
    try:
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    except ImportError as e:
        if "torchao" in str(e).lower():
            raise ImportError(
                f"{e}\n\n"
                "[MERGE] torchao 버전 불일치 — 해결 방법:\n"
                "  1) !pip install 'torchao>=0.16.0' 실행 후 런타임 재시작\n"
                "  2) 또는 train.py에 --skip_merge 추가하고 merge_lora.py로 별도 머지"
            ) from e
        raise
    print("[MERGE] merge_and_unload() 실행 중...")
    merged = peft_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[MERGE] 완료: {merged_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # VRAM 단편화 방지 — OOM 복구 시 메모리 재사용률 향상
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()

    # 경로 자동 구성
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "lora_adapter")
    if args.merged_dir is None:
        args.merged_dir = os.path.join(args.base_dir, "merged_model")

    os.makedirs(args.base_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── GPU 환경 감지 ──────────────────────────────────────────────────────
    gpu    = detect_gpu()
    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        raise SystemExit(
            "[ERROR] CUDA GPU를 찾을 수 없습니다. GPU 드라이버/CUDA Toolkit을 확인하세요.\n"
            "  nvidia-smi\n"
            "  python -c \"import torch; print(torch.version.cuda)\"\n"
            "  RTX 50xx(Blackwell)의 경우 PyTorch >= 2.7 필요"
        )

    print("=" * 60)
    print("[INIT] RTX 5060 최적화 QLoRA 파인튜닝")
    print("=" * 60)
    print(f"  GPU       : {gpu['name']} ({gpu['vram_gb']:.1f}GB)")
    print(f"  BF16      : {'지원' if gpu['bf16'] else '미지원 (fp16 사용)'}")
    print(f"  RTX 50xx  : {'감지됨 ✓' if gpu['is_rtx50xx'] else '아님'}")
    print(f"  CC        : {gpu['cc_major']}.x")

    print(f"  base_dir  : {args.base_dir}")

    # ── Attention 구현 선택 ────────────────────────────────────────────────
    attn_impl    = detect_flash_attention(gpu["cc_major"]) if has_cuda else "eager"
    use_bf16     = gpu["bf16"] if has_cuda else False
    compute_dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if has_cuda else torch.float32
    )

    # ── skip_train: 기존 어댑터로 merge만 ──────────────────────────────────
    if args.skip_train:
        print("[INFO] --skip_train: merge 단계로 이동")
        if not args.skip_merge:
            merge_and_save(args.model_id, args.output_dir, args.merged_dir)
        return

    # ── 1. 토크나이저 로드 ────────────────────────────────────────────────
    print(f"\n[INIT] 토크나이저: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. 데이터셋 빌드 ──────────────────────────────────────────────────
    print("\n[DATA] 데이터셋 빌드...")
    train_ds, eval_ds = build_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        output_format="text",
    )

    # ── LoRA 설정 (OOM 루프 외부에서 한 번만 생성) ─────────────────────────
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ── 모니터 콜백 (OOM 루프 외부에서 생성, 재시도 시 유지) ──────────────
    monitor_cb = TrainingMonitorCallback(
        output_dir=args.base_dir,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        eval_batch_size=1,
        max_eval_batches=20,
        max_length=args.max_length,
    )

    # ── Drive 백업 콜백 (drive_backup_dir 지정 시에만 활성) ───────────────
    drive_cb = None
    if args.drive_backup_dir:
        drive_cb = DriveBackupCallback(
            adapter_dir=args.output_dir,
            drive_backup_dir=args.drive_backup_dir,
        )
        print(f"[BACKUP] Drive 실시간 백업 활성화: {args.drive_backup_dir}")

    callbacks = [monitor_cb] + ([drive_cb] if drive_cb else [])

    # ── OOM 자동 복구 루프 ────────────────────────────────────────────────
    fallback_schedule = get_fallback_schedule(
        args.batch_size, args.max_length, args.grad_accum
    )

    if args.no_oom_fallback:
        # OOM 복구 비활성화 → 기본값 단 한 번만 시도
        fallback_schedule = fallback_schedule[:1]

    trainer    = None
    final_bs   = args.batch_size
    final_sl   = args.max_length
    final_ga   = args.grad_accum

    for attempt, (bs, sl, ga) in enumerate(fallback_schedule):
        if attempt > 0:
            print(f"\n[OOM] 복구 시도 {attempt}/{len(fallback_schedule) - 1}: "
                  f"batch={bs}  seq_len={sl}  grad_accum={ga}")

        try:
            # ── 3. 모델 로드 ──────────────────────────────────────────────
            if attempt == 0:
                print(f"\n[MODEL] 로드: {args.model_id}")
                print(f"  batch={bs}  seq_len={sl}  grad_accum={ga}  "
                      f"attn={attn_impl}  dtype={'bf16' if use_bf16 else 'fp16'}")
            model = build_model(args.model_id, compute_dtype, attn_impl, has_cuda)

            # ── 4. SFTConfig ───────────────────────────────────────────────
            sft_config = SFTConfig(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=max(1, bs // 2),
                gradient_accumulation_steps=ga,
                # gradient checkpointing: 중간 활성화 재계산 → VRAM ↓
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                learning_rate=args.lr,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                weight_decay=0.01,
                max_grad_norm=0.3,
                optim=args.optim,
                bf16=use_bf16,
                fp16=has_cuda and not use_bf16,
                logging_steps=args.logging_steps,
                save_strategy="steps",
                save_steps=args.save_steps,
                save_total_limit=3,
                eval_strategy="epoch",
                load_best_model_at_end=False,
                dataset_text_field="text",
                max_length=sl,
                packing=(attn_impl == "flash_attention_2"),
                seed=args.seed,
                report_to="none",
            )

            # ── 5. 트레이너 생성 ───────────────────────────────────────────
            trainer = MonitoredSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=peft_config,
                processing_class=tokenizer,
                callbacks=callbacks,
            )
            monitor_cb.trainer = trainer

            # ── 6. 체크포인트 이어받기 ────────────────────────────────────
            checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
            resume = max(checkpoints, key=os.path.getmtime) if checkpoints else None
            if resume:
                print(f"[INFO] 이어받기: {resume}")

            # ── 7. 학습 실행 ──────────────────────────────────────────────
            print("\n[TRAIN] 학습 시작...")
            trainer.train(resume_from_checkpoint=resume)

            final_bs, final_sl, final_ga = bs, sl, ga
            break  # 성공 → 루프 탈출

        except torch.cuda.OutOfMemoryError as oom:
            # OOM 발생 → 메모리 해제 후 다음 설정 시도
            print(f"\n[OOM] CUDA 메모리 부족: {oom}")

            # 모델/트레이너 삭제 및 캐시 정리
            for obj_name in ("trainer", "model"):
                obj = locals().get(obj_name)
                if obj is not None:
                    del obj
            gc.collect()
            torch.cuda.empty_cache()

            if attempt >= len(fallback_schedule) - 1:
                print("[OOM] 모든 설정에서 OOM 발생 — GPU 메모리가 부족합니다.")
                print("      추가 조치: bitsandbytes 또는 DeepSpeed ZeRO 설정 확인")
                raise
            continue

        except Exception as e:
            # OOM 외 오류 — 부분 로그 저장 후 재발생
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            if monitor_cb._logs:
                try:
                    plot_curves(monitor_cb._logs, args.base_dir)
                    print(f"[INFO] 부분 그래프 저장: {args.base_dir}")
                except Exception:
                    pass
            traceback.print_exc()
            raise

    # ── 8. LoRA 어댑터 저장 ─────────────────────────────────────────────
    if trainer is None:
        raise RuntimeError("학습이 완료되지 않았습니다.")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[DONE] LoRA adapter: {args.output_dir}")

    # ── 9. Merge ─────────────────────────────────────────────────────────
    if not args.skip_merge:
        merge_and_save(args.model_id, args.output_dir, args.merged_dir)
    else:
        print(f"[SKIP] merge 건너뜀 — 나중에: python train.py --skip_train")

    # ── 최종 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[완료] 파이프라인 종료")
    print("=" * 60)
    print(f"  GPU             : {gpu['name']}")
    print(f"  실제 학습 설정   : batch={final_bs}  seq_len={final_sl}  "
          f"grad_accum={final_ga}")
    print(f"  유효 배치 크기   : {final_bs * final_ga}")
    print(f"  LoRA adapter    : {args.output_dir}")
    print(f"  CSV 로그        : {os.path.join(args.base_dir, 'training_logs.csv')}")
    if not args.skip_merge:
        print(f"  Merged model    : {args.merged_dir}")
        print(f"\n  다음 단계:")
        print(f"    python export_gguf.py --merged_dir {args.merged_dir} "
              f"--output_dir {args.base_dir}")


if __name__ == "__main__":
    main()
