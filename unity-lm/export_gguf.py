"""
export_gguf.py — Merged HuggingFace 모델 → GGUF 변환 및 양자화
환경  : Ubuntu 22.04 / Python 3.10+
목표  : GTX 1060 3GB 추론 최적화

출력 파일:
  qwen-unity.gguf          — F16 기준 GGUF (약 3.1GB, 양자화 원본)
  qwen-unity-q4_k_m.gguf  — Q4_K_M 양자화 (약 0.95GB, 권장)
  qwen-unity-q4_k_s.gguf  — Q4_K_S 양자화 (약 0.90GB)
  qwen-unity-iq4_xs.gguf  — IQ4_XS 양자화 (약 0.85GB, 최소)

필요 조건:
  1) llama.cpp 빌드 완료:
       git clone https://github.com/ggml-org/llama.cpp
       cd llama.cpp
       cmake -B build -DGGML_CUDA=ON    # GPU 가속 (선택사항)
       cmake --build build --config Release -j$(nproc)

  2) GGUF 변환 Python 의존성:
       pip install gguf numpy

실행 예:
  # 기본 실행 (merged_model → 현재 디렉토리에 GGUF 생성)
  python export_gguf.py

  # 경로 명시
  python export_gguf.py --merged_dir ./merged_model --output_dir ./gguf_output

  # F16 변환 건너뜀 (이미 존재할 때 양자화만 재실행)
  python export_gguf.py --skip_f16

  # 특정 양자화 형식만
  python export_gguf.py --quant_types Q4_K_M
"""

import argparse
import os
import subprocess
import sys


# ──────────────────────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HuggingFace → GGUF 변환 및 양자화 (GTX 1060 3GB 최적화)"
    )
    p.add_argument("--merged_dir",    default="./merged_fp16_model",
                   help="Merged HuggingFace 모델 디렉토리 (train.py 출력)")
    p.add_argument("--output_dir",    default="./output",
                   help="GGUF 파일 출력 디렉토리 (기본: ./output)")
    p.add_argument("--llama_cpp_dir", default="./llama-cpp",
                   help="llama.cpp 루트 디렉토리")
    p.add_argument("--output_name",   default="qwen-unity",
                   help="출력 파일 기본 이름 (확장자 제외)")
    p.add_argument("--quant_types",   nargs="+",
                   default=["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "IQ3_M"],
                   help="생성할 양자화 형식 (기본 5종: Q4_K_M Q4_K_S IQ4_XS Q3_K_M IQ3_M)")
    p.add_argument("--skip_f16",      action="store_true",
                   help="F16 GGUF 변환 건너뜀 (이미 존재할 때 양자화만 재실행)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def run(cmd: list, desc: str = ""):
    """
    외부 명령 실행 — 실패 시 즉시 스크립트 종료
    명령과 설명을 출력하여 진행 상황을 추적
    """
    print(f"\n[RUN] {desc}")
    print(f"      $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] 명령 실패 (exit code: {result.returncode})")
        print(f"        실패한 명령: {' '.join(str(c) for c in cmd)}")
        sys.exit(result.returncode)


def file_size_str(path: str) -> str:
    """파일 크기를 사람이 읽기 쉬운 형식으로 반환 (GB/MB)"""
    size = os.path.getsize(path)
    if size >= 1e9:
        return f"{size / 1e9:.2f}GB"
    return f"{size / 1e6:.1f}MB"


# ──────────────────────────────────────────────────────────────────────────────
# llama.cpp 확인
# ──────────────────────────────────────────────────────────────────────────────

def check_llama_cpp(llama_dir: str) -> tuple[str, str]:
    """
    llama.cpp 디렉토리 유효성 검사

    Returns:
        (convert_script 경로, llama-quantize 바이너리 경로)

    필요 파일:
      - convert_hf_to_gguf.py   : HuggingFace → GGUF 변환
      - build/bin/llama-quantize : GGUF → 양자화 GGUF 변환
    """
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    # Windows 프리빌드 패키지: llama-quantize.exe가 루트에 바로 있음
    quantize_bin = os.path.join(llama_dir, "llama-quantize.exe")
    if not os.path.isfile(quantize_bin):
        quantize_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize")

    if not os.path.isfile(convert_script):
        print(f"\n[ERROR] convert_hf_to_gguf.py 없음: {convert_script}")
        print("  llama.cpp를 클론하세요:")
        print("    git clone https://github.com/ggml-org/llama.cpp")
        sys.exit(1)

    if not os.path.isfile(quantize_bin):
        print(f"\n[ERROR] llama-quantize 없음: {quantize_bin}")
        print("  llama.cpp를 빌드하세요:")
        print(f"    cd {llama_dir}")
        print("    cmake -B build -DGGML_CUDA=ON    # GPU 가속 (선택사항)")
        print("    cmake --build build --config Release -j$(nproc)")
        sys.exit(1)

    print(f"[CHECK] convert_hf_to_gguf.py : {convert_script}")
    print(f"[CHECK] llama-quantize         : {quantize_bin}")
    return convert_script, quantize_bin


# ──────────────────────────────────────────────────────────────────────────────
# 변환 단계
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_f16_gguf(convert_script: str, merged_dir: str, output_path: str):
    """
    HuggingFace 모델 → F16 GGUF 변환

    F16(반정밀도 부동소수점)으로 저장하는 이유:
      - 이후 모든 양자화의 기준 파일로 사용
      - BF16보다 llama.cpp 호환성이 높음
      - 모델 크기: Qwen2.5-1.5B 기준 약 3.1GB

    GTX 1060 3GB 호환 양자화 우선순위 (크기 순):
      Q4_K_M  ~0.95GB ← 권장 (품질·속도·크기 균형)
      Q4_K_S  ~0.90GB
      IQ4_XS  ~0.85GB
      Q3_K_M  ~0.77GB ← VRAM 빠듯할 때
      IQ3_M   ~0.70GB ← 최소 크기
    """
    run(
        [
            sys.executable, convert_script,
            merged_dir,
            "--outfile", output_path,
            "--outtype", "f16",
        ],
        desc=f"HuggingFace → F16 GGUF: {os.path.basename(output_path)}",
    )


def quantize_gguf(quantize_bin: str, input_path: str,
                  output_path: str, quant_type: str):
    """
    F16 GGUF → 양자화 GGUF 변환

    양자화 형식 선택 기준 (GTX 1060 3GB):
      Q4_K_M  : K-quants 중형 (권장) — 품질·속도·크기 균형
      Q4_K_S  : K-quants 소형        — Q4_K_M보다 약간 작고 빠름
      IQ4_XS  : i-quants 초소형      — 최소 크기, 약간 낮은 품질
    """
    run(
        [quantize_bin, input_path, output_path, quant_type],
        desc=f"{quant_type} 양자화: {os.path.basename(output_path)}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 경로 설정 ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # 출력 파일 경로 정의
    # 예: ./qwen-unity.gguf, ./qwen-unity-q4_k_m.gguf
    f16_path  = os.path.join(args.output_dir, f"{args.output_name}.gguf")
    name_base = os.path.join(args.output_dir, args.output_name)

    print("=" * 60)
    print("[EXPORT] GGUF 변환 및 양자화 파이프라인 시작")
    print("=" * 60)
    print(f"  입력  : {args.merged_dir}")
    print(f"  출력  : {args.output_dir}")
    print(f"  형식  : F16 → {', '.join(args.quant_types)}")

    # ── llama.cpp 확인 ────────────────────────────────────────────────────
    print(f"\n[SETUP] llama.cpp 확인: {args.llama_cpp_dir}")
    convert_script, quantize_bin = check_llama_cpp(args.llama_cpp_dir)

    # ── Merged 모델 존재 확인 ────────────────────────────────────────────
    if not os.path.isdir(args.merged_dir):
        print(f"\n[ERROR] Merged 모델 없음: {args.merged_dir}")
        print("  train.py를 먼저 실행하세요:")
        print("    python train.py")
        sys.exit(1)

    # ── Step 1: HuggingFace → F16 GGUF ───────────────────────────────────
    print("\n" + "─" * 60)
    print("[STEP 1/2] HuggingFace → F16 GGUF")
    print("─" * 60)

    if args.skip_f16 and os.path.isfile(f16_path):
        print(f"[SKIP] F16 GGUF 이미 존재: {f16_path} ({file_size_str(f16_path)})")
    else:
        convert_to_f16_gguf(convert_script, args.merged_dir, f16_path)
        print(f"[OK  ] F16 GGUF: {f16_path} ({file_size_str(f16_path)})")

    # ── Step 2: 양자화 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"[STEP 2/2] 양자화: {', '.join(args.quant_types)}")
    print("─" * 60)

    # 양자화 타입명을 소문자+언더스코어로 변환 → 파일명에 사용
    # 예: Q4_K_M → qwen-unity-q4_k_m.gguf
    quant_paths = {}
    for qtype in args.quant_types:
        out_path = f"{name_base}-{qtype.lower()}.gguf"
        quantize_gguf(quantize_bin, f16_path, out_path, qtype)
        quant_paths[qtype] = out_path
        print(f"[OK  ] {qtype:<10}: {out_path} ({file_size_str(out_path)})")

    # ── 결과 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[완료] GGUF 변환 및 양자화 파이프라인 완료")
    print("=" * 60)
    print("\n생성된 파일:")
    print(f"  {f16_path:<50} {file_size_str(f16_path):>8}  (F16 기준)")
    for qtype, path in quant_paths.items():
        print(f"  {path:<50} {file_size_str(path):>8}  ({qtype})")

    print("\nGTX 1060 3GB 추론 권장 순서:")
    print("  1순위: Q4_K_M  — 품질·속도·크기 균형, VRAM 여유 있음")
    print("  2순위: Q4_K_S  — Q4_K_M보다 약간 빠름")
    print("  3순위: IQ4_XS  — VRAM 부족 시 최후 수단")

    q4km_path = quant_paths.get("Q4_K_M", f"{name_base}-q4_k_m.gguf")
    quantize_cli = os.path.join(args.llama_cpp_dir, "build", "bin", "llama-cli")

    print(f"\n추론 실행 예:")
    print(f"  {quantize_cli} \\")
    print(f"    -m {q4km_path} \\")
    print(f"    -n 512 --temp 0.1 -ngl 99 \\")
    print(f'    -p "<|im_start|>user\\nHow do I move a GameObject with Rigidbody?'
          f'<|im_end|>\\n<|im_start|>assistant\\n"')
    print()


if __name__ == "__main__":
    main()
