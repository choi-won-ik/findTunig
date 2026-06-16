"""
merge_lora.py — LoRA 어댑터를 FP16 base에 merge하여 단일 HF 모델로 저장
출력: ./merged_fp16_model  (GGUF 변환 입력으로 사용)

주의: merge 시에는 base를 4bit가 아닌 FP16으로 로드해야 가중치 합성이 정확합니다.

그래프 출력 (train.py 학습 완료 후 trainer_state.json이 있을 때):
  - loss_curve.png    : train loss / eval loss vs step
  - accuracy_curve.png: eval accuracy (≈ exp(-eval_loss)) vs epoch
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading

# daemon 스레드의 UnicodeDecodeError를 억제 (huggingface_hub 내부 subprocess 인코딩 문제)
if sys.platform == 'win32':
    _orig_excepthook = threading.excepthook
    def _thread_excepthook(args):
        if args.exc_type is UnicodeDecodeError:
            return  # CP949↔UTF-8 충돌 무시
        _orig_excepthook(args)
    threading.excepthook = _thread_excepthook

# Windows에서 subprocess가 CP949 출력을 UTF-8로 디코딩하려다 실패하는 문제 방지
# text=True / encoding='utf-8' / universal_newlines=True 모두 처리
_orig_popen_init = subprocess.Popen.__init__
def _popen_utf8_errors(self, *args, **kwargs):
    if (kwargs.get("encoding") == "utf-8"
            or kwargs.get("text")
            or kwargs.get("universal_newlines")):
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
    _orig_popen_init(self, *args, **kwargs)
subprocess.Popen.__init__ = _popen_utf8_errors

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def plot_training_graphs(adapter_dir: str, out_dir: str) -> None:
    state_path = os.path.join(adapter_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"[GRAPH] trainer_state.json 없음 ({state_path}) — 그래프 생성 건너뜀")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[GRAPH] matplotlib 미설치 — 그래프 생성 건너뜀 (pip install matplotlib)")
        return

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    # train loss: "loss" 키가 있는 항목 (step 단위)
    train_steps, train_losses = [], []
    # eval loss: "eval_loss" 키가 있는 항목 (epoch 단위)
    eval_epochs, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_epochs.append(entry.get("epoch", len(eval_epochs)))
            eval_losses.append(entry["eval_loss"])

    os.makedirs(out_dir, exist_ok=True)

    # ── Loss 그래프 ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if train_steps:
        ax.plot(train_steps, train_losses, label="Train Loss", linewidth=1.5, color="#2196F3")
    if eval_epochs and train_steps:
        # eval epoch → step 축으로 환산 (마지막 step 기준)
        total_steps = train_steps[-1]
        total_epochs = state.get("epoch", eval_epochs[-1])
        steps_per_epoch = total_steps / total_epochs if total_epochs else 1
        eval_steps = [e * steps_per_epoch for e in eval_epochs]
        ax.plot(eval_steps, eval_losses, label="Eval Loss", linewidth=1.8,
                color="#F44336", marker="o", markersize=5)
    elif eval_epochs:
        ax.plot(eval_epochs, eval_losses, label="Eval Loss", linewidth=1.8,
                color="#F44336", marker="o", markersize=5)

    ax.set_xlabel("Step" if train_steps else "Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Eval Loss Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[GRAPH] Loss 그래프 저장: {loss_path}")

    # ── Accuracy 그래프 (eval_loss → exp(-eval_loss) 근사) ────────────────
    if not eval_losses:
        print("[GRAPH] eval_loss 데이터 없음 — accuracy 그래프 생성 건너뜀")
        return

    accuracies = [math.exp(-l) for l in eval_losses]
    x_vals = eval_epochs if eval_epochs else list(range(1, len(accuracies) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_vals, accuracies, label="Approx. Accuracy (exp(−eval_loss))",
            linewidth=1.8, color="#4CAF50", marker="o", markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (approx.)")
    ax.set_title("Eval Accuracy Curve\n(approximated as exp(−eval_loss))")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[GRAPH] Accuracy 그래프 저장: {acc_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--adapter", default="./lora_adapter")
    p.add_argument("--out", default="./merged_fp16_model")
    args = p.parse_args()

    # ── 학습 그래프 생성 (모델 로드 전에 먼저 수행) ────────────────────────
    plot_training_graphs(args.adapter, args.out)

    # FP16 base 로드 (CPU에서 안전하게 merge; VRAM 부담 없음)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu"
    )

    # 어댑터 결합 후 가중치에 병합
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)

    # 어댑터에 토크나이저가 있으면 사용, 없으면 base에서 로드
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)

    print(f"[DONE] merged FP16 model saved to: {args.out}")


if __name__ == "__main__":
    main()
