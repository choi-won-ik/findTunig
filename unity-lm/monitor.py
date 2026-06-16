"""
monitor.py — 학습 모니터링 및 시각화 모듈
환경: Ubuntu 22.04 / Python 3.10+

구성:
  MonitoredSFTTrainer    — compute_loss에서 Train Accuracy를 직접 누적하는 SFTTrainer 서브클래스
  TrainingMonitorCallback — CSV 기록 + eval accuracy 계산 + 그래프 자동 생성
  plot_curves            — loss_curve.png / accuracy_curve.png 생성 (1920×1080, dpi=300)

Token-level Accuracy 계산 전략:
  - Train : compute_loss 내에서 chunk 단위 argmax → 로그 주기(logging_steps)마다 평균
  - Eval  : on_evaluate 콜백에서 eval_dataset을 배치 단위로 직접 추론
             (전체 logits를 메모리에 쌓지 않아 VRAM 절약)

Perplexity:
  math.exp(loss) 로 직접 계산 — 추가 연산 불필요

사용 예:
  from monitor import MonitoredSFTTrainer, TrainingMonitorCallback, plot_curves

  callback = TrainingMonitorCallback(
      output_dir="./output",
      eval_dataset=eval_ds,
      tokenizer=tokenizer,
  )
  trainer = MonitoredSFTTrainer(
      ...,
      callbacks=[callback],
  )
  callback.trainer = trainer   # 트레이너 참조 설정 (eval 시 모델 접근용)
  trainer.train()
"""

import csv
import math
import os
import shutil
import traceback

import matplotlib
# 헤드리스 서버(학습 서버)에서도 PNG 생성 가능하도록 비대화형 백엔드 사용
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

# CSV에 저장할 컬럼 순서 (변경 시 헤더와 행 모두 일치해야 함)
FIELDNAMES = [
    "epoch", "step",
    "train_loss",      "eval_loss",
    "train_accuracy",  "eval_accuracy",
    "learning_rate",
    "perplexity",      "eval_perplexity",
]


# ──────────────────────────────────────────────────────────────────────────────
# MonitoredSFTTrainer
# ──────────────────────────────────────────────────────────────────────────────

class MonitoredSFTTrainer(SFTTrainer):
    """
    Token-level Training Accuracy를 compute_loss 단계에서 직접 계산하는 SFTTrainer

    동작 원리:
      compute_loss()에서 모델 출력 logits을 no_grad 상태로 chunk 처리 →
      logging_steps마다 log() 호출 시 누산값을 train_accuracy로 주입 →
      TrainingMonitorCallback.on_log()가 CSV에 기록

    메모리 최적화:
      - chunk_size=128 포지션씩 argmax 처리 (전체 logits를 한 번에 유지하지 않음)
      - detach()로 계산 그래프에서 분리하여 역전파에 영향 없음
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_acc_correct: int = 0
        self._train_acc_total:   int = 0

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        SFTTrainer의 compute_loss에 token-level accuracy 누적 추가

        super().compute_loss(return_outputs=True)로 outputs을 받아
        기존 loss 계산 로직을 그대로 유지하면서 logits만 추가로 분석
        """
        # 부모 클래스의 loss 계산 위임 (packing, label masking 등 보존)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # logits에서 accuracy 계산 — no_grad + detach로 역전파 영향 없음
        with torch.no_grad():
            logits = outputs.logits.detach()    # (B, L, V)
            labels = inputs.get("labels")       # (B, L)

            if labels is not None and logits.shape[1] > 1:
                # 128 포지션씩 chunk 처리 → 최대 128 × B × V × 2byte ≈ 37MB (fp16)
                chunk_size = 128
                for c in range(0, logits.shape[1] - 1, chunk_size):
                    end = min(c + chunk_size, logits.shape[1] - 1)

                    # logit[t] → label[t+1] 예측 (Causal LM의 shift 구조)
                    chunk_pred = logits[:, c:end, :].argmax(dim=-1)
                    chunk_lbl  = labels[:, c + 1:end + 1]
                    mask       = chunk_lbl != -100  # 패딩·무시 토큰 제외

                    if mask.sum() > 0:
                        self._train_acc_correct += ((chunk_pred == chunk_lbl) & mask).sum().item()
                        self._train_acc_total   += mask.sum().item()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):
        """
        logging_steps마다 호출 — train_accuracy를 logs에 주입 후 누산기 초기화

        콜백의 on_log()가 logs["train_accuracy"]를 읽어 CSV에 기록
        """
        if "loss" in logs and self._train_acc_total > 0:
            logs["train_accuracy"] = self._train_acc_correct / self._train_acc_total
            # 다음 logging_steps 구간을 위한 초기화
            self._train_acc_correct = 0
            self._train_acc_total   = 0

        super().log(logs, *args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# TrainingMonitorCallback
# ──────────────────────────────────────────────────────────────────────────────

class TrainingMonitorCallback(TrainerCallback):
    """
    학습 지표를 CSV로 기록하고 종료 시 시각화 그래프를 자동 생성하는 콜백

    이벤트:
      on_evaluate  — eval_accuracy 배치 단위 계산 + perplexity 계산
      on_log       — CSV 기록 (train + eval 지표 통합)
      on_train_end — loss_curve.png / accuracy_curve.png 생성

    생성 파일:
      {output_dir}/training_logs.csv
      {output_dir}/loss_curve.png
      {output_dir}/accuracy_curve.png

    사용:
      callback = TrainingMonitorCallback(output_dir, eval_dataset, tokenizer)
      trainer  = MonitoredSFTTrainer(callbacks=[callback], ...)
      callback.trainer = trainer   # ← 트레이너 참조를 직접 설정 (필수)
    """

    def __init__(
        self,
        output_dir:       str,
        eval_dataset,
        tokenizer,
        eval_batch_size:   int = 1,
        max_eval_batches:  int = 20,
        max_length:        int = 1024,
    ):
        """
        Args:
            output_dir:      CSV·그래프 저장 경로
            eval_dataset:    "text" 필드 Dataset (build_dataset() 반환값)
            tokenizer:       Qwen2.5 토크나이저
            eval_batch_size: eval accuracy 배치 크기 (VRAM 절약을 위해 기본 1)
            max_eval_batches:accuracy 계산에 사용할 최대 배치 수 (기본 20)
            max_length:      토크나이즈 시 최대 토큰 수
        """
        self.output_dir      = output_dir
        self.eval_dataset    = eval_dataset
        self.tokenizer       = tokenizer
        self.eval_batch_size = eval_batch_size
        self.max_eval_batches = max_eval_batches
        self.max_length      = max_length
        self.trainer         = None  # 트레이너 생성 후 외부에서 직접 설정

        self.csv_path = os.path.join(output_dir, "training_logs.csv")

        # on_evaluate 결과를 on_log까지 임시 보관 (두 이벤트가 서로 다른 타이밍에 발생)
        self._pending_eval: dict = {}

        # 전체 로그 목록 (그래프 생성용, 메모리 내 보관)
        self._logs: list[dict] = []

        # CSV 초기화 (헤더 작성)
        os.makedirs(output_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        print(f"[MONITOR] CSV 초기화: {self.csv_path}")

    @staticmethod
    def _fmt(v, digits: int = 6):
        if v is None or v == "":
            return ""
        try:
            return round(float(v), digits)
        except (TypeError, ValueError):
            return ""

    # ── eval accuracy 계산 ────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_eval_accuracy(self, model) -> float:
        """
        Eval 데이터셋에서 Token-level Accuracy 계산

        - 배치 단위 추론으로 전체 logits를 메모리에 쌓지 않음
        - 각 배치의 logits도 64 포지션 chunk로 나눠 argmax 처리
        - max_eval_batches 제한으로 긴 eval 시간 방지

        Returns:
            float: token-level accuracy (0.0 ~ 1.0)
        """
        device      = next(model.parameters()).device
        was_training = model.training
        model.eval()

        # "text" 필드를 토크나이즈하여 DataLoader에 공급
        def _tokenize(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        tok_ds = self.eval_dataset.map(
            _tokenize, batched=True, remove_columns=["text"]
        )
        tok_ds.set_format("torch")

        # DataCollatorForLanguageModeling(mlm=False):
        #   - 배치 내 패딩 → labels = input_ids (패딩 위치는 -100)
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        loader = DataLoader(
            tok_ds,
            batch_size=self.eval_batch_size,
            collate_fn=collator,
            shuffle=False,
        )

        total_correct = 0
        total_tokens  = 0

        for i, batch in enumerate(loader):
            if i >= self.max_eval_batches:
                break

            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits  = outputs.logits   # (B, L, V)
            labels  = batch["labels"]  # (B, L)

            if logits.shape[1] <= 1:
                continue

            # 64 포지션 chunk 처리 → 최대 1 × 64 × 151936 × 2byte ≈ 18MB
            chunk_size = 64
            for c in range(0, logits.shape[1] - 1, chunk_size):
                end        = min(c + chunk_size, logits.shape[1] - 1)
                chunk_pred = logits[:, c:end, :].argmax(dim=-1)
                chunk_lbl  = labels[:, c + 1:end + 1]
                mask       = chunk_lbl != -100

                total_correct += ((chunk_pred == chunk_lbl) & mask).sum().item()
                total_tokens  += mask.sum().item()

        # 학습 모드 복원
        if was_training:
            model.train()

        return total_correct / total_tokens if total_tokens > 0 else 0.0

    # ── TrainerCallback 이벤트 핸들러 ─────────────────────────────────────

    def on_evaluate(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        metrics: dict = None,
        **kwargs,
    ):
        """
        eval 루프 완료 직후 호출

        eval_loss → perplexity 계산
        eval_accuracy → 배치 단위 추론으로 계산
        결과를 _pending_eval에 저장 → on_log에서 CSV 기록에 병합
        """
        if metrics is None:
            return

        eval_loss       = metrics.get("eval_loss")
        eval_perplexity = math.exp(eval_loss) if eval_loss is not None else None

        # trainer 참조를 통해 모델에 접근하여 eval accuracy 계산
        eval_accuracy = None
        if self.trainer is not None:
            try:
                eval_accuracy = self._compute_eval_accuracy(self.trainer.model)
                print(
                    f"\n[MONITOR] step={state.global_step}  "
                    f"eval_accuracy={eval_accuracy:.4f}  "
                    f"eval_perplexity={eval_perplexity:.2f}"
                )
            except Exception as e:
                print(f"[MONITOR][WARN] eval accuracy 계산 실패: {e}")

        # on_log가 아직 train loss를 읽지 않았을 수 있으므로 임시 저장
        self._pending_eval = {
            "eval_loss":       eval_loss,
            "eval_accuracy":   eval_accuracy,
            "eval_perplexity": eval_perplexity,
        }

    def on_log(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        logs:    dict = None,
        **kwargs,
    ):
        """
        logging_steps마다 호출 — train 지표 + 가장 최근 eval 지표를 CSV에 기록

        logs 딕셔너리는 MonitoredSFTTrainer.log()가 train_accuracy를 주입한 뒤 전달됨
        """
        if logs is None:
            return

        # eval 전용 on_log(eval_loss만 있고 loss 없음)는 on_evaluate에서 처리됨
        if "eval_loss" in logs and "loss" not in logs:
            return

        train_loss = logs.get("loss")
        if train_loss is None:
            return

        # perplexity = exp(cross-entropy loss)
        perplexity = math.exp(train_loss)

        row = {
            "epoch":           round(float(state.epoch or 0), 4),
            "step":            state.global_step,
            "train_loss":      self._fmt(train_loss),
            "eval_loss":       self._fmt(self._pending_eval.get("eval_loss")),
            "train_accuracy":  self._fmt(logs.get("train_accuracy")),
            "eval_accuracy":   self._fmt(self._pending_eval.get("eval_accuracy")),
            "learning_rate":   logs.get("learning_rate", ""),
            "perplexity":      self._fmt(perplexity, 4),
            "eval_perplexity": self._fmt(self._pending_eval.get("eval_perplexity"), 4),
        }

        self._logs.append(row)

        # CSV 추가 기록 — 오류 발생해도 학습은 계속
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        except Exception as e:
            print(f"[MONITOR][WARN] CSV 기록 실패: {e}")

    def on_train_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """학습 완료(또는 중단) 후 마지막 step 보완 기록 + 그래프 자동 생성"""
        # ── 마지막 step 누락 보완 ─────────────────────────────────────────
        # logging_steps 주기와 관계없이 최종 global_step이 반드시 CSV에 포함되도록 보장
        final_step = state.global_step
        if not self._logs or self._logs[-1]["step"] != final_step:
            last_train = next(
                (e for e in reversed(state.log_history) if "loss" in e),
                None,
            )
            if last_train:
                train_loss = last_train["loss"]
                row = {
                    "epoch":           round(float(state.epoch or 0), 4),
                    "step":            final_step,
                    "train_loss":      self._fmt(train_loss),
                    "eval_loss":       self._fmt(self._pending_eval.get("eval_loss")),
                    "train_accuracy":  self._fmt(last_train.get("train_accuracy")),
                    "eval_accuracy":   self._fmt(self._pending_eval.get("eval_accuracy")),
                    "learning_rate":   last_train.get("learning_rate", ""),
                    "perplexity":      self._fmt(math.exp(train_loss), 4),
                    "eval_perplexity": self._fmt(self._pending_eval.get("eval_perplexity"), 4),
                }
                self._logs.append(row)
                try:
                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
                    print(f"[MONITOR] 최종 step {final_step} 보완 기록 완료")
                except Exception as e:
                    print(f"[MONITOR][WARN] 최종 step CSV 기록 실패: {e}")

        print(f"\n[MONITOR] 그래프 생성 중 ({len(self._logs)}개 로그 포인트)...")
        try:
            plot_curves(self._logs, self.output_dir)
        except Exception as e:
            print(f"[MONITOR][ERROR] 그래프 생성 실패: {e}")
            traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# plot_curves — PNG 그래프 생성
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(logs: list, output_dir: str, dpi: int = 300):
    """
    학습 로그로부터 Loss 및 Accuracy 그래프를 PNG로 저장

    생성 파일:
      {output_dir}/loss_curve.png      — Training / Validation Loss
      {output_dir}/accuracy_curve.png  — Training / Validation Accuracy (%)

    Args:
        logs:       TrainingMonitorCallback._logs 형식의 dict 리스트
        output_dir: 저장 디렉토리
        dpi:        해상도 (기본 300, figsize=(19.2, 10.8)로 1920×1080 보장)
    """
    if not logs:
        print("[MONITOR][WARN] 로그 없음 — 그래프를 생성하지 않습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── 데이터 추출 헬퍼 ──────────────────────────────────────────────────
    def _series(key: str) -> tuple[list, list]:
        """(steps, values) — None/빈값은 건너뜀"""
        xs, ys = [], []
        for row in logs:
            v = row.get(key)
            step = row.get("step")
            if v not in (None, "", "nan") and step is not None:
                try:
                    xs.append(int(step))
                    ys.append(float(v))
                except (TypeError, ValueError):
                    pass
        return xs, ys

    # ── 공통 스타일 설정 ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size":       12,
        "axes.titlesize":  18,
        "axes.labelsize":  14,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # ── Loss 그래프 ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    xl, yl = _series("train_loss")
    if xl:
        ax.plot(xl, yl, label="Train Loss", color="#2196F3",
                linewidth=2, alpha=0.9)

    xe, ye = _series("eval_loss")
    if xe:
        ax.plot(xe, ye, label="Validation Loss", color="#F44336",
                linewidth=2, linestyle="--", marker="o", markersize=6)

    ax.set_title("Training and Validation Loss", fontweight="bold", pad=15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # y축 하단 여백 (0 근처 값이 잘리지 않도록)
    all_loss = yl + ye
    if all_loss:
        y_min = max(0.0, min(all_loss) - (max(all_loss) - min(all_loss)) * 0.05)
        ax.set_ylim(bottom=y_min)

    fig.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[MONITOR] Loss 그래프: {loss_path}")

    # ── Accuracy 그래프 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    xt, yt = _series("train_accuracy")
    if xt:
        # 0~1 → 0~100% 변환
        ax.plot(xt, [v * 100 for v in yt], label="Train Accuracy",
                color="#4CAF50", linewidth=2, alpha=0.9)

    xv, yv = _series("eval_accuracy")
    if xv:
        ax.plot(xv, [v * 100 for v in yv], label="Validation Accuracy",
                color="#FF9800", linewidth=2, linestyle="--",
                marker="o", markersize=6)

    ax.set_title("Training and Validation Accuracy", fontweight="bold", pad=15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[MONITOR] Accuracy 그래프: {acc_path}")


# ──────────────────────────────────────────────────────────────────────────────
# DriveBackupCallback — save_steps마다 체크포인트를 Google Drive로 백업
# ──────────────────────────────────────────────────────────────────────────────

class DriveBackupCallback(TrainerCallback):
    """
    save_steps마다 체크포인트를 Google Drive로 실시간 백업하는 콜백.

    로컬 저장 완료 직후(on_save) Drive로 복사하므로 세션이 끊겨도
    Drive의 최신 checkpoint-N 에서 resume 가능.

    학습 완료 시(on_train_end) 최종 어댑터도 Drive에 복사.

    추천 이유(vs output_dir을 Drive로 직접 설정):
      - Drive I/O는 로컬 NVMe/tmpfs 대비 ~10× 느림. output_dir을 Drive로 두면
        save_steps마다 수백MB를 Drive에 쓰면서 학습이 멈춤.
      - 이 콜백은 로컬 저장 완료 후 비동기 복사하므로 학습 속도에 거의 영향 없음.
      - 단, 로컬 저장~Drive 복사 사이 세션이 끊기면 해당 checkpoint는 누락됨.
        save_steps를 충분히 작게 유지하면 손실 구간이 최소화됨.
    """

    def __init__(self, adapter_dir: str, drive_backup_dir: str):
        """
        Args:
            adapter_dir:      LoRA 어댑터 저장 경로 (SFTConfig.output_dir과 동일)
            drive_backup_dir: Drive 백업 목적지 (예: /content/drive/.../output_colab)
        """
        self.adapter_dir      = adapter_dir
        self.drive_backup_dir = drive_backup_dir
        os.makedirs(drive_backup_dir, exist_ok=True)

    def _copy_tree(self, src: str, dst: str, label: str = "") -> None:
        """src 디렉토리를 dst로 복사. 실패 시 학습을 중단하지 않고 경고만 출력."""
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            size_mb = sum(
                os.path.getsize(os.path.join(dp, fn))
                for dp, _dn, fns in os.walk(dst)
                for fn in fns
            ) / 1e6
            print(f"[BACKUP] {label} → {dst}  ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"[BACKUP][WARN] {label} Drive 복사 실패 (학습은 계속): {exc}")

    def on_save(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """체크포인트 로컬 저장 완료 직후 Drive로 복사"""
        ckpt_name = f"checkpoint-{state.global_step}"
        ckpt_src  = os.path.join(args.output_dir, ckpt_name)
        ckpt_dst  = os.path.join(self.drive_backup_dir, ckpt_name)
        if os.path.isdir(ckpt_src):
            self._copy_tree(ckpt_src, ckpt_dst, ckpt_name)

    def on_train_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """학습 완료 후 최종 어댑터를 Drive에 백업 (trainer.save_model() 이후 호출됨)"""
        lora_dst = os.path.join(self.drive_backup_dir, "lora_adapter")
        if os.path.isdir(self.adapter_dir):
            self._copy_tree(self.adapter_dir, lora_dst, "lora_adapter (최종)")
