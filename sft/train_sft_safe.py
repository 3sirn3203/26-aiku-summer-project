import json
import os
import sys
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model

try:
    import psutil
except ImportError:
    psutil = None

# CUDA VRAM 메모리 단편화 방지 억제 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================================================================
# 공용 리소스 안전 판정 함수
# ⚠️ 아래 판정 로직(threshold, 집계 방식)은 기존 코드와 100% 동일합니다.
#    - GPU VRAM: 디바이스 전체 기준(torch.cuda.mem_get_info) 그대로
#    - 시스템 RAM: OS 전체 기준(psutil.virtual_memory) 그대로
#    - 우리 프로그램 자식 프로세스(워커) 재귀 합산 그대로
#    - 멀티스레드는 RSS에 이미 포함되어 별도 처리 불필요 (기존과 동일)
#    콜백 전용 로직이었던 것을 여러 지점(데이터 로드 전/모델 로딩 전/학습 중)에서
#    재사용할 수 있도록 함수로만 추출했습니다.
# =========================================================================
def check_resource_safety(
    stage: str,
    max_vram_pct=0.92,
    min_free_vram_mb=1000,
    min_free_ram_gb=8.0,
    max_ram_use_ratio_of_avail=0.85,
):
    """리소스가 안전하면 (True, ""), 위험하면 (False, 사유) 반환."""
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_bytes = total_bytes - free_bytes
        vram_pct = used_bytes / total_bytes
        free_mb = free_bytes / (1024 * 1024)

        if vram_pct > max_vram_pct or free_mb < min_free_vram_mb:
            return False, (
                f"[{stage}] GPU VRAM 위험수치! 사용률 {vram_pct * 100:.1f}% "
                f"| 남은 VRAM: {free_mb:.1f} MB"
            )

    if psutil is not None:
        try:
            curr_proc = psutil.Process()
            children = curr_proc.children(recursive=True)
            my_total_rss_bytes = curr_proc.memory_info().rss + sum(
                c.memory_info().rss for c in children if c.is_running()
            )
            my_total_rss_gb = my_total_rss_bytes / (1024 ** 3)

            sys_mem = psutil.virtual_memory()
            avail_ram_gb = sys_mem.available / (1024 ** 3)

            if avail_ram_gb < min_free_ram_gb:
                return False, (
                    f"[{stage}] 시스템 여유 RAM 부족! (남은 여유 RAM: {avail_ram_gb:.2f} GB "
                    f"< 최소 보장: {min_free_ram_gb} GB)"
                )

            if my_total_rss_gb > 4.0 and my_total_rss_gb > (avail_ram_gb * max_ram_use_ratio_of_avail):
                return False, (
                    f"[{stage}] 내 프로그램 메모리가 여유 RAM의 {max_ram_use_ratio_of_avail*100:.0f}%를 초과! "
                    f"내 프로그램 점유: {my_total_rss_gb:.2f} GB | 현재 여유 RAM: {avail_ram_gb:.2f} GB"
                )
        except Exception:
            pass

    return True, ""


def ensure_resources_safe_or_raise(stage: str, **kwargs):
    """데이터 로드/모델 로딩처럼 Trainer 콜백 범위 밖의 단계에서 수동으로 호출."""
    ok, reason = check_resource_safety(stage, **kwargs)
    if not ok:
        print(f"\n🚨 [RESOURCE WARNING] {reason}")
        print("🛑 OOM 방지를 위해 진행 전 안전하게 중단합니다.")
        raise RuntimeError(reason)


# --- GPU VRAM & System RAM 실시간 감시 콜백 (학습 스텝 중) ---
class ResourceSafetyCallback(TrainerCallback):
    def __init__(self, max_vram_pct=0.92, min_free_vram_mb=1000, min_free_ram_gb=8.0, max_ram_use_ratio_of_avail=0.85):
        self.max_vram_pct = max_vram_pct
        self.min_free_vram_mb = min_free_vram_mb
        self.min_free_ram_gb = min_free_ram_gb
        self.max_ram_use_ratio_of_avail = max_ram_use_ratio_of_avail

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            curr_vram_bytes = torch.cuda.memory_allocated()
            max_vram_bytes = torch.cuda.max_memory_allocated()
            free_bytes, total_bytes = torch.cuda.mem_get_info()

            curr_vram_gb = curr_vram_bytes / (1024 ** 3)
            max_vram_gb = max_vram_bytes / (1024 ** 3)
            total_vram_gb = total_bytes / (1024 ** 3)
            vram_pct = (total_bytes - free_bytes) / total_bytes * 100

            ram_info = ""
            if psutil is not None:
                sys_mem = psutil.virtual_memory()
                avail_ram_gb = sys_mem.available / (1024 ** 3)
                ram_info = f" | 💻 여유 RAM: {avail_ram_gb:.1f} GB"

            print(f"📊 [Step {state.global_step:4d}] 🟢 VRAM: {curr_vram_gb:.2f} GB / {total_vram_gb:.1f} GB ({vram_pct:.1f}%) | Peak VRAM: {max_vram_gb:.2f} GB{ram_info}")

    def on_step_end(self, args, state, control, **kwargs):
        ok, reason = check_resource_safety(
            stage=f"Step {state.global_step}",
            max_vram_pct=self.max_vram_pct,
            min_free_vram_mb=self.min_free_vram_mb,
            min_free_ram_gb=self.min_free_ram_gb,
            max_ram_use_ratio_of_avail=self.max_ram_use_ratio_of_avail,
        )
        if not ok:
            print(f"\n🚨 [RESOURCE WARNING] {reason}")
            print("🛑 OOM 강제 종료 방지를 위해 학습을 안전하게 조기 중단(Graceful Stop)합니다.")
            control.should_training_stop = True
        return control


# --- SQL 부분에만 Loss를 부여하는 DataCollator ---
class DataCollatorForSQLSFT:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        prompts = [ex["prompt"] for ex in examples]
        responses = [ex["response"] for ex in examples]
        full_texts = [p + r for p, r in zip(prompts, responses)]

        batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        labels = batch["input_ids"].clone()

        for i in range(len(examples)):
            prompt_ids = self.tokenizer.encode(prompts[i], add_special_tokens=False)
            prompt_len = min(len(prompt_ids), 1024)
            labels[i, :prompt_len] = -100

        if self.tokenizer.pad_token_id is not None:
            labels[batch["input_ids"] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


# --- 동적 상대 경로 설정 ---
DATASET_CHOICE = os.environ.get("SFT_DATASET", "spider_raw")  # "spider_raw" 또는 "augmented"
DATASET_FILES = {
    "augmented": ["verified_augmented_train.json", "augmented_train.json"],
    "spider_raw": ["train_spider.json"],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", f"qwen2.5_coder_0.5b_sql_{DATASET_CHOICE}")
EMERGENCY_DIR = os.path.join(BASE_DIR, "models", f"EMERGENCY_{DATASET_CHOICE}")


def load_schemas(tables_file: str) -> dict:
    with open(tables_file, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    db_schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        schema_str = f"Database: {db_id}\n"
        for t_idx, table_name in enumerate(db["table_names_original"]):
            cols = [
                f"{c[1]} ({db['column_types'][c_idx]})"
                for c_idx, c in enumerate(db["column_names_original"])
                if c[0] == t_idx
            ]
            schema_str += f"- Table `{table_name}`: {', '.join(cols)}\n"
        db_schemas[db_id] = schema_str.strip()
    return db_schemas


def build_dataset(data_file: str, db_schemas: dict) -> Dataset:
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        schema = db_schemas.get(item["db_id"], "")
        if len(schema) > 2500:
            schema = schema[:2500] + "... (truncated for memory safety)"

        prompt = (
            f"### DB Schema:\n{schema}\n"
            f"### User Query:\n{item['question']}\n\n"
            f"### Executable Plan:\n"
            f"1. Analyze relevant tables and columns from the user query.\n"
            f"2. Apply condition filters and necessary JOINs.\n"
            f"3. Select the required fields.\n\n"
            f"### SQL:\n"
        )
        response = f"{item['query']}<|im_end|>"
        samples.append({"prompt": prompt, "response": response})
    return Dataset.from_list(samples)


def main():
    # =====================================================================
    # [데이터 로드/모델 로딩 단계 OOM 방지] Trainer 콜백 범위 밖 구간을 수동 체크
    # =====================================================================
    ensure_resources_safe_or_raise("데이터 로드 전")

    print("📌 1. 데이터셋 로드...")
    db_schemas = load_schemas(os.path.join(DATA_DIR, "tables.json"))

    verified_file = None
    for name in DATASET_FILES[DATASET_CHOICE]:
        candidate = os.path.join(DATA_DIR, name)
        if os.path.exists(candidate):
            verified_file = candidate
            break
    if verified_file is None:
        raise FileNotFoundError(
            f"'{DATASET_CHOICE}'에 해당하는 파일을 찾을 수 없습니다: {DATASET_FILES[DATASET_CHOICE]}"
        )
    print(f"✨ 사용할 데이터셋 ({DATASET_CHOICE}): {verified_file}")

    train_dataset = build_dataset(verified_file, db_schemas)
    print(f"📊 학습에 사용될 총 샘플 수: {len(train_dataset)}개")

    print("📌 2. 모델 및 토크나이저 초기화...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =====================================================================
    # [스파이크 방지 - 보조] 비슷한 길이끼리 배치를 묶기 위한 length 컬럼 계산
    # (group_by_length=True와 함께 사용. 배치 간 padding 편차를 줄여 예측 가능성을 높임)
    # =====================================================================
    """
    def _add_length(example):
        ids = tokenizer(
            example["prompt"] + example["response"],
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )["input_ids"]
        return {"length": len(ids)}

    print("📌 2-1. 길이 분포 계산 (group_by_length용)...")
    train_dataset = train_dataset.map(_add_length, num_proc=1, desc="길이 계산 중")
    """

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    # =====================================================================
    # [스파이크 방지 - 핵심] per_device_batch=1로 고정.
    # 배치 크기가 1이면 "배치 안에 긴 시퀀스 여러 개가 우연히 몰려 padding이
    # 폭증하는" 조합 자체가 원천적으로 불가능해집니다. 유효 배치(16)는
    # gradient_accumulation_steps로만 맞춥니다.
    # =====================================================================
    EFFECTIVE_BATCH = 16
    per_device_batch = 1

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
        grad_accum = max(1, EFFECTIVE_BATCH // (world_size * per_device_batch))
        print(f"🌐 [Multi-GPU DDP] Rank {local_rank}/{world_size} 활성화 | GPU당 배치: {per_device_batch} | 누적 스텝: {grad_accum}")
    else:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        grad_accum = EFFECTIVE_BATCH // per_device_batch
        print(f"🖥️ [Single GPU / Standalone] Device: {device_map} | GPU당 배치: {per_device_batch} | 누적 스텝: {grad_accum}")

    ensure_resources_safe_or_raise("모델 로딩 전")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    base_model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    collator = DataCollatorForSQLSFT(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "tmp_checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch,   # 1로 고정 (스파이크 방지)
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        optim="adamw_torch",
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False
        # group_by_length=True,          # [스파이크 방지] 비슷한 길이끼리 배치 구성
        # length_column_name="length",
    )

    safety_callback = ResourceSafetyCallback(
        max_vram_pct=0.92,
        min_free_vram_mb=1000,
        min_free_ram_gb=8.0,
        max_ram_use_ratio_of_avail=0.85
    )
    try:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=collator,
            processing_class=tokenizer,
            args=training_args,
            callbacks=[safety_callback],
        )
    except TypeError:
        try:
            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
                args=training_args,
                callbacks=[safety_callback],
            )
        except TypeError:
            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                data_collator=collator,
                args=training_args,
                callbacks=[safety_callback],
            )

    print("🚀 3. SFT + LoRA 학습 진행...")

    # =====================================================================
    # [OOM 예외 처리] 스텝 도중 실제 OOM이 터져도 크래시로 끝내지 않고
    # LoRA 어댑터(가벼움)만이라도 긴급 저장 후 안전하게 종료
    # =====================================================================
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n🚨 [FATAL] 학습 중 CUDA OOM 발생: {e}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                os.makedirs(EMERGENCY_DIR, exist_ok=True)
                model.save_pretrained(EMERGENCY_DIR)   # LoRA 어댑터만 저장 (가볍고 빠름)
                tokenizer.save_pretrained(EMERGENCY_DIR)
                print(f"💾 긴급 LoRA 어댑터 저장 완료: {EMERGENCY_DIR}")
            except Exception as save_err:
                print(f"⚠️ 긴급 저장 실패: {save_err}")
            sys.exit(1)
        else:
            raise

    print("💾 4. LoRA 가중치 병합 및 배포용 단일 모델 저장...")
    del trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = model.merge_and_unload()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"🎉 완료! 추론/배포용 모델 저장 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()