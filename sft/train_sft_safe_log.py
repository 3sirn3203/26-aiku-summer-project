import json
import os
import sys
import time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model

try:
    import psutil
except ImportError:
    psutil = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "training_monitor.log")


def log(msg: str):
    """콘솔 출력 + 동시에 파일에도 append (타임스탬프 포함)"""
    print(msg)
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


# =========================================================================
# 공용 리소스 안전 판정 함수 (판정 로직 100% 동일, 변경 없음)
# =========================================================================
def check_resource_safety(
    stage: str,
    max_vram_pct=0.92,
    min_free_vram_mb=1000,
    min_free_ram_gb=8.0,
    max_ram_use_ratio_of_avail=0.85,
):
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
    ok, reason = check_resource_safety(stage, **kwargs)
    if not ok:
        log(f"\n🚨 [RESOURCE WARNING] {reason}")
        log("🛑 OOM 방지를 위해 진행 전 안전하게 중단합니다.")
        raise RuntimeError(reason)


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

            log(f"📊 [Step {state.global_step:4d}] 🟢 VRAM: {curr_vram_gb:.2f} GB / {total_vram_gb:.1f} GB ({vram_pct:.1f}%) | Peak VRAM: {max_vram_gb:.2f} GB{ram_info}")

    def on_step_end(self, args, state, control, **kwargs):
        ok, reason = check_resource_safety(
            stage=f"Step {state.global_step}",
            max_vram_pct=self.max_vram_pct,
            min_free_vram_mb=self.min_free_vram_mb,
            min_free_ram_gb=self.min_free_ram_gb,
            max_ram_use_ratio_of_avail=self.max_ram_use_ratio_of_avail,
        )
        if not ok:
            log(f"\n🚨 [RESOURCE WARNING] {reason}")
            log("🛑 OOM 강제 종료 방지를 위해 학습을 안전하게 조기 중단(Graceful Stop)합니다.")
            control.should_training_stop = True
        return control


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


DATASET_CHOICE = os.environ.get("SFT_DATASET", "spider_raw")
DATASET_FILES = {
    "augmented": ["verified_augmented_train.json", "augmented_train.json"],
    "spider_raw": ["train_spider.json"],
}

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
    log(f"📝 로그 파일 경로: {LOG_FILE_PATH}")
    ensure_resources_safe_or_raise("데이터 로드 전")

    log("📌 1. 데이터셋 로드...")
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
    log(f"✨ 사용할 데이터셋 ({DATASET_CHOICE}): {verified_file}")

    train_dataset = build_dataset(verified_file, db_schemas)
    log(f"📊 학습에 사용될 총 샘플 수: {len(train_dataset)}개")

    log("📌 2. 모델 및 토크나이저 초기화...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    """
    def _add_length(example):
        ids = tokenizer(
            example["prompt"] + example["response"],
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )["input_ids"]
        return {"length": len(ids)}

    log("📌 2-1. 길이 분포 계산 (group_by_length용)...")
    train_dataset = train_dataset.map(_add_length, num_proc=1, desc="길이 계산 중")
    """
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    EFFECTIVE_BATCH = 16
    per_device_batch = 1

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
        grad_accum = max(1, EFFECTIVE_BATCH // (world_size * per_device_batch))
        log(f"🌐 [Multi-GPU DDP] Rank {local_rank}/{world_size} 활성화 | GPU당 배치: {per_device_batch} | 누적 스텝: {grad_accum}")
    else:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        grad_accum = EFFECTIVE_BATCH // per_device_batch
        log(f"🖥️ [Single GPU / Standalone] Device: {device_map} | GPU당 배치: {per_device_batch} | 누적 스텝: {grad_accum}")

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
        per_device_train_batch_size=per_device_batch,
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
        remove_unused_columns=False,
        # group_by_length=True,
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

    log("🚀 3. SFT + LoRA 학습 진행...")

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"\n🚨 [FATAL] 학습 중 CUDA OOM 발생: {e}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                os.makedirs(EMERGENCY_DIR, exist_ok=True)
                model.save_pretrained(EMERGENCY_DIR)
                tokenizer.save_pretrained(EMERGENCY_DIR)
                log(f"💾 긴급 LoRA 어댑터 저장 완료: {EMERGENCY_DIR}")
            except Exception as save_err:
                log(f"⚠️ 긴급 저장 실패: {save_err}")
            sys.exit(1)
        else:
            raise

    log("💾 4. LoRA 가중치 병합 및 배포용 단일 모델 저장...")
    del trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = model.merge_and_unload()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"🎉 완료! 추론/배포용 모델 저장 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()