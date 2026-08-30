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

# --- GPU VRAM & System RAM (멀티스레드 포함) 안전 감시 및 실시간 모니터링 콜백 ---
class ResourceSafetyCallback(TrainerCallback):
    def __init__(self, max_vram_pct=0.92, min_free_vram_mb=1000, min_free_ram_gb=8.0, max_ram_use_ratio_of_avail=0.85):
        self.max_vram_pct = max_vram_pct
        self.min_free_vram_mb = min_free_vram_mb
        self.min_free_ram_gb = min_free_ram_gb
        self.max_ram_use_ratio_of_avail = max_ram_use_ratio_of_avail

    def on_log(self, args, state, control, logs=None, **kwargs):
        """매 logging_steps (예: 20스텝) 마다 실시간 자원 상태를 콘솔에 출력하여 눈으로 확인하도록 보장"""
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
        # 1. GPU VRAM 실시간 모니터링 및 안전 검사
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            vram_pct = used_bytes / total_bytes
            free_mb = free_bytes / (1024 * 1024)

            if vram_pct > self.max_vram_pct or free_mb < self.min_free_vram_mb:
                print(f"\n🚨 [RESOURCE WARNING] GPU VRAM 위험수치 도달!")
                print(f"   현재 사용률: {vram_pct * 100:.1f}% | 남은 VRAM: {free_mb:.1f} MB")
                print("🛑 OOM 강제 종료 방지를 위해 학습을 안전하게 조기 중단(Graceful Stop)합니다.")
                control.should_training_stop = True
                return control

        # 2. System RAM (메인 프로세스 + 모든 멀티스레드/자식 프로세스 합산) 합리적 감시
        if psutil is not None:
            try:
                curr_proc = psutil.Process()
                children = curr_proc.children(recursive=True)
                my_total_rss_bytes = curr_proc.memory_info().rss + sum(c.memory_info().rss for c in children if c.is_running())
                my_total_rss_gb = my_total_rss_bytes / (1024 ** 3)

                sys_mem = psutil.virtual_memory()
                avail_ram_gb = sys_mem.available / (1024 ** 3)  # 실제 사용 가능한 여유 RAM

                if avail_ram_gb < self.min_free_ram_gb:
                    print(f"\n🚨 [RESOURCE WARNING] 시스템 여유 RAM 부족! (남은 여유 RAM: {avail_ram_gb:.2f} GB < 최소 보장: {self.min_free_ram_gb} GB)")
                    print("🛑 OS OOM Killer 동작 방지를 위해 학습을 안전하게 조기 중단(Graceful Stop)합니다.")
                    control.should_training_stop = True
                    return control

                if my_total_rss_gb > 4.0 and my_total_rss_gb > (avail_ram_gb * self.max_ram_use_ratio_of_avail):
                    print(f"\n🚨 [RESOURCE WARNING] 내 프로그램 메모리가 남은 여유 RAM의 {self.max_ram_use_ratio_of_avail*100:.0f}%를 초과!")
                    print(f"   내 프로그램 점유: {my_total_rss_gb:.2f} GB | 현재 여유 RAM: {avail_ram_gb:.2f} GB")
                    print("🛑 메모리 폭증(Memory Leak) 방지를 위해 학습을 안전하게 조기 중단(Graceful Stop)합니다.")
                    control.should_training_stop = True
                    return control

            except Exception:
                pass

        return control

# --- SQL 부분에만 Loss를 부여하는 O(1) 슬라이싱 완벽 호환 DataCollator ---
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
            max_length=1024,                     # 12GB VRAM 완전 안전 상한선
            return_tensors="pt"
        )
        labels = batch["input_ids"].clone()

        for i in range(len(examples)):
            # prompt 부분의 토큰 길이 계산 (선형 탐색 완전 제거: O(1) 슬라이싱)
            prompt_ids = self.tokenizer.encode(prompts[i], add_special_tokens=False)
            prompt_len = min(len(prompt_ids), 1024)
            labels[i, :prompt_len] = -100        # Prompt 영역 Loss 마스킹

        # 패딩 토큰 영역도 Loss 마스킹 (-100)
        if self.tokenizer.pad_token_id is not None:
            labels[batch["input_ids"] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

# --- 동적 상대 경로 설정 (현재 스크립트 파일 위치 기준) ---
DATASET_CHOICE = os.environ.get("SFT_DATASET", "spider_raw") # "spider_raw" 또는 "augmented"
DATASET_FILES = {
    "augmented": ["verified_augmented_train.json", "augmented_train.json"],
    "spider_raw": ["train_spider.json"],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", f"qwen2.5_coder_0.5b_sql_{DATASET_CHOICE}")

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
        # 스키마가 비정상적으로 길 경우 최대 2,500자로 안전 자르기 (OOM 1차 사전에 방지)
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

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    # Multi-GPU DDP 및 단일 GPU 실행 대응 분기
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
        per_device_batch = 2
        grad_accum = max(1, 16 // (world_size * per_device_batch))
        print(f"🌐 [Multi-GPU DDP] Rank {local_rank}/{world_size} 활성화 | GPU당 배치: {per_device_batch} | 누적 스텝: {grad_accum}")
    else:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        per_device_batch = 2
        grad_accum = 8
        print(f"🖥️ [Single GPU / Standalone] Device: {device_map} | GPU당 배치: {per_device_batch} | 누적 스텝: {grad_accum}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    # Gradient Checkpointing 사용 시 PEFT/LoRA 입력 그래디언트 연결 필수 함수
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

    # O(1) 슬라이싱 완벽 호환 콜레이터 생성
    collator = DataCollatorForSQLSFT(tokenizer=tokenizer)

    # --- 속도 및 메모리 최적화 학습 설정 ---
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "tmp_checkpoints"),
        num_train_epochs=1,                     # 1 에폭 (70,227개 완독)
        per_device_train_batch_size=per_device_batch, # 마이크로 배치 2
        gradient_accumulation_steps=grad_accum, # 자동 분기 (유효 배치=16 유지)
        gradient_checkpointing=True,            # VRAM 사용 최적화
        gradient_checkpointing_kwargs={"use_reentrant": False}, # PyTorch DDP + LoRA 호환 옵션
        ddp_find_unused_parameters=False,       # DDP 충돌 방지
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,                       # 최신/구버전 공통 호환
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,                        # 1000 스텝마다 중간 체크포인트 자동 저장
        save_total_limit=2,                     # 최근 2개 체크포인트 유지
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        optim="adamw_torch",
        dataloader_num_workers=2,               # CPU 코어 독점 방지
        report_to="none",                       # 외부 로깅 방지
        remove_unused_columns=False,            # text 컬럼 유지
    )

    # 모든 Transformers 버전 호환 Trainer 생성
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

    print("🚀 3. SFT + LoRA 학습 진행 (1 에폭 / 속도 및 멀티 GPU 최적화 버전)...")
    trainer.train()

    print("💾 4. LoRA 가중치 병합 및 배포용 단일 모델 저장...")
    # OOM 방지를 위해 Trainer 및 CUDA 메모리 청소
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