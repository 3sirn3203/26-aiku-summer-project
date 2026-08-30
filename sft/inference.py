import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen2.5_0.5b_sql_coder")

class SQLGenerator:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        self.stop_token_id = self.tokenizer.encode("<|im_end|>")[0]

    def generate(self, schema: str, query: str) -> str:
        prompt = (
            f"### DB Schema:\n{schema.strip()}\n"
            f"### User Query:\n{query.strip()}\n\n"
            f"### Executable Plan:\n"
            f"1. Analyze relevant tables and columns from the user query.\n"
            f"2. Apply condition filters and necessary JOINs.\n"
            f"3. Select the required fields.\n\n"
            f"### SQL:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.stop_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_sql = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return self._clean_sql(raw_sql)

    @staticmethod
    def _clean_sql(sql_str: str) -> str:
        cleaned = re.sub(r"^```sql\s*", "", sql_str, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        return cleaned.strip()

if __name__ == "__main__":
    generator = SQLGenerator()

    sample_schema = "Database: cinema\n- Table `movie`: Movie_ID (int), Title (text), Director (text), Year (int)"
    sample_query = "Find all movies directed by Christopher Nolan released after 2010."

    result_sql = generator.generate(sample_schema, sample_query)
    print("▶ 테스트 생성 SQL:\n", result_sql)