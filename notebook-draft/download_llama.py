from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 모델 및 토크나이저 이름
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"

# 로컬 저장 경로
local_dir = "./model"

# 모델 및 토크나이저 다운로드 및 저장
def download_model():
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # 모델 다운로드
    print("모델 다운로드 중...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(local_dir)
    
    # 토크나이저 다운로드
    print("토크나이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)
    
    print(f"모델이 '{local_dir}' 폴더에 성공적으로 다운로드되었습니다.")

if __name__ == "__main__":
    download_model()