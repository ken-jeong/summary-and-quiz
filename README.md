# LLaMA로 만드는 나만의 학습 도우미: 한국어 텍스트 요약 & 퀴즈 생성 AI

> 1학년 겨울학기 비교과 프로젝트 (2025.01)

---

## 들어가며

대학생이라면 누구나 공감할 것이다. 시험 기간이 다가오면 산더미 같은 강의 자료와 씨름하며 "이걸 언제 다 정리하지?"라는 생각에 막막해지는 순간이.

2006년에 출시된 플래시카드 암기 앱 **Anki**는 이런 문제를 해결하려는 시도였다. 하지만 결정적인 한계가 있었다. **사용자가 직접 내용을 요약하고, 정해진 형식에 맞춰 카드를 만들어야 한다는 점**이다. 정작 시간이 없어서 앱을 쓰려는 건데, 카드 만드는 데 시간을 써야 한다니 본말이 전도된 느낌이었다.

그래서 생각했다. **AI가 요약도 해주고, 퀴즈도 만들어주면 되지 않을까?**

이 프로젝트는 그렇게 시작됐다.

---

## 무엇을 만들었나

**긴 한국어 텍스트를 넣으면, 요약본과 주관식 퀴즈가 자동으로 생성되는 AI 학습 도우미**를 만들었다.

```
[입력] 뉴스 기사, 강의 노트, 교재 내용 등 긴 텍스트
         ↓
[처리] LLaMA 3.2 Korean 모델
         ↓
[출력] 핵심 요약 + 복습용 퀴즈(문제 & 정답)
```

### 실제 동작 예시

```python
from src import SummaryQuizGenerator

text = """
엔비디아가 차세대 게이밍 GPU '지포스 RTX 50시리즈'를 내놓으며
PC용 AI칩 시장을 겨냥한다. RTX 5090의 경우 초당 데이터 전송량이
1.8TB로, 이전 모델인 RTX 4090보다 두 배 향상됐다...
"""

with SummaryQuizGenerator() as ai:
    result = ai.summarize_and_quiz(text)

print(result["summary"])
# → 엔비디아가 블랙웰 아키텍처 기반 RTX 50 시리즈를 발표했다.
#   최상위 모델 5090은 전작 대비 2배 빠른 전송 속도를 자랑한다.

print(result["quiz"])
# → 문제: RTX 5090의 초당 데이터 전송량은 얼마인가?
#   정답: 1.8TB
```

---

## 왜 LLaMA 3.2 Korean을 선택했나

처음에는 OpenAI API를 고려했다. 하지만 몇 가지 이유로 로컬 모델을 선택했다.

| 고려 사항 | OpenAI API | 로컬 LLaMA |
|-----------|------------|------------|
| 비용 | 사용량 비례 과금 | 초기 셋업 후 무료 |
| 한국어 성능 | 좋음 | **Bllossom 파인튜닝으로 매우 좋음** |
| 프라이버시 | 데이터가 외부 서버로 | 내 컴퓨터에서 처리 |
| 오프라인 | 불가능 | **가능** |

특히 **[Bllossom/llama-3.2-Korean-Bllossom-AICA-5B](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-AICA-5B)** 모델은 한국어에 특화되어 파인튜닝된 모델이라 한국어 요약 품질이 인상적이었다.

---

## 개발 환경

| 구분 | 사양 |
|------|------|
| GPU | NVIDIA GeForce GTX 1070 (8GB VRAM) |
| CUDA / cuDNN | 12.4 / 9.6 |
| OS | Windows 10 |
| Python | 3.12.8 (Miniconda) |
| Framework | PyTorch 2.5.1 + cu124 |
| IDE | VS Code + Jupyter Notebook |

5B 파라미터 모델을 8GB VRAM에서 돌리기 위해 `bfloat16` 정밀도를 사용했다. 풀 정밀도(fp32) 대비 메모리 사용량을 절반으로 줄이면서도 성능 저하는 거의 없었다.

*로컬 환경의 하드웨어 스펙 부족으로 학교 실습실보다 10배 느린 성능을 기록하게 되었는데, 이 프로젝트를 계기로 원활한 개발을 위해 최신 그래픽카드를 도입하기로 결심했다. ㅠㅠ*

---

## 프로젝트 구조

```
summary-and-quiz/
│
├── src/                      # 핵심 소스 코드
│   ├── config.py            # 설정값 관리 (모델명, 토큰 수 등)
│   ├── generator.py         # 요약/퀴즈 생성 로직
│   └── cli.py               # 터미널에서 사용하기
│
├── tests/                    # 단위 테스트
├── notebook_study/           # 모델 학습 & 실험 기록
├── notebook_draft/           # 프로토타입 코드
├── main.ipynb               # 데모 노트북
└── requirements.txt
```

---

## 설치 및 사용법

### 1. 설치

```bash
git clone https://github.com/your-repo/summary-and-quiz.git
cd summary-and-quiz
pip install -r requirements.txt
```

### 2. Python에서 사용

```python
from src import SummaryQuizGenerator, Config

# 기본 설정으로 사용
with SummaryQuizGenerator() as gen:
    summary = gen.summarize("긴 텍스트...")
    quiz = gen.create_quiz(summary)

# 커스텀 설정
config = Config(
    temperature=0.3,      # 높을수록 창의적인 출력
    max_new_tokens=512    # 더 긴 출력 허용
)
gen = SummaryQuizGenerator(config)
```

### 3. 터미널에서 사용

```bash
# 파일에서 읽어서 요약
python -m src.cli --mode summary --input lecture.txt

# 요약 + 퀴즈 동시 생성
python -m src.cli --mode both --input article.txt --output result.txt

# 상세 로그 출력
python -m src.cli --mode both --input article.txt --verbose
```

---

## 개발 과정에서 배운 것들

### 1. Temperature의 미묘한 차이

처음에는 temperature를 0.7로 설정했다가 요약 결과가 매번 달라지는 문제가 있었다. 학습 도구로서의 **일관성**이 중요하다고 판단해 0.1로 낮췄다.

```python
# 창의적 글쓰기: temperature = 0.7~1.0
# 요약/번역 등 일관성 필요: temperature = 0.1~0.3
```

### 2. 토큰 패턴 추출의 함정

LLaMA 모델의 출력에서 실제 응답만 추출하려면 특수 토큰을 파싱해야 했다.

```python
# 모델 출력 형식
<|start_header_id|>assistant<|end_header_id|>실제 응답<|eot_id|>
```

처음에는 이 패턴만 있으면 된다고 생각했는데, 가끔 형식이 미묘하게 달라지는 경우가 있었다. **폴백 패턴을 추가해서 안정성을 높였다.**

### 3. 메모리 관리의 중요성

5B 모델은 VRAM을 거의 8GB 가까이 사용한다. 모델을 로드한 채로 다른 작업을 하면 OOM 에러가 발생했다. Context Manager 패턴을 적용해서 사용 후 자동으로 메모리를 해제하도록 했다.

```python
# 권장: with 문 사용시 자동 정리
with SummaryQuizGenerator() as gen:
    result = gen.summarize(text)
# 여기서 자동으로 GPU 메모리 해제
```

---

## 실험 노트북 가이드

`notebook_study/` 폴더에는 개발 과정에서 실험한 코드들이 있다.

| 노트북 | 내용 |
|--------|------|
| `0_download_model.ipynb` | HuggingFace에서 모델 다운로드 |
| `1_model_test.ipynb` | 모델 추론 테스트 |
| `2_download_dataset.ipynb` | 네이버 뉴스 요약 데이터셋 다운로드 |
| `3_eda.ipynb` | 데이터 탐색 (문서 길이 분포, 압축률 등) |
| `4_train.ipynb` | T5 모델 파인튜닝 실습 |
| `5_visualization.ipynb` | 학습 결과 시각화 |
| `6_evaluate.ipynb` | ROUGE, BERTScore 평가 |

파인튜닝 실습에서는 `noahkim/KoT5_news_summarization` 모델과 `daekeun-ml/naver-news-summarization-ko` 데이터셋을 사용했다.

---

## 개발 일정

| 기간 | 작업 내용 |
|------|-----------|
| 01/13 ~ 15 | 주제 선정 및 기획 |
| 01/16 ~ 17 | 모델 비교 및 선정 |
| 01/18 ~ 20 | 파인튜닝 실습 |
| 01/21 ~ 22 | 코드 구현 및 테스트 |
| 01/23 ~ 24 | 정리 및 발표 |

---

## 앞으로 해볼 것들

- [ ] **Gradio/Streamlit 웹 UI** 추가
- [ ] **4-bit 양자화**로 더 가벼운 모델 만들기
- [ ] 퀴즈 난이도 조절 기능
- [ ] PDF 파일 직접 입력 지원
- [ ] 생성된 퀴즈를 Anki 형식으로 내보내기

---

## 마치며

2주라는 짧은 시간이었지만, LLM을 로컬에서 돌려보고 실제 문제 해결에 적용해본 의미 있는 경험이었다. 특히 한국어 특화 모델의 품질이 생각보다 좋아서 놀랐다.

앞으로 이 프로젝트를 발전시켜서 실제로 시험 기간에 써볼 수 있는 수준까지 만들어보고 싶다.

---

## License

Apache License 2.0
