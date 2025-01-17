# Third-Party Licenses

이 프로젝트에서 사용한 모델 및 데이터셋의 라이선스 정보입니다.

---

## Models

### Bllossom/llama-3.2-Korean-Bllossom-AICA-5B

| 항목 | 내용 |
|------|------|
| **License** | LLaMA 3.2 Community License |
| **Source** | https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-AICA-5B |
| **Description** | Meta의 LLaMA-3.2를 기반으로 한국어에 특화되어 파인튜닝된 모델 |

**주요 라이선스 조건:**
- 비상업적/상업적 사용 가능
- 월간 활성 사용자(MAU) 7억 명 이상인 서비스에서 사용 시 Meta의 별도 허가 필요
- 모델 출력물을 다른 LLM 학습에 사용 금지

**학습 데이터:**
- Hugging Face 공개 한국어 LLM 사전학습 데이터
- AI-Hub, KISTI AI 데이터
- 한국어 시각-언어 관련 학습 데이터
- 자체 제작 한국어 시각-언어 Instruction Tuning 데이터

---

### noahkim/KoT5_news_summarization

| 항목 | 내용 |
|------|------|
| **License** | Unknown (원본 저장소 삭제됨) |
| **Status** | The original repository has been removed |
| **Description** | 한국어 뉴스 요약을 위해 파인튜닝된 T5 모델 |

**참고:** 이 모델은 `notebook_study/` 실험 과정에서만 사용되었으며, 최종 프로덕션 코드에는 포함되지 않습니다.

---

## Datasets

### daekeun-ml/naver-news-summarization-ko

| 항목 | 내용 |
|------|------|
| **License** | Apache License 2.0 |
| **Source** | https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko |
| **Description** | 네이버 뉴스 기사와 요약문 쌍으로 구성된 한국어 요약 데이터셋 |

**데이터셋 구성:**
- Train: 22,194 samples
- Validation: 2,466 samples
- Test: 2,740 samples

---

## Libraries

이 프로젝트는 다음 오픈소스 라이브러리를 사용합니다:

| Library | License |
|---------|---------|
| PyTorch | BSD-3-Clause |
| Transformers (Hugging Face) | Apache-2.0 |
| pandas | BSD-3-Clause |
| datasets | Apache-2.0 |
| rouge-score | Apache-2.0 |
| bert-score | MIT |

---

## License Compatibility

이 프로젝트의 소스 코드는 **Apache License 2.0**으로 배포됩니다.

단, 모델을 사용할 때는 **LLaMA 3.2 Community License** 조건을 준수해야 합니다.
자세한 내용은 [Meta의 LLaMA 라이선스](https://llama.meta.com/llama3_2/license/)를 참조하세요.
