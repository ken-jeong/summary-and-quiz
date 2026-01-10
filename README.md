# 한국어 텍스트 요약 및 퀴즈 생성 AI
LLaMA-3.2를 활용한 학습 보조 프로그램
(1학년 겨울학기 비교과 프로젝트)

## 1. 계획(Planning)
### 🛠 Development Environment
| Category        | Specification                              |
| :-------------- | :----------------------------------------- |
| **GPU**         | NVIDIA GeForce GTX 1070 (8GB VRAM)         |
| **CUDA**        | 12.4                                       |
| **cuDNN**       | 9.6                                        |
| **OS**          | Windows 10 (Native)                        |
| **Language**    | Python 3.12.8                              |
| **Environment** | Miniconda                                  |
| **Framework**   | PyTorch 2.5.1 + cu124                      |
| **IDE**         | Visual Studio Code (with Jupyter Notebook) |

### 📅 Schedule
| Date             | Workflow                        |
| :--------------- | :------------------------------ |
| 2025. 01. 13-15. | Topics & Plans                  |
| 2025. 01. 16-17. | Models & Evaluation             |
| 2025. 01. 18-20. | Fine-tuning & Evaluation        |
| 2025. 01. 21-22. | Program Implementation & Testing |
| 2025. 01. 23-24. | Portfolio Presentation          |

### Topics
- **문제 정의**
	- 2006년에 개발된 플래식카드식 암기 프로그램 anki에서 아이디어를 얻음.
	- 이 프로그램은 사용자가 직접 학습한 내용을 요약해서 정리하고,
	- 이를 프로그램이 요구하는 형식에 맞춰 다시 작성해서 사용해야 함.

- **개선 방향**
	- 현재는 AI 기술이 각광 받고 있는 시대인 만큼,
	- 학습 내용의 요약과 퀴즈 생성에 AI를 활용해보고자 함.

- **주제**
	- 한국어 텍스트 요약 및 퀴즈 생성 AI

- **목표**
	- 긴 한국어 텍스트 자료를 요약하여 학습 시간을 절감
	- 요약 내용을 기반으로 주관식 퀴즈를 자동으로 생성

## 2. 설계(Design) 및 구현(Implementation)
- `main.ipynb`: 프로젝트 완성 파일입니다.

### 2.1. notebook_study 폴더
|             파일             |           설명            |
| :------------------------: | :---------------------: |
|  `0_download_model.ipynb`  |  파인튜닝 실습에 사용한 모델을 다운로드  |
|    `1_model_test.ipynb`    |  로컬에 설치된 모델의 사용법을 테스트   |
| `2_download_dataset.ipynb` | 파인튜닝 실습에 사용한 데이터셋을 다운로드 |
|       `3_eda.ipynb`        |    로컬에 설치된 데이터셋 EDA     |
|      `4_train.ipynb`       |         파인튜닝 실습         |
|  `5_visualization.ipynb`   |         데이터 시각화         |
|     `6_evaluate.ipynb`     |        요약 모델 평가         |
- 파인튜닝 실습을 위해 테스트한 코드들입니다.

### 2.2. notebook_draft 폴더
|         파일          |       설명       |
| :-----------------: | :------------: |
| `download_llama.py` | 라마 한국어 모델 다운로드 |
|     `v1.ipynb`      |    모델 활용 1안    |
|     `v2.ipynb`      |    모델 활용 2안    |
|    `test.ipynb`     |  main 파일의 초안   |
- main.py 파일 구현을 위해 테스트한 코드들입니다.
