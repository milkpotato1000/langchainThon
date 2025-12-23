# 🤖 Job Helper Bot - AI 기반 취업 도우미 RAG 시스템

> **채용 공고 분석부터 자소서 작성, 모의 면접까지 한 번에!**  
> RAG(Retrieval-Augmented Generation) 기술을 활용한 올인원 취업 준비 솔루션입니다.

---

## 📌 1. 프로젝트 개요
**Job Helper Bot**은 취업 준비생들이 겪는 정보 비대칭과 준비 과정의 번거로움을 해결하기 위해 개발되었습니다.  
채용 공고 URL만 입력하면 **직무 분석, 기업 정보 탐색, 맞춤형 예상 질문 생성, 모의 면접 코칭**까지 전 과정을 AI가 도와줍니다.

### 💡 기획 의도
- 🧩 **복잡한 공고 분석**: 수많은 채용 사이트(원티드, 사람인, 잡코리아)의 긴 공고를 핵심만 요약
- 🔍 **기업 리서치 자동화**: 지원 기업의 비전, 인재상, 최신 뉴스를 찾느라 허비되는 시간 단축
- 🎯 **맞춤형 면접 대비**: 내 이력서와 공고를 매칭하여 "진짜 나올 법한" 질문과 답변을 미리 준비

---

## ✨ 2. 주요 기능

### 1️⃣ 채용 공고 자동 분석 (Parsing & Summary)
- **멀티 플랫폼 지원**: 원티드, 사람인, 잡코리아 등 다양한 채용 사이트의 URL 지원
- **스마트 크롤링**: Selenium을 활용해 동적 웹페이지(Dynamic Rendering)의 정보도 완벽하게 수집
- **핵심 정보 구조화**: LLM이 공고 내용을 분석하여 `주요 업무`, `자격 요건`, `우대 사항`으로 깔끔하게 정리

### 2️⃣ 기업 심층 분석 리포트 (Company Intel)
- **공식 홈페이지 크롤링**: 기업 URL 입력 시 비전, 미션, 인재상 등 "기업 문화" 정보를 자동 추출
- **실시간 뉴스 모니터링**: Google News RSS와 연동하여 해당 기업 관련 최신 기사 TOP 5 제공

### 3️⃣ RAG 기반 이력서 매칭 (Resume Indexing)
- **모든 포맷 지원**: PDF, DOCX, TXT 파일 등 다양한 형식의 이력서 파싱
- **벡터 검색(Vector Search)**: 이력서 내용을 청킹(Chunking) 및 임베딩하여, 질문과 가장 연관성 높은 경험을 찾아냄

### 4️⃣ AI 면접 코치 (Interactive Mock Interview)
- **킬러 문항 생성**: "이 공고의 이 요건을 당신의 이력서의 저 경험으로 어떻게 증명하겠습니까?"와 같은 구체적 질문 생성
- **STAR 답변 가이드**: 상황(S)-과제(T)-행동(A)-성과(R) 프레임워크에 맞춘 답변 초안 작성
- **정량적 피드백**: 사용자 답변을 `문제해결력`, `데이터지표`, `실행력`, `협업`, `고객관점` 등 5가지 지표로 채점하고 피드백 제공

---

## 🛠️ 3. 기술 스택 (Tech Stack)

| 분류 | 기술 | 비고 |
| :-- | :-- | :-- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white) | |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | 직관적인 대화형 UI 구성 |
| **LLM & AI** | ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white) | GPT-4o, Text-Embedding-3 |
| **Crawling** | ![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat&logo=selenium&logoColor=white) | Chrome Driver (Headless Mode) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-DataFrame-150458?style=flat&logo=pandas&logoColor=white) | 데이터 정형화 및 관리 |

---

## ⚙️ 4. 시스템 흐름도 (Workflow)

```mermaid
graph TD
    A[사용자 입력 (채용공고 URL)] --> B(Selenium Crawler);
    B --> C{데이터 정제};
    C -->|LLM 구조화| D[공고 요약 정보];
    C -->|기업 홈페이지/뉴스| E[기업 메타 정보];
    
    F[이력서 업로드] --> G(Text Parser);
    G --> H(OpenAI Embeddings);
    H --> I[Vector Store (Memory)];
    
    D & I --> J(LLM - 면접 질문 생성);
    J --> K[사용자 답변 입력];
    K --> L(LLM - 코칭/평가);
    L --> M[최종 피드백 결과];
```

---

## 🚀 5. 시작하기 (Getting Started)

### 사전 요구사항
- Chrome Browser (최신 버전)
- OpenAI API Key

### 설치 및 실행

**1. 저장소 클론 (Clone)**
```bash
git clone https://github.com/milkpotato1000/langchainThon.git
cd langchainThon
```

**2. 패키지 설치**
```bash
pip install streamlit openai selenium beautifulsoup4 pandas numpy html2text pypdf docx2txt
```

**3. 앱 실행**
```bash
streamlit run "Job_Helper_Bot(ver.3).py"
```

---

## 📸 6. 실행 화면 (Preview)
| **1. 메인 화면 & 공고 분석** | **2. 면접 질문 생성 & 답변 피드백** |
| :--: | :--: |
| *(스크린샷 영역)* | *(스크린샷 영역)* |
