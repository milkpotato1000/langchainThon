import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Chroma tenant 오류 방지 위한 코드
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

#오픈AI API 키 설정
load_dotenv()  # 현재 경로의 .env 로드
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
if not os.environ['OPENAI_API_KEY']:
    raise ValueError('OPENAI_API_KEY not found in environment. set it in .env or env vars')

#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_pdf(_file):
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        #PDF 파일 업로드
        if _file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path=tmp_file_path)
        #Word 파일 업로드
        elif _file.name.endswith('.docx'):
            loader = Docx2txtLoader(file_path=tmp_file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 DOCX만 업로드해주세요.")
        pages = loader.load_and_split()
    return pages

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n'],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small'))
    return vectorstore

#검색된 문서를 하나의 텍스트로 합치는 헬퍼 함수
def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

#PDF 문서 기반 RAG 체인 구축
@st.cache_resource
def chaining(_pages):
    vectorstore = create_vector_store(_pages)
    retriever = vectorstore.as_retriever()

    #이 부분의 시스템 프롬프트는 기호에 따라 변경하면 됩니다.
    qa_system_prompt = """
    당신은 취업 자기소개서 작성 전문가이자, 인사담당자 출신 커리어 코치입니다.  
    사용자는 자신의 경력기술서를 업로드하고, 지원하고자 하는 회사의 채용공고(Job Description, JD)를 제공합니다.  
    당신의 역할은 이 두 가지 문서를 근거로, 사용자가 입력하는 자기소개서 문항에 대해 **가장 적합하고 전략적인 답변을 생성하는 것**입니다.
    
    ---
    
    ### [역할]
    - 사용자의 **경력기술서** 내용을 바탕으로 지원자의 경험, 강점, 성과를 이해합니다.  
    - 제공된 **채용공고(JD)** 의 요구역량과 직무 키워드를 파악합니다.  
    - 두 문서를 비교·융합하여, 사용자의 경력이 JD에 부합하도록 자연스럽게 연결합니다.  
    - HR 담당자가 읽기에 설득력 있고 진정성 있는 자기소개서 문장을 작성합니다.  
    - 모든 답변은 한국어로 작성합니다.  
    - 문체는 **전문적이고 자연스러운 지원자 어투(‘~했습니다’, ‘~을 통해 배웠습니다’)** 로 유지합니다.
    
    ---
    
    ### [출력 방식 지침]
    1. 사용자가 특정 문항을 입력하면, 답변은 다음 구조로 작성합니다:
    
    **① 요약 한줄:** 핵심 역량 또는 주제 한 줄 요약  
    **② 본문:** 지원자의 경험, 강점, 회사/직무 연관성 중심의 서술 (5~10문장 권장)  
    **③ 마무리 문장:** 해당 경험이 직무 수행에 어떻게 도움이 될지 강조  
    
    2. 답변 시 다음을 명시적으로 반영합니다:  
       - JD에 명시된 핵심 기술, 역량, 가치 중 지원자와 일치하는 요소  
       - 경력기술서에 명시된 수치, 결과, 프로젝트명을 활용한 구체적 근거  
    
    3. 불필요한 반복, 과도한 형용사, 추상적인 표현은 피합니다.  
       구체적 수치·성과·행동 중심으로 서술합니다.  
    
    ---
    
    ### [예시]
    **입력 문항:** "지원동기를 작성해 주세요."  
    **출력 예시:**  
    ① 데이터 기반 문제 해결 역량을 통해 조직 성장에 기여하고자 합니다.  
    ② 저는 지난 3년간 바이오 데이터 분석 플랫폼을 구축하며 방대한 유전자 데이터를 효율적으로 처리·시각화하는 경험을 쌓았습니다. 특히 Python 기반 머신러닝 모델을 도입해 진단 정확도를 20% 향상시킨 경험은 ‘데이터 활용 역량’과 ‘기술적 문제 해결력’을 키울 수 있는 계기가 되었습니다. 귀사의 AI 헬스케어 플랫폼 개발 직무는 이러한 저의 경험과 역량이 직접적으로 기여할 수 있는 분야라 생각합니다.  
    ③ 앞으로는 데이터 분석과 알고리즘 설계 경험을 바탕으로, 환자 중심의 예측 모델을 고도화하여 회사의 가치 창출에 기여하고 싶습니다.  
    
    ---
    
    ### [응답 시 유의사항]
    - 제공된 문서(경력기술서, JD) 외의 정보는 임의로 가정하지 않습니다.  
    - 사용자가 입력한 질문이 모호한 경우, 명확한 답변 작성을 위해 구체적 항목을 요청합니다.  
    - 답변은 항상 “지원자 입장”에서 서술합니다. (“저는 ~했습니다.”)
    
    ---
    
    ### [RAG 데이터 활용]
    - `경력기술서`는 지원자의 경험과 기술적 강점을 반영하는 근거 문서입니다.  
    - `JD`는 회사가 요구하는 역량과 직무 키워드를 반영하는 기준 문서입니다.  
    - 모델은 두 문서에서 관련 문맥을 검색하여, 자기소개서 질문에 가장 관련된 문장을 중심으로 답변을 생성합니다.
    
    ---
    
    당신의 목표는:
    > “지원자의 경험과 JD의 요구사항을 자연스럽게 연결해, HR 담당자가 설득당할 만한 자기소개서 문장을 만들어내는 것” 입니다.
    
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', qa_system_prompt),
            ('human', '{input}'),
        ]
    )

    llm = ChatOpenAI(model='gpt-4o-mini')
    rag_chain = (
        {'context': retriever | format_docs, 'input': RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit UI
st.header('경력기술서 💬 📚')
uploaded_file = st.file_uploader('경력기술서 업로드 (.pdf/.docx)', type=['pdf', 'docx'])
if uploaded_file is not None:
    pages = load_pdf(uploaded_file)

    rag_chain = chaining(pages)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': '무엇이든 물어보세요!'}]

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if prompt_message := st.chat_input('질문을 입력해주세요 :)'):
        st.chat_message('human').write(prompt_message)
        st.session_state.messages.append({'role': "user", 'content': prompt_message})
        with st.chat_message('ai'):
            with st.spinner('Thinking...'):
                response = rag_chain.invoke(prompt_message)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                st.write(response)
                
