import os
from dotenv import load_dotenv
import tempfile
from chromadb import Client
from chromadb.config import Settings
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import MultiRetrievalQAChain


#오픈AI API 키 설정
load_dotenv()  # 현재 경로의 .env 로드
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
if not os.environ['OPENAI_API_KEY']:
    raise ValueError('OPENAI_API_KEY not found in environment. set it in .env or env vars')

@st.cache_resource #cache_resource로 한번 실행한 결과 캐싱해두기
# CV 불러오기 (PDF/Word)
def load_cv(_file):
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        
    #PDF 파일 업로드
    if _file.name.endswith('.pdf'):
        loader = PyPDFLoader(file_path=tmp_file_path)
    
    #Word 파일 업로드
    elif _file.name.endswith('.docx'):
        loader = Docx2txtLoader(file_path=tmp_file_path)
    
    # 파일 형식이 틀릴경우 에러 메세지 출력
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 DOCX만 업로드해주세요.")
    
    pages = loader.load()
    return pages

@st.cache_resource
# JD 불러오기 (URL)
def load_jd(_url: str):
    import warnings
    from urllib3.exceptions import InsecureRequestWarning
    warnings.simplefilter('ignore', InsecureRequestWarning)
    
    loader = WebBaseLoader(_url)
    loader.requests_kwargs = {'verify': False}  # SSL 검증 비활성화
    return loader.load()


@st.cache_resource
#문서 나누기
def chunk_documents(docs, source_label, chunk_size=500, chunk_overlap=100):
    '''
    페이지(Document list)들을 청크로 나누고 metadata를 추가하는 함수
    - docs: Document list
    - source_label: 'cv' 또는 'jd' 등 source 표시
    - return: 청크가 나뉜 Document 리스트
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' '],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    split_docs = text_splitter.split_documents(docs)

    split_docs_with_meta = [
        Document(page_content=d.page_content,
                 metadata={'source': source_label, 'chunk_id': str(i)},)
        for i, d in enumerate(split_docs)
    ]

    return split_docs_with_meta

@st.cache_resource
#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
def create_vector_store(_cv, _jd):
    '''
    CV, JD 문서를 Chroma 벡터스토어로 변환
    - st.cache_resource 제거 → 반복 호출해도 누적 문제 없음
    '''
    # 청크 생성
    cv_docs = chunk_documents(_cv, 'cv')
    jd_docs = chunk_documents(_jd, 'jd')
    
    # 클라이언트 및 임베딩 모델 초기화
    client_chroma = Client(Settings(is_persistent=False))
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=cv_docs+jd_docs,
        embedding=embeddings,
        client=client_chroma,
    )

    return vectorstore

@st.cache_resource
#RAG 체인 구축
def chaining(_cv, _jd):
    vectorstore = create_vector_store(_cv, _jd)
    
    # -----------------------------
    # Retriever 설정
    # -----------------------------
    # CV Retriever
    cv_retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'filter': {'source': 'cv'}}
    )
    # JD Retriever
    jd_retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'filter': {'source': 'jd'}}
    )
    # Default Retriever
    default_retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 6}
    )

    # -----------------------------
    # System prompt 생성
    # -----------------------------
    system_prompt = '''
    당신은 커리어 및 채용 관련 질의응답을 돕는 AI 어시스턴트입니다.  
    아래 제공된 문서들을 참고하여, 사용자의 질문에 정확하고 논리적으로 답변하세요.  

    - 사용자의 **경력, 경험, 이력 관련 질문**이라면 CV(이력서)를 우선적으로 참고하세요.  
    - **지원하는 회사, 직무, 채용 요건** 관련 질문이라면 JD(공고문)를 우선적으로 참고하세요.  
    - 질문의 맥락상 두 문서가 모두 관련되어 있다면, **균형 있게 통합하여 답변**하세요.  
    - 문서에 직접적인 정보가 없을 경우, 일반적인 HR/커리어 상식과 논리를 기반으로 보완해 설명할 수 있습니다.
    - 자기소개서 작성 또는 면접 질문 생성 요청시 각각 아래와 같은 지침을 따릅니다.

    ** 자기소개서 작성 지침**
    - 자연스럽고 억지스럽지 않은 문장으로 꼭 필요한 부분에만 CV 내용을 활용합니다.
    - 경험과 관련된 사례를 들어야 할 경우 반드시 CV에 기술된 내용만을 사실대로 말합니다.
    - 작성된 경험들을 적절하게 JD에서 요구하는 능력 및 취업 후 성과 기여가 가능함의 근거로써 활용합니다.
    - 자기소개서 작성시 글자수 제한은 한글을 기준으로 띄어쓰기를 포함하여 계산합니다.
    - 질문지에 질문의 항목 또는 번호가 있더라도, 문장 형태로 답을 작성합니다.

    ** 면접 질문 생성 지침**
    - 회사 JD를 참고하여 지원자의 능력을 평가할 수 있을만한 질문을 생성합니다.
    - 일반적인 HR/커리어 상식과 논리를 기반으로 면접 질문을 생성 할 수 있습니다.

    **언어 스타일 지침**
    - 답변은 언제나 **전문적이고 신뢰감 있는 어투**로 작성합니다.  
    - 문장은 명료하고 간결하게 유지하되, 비즈니스 상황에 맞는 적절한 어휘를 사용합니다.  
    - 필요 시 가벼운 이모지를 활용하여 자연스럽게 친근함을 더할 수 있으나, 과도하게 사용하지 않습니다.  
        
    문서 내용:
    {context}
    '''

    # -----------------------------
    # ChatPromptTemplate 생성
    # -----------------------------
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', '{question}')
    ])


    # -----------------------------
    # MultiRetrievalQAChain 구성
    # -----------------------------
    retriever_infos = [
        {
            'name': 'cv',
            'description': 'user의 경력(CV) 및 이전 회사 관련 질문',
            'retriever': cv_retriever,
            'prompt': qa_prompt
        },
        {
            'name': 'jd',
            'description': '지원하는 회사 및 직무기술서(JD) 관련 질문',
            'retriever': jd_retriever,
            'prompt': qa_prompt
        },
    ]

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.4)

    rag_chain = MultiRetrievalQAChain.from_retrievers(
        llm=llm,
        retriever_infos=retriever_infos,
        default_retriever=default_retriever,
    )

    return rag_chain


# -----------------------------
# Streamlit UI
# -----------------------------

# CV 파일 업로드
st.header('경력기술서(CV) & JD URL 입력 💬 📚 ')
cv_file = st.file_uploader('경력기술서(.pdf/.docx)를 업로드하세요.', type=['pdf', 'docx'])
jd_url = st.text_input('JD(URL)을 입력하세요.', placeholder='예: https://example.com/job_description')

if cv_file is not None and jd_url:
    st.success('CV와 JD가 모두 업로드/입력되었습니다. RAG 체인을 생성합니다.')
    
    cv = load_cv(cv_file)
    jd = load_jd(jd_url)

    rag_chain = chaining(cv, jd)

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
                st.session_state['messages'].append({'role': 'assistant', 'content': response['result']})
                
                # LLM 답변 출력
                st.markdown('🧠 **GPT-4o-mini 답변:**')
                st.write(response['result'])
                
else:
    st.info('경력기술서와 JD URL을 모두 입력해야 다음 단계로 진행됩니다.')