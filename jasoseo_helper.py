import os
import tempfile
import urllib3
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

#Chroma tenant 오류 방지 위한 코드
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

#오픈AI API 키 설정
load_dotenv()  # 현재 경로의 .env 로드
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
if not os.environ['OPENAI_API_KEY']:
    raise ValueError('OPENAI_API_KEY not found in environment. set it in .env or env vars')

#경력기술서 로딩하기
@st.cache_resource #cache_resource로 한번 실행한 결과 캐싱해두기
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

#직무기술서 로딩하기
@st.cache_resource 
def load_jd(_url: str):
    loader = WebBaseLoader(_url)
    loader.requests_kwargs = {'verify': False}  # SSL 검증 비활성화
    return loader.load()

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def chunk_documents(pages, source_label, chunk_size=500, chunk_overlap=100):
    '''
    페이지(Document list)들을 청크로 나누고 metadata를 추가하는 함수
    - pages: Document list
    - source_label: 'cv' 또는 'jd' 등 source 표시
    - return: 청크가 나뉜 Document 리스트
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n'],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    split_docs = text_splitter.split_documents(pages)

    split_docs_with_meta = [
        Document(page_content=d.page_content,
                 metadata={'source': source_label, 'chunk_id': str(i)})
        for i, d in enumerate(split_docs)
    ]

    return split_docs_with_meta

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_cv, _jd):
    cv_docs = chunk_documents(_cv, 'cv')
    jd_docs = chunk_documents(_jd, 'jd')

    # OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    
    # 새 vectorstore 생성
    vectorstore = Chroma.from_documents(
        documents=(cv_docs+jd_docs),
        embedding=embeddings_model,
        persist_directory=None
    )
    return vectorstore

#검색된 문서를 하나의 텍스트로 union
def format_docs(docs):
    context = '\n\n'.join([d.page_content for d in docs])

#RAG 체인 구축
@st.cache_resource
def chaining(_cv, _jd):
    vectorstore = create_vector_store(_cv, _jd)

    # 벡터스토어에 저장된 CV 문서(청크) 수
    cv_docs_count = len([
        d for d in vectorstore._collection.get(include=['metadatas', 'documents'])['metadatas']
        if d['source'] == 'cv'
    ])
    
    # 벡터스토어에 저장된 CV 문서(청크) 수
    jd_docs_count = len([
        d for d in vectorstore._collection.get(include=["metadatas", "documents"])['metadatas']
        if d['source'] == 'jd'
    ])

    # CV만 검색
    cv_retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': cv_docs_count/2, 'fetch_k': cv_docs_count, 'filter': {'source': 'cv'}}
    )
    
    # JD만 검색
    jd_retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': jd_docs_count/2, 'fetch_k': jd_docs_count, 'filter': {'source': 'jd'}}
    )


    #이 부분의 시스템 프롬프트는 기호에 따라 변경하면 됩니다.
    qa_system_prompt = """
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    Please answer in Korean and use respectful language.\
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
    pages = load_cv(uploaded_file)

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
                
