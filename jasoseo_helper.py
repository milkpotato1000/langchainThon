import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
import requests

st.title("💬 자기소개서 작성 도우미")

# 1. API KEY
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# 2. 경력기술서 업로드
docs = []
uploaded_file = st.file_uploader("경력기술서 업로드 (.pdf/.docx)", type=['pdf','docx'])
if uploaded_file:
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(uploaded_file)
    else:
        loader = Docx2txtLoader(uploaded_file)
    docs = loader.load_and_split()

# 3. JD 입력 (URL)
jd_url = st.text_input('JD URL 입력')

jd_docs = []
if jd_url:
    try:
        res = requests.get(jd_url, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text()
        jd_docs = [Document(page_content=text)]
    except Exception as e:
        st.warning(f"JD URL 로딩 실패: {e}")

# 4. VectorStore 생성 준비
all_docs = docs + jd_docs
if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n'],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
    )

    split_docs = text_splitter.split_documents(all_docs)

    # OpenAI 임베딩 모델
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

    # Chroma 벡터스토어 생성 (persist_directory 지정하면 재사용 가능)
    persist_dir = './chroma_resume_db'
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings_model,
        persist_directory=persist_dir,
        collection_name='resume_seungwoo'
    )

    # Retriever 준비
    retriever = vectorstore.as_retriever()
    st.success("VectorStore 생성 완료 ✅")
else:
    st.info("업로드된 경력기술서 또는 JD URL이 없습니다.")
    
# 5. RAG 프롬프트
qa_system_prompt = """
경력기술서와 JD 정보를 바탕으로 자기소개서 질문에 대해 한글로 답변하세요. 
질문: {input}
관련 내용: {context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}")
])
llm = ChatOpenAI(model="gpt-4o-mini")
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
     "input": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 6. Streamlit chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content":"자기소개서 질문을 입력해주세요!"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    response = rag_chain.invoke(prompt)
    st.session_state["messages"].append({"role":"assistant","content":response})
    st.chat_message("assistant").write(response)
