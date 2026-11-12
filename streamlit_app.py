import os

# --- FAISS 안정화: faiss/numpy import 전에 설정 (커널·앱 멈춤 예방) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
faiss.omp_set_num_threads(1)

import tempfile
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# ====================================================================
#  📄🔍 PDF RAG + 웹 검색 챗봇 — 미니 프로젝트 스켈레톤
#
#  ✅ 정답 버전: 모든 [TODO]가 채워진 해설용 코드입니다.
#     (공통 주제: 「행정업무운영 편람」 기반 공문서 작성 안내 챗봇)
#
#  오늘 노트북에서 배운 코드를 아래 [TODO]에 조립하면 나만의 웹 앱이 됩니다.
#  - [TODO 1~4] : PDF → retriever → 도구 만들기 (참고: 11.AgentRAG)
#  - [TODO 5~7] : create_agent로 에이전트 구성 (참고: 11.AgentRAG)
#  - Streamlit UI 부분은 완성되어 있으니 그대로 쓰면 됩니다.
#
#  실행: python -m streamlit run streamlit_app.py
#        (Mac은 python3 -m streamlit run streamlit_app.py)
# ====================================================================


# --------------------------------------------------------------------
# 1. PDF -> retriever 도구 만들기  (참고: 11.AgentRAG 노트북)
# --------------------------------------------------------------------
def build_pdf_tool(uploaded_files):
    # 업로드된 PDF들을 임시 파일로 저장하며 로드합니다. (이 부분은 그대로 사용)
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        all_docs.extend(PyPDFLoader(tmp_path).load())

    # [TODO 1] 텍스트 분할기를 생성하고 문서를 분할하세요.
    # 조건: chunk_size=1000, chunk_overlap=200
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # [TODO 2] 분할된 문서와 OpenAIEmbeddings("text-embedding-3-small")로
    #          FAISS 벡터스토어를 생성하세요.
    vector = FAISS.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    # [TODO 3] 벡터스토어를 retriever로 변환하세요. (search_kwargs={"k": 5})
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    # [TODO 4] 아래 함수를 @tool 데코레이터로 감싸고,
    #          1. 도구 설명을 작성하세요. 
    #          2. retriever로 query를 검색한 결과를 반환하도록 완성하세요.
    #          docstring은 에이전트가 도구를 고르는 기준이 되므로 명확하게 쓰세요!
    @tool
    def pdf_search(query: str) -> str:
        """
        업로드된 「행정업무운영 편람」에서 공문서 작성 규칙과 행정업무 처리 방법을 검색합니다.

        다음과 같은 질문에 사용합니다:
        - 공문서의 구성(두문·본문·결문)과 작성 원칙
        - 날짜·시간·금액 등의 표기 방법
        - 항목 구분 기호의 순서, 붙임과 '끝' 표시 방법
        - 기안, 검토·협조, 결재(전결·대결), 문서의 등록·시행 절차
        - 공문서의 성립과 효력 발생 시기

        공문서 작성이나 행정업무 처리에 관한 질문에는 반드시 이 도구를 가장 먼저 사용하세요.
        입력값(query)은 찾고 싶은 규칙을 담은 짧은 문장이어야 합니다. 예) '날짜 표기 방법'
        """
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

    return pdf_search


# --------------------------------------------------------------------
# 2. create_agent 로 에이전트 구성  (참고: 11.AgentRAG 노트북)
# --------------------------------------------------------------------
def build_agent(tools):
    # [TODO 5] gpt-4o-mini, temperature=0 인 ChatOpenAI 모델을 생성하세요.
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # [TODO 6] 나만의 시스템 프롬프트를 작성하세요.
    # 포함할 것: ① 챗봇의 역할/페르소나  ② 도구 사용 우선순위(pdf_search 먼저)
    #           ③ 답변 언어·말투
    system_prompt = (
        # ① 역할 / 페르소나
        "당신은 공공기관 직원의 공문서 작성을 돕는 '공문서 작성 도우미'입니다.\n"
        "업로드된 「행정업무운영 편람」을 바탕으로 공문서 작성 규칙과 행정업무 처리 방법을 안내하고,\n"
        "그 밖의 일반적인 질문에도 웹 검색으로 답할 수 있습니다.\n\n"

        # ② 도구 사용 우선순위
        "[도구 선택 규칙]\n"
        "1. 공문서 작성·행정업무에 관한 질문(표기법, 문서 구성, 붙임·끝 표시, 기안·결재 절차 등)은\n"
        "   반드시 pdf_search를 먼저 사용하고, 검색된 편람 내용을 근거로 답하세요.\n"
        "   질문에 주제가 여러 개 섞여 있으면 주제별로 나누어 pdf_search를 여러 번 호출하세요.\n"
        "2. 편람과 관계없는 일반 질문(최신 소식, 상식, 기관·제도 정보, 날씨 등)은\n"
        "   pdf_search 없이 바로 tavily_search로 웹을 검색해 답하세요.\n"
        "3. pdf_search로 찾았는데 편람에 해당 내용이 없으면, tavily_search로 보충할 수 있습니다.\n"
        "   이때는 편람 기준이 아니라 웹 검색 결과라는 점을 분명히 밝히세요.\n"
        "4. 두 도구 모두에서 근거를 찾지 못하면 추측하지 말고 '자료에서 찾을 수 없습니다'라고 답하세요.\n\n"

        "[답변 규칙]\n"
        "- 편람 내용으로 답할 때는 규칙을 먼저 한 문장으로 말하고, 필요하면 올바른 예시를 함께 보여 주세요.\n"
        "- 편람에 나온 표기·형식을 임의로 바꾸거나 지어내지 마세요.\n"
        "- 답변 끝에 근거를 표시하세요: 편람을 썼으면 '(근거: 행정업무운영 편람)',\n"
        "  웹 검색을 썼으면 '(근거: 웹 검색)'과 참고한 출처 사이트 이름을 적으세요.\n\n"

        # ③ 답변 언어·말투
        "[말투]\n"
        "- 항상 한국어로, 친절하고 정중한 해요체로 답하세요.\n"
        "- 핵심부터 간결하게 말하고, 여러 항목은 목록으로 정리하세요."
    )

    # [TODO 7] create_agent로 에이전트를 생성해 반환하세요.
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


# --------------------------------------------------------------------
# 3. Streamlit UI  (완성 코드 — 수정 없이 사용 가능)
# --------------------------------------------------------------------
# 💡 커스터마이즈 아이디어: page_title / page_icon / title 을 내 챗봇 이름으로 바꿔보세요.
st.set_page_config(page_title="기보 공문서 작성 도우미", page_icon="📝")

st.logo("kibo_logo.png")   # 기금 로고 (아래 주의사항 참고)

st.title("📝 기보 공문서 작성 도우미", text_alignment="center")
st.caption("「행정업무운영 편람」을 근거로 공문서 작성 규칙을 안내합니다. 기금 내부 규정이 따로 있으면 그 규정을 우선 확인하세요.", text_alignment="center")
st.image("banner.png")   # 같은 폴더에 있는 배너 이미지

with st.sidebar:
    st.header("설정")
    openai_key = st.text_input("OpenAI API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    uploaded_files = st.file_uploader("PDF 업로드", type="pdf", accept_multiple_files=True)
    build = st.button("에이전트 만들기")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

if build:
    if not (openai_key and tavily_key and uploaded_files):
        st.warning("API 키 2개와 PDF를 모두 입력하세요.")
    else:
        with st.spinner("문서 인덱싱 및 에이전트 생성 중..."):
            tools = [build_pdf_tool(uploaded_files), TavilySearch(max_results=5)]
            st.session_state.agent = build_agent(tools)
            st.session_state.messages = []  # 새 에이전트를 만들면 대화 기록도 새로 시작
        st.success("준비 완료! 아래에 질문하세요.")

if "agent" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 지금까지의 대화를 화면에 다시 그립니다.
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    query = st.chat_input("질문을 입력하세요")
    if query:
        st.chat_message("user").write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # 이전 대화 전체를 함께 보내야 에이전트가 앞의 대화를 기억합니다.
                result = st.session_state.agent.invoke(
                    {"messages": st.session_state.messages}
                )
            answer = result["messages"][-1].content
            st.write(answer)
            
        #     # 실제로 검색된 원문을 펼쳐 볼 수 있게 보여 줍니다.
        #     with st.expander("🔎 검색된 근거 원문 보기"):
        #         for m in result["messages"]:
        #             if m.type == "tool":
        #                 st.markdown(f"**{m.name}**")
        #                 st.text(m.content)

        st.session_state.messages.append({"role": "assistant", "content": answer})
       