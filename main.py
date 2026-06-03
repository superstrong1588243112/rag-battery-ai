import streamlit as st

from config import Config
from utils.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="锂电池知识问答助手",
    page_icon="🔋",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f7f9;
        color: #111827;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    .answer-panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 24px 28px;
        margin: 12px 0 22px 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .answer-panel h1,
    .answer-panel h2,
    .answer-panel h3 {
        margin-top: 1.1rem;
        margin-bottom: 0.55rem;
        color: #111827;
        line-height: 1.35;
    }
    .answer-panel p,
    .answer-panel li {
        color: #1f2937;
        font-size: 16px;
        line-height: 1.85;
    }
    .answer-panel ul,
    .answer-panel ol {
        padding-left: 1.35rem;
    }
    .answer-panel strong {
        color: #111827;
    }
    .source-chip {
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 6px;
        padding: 3px 8px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #f9fafb;
        color: #374151;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    rag = RAGPipeline()
    rag.init_vector_db()
    return rag


def set_example_question(question):
    st.session_state.main_question = question


if "main_question" not in st.session_state:
    st.session_state.main_question = ""


st.title("锂电池知识问答助手")
st.caption("面向双栏 PDF 文献的知识库 RAG。检索使用 TF-IDF/FAISS 向量与 BM25 融合，回答使用 OpenAI。")

with st.sidebar:
    st.subheader("知识库")
    st.write(f"文档目录：`{Config.DOCUMENT_FOLDER}`")
    st.write(f"索引目录：`{Config.VECTOR_DB_PATH}`")
    st.write(f"LLM：`{Config.OPENAI_MODEL}`")
    st.write(f"Top K：`{Config.TOP_K}`")
    st.divider()
    st.caption("云端建议随项目提交预生成索引。替换 PDF 后，请本地重建索引再重新部署。")

try:
    with st.spinner("正在加载知识库索引..."):
        rag_pipeline = get_rag_pipeline()
except Exception as exc:
    st.error(f"知识库初始化失败：{exc}")
    st.stop()

examples = [
    "锂离子电池中 SEI 膜的作用是什么？",
    "法拉第定律在电化学中如何应用？",
    "非水液体电解质需要满足哪些性能要求？",
    "正极材料界面问题有哪些改善方法？",
    "锂离子电池失效分析通常关注哪些因素？",
]

cols = st.columns(len(examples))
for col, question in zip(cols, examples):
    with col:
        st.button(question, on_click=set_example_question, args=(question,), use_container_width=True)

with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "问题",
        key="main_question",
        height=96,
        placeholder="输入一个锂电池、电化学、材料或测试方法相关问题",
    )
    submitted = st.form_submit_button("检索并回答", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("请输入问题。")
        st.stop()

    with st.spinner("正在检索知识库并生成回答..."):
        result = rag_pipeline.query(question.strip())

    st.subheader("回答")
    st.markdown("<div class='answer-panel'>", unsafe_allow_html=True)
    st.markdown(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)

    confidence = result["confidence"]
    detail = result["confidence_explanation"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("整体置信度", f"{confidence:.1%}")
    metric_cols[1].metric("LLM 置信度", f"{detail['llm_confidence']:.1%}")
    metric_cols[2].metric("检索相似度", f"{detail['avg_retrieval_similarity']:.1%}")
    metric_cols[3].metric("参考片段", str(detail["context_count"]))
    st.progress(min(1.0, max(0.0, confidence)))

    st.subheader("参考片段")
    for index, ctx in enumerate(result["contexts"], start=1):
        doc = ctx["document"]
        meta = doc.metadata
        source = meta.get("source_ref") or meta.get("source", "未知来源")
        title = f"参考资料 {index} · {source} · 综合相似度 {ctx['combined_score']:.1%}"
        with st.expander(title, expanded=index <= 2):
            chips = [
                f"主题：{meta.get('topic', '未分类')}",
                f"向量：{ctx.get('vector_similarity', 0):.1%}",
                f"BM25：{ctx.get('bm25_score', 0):.1%}",
            ]
            st.markdown(" ".join(f"<span class='source-chip'>{chip}</span>" for chip in chips), unsafe_allow_html=True)
            st.markdown(doc.page_content)

    with st.expander("发送给大模型的提示词"):
        st.code(result.get("prompt", ""), language="markdown")
