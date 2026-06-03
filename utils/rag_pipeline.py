import os
import re

from config import Config
from utils.document_loader import load_documents, split_documents
from utils.vector_db import VectorDB

if Config.USE_OPENAI:
    from openai import OpenAI
else:
    from dashscope import Generation
    os.environ["DASHSCOPE_API_KEY"] = Config.DASHSCOPE_API_KEY or ""


class RAGPipeline:
    def __init__(self):
        self.vector_db = VectorDB()
        self.openai_client = None
        if Config.USE_OPENAI and Config.OPENAI_API_KEY:
            self.openai_client = OpenAI(
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_API_BASE,
            )

    def init_vector_db(self):
        if not Config.FORCE_REBUILD_INDEX and VectorDB.has_index(Config.VECTOR_DB_PATH):
            self.vector_db.load(Config.VECTOR_DB_PATH)
            return

        print("[RAG] 创建新的混合检索索引")
        documents = load_documents(Config.DOCUMENT_FOLDER)
        print(f"[RAG] 加载页面文档数: {len(documents)}")
        chunks = split_documents(documents, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        print(f"[RAG] 切分片段数: {len(chunks)}")
        self.vector_db.create_index(chunks)
        self.vector_db.save(Config.VECTOR_DB_PATH)

    def hybrid_search(self, query, top_k=5):
        return self.vector_db.search(query, top_k)

    def generate_prompt(self, query, contexts):
        context_blocks = []
        for index, ctx in enumerate(contexts, start=1):
            doc = ctx["document"]
            meta = doc.metadata
            source = meta.get("source_ref") or meta.get("source", "未知来源")
            topic = meta.get("topic", "未分类")
            content = doc.page_content[:1800]
            context_blocks.append(
                f"[参考资料 {index}]\n"
                f"来源: {source}\n"
                f"主题: {topic}\n"
                f"综合相似度: {ctx['combined_score']:.4f}\n"
                f"向量相似度: {ctx.get('vector_similarity', 0):.4f}\n"
                f"BM25分数: {ctx.get('bm25_score', 0):.4f}\n"
                f"内容:\n{content}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        return f"""你是一位严谨的锂电池领域研究助理。请只基于给定参考资料回答用户问题。

## 参考资料
{context_text}

## 用户问题
{query}

## 回答目标
请给出“详细但不空泛”的专业解释。不要只列短要点；每个方面都要解释它为什么重要、作用机理是什么、对电池性能或工程实践有什么影响。

## 推荐输出结构
### 核心结论
用 2-4 句话概括答案，直接回答用户问题。

### 分方面详解
按 3-6 个方面展开。每个方面使用下面格式：
**1. 方面名称**
- **是什么**：定义或资料中的关键表述。
- **为什么重要**：说明背后的电化学/材料学机理。
- **影响与应用**：说明对容量、循环寿命、倍率、安全性、界面稳定性、成本或工艺的影响。
- **资料依据**：指出来自哪几个参考资料编号。

### 工程或研究启示
总结这些内容对材料设计、测试分析或电池开发的实际启示。

### 注意事项
说明参考资料不足、存在争议、适用范围或不能过度推断的地方。若资料不足，请明确写出。

### 参考资料
列出实际使用的参考资料编号和来源页。

## 写作要求
1. 必须基于参考资料，不要编造参考资料中没有的信息。
2. 优先保留资料中的专业术语、化学式、材料名、测试方法和因果关系。
3. 解释要具体，每个方面至少 3-5 句话；如果某方面资料不足，要说明不足，而不是强行扩写。
4. 使用 Markdown 标题、加粗关键词和项目符号，让页面显示清晰。
5. 最后一行给出置信度，格式必须为：置信度: 0.00 到 1.00。
"""

    def _parse_response(self, raw_text):
        answer = raw_text.strip()
        confidence = 0.55
        used_references = []

        match = re.search(r"置信度\s*[:：]\s*([01](?:\.\d+)?)", raw_text)
        if match:
            try:
                confidence = float(match.group(1))
            except ValueError:
                confidence = 0.55
            answer = raw_text[:match.start()].strip()

        refs_match = re.search(r"(参考资料|引用|资料编号)\s*[:：]?\s*(.*)", answer, flags=re.S)
        if refs_match:
            refs_text = refs_match.group(2)
            used_references = re.findall(r"参考资料\s*\d+|\[\s*\d+\s*\]|\d+", refs_text)

        return self._clean_answer(answer), min(1.0, max(0.0, confidence)), used_references

    def _clean_answer(self, text):
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _openai_models_to_try(self):
        models = [Config.OPENAI_MODEL]
        for model in getattr(Config, "OPENAI_MODEL_CANDIDATES", []):
            if model not in models:
                models.append(model)
        return models

    def _call_openai(self, prompt):
        if not self.openai_client:
            raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 OpenAI")

        last_error = None
        for model in self._openai_models_to_try():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是严谨的锂电池领域研究助理。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_error = exc
                message = str(exc)
                retryable_model_error = (
                    "model_not_found" in message
                    or "No available channel for model" in message
                    or "does not exist" in message
                )
                if not retryable_model_error:
                    break

        raise last_error

    def _call_dashscope(self, prompt):
        response = Generation.call(
            model="qwen-max",
            prompt=prompt,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
        if response is None or response.output is None:
            raise RuntimeError("大模型返回为空")
        if response.status_code != 200:
            raise RuntimeError(f"DashScope 调用失败: {response.status_code}")
        return response.output.text

    def query(self, question):
        contexts = self.hybrid_search(question, Config.TOP_K)
        if not contexts:
            return {
                "answer": "知识库中未检索到与该问题相关的资料。请换一种问法，或先重建/扩充知识库。",
                "confidence": 0.1,
                "confidence_explanation": {
                    "llm_confidence": 0.0,
                    "avg_retrieval_similarity": 0.0,
                    "context_count": 0,
                    "top_similarity": 0.0,
                },
                "prompt": "",
                "contexts": [],
                "used_references": [],
            }

        prompt = self.generate_prompt(question, contexts)
        try:
            raw_text = self._call_openai(prompt) if Config.USE_OPENAI else self._call_dashscope(prompt)
            answer, llm_confidence, used_references = self._parse_response(raw_text)
        except Exception as exc:
            answer = f"大模型调用失败: {exc}"
            llm_confidence = self.calculate_fallback_confidence(contexts)
            used_references = []

        avg_similarity = sum(ctx["combined_score"] for ctx in contexts) / len(contexts)
        final_confidence = 0.65 * llm_confidence + 0.35 * avg_similarity

        return {
            "answer": answer,
            "confidence": min(1.0, max(0.0, final_confidence)),
            "confidence_explanation": {
                "llm_confidence": llm_confidence,
                "avg_retrieval_similarity": avg_similarity,
                "context_count": len(contexts),
                "top_similarity": contexts[0]["combined_score"],
            },
            "prompt": prompt,
            "contexts": contexts,
            "used_references": used_references,
        }

    def calculate_fallback_confidence(self, contexts):
        if not contexts:
            return 0.1
        avg_similarity = sum(ctx["combined_score"] for ctx in contexts) / len(contexts)
        return min(0.75, max(0.25, avg_similarity))
