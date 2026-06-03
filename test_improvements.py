import importlib.util
import unittest
from pathlib import Path

from utils.simple_document import Document

from config import Config
from utils.document_loader import extract_text_by_columns_fitz, is_two_column_layout_fitz
from utils.rag_pipeline import RAGPipeline
from utils.vector_db import VectorDB


class ImprovementTests(unittest.TestCase):
    def test_runtime_dependencies_are_declared(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        for package in ["PyMuPDF", "pdfplumber", "openai", "jieba", "scikit-learn", "Pillow", "rich"]:
            self.assertIn(package, requirements)

    def test_core_files_compile(self):
        for file_name in [
            "main.py",
            "config.py",
            "utils/document_loader.py",
            "utils/vector_db.py",
            "utils/rag_pipeline.py",
        ]:
            source = Path(file_name).read_text(encoding="utf-8")
            compile(source, file_name, "exec")

    def test_pdf_layout_extraction_first_page(self):
        import fitz

        pdf_files = sorted(Path(Config.DOCUMENT_FOLDER).glob("*.pdf"))
        self.assertTrue(pdf_files, "未找到 PDF 测试文件")
        with fitz.open(str(pdf_files[0])) as pdf:
            text = extract_text_by_columns_fitz(pdf[0])
            self.assertGreater(len(text), 80)
            self.assertNotIn("万方数据\n万方数据", text)
            self.assertIsInstance(is_two_column_layout_fitz(pdf[0]), bool)

    def test_vector_and_bm25_hybrid_search(self):
        docs = [
            Document(page_content="SEI 膜可以阻止电解液继续分解，同时允许 Li+ 通过。", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="法拉第定律描述电量与电极反应物质量之间的关系。", metadata={"source": "b.pdf", "page": 2}),
            Document(page_content="相图用于分析材料相变和稳定区域。", metadata={"source": "c.pdf", "page": 3}),
        ]
        db = VectorDB()
        db.create_index(docs)
        results = db.search("SEI 膜 有什么作用", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("SEI", results[0]["document"].page_content)
        self.assertIn("vector_similarity", results[0])
        self.assertIn("bm25_score", results[0])

    def test_prompt_and_response_parse_without_api(self):
        rag = RAGPipeline()
        doc = Document(page_content="电解质负责在正负极之间传输 Li+。", metadata={"source_ref": "demo.pdf 第1页", "topic": "电解质"})
        prompt = rag.generate_prompt("电解质的作用是什么？", [{"document": doc, "combined_score": 0.8, "vector_similarity": 0.7, "bm25_score": 0.9}])
        self.assertIn("只基于给定参考资料", prompt)
        answer, confidence, refs = rag._parse_response("电解质传输 Li+。\n\n参考资料: 1\n置信度: 0.82")
        self.assertIn("Li+", answer)
        self.assertAlmostEqual(confidence, 0.82)


if __name__ == "__main__":
    unittest.main()
