# 锂电池知识问答助手部署版

这是一个 Streamlit RAG 应用，面向锂电池 PDF 文献问答。

## 已包含

- 双栏 PDF 解析与清洗逻辑
- TF-IDF/FAISS 向量检索 + BM25 混合检索
- OpenAI 兼容接口调用
- 预生成索引目录：`data/processed/faiss_index`
- Streamlit Secrets 读取，避免提交 API key

## 本地运行

```powershell
pip install -r requirements.txt
streamlit run main.py
```

本地可以创建 `.env`：

```env
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://api.fe8.cn/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_MODEL_CANDIDATES=gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1
```

## Streamlit Community Cloud 部署

1. 将本目录上传到 GitHub 仓库。
2. 打开 https://share.streamlit.io/ 。
3. 选择 GitHub 仓库、分支和入口文件 `main.py`。
4. 在 Advanced settings / Secrets 中填写：

```toml
OPENAI_API_KEY = "your_key"
OPENAI_API_BASE = "https://api.fe8.cn/v1"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL_CANDIDATES = "gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1"
```

5. 点击 Deploy。

部署完成后，Streamlit 会给出公网地址，格式通常类似：

```text
https://<app-name>.streamlit.app
```

## 注意

- 不要把 `.env`、API key、测试 key 文件提交到 GitHub。
- 如果替换 PDF 或修改解析策略，请先在本地重建索引，再重新提交 `data/processed/faiss_index/index.faiss` 和 `meta.pkl`。
- OpenAI 调用不是免费资源，会消耗你的中转渠道额度。
