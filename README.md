# Local RAG Document QA System | 本地 RAG 文档问答系统

这是一个基于 `Flask + Ollama + Tesseract OCR` 的本地 RAG（Retrieval-Augmented Generation，检索增强生成）项目。它提供了一个简单的网页界面，支持上传文本、PDF 和图片文件，自动提取内容并构建本地知识库，再结合 Ollama 本地大模型完成问答。

项目适合课程作业演示、离线知识问答原型、机械工程类资料的本地检索增强问答场景。

## 功能概览

- 支持上传 `txt`、`pdf`、`png`、`jpg`、`jpeg` 文件
- 对文本文件直接读取内容
- 对 PDF 使用 `PyPDF2` 提取文本
- 对图片使用 `Tesseract OCR` 识别中英文内容
- 使用 Ollama 的 `nomic-embed-text` 生成文本向量
- 基于余弦相似度检索最相关的文本块
- 使用 Ollama 大模型生成结合上下文的回答
- 前端展示回答结果与命中的上下文片段
- 支持一键重置本地知识库

## 技术栈

- 后端：Flask
- 大模型服务：Ollama
- Embedding 模型：`nomic-embed-text`
- 对话模型：`qwen3:8b`
- OCR：Tesseract OCR
- PDF 解析：PyPDF2
- 前端：原生 HTML/CSS/JavaScript

## 目录说明

```text
.
├─ app1.py                  # Flask 主程序，包含上传、检索、问答、重置接口
├─ requirements.txt         # Python 依赖
├─ README.md
├─ dummy_document.txt       # 示例知识库文档
├─ static/
│  └─ welcome_bg.png        # 首页欢迎区背景图
├─ templates/
│  └─ index.html            # 前端页面模板
└─ uploaded_documents/      # 运行时上传目录（仓库中仅保留空目录）
```

## 项目实现逻辑

### 1. 文档导入

用户通过网页上传文件，后端会先保存到 `uploaded_documents/`，再根据文件类型执行不同的文本提取流程：

- `txt`：直接读取文本
- `pdf`：提取每页文字内容
- 图片：调用 `pytesseract` 做 OCR

### 2. 文本切块

提取出的文本按空行进行初步切块。每个块长度过短时会被忽略，以降低无效内容进入知识库的概率。

### 3. 向量化与存储

每个文本块会通过 Ollama 的 embedding 接口转换为向量，并临时保存在内存中的 `documents_data` 结构里：

- `chunks`：文本块
- `embeddings`：对应向量
- `sources`：来源文件名

当前版本不使用数据库或向量库，服务重启后知识库会丢失，这是一个轻量课程项目原型的实现方式。

### 4. 查询与生成

用户提问后，系统会：

1. 先将问题转换为 embedding
2. 与知识库中的文本块做余弦相似度计算
3. 取最相关的若干片段作为上下文
4. 拼接提示词发送给 Ollama 大模型生成答案

### 5. 知识库重置

点击重置按钮后，系统会清空：

- 内存中的所有向量与文本块
- `uploaded_documents/` 下的运行时文件

## 本地运行

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装并启动 Ollama

请先安装 Ollama，并拉取项目使用的模型：

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama serve
```

默认配置写在 `app1.py` 中：

- `OLLAMA_HOST = http://localhost:11434`
- `LLM_MODEL = qwen3:8b`
- `EMBEDDING_MODEL = nomic-embed-text`

### 3. 安装 Tesseract OCR

如果要处理图片，请安装 Tesseract OCR，并确保系统中存在中文语言包 `chi_sim`。

Windows 下通常还需要把 Tesseract 安装目录加入系统环境变量 `PATH`。

### 4. 启动项目

```bash
python app1.py
```

浏览器访问：

```text
http://localhost:5001
```

## 接口说明

### `POST /api/upload_document`

上传并处理文档，支持：

- `txt`
- `pdf`
- `png`
- `jpg`
- `jpeg`

### `POST /api/rag_enhance`

提交问题，返回：

- `enhanced_text`：模型回答
- `retrieved_contexts`：命中的上下文片段
- `context_used_for_prompt`：拼接进提示词的上下文文本

### `POST /api/reset_knowledge_base`

清空当前知识库和已上传文件。

## 当前项目特点与局限

### 优点

- 结构简单，适合作为 RAG 入门项目
- 纯本地运行，不依赖云端 API
- 同时支持文本、PDF 和图片 OCR
- 前端交互完整，适合演示

### 局限

- 知识库仅保存在内存中，重启后不会持久化
- 切块逻辑较简单，尚未做更细粒度文本分段
- 没有使用专业向量数据库
- 没有用户认证、日志审计和生产部署配置

## 适合后续扩展的方向

- 引入 `FAISS`、`Chroma` 或其他向量库
- 增加多轮对话上下文管理
- 优化中文文本切块策略
- 增加上传历史和知识库管理界面
- 为 OCR 和 PDF 处理增加异常提示与预览
- 补充 Docker 部署与配置文件

## 课程项目分析结论

从当前目录内容来看，这个项目已经具备一个完整的课程展示原型：

- 有后端服务
- 有前端交互界面
- 有本地模型调用
- 有文档检索增强问答流程
- 有示例知识文档

如果作为 GitHub 仓库公开展示，建议至少保留以下文件：

- `app1.py`
- `templates/index.html`
- `static/welcome_bg.png`
- `dummy_document.txt`
- `requirements.txt`
- `README.md`

而 `.idea/` 和运行时上传文件不建议提交到仓库，因此已通过 `.gitignore` 排除。

---

## English Version

### Introduction

This project is a lightweight local RAG (Retrieval-Augmented Generation) application built with `Flask`, `Ollama`, and `Tesseract OCR`. It provides a simple web interface for uploading documents or images, extracting text, building a temporary local knowledge base, retrieving relevant context, and generating answers with a local LLM.

It is suitable for course demos, offline QA prototypes, and small local knowledge retrieval tasks.

### Features

- Supports `txt`, `pdf`, `png`, `jpg`, and `jpeg` uploads
- Reads text files directly
- Extracts PDF text with `PyPDF2`
- Performs OCR on images with `pytesseract`
- Generates embeddings with Ollama `nomic-embed-text`
- Retrieves relevant chunks using cosine similarity
- Generates context-aware answers with local Ollama models
- Shows both answers and retrieved context in the frontend
- Supports one-click knowledge-base reset

### Tech Stack

- Backend: Flask
- Model service: Ollama
- Embedding model: `nomic-embed-text`
- LLM: `qwen3:8b`
- OCR: Tesseract OCR
- PDF parser: PyPDF2
- Frontend: HTML, CSS, JavaScript

### Project Structure

```text
.
├─ app1.py
├─ requirements.txt
├─ README.md
├─ dummy_document.txt
├─ static/
│  └─ welcome_bg.png
├─ templates/
│  └─ index.html
└─ uploaded_documents/
```

### How It Works

1. Users upload text, PDF, or image files through the web interface.
2. The backend extracts text based on file type.
3. Extracted text is split into chunks, mainly by blank lines.
4. Each chunk is converted into an embedding through Ollama.
5. The embeddings are stored in memory in the `documents_data` structure.
6. When a query is submitted, the system computes cosine similarity against stored chunks.
7. The most relevant chunks are inserted into the prompt and sent to the local LLM.

### Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Pull and start Ollama models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama serve
```

Install Tesseract OCR if image OCR is needed, and make sure the Chinese language pack `chi_sim` is available.

Start the app:

```bash
python app1.py
```

Then open:

```text
http://localhost:5001
```

### API Endpoints

#### `POST /api/upload_document`

Uploads and processes a document.

#### `POST /api/rag_enhance`

Accepts a user query and returns:

- `enhanced_text`
- `retrieved_contexts`
- `context_used_for_prompt`

#### `POST /api/reset_knowledge_base`

Clears the in-memory knowledge base and uploaded runtime files.

### Strengths

- Lightweight and easy to understand
- Fully local workflow without cloud API dependency
- Supports plain text, PDF, and image OCR in one project
- Includes a complete frontend interaction flow for demonstrations

### Limitations

- The knowledge base is stored only in memory
- The chunking strategy is simple
- No vector database is used
- No authentication, persistence, or production deployment configuration

### Suggested Improvements

- Add a vector database such as `FAISS` or `Chroma`
- Improve Chinese text chunking
- Support multi-turn conversation memory
- Add upload history and knowledge-base management UI
- Improve OCR and PDF error handling
- Add Docker deployment support
