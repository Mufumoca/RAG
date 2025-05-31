import os
import io
import base64  # For potential future image display, not directly used for OCR text in RAG
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import json
import numpy as np
import pytesseract  # For OCR
from PIL import Image  # For image processing with Pytesseract
import PyPDF2  # For PDF processing
import time  # For unique filenames
import shutil  # Added for directory operations

# --- 配置 ---
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"  # 您之前使用的模型，可以替换
TOP_K_RESULTS = 3  # 检索最相关的K个文本块
UPLOAD_FOLDER = 'uploaded_documents'  # 用于存储上传的文件
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB 上传限制

# --- 全局变量存储文档数据 ---
documents_data = {
    "chunks": [],
    "embeddings": [],
    "sources": []  # 用于追踪每个chunk的来源文件名
}


# --- 辅助函数 ---
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_ollama_embedding(text, model=EMBEDDING_MODEL):
    """从Ollama获取文本的嵌入向量"""
    if not text or not text.strip():
        print("警告: 尝试获取空文本的嵌入。")
        return None
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": model, "prompt": text.strip()}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"错误：无法从Ollama获取嵌入: {e}")
        return None
    except KeyError:
        print(f"错误：Ollama的嵌入响应中没有'embedding'字段。响应: {response.text}")
        return None


def generate_ollama_response(prompt, model=LLM_MODEL):
    """使用Ollama生成文本"""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json().get("response", "Ollama未返回文本。")
    except requests.exceptions.RequestException as e:
        print(f"错误：调用Ollama生成API时出错: {e}")
        return f"Ollama API 错误: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"错误：解码Ollama响应时出错: {e}, 响应: {response.text}")
        return "解码Ollama响应时出错。"


def process_text_to_chunks(text, source_filename):
    """将文本分割成块，并为其生成嵌入"""
    # 简化分块：按空行分割段落，或固定长度分割
    raw_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_chunks:  # 如果按段落分割失败，尝试按句子或固定长度
        # 更精细的分块逻辑可以放在这里，例如使用nltk分句或Langchain的TextSplitters
        # 为简化，这里仅在没有段落时将整个文本作为一个块（如果不太长）
        if text.strip() and len(text.strip()) > 50:  # 避免太短的无意义块
            raw_chunks = [text.strip()]
        elif text.strip():  # 非常短的文本
            raw_chunks = [text.strip()]

    print(f"正在为来自 '{source_filename}' 的 {len(raw_chunks)} 个原始文本块生成嵌入...")
    new_chunks_count = 0
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text) < 20:  # 忽略过短的块
            print(f"跳过过短的块: '{chunk_text[:30]}...'")
            continue
        embedding = get_ollama_embedding(chunk_text)
        if embedding:
            documents_data["chunks"].append(chunk_text)
            documents_data["embeddings"].append(embedding)
            documents_data["sources"].append(source_filename)  # 记录来源
            new_chunks_count += 1
        # print(f"已处理来自 '{source_filename}' 的块 {i + 1}/{len(raw_chunks)}")
    print(f"成功从 '{source_filename}' 加载并嵌入 {new_chunks_count} 个有效文本块。")
    return new_chunks_count > 0


def extract_text_from_pdf(filepath):
    """从PDF文件中提取文本"""
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        print(f"成功从PDF '{filepath}' 提取文本。")
    except Exception as e:
        print(f"错误：从PDF '{filepath}' 提取文本失败: {e}")
    return text


def ocr_image_to_text(filepath):
    """使用Tesseract OCR从图像文件中提取文本"""
    text = ""
    try:
        # 指定中文简体语言包 chi_sim
        text = pytesseract.image_to_string(Image.open(filepath), lang='chi_sim+eng')  # 同时尝试英文和简体中文
        print(f"成功从图像 '{filepath}' OCR提取文本。")
    except Exception as e:
        print(f"错误：OCR处理图像 '{filepath}' 失败: {e}")
        print("请确保已正确安装Tesseract OCR并配置了语言包 (例如 chi_sim for Chinese)。")
    return text


def find_relevant_chunks(query_embedding, top_k=TOP_K_RESULTS):
    """根据查询嵌入找到最相关的文本块"""
    if not documents_data["embeddings"] or query_embedding is None:
        return [], []  # 返回空的chunks和sources

    query_emb_np = np.array(query_embedding)
    chunk_embeddings_np = np.array(documents_data["embeddings"])

    similarities = [
        np.dot(query_emb_np, chunk_emb) / (np.linalg.norm(query_emb_np) * np.linalg.norm(chunk_emb))
        if np.linalg.norm(query_emb_np) * np.linalg.norm(chunk_emb) != 0 else 0  # 避免除以零
        for chunk_emb in chunk_embeddings_np
    ]

    sorted_indices = np.argsort(similarities)[::-1]

    relevant_chunks = []
    relevant_sources = []  # 存储对应块的来源
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i]
        # 可以设置一个相似度阈值，例如 similarities[idx] > 0.5
        if similarities[idx] > 0.1:  # 降低阈值以获取更多潜在相关块
            relevant_chunks.append(documents_data["chunks"][idx])
            relevant_sources.append(documents_data["sources"][idx])
    return relevant_chunks, relevant_sources


# --- Flask路由 ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    # 仅用于开发时提供欢迎页背景图，生产环境应由Web服务器处理
    return send_from_directory('static', filename)


@app.route('/api/upload_document', methods=['POST'])
def upload_document_api():
    if 'file' not in request.files:
        return jsonify({"error": "请求中未找到文件部分 (No file part in the request)"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件 (No selected file)"}), 400

    if file and allowed_file(file.filename):
        # 为避免文件名冲突，可以添加时间戳或UUID
        original_filename = file.filename
        timestamp = str(int(time.time()))
        filename = timestamp + "_" + original_filename

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(filepath)
            print(f"文件 '{filename}' 已成功上传到 '{filepath}'")

            text_to_embed = ""
            file_ext = original_filename.rsplit('.', 1)[1].lower()

            if file_ext == 'txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_to_embed = f.read()
            elif file_ext == 'pdf':
                text_to_embed = extract_text_from_pdf(filepath)
            elif file_ext in {'png', 'jpg', 'jpeg'}:
                text_to_embed = ocr_image_to_text(filepath)

            if text_to_embed.strip():
                if process_text_to_chunks(text_to_embed, original_filename):  # 使用原始文件名作为来源标识
                    return jsonify({"message": f"文件 '{original_filename}' 已成功处理并添加到知识库。",
                                    "filename": original_filename}), 200
                else:
                    # Potentially remove uploaded file if no chunks were processed
                    # os.remove(filepath)
                    return jsonify({"error": f"未能从文件 '{original_filename}' 中提取有效文本块进行处理。"}), 500
            else:
                # os.remove(filepath) # 可选：如果未提取到文本，删除上传的空文件
                return jsonify({"error": f"未能从文件 '{original_filename}' 中提取任何文本。"}), 400

        except Exception as e:
            print(f"处理上传文件 '{filename}' 时出错: {e}")
            # if os.path.exists(filepath): os.remove(filepath) # 清理部分上传的文件
            return jsonify({"error": f"处理文件时出错: {str(e)}"}), 500
    else:
        return jsonify({"error": "文件类型不允许 (File type not allowed)"}), 400


@app.route('/api/rag_enhance', methods=['POST'])
def rag_enhance_api():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "必须提供查询 (Query is required)"}), 400

    query_embedding = get_ollama_embedding(query)
    if query_embedding is None:
        return jsonify({"error": "无法从Ollama获取查询嵌入 (Failed to get query embedding)"}), 500

    if not documents_data["chunks"]:
        print("警告：知识库为空。Ollama将在没有额外上下文的情况下响应。")
        relevant_chunks, relevant_sources = [], []
    else:
        relevant_chunks, relevant_sources = find_relevant_chunks(query_embedding)

    context_str_parts = []
    if relevant_chunks:
        for i, chunk in enumerate(relevant_chunks):
            source_info = f"(来自: {relevant_sources[i]})" if i < len(relevant_sources) else ""
            context_str_parts.append(f"{chunk} {source_info}")
    context_str = "\n\n".join(context_str_parts)

    if context_str:
        prompt = f"请根据以下上下文回答问题。\n\n上下文:\n{context_str}\n\n问题: {query}\n\n答案:"
    else:
        prompt = f"请回答问题: {query}\n\n答案:"

    print(f"\n--- 发送到Ollama的提示 ---\n{prompt}\n--------------------------\n")
    enhanced_text = generate_ollama_response(prompt)

    # 准备用于前端显示的上下文信息
    context_display = []
    if relevant_chunks:
        for i, chunk in enumerate(relevant_chunks):
            source = relevant_sources[i] if i < len(relevant_sources) else "未知来源"
            context_display.append({"text": chunk, "source": source})

    return jsonify({
        "enhanced_text": enhanced_text,
        "context_used_for_prompt": context_str if context_str else "（未从本地文档检索到特定上下文）",
        "retrieved_contexts": context_display  # 新增字段，用于前端更友好地展示上下文
    })


@app.route('/api/reset_knowledge_base', methods=['POST'])
def reset_knowledge_base_api():
    """
    重置知识库：
    1. 删除 uploaded_documents 文件夹中的所有文件。
    2. 清空全局的 documents_data 变量。
    """
    global documents_data
    folder = app.config['UPLOAD_FOLDER']
    try:
        # 1. 删除文件夹中的所有内容
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'无法删除 {file_path}. 原因: {e}')
                    # Optionally, collect errors and report them

        # 2. 重置内存中的文档数据
        documents_data = {
            "chunks": [],
            "embeddings": [],
            "sources": []
        }

        print("知识库已成功重置。")
        return jsonify({"message": "知识库已成功重置。"}), 200
    except Exception as e:
        print(f"重置知识库时出错: {e}")
        return jsonify({"error": f"重置知识库时出错: {str(e)}"}), 500


# --- 应用启动 ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"创建上传文件夹: {UPLOAD_FOLDER}")

    # 应用启动时不再加载固定文档，知识库将通过上传动态构建
    print("知识库将通过文件上传动态构建。")
    print(f"\nFlask应用正在运行于 http://localhost:5001")
    print(f"确保Ollama服务正在运行，并且模型 '{LLM_MODEL}' 和 '{EMBEDDING_MODEL}' 已下载。")
    print(f"上传的文件将存储在: '{UPLOAD_FOLDER}'")
    app.run(debug=True, port=5001)
