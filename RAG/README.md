
# 🔍 RAG 检索增强生成系统

基于 **LangChain + LangGraph + Chroma + DeepSeek** 构建的检索增强生成系统，具备完整的文档处理流水线、语义去重、查询改写、多轮对话管理和置信度评估能力。

---

## ✨ 核心特性

- **完整文档处理管线**：加载 → 分割 → 去重 → 向量化 → 存储，一条命令完成索引构建
- **双重去重机制**：MD5 精确去重 + 余弦相似度语义去重，最大限度减少索引冗余
- **查询改写**：融合对话历史进行指代消解，将多轮对话中的模糊问题转为独立完整的检索查询
- **置信度评估**：生成答案后由独立 Evaluator 节点打分，低分自动触发二次检索与重新生成
- **可配置向量存储**：支持内存模式快速验证，或 Chroma 磁盘持久化模式保留索引
- **多集合管理**：支持创建、加载、删除多个索引集合，适配多知识库场景
- **FastAPI 服务**：提供标准化 RESTful API，内置请求日志、异常处理与统计监控

---

## 🧱 系统架构

```
                         ┌─────────────────┐
                         │    用户提问      │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   查询改写节点   │  ← 结合对话历史消解指代
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   文档检索节点   │  ← Chroma 向量相似度搜索
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   上下文拼接     │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   答案生成节点   │  ← DeepSeek 基于上下文生成
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   质量评估节点   │  ← LLM 置信度评分（0~1）
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  分数 ≥ 0.6？   │
                         └───┬──────────┬──┘
                             │否        │是
                    ┌────────▼──┐  ┌───▼──────────┐
                    │ 二次检索   │  │ 返回最终答案  │
                    └───────────┘  └──────────────┘
```

### 文档处理离线管线

```
原始文档 → 递归分块 → MD5精确去重 → 向量语义去重 → Chroma索引存储
                ↑                            │
          chunk_size=500               余弦相似度阈值0.9
          overlap=100
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（推荐）或智谱 API Key

### 安装依赖

```bash
pip install langchain langchain-openai langchain-chroma langchain-community \
            langgraph python-dotenv numpy sentence-transformers fastapi uvicorn
```

### 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
# 可选，用于语义嵌入
ZHIPUAI_API_KEY=your_zhipu_api_key
ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

### 运行示例

```python
from rag import RAGChain, RAGConfig

# 初始化配置
config = RAGConfig(chunk_size=500, chunk_overlap=100, top_k=3, deduplicate=True)
rag = RAGChain(config)

# 加载示例文档
texts = ["LangChain 是一个用于开发大型语言模型应用的开源框架..."]
metadatas = [{"source": "intro.txt"}]
rag.index_documents(texts, metadatas, collection_name="default")

# 开始多轮问答
print(rag.query("什么是LangChain？"))
# 第二轮自然使用历史，自动改写问题
print(rag.query("它的核心组件有哪些？"))
```

---

## 📁 项目结构

```
├── rag.py                  # RAG核心：文档处理、检索、生成、评估
├── main.py                 # FastAPI 服务入口
├── .env                    # 环境变量配置
└── README.md
```

---

## 🔧 核心设计

### 1. 分块策略

采用 `RecursiveCharacterTextSplitter`，按段落 → 句子 → 空格优先级递归分割。  
参数 `chunk_size=500, overlap=100`，在语义完整性与检索粒度间取得平衡。

### 2. 去重机制

| 步骤 | 方法 | 解决什么 | 局限 |
|------|------|----------|------|
| 精确去重 | MD5 哈希 | 完全重复的文本块 | 换一种说法即绕过 |
| 语义去重 | 余弦相似度（阈值 0.9） | 同义不同措辞的冗余 | 依赖嵌入模型领域适配 |

### 3. 查询改写

在多轮对话中，用户后续问题常含指代词（“它”、“这个”）或信息省略。系统利用对话历史调用 LLM 将问题改写为独立完整查询，显著提升检索命中率。

### 4. 评估与重试

生成回答后，独立 Evaluator 节点对回答质量进行 0~1 评分。  
低于 0.6 触发二次检索与重新生成；连续低分由最大重试次数保护，防止死循环。

### 5. 对话历史管理

全局维护对话历史，每条 `query()` 自动追加问答对。  
提供 `clear_history()` 方法重置会话，支持“不使用历史”选项（安全备份恢复机制）。

---

## 📊 API 接口

启动服务：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

主要端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 系统健康检查 |
| `POST` | `/index` | 批量索引文档 |
| `POST` | `/query` | 提交问答请求 |
| `GET` | `/history` | 获取对话历史 |
| `DELETE` | `/history` | 清除对话历史 |
| `GET` | `/stats` | 查看运行统计 |

---

## ⚙️ 配置参数

`RAGConfig` 可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.3 | 模型生成温度 |
| `chunk_size` | 500 | 文档分块大小（字符） |
| `chunk_overlap` | 100 | 相邻块重叠字符数 |
| `top_k` | 3 | 检索返回文档数 |
| `dedup_threshold` | 0.9 | 语义去重相似度阈值 |
| `deduplicate` | True | 是否启用去重 |

---

## 📝 适用场景

- 企业知识库问答
- 技术文档检索助手
- 教育辅助系统
- 内部信息查询平台

---

本项目仅用于学习与演示。模型 API 调用需遵循对应服务商的使用协议。
```
