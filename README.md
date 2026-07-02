# fNIRS 深度学习多智能体自助平台

这是一个本地单用户平台，面向 fNIRS 数据处理、深度学习实验、RAG 知识库、多智能体对话、可解释性和报告生成。

## 文档

- [设计文档](docs/design_document.md)
- [架构文档](docs/architecture_document.md)
- [源程序说明文档](docs/source_program_document.md)

## 功能

- FastAPI 后端 + Vue 3 工作台。
- SQLite 元数据存储，文件存储位于 `artifacts/`。
- 本地 Ollama 模型配置：平台默认使用 qwen3:8b 作为 Chat 模型，使用 qwen3-embedding:8b 作为 Embedding 模型。上述模型名称并非固定要求，仅为系统默认配置。部署或二次开发时，可根据本机 Ollama 已安装模型进行替换；模型名称可通过前端设置页、/api/settings 接口、环境变量，或 backend/services.py 中的 DEFAULT_CHAT_MODEL 与 DEFAULT_EMBEDDING_MODEL 进行修改。
- 知识库上传、chunk、向量索引、Top-K 检索。
- 多智能体编排：Supervisor、RAG、Data、Preprocess、Modeling、Experiment、Explain、Report、Reviewer。
- fNIRS 数据上传：`.snirf`、`.nirs`、`.mat`、`.csv`、`.zip`、`.json`。
- 端到端 quick experiment：预处理、subject-wise validation、指标、解释、Markdown 报告。

## 启动
Python版本3.12.13

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

另一个终端：

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

访问：

```text
http://127.0.0.1:5173
```

## Ollama

```powershell
ollama pull qwen3:8b
ollama pull qwen3-embedding:8b
ollama serve
```

如果 Ollama 或 embedding 暂不可用，平台仍能启动；知识库 embedding 会使用本地 hashing fallback，聊天会给出检索式 fallback 回答。

## 验证

```powershell
python -m pytest
cd frontend
npm run build
```
