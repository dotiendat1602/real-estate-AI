# estatein-ai-service (FastAPI + pgvector)

## Documentation

- [`docs/README.md`](docs/README.md): cổng vào toàn bộ tài liệu kỹ thuật.
- [`docs/code-reading/README.md`](docs/code-reading/README.md): tài liệu đọc hiểu code theo file, hàm, luồng dữ liệu và playbook debug.
- [`docs/rag-system/README.md`](docs/rag-system/README.md): tài liệu kiến trúc/luồng RAG.
- [`docs/evaluation/README.md`](docs/evaluation/README.md): tài liệu benchmark/evaluation.

## 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install -e .

 ```
### Run project
```bash
python -m app.main
```
