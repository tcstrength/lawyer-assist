# Lawyer assistant
Project này sử dụng FastAPI cho backend, Streamlit cho frontend và Qdrant database để lưu trữ và tìm kiếm tài liệu dưới dạng vector.

## Backend hỗ trợ các API sau:
- `/chat` : để chạy inference LLM models. Hiện tại, project chạy thực nghiệm trên hai LLM models: Gemma2 gốc của Google (https://huggingface.co/google/gemma-2-2b-it) và Gemma2 đã finetune với dữ liệu lấy từ https://thuvienphapluat.vn/hoi-dap-phap-luat
- `/retrieval` : để lấy các tài liệu liên quan tới câu hỏi của người dùng. Project sử dụng hai embedding models: halong_embedding (https://huggingface.co/hiieu/halong_embedding) và all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Frontend cung cấp giao diện chat và hiển thị kết quả truy xuất tài liệu từ các embedding model khác nhau.

## Requirements
### Các framework sử dụng
Docker (19.03+), Docker Compose (1.27.0+), Poetry

### Biến môi trường (file .env)
```
# Token để truy cập vào model Gemma2 đã finetune
GEMA_HF_TOKEN=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
# Token để truy cập vào các model khác
HF_TOKEN=<token>
# Cohere api key, để sử dụng dịch vụ của Cohere
COHERE_API_KEY=<cohere-api-key>

# Đường dẫn tới các thư mục data
DATA_ARTICLE_DIR=data/article
DATA_QNA_DIR=data/qna
DATA_SRC_URL_PREFIX=https://thuvienphapluat.vn/hoi-dap-phap-luat/tien-te-ngan-hang

# URL của các APIs trong backend, dùng cho frontend
LAWYER_API_URL=http://0.0.0.0:9000

MODEL_VECTORIZER_ID=dangvantuan/vietnamese-embedding
MODEL_CHAT_FT_ID=tcstrength/gemma-2b-lawyer-assist

# Qdrant URL, để backend truy vấn tài liệu
QDRANT_URL=http://0.0.0.0:6333
```

### Xây dựng và chạy các container
Để build và chạy các container, sử dụng lệnh sau:
```
docker-compose up --build
```

Lệnh này sẽ khởi chạy các services sau:

backend: Ứng dụng FastAPI hỗ trợ inference các LLM models.
frontend: Ứng dụng Streamlit tương tác với backend và hiển thị giao diện trò chuyện và kết quả mô hình.
qdrant: Qdrant database dùng cho các thao tác tìm kiếm vector.

### Truy cập các services
- Frontend: Truy cập http://localhost:8505 để vào ứng dụng Streamlit.
- Backend: Truy cập http://localhost:9000/docs để xem thông tin các APIs.
- Qdrant: Truy cập http://localhost:6333/dashboard# để xem thông tin Qdrant database.

### Dừng các container
Để dừng và xóa các container, sử dụng lệnh sau:
```
docker-compose down
```

