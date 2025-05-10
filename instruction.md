📘 NFRS Chatbot Backend (RAG-based) - Instruction Manual
This guide provides a complete walkthrough for setting up, running, and maintaining a Django REST Framework-based chatbot backend for Nepal’s NFRS system. It integrates OpenAI, supports multilingual response (English/Nepali), user-specific document embeddings, and robust chat history.
________________________________________
🚀 Tech Stack
•	Backend Framework: Django + Django REST Framework
•	Database: PostgreSQL
•	Embedding & Chat Completion: OpenAI API
•	Vector Database: FAISS (lightweight & fast)
•	Document Parsing: PyMuPDF (fitz)
•	Multilingual: Google Translate API / OpenAI translation (for Nepali)
•	User PDF Knowledge Memory: Temporary in-browser or user-scoped vector store
•	Optional Libraries: LangChain (stable + light usage)
•	Containerization: Docker + Docker Compose
•	Virtual Environment: venv
________________________________________
📁 Project Structure
backend_nfrs/
├── api/
│   ├── users/           # Authentication & profile logic
│   ├── chat/            # Chat history & multilingual pipeline
│   ├── knowledge/       # Admin & user-uploaded PDFs + vector store
│   ├── __init__.py
│   └── urls.py
├── backend_nfrs/       # Django project folder
│   ├── settings/
│   │   ├── base.py
│   │   ├── dev.py
│   │   └── prod.py
│   └── urls.py
├── media/              # Uploaded PDFs
├── vector_store/       # FAISS index files
├── manage.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
________________________________________
🔐 .env File Example
DEBUG=True
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_key
POSTGRES_DB=nfrs_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123
POSTGRES_HOST=db
POSTGRES_PORT=5432
DEFAULT_LANGUAGE=en
________________________________________
🧠 Features Overview
1. ✅ Authentication
•	Signup, login, JWT-based authentication
•	Profile management
•	Reusable logic for all endpoints
2. 💬 Chat History
•	ChatGPT-like history for each user
•	Context-aware questions and replies
•	Each chat session scoped per user
3. 📚 Knowledge Base (Admin & User Uploads)
•	Admin uploads NFRS PDFs via Django Admin
•	Users can upload finance-related PDFs (validated by title/content semantics)
•	Embeddings generated via OpenAI and stored in FAISS
•	Invalid/non-financial docs are rejected
4. 🌐 Language Support
•	Default: English
•	If user selects Nepali, chatbot replies purely in Nepali
•	Chatbot uses same RAG logic in both languages
5. 🧠 RAG Pipeline
•	Document chunks vectorized
•	User queries retrieve top-matching docs
•	Context + query passed to OpenAI ChatCompletion
6. ⚙️ Environment Handling
•	Modular settings/ for dev and prod
•	Easy toggle using DJANGO_SETTINGS_MODULE env var
7. 🐳 Docker + 🐍 Virtualenv
•	Docker and virtualenv both supported for flexibility
•	Developer friendly
________________________________________
🧪 Local Dev Setup (virtualenv)
git clone https://github.com/your/repo.git
cd backend_nfrs
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
________________________________________
🐳 Docker-based Setup
docker-compose up --build
📌 App will be available at http://localhost:8000
________________________________________
📤 Admin Panel (PDF Upload)
1.	Go to /admin
2.	Upload PDFs related to NFRS, laws, accounts
3.	App auto-generates embeddings (FAISS + OpenAI)
________________________________________
📤 User PDF Uploads
•	POST via /api/knowledge/user-upload/
•	Only finance/NFRS PDFs accepted (checked using prompt-engineering filters)
•	If valid, embedded & saved in user-isolated vector space
________________________________________
💬 Chat Flow (RAG)
1.	User sends a message via /api/chat/
2.	History fetched + FAISS search from admin/user knowledgebase
3.	Context + query passed to OpenAI API
4.	Response saved and returned
5.	If selected language is Nepali, response is translated
________________________________________
📂 APIs Summary
•	POST /api/auth/signup/
•	POST /api/auth/login/
•	POST /api/chat/ — Ask question
•	GET /api/chat/history/ — Retrieve previous chats
•	POST /api/knowledge/upload/ — Admin only
•	POST /api/knowledge/user-upload/ — Validates NFRS user PDFs
________________________________________
🌐 Multilingual Support Logic
•	User selects preferred language: en or ne
•	Language stored in profile or session
•	Chatbot dynamically responds in selected language
________________________________________
🤖 Prompt to Use the System (for AI/LLM)
You are a financial assistant chatbot for Nepal’s NFRS system. Your knowledge base is formed by government and accounting documents uploaded by admin or user. Your job is to answer any financial/legal queries based strictly on the embedded documents using a professional and helpful tone. If user has set Nepali as their preferred language, respond only in fluent Nepali.

Reject answering questions not related to accounting or NFRS.
________________________________________
📌 Notes
•	Always use dev.py locally, prod.py in production
•	.env values should be correct and secured
•	Vector store stored in /vector_store/
•	Nepali support improves accessibility for Nepal-based users


✅ You're ready to build and scale a smart, secure and multilingual RAG-based NFRS chatbot tailored for Nepal!
