# Nepal Financial Reporting Standards Assistant Chatbot (NFRS)

An intelligent document-based conversational AI system built to assist with Nepal Financial Reporting Standards (NFRS). This application leverages modern NLP technology to provide accurate, context-aware responses based on financial regulations, accounting standards, and organizational policies relevant in Nepal.

![NFRS Assistant](https://github.com/NikeGunn/imagess/blob/main/Nepal%20Financial%20Reporting%20Standards%20(NFRS)/nfrs.png?raw=true)

## ✨ Key Features

- **Secure Authentication** - JWT-based user management system
- **Multilingual Support** - Seamlessly switch between English and Nepali
- **Smart Document Handling** - Upload, process, and semantically search through PDF and TXT documents
- **Vector-Based Retrieval** - FAISS-powered semantic search for accurate information retrieval
- **AI-Powered Conversations** - Context-aware responses with document references
- **Enterprise Integration** - Comprehensive RESTful API

## 🛠️ Technology Architecture

| Component | Technologies |
|-----------|-------------|
| **Backend** | Django, Django REST Framework |
| **Database** | PostgreSQL |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **AI/NLP** | OpenAI GPT models, Embeddings API |
| **Deployment** | Docker, Docker Compose |
| **Translation** | Google Cloud Translation API |

## 📂 Project Structure

```
NFRS_Assistant/
├── api/
│   ├── users/         # Authentication and user management
│   ├── chat/          # Conversation engine and message handling
│   ├── knowledge/     # Document processor and vector search
├── media/             # Document storage
├── vector_store/      # FAISS indexes
├── NFRS_Assistant/    # Core configuration
│   ├── settings/      # Environment-specific settings
```

## 🚀 Setup and Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- OpenAI API key
- Google Cloud Translation credentials

### Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/NFRS_Assistant.git
   cd NFRS_Assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration
   ```

5. **Setup database**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. **Launch development server**
   ```bash
   python manage.py runserver
   ```

### Production Deployment

```bash
# Clone repository
git clone https://github.com/yourusername/NFRS_Assistant.git
cd NFRS_Assistant

# Configure environment
cp .env.sample .env
# Edit .env with production settings

# Deploy with Docker
docker-compose up -d --build
```

Access the application at http://localhost

## 📄 API Documentation

Interactive API documentation is available at:
- Swagger UI: `/api/docs/`
- ReDoc: `/api/redoc/`

## ⚙️ Configuration

| Environment Variable | Description |
|----------------------|-------------|
| `SECRET_KEY` | Django security key |
| `DEBUG` | Development mode (True/False) |
| `ALLOWED_HOSTS` | Comma-separated list of domains |
| `POSTGRES_DB` | Database name |
| `POSTGRES_USER` | Database username |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_HOST` | Database host |
| `POSTGRES_PORT` | Database port |
| `OPENAI_API_KEY` | OpenAI API authentication |
| `DEFAULT_LANGUAGE` | Default interface language (en/ne) |

## 🌐 API Endpoints Guide

### 1. Authentication & User Management

#### Authentication
- **Login (Get Token)**
  ```
  POST /api/v1/users/token/
  Body: {"username": "admin", "password": "admin123"}
  ```

- **Refresh Token**
  ```
  POST /api/v1/users/token/refresh/
  Body: {"refresh": "your_refresh_token"}
  ```

#### User Management
- **Register New User**
  ```
  POST /api/v1/users/register/
  Body: {
    "username": "newuser",
    "email": "newuser@example.com",
    "password": "SecurePassword123",
    "password2": "SecurePassword123",
    "preferred_language": "en"
  }
  ```

- **Get User Profile**
  ```
  GET /api/v1/users/profile/
  Headers: Authorization: Bearer your_access_token
  ```

- **Update User Profile**
  ```
  PUT /api/v1/users/profile/
  Headers: Authorization: Bearer your_access_token
  Body: {"preferred_language": "en"}
  ```

- **Change Password**
  ```
  POST /api/v1/users/change-password/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "old_password": "current_password",
    "new_password": "new_password",
    "new_password2": "new_password"
  }
  ```

#### API Keys
- **List API Keys**
  ```
  GET /api/v1/users/api-keys/
  Headers: Authorization: Bearer your_access_token
  ```

- **Create API Key**
  ```
  POST /api/v1/users/api-keys/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "name": "My Application",
    "description": "API key for my custom integration"
  }
  ```

- **Get API Key Details**
  ```
  GET /api/v1/users/api-keys/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

- **Delete API Key**
  ```
  DELETE /api/v1/users/api-keys/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

### 2. Chat Conversations

#### Conversations
- **List Conversations**
  ```
  GET /api/v1/chat/conversations/
  Headers: Authorization: Bearer your_access_token
  ```

- **Create New Conversation**
  ```
  POST /api/v1/chat/conversations/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "title": "New NFRS Inquiry",
    "language": "en"
  }
  ```

- **Get Conversation Details**
  ```
  GET /api/v1/chat/conversations/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

- **Update Conversation**
  ```
  PUT /api/v1/chat/conversations/{id}/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "title": "Updated Conversation Title",
    "language": "en"
  }
  ```

- **Delete Conversation**
  ```
  DELETE /api/v1/chat/conversations/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

#### Messages
- **Send Message**
  ```
  POST /api/v1/chat/messages/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "conversation_id": 1,
    "content": "What are the fire safety regulations for a medium-sized office building?",
    "language": "en"
  }
  ```

- **Get Message Details**
  ```
  GET /api/v1/chat/messages/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

- **Translate Message**
  ```
  POST /api/v1/chat/translate/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "text": "What are the fire safety regulations for a medium-sized office building?",
    "source_language": "en",
    "target_language": "ne"
  }
  ```

### 3. Knowledge Base

#### Documents
- **List Documents**
  ```
  GET /api/v1/knowledge/documents/
  Headers: Authorization: Bearer your_access_token
  ```

- **Get Document Details**
  ```
  GET /api/v1/knowledge/documents/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

- **Upload Document**
  ```
  POST /api/v1/knowledge/documents/upload/
  Headers: Authorization: Bearer your_access_token
  Body: FormData {
    file: (binary file),
    title: "Fire Safety Guidelines",
    description: "Official fire safety guidelines for commercial buildings",
    language: "en",
    is_public: "true"
  }
  ```

- **Admin Upload Document**
  ```
  POST /api/v1/knowledge/documents/admin-upload/
  Headers: Authorization: Bearer your_access_token
  Body: FormData {
    file: (binary file),
    title: "Official NFRS Guidelines",
    description: "Comprehensive NFRS regulations and procedures",
    language: "en",
    is_public: "true"
  }
  ```

- **Update Document**
  ```
  PUT /api/v1/knowledge/documents/{id}/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "title": "Updated Document Title",
    "description": "Updated document description",
    "is_public": true
  }
  ```

- **Delete Document**
  ```
  DELETE /api/v1/knowledge/documents/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

#### Vector Search
- **Search Knowledge Base**
  ```
  POST /api/v1/knowledge/search/
  Headers: Authorization: Bearer your_access_token
  Body: {
    "query": "What are the fire safety regulations for office buildings?",
    "top_k": 5,
    "filter_document_ids": []
  }
  ```

#### Vector Indices (Admin Only)
- **List Vector Indices**
  ```
  GET /api/v1/knowledge/indices/
  Headers: Authorization: Bearer your_access_token
  ```

- **Get Vector Index Details**
  ```
  GET /api/v1/knowledge/indices/{id}/
  Headers: Authorization: Bearer your_access_token
  ```

- **Rebuild Vector Index**
  ```
  POST /api/v1/knowledge/indices/rebuild/
  Headers: Authorization: Bearer your_access_token
  ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

## 📜 License

[MIT License](LICENSE)
