# NFRS Assistant Chatbot

A chatbot application for the Nepal Forest Research and Survey (NFRS) department, built using Django, PostgreSQL, FAISS for vector search, and the OpenAI API.

## Features

- JWT-based authentication system
- Multilingual support (English and Nepali)
- Document uploading and processing (PDF, TXT)
- Vector search for semantic document retrieval
- Conversational chat interface with document references
- RESTful API for integration with other systems

## Tech Stack

- **Backend**: Django, Django REST Framework
- **Database**: PostgreSQL
- **Vector Storage**: FAISS
- **NLP**: OpenAI API (GPT and Embeddings)
- **Containerization**: Docker, Docker Compose
- **Translation**: Google Cloud Translation API

## Project Structure

```
NFRS_Assistant/
├── api/
│   ├── users/         # User authentication and profile management
│   ├── chat/          # Conversation and message handling
│   ├── knowledge/     # Document management and vector search
├── media/             # Uploaded documents storage
├── vector_store/      # FAISS index files
├── NFRS_Assistant/    # Core Django project
│   ├── settings/      # Environment-specific settings
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- OpenAI API key
- Google Cloud Translation credentials (for multilingual support)

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NFRS_Assistant.git
   cd NFRS_Assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file from the sample:
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration
   ```

5. Run database migrations:
   ```bash
   python manage.py migrate
   ```

6. Create a superuser:
   ```bash
   python manage.py createsuperuser
   ```

7. Run the development server:
   ```bash
   python manage.py runserver
   ```

### Docker Setup (Production)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NFRS_Assistant.git
   cd NFRS_Assistant
   ```

2. Create a `.env` file from the sample:
   ```bash
   cp .env.sample .env
   # Edit .env with your production configuration
   ```

3. Build and start the Docker containers:
   ```bash
   docker-compose up -d --build
   ```

4. Access the application at http://localhost

## API Documentation

API documentation is available at `/api/docs/` or `/api/redoc/` endpoints when running the server.

## Environment Variables

The following environment variables can be configured in the `.env` file:

- `SECRET_KEY`: Django secret key
- `DEBUG`: Debug mode (True/False)
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts
- `POSTGRES_DB`: PostgreSQL database name
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `OPENAI_API_KEY`: OpenAI API key
- `DEFAULT_LANGUAGE`: Default language for the chatbot (en/ne)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

[MIT License](LICENSE)