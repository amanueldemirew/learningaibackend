# FastAPI Backend with SQLModel

A modern FastAPI backend application using SQLModel for database operations, designed for a learning management system.

## Features

- FastAPI for high-performance API development
- SQLModel for SQL database operations
- Pydantic for data validation
- JWT authentication
- Environment-based configuration
- Structured project layout
- File upload and management
- Course management system
- User authentication and authorization
- Table of Contents (TOC) management
- AI Integration with Google's Generative AI
- PDF processing and OCR capabilities

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── auth.py
│   │       │   ├── user.py
│   │       │   ├── file.py
│   │       │   └── courses/
│   │       └── router.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   └── session.py
│   ├── models/
│   │   ├── base.py
│   │   ├── user.py
│   │   ├── course.py
│   │   └── file.py
│   ├── schemas/
│   │   ├── user.py
│   │   ├── course.py
│   │   ├── file.py
│   │   ├── content.py
│   │   └── toc.py
│   ├── services/
│   ├── utils/
│   └── main.py
├── .env
├── .env.example
└── pyproject.toml
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies using pip:

```bash
pip install .
```

Or install in development mode:

```bash
pip install -e .
```

3. Copy `.env.example` to `.env` and update the variables:

```bash
cp .env.example .env
```

4. Run the application:

```bash
uvicorn app.main:app --reload
```

## Dependencies

The project uses modern Python packaging with `pyproject.toml`. Key dependencies include:

- FastAPI and related packages
- SQLModel for database operations
- Pydantic for data validation
- Google Generative AI integration
- LlamaIndex for AI/ML capabilities
- PDF processing tools (PyMuPDF, pdf2image)
- OCR capabilities (pytesseract)
- Authentication (python-jose, passlib)
- Environment management (python-dotenv)

## API Documentation

Once the application is running, you can access:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

The API includes the following main endpoints:

- `/api/v1/auth/*` - Authentication endpoints
- `/api/v1/users/*` - User management
- `/api/v1/courses/*` - Course management
- `/api/v1/files/*` - File management

## Features in Detail

### Authentication

- JWT-based authentication
- User registration and login
- Password hashing and verification

### User Management

- User CRUD operations
- Role-based access control
- User profile management

### Course Management

- Course creation and management
- Course content organization
- Table of Contents (TOC) structure
- Course enrollment

### File Management

- File upload and storage
- File metadata management
- Secure file access

### AI Integration

- Google Generative AI integration
- Document processing with LlamaIndex
- PDF text extraction and OCR capabilities
