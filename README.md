# Aviator Chatbot Backend (chatbot_agents)

A production-ready FastAPI backend system that powers the Aviator Chatbot application. Built with Python, FastAPI, PostgreSQL with pgvector, and advanced RAG (Retrieval-Augmented Generation) capabilities for intelligent document processing and conversational AI.

## 🚀 Latest Features & Updates

### 🔐 Dynamic User Authentication System
- **PostgreSQL-based user management** with secure password hashing
- **Role-based access control** (Support, Client, Tester roles)
- **JWT token authentication** with session validation
- **Automatic user table creation** with default accounts
- **Password security** with bcrypt hashing algorithms

### 💬 Advanced Chat & Messaging
- **Streaming response generation** with real-time updates
- **Chat session management** with persistent storage
- **Message history tracking** with PostgreSQL integration
- **User feedback system** with like/dislike analytics
- **Context-aware conversations** with memory management

### 📚 Intelligent Document Processing
- **RAG-powered document search** using pgvector similarity
- **Multi-format support** (PDF, TXT, Markdown)
- **Semantic document processing** with embedding generation
- **Vector storage optimization** with ChromaDB and pgvector
- **Document metadata management** with categorization

### 📊 Analytics & Feedback Management
- **Comprehensive feedback tracking** with user interaction data
- **Paginated API responses** for optimal performance
- **Chat analytics** with session and message metrics
- **User behavior tracking** for system optimization

### 🛠️ External Service Integration
- **Internet search capabilities** via external APIs
- **Ticket creation system** for support workflows
- **Advertisement service integration** for business logic
- **Modular tool architecture** for extensibility

## 🏗️ System Architecture

### Backend Architecture
```
chatbot_agents/
├── api/                     # FastAPI Application Layer
│   ├── main.py             # FastAPI app configuration & middleware
│   ├── auth_api.py         # User authentication endpoints
│   ├── chat_api.py         # Chat & messaging endpoints
│   ├── documents_api.py    # Document management endpoints
│   ├── history_api.py      # Chat history endpoints
│   ├── dependencies.py     # Shared dependencies & utilities
│   └── schemas.py          # Pydantic data models
├── core/                   # Business Logic Layer
│   ├── chatbot.py         # Main chatbot orchestration
│   ├── common/            # Shared utilities
│   │   ├── config.py      # Configuration management
│   │   └── prompt_manager.py  # AI prompt templates
│   └── services/          # Business Services
│       ├── user_service.py     # User management service
│       ├── history_service.py  # Chat history service
│       ├── document_service.py # Document processing
│       ├── rag_retrive_service.py  # RAG retrieval logic
│       ├── internet_search_service.py  # Web search
│       ├── ticket_creation_service.py  # Support tickets
│       └── adservice_api_service.py    # Advertisement API
├── tools/                  # AI Tool Implementations
│   ├── rag_tool.py        # RAG document search tool
│   ├── internet_search_tool.py  # Web search tool
│   ├── ticket_create_tool.py    # Ticket creation tool
│   ├── adservice_api_tool.py    # Advertisement tool
│   └── default_response_tool.py # Fallback responses
├── vector/                 # Vector Database Layer
│   ├── vectorstore_manager.py      # Vector store abstraction
│   ├── chromavector_manager.py     # ChromaDB implementation
│   ├── pgvector_manager.py         # PostgreSQL pgvector
│   ├── embedding_manager.py        # Embedding generation
│   └── semantic_document_processor.py  # Document processing
├── agent/                  # AI Agent Management
│   └── agent_manager.py   # Agent orchestration & lifecycle
├── utils/                  # Utility Layer
│   └── logger_config.py   # Logging configuration
├── streamlit_ui/          # Legacy Streamlit Interface
└── ui/                    # Alternative UI Components
```

### Data Flow Architecture
```
Client Request → FastAPI Router → Authentication → Business Service → 
Vector Database → AI Agent → Tool Execution → Response Generation → 
Streaming Response → Client
```

## 🛠️ Technology Stack

### Core Technologies
- **FastAPI 0.104+** - High-performance async web framework
- **Python 3.9+** - Modern Python with type hints
- **PostgreSQL 12+** - Primary database with pgvector extension
- **ChromaDB** - Vector database for document embeddings
- **SQLAlchemy 2.0** - Modern ORM with async support
- **Pydantic 2.0** - Data validation and serialization

### AI & Machine Learning
- **OpenAI GPT Models** - Large language model integration
- **Sentence Transformers** - Text embedding generation
- **LangChain** - AI application framework
- **Vector Similarity Search** - Semantic document retrieval
- **RAG (Retrieval-Augmented Generation)** - Context-aware responses

### External Integrations
- **Google Search API** - Internet search capabilities
- **Advertisement APIs** - Business service integration
- **Ticket Systems** - Support workflow automation
- **CORS Middleware** - Cross-origin request handling

### Development & Deployment
- **Uvicorn** - ASGI server for production
- **Poetry/pip** - Dependency management
- **Docker** - Containerization support
- **Environment Variables** - Configuration management

## 📋 Prerequisites & Setup

### System Requirements
- **Python 3.9+** with pip or poetry
- **PostgreSQL 12+** with pgvector extension
- **ChromaDB** for vector storage
- **OpenAI API Key** for language model access
- **Google Search API Key** (optional)

### Environment Variables (.env)
```env
# Database Configuration
PGVECTOR_CONNECTION_STRING=postgresql+psycopg://postgres:admin@localhost:5432/vectordb

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1500

# API Configuration
API_HOST=127.0.0.1
API_PORT=8001
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External Services
GOOGLE_SEARCH_API_KEY=your-google-api-key
GOOGLE_SEARCH_ENGINE_ID=your-search-engine-id
ADSERVICE_API_URL=https://api.example.com/ads
TICKET_SERVICE_URL=https://tickets.example.com/api

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/chatbot.log
```

## 🚀 Installation & Running

### 1. Environment Setup
```bash
# Clone/navigate to project directory
cd chatbot_agents

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup
```sql
-- PostgreSQL setup
CREATE DATABASE vectordb;
\c vectordb;
CREATE EXTENSION vector;

-- Verify pgvector installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 4. Initialize Database
```bash
# Run database initialization (creates tables and default users)
python -c "
from core.services.user_service import UserService
from core.services.history_service import HistoryService
UserService()  # Creates users table with default accounts
HistoryService()  # Creates chat history tables
print('Database initialized successfully')
"
```

### 5. Start the API Server
```bash
# Development server
python run_api.py

# Production server
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# API available at http://127.0.0.1:8001
# Interactive docs at http://127.0.0.1:8001/docs
```

## 🔐 Authentication & User Management

### Default User Accounts
| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `admin` | `admin123` | Support | Full system access |
| `kamala` | `admin` | Support | Full system access |
| `tester` | `test123` | Tester | Chat, documents, feedback |
| `client1` | `client123` | Client | Chat and feedback only |

### User Service Features
- **Secure password hashing** using bcrypt
- **Role-based access control** with enum validation
- **User profile management** with metadata
- **Session validation** with JWT tokens
- **Database-driven user storage** with PostgreSQL

### Authentication Flow
```python
# Login Process
1. User submits credentials → auth_api.py
2. UserService validates password → user_service.py
3. JWT token generated → return user profile
4. Frontend stores session data → subsequent API calls

# Authorization Process
1. API request with user context
2. Role validation against endpoint requirements
3. Business logic execution with user permissions
4. Data filtering based on user role
```

## 💬 Chat System Architecture

### Chat Flow
```
User Message → FastAPI Router → Chat Service → Agent Manager → 
Tool Selection → RAG Retrieval → LLM Processing → 
Response Generation → Streaming Output → Client
```

### Core Components

#### 1. Chat API (`api/chat_api.py`)
- **Message processing** with user context
- **Session management** with chat ID tracking
- **Streaming responses** with async generators
- **Feedback collection** and storage
- **Error handling** with graceful fallbacks

#### 2. Agent Manager (`agent/agent_manager.py`)
- **Tool orchestration** and selection
- **Context management** across conversations
- **Response optimization** for user queries
- **Memory management** for long conversations

#### 3. RAG Service (`core/services/rag_retrive_service.py`)
- **Document retrieval** using vector similarity
- **Context ranking** and relevance scoring
- **Multi-document synthesis** for comprehensive answers
- **Embedding optimization** for search accuracy

### Chat Features
- **Persistent chat sessions** with database storage
- **Context-aware responses** using conversation history
- **Document-grounded answers** via RAG implementation
- **Real-time streaming** with WebSocket support
- **Multi-turn conversations** with memory retention

## 📚 Document Management System

### Document Processing Pipeline
```
File Upload → Format Detection → Text Extraction → 
Chunking → Embedding Generation → Vector Storage → 
Metadata Indexing → Search Index Update
```

### Core Components

#### 1. Document API (`api/documents_api.py`)
- **File upload handling** with validation
- **Metadata management** (category, status, access)
- **Bulk operations** (upload, update, delete)
- **Pagination support** for large datasets
- **Search and filtering** capabilities

#### 2. Document Service (`core/services/document_service.py`)
- **Multi-format processing** (PDF, TXT, Markdown)
- **Text extraction** and cleaning
- **Metadata extraction** and enhancement
- **Storage optimization** and management

#### 3. Vector Storage (`vector/`)
- **ChromaDB integration** for vector storage
- **pgvector support** for PostgreSQL vectors
- **Embedding management** with caching
- **Similarity search** optimization

### Document Features
- **Intelligent text extraction** from multiple formats
- **Semantic chunking** for optimal retrieval
- **Metadata enrichment** with automatic tagging
- **Version control** and update tracking
- **Access control** based on user roles

## 🛠️ Tools & External Services

### Available Tools

#### 1. RAG Tool (`tools/rag_tool.py`)
- **Semantic document search** using vector similarity
- **Context retrieval** from uploaded documents
- **Relevance scoring** and ranking
- **Multi-document synthesis** for comprehensive answers

#### 2. Internet Search Tool (`tools/internet_search_tool.py`)
- **Google Search API integration** for real-time information
- **Result filtering** and relevance checking
- **Content summarization** for concise answers
- **Source attribution** and link provision

#### 3. Ticket Creation Tool (`tools/ticket_create_tool.py`)
- **Support ticket generation** for user issues
- **Priority assignment** based on issue type
- **Automatic routing** to appropriate teams
- **Status tracking** and updates

#### 4. Advertisement Service Tool (`tools/adservice_api_tool.py`)
- **Product recommendation** integration
- **Contextual advertising** based on conversation
- **Revenue optimization** through targeted ads
- **Performance tracking** and analytics

### Tool Architecture
```python
# Tool Interface
class BaseTool:
    def execute(self, query: str, context: Dict) -> Dict:
        pass
    
    def validate_input(self, query: str) -> bool:
        pass
    
    def format_output(self, result: Any) -> str:
        pass
```

## 📊 Analytics & Monitoring

### Feedback System
- **User interaction tracking** (likes, dislikes, ratings)
- **Response quality metrics** for system improvement
- **Usage analytics** for feature optimization
- **Performance monitoring** for response times

### Logging & Monitoring
```python
# Comprehensive logging
- API request/response logging
- Error tracking with stack traces
- Performance metrics (response times, throughput)
- User behavior analytics
- System health monitoring
```

### Database Analytics
- **Chat session analysis** with user engagement metrics
- **Document usage statistics** for content optimization
- **Tool usage patterns** for feature prioritization
- **Error pattern analysis** for system improvement

## 🔌 API Endpoints Reference

### Authentication Endpoints
```python
POST   /auth/login              # User authentication
GET    /auth/user/{username}    # Get user profile
POST   /auth/validate          # Session validation
```

### Chat Endpoints
```python
POST   /chat/                  # Send message and get response
GET    /history/user_history/{username}  # Get user's chat history
GET    /history/{username}/{chat_id}     # Get specific chat messages
POST   /chat/feedback          # Submit message feedback
GET    /chat/feedback/history/{username} # Get feedback history (paginated)
```

### Document Endpoints
```python
GET    /documents/             # Get paginated document list
POST   /documents/upload       # Upload new document(s)
PATCH  /documents/{id}         # Update document metadata
DELETE /documents/{id}         # Delete document
GET    /documents/search       # Search documents by content
```

### System Endpoints
```python
GET    /health                 # System health check
GET    /metrics                # System metrics and statistics
GET    /docs                   # Interactive API documentation
GET    /redoc                  # Alternative API documentation
```

## 🔧 Configuration Management

### Configuration Files

#### 1. Main Configuration (`config.py`)
```python
# Basic application configuration
DATABASE_URL = "postgresql://..."
API_HOST = "127.0.0.1"
API_PORT = 8001
```

#### 2. Common Configuration (`core/common/config.py`)
```python
# Shared configuration constants
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1500
TEMPERATURE = 0.7
```

#### 3. Environment Variables
- **Database connections** with connection pooling
- **API keys** for external services
- **Feature flags** for A/B testing
- **Performance tuning** parameters

### Configuration Management
- **Environment-based configuration** (dev, staging, prod)
- **Secret management** with environment variables
- **Dynamic configuration** updates without restart
- **Configuration validation** on application startup

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - PGVECTOR_CONNECTION_STRING=postgresql://postgres:password@db:5432/vectordb
    depends_on:
      - db
  
  db:
    image: ankane/pgvector
    environment:
      - POSTGRES_DB=vectordb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Production Configuration
```env
# Production environment variables
NODE_ENV=production
LOG_LEVEL=WARNING
DATABASE_POOL_SIZE=20
MAX_CONNECTIONS=100
CORS_ORIGINS=["https://aviator-chat.com"]
```

### Performance Optimization
- **Connection pooling** for database efficiency
- **Async processing** for concurrent requests
- **Caching strategies** for frequently accessed data
- **Load balancing** for high availability

## 🧪 Testing & Quality Assurance

### Testing Strategy
```python
# Test structure
tests/
├── unit/               # Unit tests for individual components
├── integration/        # Integration tests for services
├── api/               # API endpoint tests
└── e2e/               # End-to-end workflow tests
```

### Test Categories
- **Unit Tests**: Service classes, utility functions, data models
- **Integration Tests**: Database operations, external API calls
- **API Tests**: Endpoint functionality, authentication, authorization
- **Performance Tests**: Load testing, stress testing, benchmarking

### Quality Metrics
- **Code coverage** target: 80%+
- **Performance benchmarks** for response times
- **Error rate monitoring** for system reliability
- **Security testing** for vulnerability assessment

## 🔒 Security Implementation

### Security Features
- **JWT token authentication** with secure key management
- **Password hashing** using bcrypt with salt
- **Input validation** with Pydantic models
- **SQL injection prevention** with parameterized queries
- **CORS configuration** for cross-origin security
- **Rate limiting** for API endpoint protection

### Security Best Practices
```python
# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Input validation
class UserInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    chat_id: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9-_]+$')
```

### Data Protection
- **Personal data encryption** for sensitive information
- **Database connection security** with SSL/TLS
- **Environment variable protection** for secrets
- **User session management** with secure tokens

## 📈 Performance & Monitoring

### Performance Metrics
- **API response times** with percentile tracking
- **Database query optimization** with index analysis
- **Vector search performance** with similarity thresholds
- **Memory usage monitoring** for resource optimization

### Monitoring Tools
```python
# Built-in monitoring
- Request/response logging with timing
- Error tracking with detailed stack traces
- Performance metrics collection
- System resource monitoring

# External monitoring integration
- Prometheus metrics export
- Grafana dashboard support
- APM tool integration
- Log aggregation systems
```

### Optimization Strategies
- **Database indexing** for query performance
- **Connection pooling** for resource efficiency
- **Caching layers** for frequently accessed data
- **Async processing** for I/O-bound operations

## 🐛 Troubleshooting Guide

### Common Issues

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U postgres -d vectordb

# Verify pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';
```

#### API Server Issues
```bash
# Check server logs
tail -f logs/chatbot.log

# Test API health
curl http://127.0.0.1:8001/health

# Verify environment variables
python -c "import os; print(os.getenv('PGVECTOR_CONNECTION_STRING'))"
```

#### Vector Database Issues
```python
# Test vector operations
from vector.pgvector_manager import PgVectorManager
manager = PgVectorManager()
manager.test_connection()
```

### Performance Issues
- **Slow queries**: Check database indexes and query optimization
- **High memory usage**: Monitor vector storage and embedding cache
- **API timeouts**: Adjust timeout settings and async configurations
- **Vector search slow**: Optimize similarity thresholds and indexing

### Error Handling
```python
# Common error patterns
- Authentication errors: Check user credentials and JWT tokens
- Authorization errors: Verify user roles and permissions
- Database errors: Check connection and query syntax
- External API errors: Verify API keys and service availability
```

## 🔧 Development Guidelines

### Code Standards
- **Type hints** required for all functions
- **Docstrings** for all public methods and classes
- **Error handling** with appropriate exception types
- **Logging** at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Input validation** using Pydantic models

### Development Workflow
```bash
# Development setup
1. Create feature branch from main
2. Set up development environment
3. Write tests for new functionality
4. Implement feature with proper documentation
5. Run test suite and linting
6. Submit pull request with detailed description
```

### Code Review Checklist
- [ ] Type hints added for all new functions
- [ ] Unit tests written and passing
- [ ] Integration tests for external dependencies
- [ ] Error handling implemented appropriately
- [ ] Logging added for debugging support
- [ ] Documentation updated for API changes
- [ ] Security considerations reviewed
- [ ] Performance impact assessed

## 📦 Dependencies & Requirements

### Core Dependencies
```txt
# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Database
sqlalchemy>=2.0.0
psycopg[binary]>=3.1.0
pgvector>=0.2.0

# AI & ML
openai>=1.0.0
langchain>=0.1.0
sentence-transformers>=2.2.0
chromadb>=0.4.0

# Utilities
python-multipart
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
aiofiles
```

### Development Dependencies
```txt
# Testing
pytest>=7.0.0
pytest-asyncio
httpx
pytest-cov

# Code Quality
black
flake8
mypy
isort

# Documentation
sphinx
sphinx-autodoc-typehints
```

## 📄 License & Contributing

### License
MIT License - see the LICENSE file for details.

### Contributing Guidelines
1. **Fork the repository** and create a feature branch
2. **Follow Python PEP 8** style guidelines
3. **Add comprehensive tests** for new functionality
4. **Update documentation** for API changes
5. **Ensure type safety** with mypy validation
6. **Submit detailed pull requests** with testing evidence

### Support & Community
- **Issue tracking**: GitHub Issues for bug reports and feature requests
- **Documentation**: Comprehensive API docs at `/docs` endpoint
- **Development chat**: Discord/Slack for real-time collaboration
- **Code reviews**: Thorough review process for quality assurance

---

**Version**: 2.0.0  
**Last Updated**: January 2025  
**Python Version**: 3.9+  
**Maintainer**: Backend Development Team

For API documentation, visit the running server at `http://127.0.0.1:8001/docs` for interactive documentation. 