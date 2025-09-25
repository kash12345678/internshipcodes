import os
import asyncio
import logging
import functools
import time
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from marqo import Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception, RetryError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional, Any
import httpx
import re
import json
import hashlib

# Try to import redis, but handle if not available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    logging.warning("Redis not available. Caching will be memory-only.")

# -------------------------
# Configuration Settings
# -------------------------
class Settings(BaseSettings):
    marqo_url: str = "http://localhost:8882"
    index_name: str = "community_assistance_index"
    max_response_length: int = 2000
    max_references: int = 5
    max_related_questions: int = 3
    batch_size: int = 50
    max_batch_retries: int = 3
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 300
    cache_enabled: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8001
    reload: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# -------------------------
# Setup
# -------------------------
app = FastAPI(title="Community Assistance API", version="1.0.0")
logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))

# Retry config for Tenacity
MARQO_RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_fixed(2),
    "reraise": True,
}

# Global clients
marqo_client = None
redis_client = None

# In-memory cache fallback
memory_cache = {}
memory_cache_max_size = 1000

# Topic data for fallback responses
topic_data = {
    "education": {
        "short": "Education help includes FAFSA guidance, GED programs, and tutoring.",
        "long": "Education assistance covers a wide range of services including FAFSA guidance for financial aid, GED programs for completing high school equivalency, tutoring for academic support, and access to free online learning platforms.",
        "references": ["https://studentaid.gov", "https://www.khanacademy.org", "https://www.unicef.org/education"],
        "related": ["How to apply for FAFSA and what documents are needed?", "Are there free GED programs near me?"]
    },
    "food": {
        "short": "Food assistance includes food banks and meal programs.",
        "long": "Food assistance includes access to community food banks, meal delivery programs for seniors and vulnerable groups, nutrition education, and emergency grocery support for families in need.",
        "references": ["https://www.feedingamerica.org", "https://www.fns.usda.gov/snap", "https://www.wfp.org/food-assistance"],
        "related": ["Where is the nearest food bank?", "Am I eligible for SNAP benefits?"]
    },
    "transport": {
        "short": "Transport help includes bus passes and ride assistance.",
        "long": "Transport assistance services provide free or subsidized bus passes, special transport for disabled individuals, ride assistance for medical visits, and community shuttle programs.",
        "references": ["https://www.transit.dot.gov", "https://www.uber.com/us/en/ride/", "https://www.lyft.com"],
        "related": ["How do I apply for reduced bus fare?", "Is there transport help for medical appointments?"]
    },
    "healthcare": {
        "short": "Healthcare support includes clinics, mental health, and wellness programs.",
        "long": "Healthcare support covers free and low-cost clinics, preventive screenings, prescription assistance, mental health counseling, addiction recovery programs, and wellness resources for families.",
        "references": ["https://www.who.int", "https://www.cdc.gov", "https://www.healthcare.gov"],
        "related": ["Where can I find a free clinic nearby?", "What mental health services are available?"]
    }
}

# -------------------------
# Caching Utilities
# -------------------------
def generate_cache_key(topic: str, detail: str, query: Optional[str] = None) -> str:
    key_data = f"{topic}:{detail}:{query or ''}"
    return hashlib.md5(key_data.encode()).hexdigest()

async def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    if not settings.cache_enabled:
        return None
        
    try:
        if redis_client and REDIS_AVAILABLE:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.info(f"✅ Cache hit from Redis for key: {cache_key[:8]}...")
                cache_stats.record_hit()
                return json.loads(cached)
        
        if cache_key in memory_cache:
            cached_data = memory_cache[cache_key]
            if time.time() - cached_data.get('_cached_at', 0) < settings.cache_ttl:
                logger.info(f"✅ Cache hit from memory for key: {cache_key[:8]}...")
                cache_stats.record_hit()
                return cached_data['data']
            else:
                del memory_cache[cache_key]
                cache_stats.record_miss()
                
    except Exception as e:
        logger.warning(f"⚠️ Cache read error: {e}")
        cache_stats.record_miss()
        
    cache_stats.record_miss()
    return None

async def set_cached_response(cache_key: str, data: Dict[str, Any]) -> None:
    if not settings.cache_enabled:
        return
        
    try:
        cache_data = {
            'data': data,
            '_cached_at': time.time()
        }
        
        if redis_client and REDIS_AVAILABLE:
            await redis_client.setex(
                cache_key, 
                settings.cache_ttl, 
                json.dumps(cache_data)
            )
        
        if len(memory_cache) >= memory_cache_max_size:
            oldest_key = next(iter(memory_cache))
            del memory_cache[oldest_key]
            
        memory_cache[cache_key] = cache_data
        logger.info(f"💾 Cached response for key: {cache_key[:8]}...")
        
    except Exception as e:
        logger.warning(f"⚠️ Cache write error: {e}")

async def invalidate_topic_cache(topic: str) -> None:
    if not settings.cache_enabled:
        return
        
    try:
        keys_to_remove = []
        for key in memory_cache.keys():
            if topic in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del memory_cache[key]
            
        logger.info(f"🗑️ Invalidated {len(keys_to_remove)} cache entries for topic: {topic}")
        
    except Exception as e:
        logger.warning(f"⚠️ Cache invalidation error: {e}")

def cache_response(ttl: int = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.cache_enabled:
                cache_stats.record_miss()
                return await func(*args, **kwargs)
                
            topic = kwargs.get('topic', '')
            detail = kwargs.get('detail', 'short')
            query = kwargs.get('query', None)
            q = kwargs.get('q', None)
            topic_name = kwargs.get('topic_name', '')
            
            if q:
                cache_key = generate_cache_key('search', q, str(kwargs.get('limit', 10)))
            elif topic_name:
                cache_key = generate_cache_key('topic', topic_name)
            else:
                cache_key = generate_cache_key(topic, detail, query)
                
            cache_ttl = ttl or settings.cache_ttl
            
            cached_response = await get_cached_response(cache_key)
            if cached_response:
                return cached_response
            
            response = await func(*args, **kwargs)
            
            if response and isinstance(response, dict):
                await set_cached_response(cache_key, response)
            
            return response
        return wrapper
    return decorator

# -------------------------
# Request / Response Models
# -------------------------
class DocsInput(BaseModel):
    docs: List[Dict[str, Any]] = Field(..., description="List of documents to add to the index")

class QAResponse(BaseModel):
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The answer to the question")
    references: List[str] = Field(..., description="List of reference URLs")
    related_questions: List[str] = Field(..., description="List of related questions")
    truncated: bool = Field(False, description="Indicates if response was truncated")

class AddDocsResponse(BaseModel):
    status: str = Field(..., description="Operation status")
    added: int = Field(..., description="Number of documents added")
    skipped: int = Field(..., description="Number of documents skipped")
    batches_processed: int = Field(..., description="Number of batches processed")
    total_batches: int = Field(..., description="Total number of batches")
    timestamp: float = Field(..., description="Operation timestamp")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    status_code: int = Field(..., description="HTTP status code")

class CacheStatsResponse(BaseModel):
    memory_cache_size: int = Field(..., description="Number of items in memory cache")
    redis_connected: bool = Field(..., description="Whether Redis is connected")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    cache_hits: int = Field(..., description="Total cache hits")
    cache_misses: int = Field(..., description="Total cache misses")
    hit_rate: float = Field(..., description="Cache hit rate")

# -------------------------
# Response Size Control Utilities
# -------------------------
def truncate_text(text: str, max_length: int) -> tuple[str, bool]:
    if len(text) <= max_length:
        return text, False
    return text[:max_length].rsplit(' ', 1)[0] + "...", True

def clean_and_deduplicate_references(references: List[str]) -> List[str]:
    unique_refs = set()
    cleaned_refs = []
    
    for ref in references:
        cleaned_ref = ref.strip()
        if not cleaned_ref:
            continue
            
        if not cleaned_ref.startswith(('http://', 'https://')):
            cleaned_ref = 'https://' + cleaned_ref
            
        if cleaned_ref not in unique_refs:
            unique_refs.add(cleaned_ref)
            cleaned_refs.append(cleaned_ref)
            
    return cleaned_refs[:settings.max_references]

def smart_truncate_response(answer: str, references: List[str], related_questions: List[str]) -> tuple[str, List[str], List[str], bool]:
    was_truncated = False
    
    if len(answer) > settings.max_response_length:
        answer, was_truncated = truncate_text(answer, settings.max_response_length)
    
    if len(references) > settings.max_references:
        references = references[:settings.max_references]
        was_truncated = True
    
    if len(related_questions) > settings.max_related_questions:
        related_questions = related_questions[:settings.max_related_questions]
        was_truncated = True
    
    return answer, references, related_questions, was_truncated

# -------------------------
# RAG Concatenation Utilities
# -------------------------
def concatenate_rag_content(hits: List[Dict]) -> str:
    if not hits:
        return ""
    
    content_parts = []
    seen_content = set()
    
    for hit in hits:
        content = hit.get("content", "").strip()
        if content and content not in seen_content:
            seen_content.add(content)
            content_parts.append(content)
    
    concatenated = " ".join(content_parts)
    concatenated = re.sub(r'\s+', ' ', concatenated).strip()
    
    return concatenated

def extract_references_from_hits(hits: List[Dict]) -> List[str]:
    references = set()
    
    for hit in hits:
        metadata = hit.get("metadata", {})
        hit_references = metadata.get("references", [])
        
        if isinstance(hit_references, list):
            for ref in hit_references:
                if isinstance(ref, str) and ref.strip():
                    references.add(ref.strip())
        elif isinstance(hit_references, str):
            for ref in hit_references.split(','):
                if ref.strip():
                    references.add(ref.strip())
    
    return clean_and_deduplicate_references(list(references))

# -------------------------
# Custom Exceptions
# -------------------------
class MarqoConnectionError(HTTPException):
    def __init__(self, detail: str = "Marqo service unavailable"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )

class IndexNotReadyError(HTTPException):
    def __init__(self, detail: str = "Index is not ready"):
        super().__init__(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=detail
        )

class DocumentOperationError(HTTPException):
    def __init__(self, detail: str = "Document operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class ResponseTooLargeError(HTTPException):
    def __init__(self, detail: str = "Response too large"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

# -------------------------
# Polling Functions
# -------------------------
async def wait_for_index_ready(timeout: int = 30, check_interval: int = 1) -> bool:
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout:
        attempt += 1
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.marqo_url}/indexes/{settings.index_name}/stats",
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Index ready after {attempt} attempts")
                    return True
                    
        except (httpx.HTTPError, ConnectionError) as e:
            logger.debug(f"Index not ready yet (attempt {attempt}): {e}")
        
        await asyncio.sleep(min(check_interval * (1.5 ** attempt), 5))
    
    logger.error(f"❌ Index not ready after {timeout} seconds")
    raise IndexNotReadyError(f"Index not ready after {timeout} seconds")

async def wait_for_marqo_health(timeout: int = 30) -> bool:
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout:
        attempt += 1
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.marqo_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Marqo server healthy")
                    return True
                    
        except (httpx.HTTPError, ConnectionError) as e:
            logger.debug(f"Marqo not healthy yet (attempt {attempt}): {e}")
        
        await asyncio.sleep(min(1 * (1.5 ** attempt), 3))
    
    logger.error("❌ Marqo server not healthy")
    raise MarqoConnectionError("Marqo server not healthy after timeout")

# -------------------------
# Simple Circuit Breaker Simulation
# -------------------------
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            if self.is_open:
                if current_time - self.last_failure_time > self.recovery_timeout:
                    logger.info("⚠️ Circuit breaker half-open - testing recovery")
                    self.is_open = False
                else:
                    raise MarqoConnectionError("Service temporarily unavailable due to repeated failures")
            
            try:
                result = await func(*args, **kwargs)
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.failure_threshold:
                    self.is_open = True
                    logger.error(f"🔌 Circuit breaker opened after {self.failure_count} failures")
                
                raise
        return wrapper

marqo_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

# -------------------------
# Retry Wrapper
# -------------------------
def with_marqo_retry(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            retry_decorator = retry(**MARQO_RETRY_CONFIG)
            if asyncio.iscoroutinefunction(func):
                wrapped = retry_decorator(func)
                return await wrapped(*args, **kwargs)
            else:
                wrapped = retry_decorator(func)
                return wrapped(*args, **kwargs)
        except RetryError as e:
            logger.error(f"All retry attempts failed: {e}")
            raise MarqoConnectionError("Service unavailable after multiple retry attempts")
        except Exception as e:
            logger.error(f"Unexpected error in retry wrapper: {e}")
            raise
    return wrapper

# -------------------------
# Cache Statistics
# -------------------------
class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

cache_stats = CacheStats()

# -------------------------
# Initialize Clients
# -------------------------
@app.on_event("startup")
async def startup_event():
    global marqo_client, redis_client, app_start_time
    
    app_start_time = time.time()
    
    try:
        if not await wait_for_marqo_health():
            logger.error("Marqo server not available during startup")
            marqo_client = None
        else:
            marqo_client = Client(url=settings.marqo_url)
            try:
                marqo_client.get_indexes()
                logger.info("✅ Marqo client initialized and connected")
            except Exception as e:
                logger.error(f"❌ Marqo client connected but API failed: {e}")
                marqo_client = None
    except Exception as e:
        logger.error(f"❌ Failed to init Marqo client: {e}")
        marqo_client = None

    try:
        if settings.cache_enabled and REDIS_AVAILABLE:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("✅ Redis client initialized and connected")
        else:
            if not REDIS_AVAILABLE:
                logger.info("ℹ️ Redis not available, using memory cache only")
            else:
                logger.info("ℹ️ Caching is disabled")
            redis_client = None
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed, using memory cache only: {e}")
        redis_client = None

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client and REDIS_AVAILABLE:
        await redis_client.close()
        logger.info("✅ Redis client closed")

# -------------------------
# Health check
# -------------------------
@app.get("/health", response_model=Dict[str, Any])
async def health():
    marqo_health = False
    redis_health = False
    
    if marqo_client:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.marqo_url}/health", timeout=5)
                marqo_health = response.status_code == 200
        except Exception as e:
            logger.error(f"Marqo health check failed: {e}")
    
    if redis_client and REDIS_AVAILABLE:
        try:
            await redis_client.ping()
            redis_health = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
    
    overall_health = marqo_health and (not settings.cache_enabled or redis_health or not redis_client)
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "marqo_connected": marqo_health,
        "redis_connected": redis_health,
        "cache_enabled": settings.cache_enabled,
        "client_initialized": marqo_client is not None,
        "timestamp": time.time()
    }

# -------------------------
# Cache Management Endpoints
# -------------------------
@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    return {
        "memory_cache_size": len(memory_cache),
        "redis_connected": redis_client is not None and REDIS_AVAILABLE,
        "cache_enabled": settings.cache_enabled,
        "cache_hits": cache_stats.hits,
        "cache_misses": cache_stats.misses,
        "hit_rate": cache_stats.hit_rate
    }

@app.post("/cache/clear")
async def clear_cache():
    try:
        memory_cache.clear()
        
        if redis_client and REDIS_AVAILABLE:
            await redis_client.flushdb()
        
        logger.info("🗑️ Cache cleared successfully")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.post("/cache/invalidate/{topic}")
async def invalidate_cache(topic: str):
    try:
        await invalidate_topic_cache(topic)
        return {"status": "success", "message": f"Cache invalidated for topic: {topic}"}
    except Exception as e:
        logger.error(f"❌ Error invalidating cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate cache: {str(e)}"
        )

# -------------------------
# Document Management with Batching
# -------------------------
async def get_existing_document_ids():
    if not marqo_client:
        return set()
    
    try:
        results = marqo_client.index(settings.index_name).search("", limit=1000) or {}
        existing_ids = {hit["_id"] for hit in results.get("hits", [])}
        return existing_ids
    except Exception as e:
        logger.warning(f"Could not fetch existing document IDs: {e}")
        raise DocumentOperationError(f"Failed to fetch document IDs: {str(e)}")

async def safe_add_documents(documents, batch_size: int = None):
    if batch_size is None:
        batch_size = settings.batch_size
        
    if not marqo_client:
        return {"added": 0, "skipped": len(documents)}
    
    try:
        existing_ids = await get_existing_document_ids()
        documents_to_add = [doc for doc in documents if doc["_id"] not in existing_ids]
        
        if not documents_to_add:
            logger.info("✅ All documents already exist in index")
            return {"added": 0, "skipped": len(documents)}
        
        added_count = 0
        total_batches = (len(documents_to_add) + batch_size - 1) // batch_size
        successful_batches = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(documents_to_add))
            batch = documents_to_add[start_idx:end_idx]
            
            for attempt in range(settings.max_batch_retries):
                try:
                    results = marqo_client.index(settings.index_name).add_documents(
                        batch, 
                        tensor_fields=["content", "title"],
                        client_batch_size=min(10, len(batch))
                    )
                    added_count += len(batch)
                    successful_batches += 1
                    logger.info(f"✅ Batch {batch_num + 1}/{total_batches}: Added {len(batch)} documents")
                    break
                    
                except Exception as batch_error:
                    if attempt < settings.max_batch_retries - 1:
                        logger.warning(f"⚠️ Batch {batch_num + 1} failed (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        logger.error(f"❌ Batch {batch_num + 1} failed after {settings.max_batch_retries} attempts: {batch_error}")
        
        skipped_count = len(documents) - added_count
        logger.info(f"✅ Added {added_count} new documents, skipped {skipped_count} existing ones in {successful_batches}/{total_batches} batches")
        return {
            "added": added_count, 
            "skipped": skipped_count,
            "batches_processed": successful_batches,
            "total_batches": total_batches
        }
        
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        raise DocumentOperationError(f"Failed to add documents: {str(e)}")

# -------------------------
# Main Endpoints with Caching
# -------------------------
@app.get("/init-marqo-client", response_model=Dict[str, Any])
@marqo_circuit_breaker
@with_marqo_retry
async def init_marqo_with_client():
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")

    try:
        indexes = marqo_client.get_indexes()
        index_exists = any(idx.get('indexName') == settings.index_name for idx in indexes.get('results', []))

        action_taken = "none"
        
        if not index_exists:
            marqo_client.create_index(settings.index_name)
            logger.info(f"✅ Created new index: {settings.index_name}")
            
            if not await wait_for_index_ready():
                raise IndexNotReadyError("Index creation timeout - index not ready")
                
            action_taken = "created"
        else:
            logger.info(f"ℹ️ Index already exists: {settings.index_name}")
            action_taken = "exists"

        documents = []
        for topic, data in topic_data.items():
            documents.append({
                "_id": f"doc_{topic}",
                "content": data["long"],
                "topic": topic,
                "references": data["references"],
                "title": f"{topic.capitalize()} Assistance"
            })

        add_result = await safe_add_documents(documents)

        async with httpx.AsyncClient() as client:
            refresh_response = await client.post(f"{settings.marqo_url}/indexes/{settings.index_name}/refresh")

        return {
            "status": "success", 
            "index": settings.index_name,
            "action_taken": action_taken,
            "documents_added": add_result["added"],
            "documents_skipped": add_result["skipped"],
            "refresh_status": refresh_response.status_code,
            "message": "Index initialized safely without data loss",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error initializing index: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize index: {str(e)}"
        )

@app.post("/add-docs", response_model=AddDocsResponse)
@marqo_circuit_breaker
@with_marqo_retry
async def add_docs(input_data: DocsInput):
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")

    try:
        for doc in input_data.docs:
            if "_id" not in doc:
                doc["_id"] = f"custom_doc_{abs(hash(str(doc))) % (10**12)}"
        
        add_result = await safe_add_documents(input_data.docs)
        return {
            "status": "success", 
            "added": add_result["added"],
            "skipped": add_result["skipped"],
            "batches_processed": add_result.get("batches_processed", 0),
            "total_batches": add_result.get("total_batches", 0),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        if isinstance(e, HTTPException):
            raise
        raise DocumentOperationError(f"Failed to add documents: {str(e)}")

@app.get("/ask", response_model=QAResponse, responses={
    404: {"model": ErrorResponse, "description": "Topic not found"},
    503: {"model": ErrorResponse, "description": "Service unavailable"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
@marqo_circuit_breaker
@cache_response(ttl=300)
async def ask_question(
    topic: str = Query(..., description="Choose a topic: education, food, transport, healthcare"),
    detail: str = Query("short", description="Choose 'short' or 'long' answer"),
    query: Optional[str] = Query(None, description="Optional specific question for RAG")
):
    if query:
        if not marqo_client:
            if topic.lower() in topic_data:
                data = topic_data[topic.lower()]
                answer = data["long"] if detail == "long" else data["short"]
                answer, refs, related, truncated = smart_truncate_response(
                    answer, data["references"], data["related"]
                )
                return {
                    "question": query,
                    "answer": f"Search unavailable. {answer}",
                    "references": refs,
                    "related_questions": related,
                    "truncated": truncated
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Topic '{topic}' not found and search unavailable"
                )
        
        try:
            results = marqo_client.index(settings.index_name).search(
                q=query,
                filter_string=f"topic:{topic}",
                limit=3
            ) or {}
            
            hits = results.get('hits', [])
            if hits:
                rag_content = concatenate_rag_content(hits)
                references = extract_references_from_hits(hits)
                related_questions = [f"More about {topic} programs", f"How to apply for {topic} assistance"]
                
                answer, references, related_questions, truncated = smart_truncate_response(
                    rag_content, references, related_questions
                )
                
                return {
                    "question": query,
                    "answer": answer,
                    "references": references,
                    "related_questions": related_questions,
                    "truncated": truncated
                }
            else:
                if topic.lower() in topic_data:
                    data = topic_data[topic.lower()]
                    answer = data["long"] if detail == "long" else data["short"]
                    answer, refs, related, truncated = smart_truncate_response(
                        f"No specific results found. {answer}", 
                        data["references"], 
                        data["related"]
                    )
                    return {
                        "question": query,
                        "answer": answer,
                        "references": refs,
                        "related_questions": related,
                        "truncated": truncated
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No information found about '{query}' in topic '{topic}'"
                    )
        except Exception as e:
            logger.error(f"❌ Error running query: {e}")
            if topic.lower() in topic_data:
                data = topic_data[topic.lower()]
                answer = data["long"] if detail == "long" else data["short"]
                answer, refs, related, truncated = smart_truncate_response(
                    f"Search error. {answer}", 
                    data["references"], 
                    data["related"]
                )
                return {
                    "question": query,
                    "answer": answer,
                    "references": refs,
                    "related_questions": related,
                    "truncated": truncated
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Search failed and topic '{topic}' not found: {str(e)}"
                )
    
    if topic.lower() in topic_data:
        data = topic_data[topic.lower()]
        answer = data["long"] if detail == "long" else data["short"]
        answer, refs, related, truncated = smart_truncate_response(
            answer, data["references"], data["related"]
        )
        
        return {
            "question": f"Tell me about {topic} assistance",
            "answer": answer,
            "references": refs,
            "related_questions": related,
            "truncated": truncated
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic '{topic}' not found. Available topics: {', '.join(topic_data.keys())}"
        )

@app.get("/search", response_model=Dict[str, Any])
@marqo_circuit_breaker
@cache_response(ttl=300)
async def search_documents(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return", ge=1, le=50)
):
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")
    
    try:
        results = marqo_client.index(settings.index_name).search(
            q=q,
            limit=limit
        ) or {}
        
        hits = []
        for hit in results.get('hits', []):
            hits.append({
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "content": hit.get("content", "")[:500] + "..." if len(hit.get("content", "")) > 500 else hit.get("content", ""),
                "topic": hit.get("topic", ""),
                "title": hit.get("title", "")
            })
        
        return {
            "query": q,
            "hits": hits,
            "processing_time_ms": results.get("processingTimeMs", 0),
            "total_hits": len(hits)
        }
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/topics/{topic_name}", response_model=Dict[str, Any])
@cache_response(ttl=600)
async def get_topic_info(topic_name: str):
    if topic_name.lower() in topic_data:
        return {
            "topic": topic_name,
            "data": topic_data[topic_name.lower()]
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic '{topic_name}' not found. Available topics: {', '.join(topic_data.keys())}"
        )

@app.get("/topics", response_model=Dict[str, Any])
@cache_response(ttl=600)
async def list_topics():
    return {
        "topics": list(topic_data.keys()),
        "count": len(topic_data)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
