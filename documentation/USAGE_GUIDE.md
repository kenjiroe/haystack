# Haystack Usage Guide (คู่มือการใช้งาน Haystack)

## 🎯 สารบัญ

1. [การติดตั้งและการตั้งค่า](#การติดตั้งและการตั้งค่า)
2. [แนวคิดพื้นฐาน](#แนวคิดพื้นฐาน)
3. [การสร้าง Component แรก](#การสร้าง-component-แรก)
4. [การสร้าง Pipeline](#การสร้าง-pipeline)
5. [การทำงานกับ Documents](#การทำงานกับ-documents)
6. [การสร้างระบบ RAG](#การสร้างระบบ-rag)
7. [การใช้งาน LLM](#การใช้งาน-llm)
8. [การทดสอบ](#การทดสอบ)
9. [การใช้งานขั้นสูง](#การใช้งานขั้นสูง)
10. [การแก้ไขปัญหา](#การแก้ไขปัญหา)

---

## การติดตั้งและการตั้งค่า

### 🔧 การติดตั้งพื้นฐาน

```bash
# ติดตั้ง Haystack
pip install haystack-ai

# สำหรับการพัฒนา
pip install -e .
pip install hatch
python -m hatch env create test
```

### 🌐 การตั้งค่า API Keys

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Cohere
export COHERE_API_KEY="your-api-key-here"

# HuggingFace
export HUGGINGFACE_API_TOKEN="your-token-here"
```

### ✅ การตรวจสอบการติดตั้ง

```python
import haystack
print(f"Haystack version: {haystack.__version__}")

# ทดสอบ import พื้นฐาน
from haystack import component, Pipeline, Document
print("✅ Haystack ready to use!")
```

---

## แนวคิดพื้นฐาน

### 🧩 Components
**Components** คือหน่วยการทำงานพื้นฐานใน Haystack:
- รับ input ประมวลผล และส่ง output
- สามารถเชื่อมต่อกันเป็น pipeline ได้
- มี built-in components หรือสร้างเองได้

### 🔗 Pipelines
**Pipelines** คือการเชื่อมต่อ components เข้าด้วยกัน:
- กำหนด flow การทำงาน
- จัดการการส่งข้อมูลระหว่าง components
- สามารถมี branches และ loops ได้

### 📄 Documents
**Documents** คือโครงสร้างข้อมูลสำหรับเก็บข้อความ:
- มี content และ metadata
- ใช้สำหรับการค้นหาและการประมวลผล
- สามารถแปลงจากไฟล์ต่างๆ ได้

---

## การสร้าง Component แรก

### 📝 Component พื้นฐาน

```python
from haystack import component

@component
class TextProcessor:
    @component.output_types(processed_text=str, word_count=int)
    def run(self, text: str) -> dict:
        processed = text.strip().upper()
        word_count = len(text.split())
        
        return {
            "processed_text": processed,
            "word_count": word_count
        }

# การใช้งาน
processor = TextProcessor()
result = processor.run(text="hello world")
print(result)  # {'processed_text': 'HELLO WORLD', 'word_count': 2}
```

### 🎛️ Component แบบมี Configuration

```python
@component
class CustomAnalyzer:
    def __init__(self, min_word_length: int = 3, case_sensitive: bool = False):
        self.min_word_length = min_word_length
        self.case_sensitive = case_sensitive
    
    @component.output_types(filtered_words=list, stats=dict)
    def run(self, text: str) -> dict:
        words = text.split()
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        
        filtered = [w for w in words if len(w) >= self.min_word_length]
        
        return {
            "filtered_words": filtered,
            "stats": {
                "total_words": len(words),
                "filtered_words": len(filtered),
                "average_length": sum(len(w) for w in filtered) / len(filtered) if filtered else 0
            }
        }
```

### 🔄 Async Component

```python
import asyncio

@component
class AsyncProcessor:
    @component.output_types(result=str)
    async def run(self, text: str, delay: float = 1.0) -> dict:
        await asyncio.sleep(delay)
        return {"result": f"Processed after {delay}s: {text}"}
```

---

## การสร้าง Pipeline

### 🚇 Pipeline เส้นตรง

```python
from haystack import Pipeline

# สร้าง components
processor = TextProcessor()
analyzer = CustomAnalyzer(min_word_length=4)

# สร้าง pipeline
pipeline = Pipeline()
pipeline.add_component("processor", processor)
pipeline.add_component("analyzer", analyzer)

# เชื่อมต่อ components
pipeline.connect("processor.processed_text", "analyzer.text")

# รัน pipeline
result = pipeline.run({
    "processor": {"text": "Hello wonderful world of programming"}
})

print(result["analyzer"]["filtered_words"])
```

### 🔀 Pipeline แบบมี Branches

```python
@component
class Router:
    @component.output_types(short_text=str, long_text=str)
    def run(self, text: str, threshold: int = 50) -> dict:
        if len(text) <= threshold:
            return {"short_text": text, "long_text": ""}
        else:
            return {"short_text": "", "long_text": text}

@component 
class ShortTextHandler:
    @component.output_types(result=str)
    def run(self, text: str) -> dict:
        return {"result": f"Short: {text.upper()}"}

@component
class LongTextHandler:
    @component.output_types(result=str)
    def run(self, text: str) -> dict:
        return {"result": f"Long: {text[:20]}..."}

# สร้าง branched pipeline
pipeline = Pipeline()
pipeline.add_component("router", Router())
pipeline.add_component("short_handler", ShortTextHandler())
pipeline.add_component("long_handler", LongTextHandler())

pipeline.connect("router.short_text", "short_handler.text")
pipeline.connect("router.long_text", "long_handler.text")
```

### 🔄 Pipeline แบบมี Loop

```python
@component
class IterativeImprover:
    @component.output_types(improved_text=str, should_continue=bool)
    def run(self, text: str, iteration: int = 0, max_iterations: int = 3) -> dict:
        # ปรับปรุงข้อความ
        improved = text.replace("  ", " ").strip()
        
        # ตรวจสอบว่าต้องทำต่อไหม
        should_continue = iteration < max_iterations and "  " in text
        
        return {
            "improved_text": improved,
            "should_continue": should_continue
        }

# Pipeline ที่มี feedback loop
pipeline = Pipeline()
pipeline.add_component("improver", IterativeImprover())
# เชื่อมต่อ output กลับไป input สำหรับ loop
pipeline.connect("improver.improved_text", "improver.text")
```

---

## การทำงานกับ Documents

### 📄 การสร้าง Documents

```python
from haystack import Document

# สร้าง document พื้นฐาน
doc1 = Document(
    content="Haystack is an amazing framework for building search systems.",
    meta={"source": "documentation", "category": "technical"}
)

# สร้าง document จาก dictionary
doc_data = {
    "content": "Machine learning revolutionizes data processing.",
    "meta": {"author": "AI Expert", "date": "2024-01-15"}
}
doc2 = Document(**doc_data)

# สร้าง document list
documents = [doc1, doc2]
print(f"Created {len(documents)} documents")
```

### 🔧 การประมวลผล Documents

```python
from haystack.components.preprocessors import DocumentSplitter

# แยก documents เป็นส่วนเล็ก
splitter = DocumentSplitter(
    split_by="word",      # แยกตามคำ
    split_length=50,      # 50 คำต่อส่วน
    split_overlap=10      # overlap 10 คำ
)

# สร้าง pipeline สำหรับประมวลผล documents
doc_pipeline = Pipeline()
doc_pipeline.add_component("splitter", splitter)

result = doc_pipeline.run({
    "splitter": {"documents": documents}
})

split_docs = result["splitter"]["documents"]
print(f"Split into {len(split_docs)} chunks")
```

### 🗄️ การจัดเก็บ Documents

```python
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter

# สร้าง document store
document_store = InMemoryDocumentStore()

# สร้าง writer component
writer = DocumentWriter(document_store=document_store)

# สร้าง indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("writer", writer)
indexing_pipeline.connect("splitter", "writer")

# Index documents
result = indexing_pipeline.run({
    "splitter": {"documents": documents}
})

print(f"Indexed {result['writer']['documents_written']} documents")
```

---

## การสร้างระบบ RAG

### 🔍 Retrieval (การค้นหา)

```python
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# สร้าง retriever
retriever = InMemoryBM25Retriever(document_store=document_store)

# ทดสอบการค้นหา
search_pipeline = Pipeline()
search_pipeline.add_component("retriever", retriever)

search_result = search_pipeline.run({
    "retriever": {
        "query": "machine learning framework",
        "top_k": 3
    }
})

retrieved_docs = search_result["retriever"]["documents"]
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}. Score: {doc.score:.3f}")
    print(f"   Content: {doc.content[:100]}...")
```

### 📝 Generation (การสร้างคำตอบ)

```python
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

# สร้าง prompt template
template = """
Based on the following context, answer the question.

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}

Answer:
"""

# สร้าง components
prompt_builder = PromptBuilder(template=template)
generator = OpenAIGenerator(
    model="gpt-3.5-turbo-instruct",
    generation_kwargs={"max_tokens": 200}
)

# สร้าง RAG pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("generator", generator)

# เชื่อมต่อ components
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
```

### 🤖 การใช้งาน RAG Pipeline

```python
def ask_question(question: str, top_k: int = 3):
    """ถามคำถามผ่านระบบ RAG"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return None
    
    try:
        result = rag_pipeline.run({
            "retriever": {
                "query": question,
                "top_k": top_k
            },
            "prompt_builder": {
                "question": question
            }
        })
        
        answer = result["generator"]["replies"][0]
        retrieved_docs = result["retriever"]["documents"]
        
        print(f"❓ Question: {question}")
        print(f"✅ Answer: {answer}")
        print(f"📚 Based on {len(retrieved_docs)} documents")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ทดสอบ
ask_question("What is Haystack used for?")
ask_question("How does machine learning work?")
```

---

## การใช้งาน LLM

### 🎯 Chat-based Generation

```python
from haystack.components.generators import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

# สร้าง chat generator
chat_generator = OpenAIChatGenerator(
    model="gpt-3.5-turbo",
    generation_kwargs={"temperature": 0.7}
)

# สร้าง chat messages
messages = [
    ChatMessage.from_system("You are a helpful AI assistant specialized in explaining technical concepts."),
    ChatMessage.from_user("Explain how neural networks work in simple terms.")
]

# สร้าง chat pipeline
chat_pipeline = Pipeline()
chat_pipeline.add_component("chat_generator", chat_generator)

# ใช้งาน
result = chat_pipeline.run({
    "chat_generator": {"messages": messages}
})

response = result["chat_generator"]["replies"][0]
print(response.content)
```

### 🛠️ Custom LLM Wrapper

```python
@component
class CustomLLMWrapper:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
    @component.output_types(reply=str, metadata=dict)
    def run(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 100
    ) -> dict:
        """Custom LLM wrapper with additional processing"""
        
        # เพิ่ม context และ formatting
        enhanced_prompt = f"""
        Context: You are an expert assistant.
        
        User Query: {prompt}
        
        Instructions: Provide a clear and concise answer.
        
        Response:
        """
        
        try:
            # ใช้ OpenAI (ต้องมี API key)
            from openai import OpenAI
            client = OpenAI()
            
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=enhanced_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            reply = response.choices[0].text.strip()
            metadata = {
                "model": self.model_name,
                "tokens_used": response.usage.total_tokens,
                "temperature": temperature
            }
            
            return {"reply": reply, "metadata": metadata}
            
        except Exception as e:
            return {
                "reply": f"Error: {str(e)}",
                "metadata": {"error": True}
            }
```

---

## การทดสอบ

### 🧪 Unit Testing Components

```python
import pytest
from haystack import component

@component
class SimpleAdder:
    @component.output_types(result=int)
    def run(self, a: int, b: int) -> dict:
        return {"result": a + b}

def test_simple_adder():
    """ทดสอบ component พื้นฐาน"""
    adder = SimpleAdder()
    result = adder.run(a=5, b=3)
    
    assert result["result"] == 8
    assert isinstance(result["result"], int)

def test_simple_adder_negative():
    """ทดสอบกับเลขลบ"""
    adder = SimpleAdder()
    result = adder.run(a=-5, b=3)
    
    assert result["result"] == -2

# รันเทสต์
# python -m pytest test_components.py -v
```

### 🔗 Testing Pipelines

```python
def test_text_processing_pipeline():
    """ทดสอบ pipeline สมบูรณ์"""
    processor = TextProcessor()
    analyzer = CustomAnalyzer(min_word_length=3)
    
    pipeline = Pipeline()
    pipeline.add_component("processor", processor)
    pipeline.add_component("analyzer", analyzer)
    pipeline.connect("processor.processed_text", "analyzer.text")
    
    result = pipeline.run({
        "processor": {"text": "hello world"}
    })
    
    # ตรวจสอบผลลัพธ์
    assert "analyzer" in result
    assert "filtered_words" in result["analyzer"]
    assert len(result["analyzer"]["filtered_words"]) >= 0

@pytest.mark.integration
def test_rag_pipeline_integration():
    """ทดสอบ RAG pipeline (ต้องมี API key)"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # ทดสอบ RAG pipeline
    documents = [Document(content="Test content about AI")]
    
    # TODO: สร้างและทดสอบ RAG pipeline
    assert True  # placeholder
```

### 📊 การรันเทสต์

```bash
# รันเทสต์ทั้งหมด
python -m hatch run test:unit

# รันเทสต์เฉพาะไฟล์
python -m hatch run test:pytest test_components.py -v

# รันเทสต์ integration
python -m hatch run test:integration

# รันเทสต์พร้อม coverage
python -m hatch run test:pytest --cov=haystack --cov-report=html
```

---

## การใช้งานขั้นสูง

### 🔧 Custom Document Store

```python
from haystack.document_stores.types import DocumentStore
from typing import List, Optional

class CustomDocumentStore(DocumentStore):
    """Custom document store implementation"""
    
    def __init__(self):
        self.documents = {}
        self.next_id = 1
    
    def write_documents(self, documents: List[Document]) -> int:
        """เขียน documents ลง store"""
        written = 0
        for doc in documents:
            if not doc.id:
                doc.id = str(self.next_id)
                self.next_id += 1
            self.documents[doc.id] = doc
            written += 1
        return written
    
    def filter_documents(self, filters: Optional[dict] = None) -> List[Document]:
        """กรอง documents ตามเงื่อนไข"""
        docs = list(self.documents.values())
        
        if filters:
            # ใช้ logic กรองตาม filters
            pass
            
        return docs
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """ลบ documents"""
        for doc_id in document_ids:
            self.documents.pop(doc_id, None)
```

### 🎛️ Pipeline Configuration

```python
# บันทึก pipeline configuration
pipeline_config = {
    "components": {
        "retriever": {
            "type": "InMemoryBM25Retriever",
            "init_parameters": {
                "document_store": {"type": "InMemoryDocumentStore"}
            }
        },
        "prompt_builder": {
            "type": "PromptBuilder", 
            "init_parameters": {
                "template": "Context: {{documents}}\nQuestion: {{question}}\nAnswer:"
            }
        }
    },
    "connections": [
        {"sender": "retriever.documents", "receiver": "prompt_builder.documents"}
    ]
}

# บันทึกเป็นไฟล์
import json
with open("pipeline_config.json", "w") as f:
    json.dump(pipeline_config, f, indent=2)
```

### 🚀 Performance Optimization

```python
@component
class CachedRetriever:
    """Retriever พร้อม caching"""
    
    def __init__(self, base_retriever, cache_size: int = 1000):
        self.base_retriever = base_retriever
        self.cache = {}
        self.cache_size = cache_size
    
    @component.output_types(documents=List[Document])
    def run(self, query: str, top_k: int = 10) -> dict:
        # ตรวจสอบ cache
        cache_key = f"{query}:{top_k}"
        if cache_key in self.cache:
            return {"documents": self.cache[cache_key]}
        
        # ค้นหาจริง
        result = self.base_retriever.run(query=query, top_k=top_k)
        documents = result["documents"]
        
        # บันทึกลง cache
        if len(self.cache) >= self.cache_size:
            # ลบ entry เก่าสุด
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = documents
        return {"documents": documents}

# ใช้งาน Async Pipeline
from haystack.core.pipeline import AsyncPipeline

async def run_async_pipeline():
    """รัน pipeline แบบ async"""
    pipeline = AsyncPipeline()
    
    # เพิ่ม async components
    async_processor = AsyncProcessor()
    pipeline.add_component("processor", async_processor)
    
    result = await pipeline.run({
        "processor": {"text": "Hello async world", "delay": 0.5}
    })
    
    return result
```

---

## การแก้ไขปัญหา

### 🐛 ปัญหาที่พบบ่อย

#### 1. Import Errors
```bash
# ปัญหา: ModuleNotFoundError
❌ ModuleNotFoundError: No module named 'haystack'

# แก้ไข:
pip install haystack-ai
# หรือ
pip install -e .
```

#### 2. API Key Issues
```bash
# ปัญหา: API key ไม่ถูกต้อง
❌ InvalidAPIKey: Invalid API key provided

# แก้ไข:
export OPENAI_API_KEY="sk-your-actual-key-here"
echo $OPENAI_API_KEY  # ตรวจสอบ
```

#### 3. Pipeline Connection Errors
```python
# ปัญหา: การเชื่อมต่อ pipeline ผิด
❌ PipelineConnectError: No component named 'invalid_component'

# แก้ไข: ตรวจสอบชื่อ component
pipeline.add_component("correct_name", component)
pipeline.connect("correct_name.output", "next_component.input")
```

#### 4. Component Input/Output Mismatch
```python
# ปัญหา: Type mismatch
❌ ComponentError: Expected 'str' but got 'list'

# แก้ไข: ตรวจสอบ input/output types
@component.output_types(result=str)  # ระบุ type ชัดเจน
def run(self, input_data: str) -> dict:  # ใช้ type hints
    return {"result": str(input_data)}
```

### 🔍 Debugging Tips

#### 1. Enable Detailed Logging
```python
import logging

# เปิด debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("haystack")
logger.setLevel(logging.DEBUG)
```

#### 2. Pipeline Visualization
```python
# ดู pipeline structure
pipeline.show()  # แสดง graph
print(pipeline.get_component_names())  # แสดงรายชื่อ components
```

#### 3. Step-by-step Debugging
```python
# รัน component ทีละตัว
component_result = component.run(**inputs)
print(f"Component output: {component_result}")

# ตรวจสอบ intermediate results
result = pipeline.run(inputs, include_outputs_from=["component1", "component2"])
```

### 📚 Resources สำหรับความช่วยเหลือ

1. **Official Documentation**: https://docs.haystack.deepset.ai/
2. **GitHub Issues**: https://github.com/deepset-ai/haystack/issues
3. **Discord Community**: https://discord.com/invite/VBpFzsgRVF
4. **Stack Overflow**: Tag `haystack`
5. **Examples**: https://haystack.deepset.ai/tutorials

### ⚡ Performance Tips

```python
# 1. ใช้ batch processing
documents = [doc1, doc2, doc3, ...]  # ส่งเป็น batch

# 2. ใช้ async สำหร�rงาน I/O intensive
async def process_many_queries(queries):
    tasks = [pipeline.run_async({"query": q}) for q in queries]
    return await asyncio.gather(*tasks)

# 3. ใช้ caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_embedding(text):
    return generate_embedding(text)

# 4. ปรับแต่ง batch sizes
splitter = DocumentSplitter(split_length=100)  # เล็กลง = เร็วขึ้น
```

---

## 🎉 สรุป

Haystack เป็น framework ที่ทรงพลังสำหรับสร้างแอปพลิเคชัน AI และระบบค้นหา ด้วยแนวคิดของ **Components** และ **Pipelines** คุณสามารถ:

✅ **สร้าง RAG systems** ที่ซับซ้อน  
✅ **ทำงานกับ LLMs** หลากหลายตัว  
✅ **ประมวลผล documents** อย่างมีประสิทธิภาพ  
✅ **ทดสอบและ debug** ได้ง่าย  
✅ **Scale** ตามความต้องการ  

### 🚀 Next Steps
1. ทดลองรันตัวอย่างในไฟล์ `basic_example.py` และ `advanced_example.py`
2. สร้าง custom component สำหรับงานเฉพาะของคุณ
3. ลองใช้ LLM providers ต่างๆ
4. สร้าง RAG system สำหรับข้อมูลของคุณ
5. ศึกษา advanced features เช่น agents และ tools

---

**Happy Building with Haystack! 🏗️🤖**