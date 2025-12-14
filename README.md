# DATCOM-TFM: Industrial Rule Extraction & Streaming Evaluation

An end-to-end system for extracting operational rules from industrial documentation using LLMs and evaluating them in real-time against streaming sensor data.

## Key Features

- **LLM-based Rule Extraction**: Automatically extract operational rules from technical documentation using LangGraph workflows
- **Rule Consolidation**: Intelligent deduplication, merging, and simplification of extracted rules
- **Real-time Streaming Engine**: Evaluate rules against sensor data with O(1) complexity using River
- **REST API**: Full-featured API for document management, rule extraction, and consolidation
- **Structured Output**: Pydantic-validated LLM responses for reliable rule generation

---

## Architecture Overview

The system is organized into three independent layers that communicate through well-defined interfaces:

```mermaid
graph TB
    subgraph agent_layer [Agent Layer]
        EW[Extraction Workflow]
        CW[Consolidation Workflow]
        LLM[LLM Provider]
        VS[Vector Store]
    end

    subgraph api_layer [API Layer]
        API[FastAPI Application]
        DB[(PostgreSQL)]
        QD[(Qdrant)]
        JE[Job Executor]
    end

    subgraph streaming_layer [Streaming Layer]
        RC[RuleChecker]
        BS[BufferedStatus]
        ES[EventSink]
    end

    API --> JE
    JE --> EW
    JE --> CW
    EW --> LLM
    EW --> VS
    CW --> LLM
    API --> DB
    API --> QD
    
    RC --> BS
    BS --> ES
    DB -.->|DatabaseRuleLoader| RC
```

| Layer | Purpose | Key Technology |
|-------|---------|----------------|
| **Agent** | Rule extraction and consolidation workflows | LangGraph, LangChain |
| **API** | Document management, job orchestration | FastAPI, SQLAlchemy |
| **Streaming** | Real-time rule evaluation | River, Pandas |

---

## Part 1: Rule Extraction and Consolidation

### Extraction Workflow

The extraction workflow processes document chunks through a multi-stage LangGraph pipeline:

```mermaid
flowchart LR
    A[gather_context] --> B[ground_chunk]
    B --> C[extract_rules]
    C --> D[resolve_sensors]
    D --> E[parse_time]
    E --> F((END))
    
    B -.->|optional| C
```

**Stages:**

1. **gather_context**: Retrieves relevant context from Qdrant vector store
2. **ground_chunk** *(optional)*: Enriches context with external knowledge via web search
3. **extract_rules**: LLM generates Python rules using structured output
4. **resolve_sensors**: Maps natural language sensor descriptions to sensor IDs
5. **parse_time**: Converts time expressions to the `status.get()` grammar

**Rule Format:**

Rules are Python functions using the `status.get()` API:

```python
def column_high_pressure_alert(status) -> str | None:
    """Alert when column pressure exceeds safety limit."""
    current = status.get("PT-101", "0")           # Current value
    avg_1h = status.get("PT-101", "1h:", "mean")  # 1-hour rolling mean
    
    if current and current > 15.5 and avg_1h and avg_1h > 14.0:
        return "column_high_pressure_alert"
    return None
```

### Consolidation Workflow

After extraction, rules are consolidated to remove redundancy and improve quality:

```mermaid
flowchart LR
    A[analyze_rules] --> B[consolidate_with_llm]
    B --> C[verify_consolidated]
    C --> D[finalize]
    D --> E((END))
```

**Consolidation Actions:**

| Action | Description |
|--------|-------------|
| `REMOVE` | Delete duplicate or invalid rules |
| `MERGE` | Combine similar rules into a single comprehensive rule |
| `SIMPLIFY` | Reduce complexity while preserving semantics |

### Agent Layer Files

| File | Description |
|------|-------------|
| [`src/agent/application/extraction_workflow.py`](src/agent/application/extraction_workflow.py) | LangGraph extraction workflow with sensor resolution and time parsing |
| [`src/agent/application/consolidation_workflow.py`](src/agent/application/consolidation_workflow.py) | LangGraph consolidation workflow for rule optimization |
| [`src/agent/application/extraction_use_case.py`](src/agent/application/extraction_use_case.py) | High-level extraction orchestration |
| [`src/agent/application/consolidation_use_case.py`](src/agent/application/consolidation_use_case.py) | High-level consolidation orchestration |
| [`src/agent/domain/models.py`](src/agent/domain/models.py) | Domain models: `PythonRule`, `ExtractedRules`, `SensorMappings`, `RuleConsolidations` |
| [`src/agent/application/extraction_prompts.py`](src/agent/application/extraction_prompts.py) | Extraction-related LLM prompts |
| [`src/agent/application/consolidation_prompts.py`](src/agent/application/consolidation_prompts.py) | Consolidation LLM prompts |
| [`src/agent/domain/time/`](src/agent/domain/time/) | Time expression parsing: `TimeDelta`, `TimeDeltaInterval`, `Statistic` |

---

## Part 2: Streaming Engine

The streaming engine evaluates rules against time-series sensor data with **O(1) per-update complexity**.

### Architecture

```mermaid
flowchart LR
    subgraph input [Input]
        DF[DataFrame]
        RT[Real-time Stream]
    end
    
    subgraph checker [RuleChecker]
        BS[BufferedStatus]
        RV[River Rolling Stats]
        RC[Rule Execution]
    end
    
    subgraph output [Output]
        IM[InMemoryEventSink]
        CSV[CSVEventSink]
    end
    
    DF --> BS
    RT --> BS
    BS --> RV
    RV --> RC
    RC --> IM
    RC --> CSV
```

### Key Concepts

**1. BufferedStatus with River**

The `BufferedStatus` class implements the `status.get()` API using [River](https://riverml.xyz/) for efficient rolling statistics:

```python
# Initialize checker with rules
checker = RuleChecker(rules=rules, config=CheckerConfig(granularity="1min"))

# Process DataFrame in single pass
events_df = checker.check_dataframe(sensor_data)
```

**2. Time Expression Grammar**

| Expression | Type | Description | Example |
|------------|------|-------------|---------|
| `"0"` | Point | Current value | `status.get("PT-101", "0")` |
| `"5m"` | Point | Value 5 minutes ago | `status.get("PT-101", "5m")` |
| `"10m:"` | Interval | Last 10 minutes to now | `status.get("PT-101", "10m:", "mean")` |
| `"1h:5m"` | Interval | From 1 hour ago to 5 minutes ago | `status.get("PT-101", "1h:5m", "max")` |

**3. Supported Statistics**

| Statistic | Description | River Class |
|-----------|-------------|-------------|
| `mean` | Rolling mean | `Rolling(Mean())` |
| `max` | Rolling maximum | `RollingMax` |
| `min` | Rolling minimum | `RollingMin` |
| `std` | Rolling standard deviation | `Rolling(Var())` → sqrt |
| `variance` | Rolling variance | `Rolling(Var())` |
| `sum` | Rolling sum | `Rolling(Sum())` |
| `quantile` | Rolling quantile | `RollingQuantile` |
| `iqr` | Interquartile range | `RollingIQR` |
| `mode` | Rolling mode | `RollingMode` |
| `abs_max` | Rolling absolute maximum | `RollingAbsMax` |

### Streaming Layer Files

| File | Description |
|------|-------------|
| [`src/streaming/application/rule_checker.py`](src/streaming/application/rule_checker.py) | Main `RuleChecker` class for rule evaluation |
| [`src/streaming/infrastructure/buffered_status.py`](src/streaming/infrastructure/buffered_status.py) | River-based `BufferedStatus` with O(1) rolling windows |
| [`src/streaming/infrastructure/event_sink.py`](src/streaming/infrastructure/event_sink.py) | Event storage: `InMemoryEventSink`, `CSVEventSink` |
| [`src/streaming/infrastructure/rule_loader.py`](src/streaming/infrastructure/rule_loader.py) | `DatabaseRuleLoader` to bridge API and Streaming layers |
| [`src/streaming/domain/models.py`](src/streaming/domain/models.py) | Domain models: `RuleEvent`, `RuleMetadata`, `CheckerConfig` |

---

## REST API

The API layer provides endpoints for document management and rule extraction:

| Router | Prefix | Description |
|--------|--------|-------------|
| Collections | `/collections` | Create, list, update, delete document collections |
| Documents | `/documents` | Upload, list, delete documents within collections |
| Sensors | `/sensors` | Import and manage sensor definitions |
| Processing | `/processing` | Create and execute extraction jobs |
| Rules | `/rules` | List, filter, and manage extracted rules |
| Consolidation | `/consolidation` | Create and execute consolidation jobs |

**API Documentation:** Available at `/docs` (Swagger UI) when the server is running.

### API Layer Files

| File | Description |
|------|-------------|
| [`src/api/main_app.py`](src/api/main_app.py) | FastAPI application entry point |
| [`src/api/routers/`](src/api/routers/) | API route handlers |
| [`src/api/application/job_executor.py`](src/api/application/job_executor.py) | Extraction job execution |
| [`src/api/application/consolidation_executor.py`](src/api/application/consolidation_executor.py) | Consolidation job execution |
| [`src/api/application/qdrant_sync_service.py`](src/api/application/qdrant_sync_service.py) | Qdrant vector store synchronization |
| [`src/api/domain/models.py`](src/api/domain/models.py) | SQLAlchemy models: `Collection`, `Document`, `Rule`, `Job` |
| [`src/api/infrastructure/repositories.py`](src/api/infrastructure/repositories.py) | Data access layer |

---

## Project Structure

```
src/
├── agent/                      # Rule extraction and consolidation
│   ├── application/            # Workflows and use cases
│   │   ├── extraction_workflow.py
│   │   ├── consolidation_workflow.py
│   │   ├── extraction_use_case.py
│   │   ├── consolidation_use_case.py
│   │   ├── extraction_prompts.py
│   │   └── consolidation_prompts.py
│   ├── domain/                 # Domain models and protocols
│   │   ├── models.py
│   │   ├── protocols.py
│   │   └── time/               # Time expression parsing
│   └── infrastructure/         # External integrations
│       ├── llm.py
│       ├── vector_stores.py
│       └── embeddings.py
│
├── api/                        # REST API layer
│   ├── main_app.py
│   ├── routers/                # API endpoints
│   ├── application/            # Services and executors
│   ├── domain/                 # Models and schemas
│   └── infrastructure/         # Database, storage
│
├── streaming/                  # Real-time rule evaluation
│   ├── application/
│   │   └── rule_checker.py
│   ├── domain/
│   │   ├── models.py
│   │   └── protocols.py
│   └── infrastructure/
│       ├── buffered_status.py
│       ├── event_sink.py
│       └── rule_loader.py
│
└── config.py                   # Application configuration

demo/                           # Demonstration notebooks
├── agent.ipynb                 # Rule extraction and consolidation
├── streaming.ipynb             # Streaming engine usage
└── evaluation.ipynb            # Performance benchmarks
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- PostgreSQL 15+
- Qdrant vector database
- OpenAI API key (or alternative LLM provider)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DATCOM-TFM.git
cd DATCOM-TFM

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Start Services

```bash
# Start PostgreSQL and Qdrant
docker compose up -d

# Optional: Start Langfuse for observability
docker compose -f docker-compose.langfuse.yml up -d
```

### Environment Configuration

Create a `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/datcom

# LLM Provider
OPENAI_API_KEY=your-api-key

# Qdrant
QDRANT_URL=http://localhost:6333

# Optional: Langfuse
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

### Run the API

```bash
# Development
uvicorn src.api.main_app:app --reload

# Production
uvicorn src.api.main_app:app --host 0.0.0.0 --port 8000
```

---

## Demo Notebooks

| Notebook | Description |
|----------|-------------|
| [`demo/agent.ipynb`](demo/agent.ipynb) | Complete walkthrough of rule extraction and consolidation workflows |
| [`demo/streaming.ipynb`](demo/streaming.ipynb) | Streaming engine usage with synthetic data and visualizations |
| [`demo/evaluation.ipynb`](demo/evaluation.ipynb) | Performance benchmarks: extraction accuracy, O(1) verification, memory stability |

---

## License

MIT License - see [LICENSE](LICENSE) for details.

