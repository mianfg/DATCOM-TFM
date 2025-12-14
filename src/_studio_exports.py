"""Graph exports for LangGraph Studio."""

from pathlib import Path
from src.agent.infrastructure.container import get_agent_container

# Get the agent container
agent_container = get_agent_container()

# Load sample documents into Qdrant for LangGraph Studio demo
try:
    resources_dir = Path("./resources/c3c4_splitter")
    sample_docs = [f for f in resources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".csv"]]
    
    if sample_docs:
        use_case = agent_container.rule_extraction_use_case()
        use_case.load_documents_only(sample_docs[:1])  # Load just first doc for demo
        print(f"✓ Loaded {len(sample_docs[:1])} sample documents into Qdrant for Studio")
except Exception as e:
    print(f"⚠️  Warning: Could not load sample documents: {e}")

# Export the rule extraction graph for LangGraph Studio
rule_extraction_graph = agent_container.rule_extraction_use_case()._ensure_workflow().graph
