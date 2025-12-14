"""LangGraph workflows for Agent layer - moved from src/application/workflows/"""

import ast
from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from loguru import logger

from src.agent.application.extraction_prompts import (
    grounding_prompt,
    rule_extraction_prompt,
    sensor_resolution_json_fallback,
    sensor_resolution_prompt,
    time_parsing_json_fallback,
    time_parsing_prompt,
)
from src.agent.domain.models import AgentState, ChunkModel, ExtractedRules, SensorMappings, TimeMapping
from src.agent.domain.protocols import LLMProvider, VectorStoreProvider


class StatusCallExtractor(ast.NodeVisitor):
    """AST visitor to extract status.get() calls from Python code."""

    def __init__(self) -> None:
        """Initialize the extractor with an empty calls list."""
        self.calls: list[tuple[str | None, str | None, str | None]] = []

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visit Call nodes and extract status.get() calls.

        Args:
            node: AST Call node to visit
        """
        # Check if this is a status.get(...) call
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "status"
        ):
            # Extract arguments
            sensor_id: str | None = None
            time_expr: str | None = None
            statistic: str | None = None

            if len(node.args) >= 1:
                sensor_id = ast.literal_eval(node.args[0]) if isinstance(node.args[0], ast.Constant) else None

            if len(node.args) >= 2:
                time_expr = ast.literal_eval(node.args[1]) if isinstance(node.args[1], ast.Constant) else None

            if len(node.args) >= 3:
                statistic = ast.literal_eval(node.args[2]) if isinstance(node.args[2], ast.Constant) else None

            self.calls.append((sensor_id, time_expr, statistic))

        # Continue visiting child nodes
        self.generic_visit(node)


class RuleExtractionWorkflow:
    """
    LangGraph workflow for extracting operational rules from a single document chunk.

    Supports multi-collection isolation: each workflow run can specify which
    Qdrant collection to search for context (enables project independence).

    Simplified workflow: gather_context → extract_rules → END
    Each graph execution processes exactly one chunk.
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        llm_provider: LLMProvider,
        min_suggested_searches: int = 2,
        max_suggested_searches: int = 4,
        hard_limit_searches: int = 4,
    ):
        """
        Initialize the workflow.

        Args:
            vector_store: Vector store provider (creates collection-specific retrievers)
            llm_provider: LLM provider
            min_suggested_searches: Minimum searches to suggest to LLM
            max_suggested_searches: Maximum searches to suggest to LLM
            hard_limit_searches: Hard limit on searches performed (overrides LLM)
        """
        self.vector_store = vector_store
        self.llm = llm_provider.get_llm()

        # Grounding configuration
        self.grounding_min = min_suggested_searches
        self.grounding_max = max_suggested_searches
        self.grounding_limit = hard_limit_searches

        # Try to create structured output LLMs (not all providers support this)
        self.structured_llm = None
        self.sensor_mapping_llm = None
        self.time_mapping_llm = None

        try:
            self.structured_llm = self.llm.with_structured_output(ExtractedRules)
            self.sensor_mapping_llm = self.llm.with_structured_output(SensorMappings)
            self.time_mapping_llm = self.llm.with_structured_output(TimeMapping)
            logger.info("✓ Structured output supported by LLM provider")
        except Exception as e:
            logger.warning(f"Structured output not supported by LLM provider: {e}")
            logger.info("Will use fallback JSON parsing")

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph workflow with conditional grounding stage, sensor resolution, and time parsing."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("ground_with_search", self._ground_with_search)
        workflow.add_node("extract_rules", self._extract_rules)
        workflow.add_node("resolve_sensors", self._resolve_sensors)
        workflow.add_node("parse_time_expressions", self._parse_time_expressions)

        # Conditional routing after context gathering
        workflow.add_conditional_edges(
            "gather_context", self._should_ground, {"ground": "ground_with_search", "skip": "extract_rules"}
        )

        # If grounding was used, go to extract_rules
        workflow.add_edge("ground_with_search", "extract_rules")

        # After extraction, resolve sensors
        workflow.add_edge("extract_rules", "resolve_sensors")

        # After sensor resolution, parse time expressions
        workflow.add_edge("resolve_sensors", "parse_time_expressions")

        # After time parsing, verify rules
        workflow.add_node("verify_rules", self._verify_rules)
        workflow.add_edge("parse_time_expressions", "verify_rules")

        # End after verification
        workflow.add_edge("verify_rules", END)

        # Set entry point
        workflow.set_entry_point("gather_context")

        return workflow.compile()

    async def arun(
        self,
        chunk,
        collection_name: str,
        collection_id: int,
        sensors: list[dict],
        use_grounding: bool = True,
        config: dict | None = None,
    ) -> dict:
        """
        Async version of run method for parallel execution.

        Args:
            chunk: Document chunk to process
            collection_name: Qdrant collection name
            collection_id: Collection ID for sensor lookup
            sensors: Available sensors for resolution
            use_grounding: Whether to use web grounding
            config: Optional LangGraph config

        Returns:
            State dict with extracted rules and metadata
        """
        # Convert chunk to dict format if needed
        if hasattr(chunk, "to_dict"):
            chunk_dict = chunk.to_dict()
        elif hasattr(chunk, "page_content"):
            chunk_dict = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
            }
        else:
            chunk_dict = chunk

        initial_state = {
            "chunk": chunk_dict,
            "collection_name": collection_name,
            "collection_id": collection_id,
            "sensors": sensors,
            "use_grounding": use_grounding,
            "messages": [],
            "context": "",
            "grounding_info": "",
            "extracted_rules": {},
            "context_chunks": [],
            "grounding_searches": [],
            "metadata": {},
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result

    def _should_ground(self, state: AgentState) -> str:
        """
        Conditional routing: decide whether to use grounding.

        Returns:
            "ground" if use_grounding is True
            "skip" if use_grounding is False
        """
        use_grounding = state.get("use_grounding", True)  # Default to True for backwards compatibility

        if use_grounding:
            logger.info("🌐 Grounding enabled - will perform web search")
            return "ground"
        else:
            logger.info("⏭️  Grounding disabled - skipping web search")
            # Set empty grounding info
            state["grounding_info"] = ""
            return "skip"

    def _normalize_chunk(self, chunk: Document | dict | None) -> Document | None:
        """
        Normalize chunk input to a Document object.

        Handles multiple input formats:
        - Document object (from programmatic calls)
        - Dict (from LangGraph Studio or serialized state)
        - ChunkModel (Pydantic model)

        Args:
            chunk: Document, dict, or ChunkModel with 'page_content' and 'metadata'

        Returns:
            Document object or None if chunk is invalid
        """
        if chunk is None:
            return None

        # Already a Document object
        if isinstance(chunk, Document):
            return chunk

        # Pydantic ChunkModel
        if isinstance(chunk, ChunkModel):
            return chunk.to_document()

        # Dict from LangGraph Studio - validate and convert
        if isinstance(chunk, dict):
            try:
                # Use Pydantic for validation
                chunk_model = ChunkModel(**chunk)
                return chunk_model.to_document()
            except Exception as e:
                logger.error(f"Invalid chunk dict format: {e}")
                return None

        # Unknown type
        logger.warning(f"Unknown chunk type: {type(chunk)}")
        return None

    def _gather_context(self, state: AgentState) -> AgentState:
        """Gather relevant context from specified Qdrant collection for the input chunk."""
        chunk_input = state.get("chunk")
        collection_name = state.get("collection_name")

        # Normalize chunk (handles both Document objects and dicts from LangGraph Studio)
        chunk = self._normalize_chunk(chunk_input)

        if not chunk:
            logger.error("No chunk provided or invalid chunk format")
            state["context"] = ""
            state["metadata"] = {"error": "No chunk provided or invalid chunk format"}
            return state

        if not collection_name:
            logger.error("No collection_name provided")
            state["context"] = ""
            state["metadata"] = {"error": "No collection_name provided - specify which Qdrant collection to search"}
            return state

        chunk_text = chunk.page_content
        chunk_source = chunk.metadata.get("source", "unknown")

        logger.info(f"📚 Gathering context for chunk from: {chunk_source} (collection: {collection_name})")

        # Create collection-specific retriever
        retriever = self.vector_store.get_retriever(collection_name=collection_name, search_kwargs={"k": 5})

        # Use current chunk as query to retrieve related context from the specified collection
        context_docs = retriever.invoke(chunk_text[:500])  # Use first 500 chars as query

        # Combine context
        context = "\n\n".join(
            [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in context_docs]
        )

        # Capture context chunks for observability (with metadata and rank)
        context_chunks_data = []
        for rank, doc in enumerate(context_docs, 1):
            chunk_data = {
                "rank": rank,
                "content_preview": doc.page_content[:500],
                "chunk_id": doc.metadata.get("chunk_id"),  # May be None if not set
                "qdrant_point_id": doc.metadata.get("qdrant_point_id"),  # From Qdrant metadata
                "document_filename": doc.metadata.get("source", "unknown"),
                "relevance_score": doc.metadata.get("_score"),  # Qdrant similarity score if available
            }
            context_chunks_data.append(chunk_data)

        state["context"] = context
        state["context_chunks"] = context_chunks_data  # Store for observability
        state["metadata"] = {
            "num_retrieved_docs": len(context_docs),
            "sources": [doc.metadata.get("source", "unknown") for doc in context_docs],
            "chunk_source": chunk_source,
            "collection_name": collection_name,
        }

        logger.info(f"✓ Retrieved {len(context_docs)} related chunks from collection '{collection_name}'")
        return state

    def _ground_with_search(self, state: AgentState) -> AgentState:
        """
        Ground the extraction with external knowledge using web search.

        Uses Chain-of-Thought reasoning to:
        1. Identify what concepts need clarification
        2. Make targeted web searches
        3. Synthesize findings for rule extraction
        """
        chunk_input = state.get("chunk")
        context = state.get("context", "")

        # Normalize chunk
        chunk = self._normalize_chunk(chunk_input)

        if not chunk:
            logger.warning("No chunk for grounding")
            state["grounding_info"] = ""
            return state

        chunk_text = chunk.page_content
        chunk_source = chunk.metadata.get("source", "unknown")

        logger.info(f"🌐 Grounding with web search for chunk: {chunk_source}")

        # Use Chain-of-Thought to identify search queries
        cot_prompt = grounding_prompt(
            chunk_text=chunk_text,
            context=context,
            grounding_min=self.grounding_min,
            grounding_max=self.grounding_max,
        )

        try:
            # Get search queries using CoT
            response = self.llm.invoke(cot_prompt)
            import json
            import re

            # Extract JSON array from response
            content = response.content.strip()
            if "```" in content:
                match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            queries = json.loads(content)

            if not isinstance(queries, list):
                logger.warning("Search queries not in list format")
                state["grounding_info"] = ""
                return state

            logger.info(f"📋 LLM suggested {len(queries)} search queries (hard limit: {self.grounding_limit})")

            # Perform web searches using Tavily
            search_results = []
            try:
                import os

                from tavily import TavilyClient

                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if not tavily_api_key:
                    logger.warning("TAVILY_API_KEY not set, skipping web search")
                    state["grounding_info"] = ""
                    return state

                tavily = TavilyClient(api_key=tavily_api_key)

                # Apply hard limit on number of searches
                for i, query in enumerate(queries[: self.grounding_limit], 1):
                    logger.info(f"🔍 Search {i}/{min(len(queries), self.grounding_limit)}: {query}")
                    try:
                        result = tavily.search(query=query, max_results=3)
                        search_results.append({"query": query, "results": result.get("results", [])})
                    except Exception as e:
                        logger.error(f"Search failed for '{query}': {e}")
                        continue

            except ImportError:
                logger.warning("tavily-python not installed, skipping web search")
                state["grounding_info"] = ""
                return state

            # Synthesize search results into grounding information
            grounding_text = self._synthesize_grounding(search_results, chunk_text)

            # Store search results for observability
            grounding_searches_data = []
            for rank, search in enumerate(search_results, 1):
                search_data = {
                    "search_rank": rank,
                    "search_query": search["query"],
                    "search_results": search["results"],  # Full Tavily results
                }
                grounding_searches_data.append(search_data)

            state["grounding_info"] = grounding_text
            state["grounding_searches"] = grounding_searches_data  # Store for observability
            state["metadata"]["search_queries"] = queries
            state["metadata"]["num_searches"] = len(search_results)

            logger.info(f"✓ Grounding complete with {len(search_results)} searches")

        except Exception as e:
            logger.error(f"Grounding failed: {e}")
            logger.exception("Full error:")
            state["grounding_info"] = ""

        return state

    def _synthesize_grounding(self, search_results: list[dict], chunk_text: str) -> str:
        """Synthesize search results into concise grounding information."""
        if not search_results:
            return ""

        synthesis_parts = []

        for search in search_results:
            query = search["query"]
            results = search["results"]

            synthesis_parts.append(f"\n🔍 Query: {query}")

            for result in results[:2]:  # Top 2 results per query
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                if content:
                    # Truncate to ~200 chars
                    summary = content[:200] + "..." if len(content) > 200 else content
                    synthesis_parts.append(f"  • {title}: {summary}")
                    synthesis_parts.append(f"    Source: {url}")

        return "\n".join(synthesis_parts)

    def _extract_rules(self, state: AgentState) -> AgentState:
        """Extract operational rules from the chunk as Python functions using structured output."""
        chunk_input = state.get("chunk")
        context = state.get("context", "")
        grounding_info = state.get("grounding_info", "")

        # Normalize chunk (handles both Document objects and dicts from LangGraph Studio)
        chunk = self._normalize_chunk(chunk_input)

        if not chunk:
            logger.error("No chunk provided or invalid chunk format")
            state["extracted_rules"] = {"rules": [], "error": "No chunk provided or invalid chunk format"}
            return state

        chunk_text = chunk.page_content
        chunk_source = chunk.metadata.get("source", "unknown")

        logger.info(f"🤖 Extracting Python rules from chunk: {chunk_source}")

        prompt = rule_extraction_prompt(
            chunk_text=chunk_text,
            context=context,
            grounding_info=grounding_info,
        )

        # Generate rules using structured output or fallback to JSON parsing
        logger.debug("Sending request to LLM...")
        try:
            if self.structured_llm is not None:
                # Use structured output (OpenAI, Claude, etc.)
                logger.debug("Using structured output...")
                extracted: ExtractedRules = self.structured_llm.invoke(prompt)

                # Convert to dict for storage in state
                rules_dict = {"rules": [rule.model_dump() for rule in extracted.rules], "count": len(extracted.rules)}

                state["extracted_rules"] = rules_dict
                logger.info(f"✓ Extracted {len(extracted.rules)} Python rules successfully")
            else:
                # Fallback: Request JSON format and parse manually (DeepSeek, Ollama, etc.)
                logger.debug("Using JSON fallback parsing...")
                json_prompt = (
                    prompt
                    + """

IMPORTANT: Return your response as a valid JSON object with this exact structure:
{{
  "rules": [
    {{
      "rule_name": "function_name_in_snake_case",
      "rule_description": "Brief description",
      "rule_reasoning": "Why this rule exists",
      "rule_source": "Section reference",
      "rule_body": "def function_name(status) -> str:\\n    # function code\\n    return None",
      "rule_type": "safety|operational|maintenance|optimization"
    }}
  ]
}}

Return ONLY the JSON object, no additional text before or after."""
                )

                response = self.llm.invoke(json_prompt)
                rules_dict = self._parse_json_response(response.content)

                state["extracted_rules"] = rules_dict
                logger.info(f"✓ Extracted {rules_dict.get('count', 0)} Python rules successfully (JSON fallback)")

        except Exception as e:
            logger.error(f"Failed to extract rules: {e}")
            logger.exception("Full error:")
            state["extracted_rules"] = {"rules": [], "error": str(e)}

        return state

    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from LLM when structured output is not available.

        Handles common issues like markdown code blocks and validates against Pydantic schema.
        """
        import json
        import re

        # Remove markdown code blocks if present
        cleaned = response_text.strip()
        if "```" in cleaned:
            # Extract content between code blocks
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)

        try:
            # Parse JSON
            data = json.loads(cleaned)

            # Validate and convert to Pydantic models
            extracted = ExtractedRules(**data)

            # Return as dict
            return {"rules": [rule.model_dump() for rule in extracted.rules], "count": len(extracted.rules)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return {"rules": [], "error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to validate rules: {e}")
            return {"rules": [], "error": f"Rule validation failed: {str(e)}"}

    def _resolve_sensors(self, state: AgentState) -> AgentState:
        """
        Resolve natural language sensor references to sensor IDs.

        For each status.get() call in each rule body:
        1. Extract the natural language sensor name
        2. Use LLM to map it to the correct sensor_id from the collection's sensors
        3. Replace the natural language with the sensor_id
        4. Set rule_status to 'verified' if all sensors found, 'sensors_not_found' otherwise
        """
        import re

        from src.agent.domain.models import SensorParsingStatus

        logger.info("🔍 Resolving sensors in extracted rules...")

        sensors = state.get("sensors", [])
        extracted_rules = state.get("extracted_rules", {})

        # Skip if no sensors or no rules
        if not sensors:
            logger.warning("⚠️  No sensors available for resolution")
            # Mark all rules as SENSORS_NOT_FOUND
            if isinstance(extracted_rules, dict) and "rules" in extracted_rules:
                for rule in extracted_rules["rules"]:
                    rule["sensor_parsing_status"] = SensorParsingStatus.SENSORS_NOT_FOUND
            return state

        if not extracted_rules or (isinstance(extracted_rules, dict) and not extracted_rules.get("rules")):
            logger.info("No rules to resolve")
            return state

        # Format sensors for LLM prompt
        sensors_info = "\n".join(
            [
                f"- sensor_id: {s['sensor_id']}, name: {s['name']}, unit: {s.get('unit', 'N/A')}, description: {s.get('description', 'N/A')}"
                for s in sensors
            ]
        )

        # Process each rule
        rules_list = extracted_rules.get("rules", []) if isinstance(extracted_rules, dict) else []

        for rule in rules_list:
            rule_body = rule.get("rule_body", "")

            # Store the original rule body before sensor resolution
            rule["rule_body_original"] = rule_body

            # Find all status.get() calls in the rule body
            # Pattern: status.get("anything", "time expression")
            pattern = r'status\.get\(["\']([^"\']+)["\']'
            matches = re.findall(pattern, rule_body)

            if not matches:
                # No status.get() calls found - this is unexpected for operational rules
                rule["sensor_parsing_status"] = SensorParsingStatus.NO_SENSORS
                logger.warning(
                    f"⚠️  No status.get() calls found in rule: {rule.get('rule_name')} - marking as NO_SENSORS"
                )
                continue

            logger.info(f"Found {len(matches)} sensor references in rule: {rule.get('rule_name')}")

            # Use LLM to resolve all sensors at once
            resolved_sensors = self._resolve_sensors_with_llm(matches, sensors_info, rule_body)

            # Replace natural language with sensor IDs
            all_resolved = True
            new_rule_body = rule_body

            for original_sensor, resolved_id in resolved_sensors.items():
                if resolved_id:
                    # Replace the natural language sensor with the sensor_id
                    new_rule_body = new_rule_body.replace(f'"{original_sensor}"', f'"{resolved_id}"')
                    new_rule_body = new_rule_body.replace(f"'{original_sensor}'", f"'{resolved_id}'")
                    logger.debug(f"  ✓ Resolved '{original_sensor}' → '{resolved_id}'")
                else:
                    all_resolved = False
                    logger.warning(f"  ⚠️  Could not resolve sensor: '{original_sensor}'")

            # Update rule body (final) and status
            rule["rule_body"] = new_rule_body
            rule["sensor_parsing_status"] = (
                SensorParsingStatus.OK if all_resolved else SensorParsingStatus.SENSORS_NOT_FOUND
            )

            status_emoji = "✅" if all_resolved else "⚠️"
            logger.info(f"{status_emoji} Rule '{rule.get('rule_name')}': {rule['sensor_parsing_status']}")

        # Update state
        state["extracted_rules"] = extracted_rules
        logger.info("✓ Sensor resolution complete")

        return state

    def _resolve_sensors_with_llm(
        self, sensor_references: list[str], sensors_info: str, rule_body: str
    ) -> dict[str, str | None]:
        """
        Use LLM to map natural language sensor references to sensor IDs.

        Args:
            sensor_references: List of natural language sensor names from status.get() calls
            sensors_info: Formatted string of available sensors
            rule_body: The full rule body for context

        Returns:
            Dict mapping original sensor name to resolved sensor_id (or None if not found)
        """
        prompt = sensor_resolution_prompt(
            sensors_info=sensors_info,
            rule_body=rule_body,
            sensor_references=sensor_references,
        )

        try:
            if self.sensor_mapping_llm is not None:
                # Use structured output
                result: SensorMappings = self.sensor_mapping_llm.invoke(prompt)
                # Convert to dict format
                return {m.sensor_description: m.sensor_id for m in result.mappings}
            else:
                # Fallback: Request JSON format and parse manually
                json_prompt = sensor_resolution_json_fallback(prompt)

                response = self.llm.invoke(json_prompt)
                import json
                import re

                # Clean response
                content = response.content.strip()
                if "```" in content:
                    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
                    if match:
                        content = match.group(1)

                # Parse JSON and validate with Pydantic
                data = json.loads(content)
                validated = SensorMappings(**data)
                return {m.sensor_description: m.sensor_id for m in validated.mappings}

        except Exception as e:
            logger.error(f"Failed to resolve sensors with LLM: {e}")
            # Return all as unresolved
            return {ref: None for ref in sensor_references}

    def _parse_time_expressions(self, state: AgentState) -> AgentState:
        """
        Parse natural language time expressions and convert to structured format.

        Converts expressions like:
        - "current" → "0"
        - "average over the last 30 minutes" → "30m:", "mean"
        - "standard deviation over the last 10 minutes" → "10m:", "std"

        Sets time_parsing_status for each rule.
        """
        import re

        from src.agent.domain.models import TimeParsingStatus
        from src.agent.domain.time import TimeDeltaInterval

        logger.info("⏰ Parsing time expressions in rules...")

        extracted_rules = state.get("extracted_rules", {})

        if not extracted_rules or (isinstance(extracted_rules, dict) and not extracted_rules.get("rules")):
            logger.info("No rules to parse")
            return state

        # Process each rule
        rules_list = extracted_rules.get("rules", []) if isinstance(extracted_rules, dict) else []

        for rule in rules_list:
            rule_body = rule.get("rule_body", "")
            rule_name = rule.get("rule_name", "unknown")

            # Find all status.get() calls
            pattern = r"status\.get\(([^)]+)\)"
            matches = re.findall(pattern, rule_body)

            if not matches:
                # No status.get() calls
                rule["time_parsing_status"] = TimeParsingStatus.OK
                continue

            logger.info(f"Parsing {len(matches)} status.get() calls in rule: {rule_name}")

            all_parsed_ok = True
            new_rule_body = rule_body

            for match in matches:
                # Parse the arguments
                args = self._parse_status_get_args(match)

                if not args or len(args) < 2:
                    logger.warning(f"  ⚠️  Could not parse status.get() arguments: {match}")
                    all_parsed_ok = False
                    continue

                sensor_id = args[0]
                time_expr = args[1]

                # Use LLM to parse the entire time expression (including statistic if present)
                try:
                    parsed_result = self._parse_time_expression_with_llm(time_expr)

                    if not parsed_result:
                        logger.warning(f"  ⚠️  Failed to parse time expression: '{time_expr}'")
                        all_parsed_ok = False
                        continue

                    time_str = parsed_result["time"]
                    statistic = parsed_result.get("statistic")

                    # Validate the time format
                    try:
                        time_interval = TimeDeltaInterval.from_str(time_str)

                        # Construct the new status.get() call
                        old_call = f"status.get({match})"

                        if time_interval.is_interval():
                            # Interval - must have statistic
                            if not statistic:
                                logger.warning(f"  ⚠️  Interval '{time_str}' requires a statistic but none provided")
                                all_parsed_ok = False
                                continue

                            # Reconstruct with 3 parameters: sensor, time, statistic
                            new_call = f'status.get("{sensor_id}", "{time_str}", "{statistic}")'
                            logger.debug(f"  ✓ Interval: {old_call} → {new_call}")

                        else:
                            # Point - no statistic needed
                            new_call = f'status.get("{sensor_id}", "{time_str}")'
                            logger.debug(f"  ✓ Point: {old_call} → {new_call}")

                        # Replace in rule body
                        new_rule_body = new_rule_body.replace(old_call, new_call)

                    except Exception as e:
                        logger.warning(f"  ⚠️  Invalid time format '{time_str}': {e}")
                        all_parsed_ok = False
                        continue

                except Exception as e:
                    logger.warning(f"  ⚠️  Error parsing time expression '{time_expr}': {e}")
                    all_parsed_ok = False
                    continue

            # Update rule body with parsed expressions
            rule["rule_body"] = new_rule_body

            # Set time parsing status
            if all_parsed_ok:
                rule["time_parsing_status"] = TimeParsingStatus.OK
                logger.info(f"✅ Rule '{rule_name}': time expressions parsed successfully")
            else:
                rule["time_parsing_status"] = TimeParsingStatus.PARSE_ERROR
                logger.warning(f"⚠️  Rule '{rule_name}': time parsing errors occurred")

        # Update state
        state["extracted_rules"] = extracted_rules
        logger.info("✓ Time expression parsing complete")

        return state

    def _parse_status_get_args(self, args_str: str) -> list[str] | None:
        """
        Parse the arguments from a status.get() call.

        Args:
            args_str: The arguments string (e.g., '"sensor_id", "time"' or '"sensor_id", "time", "stat"')

        Returns:
            List of argument values (without quotes)
        """
        import re

        # Pattern to match quoted strings
        pattern = r'["\']([^"\']+)["\']'
        matches = re.findall(pattern, args_str)

        return matches if matches else None

    def _parse_time_expression_with_llm(self, natural_language_expr: str) -> dict | None:
        """
        Use LLM to parse natural language time expression into structured format.

        Examples:
            "current" → {"time": "0", "statistic": null}
            "average over the last ten minutes" → {"time": "10m:", "statistic": "mean"}
            "mean from the last hour" → {"time": "1h:", "statistic": "mean"}
            "mean from the hour before the last hour" → {"time": "2h:1h", "statistic": "mean"}
            "an hour ago" → {"time": "1h", "statistic": null}
            "standard deviation over the last 10 minutes" → {"time": "10m:", "statistic": "std"}

        Args:
            natural_language_expr: Natural language time expression with optional statistic

        Returns:
            Dict with "time" and optional "statistic" keys, or None if parsing fails
        """
        prompt = time_parsing_prompt(natural_language_expr)

        try:
            if self.time_mapping_llm is not None:
                # Use structured output
                result: TimeMapping = self.time_mapping_llm.invoke(prompt)
                return {
                    "time": result.time_expression,
                    "statistic": result.time_statistic,
                }
            else:
                # Fallback: Request JSON format and parse manually
                import json
                import re

                json_prompt = time_parsing_json_fallback(prompt)

                response = self.llm.invoke(json_prompt)
                content = response.content.strip()

                # Extract JSON from markdown if present
                if "```" in content:
                    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()

                # Parse JSON and validate with Pydantic
                data = json.loads(content)
                validated = TimeMapping(**data)
                return {
                    "time": validated.time_expression,
                    "statistic": validated.time_statistic,
                }

        except Exception as e:
            logger.error(f"Failed to parse time expression with LLM: {e}")
            return None

    def _verify_rules(self, state: AgentState) -> AgentState:
        """
        Verify rules for correctness:
        1. Syntax check (AST parsing)
        2. Sensor ID validation (against collection's sensor list)
        3. Time expression validation (using TimeDeltaInterval)
        4. Statistic validation (against Statistic enum)

        Sets verification_status for each rule.
        """
        import ast

        from src.agent.domain.models import VerificationStatus
        from src.agent.domain.time import Statistic, TimeDeltaInterval

        logger.info("✅ Verifying rules...")

        sensors = state.get("sensors", [])
        extracted_rules = state.get("extracted_rules", {})

        if not extracted_rules or (isinstance(extracted_rules, dict) and not extracted_rules.get("rules")):
            logger.info("No rules to verify")
            return state

        # Build set of valid sensor IDs
        valid_sensor_ids = {s["sensor_id"] for s in sensors} if sensors else set()

        # Get valid statistics
        valid_statistics = set(Statistic.get_available_statistics())

        # Process each rule
        rules_list = extracted_rules.get("rules", []) if isinstance(extracted_rules, dict) else []

        for rule in rules_list:
            rule_body = rule.get("rule_body", "")
            rule_name = rule.get("rule_name", "unknown")

            # Step 1: Syntax validation
            try:
                tree = ast.parse(rule_body)
                logger.debug(f"  ✓ Syntax valid for rule: {rule_name}")
            except SyntaxError as e:
                logger.warning(f"  ⚠️  Syntax error in rule '{rule_name}': {e}")
                rule["verification_status"] = VerificationStatus.SYNTAX_ERROR
                continue

            # Step 2 & 3 & 4: Extract and validate status.get() calls using AST
            try:
                extractor = StatusCallExtractor()
                extractor.visit(tree)
                status_calls = extractor.calls

                if not status_calls:
                    # No status.get() calls - this is fine, just mark as OK
                    rule["verification_status"] = VerificationStatus.OK
                    logger.debug(f"  ✓ No status.get() calls in rule: {rule_name}")
                    continue

                all_valid = True

                for sensor_id, time_expr, statistic in status_calls:
                    # Validate sensor ID
                    if sensor_id not in valid_sensor_ids:
                        logger.warning(f"  ⚠️  Invalid sensor '{sensor_id}' in rule '{rule_name}'")
                        rule["verification_status"] = VerificationStatus.INVALID_SENSOR
                        all_valid = False
                        break

                    # Validate time expression
                    try:
                        time_interval = TimeDeltaInterval.from_str(time_expr)

                        # Validate statistic
                        if time_interval.is_interval():
                            # Interval requires statistic
                            if not statistic:
                                logger.warning(f"  ⚠️  Interval '{time_expr}' requires statistic in rule '{rule_name}'")
                                rule["verification_status"] = VerificationStatus.INVALID_STATISTIC
                                all_valid = False
                                break

                            if statistic not in valid_statistics:
                                logger.warning(f"  ⚠️  Invalid statistic '{statistic}' in rule '{rule_name}'")
                                rule["verification_status"] = VerificationStatus.INVALID_STATISTIC
                                all_valid = False
                                break
                        else:
                            # Point - should not have statistic
                            if statistic:
                                logger.warning(
                                    f"  ⚠️  Point query '{time_expr}' should not have statistic in rule '{rule_name}'"
                                )
                                rule["verification_status"] = VerificationStatus.INVALID_STATISTIC
                                all_valid = False
                                break

                    except Exception as e:
                        logger.warning(f"  ⚠️  Invalid time expression '{time_expr}' in rule '{rule_name}': {e}")
                        rule["verification_status"] = VerificationStatus.INVALID_TIME
                        all_valid = False
                        break

                if all_valid:
                    rule["verification_status"] = VerificationStatus.OK
                    logger.info(f"✅ Rule '{rule_name}': all validations passed")

            except Exception as e:
                logger.error(f"  ⚠️  Error verifying rule '{rule_name}': {e}")
                rule["verification_status"] = VerificationStatus.SYNTAX_ERROR

        # Update state
        state["extracted_rules"] = extracted_rules
        logger.info("✓ Rule verification complete")

        return state

    def run(
        self,
        chunk: Document | dict | ChunkModel,
        collection_name: str,
        collection_id: int,
        sensors: list[dict],
        use_grounding: bool = True,
        config: dict | None = None,
    ) -> dict:
        """
        Run the workflow to extract rules from a single chunk.

        Args:
            chunk: Document, dict, or ChunkModel to process
            collection_name: Qdrant collection name to search for context
            collection_id: Database ID of the collection (for sensor lookup)
            sensors: List of available sensors for this collection
            use_grounding: Whether to use web grounding stage (default: True)
            config: Optional LangGraph config (e.g., {"callbacks": [handler], "run_name": "..."})

        Returns:
            State dict with extracted_rules and metadata
        """
        # Ensure chunk is in dict format for JSON serialization
        if isinstance(chunk, Document):
            chunk_dict = ChunkModel.from_document(chunk).to_dict()
        elif isinstance(chunk, ChunkModel):
            chunk_dict = chunk.to_dict()
        else:
            chunk_dict = chunk  # Already a dict

        initial_state = {
            "chunk": chunk_dict,
            "collection_name": collection_name,
            "collection_id": collection_id,
            "sensors": sensors,
            "use_grounding": use_grounding,
            "context": "",
            "grounding_info": "",
            "extracted_rules": "",
            "context_chunks": [],  # For observability
            "grounding_searches": [],  # For observability
            "metadata": {},
            "messages": [],
        }

        # Pass config to graph.invoke() for callbacks and other LangGraph options
        result = self.graph.invoke(initial_state, config=config)
        return result
