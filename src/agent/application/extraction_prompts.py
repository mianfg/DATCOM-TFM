"""Prompts for the rule extraction workflow."""


def grounding_prompt(
    chunk_text: str,
    context: str,
    grounding_min: int,
    grounding_max: int,
) -> str:
    """
    Generate prompt for identifying web search queries to ground rule extraction.

    Args:
        chunk_text: The document chunk text (first 1000 chars)
        context: Context from similar documents (first 1000 chars)
        grounding_min: Minimum number of search queries to suggest
        grounding_max: Maximum number of search queries to suggest

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert industrial engineer analyzing process documentation.

Your task is to identify what external knowledge would help extract better operational rules from this document chunk.

DOCUMENT CHUNK:
{chunk_text[:1000]}

CONTEXT FROM SIMILAR DOCUMENTS:
{context[:1000]}

🧠 CHAIN OF THOUGHT PROCESS:

Step 1: What are the key technical concepts, equipment, or processes mentioned?
Step 2: Which concepts might need external clarification (industry standards, best practices, typical ranges)?
Step 3: What specific information would help extract more accurate operational rules?

Based on your analysis, provide {grounding_min}-{grounding_max} focused web search queries that would clarify:
- Industry standards for mentioned processes
- Typical operational ranges or thresholds
- Safety considerations for equipment/chemicals mentioned
- Best practices for the process described

Return ONLY a JSON array of search queries, no other text:
["query 1", "query 2", ...]

Keep queries specific and technical."""


def rule_extraction_prompt(
    chunk_text: str,
    context: str,
    grounding_info: str | None,
) -> str:
    """
    Generate prompt for extracting operational rules as Python functions.

    Args:
        chunk_text: The document chunk text to analyze
        context: Additional context from related documents
        grounding_info: External knowledge from web search (or None)

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert industrial engineer and Python programmer specializing in extracting operational rules from documentation.

Your task is to analyze industrial process documentation and extract operational rules as Python functions.

CRITICAL REQUIREMENTS:

1. PYTHON FUNCTION FORMAT:
   - Function name must be in snake_case (e.g., 'column_high_pressure_alert')
   - Must accept one parameter: 'status'
   - Must return a descriptive string in snake_case if condition is met, otherwise return None
   - Use status.get() API with natural language time expressions

2. NATURAL LANGUAGE TIME EXPRESSIONS:
   You can write time expressions in natural language, such as:
   - "current" or "current temperature" - for current/latest value
   - "5 minutes ago" - for a specific point in the past
   - "average over the last 10 minutes" - for statistics over time intervals
   - "maximum pressure in the last hour" - for max value in an interval
   - "standard deviation over the last 30 minutes" - for variability measures
   - "temperature from 2 hours ago to 1 hour ago" - for intervals between two points
   
   Available statistics for intervals:
   - average/mean - average value over time
   - maximum/max - maximum value over time
   - minimum/min - minimum value over time
   - standard deviation/std - variability measure
   - variance - variance measure

3. COMPREHENSIVE ANALYSIS:
   - Extract ALL operational rules from the documentation
   - Create rules for explicit conditions AND inferred operational knowledge
   - Use your engineering expertise to identify important monitoring conditions
   - Include safety rules, operational limits, and process optimization rules

4. RULE CATEGORIZATION:
   - rule_type should be: 'safety', 'operational', 'maintenance', or 'optimization'
   - Safety: Critical alarms, safety limits, emergency conditions
   - Operational: Normal operating conditions, process control
   - Maintenance: Service intervals, equipment checks
   - Optimization: Efficiency improvements, quality control

EXAMPLES:

Example 1 - Safety Rule with Current Value:
rule_name: column_high_pressure_alert
rule_description: Alert when column pressure exceeds safety limit
rule_reasoning: Critical alarm set at 15.5 kg/cm² to prevent overpressure scenarios
rule_source: Section 7. Process Safety Considerations
rule_type: safety
rule_body: def column_high_pressure_alert(status) -> str:
    current_pressure = status.get("column pressure", "current")
    if current_pressure and current_pressure > 15.5:
        return "column_high_pressure_alert"
    return None

Example 2 - Operational Rule with Average:
rule_name: high_temperature_sustained
rule_description: Alert when average temperature stays above limit for extended period
rule_reasoning: Sustained high temperature indicates thermal stress and requires intervention
rule_source: Section 3. Temperature Control
rule_type: operational
rule_body: def high_temperature_sustained(status) -> str:
    avg_temp = status.get("column temperature", "average over the last 10 minutes")
    if avg_temp and avg_temp > 490:
        return "high_temperature_sustained"
    return None

Example 3 - Stability Check with Standard Deviation:
rule_name: pressure_stability_check
rule_description: Alert when pressure variation exceeds acceptable range
rule_reasoning: High standard deviation indicates unstable process control
rule_source: Section 4. Process Control
rule_type: operational
rule_body: def pressure_stability_check(status) -> str:
    pressure_std = status.get("column pressure", "standard deviation over the last 10 minutes")
    if pressure_std and pressure_std > 0.5:
        return "pressure_stability_check"
    return None

Example 4 - Flow Disturbance Detection:
rule_name: flow_rate_disturbance
rule_description: Alert when current flow deviates significantly from recent average
rule_reasoning: Sudden flow changes indicate process disturbances
rule_source: Section 5. Flow Control
rule_type: operational
rule_body: def flow_rate_disturbance(status) -> str:
    current_flow = status.get("inlet flow", "current")
    avg_flow = status.get("inlet flow", "average over the last 30 minutes")
    if current_flow and avg_flow and abs(current_flow - avg_flow) / avg_flow > 0.15:
        return "flow_rate_disturbance"
    return None

CURRENT CHUNK TO ANALYZE:
{chunk_text}

ADDITIONAL CONTEXT FROM ALL DOCUMENTS:
{context}

EXTERNAL KNOWLEDGE (from web search):
{grounding_info if grounding_info else "No external knowledge available"}

Extract ALL operational rules from the CURRENT CHUNK as Python functions.
Use the external knowledge to inform:
- Accurate thresholds and limits based on industry standards
- Safety considerations from best practices
- Typical operational ranges for equipment/processes mentioned

🚨 CRITICAL: Start creating rules immediately! Don't wait to find all sensors - create rules with what you have!"""


def sensor_resolution_prompt(
    sensors_info: str,
    rule_body: str,
    sensor_references: list[str],
) -> str:
    """
    Generate prompt for mapping natural language sensor references to sensor IDs.

    Args:
        sensors_info: Formatted string of available sensors
        rule_body: The full rule body for context
        sensor_references: List of natural language sensor names to resolve

    Returns:
        Formatted prompt string
    """
    return f"""You are a sensor mapping expert. Your task is to map natural language sensor references to their correct sensor IDs.

AVAILABLE SENSORS:
{sensors_info}

RULE BODY (for context):
{rule_body}

SENSOR REFERENCES TO RESOLVE:
{chr(10).join([f"- {s}" for s in sensor_references])}

For each sensor reference, identify the matching sensor_id from the available sensors list.
Consider the sensor name, description, and unit to find the best match.
If a sensor cannot be matched, use null as the sensor_id value."""


def sensor_resolution_json_fallback(base_prompt: str) -> str:
    """
    Add JSON format instructions to sensor resolution prompt for fallback mode.

    Args:
        base_prompt: The base sensor resolution prompt

    Returns:
        Prompt with JSON format instructions appended
    """
    return (
        base_prompt
        + """

Return your response as a JSON object with this structure:
{
  "mappings": [
    {"sensor_description": "column temperature", "sensor_id": "14TI0041"},
    {"sensor_description": "unknown sensor", "sensor_id": null}
  ]
}

IMPORTANT: Return ONLY the JSON object, no additional text."""
    )


def time_parsing_prompt(natural_language_expr: str) -> str:
    """
    Generate prompt for parsing natural language time expressions.

    Args:
        natural_language_expr: The natural language time expression to parse

    Returns:
        Formatted prompt string
    """
    return f"""Parse the following natural language time expression into structured format.

TIME UNITS: us (microseconds), ms (milliseconds), s (seconds), m (minutes), h (hours), d (days)

TIME FORMATS:
- Point (single moment): "0" for current, "5m" for 5 minutes ago, "1h30m" for 1 hour 30 min ago
- Interval (range): "5m:" for last 5 minutes, "10h:2m" for from 10 hours ago to 2 minutes ago

STATISTICS (only for intervals): mean, max, min, std, variance

EXAMPLES:
- "current" → time_expression="0", time_statistic=null
- "5 minutes ago" → time_expression="5m", time_statistic=null
- "an hour ago" → time_expression="1h", time_statistic=null
- "average over the last ten minutes" → time_expression="10m:", time_statistic="mean"
- "mean from the last hour" → time_expression="1h:", time_statistic="mean"
- "mean from the hour before the last hour" → time_expression="2h:1h", time_statistic="mean"
- "standard deviation over the last 30 minutes" → time_expression="30m:", time_statistic="std"
- "maximum pressure in the last hour" → time_expression="1h:", time_statistic="max"
- "last 5 minutes" → time_expression="5m:", time_statistic=null

Input: "{natural_language_expr}"

Parse this time expression. time_statistic should be null for time points."""


def time_parsing_json_fallback(base_prompt: str) -> str:
    """
    Add JSON format instructions to time parsing prompt for fallback mode.

    Args:
        base_prompt: The base time parsing prompt

    Returns:
        Prompt with JSON format instructions appended
    """
    return (
        base_prompt
        + """

Return ONLY a JSON object with this structure:
{
  "time_description": "the input expression",
  "time_expression": "the parsed time format",
  "time_statistic": "statistic or null"
}"""
    )
