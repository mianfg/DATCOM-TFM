"""Prompts for the rule consolidation workflow."""


def consolidation_prompt(
    sensors_info: str,
    available_statistics: str,
    rules_info: str,
) -> str:
    """
    Generate prompt for consolidating and optimizing extracted rules.

    Args:
        sensors_info: Formatted string of available sensors
        available_statistics: Comma-separated list of available statistics
        rules_info: JSON string of rules to consolidate

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert at optimizing Python rule sets for industrial process monitoring.

AVAILABLE SENSORS:
{sensors_info}

AVAILABLE STATISTICS: {available_statistics}

RULES TO CONSOLIDATE:
{rules_info}

Your task:
1. Identify REDUNDANT rules:
   - Exact duplicates (same sensor, same threshold, same logic)
   - Semantic duplicates (different wording but identical meaning)
   - Subset rules (one rule is contained in another)

2. Identify rules that can be MERGED:
   - Multiple conditions for same alert (combine with AND)
   - Same action for different triggers (combine with OR)
   - Hierarchical alerts (warning + critical thresholds for same sensor)

3. Identify rules that can be SIMPLIFIED:
   - Redundant conditions (e.g., temp > 100 AND temp > 90 → temp > 100)
   - Overly complex boolean logic that can be simplified
   - Constant expressions that can be pre-computed

For each consolidation, provide:
- action_type: "remove" (redundant) | "merge" (combine multiple) | "simplify" (optimize one)
- input_rule_ids: List of rule IDs being consolidated (use the "id" field from rules)
- output_rule: The consolidated rule (null if just removing), following the same schema
- confidence: 0.0-1.0 (your confidence in this consolidation)
- reasoning: Brief explanation of why this consolidation makes sense

IMPORTANT:
- Preserve ALL required fields in output_rule: rule_name, rule_description, rule_reasoning, rule_source, rule_body, rule_type
- Keep sensor IDs and time expressions exactly as they are in rule_body
- Only suggest changes you're confident about (>= 0.5)
- If merging rules from different sources, combine their rule_source values"""


def consolidation_json_fallback(base_prompt: str) -> str:
    """
    Add JSON format instructions to consolidation prompt for fallback mode.

    Args:
        base_prompt: The base consolidation prompt

    Returns:
        Prompt with JSON format instructions appended
    """
    return (
        base_prompt
        + """

Return JSON:
{
  "consolidations": [
    {
      "action_type": "remove|merge|simplify",
      "input_rule_ids": [1, 2],
      "output_rule": {...} or null,
      "confidence": 0.95,
      "reasoning": "explanation here"
    }
  ]
}

Return ONLY the JSON object, no additional text."""
    )
