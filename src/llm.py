"""LLM-related functionality for generating scheduling policies"""

import re
from pathlib import Path
import statistics
import time
import litellm
from litellm import completion
from prompts import POLICY_GENERATION_SYSTEM_PROMPT
from eudoxia.__main__ import SCHEDULER_TEMPLATE


# Global cost tracking
_response_costs = []
_last_request_cost = None  # Track cost of most recent request


def build_system_context(files=[], sections={}):
    """Build the system context from markdown files and additional sections.

    Args:
        files: List of filenames in src/markdown/ to read and include
        sections: Dict mapping section names to content strings

    Returns:
        Combined context string
    """
    context_parts = []
    markdown_dir = Path("src/markdown")

    for filename in files:
        filepath = markdown_dir / filename
        with open(filepath, "r") as f:
            content = f.read()
            context_parts.append(f"# {filename}\n{content}")

    for section_name, content in sections.items():
        context_parts.append(f"# {section_name}\n{content}")

    return "\n\n".join(context_parts)


def completion_with_retry(
    model,
    messages,
    temperature,
    reasoning_effort=None,
    max_retries=10,
    initial_delay=1.0,
):
    """Call LiteLLM completion with exponential backoff retry logic.

    Args:
        model: The LLM model to use
        messages: List of message dicts
        temperature: Temperature parameter
        reasoning_effort: Optional reasoning effort parameter
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry

    Returns:
        The completion response

    Raises:
        Exception: If all retries are exhausted
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            return response
        except Exception as e:
            error_message = str(e).lower()

            # Check if it's an overloaded endpoint error
            is_overloaded = any(
                keyword in error_message
                for keyword in ["overloaded", "rate limit", "too many requests", "529"]
            )

            if is_overloaded:
                print(
                    f"\n⚠️  LLM endpoint is overloaded (attempt {attempt + 1}/{max_retries})"
                )
            else:
                print(
                    f"\n⚠️  LLM request failed: {e} (attempt {attempt + 1}/{max_retries})"
                )

            # If this was the last attempt, raise the exception
            if attempt == max_retries - 1:
                print(f"\n❌ All {max_retries} retry attempts exhausted. Giving up.")
                raise

            # Wait with exponential backoff
            print(f"   Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff


def generate_policy_with_llm(
    user_request,
    system_context,
    feedback_history,
    model,
    temperature,
):
    """Use LLM to generate a new policy based on context and request

    Args:
        user_request: The initial user request for policy generation
        system_context: System context built from markdown files
        feedback_history: Optional list of dicts with keys 'policy_code', 'feedback'
                         representing previous attempts and their outcomes
        model: The LLM model to use for generation
        temperature: Temperature parameter for LLM sampling

    Returns:
        tuple: (generated_content, messages, llm_params) where:
            - generated_content: The LLM's response content
            - messages: The full message list sent to the LLM
            - llm_params: Dict with model, temperature, reasoning_effort used
    """

    assert system_context is not None, "System context must be provided"

    # Use the system prompt template from prompts module
    system_prompt = POLICY_GENERATION_SYSTEM_PROMPT.format(context=system_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]

    # Add feedback history if available
    if feedback_history:
        for i, entry in enumerate(feedback_history):
            # Add assistant's previous attempt
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Here's my attempt (iteration {i + 1}):\n\n```python\n{entry['policy_code']}\n```",
                }
            )
            # Add user's feedback
            messages.append({"role": "user", "content": entry["feedback"]})

    reasoning_effort = None
    if model.startswith("gpt-5") or model.startswith("claude-opus-4"):
        # leverage advanced capabilities and override the reasoning effort
        reasoning_effort = "high"
        temperature = 1

    print(f"Now sending {len(messages)} messages as context to {model}...")
    response = completion_with_retry(
        model=model,
        messages=messages,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    print(f"got response from {model}")

    llm_params = {
        "model": model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }

    return response.choices[0].message.content, messages, llm_params


def generate_policy(
    user_request, feedback_history, model, temperature, policy_key, verbose=False
):
    """Generate a policy using LLM.

    This function encapsulates all policy generation work:
    - Building context
    - Generating policy with LLM
    - Cleaning the generated code
    - Extracting the scheduler key

    Args:
        user_request: The prompt for policy generation
        verbose: Whether to print detailed output
        feedback_history: Optional list of previous attempts with feedback
        model: The LLM model to use for generation
        temperature: Temperature parameter for LLM sampling
        policy_key: The policy key to use for registering the scheduler

    Returns:
        dict: Dictionary containing:
            - policy_code: The cleaned generated code
            - llm_messages: Full message list sent to LLM
            - llm_params: Dict with model, temperature, reasoning_effort used
    """
    import re

    # Build context
    if verbose:
        print("Building system context from markdown files...")

    # start at the init, stripping the prior imports (because we instruct the LLM not to import anything else)
    #
    # TODO: why not just let it import what it needs, and run the generated .py as its own process?
    starter_template = SCHEDULER_TEMPLATE[
        SCHEDULER_TEMPLATE.index("@register_scheduler_init") :
    ]
    starter_template = starter_template.format(scheduler_name="example")
    system_context = build_system_context(
        files=["eudoxia_bauplan.md"],
        sections={"Starter Scheduler Template": f"```python\n{starter_template}\n```"},
    )

    # Generate policy
    if verbose:
        print("\nGenerating new policy with LLM...")
        if feedback_history:
            print(f"Using feedback from {len(feedback_history)} previous attempt(s)")

    generated_code, llm_messages, llm_params = generate_policy_with_llm(
        user_request, system_context, feedback_history, model, temperature
    )

    # Clean the code
    generated_code = clean_generated_code(generated_code)

    # Extract the policy key from the generated code to verify it matches
    # Handle both @register_scheduler("key") and @register_scheduler(key="key") syntax
    key_match = re.search(r'@register_scheduler\((?:key=)?"([^"]+)"\)', generated_code)
    extracted_key = key_match.group(1) if key_match else "generated_policy"
    assert extracted_key == policy_key, (
        f"Extracted key '{extracted_key}' does not match expected key '{policy_key}'"
    )

    return {
        "policy_code": generated_code,
        "llm_messages": llm_messages,
        "llm_params": llm_params,
    }


def clean_generated_code(generated_code):
    """Clean up the generated code - remove markdown formatting if present"""
    if "```python" in generated_code or "```" in generated_code:
        # Extract code from markdown code blocks
        code_match = re.search(r"```(?:python)?\n(.*?)\n```", generated_code, re.DOTALL)
        if code_match:
            return code_match.group(1)
        else:
            # Fallback: just remove the backticks
            return generated_code.replace("```python", "").replace("```", "").strip()
    return generated_code


def setup_cost_tracking():
    """Set up cost tracking callback for litellm"""

    def track_cost_callback(
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ):
        global _last_request_cost
        try:
            response_cost = kwargs.get("response_cost", 0)
            print("\n\nResponse_cost", response_cost)
            # Store cost in global array
            _response_costs.append(response_cost)
            # Track cost of most recent request
            _last_request_cost = response_cost
        except Exception:
            # Silently ignore cost tracking errors
            pass

    litellm.success_callback = [track_cost_callback]


def reset_cost_tracking():
    """Reset the cost tracking array"""
    global _response_costs, _last_request_cost
    _response_costs = []
    _last_request_cost = None


def get_last_request_cost():
    """Get the cost of the most recent LLM request.

    Returns:
        float: Cost of the last request, or 0.0 if not available
    """
    return _last_request_cost if _last_request_cost is not None else 0.0


def get_cost_statistics():
    """Get cost statistics from tracked responses

    Returns:
        dict: Dictionary with 'total_cost' and 'median_cost' keys
    """
    if not _response_costs:
        return {"total_cost": 0.0, "median_cost": 0.0, "num_requests": 0}

    total_cost = sum(_response_costs)
    median_cost = statistics.median(_response_costs)

    return {
        "total_cost": total_cost,
        "median_cost": median_cost,
        "num_requests": len(_response_costs),
    }
