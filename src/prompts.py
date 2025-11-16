# System prompt template for policy generation
POLICY_GENERATION_SYSTEM_PROMPT = """{context}

# Instructions

Based on the above context about the Eudoxia simulator and Bauplan scheduling, generate a new scheduling policy.

Your response should be ONLY valid Python code that:
1. Uses the @register_scheduler_init decorator with the EXACT key provided in the user request (do NOT generate your own key)
2. Uses the @register_scheduler decorator with the SAME EXACT key provided in the user request
3. Implements the init function to set up any necessary state on the scheduler object `s`
4. Implements the scheduler function with the correct signature: (s, failures: List[Failure], pipelines: List[Pipeline]) -> Tuple[List[Suspend], List[Assignment]]
5. Returns proper suspensions and assignments lists
6. Follow the access patterns from the examples, especially for accessing executor pools and creating Assignment objects

Available classes and their usage based on the examples:
- Priority.QUERY, Priority.INTERACTIVE, Priority.BATCH_PIPELINE - priority levels
- Assignment(ops=op_list, cpu=cpu_amount, ram=ram_amount, priority=priority, pool_id=pool_id) - to create assignments
- Suspend(cid, pool_id) - to create suspensions
- s.executor.num_pools - number of available pools
- s.executor.pools[i].avail_cpu_pool - available CPU in pool i
- s.executor.pools[i].avail_ram_pool - available RAM in pool i
- s.executor.pools[i].max_cpu_pool - max CPU in pool i
- s.executor.pools[i].max_ram_pool - max RAM in pool i
- Pipeline has .priority and .values (list of operators)
- Failure has .priority, .ops, .cid, .pool_id, .ram, .cpu, .error

Do NOT include any explanations, markdown formatting, or import statements. Return ONLY the Python code for the policy functions, and use the function docstring
to briefly explain the logic of your policy, as it if were a normal comment in the code. If you make use of any new helper functions, define them within the same code block and make sure to add the relevant
import statements inside the function itself, should they be needed.
"""


def get_user_request(policy_key: str) -> str:
    """Generate user request with the provided policy key.

    Args:
        policy_key: The policy key that the LLM should use in @register_scheduler decorators

    Returns:
        The formatted user request string
    """
    return f"""
Starting from the naive policy provided as example, try to improve it, for example leveraging the concept of priority.
Start with small improvements first, targeting obvious flaws, and make the policy complex gradually, only after you
have working code and a direction for improvement. Make sure to consider the results of previous attempts and the feedback provided
as you generate a new policy.

IMPORTANT: Use the following EXACT key in both @register_scheduler_init and @register_scheduler decorators: "{policy_key}"
Do NOT generate your own key - you MUST use exactly: "{policy_key}"
""".strip()
