import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from vllm import RequestOutput, CompletionOutput
from transformers import PreTrainedTokenizer

####################################################################################################
# The following functions are used to interact with the vLLM server
# - For sampling conversations
# - For getting rewards
####################################################################################################


def sample_conversations(
    problems: List[str],
    answers: List[str],
    problem_indices: Optional[List[int]] = None,
    meta: dict = {},
    server_port: int = 8000,
    num_samples_per_problem: int = 1,
    tokenizer: PreTrainedTokenizer = None,
    return_metadata: bool = False,
) -> Union[List[RequestOutput], Tuple[List[RequestOutput], List[Dict[str, Any]]]]:
    server_url = f"http://localhost:{server_port}/sample_conversations"

    actual_problems = []
    actual_problem_indices = []
    for problem in problems:
        actual_problems.extend([problem] * num_samples_per_problem)

    if problem_indices is not None:
        if len(problem_indices) != len(problems):
            raise ValueError(
                f"Length mismatch: problem_indices={len(problem_indices)} vs problems={len(problems)}"
            )
        for problem_idx in problem_indices:
            actual_problem_indices.extend([int(problem_idx)] * num_samples_per_problem)

    answers = [str(answer) for answer in answers]
    request_json = {"problems": actual_problems, "meta": meta, "answers": answers}
    if problem_indices is not None:
        request_json["problem_indices"] = actual_problem_indices

    response = requests.post(
        server_url, json=request_json
    )
    response.raise_for_status()

    response_payload = response.json()

    if isinstance(response_payload, dict) and "conversations" in response_payload:
        rollout_metadata = response_payload["conversations"]
    else:
        rollout_metadata = [
            {"conversation_messages": item} for item in response_payload
        ]

    request_outputs = []
    for item in rollout_metadata:
        conversation_messages = item["conversation_messages"]
        request_output = RequestOutput(
            request_id="",
            prompt="",
            outputs=[
                CompletionOutput(
                    index=0,
                    text=tokenizer.apply_chat_template(
                        conversation_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    ),
                    token_ids=tokenizer.apply_chat_template(
                        conversation_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    ),
                    cumulative_logprob=0.0,
                    logprobs=[],
                )
            ],
            prompt_token_ids=[],
            prompt_logprobs=[],
            finished=True,
        )
        request_outputs.append(request_output)

    if return_metadata:
        return request_outputs, rollout_metadata

    return request_outputs


def get_end_rm_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_end_rm_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_thinking_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_thinking_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_end_of_conversation_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_end_of_conversation_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_length_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_length_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


####################################################################################################


def wait_batch(server_port: int = 8000):
    """
    Sends a request to the FastAPI server's /wait_batch endpoint.
    """
    server_url = f"http://localhost:{server_port}/wait_batch"

    response = requests.get(server_url)
    response.raise_for_status()

    return response.json()

def get_batch_metrics(server_port: int = 8000) -> dict:
    server_url = f"http://localhost:{server_port}/get_batch_metrics"
    response = requests.get(server_url)
    response.raise_for_status()
    return response.json()