import os
import math
import wandb
import hydra
import uvicorn
import threading
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.classroom import Classroom, Conversation
from config.train_rl_model import RLModelTrainingConfig
from src.utils.utils import init_logger

logger = init_logger()

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

lock = threading.Lock()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

classroom: Classroom = None
config: RLModelTrainingConfig = None
app = FastAPI()


class ConversationSampleRequest(BaseModel):
    problems: List[str]
    answers: List[str]
    problem_indices: Optional[List[int]] = None
    meta: dict = {}


class RewardRequest(BaseModel):
    conversations: list[str]


@app.post("/sample_conversations")
def sample_conversations(request: ConversationSampleRequest):
    global classroom, config

    problems = request.problems
    answers = request.answers
    problem_indices = request.problem_indices
    meta = request.meta
    reward_type = config.reward_model.reward_type
    compute_initial_attempt = True if reward_type in ["delta", "discrete_delta"] else False
    discrete_threshold = config.reward_model.discrete_threshold

    conversations = None
    with lock:
        conversations = classroom.sample_conversations(
            problems=problems,
            answers=answers,
            problem_indices=problem_indices,
            meta=meta,
            compute_initial_attempt=compute_initial_attempt,
        )

    accuracy_and_rewards = [classroom.get_end_rm_reward(c, reward_type, discrete_threshold) for c in conversations]
    accuracy = [a for a, _, _ in accuracy_and_rewards]
    accuracy_rewards = [r for _, r, _ in accuracy_and_rewards]
    end_rm_rewards   = [t for _, _, t in accuracy_and_rewards]

    problem_accuracy_rewards: dict[int, list[float]] = {}
    for conv, acc_reward in zip(conversations, accuracy_rewards):
        if acc_reward is None:
            continue
        try:
            acc_value = float(acc_reward)
        except (TypeError, ValueError):
            continue
        problem_accuracy_rewards.setdefault(conv.problem_idx, []).append(acc_value)

    def _binary_entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    problem_diversity_rows = []
    for problem_idx, rewards in sorted(problem_accuracy_rewards.items()):
        count = len(rewards)
        mean_reward = sum(rewards) / count if count > 0 else 0.0
        variance = mean_reward * (1.0 - mean_reward)
        entropy = _binary_entropy(mean_reward)
        problem_diversity_rows.append(
            {
                "problem_idx": problem_idx,
                "num_samples": count,
                "accuracy_mean": mean_reward,
                "accuracy_variance": variance,
                "accuracy_entropy": entropy,
            }
        )

    diversity_mean_variance = (
        sum(row["accuracy_variance"] for row in problem_diversity_rows)
        / len(problem_diversity_rows)
        if problem_diversity_rows
        else 0.0
    )
    diversity_mean_entropy = (
        sum(row["accuracy_entropy"] for row in problem_diversity_rows)
        / len(problem_diversity_rows)
        if problem_diversity_rows
        else 0.0
    )

    df_problem_diversity = pd.DataFrame.from_records(problem_diversity_rows)

    thinking_rewards = [classroom.get_thinking_reward(c) for c in conversations]
    eoc_rewards      = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    length_rewards   = [classroom.get_length_reward(c) for c in conversations]

    turn_stats = [c.get_logging_stats() for c in conversations]
    total_teacher_turns = [s["total_teacher_turns"] for s in turn_stats]
    participating_teacher_turns = [s["participating_teacher_turns"] for s in turn_stats]
    first_reject_turns = [s["first_reject_turn"] for s in turn_stats]
    mean_turn_rewards = [s["mean_turn_reward"] for s in turn_stats]
    mean_pedagogical_rewards = [s["mean_pedagogical_reward"] for s in turn_stats]

    ok_rates = [c.get_judge_ok_rate() for c in conversations]
    ok_rates_valid = [r for r in ok_rates if r is not None]

    df_table = classroom.to_pd_latest()
    df_table["accuracy"]              = accuracy
    df_table["accuracy_reward"]       = accuracy_rewards
    df_table["end_rm_reward"]         = end_rm_rewards
    df_table["thinking_reward"]       = thinking_rewards
    df_table["end_of_conversation_reward"] = eoc_rewards
    df_table["length_reward"]         = length_rewards
    df_table["total_teacher_turns"] = total_teacher_turns
    df_table["participating_teacher_turns"] = participating_teacher_turns
    df_table["first_reject_turn"] = first_reject_turns
    df_table["mean_turn_reward"] = mean_turn_rewards
    df_table["mean_pedagogical_reward"] = mean_pedagogical_rewards
    df_table["total_reward"] = [
        e + t + o + l
        for e, t, o, l in zip(end_rm_rewards, thinking_rewards, eoc_rewards, length_rewards)
    ]
    df_table = df_table.astype(str)

    turn_rows = []
    for conv in conversations:
        teacher_indices = [
            idx for idx, message in enumerate(conv.conversation)
            if message.get("role") == "teacher"
        ]
        rule_names = sorted(
            set(conv.judge_decisions.keys())
            | {rule for rules in conv.turn_judge_decisions.values() for rule in rules}
        )

        for turn_idx in range(conv.total_teacher_turns):
            turn_reward = conv.turn_rewards.get(turn_idx)
            teacher_message = None
            student_message = None

            if turn_idx < len(teacher_indices):
                teacher_msg_idx = teacher_indices[turn_idx]
                teacher_message = conv.conversation[teacher_msg_idx]

                for msg_idx in range(teacher_msg_idx - 1, -1, -1):
                    candidate = conv.conversation[msg_idx]
                    if candidate.get("role") == "student":
                        student_message = candidate
                        break

            decisions_by_rule = conv.turn_judge_decisions.get(turn_idx, {})
            turn_row = {
                "problem_idx": conv.problem_idx,
                "answer": conv.answer,
                "conversation_id": conv.conversation_id,
                "turn_idx": turn_idx,
                "turn_group": turn_idx + 1,
                "student_message": student_message,
                "teacher_message": teacher_message,
                "turn_reward": turn_reward,
                "turn_accuracy_reward": conv.turn_accuracy_rewards.get(turn_idx),
                "turn_think_reward": conv.turn_think_rewards.get(turn_idx),
                "turn_eoc_reward": conv.turn_eoc_rewards.get(turn_idx),
                "turn_length_reward": conv.turn_length_rewards.get(turn_idx),
                "turn_pedagogical_alignment": conv.turn_pedagogical_alignment.get(turn_idx),
                "is_participating_turn": turn_idx < conv.participating_teacher_turns,
                "is_reject_turn": conv.first_reject_turn == turn_idx,
            }

            for rule_name in rule_names:
                decisions = decisions_by_rule.get(rule_name, [])
                turn_row[rule_name] = [
                    {
                        "reasoning": decision.reasoning,
                        "decision": decision.decision.name,
                    }
                    for decision in decisions
                ]

            turn_rows.append(turn_row)

    turn_group_stats = {}
    for row in turn_rows:
        group_key = row["turn_group"]
        if group_key not in turn_group_stats:
            turn_group_stats[group_key] = {
                "turn_rewards": [],
                "turn_accuracy_rewards": [],
                "turn_think_rewards": [],
                "turn_eoc_rewards": [],
                "turn_length_rewards": [],
                "turn_pedagogical_alignment": [],
            }

        if row["turn_reward"] is not None:
            turn_group_stats[group_key]["turn_rewards"].append(row["turn_reward"])
        if row["turn_accuracy_reward"] is not None:
            turn_group_stats[group_key]["turn_accuracy_rewards"].append(
                row["turn_accuracy_reward"]
            )
        if row["turn_think_reward"] is not None:
            turn_group_stats[group_key]["turn_think_rewards"].append(
                row["turn_think_reward"]
            )
        if row["turn_eoc_reward"] is not None:
            turn_group_stats[group_key]["turn_eoc_rewards"].append(
                row["turn_eoc_reward"]
            )
        if row["turn_length_reward"] is not None:
            turn_group_stats[group_key]["turn_length_rewards"].append(
                row["turn_length_reward"]
            )

        if row["turn_pedagogical_alignment"] is not None:
            turn_group_stats[group_key]["turn_pedagogical_alignment"].append(
                row["turn_pedagogical_alignment"]
            )

    grouped_turn_log = {}
    for group_key, values in turn_group_stats.items():
        rewards = values["turn_rewards"]
        accuracy = values["turn_accuracy_rewards"]
        think = values["turn_think_rewards"]
        eoc = values["turn_eoc_rewards"]
        length = values["turn_length_rewards"]
        pedagogical = values["turn_pedagogical_alignment"]
        grouped_turn_log[f"turn_group/{group_key}/mean_turn_reward"] = (
            sum(rewards) / len(rewards) if rewards else 0.0
        )
        grouped_turn_log[f"turn_group/{group_key}/mean_accuracy_reward"] = (
            sum(accuracy) / len(accuracy) if accuracy else 0.0
        )
        grouped_turn_log[f"turn_group/{group_key}/mean_think_reward"] = (
            sum(think) / len(think) if think else 0.0
        )
        grouped_turn_log[f"turn_group/{group_key}/mean_eoc_reward"] = (
            sum(eoc) / len(eoc) if eoc else 0.0
        )
        grouped_turn_log[f"turn_group/{group_key}/mean_length_reward"] = (
            sum(length) / len(length) if length else 0.0
        )
        grouped_turn_log[f"turn_group/{group_key}/mean_pedagogical_alignment"] = (
            sum(pedagogical) / len(pedagogical) if pedagogical else 0.0
        )
        grouped_turn_log[f"turn_group/{group_key}/count"] = len(rewards)

    df_turn_table = pd.DataFrame.from_records(turn_rows).astype(str)

    if config.logging.wandb:
        step = len(classroom.conversation_sets)
        log_dict = {
            f"batch_{step}": wandb.Table(dataframe=df_table),
            f"turns_batch_{step}": wandb.Table(dataframe=df_turn_table),
            f"problem_diversity_batch_{step}": wandb.Table(dataframe=df_problem_diversity),
        }
        log_dict.update(grouped_turn_log)
        log_dict.update(
            {
                "problem_diversity/mean_accuracy_variance": diversity_mean_variance,
                "problem_diversity/mean_accuracy_entropy": diversity_mean_entropy,
            }
        )
        wandb.log(log_dict, step=step)

    structured_conversations = []
    for conv in conversations:
        accuracy, accuracy_reward = conv.get_end_rm_reward(reward_type, discrete_threshold)
        structured_conversations.append(
            {
                "problem_idx": conv.problem_idx,
                "conversation_id": conv.conversation_id,
                "conversation_messages": conv.get_trainable_representation(),
                "turn_rewards": {
                    str(turn_idx): reward
                    for turn_idx, reward in conv.turn_rewards.items()
                },
                "turn_accuracy_rewards": {
                    str(turn_idx): reward
                    for turn_idx, reward in conv.turn_accuracy_rewards.items()
                },
                "turn_think_rewards": {
                    str(turn_idx): reward
                    for turn_idx, reward in conv.turn_think_rewards.items()
                },
                "turn_eoc_rewards": {
                    str(turn_idx): reward
                    for turn_idx, reward in conv.turn_eoc_rewards.items()
                },
                "turn_length_rewards": {
                    str(turn_idx): reward
                    for turn_idx, reward in conv.turn_length_rewards.items()
                },
                "turn_groups": {
                    str(turn_idx): turn_idx + 1
                    for turn_idx in range(conv.total_teacher_turns)
                },
                "turn_messages": {
                    str(row["turn_idx"]): {
                        "student_message": row["student_message"],
                        "teacher_message": row["teacher_message"],
                    }
                    for row in turn_rows
                    if row["conversation_id"] == conv.conversation_id
                },
                "turn_pedagogical_alignment": {
                    str(turn_idx): reward
                    for turn_idx, reward in conv.turn_pedagogical_alignment.items()
                },
                "total_teacher_turns": conv.total_teacher_turns,
                "participating_teacher_turns": conv.participating_teacher_turns,
                "first_reject_turn": conv.first_reject_turn,
                "accuracy": accuracy,
                "accuracy_reward": accuracy_reward,
                "aggregated_turn_reward": conv.get_turn_aggregated_reward(),
            }
        )

    return {"conversations": structured_conversations}


@app.post("/get_end_rm_reward")
def get_end_rm_reward(request: RewardRequest):
    global classroom, config
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    reward_type = config.reward_model.reward_type
    discrete_threshold = config.reward_model.discrete_threshold
    rewards = [classroom.get_end_rm_reward(c, reward_type, discrete_threshold) for c in conversations]
    return [total for _, _, total in rewards]


@app.post("/get_thinking_reward")
def get_thinking_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_thinking_reward(c) for c in conversations]
    return rewards


@app.post("/get_end_of_conversation_reward")
def get_end_of_conversation_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    return rewards


@app.post("/get_length_reward")
def get_length_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_length_reward(c) for c in conversations]
    return rewards


@app.get("/wait_batch")
def wait_batch():
    # This endpoint waits (blocks) until the current batch (if any) is finished.
    with lock:
        return {"message": "Batch has been run."}

@app.get("/get_batch_metrics")
def get_batch_metrics():
    global classroom, config
    conversations = classroom.conversation_sets[-1]
    reward_type = config.reward_model.reward_type
    discrete_threshold = config.reward_model.discrete_threshold
    accuracy = [c.get_end_rm_reward(reward_type, discrete_threshold)[0] or 0.0 for c in conversations]
    accuracy_rewards = [c.get_end_rm_reward(reward_type, discrete_threshold)[1] or 0.0 for c in conversations]
    ok_rates = [c.get_judge_ok_rate() for c in conversations]
    ok_rates_valid = [r for r in ok_rates if r is not None]
    turn_stats = [c.get_logging_stats() for c in conversations]
    total_turns = [s["total_teacher_turns"] for s in turn_stats]
    participating_turns = [s["participating_teacher_turns"] for s in turn_stats]
    return {
        "accuracy": sum(accuracy) / len(accuracy),
        "accuracy_rewards": sum(accuracy_rewards) / len(accuracy_rewards),
        "judge_ok_rate": sum(ok_rates_valid) / len(ok_rates_valid) if ok_rates_valid else None,
        "avg_total_teacher_turns": sum(total_turns) / len(total_turns) if total_turns else 0.0,
        "avg_participating_teacher_turns": sum(participating_turns) / len(participating_turns) if participating_turns else 0.0,
    }


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):
    global classroom, config

    # We merge the config with the defaults
    default_config = OmegaConf.structured(RLModelTrainingConfig)

    # Merge loaded config with defaults
    cfg = OmegaConf.merge(
        default_config, cfg
    )  # Unspecified keys will use defaults from RLModelTrainingConfig

    config = cfg

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project + "-server",
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    classroom = Classroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        os.path.join(cfg.logging.save_dir, "policy"),
        log_file_path=None,  # hydra_cfg['runtime']['output_dir']
    )

    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()
