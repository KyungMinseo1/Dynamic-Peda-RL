# eval_integrated.py

import os
import hydra
import wandb
import warnings
from typing import List, Dict, Optional
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv

from src.classroom import Classroom, JudgeDecision, Conversation, ConversationType
from utils.data import load_datasets
from config.eval import EvalConfig
from src.utils.utils import init_logger

load_dotenv()
logger = init_logger()
cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────

def check_judge_decision(
    checklist_name: str,
    conversations: List[Conversation],
    prefix: str = "",
) -> float:
    valid_convs = [
        c for c in conversations
        if checklist_name in c.judge_decisions
        and len(c.judge_decisions[checklist_name]) > 0
    ]
    if not valid_convs:
        print(f"  [{prefix}{checklist_name.upper()}] No data")
        return float("nan")

    reject_rates = []
    for conv in valid_convs:
        decisions = [d.decision for d in conv.judge_decisions[checklist_name]]
        reject_rates.append(decisions.count(JudgeDecision.REJECT) / len(decisions))

    mean = sum(reject_rates) / len(reject_rates)
    print(f"  [{prefix}{checklist_name.upper()}] reject_rate={mean:.4f} (n={len(valid_convs)})")
    return mean


def get_accuracy_split_by_judge(
    checklist_name: str,
    conversations: List[Conversation],
    use_constructivist: bool,
    null_accuracy: float = 0.0,
    prefix: str = "",
):
    rejected_accs, accepted_accs = [], []
    failed_dict_attr = "constructivist_failed_judges" if use_constructivist else "constructivist_failed_judges"
    for conv in conversations:
        if use_constructivist:
            if checklist_name not in conv.constructivist_failed_judges:
                continue
            is_rejected = conv.constructivist_failed_judges[checklist_name]
        else:
            if checklist_name not in conv.judge_decisions:
                continue
            decisions = [d.decision for d in conv.judge_decisions[checklist_name]]
            is_rejected = any(d == JudgeDecision.REJECT for d in decisions)

        acc = conv.get_accuracy_reward()
        bucket = rejected_accs if is_rejected else accepted_accs
        bucket.append(acc if acc is not None else null_accuracy)

    safe_mean = lambda lst: sum(lst) / len(lst) if lst else float("nan")
    r_mean, a_mean = safe_mean(rejected_accs), safe_mean(accepted_accs)

    print(f"  [{prefix}{checklist_name.upper()}] REJECTED n={len(rejected_accs)} acc={r_mean:.4f} | ACCEPTED n={len(accepted_accs)} acc={a_mean:.4f}")
    return r_mean, a_mean, len(rejected_accs), len(accepted_accs)


def get_overall_accuracy_split(
    conversations: List[Conversation],
    use_constructivist: bool,
    null_accuracy: float = 0.0,
    prefix: str = "",
):
    rejected_accs, accepted_accs = [], []
    no_solution = 0

    for conv in conversations:
        acc = conv.get_accuracy_reward()
        if acc is None:
            no_solution += 1
            acc = null_accuracy

        if use_constructivist:
            is_rejected = any(conv.constructivist_failed_judges.values())
        else:
            if conv.judge_decisions:
                is_rejected = any(
                    d.decision == JudgeDecision.REJECT
                    for decisions in conv.judge_decisions.values()
                    for d in decisions
                )
            else:
                is_rejected = conv.failed_judges

        (rejected_accs if is_rejected else accepted_accs).append(acc)

    safe_mean = lambda lst: sum(lst) / len(lst) if lst else float("nan")
    r, a = safe_mean(rejected_accs), safe_mean(accepted_accs)
    print(f"  [{prefix}OVERALL] REJECTED n={len(rejected_accs)} acc={r:.4f} | ACCEPTED n={len(accepted_accs)} acc={a:.4f} | no_solution={no_solution}")
    return r, a


def get_recovery_regression_rate(conversations: List[Conversation], prefix: str = "") -> Dict:
    recovery, regression, maintained, stuck = [], [], [], []
    for conv in conversations:
        pre, post = conv.get_initial_rm_reward(), conv.get_accuracy_reward()
        if pre is None or post is None:
            continue
        if   pre == 0.0 and post > 0.0: recovery.append(conv)
        elif pre > 0.0 and post == 0.0: regression.append(conv)
        elif pre > 0.0 and post > 0.0:  maintained.append(conv)
        else:                            stuck.append(conv)

    total = len(recovery) + len(regression) + len(maintained) + len(stuck)
    safe_rate = lambda lst: len(lst) / total if total > 0 else float("nan")
    for label, lst in [("Recovery", recovery), ("Regression", regression), ("Maintained", maintained), ("Stuck", stuck)]:
        print(f"  [{prefix}{label}] {len(lst):3d}/{total} = {safe_rate(lst):.3f}")
    return {
        f"{prefix}recovery_rate":   safe_rate(recovery),
        f"{prefix}regression_rate": safe_rate(regression),
        f"{prefix}maintained_rate": safe_rate(maintained),
        f"{prefix}stuck_rate":      safe_rate(stuck),
    }


def get_turn_accuracy_stats(conversations: List[Conversation], prefix: str = "") -> Dict:
    pairs = []
    for conv in conversations:
        acc = conv.get_accuracy_reward()
        if acc is None:
            continue
        turns = sum(1 for m in conv.conversation if m["role"] == "teacher")
        pairs.append((turns, acc))
    if not pairs:
        return {}

    buckets: Dict[int, list] = {}
    for t, a in pairs:
        buckets.setdefault(t, []).append(a)

    results = {}
    print(f"  [{prefix}Turn→Acc]")
    for t in sorted(buckets):
        accs = buckets[t]
        m = sum(accs) / len(accs)
        print(f"    turns={t:2d}: n={len(accs):3d} acc={m:.4f}")
        results[f"{prefix}acc_at_{t}_turns"] = m

    if len(pairs) > 1:
        turns_l = [t for t, _ in pairs]
        accs_l  = [a for _, a in pairs]
        mt, ma = sum(turns_l)/len(turns_l), sum(accs_l)/len(accs_l)
        cov   = sum((t-mt)*(a-ma) for t, a in zip(turns_l, accs_l))
        std_t = sum((t-mt)**2 for t in turns_l)**0.5
        std_a = sum((a-ma)**2 for a in accs_l)**0.5
        corr  = cov / (std_t * std_a) if std_t * std_a > 0 else float("nan")
        print(f"  [{prefix}Turn-Acc Pearson r] {corr:.4f}")
        results[f"{prefix}turn_acc_pearson_r"] = corr
    return results


def get_judge_pass_rate(conversations: List[Conversation], use_constructivist: bool, prefix: str = "") -> Dict:
    total = len(conversations)
    if use_constructivist:
        judged   = [c for c in conversations if len(c.constructivist_failed_judges) > 0]
        passed   = [c for c in judged if not any(c.constructivist_failed_judges.values())]
    else:
        judged   = [c for c in conversations if len(c.judge_decisions) > 0]
        passed   = [c for c in judged if not c.failed_judges]

    overall_rate = len(passed) / total if total > 0 else float("nan")
    judged_rate  = len(passed) / len(judged) if judged else float("nan")
    print(f"  [{prefix}JudgePassRate] {len(passed)}/{total}={overall_rate:.4f} (judged only={judged_rate:.4f})")
    return {
        f"{prefix}judge_pass_rate":             overall_rate,
        f"{prefix}judge_pass_rate_judged_only": judged_rate,
    }


# ──────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────

def compute_all_metrics(
    conversations: List[Conversation],
    problems: List[str],
    times_num: int,
    use_constructivist: bool,
    recompute_initial: bool,
    checklist_names: List[str],
    prefix: str,
) -> Dict:
    metrics = {}
    problem_num = len(problems)

    # ── 1. Accuracy (post-tutoring) ──
    accuracy_rewards = []
    for i in range(problem_num):
        current = [
            conversations[i * times_num + j].get_accuracy_reward() or 0.0
            for j in range(times_num)
        ]
        accuracy_rewards.append(sum(current) / len(current))
    acc_mean = sum(accuracy_rewards) / len(accuracy_rewards)
    print(f"  [{prefix}AccuracyReward] {acc_mean:.4f}")
    metrics[f"{prefix}accuracy_reward_mean"] = acc_mean

    # ── 2. Delta (pre→post) ──
    if recompute_initial:
        deltas, init_rewards = [], []
        for i in range(problem_num):
            d_cur, r_cur = [], []
            for j in range(times_num):
                conv = conversations[i * times_num + j]
                post = conv.get_accuracy_reward() or 0.0
                pre  = conv.get_initial_rm_reward() or 0.0
                d_cur.append(post - pre)
                r_cur.append(pre)
            deltas.append(sum(d_cur) / len(d_cur))
            init_rewards.append(sum(r_cur) / len(r_cur))
        delta_mean = sum(deltas) / len(deltas)
        init_mean  = sum(init_rewards) / len(init_rewards)
        print(f"  [{prefix}Delta] {delta_mean:.4f}  [InitRM] {init_mean:.4f}")
        metrics[f"{prefix}delta_mean"]           = delta_mean
        metrics[f"{prefix}initial_rm_mean"]      = init_mean

    # ── 3. Judge reject rates ──
    print(f"\n  [{prefix}Judge Reject Rates]")
    for name in checklist_names:
        rate = check_judge_decision(name, conversations, prefix=prefix)
        metrics[f"{prefix}reject_{name}"] = rate

    # ── 4. Judge-Accuracy split ──
    print(f"\n  [{prefix}Judge-Accuracy Split]")
    for name in checklist_names:
        r_mean, a_mean, r_n, a_n = get_accuracy_split_by_judge(
            name, conversations, use_constructivist, prefix=prefix
        )
        metrics.update({
            f"{prefix}{name}_rejected_acc": r_mean,
            f"{prefix}{name}_accepted_acc": a_mean,
            f"{prefix}{name}_rejected_n":   r_n,
        })

    overall_r, overall_a = get_overall_accuracy_split(
        conversations, use_constructivist, prefix=prefix
    )
    metrics[f"{prefix}overall_rejected_acc"] = overall_r
    metrics[f"{prefix}overall_accepted_acc"] = overall_a

    # ── 5. Judge pass rate ──
    print(f"\n  [{prefix}Judge Pass Rate]")
    metrics.update(get_judge_pass_rate(conversations, use_constructivist, prefix=prefix))

    # ── 6. Turn-Accuracy ──
    print(f"\n  [{prefix}Turn-Accuracy]")
    metrics.update(get_turn_accuracy_stats(conversations, prefix=prefix))

    # ── 7. Recovery/Regression ──
    if recompute_initial:
        print(f"\n  [{prefix}Recovery/Regression]")
        metrics.update(get_recovery_regression_rate(conversations, prefix=prefix))

    return metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

@hydra.main(config_path="config/eval_integrated", version_base=None)
def main(cfg: EvalConfig):
    default_config = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    classroom = Classroom(
        cfg.student_model, cfg.teacher_model,
        cfg.judge_model, cfg.reward_model,
        cfg.generation, None,
    )

    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    _problems = eval_data["problem"]
    _answers  = eval_data["answer"]
    N = cfg.num_samples_per_problem
    problems = [p for p in _problems for _ in range(N)]
    answers  = [a for a in _answers  for _ in range(N)]

    # ════════════════════════════════════════
    # 1. Sample once
    # ════════════════════════════════════════
    classroom.generation_cfg.ignore_rejected_judge = True

    dataset_id = cfg.dataset.eval_datasets[0].name_or_path.replace("/", "_")
    initial_cache_path = f"cache/initial_attempts/{dataset_id}_seed{cfg.seed}.json"

    # For integrated eval we only want post-hoc judge evaluation controlled by
    # run_constructivist/run_original, so we disable rollout-time judging here.
    original_num_judge_attempts = classroom.generation_cfg.number_judge_attempts
    classroom.generation_cfg.number_judge_attempts = 0
    try:
        convs = classroom.sample_conversations(
            problems, answers,
            compute_initial_attempt=False,
            original_prompts=False,
        )
    finally:
        classroom.generation_cfg.number_judge_attempts = original_num_judge_attempts

    if cfg.recompute_initial_attempts:
        cache_hit = classroom.load_initial_attempts(convs, initial_cache_path)
        if not cache_hit:
            logger.info("Cache miss → computing initial attempts...")
            messages = [conv.get_student_no_tutor_attempt() for conv in convs]
            responses = classroom.student_model.run_batch(
                messages, classroom.sampling_params_student_solution
            )
            classroom.student_model.sleep()

            for conv, response in zip(convs, responses):
                conv.add_initial_attempts([output.text for output in response.outputs])

            all_prompts, all_answers_r, lengths = [], [], []
            for conv in convs:
                prompts = conv.get_initial_solutions_for_reward()
                lengths.append(len(prompts))
                all_prompts.extend(prompts)
                all_answers_r.extend([conv.answer] * len(prompts))

            rewards = classroom._compute_constructivist_rewards_from_prompts(
                all_prompts, all_answers_r
            )
            for conv in convs:
                curr_len = lengths.pop(0)
                conv.add_initial_rewards(rewards[:curr_len])
                rewards = rewards[curr_len:]

            classroom.save_initial_attempts(convs, initial_cache_path)
            logger.info(f"Saved initial attempts → {initial_cache_path}")
        else:
            logger.info(f"Loaded initial attempts from cache → {initial_cache_path}")

    all_metrics = {}

    # ════════════════════════════════════════
    # 2. Common Metrics (accuracy, delta, turn-accuracy, recovery/regression)
    # ════════════════════════════════════════
    print("\n[Common Metrics]")

    acc_rewards = []
    for i in range(len(_problems)):
        current = [convs[i*N+j].get_accuracy_reward() or 0.0 for j in range(N)]
        acc_rewards.append(sum(current) / N)
    acc_mean = sum(acc_rewards) / len(acc_rewards)
    print(f"  Accuracy Reward mean: {acc_mean:.4f}")
    all_metrics["accuracy_reward_mean"] = acc_mean

    if cfg.recompute_initial_attempts:
        deltas, init_rewards = [], []
        for i in range(len(_problems)):
            d_cur, r_cur = [], []
            for j in range(N):
                conv = convs[i*N+j]
                post = conv.get_accuracy_reward() or 0.0
                pre  = conv.get_initial_rm_reward() or 0.0
                d_cur.append(post - pre)
                r_cur.append(pre)
            deltas.append(sum(d_cur) / N)
            init_rewards.append(sum(r_cur) / N)
        delta_mean = sum(deltas) / len(deltas)
        init_mean  = sum(init_rewards) / len(init_rewards)
        print(f"  Delta mean:      {delta_mean:.4f}")
        print(f"  Initial RM mean: {init_mean:.4f}")
        all_metrics["delta_mean"]      = delta_mean
        all_metrics["initial_rm_mean"] = init_mean

        print("\n[Recovery / Regression]")
        all_metrics.update(get_recovery_regression_rate(convs))

    print("\n[Turn-Accuracy Stats]")
    all_metrics.update(get_turn_accuracy_stats(convs))

    # ════════════════════════════════════════
    # 3. Judge Evaluation (reject rates, judge-accuracy split, pass rates)
    # ════════════════════════════════════════
    judge_sets = {}
    if cfg.run_constructivist:
        judge_sets["c/"] = {
            "prompts_paths": dict(cfg.generation.judges_rules_upgraded_prompts_paths),
            "store_in_constructivist": True,
            "use_constructivist": True,
        }
    if cfg.run_original:
        judge_sets["o/"] = {
            "prompts_paths": dict(cfg.generation.judges_rules_prompts_paths),
            "store_in_constructivist": False,
            "use_constructivist": False,
        }

    if not judge_sets:
        print("\n[Judge Evaluation]")
        print("  No judge sets enabled. Skipping judge evaluation.")
    else:
        for prefix, judge_cfg in judge_sets.items():
            print(f"\n{'='*40}")
            print(f"Judge set: {prefix}  checklists: {list(judge_cfg['prompts_paths'].keys())}")
            print(f"{'='*40}")

            classroom.judge_conversations_with_paths(
                convs,
                prompts_paths=judge_cfg["prompts_paths"],
                store_in_constructivist=judge_cfg["store_in_constructivist"],
            )

            checklist_names = list(judge_cfg["prompts_paths"].keys())

            # Reject rates
            print(f"\n[{prefix}Judge Reject Rates]")
            for name in checklist_names:
                rate = check_judge_decision(name, convs, prefix=prefix)
                all_metrics[f"{prefix}reject_{name}"] = rate

            # Judge-Accuracy split
            print(f"\n[{prefix}Judge-Accuracy Split]")
            for name in checklist_names:
                r_mean, a_mean, r_n, a_n = get_accuracy_split_by_judge(
                    name, convs,
                    use_constructivist=judge_cfg["use_constructivist"],
                    prefix=prefix,
                )
                all_metrics.update({
                    f"{prefix}{name}_rejected_acc": r_mean,
                    f"{prefix}{name}_accepted_acc": a_mean,
                    f"{prefix}{name}_rejected_n":   r_n,
                })

            overall_r, overall_a = get_overall_accuracy_split(
                convs,
                use_constructivist=judge_cfg["use_constructivist"],
                prefix=prefix,
            )
            all_metrics[f"{prefix}overall_rejected_acc"] = overall_r
            all_metrics[f"{prefix}overall_accepted_acc"] = overall_a

            # Pass rate
            print(f"\n[{prefix}Judge Pass Rate]")
            all_metrics.update(get_judge_pass_rate(
                convs,
                use_constructivist=judge_cfg["use_constructivist"],
                prefix=prefix,
            ))

    # ════════════════════════════════════════
    # 4. Ped-RM scoring
    # ════════════════════════════════════════
    ped_scores = None
    if cfg.score_using_pedagogical_reward:
        from utils.pedagogical_reward import score_each_conversation
        import gc, torch

        df_for_ped = classroom.to_pd_latest()

        del classroom.student_model
        del classroom.teacher_model
        del classroom.judge_model

        gc.collect()
        torch.cuda.empty_cache()

        ped_scores = score_each_conversation(df_for_ped, cfg.pedagogical_reward_model)
        per_conv_ped = [
            sum(float(s) for s in score) / len(score) if len(score) > 0 else 0.0
            for score in ped_scores
        ]
        ped_macro = sum(per_conv_ped) / len(per_conv_ped) if per_conv_ped else 0.0
        ped_micro_den = sum(len(score) for score in ped_scores)
        ped_micro = (
            sum(float(s) for score in ped_scores for s in score) / ped_micro_den
            if ped_micro_den > 0 else 0.0
        )

        print(f"\n[Ped-RM] macro={ped_macro:.4f} micro={ped_micro:.4f}")
        all_metrics["ped_rm_reward_macro_avg"] = ped_macro
        all_metrics["ped_rm_reward_micro_avg"] = ped_micro

    # ════════════════════════════════════════
    # 5. wandb logging
    # ════════════════════════════════════════
    if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
        wandb.log(all_metrics)

        classroom.conversation_sets.append(convs)
        df = classroom.to_pd_latest()
        if ped_scores is not None:
            df["ped_rm_reward"] = ped_scores
        df["accuracy_reward"]    = [classroom.get_accuracy_reward(c) for c in convs]
        df["thinking_reward"]    = [classroom.get_thinking_reward(c) for c in convs]
        df["eoc_reward"]         = [classroom.get_end_of_conversation_reward(c) for c in convs]
        df["length_reward"]      = [classroom.get_length_reward(c) for c in convs]
        ped_alignment_rewards = []
        for c in convs:
            try:
                ped_alignment_rewards.append(classroom.get_pedagogical_alignment_reward(c))
            except ZeroDivisionError:
                ped_alignment_rewards.append(0.0)
        df["pedagogical_reward"] = ped_alignment_rewards
        df["total_reward"] = (
            df["accuracy_reward"] + df["thinking_reward"] +
            df["eoc_reward"] + df["length_reward"] + df["pedagogical_reward"]
        )
        wandb.log({"conversations": wandb.Table(dataframe=df.astype(str))})
        wandb.finish()

    os._exit(0)

if __name__ == "__main__":
    main()
