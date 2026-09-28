"""Task03 breakdowns from the verifier's committed simulator records."""
from rlebench.runtime.scoring import pocket_quality


def metrics(state, result):
    config = state["config"]
    records = [state["results"].get(str(i), {}) for i in range(len(config["plan"]))]
    if config["mode"] == "hidden_com":
        totals = dict(correct=0, submitted=0, control_steps=0)
        extra = {}
        for i, record in enumerate(records, 1):
            evidence = record.get("evidence", {})
            answer = evidence.get("answer")
            submitted = int(answer in ("A", "B", "C", "D"))
            correct = int(bool(submitted) and answer == evidence.get("quadrant"))
            trial = dict(correct=correct, submitted=submitted, control_steps=record.get("steps", 0))
            extra.update({f"trial_{i}_{key}": value for key, value in dict(reward=float(correct), **trial).items()})
            for key, value in trial.items():
                totals[key] += value
        return dict(totals, **extra)

    quality = result["reward"]
    steps = state["dev_steps"]
    extra = {}
    if config["mode"] == "pocket":
        evidence = records[0].get("evidence", {})
        quality = pocket_quality(evidence)
        steps = sum(record.get("steps", 0) for record in records)
        counters = evidence.get("counters", {})
        keys = ("optimal_qtm", "actual_qtm", "unclassified_transitions", "recoveries")
        if all(type(counters.get(k)) is int and counters[k] >= 0 for k in keys):
            extra.update({key: counters[key] for key in keys})
    return dict(extra, quality=quality, task_score=100*quality,
                component_outcome=quality, component_efficiency=0.,
                interaction_steps=steps, interaction_budget=config["budget"])
