"""Scores only committed private evidence. No simulator or agent artifact imports."""
import math


def pocket_quality(evidence):
    counters = evidence.get("counters", {})
    keys = ("optimal_qtm", "actual_qtm", "unclassified_transitions", "recoveries")
    if not evidence.get("success") or not all(type(counters.get(k)) is int and counters[k] >= 0 for k in keys):
        return 0.
    optimal, actual = counters["optimal_qtm"], counters["actual_qtm"]
    return 2. ** -max(0., actual/max(1, optimal)-(optimal > 0))


def score(state, weights=(.8, .2), metrics=None):
    config = state["config"]
    mode, plan = config["mode"], config["plan"]
    scores, successes = [], 0
    for index in range(len(plan)):
        record = state["results"].get(str(index), {})
        evidence = record.get("evidence", {})
        success = bool(evidence.get("success", False))
        if mode == "task01":
            value = float(success)
        elif mode == "task02":
            value = float(evidence.get("score", 0))
        elif mode == "tabletop":
            value = float(evidence.get("quality", 0))
            success = value  # Legacy tabletop success_rate is continuous quality.
        elif mode == "hidden_com":
            answer = evidence.get("answer")
            success = answer in ("A", "B", "C", "D") and answer == evidence.get("quadrant")
            value = float(success)
        elif mode == "pocket":
            value = pocket_quality(evidence)
            if evidence.get("counters", {}).get("unclassified_transitions"):
                value = min(value, .5)
        else:
            raise ValueError("unknown scoring mode")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid evidence")
        scores.append(value)
        successes += success
    value = sum(scores)/len(plan)
    if mode == "task01":
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1):
            raise ValueError("invalid weights")
        efficiency = max(0., 1-state["dev_steps"]/config["budget"])
        value *= weights[0]+weights[1]*efficiency
    result = dict(reward=value, success_rate=successes/len(plan),
                trials_total=len(plan), trials_recorded=len(state["results"]),
                trials_attempted=sum(r["steps"] > 0 for r in state["results"].values()),
                development_steps=state["dev_steps"], infrastructure_failures=state["failures"])
    if metrics is not None:
        extra = metrics(state, dict(result))
        if result.keys() & extra.keys():
            raise ValueError("task metrics cannot overwrite common scores")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in extra.values()):
            raise ValueError("task metrics must be finite numbers")
        result.update(extra)
    return result
