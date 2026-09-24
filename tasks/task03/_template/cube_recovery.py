"""Public dropped-cube rescue; no simulator implementation or state access."""

def recover_drop(sim):
    """Restore the dropped cube on its support; costs one existing-budget step."""
    return sim._request("recover_drop")
