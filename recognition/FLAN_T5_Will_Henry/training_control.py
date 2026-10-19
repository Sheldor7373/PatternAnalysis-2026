"""Keep model-selection decisions independent of session interruptions."""


def selection_point(finished_epoch, step, planned_steps):
    return finished_epoch or step >= planned_steps


def update_selection(best, stale, loss, eligible, min_delta):
    if not eligible:
        return best, stale, False
    if loss < best - min_delta:
        return loss, 0, True
    return best, stale + 1, False
