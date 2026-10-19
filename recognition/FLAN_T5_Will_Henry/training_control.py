"""Keep model-selection decisions independent of session interruptions."""

import math


def selection_point(finished_epoch, step, planned_steps):
    return finished_epoch or step >= planned_steps


def update_selection(best, stale, loss, eligible, min_delta):
    if not eligible:
        return best, stale, False
    if loss < best - min_delta:
        return loss, 0, True
    return best, stale + 1, False


def validate_best_reference(best, recorded_step, published_step, latest_step):
    has_best = math.isfinite(best)
    if (has_best != (recorded_step is not None) or recorded_step != published_step
            or (recorded_step is not None and recorded_step > latest_step)):
        raise ValueError("Published best checkpoint does not match resumable training state")
