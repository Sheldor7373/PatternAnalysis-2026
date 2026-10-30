"""Explicit, auditable batch transitions at completed optimizer boundaries."""

from itertools import islice
import math

from torch.utils.data import Sampler


class RemainingBatches(Sampler):
    """Slice the original RandomSampler stream before regrouping examples.

    Share the sampler's generator with DataLoader. Its iterator consumes a base
    seed before RandomSampler draws the permutation, just as shuffle=True does.
    A standalone randperm(seed + epoch) is therefore not equivalent.
    """

    def __init__(self, sampler, batch_size, offset=0):
        self.sampler, self.batch_size, self.offset = sampler, batch_size, offset
        if batch_size < 1 or not 0 <= offset <= len(sampler):
            raise ValueError("Invalid remaining-example cursor")

    def __iter__(self):
        indices = iter(islice(self.sampler, self.offset, None))
        while batch := list(islice(indices, self.batch_size)):
            yield batch

    def __len__(self):
        return math.ceil((len(self.sampler) - self.offset) / self.batch_size)


def checkpoint_cursor(state, original_arguments, size):
    epoch = state["epoch"]
    sample = state.get("next_sample", min(state["next_batch"] * original_arguments["batch_size"], size))
    if epoch < 0 or not 0 <= sample <= size:
        raise ValueError("Invalid checkpoint example cursor")
    if sample == size:
        return epoch + 1, 0
    return epoch, sample


def remaining_updates(size, epochs, epoch, offset, batch, accumulate):
    if epoch >= epochs:
        return 0
    per_epoch = math.ceil(math.ceil(size / batch) / accumulate)
    return math.ceil(math.ceil((size - offset) / batch) / accumulate) + (epochs - epoch - 1) * per_epoch


def examples_after_updates(size, epoch, offset, batch, accumulate, updates, epochs):
    """Include odd final batches and short final accumulation groups exactly."""
    consumed = epoch * size + offset
    for current in range(epoch, epochs):
        remaining = size - (offset if current == epoch else 0)
        count = math.ceil(math.ceil(remaining / batch) / accumulate)
        taken = min(updates, count)
        consumed += min(remaining, taken * batch * accumulate)
        updates -= taken
        if updates <= 0:
            break
    return min(consumed, size * epochs)


def transition_policy(state, original, new, size):
    """Preserve the next LR, actual Adam step and original example milestones."""
    if original.get("max_steps") is not None or new.get("max_steps") is not None:
        raise ValueError("Batch transitions require an epoch-defined example budget, without max_steps")
    previous_policy = state.get("scheduler_policy")
    # A same-effective-batch recovery retains the original update schedule.
    # It is safe to re-anchor from it after validating that schedule below.
    # A prior example-anchored batch change still needs a separate policy.
    if previous_policy and previous_policy.get("kind") != "original_updates":
        raise ValueError("A second effective-batch transition requires a separately reviewed policy")
    epoch, offset = checkpoint_cursor(state, original, size)
    if epoch >= new["epochs"]:
        raise ValueError("No training examples remain for a transition")
    old_total = remaining_updates(size, original["epochs"], 0, 0,
                                  original["batch_size"], original["accumulate"])
    warmup_steps = int(old_total * 0.05)
    if previous_policy:
        expected = {"size": size, "epochs": original["epochs"],
                    "batch_size": original["batch_size"], "accumulate": original["accumulate"],
                    "original_total_steps": old_total, "original_warmup_steps": warmup_steps,
                    "final_examples": size * original["epochs"], "planned_steps": old_total}
        if any(previous_policy.get(key) != value for key, value in expected.items()):
            raise ValueError("Previous original-update policy does not match the checkpoint configuration")
    anchor_lrs = [group["lr"] for group in state["optimizer"]["param_groups"]]
    if state["scheduler"]["last_epoch"] != state["step"]:
        raise ValueError("Scheduler and actual optimizer step disagree")
    if state["scheduler"]["_last_lr"] != anchor_lrs:
        raise ValueError("Scheduler and optimizer next learning rates disagree")
    unchanged_effective = original["batch_size"] * original["accumulate"] == new["batch_size"] * new["accumulate"]
    policy = {"version": 1, "kind": "original_updates" if unchanged_effective else "anchored_examples",
              "anchor_step": state["step"], "anchor_epoch": epoch, "anchor_offset": offset,
              "anchor_examples": epoch * size + offset, "anchor_lrs": anchor_lrs,
              "base_lrs": list(state["scheduler"]["base_lrs"]), "size": size,
              "epochs": new["epochs"], "batch_size": new["batch_size"], "accumulate": new["accumulate"],
              "original_total_steps": old_total, "original_warmup_steps": warmup_steps,
              "warmup_examples": examples_after_updates(size, 0, 0, original["batch_size"],
                                                         original["accumulate"], warmup_steps, original["epochs"]),
              "final_examples": size * new["epochs"],
              "planned_steps": state["step"] + remaining_updates(size, new["epochs"], epoch, offset,
                                                                  new["batch_size"], new["accumulate"])}
    if unchanged_effective and policy["planned_steps"] != old_total:
        raise ValueError("Equal-effective-batch transition would change the original update schedule")
    return policy


def lr_factor(policy, actual_step, group=0):
    if policy["kind"] == "original_updates":
        warm, total = policy["original_warmup_steps"], policy["original_total_steps"]
        return actual_step / max(1, warm) if actual_step < warm else max(0.0, (total - actual_step) / max(1, total - warm))
    anchor, base = policy["anchor_lrs"][group], policy["base_lrs"][group]
    if actual_step <= policy["anchor_step"]:
        return anchor / base
    progress = examples_after_updates(policy["size"], policy["anchor_epoch"], policy["anchor_offset"],
                                     policy["batch_size"], policy["accumulate"],
                                     actual_step - policy["anchor_step"], policy["epochs"])
    start, warm, final = policy["anchor_examples"], policy["warmup_examples"], policy["final_examples"]
    if start < warm:
        lr = (anchor + (base - anchor) * (progress - start) / (warm - start)
              if progress <= warm else base * max(0.0, (final - progress) / max(1, final - warm)))
    else:
        lr = anchor * max(0.0, (final - progress) / max(1, final - start))
    return lr / base


def epoch_extension_policy(state, original, new, size):
    """Explicit one-epoch LR restart from a completed checkpoint, retaining Adam."""
    epoch, offset = checkpoint_cursor(state, original, size)
    if epoch != original['epochs'] or offset or state.get('consumed_examples') != epoch * size:
        raise ValueError('Epoch extension requires an exactly completed original exposure budget')
    if new['epochs'] != original['epochs'] + 1 or not math.isfinite(new['lr']) or new['lr'] <= 0:
        raise ValueError('Restart requires exactly one additional epoch and a finite positive LR')
    if new['batch_size'] != original['batch_size'] or new['accumulate'] != original['accumulate']:
        raise ValueError('Epoch extension must preserve batch and accumulation')
    if original.get('max_steps') is not None or new.get('max_steps') is not None:
        raise ValueError('Epoch extension requires an exposure-defined budget')
    old_lrs = [g['lr'] for g in state['optimizer']['param_groups']]
    if state['scheduler']['last_epoch'] != state['step'] or state['scheduler']['_last_lr'] != old_lrs:
        raise ValueError('Source scheduler/optimizer counters or rates disagree')
    if any(lr != 0 for lr in old_lrs):
        raise ValueError('Source schedule must be completed at zero LR')
    rates = [new['lr']] * len(old_lrs)
    return {'version':1,'kind':'restarted_epoch_examples','anchor_step':state['step'],
            'anchor_epoch':epoch,'anchor_offset':0,'anchor_examples':epoch*size,
            'anchor_lrs':rates,'base_lrs':rates,'source_next_lrs':old_lrs,
            'size':size,'epochs':new['epochs'],'batch_size':new['batch_size'],'accumulate':new['accumulate'],
            'warmup_examples':epoch*size,'final_examples':size*new['epochs'],
            'planned_steps':state['step']+remaining_updates(size,new['epochs'],epoch,0,new['batch_size'],new['accumulate']),
            'restart_note':'New peak LR at first additional update; linear decay by examples to zero, no new warmup; Adam moments and global update counter preserved'}


def restore_optimizer_schedule(optimizer, scheduler, state, policy, restart=False):
    optimizer.load_state_dict(state['optimizer'])
    if restart:
        if policy['kind'] != 'restarted_epoch_examples':
            raise ValueError('Unrecognised explicit restart policy')
        scheduler.base_lrs = list(policy['base_lrs'])
        scheduler.last_epoch = state['step']
        scheduler._step_count = state['step'] + 1
        scheduler._last_lr = list(policy['anchor_lrs'])
        for group, base, rate in zip(optimizer.param_groups, scheduler.base_lrs, scheduler._last_lr):
            group['initial_lr'], group['lr'] = base, rate
    else:
        scheduler.load_state_dict(state['scheduler'])


def validate_arguments(arguments, original, transition=False, schedule_restart=False):
    excluded = {"resume", "device", "max_minutes", "cpu_threads"}
    if transition:
        excluded |= {"output", "batch_size", "accumulate", "transition_from"}
    if schedule_restart:
        if not transition:raise ValueError('A new schedule requires a fresh transition, not resume')
        excluded |= {'epochs','lr','restart_schedule'}
    defaults = {"checkpoint_minutes": 0.0, "eval_minutes": 0.0, "eval_at_start": False, "transition_from": None,'restart_schedule':False}
    changed = [key for key, value in arguments.items()
               if key not in excluded and original.get(key, defaults.get(key)) != value]
    if changed:
        raise ValueError(f"{'Transition' if transition else 'Resume'} configuration changed: {changed}")
