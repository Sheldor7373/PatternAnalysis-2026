"""CPU-only transition regressions using actual DataLoader and tiny offline T5."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, RandomSampler

from checkpoint_storage import file_digests, publish_checkpoint, resolve_checkpoint
from dataset import write_jsonl
from modules import load_checkpoint
import test_model as model_tests
import train
from training_transition import (RemainingBatches, checkpoint_cursor, examples_after_updates,
                                 lr_factor, remaining_updates, transition_policy, validate_arguments)


class CursorPolicyTests(unittest.TestCase):
    def test_actual_loader_order_and_coverage_at_unaligned_offsets(self):
        for size in (11, 17):
            for epoch in (0, 1, 2):
                generator = torch.Generator().manual_seed(3710 + epoch)
                original = list(DataLoader(list(range(size)), batch_size=1, shuffle=True, generator=generator))
                expected = [int(t[0]) for t in original]
                for offset in (2, 6, size - 1):
                    for batch in (2, 4, 8):
                        generator = torch.Generator().manual_seed(3710 + epoch)
                        sampler = RandomSampler(list(range(size)), generator=generator)
                        loader = DataLoader(list(range(size)), batch_sampler=RemainingBatches(sampler, batch, offset),
                                            generator=generator)
                        actual = [int(i) for row in loader for i in row]
                        self.assertEqual(actual, expected[offset:])
                        self.assertEqual(len(loader), math.ceil((size - offset) / batch))
                        self.assertEqual(len(set(actual)), size - offset)

    def test_legacy_cursor_normalizes_odd_final_batch(self):
        self.assertEqual(checkpoint_cursor({"epoch": 0, "next_batch": 6}, {"batch_size": 2}, 11), (1, 0))
        self.assertEqual(checkpoint_cursor({"epoch": 0, "next_batch": 2}, {"batch_size": 1}, 11), (0, 2))
        self.assertEqual(checkpoint_cursor({"epoch": 0, "next_batch": 2, "next_sample": 6},
                                           {"batch_size": 4}, 11), (0, 6))
        self.assertEqual(remaining_updates(11, 2, 0, 2, 4, 2), 4)
        self.assertEqual([examples_after_updates(11, 0, 2, 4, 2, u, 2) for u in range(5)],
                         [2, 10, 11, 19, 22])

    def make_state(self, size=101, step=2):
        original = {"epochs": 3, "batch_size": 1, "accumulate": 2, "max_steps": None}
        total = remaining_updates(size, 3, 0, 0, 1, 2)
        warm = int(total * .05)
        lr = 1e-4 * (step / warm if step < warm else (total - step) / (total - warm))
        state = {"epoch": 0, "next_batch": step * 2, "step": step,
                 "optimizer": {"param_groups": [{"lr": lr}]},
                 "scheduler": {"last_epoch": step, "_last_lr": [lr], "base_lrs": [1e-4]}}
        return state, original

    def test_warmup_anchor_then_original_example_milestones(self):
        state, original = self.make_state()
        policy = transition_policy(state, original, {**original, "batch_size": 4, "accumulate": 1}, 101)
        anchor = state["optimizer"]["param_groups"][0]["lr"]
        self.assertEqual(policy["anchor_examples"], 4)
        self.assertEqual(policy["warmup_examples"], 14)
        self.assertEqual(lr_factor(policy, 2) * 1e-4, anchor)
        self.assertAlmostEqual(lr_factor(policy, 3) * 1e-4, anchor + (1e-4 - anchor) * 4 / 10)
        self.assertAlmostEqual(lr_factor(policy, 5), (303 - 16) / (303 - 14))
        self.assertEqual(lr_factor(policy, policy["planned_steps"]), 0)
        # Explicitly demonstrate that a later LambdaLR reconstructs the callable.
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(policy, s))
        scheduler.load_state_dict({**scheduler.state_dict(), "last_epoch": 2, "_last_lr": [anchor]})
        optimizer.param_groups[0]["lr"] = anchor
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], lr_factor(policy, 3) * 1e-4)

    def test_decay_and_equal_effective_batch_keep_next_lr(self):
        state, original = self.make_state(step=10)
        policy = transition_policy(state, original, {**original, "batch_size": 4, "accumulate": 1}, 101)
        self.assertEqual(lr_factor(policy, 10) * 1e-4, state["optimizer"]["param_groups"][0]["lr"])
        self.assertLess(lr_factor(policy, 11), lr_factor(policy, 10))
        equal = transition_policy(state, original, {**original, "batch_size": 2, "accumulate": 1}, 101)
        self.assertEqual(equal["kind"], "original_updates")
        self.assertEqual(equal["planned_steps"], 153)
        self.assertAlmostEqual(lr_factor(equal, 11), (153 - 11) / (153 - 7))

    def test_configuration_and_state_guards_remain_strict(self):
        old = {"batch_size": 1, "accumulate": 2, "seed": 3710, "output": "old", "transition_from": None}
        new = {**old, "batch_size": 4, "accumulate": 1, "output": "new", "transition_from": "immutable"}
        validate_arguments(new, old, True)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_arguments({**new, "seed": 42}, old, True)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_arguments(new, old, False)
        state, original = self.make_state()
        state["scheduler"]["last_epoch"] += 1
        with self.assertRaisesRegex(ValueError, "step disagree"):
            transition_policy(state, original, {**original, "batch_size": 4}, 101)


class TransitionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_tests.ModelTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        model_tests.ModelTests.tearDownClass()

    def state(self, output):
        return torch.load(resolve_checkpoint(output / "latest") / "trainer.pt", map_location="cpu", weights_only=False)

    def setup_run(self, root, mode="full"):
        config_path = model_tests.ModelTests.path / "config.json"
        config = json.loads(config_path.read_text())
        config["dropout_rate"] = 0.1
        config_path.write_text(json.dumps(config))
        data = root / "data"
        data.mkdir()
        rows = [{"id": str(i), "input": "No effusion", "target": "Fluid absent" if i % 2 else "absent"}
                for i in range(11)]
        write_jsonl(data / "train.jsonl", rows)
        write_jsonl(data / "dev.jsonl", rows[:2])
        common = ["--model", str(model_tests.ModelTests.path), "--data", str(data), "--mode", mode, "--device", "cpu",
                  "--batch-size", "1", "--accumulate", "2", "--epochs", "2", "--dev-limit", "2",
                  "--cpu-threads", "1", "--precision", "fp32", "--log-every", "1"]
        source = root / "source"
        train.main(train.parser().parse_args(common + ["--output", str(source), "--max-minutes", "0.0000001"]))
        parent = resolve_checkpoint(source / "latest")
        candidate = common + ["--batch-size", "4", "--accumulate", "1", "--transition-from", str(parent)]
        return source, parent, candidate

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_handoff_preserves_weights_adam_rng_and_cumulative_state(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, parent, candidate = self.setup_run(root)
            old = self.state(source)
            immutable_digests = file_digests(parent)
            target = root / "transition"
            real_append = train.append_jsonl

            def stop_at_handoff(path, entry):
                real_append(path, entry)
                if entry["event"] == "transition_handoff":
                    raise RuntimeError("Test handoff boundary")

            with patch("train.append_jsonl", side_effect=stop_at_handoff), self.assertRaisesRegex(RuntimeError, "handoff boundary"):
                train.main(train.parser().parse_args(candidate + ["--output", str(target)]))
            new = self.state(target)
            for field in ("optimizer", "scheduler", "step", "best", "best_step", "stale", "python_rng", "torch_rng", "cuda_rng"):
                self.assert_nested_equal(old[field], new[field])
            self.assertGreaterEqual(new["elapsed_seconds"], old["elapsed_seconds"])
            self.assertEqual((new["epoch"], new["next_sample"], new["consumed_examples"]), (0, 2, 2))
            self.assertEqual(new["monitor"]["epoch_tokens"], old["monitor"]["epoch_tokens"])
            self.assertEqual(new["monitor"]["epoch_sum"], old["monitor"]["epoch_sum"])
            self.assertEqual(new["monitor"]["window_steps"], 0)
            old_model, _, _ = load_checkpoint(parent, torch.device("cpu"))
            new_model, _, settings = load_checkpoint(target / "latest", torch.device("cpu"))
            self.assert_nested_equal(old_model.state_dict(), new_model.state_dict())
            self.assertEqual(settings["transition"]["source_checkpoint"], str(parent.resolve()))
            self.assertEqual(settings["dev_batch_size"], 1)
            self.assertEqual(immutable_digests, file_digests(parent))

    def test_lora_adapter_handoff_keeps_all_optimizer_moments_and_weights(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, parent, candidate = self.setup_run(root, mode="lora")
            target = root / "lora-transition"
            real_append = train.append_jsonl

            def interrupt(path, entry):
                real_append(path, entry)
                if entry["event"] == "transition_handoff":
                    raise RuntimeError("LoRA handoff boundary")

            with patch("train.append_jsonl", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "LoRA handoff"):
                train.main(train.parser().parse_args(candidate + ["--output", str(target)]))
            old, new = self.state(source), self.state(target)
            self.assert_nested_equal(old["optimizer"], new["optimizer"])
            self.assertTrue(all(int(item["step"]) == 1 for item in new["optimizer"]["state"].values()))
            a, _, _ = load_checkpoint(parent, torch.device("cpu"))
            b, _, _ = load_checkpoint(target / "latest", torch.device("cpu"))
            self.assert_nested_equal(a.state_dict(), b.state_dict())

    def test_transition_resume_matches_uninterrupted_and_exact_final_exposure(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, parent, candidate = self.setup_run(root)
            whole, resumed = root / "whole", root / "resumed"
            train.main(train.parser().parse_args(candidate + ["--output", str(whole)]))
            train.main(train.parser().parse_args(candidate + ["--output", str(resumed), "--max-minutes", "0.0000001"]))
            partial = self.state(resumed)
            self.assertEqual((partial["step"], partial["next_sample"]), (2, 6))
            events = list(map(json.loads, (resumed / "history.jsonl").read_text().splitlines()))
            first = next(e for e in events if e["event"] == "train_window")
            self.assertEqual(first["learning_rate"], self.state(source)["optimizer"]["param_groups"][0]["lr"])
            train.main(train.parser().parse_args(candidate + ["--output", str(resumed), "--resume"]))
            left, right = self.state(whole), self.state(resumed)
            self.assertEqual((right["step"], right["epoch"], right["next_sample"], right["consumed_examples"]), (7, 2, 0, 22))
            for field in ("optimizer", "scheduler", "scheduler_policy", "best", "best_step", "stale"):
                self.assert_nested_equal(left[field], right[field])
            a, _, _ = load_checkpoint(whole / "latest", torch.device("cpu"))
            b, _, _ = load_checkpoint(resumed / "latest", torch.device("cpu"))
            self.assert_nested_equal(a.state_dict(), b.state_dict())
            self.assertEqual(right["optimizer"]["param_groups"][0]["lr"], 0)

    def test_historical_best_is_carried_and_inconsistent_best_rejected(self):
        with tempfile.TemporaryDirectory(dir=model_tests.ModelTests.path) as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source, parent, candidate = self.setup_run(root)
            state = self.state(source)
            model, tokenizer, settings = load_checkpoint(parent, torch.device("cpu"))
            publish_checkpoint(source / "best", lambda stage: model.save(stage, tokenizer, settings))
            state.update(best=1.23, best_step=1, stale=1)

            def writer(stage):
                model.save(stage, tokenizer, settings)
                torch.save(state, stage / "trainer.pt")

            historical = publish_checkpoint(source / "latest", writer)
            candidate[candidate.index("--transition-from") + 1] = str(historical)
            target = root / "historical"
            real_append = train.append_jsonl

            def interrupt(path, entry):
                real_append(path, entry)
                if entry["event"] == "transition_handoff":
                    raise RuntimeError("Test handoff")

            with patch("train.append_jsonl", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "Test handoff"):
                train.main(train.parser().parse_args(candidate + ["--output", str(target)]))
            self.assertEqual(self.state(target)["stale"], 1)
            self.assertEqual(self.state(target)["best"], 1.23)
            best_settings = json.loads((resolve_checkpoint(target / "best") / "experiment.json").read_text())
            self.assertEqual(best_settings["checkpoint_step"], 1)
            # Publish a later best without updating the resumable state.
            publish_checkpoint(source / "best", lambda stage: model.save(stage, tokenizer, {**settings, "checkpoint_step": 2}))
            with self.assertRaisesRegex(ValueError, "Published best checkpoint"):
                train.main(train.parser().parse_args(candidate + ["--output", str(root / "invalid")]))


if __name__ == "__main__":
    unittest.main()
