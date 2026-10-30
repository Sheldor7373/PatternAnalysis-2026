# Reproducing the recorded training protocol

`configs/historical_training.json` is a data-free export of the saved run
manifests, checkpoint markers and observed history. It records actual protected
arguments, transition cursors, learning-rate policy, model/trainer hashes and
epoch-end selections. Paths are role placeholders. Saved `resume` and runtime
guard arguments describe the saved session, not necessarily the first invocation.
The initial full run has history but no surviving final session-metrics file;
that absence is explicit. A superseded fresh LoRA trial is excluded.

All retained baseline training used AdamW, initial learning rate 0.0001, weight
decay 0.01, 5% warm-up, clipping at 1, source/target caps 384/256, seed 3710,
BF16, three epochs and the same train-derived development monitor. The original
pretrained assets and prepared dataset revisions are pinned in the main README.

| Retained phase | Microbatch x accumulation | End optimizer step | Examples consumed |
| --- | --- | --- | --- |
| LoRA initial | 1 x 2 | 8,750 | 17,500 |
| LoRA continuation | 4 x 1 | 77,434 | 292,233 |
| Full initial | 1 x 2 | 10,181 | 20,362 |
| Full recovery | 1 x 2 | 40,779 | 81,558 |
| Full continuation | 4 x 1 | 93,449 | 292,233 |
| Separate fourth full epoch | 4 x 1 | 117,802 | 389,644 |

Increasing effective batch changes optimisation. The continuation preserved
weights, Adam state, RNG and data position and anchored the remaining decay to
examples consumed and the next learning rate, rather than restarting warm-up.
LoRA and full switches preserved next rates 9.89590312146557e-5 and
7.588554386116575e-5 respectively. Paused/replayed tails and different session
hardware make wall time and update count unsuitable as identical exposure axes.
The separate fourth epoch explicitly restarted a one-epoch, no-warm-up linear
schedule at 3e-5, retaining Adam/RNG/data lineage. It is an extension, not part of
the original three-model held-out comparison.

## Fresh replay commands

From this project directory, after preparing the pinned data and model assets:

```sh
python replay_training.py --scenario full --model artifacts/flan-t5-small --data data/prepared --output-root runs/replay-full --plan
python replay_training.py --scenario lora --model artifacts/flan-t5-small --data data/prepared --output-root runs/replay-lora --plan
```

Inspect the JSON command plan; replace `--plan` with `--execute` to train.
The helper verifies canonical JSON-row content/order hashes for TRAIN and DEV,
executes the recorded boundaries and checks actual checkpoint steps/exposures.
Its new `--stop-at-step` control pauses without shortening the full schedule;
this is tested replay support, not a claim these CLI strings existed historically.
Interrupted phases resume from their immutable latest checkpoint. Operational
time guards are omitted from a fresh replay; epoch/example targets stay finite.

```sh
python replay_training.py --scenario extra --model artifacts/flan-t5-small --data data/prepared --output-root runs/replay-extra --source-checkpoint runs/replay-full/full-b4/latest --plan
python -m unittest test_step_pause test_replay_training test_epoch_extension test_training_transition test_train test_training_control test_checkpoint_storage
```

Baseline selected model markers are full
`fda47b10d94ba377c4ed1279c16b7b4d994a21b3d4019f20dfbf2298f863d358`
and LoRA `2f5800e64b514c0b2ac4c35ea04422cf20069d9516b3e7adff408f92f1b02053`.
The manifest contains directory and payload identities for each available anchor.
These identify historical artifacts, not downloadable weights. Fresh runs produce
new metadata hashes; floating-point reproduction across hardware is not promised.
No dataset, checkpoint or machine-specific path is included in this provenance.
