# RLVR Experiment Roadmap

This repo should stay thin: keep task-specific environments, verifiers, configs,
and analysis code here, and use `tinker-cookbook` for the training loop.

## Stages

1. Countdown arithmetic RLVR
   - Goal: understand sparse verifiable rewards, format failures, group size,
     sampling variance, and reward hacking.
   - Reward: exact expression parsing, exact number multiset match, exact target
     equality.
   - Dataset: `stzhao/tinyzero-countdown-data`.

2. GSM8K answer RLVR
   - Goal: move from symbolic expression checking to natural-language math and
     answer extraction.
   - Reward: extract final answer and compare to ground truth.
   - Dataset: `allenai/RLVR-GSM`.

3. Code execution RLVR
   - Goal: learn sandboxed execution rewards, flaky tests, partial credit, and
     hidden-test generalization.
   - Reward: unit-test pass rate in a sandbox.

4. Tool-use or search RL
   - Goal: learn multi-step environments, observation design, tool costs, and
     action constraints.
   - Reward: task success minus tool/cost penalties.

5. Preference or RLHF-style training
   - Goal: study noisy learned rewards after the deterministic verifier path is
     understood.
   - Reward: reward model or pairwise preference signal.

## Repo Layout

```text
configs/rlvr/        Training config entrypoints for each stage
docs/                Experiment plans and run notes
src/posttraining/rlvr/
  countdown.py       Stage 1 environment, dataset builder, verifier
tests/rlvr/          Reward/verifier tests
outputs/             Local training logs, ignored by git
data/                Local cached/prepared data, ignored by git
```

## Stage 1 Metrics

Track these separately; aggregate reward alone hides the useful failure modes.

- `answer_tag`: model emitted a usable `<answer>...</answer>` block.
- `parse`: answer expression parsed successfully.
- `numbers`: expression used exactly the provided number multiset.
- `correct`: expression evaluated exactly to the target.
- `reward`: scalar reward used for training.

Recommended first run: 3-number examples only, small model, short max token
budget, no KL penalty, and enough samples per prompt to see within-prompt reward
variance.
