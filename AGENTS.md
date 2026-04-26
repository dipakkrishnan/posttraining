# Project Notes

- This repo is a thin post-training experiment layer over `tinker-cookbook`.
- Keep task-specific environments, verifiers, configs, docs, and tests here.
- Use `tinker-cookbook` for core Tinker training loops and abstractions.
- For RLVR work, make reward/verifier tests pass before launching a paid Tinker run.
- The current RLVR learning path is documented in `docs/rlvr-roadmap.md`.
- Stage 1 is Countdown arithmetic RLVR using `stzhao/tinyzero-countdown-data`.
- Do not start Tinker training runs unless the user explicitly asks, because they may
  consume remote training credits.
