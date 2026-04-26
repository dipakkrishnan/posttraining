# Countdown RLVR Task Diagram

```text
                    COUNTDOWN RLVR TASK

        +--------------------------------------------+
        | Dataset row                                |
        |                                            |
        | nums = [2, 3, 4]                           |
        | target = 10                                |
        +---------------------+----------------------+
                              |
                              v
        src/posttraining/rlvr/countdown.py
        CountdownEnv.get_question()

        +--------------------------------------------+
        | Prompt to model                            |
        |                                            |
        | Using [2, 3, 4], create an equation = 10.  |
        | Use each number once.                      |
        | Return final expression in <answer> tags.  |
        +---------------------+----------------------+
                              |
                              v
        tinker-cookbook rollout sampling

        +--------------------------------------------+
        | Model response                             |
        |                                            |
        | <think>3 * 4 = 12, 12 - 2 = 10</think>     |
        | <answer>(3 * 4) - 2</answer>               |
        +---------------------+----------------------+
                              |
                              v
        src/posttraining/rlvr/countdown.py
        verify_countdown()

        +---------------+---------------+---------------+
        | answer tag    | safe parse    | exact numbers |
        |               |               |               |
        | yes           | yes           | [2, 3, 4]     |
        +-------+-------+-------+-------+-------+-------+
                |               |               |
                v               v               v
            <answer>       AST parser       Counter match
              found        + Fraction       no extra nums
                            arithmetic      no reused nums

                              |
                              v

        +--------------------------------------------+
        | Expression evaluation                      |
        |                                            |
        | (3 * 4) - 2                                |
        | 12 - 2                                     |
        | 10                                         |
        +---------------------+----------------------+
                              |
                              v

        +--------------------------------------------+
        | Reward                                     |
        |                                            |
        | format = 1                                 |
        | correct = 1                                |
        | reward = correct + small format component  |
        +---------------------+----------------------+
                              |
                              v

        tinker-cookbook RL update

        +--------------------------------------------+
        | Policy update                              |
        |                                            |
        | Increase likelihood of successful attempts |
        | Decrease likelihood of failed attempts     |
        +--------------------------------------------+
```

## Code Paths

- Training entrypoint: `src/posttraining/rlvr/countdown_train.py`
- Dataset, env, verifier: `src/posttraining/rlvr/countdown.py`
- CLI playground: `src/posttraining/rlvr/countdown_cli.py`
- Verifier tests: `tests/rlvr/test_countdown.py`
- CLI tests: `tests/rlvr/test_countdown_cli.py`
- Data intuition notebook: `notebooks/rlvr_data_playground.ipynb`
- Run dashboard notebook: `notebooks/rlvr_run_dashboard.ipynb`

## Mental Model

The model is not trained on the dataset solution. It samples attempts, the
verifier deterministically scores each attempt, and RL shifts probability mass
toward responses that parse, use the right numbers, and hit the target.

```text
same prompt, group_size = 8

sample 0 -> reward -0.1
sample 1 -> reward  0.0
sample 2 -> reward  1.0
sample 3 -> reward -0.1
...

advantages = rewards - mean(group_rewards)
```

Positive-advantage completions are pushed up. Negative-advantage completions are
pushed down.
