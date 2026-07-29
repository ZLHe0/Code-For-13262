# ReSkill

An easy-to-configure and extensible veRL extension for agent RL training with
skill co-evolution.

Also the official code for the paper:
**ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL**.

## System Overview

ReSkill combines three pieces:

- **RL training with per-turn skill customization**: veRL handles distributed RL, while
  ReSkill follows the [verl-agent](https://github.com/langfengq/verl-agent)
  design of decomposing multi-turn agent rollouts and adds skill loading into
  each turn.
- **RL-in-the-loop skill creation**: ReSkill adapts the structure of
  [Anthropic's skill creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
  into an RL feedback loop for analyzing rollout experience and proposing skill
  updates during training.
- **Skill versioning and sampling**: ReSkill tracks skill versions, loads active
  skills, samples/testing skill bundles, and supports skill-policy
  co-evolution over training.

## Installation

```bash
git clone https://github.com/anonymous/Code-For-13262.git
cd Code-For-13262
git submodule update --init --recursive verl
pip install -e .
```

Install only the benchmark and backend extras you need:

```bash
pip install -e ".[<env>,vllm]"
```

For the validated ALFWorld + vLLM CUDA 12.4 stack:

```bash
pip install -e ".[alfworld,vllm]" \
  -c requirements/constraints-alfworld-vllm-cu124.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Set `ALFWORLD_DATA` to the downloaded ALFWorld data directory before running
ALFWorld experiments.

The current benchmark extras are `alfworld`, `search`, and `scienceworld`.
Additional environment support will be added over time. The pinned veRL
dependency is recorded in `REQUIRED_VERL.txt`.

## Usage

Prepare data for an environment:

```bash
python scripts/data_prep/prepare_<env>.py --output_dir data/<env>
```

Run training:

```bash
python scripts/train.py --config-name <env>
```

Concrete configs live under `configs/`, and cluster launch examples live under
`scripts/launch/`.

## Customize ReSkill

ReSkill is designed so both sides of the co-evolution loop can be customized.

- **Policy side**: customize the environment, rollout format, action projection,
  rewards, group rollout settings, and backend profiles.
- **Skill side**: customize skill-generation prompts, trigger behavior, active
  skill budgets, version testing/sampling, and skill library persistence.

## Validated Environments

The current release includes validated examples for ALFWorld, agentic search,
and ScienceWorld. More environments will be added in subsequent releases.

## Roadmap

- Track newer veRL releases.
- Add SGLang rollout backend support.
- Add backend config profiles for vLLM and SGLang.
- Expand validated environment examples.

## Acknowledgements

This codebase builds on [veRL](https://github.com/volcengine/verl) and
[verl-agent](https://github.com/langfengq/verl-agent). We extend their trainer
path with the ReSkill skill evolution pipeline.

## License

Apache 2.0
