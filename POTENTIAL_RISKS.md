# Potential Risks and Responsible Use

This file accompanies the code for the paper *ReSkill: Reconciling Skill Creation with Policy
Optimization in Agentic RL*. It describes the potential risks of running this code and the
related operations, and what we recommend to reduce them. ReSkill is research code for
training LLM agents on public benchmarks. It is not a product and has not been tested for
deployment.

## Intended use

- Research on reinforcement learning of LLM agents with skills, in the benchmark
  environments included here (ALFWorld, ScienceWorld, retrieval-based search QA) or similar
  ones.
- It is not intended to drive agents that act on real systems, real users or real accounts
  without further testing and human oversight.

## 1. Skills written by an LLM can be wrong

- During training, an LLM (the skill creator) reads rollouts and writes or revises skills,
  and these skills are inserted into the agent's prompt.
- A skill can be wrong, too broad, or conflict with what the policy has learned. The paper
  shows that adopting skills without testing them lowers the final performance (Section 4.3,
  Table 2).
- What the code does: each revised skill is tested against the current version within the
  same rollout group before it is adopted (paper Appendix A.2), and skill versions are tracked
  in `skill_registry.json`.
- What we recommend: keep skill testing enabled, and read the skill bank before reusing it.

## 2. Skills are text that enters the agent's prompt

- Skills are saved as plain-text `SKILL.md` files under `<save_dir>/skill_library/`, with
  their versions in `<save_dir>/skill_registry.json`. Anyone
  who can edit these files, or the prompts in `reskill/skill_creator/prompts/`, can change what the
  agent is told to do.
- Treat skill files from other sources as untrusted input and review them before loading.
  Before using a skill library outside the environment it was evolved in, check each skill's
  trigger conditions and `when_to_use` field.
- Models may over-apply a skill where it does not fit, or mix skill text into their actions,
  especially smaller models (paper Limitations and Appendix E).

## 3. Data sent to LLM APIs

- The skill creator calls an LLM through `reskill/utils/llm_client.py` (in both
  `framework/` and `reproduction/`): an OpenAI-compatible
  endpoint (default `http://localhost:8000/v1`, e.g. a local vLLM server), AWS Bedrock, or
  the Anthropic API.
- If you point it to an external API, rollout text (task descriptions, observations and agent
  actions) is sent to that provider. With the included public benchmarks this text contains
  no personal data. If you add your own environments or data, check what the rollouts contain
  before sending them out, and follow the provider's terms of use.
- Pass API keys through environment variables (e.g. `ANTHROPIC_API_KEY`) or a local config
  file, and do not commit them.
- External API calls are billed; the skill creator calls the LLM at each skill evolution step.

## 4. Search environment

- The search environment sends queries to the retrieval service set in
  `configs/search.yaml` (default `http://localhost:8000/retrieve`). If you change it to a
  remote or live search service, the agent's queries leave your machine.

## 5. Compute and environmental cost

- Each run in the paper used nodes with 8×A100 GPUs and took roughly a day (paper
  Appendix A.1). ReSkill adds no extra rollouts over GRPO, but skill creation adds LLM calls
  and some training and inference overhead (paper Figure 5).
- Start from the provided configs, and avoid sweeps you do not need.

## 6. Scope of the results

- The results are on English public benchmarks in five domains with Qwen3-4B-Instruct-2507 and
  Qwen3-8B. They may not carry over to other languages, models or tasks, and the skill
  evolution frequency needs tuning per domain (paper Limitations).
