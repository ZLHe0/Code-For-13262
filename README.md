# Code for Submission 13262

Anonymous code release accompanying the paper submission
**ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL**.

This repository is organized into two self-contained parts:

- **[`reproduction/`](reproduction/)** — the code used to reproduce the paper's
  experiments (ALFWorld, agentic search, ScienceWorld), with per-domain configs
  and launch scripts.
- **[`framework/`](framework/)** — the easy-to-configure, extensible veRL
  extension for skill × policy co-evolution, intended for general open-source
  use and for building on top of ReSkill.

Each folder has its own `README.md` with installation and usage instructions,
and each pins the same validated veRL base as a git submodule:

```bash
git submodule update --init --recursive
```

## Potential Risks

See [`POTENTIAL_RISKS.md`](POTENTIAL_RISKS.md) for the intended use of this code, the
potential risks of running it and the related operations, and our recommendations.

## License

Apache 2.0. See [`framework/LICENSE`](framework/LICENSE).
