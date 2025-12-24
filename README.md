# AI for distributed system design

Reference implementation for the paper [AI for Distributed System Design: Scalable Cloud Optimization Through Repeated LLMs Sampling And Simulators](https://arxiv.org/abs/2510.18897), presented at [Post-AI Formal Methods at AAAI 2026](https://www.p-ai-fm.com/).

## Overview
The main idea in the paper is to use repeated sampling from frontier LLMs to improve on existing FaaS scheduling policies. Aside from the usual reasoning loop, the system relies on two main ideas:

* policies as code: expressing scheduling policies as Python functions leverages LLM code generation abilities and allows human inspection;
* rewards from simulations: a FaaS simulator ([Eudoxia](https://arxiv.org/abs/2505.13750)) is used to evaluate generated policies quickly and deterministically: the simulator generates the "reward" signal for the LLM to self-improve over multiple iterations.

Note on AI-assisted coding: the code in this repository has been generated with the help of Claude Code ("Resistance is futile"). Personally, this repo represents the second time I used LLMs to write the majority of the code as a first stab, and then intervening by cleaning things up: it was an interesting experience in itself and worth reflecting on.

## Setup

### API Keys for hosted LLMs

Create a `.env` file in the `src` directory by copying `local.env` and filling it with your API keys. Since the paper reports results from both Anthropic and OpenAI models, the local file includes placeholders for both, and the `main.py` script *asserts* on both keys being present - please modify as needed if you wish to use only one of the two providers.

### Python environment

We use `uv` to manage the Python environment. To set up the environment, just run:

```bash
uv sync
```

### Prompts

The `markdown` folder contains the "system prompts" used to prime the LLMs. The existing prompts are a combination of general knowledge about FaaS scheduling and specific instructions to implement the desired policies as proper Eudoxia-compatible Python functions. You can modify these prompts to experiment with different instructions or information.

## How to run an experiment

If you wish to use default settings, you can just run a reasoning loop over the pre-defined amount of iterations with:

```bash
uv run python src/main.py
```

from the project root directory. This will kick off a reasoning loop using the default parameters, and print the progress in the terminal: please refer to the paper for additional context on LLM sampling. For a list of all the supported command-line arguments, check the `main.py`
argument parser directly.

## Bonus: an RL loop with Tinker

As an alternative to the iterative sampling approach, you can fine-tune an LLM using reinforcement learning with [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook). The `training.py` script implements a LoRA-based RL loop where the simulator reward guides weight updates. 

The script is only a first stub in moving to fine-tuning and it was not included in the original paper - make sure to have a `TINKER_API_KEY` set in your `.env` file to run the training script:

```bash
uv run python src/training.py
```

Training logs are saved to `src/training_logs/`. To visualize training progress and browse generated policies:

```bash
uv run streamlit run src/training_dashboard.py
```

## License
This code is released "as is" under the MIT License. See the [LICENSE](LICENSE) file for details.
