# Lunar Lander Agent Explorer

## Project overview

I built this project to make a saved Deep Q-Network (DQN) LunarLander agent easier to inspect. The Streamlit app lets me run a seeded episode, watch the replay, and look at the reward and action telemetry. A separate Benchmark tab compares the fixed agent with a seeded random policy using saved evaluation results.

The DQN checkpoint came from DeepLearning.AI coursework. I kept the original model and added a NumPy inference path, a Gymnasium evaluation pipeline, the explorer UI, and Docker packaging, with AI assistance during parts of the application work. The app does not train or change the model.

## Features

- **Saved DQN policy:** Runs greedy inference from the checked-in NumPy weight file.
- **Random baseline:** Selects uniformly random actions from a seed-based NumPy generator.
- **Episode controls:** Choose DQN or Random, select an integer seed, and record an episode of up to 1,000 steps.
- **Replay and telemetry:** View a saved seed-1000 DQN replay or generate a new video. New episodes show cumulative reward, simulator outcome, step count, action counts, and per-step telemetry; the trace can be downloaded as CSV.
- **Optional episode explanation:** After an episode finishes, request a plain-English explanation from a local Ollama model. The app sends compact episode facts and sampled trajectory points; Ollama does not control the lander.
- **Saved benchmark:** Compare DQN and Random results across the same 50 Gymnasium v3 seeds, inspect individual seeds, and download the raw results.
- **Model checks:** A development script compares TensorFlow predictions from the original H5 checkpoint with the NumPy implementation.
- **Tests and packaging:** Includes a saved-benchmark integrity test and a Docker image that has been built and smoke-tested locally.
- **Render guide:** The README documents a possible Docker-based Render deployment. The project has not been deployed there.

## Architecture

```text
  +------+       +------------------+
  | User | ----> |  Streamlit App   |
  +------+       +--------+---------+
                           |
                           v
                   +-------+--------+
                   | Episode Runner |
                   +-------+--------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
      +------+--------+           +------+--------------+
      | DQN Policy    |           | Random Policy      |
      +------+--------+           +------+--------------+
             ^                           |
             |                           |
      +------+--------+                  |
      | Saved NumPy   |                  |
      | Weights       |                  |
      +---------------+                  |
             +-------------+-------------+
                           |
                           v
                  +--------+---------+
                  | Gymnasium        |
                  | Environment      |
                  +--------+---------+
                           |
                           v
                  +--------+---------+
                  | Episode Results  |
                  | / Telemetry      |
                  | / Generated      |
                  | Replay           |
                  +------------------+

  App also loads these saved inputs:
  +----------------------+   +------------------+
  | Saved Benchmark Data |   | Saved Replay    |
  +----------------------+   +------------------+
```

The Streamlit app reads the saved benchmark files and illustrative replay for their respective tabs. When a user runs an episode, the app calls the simulation runner with a policy and seed. The runner uses the DQN weights or the seeded random policy to act in Gymnasium, then returns episode data and an optional generated video for display.

The optional explainer runs only after the user presses **Explain This Episode**. It sends episode metrics, a bounded summary of consecutive actions, and up to eight sampled trajectory points to the local Ollama chat API. The app displays the returned text; the model is not part of the simulation or policy path.

## How it works

1. On startup, `app.py` calls `load_benchmark()`. It reads `results/episodes.csv` and `results/summary.json`, checks the expected seed rows, and verifies the saved NumPy weight hash against the summary. This loads existing results; it does not start a new benchmark.
2. In **Episode replay**, the user selects DQN or Random, enters a seed, and starts a run. Before a new run, the app displays the saved seed-1000 DQN replay if that asset is available.
3. `run_episode()` creates a fresh Gymnasium `LunarLander-v3` environment and resets it with the selected seed. For DQN, it loads the saved NumPy weights and selects the action with the highest Q-value. For Random, it chooses uniformly from the four actions using a NumPy generator seeded from the selected seed.
4. The runner accumulates the environment's rewards and records the episode outcome, step count, action counts, and—when requested—per-step reward, cumulative reward, action, position, and angle. Episode recording samples rendered frames into a temporary MP4.
5. The app displays the generated video and telemetry in the current Streamlit session. It does not overwrite the saved benchmark CSV, summary, or seed-1000 replay.

The random policy is a useful reference because it shows how the environment scores untrained, unstructured action choices under the same selected seeds. It is a baseline for comparison, not a second trained agent.

## Model and policy details

The saved DQN weights are three weight-and-bias pairs exported from the original Dense network. `lander/policy.py` implements inference in NumPy: it applies two 64-unit ReLU layers and a four-value linear output layer, then uses `argmax` to choose an action. The network shape is **8 → 64 → 64 → 4**.

The policy accepts an eight-value observation, and the simulator exposes four discrete actions: coast, left engine, main engine, and right engine. The environment's reward is accumulated over the episode; it is a score, not by itself a measure of whether the lander came to rest safely.

The Random option uses a seeded NumPy generator. Reusing the same seed and policy makes runs repeatable in the tested setup. Seeds make it easier to compare policies under matching starting conditions and to reproduce an episode when investigating its telemetry.

The DQN weights are fixed at runtime. The project does not retrain them. `models/model_info.json` records the original checkpoint hash, architecture, and training provenance. The original H5 checkpoint is not part of the runtime image or GitHub upload; `scripts/verify_model.py` needs that original file at `models/original_dqn.h5` to repeat the TensorFlow-versus-NumPy check.

## Evaluation and benchmark results

The saved Gymnasium `LunarLander-v3` evaluation compares the fixed DQN with a uniform-random baseline on the same **50 seeds, 1000–1049**. Each policy has 50 recorded episodes, each with a 1,000-step cap. These figures come from `results/summary.json`; this documentation update did not rerun the evaluation.

| Policy | Episodes | Mean reward |
| --- | ---: | ---: |
| Saved DQN | 50 | 215.36 |
| Seeded random baseline | 50 | -213.18 |

In this saved sample, the DQN's mean cumulative reward is higher than the random policy's. Reward **≥ 200 is only a score threshold**, not proof of a successful landing. The app reports simulator outcome separately for each episode.

This evaluation is distinct from the original training record. The supplied course checkpoint was trained in OpenAI Gym `LunarLander-v2`; its metadata records a rolling mean reward of **200.05 over the final 100 training episodes at episode 455**. The saved 50-seed evaluation instead transfers that checkpoint to Gymnasium `LunarLander-v3`; it is not a reproduction of the training run.

The `assets/evaluation_seed_1000.mp4` file is one illustrative replay from the earlier evaluation environment. It is separate from the 50-seed benchmark and is not an average episode or a benchmark summary.

## Demo and screenshots

There are no genuine application screenshots in the repository yet. I will add browser captures after reviewing the current UI at desktop and narrow-screen sizes. I have not added placeholder image paths or a hosted demo link.

The saved seed-1000 replay is available at [`assets/evaluation_seed_1000.mp4`](assets/evaluation_seed_1000.mp4). It is a single DQN episode; the app can also generate a new replay for the selected policy and seed.

## Tech stack

- **Language and UI:** Python 3.12, Streamlit 1.50, pandas.
- **Simulation and inference:** Gymnasium 1.2.2 with Box2D (`box2d-py` 2.3.5), NumPy 2.2.6.
- **Replay output:** ImageIO 2.37 and imageio-ffmpeg 0.6.0.
- **Optional local explanations:** Ollama's local HTTP API and the `qwen2.5:3b-instruct-q4_0` model. The integration uses Python's standard library and adds no application dependency.
- **Model verification:** TensorFlow 2.20 and h5py 3.14, installed from `requirements-dev.txt`; these are not needed for app inference.
- **Testing and packaging:** Python `unittest`, Streamlit `AppTest`, Docker.
- **Deployment documentation:** Render Docker web-service instructions. No Render deployment configuration or live service is present.

The exact runtime pins are in `requirements.txt`; development verification pins are in `requirements-dev.txt`.

## Project structure

```text
.
├── app.py
├── lander/
│   ├── benchmark.py
│   ├── explainer.py
│   ├── policy.py
│   └── simulation.py
├── models/
│   ├── dqn_weights.npz
│   └── model_info.json
├── results/
│   ├── episodes.csv
│   ├── smoke_check.json
│   └── summary.json
├── assets/
│   └── evaluation_seed_1000.mp4
├── scripts/
│   ├── evaluate.py
│   ├── export_model.py
│   └── verify_model.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_explainer.py
│   └── test_explainer_app.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

`app.py` contains the Streamlit tabs and display logic. The `lander/` package separates model inference, episode simulation, and saved benchmark loading. `scripts/` contains export, evaluation, and verification utilities. `results/` stores the saved Gymnasium v3 evaluation and model smoke-check output.

## Local setup and execution

Use Python 3.12. If the project-local `.venv` does not exist yet, create it with the first command; then activate it and install the pinned runtime dependencies there:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install setuptools wheel swig
python -m pip install --no-build-isolation -r requirements.txt
streamlit run app.py
```

On macOS, a Box2D source build may also need the Xcode Command Line Tools. Once Streamlit starts, open the local URL it prints (normally `http://localhost:8501`). Press **Ctrl+C** in the terminal to stop the app.

### Optional local Ollama episode explanation

The core app does not require Ollama. To enable the explainer on macOS, install the Ollama app from the [official macOS download page](https://ollama.com/download/mac), open it so its local service is running, then download the selected model:

```sh
ollama pull qwen2.5:3b-instruct-q4_0
ollama list
```

The quantized model is about 1.8 GB. Keep the Ollama service running, launch the app with `streamlit run app.py`, generate an episode, then press **Explain This Episode** below its telemetry. Generation is opt-in and has a 90-second timeout. If Ollama is unreachable, open the Ollama app (or start its service with `ollama serve`); if the model is missing, run the `ollama pull` command above. Episodes, replays, telemetry, and benchmark results remain available without Ollama.

To run the saved-results integrity test:

```sh
python -m unittest discover -s tests
```

To repeat the model parity check, install the development requirements and provide the original H5 checkpoint at `models/original_dqn.h5` first. That file is intentionally excluded from the repository and is not needed to run the app.

```sh
python -m pip install --no-build-isolation -r requirements-dev.txt
python scripts/verify_model.py
```

The verifier writes `results/smoke_check.json`. `scripts/evaluate.py` writes new benchmark files; do not run it unless you intend to create a new evaluation output.

## Docker instructions

These commands previously built and ran the `linux/amd64` image locally, before the optional explainer was added. Rebuild to include the current source. On Apple silicon, Docker uses emulation for this target. The host port is restricted to this machine, while Streamlit listens on `0.0.0.0` inside the container and uses the supplied `PORT` value.

```sh
docker build --platform linux/amd64 -t lunar-lander-explorer:local .
docker run --rm --detach --name lunar-lander-smoke \
  -p 127.0.0.1:8080:8080 -e PORT=8080 lunar-lander-explorer:local
curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:8080/_stcore/health
curl --silent --show-error --output /dev/null \
  --write-out 'root_http=%{http_code}\n' --max-time 10 http://127.0.0.1:8080
docker stop lunar-lander-smoke
```

The health check should return `ok`, and the root request should report HTTP 200. The health path is Streamlit's `/_stcore/health` endpoint. The local image tested for this project was about 1.43 GB; the container test does not prove that the image or app fits Render Free's memory limit.

## Render deployment instructions

The project is **not deployed on Render**. To try a deployment after reviewing the repository:

1. In the Render Dashboard, create a **Web Service** and connect the GitHub repository and branch.
2. Select **Docker** and use the root `Dockerfile` (`/Dockerfile`). Leave the start command unchanged so the image's `CMD` runs Streamlit.
3. Select the **Free** compute plan and a region.
4. Set the HTTP health-check path to `/_stcore/health`.
5. After the first deploy, open the hosted app and check the health endpoint, saved replay, benchmark, and a newly generated episode.

The Dockerfile binds Streamlit to `0.0.0.0` and follows Render's injected `PORT` (defaulting to 8080 in this Dockerfile). Render web services expect the process to bind on `0.0.0.0`; its default expected port is 10000, but it can provide a configured port to the container. The local health check has passed; hosted startup, episode behavior, and memory use are still unverified. See Render's [Docker deployment guide](https://render.com/docs/docker), [web-service setup](https://render.com/docs/web-services), and [health-check documentation](https://render.com/docs/health-checks).

Render currently lists the Free web-service plan at **0.1 CPU and 512 MB RAM**. Free services spin down after 15 minutes without inbound traffic and take about a minute to start again. Their filesystem is ephemeral, and Free services cannot attach persistent disks. The app's memory use on this plan has not been measured, so startup or episode generation may exceed the available memory. Free plans include **750 instance-hours per workspace per calendar month**; spun-down services do not consume those hours. Outbound bandwidth and build-pipeline usage also have monthly limits. Exceeding these limits can suspend services or builds, or lead to billable overages depending on payment and spend-limit settings. Check current account usage and billing before deploying. See Render's [Free plan limits](https://render.com/docs/free) and [compute plans](https://render.com/docs/compute-plans).

## Testing and verification

These results are from earlier project verification milestones; they were not all rerun while editing this README.

- **Model parity:** `python scripts/verify_model.py` compared the original H5 model with the NumPy implementation. The recorded `results/smoke_check.json` reports 100% action agreement, all tested Q-values within the configured `1e-4` tolerance, and a maximum single-observation absolute difference of about `1.14e-5`. The script also repeated a seeded episode consistently. The original H5 checkpoint is required to rerun this check.
- **Tests after the explainer change:** `./.venv/bin/python -m unittest discover -s tests -v` passed 8 tests. They include saved-benchmark integrity, mocked Ollama success/unavailable/error/empty-response handling, and Streamlit AppTest checks for opt-in behavior, generated explanation display, an Ollama-unavailable message, the saved replay, and the Benchmark tab. The AppTest uses a mocked LLM response; it does not require or verify a live Ollama model.
- **Earlier Streamlit checks:** A prior `streamlit.testing.v1.AppTest` harness checked DQN and Random runs, repeated seeds, telemetry/action counts, and MP4 decoding without app exceptions. Browser-level visual review and playback were not verified.
- **Earlier local Docker check:** `docker build --platform linux/amd64 -t lunar-lander-explorer:local .` succeeded before the Ollama feature was added. At that time the health endpoint returned `ok`, the root returned HTTP 200, and an in-container smoke check loaded the model, validated the benchmark, decoded the saved replay, and ran a seed-1000 DQN episode. The image has not been rebuilt with the explainer change. Ollama is not installed in this environment, so live local generation is unverified.

The saved Mac model smoke check recorded seed 1000 at reward `265.038206`, 344 steps, and simulator outcome `Landed / at rest`. The Docker run recorded a slightly different cumulative reward (`265.038297`) for the same 344-step outcome. These are single smoke-check episodes, not the 50-seed benchmark. Small floating-point differences across runtime platforms are possible.

## Limitations and future improvements

- The checkpoint was trained in Gym `LunarLander-v2` and evaluated in Gymnasium `LunarLander-v3`. The environment versions differ, and 50 fixed seeds do not establish general reliability.
- The saved evaluation uses one seed range and default gravity/no wind. More seeds and varied environment conditions would give a broader evaluation.
- There is one checked-in illustrative replay. The app generates new replays on demand, but the saved clip is not a summary of the benchmark.
- Render Free's memory limit may be too small for the image or video-generating episode flow. Measure hosted memory before relying on the free service; the 1.43 GB local image size is a separate disk-size measure, not RAM use.
- The project has no genuine screenshots or verified live deployment yet. Those should be added only after browser review and a successful hosted smoke test.
- The optional explainer can produce plausible but incorrect interpretations. It receives sparse trajectory samples rather than every observation and cannot establish causation or inspect unrecorded simulator state. Ollama is not packaged in Docker; the integration currently targets `localhost:11434`, so a Docker or Render app cannot reach the user's Mac-local Ollama service.
- Further work could evaluate explanation quality against human-reviewed examples and configure an explicit Ollama endpoint for hosted use. Neither has been implemented or tested.

## What I learned

This project helped me practice keeping training separate from inference: the app loads fixed weights and does not need to retrain the agent. I also worked with seeded runs to make episodes repeatable, compared a policy with a simple baseline, and separated the model, simulator, benchmark loader, and UI into distinct modules. Building and testing the Docker image showed me how a local container check can verify startup and packaged assets, while still leaving deployment-specific questions—especially memory and cold starts—to be tested on the host.
