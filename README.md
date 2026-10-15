# Lunar Lander Agent Explorer

An interactive Streamlit explorer for Krrish Khera's saved Deep Q Network (DQN) Lunar Lander agent. The app replays seeded episodes, shows actions and reward over time, and compares the fixed checkpoint with a random baseline.

## What is measured

The original course notebook trained in **OpenAI Gym LunarLander-v2** and recorded a rolling mean reward of **200.05 over its last 100 training episodes at episode 455**. This is a historical training result.

The subsequent evaluation ran the same saved checkpoint, without retraining, in **Gymnasium LunarLander-v3**. On predetermined seeds 1000–1049, the greedy DQN's mean reward was **215.36** (sample SD **65.99**); the uniform random baseline's mean was **−213.18** (sample SD **102.15**). The DQN scored at least 200 in 36/50 episodes. That is a **reward threshold**, not a physical landing success rate. See [raw results](results/episodes.csv) and [evaluation metadata](results/summary.json) for all runs, conditions, and dependency versions.

The trained model is `models/original_dqn.h5`; the app uses its extracted Dense weights in `models/dqn_weights.npz` for lightweight NumPy inference. The original checkpoint is preserved. Model parity and a repeatable seed smoke check are in `results/smoke_check.json`.

## Run locally

Use Python 3.12. On macOS, install Xcode command-line tools if needed; Box2D builds native code and uses SWIG.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install setuptools wheel swig
python -m pip install --no-build-isolation -r requirements.txt
python -m unittest discover -s tests
streamlit run app.py
```

Use the **Episode replay** tab to choose DQN or random and a seed, then run one bounded episode. The app shows a saved DQN replay of seed 1000 before a live run. Telemetry and action counts are generated for live runs. **Benchmark** shows paired raw results and summary statistics; **Methodology** explains what can and cannot be concluded. No training happens in the app.

For TensorFlow verification of the extracted weights, install `requirements-dev.txt` in the same environment and run `python scripts/verify_model.py`. To reproduce the evaluation, run `python scripts/evaluate.py --episodes 50` (this overwrites the saved result files, so keep a copy if needed).

## Deploy on Render (Docker)

The app is packaged by the repository's `Dockerfile`. Render can build that Dockerfile from a linked GitHub repository. The container command binds Streamlit to `0.0.0.0` and uses Render's `PORT` environment variable (Render's default is `10000`). The saved NumPy weights, benchmark files, and seed-1000 replay are copied into the image, so the app does not depend on persistent writable storage for its current core flow.

Before deployment, push the reviewed project files to the GitHub repository. In the Render Dashboard:

1. Choose **New → Web Service**, connect the GitHub account, and select this repository and branch.
2. Set **Language** to **Docker** and use the repository-root `Dockerfile` (the default path).
3. Choose a region and the **Free** compute plan, then create the service. Keep the Dockerfile's start command unless intentionally overriding it.
4. In service settings, configure the HTTP health-check path as `/_stcore/health`. This Streamlit endpoint returned `ok` in the already completed local Docker test; the Render-hosted endpoint still needs to be verified after deployment.
5. After the first deploy succeeds, open the Render URL and verify the health endpoint, benchmark, saved replay, and a generated episode. Only then record the URL and describe the service as deployed.

Render's Free web services currently provide 0.1 CPU and 512 MB RAM, spin down after 15 minutes without inbound traffic, and can take about a minute to start again. The app's memory use on that plan has not been measured; verify both startup and an episode run on the deployed service, and move to a larger plan if it runs out of memory. The filesystem is ephemeral and local changes are lost on restart/redeploy; this app's model, benchmark, and sample replay are shipped as image files. Free usage also has monthly instance-hour, bandwidth, and build-pipeline limits; check the account's current usage and billing settings because exceeding included allowances can suspend services or result in charges depending on account setup. Render describes Free instances as suitable for hobby projects and previews, not production. See Render's [Docker deploy guide](https://render.com/docs/docker), [web service setup and port binding](https://render.com/docs/web-services), [health checks](https://render.com/docs/health-checks), [compute plans](https://render.com/docs/compute-plans), and [Free instance limitations](https://render.com/docs/free).

Build and test the same `linux/amd64` image locally before pushing changes:

```sh
docker build --platform linux/amd64 -t lunar-lander-explorer:local .
docker run --rm --platform linux/amd64 -p 127.0.0.1:8080:8080 -e PORT=8080 lunar-lander-explorer:local
# Visit http://localhost:8080/_stcore/health and open the app in a browser.
```

**Deployment status:** this repository has not been pushed to GitHub and the app has not been deployed to Render. The previously completed local Docker health and application smoke checks do not verify a hosted deployment.

## Provenance and limits

The original DQN training notebook, helpers, and checkpoint come from Krrish's DeepLearning.AI Coursera coursework, *Unsupervised Learning, Recommenders, Reinforcement Learning*. The standalone inference, evaluation, app, and deployment packaging are subsequent extensions prepared with AI assistance. Original assignment files remain local under ignored `originals/` and are not included in the Render image. Review course sharing terms before publishing them.

The v2 training record and v3 evaluation use different environment versions and measurement conditions. The 50 evaluation seeds are a fixed sample, not a guarantee of performance on new seeds. The saved seed 1000 replay is illustrative, not the benchmark average. The optional LLM explanation is not part of the verified core app.
