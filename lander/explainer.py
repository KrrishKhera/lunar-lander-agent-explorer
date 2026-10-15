"""Optional, local Ollama explanations for completed Lunar Lander episodes."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = 'qwen2.5:3b-instruct-q4_0'
OLLAMA_URL = 'http://localhost:11434/api/chat'
TIMEOUT_SECONDS = 90
MAX_ACTION_RUNS = 12
MAX_TRAJECTORY_POINTS = 8


class ExplainerError(RuntimeError):
    """A user-facing error from the optional local explainer."""


def summarize_episode(result, trace):
    """Return a bounded summary of recorded episode facts for the LLM."""
    runs = []
    for row in trace:
        action = row.get('action')
        step = int(row.get('step', 0))
        if not action or step < 1:
            continue
        if runs and runs[-1]['action'] == action:
            runs[-1]['end_step'] = step
        else:
            runs.append({'action': action, 'start_step': step, 'end_step': step})

    omitted_runs = max(0, len(runs) - MAX_ACTION_RUNS)
    if omitted_runs:
        runs = runs[:MAX_ACTION_RUNS // 2] + runs[-(MAX_ACTION_RUNS // 2):]

    points = []
    if trace:
        count = min(MAX_TRAJECTORY_POINTS, len(trace))
        indices = sorted({round(i * (len(trace) - 1) / max(1, count - 1))
                          for i in range(count)})
        for index in indices:
            row = trace[index]
            points.append({key: row[key] for key in ('step', 'x', 'y', 'angle')
                           if key in row})

    return {
        'recorded_facts': {
            'policy': result['agent'],
            'seed': result['seed'],
            'cumulative_reward': result['reward'],
            'steps': result['steps'],
            'simulator_outcome': result['outcome'],
            'terminated': result['terminated'],
            'truncated': result['truncated'],
            'action_counts': dict(zip(
                ('Coast', 'Left engine', 'Main engine', 'Right engine'),
                result['action_counts'])),
            'action_runs_sample': runs,
            'action_runs_omitted': omitted_runs,
            'trajectory_samples': points,
            'trajectory_note': 'x, y, and angle samples are recorded after actions; not every step is included.',
        }
    }


def build_messages(summary):
    user_content = (
        'Explain this completed Lunar Lander episode in plain English. Cover the '
        'recorded outcome, notable action or sampled trajectory patterns, and '
        'plausible explanations. Separate statements supported directly by the '
        'recorded facts from hypotheses. Do not claim an action caused an outcome, '
        'infer unrecorded physics or contact state, or treat reward >= 200 as proof '
        'of a successful landing. Mention when the sparse samples limit certainty. '
        'Keep the response concise (about 3 short paragraphs).\n\n'
        'Episode data (JSON):\n'
        + json.dumps(summary, separators=(',', ':'), allow_nan=False)
    )
    return [
        {'role': 'system', 'content': 'You explain logged simulation data cautiously. Do not invent observations.'},
        {'role': 'user', 'content': user_content},
    ]


def explain_episode(result, trace, model=DEFAULT_MODEL, timeout=TIMEOUT_SECONDS,
                    opener=urlopen):
    """Call the local Ollama chat API on explicit user request only."""
    payload = json.dumps({
        'model': model,
        'messages': build_messages(summarize_episode(result, trace)),
        'stream': False,
        'options': {'temperature': 0.2},
    }).encode('utf-8')
    request = Request(OLLAMA_URL, data=payload,
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with opener(request, timeout=timeout) as response:
            body = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:300]
        hint = ' Run `ollama pull ' + model + '` if that model is missing.' if exc.code == 404 else ''
        raise ExplainerError(f'Ollama returned HTTP {exc.code}: {detail}.{hint}') from exc
    except TimeoutError as exc:
        raise ExplainerError(
            f'Ollama took longer than {timeout} seconds. Confirm the model is loaded and retry.'
        ) from exc
    except (URLError, OSError) as exc:
        raise ExplainerError(
            'Could not reach local Ollama. Install and start Ollama, then retry; '
            'the episode and other app features remain available.'
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExplainerError('Ollama returned an invalid response. Check the local Ollama service and retry.') from exc

    message = body.get('message') if isinstance(body, dict) else None
    content = message.get('content', '') if isinstance(message, dict) else ''
    if not isinstance(content, str) or not content.strip():
        raise ExplainerError(
            f'Ollama returned an empty explanation. Check that `{model}` is available with `ollama list`, then retry.'
        )
    return content.strip()
