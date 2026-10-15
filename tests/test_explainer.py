import json
import unittest
from io import BytesIO
from urllib.error import HTTPError, URLError
from unittest.mock import Mock

from lander.explainer import (
    DEFAULT_MODEL,
    ExplainerError,
    explain_episode,
    summarize_episode,
)


def episode_fixture():
    result = {
        'agent': 'DQN', 'seed': 1000, 'reward': 215.25, 'steps': 20,
        'outcome': 'Landed / at rest', 'terminated': True, 'truncated': False,
        'action_counts': [5, 4, 8, 3],
    }
    trace = [
        {'step': step, 'reward': 1.0, 'total_reward': float(step),
         'action': ('Main engine' if step < 15 else 'Left engine'),
         'x': step / 10, 'y': step / 20, 'angle': step / 100}
        for step in range(1, 21)
    ]
    return result, trace


class ExplainerTests(unittest.TestCase):
    def test_summary_uses_real_episode_facts_and_bounds_trace(self):
        result, trace = episode_fixture()
        summary = summarize_episode(result, trace)['recorded_facts']
        self.assertEqual(summary['cumulative_reward'], 215.25)
        self.assertEqual(summary['simulator_outcome'], 'Landed / at rest')
        self.assertEqual(summary['action_counts']['Main engine'], 8)
        self.assertEqual(len(summary['trajectory_samples']), 8)
        self.assertEqual(len(summary['action_runs_sample']), 2)
        self.assertNotIn('reward', summary['trajectory_samples'][0])

    def test_calls_local_api_and_returns_explanation(self):
        result, trace = episode_fixture()
        opener = Mock(return_value=BytesIO(json.dumps({
            'message': {'content': 'Recorded outcome: at rest. Hypothesis: the sampled path suggests a controlled descent.'}
        }).encode()))
        answer = explain_episode(result, trace, opener=opener)
        self.assertIn('Hypothesis', answer)
        request = opener.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'http://localhost:11434/api/chat')
        self.assertEqual(payload['model'], DEFAULT_MODEL)
        self.assertFalse(payload['stream'])
        self.assertIn('trajectory_samples', payload['messages'][1]['content'])

    def test_unavailable_service_has_practical_message(self):
        result, trace = episode_fixture()
        with self.assertRaisesRegex(ExplainerError, 'Install and start Ollama'):
            explain_episode(result, trace, opener=Mock(side_effect=URLError('connection refused')))
        with self.assertRaisesRegex(ExplainerError, 'Confirm the model is loaded'):
            explain_episode(result, trace, opener=Mock(side_effect=TimeoutError()))

    def test_missing_model_and_empty_response_are_graceful(self):
        result, trace = episode_fixture()
        missing = HTTPError('http://localhost:11434/api/chat', 404, 'Not Found', {},
                            BytesIO(b'{"error":"model not found"}'))
        with self.assertRaisesRegex(ExplainerError, 'ollama pull'):
            explain_episode(result, trace, opener=Mock(side_effect=missing))
        empty = BytesIO(b'{"message":{"content":"  "}}')
        with self.assertRaisesRegex(ExplainerError, 'ollama list'):
            explain_episode(result, trace, opener=Mock(return_value=empty))

    def test_server_error_and_invalid_payload_are_graceful(self):
        result, trace = episode_fixture()
        failed = HTTPError('http://localhost:11434/api/chat', 500, 'Server Error', {},
                           BytesIO(b'backend error'))
        with self.assertRaisesRegex(ExplainerError, 'HTTP 500'):
            explain_episode(result, trace, opener=Mock(side_effect=failed))
        invalid = BytesIO(b'not-json')
        with self.assertRaisesRegex(ExplainerError, 'invalid response'):
            explain_episode(result, trace, opener=Mock(return_value=invalid))


if __name__ == '__main__':
    unittest.main()
