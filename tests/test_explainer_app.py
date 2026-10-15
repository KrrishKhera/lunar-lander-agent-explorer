import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from lander.explainer import ExplainerError


class ExplainerAppTests(unittest.TestCase):
    def test_completed_episode_explainer_is_opt_in_and_renders_answer(self):
        result = {
            'agent': 'DQN', 'seed': 1000, 'reward': 12.5, 'steps': 2,
            'outcome': 'Terminated', 'score_ge_200': False, 'terminated': True,
            'truncated': False, 'action_counts': [1, 0, 1, 0],
        }
        trace = [
            {'step': 1, 'reward': 1.0, 'total_reward': 1.0, 'action': 'Coast',
             'x': 0.0, 'y': 0.5, 'angle': 0.1},
            {'step': 2, 'reward': 11.5, 'total_reward': 12.5, 'action': 'Main engine',
             'x': 0.1, 'y': 0.4, 'angle': 0.0},
        ]
        answer = 'Recorded outcome: terminated. Hypothesis: descent may have been too fast.'
        app = AppTest.from_file('app.py')
        with patch('lander.explainer.explain_episode', return_value=answer) as mocked:
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(mocked.call_count, 0)
            self.assertTrue(any(tab.label == 'Benchmark' for tab in app.tabs))
            self.assertTrue(any('Recorded 50-seed Gymnasium v3 evaluation' in item.value
                                for item in app.subheader))
            self.assertTrue(any('Saved illustrative DQN replay' in item.value for item in app.caption))
            app.session_state['episode'] = (result, trace, None)
            app.run(timeout=30)
            app.button(key='explain_episode').click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(any(answer in item.value for item in app.markdown))
        self.assertTrue(any('AI-generated interpretation' in item.value for item in app.info))

    def test_unavailable_ollama_error_is_shown_without_app_exception(self):
        result = {
            'agent': 'Random', 'seed': 9, 'reward': -4.0, 'steps': 1,
            'outcome': 'Terminated', 'score_ge_200': False, 'terminated': True,
            'truncated': False, 'action_counts': [0, 1, 0, 0],
        }
        trace = [{'step': 1, 'reward': -4.0, 'total_reward': -4.0,
                  'action': 'Left engine', 'x': 0.0, 'y': 0.1, 'angle': 0.0}]
        app = AppTest.from_file('app.py')
        app.session_state['episode'] = (result, trace, None)
        message = 'Could not reach local Ollama. Install and start Ollama, then retry.'
        with patch('lander.explainer.explain_episode', side_effect=ExplainerError(message)):
            app.run(timeout=30)
            app.button(key='explain_episode').click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(message in item.value for item in app.error))


if __name__ == '__main__':
    unittest.main()
