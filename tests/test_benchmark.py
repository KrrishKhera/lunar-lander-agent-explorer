import statistics
import unittest

from lander.benchmark import load_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_saved_statistics_match_raw_episodes(self):
        summary, rows = load_benchmark()
        for agent in ('DQN', 'Random'):
            rewards = [row['reward'] for row in rows if row['agent'] == agent]
            stats = summary['statistics'][agent]
            self.assertEqual(len(rewards), stats['episodes'])
            self.assertAlmostEqual(statistics.mean(rewards), stats['mean_reward'], places=6)
            self.assertAlmostEqual(statistics.stdev(rewards), stats['std_reward'], places=6)
            self.assertAlmostEqual(
                sum(reward >= 200 for reward in rewards) / len(rewards),
                stats['score_ge_200_rate'], places=6)


if __name__ == '__main__':
    unittest.main()
