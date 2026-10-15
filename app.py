"""Interactive explorer for the user's fixed Lunar Lander DQN checkpoint."""
import pandas as pd
import streamlit as st

from lander.benchmark import load_benchmark
from lander.policy import ROOT
from lander.simulation import ACTIONS, run_episode

st.set_page_config(page_title='Lunar Lander Agent Explorer', page_icon='🌙', layout='wide')
st.markdown('''
<style>
.stApp {background: #0a1020; color: #e9effb;}
.block-container {max-width: 1240px; padding: 2.4rem clamp(1rem, 4vw, 3rem) 3rem;}
h1, h2, h3 {letter-spacing: -.035em;}
h1 {margin: .25rem 0 .5rem; font-size: clamp(2rem, 4vw, 3rem);}
h2, h3 {margin-top: 1.6rem;}
[data-testid="stMetric"] {background: linear-gradient(145deg, #151f36, #111a2e);
  border: 1px solid #2d3d5c; border-radius: 14px; padding: 1rem 1.1rem;}
[data-testid="stMetricLabel"] {color:#aebdd7;}
.eyebrow {color:#84c8d5; font-size:.78rem; font-weight:650; text-transform:uppercase; letter-spacing:.16em;}
.subtitle {color:#b5c3da; font-size:1.05rem; line-height:1.65; max-width:790px; margin-bottom:1.6rem;}
[data-testid="stTabs"] [role="tablist"] {gap:.55rem; border-bottom:1px solid #263550;}
[data-testid="stTabs"] button[role="tab"] {padding:.7rem 1rem;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#8bd3df;}
div.stButton > button[kind="primary"] {min-height:2.8rem; font-weight:650;}
div[data-testid="stDataFrame"] {border:1px solid #263550; border-radius:10px; overflow:hidden;}
@media (max-width: 700px) {
  .block-container {padding:1.3rem 1rem 2rem;}
  [data-testid="stMetric"] {padding:.8rem;}
  [data-testid="stTabs"] [role="tablist"] {gap:.15rem;}
  [data-testid="stTabs"] button[role="tab"] {padding:.55rem .45rem; font-size:.85rem;}
}
</style>
''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Fixed checkpoint · Gymnasium LunarLander-v3</div>', unsafe_allow_html=True)
st.title('Lunar Lander Agent Explorer')
st.markdown('<div class="subtitle">Watch a seeded DQN episode, inspect its decisions, and compare the saved 50-seed evaluation with a random policy.</div>', unsafe_allow_html=True)

try:
    summary, rows = load_benchmark()
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Benchmark data could not be loaded: {exc}')
    st.stop()

replay_tab, benchmark_tab, method_tab = st.tabs(['Episode replay', 'Benchmark', 'Methodology'])

with replay_tab:
    st.subheader('Run an episode')
    st.caption('Choose a policy and seed, then record a bounded episode (at most 1,000 steps). Repeating a seed and policy is deterministic in this setup.')
    controls, display = st.columns([1, 2], gap='large')
    with controls:
        agent = st.selectbox('Policy', ['DQN', 'Random'])
        seed = st.number_input('Seed', min_value=0, max_value=1_000_000, value=1000, step=1,
                               help='The benchmark used seeds 1000–1049. You can explore any seed in this range.')
        run = st.button('Run episode', type='primary', width='stretch')
        if run:
            st.session_state.pop('episode', None)
            try:
                with st.spinner('Simulating and rendering the episode…'):
                    result, trace, video = run_episode(int(seed), agent, record=True, trace=True)
                st.session_state['episode'] = (result, trace, video)
            except Exception as exc:
                st.error(f'Episode could not be generated: {exc}. Check the selected seed and local simulation dependencies, then try again.')
        if 'episode' not in st.session_state:
            st.info('A saved DQN replay for seed 1000 is ready. Run an episode to see telemetry.')
    with display:
        episode = st.session_state.get('episode')
        if episode:
            result, trace, video = episode
            if video:
                st.video(video)
                st.caption(f"Newly generated replay · {result['agent']} policy · seed {result['seed']}")
            else:
                st.warning('The episode ran, but its video could not be rendered. Episode metrics and telemetry are still available.')
        else:
            replay = ROOT / 'assets/evaluation_seed_1000.mp4'
            if replay.exists():
                st.video(str(replay))
                st.caption('Saved illustrative DQN replay · seed 1000 · generated in the prior evaluation environment')
            else:
                st.warning('Saved replay is unavailable. Run an episode to generate one.')

    if episode:
        result, trace, _ = episode
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total reward', f"{result['reward']:.1f}")
        c2.metric('Observed outcome', result['outcome'])
        c3.metric('Steps', result['steps'])
        c4.metric('Reward ≥ 200', 'Yes' if result['score_ge_200'] else 'No')
        st.caption('Reward ≥ 200 is a score threshold, not a landing success measure. Outcome is derived from the simulator termination state.')
        chart_left, chart_right = st.columns([2, 1], gap='large')
        with chart_left:
            st.markdown('#### Reward trajectory')
            frame = pd.DataFrame(trace)
            st.line_chart(frame.set_index('step')[['total_reward']], height=270)
            with st.expander('Per-step telemetry'):
                st.dataframe(frame, width='stretch', hide_index=True)
                st.download_button('Download telemetry CSV', frame.to_csv(index=False),
                                   file_name=f"episode_{result['agent'].lower()}_{result['seed']}.csv",
                                   mime='text/csv')
        with chart_right:
            st.markdown('#### Action counts')
            counts = pd.DataFrame({'Action': ACTIONS, 'Steps': result['action_counts']}).set_index('Action')
            st.bar_chart(counts, height=270)

with benchmark_tab:
    st.subheader('Recorded 50-seed Gymnasium v3 evaluation')
    st.caption('Fixed DQN checkpoint and uniform random policy, evaluated on the same seeds 1000–1049 in Gymnasium LunarLander-v3. Values below come from the saved episode results.')
    stats = summary['statistics']
    c1, c2, c3 = st.columns(3)
    c1.metric('DQN mean reward · n=50', f"{stats['DQN']['mean_reward']:.1f}")
    c2.metric('Random mean reward · n=50', f"{stats['Random']['mean_reward']:.1f}")
    c3.metric('Mean reward difference', f"{stats['DQN']['mean_reward'] - stats['Random']['mean_reward']:.1f}")
    metrics = pd.DataFrame({
        'Policy': ['DQN', 'Random'],
        'Episodes': [stats[a]['episodes'] for a in ('DQN', 'Random')],
        'Mean reward': [stats[a]['mean_reward'] for a in ('DQN', 'Random')],
        'SD': [stats[a]['std_reward'] for a in ('DQN', 'Random')],
        'Median': [stats[a]['median_reward'] for a in ('DQN', 'Random')],
        'Min': [stats[a]['min_reward'] for a in ('DQN', 'Random')],
        'Max': [stats[a]['max_reward'] for a in ('DQN', 'Random')],
        'Reward ≥ 200 threshold': [f"{stats[a]['score_ge_200_rate']:.0%}" for a in ('DQN', 'Random')],
    })
    st.caption('Reward ≥ 200 is only a score threshold; it does not indicate landing success.')
    st.dataframe(metrics, width='stretch', hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format='%.1f')
                                for name in ('Mean reward', 'SD', 'Median', 'Min', 'Max')})
    frame = pd.DataFrame(rows)
    paired = frame.pivot(index='seed', columns='agent', values='reward')[['DQN', 'Random']]
    st.markdown('#### Reward by seed')
    st.line_chart(paired, height=340)
    chosen = st.select_slider('Inspect a benchmark seed', options=summary['seeds'], value=1000)
    st.dataframe(frame[frame.seed == chosen][['agent', 'reward', 'steps', 'outcome', 'score_ge_200']],
                 width='stretch', hide_index=True)
    with st.expander('All raw episode results'):
        st.dataframe(frame, width='stretch', hide_index=True)
        st.download_button('Download benchmark CSV', (ROOT / 'results/episodes.csv').read_bytes(),
                           file_name='lunar_lander_v3_episodes.csv', mime='text/csv')

with method_tab:
    st.subheader('What these results mean')
    st.markdown('''
#### Agent and provenance
The original DeepLearning.AI course DQN uses an eight-value observation vector and Dense layers of sizes **64 → 64 → 4**. This app loads the saved checkpoint’s exported NumPy weights and selects the highest-value action (greedy inference). The model is fixed: this application does not train or modify it.

#### Evaluation protocol
The saved evaluation creates a fresh Gymnasium `LunarLander-v3` environment per episode. The fixed DQN and uniform random baseline share seeds **1000–1049**, with default gravity, no wind, and a 1,000-step cap. The selected replay is one illustrative episode, not the mean episode. Sample standard deviation summarizes variation within these 50 recorded episodes.

#### Historical training result
The original course notebook reports a rolling mean reward of **200.05** over the final 100 *training* episodes at episode 455 in OpenAI Gym `LunarLander-v2`. This historical v2 result is not the Gymnasium v3 evaluation shown in the Benchmark tab.

#### Interpretation and limitations
Training and evaluation use different environment versions; evaluation over 50 fixed seeds does not establish general reliability. **Reward ≥ 200 is a score threshold, not a landing-success label.** Simulator outcomes and per-step telemetry are available for episodes generated in this app.
''')
    st.caption(f"Evaluated model SHA-256: {summary['model_sha256']} · Gymnasium {summary['versions']['gymnasium']} · NumPy {summary['versions']['numpy']}")
