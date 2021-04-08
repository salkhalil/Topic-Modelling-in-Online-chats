from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from Saving import *
from tsmoothie.smoother import *

fc = pd.read_csv('full_corpus.csv', index_col='Unnamed: 0')
pypl = fc.loc[(fc.Comp == 'pypl') & (fc.message != 'n')].iloc[-5000:]
embs = openPk('full_embs.pkl')
embs = embs[5]
t_vecs = openPk('topNg_vecs.pkl')
t_vecs = t_vecs[5]

comps = ['aapl', 'abb', 'amzn', 'aon', 'bmy', 'cern', 'csco', 'ebay', 'hsbc', 'jpm', 'mmm', 'nflx', 'pypl']
tg_words = pd.read_csv('topic_words_wGram.csv', index_col='Topic').iloc[1:]
tNg_words = pd.read_csv('topic_words_noGram.csv', index_col='Topic').iloc[1:]

def window_check(t1, t2, window):
    t1_sec, t2_sec = time.mktime(t1), time.mktime(t2)
    wind_sec = window*60**2
    if t2_sec - t1_sec < wind_sec and t2_sec - t1_sec > 0:
        return True
    return False

def get_time(df, ind, date_col='createdAt'):
    return df[date_col].iloc[ind]
def parse_dtime(time_str):
    parsed_dtime = time.strptime(str(time_str), '%Y-%m-%d %H:%M:%S')
    return parsed_dtime

def split_df(df, window):
    inds = df.index.values
    blocks = []
    cur_ind = 0
    while cur_ind < len(inds):
        j = 1
        cur_t = get_time(df, cur_ind)

        if cur_ind + j >= len(inds):
            blocks.append((cur_ind, cur_ind))
            break
        next_t = get_time(df, cur_ind+j)

        while window_check(parse_dtime(cur_t), parse_dtime(next_t), window) and cur_ind+j < len(inds):
            j += 1
            if cur_ind+j < len(inds):
                next_t = get_time(df, cur_ind+j)

        blocks.append((cur_ind, cur_ind+j-1))
        cur_ind += j
    return blocks

def calc_score(dists):
    dists_norm = minmax_scale(dists)
    score = np.mean(dists_norm)
    return score

def calc_block_dev(df, block, embs, t_vecs, mod = 'top_Ng'):
    block_start, block_end = block
    block_df = df.iloc[block_start:block_end+1]

    inds = block_df.index.values
    nd = embs.shape[1]
    main_dists = []
    start_dists = []
    block_size = len(inds)
    if block_size == 1:
        main_dists.append(0)
        start_dists.append(0)
        return 0, 0, block_df['top_Ng'].iloc[0]
    else:
        start_bs = block_size // 10
        if start_bs == 0:
            start_bs = 1
        first_inds = inds[:start_bs]
        start_vec = np.zeros(nd)
        for i in first_inds:
            start_vec += embs[i, :]
        start_vec /= start_bs

        for i in inds[start_bs:]:
            dist_vec = embs[i,:] - start_vec
            start_dists.append(np.linalg.norm(dist_vec))

        tops = block_df[mod].value_counts().index.values
        main_top = tops[0]
        if main_top == -1:
            if len(tops) > 1:
                main_top = tops[1]
            else:
                main_dists = [0]
                dfm = calc_score(main_dists)
                dfs = calc_score(start_dists)
                return dfs, dfm, main_top

        main_t_vec = t_vecs[main_top, :]
        nTops = t_vecs.shape[0]
        dist_dict = {}
        dist_dict[-1] = 0
        for i in range(nTops):
            cur_top = t_vecs[i, :]
            sim = np.linalg.norm(cur_top - main_t_vec)
            dist_dict[i] = sim
        main_dists = block_df[mod].apply(lambda x: dist_dict[x])
        main_dists = main_dists.values

        dfm = np.mean(main_dists)
        dfs = np.mean(start_dists)

        return dfs, dfm, main_top

def add_memory(scores, disc_factor, window=10):
    new_scores = np.zeros(len(scores))
    new_scores[0] = scores[0]
    cum_sums = []
    for i in range(len(scores)-1):
        cum_sums.append(scores[i])
        if len(cum_sums) < window:
            disc_scores = np.array([(disc_factor**(len(cum_sums)-j))*c for j,c in enumerate(cum_sums)])
        else:
            disc_scores = np.array([(disc_factor**(len(cum_sums)-j))*c for j,c in enumerate(cum_sums[-window:])])
        new_scores[i+1] = scores[i+1] + np.sum(disc_scores)
    return new_scores

def check_split(blocks, thresh):
    block_dif = [b[1] - b[0] for b in blocks]
    sing_bc = 0
    for d in block_dif:
        if d == 1:
            sing_bc += 1
    if (sing_bc / len(block_dif)) < thresh:
        return True
    return False

def split_df_auto(df, thresh, precision=5):
        wind = 1
        while True:
            blocks = split_df(df, wind)
            if check_split(blocks, thresh):
                return blocks, wind
            wind += precision

def midpoint_times(t1, t2):
    new1 = parse_dtime(t1); new2 = parse_dtime(t2)

    mp = (time.mktime(new2) + time.mktime(new1)) / 2

    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mp))


def get_block_devs(fig, comp, wind, s, disc_f):
    our_df = fc.loc[fc.Comp == comp]
    blocks = split_df(our_df, wind)

    tops, dfs_scores, dfm_scores = [], [], []
    b_start, b_end = [], []
    for b in blocks:
        b_start.append(our_df['createdAt'].iloc[b[0]])
        b_end.append(our_df['createdAt'].iloc[b[1]])
        dfs, dfm, top = calc_block_dev(our_df, b, embs, t_vecs)
        tops.append(top)
        dfs_scores.append(dfs)
        dfm_scores.append(dfm)

    dfm_scores = add_memory(dfm_scores, disc_f)
    dfs_scores = add_memory(dfs_scores, disc_f)

    if s == 'conv':
        smoother = ConvolutionSmoother(window_len=20, window_type='ones')
        smoother.smooth(dfm_scores)
        dfm_scores = smoother.smooth_data[0]
        smoother.smooth(dfs_scores)
        dfs_scores = smoother.smooth_data[0]
    if s == 'k':
        smoother = KalmanSmoother(component='level_trend',
                          component_noise={'level':0.1, 'trend':0.1})
        smoother.smooth(dfm_scores)
        dfm_scores = smoother.smooth_data[0]
        smoother.smooth(dfs_scores)
        dfs_scores = smoother.smooth_data[0]


    dfm_df = pd.DataFrame.from_dict({'Start': b_start, 'End': b_end, 'score': dfm_scores, 'Tops': tops})
    dfs_df = pd.DataFrame.from_dict({'Start': b_start, 'End': b_end, 'score': dfs_scores, 'Tops': tops})

    dfm_df['mp'] = np.vectorize(midpoint_times)(dfm_df['Start'], dfm_df['End'])
    dfs_df['mp'] = dfm_df['mp']

    fig.add_trace(go.Scatter(
        x=dfs_df.mp,
        y=dfs_df.score,
        hovertext = ['Start: {} \nEnd: {} Main Topic: {}'.format(i, j, k) for i, j, k in dfs_df[['Start', 'End', 'Tops']].values],
        name="Dev from Start: {}".format(comp)        # this sets its legend entry
    ))


    fig.add_trace(go.Scatter(
        x=dfm_df.mp,
        y=dfm_df.score,
        hovertext = ['Start: {} \nEnd: {} Main Topic: {}'.format(i, j, k) for i, j, k in dfm_df[['Start', 'End', 'Tops']].values],
        name="Dev from Main: {}".format(comp)       # this sets its legend entry
    ))

    main_top = our_df['top_Ng'].value_counts().index.values[0]

    if main_top == -1:
        main_top = our_df['top_Ng'].value_counts().index.values[1]
    topic_words = tNg_words['Words'].iloc[main_top]
    topic_words = ', '.join(topic_words.split(';'))

    stats  = '**{} stats:**\n**Dev-from-Main Score:** {}\n'.format(comp, str(calc_score(dfm_scores)))
    stats += '**Dev-from-Start Score:** {}\n'.format(str(calc_score(dfs_scores)))
    stats += '**Main topic:** {} \n**Topic words:** {}\n\n'.format(main_top, topic_words)
    return main_top, stats
