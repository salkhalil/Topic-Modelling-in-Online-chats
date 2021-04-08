import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pickle as pk
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import minmax_scale
from Saving import *
from tsmoothie.smoother import *
from block_dev import *

fc = pd.read_csv('full_corpus.csv', index_col='Unnamed: 0')
pypl = fc.loc[(fc.Comp == 'pypl') & (fc.message != 'n')].iloc[-5000:]
embs = openPk('full_embs.pkl')
embs = embs[5]
t_vecs = openPk('topNg_vecs.pkl')
t_vecs = t_vecs[5]

t_vecs_plot = openPk('top_vecs4plot.pkl')
comps = ['aapl', 'abb', 'amzn', 'aon', 'bmy', 'cern', 'csco', 'ebay', 'hsbc', 'jpm', 'mmm', 'nflx', 'pypl']
tg_words = pd.read_csv('topic_words_wGram.csv', index_col='Topic').iloc[1:]
tNg_words = pd.read_csv('topic_words_noGram.csv', index_col='Topic').iloc[1:]

fig = go.Figure(data=go.Scatter(x=pypl['createdAt'], y=pypl['deviation'], hovertext=pypl['message']))

Deviation = [
    dbc.CardHeader(html.H5('Measuring average devtiation over time - PayPal')),
    dbc.CardBody([
        dbc.Row([
            dcc.Graph(id='topic-deviation', figure = fig)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='wind-num', children='hdbdhbs'),
                dcc.Slider(id='wind-slider', min = 2, max = 100, step=1, value=4)
            ])
        ])
    ])
]

Top_vec_comp = [
    dbc.CardHeader(html.H5('Comparing Topic Vectors')),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id = 't_set1',
                    options = [
                        {'label': 'No-Grams 2D', 'value': 'ng2D'},
                        {'label': 'No-Grams 5D r-dims', 'value': 'ng5D'},
                        {'label': 'With-Grams 2D', 'value': 'g2D'},
                        {'label': 'With-Grams 5D r-dims', 'value': 'g5D'},
                    ],
                    placeholder = 'Graph 1',
                    value = 'ng2D',
                ), width={'size': 3}
            ),
            dbc.Col(
                dcc.Dropdown(
                    id = 't_set2',
                    options = [
                        {'label': 'No-Grams 2D', 'value': 'ng2D'},
                        {'label': 'No-Grams 5D r-dims', 'value': 'ng5D'},
                        {'label': 'With-Grams 2D', 'value': 'g2D'},
                        {'label': 'With-Grams 5D r-dims', 'value': 'g5D'},
                    ],
                    placeholder = 'Graph 1',
                    value = 'ng5D',
                ), width={'size': 3}
            )
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='t_graph1')
            ),
            dbc.Col(
                dcc.Graph(id='t_graph2')
            )
        ])
    ])
]

dev_from_start = [
    dbc.CardHeader(html.H5('Measuring deviation from start of chat')),
    dbc.CardBody([
        dbc.Row([
            dcc.Graph(id='dev_s_g')
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='window-size', children='Enter starting window size:'),
                dcc.Slider(id='dev-wind-slider', min = 1, max = 100, step=1, value=4)
            ]),
            dbc.Col(
                dcc.Dropdown(
                    id = 'ds_comp',
                    options = [{'label': i, 'value': i} for i in comps],
                    placeholder = 'Company',
                    value = 'aapl',
                ), width={'size': 3}
            )
        ])
    ])
]

dev_from_main = [
    dbc.CardHeader(html.H5('Measuring deviation from main topic')),
    dbc.CardBody([
        dbc.Row([
            dcc.Graph(id='dev_m_g')
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id = 'dm_comp',
                    options = [{'label': i, 'value': i} for i in comps],
                    placeholder = 'Company',
                    value = 'aapl'
                ), width={'size': 3}
            ),
            dbc.Col(
                dcc.Dropdown(
                    id = 'smoother',
                    options = [
                        {'label': 'No Smoothing', 'value': 'n'},
                        {'label': 'Exponential', 'value': 'exp'},
                        {'label': 'Convolutional', 'value': 'conv'},
                        {'label': 'Kalman', 'value': 'k'}
                    ],
                    placeholder = 'Smoother',
                    value = 'n'
                ), width={'size': 3, 'offset': 3}
            )
        ]),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(id='stats', style={'white-space': 'pre-wrap'})
            )
        )
    ])
]

block_devs = [
    dbc.CardHeader(html.H5('Deviation Scores per Block')),
    dbc.CardBody([
        dbc.Row([
            dcc.Graph(id='block_dev')
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='b_wind', children='Enter block window size:'),
                dcc.Slider(id='b-wind-slider', min = 1, max = 200, step=2, value=50),
                html.Div(id='b-disc', children='Discount factor:'),
                dcc.Slider('b_disc_f', min=0, max = 1, step=0.02, value=0.3)
            ]),
            dbc.Col([
                dcc.Dropdown(
                    id = 'bd_comp',
                    options = [{'label': i, 'value': i} for i in comps],
                    placeholder = 'Company',
                    value = ['aapl'],
                    multi = True
                ),
                dcc.Checklist(
                    id='auto',
                    options = [{'label': 'Auto-Block', 'value': 'ab'}],
                    value = ['ab']
                )], width={'size': 3}
            )
        ]),
        dbc.Row([
            dbc.Col(
            dcc.Markdown(id='block_stats', style={'white-space': 'pre-wrap'})
            ),
            dbc.Col(
                dcc.Dropdown(
                    id = 'block_smoother',
                    options = [
                        {'label': 'No Smoothing', 'value': 'n'},
                        {'label': 'Convolutional', 'value': 'conv'},
                        {'label': 'Kalman', 'value': 'k'}
                    ],
                    placeholder = 'Smoother',
                    value = 'n'
                ), width={'size': 3, 'offset': 3})
            ])
    ])
]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    html.H1('Topic Modelling', style={'align': 'center'}),
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(Deviation))
            ], style={'marginTop': 10}),
        dbc.Row([
            dbc.Col(dbc.Card(Top_vec_comp))
            ], style={'marginTop': 10}),
        dbc.Row([
            dbc.Col(dbc.Card(dev_from_start))
        ], style={'marginTop': 10}),
        dbc.Row([
            dbc.Col(dbc.Card(dev_from_main))
        ], style={'marginTop': 10}),
        dbc.Row([
            dbc.Col(dbc.Card(block_devs))
        ], style={'marginTop': 10})
    ])
])
# Block-level Deviation
@app.callback(
    Output('b-wind-slider', 'value'),
    [Input('auto', 'value'), Input('bd_comp', 'value')]
)
def check(val, comps):
    if val == ['ab']:
        winds = []
        for comp in comps:
            our_df = fc.loc[(fc.Comp == comp) & (fc.t2v != 'n')]
            _, wind = split_df_auto(our_df, 0.05)
            winds.append(wind)
        return max(winds)
    return 50
@app.callback(
    Output('b-disc', 'children'),
    [Input('b_disc_f', 'value')]
)
def show_df(val):
    return 'Enter discount size: ' + str(val)
@app.callback(
    Output('b_wind', 'children'),
    [Input('b-wind-slider', 'value')]
)
def show_bs(w):
    return 'Block Size: {} hours'.format(w)

@app.callback(
    [Output('block_dev', 'figure'), Output('block_stats', 'children')],
    [Input('bd_comp', 'value'), Input('b-wind-slider', 'value'), Input('block_smoother', 'value'), Input('b_disc_f', 'value')]
)
def block_main(comps, wind, s, disc_f):
    fig = go.Figure()
    stats = ''
    main_ts = []
    for comp in comps:
        main_top, stat = get_block_devs(fig, comp, wind, s, disc_f)
        main_ts.append(main_top)
        stats += stat

    fig.update_layout(
        title="Deviation per block for {}".format(', '.join(comps)),
        xaxis_title="Block Number",
        yaxis_title="Deviation",
        legend_title="Legend"
    )

    return fig, stats
# Dev from main callbacks
@app.callback(
    Output('dev_m_g', 'figure'),
    Output('stats', 'children'),
    [Input('dm_comp', 'value'), Input('smoother', 'value')]
)
def dev_from_main(comp, s):
    our_df = fc.loc[(fc.Comp == comp) & (fc.t2v != 'n')]
    main_top = our_df['top_Ng'].value_counts().index.values[0]

    if main_top == -1:
        main_top = our_df['top_Ng'].value_counts().index.values[1]

    main_top_vec = t_vecs[main_top, :]

    nTops = t_vecs.shape[0]
    main_vec = t_vecs[main_top, :]
    dist_dict = {}
    dist_dict[-1] = 0
    for i in range(nTops):
        cur_top = t_vecs[i, :]
        similarity = np.linalg.norm(cur_top - main_vec)
        dist_dict[i] = similarity


    dists = our_df['top_Ng'].apply(lambda x: dist_dict[x])

    if s == 'exp':
        smoother = ExponentialSmoother(window_len=20, alpha=0.1)
        smoother.smooth(dists)
        dists = smoother.smooth_data[0]

    if s == 'conv':
        smoother = ConvolutionSmoother(window_len=20, window_type='ones')
        smoother.smooth(dists)
        dists = smoother.smooth_data[0]

    if s == 'k':
        smoother = KalmanSmoother(component='level_trend',
                          component_noise={'level':0.1, 'trend':0.1})
        smoother.smooth(dists)
        dists = smoother.smooth_data[0]

    fig = go.Figure(data=go.Scatter(x=our_df['createdAt'], y=dists, hovertext=our_df['message']))

    max_dist, min_dist = dists.max(), dists.min()
    dists_norm = (dists -  min_dist) / (max_dist - min_dist)
    score =  1 - np.mean(dists_norm)

    topic_words = tNg_words['Words'].iloc[main_top]
    topic_words = ', '.join(topic_words.split(';'))
    desc = '**Main topic:** {} \n**Topic words:** {} \n**Score:** {}'.format(str(main_top), topic_words, str(score))
    return fig, desc

# Dev from start callbacks
@app.callback(
    Output('window-size', 'children'),
    [Input('dev-wind-slider', 'value')]
)
def set_wind_text(val):
    return "Starting window size: {}".format(str(val))

@app.callback(
    Output('dev_s_g', 'figure'),
    [Input('dev-wind-slider', 'value'), Input('ds_comp', 'value')]
)
def plot_dFs(wind, comp):
    our_df = fc.loc[(fc.Comp == comp) & (fc.t2v != 'n')]
    inds = our_df.index.values
    nd = embs.shape[1]
    # get first average of first n messages for starting point
    firstInds = inds[:wind]
    start_vec = np.zeros(nd)
    for i in firstInds:
        start_vec += embs[i, :]
    start_vec /= wind

    dists = []
    dates = []
    msg = []
    for i in inds[wind:]:
        dist_vec = embs[i,:] - start_vec
        dists.append(np.linalg.norm(dist_vec))
        dates.append(fc['createdAt'].iloc[i])
        msg.append(fc['message'].iloc[i])

    fig = go.Figure(data=go.Scatter(x=dates, y=dists, hovertext=msg))
    return fig
# comparing t-vecs callbacks
@app.callback(
    Output('t_graph1', 'figure'),
    [Input('t_set1', 'value')]
)
def plot_g1(val):
    switcher = {'ng2D': t_vecs_plot['ng'][2],
                'ng5D': t_vecs_plot['ng'][5],
                'g2D': t_vecs_plot['g'][2],
                'g5D': t_vecs_plot['g'][5]}

    t_switcher = {'ng2D': tNg_words,
                    'ng5D': tNg_words,
                    'g2D': tg_words,
                    'g5D': tg_words}

    df = pd.DataFrame(data=switcher[val], columns=['x', 'y'])
    fig = go.Figure(data=go.Scatter(x=df['x'], y=df['y'], hovertext=t_switcher[val].Words, mode='markers'))

    return fig

@app.callback(
    Output('t_graph2', 'figure'),
    [Input('t_set2', 'value')]
)
def plot_g2(val):
    switcher = {'ng2D': t_vecs_plot['ng'][2],
                'ng5D': t_vecs_plot['ng'][5],
                'g2D': t_vecs_plot['g'][2],
                'g5D': t_vecs_plot['g'][5]}

    t_switcher = {'ng2D': tNg_words,
                    'ng5D': tNg_words,
                    'g2D': tg_words,
                    'g5D': tg_words}

    df = pd.DataFrame(data=switcher[val], columns=['x', 'y'])
    fig = go.Figure(data=go.Scatter(x=df['x'], y=df['y'], hovertext=t_switcher[val].Words, mode='markers'))

    return fig

# paypal plot callbacks

@app.callback(
    Output('wind-num', 'children'),
    [Input('wind-slider', 'value')]
)
def show_num(w):
    return 'Window Size: {}'.format(str(w))

@app.callback(
    Output('topic-deviation', 'figure'),
    [Input('wind-slider', 'value')]
)
def plot_deviation(window):
    devs = []
    inds = pypl.index.values
    last_embs = np.zeros((window, embs.shape[1]))
    last_devs = np.zeros(window)
    rng = range(len(inds))
    devs = []
    for i in rng:
        avg_dev = np.mean(last_devs)
        dev = np.linalg.norm(embs[i, :] - last_embs[(i-1)%window, :])
        last_embs[i%window, :] = embs[i,:]
        last_devs[i%window] = dev
        devs.append((dev-avg_dev)/(avg_dev+1))

    fig = go.Figure(data=go.Scatter(x=pypl['createdAt'], y=devs, hovertext=pypl['message']))
    fig.update_layout(title_text='Topic Deviation with window size: {}'.format(str(window)),
                      title_x=0.5,
                      yaxis_title='Deviation')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
