from plotly.subplots import make_subplots
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from api import get_window_data, get_whole_data
import dash_bootstrap_components as dbc

from main import app

layout = html.Div(
    [
        dbc.Row(dbc.Col(html.Div([
            html.Div(
                [html.H2("Simulation of EEG", className="graph__title")]
            ),
            dcc.Graph(id='live-update-graph'),
            dcc.Interval(
                id='window-update',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ], className='something'))),
        dbc.Row(
            [
                dbc.Col(html.Div([
                    html.H4(
                        ['Simulation Parameters'], className="subtitle padded"
                    ),
                    html.Hr(),
                    html.H6(
                        ['Chosen Dataset:'], className="subtitle padded"
                    ),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[
                            {'label': 'Cue Set 1', 'value': 'cue_set0'},
                            {'label': 'Cue Set 2', 'value': 'cue_set1'}
                        ],
                        value='cue_set0',
                        clearable=False
                    ),
                    html.H6(
                        ['Chosen Dataset:'], className="subtitle padded",
                        style={"margin-top": "15px"}
                    ),
                    dcc.Dropdown(
                        id='channel-dropdown',
                        options=[
                            {'label': 'Channel 1', 'value': '0'},
                            {'label': 'Channel 2', 'value': '1'},
                            {'label': 'Channel 3', 'value': '2'},
                            {'label': 'Channel 4', 'value': '3'},
                            {'label': 'Channel 5', 'value': '4'},
                            {'label': 'Channel 6', 'value': '5'},
                            {'label': 'Channel 7', 'value': '6'},
                            {'label': 'Channel 8', 'value': '7'},
                            {'label': 'Channel 9', 'value': '8'}
                        ],
                        value='4',
                        clearable=False
                    )
                ]), width="auto"),
                dbc.Col(html.Div([
                    html.H4(
                        ['Simulation Metrics'], className="subtitle padded"
                    ),
                    html.Hr(),
                    html.Table()
                ]))
            ]
        )
    ]
)


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('window-update', 'n_intervals'),
              Input('channel-dropdown', 'value'))
def update_graph_live(interval, channel):
    new_window = get_window_data(channel)
    whole_window = get_whole_data(channel)

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                        subplot_titles=(f'Sliding window of Channel {int(channel) + 1}',
                                        f'Trail of Channel {int(channel) + 1}'))

    fig.add_trace({
        'x': new_window.index.values,
        'y': new_window.iloc[:, 0].values
    }, row=1, col=1)

    fig.add_trace({
        'x': whole_window.index,
        'y': whole_window.iloc[:, 0].values,
    }, row=2, col=1)

    fig.update_layout(showlegend=False)

    return fig
