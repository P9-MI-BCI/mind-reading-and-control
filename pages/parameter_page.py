import numpy as np
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import utility.dash_util as drc

from main import app

layout = html.Div([
    html.H6("Parameters"),
    html.Hr(),
    html.P(
        "Change the parameters for pipeline."
    ),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Cue Set 1', 'value': 'cue_set0'},
            {'label': 'Cue Set 2', 'value': 'cue_set1'}
        ],
        value='cue_set0'
    ),
    drc.card(
        id="button-card",
        children=[
            drc.create_slider(
                name="Threshold",
                id="slider-threshold",
                min=0,
                max=1,
                value=0.5,
                step=0.01,
            ),
            html.Button(
                "Reset Threshold",
                id="button-zero-threshold",
            ),
        ],
    ),
])

@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value
