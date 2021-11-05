import pandas as pd
import plotly.express as px

# Dash imports
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

df = pd.read_csv('cue_set1.csv')

app = dash.Dash(__name__)

app.title = "BCI Real-Time Movement Detection"

colors = {
    'background': 'grey',
    'text': 'black'
}

# fig.update_layout(
#     plot_bgcolor=colors['background'],
#     paper_bgcolor=colors['background'],
#     font_color=colors['text']
# )


eeg_channels = range(0,9)

app.layout = html.Div(children=[
    # html.H1(
    #     children='Hello Fuckheads',
    #     style={
    #         'textAlign': 'center',
    #         'color': colors['text']
    #     }
    # ),
    #
    # html.Div(children='Below is your fucking data. So long.', style={
    #     'textAlign': 'center',
    #     'color': colors['text']
    # }),

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
            {'label': 'Channel 9', 'value': '8'},

        ],
        value='0'
    ),

    html.Div([
        html.Div([
            dcc.Graph(
                id='emg-channel-graph',
                figure=px.line(df['12'])
            )
        ]),
    ])

])


# @app.callback(
#     Output('eeg-channel-graph', 'figure'),
#     Input('channel-dropdown', 'value')
# )
# def update_graph(selected_channel):
#     channel_df = df[selected_channel]
#
#     fig = px.line(channel_df)
#
#     return fig


if __name__ == '__main__':
    app.run_server(debug=True)
