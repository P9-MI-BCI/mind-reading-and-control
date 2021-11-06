import pandas as pd
import plotly.express as px

# Dash imports
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

df = pd.read_csv('cue_set1.csv')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "BCI Real-Time Movement Detection"

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

layout = dict(
        font={"color": "#fff"},
        height=200,
        xaxis={
            "showline": True,
            "zeroline": False,
            "title": "Frequency (Hz)",
        },
        yaxis={
            "showgrid": True,
            "showline": True,
            "zeroline": False,
            "title": "Amplitude (µV)",
        },
    )

channel = 4
channel_df = pd.DataFrame({
    "Channel": "4",
    "Amplitude": df.iloc[:,channel],
    "Frequency": df.index
})
eeg_graph = dcc.Graph(
    id='eeg-graph',
    figure=px.line(channel_df, x="Frequency", y="Amplitude", color="Channel"),

)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([eeg_graph]),

    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
