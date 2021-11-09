import pandas as pd

# Dash imports
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Application imports
from app import app
from pages import simulation_page, parameter_page


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
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
        html.H3("Navigation", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Simulation", href="/", active="exact"),
                dbc.NavLink("Parameter Configuration", href="/param-config", active="exact"),
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

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return simulation_page.layout
    elif pathname == "/param-config":
        return parameter_page.layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=3000)
