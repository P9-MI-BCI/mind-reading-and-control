import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from pages import simulation_page, parameter_page

from flask import request
from main import app, server


layout = html.Div(
    [
        html.H4("Navigation", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Simulation", href="/simulation/", active="exact"),
                dbc.NavLink("Parameter Configuration", href="/param-config", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "18rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    },
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/simulation/":
        return simulation_page.layout
    elif pathname == "/param-config":
        return parameter_page.layout
    # If the user tries to reach a different page, return a 404 message
    return "404: Not found"
