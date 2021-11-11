import pandas as pd

# Dash imports
from dash import dcc
from dash import html

# Application imports
from flask import request
from main import app, server
from pages import sidebar


content = html.Div(id="page-content", style={
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
})

app.layout = html.Div([dcc.Location(id="url"), sidebar.layout, content])


if __name__ == "__main__":
    app.run_server(debug=True, port=3000)
