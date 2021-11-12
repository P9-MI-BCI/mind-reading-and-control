from dash import html
from dash import dcc

"""Following utility has been found from various Dash.com/gallery open-source examples"""


# Return a dash definition of an HTML table for a dataframe
def make_dash_table(df):
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def card(children, **kwargs):
    return html.Section(className="card", children=children, **_omit(["style"], kwargs))


def create_slider(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
        ],
    )
