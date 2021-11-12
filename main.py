import pandas as pd
import json
import logging
from classes.Dict import AttrDict
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger
from flask import Flask
import dash
import dash_bootstrap_components as dbc

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings

server = Flask(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], server=server, url_base_pathname='/simulation/',
                suppress_callback_exceptions=True, update_title=None)
app.title = "BCI Real-Time Movement Detection"


@server.route('/simulation/')
def main():
    with open('config.json') as config_file, open('script_parameters.json') as script_parameters:
        config = json.load(config_file)['cue_set0']  # Choose config
        script_params = json.load(script_parameters)  # Load script parameters

    script_params = AttrDict(script_params)
    config = AttrDict(config)

    dispatch(script_params, config)


if __name__ == '__main__':
    main()
