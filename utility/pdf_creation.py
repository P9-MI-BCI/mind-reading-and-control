import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib.backends.backend_pdf import PdfPages
from definitions import OUTPUT_PATH
from utility.logger import get_logger


def convert_dataframe_to_fig(df: pd.DataFrame) -> plt.figure():
    fig, ax = plt.subplots()
    ax.axis('off')  # removes axis from plot
    ax.table(cellText=df.values, cellLoc='left', colLabels=df.columns, rowLabels=df.index, loc='center')
    return fig


def save_results_to_pdf(results: Dict[str, pd.DataFrame], name: str):
    path = os.path.join(OUTPUT_PATH, name)
    with PdfPages(path) as pdf:
        for key, df in results.items():
            fig = convert_dataframe_to_fig(df)
            plt.title(key)
            pdf.savefig(fig, bbox_inches='tight', dpi=100)
            get_logger().info(f'Writing scores to pdf file at {path}')
