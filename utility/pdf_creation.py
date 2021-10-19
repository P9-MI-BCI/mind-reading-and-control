import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib.backends.backend_pdf import PdfPages
from definitions import OUTPUT_PATH
from utility.logger import get_logger
from classes.Window import Window


def convert_dataframe_to_fig(df: pd.DataFrame) -> plt.figure():
    fig, ax = plt.subplots()
    ax.axis('off')  # removes axis from plot
    ax.table(cellText=df.values, cellLoc='left', colLabels=df.columns, rowLabels=df.index, loc='center')
    return fig


def add_pdf_text_page(text: str, font_size: int = 24) -> plt.figure():
    text_page = plt.figure()
    text_page.clf()
    text_page.text(0.5, 0.5, text, size=font_size, ha="center")
    return text_page


def save_results_to_pdf(train_data: [Window], test_data: [Window], results: Dict[str, pd.DataFrame], file_name: str):
    path = os.path.join(OUTPUT_PATH, file_name)

    with PdfPages(path) as pdf:

        pdf.savefig(add_pdf_text_page('Train Windows'))
        for window in train_data:
            pdf.savefig(window.plot(show=False))
            plt.close()

        pdf.savefig(add_pdf_text_page('Test Windows'))
        for window in test_data:
            pdf.savefig(window.plot(show=False))
            plt.close()

        for key, df in results.items():
            df = df.round(3)
            fig = convert_dataframe_to_fig(df)
            plt.title(key)
            pdf.savefig(fig, bbox_inches='tight', dpi=100)
            plt.close()

        get_logger().info(f'Pdf file written to {path}')
