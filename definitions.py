from dotenv import load_dotenv
import os

load_dotenv()

get_env = os.getenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.getenv("DATASET_PATH"))
INPUT_PATH = os.path.join(ROOT_DIR, "input")
OUTPUT_PATH = os.path.join(os.getenv("OUTPUT_PATH"))
DB_PATH = os.path.join(os.getenv("DB_PATH"))
CONFIG_PATH = os.path.normpath(os.getenv("CONFIG_PATH"))
