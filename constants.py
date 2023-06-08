import os
from dotenv import load_dotenv
from chromadb.config import Settings

import torch

load_dotenv()

runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the folder for storing database

if runtime_device == "cuda":
    local_persist_directory = os.environ.get('PERSIST_DIRECTORY_GPU')
else:
    local_persist_directory = os.environ.get('PERSIST_DIRECTORY_CPU')

PERSIST_DIRECTORY = local_persist_directory

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)
