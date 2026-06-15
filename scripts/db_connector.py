from sqlalchemy import create_engine
import os
from pathlib import Path

def get_engine():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    except Exception:
        pass
    db_url = os.environ.get("DB_URL_1")
    return create_engine(db_url)

def get_engine_db():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    except Exception:
        pass
    db_url = os.environ.get("DB_URL_2")
    return create_engine(db_url)

