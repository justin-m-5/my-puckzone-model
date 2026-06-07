# db.py

from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

def get_client():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_client()

def fetch_all(table, query):
    """Fetch all rows from a table using pagination."""
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        res = query.range(offset, offset + page_size - 1).execute()
        rows = res.data
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def upsert_rows(table, rows, on_conflict=None):
    """Upsert rows into a Supabase table."""
    if not rows:
        return None
    query = supabase.table(table).upsert(rows, on_conflict=on_conflict)
    return query.execute()