# conftest.py  (repository root)
"""
Root-level pytest conftest that prevents db.py from trying to connect to
Supabase when the test suite is collected.

All pipeline unit tests use in-memory DataContext objects (see tests/conftest.py)
and never call the real Supabase client.  The mock below intercepts the
module-level ``supabase = get_client()`` call in db.py so that importing
any module that transitively imports db.py does not require credentials.
"""

import sys
from unittest.mock import MagicMock

# Inject a mock 'db' module before any test file imports it.
# This must happen before conftest.py in tests/ (or any test file) is loaded.
_mock_db = MagicMock()
_mock_db.supabase = MagicMock()
_mock_db.fetch_all = MagicMock(return_value=[])
sys.modules.setdefault("db", _mock_db)
