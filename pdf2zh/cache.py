import logging
import os
import json
import time
from peewee import (
    Model,
    SqliteDatabase,
    AutoField,
    CharField,
    TextField,
    SQL,
    DatabaseError,
    OperationalError,
)
from typing import Optional


# we don't init the database here
db = SqliteDatabase(None)
logger = logging.getLogger(__name__)

# Several translation threads write to the cache at the same time. With the old
# 1 second timeout a busy database made `set` fail silently (the translation was
# then lost from the cache and had to be paid for / waited for again after
# "Continue"). Wait longer and retry a few times instead.
BUSY_TIMEOUT_MS = 10000
SET_RETRIES = 3


class _TranslationCache(Model):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    translate_engine_params = TextField()
    original_text = TextField()
    translation = TextField()

    class Meta:
        database = db
        constraints = [
            SQL(
                """
            UNIQUE (
                translate_engine,
                translate_engine_params,
                original_text
                )
            ON CONFLICT REPLACE
            """
            )
        ]


class TranslationCache:
    @staticmethod
    def _sort_dict_recursively(obj):
        if isinstance(obj, dict):
            return {
                k: TranslationCache._sort_dict_recursively(v)
                for k in sorted(obj.keys())
                for v in [obj[k]]
            }
        elif isinstance(obj, list):
            return [TranslationCache._sort_dict_recursively(item) for item in obj]
        return obj

    def __init__(self, translate_engine: str, translate_engine_params: dict = None):
        assert (
            len(translate_engine) < 20
        ), "current cache require translate engine name less than 20 characters"
        self.translate_engine = translate_engine
        self.replace_params(translate_engine_params)

    # The program typically starts multi-threaded translation
    # only after cache parameters are fully configured,
    # so thread safety doesn't need to be considered here.
    def replace_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params = params
        params = self._sort_dict_recursively(params)
        self.translate_engine_params = json.dumps(params)

    def update_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params.update(params)
        self.replace_params(self.params)

    def add_params(self, k: str, v):
        self.params[k] = v
        self.replace_params(self.params)

    # Since peewee and the underlying sqlite are thread-safe,
    # get and set operations don't need locks.
    def get(self, original_text: str) -> Optional[str]:
        # A failed read must behave like a cache miss. Raising here would make
        # the caller retry the whole paragraph forever.
        try:
            result = _TranslationCache.get_or_none(
                translate_engine=self.translate_engine,
                translate_engine_params=self.translate_engine_params,
                original_text=original_text,
            )
        except Exception as e:
            logger.debug(f"Error reading cache: {e}")
            return None
        return result.translation if result else None

    def set(self, original_text: str, translation: str):
        if translation is None:
            return
        for attempt in range(SET_RETRIES):
            try:
                _TranslationCache.create(
                    translate_engine=self.translate_engine,
                    translate_engine_params=self.translate_engine_params,
                    original_text=original_text,
                    translation=translation,
                )
                return
            except Exception as e:
                logger.debug(f"Error setting cache (try {attempt + 1}): {e}")
                if attempt < SET_RETRIES - 1:
                    time.sleep(0.2 * (attempt + 1))


def _open_db(cache_db_path: str):
    db.init(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": BUSY_TIMEOUT_MS,
        },
    )
    db.create_tables([_TranslationCache], safe=True)


def init_db(remove_exists=False):
    cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "pdf2zh")
    os.makedirs(cache_folder, exist_ok=True)
    # The current version does not support database migration, so add the version number to the file name.
    cache_db_path = os.path.join(cache_folder, "cache.v1.db")
    if remove_exists and os.path.exists(cache_db_path):
        os.remove(cache_db_path)
    try:
        _open_db(cache_db_path)
    except OperationalError:
        raise  # e.g. "database is locked": the file is fine, do not touch it
    except DatabaseError as e:
        # "database disk image is malformed" (for example after a hard kill
        # while writing). Keep the broken file for inspection and start fresh
        # instead of crashing the whole program at import time.
        broken = cache_db_path + ".corrupt"
        logger.warning(f"Cache database is damaged ({e}); moving it to {broken}")
        try:
            db.close()
        except Exception:
            pass
        try:
            os.replace(cache_db_path, broken)
        except OSError:
            pass
        _open_db(cache_db_path)


def init_test_db():
    import tempfile

    cache_db_path = tempfile.mktemp(suffix=".db")
    test_db = SqliteDatabase(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    test_db.bind([_TranslationCache], bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables([_TranslationCache], safe=True)
    return test_db


def clean_test_db(test_db):
    test_db.drop_tables([_TranslationCache])
    test_db.close()
    db_path = test_db.database
    if os.path.exists(db_path):
        os.remove(test_db.database)
    wal_path = db_path + "-wal"
    if os.path.exists(wal_path):
        os.remove(wal_path)
    shm_path = db_path + "-shm"
    if os.path.exists(shm_path):
        os.remove(shm_path)


init_db()