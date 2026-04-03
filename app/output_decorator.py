import json
from functools import wraps
from pathlib import Path
import os

_log_path_str = os.getenv("LOG_FILE_PATH", "./data/logs.jsonl")
LOG_FILE = Path(_log_path_str)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_json(endpoint_name: str):
    """Sync decorator to log endpoint input/output to a jsonl file"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                output_data = result.dict() if hasattr(result, "dict") else result
            except Exception as e:
                output_data = {"error": str(e)}
                raise
            finally:
                log_entry = {
                    "endpoint": endpoint_name,
                    "output": output_data,
                }
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            return result
        return wrapper
    return decorator