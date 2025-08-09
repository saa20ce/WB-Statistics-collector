import os
import sys
import time
import json
import math
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://statistics-api.wildberries.ru"
SECTIONS_ALL = ["sales", "returns", "commissions"]


COMMISSIONS_WINDOW_DAYS = 30


DEFAULT_DEDUP_KEYS = {
    "sales": ["saleID"],        
    "returns": ["returnID"],    
    "commissions": ["srid"],    
}

def mk_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("wb")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger

def session_with_retries(api_key: str, total: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": api_key})
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def sleep_respecting_retry_after(resp: requests.Response, fallback: float = 1.0, logger: Optional[logging.Logger] = None):
    ra = resp.headers.get("Retry-After")
    try:
        delay = float(ra) if ra is not None else fallback
    except ValueError:
        delay = fallback
    if logger:
        logger.warning(f"Rate limit/temporary error, sleeping {delay:.1f}s (Retry-After={ra})")
    time.sleep(delay)

def safe_request_json(
    s: requests.Session, method: str, url: str, *, params=None, json_body=None, timeout=60, logger: Optional[logging.Logger]=None
):
    while True:
        resp = s.request(method, url, params=params, json=json_body, timeout=timeout)
        if resp.status_code == 429 or resp.status_code in (500, 502, 503, 504):
            sleep_respecting_retry_after(resp, fallback=1.0, logger=logger)
            continue
        if not resp.ok:
            text = resp.text[:1000]
            raise RuntimeError(f"{method} {url} failed {resp.status_code}: {text}")
        try:
            return resp.json()
        except ValueError:
            raise RuntimeError(f"{method} {url} returned non-JSON (status {resp.status_code})")

def date_chunks(start: datetime, end: datetime, days: int) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt

def ensure_outdir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def write_df(df: pd.DataFrame, path: Path, fmt: str):
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("format must be csv or parquet")

def load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def save_checkpoint(path: Path, data: Dict[str, Any]):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def collect_sales(s: requests.Session, since: datetime, logger: logging.Logger) -> pd.DataFrame:
    url = f"{BASE}/api/v1/supplier/sales"
    params = {"dateFrom": since.strftime("%Y-%m-%d"), "flag": 0}
    logger.info("Fetching sales …")
    data = safe_request_json(s, "GET", url, params=params, logger=logger)
    df = pd.DataFrame(data)
    return df

def collect_returns(s: requests.Session, since: datetime, logger: logging.Logger) -> pd.DataFrame:
    url = f"{BASE}/api/v1/supplier/returns"
    params = {"dateFrom": since.strftime("%Y-%m-%d")}
    logger.info("Fetching returns …")
    data = safe_request_json(s, "GET", url, params=params, logger=logger)
    df = pd.DataFrame(data)
    return df

def collect_commissions(s: requests.Session, start: datetime, end: datetime, logger: logging.Logger,
                        ckpt: Dict[str, Any], ckpt_key: str) -> pd.DataFrame:
    url = f"{BASE}/api/v1/supplier/reportDetailByPeriod"
    rows: List[Dict[str, Any]] = []
    logger.info(f"Fetching commissions in windows of {COMMISSIONS_WINDOW_DAYS} days …")

    for d1, d2 in date_chunks(start, end, COMMISSIONS_WINDOW_DAYS):
        key = f"{d1.date()}_{d2.date()}"
        if ckpt.get(ckpt_key, {}).get(key) == "done":
            logger.info(f"Skip window {key} (checkpoint).")
            continue

        payload = {"dateFrom": d1.strftime("%Y-%m-%d"), "dateTo": d2.strftime("%Y-%m-%d")}
        chunk = safe_request_json(s, "POST", url, json_body=payload, logger=logger)
        if isinstance(chunk, list):
            rows.extend(chunk)
        else:
            
            rows.extend(chunk.get("data", []))
        ckpt.setdefault(ckpt_key, {})[key] = "done"

    return pd.DataFrame(rows)

def drop_dupes(df: pd.DataFrame, keys: List[str], logger: logging.Logger) -> pd.DataFrame:
    if not len(df) or not keys or any(k not in df.columns for k in keys):
        return df
    before = len(df)
    df = df.drop_duplicates(subset=keys)
    after = len(df)
    removed = before - after
    if removed:
        logger.info(f"Deduplicated by {keys}: removed {removed} rows")
    return df

def parse_args():
    p = argparse.ArgumentParser(description="WB Statistics collector")
    p.add_argument("--api-key", default=os.getenv("WB_API_KEY"), help="API key (Statistics). Or set WB_API_KEY env.")
    p.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD (default: today-90d)")
    p.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--sections", nargs="+", default=SECTIONS_ALL, choices=SECTIONS_ALL, help="Which sections to collect")
    p.add_argument("--out", default="out", help="Output dir")
    p.add_argument("--format", default="csv", choices=["csv", "parquet"])
    p.add_argument("--log", default="INFO", help="Log level")
    p.add_argument("--sales-keys", nargs="*", default=DEFAULT_DEDUP_KEYS["sales"])
    p.add_argument("--returns-keys", nargs="*", default=DEFAULT_DEDUP_KEYS["returns"])
    p.add_argument("--commissions-keys", nargs="*", default=DEFAULT_DEDUP_KEYS["commissions"])
    return p.parse_args()

def main():
    args = parse_args()
    logger = mk_logger(args.log)

    if not args.api_key:
        logger.error("API key required. Pass --api-key or set WB_API_KEY env.")
        sys.exit(1)

    date_to = datetime.utcnow().date() if not args.date_to else datetime.strptime(args.date_to, "%Y-%m-%d").date()
    date_from = (date_to - timedelta(days=90)) if not args.date_from else datetime.strptime(args.date_from, "%Y-%m-%d").date()

    start_dt = datetime.combine(date_from, datetime.min.time())
    end_dt = datetime.combine(date_to, datetime.min.time())

    out_dir = ensure_outdir(Path(args.out) / f"{date_from}_{date_to}")
    ckpt_path = out_dir / "_checkpoint.json"
    ckpt = load_checkpoint(ckpt_path)

    s = session_with_retries(args.api_key)

    try:
        if "sales" in args.sections:
            df = collect_sales(s, start_dt, logger)
            df = drop_dupes(df, args.sales_keys, logger)
            write_df(df, out_dir / f"sales.{args.format}", args.format)
            logger.info(f"Saved sales → {out_dir / f'sales.{args.format}'}")

        if "returns" in args.sections:
            df = collect_returns(s, start_dt, logger)
            df = drop_dupes(df, args.returns_keys, logger)
            write_df(df, out_dir / f"returns.{args.format}", args.format)
            logger.info(f"Saved returns → {out_dir / f'returns.{args.format}'}")

        if "commissions" in args.sections:
            df = collect_commissions(s, start_dt, end_dt, logger, ckpt, "commissions")
            df = drop_dupes(df, args.commissions_keys, logger)
            write_df(df, out_dir / f"commissions.{args.format}", args.format)
            logger.info(f"Saved commissions → {out_dir / f'commissions.{args.format}'}")

        save_checkpoint(ckpt_path, ckpt)
        logger.info("Done.")

    except Exception as e:
        logger.exception(f"Failed: {e}")
        save_checkpoint(ckpt_path, ckpt)
        sys.exit(2)

if __name__ == "__main__":
    main()
"""
Запуск:
1) через переменную окружения
export WB_API_KEY="ваш_ключ"
python wb_collect_v2.py --sections sales returns commissions --format parquet --out data

# 2) с явными датами и CSV
python wb_collect_v2.py --api-key "ваш_ключ" --from 2025-05-10 --to 2025-08-08 --format csv

"""

