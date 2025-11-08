#!/usr/bin/env python3
"""
Incrementally update Qlib CN day dataset using AkShare.
- Detect last trading date in ~/.qlib/qlib_data/cn_data/calendars/day.txt
- Fetch daily OHLCV(amount) for a target universe (default: CSI300) from last+1 to today
- Append to existing Qlib binary store via scripts/dump_bin.py dump_update

Usage:
  python scripts/qlib_update_cn_from_akshare.py --universe csi300 --start auto --end today \
    --provider_dir ~/.qlib/qlib_data/cn_data --workers 8
"""
import argparse
import os
from pathlib import Path
import sys
import datetime as dt
import pandas as pd

# Ensure local qlib repo is importable for dump_bin
QLIB_REPO = Path(r"G:\test\qlib")
if QLIB_REPO.exists():
    sys.path.insert(0, str(QLIB_REPO / 'scripts'))


def read_last_date(provider_dir: Path) -> pd.Timestamp:
    cal = provider_dir / 'calendars' / 'day.txt'
    if not cal.exists():
        return pd.Timestamp('2000-01-01')
    try:
        s = cal.read_text(encoding='utf-8').strip().splitlines()
        return pd.to_datetime(s[-1]) if s else pd.Timestamp('2000-01-01')
    except Exception:
        return pd.Timestamp('2000-01-01')


def fetch_universe(universe: str) -> list[str]:
    # Prefer Qlib instruments if available
    try:
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=os.path.expanduser('~/.qlib/qlib_data/cn_data'), region='cn')
        if universe.lower() == 'csi300':
            inst = D.instruments('csi300')
            codes = D.list_instruments(instruments=inst, as_list=True)
        else:
            inst = D.instruments('all')
            codes = D.list_instruments(instruments=inst, as_list=True)
        # Qlib codes like SH600000 -> to AkShare 600000
        def to_ak(code: str) -> str:
            s = str(code).upper()
            if s.startswith('SH') or s.startswith('SZ'):
                return s[2:]
            return s
        return [to_ak(c) for c in codes]
    except Exception:
        # Fallback to AkShare full A-share list
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        return df['代码'].astype(str).tolist()


def fetch_ohlcv(codes: list[str], start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    rows = []
    start_ak = start.replace('-', '')
    end_ak = end.replace('-', '')
    for i, code in enumerate(codes):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start_ak, end_date=end_ak, adjust='qfq')
            if isinstance(df, pd.DataFrame) and not df.empty and '日期' in df.columns:
                df2 = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','成交额':'amount'})
                df2['symbol'] = code
                df2['date'] = pd.to_datetime(df2['date'])
                rows.append(df2[['date','symbol','open','high','low','close','volume','amount']])
        except Exception:
            continue
        if (i+1) % 100 == 0:
            print(f"Fetched {i+1}/{len(codes)}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Drop duplicates (date,symbol)
    out = out.drop_duplicates(['date','symbol']).sort_values(['symbol','date'])
    return out


def dump_update(csv_path: Path, provider_dir: Path, workers: int = 8):
    # Use qlib's dump_bin.py dump_update
    from dump_bin import DumpDataUpdate  # type: ignore
    d = DumpDataUpdate(
        data_path=str(csv_path),
        qlib_dir=str(provider_dir),
        freq='day',
        max_workers=workers,
        date_field_name='date',
        file_suffix='.csv',
        symbol_field_name='symbol',
        include_fields='open,high,low,close,volume,amount',
    )
    d.dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', default='csi300', choices=['csi300','all'])
    ap.add_argument('--provider_dir', default='~/.qlib/qlib_data/cn_data')
    ap.add_argument('--start', default='auto')
    ap.add_argument('--end', default='today')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    provider_dir = Path(os.path.expanduser(args.provider_dir))
    provider_dir.mkdir(parents=True, exist_ok=True)

    last = read_last_date(provider_dir)
    start = (last + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if args.start == 'auto' else args.start
    end = dt.date.today().strftime('%Y-%m-%d') if args.end == 'today' else args.end

    print(f"Updating Qlib CN/day from {start} to {end} for universe={args.universe}")

    codes = fetch_universe(args.universe)
    if not codes:
        print('No codes fetched; abort')
        return 1

    df = fetch_ohlcv(codes, start, end)
    if df.empty:
        print('No incremental OHLCV to update; calendar may still be old.')
        return 0

    tmp_csv = provider_dir / f"ak_update_{start}_to_{end}.csv"
    df.to_csv(tmp_csv, index=False)
    print(f"Saved incremental CSV: {tmp_csv}  rows={len(df)}")

    dump_update(tmp_csv, provider_dir, args.workers)
    print('✅ Qlib dump update done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
