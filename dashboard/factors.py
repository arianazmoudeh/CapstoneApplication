import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .models import PriceSnapshot

CACHE_KEY = "french-em-factors-v1"
SOURCE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


class FactorDataError(ValueError):
    pass


def read_monthly(content, columns):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        lines = archive.read(member).decode("utf-8-sig").splitlines()
    first = next(i for i, line in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", line))
    last = first
    while last < len(lines) and re.match(r"^\s*\d{6}\s*,", lines[last]):
        last += 1
    frame = pd.read_csv(io.StringIO("\n".join(lines[first - 1:last])), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame = frame[columns].apply(pd.to_numeric, errors="raise")
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m")
    if not frame.index.is_unique:
        raise FactorDataError("The factor data contain duplicate months.")
    frame = frame.replace([-99.99, -999], np.nan)
    return frame.where(np.isfinite(frame)).sort_index()


def download_factors():
    def load(filename, columns):
        with urlopen(SOURCE + filename, timeout=10) as response:
            return read_monthly(response.read(), columns)

    with ThreadPoolExecutor(max_workers=2) as pool:
        factors = pool.submit(load, "Emerging_5_Factors_CSV.zip", ["HML", "RMW"])
        momentum = pool.submit(load, "Emerging_MOM_Factor_CSV.zip", ["WML"])
        data = factors.result().join(momentum.result(), how="inner").dropna()
    current_month = pd.Timestamp(datetime.now(timezone.utc).date()).replace(day=1)
    data = data.loc[data.index < current_month]
    if data.empty:
        raise FactorDataError("No complete monthly factor data were found.")
    return {
        "rows": [
            {"month": month.strftime("%Y-%m-%d"), "value": float(row.HML),
             "profitability": float(row.RMW), "momentum": float(row.WML)}
            for month, row in data.tail(6).iloc[::-1].iterrows()
        ],
    }


def get_factor_data(force=False):
    now = datetime.now(timezone.utc)
    saved = PriceSnapshot.objects.filter(key=CACHE_KEY).first()
    if saved and ((saved.refresh_after > now and not force) or now - saved.fetched_at < timedelta(minutes=1)):
        data = saved.data
        stale = now - saved.fetched_at > timedelta(days=1)
    else:
        try:
            data = download_factors()
        except Exception:
            if not saved:
                raise FactorDataError("Kenneth French data are unavailable right now. Please try again later.") from None
            saved.refresh_after = now + timedelta(minutes=5)
            saved.save(update_fields=["refresh_after"])
            data, stale = saved.data, True
        else:
            PriceSnapshot.objects.update_or_create(key=CACHE_KEY, defaults={
                "data": data, "fetched_at": now,
                "refresh_after": now + timedelta(days=1),
            })
            stale = False
    return {"rows": [{**row, "month": datetime.strptime(row["month"], "%Y-%m-%d")} for row in data["rows"]]}, stale
