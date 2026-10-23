"""CRSP data via WRDS.

Your WRDS subscription gives the daily stock file and the security history, but not the
S&P 500 constituents table (permission denied). So index membership still comes from the
GitHub membership file, and each (ticker, date) is mapped to CRSP's permanent ID, PERMNO,
using the ticker history in crsp.stksecurityinfohist. PERMNOs are never reused, which
removes the ticker-reuse problem for good.
"""
import pandas as pd

SEC_COLS = ["permno", "secinfostartdt", "secinfoenddt", "ticker", "shareclass",
            "securitytype", "sharetype", "issuernm"]
CRSP_END = pd.Timestamp("2025-12-31")          # last date in your CRSP daily file
OPEN_END = pd.Timestamp("2099-12-31")          # end date for spells still open today

# The membership file labels some companies with tickers CRSP never saw on an exchange:
# post-bankruptcy OTC tickers ending in Q, and later names of companies that were renamed
# or merged after leaving the index. Map them to the ticker the company traded under
# while it was in the S&P 500. Add rows here when the report shows a stay NOT MATCHED.
TICKER_ALIASES = {
    # bankruptcies: OTC ticker after failing -> exchange ticker while in the index
    "LEHMQ": "LEH",    # Lehman Brothers
    "WAMUQ": "WM",     # Washington Mutual
    "MTLQQ": "GM",     # old General Motors (Motors Liquidation)
    "EKDKQ": "EK",     # Eastman Kodak
    "ENRNQ": "ENE",    # Enron
    "WCOEQ": "WCOM",   # WorldCom
    "AAMRQ": "AMR",    # AMR (American Airlines)
    "UAWGQ": "UAL",    # UAL (United Airlines)
    "DALRQ": "DAL",    # Delta Air Lines (2005)
    "ABKFQ": "ABK",    # Ambac Financial
    "NRTLQ": "NT",     # Nortel Networks
    "DPHIQ": "DPH",    # Delphi
    "VSTNQ": "VC",     # Visteon
    "CPNLQ": "CPN",    # Calpine
    "BHMSQ": "BS",     # Bethlehem Steel
    "OWENQ": "OWC",    # Owens Corning
    "CITGQ": "CIT",    # CIT Group
    "GAPTQ": "GAP",    # Great Atlantic & Pacific Tea
    "FLTWQ": "FLE",    # Fleetwood Enterprises
    "FLMIQ": "FLM",    # Fleming Companies
    "ANRZQ": "ANR",    # Alpha Natural Resources
    "RSHCQ": "RSH",    # RadioShack
    "CCTYQ": "CC",     # Circuit City
    "DCNAQ": "DCN",    # Dana Corp
    "WNDXQ": "WIN",    # Winn-Dixie
    "BTUUQ": "BTU",    # Peabody Energy
    "SUNEQ": "WFR",    # MEMC Electronic Materials (later SunEdison)
    # other later names / formats
    "FNMA": "FNM",     # Fannie Mae on the NYSE until 2010
    "FMCC": "FRE",     # Freddie Mac on the NYSE until 2010
    "AABA": "YHOO",    # Yahoo, renamed Altaba
    "DXC": "CSC",      # Computer Sciences Corp (the 1996-2015 stay)
    "VIAV": "JDSU",    # JDS Uniphase
    "KATE": "LIZ",     # Liz Claiborne
    "KDP": "DPS",      # Dr Pepper Snapple (the 2008-2018 stay)
    "RDS.A": "RD",     # Royal Dutch Petroleum
    "INCLF": "N",      # Inco
    "AZA.A": "AZA",    # Alza
    "EHC": "HRC",      # HealthSouth
    "ACKH": "ACK",     # Armstrong World Industries
    "SGID": "SGI",     # Silicon Graphics
    "FWLT": "FWC",     # Foster Wheeler
    "FL": "Z",         # Woolworth / Venator (the 1996-1999 stay)
    "ATGE": "DV",      # DeVry
    "IAC": "IACI",     # IAC/InterActiveCorp (the 2006-2008 stay)
    "YRCW": "YELL",    # Yellow Corp
}
# Partial matches that turned out to be a different, small company using the ticker
# during part of the stay. Their stays are dropped (counted as unmatched).
BAD_MATCHES = {"MWI", "SFS"}

# Aliases are a FALLBACK: a stay is first matched on its own ticker, and the alias is
# tried only if that finds nothing. So 'KDP' in 2022 (Keurig Dr Pepper, its own ticker)
# is unaffected by the 'KDP' -> 'DPS' alias used for the 2008-2018 stay.


def connect(username: str):
    """Connect to WRDS. Uses the saved .pgpass login; approve Duo if asked."""
    import wrds
    return wrds.Connection(wrds_username=username)


def load_security_history(db) -> pd.DataFrame:
    """Every ticker each PERMNO has had, with the dates it applied (about 190k rows)."""
    sql = f"select {', '.join(SEC_COLS)} from crsp.stksecurityinfohist"
    df = db.raw_sql(sql, date_cols=["secinfostartdt", "secinfoenddt"])
    df["ticker"] = df["ticker"].str.strip()
    df["shareclass"] = df["shareclass"].fillna("").str.strip()
    return df


def split_ticker(t: str) -> tuple:
    """'BRK.B' -> ('BRK', 'B'). Tickers without a dot have no class: 'AAPL' -> ('AAPL', '')."""
    if "." in t:
        base, cls = t.split(".", 1)
        return base, cls
    return t, ""


def membership_spells(membership: pd.DataFrame) -> pd.DataFrame:
    """Turn the change-date membership table into one row per continuous stay in the index:
    ticker, start (first day in), end (last day in). Stays still open get end = 2099-12-31."""
    spells, active = [], {}
    for d, row in membership.iterrows():
        cur = row["tickers"]
        for t in cur - active.keys():
            active[t] = d
        for t in [t for t in active if t not in cur]:
            spells.append((t, active.pop(t), d - pd.Timedelta(days=1)))
    spells += [(t, s, OPEN_END) for t, s in active.items()]
    return pd.DataFrame(spells, columns=["ticker", "start", "end"]).sort_values(["ticker", "start"], ignore_index=True)


def map_spells_to_permno(spells: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Match each index stay to the PERMNO(s) that used that ticker during it.
    Returns intervals: ticker, spell start/end, permno, from, to, issuer name.
    A ticker with a dot ('BRK.B') must also match the share class; one without a dot
    matches any class."""
    sp = spells.copy()
    sp[["base", "cls"]] = pd.DataFrame(sp["ticker"].map(split_ticker).tolist(), index=sp.index)
    m = sp.merge(sec.rename(columns={"ticker": "base"}), on="base", how="inner")
    m = m[(m["cls"] == "") | (m["shareclass"] == m["cls"])]
    m["from"] = m[["start", "secinfostartdt"]].max(axis=1)
    m["to"] = m[["end", "secinfoenddt"]].min(axis=1)
    m = m[m["from"] <= m["to"]]
    iv = (m.groupby(["ticker", "start", "end", "permno"], as_index=False)
            .agg(**{"from": ("from", "min"), "to": ("to", "max"), "issuer": ("issuernm", "last")}))
    return iv.sort_values(["ticker", "from"], ignore_index=True)


def spell_coverage(spells: pd.DataFrame, intervals: pd.DataFrame, data_end=CRSP_END) -> pd.DataFrame:
    """For each index stay: how many calendar days fall inside the CRSP data period, how many
    of those are matched to a PERMNO, and how many different PERMNOs matched."""
    sp = spells.copy()
    sp["end_c"] = sp["end"].clip(upper=data_end)
    sp = sp[sp["start"] <= sp["end_c"]]
    sp["days"] = (sp["end_c"] - sp["start"]).dt.days + 1
    iv = intervals.copy()
    iv["to_c"] = iv["to"].clip(upper=data_end)
    iv["days"] = ((iv["to_c"] - iv["from"]).dt.days + 1).clip(lower=0)
    agg = iv.groupby(["ticker", "start"]).agg(matched_days=("days", "sum"), n_permnos=("permno", "nunique"),
                                             permnos=("permno", lambda s: ",".join(map(str, sorted(set(s))))),
                                             issuer=("issuer", "last"))
    out = sp.merge(agg, on=["ticker", "start"], how="left")
    out["matched_days"] = out["matched_days"].fillna(0)
    out["n_permnos"] = out["n_permnos"].fillna(0).astype(int)
    out["coverage"] = (out["matched_days"] / out["days"]).clip(upper=1)
    return out.drop(columns="end_c")



def map_spells_best(spells: pd.DataFrame, sec: pd.DataFrame, aliases: dict = None, data_end=CRSP_END,
                    window_days: int = 60) -> pd.DataFrame:
    """Recommended matcher. One PERMNO per index stay.

    The membership file often labels a company with a LATER ticker for its whole stay
    (NextEra shows as 'NEE' in 1996, when it traded as 'FPL'), and tickers get reused
    ('C' was Chrysler before it was Citigroup). So for each stay:
      1. candidates = every PERMNO that held the ticker at any time during the stay
         (+/- window_days, because removals and ticker changes are a few days apart)
      2. pick the candidate whose whole CRSP life overlaps the stay the most; on a tie,
         the one holding the ticker closest to the end of the stay
      3. assign that PERMNO to the stay, clipped to the PERMNO's life
    A PERMNO survives renames, so FPL -> NEE is one company throughout."""
    sp = spells.copy()
    sp["end_c"] = sp["end"].clip(upper=data_end)
    sp = sp[sp["start"] <= sp["end_c"]]
    aliases = aliases or {}
    sp["lookup"] = sp["ticker"].map(lambda t: aliases.get(t, t))
    sp[["base", "cls"]] = pd.DataFrame(sp["lookup"].map(split_ticker).tolist(), index=sp.index)
    m = sp.merge(sec.rename(columns={"ticker": "base"}), on="base", how="inner")
    m = m[(m["cls"] == "") | (m["shareclass"] == m["cls"])]
    w = pd.Timedelta(days=window_days)
    m = m[(m["secinfostartdt"] <= m["end_c"] + w) & (m["secinfoenddt"] >= m["start"] - w)].copy()

    life = sec.groupby("permno").agg(life_start=("secinfostartdt", "min"), life_end=("secinfoenddt", "max"))
    m = m.join(life, on="permno")
    m["from"] = m[["start", "life_start"]].max(axis=1)
    m["to"] = m[["end", "life_end"]].min(axis=1)
    m["overlap"] = ((m[["end_c", "life_end"]].min(axis=1) - m["from"]).dt.days + 1).clip(lower=0)
    before = (m["secinfostartdt"] - m["end_c"]).dt.days.clip(lower=0)
    after = (m["end_c"] - m["secinfoenddt"]).dt.days.clip(lower=0)
    m["dist_to_end"] = before + after

    m = m.sort_values(["overlap", "dist_to_end"], ascending=[False, True])
    n_cand = m.groupby(["ticker", "start"])["permno"].transform("nunique")
    best = m.groupby(["ticker", "start"]).head(1).copy()
    best["n_candidates"] = n_cand.loc[best.index]
    best = best[best["overlap"] > 0]
    out = best[["ticker", "start", "end", "permno", "from", "to", "issuernm", "n_candidates", "lookup"]]
    return out.rename(columns={"issuernm": "issuer"}).sort_values(["ticker", "start"], ignore_index=True)


def fill_gaps(spells: pd.DataFrame, intervals: pd.DataFrame, sec: pd.DataFrame,
              data_end=CRSP_END, min_gap_days: int = 7) -> pd.DataFrame:
    """Second pass. Some companies kept their index seat through a merger or spin-off that
    gave them a NEW PERMNO (Comcast in 2002, News Corp / Fox in 2013). map_spells_best picks
    the later PERMNO, which leaves the early part of the stay uncovered. For each uncovered
    stretch, look up which PERMNO held the ticker on those exact dates and add it.
    Only gaps are filled, so a company that the best PERMNO already covers (Citigroup) can
    never be replaced by an older user of the ticker (Chrysler)."""
    gaps = []
    covered = intervals.groupby(["ticker", "start"]).agg(f=("from", "min"), t=("to", "max"), lookup=("lookup", "first"))
    for _, sp in spells.iterrows():
        end_c = min(sp["end"], data_end)
        if sp["start"] > end_c:
            continue
        key = (sp["ticker"], sp["start"])
        if key in covered.index:
            f, t, look = covered.loc[key, "f"], covered.loc[key, "t"], covered.loc[key, "lookup"]
            segs = [(sp["start"], f - pd.Timedelta(days=1)), (t + pd.Timedelta(days=1), end_c)]
        else:
            look = sp["ticker"]
            segs = [(sp["start"], end_c)]
        for a, b in segs:
            if (b - a).days + 1 >= min_gap_days:
                gaps.append({"ticker": look, "start": a, "end": b,
                             "orig_ticker": sp["ticker"], "orig_start": sp["start"], "orig_end": sp["end"]})
    if not gaps:
        return intervals
    g = pd.DataFrame(gaps)
    extra = map_spells_to_permno(g[["ticker", "start", "end"]], sec)
    extra = extra.merge(g, on=["ticker", "start", "end"])
    extra = extra.assign(lookup=extra["ticker"], ticker=extra["orig_ticker"], start=extra["orig_start"],
                         end=extra["orig_end"], n_candidates=0)
    extra = extra[["ticker", "start", "end", "permno", "from", "to", "issuer", "n_candidates", "lookup"]]
    return pd.concat([intervals, extra], ignore_index=True).sort_values(["ticker", "start", "from"], ignore_index=True)


def map_all(spells: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Full mapping: own ticker first, aliases only for stays still unmatched, then gap filling.
    The 'lookup' column records which ticker was actually used for each interval."""
    iv = map_spells_best(spells, sec)
    matched = set(zip(iv["ticker"], iv["start"]))
    rest = spells[[(t, s) not in matched and t in TICKER_ALIASES for t, s in zip(spells["ticker"], spells["start"])]]
    if len(rest):
        iv = pd.concat([iv, map_spells_best(rest, sec, TICKER_ALIASES)], ignore_index=True)
    iv = fill_gaps(spells, iv, sec)
    return iv[~iv["ticker"].isin(BAD_MATCHES)].reset_index(drop=True)



# ====================================================================== step B: daily data
DAILY_COLS = ["permno", "dlycaldt", "dlyret", "dlyprc", "dlyvol", "dlybid", "dlyask", "dlycap", "dlydelflg"]


def download_daily(db, permnos, start: str, end: str, out_dir, batch: int = 100) -> None:
    """Download the daily file for the given PERMNOs, in batches, one parquet per batch.
    Batches already on disk are skipped, so a crash or Duo timeout can simply be re-run."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    permnos = sorted(int(p) for p in permnos)
    for i in range(0, len(permnos), batch):
        path = out / f"daily_{i // batch:03d}.parquet"
        if path.exists():
            continue
        chunk = permnos[i:i + batch]
        sql = (f"select {', '.join(DAILY_COLS)} from crsp.stkdlysecuritydata "
               f"where permno in ({', '.join(map(str, chunk))}) "
               f"and dlycaldt between '{start}' and '{end}'")
        df = db.raw_sql(sql, date_cols=["dlycaldt"])
        df.to_parquet(path)
        print(f"  batch {i // batch + 1}/{-(-len(permnos) // batch)}: {len(df):,} rows")


def load_daily(out_dir) -> pd.DataFrame:
    from pathlib import Path
    files = sorted(Path(out_dir).glob("daily_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["permno"] = df["permno"].astype(int).astype(str)          # column names must be strings
    for c in ["dlyret", "dlyprc", "dlyvol", "dlybid", "dlyask", "dlycap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def grid(daily: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long daily table -> date x permno grid for one column."""
    return daily.pivot(index="dlycaldt", columns="permno", values=col).sort_index().rename_axis(index="date", columns=None)


def total_return_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound CRSP total returns into a price-like series (what 'adjusted close' means).
    Days with no return stay NaN; compounding continues after them."""
    return (1 + returns).cumprod(skipna=True).where(returns.notna())


def intervals_mask(intervals: pd.DataFrame, dates: pd.DatetimeIndex, permnos) -> pd.DataFrame:
    """Boolean date x permno grid: True when that PERMNO was an S&P 500 member that day."""
    mask = pd.DataFrame(False, index=dates, columns=[str(p) for p in permnos])
    for _, r in intervals.iterrows():
        col = str(int(r["permno"]))
        if col in mask.columns:
            mask.loc[(dates >= r["from"]) & (dates <= r["to"]), col] = True
    return mask