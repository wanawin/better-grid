import re
import itertools
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants
# -----------------------------
DIGITS = [str(i) for i in range(10)]
DRAW_ORDER = {"Morning": 0, "Midday": 1, "Day": 1, "Evening": 2, "Night": 3}

_MIRROR = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}

# Matches LotteryPost-like tab-delimited TXT:
#   Fri, Jan 9, 2026\tGeorgia\tCash 3 Evening\t0-0-2
#   Sat, Jan 10, 2026\tTexas\tPick 3 Day\t3-6-2, Fireball: 7
_LOTTERYPOST_PAT = re.compile(
    r"^(?P<date>[A-Za-z]{3},\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4})\t"
    r"(?P<state>[^\t]+)\t"
    r"(?:(?:Pick|Cash)\s*3)\s+(?P<draw>Morning|Midday|Day|Evening|Night)\t"
    r"(?P<num>\d-\d-\d)",
    re.IGNORECASE,
)


def parse_history_text(raw: str) -> pd.DataFrame:
    """Parse tab-delimited Pick 3 / Cash 3 history.

    Returns columns:
      - date (date)
      - state (str)
      - draw (str)
      - num (str 'XYZ')

    Sorting is chronological per stream (date + draw_order).
    """
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LOTTERYPOST_PAT.search(line)
        if not m:
            continue
        date_str = m.group("date")
        state = m.group("state").strip()
        draw = m.group("draw").title()
        num = m.group("num").replace("-", "")
        try:
            d = dt.datetime.strptime(date_str, "%a, %b %d, %Y").date()
        except Exception:
            continue
        if len(num) == 3 and num.isdigit():
            rows.append((d, state, draw, num))

    df = pd.DataFrame(rows, columns=["date", "state", "draw", "num"])
    if df.empty:
        return df

    df["draw_order"] = df["draw"].map(DRAW_ORDER).fillna(99).astype(int)
    df = df.sort_values(["state", "date", "draw_order"]).reset_index(drop=True)
    return df


# -----------------------------
# Small helpers
# -----------------------------

def box_key(num: str) -> str:
    return "".join(sorted(list(num), key=int))


def parity_sig(num: str) -> str:
    ev = sum((int(d) % 2 == 0) for d in num)
    if ev == 3:
        return "EEE"
    if ev == 2:
        return "EEO"
    if ev == 1:
        return "EOO"
    return "OOO"


def _mod10(x: int) -> int:
    return x % 10


def compute_pos_followers(stream_nums: list[str], upto_exclusive: int, lookback: int | None) -> list[dict[str, dict[str, int]]]:
    """counts[pos][prev_digit][next_digit] = count

    Uses transitions fully contained in [start, upto_exclusive).
    """
    counts = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    if upto_exclusive <= 1:
        return counts

    end = int(upto_exclusive)
    start = 1
    if lookback is not None:
        start = max(1, end - int(lookback))

    for i in range(start, end):
        prev = str(stream_nums[i - 1]).zfill(3)
        nxt = str(stream_nums[i]).zfill(3)
        for pos in range(3):
            a = prev[pos]
            b = nxt[pos]
            counts[pos][a][b] += 1

    return counts


def cond_prob(counts_pos: dict[str, dict[str, int]], prev_digit: str, next_digit: str, alpha: float) -> float:
    """Laplace-smoothed P(next_digit | prev_digit) for a single position."""
    dct = counts_pos.get(prev_digit)
    if not dct:
        return 1.0 / 10.0
    total = sum(dct.values())
    if total <= 0:
        return 1.0 / 10.0
    return (dct.get(next_digit, 0) + alpha) / (total + 10.0 * alpha)


def top_followers(counts_pos: dict[str, dict[str, int]], prev_digit: str, k: int) -> list[str]:
    dct = counts_pos.get(prev_digit, {})
    items = sorted(dct.items(), key=lambda kv: (-kv[1], int(kv[0])))
    return [d for d, _ in items[: int(k)]]


# -----------------------------
# Grid builders
# -----------------------------

def compute_drought(stream_nums: list[str], upto_exclusive: int, window: int | None = None) -> list[dict[str, int]]:
    """drought[pos][digit] = draws since last seen for that digit at that position."""
    if window is None:
        start = 0
    else:
        start = max(0, upto_exclusive - int(window))

    last_seen = [{d: None for d in DIGITS} for _ in range(3)]
    for i in range(start, upto_exclusive):
        rel = i - start
        s = str(stream_nums[i]).zfill(3)
        for pos in range(3):
            last_seen[pos][s[pos]] = rel

    drought = []
    for pos in range(3):
        dct = {}
        for d in DIGITS:
            if last_seen[pos][d] is None:
                dct[d] = (upto_exclusive - start)
            else:
                dct[d] = (upto_exclusive - start - 1) - last_seen[pos][d]
        drought.append(dct)
    return drought


def build_due_grid(stream_nums: list[str], upto_exclusive: int, rows: int = 4, window: int | None = None):
    drought = compute_drought(stream_nums, upto_exclusive, window=window)
    cols: list[list[str]] = []
    for pos in range(3):
        items = sorted(drought[pos].items(), key=lambda kv: (-kv[1], int(kv[0])))
        top = [d for d, _ in items[: int(rows)]]
        cols.append(top)

    # grid is rows x 3 (Hundreds, Tens, Ones)
    grid = [[cols[c][r] for c in range(3)] for r in range(rows)]
    return grid, drought


def build_seed_grid(
    stream_nums: list[str],
    upto_exclusive: int,
    rows: int = 4,
    follower_lookback: int | None = 200,
    follower_k: int = 2,
    w_carry: float = 6.0,
    w_neighbor: float = 3.0,
    w_mirror: float = 2.0,
):
    """Seed-conditioned 3x4 grid.

    For each position, choose 4 digits from the pool:
      - seed digit at that position
      - +/- 1 neighbors (wrap)
      - mirror(seed)
      - top follower_k followers after that seed digit (same stream, same position)

    Rank the 4 digits by: follower_count + weights above.
    """
    # fallback early
    if upto_exclusive < 1:
        return build_due_grid(stream_nums, upto_exclusive, rows=rows, window=None)

    drought = compute_drought(stream_nums, upto_exclusive, window=None)

    # seed is last known draw
    seed = str(stream_nums[upto_exclusive - 1]).zfill(3)
    followers = compute_pos_followers(stream_nums, upto_exclusive - 1, lookback=follower_lookback)

    cols: list[list[str]] = []
    for pos in range(3):
        s = int(seed[pos])
        n_plus = _mod10(s + 1)
        n_minus = _mod10(s - 1)
        m = _MIRROR.get(s, s)

        fdict = followers[pos].get(seed[pos], {})
        tf = sorted(fdict.items(), key=lambda kv: (-kv[1], int(kv[0])))[: int(follower_k)]
        tf_digits = [int(d) for d, _ in tf]

        cand = set([s, n_plus, n_minus, m] + tf_digits)

        scored = []
        for d in cand:
            score = 0.0
            if d == s:
                score += w_carry
            if d in (n_plus, n_minus):
                score += w_neighbor
            if d == m:
                score += w_mirror
            score += float(fdict.get(str(d), 0))
            scored.append((d, score))

        scored_sorted = sorted(scored, key=lambda t: (-t[1], t[0]))[: int(rows)]
        col = [str(d) for d, _ in scored_sorted]

        # pad using overdue digits if needed
        if len(col) < rows:
            due_items = sorted(drought[pos].items(), key=lambda kv: (-kv[1], int(kv[0])))
            for dd, _ in due_items:
                if dd not in col:
                    col.append(dd)
                if len(col) >= rows:
                    break

        cols.append(col)

    grid = [[cols[c][r] for c in range(3)] for r in range(rows)]
    return grid, drought


def grid_to_cols(grid: list[list[str]]) -> tuple[list[str], list[str], list[str]]:
    rows = len(grid)
    colH = [grid[r][0] for r in range(rows)]
    colT = [grid[r][1] for r in range(rows)]
    colO = [grid[r][2] for r in range(rows)]
    return colH, colT, colO


def slot_labels(rows: int = 4) -> list[str]:
    return [chr(ord("A") + i) for i in range(rows)]


def grid_slot_hit_stats(stream_nums: list[str], grid_builder, rows: int, follower_lookback: int | None, last_n: int = 400):
    """Recompute slot hit counts for the current grid formula.

    Uses walk-forward: for each i-1 -> i, build grid from history up to i.
    Returns:
      - slot_counts dict for 12 slots
      - in_pos_rate per position (H/T/O)
      - any2_digits_anywhere_rate (counts duplicates)
      - any2_unique_anywhere_rate
    """
    n_total = 0
    letters = slot_labels(rows)
    slot_counts = {f"{pos}{L}": 0 for pos in (1, 2, 3) for L in letters}

    in_pos = [0, 0, 0]
    any2_anywhere = 0
    any2_unique = 0

    end = len(stream_nums)
    start = max(1, end - int(last_n))

    for i in range(start, end):
        # grid built from history up to i (seed is i-1)
        if grid_builder == "seed":
            g, _ = build_seed_grid(stream_nums, i, rows=rows, follower_lookback=follower_lookback)
        else:
            g, _ = build_due_grid(stream_nums, i, rows=rows, window=None)

        colH, colT, colO = grid_to_cols(g)
        winner = str(stream_nums[i]).zfill(3)

        # anywhere in grid
        gset = set(colH + colT + colO)
        if sum(1 for d in winner if d in gset) >= 2:
            any2_anywhere += 1
        if len(set(winner) & gset) >= 2:
            any2_unique += 1

        # per-position
        wH, wT, wO = winner[0], winner[1], winner[2]
        if wH in colH:
            in_pos[0] += 1
            slot_counts[f"1{letters[colH.index(wH)]}"] += 1
        if wT in colT:
            in_pos[1] += 1
            slot_counts[f"2{letters[colT.index(wT)]}"] += 1
        if wO in colO:
            in_pos[2] += 1
            slot_counts[f"3{letters[colO.index(wO)]}"] += 1

        n_total += 1

    rates = {
        "H": (in_pos[0] / n_total) if n_total else 0.0,
        "T": (in_pos[1] / n_total) if n_total else 0.0,
        "O": (in_pos[2] / n_total) if n_total else 0.0,
        "Any2Anywhere": (any2_anywhere / n_total) if n_total else 0.0,
        "Any2UniqueAnywhere": (any2_unique / n_total) if n_total else 0.0,
        "N": n_total,
    }
    return slot_counts, rates


# -----------------------------
# Candidate generation (STRAIGHTS-first)
# -----------------------------

def top4_mass_for_position(counts, seed_digit: str, pos: int, col_digits: list[str], alpha: float) -> float:
    return float(sum(cond_prob(counts[pos], seed_digit, d, alpha) for d in col_digits))


def choose_rescue_digits(counts, seed_digit: str, pos: int, already: set[str], k: int) -> list[str]:
    dct = counts[pos].get(seed_digit, {})
    items = sorted(dct.items(), key=lambda kv: (-kv[1], int(kv[0])))
    out = []
    for d, _ in items:
        if d in already:
            continue
        out.append(d)
        if len(out) >= int(k):
            break
    # if no follower info, fallback to all digits not already
    if len(out) < int(k):
        for d in DIGITS:
            if d not in already and d not in out:
                out.append(d)
            if len(out) >= int(k):
                break
    return out


def generate_straights(colH: list[str], colT: list[str], colO: list[str]) -> list[str]:
    return [h + t + o for h in colH for t in colT for o in colO]


def score_straight(straight: str, seed: str, counts, alpha: float) -> float:
    s = str(seed).zfill(3)
    x = str(straight).zfill(3)
    # product of position-conditional probabilities
    p = 1.0
    for pos in range(3):
        p *= cond_prob(counts[pos], s[pos], x[pos], alpha)
    return float(p)


def build_candidates_for_seed(
    stream_nums: list[str],
    seed_index: int,
    grid_mode: str,
    follower_lookback: int | None,
    follower_k: int,
    alpha: float,
    rescue_enabled: bool,
    rescue_k: int,
    rescue_mass_threshold: float,
    rescue_scope: str,
):
    """Build candidates for the draw at seed_index (seed is draw seed_index-1).

    seed_index is the index of the next draw we are predicting (exclusive).

    Returns:
      grid, cols, candidates(list), counts(followers counts), rescue_info
    """
    rows = 4

    # follower counts for scoring and rescue
    counts = compute_pos_followers(stream_nums, seed_index, lookback=follower_lookback)

    # build grid using history up to seed_index
    if grid_mode == "seed":
        grid, _ = build_seed_grid(
            stream_nums,
            seed_index,
            rows=rows,
            follower_lookback=follower_lookback,
            follower_k=follower_k,
        )
    else:
        grid, _ = build_due_grid(stream_nums, seed_index, rows=rows, window=None)

    colH, colT, colO = grid_to_cols(grid)

    # base straights (64)
    base = generate_straights(colH, colT, colO)
    candidates = list(base)

    rescue_info = {
        "triggered": False,
        "position": None,
        "mass": None,
        "added": 0,
        "digits": [],
    }

    if not rescue_enabled or seed_index < 1:
        return grid, (colH, colT, colO), candidates, counts, rescue_info

    seed = str(stream_nums[seed_index - 1]).zfill(3)

    masses = {
        "H": top4_mass_for_position(counts, seed[0], 0, colH, alpha),
        "T": top4_mass_for_position(counts, seed[1], 1, colT, alpha),
        "O": top4_mass_for_position(counts, seed[2], 2, colO, alpha),
    }

    def add_rescue_for(pos_key: str):
        pos_map = {"H": 0, "T": 1, "O": 2}
        pos = pos_map[pos_key]
        already = set([colH, colT, colO][pos])
        rdigits = choose_rescue_digits(counts, seed[pos], pos, already, rescue_k)

        # add 16*K combos by swapping the target column
        added = 0
        if pos_key == "H":
            for h in rdigits:
                for t in colT:
                    for o in colO:
                        candidates.append(h + t + o)
                        added += 1
        elif pos_key == "T":
            for t in rdigits:
                for h in colH:
                    for o in colO:
                        candidates.append(h + t + o)
                        added += 1
        else:
            for o in rdigits:
                for h in colH:
                    for t in colT:
                        candidates.append(h + t + o)
                        added += 1
        return rdigits, added

    # decide rescue positions
    weak_positions = sorted(masses.items(), key=lambda kv: (kv[1], kv[0]))

    if rescue_scope == "Weakest position only":
        pos_key, mass = weak_positions[0]
        if mass <= rescue_mass_threshold:
            rdigits, added = add_rescue_for(pos_key)
            rescue_info.update({"triggered": True, "position": pos_key, "mass": float(mass), "added": int(added), "digits": rdigits})
    else:
        # All positions below threshold
        triggered_any = False
        total_added = 0
        all_rdigits = {}
        for pos_key, mass in weak_positions:
            if mass <= rescue_mass_threshold:
                rdigits, added = add_rescue_for(pos_key)
                triggered_any = True
                total_added += added
                all_rdigits[pos_key] = rdigits
        if triggered_any:
            rescue_info.update({"triggered": True, "position": "MULTI", "mass": float(min(masses.values())), "added": int(total_added), "digits": all_rdigits})

    # dedupe straights (keep order)
    seen = set()
    deduped = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    candidates = deduped

    return grid, (colH, colT, colO), candidates, counts, rescue_info


def rank_candidates(
    candidates: list[str],
    seed: str,
    counts,
    alpha: float,
    drop_all_even: bool,
    drop_all_odd: bool,
    keep_top_pct: float | None,
):
    rows = []
    for s in candidates:
        p = score_straight(s, seed, counts, alpha)
        rows.append({
            "straight": s,
            "box": box_key(s),
            "parity": parity_sig(s),
            "score": p,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # parity trims
    if drop_all_even:
        df = df[df["parity"] != "EEE"]
    if drop_all_odd:
        df = df[df["parity"] != "OOO"]

    df = df.sort_values(["score", "straight"], ascending=[False, True]).reset_index(drop=True)

    # percentile trim
    if keep_top_pct is not None:
        pct = float(keep_top_pct)
        pct = max(0.0, min(100.0, pct))
        k = max(1, int(np.ceil(len(df) * (pct / 100.0))))
        df = df.iloc[:k].reset_index(drop=True)

    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def aggregate_boxes(df_straights: pd.DataFrame) -> pd.DataFrame:
    if df_straights.empty:
        return df_straights

    grp = df_straights.groupby("box")
    rows = []
    for box, g in grp:
        best = g.sort_values(["score", "straight"], ascending=[False, True]).iloc[0]
        rows.append({
            "box": box,
            "best_straight": best["straight"],
            "best_score": float(best["score"]),
            "num_straights": int(len(g)),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["best_score", "box"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# -----------------------------
# Walk-forward backtest
# -----------------------------

def walk_forward_backtest(
    stream_nums: list[str],
    grid_mode: str,
    follower_lookback: int | None,
    follower_k: int,
    alpha: float,
    rescue_enabled: bool,
    rescue_k: int,
    rescue_mass_threshold: float,
    rescue_scope: str,
    drop_all_even: bool,
    drop_all_odd: bool,
    keep_top_pct: float | None,
    start_at: int = 30,
):
    hits_box = 0
    hits_straight = 0
    n = 0

    for i in range(max(2, int(start_at)), len(stream_nums)):
        grid, cols, cand, counts, _ = build_candidates_for_seed(
            stream_nums,
            seed_index=i,
            grid_mode=grid_mode,
            follower_lookback=follower_lookback,
            follower_k=follower_k,
            alpha=alpha,
            rescue_enabled=rescue_enabled,
            rescue_k=rescue_k,
            rescue_mass_threshold=rescue_mass_threshold,
            rescue_scope=rescue_scope,
        )
        seed = str(stream_nums[i - 1]).zfill(3)
        ranked = rank_candidates(cand, seed, counts, alpha, drop_all_even, drop_all_odd, keep_top_pct)
        if ranked.empty:
            continue
        winner = str(stream_nums[i]).zfill(3)

        if winner in set(ranked["straight"].tolist()):
            hits_straight += 1
        if box_key(winner) in set(ranked["box"].tolist()):
            hits_box += 1
        n += 1

    return {
        "N": n,
        "StraightHitRate": (hits_straight / n) if n else 0.0,
        "BoxHitRate": (hits_box / n) if n else 0.0,
    }


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Pick-3 3x4 Seed Grid -> Straights", layout="wide")

st.title("Pick-3: 3×4 Seed Grid → Straights-first list (same stream → same stream)")

st.caption(
    "This app keeps a fixed 3×4 grid but rebuilds it every run from the previous draw (same stream). "
    "It generates a ranked STRAIGHT list first (64 core, plus optional targeted rescue), then trims with simple filters."
)

with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader(
        "Upload Pick-3 history TXT",
        type=["txt"],
        help=(
            "Tab-delimited lines like: 'Fri, Jan 9, 2026\tGeorgia\tCash 3 Evening\t0-0-2'. "
            "Both 'Pick 3' and 'Cash 3' formats are supported."
        ),
    )

    grid_mode = st.selectbox(
        "Grid formula",
        options=["Seed-conditioned (recommended)", "Due digits (legacy)",],
        index=0,
        help=(
            "Seed-conditioned = carryover digit + ±1 neighbors + mirror + learned followers (same stream). "
            "Due digits = top-overdue digits per position (not seed-aware)."
        ),
    )

    follower_lookback = st.number_input(
        "Follower lookback (draws)",
        min_value=30,
        max_value=5000,
        value=200,
        step=10,
        help=(
            "How many prior same-stream draws to learn follower behavior from. "
            "Smaller = more recent/adaptive. Larger = more stable."
        ),
    )

    follower_k = st.slider(
        "Followers injected into each column",
        min_value=0,
        max_value=4,
        value=2,
        step=1,
        help=(
            "How many 'top follower digits' (by position) to include in the candidate pool for each column. "
            "These help keep winners from never being generated."
        ),
    )

    st.divider()
    st.header("Rescue expansion")

    rescue_enabled = st.checkbox(
        "Enable rescue expansion",
        value=True,
        help=(
            "If a position is historically 'wild' for the current seed digit, add extra straights by swapping that position "
            "with the next-best follower digits not already in the column. This boosts coverage without exploding the list."
        ),
    )

    rescue_scope = st.selectbox(
        "Rescue scope",
        options=["Weakest position only", "All weak positions"],
        index=0,
        help=(
            "Weakest position only = adds the smallest number of combos (best default). "
            "All weak positions = more coverage, bigger list."
        ),
        disabled=not rescue_enabled,
    )

    rescue_k = st.slider(
        "Rescue digits (K)",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help=(
            "How many non-grid follower digits to try in the rescued position. K=2 adds 32 combos (16×2)."
        ),
        disabled=not rescue_enabled,
    )

    rescue_mass_threshold = st.slider(
        "Rescue trigger threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.55,
        step=0.01,
        help=(
            "We compute how much probability mass is covered by the 4 grid digits at a position. "
            "If that mass is ≤ this threshold, the position is considered 'wild' and rescue activates."
        ),
        disabled=not rescue_enabled,
    )

    st.divider()
    st.header("Trims & ranking")

    alpha = st.slider(
        "Probability smoothing (alpha)",
        min_value=0.2,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help=(
            "Laplace smoothing for follower probabilities. Alpha=1 is usually stable. Larger = flatter probabilities."
        ),
    )

    keep_top_pct = st.slider(
        "Keep top % of straights (by score)",
        min_value=5,
        max_value=100,
        value=35,
        step=5,
        help=(
            "Percentile trim after generation. This is meant to reduce cost AFTER we already ensured the winner is likely present."
        ),
    )

    drop_all_even = st.checkbox(
        "Drop EEE (all even)",
        value=False,
        help=(
            "Optional. Dropping EEE can cut the list, but can also remove real winners depending on the stream."
        ),
    )

    drop_all_odd = st.checkbox(
        "Drop OOO (all odd)",
        value=False,
        help=(
            "Optional. Dropping OOO can cut the list, but can also remove real winners depending on the stream."
        ),
    )


if not uploaded:
    st.info("Upload your Pick-3 history TXT to begin.")
    st.stop()

raw = uploaded.read().decode("utf-8", errors="ignore")
df = parse_history_text(raw)

if df.empty:
    st.error("Couldn't parse any draws from that file. Make sure it's tab-delimited and includes Pick 3 / Cash 3 lines.")
    st.stop()

# State + stream selection
states = sorted(df["state"].unique().tolist())
colA, colB, colC = st.columns([1, 1, 2])

with colA:
    state = st.selectbox(
        "State",
        options=states,
        index=0,
        help="Which state/lottery the history file contains.",
    )

sub = df[df["state"] == state].copy()
streams = sorted(sub["draw"].unique().tolist(), key=lambda s: DRAW_ORDER.get(s, 99))

with colB:
    stream = st.selectbox(
        "Draw stream",
        options=streams,
        index=0,
        help="We model same-stream transitions (e.g., Midday→Midday).",
    )

stream_df = sub[sub["draw"] == stream].copy().sort_values(["date"]).reset_index(drop=True)
stream_nums = stream_df["num"].astype(str).tolist()

with colC:
    st.write(
        f"**{state} – {stream}** history: {len(stream_nums)} draws from {stream_df['date'].min()} to {stream_df['date'].max()}"
    )

# Grid build + candidates for the most recent seed
if len(stream_nums) < 2:
    st.error("Not enough draws in this stream to build a seed-based grid.")
    st.stop()

mode_key = "seed" if grid_mode.startswith("Seed") else "due"

# slot hit stats (recomputed for the selected formula)
slot_counts, rates = grid_slot_hit_stats(
    stream_nums,
    grid_builder=mode_key,
    rows=4,
    follower_lookback=int(follower_lookback),
    last_n=min(800, max(50, len(stream_nums) - 1)),
)

seed = str(stream_nums[-1]).zfill(3)

# Build candidates for "next" draw (seed_index = len(stream_nums))
# We use the last known draw as seed.
next_index = len(stream_nums)

grid, (colH, colT, colO), candidates, counts, rescue_info = build_candidates_for_seed(
    stream_nums,
    seed_index=next_index,
    grid_mode=mode_key,
    follower_lookback=int(follower_lookback),
    follower_k=int(follower_k),
    alpha=float(alpha),
    rescue_enabled=bool(rescue_enabled),
    rescue_k=int(rescue_k),
    rescue_mass_threshold=float(rescue_mass_threshold),
    rescue_scope=rescue_scope,
)

ranked = rank_candidates(
    candidates,
    seed=seed,
    counts=counts,
    alpha=float(alpha),
    drop_all_even=bool(drop_all_even),
    drop_all_odd=bool(drop_all_odd),
    keep_top_pct=float(keep_top_pct) if keep_top_pct < 100 else None,
)

boxes = aggregate_boxes(ranked)

# -----------------------------
# Display grid + mapping
# -----------------------------

st.subheader("Current 3×4 grid (labeled slots)")

letters = slot_labels(4)

grid_table = []
for r in range(4):
    grid_table.append({
        "Rank": letters[r],
        "Hundreds (1)": grid[r][0],
        "Tens (2)": grid[r][1],
        "Ones (3)": grid[r][2],
    })

df_grid = pd.DataFrame(grid_table)

col1, col2 = st.columns([1.1, 1])

with col1:
    st.dataframe(df_grid, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Slot mapping (12 cells):**")
    st.markdown(
        "- Hundreds row slots: "
        + ", ".join([f"**1{L}={colH[i]}**" for i, L in enumerate(letters)])
    )
    st.markdown(
        "- Tens row slots: "
        + ", ".join([f"**2{L}={colT[i]}**" for i, L in enumerate(letters)])
    )
    st.markdown(
        "- Ones row slots: "
        + ", ".join([f"**3{L}={colO[i]}**" for i, L in enumerate(letters)])
    )

    if rescue_info.get("triggered"):
        st.success(
            f"Rescue ON: triggered on position **{rescue_info['position']}** (top-4 mass={rescue_info['mass']:.3f}). "
            f"Added {rescue_info['added']} straights."
        )
        st.write(f"Rescue digits: {rescue_info['digits']}")
    else:
        st.info("Rescue did not trigger for this seed (or rescue is OFF).")

st.subheader("Recomputed historical grid-hit stats for this formula")

colS1, colS2, colS3, colS4 = st.columns(4)
with colS1:
    st.metric("Hundreds digit in its grid column", f"{rates['H']*100:.1f}%")
with colS2:
    st.metric("Tens digit in its grid column", f"{rates['T']*100:.1f}%")
with colS3:
    st.metric("Ones digit in its grid column", f"{rates['O']*100:.1f}%")
with colS4:
    st.metric("≥2 winner digits anywhere in grid", f"{rates['Any2Anywhere']*100:.1f}%")

st.caption("Rates are computed walk-forward (no look-ahead), using the last ~N transitions in this stream.")

# Slot ranking (least -> most)
slot_rank = sorted(slot_counts.items(), key=lambda kv: (kv[1], kv[0]))
slot_rank_str = ", ".join([f"{k} ({v})" for k, v in slot_rank])

with st.expander("Slot hit ranking (least → most)"):
    st.write(slot_rank_str)

# -----------------------------
# Outputs
# -----------------------------

st.subheader("Ranked STRAIGHT list (play order = most likely straight first)")

colO1, colO2 = st.columns([1.2, 1])
with colO1:
    st.write(f"Seed (previous {stream} draw): **{seed}**")
    st.write(f"Generated candidates: **{len(candidates)}** (base=64 + rescue)")
    st.write(f"After trims: **{len(ranked)}**")

with colO2:
    # show parity distribution
    if not ranked.empty:
        par_counts = ranked["parity"].value_counts().to_dict()
        st.write("Parity counts (after trims):")
        st.json(par_counts)

st.dataframe(ranked, use_container_width=True, hide_index=True)

st.download_button(
    "Download straights (CSV)",
    data=ranked.to_csv(index=False).encode("utf-8"),
    file_name=f"pick3_{state}_{stream}_straights.csv".replace(" ", "_"),
    mime="text/csv",
)

st.subheader("BOX summary (best straight per box)")

st.dataframe(boxes, use_container_width=True, hide_index=True)

st.download_button(
    "Download boxes (CSV)",
    data=boxes.to_csv(index=False).encode("utf-8"),
    file_name=f"pick3_{state}_{stream}_boxes.csv".replace(" ", "_"),
    mime="text/csv",
)

# -----------------------------
# Backtest
# -----------------------------

with st.expander("Walk-forward backtest (same stream)"):
    st.write(
        "This evaluates how often the winner is contained by the generated list when we walk through history, "
        "always using only past data (no look-ahead)."
    )

    start_at = st.number_input(
        "Backtest start index (warmup)",
        min_value=5,
        max_value=max(10, len(stream_nums) - 1),
        value=min(30, max(5, len(stream_nums) // 5)),
        step=1,
        help=(
            "We skip the first N draws so the follower tables have some data. "
            "Higher = more stable but fewer test points."
        ),
    )

    run_bt = st.button("Run backtest")
    if run_bt:
        res = walk_forward_backtest(
            stream_nums,
            grid_mode=mode_key,
            follower_lookback=int(follower_lookback),
            follower_k=int(follower_k),
            alpha=float(alpha),
            rescue_enabled=bool(rescue_enabled),
            rescue_k=int(rescue_k),
            rescue_mass_threshold=float(rescue_mass_threshold),
            rescue_scope=rescue_scope,
            drop_all_even=bool(drop_all_even),
            drop_all_odd=bool(drop_all_odd),
            keep_top_pct=float(keep_top_pct) if keep_top_pct < 100 else None,
            start_at=int(start_at),
        )
        st.success(
            f"Tested {res['N']} transitions | Straight hit rate: {res['StraightHitRate']*100:.1f}% | "
            f"Box hit rate: {res['BoxHitRate']*100:.1f}%"
        )

