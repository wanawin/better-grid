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
        raw_num = m.group("num").strip()
        num = raw_num.replace(" ", "").replace("-", "")
        try:
            d = dt.datetime.strptime(date_str, "%a, %b %d, %Y").date()
        except Exception:
            continue
        if len(num) == 3 and num.isdigit():
            rows.append((d, state, draw, raw_num, num))

    df = pd.DataFrame(rows, columns=["date", "state", "draw", "raw_num", "num"])
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



def digits_str(x, width=3):
    """Return a zero-padded digit string of fixed width.

    Accepts int (e.g., 77), str (e.g., '1-7-7', '177'), or an iterable of digits.
    Always returns exactly `width` numeric characters (left-padded with zeros if needed).
    """
    if x is None:
        return '0' * width

    # Common case: already a digit string
    if isinstance(x, str):
        digits = ''.join(ch for ch in x if ch.isdigit())
        if digits == '':
            return '0' * width
        if len(digits) < width:
            return digits.zfill(width)
        if len(digits) > width:
            return digits[-width:]
        return digits

    if isinstance(x, int):
        return f"{x:0{width}d}"[-width:]

    if isinstance(x, (list, tuple)):
        digits = ''.join(str(int(d)) for d in x)
        if len(digits) < width:
            return digits.zfill(width)
        if len(digits) > width:
            return digits[-width:]
        return digits

    # Fallback
    return digits_str(str(x), width=width)

def parity_sig(num: str) -> str:
    ev = sum((int(d) % 2 == 0) for d in num)
    if ev == 3:
        return "EEE"
    if ev == 2:
        return "EEO"
    if ev == 1:
        return "EOO"
    return "OOO"


# --- Derived feature helpers ---

def digit_sum(num: str) -> int:
    s = digits_str(num, width=3)
    return int(s[0]) + int(s[1]) + int(s[2])


def parity_pattern(num: str) -> str:
    # EEE / EEO / EOO / OOO (based on count of even digits)
    return parity_sig(digits_str(num, width=3))


def aggregate_boxes(ranked_df: pd.DataFrame) -> pd.DataFrame:
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame(columns=['box','straight','score','score_base','boost_mult','sum','parity','unique','carry_ct','mirror_ct','neighbor_ct'])
    df = ranked_df.copy()
    if 'box' not in df.columns:
        df['box'] = df['straight'].astype(str).apply(box_key)
    # ranked_df is already sorted best-first; keep the best straight per box
    df = df.drop_duplicates('box', keep='first').reset_index(drop=True)
    # Nice column order
    cols = [c for c in ['box','straight','score','score_base','boost_mult','sum','parity','unique','carry_ct','mirror_ct','neighbor_ct'] if c in df.columns]
    other = [c for c in df.columns if c not in cols]
    return df[cols + other]


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
    w_carry: float = 1000.0,
    w_neighbor: float = 50.0,
    w_mirror: float = 900.0,
):
    """Seed-conditioned 3x4 grid.

    IMPORTANT BEHAVIOR (to prevent confusion):
    - This grid is built so that ROWS correspond to transformations of the seed:
        Row A = seed
        Row B = seed + 1 (per-digit, mod 10)
        Row C = seed - 1 (per-digit, mod 10)
        Row D = mirror(seed)
      (Duplicates are de-duped per-column; any missing slots are padded from followers/overdue.)

    This keeps the grid stable (3x4) while preserving your manual row-order logic.

    Returns:
      grid: rows x 3 digits (strings)
      drought: per-position drought dicts
      dbg: per-position debug payload to explain choices
    """
    # fallback early
    if upto_exclusive < 1:
        grid, drought = build_due_grid(stream_nums, upto_exclusive, rows=rows, window=None)
        return grid, drought, {"mode": "due_fallback"}

    drought = compute_drought(stream_nums, upto_exclusive, window=None)

    # seed is last known draw in this stream slice
    seed = str(stream_nums[upto_exclusive - 1]).zfill(3)

    # follower counts keyed by (position, prev_digit -> next_digit counts)
    followers = compute_pos_followers(stream_nums, upto_exclusive - 1, lookback=follower_lookback)

    cols: list[list[str]] = []
    dbg: dict = {"seed": seed, "per_pos": []}

    for pos in range(3):
        s = int(seed[pos])
        n_plus = _mod10(s + 1)
        n_minus = _mod10(s - 1)
        m = _MIRROR.get(s, s)

        fdict = followers[pos].get(seed[pos], {})
        tf = sorted(fdict.items(), key=lambda kv: (-kv[1], int(kv[0])))[: int(follower_k)]
        tf_digits = [int(d) for d, _ in tf]
        # base row-order digits: seed row, +1 row, -1 row, mirror row
        base_order: list[int] = []
        for d in [s, n_plus, n_minus, m]:
            if d not in base_order:
                base_order.append(d)

        fdict = followers[pos].get(seed[pos], {})
        tf = sorted(fdict.items(), key=lambda kv: (-kv[1], int(kv[0])))[: int(follower_k)]
        tf_digits = [int(d) for d, _ in tf]

        # optional pool: top followers (used only when duplicates shrink the base_order)
        cand_optional: list[int] = []
        for d in tf_digits:
            if d not in base_order and d not in cand_optional:
                cand_optional.append(d)

        # score optional digits
        scored_optional: list[tuple[int, float, dict]] = []
        for d in cand_optional:
            comp = {
                "carry": float(w_carry if d == s else 0.0),
                "mirror": float(w_mirror if d == m else 0.0),
                "neighbor": float(w_neighbor if d in (n_plus, n_minus) else 0.0),
                "follower_count": float(fdict.get(str(d), 0)),
            }
            score = comp["carry"] + comp["mirror"] + comp["neighbor"] + comp["follower_count"]
            scored_optional.append((d, score, comp))

        scored_optional_sorted = sorted(scored_optional, key=lambda t: (-t[1], t[0]))

        chosen = base_order.copy()
        chosen_scores: dict[int, float] = {}

        # record scores for base digits too (for transparency; ordering is fixed)
        for d in chosen:
            comp = {
                "carry": float(w_carry if d == s else 0.0),
                "mirror": float(w_mirror if d == m else 0.0),
                "neighbor": float(w_neighbor if d in (n_plus, n_minus) else 0.0),
                "follower_count": float(fdict.get(str(d), 0)),
            }
            chosen_scores[d] = comp["carry"] + comp["mirror"] + comp["neighbor"] + comp["follower_count"]

        # fill remaining slots from followers
        for d, score, _comp in scored_optional_sorted:
            if len(chosen) >= rows:
                break
            if d not in chosen:
                chosen.append(d)
                chosen_scores[d] = score

        # pad using overdue digits if still short
        if len(chosen) < rows:
            due_items = sorted(drought[pos].items(), key=lambda kv: (-kv[1], int(kv[0])))
            for dd, _ in due_items:
                di = int(dd)
                if di not in chosen:
                    chosen.append(di)
                    # drought fill gets a tiny score just for debugging
                    chosen_scores[di] = -9999.0
                if len(chosen) >= rows:
                    break

        # final order is the fixed row-order (seed, +1, -1, mirror), then any padding
        final = chosen[: int(rows)]

        cols.append([str(d) for d in final])

        dbg["per_pos"].append(
            {
                "pos": pos,
                "seed_digit": str(s),
                "mirror_digit": str(m),
                "neighbors": [str(n_plus), str(n_minus)],
                "top_followers": [(str(d), int(c)) for d, c in tf],
                "base_order": [str(d) for d in base_order],
                "optional_scored": [(str(d), float(score), comp) for d, score, comp in scored_optional_sorted],
                "chosen_final": [str(d) for d in final],
                "chosen_scores": {str(k): float(v) for k, v in chosen_scores.items()},
            }
        )

    grid = [[cols[c][r] for c in range(3)] for r in range(rows)]
    return grid, drought, dbg



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
            g, _, _ = build_seed_grid(stream_nums, i, rows=rows, follower_lookback=follower_lookback)
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
        grid, _, dbg = build_seed_grid(
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
    keep_top_n: int | None,
):
    """Rank candidate straights.

    Base score comes from the position-wise follower table (seed-position conditional).
    Then we apply a small inclusion boost (carry/mirror/+/-1 overlap with the seed)
    so we can improve coverage *without* changing grid ordering.
    """

    seed = digits_str(seed)
    seed_digits = set(seed)
    mirror_digits = {str(_MIRROR[int(d)]) for d in seed_digits}
    neighbor_digits = {str((int(d) + 1) % 10) for d in seed_digits} | {str((int(d) - 1) % 10) for d in seed_digits}

    rows = []
    for s in candidates:
        s = digits_str(s)
        base = float(score_straight(s, seed, counts, alpha))

        carry_ct = sum(ch in seed_digits for ch in s)
        mirror_ct = sum(ch in mirror_digits for ch in s)
        neighbor_ct = sum(ch in neighbor_digits for ch in s)

        # Multiplicative boost, capped (keeps things stable)
        boost_mult = 1.0 + 0.25 * carry_ct + 0.15 * mirror_ct + 0.10 * neighbor_ct
        boost_mult = min(2.5, boost_mult)
        score = base * boost_mult

        rows.append(
            {
                "straight": s,
                "score": score,
                "score_base": base,
                "boost_mult": boost_mult,
                "carry_ct": carry_ct,
                "mirror_ct": mirror_ct,
                "neighbor_ct": neighbor_ct,
                "box": box_key(s),
                "sum": digit_sum(s),
                "parity": parity_pattern(s),
                "unique": len(set(s)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["score", "score_base"], ascending=False).reset_index(drop=True)

    if drop_all_even:
        df = df[df["parity"] != "EEE"]
    if drop_all_odd:
        df = df[df["parity"] != "OOO"]

    if keep_top_n is not None:
        try:
            n = int(keep_top_n)
        except Exception:
            n = None
        if n is not None and n > 0:
            df = df.head(n)

    return df.reset_index(drop=True)


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
    keep_top_n: int | None,
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
        ranked = rank_candidates(cand, seed, counts, alpha, drop_all_even, drop_all_odd, keep_top_n)
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

    debug_show_parsed = st.checkbox(
        "Debug: show parsed history rows",
        value=False,
        help=(
            "Shows the last parsed rows for the selected stream, including the raw '1-7-7' token and the parsed '177'. "
            "Useful to confirm the app isn't misreading results (for example, 177 vs 077)."
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
    # Tight grid: no follower-digit injection into columns (set to 0).
    follower_k = 0

    st.divider()
    st.header("Rescue expansion")

    rescue_enabled = st.checkbox(
        "Enable rescue expansion",
        value=False,
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

    keep_top_n = st.slider(
        "Keep top N straights (after scoring)",
        min_value=20,
        max_value=300,
        value=120,
        step=1,
        help=(
            "Keeps the top N ranked STRAIGHTS after scoring. "
            "Use this when the winner keeps dropping: bump N up to widen coverage. "
            "If N is larger than the available list, the app will just keep them all."
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

# --- Debug: show parsed history rows and validate parsing ---
if 'debug_show_parsed' in globals() and debug_show_parsed:
    with st.expander("Debug: parsed history (this stream)", expanded=True):
        cols = [c for c in ["date", "state", "draw", "raw_num", "num"] if c in stream_df.columns]
        dbg = stream_df[cols].copy()
        if "raw_num" in dbg.columns:
            expected = (
                dbg["raw_num"].astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("-", "", regex=False)
                .str.zfill(3)
            )
            actual = dbg["num"].astype(str).str.zfill(3)
            mism = dbg.loc[expected != actual]
            st.write(f"Parsing mismatches: **{len(mism)}**")
            if len(mism):
                st.dataframe(mism.tail(50), use_container_width=True)
        find_num = st.text_input("Find a specific result (optional, e.g., 177)", value="")
        view = dbg if not find_num.strip() else dbg.loc[dbg["num"].astype(str).str.zfill(3) == find_num.strip().zfill(3)]
        st.caption("Tip: if you see '077' in the history, that can be an actual draw. This table shows both the raw token (e.g. '1-7-7') and the parsed 3-digit string.")
        st.dataframe(view.tail(80), use_container_width=True)

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
    keep_top_n=int(keep_top_n),
)

boxes = aggregate_boxes(ranked)

# -----------------------------
# Display grid + mapping
# -----------------------------

st.subheader("Current 3×4 grid (labeled slots)")
st.caption("This grid is built from the previous draw (seed) in the selected stream. Row A is **not** the seed — it's just the top-ranked digit in each position column.")
st.write(f"Seed (previous {stream} draw): **{seed}**")

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
    mirror_seed = "".join(str(_MIRROR[int(d)]) for d in seed)
    st.write(f"Mirror of seed: **{mirror_seed}**  (0<->5, 1<->6, 2<->7, 3<->8, 4<->9)")
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
            keep_top_n=int(keep_top_n),
            start_at=int(start_at),
        )
        st.success(
            f"Tested {res['N']} transitions | Straight hit rate: {res['StraightHitRate']*100:.1f}% | "
            f"Box hit rate: {res['BoxHitRate']*100:.1f}%"
        )

