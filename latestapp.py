import os
import re
import math
import random
import csv
import ast          # ← move here
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript
from uuid import uuid4
from streamlit_scroll_to_top import scroll_to_here
import time
import pandas as pd  # you already have this; OK to ignore if present
from pathlib import Path


# ensure keys exist
if "_scroll_to_top" not in st.session_state:
    st.session_state["_scroll_to_top"] = False
if "_scroll_token" not in st.session_state:
    st.session_state["_scroll_token"] = ""


st.set_page_config(page_title="Trial Presenter", page_icon="🖼️", layout="wide")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
CSV_DIR = os.path.join(STATIC_DIR, "CSVs")
os.makedirs(CSV_DIR, exist_ok=True)

# ======================== ORDERINGS ========================
# ORDERINGS = {
#     1: {"rep":[2,3,1,3,1,2,2,3,1,2,1,3,3,2,1,1,3,2],
#         "expl":[2,2,2,1,1,1,3,3,3,3,3,3,1,1,1,2,2,2],
#         "num":[6,4,7,15,13,16,30,22,36,19,25,33,10,27,24,12,9,1]},
#     2: {"rep":[3,2,1,3,1,2,2,1,3,3,1,2,1,3,2,1,3,2],
#         "expl":[2,2,2,1,1,1,3,3,3,2,2,2,1,1,1,3,3,3],
#         "num":[4,6,7,15,13,16,30,36,22,9,12,1,24,10,27,25,33,19]},
#     3: {"rep":[1,2,3,2,3,1,2,3,1,2,3,1,3,2,1,3,2,1],
#         "expl":[1,1,1,3,3,3,2,2,2,1,1,1,2,2,2,3,3,3],
#         "num":[13,16,15,30,22,36,6,4,7,27,10,24,9,1,12,33,19,25]},
#     4: {"rep":[1,3,2,3,1,2,1,2,3,3,2,1,1,3,2,3,1,2],
#         "expl":[3,3,3,2,2,2,1,1,1,3,3,3,1,1,1,2,2,2],
#         "num":[36,22,30,4,7,6,13,16,15,33,19,25,24,10,27,9,12,1]},
#     5: {"rep":[3,1,2,1,2,3,3,2,1,2,3,1,2,1,3,1,2,3],
#         "expl":[1,1,1,2,2,2,3,3,3,2,2,2,3,3,3,1,1,1],
#         "num":[15,13,16,7,6,4,22,30,36,1,9,12,19,25,33,24,27,10]},
# }

ORDERINGS = {
    1: {
        "rep":  [2,3,1,3,1,2,2,3,1,2,1,3,3,2,1,1,3,2],
        "expl": [2,2,2,1,1,1,3,3,3,3,3,3,1,1,1,2,2,2],
        "num":  [52, 3, 4, 57, 6, 8, 62, 12, 64, 9, 14, 63, 5, 61, 60, 55, 53, 2],
    },
    2: {
        "rep":  [3,2,1,3,1,2,2,1,3,3,1,2,1,3,2,1,3,2],
        "expl": [2,2,2,1,1,1,3,3,3,2,2,2,1,1,1,3,3,3],
        "num":  [3, 52, 4, 57, 6, 8, 62, 64, 12, 53, 55, 2, 60, 5, 61, 14, 63, 9],
    },
    3: {
        "rep":  [1,2,3,2,3,1,2,3,1,2,3,1,3,2,1,3,2,1],
        "expl": [1,1,1,3,3,3,2,2,2,1,1,1,2,2,2,3,3,3],
        "num":  [6, 8, 57, 62, 12, 64, 52, 3, 4, 61, 5, 60, 53, 2, 55, 63, 9, 14],
    },
    4: {
        "rep":  [1,3,2,3,1,2,1,2,3,3,2,1,1,3,2,3,1,2],
        "expl": [3,3,3,2,2,2,1,1,1,3,3,3,1,1,1,2,2,2],
        "num":  [64, 12, 62, 3, 4, 52, 6, 8, 57, 63, 9, 14, 60, 5, 61, 53, 55, 2],
    },
    5: {
        "rep":  [3,1,2,1,2,3,3,2,1,2,3,1,2,1,3,1,2,3],
        "expl": [1,1,1,2,2,2,3,3,3,2,2,2,3,3,3,1,1,1],
        "num":  [57, 6, 8, 4, 52, 3, 12, 62, 64, 2, 53, 55, 9, 14, 63, 60, 61, 5],
    },
}


# ======================== LABELS ========================
REP_LABELS  = {1: "Text", 2: "TimeTable", 3: "DiffusionModel"}
EXPL_LABELS = {1: "Text Explanation", 2: "Counterfactual", 3: "InputOutputTrace"}

# ======================== MAPPINGS (leftovers → replacement) ========================
# rem1_map = {1:28, 4:31, 7:34, 10:37, 13:40, 16:43, 19:46, 22:49, 25:52}
# mod0_map = {6:39, 9:42, 12:45, 15:57, 24:60, 27:72, 30:75, 33:81, 36:84}

rem1_map = {2: 15, 3: 16, 4: 18, 5: 20, 6: 21, 8: 22, 9: 24, 12: 25, 14: 26}
mod0_map = {52: 65, 53: 66, 55: 67, 57: 69, 60: 71, 61: 72, 62: 73, 63: 76, 64: 77}

def mapped_number(n: int) -> int | None:
    if 0 < n < 51: return rem1_map.get(n)
    if 50 < n < 101: return mod0_map.get(n)
    return None

# ======================== PATH HELPERS ========================
def prompt_path(num: int, version: int) -> str:
    return os.path.join(STATIC_DIR, f"prompt_p_{num}_v{version}_base.png")

def rep_path(rep_type: int, num: int, version: int) -> str:
    # type 1 -> text_timetable; types 2/3 -> it_p_
    if rep_type == 1:
        return os.path.join(STATIC_DIR, f"text_timetable_{num}_v{version}.png")
    # if rep_type in (2, 3):
    #     return os.path.join(STATIC_DIR, f"it_p_{num}_v_{version}.png")
    if rep_type == 2:
        return os.path.join(STATIC_DIR, f"it_p_{num}_v_{version}.png")
    if rep_type == 3:
        return os.path.join(STATIC_DIR, f"diffusion_model_{num}_v{version}.png")
    return ""

def expl_path(expl_type: int, num: int) -> str:
    if expl_type == 1:
        return os.path.join(STATIC_DIR, f"explanation_{num:03d}.png")
    if expl_type in (2, 3):
        return os.path.join(STATIC_DIR, f"schedule_card_{num}.png")
    return ""

# def show_image_or_warn(path: str, label: str = ""):
#     if os.path.exists(path):
#         st.image(path, use_column_width=True, caption=label if label else None)
#     else:
#         st.warning(f"Missing file: {os.path.relpath(path, BASE_DIR)}")

def show_image_or_warn(path: str, label: str = "", custom_size=0):
    if os.path.exists(path):
        # replace deprecated arg:
        if custom_size == 0:
            # st.image(path, use_container_width=True, caption=label if label else None, width=200)
            st.image(path, use_container_width=True, caption=label if label else None)
        else:
            # st.image(path, use_container_width=True, caption=label if label else None, width=400)
            st.image(path, caption=label if label else None, width=custom_size)

    else:
        st.warning(f"Missing file: {os.path.relpath(path, BASE_DIR)}")

def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().split()).lower()

# Build a natural sentence from the template and the current answers
PLACEHOLDER_RE = re.compile(r"\{(days|time_of_day|number_of_classes|course_level_minimum)\}")

def assemble_attempted_sentence(template_text: str | None, answers: dict) -> str:
    if not template_text:
        # Fallback if no template (won't happen in your current flow)
        parts = []
        if answers.get("days"): parts.append(f"I want classes on {answers['days']}")
        if answers.get("time_of_day"): parts.append(f"in the {answers['time_of_day']} period")
        if answers.get("number_of_classes"): parts.append(f"with {answers['number_of_classes']} classes per day")
        if answers.get("course_level_minimum"): parts.append(f"at or above the {answers['course_level_minimum']}-level")
        return ", ".join(parts) + "."
    def repl(m):
        key = m.group(1)
        val = answers.get(key)
        return str(val) if val not in (None, "") else f"{{{key}}}"
    return PLACEHOLDER_RE.sub(repl, template_text)

# Put near ALLOWED_PARAMS / POOLS
ANY_ALL_CHOICES = ["Any/All"]

def _with_any_all(options):
    """Prepend 'Any' and 'All' (without duplicates) to a list of options."""
    seen = set()
    out = []
    for x in ANY_ALL_CHOICES + list(options or []):
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# @st.cache_data
def build_option_pools(df: pd.DataFrame) -> dict:
    """Build dropdown pools by scanning the whole CSV (so you control options centrally)."""
    cols = {c.lower(): c for c in df.columns}

    def collect_days():
        vals=set()
        for cname in ("days","preferred_days","v1_days","v1_preferred_days"):
            if cname in cols:
                for v in df[cols[cname]].dropna():
                    L = _parse_listish(v)
                    if isinstance(L, list):
                        disp = _display_days(L)
                        if disp: vals.add(disp)
                    elif isinstance(L, str) and L.strip():
                        vals.add(_display_days(L))
        return sorted(vals)

    def collect_str(name_candidates, default):
        vals=set()
        for cname in name_candidates:
            if cname in cols:
                for v in df[cols[cname]].dropna():
                    s = str(v).strip().lower()
                    if s and s != "none": vals.add(s)
        return sorted(vals) if vals else default

    def collect_int(name_candidates, default):
        vals=set()
        for cname in name_candidates:
            if cname in cols:
                for v in df[cols[cname]].dropna():
                    try:
                        vals.add(int(str(v).strip()))
                    except Exception:
                        pass
        return sorted(vals) if vals else default

    pools = {
        "days": collect_days() or [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Monday and Wednesday", "Monday and Thursday", "Monday and Friday",
    "Tuesday and Wednesday", "Tuesday and Thursday", "Tuesday and Friday",
    "Wednesday and Thursday", "Wednesday and Friday"
],
        "time_of_day": collect_str(("time_of_day","timewindow","time_window","v1_time_of_day"),
                                   ["morning","afternoon","evening"]),
        "number_of_classes": collect_int(("number_of_classes","max_classes_per_day"), [1,2,3]),
        "course_level_minimum": collect_int(("course_level_minimum","min_level"), [200,300,400]),
    }
    return pools

def _parse_listish(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip()
    if not s: return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)): return list(v)
        return v
    except Exception:
        pass
    # fallback: list-ish by delimiters
    if any(d in s for d in [",", "&"]):
        return [p.strip() for p in re.split(r"[,&]", s) if p.strip()]
    return s

def _display_days(val):
    if isinstance(val, list): 
        return " & ".join([str(d).strip().capitalize() for d in val if str(d).strip()])
    if isinstance(val, str):
        parts = [p.strip().capitalize() for p in re.split(r"[,&]", val) if p.strip()]
        return " & ".join(parts) if parts else val.strip()
    return str(val)

def _sync_nav_from_widget():
    # called when the user changes the sidebar number_input
    st.session_state["nav_trial_idx"] = int(st.session_state["trial_idx_widget"])
    # optional: when user jumps via the widget, reset to first substep
    st.session_state["nav_substep"] = 1


def debug_prompt_lookup(n: int):
    st.info("🔎 Prompt lookup debug", icon="🔍")
    # What CSV did we load?
    st.write("OPTIONS_DF shape:", getattr(OPTIONS_DF, "shape", None))
    st.write("Columns:", list(getattr(OPTIONS_DF, "columns", [])))

    # Show key distribution
    key_cols = [c for c in OPTIONS_DF.columns if c.lower() in ("num","promptnum","prompt_num","id")]
    st.write("Detected key columns:", key_cols or "(none)")

    # What ID are we trying to fetch?
    st.write("Requested ID:", n)

    # Does the index map know this ID?
    idx = OPTIONS_INDEX.get(int(n)) if isinstance(n, (int,float,str)) and str(n).strip().isdigit() else None
    st.write("Resolved row index (0-based) from OPTIONS_INDEX:", idx)

    # If not mapped, show why we’ll fall back
    if idx is None:
        st.warning("ID not found in index — falling back to the first row (this would make the prompt repeat).")

    # Show a small preview of the row we’re actually using
    row = OPTIONS_DF.iloc[idx if idx is not None else 0]
    preview_keys = [c for c in OPTIONS_DF.columns if c.lower() in ("num","promptnum","prompt_num","id")]
    st.write("Row keys:", {c: row[c] for c in preview_keys})
    # Adjust the column name below to your template column
    tmpl_col = "template" if "template" in OPTIONS_DF.columns else (
        "fillin_template" if "fillin_template" in OPTIONS_DF.columns else None
    )
    if tmpl_col:
        st.write(f"Template column '{tmpl_col}' preview:", (row[tmpl_col] or "")[:200])
    else:
        st.error("No template column (v1_template/fillin_template) found in CSV.")


# ======================== CSV (options) LOADING ========================
CSV_PATH_CANDIDATES = [
    # os.path.join(BASE_DIR, "generated_prompts_texts.csv"),
    os.path.join(STATIC_DIR, "generated_prompts_texts.csv")
]

@st.cache_data
def _extract_first_promptnum_like(text: str) -> int | None:
    """
    Try to infer a prompt number from a string using file-name patterns.
    Looks for:
      - p_{num}    (e.g., 'prompt_p_39_v1_base.png', 'it_p_72_v_1.png')
      - text_timetable_{num}
      - schedule_card_{num}
      - explanation_{num}
    Returns the first match as int, else None.
    """
    if not isinstance(text, str) or not text:
        return None
    patterns = [
        r"\bp_(\d+)\b",                    # p_39
        r"text_timetable[_\- ]?(\d+)",     # text_timetable_39
        r"schedule_card[_\- ]?(\d+)",      # schedule_card_39
        r"explanation[_\- ]?(\d+)",        # explanation_039 or explanation_39
        r"\b(\d{1,3})\b"                   # fallback lone number
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None

@st.cache_data
def load_options_df_and_index(path_candidates: list[str]):
    """
    Load the entire options CSV once, normalize columns, and build an index:
      - df: the DataFrame (cols lowercased)
      - key_col: detected explicit key column (promptnum/prompt_num/num/id) or None
      - idx: dict[int -> row_index], derived from key_col or by inferring from cell strings
    """
    df = pd.DataFrame()
    key_col = None
    for p in path_candidates:
        if os.path.exists(p):
            try:
                tmp = pd.read_csv(p)
                if tmp is not None and not tmp.empty:
                    df = tmp
                    break
            except Exception:
                continue
    if df.empty:
        return pd.DataFrame(), None, {}

    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # try explicit key column
    for kc in ("promptnum", "prompt_num", "num", "id"):
        if kc in df.columns:
            key_col = kc
            break

    index_map: dict[int, int] = {}

    if key_col:
        # use explicit key col (coerce to int if possible)
        for i, v in enumerate(df[key_col].tolist()):
            try:
                n = int(v)
                index_map[n] = i
            except Exception:
                # ignore non-int rows
                pass

    # If still missing some rows or no key_col, try to infer from any cell
    if not key_col or len(index_map) < len(df):
        for i in range(len(df)):
            row = df.iloc[i]
            # scan all cells for a usable number
            found = None
            for c in df.columns:
                val = row[c]
                if pd.isna(val): 
                    continue
                n = _extract_first_promptnum_like(str(val))
                if n is not None:
                    found = n
                    # prefer not to overwrite explicit mapping
                    if n not in index_map:
                        index_map[n] = i
                    break
            # nothing found → leave for fallback

    return df, key_col, index_map

OPTIONS_DF, OPTIONS_KEYCOL, OPTIONS_INDEX = load_options_df_and_index(CSV_PATH_CANDIDATES)

POOLS = build_option_pools(OPTIONS_DF) if OPTIONS_DF is not None and not OPTIONS_DF.empty else {
    "days": ["Monday","Tuesday","Wednesday","Thursday","Friday","Monday & Wednesday","Tuesday & Thursday"],
    "time_of_day": ["morning","afternoon","evening"],
    "number_of_classes": [1,2,3,4],
    "course_level_minimum": [100,200,300,400],
}


def get_options_for_prompt(num: int) -> list[str]:
    """
    Get [Base, Perturb1, Perturb2] for a given prompt num from OPTIONS_DF.
    Uses OPTIONS_INDEX for precise row mapping; falls back to first row.
    """
    # cols = ["V1_Base_Text", "V1_Perturb1_Text", "V1_Perturb2_Text"]
    cols = ["V1_Base_Text"]
    if OPTIONS_DF is not None and not OPTIONS_DF.empty:
        if isinstance(num, str):
            try:
                num = int(num)
            except Exception:
                pass
        # lookup via robust index
        if isinstance(num, (int, float)) and int(num) in OPTIONS_INDEX:
            r = OPTIONS_DF.iloc[OPTIONS_INDEX[int(num)]]
        else:
            # fallback to first row
            r = OPTIONS_DF.iloc[0]
        out = []
        for c in cols:
            if c in OPTIONS_DF.columns and pd.notna(r[c]):
                s = str(r[c]).strip()
                if s:
                    out.append(s)
        if out:
            return out
    # ultimate fallback
    return ["(missing) V1_Base_Text", "(missing) V1_Perturb1_Text", "(missing) V1_Perturb2_Text"]

# ---------- FILL-IN (CSV-driven) ----------
import ast

FILLIN_PARAMS = ["days", "time_of_day", "number_of_classes", "course_level_minimum"]


def trigger_scroll_top():
    st.session_state["scroll_to_top"] = True



def _canon_tod(val):
    if val is None: return None
    s = str(val).strip().lower()
    return s if s and s != "none" else None

def _canon_int(val):
    if val is None: return None
    try:
        s = str(val).strip().lower()
        if not s or s == "none": return None
        return int(s)
    except Exception:
        return None

def get_truth_params_for_prompt(num: int) -> dict:
    """Read truth params from the central CSV row for this prompt (using your OPTIONS_DF/INDEX)."""
    truth = {k: None for k in FILLIN_PARAMS}
    if OPTIONS_DF is None or OPTIONS_DF.empty:
        return truth

    row = OPTIONS_DF.iloc[OPTIONS_INDEX.get(int(num), 0)] if isinstance(num, (int, float)) else OPTIONS_DF.iloc[0]
    cols = {c.lower(): c for c in OPTIONS_DF.columns}

    # days / preferred_days variants
    for name in ("days","preferred_days","v1_days","v1_preferred_days"):
        if name in cols:
            d = _parse_listish(row[cols[name]])
            truth["days"] = d if d not in ("", "none", None, []) else None
            break

    # time_of_day variants
    for name in ("time_of_day","timewindow","time_window","v1_time_of_day"):
        if name in cols:
            truth["time_of_day"] = _canon_tod(row[cols[name]])
            break

    # number_of_classes / max_classes_per_day variants
    for name in ("number_of_classes","max_classes_per_day"):
        if name in cols:
            truth["number_of_classes"] = _canon_int(row[cols[name]])
            break

    # course_level_minimum / min_level variants
    for name in ("course_level_minimum","min_level"):
        if name in cols:
            truth["course_level_minimum"] = _canon_int(row[cols[name]])
            break

    return truth

# --- Template-driven helpers ---

ALLOWED_PARAMS = {"days", "time_of_day", "number_of_classes", "course_level_minimum"}

# def get_template_for_prompt(num: int) -> str:
#     """Return a template string for the prompt row. Prefers 'template'/'v1_template', falls back to 'v1_base_text'."""
#     if OPTIONS_DF is None or OPTIONS_DF.empty:
#         return ""
#     row = OPTIONS_DF.iloc[OPTIONS_INDEX.get(int(num), 0)]
#     for c in ("template", "v1_template", "v1_base_text"):
#         if c in OPTIONS_DF.columns and pd.notna(row[c]):
#             return str(row[c]).strip()
#     return ""

def get_template_for_prompt(num: int, template_hint: str | None = None) -> str:
    """
    Prefer 'template'/'v1_template'. If those are absent, return a safe default.
    Uses _row_for_num_or_template so we don't rely solely on numeric keys.
    """
    if OPTIONS_DF is None or OPTIONS_DF.empty:
        return "Please create a schedule on {days} in the {time_of_day}, with {number_of_classes} classes per day, at or above the {course_level_minimum} level."

    row = _row_for_num_or_template(num, template_hint)
    for c in ("template", "v1_template"):
        if c in OPTIONS_DF.columns and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()

    # If no explicit template column, allow fallback to base text (still contains the values)
    for c in ("v1_base_text", "V1_Base_Text"):
        if c in OPTIONS_DF.columns and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()

    return "Please create a schedule on {days} in the {time_of_day}, with {number_of_classes} classes per day, at or above the {course_level_minimum} level."


def extract_placeholders(tmpl: str) -> list[str]:
    """Return placeholders like {days} that match ALLOWED_PARAMS."""
    if not tmpl:
        return []
    names = re.findall(r"\{([A-Za-z0-9_]+)\}", tmpl)
    return [n for n in names if n in ALLOWED_PARAMS]



# --- Template-driven helpers (add these) ---
# --- Fill-in template (tokenized) helpers: keep a single copy only ---
ALLOWED_PARAMS = {"days", "time_of_day", "number_of_classes", "course_level_minimum"}

def get_template_for_prompt(num: int) -> str:
    """
    Return a template string for the prompt row.
    Prefers 'template'/'v1_template', otherwise falls back to a sensible default
    that includes placeholders so dropdowns always render.
    """
    if OPTIONS_DF is not None and not OPTIONS_DF.empty:
        row = OPTIONS_DF.iloc[OPTIONS_INDEX.get(int(num), 0)]
        for c in ("template", "v1_template"):
            if c in OPTIONS_DF.columns and pd.notna(row[c]) and str(row[c]).strip():
                return str(row[c]).strip()

    # Fallback ensures placeholders exist
    return "Please create a schedule on {days} in the {time_of_day}, with {number_of_classes} classes per day, at or above the {course_level_minimum} level."

def _tokenize_template(tmpl: str):
    """Yield ('text', str) and ('param', name) tokens from a {param}-style template."""
    if not tmpl:
        return [("text", "")]
    out = []
    i = 0
    for m in re.finditer(r"\{([A-Za-z0-9_]+)\}", tmpl):
        if m.start() > i:
            out.append(("text", tmpl[i:m.start()]))
        out.append(("param", m.group(1)))
        i = m.end()
    if i < len(tmpl):
        out.append(("text", tmpl[i:]))
    return out

def _chunk_tokens_for_row(tokens, max_units=12):
    """
    Simple layout: each token consumes 'units' of width; wrap to next row if needed.
    """
    def units(tok):
        kind, val = tok
        if kind == "param":
            return 4
        n = max(1, len(val)//14 + 1)
        return min(8, n)

    rows, cur, used = [], [], 0
    for t in tokens:
        u = units(t)
        if used + u > max_units and cur:
            rows.append(cur)
            cur, used = [], 0
        cur.append((t, u))
        used += u
    if cur:
        rows.append(cur)
    return rows





# # ===== Ground-truth extraction from the EXACT variant text shown =====
# import re

# def _ref_col_lower_for(rep_num: int, expl_num: int) -> str:
#     """
#     Map representation_num & explanation_num to the correct text column in OPTIONS_DF.
#     Assumes rep: 1->V1, 2->V2 ; expl: 1->Base, 2->Perturb1, 3->Perturb2
#     and that columns are lowercased: v1_base_text, v1_perturb1_text, ...
#     """
#     v = "v1" if int(rep_num) == 1 else "v2"
#     if int(expl_num) == 1:
#         return f"{v}_base_text"
#     elif int(expl_num) == 2:
#         return f"{v}_perturb1_text"
#     else:
#         return f"{v}_perturb2_text"

# # --- Parsers on fully-instantiated prompt text ---
# _DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
# _time_pat  = re.compile(r"\b(morning|afternoon|evening)\b", re.I)
# _level_pat = re.compile(r"(\d{3})-level", re.I)
# _max_pat   = re.compile(r"(?:at\s+most|max)\s+(\d+)\s+(?:classes|class(?:es)?\s+per\s+day|per\s+day)", re.I)

# def _extract_days(text: str) -> str:
#     t = text or ""
#     hits = []
#     for d in _DAYS:
#         for m in re.finditer(d, t, flags=re.I):
#             hits.append((m.start(), d))
#     if not hits:
#         return ""
#     ordered, seen = [], set()
#     for _, d in sorted(hits):  # left-to-right
#         if d not in seen:
#             seen.add(d)
#             ordered.append(d)
#     return ordered[0] if len(ordered) == 1 else (", ".join(ordered[:-1]) + " and " + ordered[-1])

# def _extract_time_of_day(text: str) -> str:
#     m = _time_pat.search(text or "")
#     return (m.group(1) or "").lower() if m else ""

# def _extract_max_classes(text: str) -> str:
#     m = _max_pat.search(text or "")
#     return m.group(1) if m else ""

# def _extract_min_level(text: str) -> str:
#     m = _level_pat.search(text or "")
#     return m.group(1) if m else ""







# ---- Helpers to grab a row & column safely (columns are lowercased in OPTIONS_DF) ----
import re
import pandas as pd

def _norm_space(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _row_for_num_or_template(num: int | None, template_hint: str | None):
    """
    1) Try numeric lookup via OPTIONS_INDEX.
    2) If that fails, match by Template text (case/space-insensitive).
    3) Fallback to the first row.
    """
    if OPTIONS_DF is None or OPTIONS_DF.empty:
        return None

    # 1) numeric key
    try:
        n = int(num) if num is not None else None
    except Exception:
        n = None
    if n in OPTIONS_INDEX:
        return OPTIONS_DF.iloc[OPTIONS_INDEX[n]]

    # 2) template text match
    if template_hint:
        t_norm = _norm_space(template_hint)
        cols = {c.lower(): c for c in OPTIONS_DF.columns}
        for name in ("template", "v1_template"):
            if name in cols:
                col = cols[name]
                for i in range(len(OPTIONS_DF)):
                    cand = _norm_space(str(OPTIONS_DF.iloc[i].get(col, "")))
                    if cand == t_norm:
                        return OPTIONS_DF.iloc[i]

    # 3) fallback
    return OPTIONS_DF.iloc[0]

def lookup_truth_prompt_text(num: int, template_hint: str | None = None):
    """
    Return (truth_prompt_text, truth_prompt_col).
    Prefer V1_Base_Text (or v1_base_text). If empty, try a few fallbacks.
    """
    row = _row_for_num_or_template(num, template_hint)
    if row is None:
        return "", ""

    cols = {c.lower(): c for c in OPTIONS_DF.columns}
    # strict preference: V1 base text (this is your “ground truth” column)
    for lc in ("v1_base_text",):
        if lc in cols:
            txt = str(row.get(cols[lc], "") or "").strip()
            if txt:
                return txt, cols[lc]

    # gentle fallbacks (only if you really need them)
    for lc in ("v2_base_text", "v1_perturb1_text", "v1_perturb2_text"):
        if lc in cols:
            txt = str(row.get(cols[lc], "") or "").strip()
            if txt:
                return txt, cols[lc]

    return "", ""


# --- put near your other helpers ---
_WORD_NUMS = {
    "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10
}

def _as_int(tok: str | None):
    if not tok:
        return None
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return _WORD_NUMS.get(tok)

# Strong, explicit patterns first (covers “at most”, “maximum of”, etc.)
_MAX_PATS = [
    # at most / maximum of / no more than / up to / not exceeding ... classes
    re.compile(r"\b(?:at\s+most|max(?:imum)?\s*(?:of)?|no\s+more\s+than|up\s+to|not\s+exceed(?:ing)?|no\s+greater\s+than)\s+(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+classes?\b", re.I),

    # plain “2 classes per day”
    re.compile(r"\b(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+classes?\s+(?:per|each)\s+day\b", re.I),

    # “classes per day = 2”, “classes per day: 2”
    re.compile(r"\bclasses?\s+(?:per|each)\s+day\s*(?:=|:)\s*(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I),

    # shorthand “2 classes/day”
    re.compile(r"\b(?P<n>\d+)\s+classes?\s*/\s*day\b", re.I),
]

# Strictly extract max classes as a single numeral 1|2|3, tied to "class(es)"
_MAX_CLASSES_PATS = [
    re.compile(r"\b([123])\s+classes?\b(?:\s+per\s+day\b)?", re.I),             # "2 classes [per day]"
    re.compile(r"\bclasses?\b(?:\s+per\s+day\b)?\s*(?:=|:)?\s*([123])\b", re.I) # "classes per day: 2" / "classes = 2"
]

def _extract_max_classes(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"(?<!\S)([123])(?!\S)", text)  # start/space, 1|2|3, end/space
    return int(m.group(1)) if m else None



def _row_for_num(num: int):
    """Return the row Series for this prompt number using OPTIONS_INDEX; fallback to first row."""
    if OPTIONS_DF is None or OPTIONS_DF.empty:
        return None
    idx = OPTIONS_INDEX.get(int(num), None)
    return OPTIONS_DF.iloc[idx if idx is not None else 0]

def _get_col(row, names: list[str], default: str = "") -> str:
    """Fetch first available column from `names` (case-insensitive)."""
    if row is None:
        return default
    cols = {c.lower(): c for c in OPTIONS_DF.columns}
    for n in names:
        ln = n.lower()
        if ln in cols:
            val = row.get(cols[ln], default)
            return "" if pd.isna(val) else str(val)
    return default

# ---- Use the Template to detect which placeholders matter for THIS prompt ----
_DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# time: allow singular/plural variants
_time_pat  = re.compile(r"\b(morning|mornings|afternoon|afternoons|evening|evenings)\b", re.I)

# level: "300-level", "300 level", "level 300"
_level_pats = [
    re.compile(r"\b(\d{3})[- ]?level\b", re.I),
    re.compile(r"\blevel\s*(\d{3})\b", re.I),
]

# classes/day:
#   - "at most 2 classes per day", "maximum of 2 classes", "no more than 2 classes"
#   - plus a plain fallback: "2 classes per day"
_max_pats = [
    re.compile(r"\b(?:at\s+most|max(?:imum)?\s+of|no\s+more\s+than|up\s+to)\s+(\d+)\s+classes?\b", re.I),
    re.compile(r"\b(\d+)\s+classes?\s+per\s+day\b", re.I),
]

def _extract_days(text: str) -> list[str] | None:
    t = text or ""
    hits = []
    for d in _DAYS:
        for m in re.finditer(rf"\b{re.escape(d)}\b", t, flags=re.I):
            hits.append((m.start(), d))
    if not hits:
        return None
    ordered, seen = [], set()
    for _, d in sorted(hits):
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered or None

def _extract_time_of_day(text: str) -> str | None:
    m = _time_pat.search(text or "")
    if not m: 
        return None
    base = m.group(1).lower()
    # normalize plurals
    return base[:-1] if base.endswith("s") else base

def _extract_min_level(text: str) -> int | None:
    t = text or ""
    for pat in _level_pats:
        m = pat.search(t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

# def _extract_max_classes(text: str) -> int | None:
#     t = text or ""
#     for pat in _max_pats:
#         m = pat.search(t)
#         if m:
#             try:
#                 return int(m.group(1))
#             except Exception:
#                 pass
#     return None


# ---- The new canonical truth getter (uses Template + V1_Base_Text only) ----
def get_truth_params_for_prompt(num: int, template_hint: str | None = None) -> dict:
    """
    Build truths using:
      - which placeholders are present in Template (template or v1_template)
      - values parsed from V1_Base_Text (the canonical ground-truth instantiation)
    Uses template-based row fallback if numeric key is missing/misaligned.
    """
    row = _row_for_num_or_template(num, template_hint)
    if row is None:
        return {k: None for k in ("days","time_of_day","number_of_classes","course_level_minimum")}

    # which placeholders matter?
    tmpl = ""
    for c in ("template", "v1_template"):
        if c in OPTIONS_DF.columns and pd.notna(row[c]) and str(row[c]).strip():
            tmpl = str(row[c]).strip()
            break
    present = set(re.findall(r"\{([A-Za-z0-9_]+)\}", tmpl or ""))

    # ground-truth text to parse
    gt = ""
    for c in ("v1_base_text", "V1_Base_Text"):
        if c in OPTIONS_DF.columns and pd.notna(row[c]) and str(row[c]).strip():
            gt = str(row[c]).strip()
            break

    days_list   = _extract_days(gt)
    time_of_day = _extract_time_of_day(gt)
    max_classes = _extract_max_classes(gt)
    min_level   = _extract_min_level(gt)

    return {
        "days":                 (days_list   if "days" in present else None),
        "time_of_day":          (time_of_day if "time_of_day" in present else None),
        "number_of_classes":    (max_classes if "number_of_classes" in present else None),
        "course_level_minimum": (min_level   if "course_level_minimum" in present else None),
    }





def _truth_from_text(text: str) -> dict:
    return {
        "days_truth":                 _extract_days(text),
        "time_of_day_truth":          _extract_time_of_day(text),
        "number_of_classes_truth":    _extract_max_classes(text),
        "course_level_minimum_truth": _extract_min_level(text),
    }

def truth_from_reference(mapped_prompt_num: int, rep_num: int, expl_num: int) -> dict:
    """
    Look up the EXACT variant text from OPTIONS_DF (already loaded & lowercased),
    parse the 4 truth fields, and also return the raw prompt text used.
    """
    try:
        if OPTIONS_DF is None or OPTIONS_DF.empty:
            return {
                "days_truth": "", "time_of_day_truth": "",
                "number_of_classes_truth": "", "course_level_minimum_truth": "",
                "truth_prompt_text": "", "truth_prompt_col": ""
            }

        idx = OPTIONS_INDEX.get(int(mapped_prompt_num))
        row = OPTIONS_DF.iloc[idx] if idx is not None else OPTIONS_DF.iloc[0]

        # col = _ref_col_lower_for(rep_num, expl_num)
        col = "V1_Base_Text"

        # Pull text from lower-case column if available; fallback to TitleCase header
        if col in OPTIONS_DF.columns:
            raw_text = str(row.get(col, "") or "")
            used_col = col
        else:
            fallback = col.upper().replace("_", " ").title().replace(" ", "_")
            raw_text = str(row.get(fallback, "") or "")
            used_col = fallback

        truths = _truth_from_text(raw_text)

        # Optional: sanitize for CSV (single-line)
        safe_text = raw_text.replace("\r", " ").replace("\n", " ").strip()

        return {
            **truths,
            "truth_prompt_text": safe_text,
            "truth_prompt_col": used_col,
        }
    except Exception:
        return {
            "days_truth": "", "time_of_day_truth": "",
            "number_of_classes_truth": "", "course_level_minimum_truth": "",
            "truth_prompt_text": "", "truth_prompt_col": ""
        }







def render_fillin_template_ui(mapped_num_val: int, key: str, pools: dict, template_text: str, key_ns: str = ""):
    """
    PARAMS-ONLY MODE:
      - Ignore template_text entirely (no original prompt text).
      - Always render 4 dropdowns: days, time_of_day, number_of_classes, course_level_minimum
      - Each dropdown includes 'Any' and 'All' options.
    Returns (truth, shown_params_list, complete_flag, attempted_sentence).
    """
    # truth = get_truth_params_for_prompt(mapped_num_val)
    template_text = get_template_for_prompt(mapped_num_val)   # or however you already computed it
    truth = get_truth_params_for_prompt(mapped_num_val, template_text)


    # init state bucket
    if "fill_answers" not in st.session_state:
        st.session_state.fill_answers = {}
    if key not in st.session_state.fill_answers:
        st.session_state.fill_answers[key] = {}
    answers = st.session_state.fill_answers[key]

    # Render the 4 dropdowns in a 2x2 grid
    PARAM_ORDER = ["days", "time_of_day", "number_of_classes", "course_level_minimum"]
    labels = {
        "days": "Days",
        "time_of_day": "Time of day",
        "number_of_classes": "Max classes/day",
        "course_level_minimum": "Min course level",
    }

    # Build option lists with Any/All
    opts_map = {
        "days": _with_any_all(pools.get("days", [])),
        "time_of_day": _with_any_all(pools.get("time_of_day", [])),
        "number_of_classes": _with_any_all(pools.get("number_of_classes", [])),
        "course_level_minimum": _with_any_all(pools.get("course_level_minimum", [])),
    }

    cols = st.columns(2)
    for i, pname in enumerate(PARAM_ORDER):
        label = labels[pname]
        opts = opts_map[pname]

        # Prefer previously chosen answer; else default to truth if present; else "Any"
        current = answers.get(pname, None)
        if current is None:
            # Normalize truth to a string display if needed
            # base = truth.get(pname)
            # current = base if base not in (None, "") else "Any"
            # inside render_fillin_template_ui:
            base = truth.get(pname)
            if pname == "days" and base not in (None, "", []):
                current = _display_days(base)
            elif base not in (None, ""):
                current = base
            else:
                current = "Any"

        widget_key = f"{key_ns}_{key}_{pname}"
        with cols[i % 2]:
            sel = st.selectbox(
                label,
                options=opts,
                index=(opts.index(current) if current in opts else 0),
                key=widget_key,
            )
        answers[pname] = sel

    # We "showed" all 4 params in this mode
    shown = PARAM_ORDER[:]

    # For bookkeeping, create a compact attempted summary (no original prompt text)
    attempted_sentence = "; ".join(
        f"{p}={answers.get(p, '')}" for p in PARAM_ORDER
    )

    # Completion is trivially True (all four dropdowns always have a selection)
    complete = True

    # Persist attempted sentence so go_next() can save it
    st.session_state._attempted_sentence = attempted_sentence

    return truth, shown, complete, attempted_sentence






def trigger_scroll_to_top():
    """Call this right after handling Next/Prev to force a scroll-to-top on the next render."""
    st.session_state["_scroll_token"] = str(uuid4())

def render_scroll_to_top_shim():
    token = st.session_state.get("_scroll_token", "")
    components.html(
        f"""
        <div style="height:0;overflow:hidden;margin:0;padding:0">
          <script>
            try {{
              window.scrollTo({{top:0,left:0,behavior:'auto'}});
              if (window.parent && window.parent !== window) {{
                window.parent.scrollTo({{top:0,left:0,behavior:'auto'}});
              }}
            }} catch (e) {{}}
          </script>
          <!-- token:{token} -->
        </div>
        """,
        height=1,  # stays invisible; supported by components.html
    )



def fillins_csv_path() -> str:
    pid = st.session_state.participant_num or "unknown"
    return os.path.join(CSV_DIR, f"participant_{pid}_fillins.csv")

def _days_to_set(x):
    if x is None:
        return set()
    if isinstance(x, list):
        parts = x
    else:
        parts = [p.strip() for p in re.split(r"[,&]", str(x)) if p.strip()]
    return set(p.capitalize() for p in parts if p)

def _compare_days(user_days, truth_days) -> bool:
    # order-insensitive comparison, tolerant of commas/&
    return _days_to_set(user_days) == _days_to_set(truth_days)





def save_current_fillin(truth: dict, shown: list[str], key: str, mapped_num_val: int,
                        attempted_sentence: str = "", resolved_sentence: str = ""):
    answers = st.session_state.fill_answers.get(key, {})

    def same_days(ans, tru):
        if not ans or tru is None: return False
        def to_set_days(x):
            if isinstance(x, list): L = x
            else: L = [p.strip() for p in re.split(r"[,&]", str(x)) if p.strip()]
            return set(d.strip().capitalize() for d in L if d.strip())
        return to_set_days(ans) == to_set_days(truth["days"])

    checks = {}
    if "days" in shown:
        checks["days_correct"] = same_days(answers.get("days"), truth.get("days"))
    if "time_of_day" in shown:
        checks["time_of_day_correct"] = str(answers.get("time_of_day","")).strip().lower() == (truth.get("time_of_day") or "")
    if "number_of_classes" in shown:
        try:
            checks["number_of_classes_correct"] = int(str(answers.get("number_of_classes")).strip()) == (truth.get("number_of_classes") or -999)
        except Exception:
            checks["number_of_classes_correct"] = False
    if "course_level_minimum" in shown:
        try:
            checks["course_level_minimum_correct"] = int(str(answers.get("course_level_minimum")).strip()) == (truth.get("course_level_minimum") or -999)
        except Exception:
            checks["course_level_minimum_correct"] = False

    score = sum(1 for v in checks.values() if v)
    correct_all = (score == len(checks))

    series    = st.session_state.series
    trial_idx = current_trial_idx()
    rep_type  = ORDERINGS[series]["rep"][trial_idx - 1]
    expl_type = ORDERINGS[series]["expl"][trial_idx - 1]
    num       = ORDERINGS[series]["num"][trial_idx - 1]

    header = [
        "participant_num","order_num",
        "prompt_num","mapped_prompt_num",
        "representation_num","explanation_num",
        "days_sel","time_of_day_sel","number_of_classes_sel","course_level_minimum_sel",
        "days_truth","time_of_day_truth","number_of_classes_truth","course_level_minimum_truth",
        "days_correct","time_of_day_correct","number_of_classes_correct","course_level_minimum_correct",
        "score","correct_all",
        "attempted_sentence","resolved_sentence"   # <-- new columns at end
    ]

    row = [
        st.session_state.participant_num, series,
        num, mapped_num_val,
        rep_type, expl_type,
        answers.get("days"), answers.get("time_of_day"), answers.get("number_of_classes"), answers.get("course_level_minimum"),
        _display_days(truth.get("days")), truth.get("time_of_day"), truth.get("number_of_classes"), truth.get("course_level_minimum"),
        checks.get("days_correct"), checks.get("time_of_day_correct"), checks.get("number_of_classes_correct"), checks.get("course_level_minimum_correct"),
        score, correct_all,
        attempted_sentence, resolved_sentence
    ]
    append_csv(fillins_csv_path(), header, row)
    st.toast("✅ Answers saved", icon="✅")





# ======================== SURVEY ========================
SURVEY_ITEMS = [
    "From the explanation, I know how the Large Language Model (LLM) works.",
    "This explanation of how the Large Language Model (LLM) works is satisfying.",
    "This explanation of how the Large Language Model (LLM) works has sufficient detail.",
    "This explanation of how the Large Language Model (LLM) works seems complete.",
    "This explanation of how the Large Language Model (LLM) works tells me how to use it.",
    "This explanation of how the Large Language Model (LLM) works is useful to my goals.",
    "This explanation of the Large Language Model (LLM) shows me how accurate the Large Language Model (LLM) is.",
]
LIKERT = ["1","2","3","4","5","6","7"]

# ======================== SESSION DEFAULTS ========================
# --- top of script: clear the flag each run if we used it last time
if st.session_state.get("_scroll_to_top_done"):
    st.session_state["_scroll_to_top_done"] = False
if "started" not in st.session_state: st.session_state.started = False
if "participant_num" not in st.session_state: st.session_state.participant_num = ""
if "series" not in st.session_state: st.session_state.series = 1
if "version" not in st.session_state: st.session_state.version = 1
if "trial_idx" not in st.session_state: st.session_state.trial_idx = 1
# substep: 1 (main), 2 (follow-up), 3 (survey if end-of-block)
if "substep" not in st.session_state: st.session_state.substep = 1
# Unique counter for one-time JS components
if "_scroll_key" not in st.session_state: st.session_state["_scroll_key"] = 0

# ==== App stage gate ====
# Stages: 'tutorial' -> 'setup' -> 'run'
if "app_stage" not in st.session_state:
    st.session_state.app_stage = "tutorial"

if "setup_done" not in st.session_state:
    st.session_state.setup_done = False

def _go_to_setup():
    st.session_state.app_stage = "setup"
    st.rerun()

def _go_to_run():
    st.session_state.app_stage = "run"
    st.session_state.setup_done = True
    st.rerun()


# --- NAV STATE (canonical) ---
if "nav_trial_idx" not in st.session_state:
    # seed from any existing widget, or default
    # if "trial_idx_widget" in st.session_state:
    #     st.session_state.nav_trial_idx = int(st.session_state.trial_idx_widget)
    # else:
    st.session_state.nav_trial_idx = 1

if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

if "fillin_timer_start" not in st.session_state:
    st.session_state.fillin_timer_start = {}  # key -> start timestamp (float)

if "_scroll_to_top" not in st.session_state:
    st.session_state["_scroll_to_top"] = False
if "_scroll_token" not in st.session_state:
    st.session_state["_scroll_token"] = ""

# --- Confidence (per-trial) state ---
if "confidence_answers" not in st.session_state:   # key -> int (1..5)
    st.session_state.confidence_answers = {}
if "confidence_order" not in st.session_state:     # save order
    st.session_state.confidence_order = []

if st.session_state.scroll_to_top:
    scroll_to_here(0, key='top')  # Scroll to the top of the page
    st.session_state.scroll_to_top = False  # Reset the state after scrolling

def scroll():
    st.session_state.scroll_to_top = True

CONF_SUBSTEP = 2  # substep 1 = fill-in page, substep 2 = confidence page (new)


# Ranking state per (series, trial)
if "rank_orders" not in st.session_state: st.session_state.rank_orders = {}  # randomized options
if "rank_answers" not in st.session_state: st.session_state.rank_answers = {} # chosen order
if "rank_canon"  not in st.session_state: st.session_state.rank_canon  = {}  # [Base, Perturb1, Perturb2] for scoring



# put this with your other session defaults
if "saved_fillins" not in st.session_state:
    st.session_state.saved_fillins = set()   # e.g., {"1-3", "2-7"}

if "saved_rankings" not in st.session_state:
    st.session_state.saved_rankings = set()



# Survey answers keyed by (series, block#)
if "survey_answers" not in st.session_state: st.session_state.survey_answers = {}

def key_for(series: int, trial_idx: int) -> str:
    return f"{series}-{trial_idx}"

def block_num(trial_idx: int) -> int:
    return math.ceil(trial_idx / 3)

def is_block_end(trial_idx: int) -> bool:
    return (trial_idx % 3) == 0

# def substeps_max_for(trial_idx: int) -> int:
#     return 3 if is_block_end(trial_idx) else 2

# def substeps_max_for(trial_idx: int) -> int:
#     # 1 page for normal trials, 2 pages when a block survey is due
#     return 2 if is_block_end(trial_idx) else 1

# Put these near your other “constants”
CONF_SUBSTEP  = 2          # confidence page is always the 2nd page
SURVEY_SUBSTEP = 3         # block-survey page only at block ends

def substeps_max_for(trial_idx: int) -> int:
    """
    2 pages for normal trials:
      1 = fill-in, 2 = confidence
    3 pages at block end:
      1 = fill-in, 2 = confidence, 3 = block survey
    """
    return 3 if is_block_end(trial_idx) else 2


def current_trial_idx() -> int:
    # Prefer the nav key; fall back to legacy key if you haven’t migrated everything yet
    return int(st.session_state.get("nav_trial_idx", st.session_state.get("trial_idx", 1)))

def current_substep() -> int:
    return int(st.session_state.get("nav_substep", st.session_state.get("substep", 1)))


# ======================== OUTPUT CSV HELPERS ========================
def rankings_csv_path() -> str:
    pid = st.session_state.participant_num or "unknown"
    return os.path.join(CSV_DIR, f"participant_{pid}_rankings.csv")

def survey_csv_path() -> str:
    pid = st.session_state.participant_num or "unknown"
    return os.path.join(CSV_DIR, f"participant_{pid}_survey_responses.csv")

def append_csv(path: str, header: list[str], row: list):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def confidence_csv_path() -> str:
    pid = st.session_state.participant_num or "unknown"
    return os.path.join(CSV_DIR, f"participant_{pid}_confidence.csv")


# ======================== SAVE LOGIC ========================

def save_current_ranking():
    """Save ranking row for the current (series, trial) if complete."""
    series    = st.session_state.series
    trial_idx = current_trial_idx()
    key       = key_for(series, trial_idx)

    # Read the user's ranking answers
    answers = st.session_state.rank_answers.get(key)

    # Bail out unless we have 3 non-empty choices
    if not answers or len(answers) < 3:
        return
    first_raw, second_raw, third_raw = answers[0], answers[1], answers[2]
    if any(a in (None, "", "None") for a in (first_raw, second_raw, third_raw)):
        return

    # Resolve trial metadata
    num       = ORDERINGS[series]["num"][trial_idx - 1]
    rep_type  = ORDERINGS[series]["rep"][trial_idx - 1]
    expl_type = ORDERINGS[series]["expl"][trial_idx - 1]
    mapped_num_val = mapped_number(num) or num

    # Canonical options for THIS trial (from mapped prompt row)
    canon_raw = st.session_state.rank_canon.get(key) or get_options_for_prompt(mapped_num_val)
    canon = [_norm(x) for x in canon_raw]

    # Score against canonical order
    first, second, third = _norm(first_raw), _norm(second_raw), _norm(third_raw)
    score = 0
    score += 1 if len(canon) > 0 and first  == canon[0] else 0
    score += 1 if len(canon) > 1 and second == canon[1] else 0
    score += 1 if len(canon) > 2 and third  == canon[2] else 0
    correct = (score == 3)

    header = [
        "participant_num","order_num",
        "prompt_num","mapped_prompt_num",
        "representation_num","explanation_num",
        "first_rank","second_rank","third_rank",
        "score","correct"
    ]
    row = [
        st.session_state.participant_num, series,
        num, mapped_num_val, rep_type, expl_type,
        first_raw, second_raw, third_raw,
        score, correct
    ]
    append_csv(rankings_csv_path(), header, row)
    st.toast("✅ Ranking saved", icon="✅")
    # mark as saved to avoid double-writes
    st.session_state.saved_rankings.add(key)



def render_tutorial():
    st.title("Quick Tutorial")
    st.markdown("""
**What you’ll do**
1. You’ll see prompts and its corresponding output as either a text timetable, image timetable, or AI-Generated timetable.
2. Afterwards, you'll see a visual of an model's output, and from it you'll have to infer what the inputs were to produce it. 
3. You will have to select the parameters via dropdowns. The parameters are time of day, max classes per day, days to have classes, and minimum course level.
4. When done, you will click **Next** to save and advance. A timer records how long you took per trial. In total, there are 18 trials.
5. After every three trials, you’ll answer a confidence question and a short survey.

**Representation Types**
1. Text Timetable: A text list of classes and their details.
2. Image Timetable: A timetable image with the classes arranged to show classes.
3. AI-Generarted Timetable: A "timetable" image generated by an AI Model (GPT's Dall-e 3) to represent classes.

**Explanation Types** 
1. Text Explanation: Simple text explanation given by the model as to how it arrived at the schedule.
2. Schedule Card: Card showing parameters of relevant parameters for the given schedule and showing how things are satisfied.
3. Counterfactual: Given a prompt and a specific output type (one of the three mentioned above), an alternate version of the input prompt and its output are also shown to demonstrate how prompt changes affect output.
                                
**Tips**
- You can use **Prev** to go back one step as necessary.
- Your selections overwrite your previous answers each time (no duplicates).
    """)

    with st.expander("See example visualizations you will encounter:"):
#         st.markdown("""
# - **Days**: Tuesday and Thursday  
# - **Time of Day**: morning  
# - **Max classes per day**: 2  
# - **Minimum course level**: 300  
#         """)
        st.markdown("### Prompt Example:")
        st.image(os.path.join(STATIC_DIR, "prompt_p_1_v1_base.png"), width=400)

        # st.markdown("### Corresponding Text Timetable:")
        # st.image(os.path.join(STATIC_DIR, "text_timetable_1_v1.png"), width=1200)

        # st.markdown("### Corresponding Image Timetable:")
        # st.image(os.path.join(STATIC_DIR, "it_p_1_v_1.png"), width=400)

        # st.markdown("### Corresponding AI_Generated Timetable:")
        # st.image(os.path.join(STATIC_DIR, "diffusion_model_1_v1.png"), width=400)

        c1, c2, c3 = st.columns(3)  # or st.columns([1,1,1], gap="large")

        with c1:
            st.markdown("### Corresponding Text Timetable:")
            st.image(os.path.join(STATIC_DIR, "text_timetable_1_v1.png"), use_container_width=True)

        with c2:
            st.markdown("### Corresponding Image Timetable:")
            st.image(os.path.join(STATIC_DIR, "it_p_1_v_1.png"), use_container_width=True)

        with c3:
            st.markdown("### Corresponding AI_Generated Timetable:")
            st.image(os.path.join(STATIC_DIR, "diffusion_model_1_v1.png"), use_container_width=True)

        st.markdown("### Corresponding Text Explanation:")
        st.image(os.path.join(STATIC_DIR, "explanation_001.png"), width=1200)

        # st.markdown("### Corresponding Text Explanation:")
        # st.image(os.path.join(STATIC_DIR, "explanation_001.png"), width=1200)

        st.markdown("### Corresponding Schedule Card Explanation:")
        st.image(os.path.join(STATIC_DIR, "schedule_card_1.png"), width=1200)

        st.markdown("### Corresponding Counterfactual Explanation:")
        for v in [1, 2]:
            if v == 1:
                st.markdown(f"Original Prompt and Output")
            else:
                st.markdown(f"Modified Prompt and Output")
            c1, c2 = st.columns(2)
            with c1: show_image_or_warn(prompt_path(1, v), label="Prompt", custom_size=500)
            with c2: 
                show_image_or_warn(rep_path(1, 1, v), label="Output", custom_size=1000)
                # if rep_type == 3:
                #     show_image_or_warn(rep_path(1, 1, 1), custom_size=800)
                # elif rep_type == 1:
                # # show_image_or_warn(rep_path(rep_type, num, version), f"Rep v{version}")
                #     show_image_or_warn(rep_path(1, 1, 2), custom_size=1500)
                # else:
                #     show_image_or_warn(rep_path(1, 1, 2))

        st.markdown("### NOTE: Text timetable used to show the differences in output given changes to the original prompt's \"parameters\" for this counterfactual.")

    if st.button("I Understand, Continue", use_container_width=True, type="primary"):
            st.session_state["app_stage"] = "setup"
            st.rerun()

    # c1, c2 = st.columns([1,1])
    # with c1:
    #     if st.button("I Understand, Continue", use_container_width=True):
    #         st.session_state["app_stage"] = "setup"
    #         st.rerun()
    # with c2:
    #     st.button("Skip tutorial", key="btn_tutorial_skip", use_container_width=True, on_click=_go_to_setup)



def save_current_survey():
    """Save survey row for the current block (at end-of-block trials)."""
    series = st.session_state.series
    trial_idx = st.session_state.trial_idx
    blk = block_num(trial_idx)
    survey_key = f"{series}-block{blk}"
    answers = st.session_state.survey_answers.get(survey_key, [])
    if not answers or any(a is None for a in answers):
        return
    # Record explanation of the last trial in this block
    expl_type = ORDERINGS[series]["expl"][trial_idx - 1]

    header = ["participant_num","order_num","explanation_num","Q1","Q2","Q3","Q4","Q5","Q6","Q7"]
    row = [st.session_state.participant_num, series, expl_type] + answers[:7]
    append_csv(survey_csv_path(), header, row)
    st.toast("✅ Survey saved", icon="✅")

import pandas as pd
from pathlib import Path

# ---- persistent in-memory log for fill-ins (ordered) ----
if "fillins_log" not in st.session_state:    # key -> row dict
    st.session_state.fillins_log = {}
if "fillins_order" not in st.session_state:  # save order
    st.session_state.fillins_order = []

# --- Fill-in timers (per trial key) ---
if "fillin_timer_start" not in st.session_state:
    st.session_state.fillin_timer_start = {}  # key -> start timestamp (float)

# (You already have these; shown here for clarity)
# if "fillins_log" not in st.session_state: st.session_state.fillins_log = {}
# if "fillins_order" not in st.session_state: st.session_state.fillins_order = []

def _current_meta():
    """Return all meta for the current trial."""
    series    = st.session_state.series
    trial_idx = current_trial_idx()  # your helper
    num       = ORDERINGS[series]["num"][trial_idx - 1]
    mapped    = mapped_number(num) or num
    rep_type  = ORDERINGS[series]["rep"][trial_idx - 1]
    expl_type = ORDERINGS[series]["expl"][trial_idx - 1]
    key       = key_for(series, trial_idx)
    return series, trial_idx, num, mapped, rep_type, expl_type, key

def _fillins_csv_path():
    # If you already have fillins_csv_path(), use that instead:
    try:
        return fillins_csv_path()
    except NameError:
        csv_dir = Path(st.session_state.get("CSV_DIR", "CSV"))
        csv_dir.mkdir(parents=True, exist_ok=True)
        pid = st.session_state.get("participant_num", "unknown")
        return str(csv_dir / f"participant_{pid}_fillins.csv")

def _write_all_fillins_snapshot():
    """Overwrite fill-ins CSV from in-memory rows."""
    rows = [st.session_state.fillins_log[k]
            for k in st.session_state.fillins_order if k in st.session_state.fillins_log]

    cols = [
        "participant_num","order_num",
        "prompt_num","mapped_prompt_num",
        "representation_num","explanation_num",
        "days_sel","time_of_day_sel","number_of_classes_sel","course_level_minimum_sel",
        "attempted_sentence","resolved_sentence",
        "fillin_duration_sec",
        # truth fields
        "days_truth","time_of_day_truth","number_of_classes_truth","course_level_minimum_truth",
        # NEW: the actual prompt text we parsed truths from (and which column it came from)
        "truth_prompt_text", "truth_prompt_col",
    ]

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(_fillins_csv_path(), index=False)




def _write_all_confidence_snapshot():
    rows = []
    for key in st.session_state.confidence_order:
        if key not in st.session_state.confidence_answers:
            continue
        s_part, t_part = key.split("::", 1)
        series    = int(s_part)
        trial_idx = int(t_part)

        num       = ORDERINGS[series]["num"][trial_idx - 1]
        mapped    = mapped_number(num) or num
        rep_type  = ORDERINGS[series]["rep"][trial_idx - 1]
        expl_type = ORDERINGS[series]["expl"][trial_idx - 1]

        rows.append({
            "participant_num":    st.session_state.participant_num,
            "order_num":          series,
            "prompt_num":         num,
            "mapped_prompt_num":  mapped,
            "representation_num": rep_type,
            "explanation_num":    expl_type,
            "confidence_1to7":    st.session_state.confidence_answers[key],  # <- renamed
        })

    import pandas as pd
    pd.DataFrame(
        rows,
        columns=[
            "participant_num","order_num",
            "prompt_num","mapped_prompt_num",
            "representation_num","explanation_num",
            "confidence_1to7",   # <- renamed
        ],
    ).to_csv(confidence_csv_path(), index=False)



def confidence_key(series, trial_idx) -> str:
    return f"{int(series)}::{int(trial_idx)}"




def _write_all_survey_snapshot():
    """
    Overwrite participant_*_survey_responses.csv from st.session_state.survey_answers.
    Saves one row per block key in survey_order, including meta + all Q* columns.
    """
    rows = []
    all_qs = set()

    for skey in st.session_state.survey_order:
        answers = st.session_state.survey_answers.get(skey)
        if not answers:
            continue

        # Parse the key: "series::blockN"
        s_part, blk_part = skey.split("::", 1)
        series = int(s_part)
        try:
            block_idx = int(blk_part.replace("block", ""))
        except Exception:
            # fallback if format ever changes
            block_idx = _block_index_for(current_trial_idx())

        # Use the LAST trial in that block for meta columns
        trial_idx_in_block = block_idx * 3
        # Clamp to 18 (if your study has 18 trials)
        trial_idx_in_block = min(trial_idx_in_block, 18)

        num       = ORDERINGS[series]["num"][trial_idx_in_block - 1]
        mapped    = mapped_number(num) or num
        rep_type  = ORDERINGS[series]["rep"][trial_idx_in_block - 1]
        expl_type = ORDERINGS[series]["expl"][trial_idx_in_block - 1]

        # Track all question labels we’ve seen (for consistent columns)
        all_qs.update(answers.keys())

        rows.append({
            "participant_num":    st.session_state.participant_num,
            "order_num":          series,
            "block_index":        block_idx,
            "last_trial_in_block": trial_idx_in_block,
            "prompt_num":         num,
            "mapped_prompt_num":  mapped,
            "representation_num": rep_type,
            "explanation_num":    expl_type,
            **answers,  # merges Q* fields
        })

    # Stable column order: meta + sorted Qs
    all_qs = sorted(all_qs, key=lambda s: (len(s), s))  # Q1..Q10 sorts nicely
    cols = [
        "participant_num","order_num",
        "block_index","last_trial_in_block",
        "prompt_num","mapped_prompt_num",
        "representation_num","explanation_num",
        *all_qs,
    ]

    import pandas as pd
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(survey_csv_path(), index=False)

# ==== SURVEY (block-level) SNAPSHOT STATE ====
# One survey row per block (e.g., after trials 3, 6, 9, ...)

if "survey_answers" not in st.session_state:
    # key -> dict like {"Q1": 5, "Q2": 3, ...}
    st.session_state.survey_answers = {}
if "survey_order" not in st.session_state:
    # preserves save order (LIFO removal compatible)
    st.session_state.survey_order = []

# Scale and UI items (tweak as you wish)
SURVEY_SCALE_MIN = 1
SURVEY_SCALE_MAX = 7   # you asked for 1–7; change to 5 if needed

# If you already have question text, use that; otherwise generic labels:
SURVEY_ITEMS = [
    ("Q1", "The schedule was easy to understand."),
    ("Q2", "The interface felt responsive."),
    ("Q3", "My choices were clear."),
    ("Q4", "The instructions were clear."),
    ("Q5", "I felt in control."),
    ("Q6", "The workload was reasonable."),
    ("Q7", "Overall experience was positive."),
]

def survey_csv_path() -> str:
    # Use your existing CSV_DIR and participant_num
    pid = st.session_state.participant_num or "unknown"
    os.makedirs(CSV_DIR, exist_ok=True)
    return os.path.join(CSV_DIR, f"participant_{pid}_survey_responses.csv")

def _block_index_for(trial_idx: int, block_size: int = 3) -> int:
    return (trial_idx - 1) // block_size + 1

def survey_key(series: int, trial_idx: int) -> str:
    # normalize to ints to avoid KeyErrors later
    return f"{int(series)}::block{_block_index_for(int(trial_idx))}"


def go_prev():
    max_step = substeps_max_for(st.session_state.trial_idx)
    if st.session_state.substep > 1:
        st.session_state.substep -= 1
    else:
        if st.session_state.trial_idx > 1:
            st.session_state.trial_idx -= 1
            st.session_state.substep = substeps_max_for(st.session_state.trial_idx)

    st.session_state["_scroll_to_top"] = True
    st.session_state["_scroll_token"] = str(uuid4())  # unique key per click


def go_next():
    max_step = substeps_max_for(st.session_state.trial_idx)

    # Save the fill-in when leaving the combined page (substep 1)
    if st.session_state.substep == 1:
        if getattr(st.session_state, "_fillin_complete", False):
            series = st.session_state.series
            trial_idx = st.session_state.trial_idx
            num = ORDERINGS[series]["num"][trial_idx - 1]
            mapped_num_val = mapped_number(num) or num
            key = key_for(series, trial_idx)

            truth = st.session_state._fillin_truth
            shown = st.session_state._fillin_shown
            attempted_sentence = st.session_state.get("_attempted_sentence", "")

            # Build a resolved (correct) sentence from `truth`
            def truth_val(p):
                v = truth.get(p)
                if p == "days":
                    return _display_days(v) if v else ""
                return "" if v in (None, "") else str(v)

            resolved_sentence = get_template_for_prompt(mapped_num_val)
            for p in ALLOWED_PARAMS:
                resolved_sentence = resolved_sentence.replace(f"{{{p}}}", truth_val(p))

            save_current_fillin(
                truth, shown, key, mapped_num_val,
                attempted_sentence=attempted_sentence,
                resolved_sentence=resolved_sentence,
            )

    # advance
    if st.session_state.substep < max_step:
        st.session_state.substep += 1
    else:
        st.session_state.trial_idx = min(18, st.session_state.trial_idx + 1)
        st.session_state.substep = 1

    st.session_state._jump_to_top = True  # (you can rename to _scroll_to_top for consistency)
    st.session_state["_scroll_to_top"] = True

    st.session_state["_scroll_to_top"] = True
    st.session_state["_scroll_token"] = str(uuid4())  # unique key per click


# ======================== WELCOME SCREEN ========================
def welcome_screen():
    st.title("Welcome")
    st.markdown("Please enter your **Participant Number** and choose the **Order Number** to begin.")
    col1, col2 = st.columns(2)
    with col1:
        pid = st.text_input("Participant Number", value=st.session_state.participant_num)
    with col2:
        order = st.selectbox("Order Number", options=[1,2,3,4,5], index=st.session_state.series - 1)

    start = st.button("Begin ▶", key="btn_begin", type="primary")


    if start:
        st.session_state.participant_num = pid.strip() or "unknown"
        st.session_state.series = int(order)

        # initialize both legacy and unified nav keys
        st.session_state.trial_idx   = 1
        st.session_state.substep     = 1
        st.session_state.nav_trial_idx = 1
        st.session_state.nav_substep   = 1

        # mark setup as completed for the router that looks at setup_done/app_stage
        st.session_state.started    = True          # keep for any legacy checks
        st.session_state.setup_done = True          # <-- critical
        st.session_state.app_stage  = "run"         # <-- critical

        os.makedirs(CSV_DIR, exist_ok=True)
        _ = rankings_csv_path()
        _ = survey_csv_path()

        st.rerun()

# ======================== APP ========================
# ---- Stage routing ----
if st.session_state.app_stage == "tutorial":
    render_tutorial()
    st.stop()

# Your existing setup gate comes next (unchanged):
if not st.session_state.setup_done:
    # render_setup() should end by setting setup_done True and (optionally) app_stage='run'
    # If your render_setup currently sets setup_done, keep that; otherwise call _go_to_run()
    welcome_screen()   # your existing function that collects participant_num + series
    st.stop()


# if not st.session_state.started:
#     welcome_screen()
#     st.stop()

# Keep the widget showing the current nav value WITHOUT writing to the widget key.
_target_trial = int(st.session_state.get("nav_trial_idx", 1))

# If the existing widget value differs, delete it so the next render can re-init with `value=`
if st.session_state.get("trial_idx_widget", None) != _target_trial:
    if "trial_idx_widget" in st.session_state:
        del st.session_state["trial_idx_widget"]


# ----- Sidebar status (fixed identifiers once started) -----
st.sidebar.header("Session")
st.sidebar.write(f"**Participant:** {st.session_state.participant_num}")
st.sidebar.write(f"**Order (Series):** {st.session_state.series}")
st.sidebar.selectbox("Version (for main page images)", options=[1,2], key="version")
# st.sidebar.number_input("Trial (1–18)", min_value=1, max_value=18, step=1, key="trial_idx")
st.sidebar.number_input(
    "Trial (1–18)",
    min_value=1, max_value=18, step=1,
    value=_target_trial,                # <- seed from nav
    key="trial_idx_widget",
    on_change=_sync_nav_from_widget,    # <- sync user changes to nav keys
)



# series    = st.session_state.series
# version   = st.session_state.version
# trial_idx = st.session_state.trial_idx
# substep   = st.session_state.substep

series    = st.session_state.series
version   = st.session_state.version
trial_idx = current_trial_idx()     # <-- use the helper
substep   = current_substep()       # <-- use the helper


# ----- Resolve current trial -----
rep_type  = ORDERINGS[series]["rep"][trial_idx - 1]
expl_type = ORDERINGS[series]["expl"][trial_idx - 1]
num       = ORDERINGS[series]["num"][trial_idx - 1]

# ----- Header -----
st.title(f"**Trial {trial_idx}/18** · ")
# st.markdown(
#     f"**Participant {st.session_state.participant_num}** · "
#     f"**Series {series}** · **Trial {trial_idx}/18** · "
#     f"**PromptNum {num}** · **Step {substep}/{substeps_max_for(trial_idx)}**"
# )
# st.caption(
#     f"Representation: {REP_LABELS.get(rep_type, rep_type)} · "
#     f"Explanation: {EXPL_LABELS.get(expl_type, expl_type)} · "
#     f"Version selector (main only): v{version}"
# )



# ======================== PAGE BODY ========================
if substep == 1:
    # ----- Examples (prompt + representation + explanation) -----
    if expl_type == 2:
        st.markdown("### Counterfactual")
        for v in [1, 2]:
            if v == 1:
                st.markdown(f"#### Original Prompt and Output")
            else:
                st.markdown(f"#### Modified Prompt and Output")
            c1, c2 = st.columns(2)
            with c1: show_image_or_warn(prompt_path(num, v), custom_size=800)
            with c2: 
                if rep_type == 3:
                    show_image_or_warn(rep_path(rep_type, num, version), custom_size=800)
                elif rep_type == 1:
                # show_image_or_warn(rep_path(rep_type, num, version), f"Rep v{version}")
                    show_image_or_warn(rep_path(rep_type, num, version), custom_size=1500)
                else:
                    show_image_or_warn(rep_path(rep_type, num, version))

    else:
        st.markdown("### Prompt")
        # show_image_or_warn(prompt_path(num, version), f"Prompt v{version}")
        show_image_or_warn(prompt_path(num, version), custom_size=600)
        c1, c2 = st.columns(2)
        with c1:
            # st.markdown(f"### Representation ({REP_LABELS.get(rep_type, rep_type)})")
            st.markdown(f"### Representation")
            if rep_type == 2:
                show_image_or_warn(rep_path(rep_type, num, version), custom_size=600)
            else:
            # show_image_or_warn(rep_path(rep_type, num, version), f"Rep v{version}")
                show_image_or_warn(rep_path(rep_type, num, version))
        with c2:
            # st.markdown(f"### Explanation ({EXPL_LABELS.get(expl_type, expl_type)})")
            st.markdown(f"### Explanation")
            show_image_or_warn(expl_path(expl_type, num), custom_size=600)

    st.divider()

    st.markdown("### Representation With Unknown Parameters: See Below To Select Plausible Ones From Dropdowns")

    mapped_num_val = mapped_number(num) or num
    # st.caption(f"Mapping: used #{num} → leftover #{mapped_num_val}")
    follow_rep_path = rep_path(rep_type, mapped_num_val, 1)
    if rep_type == 1:
        show_image_or_warn(follow_rep_path, f"### Current Representation to consider", custom_size=1500)
    elif rep_type == 2:
        show_image_or_warn(follow_rep_path, f"### Current Representation to consider", custom_size=1000)
    else:
        show_image_or_warn(follow_rep_path, f"### Current Representation to consider", custom_size=800)



    st.divider()  # <— template at the bottom
    # after you compute: mapped_num_val = mapped_number(num) or num
    mapped_num_val = mapped_number(num) or num
    key = key_for(series, trial_idx)

    tmpl = get_template_for_prompt(mapped_num_val)
    truth, shown, complete_fillin, attempted_sentence = render_fillin_template_ui(
        mapped_num_val,
        key,
        POOLS,
        template_text=tmpl,
        key_ns=f"{series}-{trial_idx}",   # important for unique widget keys
    )

    # After rendering the 4 dropdowns:
    series, trial_idx, num, mapped, rep_type, expl_type, key = _current_meta()

    if key not in st.session_state.fillin_timer_start:
        st.session_state.fillin_timer_start[key] = time.perf_counter()

    answers = st.session_state.fill_answers.get(key, {})
    attempted_sentence = st.session_state.get("_attempted_sentence", "")

    # Optional “resolved” sentence if you still want it in the file:
    def _truth_val(p):
        v = st.session_state._fillin_truth.get(p) if hasattr(st.session_state, "_fillin_truth") else None
        if p == "days":
            try:
                return _display_days(v) if v else ""
            except Exception:
                return str(v or "")
        return "" if v in (None, "") else str(v)

    resolved_sentence = get_template_for_prompt(mapped)
    for p in ALLOWED_PARAMS:
        resolved_sentence = resolved_sentence.replace(f"{{{p}}}", _truth_val(p))

    # Gate Next
    st.session_state._fillin_complete = complete_fillin
    st.session_state._fillin_truth = truth
    st.session_state._fillin_shown = shown
    st.session_state._attempted_sentence = attempted_sentence



series    = st.session_state.series
trial_idx = current_trial_idx()
key       = confidence_key(series, trial_idx)

# --- NEW: Confidence page just after the fill-in page ---
if current_substep() == CONF_SUBSTEP:
    st.markdown("### How confident are you in your most recent answer?")
    st.caption("1 = Not confident at all, 7 = Extremely confident")


    default_val = st.session_state.confidence_answers.get(key, None)

    choice = st.radio(
        "Confidence (1–7)",
        options=list(range(1, 8)),  # 1..7
        index=(default_val - 1) if isinstance(default_val, int) and 1 <= default_val <= 7 else None,
        horizontal=True,
        key=f"conf_{series}_{trial_idx}",
    )

    if choice is not None:
        st.session_state.confidence_answers[key] = int(choice)
        if key not in st.session_state.confidence_order:
            st.session_state.confidence_order.append(key)
        _write_all_confidence_snapshot()
    else:
        st.info("Please select your confidence to continue.")





if is_block_end(current_trial_idx()) and current_substep() == SURVEY_SUBSTEP:
    series    = st.session_state.series
    trial_idx = current_trial_idx()
    skey      = survey_key(series, trial_idx)   # "series::blockN"
    st.markdown("### Block Survey")
    st.caption("Please rate each statement from 1 (strongly disagree) to 7 (strongly agree).")

    st.session_state.survey_answers.setdefault(skey, {})
    answers = st.session_state.survey_answers[skey]

    for qname, qtext in SURVEY_ITEMS:
        default = answers.get(qname)
        val = st.radio(
            f"{qname}. {qtext}",
            options=list(range(SURVEY_SCALE_MIN, SURVEY_SCALE_MAX + 1)),
            index=(default - SURVEY_SCALE_MIN) if isinstance(default, int) else None,
            horizontal=True,
            key=f"survey_{skey}_{qname}",
        )
        if val is not None:
            answers[qname] = int(val)

    if skey not in st.session_state.survey_order:
        st.session_state.survey_order.append(skey)

    # Gate navigation on completion
    all_answered = all(q in answers and isinstance(answers[q], int) for q, _ in SURVEY_ITEMS)
    if not all_answered:
        st.info("Please answer all questions to continue.")




# ===== Navigation controls (drop-in) =====
import streamlit as st

# --- safe defaults if not set elsewhere (won't override if already present) ---
if "trial_idx" not in st.session_state:
    st.session_state.trial_idx = 1
if "substep" not in st.session_state:
    st.session_state.substep = 1
if "total_trials" not in st.session_state:
    st.session_state.total_trials = 18  # adjust to your real total

def _substeps_max_for(idx: int) -> int:
    """Use your own substeps_max_for if you have it; this is a safe fallback."""
    try:
        return substeps_max_for(idx)  # type: ignore[name-defined]
    except Exception:
        # Example: 2 substeps for all trials; change to your logic if needed
        return 2




def current_trial_idx() -> int:
    return int(st.session_state.get("nav_trial_idx", 1))

def current_substep() -> int:
    return int(st.session_state.get("nav_substep", 1))



prev_col, next_col = st.columns([1, 1])


prev_col, next_col = st.columns([1, 1])

# ---------------------- PREV ----------------------
with prev_col:
    if st.button("⟵ Prev", use_container_width=True, key="btn_prev"):
        t = current_trial_idx()
        s = current_substep()

        if s > 1:
            # Only step back a page within the same trial
            # If currently on survey page, pop its most recent row
            if is_block_end(t) and s == SURVEY_SUBSTEP:
                if st.session_state.survey_order:
                    last_skey = st.session_state.survey_order.pop()
                    st.session_state.survey_answers.pop(last_skey, None)
                    _write_all_survey_snapshot()
            st.session_state["nav_substep"] = s - 1
        else:
            # LIFO removal when leaving a trial completely (you already do fill-ins/confidence here)
            # (Optional) also pop last survey row if you want strict LIFO across all entities
            if st.session_state.survey_order:
                last_skey = st.session_state.survey_order.pop()
                st.session_state.survey_answers.pop(last_skey, None)
                _write_all_survey_snapshot()
            # s == 1: leaving the current trial entirely -> drop most recently saved trial
            # Remove last saved fill-in row (LIFO) and its timer, then rewrite the CSV
            if st.session_state.get("fillins_order"):
                last_key = st.session_state.fillins_order.pop()
                st.session_state.fillins_log.pop(last_key, None)
                st.session_state.fillin_timer_start.pop(last_key, None)  # drop per-trial timer
                _write_all_fillins_snapshot()

            # Also remove the most recent confidence answer (LIFO) and rewrite
            if st.session_state.get("confidence_order"):
                last_ckey = st.session_state.confidence_order.pop()
                st.session_state.confidence_answers.pop(last_ckey, None)
                _write_all_confidence_snapshot()

            # (Optional) If you snapshot rankings and want to roll them back too:
            # if "rank_answers" in st.session_state and "save_rankings_snapshot" in globals():
            #     # If you track an order list for rankings, pop it; else, clear the current trial key:
            #     series = st.session_state.series
            #     key_for_trial = key_for(series, max(1, t - 1))  # previous trial key
            #     st.session_state.rank_answers.pop(key_for_trial, None)
            #     save_rankings_snapshot()

            # Navigate to the previous trial's last substep
            if t > 1:
                prev_t = t - 1
                st.session_state["nav_trial_idx"] = prev_t
                st.session_state["nav_substep"]   = substeps_max_for(prev_t)
            else:
                st.session_state["nav_trial_idx"] = 1
                st.session_state["nav_substep"]   = 1

        st.session_state["scroll_to_top"] = True
        st.rerun()

# ---------------------- NEXT ----------------------
with next_col:
    # Optional: gate Next on survey completion (only on the survey page)
    _allow_next = True
    if is_block_end(current_trial_idx()) and current_substep() == SURVEY_SUBSTEP:
        series = st.session_state.series
        t = current_trial_idx()
        skey = survey_key(series, t)
        answers = st.session_state.survey_answers.get(skey, {})
        _allow_next = all(
            (q in answers) and isinstance(answers[q], int)
            for q, _ in SURVEY_ITEMS
        )

    if st.button("Next ⟶", type="primary", use_container_width=True, key="btn_next", disabled=not _allow_next):
        t = current_trial_idx()
        s = current_substep()
        max_step = substeps_max_for(t)

        # ---------- SAVE on substep 1 (fill-in page) ----------
        if s == 1:
            import time

            series, trial_idx, num, mapped, rep_type, expl_type, key = _current_meta()
            answers = st.session_state.get("fill_answers", {}).get(key, {})
            attempted_sentence = st.session_state.get("_attempted_sentence", "")

            # Resolve sentence (safe fallback)
            resolved_sentence = ""
            try:
                def _truth_val(p):
                    v = getattr(st.session_state, "_fillin_truth", {}).get(p)
                    if p == "days" and "_display_days" in globals():
                        try:
                            return _display_days(v) if v else ""
                        except Exception:
                            return str(v or "")
                    return "" if v in (None, "") else str(v)
                if "get_template_for_prompt" in globals():
                    resolved_sentence = get_template_for_prompt(mapped)
                    for p in ALLOWED_PARAMS:
                        resolved_sentence = resolved_sentence.replace(f"{{{p}}}", _truth_val(p))
            except Exception:
                pass

            # Timer (3 decimals)
            start_ts = st.session_state.get("fillin_timer_start", {}).get(key, time.perf_counter())
            elapsed  = max(0.0, time.perf_counter() - start_ts)
            duration_sec = float(f"{elapsed:.3f}")

            # 3.1 Get the truths for THIS variant (mapped, rep_type, expl_type)
            truth_cols = truth_from_reference(mapped, rep_type, expl_type)  # now includes truth_prompt_text

            
            # 1) ensure you have the exact template shown
            template_text = get_template_for_prompt(mapped_num_val)  # or wherever you obtain it

            # 2) fetch the ground-truth prompt text from the CSV
            truth_prompt_text, truth_prompt_col = lookup_truth_prompt_text(mapped_num_val, template_hint=template_text)

            # 3) include them in the row you persist
            row_dict = {
                "participant_num":           st.session_state.participant_num,
                "order_num":                 series,
                "prompt_num":                num,
                "mapped_prompt_num":         mapped_num_val,
                "representation_num":        rep_type,
                "explanation_num":           expl_type,
                "days_sel":                  answers.get("days"),
                "time_of_day_sel":           answers.get("time_of_day"),
                "number_of_classes_sel":     answers.get("number_of_classes"),
                "course_level_minimum_sel":  answers.get("course_level_minimum"),
                "attempted_sentence":        attempted_sentence,
                "resolved_sentence":         resolved_sentence,
                "fillin_duration_sec":       st.session_state.get("_fillin_duration_sec", None),

                # truths (from your existing parser that reads V1_Base_Text)
                "days_truth":                _display_days(truth.get("days")),
                "time_of_day_truth":         truth.get("time_of_day"),
                "number_of_classes_truth":   truth.get("number_of_classes"),
                "course_level_minimum_truth":truth.get("course_level_minimum"),

                # NEW: persist the actual ground-truth prompt & the column we used
                "truth_prompt_text":         truth_prompt_text,
                "truth_prompt_col":          truth_prompt_col,
            }

            # save to in-memory log and snapshot to CSV as you already do
            st.session_state.fillins_log[key] = row_dict
            if key not in st.session_state.fillins_order:
                st.session_state.fillins_order.append(key)
            _write_all_fillins_snapshot()

            # row = {
            #     "participant_num":           st.session_state.participant_num,
            #     "order_num":                 series,
            #     "prompt_num":                num,
            #     "mapped_prompt_num":         mapped,
            #     "representation_num":        rep_type,
            #     "explanation_num":           expl_type,
            #     "days_sel":                  answers.get("days"),
            #     "time_of_day_sel":           answers.get("time_of_day"),
            #     "number_of_classes_sel":     answers.get("number_of_classes"),
            #     "course_level_minimum_sel":  answers.get("course_level_minimum"),
            #     "attempted_sentence":        attempted_sentence,
            #     "resolved_sentence":         resolved_sentence,
            #     "fillin_duration_sec":       duration_sec,
            #     **truth_cols,  # <- includes days_*, time_*, count_*, level_* AND truth_prompt_text (+ truth_prompt_col)
            # }
            # st.session_state.fillins_log[key] = row
            # if key not in st.session_state.fillins_order:
            #     st.session_state.fillins_order.append(key)
            # _write_all_fillins_snapshot()


        # ---------- SAVE on substep 3 (block survey page) ----------
        elif s == SURVEY_SUBSTEP and is_block_end(t):
            series = st.session_state.series
            skey   = survey_key(series, t)
            answers = st.session_state.survey_answers.get(skey, {})
            all_answered = all(
                (q in answers) and isinstance(answers[q], int)
                for q, _ in SURVEY_ITEMS
            )
            if not all_answered:
                st.warning("Please answer all survey questions before continuing.")
                st.stop()
            _write_all_survey_snapshot()

        # (Nothing to save on substep 2; confidence autosaves on change.)

        # ---------- ADVANCE NAV ----------
        if s < max_step:
            st.session_state["nav_substep"] = s + 1
        else:
            st.session_state["nav_trial_idx"] = min(18, t + 1)
            st.session_state["nav_substep"]   = 1

        # Request scroll + rerun; do NOT mutate widget key.
        st.session_state["scroll_to_top"] = True
        st.rerun()
