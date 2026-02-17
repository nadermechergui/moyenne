import uuid
import base64
import io
import re
import time
from datetime import datetime, date

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError
from PIL import Image

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Portail Mega Formation", page_icon="🧩", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],

    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "student_name", "created_at", "last_login"],

    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group",
               "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],

    # ✅ Planning structured (no Drive/base64): staff can CRUD
    "Timetable": ["row_id", "branch", "program", "group", "year", "day", "start", "end",
                  "subject_name", "teacher", "color", "room", "note", "updated_at", "staff_name"],

    # ✅ Profile pics (small base64)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    # ✅ Payments: one row per trainee_id + year
    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"] + MONTHS + ["updated_at", "staff_name"],

    # ✅ Supports links (manual)
    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "url", "uploaded_at", "staff_name"],
}

# =========================================================
# UTILS
# =========================================================
def norm(x):
    return str(x or "").strip()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_bool_str(v: bool) -> str:
    return "TRUE" if v else "FALSE"

def df_filter(df: pd.DataFrame, **kwargs):
    out = df.copy()
    for k, v in kwargs.items():
        if k in out.columns:
            out = out[out[k].astype(str).str.strip() == norm(v)]
    return out

def explain_api_error(e: APIError) -> str:
    try:
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", "") or ""
        low = text.lower()

        if status == 429 or "quota" in low or "rate" in low:
            return "⚠️ 429 Quota (Google Sheets). اعمل Reboot واستنى دقيقة.\n" + text[:260]
        if status == 403 or "permission" in low or "forbidden" in low:
            return "❌ 403 Permission. Share Google Sheet مع service account كـ Editor.\n" + text[:260]
        if status == 404 or "not found" in low:
            return "❌ 404 Not found. تأكد GSHEET_ID صحيح + Share للـ service account.\n" + text[:260]
        return "❌ Google API Error:\n" + (text[:380] if text else str(e))
    except Exception:
        return "❌ Google API Error."

def compress_image_bytes(img_bytes: bytes, max_side: int = 256, quality: int = 70) -> bytes:
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    w, h = im.size
    scale = min(max_side / max(w, h), 1.0)
    nw, nh = int(w * scale), int(h * scale)
    im = im.resize((nw, nh))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()

def html_escape(s: str) -> str:
    return (
        norm(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

def normalize_phone(s: str) -> str:
    # keep digits + plus
    s = norm(s)
    s = re.sub(r"[^\d+]", "", s)
    return s

# =========================================================
# GOOGLE SHEETS CLIENT
# =========================================================
@st.cache_resource
def creds():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    return Credentials.from_service_account_info(creds_dict, scopes=scopes)

@st.cache_resource
def gs_client():
    return gspread.authorize(creds())

@st.cache_resource
def spreadsheet():
    return gs_client().open_by_key(st.secrets["GSHEET_ID"])

# =========================================================
# SCHEMA (SAFE: no delete)
# =========================================================
def ensure_headers_safe(ws, headers: list[str]):
    # one small read
    rng = ws.get("1:1")
    row1 = rng[0] if (rng and len(rng) > 0) else []
    row1 = [norm(x) for x in row1]

    if len(row1) == 0 or all(x == "" for x in row1):
        ws.append_row(headers, value_input_option="RAW")
        return

    if row1 != headers:
        st.warning(f"⚠️ Sheet '{ws.title}' headers مختلفة. ما عملتش مسح. "
                   f"إذا تحب صحّح الهيدرز يدويًا باش تولّي مطابقة.")

def ensure_worksheets_and_headers():
    sh = spreadsheet()
    titles = [w.title for w in sh.worksheets()]
    for ws_name, headers in REQUIRED_SHEETS.items():
        if ws_name not in titles:
            sh.add_worksheet(title=ws_name, rows=4000, cols=max(18, len(headers) + 2))
            titles.append(ws_name)
        ws = sh.worksheet(ws_name)
        ensure_headers_safe(ws, headers)

def ensure_schema_once():
    # manual init only to avoid 429
    if st.session_state.get("schema_ok", False):
        return
    if not st.session_state.get("init_schema_now", False):
        return
    try:
        ensure_worksheets_and_headers()
        st.session_state.schema_ok = True
        st.session_state.init_schema_now = False
        st.success("✅ Sheets vérifiées/initialisées.")
    except APIError as e:
        st.session_state.init_schema_now = False
        st.error(explain_api_error(e))
        raise

# =========================================================
# DATA ACCESS (reduce quota)
# =========================================================
def _ss_cache_key(ws_name: str) -> str:
    return f"dfcache::{ws_name}"

def _ss_cache_ts_key(ws_name: str) -> str:
    return f"dfcache_ts::{ws_name}"

def read_df(ws_name: str, ttl_sec: int = 20) -> pd.DataFrame:
    """
    Session-state cache + small TTL to reduce 429.
    """
    key = _ss_cache_key(ws_name)
    kts = _ss_cache_ts_key(ws_name)
    now = time.time()

    if key in st.session_state and kts in st.session_state:
        if now - st.session_state[kts] < ttl_sec:
            return st.session_state[key].copy()

    try:
        ws = spreadsheet().worksheet(ws_name)
        values = ws.get_all_values()
        if len(values) <= 1:
            df = pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
        else:
            headers = values[0]
            rows = values[1:]
            df = pd.DataFrame(rows, columns=headers)
        st.session_state[key] = df
        st.session_state[kts] = now
        return df.copy()
    except APIError as e:
        # return last cached if exists
        if key in st.session_state:
            st.warning("⚠️ Google Sheets quota/erreur. عرضت آخر نسخة مخزّنة.")
            return st.session_state[key].copy()
        st.error(explain_api_error(e))
        return pd.DataFrame(columns=REQUIRED_SHEETS.get(ws_name, []))

def invalidate_df_cache(ws_name: str | None = None):
    if ws_name:
        st.session_state.pop(_ss_cache_key(ws_name), None)
        st.session_state.pop(_ss_cache_ts_key(ws_name), None)
    else:
        for k in list(st.session_state.keys()):
            if str(k).startswith("dfcache::") or str(k).startswith("dfcache_ts::"):
                st.session_state.pop(k, None)

def append_row(ws_name: str, row: dict):
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    invalidate_df_cache(ws_name)

def update_row_by_key(ws_name: str, key_cols: list[str], key_vals: list[str], updates: dict) -> bool:
    df = read_df(ws_name)
    if df.empty:
        return False

    m = df.copy()
    for c, v in zip(key_cols, key_vals):
        if c not in m.columns:
            return False
        m = m[m[c].astype(str).str.strip() == norm(v)]

    if m.empty:
        return False

    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]

    # batch update via ranges
    updates_list = []
    for col_name, val in updates.items():
        if col_name not in headers:
            continue
        col_num = headers.index(col_name) + 1
        updates_list.append((row_num, col_num, norm(val)))

    for r, c, v in updates_list:
        ws.update_cell(r, c, v)

    invalidate_df_cache(ws_name)
    return True

def delete_row_by_key(ws_name: str, key_col: str, key_val: str) -> bool:
    df = read_df(ws_name)
    if df.empty or key_col not in df.columns:
        return False
    m = df[df[key_col].astype(str).str.strip() == norm(key_val)]
    if m.empty:
        return False
    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet(ws_name)
    ws.delete_rows(row_num)
    invalidate_df_cache(ws_name)
    return True

# =========================================================
# PROFILE PICS
# =========================================================
def get_profile_pic_bytes(phone: str) -> bytes | None:
    phone = normalize_phone(phone)
    if not phone:
        return None
    df = read_df("ProfilePics")
    if df.empty:
        return None
    m = df[df["phone"].astype(str).str.strip() == phone]
    if m.empty:
        return None
    b64 = norm(m.iloc[0].get("image_b64"))
    if not b64:
        return None
    try:
        return base64.b64decode(b64.encode("utf-8"))
    except Exception:
        return None

def upsert_profile_pic(phone: str, trainee_id: str, img_bytes: bytes):
    phone = normalize_phone(phone)
    small = compress_image_bytes(img_bytes, max_side=256, quality=70)
    b64 = base64.b64encode(small).decode("utf-8")

    updated = update_row_by_key(
        "ProfilePics",
        ["phone"], [phone],
        {"trainee_id": trainee_id, "image_b64": b64, "uploaded_at": now_str()},
    )
    if not updated:
        append_row("ProfilePics", {
            "phone": phone,
            "trainee_id": trainee_id,
            "image_b64": b64,
            "uploaded_at": now_str(),
        })

# =========================================================
# PAYMENTS
# =========================================================
def ensure_payment_row(trainee_id: str, branch: str, program: str, group: str, year: str, staff_name: str):
    df = read_df("Payments")
    if not df.empty:
        m = df[(df["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
               (df["year"].astype(str).str.strip() == norm(year))]
        if not m.empty:
            return

    row = {
        "payment_id": f"PAY-{uuid.uuid4().hex[:8].upper()}",
        "trainee_id": trainee_id,
        "branch": branch,
        "program": program,
        "group": group,
        "year": year,
        "updated_at": now_str(),
        "staff_name": staff_name,
    }
    for mo in MONTHS:
        row[mo] = "FALSE"
    append_row("Payments", row)

def set_payment_month(trainee_id: str, year: str, month: str, paid: bool, staff_name: str) -> bool:
    df = read_df("Payments")
    if df.empty:
        return False
    m = df[(df["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
           (df["year"].astype(str).str.strip() == norm(year))]
    if m.empty:
        return False

    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet("Payments")
    headers = REQUIRED_SHEETS["Payments"]

    ws.update_cell(row_num, headers.index(month) + 1, safe_bool_str(paid))
    ws.update_cell(row_num, headers.index("updated_at") + 1, now_str())
    ws.update_cell(row_num, headers.index("staff_name") + 1, staff_name)

    invalidate_df_cache("Payments")
    return True

# =========================================================
# TIMETABLE
# =========================================================
def load_timetable(branch: str, program: str, group: str, year: str) -> pd.DataFrame:
    df = read_df("Timetable")
    if df.empty:
        return df
    for c in ["branch", "program", "group", "year", "day", "start", "end"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df = df[(df["branch"] == norm(branch)) &
            (df["program"] == norm(program)) &
            (df["group"] == norm(group)) &
            (df["year"] == norm(year))]
    return df.copy()

def add_timetable_row(branch: str, program: str, group: str, year: str,
                      day: str, start: str, end: str, subject_name: str,
                      teacher: str, color: str, room: str, note: str, staff_name: str):
    append_row("Timetable", {
        "row_id": f"TT-{uuid.uuid4().hex[:10].upper()}",
        "branch": branch,
        "program": program,
        "group": group,
        "year": year,
        "day": day,
        "start": start,
        "end": end,
        "subject_name": subject_name,
        "teacher": teacher,
        "color": color,
        "room": room,
        "note": note,
        "updated_at": now_str(),
        "staff_name": staff_name,
    })

def update_timetable_row(row_id: str, updates: dict) -> bool:
    updates = dict(updates)
    updates["updated_at"] = now_str()
    return update_row_by_key("Timetable", ["row_id"], [row_id], updates)

def delete_timetable_row(row_id: str) -> bool:
    return delete_row_by_key("Timetable", "row_id", row_id)

def timetable_html_grid(df: pd.DataFrame) -> str:
    """
    Build colored weekly grid (HTML).
    """
    if df.empty:
        return "<div style='padding:12px;border:1px solid #eee;border-radius:10px;'>Aucun planning.</div>"

    # normalize
    df2 = df.copy()
    for c in ["day", "start", "end", "subject_name", "teacher", "color", "room"]:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).str.strip()

    # sort by day then start
    day_order = {d: i for i, d in enumerate(DAYS)}
    df2["day_i"] = df2["day"].map(lambda x: day_order.get(x, 999))
    df2 = df2.sort_values(by=["day_i", "start", "end"])

    # group entries per day
    per_day = {d: [] for d in DAYS}
    for _, r in df2.iterrows():
        d = r.get("day", "")
        if d not in per_day:
            per_day[d] = []
        color = norm(r.get("color")) or "#E6F2FF"
        subj = html_escape(r.get("subject_name"))
        teacher = html_escape(r.get("teacher"))
        start = html_escape(r.get("start"))
        end = html_escape(r.get("end"))
        room = html_escape(r.get("room"))
        extra = f"<div style='opacity:.85;font-size:12px'>{start} - {end}</div>"
        if room:
            extra += f"<div style='opacity:.75;font-size:12px'>Salle: {room}</div>"
        cell = f"""
        <div style="
            background:{color};
            border-radius:10px;
            padding:10px;
            margin:8px 0;
            border:1px solid rgba(0,0,0,.08);
        ">
            <div style="font-weight:700">{subj}</div>
            <div style="font-size:13px;opacity:.9">{teacher}</div>
            {extra}
        </div>
        """
        per_day[d].append(cell)

    # build table
    ths = "".join([f"<th style='padding:10px;border-bottom:1px solid #ddd;text-align:left'>{d}</th>" for d in DAYS])
    tds = "".join([
        f"<td style='vertical-align:top;padding:10px;border-right:1px solid #f0f0f0;min-width:170px'>"
        + "".join(per_day[d]) +
        "</td>"
        for d in DAYS
    ])
    html = f"""
    <div style="overflow:auto;border:1px solid #eee;border-radius:12px;padding:8px">
      <table style="border-collapse:collapse;width:100%">
        <thead><tr>{ths}</tr></thead>
        <tbody><tr>{tds}</tr></tbody>
      </table>
    </div>
    """
    return html

# =========================================================
# AUTH / SESSION
# =========================================================
def ensure_session():
    st.session_state.setdefault("role", None)     # "staff" | None
    st.session_state.setdefault("user", {})
    st.session_state.setdefault("student", None)

def logout_staff():
    st.session_state.role = None
    st.session_state.user = {}

def staff_branch_login(branch: str, branch_password: str):
    df = read_df("Branches")
    if df.empty:
        return None
    df2 = df.copy()
    df2["branch"] = df2["branch"].astype(str).str.strip()
    df2["staff_password"] = df2["staff_password"].astype(str).str.strip()
    df2["is_active"] = df2["is_active"].astype(str).str.strip().str.lower()

    m = df2[(df2["branch"] == norm(branch)) &
            (df2["staff_password"] == norm(branch_password)) &
            (df2["is_active"] != "false")]
    if m.empty:
        return None
    return {"branch": norm(branch), "role": "staff"}

def student_login(phone: str, password: str):
    phone = normalize_phone(phone)
    df = read_df("Accounts")
    if df.empty:
        return None
    df2 = df.copy()
    df2["phone"] = df2["phone"].astype(str).str.strip().apply(normalize_phone)
    df2["password"] = df2["password"].astype(str).str.strip()
    m = df2[(df2["phone"] == phone) & (df2["password"] == norm(password))]
    if m.empty:
        return None
    return m.iloc[0].to_dict()

# =========================================================
# SIDEBAR (STAFF LOGIN LEFT)
# =========================================================
def sidebar_staff_login():
    st.sidebar.markdown("## 👨‍💼 Connexion Employé")

    branches_df = read_df("Branches")
    branches = sorted([x for x in branches_df.get("branch", pd.Series([])).astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []

    if st.session_state.role == "staff":
        br = norm(st.session_state.user.get("branch"))
        st.sidebar.success(f"Connecté: {br}")

        st.sidebar.divider()
        st.sidebar.markdown("### 🧰 Maintenance")
        if st.sidebar.button("Initialiser / Vérifier les Sheets", use_container_width=True, key="btn_init_schema"):
            st.session_state.init_schema_now = True
            st.rerun()

        if st.sidebar.button("Se déconnecter", use_container_width=True, key="btn_logout_staff"):
            logout_staff()
            st.rerun()
        return

    if not branches:
        st.sidebar.warning("Branches vide. Ajoutez centres + mots de passe.")
        return

    branch = st.sidebar.selectbox("Centre", branches, key="sb_branch")
    pwd = st.sidebar.text_input("Mot de passe du centre", type="password", key="sb_pwd")

    if st.sidebar.button("Connexion", use_container_width=True, key="btn_login_staff"):
        user = staff_branch_login(branch, pwd)
        if user:
            st.session_state.role = "staff"
            st.session_state.user = user
            st.sidebar.success("✅ OK")
            st.rerun()
        else:
            st.sidebar.error("Mot de passe incorrect / centre inactif.")

# =========================================================
# STUDENT PORTAL (CENTER)
# =========================================================
def student_portal_center():
    st.markdown("## 🎓 Espace Stagiaire")

    tab1, tab2, tab3 = st.tabs(["🔐 Connexion", "🆕 Inscription", "📌 Mon espace"])

    # ---------------- Login
    with tab1:
        phone = st.text_input("Téléphone", key="stud_phone")
        pwd = st.text_input("Mot de passe", type="password", key="stud_pwd")

        if st.button("Se connecter", use_container_width=True, key="btn_stud_login"):
            acc = student_login(phone, pwd)
            if acc:
                update_row_by_key("Accounts", ["phone"], [normalize_phone(phone)], {"last_login": now_str()})
                st.session_state.student = acc
                st.success("✅ Connexion réussie")
                st.rerun()
            else:
                st.error("Téléphone / mot de passe incorrect.")

        if st.button("Se déconnecter (Stagiaire)", use_container_width=True, key="btn_stud_logout"):
            st.session_state.student = None
            st.rerun()

    # ---------------- Registration
    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")

        branches_df = read_df("Branches")
        branches = sorted([x for x in branches_df.get("branch", pd.Series([])).astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre.")
            return
        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(read_df("Programs"), branch=b)
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not prog_df.empty else prog_df
        programs = sorted([x for x in prog_df.get("program_name", pd.Series([])).astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_prog")

        grp_df = df_filter(read_df("Groups"), branch=b, program_name=p)
        grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not grp_df.empty else grp_df
        groups = sorted([x for x in grp_df.get("group_name", pd.Series([])).astype(str).str.strip().tolist() if x])
        if not groups:
            st.warning("Aucun groupe.")
            return
        g = st.selectbox("Groupe", groups, key="reg_group")

        student_name = st.text_input("Nom (أي اسم تحب)", key="reg_name")
        phone = st.text_input("Téléphone (نفس رقمك عند الإدارة)", key="reg_phone")
        pwd = st.text_input("Mot de passe", type="password", key="reg_pwd")

        if st.button("Créer mon compte", use_container_width=True, key="btn_register"):
            phone_n = normalize_phone(phone)
            if not norm(student_name) or not phone_n or not norm(pwd):
                st.error("Nom + téléphone + mot de passe obligatoire.")
                return
            if len(norm(pwd)) < 4:
                st.error("Mot de passe قصير (min 4).")
                return

            acc = read_df("Accounts")
            if not acc.empty:
                accp = acc.copy()
                accp["phone"] = accp["phone"].astype(str).str.strip().apply(normalize_phone)
                if accp["phone"].eq(phone_n).any():
                    st.error("Ce téléphone est déjà inscrit.")
                    return

            tr = read_df("Trainees")
            if tr.empty:
                st.error("Aucun stagiaire.")
                return

            tr2 = tr.copy()
            for c in ["branch", "program", "group", "phone"]:
                if c in tr2.columns:
                    tr2[c] = tr2[c].astype(str).str.strip()
            tr2["phone_norm"] = tr2["phone"].apply(normalize_phone)

            candidates = tr2[(tr2["branch"] == norm(b)) &
                             (tr2["program"] == norm(p)) &
                             (tr2["group"] == norm(g)) &
                             (tr2["phone_norm"] == phone_n)]

            if candidates.empty:
                st.error("رقم الهاتف موش موجود في Trainees. الموظف لازم يسجل نفس الرقم.")
                return

            trainee_id = candidates.iloc[0]["trainee_id"]

            append_row("Accounts", {
                "phone": phone_n,
                "password": norm(pwd),
                "trainee_id": norm(trainee_id),
                "student_name": norm(student_name),
                "created_at": now_str(),
                "last_login": ""
            })
            st.success("✅ Compte créé. امشي Connexion.")

    # ---------------- My space
    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = normalize_phone(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = read_df("Trainees")
        row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not tr.empty else pd.DataFrame()
        if row.empty:
            st.error("Compte مرتبط بمتربص غير موجود.")
            return

        info = row.iloc[0].to_dict()
        branch = norm(info.get("branch"))
        program = norm(info.get("program"))
        group = norm(info.get("group"))
        full_name = norm(info.get("full_name"))

        c1, c2 = st.columns([1, 3])
        with c1:
            pic = get_profile_pic_bytes(phone)
            if pic:
                st.image(pic, caption="Photo", use_container_width=True)
            else:
                st.info("Pas de photo")

        with c2:
            st.success(f"Bienvenue {student_name or full_name} ✅")
            st.caption(f"Centre: {branch} | Spécialité: {program} | Groupe: {group} | Tél: {phone}")

            up = st.file_uploader("📸 Ajouter/Changer ma photo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="pp_upl")
            if up is not None:
                img_bytes = up.read()
                st.image(img_bytes, caption="Aperçu", width=150)
                if st.button("Enregistrer ma photo", use_container_width=True, key="pp_save"):
                    try:
                        upsert_profile_pic(phone, trainee_id, img_bytes)
                        st.success("✅ Photo enregistrée.")
                        st.rerun()
                    except APIError as e:
                        st.error(explain_api_error(e))

        t1, t2, t3, t4 = st.tabs(["📝 Notes", "🗓️ Planning", "💳 Paiements", "📎 Supports"])

        with t1:
            gr = read_df("Grades")
            grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
            if grf.empty:
                st.info("Aucune note.")
            else:
                if "date" in grf.columns:
                    grf["date_sort"] = grf["date"].astype(str)
                    grf = grf.sort_values(by=["date_sort"], ascending=False)
                st.dataframe(grf[["subject_name", "exam_type", "score", "date", "staff_name", "note"]],
                             use_container_width=True, hide_index=True)

        with t2:
            # show timetable grid
            year_now = str(datetime.now().year)
            # show available years (existing timetable for this group)
            tt_all = read_df("Timetable")
            years = []
            if not tt_all.empty:
                ttf = tt_all[
                    (tt_all["branch"].astype(str).str.strip() == branch) &
                    (tt_all["program"].astype(str).str.strip() == program) &
                    (tt_all["group"].astype(str).str.strip() == group)
                ]
                years = sorted([y for y in ttf.get("year", pd.Series([])).astype(str).str.strip().unique().tolist() if y])
            if not years:
                years = [year_now]

            sel_year = st.selectbox("Année", years, index=min(years.index(year_now), len(years)-1) if year_now in years else 0,
                                    key="stud_tt_year")
            tt = load_timetable(branch, program, group, sel_year)
            st.markdown(timetable_html_grid(tt), unsafe_allow_html=True)

        with t3:
            pay = read_df("Payments")
            if pay.empty:
                st.info("لا توجد بيانات دفوعات.")
            else:
                p2 = pay[pay["trainee_id"].astype(str).str.strip() == trainee_id].copy()
                if p2.empty:
                    st.info("لا توجد بيانات دفوعات لهذا المتكون.")
                else:
                    years = sorted([y for y in p2["year"].astype(str).str.strip().unique().tolist() if y])
                    sel_y = st.selectbox("Année", years, key="stud_pay_year")
                    rowp = p2[p2["year"].astype(str).str.strip() == sel_y].iloc[0].to_dict()

                    show = {mo: (norm(rowp.get(mo)).upper() == "TRUE") for mo in MONTHS}
                    df_show = pd.DataFrame([show])
                    st.dataframe(df_show, use_container_width=True, hide_index=True)

        with t4:
            files = read_df("CourseFiles")
            if files.empty:
                st.info("لا توجد ملفات.")
            else:
                f = files[
                    (files["branch"].astype(str).str.strip() == branch) &
                    (files["program"].astype(str).str.strip() == program) &
                    (files["group"].astype(str).str.strip() == group)
                ].copy()
                if f.empty:
                    st.info("لا توجد ملفات.")
                else:
                    f = f.sort_values(by=["uploaded_at"], ascending=False)
                    for _, r in f.iterrows():
                        subj = norm(r.get("subject_name"))
                        fname = norm(r.get("file_name"))
                        url = norm(r.get("url"))
                        st.markdown(f"**📌 {subj}** — {fname}")
                        if url:
                            st.markdown(f"[👀 Ouvrir]({url})")
                        st.divider()

# =========================================================
# STAFF WORK AREA
# =========================================================
def staff_work_center():
    st.markdown("## 🛠️ Espace Employé")
    if st.session_state.role != "staff":
        st.info("Connexion Employé من اليسار.")
        return

    staff_branch = norm(st.session_state.user.get("branch"))
    staff_name = f"Staff-{staff_branch}"
    st.success(f"Centre: {staff_branch}")

    prog_df = df_filter(read_df("Programs"), branch=staff_branch)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not prog_df.empty else prog_df
    programs = sorted([x for x in prog_df.get("program_name", pd.Series([])).astype(str).str.strip().tolist() if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not grp_df.empty else grp_df
            groups = sorted([x for x in grp_df.get("group_name", pd.Series([])).astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox("Année", [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)],
                            index=1, key="year_main")

    t_programs, t_groups, t_subjects, t_stag, t_gr, t_pay, t_tt, t_files = st.tabs(
        ["🏷️ Spécialités", "👥 Groupes", "📚 Matières", "👤 Stagiaires", "📝 Notes", "💳 Paiements", "🗓️ Planning", "📎 Supports"]
    )

    # ---------- Programs
    with t_programs:
        cur = df_filter(read_df("Programs"), branch=staff_branch)
        st.dataframe(cur[["program_name", "is_active", "created_at"]] if not cur.empty else cur,
                     use_container_width=True, hide_index=True)
        new_prog = st.text_input("Nouvelle spécialité", key="new_prog")
        if st.button("Ajouter spécialité", use_container_width=True, key="btn_add_prog"):
            if not norm(new_prog):
                st.error("Nom obligatoire.")
            else:
                append_row("Programs", {
                    "program_id": f"PR-{uuid.uuid4().hex[:8].upper()}",
                    "branch": staff_branch,
                    "program_name": norm(new_prog),
                    "is_active": "true",
                    "created_at": now_str()
                })
                st.success("✅ Ajouté.")
                st.rerun()

    # ---------- Groups
    with t_groups:
        if not program:
            st.info("اختار Spécialité من فوق.")
        else:
            cur = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            st.dataframe(cur[["group_name", "is_active", "created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)
            new_group = st.text_input("Nouveau groupe", key="new_group")
            if st.button("Ajouter groupe", use_container_width=True, key="btn_add_group"):
                if not norm(new_group):
                    st.error("Nom obligatoire.")
                else:
                    append_row("Groups", {
                        "group_id": f"GP-{uuid.uuid4().hex[:8].upper()}",
                        "branch": staff_branch,
                        "program_name": norm(program),
                        "group_name": norm(new_group),
                        "is_active": "true",
                        "created_at": now_str()
                    })
                    st.success("✅ Ajouté.")
                    st.rerun()

    # ---------- Subjects
    with t_subjects:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            cur = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            st.dataframe(cur[["subject_name", "is_active", "created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)
            subject_name = st.text_input("Nouvelle matière", key="new_subject")
            if st.button("Ajouter matière", use_container_width=True, key="btn_add_subject"):
                if not norm(subject_name):
                    st.error("Nom obligatoire.")
                else:
                    append_row("Subjects", {
                        "subject_id": f"SB-{uuid.uuid4().hex[:8].upper()}",
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "subject_name": norm(subject_name),
                        "is_active": "true",
                        "created_at": now_str()
                    })
                    st.success("✅ Ajouté.")
                    st.rerun()

    # ---------- Trainees
    with t_stag:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            cur = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            st.dataframe(cur[["full_name", "phone", "status", "created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)

            st.markdown("### ➕ Ajouter stagiaire")
            name = st.text_input("Nom & Prénom", key="add_tr_name")
            phone = st.text_input("Téléphone (obligatoire)", key="add_tr_phone")
            status = st.selectbox("Statut", ["active", "inactive"], key="add_tr_status")

            if st.button("Enregistrer stagiaire", use_container_width=True, key="btn_add_tr"):
                ph = normalize_phone(phone)
                if not norm(name) or not ph:
                    st.error("Nom + téléphone obligatoire.")
                else:
                    # prevent duplicate phone within same branch/program/group
                    existing = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
                    if not existing.empty:
                        ephones = existing["phone"].astype(str).apply(normalize_phone)
                        if ephones.eq(ph).any():
                            st.error("هذا الرقم موجود من قبل في نفس المجموعة.")
                            return

                    append_row("Trainees", {
                        "trainee_id": f"TR-{uuid.uuid4().hex[:8].upper()}",
                        "full_name": norm(name),
                        "phone": ph,
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "status": status,
                        "created_at": now_str()
                    })
                    st.success("✅ Ajouté.")
                    st.rerun()

            st.divider()
            st.markdown("### 📥 Import Excel (xlsx) : full_name + phone")
            up = st.file_uploader("Uploader Excel", type=["xlsx"], key="excel_tr")
            if up is not None:
                df = pd.read_excel(up)
                df.columns = [str(c).strip() for c in df.columns]
                st.dataframe(df.head(30), use_container_width=True)

                if st.button("✅ Importer maintenant", use_container_width=True, key="do_imp"):
                    if "full_name" not in df.columns or "phone" not in df.columns:
                        st.error("لازم الأعمدة: full_name و phone.")
                    else:
                        existing = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
                        existing_phones = set(existing["phone"].astype(str).apply(normalize_phone).tolist()) if not existing.empty else set()

                        count = 0
                        for _, r in df.iterrows():
                            fn = norm(r.get("full_name"))
                            ph = normalize_phone(r.get("phone"))
                            if not fn or not ph:
                                continue
                            if ph in existing_phones:
                                continue
                            append_row("Trainees", {
                                "trainee_id": f"TR-{uuid.uuid4().hex[:8].upper()}",
                                "full_name": fn,
                                "phone": ph,
                                "branch": staff_branch,
                                "program": norm(program),
                                "group": norm(group),
                                "status": "active",
                                "created_at": now_str(),
                            })
                            existing_phones.add(ph)
                            count += 1
                        st.success(f"✅ Import terminé: {count}")
                        st.rerun()

    # ---------- Grades
    with t_gr:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)

            if tr.empty:
                st.warning("لا يوجد stagiaires.")
            elif sub.empty:
                st.warning("زيد matières قبل.")
            else:
                tr = tr.copy()
                tr["phone_norm"] = tr["phone"].apply(normalize_phone)
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone_norm"].astype(str) + " — " + tr["trainee_id"].astype(str)

                chosen = st.selectbox("Stagiaire", tr["label"].tolist(), key="gr_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x])
                subject_name = st.selectbox("Matière", subjects, key="gr_subject")
                exam_type = st.text_input("Type examen (DS1/TP/Examen...)", key="gr_examtype")
                score = st.number_input("Note", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="gr_score")
                dt = st.date_input("Date", value=date.today(), key="gr_date")
                note = st.text_area("Remarque", key="gr_note")

                if st.button("✅ Enregistrer la note", use_container_width=True, key="btn_save_grade"):
                    if not norm(exam_type):
                        st.error("Type examen obligatoire.")
                    else:
                        append_row("Grades", {
                            "grade_id": f"GR-{uuid.uuid4().hex[:8].upper()}",
                            "trainee_id": trainee_id,
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "subject_name": norm(subject_name),
                            "exam_type": norm(exam_type),
                            "score": str(score),
                            "date": str(dt),
                            "staff_name": staff_name,
                            "note": norm(note),
                            "created_at": now_str(),
                        })
                        st.success("✅ Note enregistrée.")
                        st.rerun()

    # ---------- Payments
    with t_pay:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr = tr.copy()
                tr["phone_norm"] = tr["phone"].apply(normalize_phone)
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone_norm"].astype(str) + " — " + tr["trainee_id"].astype(str)

                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="pay_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                ensure_payment_row(trainee_id, staff_branch, program, group, year, staff_name)

                pay = read_df("Payments")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
                        (pay["year"].astype(str).str.strip() == norm(year))].copy()

                rowp = m.iloc[0].to_dict() if not m.empty else {}
                st.caption("✅ علّم الأشهر اللي خلّصهم (يولي TRUE).")

                cols = st.columns(4)
                for i, mo in enumerate(MONTHS):
                    paid = (norm(rowp.get(mo)).upper() == "TRUE")
                    with cols[i % 4]:
                        new_paid = st.checkbox(mo, value=paid, key=f"ck_{mo}_{trainee_id}_{year}")
                        if new_paid != paid:
                            set_payment_month(trainee_id, year, mo, new_paid, staff_name)
                            st.rerun()

    # ---------- Timetable (CRUD + colors)
    with t_tt:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            st.markdown("### 🗓️ Planning (الموظف يكتب ويعدّل ويفسخ)")

            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            subjects = sorted([x for x in sub.get("subject_name", pd.Series([])).astype(str).str.strip().tolist() if x]) if not sub.empty else []

            tt = load_timetable(staff_branch, program, group, year)

            # Preview grid
            st.markdown("#### 👀 Aperçu (ملوّن)")
            st.markdown(timetable_html_grid(tt), unsafe_allow_html=True)

            st.divider()
            st.markdown("#### ➕ Ajouter une séance")
            if not subjects:
                st.warning("زيد matières (tab Matières) قبل ما تعمل Planning.")
            else:
                c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
                with c1:
                    day = st.selectbox("Jour", DAYS, key="tt_day_add")
                with c2:
                    start = st.text_input("De (HH:MM)", value="18:00", key="tt_start_add")
                with c3:
                    end = st.text_input("À (HH:MM)", value="19:30", key="tt_end_add")
                with c4:
                    color = st.color_picker("Couleur", value="#E6F2FF", key="tt_color_add")

                c5, c6, c7 = st.columns([2, 1, 2])
                with c5:
                    subject_name = st.selectbox("Matière", subjects, key="tt_subject_add")
                with c6:
                    room = st.text_input("Salle", key="tt_room_add")
                with c7:
                    teacher = st.text_input("Nom du formateur", key="tt_teacher_add")

                note = st.text_input("Note (optionnel)", key="tt_note_add")

                if st.button("✅ Ajouter au planning", use_container_width=True, key="tt_add_btn"):
                    if not norm(start) or not norm(end) or not norm(teacher):
                        st.error("Start/End + Nom formateur obligatoire.")
                    else:
                        add_timetable_row(
                            staff_branch, program, group, year,
                            day=day,
                            start=norm(start),
                            end=norm(end),
                            subject_name=norm(subject_name),
                            teacher=norm(teacher),
                            color=norm(color),
                            room=norm(room),
                            note=norm(note),
                            staff_name=staff_name
                        )
                        st.success("✅ Ajouté.")
                        st.rerun()

            st.divider()
            st.markdown("#### ✏️ Modifier / 🗑️ Supprimer")

            tt = load_timetable(staff_branch, program, group, year)
            if tt.empty:
                st.info("لا توجد حصص في هذا العام.")
            else:
                # build options
                tt2 = tt.copy()
                tt2["label"] = (
                    tt2["day"].astype(str) + " | " +
                    tt2["start"].astype(str) + "-" + tt2["end"].astype(str) + " | " +
                    tt2["subject_name"].astype(str) + " | " +
                    tt2["teacher"].astype(str) + " | " +
                    tt2["row_id"].astype(str)
                )
                chosen = st.selectbox("اختار séance", tt2["label"].tolist(), key="tt_pick")
                row_id = tt2[tt2["label"] == chosen].iloc[0]["row_id"]
                row = tt2[tt2["row_id"] == row_id].iloc[0].to_dict()

                e1, e2, e3, e4 = st.columns([1.2, 1, 1, 1])
                with e1:
                    day_e = st.selectbox("Jour", DAYS, index=DAYS.index(norm(row.get("day")) if norm(row.get("day")) in DAYS else "Monday"), key="tt_day_edit")
                with e2:
                    start_e = st.text_input("De (HH:MM)", value=norm(row.get("start")), key="tt_start_edit")
                with e3:
                    end_e = st.text_input("À (HH:MM)", value=norm(row.get("end")), key="tt_end_edit")
                with e4:
                    color_e = st.color_picker("Couleur", value=norm(row.get("color")) or "#E6F2FF", key="tt_color_edit")

                e5, e6, e7 = st.columns([2, 1, 2])
                with e5:
                    if subjects:
                        # fallback if old subject missing
                        cur_sub = norm(row.get("subject_name"))
                        if cur_sub not in subjects:
                            subjects2 = subjects + [cur_sub]
                        else:
                            subjects2 = subjects
                        subject_e = st.selectbox("Matière", subjects2, index=subjects2.index(cur_sub) if cur_sub in subjects2 else 0, key="tt_subject_edit")
                    else:
                        subject_e = st.text_input("Matière", value=norm(row.get("subject_name")), key="tt_subject_edit_txt")
                with e6:
                    room_e = st.text_input("Salle", value=norm(row.get("room")), key="tt_room_edit")
                with e7:
                    teacher_e = st.text_input("Nom du formateur", value=norm(row.get("teacher")), key="tt_teacher_edit")

                note_e = st.text_input("Note (optionnel)", value=norm(row.get("note")), key="tt_note_edit")

                cbtn1, cbtn2 = st.columns(2)
                with cbtn1:
                    if st.button("✅ Sauvegarder modification", use_container_width=True, key="tt_save_edit"):
                        ok = update_timetable_row(row_id, {
                            "day": day_e,
                            "start": start_e,
                            "end": end_e,
                            "subject_name": subject_e,
                            "teacher": teacher_e,
                            "color": color_e,
                            "room": room_e,
                            "note": note_e,
                            "staff_name": staff_name,
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "year": norm(year),
                        })
                        if ok:
                            st.success("✅ Modifié.")
                            st.rerun()
                        else:
                            st.error("❌ ما لقيتش row باش نعدّلها.")
                with cbtn2:
                    if st.button("🗑️ Supprimer هذه séance", use_container_width=True, key="tt_delete"):
                        ok = delete_timetable_row(row_id)
                        if ok:
                            st.success("✅ Supprimé.")
                            st.rerun()
                        else:
                            st.error("❌ ما لقيتش row باش نفسخها.")

    # ---------- Supports (Links)
    with t_files:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            st.markdown("### 📎 Supports (Links)")
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            subjects = sorted([x for x in sub.get("subject_name", pd.Series([])).astype(str).str.strip().tolist() if x]) if not sub.empty else []
            if not subjects:
                st.warning("زيد matières قبل.")
            else:
                subj = st.selectbox("Matière", subjects, key="cf_subj")
                fname = st.text_input("Nom du fichier", key="cf_name")
                link = st.text_input("Lien (Drive/Docs/URL)", key="cf_link")

                if st.button("✅ Enregistrer fichier", use_container_width=True, key="cf_save"):
                    if not norm(link) or not norm(fname):
                        st.error("لازم اسم ملف + رابط.")
                    else:
                        append_row("CourseFiles", {
                            "file_id": f"CF-{uuid.uuid4().hex[:8].upper()}",
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "subject_name": norm(subj),
                            "file_name": norm(fname),
                            "url": norm(link),
                            "uploaded_at": now_str(),
                            "staff_name": staff_name,
                        })
                        st.success("✅ Fichier enregistré.")
                        st.rerun()

            files = read_df("CourseFiles")
            files = files[
                (files["branch"].astype(str).str.strip() == staff_branch) &
                (files["program"].astype(str).str.strip() == norm(program)) &
                (files["group"].astype(str).str.strip() == norm(group))
            ].copy() if not files.empty else pd.DataFrame()

            if not files.empty:
                st.divider()
                st.markdown("### Fichiers enregistrés")
                files = files.sort_values(by=["uploaded_at"], ascending=False)
                st.dataframe(files[["subject_name", "file_name", "url", "uploaded_at", "staff_name"]],
                             use_container_width=True, hide_index=True)

                st.caption("🗑️ Supprimer fichier")
                opts = (files["file_name"].astype(str) + " | " + files["file_id"].astype(str)).tolist()
                pick = st.selectbox("اختار ملف", opts, key="cf_del_pick")
                file_id = pick.split("|")[-1].strip() if pick else ""
                if st.button("🗑️ Supprimer", use_container_width=True, key="cf_del_btn"):
                    if file_id and delete_row_by_key("CourseFiles", "file_id", file_id):
                        st.success("✅ Supprimé.")
                        st.rerun()
                    else:
                        st.error("❌ ما لقيتش الملف.")

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_session()
    ensure_schema_once()
    sidebar_staff_login()

    if st.session_state.role == "staff":
        staff_work_center()
        st.divider()
        student_portal_center()
    else:
        # only student in center
        student_portal_center()
        st.divider()
        st.info("ℹ️ Connexion Employé موجودة في اليسار.")

if __name__ == "__main__":
    main()
