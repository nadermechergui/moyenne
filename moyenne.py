```python
# =========================================================
# Portail Mega Formation (FINAL++) - Staff can EDIT/DELETE everything
# Sheets Only + Drive Links (manual)
# - Planning: colored timetable (staff add/edit/delete rows)
# - Supports: Drive links (staff add/edit/delete rows)
# - Grades: staff add/edit/delete rows
# - Trainees: staff add/edit/delete rows
# - Payments: staff toggle months (and can delete a payment row)
# - Student sees: photo + colored timetable + payments by year + supports
# - Fixes: link_button compatibility + unique keys + cache clear
# =========================================================

import uuid
import base64
import io
import re
from datetime import datetime

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
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],

    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "student_name", "created_at", "last_login"],

    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group",
               "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],

    # Profile pics (small base64)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    # Payments
    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"] + MONTHS + ["updated_at", "staff_name"],

    # Supports: Drive manual links
    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "drive_view_url", "drive_download_url", "uploaded_at", "staff_name"],

    # Timetable (colored): staff writes schedule
    "Timetable": [
        "tt_id", "branch", "program", "group", "year",
        "day", "start_time", "end_time",
        "subject_name", "teacher_name",
        "color", "created_at", "staff_name"
    ],
}


# =========================================================
# UTILS
# =========================================================
def norm(x):
    return str(x or "").strip()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def explain_api_error(e: APIError) -> str:
    try:
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", "") or ""
        low = text.lower()
        if status == 429 or "quota" in low:
            return "⚠️ 429 Quota (Google Sheets). جرّب Reboot واستنى شوية.\n" + text[:260]
        if status == 403 or "permission" in low or "forbidden" in low:
            return "❌ 403 Permission. Share Sheet مع service account.\n" + text[:260]
        if status == 404 or "not found" in low:
            return "❌ 404 Not found. تأكد GSHEET_ID صحيح + Share للـ service account.\n" + text[:260]
        return "❌ Google API Error:\n" + (text[:400] if text else str(e))
    except Exception:
        return "❌ Google API Error."

def df_filter(df: pd.DataFrame, **kwargs):
    out = df.copy()
    for k, v in kwargs.items():
        if k in out.columns:
            out = out[out[k].astype(str).str.strip() == norm(v)]
    return out

def safe_link_button(label: str, url: str, *, key: str, use_container_width: bool = True):
    u = norm(url)
    if not u:
        st.button(label, disabled=True, use_container_width=use_container_width, key=key)
        return
    try:
        if hasattr(st, "link_button"):
            st.link_button(label, u, use_container_width=use_container_width, key=key)
        else:
            st.markdown(f"**{label}:** [{u}]({u})")
    except Exception:
        st.markdown(f"**{label}:** [{u}]({u})")

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

def normalize_color(c: str) -> str:
    c = norm(c)
    if not c:
        return "#E5E7EB"
    if re.match(r"^#[0-9A-Fa-f]{6}$", c):
        return c
    if re.match(r"^[0-9A-Fa-f]{6}$", c):
        return "#" + c
    return "#E5E7EB"

def parse_time_hhmm(t: str) -> tuple[int, int] | None:
    t = norm(t)
    m = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return hh, mm

def minutes_of(t: str) -> int | None:
    p = parse_time_hhmm(t)
    if not p:
        return None
    return p[0] * 60 + p[1]


# --- Drive link helpers (manual) ---
def extract_drive_file_id(url: str) -> str | None:
    u = norm(url)
    if not u:
        return None
    if "/file/d/" in u:
        try:
            return u.split("/file/d/")[1].split("/")[0]
        except Exception:
            return None
    if "open?id=" in u:
        try:
            return u.split("open?id=")[1].split("&")[0]
        except Exception:
            return None
    if "uc?id=" in u:
        try:
            return u.split("uc?id=")[1].split("&")[0]
        except Exception:
            return None
    return None

def to_view_and_download(url: str) -> tuple[str, str]:
    fid = extract_drive_file_id(url)
    if not fid:
        return norm(url), norm(url)
    view_url = f"https://drive.google.com/file/d/{fid}/view"
    dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
    return view_url, dl_url


# =========================================================
# AUTH CLIENTS
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
# SHEETS SETUP (SAFE: no clear)
# =========================================================
def ensure_headers_safe(ws, headers: list[str]):
    rng = ws.get("1:1")
    row1 = rng[0] if (rng and len(rng) > 0) else []
    row1 = [norm(x) for x in row1]

    if len(row1) == 0 or all(x == "" for x in row1):
        ws.append_row(headers, value_input_option="RAW")
        return

    if row1 != headers:
        st.warning(f"⚠️ Sheet '{ws.title}' headers مختلفة. ما عملتش مسح. صحّح الهيدرز يدويًا إذا تحب.")

def ensure_worksheets_and_headers():
    sh = spreadsheet()
    titles = [w.title for w in sh.worksheets()]
    for ws_name, headers in REQUIRED_SHEETS.items():
        if ws_name not in titles:
            sh.add_worksheet(title=ws_name, rows=3000, cols=max(16, len(headers) + 2))
            titles.append(ws_name)
        ws = sh.worksheet(ws_name)
        ensure_headers_safe(ws, headers)

def ensure_schema_once():
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

@st.cache_data(ttl=300, show_spinner=False)
def read_df(ws_name: str) -> pd.DataFrame:
    ws = spreadsheet().worksheet(ws_name)
    values = ws.get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

def append_row(ws_name: str, row: dict):
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_row_by_id(ws_name: str, id_col: str, id_val: str, updates: dict) -> bool:
    df = read_df(ws_name)
    if df.empty or id_col not in df.columns:
        return False
    df[id_col] = df[id_col].astype(str).str.strip()
    m = df[df[id_col] == norm(id_val)]
    if m.empty:
        return False
    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    for k, v in updates.items():
        if k in headers:
            ws.update_cell(row_num, headers.index(k) + 1, norm(v))
    st.cache_data.clear()
    return True

def delete_row_by_id(ws_name: str, id_col: str, id_val: str) -> bool:
    df = read_df(ws_name)
    if df.empty or id_col not in df.columns:
        return False
    df[id_col] = df[id_col].astype(str).str.strip()
    m = df[df[id_col] == norm(id_val)]
    if m.empty:
        return False
    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet(ws_name)
    ws.delete_rows(row_num)
    st.cache_data.clear()
    return True


# =========================================================
# PROFILE PICS
# =========================================================
def get_profile_pic_bytes(phone: str) -> bytes | None:
    df = read_df("ProfilePics")
    if df.empty or "phone" not in df.columns:
        return None
    df["phone"] = df["phone"].astype(str).str.strip()
    m = df[df["phone"] == norm(phone)]
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
    small = compress_image_bytes(img_bytes, max_side=256, quality=70)
    b64 = base64.b64encode(small).decode("utf-8")

    # upsert by phone
    df = read_df("ProfilePics")
    if not df.empty and "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).str.strip()
        m = df[df["phone"] == norm(phone)]
        if not m.empty:
            pid = m.iloc[0].get("phone")
            ok = update_row_by_id("ProfilePics", "phone", pid, {
                "trainee_id": trainee_id,
                "image_b64": b64,
                "uploaded_at": now_str()
            })
            if ok:
                return

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
    if not df.empty and {"trainee_id","year"}.issubset(set(df.columns)):
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
    if df.empty or not {"trainee_id","year"}.issubset(set(df.columns)):
        return False
    m = df[(df["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
           (df["year"].astype(str).str.strip() == norm(year))]
    if m.empty:
        return False

    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet("Payments")
    headers = REQUIRED_SHEETS["Payments"]

    if month not in headers:
        return False

    ws.update_cell(row_num, headers.index(month) + 1, "TRUE" if paid else "FALSE")
    ws.update_cell(row_num, headers.index("updated_at") + 1, now_str())
    ws.update_cell(row_num, headers.index("staff_name") + 1, staff_name)

    st.cache_data.clear()
    return True

def available_payment_years_for_trainee(trainee_id: str) -> list[str]:
    pay = read_df("Payments")
    if pay.empty or not {"trainee_id","year"}.issubset(set(pay.columns)):
        return []
    pay["trainee_id"] = pay["trainee_id"].astype(str).str.strip()
    pay["year"] = pay["year"].astype(str).str.strip()
    ys = pay[pay["trainee_id"] == norm(trainee_id)]["year"].dropna().tolist()
    ys = [y for y in ys if norm(y)]
    return sorted(list(set(ys)))


# =========================================================
# TIMETABLE (Colored)
# =========================================================
def load_timetable(branch: str, program: str, group: str, year: str) -> pd.DataFrame:
    df = read_df("Timetable")
    if df.empty:
        return df
    needed = ["branch","program","group","year","day","start_time","end_time"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame(columns=REQUIRED_SHEETS["Timetable"])
        df[c] = df[c].astype(str).str.strip()

    out = df[
        (df["branch"] == norm(branch)) &
        (df["program"] == norm(program)) &
        (df["group"] == norm(group)) &
        (df["year"] == norm(year))
    ].copy()

    # sort by start_time
    if not out.empty and "start_time" in out.columns:
        out["_m"] = out["start_time"].astype(str).apply(lambda x: minutes_of(x) if minutes_of(x) is not None else 9999)
        out = out.sort_values(by=["day","_m"], ascending=True).drop(columns=["_m"])
    return out

def render_timetable_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div style='padding:10px;border:1px solid #eee;border-radius:8px'>لا يوجد Planning بعد.</div>"

    df = df.copy()
    for c in ["day","start_time","end_time","subject_name","teacher_name","color"]:
        if c not in df.columns:
            return "<div style='padding:10px;border:1px solid #eee;border-radius:8px'>Planning غير مهيكل (columns ناقصة).</div>"

    df["slot"] = df["start_time"].astype(str).str.strip() + " - " + df["end_time"].astype(str).str.strip()
    df["color"] = df["color"].astype(str).apply(normalize_color)

    slots = df[["start_time","end_time","slot"]].drop_duplicates()
    slots["_m"] = slots["start_time"].astype(str).apply(lambda x: minutes_of(x) if minutes_of(x) is not None else 9999)
    slots = slots.sort_values(by=["_m","end_time"], ascending=True).drop(columns=["_m"])
    slot_list = slots["slot"].tolist()

    cell = {}
    color_map = {}
    for _, r in df.iterrows():
        d = norm(r.get("day"))
        s = norm(r.get("slot"))
        subj = norm(r.get("subject_name"))
        teacher = norm(r.get("teacher_name"))
        bg = normalize_color(r.get("color"))
        text = f"<div style='font-weight:700'>{subj}</div><div style='font-size:12px;opacity:.9'>{teacher}</div>"
        cell[(s, d)] = text
        color_map[(s, d)] = bg

    html = """
    <style>
      .mf-table{border-collapse:collapse;width:100%;font-family:Arial,sans-serif}
      .mf-table th,.mf-table td{border:1px solid #e5e7eb;padding:10px;vertical-align:top}
      .mf-table th{background:#f9fafb;text-align:center;font-weight:700}
      .mf-slot{white-space:nowrap;font-weight:700;background:#fff}
      .mf-empty{background:#fff}
    </style>
    <table class='mf-table'>
    """
    html += "<tr><th style='text-align:left'>Heure</th>" + "".join([f"<th>{d}</th>" for d in DAYS]) + "</tr>"

    for s in slot_list:
        html += f"<tr><td class='mf-slot'>{s}</td>"
        for d in DAYS:
            v = cell.get((s, d), "")
            if not v:
                html += "<td class='mf-empty'></td>"
            else:
                bg = color_map.get((s, d), "#E5E7EB")
                html += f"<td style='background:{bg}'>{v}</td>"
        html += "</tr>"

    html += "</table>"
    return html


# =========================================================
# AUTH / SESSION
# =========================================================
def ensure_session():
    st.session_state.setdefault("role", None)   # staff | None
    st.session_state.setdefault("user", {})     # staff session
    st.session_state.setdefault("student", None)

def logout_staff():
    st.session_state.role = None
    st.session_state.user = {}

def staff_branch_login(branch: str, branch_password: str):
    df = read_df("Branches")
    if df.empty:
        return None
    for c in ["branch","staff_password","is_active"]:
        if c not in df.columns:
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
    df = read_df("Accounts")
    if df.empty:
        return None
    for c in ["phone","password"]:
        if c not in df.columns:
            return None
    df2 = df.copy()
    df2["phone"] = df2["phone"].astype(str).str.strip()
    df2["password"] = df2["password"].astype(str).str.strip()
    m = df2[(df2["phone"] == norm(phone)) & (df2["password"] == norm(password))]
    if m.empty:
        return None
    return m.iloc[0].to_dict()


# =========================================================
# SIDEBAR STAFF LOGIN
# =========================================================
def sidebar_staff_login():
    st.sidebar.markdown("## 👨‍💼 Connexion Employé")
    branches_df = read_df("Branches")
    branches = []
    if not branches_df.empty and "branch" in branches_df.columns:
        branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x])

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
        st.sidebar.warning("Branches vide. زِد Branches في Sheet 'Branches'.")
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
# STAFF CRUD HELPERS (UI)
# =========================================================
def crud_section(title: str):
    st.markdown(f"### {title}")
    st.caption("✏️ تعديل / 🗑️ حذف")


# =========================================================
# STUDENT PORTAL
# =========================================================
def student_portal_center():
    st.markdown("## 🎓 Espace Stagiaire")
    tab1, tab2, tab3 = st.tabs(["🔐 Connexion", "🆕 Inscription", "📌 Mon espace"])

    # Login
    with tab1:
        phone = st.text_input("Téléphone", key="stud_phone")
        pwd = st.text_input("Mot de passe", type="password", key="stud_pwd")

        if st.button("Se connecter", use_container_width=True, key="btn_stud_login"):
            acc = student_login(phone, pwd)
            if acc:
                # last_login update (best effort)
                try:
                    update_row_by_id("Accounts", "phone", phone, {"last_login": now_str()})
                except Exception:
                    pass
                st.session_state.student = acc
                st.success("✅ Connexion réussie")
            else:
                st.error("Téléphone / mot de passe incorrect.")

        if st.button("Se déconnecter (Stagiaire)", use_container_width=True, key="btn_stud_logout"):
            st.session_state.student = None
            st.rerun()

    # Registration
    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")
        branches_df = read_df("Branches")
        branches = []
        if not branches_df.empty and "branch" in branches_df.columns:
            branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x])
        if not branches:
            st.warning("Aucun centre.")
            return

        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(read_df("Programs"), branch=b)
        if not prog_df.empty and "is_active" in prog_df.columns:
            prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in (prog_df["program_name"].astype(str).str.strip().tolist()
                                       if (not prog_df.empty and "program_name" in prog_df.columns) else []) if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_prog")

        grp_df = df_filter(read_df("Groups"), branch=b, program_name=p)
        if not grp_df.empty and "is_active" in grp_df.columns:
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        groups = sorted([x for x in (grp_df["group_name"].astype(str).str.strip().tolist()
                                     if (not grp_df.empty and "group_name" in grp_df.columns) else []) if x])
        if not groups:
            st.warning("Aucun groupe.")
            return
        g = st.selectbox("Groupe", groups, key="reg_group")

        student_name = st.text_input("Nom (أي اسم تحب)", key="reg_name")
        phone = st.text_input("Téléphone (نفس رقمك عند الإدارة)", key="reg_phone")
        pwd = st.text_input("Mot de passe", type="password", key="reg_pwd")

        if st.button("Créer mon compte", use_container_width=True, key="btn_register"):
            if not norm(student_name) or not norm(phone) or not norm(pwd):
                st.error("Nom + téléphone + mot de passe obligatoire.")
                return
            if len(norm(pwd)) < 4:
                st.error("Mot de passe قصير (min 4).")
                return

            acc = read_df("Accounts")
            if not acc.empty and "phone" in acc.columns:
                if acc["phone"].astype(str).str.strip().eq(norm(phone)).any():
                    st.error("Ce téléphone est déjà inscrit.")
                    return

            tr = read_df("Trainees")
            if tr.empty or "phone" not in tr.columns:
                st.error("Aucun stagiaire.")
                return

            tr2 = tr.copy()
            for c in ["branch", "program", "group", "phone"]:
                if c in tr2.columns:
                    tr2[c] = tr2[c].astype(str).str.strip()

            candidates = tr2[
                (tr2.get("branch", "") == norm(b)) &
                (tr2.get("program", "") == norm(p)) &
                (tr2.get("group", "") == norm(g)) &
                (tr2.get("phone", "") == norm(phone))
            ]

            if candidates.empty:
                st.error("رقم الهاتف موش موجود في Trainees. الموظف لازم يسجل نفس الرقم.")
                return

            trainee_id = candidates.iloc[0].get("trainee_id")

            append_row("Accounts", {
                "phone": norm(phone),
                "password": norm(pwd),
                "trainee_id": norm(trainee_id),
                "student_name": norm(student_name),
                "created_at": now_str(),
                "last_login": ""
            })
            st.success("✅ Compte créé. امشي Connexion.")

    # My Space
    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion باش تشوف الصفحة متاعك.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = read_df("Trainees")
        if tr.empty or "trainee_id" not in tr.columns:
            st.error("Trainees sheet فارغ/ناقص.")
            return
        row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy()
        if row.empty:
            st.error("Compte مرتبط بمتربص غير موجود.")
            return

        info = row.iloc[0].to_dict()
        branch = norm(info.get("branch"))
        program = norm(info.get("program"))
        group = norm(info.get("group"))

        c1, c2 = st.columns([1, 3])
        with c1:
            try:
                pic = get_profile_pic_bytes(phone)
                if pic:
                    st.image(pic, caption="Photo", use_container_width=True)
                else:
                    st.info("Pas de photo")
            except APIError as e:
                st.warning(explain_api_error(e))
                st.info("Pas de photo (Quota).")

        with c2:
            st.success(f"Bienvenue {student_name or norm(info.get('full_name'))} ✅")
            st.caption(f"Centre: {branch} | Spécialité: {program} | Groupe: {group} | Tél: {phone}")

            up = st.file_uploader("📸 Ajouter/Changer ma photo (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                  key=f"pp_upl_{trainee_id}")
            if up is not None:
                img_bytes = up.read()
                st.image(img_bytes, caption="Aperçu", width=150)
                if st.button("Enregistrer ma photo", use_container_width=True, key=f"pp_save_{trainee_id}"):
                    upsert_profile_pic(phone, trainee_id, img_bytes)
                    st.success("✅ Photo enregistrée.")
                    st.rerun()

        t1, t2, t3, t4 = st.tabs(["📝 Notes", "🗓️ Planning", "💳 Paiements", "📎 Supports"])

        with t1:
            gr = read_df("Grades")
            if gr.empty or "trainee_id" not in gr.columns:
                st.info("Aucune note.")
            else:
                grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy()
                if grf.empty:
                    st.info("Aucune note.")
                else:
                    cols = [c for c in ["subject_name", "exam_type", "score", "date", "staff_name", "note"] if c in grf.columns]
                    st.dataframe(grf[cols], use_container_width=True, hide_index=True)

        with t2:
            st.markdown("### 📅 Planning (ملوّن)")
            year_now = str(datetime.now().year)
            tt_all = read_df("Timetable")
            years_tt = []
            if not tt_all.empty and {"branch","program","group","year"}.issubset(set(tt_all.columns)):
                ttf = tt_all[
                    (tt_all["branch"].astype(str).str.strip() == branch) &
                    (tt_all["program"].astype(str).str.strip() == program) &
                    (tt_all["group"].astype(str).str.strip() == group)
                ].copy()
                if not ttf.empty:
                    years_tt = sorted(list(set(ttf["year"].astype(str).str.strip().tolist())))
            if not years_tt:
                years_tt = [year_now]

            y = st.selectbox("Année", years_tt, index=(years_tt.index(year_now) if year_now in years_tt else 0),
                             key=f"stud_tt_year_{trainee_id}")
            df_tt = load_timetable(branch, program, group, y)
            st.markdown(render_timetable_html(df_tt), unsafe_allow_html=True)

        with t3:
            years = available_payment_years_for_trainee(trainee_id)
            if not years:
                years = [str(datetime.now().year)]
            y = st.selectbox("Année", years,
                             index=(years.index(str(datetime.now().year)) if str(datetime.now().year) in years else 0),
                             key=f"stud_pay_year_{trainee_id}")

            pay = read_df("Payments")
            if pay.empty or not {"trainee_id","year"}.issubset(set(pay.columns)):
                st.info("لا توجد بيانات دفوعات.")
            else:
                m = pay[
                    (pay["trainee_id"].astype(str).str.strip() == trainee_id) &
                    (pay["year"].astype(str).str.strip() == norm(y))
                ]
                if m.empty:
                    st.info("لا توجد بيانات دفوعات لهذه السنة.")
                else:
                    rowp = m.iloc[0].to_dict()
                    show = {mo: (norm(rowp.get(mo)).upper() == "TRUE") for mo in MONTHS}
                    st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

        with t4:
            files = read_df("CourseFiles")
            if files.empty or not {"branch","program","group"}.issubset(set(files.columns)):
                st.info("لا توجد ملفات.")
            else:
                files = files[
                    (files["branch"].astype(str).str.strip() == branch) &
                    (files["program"].astype(str).str.strip() == program) &
                    (files["group"].astype(str).str.strip() == group)
                ].copy()
                if files.empty:
                    st.info("لا توجد ملفات.")
                else:
                    if "uploaded_at" in files.columns:
                        files = files.sort_values(by=["uploaded_at"], ascending=False)
                    for _, r in files.iterrows():
                        fid = norm(r.get("file_id")) or uuid.uuid4().hex
                        st.markdown(f"**📌 {norm(r.get('subject_name'))}** — {norm(r.get('file_name'))}")
                        safe_link_button("👀 Ouvrir", norm(r.get("drive_view_url")),
                                         key=f"stud_view_{trainee_id}_{fid}")
                        safe_link_button("⬇️ Télécharger", norm(r.get("drive_download_url")),
                                         key=f"stud_dl_{trainee_id}_{fid}")
                        st.divider()


# =========================================================
# STAFF AREA (FULL CRUD)
# =========================================================
def staff_work_center():
    st.markdown("## 🛠️ Espace Employé (CRUD)")
    if st.session_state.role != "staff":
        st.info("Connexion Employé من اليسار.")
        return

    staff_branch = norm(st.session_state.user.get("branch"))
    staff_name = f"Staff-{staff_branch}"

    # Programs
    prog_df = df_filter(read_df("Programs"), branch=staff_branch)
    if not prog_df.empty and "is_active" in prog_df.columns:
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in (prog_df["program_name"].astype(str).str.strip().tolist()
                                   if (not prog_df.empty and "program_name" in prog_df.columns) else []) if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            if not grp_df.empty and "is_active" in grp_df.columns:
                grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in (grp_df["group_name"].astype(str).str.strip().tolist()
                                         if (not grp_df.empty and "group_name" in grp_df.columns) else []) if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox("Année", [str(datetime.now().year), str(datetime.now().year + 1), str(datetime.now().year - 1)],
                            key="year_global")

    tab_stag, tab_gr, tab_pay, tab_plan, tab_sup = st.tabs(
        ["👤 Stagiaires (CRUD)", "📝 Notes (CRUD)", "💳 Paiements (CRUD)", "🗓️ Planning (CRUD)", "📎 Supports (CRUD)"]
    )

    # ---------------- Trainees CRUD
    with tab_stag:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            st.markdown("### 👤 Stagiaires")
            cur = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            if not cur.empty:
                show_cols = [c for c in ["trainee_id","full_name","phone","status","created_at"] if c in cur.columns]
                st.dataframe(cur[show_cols], use_container_width=True, hide_index=True)
            else:
                st.info("لا يوجد stagiaires.")

            st.divider()
            st.markdown("### ➕ Ajouter stagiaire")
            name = st.text_input("Nom", key="add_tr_name")
            phone = st.text_input("Téléphone", key="add_tr_phone")
            status = st.selectbox("Statut", ["active", "inactive"], key="add_tr_status")

            if st.button("Enregistrer", use_container_width=True, key="btn_add_tr"):
                if not norm(name) or not norm(phone):
                    st.error("Nom + téléphone obligatoire.")
                else:
                    existing = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
                    if not existing.empty and "phone" in existing.columns:
                        if existing["phone"].astype(str).str.strip().eq(norm(phone)).any():
                            st.error("❌ Téléphone موجود déjà في نفس groupe.")
                            st.stop()

                    append_row("Trainees", {
                        "trainee_id": f"TR-{uuid.uuid4().hex[:8].upper()}",
                        "full_name": norm(name),
                        "phone": norm(phone),
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "status": status,
                        "created_at": now_str(),
                    })
                    st.success("✅ Ajouté.")
                    st.rerun()

            # Edit/Delete trainee
            st.divider()
            crud_section("✏️/🗑️ Modifier أو supprimer stagiaire")
            cur = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            if cur.empty or "trainee_id" not in cur.columns:
                st.info("لا يوجد.")
            else:
                cur["label"] = cur["full_name"].astype(str) + " — " + cur["phone"].astype(str) + " — " + cur["trainee_id"].astype(str)
                chosen = st.selectbox("اختار stagiaire", cur["label"].tolist(), key="tr_edit_pick")
                tid = cur[cur["label"] == chosen].iloc[0]["trainee_id"]
                row = cur[cur["trainee_id"] == tid].iloc[0].to_dict()

                new_name = st.text_input("Nom", value=norm(row.get("full_name")), key="tr_edit_name")
                new_phone = st.text_input("Téléphone", value=norm(row.get("phone")), key="tr_edit_phone")
                new_status = st.selectbox("Statut", ["active","inactive"],
                                          index=(0 if norm(row.get("status")).lower() != "inactive" else 1),
                                          key="tr_edit_status")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save modifications", use_container_width=True, key="tr_save"):
                        # prevent duplicate phone (exclude self)
                        all_tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
                        if not all_tr.empty and "phone" in all_tr.columns and "trainee_id" in all_tr.columns:
                            dup = all_tr[
                                (all_tr["phone"].astype(str).str.strip() == norm(new_phone)) &
                                (all_tr["trainee_id"].astype(str).str.strip() != norm(tid))
                            ]
                            if not dup.empty:
                                st.error("❌ رقم الهاتف موجود عند متكون آخر.")
                                st.stop()

                        ok = update_row_by_id("Trainees", "trainee_id", tid, {
                            "full_name": new_name,
                            "phone": new_phone,
                            "status": new_status,
                        })
                        if ok:
                            st.success("✅ Updated.")
                            st.rerun()
                        else:
                            st.error("Update failed.")
                with c2:
                    if st.button("🗑️ Delete stagiaire", use_container_width=True, key="tr_del"):
                        if delete_row_by_id("Trainees", "trainee_id", tid):
                            st.success("✅ Deleted.")
                            st.rerun()
                        else:
                            st.error("Delete failed.")

    # ---------------- Grades CRUD
    with tab_gr:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)

            if tr.empty:
                st.warning("لا يوجد stagiaires.")
                st.stop()
            if sub.empty or "subject_name" not in sub.columns:
                st.warning("زيد matières في Subjects قبل.")
                st.stop()

            # Add grade
            st.markdown("### ➕ Ajouter note")
            tr = tr.copy()
            tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
            chosen = st.selectbox("Stagiaire", tr["label"].tolist(), key="gr_add_tr")
            trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x])
            subject_name = st.selectbox("Matière", subjects, key="gr_add_subj")
            exam_type = st.text_input("Type examen (DS1/TP/Examen...)", key="gr_add_exam")
            score = st.number_input("Note", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="gr_add_score")
            date = st.date_input("Date", value=datetime.now().date(), key="gr_add_date")
            note = st.text_area("Remarque", key="gr_add_note")

            if st.button("✅ Enregistrer la note", use_container_width=True, key="gr_add_btn"):
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
                        "date": str(date),
                        "staff_name": staff_name,
                        "note": norm(note),
                        "created_at": now_str(),
                    })
                    st.success("✅ Note enregistrée.")
                    st.rerun()

            # Edit/Delete grade
            st.divider()
            crud_section("✏️/🗑️ Modifier أو supprimer note")
            gr = read_df("Grades")
            if gr.empty or "grade_id" not in gr.columns:
                st.info("لا يوجد Notes.")
            else:
                grf = gr[
                    (gr.get("branch","").astype(str).str.strip() == staff_branch) &
                    (gr.get("program","").astype(str).str.strip() == norm(program)) &
                    (gr.get("group","").astype(str).str.strip() == norm(group))
                ] if {"branch","program","group"}.issubset(set(gr.columns)) else pd.DataFrame()

                if grf.empty:
                    st.info("لا يوجد Notes.")
                else:
                    # label include trainee
                    tr_map = {}
                    if "trainee_id" in tr.columns:
                        for _, r in tr.iterrows():
                            tr_map[norm(r["trainee_id"])] = norm(r.get("full_name"))

                    def mk_label(r):
                        return f"{norm(r.get('grade_id'))} | {tr_map.get(norm(r.get('trainee_id')), norm(r.get('trainee_id')))} | {norm(r.get('subject_name'))} | {norm(r.get('date'))}"

                    grf = grf.copy()
                    grf["label"] = grf.apply(mk_label, axis=1)
                    pick = st.selectbox("اختار note", grf["label"].tolist(), key="gr_edit_pick")
                    gid = grf[grf["label"] == pick].iloc[0]["grade_id"]
                    row = grf[grf["grade_id"] == gid].iloc[0].to_dict()

                    new_exam = st.text_input("Type examen", value=norm(row.get("exam_type")), key="gr_edit_exam")
                    new_score = st.text_input("Score", value=norm(row.get("score")), key="gr_edit_score")
                    new_date = st.text_input("Date (YYYY-MM-DD)", value=norm(row.get("date")), key="gr_edit_date")
                    new_note = st.text_area("Remarque", value=norm(row.get("note")), key="gr_edit_note")

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save modifications", use_container_width=True, key="gr_save"):
                            ok = update_row_by_id("Grades", "grade_id", gid, {
                                "exam_type": new_exam,
                                "score": new_score,
                                "date": new_date,
                                "note": new_note,
                            })
                            if ok:
                                st.success("✅ Updated.")
                                st.rerun()
                            else:
                                st.error("Update failed.")
                    with c2:
                        if st.button("🗑️ Delete note", use_container_width=True, key="gr_del"):
                            if delete_row_by_id("Grades", "grade_id", gid):
                                st.success("✅ Deleted.")
                                st.rerun()
                            else:
                                st.error("Delete failed.")

    # ---------------- Payments CRUD (toggle + delete row)
    with tab_pay:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            if tr.empty or "trainee_id" not in tr.columns:
                st.info("لا يوجد stagiaires.")
                st.stop()

            tr = tr.copy()
            tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
            chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="pay_pick_tr")
            trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

            ensure_payment_row(trainee_id, staff_branch, program, group, year, staff_name)

            pay = read_df("Payments")
            m = pay[
                (pay["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
                (pay["year"].astype(str).str.strip() == norm(year))
            ].copy() if (not pay.empty and {"trainee_id","year"}.issubset(set(pay.columns))) else pd.DataFrame()

            if m.empty:
                st.warning("Payment row لم يتسجل (جرّب مرة أخرى).")
            else:
                rowp = m.iloc[0].to_dict()
                cols = st.columns(4)
                for i, mo in enumerate(MONTHS):
                    paid = (norm(rowp.get(mo)).upper() == "TRUE")
                    with cols[i % 4]:
                        new_paid = st.checkbox(mo, value=paid, key=f"ck_{mo}_{trainee_id}_{year}_{staff_branch}")
                        if new_paid != paid:
                            set_payment_month(trainee_id, year, mo, new_paid, staff_name)
                            st.rerun()

                st.divider()
                crud_section("🗑️ Delete payment row (للسنة هذي)")
                pid = norm(rowp.get("payment_id"))
                if st.button("🗑️ Delete payments of this year", use_container_width=True, key="pay_del_row"):
                    if pid and delete_row_by_id("Payments", "payment_id", pid):
                        st.success("✅ Deleted.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")

    # ---------------- Timetable CRUD
    with tab_plan:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            subjects = sorted([x for x in (sub["subject_name"].astype(str).str.strip().tolist()
                                           if (not sub.empty and "subject_name" in sub.columns) else []) if x])
            if not subjects:
                st.warning("زيد matières في Subjects قبل.")
                st.stop()

            st.markdown("### ➕ Ajouter au Planning")
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 2])
            with c1:
                day = st.selectbox("Jour", DAYS, key=f"tt_day_{staff_branch}_{program}_{group}_{year}")
            with c2:
                start_time = st.text_input("De (HH:MM)", value="08:00", key=f"tt_start_{staff_branch}_{program}_{group}_{year}")
            with c3:
                end_time = st.text_input("À (HH:MM)", value="09:30", key=f"tt_end_{staff_branch}_{program}_{group}_{year}")
            with c4:
                subject_name = st.selectbox("Matière", subjects, key=f"tt_subj_{staff_branch}_{program}_{group}_{year}")
            with c5:
                color = st.color_picker("Couleur", value="#DDEEFF", key=f"tt_color_{staff_branch}_{program}_{group}_{year}")

            teacher_name = st.text_input("Nom du prof", key=f"tt_teacher_{staff_branch}_{program}_{group}_{year}")

            if st.button("✅ Ajouter", use_container_width=True, key=f"tt_add_{staff_branch}_{program}_{group}_{year}"):
                if not norm(teacher_name):
                    st.error("اسم البروف obligatoire.")
                    st.stop()
                if minutes_of(start_time) is None or minutes_of(end_time) is None:
                    st.error("الوقت لازم يكون HH:MM صحيح.")
                    st.stop()
                if minutes_of(end_time) <= minutes_of(start_time):
                    st.error("وقت النهاية لازم يكون أكبر من البداية.")
                    st.stop()

                append_row("Timetable", {
                    "tt_id": f"TT-{uuid.uuid4().hex[:10].upper()}",
                    "branch": staff_branch,
                    "program": norm(program),
                    "group": norm(group),
                    "year": norm(year),
                    "day": norm(day),
                    "start_time": norm(start_time),
                    "end_time": norm(end_time),
                    "subject_name": norm(subject_name),
                    "teacher_name": norm(teacher_name),
                    "color": normalize_color(color),
                    "created_at": now_str(),
                    "staff_name": staff_name,
                })
                st.success("✅ Added.")
                st.rerun()

            st.divider()
            st.markdown("### 📋 Preview (ملوّن)")
            existing = load_timetable(staff_branch, program, group, year)
            st.markdown(render_timetable_html(existing), unsafe_allow_html=True)

            st.divider()
            crud_section("✏️/🗑️ Modifier أو supprimer slot")
            if existing.empty or "tt_id" not in existing.columns:
                st.info("لا يوجد Planning.")
            else:
                existing = existing.copy()
                existing["label"] = existing.apply(
                    lambda r: f"{norm(r.get('tt_id'))} | {norm(r.get('day'))} | {norm(r.get('start_time'))}-{norm(r.get('end_time'))} | {norm(r.get('subject_name'))}",
                    axis=1
                )
                pick = st.selectbox("اختار slot", existing["label"].tolist(), key="tt_edit_pick")
                tt_id = existing[existing["label"] == pick].iloc[0]["tt_id"]
                row = existing[existing["tt_id"] == tt_id].iloc[0].to_dict()

                new_day = st.selectbox("Jour", DAYS, index=DAYS.index(norm(row.get("day"))) if norm(row.get("day")) in DAYS else 0, key="tt_edit_day")
                new_start = st.text_input("De (HH:MM)", value=norm(row.get("start_time")), key="tt_edit_start")
                new_end = st.text_input("À (HH:MM)", value=norm(row.get("end_time")), key="tt_edit_end")
                new_subj = st.selectbox("Matière", subjects,
                                        index=subjects.index(norm(row.get("subject_name"))) if norm(row.get("subject_name")) in subjects else 0,
                                        key="tt_edit_subj")
                new_teacher = st.text_input("Nom du prof", value=norm(row.get("teacher_name")), key="tt_edit_teacher")
                new_color = st.color_picker("Couleur", value=normalize_color(row.get("color")), key="tt_edit_color")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save modifications", use_container_width=True, key="tt_save"):
                        if minutes_of(new_start) is None or minutes_of(new_end) is None:
                            st.error("الوقت لازم يكون HH:MM صحيح.")
                            st.stop()
                        if minutes_of(new_end) <= minutes_of(new_start):
                            st.error("وقت النهاية لازم يكون أكبر من البداية.")
                            st.stop()

                        ok = update_row_by_id("Timetable", "tt_id", tt_id, {
                            "day": new_day,
                            "start_time": new_start,
                            "end_time": new_end,
                            "subject_name": new_subj,
                            "teacher_name": new_teacher,
                            "color": normalize_color(new_color),
                        })
                        if ok:
                            st.success("✅ Updated.")
                            st.rerun()
                        else:
                            st.error("Update failed.")
                with c2:
                    if st.button("🗑️ Delete slot", use_container_width=True, key="tt_del"):
                        if delete_row_by_id("Timetable", "tt_id", tt_id):
                            st.success("✅ Deleted.")
                            st.rerun()
                        else:
                            st.error("Delete failed.")

    # ---------------- Supports CRUD
    with tab_sup:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            subjects = sorted([x for x in (sub["subject_name"].astype(str).str.strip().tolist()
                                           if (not sub.empty and "subject_name" in sub.columns) else []) if x])
            if not subjects:
                st.warning("زيد matières في Subjects قبل.")
                st.stop()

            st.markdown("### ➕ Ajouter support (Drive link)")
            subj = st.selectbox("Matière", subjects, key=f"cf_subj_{staff_branch}_{program}_{group}")
            fname = st.text_input("Nom du fichier", key=f"cf_name_{staff_branch}_{program}_{group}")
            link = st.text_input("Lien Google Drive (Share link)", key=f"cf_link_{staff_branch}_{program}_{group}")

            if st.button("✅ Enregistrer fichier", use_container_width=True, key=f"cf_add_{staff_branch}_{program}_{group}"):
                if not norm(link) or not norm(fname):
                    st.error("لازم اسم ملف + رابط.")
                else:
                    view_url, dl_url = to_view_and_download(link)
                    append_row("CourseFiles", {
                        "file_id": f"CF-{uuid.uuid4().hex[:8].upper()}",
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "subject_name": norm(subj),
                        "file_name": norm(fname),
                        "drive_view_url": view_url,
                        "drive_download_url": dl_url,
                        "uploaded_at": now_str(),
                        "staff_name": staff_name,
                    })
                    st.success("✅ Ajouté.")
                    st.rerun()

            st.divider()
            st.markdown("### 📚 Liste supports")
            files = read_df("CourseFiles")
            if files.empty or "file_id" not in files.columns:
                st.info("لا يوجد.")
                st.stop()

            files = files[
                (files.get("branch","").astype(str).str.strip() == staff_branch) &
                (files.get("program","").astype(str).str.strip() == norm(program)) &
                (files.get("group","").astype(str).str.strip() == norm(group))
            ] if {"branch","program","group"}.issubset(set(files.columns)) else pd.DataFrame()

            if files.empty:
                st.info("لا يوجد.")
            else:
                if "uploaded_at" in files.columns:
                    files = files.sort_values(by=["uploaded_at"], ascending=False)

                # quick view
                view_cols = [c for c in ["subject_name","file_name","uploaded_at","staff_name"] if c in files.columns]
                st.dataframe(files[view_cols], use_container_width=True, hide_index=True)

                st.divider()
                crud_section("✏️/🗑️ Modifier أو supprimer support")
                files = files.copy()
                files["label"] = files.apply(lambda r: f"{norm(r.get('file_id'))} | {norm(r.get('subject_name'))} | {norm(r.get('file_name'))}", axis=1)
                pick = st.selectbox("اختار support", files["label"].tolist(), key="cf_edit_pick")
                file_id = files[files["label"] == pick].iloc[0]["file_id"]
                row = files[files["file_id"] == file_id].iloc[0].to_dict()

                new_subj = st.selectbox("Matière", subjects,
                                        index=subjects.index(norm(row.get("subject_name"))) if norm(row.get("subject_name")) in subjects else 0,
                                        key="cf_edit_subj")
                new_name = st.text_input("Nom du fichier", value=norm(row.get("file_name")), key="cf_edit_name")
                new_link = st.text_input("Lien Drive", value=norm(row.get("drive_view_url")), key="cf_edit_link")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save modifications", use_container_width=True, key="cf_save"):
                        if not norm(new_name) or not norm(new_link):
                            st.error("لازم اسم + رابط.")
                            st.stop()
                        view_url, dl_url = to_view_and_download(new_link)
                        ok = update_row_by_id("CourseFiles", "file_id", file_id, {
                            "subject_name": new_subj,
                            "file_name": new_name,
                            "drive_view_url": view_url,
                            "drive_download_url": dl_url,
                        })
                        if ok:
                            st.success("✅ Updated.")
                            st.rerun()
                        else:
                            st.error("Update failed.")
                with c2:
                    if st.button("🗑️ Delete support", use_container_width=True, key="cf_del"):
                        if delete_row_by_id("CourseFiles", "file_id", file_id):
                            st.success("✅ Deleted.")
                            st.rerun()
                        else:
                            st.error("Delete failed.")


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_session()
    ensure_schema_once()
    sidebar_staff_login()

    try:
        if st.session_state.role == "staff":
            staff_work_center()
            st.divider()
            student_portal_center()
        else:
            student_portal_center()
            st.divider()
            st.info("ℹ️ Connexion Employé موجودة في اليسار.")
    except APIError as e:
        st.error(explain_api_error(e))
        st.info("جرّب تعمل Reboot (Restart) و استنى دقيقة (quota).")

if __name__ == "__main__":
    main()
```
