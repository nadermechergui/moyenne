import uuid
import base64
import io
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError
from PIL import Image

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Portail Mega Formation", page_icon="🧩", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# ✅ NOTE:
# - Planning image is stored as Drive links (manual paste)
# - Timetable is structured (days/slots/subject/teacher/color) so student sees a colored table
REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],

    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],

    # student_name = free name typed by student (phone must match Trainees)
    "Accounts": ["phone", "password", "trainee_id", "student_name", "created_at", "last_login"],

    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group",
               "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],

    # Manual planning IMAGE link (Drive share link -> view + download)
    "TimetableImages": ["branch", "program", "group", "year",
                        "drive_view_url", "drive_download_url", "file_name",
                        "uploaded_at", "staff_name"],

    # Profile pics only (small base64 safe)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    # Payments per year + months
    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"] + MONTHS + ["updated_at", "staff_name"],

    # Course supports links (Drive manual paste)
    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "drive_view_url", "drive_download_url", "uploaded_at", "staff_name"],

    # ✅ Structured timetable (colored)
    "TimetableSlots": ["row_id", "branch", "program", "group", "year",
                       "day", "start", "end",
                       "subject_name", "teacher_name", "color",
                       "created_at", "updated_at", "staff_name"],
}

# =========================================================
# UTILS
# =========================================================
def norm(x):
    return str(x or "").strip()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        if status == 429 or "quota" in low or "read requests" in low:
            return "⚠️ 429 Quota (Google Sheets). جرّب Reboot واستنى شوية.\n" + text[:240]
        if status == 403 or "permission" in low or "forbidden" in low:
            return "❌ 403 Permission. Share Sheet مع service account.\n" + text[:240]
        if status == 404 or "not found" in low:
            return "❌ 404 Not found. تأكد GSHEET_ID صحيح + Share للـ service account.\n" + text[:240]
        return "❌ Google API Error:\n" + (text[:360] if text else str(e))
    except Exception:
        return "❌ Google API Error."

def safe_hex_color(x: str, default="#E5E7EB") -> str:
    s = norm(x)
    if not s:
        return default
    if not s.startswith("#"):
        s = "#" + s
    if len(s) not in (4, 7):
        return default
    ok = set("0123456789abcdefABCDEF#")
    if any(ch not in ok for ch in s):
        return default
    return s

def compress_image_bytes(img_bytes: bytes, max_side: int = 256, quality: int = 70) -> bytes:
    """For PROFILE pics only (small)."""
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

# ---------- Drive manual link helpers ----------
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

# ---------- Streamlit link button compatibility ----------
def link_button(label: str, url: str, **kwargs):
    u = norm(url)
    if not u:
        st.button(label, disabled=True, **{k: v for k, v in kwargs.items() if k in ["key", "use_container_width"]})
        return
    if hasattr(st, "link_button"):
        st.link_button(label, u, **kwargs)
    else:
        # fallback
        st.markdown(f"[{label}]({u})")

# =========================================================
# AUTH CLIENTS
# =========================================================
@st.cache_resource
def creds():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",  # ok even if we don't upload (links only)
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

    # if empty -> write headers
    if len(row1) == 0 or all(x == "" for x in row1):
        ws.append_row(headers, value_input_option="RAW")
        return

    # if mismatch -> warn only (no delete)
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

def bump_cache():
    st.session_state["cache_bust"] = st.session_state.get("cache_bust", 0) + 1

@st.cache_data(ttl=180, show_spinner=False)
def read_df(ws_name: str, cache_bust: int) -> pd.DataFrame:
    # cache_bust is used to invalidate cache when we write
    ws = spreadsheet().worksheet(ws_name)
    values = ws.get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

def df(ws_name: str) -> pd.DataFrame:
    return read_df(ws_name, st.session_state.get("cache_bust", 0))

def append_row(ws_name: str, row: dict):
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    bump_cache()

def update_row_by_key(ws_name: str, key_cols: list[str], key_vals: list[str], updates: dict) -> bool:
    d = df(ws_name)
    if d.empty:
        return False

    m = d.copy()
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

    for col_name, val in updates.items():
        if col_name not in headers:
            continue
        ws.update_cell(row_num, headers.index(col_name) + 1, norm(val))

    bump_cache()
    return True

def delete_row_by_key(ws_name: str, key_col: str, key_val: str) -> bool:
    d = df(ws_name)
    if d.empty or key_col not in d.columns:
        return False
    m = d[d[key_col].astype(str).str.strip() == norm(key_val)]
    if m.empty:
        return False
    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet(ws_name)
    ws.delete_rows(row_num)
    bump_cache()
    return True

# =========================================================
# PROFILE PICS
# =========================================================
def get_profile_pic_bytes(phone: str) -> bytes | None:
    d = df("ProfilePics")
    if d.empty:
        return None
    m = d[d["phone"].astype(str).str.strip() == norm(phone)]
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
    d = df("Payments")
    if not d.empty:
        m = d[(d["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
              (d["year"].astype(str).str.strip() == norm(year))]
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
    d = df("Payments")
    if d.empty:
        return False
    m = d[(d["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
          (d["year"].astype(str).str.strip() == norm(year))]
    if m.empty:
        return False

    idx = m.index[0]
    row_num = idx + 2
    ws = spreadsheet().worksheet("Payments")
    headers = REQUIRED_SHEETS["Payments"]

    ws.update_cell(row_num, headers.index(month) + 1, "TRUE" if paid else "FALSE")
    ws.update_cell(row_num, headers.index("updated_at") + 1, now_str())
    ws.update_cell(row_num, headers.index("staff_name") + 1, staff_name)

    bump_cache()
    return True

# =========================================================
# TIMETABLE (STRUCTURED) - CRUD
# =========================================================
def load_timetable(branch: str, program: str, group: str, year: str) -> pd.DataFrame:
    d = df("TimetableSlots")
    if d.empty:
        return pd.DataFrame(columns=REQUIRED_SHEETS["TimetableSlots"])
    for c in ["branch", "program", "group", "year", "day", "start", "end", "subject_name", "teacher_name", "color", "row_id"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    m = d[(d["branch"] == norm(branch)) &
          (d["program"] == norm(program)) &
          (d["group"] == norm(group)) &
          (d["year"] == norm(year))].copy()
    return m

def upsert_slot(row_id: str | None,
                branch: str, program: str, group: str, year: str,
                day: str, start: str, end: str,
                subject_name: str, teacher_name: str, color: str,
                staff_name: str):
    if row_id:
        ok = update_row_by_key(
            "TimetableSlots",
            ["row_id"], [row_id],
            {
                "branch": branch,
                "program": program,
                "group": group,
                "year": year,
                "day": day,
                "start": start,
                "end": end,
                "subject_name": subject_name,
                "teacher_name": teacher_name,
                "color": safe_hex_color(color),
                "updated_at": now_str(),
                "staff_name": staff_name,
            }
        )
        return ok

    new_id = f"TT-{uuid.uuid4().hex[:10].upper()}"
    append_row("TimetableSlots", {
        "row_id": new_id,
        "branch": branch,
        "program": program,
        "group": group,
        "year": year,
        "day": day,
        "start": start,
        "end": end,
        "subject_name": subject_name,
        "teacher_name": teacher_name,
        "color": safe_hex_color(color),
        "created_at": now_str(),
        "updated_at": "",
        "staff_name": staff_name,
    })
    return True

def timetable_grid_html(tt: pd.DataFrame, title: str = "") -> str:
    # build grid days x unique time slots
    css = """
    <style>
      .tt-wrap{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;}
      .tt-title{font-weight:700;margin:0 0 8px 0;}
      table.tt{border-collapse:collapse;width:100%;table-layout:fixed;}
      table.tt th, table.tt td{border:1px solid #e5e7eb;padding:8px;vertical-align:top;}
      table.tt th{background:#f9fafb;font-weight:700;text-align:center;}
      .timecell{background:#f9fafb;font-weight:600;width:130px;white-space:nowrap;}
      .slot{border-radius:10px;padding:10px;min-height:56px;}
      .slot .time{font-size:12px;font-weight:700;margin-bottom:4px;opacity:.9}
      .slot .sub{font-size:14px;font-weight:700;margin-bottom:2px}
      .slot .meta{font-size:12px;opacity:.85}
      .empty{height:56px;}
    </style>
    """
    if tt is None or tt.empty:
        return css + f"""<div class="tt-wrap"><div class="tt-title">{title}</div><div>لا يوجد جدول.</div></div>"""

    df2 = tt.copy()

    def time_key(t: str) -> int:
        s = norm(t)
        try:
            hh, mm = s.split(":")
            return int(hh) * 60 + int(mm)
        except Exception:
            return 10**9

    def slot_key(row):
        return (time_key(row.get("start", "")), time_key(row.get("end", "")))

    df2["day"] = df2["day"].astype(str).str.strip()
    df2["start"] = df2["start"].astype(str).str.strip()
    df2["end"] = df2["end"].astype(str).str.strip()
    df2["color"] = df2["color"].astype(str).apply(lambda x: safe_hex_color(x, "#E5E7EB"))

    # Keep only valid days
    days = [d for d in DAYS_FR if d in set(df2["day"].tolist())] or DAYS_FR

    # unique timeslots
    df2["slot_label"] = df2["start"] + " → " + df2["end"]
    slots = sorted(df2["slot_label"].unique().tolist(), key=lambda s: time_key(s.split("→")[0].strip()))
    # map slot+day -> possibly multiple (we will stack)
    cell_map = {}
    for _, r in df2.iterrows():
        key = (r["slot_label"], r["day"])
        cell_map.setdefault(key, []).append(r.to_dict())

    # HTML
    h = [css, '<div class="tt-wrap">']
    if title:
        h.append(f'<div class="tt-title">{title}</div>')
    h.append('<table class="tt">')
    # header row
    h.append("<tr>")
    h.append('<th class="timecell">Heure</th>')
    for dday in days:
        h.append(f"<th>{dday}</th>")
    h.append("</tr>")

    for sl in slots:
        h.append("<tr>")
        h.append(f'<td class="timecell">{sl}</td>')
        for dday in days:
            items = cell_map.get((sl, dday), [])
            if not items:
                h.append('<td><div class="empty"></div></td>')
            else:
                # stack multiple slots in same cell
                blocks = []
                for it in items:
                    color = safe_hex_color(it.get("color", "#E5E7EB"))
                    sub = norm(it.get("subject_name"))
                    teacher = norm(it.get("teacher_name"))
                    blocks.append(
                        f"""
                        <div class="slot" style="background:{color}">
                          <div class="time">{sl}</div>
                          <div class="sub">{sub}</div>
                          <div class="meta">{teacher}</div>
                        </div>
                        """
                    )
                h.append("<td>" + "".join(blocks) + "</td>")
        h.append("</tr>")
    h.append("</table></div>")
    return "".join(h)

# =========================================================
# SESSION / AUTH
# =========================================================
def ensure_session():
    st.session_state.setdefault("role", None)  # staff | None
    st.session_state.setdefault("user", {})
    st.session_state.setdefault("student", None)
    st.session_state.setdefault("cache_bust", 0)

def logout_staff():
    st.session_state.role = None
    st.session_state.user = {}

def staff_branch_login(branch: str, branch_password: str):
    d = df("Branches")
    if d.empty:
        return None

    df2 = d.copy()
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
    d = df("Accounts")
    if d.empty:
        return None
    df2 = d.copy()
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

    branches_df = df("Branches")
    branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []

    if st.session_state.role == "staff":
        br = norm(st.session_state.user.get("branch"))
        st.sidebar.success(f"Connecté: {br}")

        st.sidebar.divider()
        st.sidebar.markdown("### 🧰 Maintenance")
        if st.sidebar.button("Initialiser / Vérifier les Sheets", use_container_width=True, key="btn_init_schema"):
            st.session_state.init_schema_now = True
            st.rerun()

        if st.sidebar.button("🔄 Refresh data", use_container_width=True, key="btn_refresh"):
            bump_cache()
            st.rerun()

        if st.sidebar.button("Se déconnecter", use_container_width=True, key="btn_logout_staff"):
            logout_staff()
            st.rerun()
        return

    if not branches:
        st.sidebar.warning("Branches vide. Ajoutez centres + mots de passe.")
        return

    branch = st.sidebar.selectbox("Centre", branches, key="sb_staff_branch")
    pwd = st.sidebar.text_input("Mot de passe du centre", type="password", key="sb_staff_pwd")

    if st.sidebar.button("Connexion", use_container_width=True, key="btn_staff_login"):
        user = staff_branch_login(branch, pwd)
        if user:
            st.session_state.role = "staff"
            st.session_state.user = user
            st.sidebar.success("✅ OK")
            st.rerun()
        else:
            st.sidebar.error("Mot de passe incorrect / centre inactif.")

# =========================================================
# STUDENT PORTAL
# =========================================================
def student_portal_center():
    st.markdown("## 🎓 Espace Stagiaire")

    tab1, tab2, tab3 = st.tabs(["🔐 Connexion", "🆕 Inscription", "📌 Mon espace"])

    # ------------------ Login
    with tab1:
        phone = st.text_input("Téléphone", key="stud_phone")
        pwd = st.text_input("Mot de passe", type="password", key="stud_pwd")

        if st.button("Se connecter", use_container_width=True, key="btn_stud_login"):
            acc = student_login(phone, pwd)
            if acc:
                update_row_by_key("Accounts", ["phone"], [phone], {"last_login": now_str()})
                st.session_state.student = acc
                st.success("✅ Connexion réussie")
                st.rerun()
            else:
                st.error("Téléphone / mot de passe incorrect.")

        if st.button("Se déconnecter (Stagiaire)", use_container_width=True, key="btn_stud_logout"):
            st.session_state.student = None
            st.rerun()

    # ------------------ Registration (name free, phone must exist)
    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")

        branches_df = df("Branches")
        branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre disponible.")
            return

        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(df("Programs"), branch=b)
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_program")

        grp_df = df_filter(df("Groups"), branch=b, program_name=p)
        grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
        if not groups:
            st.warning("Aucun groupe.")
            return
        g = st.selectbox("Groupe", groups, key="reg_group")

        student_name = st.text_input("Nom (أي اسم تحب)", key="reg_name_free")
        phone = st.text_input("Téléphone (نفس رقمك عند الإدارة)", key="reg_phone")
        pwd = st.text_input("Mot de passe", type="password", key="reg_pwd")

        if st.button("Créer mon compte", use_container_width=True, key="btn_register"):
            if not norm(student_name) or not norm(phone) or not norm(pwd):
                st.error("Nom + téléphone + mot de passe obligatoire.")
                return
            if len(norm(pwd)) < 4:
                st.error("Mot de passe قصير (min 4).")
                return

            acc = df("Accounts")
            if not acc.empty and acc["phone"].astype(str).str.strip().eq(norm(phone)).any():
                st.error("Ce téléphone est déjà inscrit.")
                return

            tr = df("Trainees")
            if tr.empty:
                st.error("Aucun stagiaire enregistré par l'employé.")
                return

            tr2 = tr.copy()
            for c in ["branch", "program", "group", "phone"]:
                tr2[c] = tr2[c].astype(str).str.strip()

            candidates = tr2[(tr2["branch"] == norm(b)) &
                             (tr2["program"] == norm(p)) &
                             (tr2["group"] == norm(g)) &
                             (tr2["phone"] == norm(phone))]

            if candidates.empty:
                st.error("رقم الهاتف موش موجود في Trainees. الموظف لازم يسجل نفس الرقم.")
                return

            trainee_id = candidates.iloc[0]["trainee_id"]

            append_row("Accounts", {
                "phone": norm(phone),
                "password": norm(pwd),
                "trainee_id": norm(trainee_id),
                "student_name": norm(student_name),
                "created_at": now_str(),
                "last_login": ""
            })
            st.success("✅ Compte créé. امشي لصفحة Connexion.")

    # ------------------ My space
    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion باش تشوف النوطات والدفوعات والجدول والملفات.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = df("Trainees")
        row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not tr.empty else pd.DataFrame()
        if row.empty:
            st.error("Compte مرتبط بمتربص غير موجود.")
            return

        info = row.iloc[0].to_dict()
        branch = norm(info.get("branch"))
        program = norm(info.get("program"))
        group = norm(info.get("group"))

        c1, c2 = st.columns([1, 3])
        with c1:
            pic = get_profile_pic_bytes(phone)
            if pic:
                st.image(pic, caption="Photo", use_container_width=True)
            else:
                st.info("Pas de photo")

        with c2:
            st.success(f"Bienvenue {student_name or norm(info.get('full_name'))} ✅")
            st.caption(f"Centre: {branch} | Spécialité: {program} | Groupe: {group} | Tél: {phone}")

            up = st.file_uploader("📸 Ajouter/Changer ma photo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="profile_uploader")
            if up is not None:
                img_bytes = up.read()
                st.image(img_bytes, caption="Aperçu", width=150)
                if st.button("Enregistrer ma photo", use_container_width=True, key="btn_save_profile_pic"):
                    try:
                        upsert_profile_pic(phone, trainee_id, img_bytes)
                        st.success("✅ Photo enregistrée.")
                        st.rerun()
                    except APIError as e:
                        st.error(explain_api_error(e))

        t1, t2, t3, t4, t5 = st.tabs(["📝 Notes", "🗓️ Planning (Image)", "🗓️ Emploi du temps (Couleur)", "💳 Paiements", "📎 Supports"])

        with t1:
            gr = df("Grades")
            grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
            if grf.empty:
                st.info("Aucune note.")
            else:
                if "date" in grf.columns:
                    grf["date"] = grf["date"].astype(str)
                grf = grf.sort_values(by=[c for c in ["date", "created_at"] if c in grf.columns], ascending=False)
                cols = [c for c in ["subject_name", "exam_type", "score", "date", "staff_name", "note"] if c in grf.columns]
                st.dataframe(grf[cols], use_container_width=True, hide_index=True)

        with t2:
            year = st.selectbox("Année (Planning Image)", [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)], index=1, key="stud_pl_year")
            tt = df("TimetableImages")
            m = tt[(tt["branch"].astype(str).str.strip() == branch) &
                   (tt["program"].astype(str).str.strip() == program) &
                   (tt["group"].astype(str).str.strip() == group) &
                   (tt["year"].astype(str).str.strip() == norm(year))] if not tt.empty else pd.DataFrame()
            if m.empty:
                st.info("لا توجد Planning image للسنة هذه.")
            else:
                r = m.iloc[0].to_dict()
                st.markdown(f"**📄 {norm(r.get('file_name') or 'Planning')}**")
                # Show image directly if possible (download url)
                dl = norm(r.get("drive_download_url"))
                if dl:
                    try:
                        st.image(dl, caption="Planning", use_container_width=True)
                    except Exception:
                        pass
                link_button("👀 Ouvrir", norm(r.get("drive_view_url")), use_container_width=True, key="stud_pl_open")
                link_button("⬇️ Télécharger", norm(r.get("drive_download_url")), use_container_width=True, key="stud_pl_dl")

        with t3:
            year = st.selectbox("Année (Emploi du temps)", [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)], index=1, key="stud_tt_year")
            tt = load_timetable(branch, program, group, year)
            html = timetable_grid_html(tt, title=f"Planning ({branch} • {program} • {group} • {year})")
            components.html(html, height=560, scrolling=True)

        with t4:
            pay = df("Payments")
            if pay.empty:
                st.info("لا توجد بيانات دفوعات.")
            else:
                # show years that exist for this trainee
                pay2 = pay[pay["trainee_id"].astype(str).str.strip() == trainee_id].copy()
                years = sorted([y for y in pay2["year"].astype(str).str.strip().unique().tolist() if y])
                if not years:
                    st.info("لا توجد بيانات دفوعات.")
                else:
                    year = st.selectbox("السنة", years, index=len(years) - 1, key="stud_pay_year")
                    m = pay2[pay2["year"].astype(str).str.strip() == norm(year)]
                    if m.empty:
                        st.info("لا توجد بيانات دفوعات للسنة هذه.")
                    else:
                        rowp = m.iloc[0].to_dict()
                        show = {mo: ("✅" if (norm(rowp.get(mo)).upper() == "TRUE") else "—") for mo in MONTHS}
                        st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

        with t5:
            files = df("CourseFiles")
            files = files[(files["branch"].astype(str).str.strip() == branch) &
                          (files["program"].astype(str).str.strip() == program) &
                          (files["group"].astype(str).str.strip() == group)] if not files.empty else pd.DataFrame()
            if files.empty:
                st.info("لا توجد ملفات.")
            else:
                files = files.sort_values(by=["uploaded_at"] if "uploaded_at" in files.columns else files.columns[0], ascending=False)
                for _, r in files.iterrows():
                    st.markdown(f"**📌 {norm(r.get('subject_name'))}** — {norm(r.get('file_name'))}")
                    link_button("👀 Ouvrir", norm(r.get("drive_view_url")), use_container_width=True, key=f"v_{r.get('file_id')}")
                    link_button("⬇️ Télécharger", norm(r.get("drive_download_url")), use_container_width=True, key=f"d_{r.get('file_id')}")
                    st.divider()

# =========================================================
# STAFF AREA
# =========================================================
def staff_work_center():
    st.markdown("## 🛠️ Espace Employé")
    if st.session_state.role != "staff":
        st.info("Connexion Employé من اليسار.")
        return

    staff_branch = norm(st.session_state.user.get("branch"))
    staff_name = f"Staff-{staff_branch}"

    prog_df = df_filter(df("Programs"), branch=staff_branch)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(df("Groups"), branch=staff_branch, program_name=program)
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox(
            "Année",
            [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)],
            index=1,
            key="manage_year"
        )

    tab_stag, tab_sub, tab_gr, tab_pay, tab_plan_img, tab_tt, tab_sup = st.tabs(
        ["👤 Stagiaires", "📚 Matières", "📝 Notes", "💳 Paiements", "🗓️ Planning (Image Drive)", "🗓️ Emploi du temps (Couleur)", "📎 Supports (Liens Drive)"]
    )

    # ---------- Stagiaires ----------
    with tab_stag:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            cur = df_filter(df("Trainees"), branch=staff_branch, program=program, group=group)
            if cur.empty:
                st.info("لا يوجد stagiaires.")
            else:
                st.dataframe(cur[["full_name", "phone", "status", "created_at"]], use_container_width=True, hide_index=True)

            st.markdown("### ➕ Ajouter stagiaire")
            name = st.text_input("Nom", key="add_tr_name")
            phone = st.text_input("Téléphone", key="add_tr_phone")
            status = st.selectbox("Statut", ["active", "inactive"], key="add_tr_status")
            if st.button("Enregistrer", use_container_width=True, key="btn_add_tr"):
                if not norm(name) or not norm(phone):
                    st.error("Nom + téléphone obligatoire.")
                else:
                    # prevent duplicate phone in same group
                    exists = df_filter(df("Trainees"), branch=staff_branch, program=program, group=group)
                    if (not exists.empty) and exists["phone"].astype(str).str.strip().eq(norm(phone)).any():
                        st.error("❌ رقم الهاتف موجود déjà في نفس المجموعة.")
                    else:
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

            st.divider()
            st.markdown("### 📥 Import Excel (xlsx) : full_name + phone")
            up = st.file_uploader("Uploader Excel", type=["xlsx"], key="excel_tr")
            if up is not None:
                try:
                    df_x = pd.read_excel(up)
                    df_x.columns = [c.strip() for c in df_x.columns]
                    st.dataframe(df_x.head(20), use_container_width=True)

                    if st.button("✅ Importer maintenant", use_container_width=True, key="do_imp"):
                        if "full_name" not in df_x.columns or "phone" not in df_x.columns:
                            st.error("لازم full_name و phone.")
                        else:
                            existing = df_filter(df("Trainees"), branch=staff_branch, program=program, group=group)
                            existing_phones = set(existing["phone"].astype(str).str.strip().tolist()) if not existing.empty else set()

                            count = 0
                            for _, r in df_x.iterrows():
                                fn = norm(r.get("full_name"))
                                ph = norm(r.get("phone"))
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
                except Exception as e:
                    st.error(f"❌ Excel error: {e}")

    # ---------- Subjects ----------
    with tab_sub:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(df("Subjects"), branch=staff_branch, program=program, group=group)
            if sub.empty:
                st.info("ما فماش matières.")
            else:
                st.dataframe(sub[["subject_name", "is_active", "created_at"]], use_container_width=True, hide_index=True)

            st.markdown("### ➕ Ajouter matière")
            sname = st.text_input("Nom matière", key="add_subject_name")
            is_active = st.selectbox("Active?", ["true", "false"], key="add_subject_active")
            if st.button("Enregistrer matière", use_container_width=True, key="btn_add_subject"):
                if not norm(sname):
                    st.error("Nom matière obligatoire.")
                else:
                    append_row("Subjects", {
                        "subject_id": f"SUB-{uuid.uuid4().hex[:8].upper()}",
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "subject_name": norm(sname),
                        "is_active": is_active,
                        "created_at": now_str(),
                    })
                    st.success("✅ Matière ajoutée.")
                    st.rerun()

    # ---------- Grades ----------
    with tab_gr:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(df("Trainees"), branch=staff_branch, program=program, group=group)
            sub = df_filter(df("Subjects"), branch=staff_branch, program=program, group=group)
            sub = sub[sub["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not sub.empty else sub

            if tr.empty:
                st.warning("لا يوجد stagiaires.")
            elif sub.empty:
                st.warning("زيد matières قبل.")
            else:
                tr = tr.copy()
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
                chosen = st.selectbox("Stagiaire", tr["label"].tolist(), key="gr_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x])
                subject_name = st.selectbox("Matière", subjects, key="gr_subject")
                exam_type = st.text_input("Type examen (DS1/TP/Examen...)", key="gr_examtype")
                score = st.number_input("Note", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="gr_score")
                date = st.date_input("Date", value=datetime.now().date(), key="gr_date")
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
                            "date": str(date),
                            "staff_name": staff_name,
                            "note": norm(note),
                            "created_at": now_str(),
                        })
                        st.success("✅ Note enregistrée.")
                        st.rerun()

    # ---------- Payments ----------
    with tab_pay:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(df("Trainees"), branch=staff_branch, program=program, group=group)
            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr = tr.copy()
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="pay_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                ensure_payment_row(trainee_id, staff_branch, program, group, year, staff_name)

                pay = df("Payments")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
                        (pay["year"].astype(str).str.strip() == norm(year))].copy()
                if m.empty:
                    st.warning("Payment row مش موجود.")
                else:
                    rowp = m.iloc[0].to_dict()
                    cols = st.columns(4)
                    for i, mo in enumerate(MONTHS):
                        paid = (norm(rowp.get(mo)).upper() == "TRUE")
                        with cols[i % 4]:
                            new_paid = st.checkbox(mo, value=paid, key=f"ck_{mo}_{trainee_id}_{year}")
                            if new_paid != paid:
                                set_payment_month(trainee_id, year, mo, new_paid, staff_name)
                                st.rerun()

    # ---------- Planning image (Drive link) ----------
    with tab_plan_img:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            st.info("✅ ارفع الـ Planning كصورة في Google Drive ثم Paste الرابط هنا. (Share: Anyone with the link)")
            file_name = st.text_input("Nom du fichier (اختياري)", key="pl_name")
            share_link = st.text_input("Lien Google Drive (Share link)", key="pl_link")

            if st.button("✅ Enregistrer planning (image)", use_container_width=True, key="pl_save"):
                if not norm(share_link):
                    st.error("لازم رابط Drive.")
                else:
                    view_url, dl_url = to_view_and_download(share_link)
                    updated = update_row_by_key(
                        "TimetableImages",
                        ["branch", "program", "group", "year"],
                        [staff_branch, program, group, year],
                        {"drive_view_url": view_url, "drive_download_url": dl_url,
                         "file_name": norm(file_name) or "Planning",
                         "uploaded_at": now_str(), "staff_name": staff_name}
                    )
                    if not updated:
                        append_row("TimetableImages", {
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "year": norm(year),
                            "drive_view_url": view_url,
                            "drive_download_url": dl_url,
                            "file_name": norm(file_name) or "Planning",
                            "uploaded_at": now_str(),
                            "staff_name": staff_name,
                        })
                    st.success("✅ Planning enregistré.")
                    st.rerun()

            # preview current
            tt = df("TimetableImages")
            m = tt[(tt["branch"].astype(str).str.strip() == staff_branch) &
                   (tt["program"].astype(str).str.strip() == norm(program)) &
                   (tt["group"].astype(str).str.strip() == norm(group)) &
                   (tt["year"].astype(str).str.strip() == norm(year))] if not tt.empty else pd.DataFrame()

            if not m.empty:
                r = m.iloc[0].to_dict()
                st.divider()
                st.markdown("### Planning الحالي (Image)")
                dl = norm(r.get("drive_download_url"))
                if dl:
                    try:
                        st.image(dl, caption=norm(r.get("file_name") or "Planning"), use_container_width=True)
                    except Exception:
                        pass
                link_button("👀 Ouvrir", norm(r.get("drive_view_url")), use_container_width=True, key="pl_open_cur")
                link_button("⬇️ Télécharger", norm(r.get("drive_download_url")), use_container_width=True, key="pl_dl_cur")

    # ---------- Structured timetable (colored) ----------
    with tab_tt:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(df("Subjects"), branch=staff_branch, program=program, group=group)
            sub = sub[sub["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not sub.empty else sub
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x]) if not sub.empty else []

            st.markdown("### ➕ إضافة / تعديل Slot")
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                day = st.selectbox("Jour", DAYS_FR, key="tt_day")
            with c2:
                subject_name = st.selectbox("Matière", subjects, key="tt_subject") if subjects else st.text_input("Matière (لازم تزيدها في Matières)", key="tt_subject_txt")
                if not subjects:
                    subject_name = norm(subject_name)
            with c3:
                color = st.color_picker("Couleur", value="#E5E7EB", key="tt_color")

            c4, c5, c6 = st.columns([1, 1, 2])
            with c4:
                start = st.text_input("Heure début (HH:MM)", value="08:00", key="tt_start")
            with c5:
                end = st.text_input("Heure fin (HH:MM)", value="09:30", key="tt_end")
            with c6:
                teacher = st.text_input("Nom prof", key="tt_teacher")

            if st.button("✅ Ajouter slot", use_container_width=True, key="tt_add"):
                if not (norm(day) and norm(start) and norm(end) and norm(subject_name) and norm(teacher)):
                    st.error("لازم: day + start + end + subject + teacher.")
                else:
                    upsert_slot(
                        row_id=None,
                        branch=staff_branch, program=program, group=group, year=year,
                        day=day, start=norm(start), end=norm(end),
                        subject_name=norm(subject_name), teacher_name=norm(teacher),
                        color=safe_hex_color(color),
                        staff_name=staff_name
                    )
                    st.success("✅ Slot ajouté.")
                    st.rerun()

            st.divider()
            st.markdown("### 🧾 Slots الحاليّة + تعديل/حذف")

            tt = load_timetable(staff_branch, program, group, year)
            if tt.empty:
                st.info("لا يوجد جدول للسنة هذه.")
            else:
                # show a compact editor list
                show_cols = [c for c in ["row_id", "day", "start", "end", "subject_name", "teacher_name", "color", "updated_at", "staff_name"] if c in tt.columns]
                st.dataframe(tt[show_cols], use_container_width=True, hide_index=True)

                # pick a row to edit/delete
                ids = tt["row_id"].astype(str).str.strip().tolist() if "row_id" in tt.columns else []
                pick = st.selectbox("اختار slot باش تعدّل/تحذف", [""] + ids, key="tt_pick")
                if pick:
                    r = tt[tt["row_id"].astype(str).str.strip() == norm(pick)].iloc[0].to_dict()

                    st.markdown("#### ✏️ تعديل slot")
                    e1, e2, e3 = st.columns([2, 2, 1])
                    with e1:
                        day2 = st.selectbox("Jour", DAYS_FR, index=DAYS_FR.index(norm(r.get("day")) if norm(r.get("day")) in DAYS_FR else "Lundi"), key="tt_day2")
                    with e2:
                        # for edit, allow free text but prefer list if exists
                        if subjects:
                            subj2 = st.selectbox("Matière", subjects, index=subjects.index(norm(r.get("subject_name"))) if norm(r.get("subject_name")) in subjects else 0, key="tt_subj2")
                        else:
                            subj2 = st.text_input("Matière", value=norm(r.get("subject_name")), key="tt_subj2_txt")
                    with e3:
                        col2 = st.color_picker("Couleur", value=safe_hex_color(r.get("color"), "#E5E7EB"), key="tt_color2")

                    e4, e5, e6 = st.columns([1, 1, 2])
                    with e4:
                        st2 = st.text_input("Début", value=norm(r.get("start")), key="tt_start2")
                    with e5:
                        en2 = st.text_input("Fin", value=norm(r.get("end")), key="tt_end2")
                    with e6:
                        teach2 = st.text_input("Prof", value=norm(r.get("teacher_name")), key="tt_teacher2")

                    cbtn1, cbtn2 = st.columns(2)
                    with cbtn1:
                        if st.button("💾 Sauvegarder modification", use_container_width=True, key="tt_save_edit"):
                            ok = upsert_slot(
                                row_id=pick,
                                branch=staff_branch, program=program, group=group, year=year,
                                day=day2, start=norm(st2), end=norm(en2),
                                subject_name=norm(subj2), teacher_name=norm(teach2),
                                color=safe_hex_color(col2),
                                staff_name=staff_name
                            )
                            if ok:
                                st.success("✅ Modifié.")
                                st.rerun()
                            else:
                                st.error("❌ Modif failed.")
                    with cbtn2:
                        if st.button("🗑️ Supprimer ce slot", use_container_width=True, key="tt_delete"):
                            if delete_row_by_key("TimetableSlots", "row_id", pick):
                                st.success("✅ Supprimé.")
                                st.rerun()
                            else:
                                st.error("❌ Delete failed.")

                st.divider()
                st.markdown("### 👀 Preview tableau (ملوّن)")
                html = timetable_grid_html(tt, title=f"Planning ({staff_branch} • {program} • {group} • {year})")
                components.html(html, height=560, scrolling=True)

    # ---------- Course files links ----------
    with tab_sup:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(df("Subjects"), branch=staff_branch, program=program, group=group)
            sub = sub[sub["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not sub.empty else sub
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x]) if not sub.empty else []
            if not subjects:
                st.warning("زيد matières قبل.")
            else:
                subj = st.selectbox("Matière", subjects, key=f"cf_subj_{staff_branch}_{program}_{group}")
                fname = st.text_input("Nom du fichier", key=f"cf_name_{staff_branch}_{program}_{group}")
                link = st.text_input("Lien Google Drive (Share link)", key=f"cf_link_{staff_branch}_{program}_{group}")

                if st.button("✅ Enregistrer fichier", use_container_width=True, key=f"cf_save_{staff_branch}_{program}_{group}"):
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
                        st.success("✅ Fichier enregistré.")
                        st.rerun()

            # list current + delete
            files = df("CourseFiles")
            files = files[(files["branch"].astype(str).str.strip() == staff_branch) &
                          (files["program"].astype(str).str.strip() == norm(program)) &
                          (files["group"].astype(str).str.strip() == norm(group))] if not files.empty else pd.DataFrame()

            if not files.empty:
                st.divider()
                st.markdown("### Fichiers enregistrés")
                files = files.sort_values(by=["uploaded_at"] if "uploaded_at" in files.columns else files.columns[0], ascending=False)
                st.dataframe(files[["file_id", "subject_name", "file_name", "uploaded_at", "staff_name"]], use_container_width=True, hide_index=True)

                del_id = st.selectbox("حذف ملف (اختياري)", [""] + files["file_id"].astype(str).tolist(), key="cf_del_pick")
                if del_id and st.button("🗑️ Supprimer ce fichier", use_container_width=True, key="cf_del_btn"):
                    if delete_row_by_key("CourseFiles", "file_id", del_id):
                        st.success("✅ Deleted.")
                        st.rerun()
                    else:
                        st.error("❌ Delete failed.")

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_session()
    ensure_schema_once()
    sidebar_staff_login()

    # Important note: do NOT paste ```python in your file. Paste only this code.
    if st.session_state.role == "staff":
        staff_work_center()
        st.divider()
        student_portal_center()
    else:
        student_portal_center()
        st.divider()
        st.info("ℹ️ Connexion Employé موجودة في اليسار.")

if __name__ == "__main__":
    try:
        main()
    except APIError as e:
        st.error(explain_api_error(e))
