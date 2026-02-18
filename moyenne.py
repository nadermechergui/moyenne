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

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],
    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "student_name", "created_at", "last_login"],
    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group",
               "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],

    # Planning image links (Drive manual)
    # NOTE: some old sheets may not have "year" -> code auto-handles
    "TimetableImages": ["branch", "program", "group", "year",
                        "drive_view_url", "drive_download_url", "file_name",
                        "uploaded_at", "staff_name"],

    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"] + MONTHS + ["updated_at", "staff_name"],

    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "drive_view_url", "drive_download_url", "uploaded_at", "staff_name"],

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

def explain_api_error(e: APIError) -> str:
    try:
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", "") or ""
        low = text.lower()
        if status == 429 or "quota" in low or "read requests" in low:
            return "⚠️ 429 Quota (Google Sheets). اعمل Reboot واستنى شوية.\n" + text[:240]
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

def df_filter(df: pd.DataFrame, **kwargs):
    out = df.copy()
    for k, v in kwargs.items():
        if k in out.columns:
            out = out[out[k].astype(str).str.strip() == norm(v)]
    return out

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
        st.markdown(f"[{label}]({u})")

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
# SHEETS SETUP
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

def bump_cache():
    st.session_state["cache_bust"] = st.session_state.get("cache_bust", 0) + 1

@st.cache_data(ttl=180, show_spinner=False)
def read_df(ws_name: str, cache_bust: int) -> pd.DataFrame:
    ws = spreadsheet().worksheet(ws_name)
    values = ws.get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

def df(ws_name: str) -> pd.DataFrame:
    return read_df(ws_name, st.session_state.get("cache_bust", 0))

def ensure_cols(d: pd.DataFrame, ws_name: str) -> pd.DataFrame:
    """avoid KeyError: add missing columns as empty strings."""
    wanted = REQUIRED_SHEETS.get(ws_name, [])
    out = d.copy()
    for c in wanted:
        if c not in out.columns:
            out[c] = ""
    return out

def append_row(ws_name: str, row: dict):
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    bump_cache()

def update_row_by_key(ws_name: str, key_cols: list[str], key_vals: list[str], updates: dict) -> bool:
    d = ensure_cols(df(ws_name), ws_name)
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
    d = ensure_cols(df("ProfilePics"), "ProfilePics")
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
    updated = update_row_by_key("ProfilePics", ["phone"], [phone],
                               {"trainee_id": trainee_id, "image_b64": b64, "uploaded_at": now_str()})
    if not updated:
        append_row("ProfilePics", {"phone": phone, "trainee_id": trainee_id, "image_b64": b64, "uploaded_at": now_str()})

# =========================================================
# PAYMENTS
# =========================================================
def ensure_payment_row(trainee_id: str, branch: str, program: str, group: str, year: str, staff_name: str):
    d = ensure_cols(df("Payments"), "Payments")
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
    d = ensure_cols(df("Payments"), "Payments")
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
# TIMETABLE SLOTS
# =========================================================
def load_timetable(branch: str, program: str, group: str, year: str) -> pd.DataFrame:
    d = ensure_cols(df("TimetableSlots"), "TimetableSlots")
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
        return update_row_by_key(
            "TimetableSlots", ["row_id"], [row_id],
            {"branch": branch, "program": program, "group": group, "year": year,
             "day": day, "start": start, "end": end,
             "subject_name": subject_name, "teacher_name": teacher_name,
             "color": safe_hex_color(color),
             "updated_at": now_str(), "staff_name": staff_name}
        )

    new_id = f"TT-{uuid.uuid4().hex[:10].upper()}"
    append_row("TimetableSlots", {
        "row_id": new_id, "branch": branch, "program": program, "group": group, "year": year,
        "day": day, "start": start, "end": end,
        "subject_name": subject_name, "teacher_name": teacher_name, "color": safe_hex_color(color),
        "created_at": now_str(), "updated_at": "", "staff_name": staff_name
    })
    return True

def timetable_grid_html(tt: pd.DataFrame, title: str = "") -> str:
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

    df2["day"] = df2["day"].astype(str).str.strip()
    df2["start"] = df2["start"].astype(str).str.strip()
    df2["end"] = df2["end"].astype(str).str.strip()
    df2["color"] = df2["color"].astype(str).apply(lambda x: safe_hex_color(x, "#E5E7EB"))

    days = [d for d in DAYS_FR if d in set(df2["day"].tolist())] or DAYS_FR

    df2["slot_label"] = df2["start"] + " → " + df2["end"]
    slots = sorted(df2["slot_label"].unique().tolist(), key=lambda s: time_key(s.split("→")[0].strip()))

    cell_map = {}
    for _, r in df2.iterrows():
        key = (r["slot_label"], r["day"])
        cell_map.setdefault(key, []).append(r.to_dict())

    h = [css, '<div class="tt-wrap">']
    if title:
        h.append(f'<div class="tt-title">{title}</div>')
    h.append('<table class="tt">')
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
                h.append("<td>" + "".join(blocks) + "</td>')
        h.append("</tr>")
    h.append("</table></div>")
    return "".join(h)

# =========================================================
# AUTH / SESSION
# =========================================================
def ensure_session():
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("user", {})
    st.session_state.setdefault("student", None)
    st.session_state.setdefault("cache_bust", 0)

def logout_staff():
    st.session_state.role = None
    st.session_state.user = {}

def staff_branch_login(branch: str, branch_password: str):
    d = ensure_cols(df("Branches"), "Branches")
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
    d = ensure_cols(df("Accounts"), "Accounts")
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
# SIDEBAR
# =========================================================
def sidebar_staff_login():
    st.sidebar.markdown("## 👨‍💼 Connexion Employé")
    branches_df = ensure_cols(df("Branches"), "Branches")
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
# STUDENT PORTAL (مختصر: Planning image + timetable + payments years)
# =========================================================
def student_portal_center():
    st.markdown("## 🎓 Espace Stagiaire")
    tab1, tab2, tab3 = st.tabs(["🔐 Connexion", "🆕 Inscription", "📌 Mon espace"])

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

    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")
        branches_df = ensure_cols(df("Branches"), "Branches")
        branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre disponible.")
            return

        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(ensure_cols(df("Programs"), "Programs"), branch=b)
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_prog")

        grp_df = df_filter(ensure_cols(df("Groups"), "Groups"), branch=b, program_name=p)
        grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
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

            acc = ensure_cols(df("Accounts"), "Accounts")
            if not acc.empty and acc["phone"].astype(str).str.strip().eq(norm(phone)).any():
                st.error("Ce téléphone est déjà inscrit.")
                return

            tr = ensure_cols(df("Trainees"), "Trainees")
            if tr.empty:
                st.error("Aucun stagiaire enregistré.")
                return

            tr2 = tr.copy()
            for c in ["branch", "program", "group", "phone"]:
                tr2[c] = tr2[c].astype(str).str.strip()

            candidates = tr2[(tr2["branch"] == norm(b)) &
                             (tr2["program"] == norm(p)) &
                             (tr2["group"] == norm(g)) &
                             (tr2["phone"] == norm(phone))]
            if candidates.empty:
                st.error("رقم الهاتف موش موجود في Trainees.")
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

    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = ensure_cols(df("Trainees"), "Trainees")
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

            up = st.file_uploader("📸 Ajouter/Changer ma photo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="pp_upl")
            if up is not None:
                img_bytes = up.read()
                st.image(img_bytes, caption="Aperçu", width=150)
                if st.button("Enregistrer ma photo", use_container_width=True, key="pp_save"):
                    upsert_profile_pic(phone, trainee_id, img_bytes)
                    st.success("✅ Photo enregistrée.")
                    st.rerun()

        t1, t2, t3 = st.tabs(["🗓️ Planning Image", "🗓️ Emploi du temps (Couleur)", "💳 Paiements"])

        with t1:
            year = str(datetime.now().year)
            tt = ensure_cols(df("TimetableImages"), "TimetableImages")
            if tt.empty:
                st.info("لا توجد Planning.")
            else:
                tt2 = tt.copy()
                # if sheet has 'year', filter by it. else ignore year.
                has_year = "year" in tt2.columns
                cond = (
                    (tt2["branch"].astype(str).str.strip() == branch) &
                    (tt2["program"].astype(str).str.strip() == program) &
                    (tt2["group"].astype(str).str.strip() == group)
                )
                if has_year:
                    cond = cond & (tt2["year"].astype(str).str.strip() == norm(year))

                m = tt2[cond]
                if m.empty:
                    st.info("لا توجد Planning image.")
                else:
                    r = m.iloc[0].to_dict()
                    st.markdown(f"**📄 {norm(r.get('file_name') or 'Planning')}**")
                    dl = norm(r.get("drive_download_url"))
                    if dl:
                        try:
                            st.image(dl, caption="Planning", use_container_width=True)
                        except Exception:
                            pass
                    link_button("👀 Ouvrir", norm(r.get("drive_view_url")), use_container_width=True, key="stud_pl_open")
                    link_button("⬇️ Télécharger", norm(r.get("drive_download_url")), use_container_width=True, key="stud_pl_dl")

        with t2:
            year = str(datetime.now().year)
            tt = load_timetable(branch, program, group, year)
            html = timetable_grid_html(tt, title=f"Planning ({branch} • {program} • {group} • {year})")
            components.html(html, height=560, scrolling=True)

        with t3:
            pay = ensure_cols(df("Payments"), "Payments")
            if pay.empty:
                st.info("لا توجد بيانات دفوعات.")
            else:
                pay2 = pay[pay["trainee_id"].astype(str).str.strip() == trainee_id].copy()
                years = sorted([y for y in pay2["year"].astype(str).str.strip().unique().tolist() if y])
                if not years:
                    st.info("لا توجد بيانات دفوعات.")
                else:
                    ysel = st.selectbox("السنة", years, index=len(years) - 1, key="stud_pay_year")
                    m = pay2[pay2["year"].astype(str).str.strip() == norm(ysel)]
                    if m.empty:
                        st.info("لا توجد بيانات دفوعات للسنة هذه.")
                    else:
                        rowp = m.iloc[0].to_dict()
                        show = {mo: ("✅" if (norm(rowp.get(mo)).upper() == "TRUE") else "—") for mo in MONTHS}
                        st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

# =========================================================
# STAFF AREA (PlanningImage save uses year if exists, else ignore)
# =========================================================
def staff_work_center():
    st.markdown("## 🛠️ Espace Employé")
    if st.session_state.role != "staff":
        st.info("Connexion Employé من اليسار.")
        return

    staff_branch = norm(st.session_state.user.get("branch"))
    staff_name = f"Staff-{staff_branch}"

    prog_df = df_filter(ensure_cols(df("Programs"), "Programs"), branch=staff_branch)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(ensure_cols(df("Groups"), "Groups"), branch=staff_branch, program_name=program)
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox("Année", [str(datetime.now().year), str(datetime.now().year + 1), str(datetime.now().year - 1)], key="manage_year")

    tab_plan_img, tab_tt = st.tabs(["🗓️ Planning Image (Drive)", "🗓️ Emploi du temps (Couleur)"])

    with tab_plan_img:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            st.info("✅ ارفع الـ Planning كصورة في Google Drive ثم Paste الرابط هنا. (Share: Anyone with the link)")
            file_name = st.text_input("Nom du fichier (اختياري)", key="pl_name")
            share_link = st.text_input("Lien Google Drive (Share link)", key="pl_link")

            tt = ensure_cols(df("TimetableImages"), "TimetableImages")
            has_year = ("year" in tt.columns)

            if st.button("✅ Enregistrer planning (image)", use_container_width=True, key="pl_save"):
                if not norm(share_link):
                    st.error("لازم رابط Drive.")
                else:
                    view_url, dl_url = to_view_and_download(share_link)

                    if has_year:
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
                                "branch": staff_branch, "program": norm(program), "group": norm(group), "year": norm(year),
                                "drive_view_url": view_url, "drive_download_url": dl_url,
                                "file_name": norm(file_name) or "Planning",
                                "uploaded_at": now_str(), "staff_name": staff_name,
                            })
                    else:
                        updated = update_row_by_key(
                            "TimetableImages",
                            ["branch", "program", "group"],
                            [staff_branch, program, group],
                            {"drive_view_url": view_url, "drive_download_url": dl_url,
                             "file_name": norm(file_name) or "Planning",
                             "uploaded_at": now_str(), "staff_name": staff_name}
                        )
                        if not updated:
                            # still append with required headers list; year will be ignored by sheet if not exists? (it will be extra -> so we must NOT include year)
                            ws = spreadsheet().worksheet("TimetableImages")
                            headers = ws.row_values(1)
                            if "year" in headers:
                                append_row("TimetableImages", {
                                    "branch": staff_branch, "program": norm(program), "group": norm(group), "year": norm(year),
                                    "drive_view_url": view_url, "drive_download_url": dl_url,
                                    "file_name": norm(file_name) or "Planning",
                                    "uploaded_at": now_str(), "staff_name": staff_name,
                                })
                            else:
                                # append without year
                                ws.append_row(
                                    [staff_branch, norm(program), norm(group),
                                     view_url, dl_url, (norm(file_name) or "Planning"),
                                     now_str(), staff_name],
                                    value_input_option="USER_ENTERED"
                                )
                                bump_cache()

                    st.success("✅ Planning enregistré.")
                    st.rerun()

            # preview current
            tt = ensure_cols(df("TimetableImages"), "TimetableImages")
            if not tt.empty:
                cond = (
                    (tt["branch"].astype(str).str.strip() == staff_branch) &
                    (tt["program"].astype(str).str.strip() == norm(program)) &
                    (tt["group"].astype(str).str.strip() == norm(group))
                )
                if "year" in tt.columns:
                    cond = cond & (tt["year"].astype(str).str.strip() == norm(year))
                m = tt[cond]
                if not m.empty:
                    r = m.iloc[0].to_dict()
                    st.divider()
                    st.markdown("### Planning الحالي")
                    dl = norm(r.get("drive_download_url"))
                    if dl:
                        try:
                            st.image(dl, caption=norm(r.get("file_name") or "Planning"), use_container_width=True)
                        except Exception:
                            pass
                    link_button("👀 Ouvrir", norm(r.get("drive_view_url")), use_container_width=True, key="pl_open_cur")
                    link_button("⬇️ Télécharger", norm(r.get("drive_download_url")), use_container_width=True, key="pl_dl_cur")

    with tab_tt:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(ensure_cols(df("Subjects"), "Subjects"), branch=staff_branch, program=program, group=group)
            sub = sub[sub["is_active"].astype(str).str.strip().str.lower() != "false"].copy() if not sub.empty else sub
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x]) if not sub.empty else []

            st.markdown("### ➕ Ajouter slot")
            day = st.selectbox("Jour", DAYS_FR, key="tt_day")
            subject_name = st.selectbox("Matière", subjects, key="tt_subject") if subjects else st.text_input("Matière", key="tt_subject_txt")
            if not subjects:
                subject_name = norm(subject_name)
            start = st.text_input("Heure début (HH:MM)", value="08:00", key="tt_start")
            end = st.text_input("Heure fin (HH:MM)", value="09:30", key="tt_end")
            teacher = st.text_input("Nom prof", key="tt_teacher")
            color = st.color_picker("Couleur", value="#E5E7EB", key="tt_color")

            if st.button("✅ Ajouter", use_container_width=True, key="tt_add"):
                if not (norm(day) and norm(start) and norm(end) and norm(subject_name) and norm(teacher)):
                    st.error("لازم: day + start + end + subject + teacher.")
                else:
                    upsert_slot(None, staff_branch, program, group, year, day, norm(start), norm(end), norm(subject_name), norm(teacher), safe_hex_color(color), staff_name)
                    st.success("✅ Slot ajouté.")
                    st.rerun()

            tt = load_timetable(staff_branch, program, group, year)
            st.divider()
            st.markdown("### Preview")
            components.html(timetable_grid_html(tt, title=f"Planning ({staff_branch} • {program} • {group} • {year})"), height=560, scrolling=True)

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
        student_portal_center()
        st.divider()
        st.info("ℹ️ Connexion Employé موجودة في اليسار.")

if __name__ == "__main__":
    try:
        main()
    except APIError as e:
        st.error(explain_api_error(e))
