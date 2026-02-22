import uuid
import base64
import io
import time
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

    # ✅ Planning typed by staff (CRUD) + colors
    "Timetable": ["row_id", "branch", "program", "group", "year",
                  "day", "start", "end", "subject", "room", "teacher", "color",
                  "created_at", "updated_at", "staff_name"],

    # ✅ Profile pics (small base64)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    # ✅ Payments
    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"]
                + MONTHS + ["updated_at", "staff_name"],

    # ✅ Course supports links (Drive manual upload)
    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "drive_view_url", "drive_download_url", "uploaded_at", "staff_name"],

    # ✅ Moyennes (manual + editable)
    "Averages": ["avg_id", "trainee_id", "branch", "program", "group", "year",
                 "semester", "average", "note", "created_at", "updated_at", "staff_name"],
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
        if status == 429 or "quota" in low or "rate" in low:
            return "⚠️ 429 Quota (Google Sheets). جرّب Reboot واستنى شوية.\n" + text[:240]
        if status == 403 or "permission" in low or "forbidden" in low:
            return "❌ 403 Permission. Share Sheet مع service account.\n" + text[:240]
        if status == 404 or "not found" in low:
            return "❌ 404 Not found. تأكد GSHEET_ID صحيح + Share للـ service account.\n" + text[:240]
        return "❌ Google API Error:\n" + (text[:360] if text else str(e))
    except Exception:
        return "❌ Google API Error."

def df_filter(df: pd.DataFrame, **kwargs):
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    out = df.copy()
    for k, v in kwargs.items():
        if k in out.columns:
            out = out[out[k].astype(str).str.strip() == norm(v)]
    return out

def ensure_cols(df: pd.DataFrame, ws_name: str) -> pd.DataFrame:
    """If headers in sheet are missing some expected columns, add them in dataframe to avoid KeyError."""
    expected = REQUIRED_SHEETS.get(ws_name, [])
    if df is None:
        return pd.DataFrame(columns=expected)
    df2 = df.copy()
    for c in expected:
        if c not in df2.columns:
            df2[c] = ""
    # Keep order (expected first, then any extra)
    cols = expected + [c for c in df2.columns if c not in expected]
    return df2[cols]

def safe_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

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

# --- Drive link helpers (manual) ---
def extract_drive_file_id(url: str):
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

def to_view_and_download(url: str):
    fid = extract_drive_file_id(url)
    if not fid:
        return norm(url), norm(url)
    view_url = f"https://drive.google.com/file/d/{fid}/view"
    dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
    return view_url, dl_url

def link_btn(label: str, url: str, key: str = None):
    """Compat: if st.link_button not available, fallback to markdown link."""
    u = norm(url)
    if not u:
        st.button(label, disabled=True, key=key)
        return
    if hasattr(st, "link_button"):
        try:
            st.link_button(label, u, use_container_width=True, key=key)
            return
        except Exception:
            pass
    # fallback
    st.markdown(f"- **{label}:** {u}")

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
# SHEETS SETUP (SAFE)
# =========================================================
def ensure_headers_safe(ws, headers):
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
            sh.add_worksheet(title=ws_name, rows=4000, cols=max(18, len(headers) + 2))
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

def _sheet_read_retry(ws, max_tries=4):
    for i in range(max_tries):
        try:
            return ws.get_all_values()
        except APIError as e:
            msg = str(e).lower()
            if "quota" in msg or "429" in msg or "rate" in msg:
                time.sleep(1.2 * (i + 1))
                continue
            raise
    # last try
    return ws.get_all_values()

@st.cache_data(ttl=180, show_spinner=False)
def read_df(ws_name: str) -> pd.DataFrame:
    ws = spreadsheet().worksheet(ws_name)
    values = _sheet_read_retry(ws)
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    return ensure_cols(df, ws_name)

def append_row(ws_name: str, row: dict):
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_row_by_key(ws_name: str, key_cols, key_vals, updates: dict) -> bool:
    df = read_df(ws_name)
    df = ensure_cols(df, ws_name)
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

    for col_name, val in updates.items():
        if col_name not in headers:
            continue
        ws.update_cell(row_num, headers.index(col_name) + 1, norm(val))

    st.cache_data.clear()
    return True

def delete_row_by_key(ws_name: str, key_col: str, key_val: str) -> bool:
    df = read_df(ws_name)
    df = ensure_cols(df, ws_name)
    if df.empty or key_col not in df.columns:
        return False
    m = df[df[key_col].astype(str).str.strip() == norm(key_val)]
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
def get_profile_pic_bytes(phone: str):
    phone = norm(phone)
    if not phone:
        return None
    try:
        df = read_df("ProfilePics")
    except APIError:
        return None
    df = ensure_cols(df, "ProfilePics")
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
    df = ensure_cols(df, "Payments")
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
    df = ensure_cols(df, "Payments")
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

    ws.update_cell(row_num, headers.index(month) + 1, "TRUE" if paid else "FALSE")
    ws.update_cell(row_num, headers.index("updated_at") + 1, now_str())
    ws.update_cell(row_num, headers.index("staff_name") + 1, staff_name)

    st.cache_data.clear()
    return True

# =========================================================
# TIMETABLE (CRUD + HTML GRID)
# =========================================================
def load_timetable(branch: str, program: str, group: str, year: str) -> pd.DataFrame:
    df = read_df("Timetable")
    df = ensure_cols(df, "Timetable")
    if df.empty:
        return df
    df["branch"] = df["branch"].astype(str).str.strip()
    df["program"] = df["program"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    df["year"] = df["year"].astype(str).str.strip()
    out = df[(df["branch"] == norm(branch)) &
             (df["program"] == norm(program)) &
             (df["group"] == norm(group)) &
             (df["year"] == norm(year))].copy()
    return out

def timetable_html_grid(df: pd.DataFrame) -> str:
    df = ensure_cols(df, "Timetable")
    if df.empty:
        return "<div style='padding:10px'>Aucun planning.</div>"

    # normalize
    df2 = df.copy()
    df2["day"] = df2["day"].astype(str).str.strip()
    df2["start"] = df2["start"].astype(str).str.strip()
    df2["end"] = df2["end"].astype(str).str.strip()
    df2["subject"] = df2["subject"].astype(str).str.strip()
    df2["teacher"] = df2["teacher"].astype(str).str.strip()
    df2["room"] = df2["room"].astype(str).str.strip()
    df2["color"] = df2["color"].astype(str).str.strip()

    day_map = {d: i for i, d in enumerate(DAYS)}
    df2["day_i"] = df2["day"].map(lambda x: day_map.get(x, 99))

    # sort
    df2 = df2.sort_values(by=["day_i", "start", "end"])

    # Build table
    th = "".join([f"<th style='padding:10px;border:1px solid #ddd;background:#fafafa'>{d}</th>" for d in DAYS])
    # for each day: list cards
    tds = []
    for d in DAYS:
        items = df2[df2["day"] == d].copy()
        if items.empty:
            tds.append("<td style='vertical-align:top;padding:10px;border:1px solid #ddd;min-width:180px'>—</td>")
            continue
        blocks = []
        for _, r in items.iterrows():
            c = r.get("color") or "#E8EEF7"
            start = r.get("start") or ""
            end = r.get("end") or ""
            subj = r.get("subject") or ""
            teach = r.get("teacher") or ""
            room = r.get("room") or ""
            blocks.append(
                f"""
                <div style="background:{c};padding:10px;border-radius:10px;margin-bottom:10px;border:1px solid rgba(0,0,0,0.08)">
                  <div style="font-weight:700">{start} → {end}</div>
                  <div style="margin-top:6px"><b>{subj}</b></div>
                  <div style="opacity:0.85;margin-top:4px">{teach}</div>
                  <div style="opacity:0.75;margin-top:2px">{room}</div>
                </div>
                """
            )
        tds.append(f"<td style='vertical-align:top;padding:10px;border:1px solid #ddd;min-width:180px'>{''.join(blocks)}</td>")

    html = f"""
    <div style="overflow:auto">
    <table style="border-collapse:collapse;width:100%;min-width:1100px">
      <thead><tr>{th}</tr></thead>
      <tbody><tr>{''.join(tds)}</tr></tbody>
    </table>
    </div>
    """
    return html

# =========================================================
# AVERAGES (CRUD)
# =========================================================
def load_averages(branch: str, program: str, group: str, trainee_id: str = None) -> pd.DataFrame:
    df = read_df("Averages")
    df = ensure_cols(df, "Averages")
    if df.empty:
        return df
    df["branch"] = df["branch"].astype(str).str.strip()
    df["program"] = df["program"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    out = df[(df["branch"] == norm(branch)) &
             (df["program"] == norm(program)) &
             (df["group"] == norm(group))].copy()
    if trainee_id:
        out = out[out["trainee_id"].astype(str).str.strip() == norm(trainee_id)].copy()
    return out

# =========================================================
# AUTH / SESSION
# =========================================================
def ensure_session():
    st.session_state.setdefault("role", None)     # staff | None
    st.session_state.setdefault("user", {})
    st.session_state.setdefault("student", None)

def logout_staff():
    st.session_state.role = None
    st.session_state.user = {}

def staff_branch_login(branch: str, branch_password: str):
    df = read_df("Branches")
    df = ensure_cols(df, "Branches")
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
    df = read_df("Accounts")
    df = ensure_cols(df, "Accounts")
    if df.empty:
        return None
    df2 = df.copy()
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

    branches_df = read_df("Branches")
    branches_df = ensure_cols(branches_df, "Branches")
    branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []

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
# STUDENT PORTAL
# =========================================================
def student_portal_center():
    st.markdown("## 🎓 Espace Stagiaire")

    tab1, tab2, tab3 = st.tabs(["🔐 Connexion", "🆕 Inscription", "📌 Mon espace"])

    # ---------- Login
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

    # ---------- Register (name free, phone must exist in Trainees)
    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")

        branches_df = read_df("Branches")
        branches_df = ensure_cols(branches_df, "Branches")
        branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre.")
            return

        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(read_df("Programs"), branch=b)
        prog_df = ensure_cols(prog_df, "Programs")
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_prog")

        grp_df = df_filter(read_df("Groups"), branch=b, program_name=p)
        grp_df = ensure_cols(grp_df, "Groups")
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

            acc = read_df("Accounts")
            acc = ensure_cols(acc, "Accounts")
            if not acc.empty and acc["phone"].astype(str).str.strip().eq(norm(phone)).any():
                st.error("Ce téléphone est déjà inscrit.")
                return

            tr = read_df("Trainees")
            tr = ensure_cols(tr, "Trainees")
            if tr.empty:
                st.error("Aucun stagiaire.")
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
                "last_login": "",
            })
            st.success("✅ Compte créé. امشي Connexion.")

    # ---------- My space
    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = read_df("Trainees")
        tr = ensure_cols(tr, "Trainees")
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
                    try:
                        upsert_profile_pic(phone, trainee_id, img_bytes)
                        st.success("✅ Photo enregistrée.")
                        st.rerun()
                    except APIError as e:
                        st.error(explain_api_error(e))

        t1, t2, t3, t4, t5 = st.tabs(["📝 Notes", "🗓️ Planning", "💳 Paiements", "📎 Supports", "📊 Moyennes"])

        with t1:
            gr = read_df("Grades")
            gr = ensure_cols(gr, "Grades")
            grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
            if grf.empty:
                st.info("Aucune note.")
            else:
                grf = grf.sort_values(by=["date", "created_at"], ascending=False)
                st.dataframe(grf[["subject_name", "exam_type", "score", "date", "staff_name", "note"]],
                             use_container_width=True, hide_index=True)

        with t2:
            tt = load_timetable(branch, program, group, str(datetime.now().year))
            st.markdown(timetable_html_grid(tt), unsafe_allow_html=True)

        with t3:
            pay = read_df("Payments")
            pay = ensure_cols(pay, "Payments")

            years = []
            if not pay.empty:
                years = sorted([y for y in pay[pay["trainee_id"].astype(str).str.strip() == trainee_id]["year"].astype(str).str.strip().unique().tolist() if y])
            if not years:
                st.info("لا توجد بيانات دفوعات.")
            else:
                year_sel = st.selectbox("Année", years, key="stud_pay_year")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == trainee_id) &
                        (pay["year"].astype(str).str.strip() == norm(year_sel))].copy()
                if m.empty:
                    st.info("لا توجد بيانات لهذه السنة.")
                else:
                    rowp = m.iloc[0].to_dict()
                    show = {mo: ("✅" if (norm(rowp.get(mo)).upper() == "TRUE") else "❌") for mo in MONTHS}
                    st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

        with t4:
            files = read_df("CourseFiles")
            files = ensure_cols(files, "CourseFiles")
            files = files[(files["branch"].astype(str).str.strip() == branch) &
                          (files["program"].astype(str).str.strip() == program) &
                          (files["group"].astype(str).str.strip() == group)] if not files.empty else pd.DataFrame()
            if files.empty:
                st.info("لا توجد ملفات.")
            else:
                files = files.sort_values(by=["uploaded_at"], ascending=False)
                for _, r in files.iterrows():
                    fid = norm(r.get("file_id")) or uuid.uuid4().hex
                    st.markdown(f"**📌 {norm(r.get('subject_name'))}** — {norm(r.get('file_name'))}")
                    link_btn("👀 Ouvrir", norm(r.get("drive_view_url")), key=f"v_{fid}")
                    link_btn("⬇️ Télécharger", norm(r.get("drive_download_url")), key=f"d_{fid}")
                    st.divider()

        with t5:
            av = read_df("Averages")
            av = ensure_cols(av, "Averages")
            av = av[av["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not av.empty else pd.DataFrame()
            if av.empty:
                st.info("لا توجد Moyennes.")
            else:
                av = av.sort_values(by=["year", "semester"], ascending=False)
                show = av[["year", "semester", "average", "note", "updated_at", "staff_name"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)

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

    prog_df = df_filter(read_df("Programs"), branch=staff_branch)
    prog_df = ensure_cols(prog_df, "Programs")
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            grp_df = ensure_cols(grp_df, "Groups")
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox(
            "Année",
            [str(datetime.now().year), str(datetime.now().year + 1), str(datetime.now().year - 1)],
            key="pay_year"
        )

    tab_stag, tab_gr, tab_pay, tab_tt, tab_sup, tab_avg = st.tabs(
        ["👤 Stagiaires", "📝 Notes", "💳 Paiements", "🗓️ Planning", "📎 Supports (Liens Drive)", "📊 Moyennes"]
    )

    # ---------------- Stagiaires
    with tab_stag:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            cur = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            cur = ensure_cols(cur, "Trainees")
            st.dataframe(cur[["full_name", "phone", "status", "created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)

            st.markdown("### ➕ Ajouter stagiaire")
            name = st.text_input("Nom", key="add_tr_name")
            phone = st.text_input("Téléphone", key="add_tr_phone")
            status = st.selectbox("Statut", ["active", "inactive"], key="add_tr_status")
            if st.button("Enregistrer", use_container_width=True, key="btn_add_tr"):
                if not norm(name) or not norm(phone):
                    st.error("Nom + téléphone obligatoire.")
                else:
                    # avoid duplicate phone in same branch/program/group
                    exists = cur[cur["phone"].astype(str).str.strip() == norm(phone)] if not cur.empty else pd.DataFrame()
                    if not exists.empty:
                        st.error("⚠️ نفس رقم الهاتف موجود بالفعل في هذا groupe.")
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
                df = pd.read_excel(up)
                df.columns = [c.strip() for c in df.columns]
                st.dataframe(df.head(20), use_container_width=True)

                if st.button("✅ Importer maintenant", use_container_width=True, key="do_imp"):
                    if "full_name" not in df.columns or "phone" not in df.columns:
                        st.error("لازم full_name و phone.")
                    else:
                        existing = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
                        existing = ensure_cols(existing, "Trainees")
                        existing_phones = set(existing["phone"].astype(str).str.strip().tolist()) if not existing.empty else set()

                        count = 0
                        for _, r in df.iterrows():
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

    # ---------------- Notes (add + edit + delete)
    with tab_gr:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            tr = ensure_cols(tr, "Trainees")
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            sub = ensure_cols(sub, "Subjects")

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

                # ---- Edit/Delete note
                st.divider()
                st.markdown("### ✏️ Modifier / Supprimer une note")

                gr_all = read_df("Grades")
                gr_all = ensure_cols(gr_all, "Grades")
                if gr_all.empty:
                    st.info("لا توجد Notes.")
                else:
                    gr_all["branch"] = gr_all["branch"].astype(str).str.strip()
                    gr_all["program"] = gr_all["program"].astype(str).str.strip()
                    gr_all["group"] = gr_all["group"].astype(str).str.strip()

                    grf = gr_all[(gr_all["branch"] == norm(staff_branch)) &
                                 (gr_all["program"] == norm(program)) &
                                 (gr_all["group"] == norm(group)) &
                                 (gr_all["trainee_id"].astype(str).str.strip() == norm(trainee_id))].copy()

                    if grf.empty:
                        st.info("لا توجد Notes لهذا المتكون.")
                    else:
                        grf["label"] = (
                            grf["date"].astype(str) + " | " +
                            grf["subject_name"].astype(str) + " | " +
                            grf["exam_type"].astype(str) + " | " +
                            grf["score"].astype(str) + " | " +
                            "ID:" + grf["grade_id"].astype(str)
                        )
                        pick = st.selectbox("اختار Note", grf["label"].tolist(), key="gr_edit_pick")
                        row = grf[grf["label"] == pick].iloc[0].to_dict()
                        grade_id = norm(row.get("grade_id"))

                        subject_e = st.text_input("Matière", value=norm(row.get("subject_name")), key="gr_edit_subject")
                        examtype_e = st.text_input("Type examen", value=norm(row.get("exam_type")), key="gr_edit_examtype")
                        score_e = st.number_input("Note", min_value=0.0, max_value=20.0,
                                                  value=safe_float(row.get("score"), 0.0),
                                                  step=0.25, key="gr_edit_score")
                        date_e = st.text_input("Date (YYYY-MM-DD)", value=norm(row.get("date")), key="gr_edit_date")
                        note_e = st.text_area("Remarque", value=norm(row.get("note")), key="gr_edit_note")

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("💾 Enregistrer modification", use_container_width=True, key="gr_edit_save"):
                                ok = update_row_by_key(
                                    "Grades",
                                    ["grade_id"], [grade_id],
                                    {
                                        "subject_name": norm(subject_e),
                                        "exam_type": norm(examtype_e),
                                        "score": str(score_e),
                                        "date": norm(date_e),
                                        "note": norm(note_e),
                                        "staff_name": staff_name,
                                        "created_at": now_str(),
                                    }
                                )
                                if ok:
                                    st.success("✅ تم تعديل الـ Note")
                                    st.rerun()
                                else:
                                    st.error("❌ ما نجّمش نحدّث")
                        with c2:
                            if st.button("🗑️ Supprimer", use_container_width=True, key="gr_edit_delete"):
                                if delete_row_by_key("Grades", "grade_id", grade_id):
                                    st.success("✅ تم حذف الـ Note")
                                    st.rerun()
                                else:
                                    st.error("❌ ما نجّمش نحذف")

    # ---------------- Payments
    with tab_pay:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            tr = ensure_cols(tr, "Trainees")
            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr = tr.copy()
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="pay_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                ensure_payment_row(trainee_id, staff_branch, program, group, year, staff_name)

                pay = read_df("Payments")
                pay = ensure_cols(pay, "Payments")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
                        (pay["year"].astype(str).str.strip() == norm(year))].copy()
                rowp = m.iloc[0].to_dict() if not m.empty else {}

                cols = st.columns(4)
                for i, mo in enumerate(MONTHS):
                    paid = (norm(rowp.get(mo)).upper() == "TRUE")
                    with cols[i % 4]:
                        new_paid = st.checkbox(mo, value=paid, key=f"ck_{mo}_{trainee_id}_{year}")
                        if new_paid != paid:
                            set_payment_month(trainee_id, year, mo, new_paid, staff_name)
                            st.rerun()

    # ---------------- Timetable (CRUD) + preview
    with tab_tt:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            st.markdown("### 🗓️ Planning — الموظف يكتب الأيام/الوقت/المادة/المكون + لون")

            # subjects list for select
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            sub = ensure_cols(sub, "Subjects")
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x]) if not sub.empty else []

            if not subjects:
                st.warning("زيد matières قبل باش تختار منهم في planning.")
                st.stop()

            tt = load_timetable(staff_branch, program, group, year)

            st.markdown("#### ➕ Ajouter séance")
            c1, c2, c3 = st.columns(3)
            with c1:
                day = st.selectbox("Jour", DAYS, key="tt_day_add")
                start = st.text_input("De (HH:MM)", value="18:00", key="tt_start_add")
                end = st.text_input("À (HH:MM)", value="19:30", key="tt_end_add")
            with c2:
                subject = st.selectbox("Matière", subjects, key="tt_subject_add")
                teacher = st.text_input("Nom du prof", key="tt_teacher_add")
                room = st.text_input("Salle (optionnel)", key="tt_room_add")
            with c3:
                color = st.color_picker("Couleur", value="#E8EEF7", key="tt_color_add")

            if st.button("✅ Ajouter au planning", use_container_width=True, key="tt_add_btn"):
                append_row("Timetable", {
                    "row_id": f"TT-{uuid.uuid4().hex[:10].upper()}",
                    "branch": staff_branch,
                    "program": norm(program),
                    "group": norm(group),
                    "year": norm(year),
                    "day": norm(day),
                    "start": norm(start),
                    "end": norm(end),
                    "subject": norm(subject),
                    "room": norm(room),
                    "teacher": norm(teacher),
                    "color": norm(color),
                    "created_at": now_str(),
                    "updated_at": "",
                    "staff_name": staff_name,
                })
                st.success("✅ Ajouté.")
                st.rerun()

            st.divider()
            st.markdown("#### ✏️ Modifier / 🗑️ Supprimer séance")

            if tt.empty:
                st.info("Aucune séance.")
            else:
                tt2 = ensure_cols(tt, "Timetable").copy()
                tt2["label"] = (
                    tt2["day"].astype(str) + " | " +
                    tt2["start"].astype(str) + "-" + tt2["end"].astype(str) + " | " +
                    tt2["subject"].astype(str) + " | " +
                    tt2["teacher"].astype(str) + " | ID:" + tt2["row_id"].astype(str)
                )
                pick = st.selectbox("Séance", tt2["label"].tolist(), key="tt_pick_edit")
                row = tt2[tt2["label"] == pick].iloc[0].to_dict()
                row_id = norm(row.get("row_id"))

                e1, e2, e3 = st.columns(3)
                with e1:
                    day_e = st.selectbox("Jour", DAYS, index=max(0, DAYS.index(norm(row.get("day"))) if norm(row.get("day")) in DAYS else 0), key="tt_day_edit")
                    start_e = st.text_input("De (HH:MM)", value=norm(row.get("start")), key="tt_start_edit")
                    end_e = st.text_input("À (HH:MM)", value=norm(row.get("end")), key="tt_end_edit")
                with e2:
                    subj_e = st.selectbox("Matière", subjects, index=max(0, subjects.index(norm(row.get("subject"))) if norm(row.get("subject")) in subjects else 0), key="tt_subject_edit")
                    teacher_e = st.text_input("Nom du prof", value=norm(row.get("teacher")), key="tt_teacher_edit")
                    room_e = st.text_input("Salle", value=norm(row.get("room")), key="tt_room_edit")
                with e3:
                    color_default = norm(row.get("color")) or "#E8EEF7"
                    # ensure valid hex for color_picker
                    if not (color_default.startswith("#") and len(color_default) == 7):
                        color_default = "#E8EEF7"
                    color_e = st.color_picker("Couleur", value=color_default, key="tt_color_edit")

                cA, cB = st.columns(2)
                with cA:
                    if st.button("💾 Enregistrer modification", use_container_width=True, key="tt_save_edit"):
                        ok = update_row_by_key(
                            "Timetable",
                            ["row_id"], [row_id],
                            {
                                "day": norm(day_e),
                                "start": norm(start_e),
                                "end": norm(end_e),
                                "subject": norm(subj_e),
                                "teacher": norm(teacher_e),
                                "room": norm(room_e),
                                "color": norm(color_e),
                                "updated_at": now_str(),
                                "staff_name": staff_name,
                            }
                        )
                        if ok:
                            st.success("✅ Modifié.")
                            st.rerun()
                        else:
                            st.error("❌ ما نجّمش نحدّث.")
                with cB:
                    if st.button("🗑️ Supprimer séance", use_container_width=True, key="tt_delete_edit"):
                        if delete_row_by_key("Timetable", "row_id", row_id):
                            st.success("✅ Supprimé.")
                            st.rerun()
                        else:
                            st.error("❌ ما نجّمش نحذف.")

            st.divider()
            st.markdown("#### 👀 Preview planning (نفس اللي يشوفو المتكون)")
            tt_preview = load_timetable(staff_branch, program, group, year)
            st.markdown(timetable_html_grid(tt_preview), unsafe_allow_html=True)

    # ---------------- Supports (manual links)
    with tab_sup:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            sub = ensure_cols(sub, "Subjects")
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x]) if not sub.empty else []

            if not subjects:
                st.warning("زيد matières قبل.")
            else:
                st.info("✅ ارفع الملف يدويًا في Google Drive ثم Paste الرابط هنا (Share: Anyone with the link).")
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

            files = read_df("CourseFiles")
            files = ensure_cols(files, "CourseFiles")
            files = files[(files["branch"].astype(str).str.strip() == staff_branch) &
                          (files["program"].astype(str).str.strip() == norm(program)) &
                          (files["group"].astype(str).str.strip() == norm(group))] if not files.empty else pd.DataFrame()
            if not files.empty:
                st.divider()
                st.markdown("### Fichiers enregistrés")
                files = files.sort_values(by=["uploaded_at"], ascending=False)
                st.dataframe(files[["subject_name", "file_name", "uploaded_at", "staff_name"]],
                             use_container_width=True, hide_index=True)

    # ---------------- Averages (add + edit + delete)
    with tab_avg:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            st.markdown("### 📊 Moyennes — إضافة/تعديل/حذف")

            tr = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            tr = ensure_cols(tr, "Trainees")
            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr = tr.copy()
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="avg_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                sem = st.selectbox("Semestre", ["S1", "S2", "Annuel"], key="avg_sem")
                avg_val = st.number_input("Moyenne", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="avg_val")
                avg_note = st.text_area("Remarque (اختياري)", key="avg_note")

                if st.button("✅ Enregistrer moyenne", use_container_width=True, key="avg_save"):
                    # upsert by (trainee_id, year, semester)
                    ok = update_row_by_key(
                        "Averages",
                        ["trainee_id", "year", "semester"],
                        [trainee_id, year, sem],
                        {
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "average": str(avg_val),
                            "note": norm(avg_note),
                            "updated_at": now_str(),
                            "staff_name": staff_name,
                        }
                    )
                    if not ok:
                        append_row("Averages", {
                            "avg_id": f"AV-{uuid.uuid4().hex[:8].upper()}",
                            "trainee_id": trainee_id,
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "year": norm(year),
                            "semester": norm(sem),
                            "average": str(avg_val),
                            "note": norm(avg_note),
                            "created_at": now_str(),
                            "updated_at": now_str(),
                            "staff_name": staff_name,
                        })
                    st.success("✅ Moyenne enregistrée.")
                    st.rerun()

                st.divider()
                st.markdown("### ✏️ Modifier / 🗑️ Supprimer moyenne")

                av = load_averages(staff_branch, program, group, trainee_id=trainee_id)
                av = ensure_cols(av, "Averages")
                av["year"] = av["year"].astype(str).str.strip()
                av = av[av["year"] == norm(year)].copy()

                if av.empty:
                    st.info("لا توجد Moyennes لهذه السنة.")
                else:
                    av["label"] = (
                        av["year"].astype(str) + " | " +
                        av["semester"].astype(str) + " | " +
                        av["average"].astype(str) + " | ID:" + av["avg_id"].astype(str)
                    )
                    pick = st.selectbox("Moyenne", av["label"].tolist(), key="avg_pick")
                    row = av[av["label"] == pick].iloc[0].to_dict()
                    avg_id = norm(row.get("avg_id"))

                    sem_e = st.selectbox("Semestre", ["S1", "S2", "Annuel"],
                                         index=max(0, ["S1", "S2", "Annuel"].index(norm(row.get("semester"))) if norm(row.get("semester")) in ["S1", "S2", "Annuel"] else 0),
                                         key="avg_sem_edit")
                    avg_e = st.number_input("Moyenne", min_value=0.0, max_value=20.0,
                                            value=safe_float(row.get("average"), 0.0),
                                            step=0.25, key="avg_val_edit")
                    note_e = st.text_area("Remarque", value=norm(row.get("note")), key="avg_note_edit")

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Enregistrer modification", use_container_width=True, key="avg_edit_save"):
                            ok = update_row_by_key(
                                "Averages",
                                ["avg_id"], [avg_id],
                                {
                                    "semester": norm(sem_e),
                                    "average": str(avg_e),
                                    "note": norm(note_e),
                                    "updated_at": now_str(),
                                    "staff_name": staff_name,
                                }
                            )
                            if ok:
                                st.success("✅ Modifié.")
                                st.rerun()
                            else:
                                st.error("❌ ما نجّمش نحدّث.")
                    with c2:
                        if st.button("🗑️ Supprimer", use_container_width=True, key="avg_edit_del"):
                            if delete_row_by_key("Averages", "avg_id", avg_id):
                                st.success("✅ Supprimé.")
                                st.rerun()
                            else:
                                st.error("❌ ما نجّمش نحذف.")

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
    main()
