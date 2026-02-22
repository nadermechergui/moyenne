import uuid
import base64
import io
import time
import re
from datetime import datetime, date as dt_date

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, WorksheetNotFound
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

    # Timetable typed by staff (CRUD + colors)
    "Timetable": ["row_id", "branch", "program", "group", "year",
                  "day", "start", "end", "subject", "room", "teacher", "color",
                  "created_at", "updated_at", "staff_name"],

    # Profile pics (small base64)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    # Payments
    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"]
                + MONTHS + ["updated_at", "staff_name"],

    # Course supports links (Drive manual)
    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "drive_view_url", "drive_download_url", "uploaded_at", "staff_name"],

    # Averages (editable)
    "Averages": ["avg_id", "trainee_id", "branch", "program", "group", "year",
                 "semester", "average", "note", "created_at", "updated_at", "staff_name"],
}

# =========================================================
# UTILS
# =========================================================
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

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

def ensure_cols(df: pd.DataFrame, ws_name: str) -> pd.DataFrame:
    expected = REQUIRED_SHEETS.get(ws_name, [])
    if df is None:
        return pd.DataFrame(columns=expected)
    df2 = df.copy()
    for c in expected:
        if c not in df2.columns:
            df2[c] = ""
    # keep expected order first
    cols = expected + [c for c in df2.columns if c not in expected]
    return df2[cols]

def df_filter(df: pd.DataFrame, **kwargs):
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    out = df.copy()
    for k, v in kwargs.items():
        if k in out.columns:
            out = out[out[k].astype(str).str.strip() == norm(v)]
    return out

def safe_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def valid_hex_color(c: str, fallback="#E8EEF7"):
    c = norm(c)
    if HEX_RE.match(c):
        return c
    return fallback

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

def link_btn(label: str, url: str, key: str):
    u = norm(url)
    if not u:
        st.button(label, disabled=True, use_container_width=True, key=key)
        return
    # Streamlit versions differ. If link_button exists, use it; otherwise show a clickable markdown link.
    if hasattr(st, "link_button"):
        try:
            st.link_button(label, u, use_container_width=True, key=key)
            return
        except Exception:
            pass
    st.markdown(f"- **{label}**: {u}")

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
# SHEETS (AUTO CREATE + RETRIES)
# =========================================================
def ensure_headers_safe(ws, headers):
    try:
        rng = ws.get("1:1")
    except Exception:
        rng = []
    row1 = rng[0] if (rng and len(rng) > 0) else []
    row1 = [norm(x) for x in row1]
    if len(row1) == 0 or all(x == "" for x in row1):
        ws.append_row(headers, value_input_option="RAW")
        return
    if row1 != headers:
        st.warning(f"⚠️ Sheet '{ws.title}' headers مختلفة. ما عملتش مسح. صحّح الهيدرز يدويًا إذا تحب.")

def get_ws(ws_name: str):
    sh = spreadsheet()
    try:
        ws = sh.worksheet(ws_name)
        ensure_headers_safe(ws, REQUIRED_SHEETS[ws_name])
        return ws
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=ws_name, rows=4000, cols=max(20, len(REQUIRED_SHEETS[ws_name]) + 2))
        ensure_headers_safe(ws, REQUIRED_SHEETS[ws_name])
        return ws

def ensure_all_sheets():
    for name in REQUIRED_SHEETS.keys():
        _ = get_ws(name)

def ensure_schema_once():
    if st.session_state.get("schema_ok", False):
        return
    try:
        ensure_all_sheets()
        st.session_state.schema_ok = True
    except APIError as e:
        st.error(explain_api_error(e))

def _sheet_read_retry(ws, max_tries=4):
    last_err = None
    for i in range(max_tries):
        try:
            return ws.get_all_values()
        except APIError as e:
            last_err = e
            msg = str(e).lower()
            if "quota" in msg or "429" in msg or "rate" in msg:
                time.sleep(1.2 * (i + 1))
                continue
            raise
    if last_err:
        raise last_err
    return ws.get_all_values()

@st.cache_data(ttl=180, show_spinner=False)
def read_df(ws_name: str) -> pd.DataFrame:
    ws = get_ws(ws_name)
    values = _sheet_read_retry(ws)
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    return ensure_cols(df, ws_name)

def append_row(ws_name: str, row: dict):
    ws = get_ws(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_row_by_key(ws_name: str, key_cols, key_vals, updates: dict) -> bool:
    df = ensure_cols(read_df(ws_name), ws_name)
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
    ws = get_ws(ws_name)
    headers = REQUIRED_SHEETS[ws_name]

    for col_name, val in updates.items():
        if col_name not in headers:
            continue
        ws.update_cell(row_num, headers.index(col_name) + 1, norm(val))

    st.cache_data.clear()
    return True

def delete_row_by_key(ws_name: str, key_col: str, key_val: str) -> bool:
    df = ensure_cols(read_df(ws_name), ws_name)
    if df.empty or key_col not in df.columns:
        return False
    m = df[df[key_col].astype(str).str.strip() == norm(key_val)]
    if m.empty:
        return False
    idx = m.index[0]
    row_num = idx + 2
    ws = get_ws(ws_name)
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
        df = ensure_cols(read_df("ProfilePics"), "ProfilePics")
    except APIError:
        return None
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
    df = ensure_cols(read_df("Payments"), "Payments")
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
    df = ensure_cols(read_df("Payments"), "Payments")
    if df.empty:
        return False
    m = df[(df["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
           (df["year"].astype(str).str.strip() == norm(year))]
    if m.empty:
        return False

    idx = m.index[0]
    row_num = idx + 2
    ws = get_ws("Payments")
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
    df = ensure_cols(read_df("Timetable"), "Timetable")
    if df.empty:
        return df
    for c in ["branch", "program", "group", "year", "day", "start", "end"]:
        df[c] = df[c].astype(str).str.strip()
    return df[(df["branch"] == norm(branch)) &
              (df["program"] == norm(program)) &
              (df["group"] == norm(group)) &
              (df["year"] == norm(year))].copy()

def timetable_html_grid(df: pd.DataFrame) -> str:
    df = ensure_cols(df, "Timetable")
    if df.empty:
        return "<div style='padding:10px'>Aucun planning.</div>"

    df2 = df.copy()
    for c in ["day", "start", "end", "subject", "teacher", "room", "color"]:
        df2[c] = df2[c].astype(str).str.strip()

    day_map = {d: i for i, d in enumerate(DAYS)}
    df2["day_i"] = df2["day"].map(lambda x: day_map.get(x, 99))

    # Sort safely
    df2 = df2.sort_values(by=["day_i", "start", "end"], ascending=True)

    th = "".join([f"<th style='padding:10px;border:1px solid #ddd;background:#fafafa'>{d}</th>" for d in DAYS])
    tds = []
    for d in DAYS:
        items = df2[df2["day"] == d].copy()
        if items.empty:
            tds.append("<td style='vertical-align:top;padding:10px;border:1px solid #ddd;min-width:180px'>—</td>")
            continue

        blocks = []
        for _, r in items.iterrows():
            c = valid_hex_color(r.get("color"), "#E8EEF7")
            start = norm(r.get("start"))
            end = norm(r.get("end"))
            subj = norm(r.get("subject"))
            teach = norm(r.get("teacher"))
            room = norm(r.get("room"))

            blocks.append(
                f"""
                <div style="background:{c};padding:10px;border-radius:12px;margin-bottom:10px;border:1px solid rgba(0,0,0,0.08)">
                  <div style="font-weight:800">{start} → {end}</div>
                  <div style="margin-top:6px"><b>{subj}</b></div>
                  <div style="opacity:0.88;margin-top:4px">{teach}</div>
                  <div style="opacity:0.75;margin-top:2px">{room}</div>
                </div>
                """
            )
        tds.append(f"<td style='vertical-align:top;padding:10px;border:1px solid #ddd;min-width:180px'>{''.join(blocks)}</td>")

    return f"""
    <div style="overflow:auto">
    <table style="border-collapse:collapse;width:100%;min-width:1100px">
      <thead><tr>{th}</tr></thead>
      <tbody><tr>{''.join(tds)}</tr></tbody>
    </table>
    </div>
    """

# =========================================================
# AVERAGES
# =========================================================
def load_averages(branch: str, program: str, group: str, trainee_id: str = None) -> pd.DataFrame:
    df = ensure_cols(read_df("Averages"), "Averages")
    if df.empty:
        return df
    for c in ["branch", "program", "group", "trainee_id", "year", "semester"]:
        df[c] = df[c].astype(str).str.strip()
    out = df[(df["branch"] == norm(branch)) &
             (df["program"] == norm(program)) &
             (df["group"] == norm(group))].copy()
    if trainee_id:
        out = out[out["trainee_id"] == norm(trainee_id)].copy()
    return out

# =========================================================
# AUTH / SESSION
# =========================================================
def ensure_session():
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("user", {})
    st.session_state.setdefault("student", None)

def logout_staff():
    st.session_state.role = None
    st.session_state.user = {}

def staff_branch_login(branch: str, branch_password: str):
    df = ensure_cols(read_df("Branches"), "Branches")
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
    df = ensure_cols(read_df("Accounts"), "Accounts")
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
# SIDEBAR STAFF LOGIN
# =========================================================
def sidebar_staff_login():
    st.sidebar.markdown("## 👨‍💼 Connexion Employé")

    branches_df = ensure_cols(read_df("Branches"), "Branches")
    branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []

    if st.session_state.role == "staff":
        br = norm(st.session_state.user.get("branch"))
        st.sidebar.success(f"Connecté: {br}")

        st.sidebar.divider()
        st.sidebar.markdown("### 🧰 Maintenance")
        if st.sidebar.button("Initialiser / Vérifier les Sheets", use_container_width=True, key="btn_init_schema"):
            try:
                ensure_all_sheets()
                st.sidebar.success("✅ OK")
            except APIError as e:
                st.sidebar.error(explain_api_error(e))

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

    # --------- Login
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

    # --------- Register
    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")

        branches_df = ensure_cols(read_df("Branches"), "Branches")
        branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre.")
            return

        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(ensure_cols(read_df("Programs"), "Programs"), branch=b)
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_prog")

        grp_df = df_filter(ensure_cols(read_df("Groups"), "Groups"), branch=b, program_name=p)
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

            acc = ensure_cols(read_df("Accounts"), "Accounts")
            if not acc.empty and acc["phone"].astype(str).str.strip().eq(norm(phone)).any():
                st.error("Ce téléphone est déjà inscrit.")
                return

            tr = ensure_cols(read_df("Trainees"), "Trainees")
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

    # --------- My Space
    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = ensure_cols(read_df("Trainees"), "Trainees")
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
            gr = ensure_cols(read_df("Grades"), "Grades")
            grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
            if grf.empty:
                st.info("Aucune note.")
            else:
                grf = grf.sort_values(by=["date", "created_at"], ascending=False)
                st.dataframe(grf[["subject_name", "exam_type", "score", "date", "staff_name", "note"]],
                             use_container_width=True, hide_index=True)

        with t2:
            year_now = str(datetime.now().year)
            year_sel = st.selectbox("Année", [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)],
                                    index=1, key="stud_tt_year")
            tt = load_timetable(branch, program, group, year_sel)
            st.markdown(timetable_html_grid(tt), unsafe_allow_html=True)

        with t3:
            pay = ensure_cols(read_df("Payments"), "Payments")
            years = []
            if not pay.empty:
                years = sorted([y for y in pay[pay["trainee_id"].astype(str).str.strip() == trainee_id]["year"]
                               .astype(str).str.strip().unique().tolist() if y])
            if not years:
                st.info("لا توجد بيانات دفوعات.")
            else:
                year_pick = st.selectbox("Année", years, key="stud_pay_year")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == trainee_id) &
                        (pay["year"].astype(str).str.strip() == norm(year_pick))].copy()
                if m.empty:
                    st.info("لا توجد بيانات لهذه السنة.")
                else:
                    rowp = m.iloc[0].to_dict()
                    show = {mo: ("✅" if (norm(rowp.get(mo)).upper() == "TRUE") else "❌") for mo in MONTHS}
                    st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

        with t4:
            files = ensure_cols(read_df("CourseFiles"), "CourseFiles")
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
                    link_btn("👀 Ouvrir", norm(r.get("drive_view_url")), key=f"stud_v_{fid}")
                    link_btn("⬇️ Télécharger", norm(r.get("drive_download_url")), key=f"stud_d_{fid}")
                    st.divider()

        with t5:
            av = ensure_cols(read_df("Averages"), "Averages")
            av = av[av["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not av.empty else pd.DataFrame()
            if av.empty:
                st.info("لا توجد Moyennes.")
            else:
                av["year"] = av["year"].astype(str).str.strip()
                av["semester"] = av["semester"].astype(str).str.strip()
                av = av.sort_values(by=["year", "semester"], ascending=False)
                show = av[["year", "semester", "average", "note", "updated_at", "staff_name"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)

# =========================================================
# STAFF AREA (FULL)
# =========================================================
def staff_work_center():
    st.markdown("## 🛠️ Espace Employé")
    if st.session_state.role != "staff":
        st.info("Connexion Employé من اليسار.")
        return

    staff_branch = norm(st.session_state.user.get("branch"))
    staff_name = f"Staff-{staff_branch}"

    # Load once to reduce quota
    programs_df = ensure_cols(read_df("Programs"), "Programs")
    groups_df = ensure_cols(read_df("Groups"), "Groups")
    trainees_df = ensure_cols(read_df("Trainees"), "Trainees")
    subjects_df = ensure_cols(read_df("Subjects"), "Subjects")

    prog_df = df_filter(programs_df, branch=staff_branch)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        groups = []
        if program:
            grp_df = df_filter(groups_df, branch=staff_branch, program_name=program)
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox(
            "Année",
            [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)],
            index=1,
            key="pay_year",
        )

    tab_stag, tab_gr, tab_pay, tab_tt, tab_sup, tab_avg = st.tabs(
        ["👤 Stagiaires", "📝 Notes (CRUD)", "💳 Paiements", "🗓️ Planning (CRUD)", "📎 Supports (Liens Drive)", "📊 Moyennes (CRUD)"]
    )

    # ------------------ Stagiaires
    with tab_stag:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            cur = trainees_df.copy()
            for c in ["branch", "program", "group", "phone", "full_name"]:
                cur[c] = cur[c].astype(str).str.strip()
            cur = cur[(cur["branch"] == staff_branch) & (cur["program"] == norm(program)) & (cur["group"] == norm(group))].copy()

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
                    # prevent duplicate phone in same group/program/branch
                    if not cur.empty and cur["phone"].astype(str).str.strip().eq(norm(phone)).any():
                        st.error("❌ نفس رقم الهاتف موجود مسبقًا في نفس المجموعة.")
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
                        existing_phones = set(cur["phone"].astype(str).str.strip().tolist()) if not cur.empty else set()
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

    # ------------------ Notes CRUD
    with tab_gr:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = trainees_df.copy()
            for c in ["branch", "program", "group", "phone", "full_name", "trainee_id"]:
                tr[c] = tr[c].astype(str).str.strip()
            tr = tr[(tr["branch"] == staff_branch) & (tr["program"] == norm(program)) & (tr["group"] == norm(group))].copy()

            sub = subjects_df.copy()
            for c in ["branch", "program", "group", "subject_name", "is_active"]:
                sub[c] = sub[c].astype(str).str.strip()
            sub = sub[(sub["branch"] == staff_branch) & (sub["program"] == norm(program)) & (sub["group"] == norm(group))].copy()
            sub = sub[sub["is_active"].astype(str).str.lower() != "false"].copy()

            if tr.empty:
                st.warning("لا يوجد stagiaires.")
            elif sub.empty:
                st.warning("زيد matières قبل (Sheet Subjects).")
            else:
                tr["label"] = tr["full_name"] + " — " + tr["phone"] + " — " + tr["trainee_id"]
                chosen = st.selectbox("Stagiaire", tr["label"].tolist(), key="gr_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x])

                st.markdown("### ➕ Ajouter note")
                subject_name = st.selectbox("Matière", subjects, key="gr_subject_add")
                exam_type = st.text_input("Type examen (DS1/TP/Examen...)", key="gr_examtype_add")
                score = st.number_input("Note", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="gr_score_add")
                d = st.date_input("Date", value=datetime.now().date(), key="gr_date_add")
                note_txt = st.text_area("Remarque", key="gr_note_add")

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
                            "date": str(d),
                            "staff_name": staff_name,
                            "note": norm(note_txt),
                            "created_at": now_str(),
                        })
                        st.success("✅ Note enregistrée.")
                        st.rerun()

                st.divider()
                st.markdown("### ✏️/🗑️ Modifier أو حذف ملاحظات")
                grades = ensure_cols(read_df("Grades"), "Grades")
                grades = grades[grades["trainee_id"].astype(str).str.strip() == norm(trainee_id)].copy() if not grades.empty else pd.DataFrame()
                if grades.empty:
                    st.info("لا توجد Notes لهذا المتكون.")
                else:
                    grades["date"] = grades["date"].astype(str).str.strip()
                    grades["created_at"] = grades["created_at"].astype(str).str.strip()
                    grades = grades.sort_values(by=["date", "created_at"], ascending=False)

                    grades["label"] = (
                        grades["date"].astype(str) + " | " +
                        grades["subject_name"].astype(str) + " | " +
                        grades["exam_type"].astype(str) + " | " +
                        grades["score"].astype(str) + " | " +
                        grades["grade_id"].astype(str)
                    )
                    pick = st.selectbox("Choisir note", grades["label"].tolist(), key="gr_pick_edit")
                    row = grades[grades["label"] == pick].iloc[0].to_dict()

                    gid = norm(row.get("grade_id"))
                    c1, c2 = st.columns(2)
                    with c1:
                        s_e = st.selectbox("Matière", subjects, index=max(0, subjects.index(norm(row.get("subject_name"))) if norm(row.get("subject_name")) in subjects else 0),
                                           key="gr_subject_edit")
                        t_e = st.text_input("Type examen", value=norm(row.get("exam_type")), key="gr_examtype_edit")
                        sc_e = st.number_input("Note", min_value=0.0, max_value=20.0, value=float(safe_float(row.get("score"), 0.0)),
                                               step=0.25, key="gr_score_edit")
                    with c2:
                        # parse date safe
                        try:
                            dd = datetime.fromisoformat(norm(row.get("date"))).date()
                        except Exception:
                            dd = datetime.now().date()
                        d_e = st.date_input("Date", value=dd, key="gr_date_edit")
                        n_e = st.text_area("Remarque", value=norm(row.get("note")), key="gr_note_edit")

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("💾 Sauvegarder modification", use_container_width=True, key="gr_save_edit"):
                            ok = update_row_by_key(
                                "Grades",
                                ["grade_id"], [gid],
                                {
                                    "subject_name": norm(s_e),
                                    "exam_type": norm(t_e),
                                    "score": str(sc_e),
                                    "date": str(d_e),
                                    "note": norm(n_e),
                                    "staff_name": staff_name,
                                }
                            )
                            if ok:
                                st.success("✅ تم التعديل")
                                st.rerun()
                            else:
                                st.error("❌ ما لقايناش السطر باش نعدلو")
                    with b2:
                        if st.button("🗑️ Supprimer note", use_container_width=True, key="gr_del"):
                            if delete_row_by_key("Grades", "grade_id", gid):
                                st.success("✅ تحذفت")
                                st.rerun()
                            else:
                                st.error("❌ ما تحذفتش")

    # ------------------ Payments
    with tab_pay:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = trainees_df.copy()
            for c in ["branch", "program", "group", "phone", "full_name", "trainee_id"]:
                tr[c] = tr[c].astype(str).str.strip()
            tr = tr[(tr["branch"] == staff_branch) & (tr["program"] == norm(program)) & (tr["group"] == norm(group))].copy()

            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr["label"] = tr["full_name"] + " — " + tr["phone"] + " — " + tr["trainee_id"]
                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="pay_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                ensure_payment_row(trainee_id, staff_branch, program, group, year, staff_name)

                pay = ensure_cols(read_df("Payments"), "Payments")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
                        (pay["year"].astype(str).str.strip() == norm(year))].copy()
                rowp = m.iloc[0].to_dict()

                st.markdown("### 📌 علّم الأشهر اللي خالصين")
                cols = st.columns(4)
                for i, mo in enumerate(MONTHS):
                    paid = (norm(rowp.get(mo)).upper() == "TRUE")
                    with cols[i % 4]:
                        new_paid = st.checkbox(mo, value=paid, key=f"ck_{mo}_{trainee_id}_{year}")
                        if new_paid != paid:
                            set_payment_month(trainee_id, year, mo, new_paid, staff_name)
                            st.rerun()

    # ------------------ Timetable CRUD
    with tab_tt:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = subjects_df.copy()
            for c in ["branch", "program", "group", "subject_name", "is_active"]:
                sub[c] = sub[c].astype(str).str.strip()
            sub = sub[(sub["branch"] == staff_branch) & (sub["program"] == norm(program)) & (sub["group"] == norm(group))].copy()
            sub = sub[sub["is_active"].astype(str).str.lower() != "false"].copy()
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x])

            st.markdown("### ➕ Ajouter séance")
            c1, c2, c3 = st.columns(3)
            with c1:
                day = st.selectbox("Jour", DAYS, key="tt_day_add")
                start = st.text_input("Heure début (ex: 09:00)", key="tt_start_add")
                end = st.text_input("Heure fin (ex: 10:30)", key="tt_end_add")
            with c2:
                subject = st.selectbox("Matière", subjects if subjects else ["(no subjects)"], key="tt_subject_add")
                teacher = st.text_input("Nom du prof", key="tt_teacher_add")
                room = st.text_input("Salle (optionnel)", key="tt_room_add")
            with c3:
                col_default = "#E8EEF7"
                color = st.color_picker("Couleur", value=col_default, key="tt_color_add")

            if st.button("✅ Enregistrer séance", use_container_width=True, key="tt_save_add"):
                if not norm(start) or not norm(end) or not norm(teacher) or not norm(subject) or subject == "(no subjects)":
                    st.error("لازم وقت (من/إلى) + مادة + اسم البروف.")
                else:
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
                        "color": valid_hex_color(color, "#E8EEF7"),
                        "created_at": now_str(),
                        "updated_at": now_str(),
                        "staff_name": staff_name,
                    })
                    st.success("✅ Ajouté")
                    st.rerun()

            st.divider()
            st.markdown("### 🗓️ Preview (الجدول اللي باش يشوفوه الموظف والمتكون)")
            tt = load_timetable(staff_branch, program, group, year)
            st.markdown(timetable_html_grid(tt), unsafe_allow_html=True)

            st.divider()
            st.markdown("### ✏️/🗑️ تعديل أو حذف سِيانس")
            if tt.empty:
                st.info("لا توجد حصص.")
            else:
                tt2 = tt.copy()
                for c in ["day", "start", "end", "subject", "teacher", "room", "row_id", "color"]:
                    tt2[c] = tt2[c].astype(str).str.strip()
                day_map = {d: i for i, d in enumerate(DAYS)}
                tt2["day_i"] = tt2["day"].map(lambda x: day_map.get(x, 99))
                tt2 = tt2.sort_values(by=["day_i", "start", "end"], ascending=True)

                tt2["label"] = (
                    tt2["day"] + " | " +
                    tt2["start"] + "-" + tt2["end"] + " | " +
                    tt2["subject"] + " | " +
                    tt2["teacher"] + " | " +
                    tt2["row_id"]
                )
                pick = st.selectbox("Choisir séance", tt2["label"].tolist(), key="tt_pick_edit")
                row = tt2[tt2["label"] == pick].iloc[0].to_dict()
                rid = norm(row.get("row_id"))

                e1, e2, e3 = st.columns(3)
                with e1:
                    day_e = st.selectbox("Jour", DAYS, index=max(0, DAYS.index(norm(row.get("day"))) if norm(row.get("day")) in DAYS else 0),
                                         key="tt_day_edit")
                    start_e = st.text_input("Heure début", value=norm(row.get("start")), key="tt_start_edit")
                    end_e = st.text_input("Heure fin", value=norm(row.get("end")), key="tt_end_edit")
                with e2:
                    subj_list = subjects if subjects else [norm(row.get("subject")) or "—"]
                    idx_sub = subj_list.index(norm(row.get("subject"))) if norm(row.get("subject")) in subj_list else 0
                    subject_e = st.selectbox("Matière", subj_list, index=idx_sub, key="tt_subject_edit")
                    teacher_e = st.text_input("Nom du prof", value=norm(row.get("teacher")), key="tt_teacher_edit")
                    room_e = st.text_input("Salle", value=norm(row.get("room")), key="tt_room_edit")
                with e3:
                    color_seed = valid_hex_color(row.get("color"), "#E8EEF7")
                    color_e = st.color_picker("Couleur", value=color_seed, key="tt_color_edit")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("💾 Sauvegarder modification", use_container_width=True, key="tt_save_edit"):
                        ok = update_row_by_key(
                            "Timetable",
                            ["row_id"], [rid],
                            {
                                "day": norm(day_e),
                                "start": norm(start_e),
                                "end": norm(end_e),
                                "subject": norm(subject_e),
                                "teacher": norm(teacher_e),
                                "room": norm(room_e),
                                "color": valid_hex_color(color_e, "#E8EEF7"),
                                "updated_at": now_str(),
                                "staff_name": staff_name,
                            }
                        )
                        if ok:
                            st.success("✅ تم التعديل")
                            st.rerun()
                        else:
                            st.error("❌ ما لقايناش السطر باش نعدلو")
                with b2:
                    if st.button("🗑️ Supprimer séance", use_container_width=True, key="tt_del"):
                        if delete_row_by_key("Timetable", "row_id", rid):
                            st.success("✅ تحذفت")
                            st.rerun()
                        else:
                            st.error("❌ ما تحذفتش")

    # ------------------ Supports (Drive links manual)
    with tab_sup:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = subjects_df.copy()
            for c in ["branch", "program", "group", "subject_name", "is_active"]:
                sub[c] = sub[c].astype(str).str.strip()
            sub = sub[(sub["branch"] == staff_branch) & (sub["program"] == norm(program)) & (sub["group"] == norm(group))].copy()
            sub = sub[sub["is_active"].astype(str).str.lower() != "false"].copy()
            subjects = sorted([x for x in sub["subject_name"].astype(str).str.strip().tolist() if x])

            if not subjects:
                st.warning("زيد matières قبل (Sheet Subjects).")
            else:
                st.markdown("### ➕ Ajouter fichier (Lien Google Drive)")
                subj = st.selectbox("Matière", subjects, key="cf_subject")
                fname = st.text_input("Nom du fichier", key="cf_name")
                link = st.text_input("Lien Google Drive (Share link: Anyone with the link)", key="cf_link")

                if st.button("✅ Enregistrer fichier", use_container_width=True, key="cf_save"):
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

            files = ensure_cols(read_df("CourseFiles"), "CourseFiles")
            files = files[(files["branch"].astype(str).str.strip() == staff_branch) &
                          (files["program"].astype(str).str.strip() == norm(program)) &
                          (files["group"].astype(str).str.strip() == norm(group))] if not files.empty else pd.DataFrame()
            if not files.empty:
                st.divider()
                st.markdown("### 🗂️ Fichiers enregistrés (اللي باش يشوفهم المتكون)")
                files = files.sort_values(by=["uploaded_at"], ascending=False)
                for _, r in files.iterrows():
                    fid = norm(r.get("file_id")) or uuid.uuid4().hex
                    st.markdown(f"**📌 {norm(r.get('subject_name'))}** — {norm(r.get('file_name'))}")
                    link_btn("👀 Ouvrir", norm(r.get("drive_view_url")), key=f"stf_v_{fid}")
                    link_btn("⬇️ Télécharger", norm(r.get("drive_download_url")), key=f"stf_d_{fid}")
                    # delete
                    if st.button("🗑️ Supprimer ce fichier", use_container_width=True, key=f"cf_del_{fid}"):
                        if delete_row_by_key("CourseFiles", "file_id", norm(r.get("file_id"))):
                            st.success("✅ تحذف")
                            st.rerun()
                        else:
                            st.error("❌ ما تحذفش")
                    st.divider()

    # ------------------ Moyennes CRUD
    with tab_avg:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = trainees_df.copy()
            for c in ["branch", "program", "group", "phone", "full_name", "trainee_id"]:
                tr[c] = tr[c].astype(str).str.strip()
            tr = tr[(tr["branch"] == staff_branch) & (tr["program"] == norm(program)) & (tr["group"] == norm(group))].copy()
            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr["label"] = tr["full_name"] + " — " + tr["phone"] + " — " + tr["trainee_id"]
                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="avg_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                st.markdown("### ➕ Ajouter moyenne")
                sem = st.selectbox("Semestre", ["S1", "S2", "S3", "S4"], key="avg_sem_add")
                avg_val = st.number_input("Moyenne", min_value=0.0, max_value=20.0, value=10.0, step=0.01, key="avg_val_add")
                note = st.text_area("Remarque (optionnel)", key="avg_note_add")

                if st.button("✅ Enregistrer moyenne", use_container_width=True, key="avg_save_add"):
                    append_row("Averages", {
                        "avg_id": f"AV-{uuid.uuid4().hex[:8].upper()}",
                        "trainee_id": trainee_id,
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "year": norm(year),
                        "semester": norm(sem),
                        "average": str(avg_val),
                        "note": norm(note),
                        "created_at": now_str(),
                        "updated_at": now_str(),
                        "staff_name": staff_name,
                    })
                    st.success("✅ Ajouté")
                    st.rerun()

                st.divider()
                st.markdown("### ✏️/🗑️ تعديل أو حذف Moyennes")
                av = load_averages(staff_branch, program, group, trainee_id=trainee_id)
                if av.empty:
                    st.info("لا توجد Moyennes.")
                else:
                    av2 = av.copy()
                    for c in ["year", "semester", "average", "avg_id"]:
                        av2[c] = av2[c].astype(str).str.strip()
                    av2 = av2.sort_values(by=["year", "semester"], ascending=False)
                    av2["label"] = av2["year"] + " | " + av2["semester"] + " | " + av2["average"] + " | " + av2["avg_id"]
                    pick = st.selectbox("Choisir moyenne", av2["label"].tolist(), key="avg_pick")
                    row = av2[av2["label"] == pick].iloc[0].to_dict()
                    aid = norm(row.get("avg_id"))

                    e1, e2 = st.columns(2)
                    with e1:
                        year_e = st.text_input("Année", value=norm(row.get("year")), key="avg_year_edit")
                        sem_e = st.selectbox("Semestre", ["S1", "S2", "S3", "S4"],
                                             index=max(0, ["S1","S2","S3","S4"].index(norm(row.get("semester"))) if norm(row.get("semester")) in ["S1","S2","S3","S4"] else 0),
                                             key="avg_sem_edit")
                        avg_e = st.number_input("Moyenne", min_value=0.0, max_value=20.0,
                                                value=float(safe_float(row.get("average"), 0.0)),
                                                step=0.01, key="avg_val_edit")
                    with e2:
                        note_e = st.text_area("Remarque", value=norm(row.get("note")), key="avg_note_edit")

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("💾 Sauvegarder modification", use_container_width=True, key="avg_save_edit"):
                            ok = update_row_by_key(
                                "Averages",
                                ["avg_id"], [aid],
                                {
                                    "year": norm(year_e),
                                    "semester": norm(sem_e),
                                    "average": str(avg_e),
                                    "note": norm(note_e),
                                    "updated_at": now_str(),
                                    "staff_name": staff_name,
                                }
                            )
                            if ok:
                                st.success("✅ تم التعديل")
                                st.rerun()
                            else:
                                st.error("❌ ما لقايناش السطر")
                    with b2:
                        if st.button("🗑️ Supprimer moyenne", use_container_width=True, key="avg_del"):
                            if delete_row_by_key("Averages", "avg_id", aid):
                                st.success("✅ تحذفت")
                                st.rerun()
                            else:
                                st.error("❌ ما تحذفتش")

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
