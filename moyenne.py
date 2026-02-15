import uuid
import base64
import io
from datetime import datetime

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

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],

    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],

    "Accounts": ["phone", "password", "trainee_id", "student_name", "created_at", "last_login"],

    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group",
               "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],

    # Planning links
    "TimetableImages": ["branch", "program", "group",
                        "drive_view_url", "drive_download_url", "file_name",
                        "uploaded_at", "staff_name"],

    # Profile pics (small base64)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],

    # Payments
    "Payments": ["payment_id", "trainee_id", "branch", "program", "group", "year"] + MONTHS + ["updated_at", "staff_name"],

    # Course supports links
    "CourseFiles": ["file_id", "branch", "program", "group", "subject_name", "file_name",
                    "drive_view_url", "drive_download_url", "uploaded_at", "staff_name"],
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
        if status == 429 or "quota" in low:
            return "⚠️ 429 Quota (Google Sheets). جرّب Reboot واستنى شوية.\n" + text[:300]
        if status == 403 or "permission" in low or "forbidden" in low:
            return "❌ 403 Permission (Google Sheets). لازم Share للـ service account.\n" + text[:300]
        if status == 404 or "not found" in low:
            return "❌ 404 Not found. تأكد GSHEET_ID صحيح + Share للـ service account.\n" + text[:300]
        return "❌ Google Sheets API Error:\n" + (text[:500] if text else str(e))
    except Exception:
        return "❌ Google Sheets API Error."

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
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    im = im.resize((nw, nh))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()

# --- link button fallback (fix streamlit versions) ---
def ui_link_button(label: str, url: str, key: str | None = None, full: bool = True):
    u = norm(url)
    if not u:
        return
    if hasattr(st, "link_button"):
        st.link_button(label, u, use_container_width=full, key=key)
    else:
        st.markdown(f"🔗 [{label}]({u})")

# --- Drive link helpers ---
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
    if "id=" in u:
        try:
            return u.split("id=")[1].split("&")[0]
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

def drive_image_embed_url(any_drive_url: str) -> str | None:
    fid = extract_drive_file_id(any_drive_url)
    if not fid:
        return None
    return f"https://drive.google.com/uc?export=view&id={fid}"

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
def ensure_headers_safe(ws, headers: list[str]):
    try:
        rng = ws.get("1:1")
        row1 = rng[0] if (rng and len(rng) > 0) else []
        row1 = [norm(x) for x in row1]
    except APIError:
        row1 = []

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

# =========================================================
# SAFE READ/WRITE (IMPORTANT FIX)
# =========================================================
def read_df_safe(ws_name: str) -> pd.DataFrame:
    """
    قراءة Sheets بدون ما تطيّح الأبليكاسيون.
    ترجع DF فارغ إذا صار APIError/WorksheetNotFound.
    """
    headers = REQUIRED_SHEETS.get(ws_name, [])
    try:
        ws = spreadsheet().worksheet(ws_name)
        values = ws.get_all_values()
        if len(values) <= 1:
            return pd.DataFrame(columns=headers)
        hdr = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=hdr)
    except WorksheetNotFound:
        st.error(f"❌ Worksheet '{ws_name}' مش موجودة. اعمل Initialiser مرة واحدة.")
        return pd.DataFrame(columns=headers)
    except APIError as e:
        st.error(explain_api_error(e))
        return pd.DataFrame(columns=headers)
    except Exception as e:
        st.error(f"❌ Error reading '{ws_name}': {e}")
        return pd.DataFrame(columns=headers)

def append_row(ws_name: str, row: dict):
    ws = spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")

def update_row_by_key(ws_name: str, key_cols: list[str], key_vals: list[str], updates: dict) -> bool:
    df = read_df_safe(ws_name)
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

    return True

# =========================================================
# PROFILE PICS (small base64)
# =========================================================
def get_profile_pic_bytes(phone: str) -> bytes | None:
    df = read_df_safe("ProfilePics")
    if df.empty or "phone" not in df.columns:
        return None
    m = df[df["phone"].astype(str).str.strip() == norm(phone)]
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
    df = read_df_safe("Payments")
    if not df.empty:
        if "trainee_id" in df.columns and "year" in df.columns:
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
    df = read_df_safe("Payments")
    if df.empty:
        return False
    if "trainee_id" not in df.columns or "year" not in df.columns:
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
    return True

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
    df = read_df_safe("Branches")
    if df.empty:
        return None
    df2 = df.copy()
    for c in ["branch", "staff_password", "is_active"]:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).str.strip()
    if "is_active" in df2.columns:
        df2["is_active"] = df2["is_active"].str.lower()

    m = df2[(df2.get("branch", "") == norm(branch)) &
            (df2.get("staff_password", "") == norm(branch_password)) &
            (df2.get("is_active", "") != "false")]
    if m.empty:
        return None
    return {"branch": norm(branch), "role": "staff"}

def student_login(phone: str, password: str):
    df = read_df_safe("Accounts")
    if df.empty:
        return None
    df2 = df.copy()
    for c in ["phone", "password"]:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).str.strip()

    m = df2[(df2.get("phone", "") == norm(phone)) & (df2.get("password", "") == norm(password))]
    if m.empty:
        return None
    return m.iloc[0].to_dict()

# =========================================================
# SIDEBAR
# =========================================================
def sidebar_staff_login():
    st.sidebar.markdown("## 👨‍💼 Connexion Employé")
    branches_df = read_df_safe("Branches")
    branches = sorted([x for x in branches_df.get("branch", pd.Series(dtype=str)).astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []

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
        st.sidebar.warning("Branches vide. أضف Branches في Sheet 'Branches'.")
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
        if st.button("Se déconnecter", use_container_width=True, key="btn_stud_logout"):
            st.session_state.student = None
            st.rerun()

    with tab2:
        st.subheader("Inscription (Nom libre + Téléphone لازم يكون مسجّل عند الإدارة)")
        branches_df = read_df_safe("Branches")
        branches = sorted([x for x in branches_df.get("branch", pd.Series(dtype=str)).astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre.")
            return
        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(read_df_safe("Programs"), branch=b)
        if not prog_df.empty and "is_active" in prog_df.columns:
            prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in prog_df.get("program_name", pd.Series(dtype=str)).astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité.")
            return
        p = st.selectbox("Spécialité", programs, key="reg_prog")

        grp_df = df_filter(read_df_safe("Groups"), branch=b, program_name=p)
        if not grp_df.empty and "is_active" in grp_df.columns:
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        groups = sorted([x for x in grp_df.get("group_name", pd.Series(dtype=str)).astype(str).str.strip().tolist() if x])
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

            acc = read_df_safe("Accounts")
            if not acc.empty and "phone" in acc.columns and acc["phone"].astype(str).str.strip().eq(norm(phone)).any():
                st.error("Ce téléphone est déjà inscrit.")
                return

            tr = read_df_safe("Trainees")
            if tr.empty:
                st.error("Aucun stagiaire.")
                return

            tr2 = tr.copy()
            for c in ["branch", "program", "group", "phone"]:
                if c in tr2.columns:
                    tr2[c] = tr2[c].astype(str).str.strip()

            candidates = tr2[(tr2.get("branch", "") == norm(b)) &
                             (tr2.get("program", "") == norm(p)) &
                             (tr2.get("group", "") == norm(g)) &
                             (tr2.get("phone", "") == norm(phone))]
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
            st.success("✅ Compte créé. امشي Connexion.")
            st.rerun()

    with tab3:
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))
        student_name = norm(acc.get("student_name"))

        tr = read_df_safe("Trainees")
        if tr.empty or "trainee_id" not in tr.columns:
            st.error("Trainees sheet فارغة/فيها مشكل.")
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

        t1, t2, t3, t4 = st.tabs(["📝 Notes", "🗓️ Planning", "💳 Paiements", "📎 Supports"])

        with t1:
            gr = read_df_safe("Grades")
            if gr.empty or "trainee_id" not in gr.columns:
                st.info("Aucune note.")
            else:
                grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy()
                if grf.empty:
                    st.info("Aucune note.")
                else:
                    st.dataframe(grf[["subject_name", "exam_type", "score", "date", "staff_name", "note"]],
                                 use_container_width=True, hide_index=True)

        with t2:
            tt = read_df_safe("TimetableImages")
            if tt.empty:
                st.info("لا توجد Planning.")
            else:
                m = tt[(tt.get("branch", "")).astype(str).str.strip() == branch]
                m = m[(m.get("program", "")).astype(str).str.strip() == program]
                m = m[(m.get("group", "")).astype(str).str.strip() == group]
                if m.empty:
                    st.info("لا توجد Planning.")
                else:
                    r = m.iloc[0].to_dict()
                    st.markdown(f"**📄 {norm(r.get('file_name') or 'Planning')}**")
                    v = norm(r.get("drive_view_url"))
                    d = norm(r.get("drive_download_url"))

                    img_url = drive_image_embed_url(v or d)
                    if img_url:
                        st.image(img_url, caption="🗓️ Planning", use_container_width=True)
                    else:
                        st.warning("الرابط موش واضح لاستخراج الصورة. استعمل Drive share link صحيح.")

                    cA, cB = st.columns(2)
                    with cA:
                        ui_link_button("👀 Ouvrir", v, key=f"stud_pl_open_{trainee_id}")
                    with cB:
                        ui_link_button("⬇️ Télécharger", d, key=f"stud_pl_dl_{trainee_id}")

        with t3:
            pay = read_df_safe("Payments")
            if pay.empty or "trainee_id" not in pay.columns:
                st.info("لا توجد بيانات دفوعات.")
            else:
                pay_t = pay[pay["trainee_id"].astype(str).str.strip() == trainee_id].copy()
                if pay_t.empty:
                    st.info("لا توجد بيانات دفوعات لهذا المتكون.")
                else:
                    years = sorted([y for y in pay_t["year"].astype(str).str.strip().unique().tolist() if y], reverse=True)
                    cur_year = str(datetime.now().year)
                    default_idx = years.index(cur_year) if cur_year in years else 0
                    year = st.selectbox("📅 السنة", years, index=default_idx, key=f"stud_pay_year_{trainee_id}")

                    rowp = pay_t[pay_t["year"].astype(str).str.strip() == year].iloc[0].to_dict()
                    paid_months = [mo for mo in MONTHS if str(rowp.get(mo, "")).strip().upper() == "TRUE"]
                    unpaid_months = [mo for mo in MONTHS if mo not in paid_months]

                    cA, cB = st.columns(2)
                    with cA:
                        st.markdown("### ✅ أشهر خالصة")
                        st.success(" / ".join(paid_months) if paid_months else "حتى شهر موش خالص.")
                    with cB:
                        st.markdown("### ⏳ أشهر مازالت")
                        st.warning(" / ".join(unpaid_months) if unpaid_months else "كل الأشهر خالصة ✅")

                    show = {mo: (str(rowp.get(mo, "")).strip().upper() == "TRUE") for mo in MONTHS}
                    st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

        with t4:
            files = read_df_safe("CourseFiles")
            if files.empty:
                st.info("لا توجد ملفات.")
            else:
                f = files.copy()
                for col in ["branch", "program", "group"]:
                    if col in f.columns:
                        f[col] = f[col].astype(str).str.strip()
                f = f[(f.get("branch", "") == branch) &
                      (f.get("program", "") == program) &
                      (f.get("group", "") == group)]
                if f.empty:
                    st.info("لا توجد ملفات.")
                else:
                    if "uploaded_at" in f.columns:
                        f = f.sort_values(by=["uploaded_at"], ascending=False)
                    for _, r in f.iterrows():
                        st.markdown(f"**📌 {norm(r.get('subject_name'))}** — {norm(r.get('file_name'))}")
                        v = norm(r.get("drive_view_url"))
                        d = norm(r.get("drive_download_url"))
                        fid = norm(r.get("file_id") or uuid.uuid4().hex[:6])

                        cA, cB = st.columns(2)
                        with cA:
                            ui_link_button("👀 Ouvrir", v, key=f"v_{fid}_{trainee_id}")
                        with cB:
                            ui_link_button("⬇️ Télécharger", d, key=f"d_{fid}_{trainee_id}")
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

    prog_df = df_filter(read_df_safe("Programs"), branch=staff_branch)
    if not prog_df.empty and "is_active" in prog_df.columns:
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df.get("program_name", pd.Series(dtype=str)).astype(str).str.strip().tolist() if x])

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        program = st.selectbox("Spécialité", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(read_df_safe("Groups"), branch=staff_branch, program_name=program)
            if not grp_df.empty and "is_active" in grp_df.columns:
                grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in grp_df.get("group_name", pd.Series(dtype=str)).astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe", groups, key="manage_group") if groups else None
    with colC:
        year = st.selectbox("Année", [str(datetime.now().year - 1), str(datetime.now().year), str(datetime.now().year + 1)],
                            index=1, key="pay_year")

    tab_stag, tab_gr, tab_pay, tab_plan, tab_sup = st.tabs(
        ["👤 Stagiaires", "📝 Notes", "💳 Paiements", "🗓️ Planning (Lien Drive)", "📎 Supports (Liens Drive)"]
    )

    with tab_stag:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            cur = df_filter(read_df_safe("Trainees"), branch=staff_branch, program=program, group=group)
            if not cur.empty:
                st.dataframe(cur[["full_name", "phone", "status", "created_at"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("لا يوجد stagiaires.")

            st.markdown("### ➕ Ajouter stagiaire")
            name = st.text_input("Nom", key="add_tr_name")
            phone = st.text_input("Téléphone", key="add_tr_phone")
            status = st.selectbox("Statut", ["active", "inactive"], key="add_tr_status")
            if st.button("Enregistrer", use_container_width=True, key="btn_add_tr"):
                if not norm(name) or not norm(phone):
                    st.error("Nom + téléphone obligatoire.")
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

    with tab_gr:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(read_df_safe("Trainees"), branch=staff_branch, program=program, group=group)
            sub = df_filter(read_df_safe("Subjects"), branch=staff_branch, program=program, group=group)

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

    with tab_pay:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            tr = df_filter(read_df_safe("Trainees"), branch=staff_branch, program=program, group=group)
            if tr.empty:
                st.info("لا يوجد stagiaires.")
            else:
                tr = tr.copy()
                tr["label"] = tr["full_name"].astype(str) + " — " + tr["phone"].astype(str) + " — " + tr["trainee_id"].astype(str)
                chosen = st.selectbox("Choisir stagiaire", tr["label"].tolist(), key="pay_trainee")
                trainee_id = tr[tr["label"] == chosen].iloc[0]["trainee_id"]

                ensure_payment_row(trainee_id, staff_branch, program, group, year, staff_name)

                pay = read_df_safe("Payments")
                m = pay[(pay["trainee_id"].astype(str).str.strip() == norm(trainee_id)) &
                        (pay["year"].astype(str).str.strip() == norm(year))].copy()
                if m.empty:
                    st.warning("Payments row مش متسجّل.")
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

    with tab_plan:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            st.info("✅ ارفع الـ Planning يدويًا في Google Drive ثم Paste الرابط هنا. (Share: Anyone with the link)")
            file_name = st.text_input("Nom du fichier (اختياري)", key="pl_name")
            share_link = st.text_input("Lien Google Drive (Share link)", key="pl_link")

            if st.button("✅ Enregistrer planning", use_container_width=True, key="pl_save"):
                if not norm(share_link):
                    st.error("لازم رابط Drive.")
                else:
                    view_url, dl_url = to_view_and_download(share_link)

                    updated = update_row_by_key(
                        "TimetableImages",
                        ["branch", "program", "group"],
                        [staff_branch, program, group],
                        {"drive_view_url": view_url, "drive_download_url": dl_url,
                         "file_name": norm(file_name) or "Planning",
                         "uploaded_at": now_str(), "staff_name": staff_name}
                    )
                    if not updated:
                        append_row("TimetableImages", {
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "drive_view_url": view_url,
                            "drive_download_url": dl_url,
                            "file_name": norm(file_name) or "Planning",
                            "uploaded_at": now_str(),
                            "staff_name": staff_name,
                        })

                    st.success("✅ Planning enregistré.")
                    st.rerun()

    with tab_sup:
        if not (program and group):
            st.info("اختار spécialité + groupe.")
        else:
            sub = df_filter(read_df_safe("Subjects"), branch=staff_branch, program=program, group=group)
            subjects = sorted([x for x in sub.get("subject_name", pd.Series(dtype=str)).astype(str).str.strip().tolist() if x]) if not sub.empty else []
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
