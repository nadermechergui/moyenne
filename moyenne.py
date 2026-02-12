import uuid
import re
import base64
import io
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

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],

    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],

    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "created_at", "last_login"],
    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group", "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],
    "ExamTypes": ["examtype_id", "branch", "program_name", "group_name", "exam_type", "is_active", "created_at"],

    # جدول الأوقات كصفوف (اختياري)
    "Timetable": ["row_id", "branch", "program", "group", "day", "start", "end", "subject", "room", "teacher", "created_at"],

    # ✅ صورة جدول الأوقات (key: branch+program+group)
    "TimetableImages": ["branch", "program", "group", "image_b64", "uploaded_at"],

    # ✅ صور البروفيل (key = phone)
    "ProfilePics": ["phone", "trainee_id", "image_b64", "uploaded_at"],
}

DEFAULT_TIMETABLE_ROW = {
    "row_id": "",
    "day": "Monday",
    "start": "18:00",
    "end": "19:30",
    "subject": "",
    "room": "",
    "teacher": "",
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

        if status == 429 or "rate" in low or "quota" in low:
            return (
                "⚠️ Limite Google Sheets (429) atteinte.\n\n"
                "✅ الحل: اعمل Reboot للتطبيق واستنى دقيقة.\n"
                "Détails: " + text[:250]
            )
        if status == 403 or "permission" in low or "forbidden" in low:
            return (
                "❌ Permission refusée (403).\n\n"
                "✅ الحل:\n"
                "Google Sheet → Share → زِد service account (client_email) كـ Editor → Reboot\n\n"
                "Détails: " + text[:250]
            )
        if status == 404 or "not found" in low:
            return (
                "❌ Spreadsheet غير موجود (404).\n\n"
                "✅ الحل:\n"
                "تأكد GSHEET_ID صحيح + Share للـ service account\n\n"
                "Détails: " + text[:250]
            )
        return "❌ Google API Error:\n" + (text[:350] if text else str(e))
    except Exception:
        return "❌ Google API Error (unknown)."

# ---- smart name matching (order-insensitive) ----
def tokenize_name(s: str) -> list[str]:
    s = norm(s).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    parts = [p for p in s.split() if p]
    parts = [p for p in parts if len(p) > 1]
    return parts

def name_key(s: str) -> str:
    toks = sorted(set(tokenize_name(s)))
    return " ".join(toks)

# ---- image compression to avoid Sheets cell limits ----
def compress_image_bytes(img_bytes: bytes, max_side: int = 900, quality: int = 75) -> bytes:
    """
    تصغير + ضغط باش Base64 ما يكبرش برشا.
    max_side = 900 مناسب لصورة جدول أوقات واضحة.
    """
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if im.mode == "RGBA":
        # نحولها RGB بخلفية بيضاء
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

# =========================================================
# GSHEETS (OPTIMIZED)
# =========================================================
@st.cache_resource
def gs_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_resource
def spreadsheet():
    sheet_id = st.secrets["GSHEET_ID"]
    return gs_client().open_by_key(sheet_id)

def ensure_headers(ws, headers: list[str]):
    row1 = []
    rng = ws.get("1:1")
    if rng and len(rng) > 0:
        row1 = rng[0]
    row1 = [norm(x) for x in row1]
    if row1 != headers:
        ws.clear()
        ws.append_row(headers, value_input_option="RAW")

def ensure_worksheets_and_headers():
    try:
        sh = spreadsheet()
        titles = [w.title for w in sh.worksheets()]
        for ws_name, headers in REQUIRED_SHEETS.items():
            if ws_name not in titles:
                sh.add_worksheet(title=ws_name, rows=2000, cols=max(12, len(headers) + 2))
                titles.append(ws_name)
            ws = sh.worksheet(ws_name)
            ensure_headers(ws, headers)
        return sh
    except APIError as e:
        st.error(explain_api_error(e))
        raise

def ensure_schema_once():
    if st.session_state.get("schema_ok", False):
        return
    ensure_worksheets_and_headers()
    st.session_state.schema_ok = True

@st.cache_data(ttl=60, show_spinner=False)
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

def update_row_by_key(ws_name: str, key_cols: list[str], key_vals: list[str], updates: dict) -> bool:
    """
    Update first row where all key_cols == key_vals.
    """
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

    for col_name, val in updates.items():
        if col_name not in headers:
            continue
        col_num = headers.index(col_name) + 1
        ws.update_cell(row_num, col_num, norm(val))

    st.cache_data.clear()
    return True

def delete_group_timetable(branch: str, program: str, group: str):
    ws = spreadsheet().worksheet("Timetable")
    all_vals = ws.get_all_values()
    if not all_vals:
        return
    headers = all_vals[0]
    rows = all_vals[1:]
    to_delete = []
    for i, r in enumerate(rows, start=2):
        rdict = dict(zip(headers, r + [""] * (len(headers) - len(r))))
        if (norm(rdict.get("branch")) == norm(branch) and
            norm(rdict.get("program")) == norm(program) and
            norm(rdict.get("group")) == norm(group)):
            to_delete.append(i)
    for ridx in sorted(to_delete, reverse=True):
        ws.delete_rows(ridx)
    st.cache_data.clear()

# =========================================================
# PROFILE PICS (student uploads)
# =========================================================
def get_profile_pic_bytes(phone: str) -> bytes | None:
    df = read_df("ProfilePics")
    if df.empty:
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
    if len(small) > 80_000:
        small = compress_image_bytes(img_bytes, max_side=200, quality=60)

    b64 = base64.b64encode(small).decode("utf-8")

    updated = update_row_by_key(
        "ProfilePics",
        key_cols=["phone"],
        key_vals=[phone],
        updates={"trainee_id": trainee_id, "image_b64": b64, "uploaded_at": now_str()},
    )
    if not updated:
        append_row("ProfilePics", {
            "phone": norm(phone),
            "trainee_id": norm(trainee_id),
            "image_b64": b64,
            "uploaded_at": now_str(),
        })

# =========================================================
# TIMETABLE IMAGE (staff uploads)
# =========================================================
def get_timetable_image_bytes(branch: str, program: str, group: str) -> bytes | None:
    df = read_df("TimetableImages")
    if df.empty:
        return None
    m = df[
        (df["branch"].astype(str).str.strip() == norm(branch)) &
        (df["program"].astype(str).str.strip() == norm(program)) &
        (df["group"].astype(str).str.strip() == norm(group))
    ]
    if m.empty:
        return None
    b64 = norm(m.iloc[0].get("image_b64"))
    if not b64:
        return None
    try:
        return base64.b64decode(b64.encode("utf-8"))
    except Exception:
        return None

def upsert_timetable_image(branch: str, program: str, group: str, img_bytes: bytes):
    # ✅ نخليها أكبر من بروفيل باش تبقى واضحة
    small = compress_image_bytes(img_bytes, max_side=900, quality=75)
    if len(small) > 250_000:
        # لو كبيرة برشا، نقص أكثر
        small = compress_image_bytes(img_bytes, max_side=750, quality=70)

    b64 = base64.b64encode(small).decode("utf-8")

    updated = update_row_by_key(
        "TimetableImages",
        key_cols=["branch", "program", "group"],
        key_vals=[branch, program, group],
        updates={"image_b64": b64, "uploaded_at": now_str()},
    )
    if not updated:
        append_row("TimetableImages", {
            "branch": norm(branch),
            "program": norm(program),
            "group": norm(group),
            "image_b64": b64,
            "uploaded_at": now_str(),
        })

# =========================================================
# SESSION / AUTH
# =========================================================
def ensure_session():
    if "role" not in st.session_state:
        st.session_state.role = None  # "staff" | None
    if "user" not in st.session_state:
        st.session_state.user = {}
    if "student" not in st.session_state:
        st.session_state.student = None

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

    m = df2[
        (df2["branch"] == norm(branch)) &
        (df2["staff_password"] == norm(branch_password)) &
        (df2["is_active"] != "false")
    ]
    if m.empty:
        return None
    return {"branch": norm(branch), "role": "staff"}

def student_login(phone: str, password: str):
    df = read_df("Accounts")
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
# SIDEBAR: STAFF LOGIN ONLY
# =========================================================
def sidebar_staff_login():
    st.sidebar.markdown("## 👨‍💼 Connexion Employé")

    branches_df = read_df("Branches")
    branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []

    if st.session_state.role == "staff":
        br = norm(st.session_state.user.get("branch"))
        st.sidebar.success(f"Connecté: {br}")
        if st.sidebar.button("Se déconnecter (Employé)", use_container_width=True):
            logout_staff()
            st.rerun()
        return

    if not branches:
        st.sidebar.warning("Branches vide. Ajoutez centres + mots de passe.")
        return

    branch = st.sidebar.selectbox("Centre", branches, key="sb_staff_branch")
    pwd = st.sidebar.text_input("Mot de passe du centre", type="password", key="sb_staff_pwd")

    if st.sidebar.button("Connexion", use_container_width=True):
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

    # Connexion
    with tab1:
        st.subheader("Connexion Stagiaire")
        phone = st.text_input("Téléphone", key="stud_phone")
        pwd = st.text_input("Mot de passe", type="password", key="stud_pwd")

        if st.button("Se connecter", use_container_width=True, key="btn_stud_login"):
            acc = student_login(phone, pwd)
            if acc:
                update_row_by_key("Accounts", ["phone"], [phone], {"last_login": now_str()})
                st.session_state.student = acc
                st.success("✅ Connexion réussie")
            else:
                st.error("Téléphone / mot de passe incorrect.")

        if st.button("Se déconnecter (Stagiaire)", use_container_width=True, key="btn_stud_logout"):
            st.session_state.student = None
            st.success("Déconnecté.")
            st.rerun()

    # Inscription (name order-insensitive)
    with tab2:
        st.subheader("Inscription Stagiaire")

        branches_df = read_df("Branches")
        branches = sorted([x for x in branches_df["branch"].astype(str).str.strip().unique().tolist() if x]) if not branches_df.empty else []
        if not branches:
            st.warning("Aucun centre disponible.")
            return

        b = st.selectbox("Centre", branches, key="reg_branch")

        prog_df = df_filter(read_df("Programs"), branch=b)
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])
        if not programs:
            st.warning("Aucune spécialité pour ce centre. (Programs)")
            return
        p = st.selectbox("Spécialité", programs, key="reg_program")

        grp_df = df_filter(read_df("Groups"), branch=b, program_name=p)
        grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
        if not groups:
            st.warning("Aucun groupe pour cette spécialité. (Groups)")
            return
        g = st.selectbox("Groupe", groups, key="reg_group")

        full_name = st.text_input("Nom (اكتب اسمك كيف تحب)", key="reg_name")
        phone = st.text_input("Téléphone (unique)", key="reg_phone")
        pwd = st.text_input("Mot de passe", type="password", key="reg_pwd")

        if st.button("Créer mon compte", use_container_width=True, key="btn_register"):
            full_name_n = norm(full_name)
            phone_n = norm(phone)
            pwd_n = norm(pwd)

            if not full_name_n or not phone_n or not pwd_n:
                st.error("Nom, téléphone, mot de passe obligatoires.")
                return
            if len(pwd_n) < 4:
                st.error("Mot de passe trop court (min 4).")
                return

            acc = read_df("Accounts")
            if not acc.empty and acc["phone"].astype(str).str.strip().eq(phone_n).any():
                st.error("Ce téléphone est déjà inscrit.")
                return

            tr = read_df("Trainees")
            if tr.empty:
                st.error("Aucun stagiaire enregistré par l'employé.")
                return

            tr2 = tr.copy()
            for c in ["branch", "program", "group", "full_name"]:
                tr2[c] = tr2[c].astype(str).str.strip()

            typed_key = name_key(full_name_n)
            tr2["name_key"] = tr2["full_name"].apply(name_key)

            candidates = tr2[
                (tr2["branch"] == norm(b)) &
                (tr2["program"] == norm(p)) &
                (tr2["group"] == norm(g)) &
                (tr2["name_key"] == typed_key)
            ].copy()

            if candidates.empty:
                st.error("اسمك ما لقيتوش في القائمة متاع المجموعة (حتى مع تبديل الترتيب). كلم الإدارة باش تزيدك.")
                return

            if len(candidates) > 1:
                candidates["label"] = candidates["full_name"] + " — " + candidates["trainee_id"].astype(str)
                chosen = st.selectbox("اختر اسمك من القائمة", candidates["label"].tolist(), key="choose_candidate")
                trainee_id = candidates[candidates["label"] == chosen].iloc[0]["trainee_id"]
            else:
                trainee_id = candidates.iloc[0]["trainee_id"]

            append_row("Accounts", {
                "phone": phone_n,
                "password": pwd_n,
                "trainee_id": norm(trainee_id),
                "created_at": now_str(),
                "last_login": ""
            })
            st.success("✅ Compte créé. امشي لصفحة Connexion.")

    # Mon espace + profile pic upload
    with tab3:
        st.subheader("Mon espace")
        acc = st.session_state.get("student")
        if not acc:
            st.info("اعمل Connexion باش تشوف النوطات والجدول.")
            return

        trainee_id = norm(acc.get("trainee_id"))
        phone = norm(acc.get("phone"))

        tr = read_df("Trainees")
        row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not tr.empty else pd.DataFrame()
        if row.empty:
            st.error("Compte مرتبط بمتربص غير موجود. كلم الإدارة.")
            return

        info = row.iloc[0].to_dict()
        branch = norm(info.get("branch"))
        program = norm(info.get("program"))
        group = norm(info.get("group"))
        full_name = norm(info.get("full_name"))

        pic = get_profile_pic_bytes(phone)
        c1, c2 = st.columns([1, 3])
        with c1:
            if pic:
                st.image(pic, caption="Photo de profil", use_container_width=True)
            else:
                st.info("Pas de photo")

        with c2:
            st.success(f"Bienvenue {full_name} ✅")
            st.caption(f"Centre: {branch} | Spécialité: {program} | Groupe: {group} | Téléphone: {phone}")

            up = st.file_uploader("📸 Ajouter/Changer ma photo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="profile_uploader")
            if up is not None:
                img_bytes = up.read()
                st.image(img_bytes, caption="Aperçu", width=160)
                if st.button("Enregistrer ma photo", use_container_width=True, key="btn_save_profile_pic"):
                    try:
                        upsert_profile_pic(phone, trainee_id, img_bytes)
                        st.success("✅ Photo enregistrée.")
                        st.rerun()
                    except APIError as e:
                        st.error(explain_api_error(e))

        t1, t2, t3 = st.tabs(["📝 Notes", "🗓️ Emploi du temps", "📚 Matières"])

        with t1:
            gr = read_df("Grades")
            grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
            if grf.empty:
                st.info("Aucune note pour le moment.")
            else:
                grf = grf.sort_values(by=["date", "created_at"], ascending=False)
                st.dataframe(grf[["subject_name", "exam_type", "score", "date", "staff_name", "note"]],
                             use_container_width=True, hide_index=True)

        with t2:
            # ✅ هنا نعرض صورة جدول الأوقات (الموظف رفعها)
            img = get_timetable_image_bytes(branch, program, group)
            if img:
                st.image(img, caption="Emploi du temps", use_container_width=True)
                st.download_button(
                    "⬇️ Télécharger l'image",
                    data=img,
                    file_name=f"planning_{branch}_{program}_{group}.jpg".replace(" ", "_"),
                    mime="image/jpeg",
                    use_container_width=True,
                )
            else:
                st.info("لا توجد صورة لجدول الأوقات لهذا groupe حاليا. (الموظف لازم يرفعها)")

        with t3:
            sub = read_df("Subjects")
            subf = df_filter(sub, branch=branch, program=program, group=group) if not sub.empty else pd.DataFrame()
            if subf.empty:
                st.info("Aucune matière enregistrée.")
            else:
                st.dataframe(subf[["subject_name"]], use_container_width=True, hide_index=True)

# =========================================================
# STAFF WORK AREA (CENTER)
# =========================================================
def staff_work_center():
    st.markdown("## 🛠️ Espace Employé (Gestion)")

    if st.session_state.role != "staff":
        st.info("Connexion Employé من اليسار باش تفتح الإدارة.")
        return

    staff_branch = norm(st.session_state.user.get("branch"))
    st.success(f"Centre: {staff_branch}")

    prog_df = df_filter(read_df("Programs"), branch=staff_branch)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])

    colA, colB = st.columns(2)
    with colA:
        program = st.selectbox("Spécialité (pour gérer)", programs, key="manage_program") if programs else None
    with colB:
        group = None
        if program:
            grp_df = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
            group = st.selectbox("Groupe (pour gérer)", groups, key="manage_group") if groups else None

    t1, t2, t3, t4, t5 = st.tabs(["🏷️ Spécialités", "👥 Groupes", "📚 Matières", "👤 Stagiaires", "🗓️ Planning (Image)"])

    with t1:
        cur = df_filter(read_df("Programs"), branch=staff_branch)
        st.dataframe(cur[["program_name","is_active","created_at"]] if not cur.empty else cur,
                     use_container_width=True, hide_index=True)
        new_prog = st.text_input("Nouvelle spécialité", key="new_prog_center")
        if st.button("Ajouter spécialité", use_container_width=True):
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

    with t2:
        if not program:
            st.info("اختار Spécialité من فوق.")
        else:
            cur = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            st.dataframe(cur[["group_name","is_active","created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)
            new_group = st.text_input("Nouveau groupe", key="new_group_center")
            if st.button("Ajouter groupe", use_container_width=True):
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

    with t3:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            cur = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
            st.dataframe(cur[["subject_name","is_active","created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)
            subject_name = st.text_input("Nouvelle matière", key="new_subject_center")
            if st.button("Ajouter matière", use_container_width=True):
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

    with t4:
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
        else:
            cur = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
            st.dataframe(cur[["full_name","phone","status","created_at"]] if not cur.empty else cur,
                         use_container_width=True, hide_index=True)

            st.markdown("### ➕ Ajouter un stagiaire")
            name = st.text_input("Nom & Prénom", key="add_tr_name_center")
            phone = st.text_input("Téléphone (optionnel)", key="add_tr_phone_center")
            status = st.selectbox("Statut", ["active", "inactive"], key="add_tr_status_center")
            if st.button("Enregistrer stagiaire", use_container_width=True, key="btn_add_tr_center"):
                if not norm(name):
                    st.error("Nom obligatoire.")
                else:
                    append_row("Trainees", {
                        "trainee_id": f"TR-{uuid.uuid4().hex[:8].upper()}",
                        "full_name": norm(name),
                        "phone": norm(phone),
                        "branch": staff_branch,
                        "program": norm(program),
                        "group": norm(group),
                        "status": status,
                        "created_at": now_str()
                    })
                    st.success("✅ Ajouté.")
                    st.rerun()

    with t5:
        st.subheader("رفع صورة جدول الأوقات")
        if not (program and group):
            st.info("اختار Spécialité + Groupe.")
            return

        st.caption("✅ Upload صورة (PNG/JPG) — وبعدها المتكون يشوفها كصورة في بوابتو.")

        # preview current
        old = get_timetable_image_bytes(staff_branch, program, group)
        if old:
            st.image(old, caption="الصورة الحالية", use_container_width=True)

        up = st.file_uploader("Uploader l'image du planning", type=["png", "jpg", "jpeg"], key="tt_img_uploader")
        if up is not None:
            raw = up.read()
            st.image(raw, caption="Aperçu قبل الحفظ", use_container_width=True)

            if st.button("✅ Enregistrer l'image", use_container_width=True, key="btn_save_tt_img"):
                try:
                    upsert_timetable_image(staff_branch, program, group, raw)
                    st.success("✅ تم حفظ صورة جدول الأوقات.")
                    st.rerun()
                except APIError as e:
                    st.error(explain_api_error(e))

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
