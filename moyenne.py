import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Portail Mega Formation", page_icon="🧩", layout="wide")

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "created_at", "last_login"],
    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group", "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],
    "Timetable": ["row_id", "branch", "program", "group", "day", "start", "end", "subject", "room", "teacher", "created_at"],
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
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def norm(x):
    return str(x or "").strip()

def get_unique(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return []
    vals = df[col].astype(str).str.strip().unique().tolist()
    return sorted([v for v in vals if v])

def df_filter(df: pd.DataFrame, **kwargs):
    out = df.copy()
    for k, v in kwargs.items():
        if k in out.columns:
            out = out[out[k].astype(str).str.strip() == norm(v)]
    return out

# =========================================================
# GSHEETS
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

def open_spreadsheet():
    sheet_id = st.secrets["GSHEET_ID"]
    return gs_client().open_by_key(sheet_id)

def ensure_worksheets_and_headers():
    sh = open_spreadsheet()
    existing = {ws.title: ws for ws in sh.worksheets()}

    for ws_name, headers in REQUIRED_SHEETS.items():
        if ws_name not in existing:
            sh.add_worksheet(title=ws_name, rows=2000, cols=max(12, len(headers) + 2))
            existing[ws_name] = sh.worksheet(ws_name)

        ws = existing[ws_name]
        first_row = ws.row_values(1)
        if first_row != headers:
            ws.clear()
            ws.append_row(headers, value_input_option="RAW")

    return sh

@st.cache_data(ttl=8)
def read_df(ws_name: str) -> pd.DataFrame:
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    values = ws.get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

def append_row(ws_name: str, row: dict):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_cell(ws_name: str, row_index_1based: int, col_name: str, value):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    col_index = headers.index(col_name) + 1
    ws.update_cell(row_index_1based, col_index, value)
    st.cache_data.clear()

def delete_group_timetable(branch: str, program: str, group: str):
    sh = open_spreadsheet()
    ws = sh.worksheet("Timetable")
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
# AUTH / SESSION
# =========================================================
def ensure_session():
    if "role" not in st.session_state:
        st.session_state.role = None  # "staff" | "student"
    if "user" not in st.session_state:
        st.session_state.user = {}
    if "page" not in st.session_state:
        st.session_state.page = "Login"

def logout():
    st.session_state.role = None
    st.session_state.user = {}
    st.session_state.page = "Login"

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
# UI: LOGIN
# =========================================================
def page_login():
    st.title("🧩 Portail Mega Formation — Connexion")

    branches_df = read_df("Branches")
    branches = get_unique(branches_df, "branch")

    col1, col2 = st.columns(2, gap="large")

    # ---------- Staff Login ----------
    with col1:
        st.subheader("👨‍💼 Connexion Employé (mot de passe du centre)")
        if not branches:
            st.warning("La feuille 'Branches' est vide. Ajoutez les centres et les mots de passe.")
        else:
            branch = st.selectbox("Centre", branches, key="staff_branch")
            pwd = st.text_input("Mot de passe du centre", type="password", key="staff_pwd")
            if st.button("Se connecter (Employé)", use_container_width=True):
                user = staff_branch_login(branch, pwd)
                if user:
                    st.session_state.role = "staff"
                    st.session_state.user = user
                    st.session_state.page = "Home"
                    st.success("Connexion réussie ✅")
                    st.rerun()
                else:
                    st.error("Mot de passe incorrect / centre inactif.")

    # ---------- Student Login ----------
    with col2:
        st.subheader("🎓 Connexion Stagiaire")
        phone = st.text_input("Téléphone", key="stud_phone")
        pwd2 = st.text_input("Mot de passe", type="password", key="stud_pwd")
        if st.button("Se connecter (Stagiaire)", use_container_width=True):
            acc = student_login(phone, pwd2)
            if acc:
                # update last_login
                df = read_df("Accounts")
                idx = df.index[df["phone"].astype(str).str.strip() == norm(phone)].tolist()
                if idx:
                    update_cell("Accounts", idx[0] + 2, "last_login", now_str())

                st.session_state.role = "student"
                st.session_state.user = acc
                st.session_state.page = "Home"
                st.success("Connexion réussie ✅")
                st.rerun()
            else:
                st.error("Téléphone / mot de passe incorrect.")

    st.divider()

    # ---------- Student Registration ----------
    st.subheader("🆕 Inscription Stagiaire")
    st.caption("Choisissez Centre → Spécialité → Groupe → Nom, ثم Téléphone + Mot de passe.")

    tr = read_df("Trainees")
    if tr.empty:
        st.warning("La feuille 'Trainees' est vide. L'employé doit ajouter les stagiaires d'abord.")
        return
    if not branches:
        st.warning("Ajoutez d'abord les centres dans 'Branches'.")
        return

    b = st.selectbox("Centre", branches, key="reg_branch")
    tr_b = tr[tr["branch"].astype(str).str.strip() == norm(b)].copy()

    programs = get_unique(tr_b, "program")
    if not programs:
        st.warning("Aucune spécialité dans ce centre.")
        return
    p = st.selectbox("Spécialité", programs, key="reg_program")

    tr_bp = tr_b[tr_b["program"].astype(str).str.strip() == norm(p)].copy()
    groups = get_unique(tr_bp, "group")
    if not groups:
        st.warning("Aucun groupe dans cette spécialité.")
        return
    g = st.selectbox("Groupe", groups, key="reg_group")

    tr_f = tr_bp[tr_bp["group"].astype(str).str.strip() == norm(g)].copy()
    if tr_f.empty:
        st.warning("Aucun stagiaire dans ce groupe.")
        return

    search = st.text_input("🔎 Rechercher un nom (optionnel)", key="reg_search", placeholder="Tapez une partie du nom...")
    if search.strip():
        s = search.strip().lower()
        tr_f = tr_f[tr_f["full_name"].astype(str).str.lower().str.contains(s)]

    tr_f["label"] = tr_f["full_name"].astype(str).str.strip() + " — " + tr_f["trainee_id"].astype(str).str.strip()
    choice = st.selectbox("Choisissez votre nom", tr_f["label"].tolist(), key="reg_choice")
    trainee_id = tr_f[tr_f["label"] == choice].iloc[0]["trainee_id"]

    phone = st.text_input("Téléphone (unique)", key="reg_phone")
    pwd = st.text_input("Mot de passe", type="password", key="reg_password")

    if st.button("Créer le compte", use_container_width=True):
        phone = norm(phone)
        pwd = norm(pwd)

        if not phone or not pwd:
            st.error("Téléphone et mot de passe obligatoires.")
            return
        if len(pwd) < 4:
            st.error("Mot de passe trop court (min 4).")
            return

        acc = read_df("Accounts")
        if not acc.empty and acc["phone"].astype(str).str.strip().eq(phone).any():
            st.error("Ce téléphone est déjà inscrit. Utilisez Connexion.")
            return

        append_row("Accounts", {
            "phone": phone,
            "password": pwd,
            "trainee_id": norm(trainee_id),
            "created_at": now_str(),
            "last_login": ""
        })
        st.success("✅ Compte créé. Vous pouvez vous connecter.")

# =========================================================
# UI: HOME (SIDEBAR EMPLOYÉ / CENTRE STAGIAIRE)
# =========================================================
def page_home():
    st.title("🧩 Portail Mega Formation")

    # =========================
    # SIDEBAR: EMPLOYÉ
    # =========================
    st.sidebar.markdown("## 👨‍💼 Espace Employé")

    if st.session_state.role == "staff":
        staff_branch = norm(st.session_state.user.get("branch"))
        st.sidebar.success(f"Centre: {staff_branch}")

        tr_all = read_df("Trainees")
        tr_all = tr_all[tr_all["branch"].astype(str).str.strip() == staff_branch].copy()

        programs = get_unique(tr_all, "program")
        program = st.sidebar.selectbox("Spécialité", programs, key="sb_program") if programs else None

        groups = []
        if program:
            tr_p = tr_all[tr_all["program"].astype(str).str.strip() == norm(program)].copy()
            groups = get_unique(tr_p, "group")
        group = st.sidebar.selectbox("Groupe", groups, key="sb_group") if groups else None

        st.sidebar.divider()

        if program and group:
            # 1) Ajouter stagiaire
            with st.sidebar.expander("➕ Ajouter un stagiaire", expanded=False):
                name = st.text_input("Nom & Prénom", key="emp_add_name")
                phone = st.text_input("Téléphone (optionnel)", key="emp_add_phone")
                status = st.selectbox("Statut", ["active", "inactive"], key="emp_add_status")
                if st.button("Enregistrer", use_container_width=True, key="emp_add_save"):
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
                        st.success("✅ Stagiaire ajouté.")
                        st.rerun()

            # 2) Ajouter matière
            with st.sidebar.expander("📚 Ajouter une matière", expanded=False):
                subject_name = st.text_input("Nom de la matière", key="emp_subj_name")
                if st.button("Ajouter", use_container_width=True, key="emp_subj_add"):
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
                        st.success("✅ Matière ajoutée.")
                        st.rerun()

            # 3) Saisir note
            with st.sidebar.expander("📝 Saisir une note", expanded=True):
                trf = df_filter(tr_all, program=program, group=group)

                sub = read_df("Subjects")
                subf = df_filter(sub, branch=staff_branch, program=program, group=group) if not sub.empty else pd.DataFrame()

                if trf.empty:
                    st.warning("Aucun stagiaire dans ce groupe.")
                elif subf.empty:
                    st.warning("Ajoutez des matières d'abord.")
                else:
                    trf = trf.copy()
                    trf["label"] = trf["full_name"].astype(str).str.strip() + " — " + trf["trainee_id"].astype(str).str.strip()
                    trainee_choice = st.selectbox("Stagiaire", trf["label"].tolist(), key="emp_grade_tr")
                    trainee_id = trf[trf["label"] == trainee_choice].iloc[0]["trainee_id"]

                    subject = st.selectbox("Matière", sorted(subf["subject_name"].astype(str).str.strip().tolist()), key="emp_grade_sub")
                    exam_type = st.text_input("Type d'examen (libre)", key="emp_exam_type", placeholder="Ex: Devoir 1 / Oral / Final / ...")
                    score = st.number_input("Note /20", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="emp_score")
                    date = st.date_input("Date", key="emp_date")
                    note = st.text_input("Remarque (optionnel)", key="emp_note")

                    if st.button("Enregistrer la note", use_container_width=True, key="emp_grade_save"):
                        if not norm(exam_type):
                            st.error("Type d'examen obligatoire.")
                        else:
                            append_row("Grades", {
                                "grade_id": f"GR-{uuid.uuid4().hex[:10].upper()}",
                                "trainee_id": norm(trainee_id),
                                "branch": staff_branch,
                                "program": norm(program),
                                "group": norm(group),
                                "subject_name": norm(subject),
                                "exam_type": norm(exam_type),
                                "score": str(score),
                                "date": str(date),
                                "staff_name": f"Employé-{staff_branch}",  # no staff name
                                "note": norm(note),
                                "created_at": now_str()
                            })
                            st.success("✅ Note enregistrée.")

            # 4) Emploi du temps
            with st.sidebar.expander("🗓️ Emploi du temps (éditer)", expanded=False):
                tt = read_df("Timetable")
                ttf = df_filter(tt, branch=staff_branch, program=program, group=group) if not tt.empty else pd.DataFrame()

                if ttf.empty:
                    base = pd.DataFrame([{
                        **DEFAULT_TIMETABLE_ROW,
                        "row_id": f"TT-{uuid.uuid4().hex[:8].upper()}",
                    }])
                else:
                    base = ttf[["row_id","day","start","end","subject","room","teacher"]].copy()

                edited = st.data_editor(base, use_container_width=True, num_rows="dynamic", key="emp_tt_editor")

                if st.button("Sauvegarder", use_container_width=True, key="emp_tt_save"):
                    delete_group_timetable(staff_branch, program, group)

                    for _, row in edited.iterrows():
                        if not norm(row.get("day")):
                            continue
                        append_row("Timetable", {
                            "row_id": norm(row.get("row_id") or f"TT-{uuid.uuid4().hex[:8].upper()}"),
                            "branch": staff_branch,
                            "program": norm(program),
                            "group": norm(group),
                            "day": norm(row.get("day")),
                            "start": norm(row.get("start")),
                            "end": norm(row.get("end")),
                            "subject": norm(row.get("subject")),
                            "room": norm(row.get("room")),
                            "teacher": norm(row.get("teacher")),
                            "created_at": now_str()
                        })

                    st.success("✅ Emploi du temps sauvegardé.")
                    st.rerun()

        else:
            st.sidebar.info("Choisissez Spécialité + Groupe pour gérer les données.")

        st.sidebar.divider()
        if st.sidebar.button("Se déconnecter", use_container_width=True):
            logout()
            st.rerun()

    else:
        st.sidebar.info("Connectez-vous en tant qu'employé sur la page Login.")
        if st.sidebar.button("Aller à Login", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()

    # =========================
    # CENTRE: STAGIAIRE (READ ONLY)
    # =========================
    st.markdown("## 🎓 Espace Stagiaire")
    st.caption("Le stagiaire consulte فقط: notes, matières, emploi du temps.")

    if st.session_state.role != "student":
        st.info("Connectez-vous en tant que stagiaire depuis la page Login.")
        return

    acc = st.session_state.user
    trainee_id = norm(acc.get("trainee_id"))
    phone = norm(acc.get("phone"))

    tr = read_df("Trainees")
    row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not tr.empty else pd.DataFrame()

    if row.empty:
        st.error("Compte lié à un stagiaire introuvable. Contactez l'administration.")
        return

    info = row.iloc[0].to_dict()
    branch = norm(info.get("branch"))
    program = norm(info.get("program"))
    group = norm(info.get("group"))
    full_name = norm(info.get("full_name"))

    st.success(f"Bienvenue {full_name} ✅")
    st.caption(f"Centre: {branch} | Spécialité: {program} | Groupe: {group} | Téléphone: {phone}")

    t1, t2, t3 = st.tabs(["📝 Notes", "🗓️ Emploi du temps", "📚 Matières"])

    with t1:
        gr = read_df("Grades")
        grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
        if grf.empty:
            st.info("Aucune note pour le moment.")
        else:
            grf = grf.sort_values(by=["date","created_at"], ascending=False)
            st.dataframe(grf[["subject_name","exam_type","score","date","staff_name","note"]], use_container_width=True, hide_index=True)

    with t2:
        tt = read_df("Timetable")
        ttf = df_filter(tt, branch=branch, program=program, group=group) if not tt.empty else pd.DataFrame()
        if ttf.empty:
            st.info("Emploi du temps non disponible.")
        else:
            st.dataframe(ttf[["day","start","end","subject","room","teacher"]], use_container_width=True, hide_index=True)

    with t3:
        sub = read_df("Subjects")
        subf = df_filter(sub, branch=branch, program=program, group=group) if not sub.empty else pd.DataFrame()
        if subf.empty:
            st.info("Aucune matière enregistrée.")
        else:
            st.dataframe(subf[["subject_name"]], use_container_width=True, hide_index=True)

# =========================================================
# NAV + MAIN
# =========================================================
def sidebar_nav():
    st.sidebar.title("Portail")
    if st.session_state.role in ["staff", "student"]:
        if st.sidebar.button("🏠 Accueil", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
        if st.sidebar.button("🔐 Login", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()
    else:
        st.sidebar.info("Non connecté")
        if st.sidebar.button("🔐 Aller à Login", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()

def main():
    ensure_session()
    ensure_worksheets_and_headers()
    sidebar_nav()

    if st.session_state.page == "Login":
        page_login()
    elif st.session_state.page == "Home":
        page_home()
    else:
        page_login()

if __name__ == "__main__":
    main()
