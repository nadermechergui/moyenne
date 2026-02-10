import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Portail Mega Formation", page_icon="🧩", layout="wide")

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],

    "Programs": ["program_id", "branch", "program_name", "is_active", "created_at"],
    "Groups": ["group_id", "branch", "program_name", "group_name", "is_active", "created_at"],
    "ExamTypes": ["examtype_id", "branch", "program_name", "group_name", "exam_type", "is_active", "created_at"],

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

# ---------------- utils ----------------
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

# ---------------- gsheet ----------------
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
    return gs_client().open_by_key(st.secrets["GSHEET_ID"])

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
    ws = open_spreadsheet().worksheet(ws_name)
    values = ws.get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)

def append_row(ws_name: str, row: dict):
    ws = open_spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [norm(row.get(h, "")) for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_cell(ws_name: str, row_index_1based: int, col_name: str, value):
    ws = open_spreadsheet().worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    col_index = headers.index(col_name) + 1
    ws.update_cell(row_index_1based, col_index, value)
    st.cache_data.clear()

def delete_group_timetable(branch: str, program: str, group: str):
    ws = open_spreadsheet().worksheet("Timetable")
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

# ---------------- session/auth ----------------
def ensure_session():
    if "role" not in st.session_state:
        st.session_state.role = None
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

# ---------------- pages ----------------
def page_login():
    st.title("🧩 Portail Mega Formation — Connexion")

    branches_df = read_df("Branches")
    branches = get_unique(branches_df, "branch")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("👨‍💼 Connexion Employé (mot de passe du centre)")
        if not branches:
            st.warning("La feuille 'Branches' est vide.")
        else:
            branch = st.selectbox("Centre", branches, key="staff_branch")
            pwd = st.text_input("Mot de passe du centre", type="password", key="staff_pwd")
            if st.button("Se connecter (Employé)", use_container_width=True):
                user = staff_branch_login(branch, pwd)
                if user:
                    st.session_state.role = "staff"
                    st.session_state.user = user
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Mot de passe incorrect / centre inactif.")

    with c2:
        st.subheader("🎓 Connexion Stagiaire")
        phone = st.text_input("Téléphone", key="stud_phone")
        pwd2 = st.text_input("Mot de passe", type="password", key="stud_pwd")
        if st.button("Se connecter (Stagiaire)", use_container_width=True):
            acc = student_login(phone, pwd2)
            if acc:
                df = read_df("Accounts")
                idx = df.index[df["phone"].astype(str).str.strip() == norm(phone)].tolist()
                if idx:
                    update_cell("Accounts", idx[0] + 2, "last_login", now_str())
                st.session_state.role = "student"
                st.session_state.user = acc
                st.session_state.page = "Home"
                st.rerun()
            else:
                st.error("Téléphone / mot de passe incorrect.")

    st.divider()
    st.subheader("🆕 Inscription Stagiaire")
    st.caption("Centre → Spécialité → Groupe → Nom (déjà ajouté par l'employé) ثم Téléphone + Mot de passe.")

    tr = read_df("Trainees")
    if tr.empty or not branches:
        st.info("L'employé doit ajouter centres / stagiaires d'abord.")
        return

    b = st.selectbox("Centre", branches, key="reg_branch")
    tr_b = tr[tr["branch"].astype(str).str.strip() == norm(b)].copy()

    # programs from Programs sheet (employee controlled)
    prog_df = df_filter(read_df("Programs"), branch=b)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])

    if not programs:
        st.warning("Aucune spécialité (Programs) pour ce centre. L'employé doit les ajouter.")
        return
    p = st.selectbox("Spécialité", programs, key="reg_program")

    grp_df = df_filter(read_df("Groups"), branch=b, program_name=p)
    grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])

    if not groups:
        st.warning("Aucun groupe (Groups) pour cette spécialité. L'employé doit les ajouter.")
        return
    g = st.selectbox("Groupe", groups, key="reg_group")

    tr_f = tr_b[
        (tr_b["program"].astype(str).str.strip() == norm(p)) &
        (tr_b["group"].astype(str).str.strip() == norm(g))
    ].copy()

    if tr_f.empty:
        st.warning("Aucun stagiaire dans ce groupe. L'employé doit les ajouter.")
        return

    search = st.text_input("🔎 Rechercher un nom (optionnel)", key="reg_search")
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
            st.error("Ce téléphone est déjà inscrit.")
            return

        append_row("Accounts", {
            "phone": phone,
            "password": pwd,
            "trainee_id": norm(trainee_id),
            "created_at": now_str(),
            "last_login": ""
        })
        st.success("✅ Compte créé. Vous pouvez vous connecter.")

def staff_sidebar_controls():
    """Everything employee does is in the LEFT sidebar."""
    staff_branch = norm(st.session_state.user.get("branch"))
    st.sidebar.markdown("## 👨‍💼 Espace Employé")
    st.sidebar.success(f"Centre: {staff_branch}")

    # Employee controls programs/groups/exam types (no free typing in student side)
    st.sidebar.caption("Gestion complète: spécialités, groupes, stagiaires, matières, types d'examen, notes, emploi du temps.")

    # 1) Programs
    with st.sidebar.expander("🏷️ Spécialités (Programs)", expanded=False):
        prog_df = df_filter(read_df("Programs"), branch=staff_branch)
        prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        st.sidebar.write("Actuelles:")
        if prog_df.empty:
            st.sidebar.info("Aucune spécialité.")
        else:
            st.sidebar.dataframe(prog_df[["program_name"]], use_container_width=True, hide_index=True)

        new_prog = st.sidebar.text_input("Nouvelle spécialité", key="new_prog")
        if st.sidebar.button("Ajouter spécialité", use_container_width=True, key="btn_add_prog"):
            if not norm(new_prog):
                st.sidebar.error("Nom obligatoire.")
            else:
                append_row("Programs", {
                    "program_id": f"PR-{uuid.uuid4().hex[:8].upper()}",
                    "branch": staff_branch,
                    "program_name": norm(new_prog),
                    "is_active": "true",
                    "created_at": now_str()
                })
                st.sidebar.success("✅ Ajouté.")
                st.rerun()

    # Program selection (from Programs sheet)
    prog_df = df_filter(read_df("Programs"), branch=staff_branch)
    prog_df = prog_df[prog_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
    programs = sorted([x for x in prog_df["program_name"].astype(str).str.strip().tolist() if x])
    program = st.sidebar.selectbox("Spécialité (pour gérer)", programs, key="manage_program") if programs else None

    # 2) Groups
    with st.sidebar.expander("👥 Groupes (Groups)", expanded=False):
        if not program:
            st.sidebar.info("Choisissez une spécialité.")
        else:
            grp_df = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
            grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
            st.sidebar.write("Actuels:")
            if grp_df.empty:
                st.sidebar.info("Aucun groupe.")
            else:
                st.sidebar.dataframe(grp_df[["group_name"]], use_container_width=True, hide_index=True)

            new_group = st.sidebar.text_input("Nouveau groupe", key="new_group")
            if st.sidebar.button("Ajouter groupe", use_container_width=True, key="btn_add_group"):
                if not norm(new_group):
                    st.sidebar.error("Nom obligatoire.")
                else:
                    append_row("Groups", {
                        "group_id": f"GP-{uuid.uuid4().hex[:8].upper()}",
                        "branch": staff_branch,
                        "program_name": norm(program),
                        "group_name": norm(new_group),
                        "is_active": "true",
                        "created_at": now_str()
                    })
                    st.sidebar.success("✅ Ajouté.")
                    st.rerun()

    # Group selection (from Groups sheet)
    group = None
    if program:
        grp_df = df_filter(read_df("Groups"), branch=staff_branch, program_name=program)
        grp_df = grp_df[grp_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()
        groups = sorted([x for x in grp_df["group_name"].astype(str).str.strip().tolist() if x])
        group = st.sidebar.selectbox("Groupe (pour gérer)", groups, key="manage_group") if groups else None

    if not (program and group):
        st.sidebar.info("Choisissez spécialité + groupe.")
        return staff_branch, program, group

    st.sidebar.divider()

    # 3) Exam Types
    with st.sidebar.expander("🧾 Types d'examen (ExamTypes)", expanded=False):
        et_df = df_filter(read_df("ExamTypes"), branch=staff_branch, program_name=program, group_name=group)
        et_df = et_df[et_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()

        if et_df.empty:
            st.sidebar.info("Aucun type d'examen.")
        else:
            st.sidebar.dataframe(et_df[["exam_type"]], use_container_width=True, hide_index=True)

        new_et = st.sidebar.text_input("Nouveau type", key="new_examtype", placeholder="Ex: Devoir 1 / Oral / Final ...")
        if st.sidebar.button("Ajouter type", use_container_width=True, key="btn_add_examtype"):
            if not norm(new_et):
                st.sidebar.error("Nom obligatoire.")
            else:
                append_row("ExamTypes", {
                    "examtype_id": f"ET-{uuid.uuid4().hex[:8].upper()}",
                    "branch": staff_branch,
                    "program_name": norm(program),
                    "group_name": norm(group),
                    "exam_type": norm(new_et),
                    "is_active": "true",
                    "created_at": now_str()
                })
                st.sidebar.success("✅ Ajouté.")
                st.rerun()

    st.sidebar.divider()

    # 4) Add Trainee
    with st.sidebar.expander("➕ Ajouter un stagiaire (Trainees)", expanded=False):
        name = st.sidebar.text_input("Nom & Prénom", key="add_tr_name")
        phone = st.sidebar.text_input("Téléphone (optionnel)", key="add_tr_phone")
        status = st.sidebar.selectbox("Statut", ["active", "inactive"], key="add_tr_status")
        if st.sidebar.button("Enregistrer stagiaire", use_container_width=True, key="btn_add_tr"):
            if not norm(name):
                st.sidebar.error("Nom obligatoire.")
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
                st.sidebar.success("✅ Ajouté.")
                st.rerun()

    # 5) Add Subject
    with st.sidebar.expander("📚 Ajouter une matière (Subjects)", expanded=False):
        subject_name = st.sidebar.text_input("Nom de la matière", key="add_subj_name")
        if st.sidebar.button("Ajouter matière", use_container_width=True, key="btn_add_subj"):
            if not norm(subject_name):
                st.sidebar.error("Nom obligatoire.")
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
                st.sidebar.success("✅ Ajouté.")
                st.rerun()

    # 6) Enter Grade
    with st.sidebar.expander("📝 Saisir une note (Grades)", expanded=True):
        tr_all = df_filter(read_df("Trainees"), branch=staff_branch, program=program, group=group)
        subf = df_filter(read_df("Subjects"), branch=staff_branch, program=program, group=group)
        et_df = df_filter(read_df("ExamTypes"), branch=staff_branch, program_name=program, group_name=group)
        et_df = et_df[et_df["is_active"].astype(str).str.strip().str.lower() != "false"].copy()

        if tr_all.empty:
            st.sidebar.warning("Aucun stagiaire.")
        elif subf.empty:
            st.sidebar.warning("Ajoutez des matières d'abord.")
        elif et_df.empty:
            st.sidebar.warning("Ajoutez des types d'examen d'abord.")
        else:
            tr_all = tr_all.copy()
            tr_all["label"] = tr_all["full_name"].astype(str).str.strip() + " — " + tr_all["trainee_id"].astype(str).str.strip()
            trainee_choice = st.sidebar.selectbox("Stagiaire", tr_all["label"].tolist(), key="grade_tr")
            trainee_id = tr_all[tr_all["label"] == trainee_choice].iloc[0]["trainee_id"]

            subject = st.sidebar.selectbox("Matière", sorted(subf["subject_name"].astype(str).str.strip().tolist()), key="grade_subj")
            exam_type = st.sidebar.selectbox("Type d'examen", sorted(et_df["exam_type"].astype(str).str.strip().tolist()), key="grade_examtype")
            score = st.sidebar.number_input("Note /20", min_value=0.0, max_value=20.0, value=10.0, step=0.25, key="grade_score")
            date = st.sidebar.date_input("Date", key="grade_date")
            note = st.sidebar.text_input("Remarque (optionnel)", key="grade_note")

            if st.sidebar.button("Enregistrer note", use_container_width=True, key="btn_save_grade"):
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
                    "staff_name": f"Employé-{staff_branch}",
                    "note": norm(note),
                    "created_at": now_str()
                })
                st.sidebar.success("✅ Note enregistrée.")

    # 7) Timetable editor
    with st.sidebar.expander("🗓️ Emploi du temps (Timetable)", expanded=False):
        tt = df_filter(read_df("Timetable"), branch=staff_branch, program=program, group=group)
        if tt.empty:
            base = pd.DataFrame([{**DEFAULT_TIMETABLE_ROW, "row_id": f"TT-{uuid.uuid4().hex[:8].upper()}"}])
        else:
            base = tt[["row_id","day","start","end","subject","room","teacher"]].copy()

        edited = st.sidebar.data_editor(base, use_container_width=True, num_rows="dynamic", key="tt_editor")

        if st.sidebar.button("Sauvegarder emploi du temps", use_container_width=True, key="btn_save_tt"):
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
            st.sidebar.success("✅ Sauvegardé.")
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Se déconnecter", use_container_width=True):
        logout()
        st.rerun()

    return staff_branch, program, group

def student_center_view():
    st.markdown("## 🎓 Espace Stagiaire")
    st.caption("Lecture فقط: notes, matières, emploi du temps.")

    if st.session_state.role != "student":
        st.info("Connectez-vous en tant que stagiaire depuis Login.")
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

def page_home():
    st.title("🧩 Portail Mega Formation")
    st.caption("À gauche: Employé (gestion). Au centre: Stagiaire (consultation).")

    if st.session_state.role == "staff":
        staff_sidebar_controls()
        st.info("Vous êtes connecté en tant qu'employé. (Le centre stagiaire est فقط للطلاب المسجلين).")
    else:
        st.sidebar.markdown("## 👨‍💼 Espace Employé")
        st.sidebar.info("Connectez-vous (Login) pour gérer le centre.")
        if st.sidebar.button("Aller à Login", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()

    st.divider()
    student_center_view()

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
