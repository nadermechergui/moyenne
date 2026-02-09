import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Mega Portal (Branches)", page_icon="🧩", layout="wide")

REQUIRED_SHEETS = {
    "Branches": ["branch", "staff_password", "is_active", "created_at"],
    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "created_at", "last_login"],
    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades": ["grade_id", "trainee_id", "branch", "program", "group", "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],
    "Timetable": ["row_id", "branch", "program", "group", "day", "start", "end", "subject", "room", "teacher", "created_at"],
}

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# =========================
# GSHEETS HELPERS
# =========================
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
            sh.add_worksheet(title=ws_name, rows=2000, cols=max(12, len(headers)+2))
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
    df = pd.DataFrame(rows, columns=headers)
    return df

def append_row(ws_name: str, row: dict):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [str(row.get(h, "")).strip() for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_cell(ws_name: str, row_index_1based: int, col_name: str, value):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    col_index = headers.index(col_name) + 1
    ws.update_cell(row_index_1based, col_index, value)
    st.cache_data.clear()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_unique(df, col):
    if df.empty or col not in df.columns:
        return []
    return sorted([x for x in df[col].astype(str).str.strip().unique().tolist() if x])

# =========================
# SESSION
# =========================
def ensure_session():
    if "role" not in st.session_state:
        st.session_state.role = None  # "staff" or "student"
    if "user" not in st.session_state:
        st.session_state.user = {}
    if "page" not in st.session_state:
        st.session_state.page = "Login"

def logout():
    st.session_state.role = None
    st.session_state.user = {}
    st.session_state.page = "Login"

# =========================
# AUTH
# =========================
def staff_branch_login(branch: str, branch_password: str, staff_name: str):
    df = read_df("Branches")
    if df.empty:
        return None

    df2 = df.copy()
    df2["branch"] = df2["branch"].astype(str).str.strip()
    df2["staff_password"] = df2["staff_password"].astype(str).str.strip()
    df2["is_active"] = df2["is_active"].astype(str).str.strip().str.lower()

    m = df2[(df2["branch"] == str(branch).strip()) &
            (df2["staff_password"] == str(branch_password).strip()) &
            (df2["is_active"] != "false")]

    if m.empty:
        return None

    return {
        "branch": str(branch).strip(),
        "staff_name": str(staff_name).strip() if staff_name else "Staff",
        "role": "staff"
    }

def student_login(phone: str, password: str):
    df = read_df("Accounts")
    if df.empty:
        return None
    df2 = df.copy()
    df2["phone"] = df2["phone"].astype(str).str.strip()
    df2["password"] = df2["password"].astype(str).str.strip()
    m = df2[(df2["phone"] == str(phone).strip()) & (df2["password"] == str(password).strip())]
    if m.empty:
        return None
    return m.iloc[0].to_dict()

# =========================
# PAGES
# =========================
def page_login():
    st.title("🧩 Mega Portal — Login")

    branches_df = read_df("Branches")
    branches = get_unique(branches_df, "branch")

    c1, c2 = st.columns(2, gap="large")

    # ---- Staff login (by branch password)
    with c1:
        st.subheader("👩‍💼 Staff Login (Branch Password)")
        if not branches:
            st.warning("Sheet Branches فارغة. زيد الفروع وكلمات السر.")
        else:
            branch = st.selectbox("Branch", branches, key="staff_branch")
            staff_pwd = st.text_input("Branch Password", type="password", key="staff_branch_pwd")
            staff_name = st.text_input("Staff Name (for logs)", key="staff_name", placeholder="مثال: Olfa / Ons ...")

            if st.button("Login as Staff", use_container_width=True):
                user = staff_branch_login(branch, staff_pwd, staff_name)
                if user:
                    st.session_state.role = "staff"
                    st.session_state.user = user
                    st.session_state.page = "Staff"
                    st.success("Welcome (Staff) ✅")
                    st.rerun()
                else:
                    st.error("Wrong branch password / branch inactive.")

    # ---- Student login
    with c2:
        st.subheader("🎓 Student Login")
        phone2 = st.text_input("Phone", key="stud_phone")
        pwd2 = st.text_input("Password", type="password", key="stud_pwd")
        if st.button("Login as Student", use_container_width=True):
            acc = student_login(phone2, pwd2)
            if acc:
                # update last_login
                df = read_df("Accounts")
                idx = df.index[df["phone"].astype(str).str.strip() == str(phone2).strip()].tolist()
                if idx:
                    update_cell("Accounts", idx[0] + 2, "last_login", now_str())

                st.session_state.role = "student"
                st.session_state.user = acc
                st.session_state.page = "Student"
                st.success("Welcome (Student) ✅")
                st.rerun()
            else:
                st.error("Wrong phone/password.")

    st.divider()

    # ---- Student registration
    st.subheader("🆕 Student Registration")
    st.caption("اختار Branch → Program → Group → اسمك، وبعدها Phone + Password.")

    tr = read_df("Trainees")
    if tr.empty:
        st.warning("Trainees sheet فارغة. الإدارة لازم تزيد المتكوّنين أولاً.")
        return

    if not branches:
        st.warning("Branches sheet فارغة. زيد الفروع أولاً.")
        return

    b = st.selectbox("Branch", branches, key="reg_branch")
    tr_b = tr[tr["branch"].astype(str).str.strip() == str(b).strip()].copy()

    programs = get_unique(tr_b, "program")
    if not programs:
        st.warning("ما فماش Programs في الفرع هذا.")
        return
    p = st.selectbox("Program", programs, key="reg_program")
    tr_bp = tr_b[tr_b["program"].astype(str).str.strip() == str(p).strip()].copy()

    groups = get_unique(tr_bp, "group")
    if not groups:
        st.warning("ما فماش Groups في الاختصاص هذا.")
        return
    g = st.selectbox("Group", groups, key="reg_group")

    tr_f = tr_bp[tr_bp["group"].astype(str).str.strip() == str(g).strip()].copy()
    if tr_f.empty:
        st.warning("ما فماش متكوّنين في المجموعة هذي.")
        return

    search = st.text_input("🔎 Search name (optional)", key="reg_search", placeholder="اكتب جزء من الاسم...")
    if search.strip():
        s = search.strip().lower()
        tr_f = tr_f[tr_f["full_name"].astype(str).str.lower().str.contains(s)]

    tr_f["label"] = tr_f["full_name"].astype(str).str.strip() + "  —  " + tr_f["trainee_id"].astype(str).str.strip()
    choice = st.selectbox("Choose your name", tr_f["label"].tolist(), key="reg_choice")
    chosen_id = tr_f[tr_f["label"] == choice].iloc[0]["trainee_id"]

    phone = st.text_input("Phone (unique)", key="reg_phone")
    password = st.text_input("Password", type="password", key="reg_password")

    if st.button("Create Account", use_container_width=True):
        phone = str(phone).strip()
        password = str(password).strip()

        if not phone or not password:
            st.error("لازم Phone و Password.")
            return
        if len(password) < 4:
            st.error("المودباس قصير. خليه 4 أحرف/أرقام ولا أكثر.")
            return

        acc = read_df("Accounts")
        if not acc.empty and acc["phone"].astype(str).str.strip().eq(phone).any():
            st.error("رقم الهاتف هذا مسجّل قبل. استعمل Login.")
            return

        append_row("Accounts", {
            "phone": phone,
            "password": password,
            "trainee_id": chosen_id,
            "created_at": now_str(),
            "last_login": ""
        })
        st.success("✅ Account created. توا تنجم تعمل Login.")

def page_staff():
    user = st.session_state.user
    staff_branch = str(user.get("branch", "")).strip()
    staff_name = str(user.get("staff_name", "Staff")).strip()

    st.title(f"👩‍💼 Staff CRM — {staff_branch}")
    st.caption(f"Staff: {staff_name} | Branch locked: {staff_branch}")

    tr_all = read_df("Trainees")
    tr_all = tr_all[tr_all["branch"].astype(str).str.strip() == staff_branch].copy()

    if tr_all.empty:
        st.warning("ما فماش Trainees في الفرع هذا. عمّر Sheet Trainees.")
        if st.button("Logout"):
            logout(); st.rerun()
        return

    programs = get_unique(tr_all, "program")
    program = st.sidebar.selectbox("Program", programs) if programs else None
    tr_p = tr_all[tr_all["program"].astype(str).str.strip() == str(program).strip()].copy() if program else tr_all

    groups = get_unique(tr_p, "group")
    group = st.sidebar.selectbox("Group", groups) if groups else None
    trf = tr_p[tr_p["group"].astype(str).str.strip() == str(group).strip()].copy() if group else pd.DataFrame()

    tab1, tab2, tab3, tab4 = st.tabs(["👥 Trainees", "📚 Subjects", "📝 Grades", "🗓️ Timetable"])

    # ---- Trainees
    with tab1:
        st.subheader("Trainees (this branch)")
        if not (program and group):
            st.info("اختار Program + Group من اليسار.")
        else:
            if trf.empty:
                st.warning("ما فماش متكوّنين في المجموعة هذي.")
            else:
                st.dataframe(trf[["trainee_id","full_name","phone","program","group","status"]], use_container_width=True, hide_index=True)

            st.markdown("### ➕ Add trainee")
            with st.form("add_tr", clear_on_submit=True):
                name = st.text_input("Full name")
                phone = st.text_input("Phone (optional)")
                status = st.selectbox("Status", ["active","inactive"], index=0)
                ok = st.form_submit_button("Save", use_container_width=True)
                if ok:
                    if not name.strip():
                        st.error("لازم الاسم.")
                    else:
                        append_row("Trainees", {
                            "trainee_id": f"TR-{uuid.uuid4().hex[:8].upper()}",
                            "full_name": name.strip(),
                            "phone": phone.strip(),
                            "branch": staff_branch,
                            "program": str(program).strip(),
                            "group": str(group).strip(),
                            "status": status,
                            "created_at": now_str()
                        })
                        st.success("✅ Added.")
                        st.rerun()

    # ---- Subjects
    with tab2:
        st.subheader("Subjects (per group)")
        if not (program and group):
            st.info("اختار Program + Group من اليسار.")
        else:
            sub = read_df("Subjects")
            subf = sub[
                (sub["branch"].astype(str).str.strip() == staff_branch) &
                (sub["program"].astype(str).str.strip() == str(program).strip()) &
                (sub["group"].astype(str).str.strip() == str(group).strip())
            ].copy() if not sub.empty else pd.DataFrame()

            if subf.empty:
                st.info("No subjects yet.")
            else:
                st.dataframe(subf[["subject_id","subject_name","is_active"]], use_container_width=True, hide_index=True)

            with st.form("add_subject", clear_on_submit=True):
                subject_name = st.text_input("Subject name")
                ok = st.form_submit_button("Add", use_container_width=True)
                if ok:
                    if not subject_name.strip():
                        st.error("اكتب اسم المادة.")
                    else:
                        append_row("Subjects", {
                            "subject_id": f"SB-{uuid.uuid4().hex[:8].upper()}",
                            "branch": staff_branch,
                            "program": str(program).strip(),
                            "group": str(group).strip(),
                            "subject_name": subject_name.strip(),
                            "is_active": "true",
                            "created_at": now_str()
                        })
                        st.success("✅ Added.")
                        st.rerun()

    # ---- Grades
    with tab3:
        st.subheader("Enter grades")
        if not (program and group):
            st.info("اختار Program + Group من اليسار.")
        else:
            if trf.empty:
                st.warning("ما فماش متكوّنين في المجموعة هذي.")
            else:
                sub = read_df("Subjects")
                subf = sub[
                    (sub["branch"].astype(str).str.strip() == staff_branch) &
                    (sub["program"].astype(str).str.strip() == str(program).strip()) &
                    (sub["group"].astype(str).str.strip() == str(group).strip())
                ].copy() if not sub.empty else pd.DataFrame()

                if subf.empty:
                    st.warning("زيد مواد للمجموعة من تبويب Subjects.")
                else:
                    trf2 = trf.copy()
                    trf2["label"] = trf2["full_name"].astype(str).str.strip() + " — " + trf2["trainee_id"].astype(str).str.strip()
                    trainee_choice = st.selectbox("Trainee", trf2["label"].tolist())
                    trainee_id = trf2[trf2["label"] == trainee_choice].iloc[0]["trainee_id"]

                    subject = st.selectbox("Subject", sorted(subf["subject_name"].astype(str).str.strip().tolist()))
                    exam_type = st.selectbox("Exam type", ["Exam", "Oral", "Final", "Surprise", "Other"])
                    exam_custom = st.text_input("Custom exam type (if Other)") if exam_type == "Other" else ""
                    score = st.number_input("Score /20", min_value=0.0, max_value=20.0, value=10.0, step=0.25)
                    date = st.date_input("Date")
                    note = st.text_input("Note (optional)")

                    if st.button("Save grade", use_container_width=True):
                        et = exam_custom.strip() if exam_type == "Other" else exam_type
                        if not et:
                            st.error("اكتب نوع الامتحان.")
                        else:
                            append_row("Grades", {
                                "grade_id": f"GR-{uuid.uuid4().hex[:10].upper()}",
                                "trainee_id": str(trainee_id).strip(),
                                "branch": staff_branch,
                                "program": str(program).strip(),
                                "group": str(group).strip(),
                                "subject_name": str(subject).strip(),
                                "exam_type": et,
                                "score": str(score),
                                "date": str(date),
                                "staff_name": staff_name,
                                "note": note.strip(),
                                "created_at": now_str()
                            })
                            st.success("✅ Saved.")

                    st.divider()
                    st.markdown("### Latest grades (this group)")
                    gr = read_df("Grades")
                    if gr.empty:
                        st.info("No grades yet.")
                    else:
                        grg = gr[
                            (gr["branch"].astype(str).str.strip() == staff_branch) &
                            (gr["program"].astype(str).str.strip() == str(program).strip()) &
                            (gr["group"].astype(str).str.strip() == str(group).strip())
                        ].copy()
                        if grg.empty:
                            st.info("No grades for this group.")
                        else:
                            grg = grg.sort_values(by=["date","created_at"], ascending=False).head(50)
                            st.dataframe(grg[["trainee_id","subject_name","exam_type","score","date","staff_name"]], use_container_width=True, hide_index=True)

    # ---- Timetable
    with tab4:
        st.subheader("Timetable editor")
        if not (program and group):
            st.info("اختار Program + Group من اليسار.")
        else:
            tt = read_df("Timetable")
            ttf = tt[
                (tt["branch"].astype(str).str.strip() == staff_branch) &
                (tt["program"].astype(str).str.strip() == str(program).strip()) &
                (tt["group"].astype(str).str.strip() == str(group).strip())
            ].copy() if not tt.empty else pd.DataFrame()

            if ttf.empty:
                base = pd.DataFrame([{
                    "row_id": f"TT-{uuid.uuid4().hex[:8].upper()}",
                    "day": "Monday", "start": "18:00", "end": "19:30",
                    "subject": "", "room": "", "teacher": ""
                }])
            else:
                base = ttf[["row_id","day","start","end","subject","room","teacher"]].copy()

            edited = st.data_editor(base, use_container_width=True, num_rows="dynamic")
            if st.button("Save timetable", use_container_width=True):
                # rewrite group rows
                sh = open_spreadsheet()
                ws = sh.worksheet("Timetable")
                all_vals = ws.get_all_values()
                headers = all_vals[0]
                rows = all_vals[1:]

                to_delete = []
                for i, r in enumerate(rows, start=2):
                    rdict = dict(zip(headers, r + [""]*(len(headers)-len(r))))
                    if (rdict.get("branch","").strip() == staff_branch and
                        rdict.get("program","").strip() == str(program).strip() and
                        rdict.get("group","").strip() == str(group).strip()):
                        to_delete.append(i)

                for ridx in sorted(to_delete, reverse=True):
                    ws.delete_rows(ridx)

                for _, row in edited.iterrows():
                    if str(row.get("day","")).strip() == "":
                        continue
                    append_row("Timetable", {
                        "row_id": str(row.get("row_id") or f"TT-{uuid.uuid4().hex[:8].upper()}").strip(),
                        "branch": staff_branch,
                        "program": str(program).strip(),
                        "group": str(group).strip(),
                        "day": str(row.get("day","")).strip(),
                        "start": str(row.get("start","")).strip(),
                        "end": str(row.get("end","")).strip(),
                        "subject": str(row.get("subject","")).strip(),
                        "room": str(row.get("room","")).strip(),
                        "teacher": str(row.get("teacher","")).strip(),
                        "created_at": now_str()
                    })

                st.success("✅ Timetable saved.")
                st.rerun()

    st.divider()
    if st.button("Logout"):
        logout(); st.rerun()

def page_student():
    st.title("🎓 Student Portal")

    acc = st.session_state.user
    trainee_id = str(acc.get("trainee_id","")).strip()
    phone = str(acc.get("phone","")).strip()

    tr = read_df("Trainees")
    row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not tr.empty else pd.DataFrame()

    if row.empty:
        st.error("حسابك مربوط بtrainee_id غير موجود في Trainees. كلم الإدارة.")
        if st.button("Logout"):
            logout(); st.rerun()
        return

    info = row.iloc[0].to_dict()
    branch = str(info.get("branch","")).strip()
    program = str(info.get("program","")).strip()
    group = str(info.get("group","")).strip()

    st.success(f"مرحباً {info.get('full_name','')} ✅")
    st.caption(f"Branch: {branch} | Program: {program} | Group: {group} | Phone: {phone}")

    tab1, tab2, tab3 = st.tabs(["📝 Grades", "🗓️ Timetable", "📚 Subjects"])

    with tab1:
        gr = read_df("Grades")
        grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
        if grf.empty:
            st.info("مازال ما تمش إدخال نوطات.")
        else:
            grf = grf.sort_values(by=["date","created_at"], ascending=False)
            st.dataframe(grf[["subject_name","exam_type","score","date","staff_name","note"]], use_container_width=True, hide_index=True)

    with tab2:
        tt = read_df("Timetable")
        ttf = tt[
            (tt["branch"].astype(str).str.strip() == branch) &
            (tt["program"].astype(str).str.strip() == program) &
            (tt["group"].astype(str).str.strip() == group)
        ].copy() if not tt.empty else pd.DataFrame()

        if ttf.empty:
            st.info("جدول الأوقات موش موجود توّا.")
        else:
            show = ttf[["day","start","end","subject","room","teacher"]].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)

    with tab3:
        sub = read_df("Subjects")
        subf = sub[
            (sub["branch"].astype(str).str.strip() == branch) &
            (sub["program"].astype(str).str.strip() == program) &
            (sub["group"].astype(str).str.strip() == group)
        ].copy() if not sub.empty else pd.DataFrame()

        if subf.empty:
            st.info("ما فماش مواد متسجّلة للمجموعة توّا.")
        else:
            st.dataframe(subf[["subject_name"]], use_container_width=True, hide_index=True)

    st.divider()
    if st.button("Logout"):
        logout(); st.rerun()

# =========================
# ROUTER
# =========================
def sidebar_nav():
    st.sidebar.title("Mega Portal")
    if st.session_state.role == "staff":
        st.sidebar.success(f"Staff — {st.session_state.user.get('branch','')}")
        st.sidebar.caption("Branch password mode")
        if st.sidebar.button("Logout", use_container_width=True):
            logout(); st.rerun()
        st.session_state.page = "Staff"

    elif st.session_state.role == "student":
        st.sidebar.success("Student")
        if st.sidebar.button("Logout", use_container_width=True):
            logout(); st.rerun()
        st.session_state.page = "Student"
    else:
        st.session_state.page = "Login"

def main():
    ensure_session()
    ensure_worksheets_and_headers()
    sidebar_nav()

    if st.session_state.page == "Login":
        page_login()
    elif st.session_state.page == "Staff":
        if st.session_state.role != "staff":
            st.session_state.page = "Login"
            st.rerun()
        page_staff()
    elif st.session_state.page == "Student":
        if st.session_state.role != "student":
            st.session_state.page = "Login"
            st.rerun()
        page_student()
    else:
        page_login()

if __name__ == "__main__":
    main()
