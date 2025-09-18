
import os, re, time, tempfile, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
import streamlit.components.v1 as components

# ffmpeg
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = str(Path(ffmpeg_path).parent) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

from faster_whisper import WhisperModel
import dateparser

st.set_page_config(page_title="Whisper Small → Tasks → Jira", page_icon="🌀", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>🌀 Whisper Small → Tasks → Jira</h1>", unsafe_allow_html=True)
st.caption("CPU-only · faster-whisper small · Start button · Live progress · Stable checkboxes · Jira ADF · Enter→Next")

MODEL_NAME = "small"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"

@st.cache_resource(show_spinner=True)
def load_model_cached() -> WhisperModel:
    return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

model = load_model_cached()

with st.sidebar:
    st.subheader("⚙️ Настройки")
    lang_hint = st.selectbox("Язык аудио (подсказка)", ["auto", "ru", "kk", "en", "tr"], index=0)

def detect_lang_code(text: str) -> str:
    cyr = sum('а' <= ch.lower() <= 'я' or ch == 'ё' for ch in text)
    lat = sum('a' <= ch.lower() <= 'z' for ch in text)
    return "ru" if cyr > lat else "en"

def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+|[\n\r]+|•\s*| - ", text.strip())
    return [s.strip(" \t-—•") for s in sents if len(s.strip()) > 2]

def expand_compounds(s: str) -> List[str]:
    parts = re.split(r"\b(и|а также|затем|после этого|потом|далее|and then|and)\b", s, flags=re.IGNORECASE)
    out = []
    for p in parts:
        if re.fullmatch(r"(и|а также|затем|после этого|потом|далее|and then|and)", p, flags=re.IGNORECASE):
            continue
        frag = p.strip(" ,.;:—-")
        if frag:
            out.append(frag)
    return out or [s]

VERB_RE = r"(провести|подготовить|отправить|создать|написать|проверить|созвониться|добавить|исправить|закрыть|запланировать|согласовать|обновить|описать|развернуть|подключить|оформить|назначить|организовать|презентовать|ожидать|собрать|дать|выполнить|подтвердить|утвердить|поделиться|скинуть|зафиксировать|напомнить|подвести итоги|review|plan|schedule|deploy|implement|prepare|send|create|write|check|fix|update|investigate|present|follow up)"

def candidate_actions(text: str) -> List[str]:
    out = []
    for s in split_sentences(text):
        if not re.search(VERB_RE, s, flags=re.IGNORECASE):
            continue
        for sub in expand_compounds(s):
            sub = re.sub(r"(?i)\b(нужно|надо|будет|давайте|давай|предлагаю)\s+", "", sub).strip()
            m = re.search(VERB_RE + r".*", sub, flags=re.IGNORECASE)
            frag = sub[m.start():].strip() if m else sub
            frag = re.split(r"[.;!?]", frag)[0].strip(" ,.;:—-")
            words = frag.split()
            if len(words) > 16:
                frag = " ".join(words[:16])
            if len(frag) >= 3:
                out.append(frag)
    seen, res = set(), []
    for t in out:
        if t not in seen:
            res.append(t); seen.add(t)
    return res

def extract_tasks(text: str) -> List[Dict[str, Any]]:
    tasks = []
    for frag in candidate_actions(text):
        email = None
        m_email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', frag)
        if m_email: email = m_email.group(0)
        tasks.append({
            "id": hashlib.md5(frag.encode("utf-8")).hexdigest()[:10],
            "summary": frag[:120],
            "description": frag,
            "assignee_email": email,
            "due_text": None,
        })
    if not tasks and text:
        tasks.append({
            "id": hashlib.md5(text[:64].encode("utf-8")).hexdigest()[:10],
            "summary": "Review transcript & create tasks",
            "description": text[:2000],
            "assignee_email": None,
            "due_text": None,
        })
    return tasks

def to_adf(text: str) -> Dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text or ""}]}],
    }

def parse_due(due_text: Optional[str], ref_dt):
    if not due_text: return None
    dt = dateparser.parse(due_text, languages=["ru","en","kk","tr"], settings={"RELATIVE_BASE": ref_dt})
    return dt.date().isoformat() if dt else None

def create_jira_issue(base_url: str, email: str, api_token: str, project_key: str, task: Dict[str, Any], due_iso: Optional[str]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/rest/api/3/issue"
    auth = (email, api_token)
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": task["summary"][:120] if task["summary"] else "Task",
            "description": to_adf(task.get("description") or task["summary"]),
            "issuetype": {"name": "Task"},
        }
    }
    if due_iso:
        payload["fields"]["duedate"] = due_iso
    headers = {"Accept":"application/json","Content-Type":"application/json"}
    r = requests.post(url, auth=auth, json=payload, headers=headers, timeout=60)
    if r.status_code >= 300:
        return {"ok": False, "error": r.text}
    res = r.json()
    return {"ok": True, "key": res.get("key"), "id": res.get("id")}

# ------------------ UI ------------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1) Загрузка → Распознавание")
    file = st.file_uploader("Форматы: wav, mp3, m4a, ogg, flac, mp4, mov, mkv, webm",
                            type=["wav","mp3","m4a","ogg","flac","mp4","mov","mkv","webm"])
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
            tmp.write(file.read()); src_path = tmp.name
        st.audio(src_path)
        if st.button("▶️ Начать распознавание", type="primary"):
            prog = st.progress(0)
            elapsed_box = st.empty()
            live_text_box = st.empty()
            start_t = time.time()
            transcript_parts: List[str] = []

            kwargs = {}
            if lang_hint and lang_hint != "auto":
                kwargs["language"] = lang_hint
            segments, info = model.transcribe(
                src_path,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                **kwargs
            )
            duration = max(1.0, float(getattr(info, "duration", 1.0)))
            last_p = 0
            try:
                for seg in segments:
                    transcript_parts.append(seg.text)
                    p = min(100, int(100.0 * float(getattr(seg, "end", 0.0)) / duration))
                    if p > last_p: prog.progress(p); last_p = p
                    elapsed_box.markdown(f"⏱️ Время: **{time.time()-start_t:.1f} сек** · Прогресс: **{p}%**")
                    live_text_box.markdown(f"**Распознанный текст (on-the-fly):**\n\n{''.join(transcript_parts)}")
            finally:
                try: os.unlink(src_path)
                except Exception: pass

            final_text = "".join(transcript_parts).strip()
            st.success(f"Готово · Длительность: {duration:.1f} сек · Всего: {time.time()-start_t:.1f} сек · Язык (по тексту): {detect_lang_code(final_text)}")
            st.download_button("⬇️ Скачать транскрипт (.txt)", data=final_text, file_name="transcript.txt", mime="text/plain")
            st.session_state["final_transcript"] = final_text
            st.session_state["ref_datetime"] = datetime.now(timezone.utc).isoformat()

# After transcription
final_text = st.session_state.get("final_transcript", "")
if final_text:
    st.subheader("📝 Транскрипт (редактируемый)")
    edited_text = st.text_area("Текст", value=final_text, height=220, key="final_text_area")

    text_hash = hashlib.md5(edited_text.encode("utf-8")).hexdigest()
    if st.session_state.get("tasks_src_hash") != text_hash:
        tasks = extract_tasks(edited_text)
        st.session_state["tasks"] = tasks
        st.session_state["tasks_src_hash"] = text_hash
        # initialize checkbox states (all True)
        for t in tasks:
            st.session_state.setdefault(f"inc_{t['id']}", True)

    st.markdown("### ✅ Задачи (галочки стоят изначально — снимай лишние)")
    c1, c2 = st.columns(2)
    if c1.button("Выбрать всё"):
        for t in st.session_state.get("tasks", []):
            st.session_state[f"inc_{t['id']}"] = True
    if c2.button("Снять всё"):
        for t in st.session_state.get("tasks", []):
            st.session_state[f"inc_{t['id']}"] = False

    for t in st.session_state.get("tasks", []):
        cols = st.columns([0.08, 0.92])
        with cols[0]:
            st.checkbox("", key=f"inc_{t['id']}")
        with cols[1]:
            st.write(f"**{t['summary']}**")
            if t.get("description") and t["description"] != t["summary"]:
                st.caption(t["description"])

with col2:
    st.subheader("2) Отправка задач в Jira")
    with st.form("jira_form", clear_on_submit=False):
        jira_url = st.text_input("Jira URL", placeholder="https://your-domain.atlassian.net")
        jira_email = st.text_input("Jira email")
        jira_token = st.text_input("Jira API token", type="password")
        jira_project = st.text_input("Project Key", placeholder="PRJ")
        submitted = st.form_submit_button("🚀 Создать задачи в Jira", type="primary")

        # Inject JS: Enter → focus next input; on last input → submit the form
        components.html("""
<script>
(function(){
  const stFrame = window.frameElement; // our iframe
  if(!stFrame) return;
  const root = stFrame.closest('div[data-testid="stForm"]') || stFrame.parentElement;
  // fallback: whole document
  const doc = window.parent.document;
  const form = doc.querySelector('form[data-testid="stForm"]');
  if(!form) return;
  const inputs = form.querySelectorAll('input[type="text"], input[type="password"]');
  const submitBtn = form.querySelector('button[kind="primary"]');
  inputs.forEach((inp, idx)=>{
    inp.addEventListener('keydown', (e)=>{
      if(e.key === 'Enter'){
        e.preventDefault();
        if(idx < inputs.length-1){
          inputs[idx+1].focus();
        }else{
          submitBtn && submitBtn.click();
        }
      }
    });
  });
})();
</script>
        """, height=0)
    if submitted:
        missing = []
        if not jira_url.strip(): missing.append("URL")
        if not jira_email.strip(): missing.append("email")
        if not jira_token.strip(): missing.append("API token")
        if not jira_project.strip(): missing.append("Project Key")
        if missing:
            st.warning("Заполни поля: " + ", ".join(missing))
        else:
            chosen = [t for t in st.session_state.get("tasks", []) if st.session_state.get(f"inc_{t['id']}", True)]
            if not chosen:
                st.error("Нет выбранных задач.")
            else:
                ref_iso = st.session_state.get("ref_datetime", datetime.now(timezone.utc).isoformat())
                ref_dt = datetime.fromisoformat(ref_iso)
                results = []
                with st.spinner("Создаю задачи..."):
                    for t in chosen:
                        due_iso = parse_due(t.get("due_text"), ref_dt) if t.get("due_text") else None
                        res = create_jira_issue(jira_url, jira_email, jira_token, jira_project, t, due_iso)
                        results.append(res)
                ok = [r for r in results if r.get("ok")]
                bad = [r for r in results if not r.get("ok")]
                if ok: st.success("Создано: " + ", ".join([r.get("key") or r.get("id","?") for r in ok]))
                if bad: st.error("Ошибки: " + "; ".join([r.get("error","")[:160] for r in bad]))

st.caption("Галочки стоят изначально → снимаешь лишнее. Enter в полях Jira: переход, на последнем — отправка.")
