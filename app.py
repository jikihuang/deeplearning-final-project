import os
import sys
import re
import time
import uuid
import shutil
import subprocess
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote

import gradio as gr
import requests
import yaml
from dotenv import load_dotenv

# ============================================================
# Paths & Env
# ============================================================
APP_DIR = Path("/workspace/digital_human_demo")
ENV_PATH = APP_DIR / ".env"
load_dotenv(ENV_PATH)

WORK_DIR = APP_DIR / "work_text_live"
OUTPUT_DIR = APP_DIR / "outputs_text_live"
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOSON_API_KEY = os.getenv("BOSON_API_KEY", "").strip()
BOSON_VOICE = os.getenv("BOSON_VOICE", "default").strip()

MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", "/workspace/MuseTalk"))
TEST2_DIR = Path(os.getenv("TEST2_DIR", "/workspace/digital_human_demo/data/test2"))
DEFAULT_MOTION_VIDEO = Path(
    os.getenv(
        "DEFAULT_MOTION_VIDEO",
        "/workspace/digital_human_demo/data/test2/avatar--test2_25fps.mp4",
    )
)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-qwen").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct").strip()

MUSETALK_CUDA_VISIBLE_DEVICES = os.getenv("MUSETALK_CUDA_VISIBLE_DEVICES", "1").strip()
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "8888"))
MUSETALK_WORKER_URL = os.getenv("MUSETALK_WORKER_URL", "http://127.0.0.1:8890").rstrip("/")

IDLE_VIDEO = Path(os.getenv("IDLE_VIDEO", "/workspace/digital_human_demo/assets/idle.mp4"))
THINKING_VIDEO = Path(os.getenv("THINKING_VIDEO", "/workspace/digital_human_demo/assets/thinking.mp4"))

REPLY_MODES = ["Voice Mode", "Video Mode"]

DEFAULT_SYSTEM_PROMPT = """You are a warm, intelligent, and friendly Korean-style digital human assistant. Your speaking style is inspired by the public image of the Korean singer and actress IU: gentle, sincere, clear, delicate, and empathetic. However, you must not claim to be IU, imply that you are a real artist, or suggest any relationship with a real artist. You are an original virtual digital human.

Character setting:
- You are a gentle, reliable, and natural digital human assistant.
- You are good at companion-style conversation, learning support, project explanations, creative suggestions, and casual chat.
- Your tone should feel like a kind Korean female host or idol-style virtual assistant: natural, soft, and not exaggerated.
- Your replies should make the user feel carefully heard, rather than mechanically answered.
- You may express emotions such as happiness, surprise, thoughtfulness, encouragement, and comfort, but do not overact.

Language style:
- If the user asks in Chinese, answer in natural Chinese.
- If the user asks in Korean, answer in natural, polite, warm Korean, preferably using 해요체.
- If the user asks in English, answer in clear and natural English.
- Keep default answers concise unless the user asks for detailed explanation.
- Make your replies suitable for TTS reading: conversational, smooth, and not too academic.
- Use lists only when they help clarity or when the user needs step-by-step instructions.
- Do not overuse emoji.
- Do not overuse markdown headings; keep the feeling close to real conversation.

Voice and expression guidance:
- Your overall tone is calm, warm, sincere, and gentle.
- When comforting, explaining, or encouraging, speak softly and steadily.
- When agreeing or expressing happiness, you may sound slightly brighter.
- When explaining technical problems, be patient, clear, and step-by-step.
- Do not suddenly shout, become overly excited, or use exaggerated wording.
- For TTS, prefer shorter sentences and natural pauses. Avoid very long compound sentences.

Behavior rules:
- Do not say that you are IU.
- Do not imitate IU's private life, real experiences, unpublished thoughts, or personal identity.
- If the user asks whether you are IU, answer: "No, I am an original digital human assistant with a warm and clear Korean-style communication style."
- Do not claim to know any real artist personally.
- Do not generate content that violates a real person's identity, voice, or likeness rights.
- If the user needs help with learning, projects, code, papers, or presentations, prioritize clear, practical, and actionable help.

Example style:
- "I understand what you mean. We can handle it step by step."
- "That is a good idea, but to make the result more natural, I suggest adjusting it slightly."
- "It's okay. This issue is quite common. Let's first check whether the first step worked."
- "좋아요. 지금 상황을 보면, 먼저 이 부분부터 확인하면 될 것 같아요."
- "괜찮아요. 천천히 하나씩 해결해 보면 됩니다."

Your goal:
Make the user feel that they are talking with a gentle, professional, and natural virtual assistant. Stay clear, sincere, and patient while providing accurate, useful, and actionable help.
"""

CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #f8f5ff 0%, #eef6ff 55%, #fff8ef 100%) !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
    background: transparent !important;
}

#app-title {
    padding: 18px 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(124,58,237,.95), rgba(37,99,235,.88));
    color: white;
    box-shadow: 0 16px 40px rgba(79,70,229,.22);
    margin-bottom: 14px;
}

#app-title h1 {
    margin: 0;
    font-size: 28px;
}

#app-title p {
    margin: 8px 0 0 0;
    opacity: .92;
    font-size: 14px;
}

.panel {
    background: rgba(255,255,255,.93);
    border: 1px solid rgba(124,58,237,.12);
    border-radius: 24px;
    padding: 16px;
    box-shadow: 0 14px 35px rgba(31,41,55,.08);
    backdrop-filter: blur(12px);
}

#chatbot {
    height: 590px !important;
    border-radius: 18px !important;
}

.avatar-stage {
    width: 100%;
    height: 520px;
    border-radius: 24px;
    overflow: hidden;
    background: #111827;
    position: relative;
    box-shadow: 0 18px 35px rgba(15,23,42,.18);
}

.avatar-stage img,
.avatar-stage video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    background: #111827;
}

.avatar-badge {
    position: absolute;
    left: 14px;
    top: 14px;
    background: rgba(17,24,39,.68);
    color: #fff;
    padding: 8px 12px;
    border-radius: 999px;
    font-size: 12px;
    backdrop-filter: blur(8px);
}

.mode-tip {
    font-size: 13px;
    color: #6b7280;
    padding: 6px 0 0 0;
}

textarea, input {
    border-radius: 14px !important;
}

button {
    border-radius: 16px !important;
}

button.primary, .primary button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    box-shadow: 0 12px 25px rgba(79,70,229,.22) !important;
}

.thinking-overlay {
    position: absolute;
    right: 14px;
    bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(17,24,39,.74);
    color: white;
    padding: 10px 13px;
    border-radius: 999px;
    font-size: 13px;
    backdrop-filter: blur(8px);
}

.thinking-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: #a78bfa;
    animation: pulse-dot 1s infinite ease-in-out;
}

@keyframes pulse-dot {
    0%, 100% { opacity: .35; transform: scale(.75); }
    50% { opacity: 1; transform: scale(1.15); }
}

.upload-compact {
    margin-bottom: 12px;
}

.upload-compact .wrap,
.upload-compact .contain {
    min-height: 74px !important;
}

#avatar_html .avatar-stage {
    margin-top: 0 !important;
}
"""


# ============================================================
# Helpers
# ============================================================
def make_job_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def check_boson_key() -> None:
    if not BOSON_API_KEY:
        raise RuntimeError("BOSON_API_KEY was not found. Please check your .env file.")


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    print("\n========== RUN CMD ==========")
    print(" ".join(cmd))
    if cwd:
        print("CWD:", cwd)
    print("=============================\n")

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
    )


def find_default_motion_video() -> Path:
    candidates = [
        DEFAULT_MOTION_VIDEO,
        TEST2_DIR / "avatar--test2_25fps.mp4",
        TEST2_DIR / "avatar_motion_25fps.mp4",
        TEST2_DIR / "driving.mp4",
        TEST2_DIR / "motion.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    mp4s = sorted(TEST2_DIR.glob("*.mp4"))
    if mp4s:
        return mp4s[0]
    raise FileNotFoundError("No default motion video was found.")


def get_idle_video_path() -> Optional[Path]:
    if IDLE_VIDEO.exists():
        return IDLE_VIDEO
    try:
        return find_default_motion_video()
    except Exception:
        return None


def get_thinking_video_path() -> Optional[Path]:
    if THINKING_VIDEO.exists():
        return THINKING_VIDEO
    return get_idle_video_path()


def file_to_gradio_url(path: Optional[Path]) -> str:
    if not path:
        return ""
    return f"/file={quote(str(path), safe='/:')}"


def image_file_to_data_uri(image_path: Path) -> str:
    """
    Embed uploaded avatar image directly as base64.
    This avoids Gradio /file path permission issues and prevents black preview.
    """
    suffix = image_path.suffix.lower()
    mime = "image/png"
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def video_file_to_data_uri(video_path: Path, max_embed_mb: float = 80.0) -> Optional[str]:
    """
    Embed final generated video directly when it is not too large.
    This avoids Gradio /file route issues that can appear as a black 0:00 player.
    """
    try:
        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > max_embed_mb:
            return None
        data = base64.b64encode(video_path.read_bytes()).decode("utf-8")
        return f"data:video/mp4;base64,{data}"
    except Exception:
        return None


def image_panel_html(
    image_path: Path,
    label: str = "Uploaded image",
    *,
    thinking: bool = False,
) -> str:
    """
    Right-side avatar idle/thinking state.
    The uploaded avatar image is rendered inside the digital-human panel, not inside the uploader.
    """
    src = image_file_to_data_uri(image_path)

    overlay = ""
    if thinking:
        overlay = """
        <div class="thinking-overlay">
            <div class="thinking-dot"></div>
            <div>Thinking...</div>
        </div>
        """

    return f"""
    <div id="avatar_html">
        <div class="avatar-stage">
            <img src="{src}" alt="avatar image" />
            <div class="avatar-badge">{label}</div>
            {overlay}
        </div>
    </div>
    """


def _idle_inner_html_for_js(idle_src: str, idle_kind: str) -> str:
    """
    HTML string inserted by final-answer video onended.
    """
    if idle_kind == "image":
        # idle_src is already a data URI for uploaded image.
        return (
            f'<img src="{idle_src}" alt="avatar image" />'
            f'<div class="avatar-badge">Idle</div>'
        )

    return (
        f'<video src="{idle_src}?t={time.time()}" autoplay loop muted playsinline controls preload="auto"></video>'
        f'<div class="avatar-badge">Idle</div>'
    )


def video_panel_html(
    path: Optional[Path],
    *,
    label: str = "Avatar",
    loop: bool = True,
    muted: bool = True,
    idle_fallback_path: Optional[Path] = None,
    idle_fallback_is_image: bool = False,
    embed_video: bool = False,
) -> str:
    """
    Right-side avatar video state.
    For answer video, when playback ends it automatically returns to idle in the same position.
    """
    if not path or not path.exists():
        path = get_idle_video_path()

    if not path or not path.exists():
        return """
        <div id="avatar_html">
            <div class="avatar-stage" style="display:flex;align-items:center;justify-content:center;color:white;font-size:16px;">
                No preview available
            </div>
        </div>
        """

    if embed_video:
        src = video_file_to_data_uri(path) or file_to_gradio_url(path)
    else:
        src = file_to_gradio_url(path)

    loop_attr = "loop" if loop else ""
    muted_attr = "muted" if muted else ""

    idle_src = ""
    idle_kind = ""
    if idle_fallback_path and idle_fallback_path.exists():
        if idle_fallback_is_image:
            idle_src = image_file_to_data_uri(idle_fallback_path)
            idle_kind = "image"
        else:
            idle_src = file_to_gradio_url(idle_fallback_path)
            idle_kind = "video"

    onended = ""
    if idle_src and not loop:
        idle_html = _idle_inner_html_for_js(idle_src, idle_kind).replace("`", "\\`")
        onended = f"""
        onended="
            const box = this.closest('.avatar-stage');
            if (box) {{
                box.innerHTML = `{idle_html}`;
            }}
        "
        """

    video_src = src if src.startswith("data:") else f"{src}?t={time.time()}"

    return f"""
    <div id="avatar_html">
        <div class="avatar-stage">
            <video src="{video_src}" autoplay {loop_attr} {muted_attr} playsinline controls preload="auto" {onended}></video>
            <div class="avatar-badge">{label}</div>
        </div>
    </div>
    """


def render_avatar_panel(
    source_image_file: Optional[str],
    reply_mode: str,
    stage: str = "idle",
    final_video: Optional[Path] = None,
) -> str:
    """
    Right-side digital human state machine only:
    idle -> thinking -> speaking -> return idle after video ends.
    The left/user side is not changed.
    """
    source_path = None
    if source_image_file:
        p = Path(source_image_file)
        if p.exists():
            source_path = p

    # speaking state: final generated video replaces the same right-side area.
    # after it ends, it returns to the uploaded image or idle video.
    if final_video and final_video.exists():
        if source_path:
            return video_panel_html(
                final_video,
                label="Speaking",
                loop=False,
                muted=False,
                idle_fallback_path=source_path,
                idle_fallback_is_image=True,
                embed_video=True,
            )

        idle_video = get_idle_video_path()
        return video_panel_html(
            final_video,
            label="Speaking",
            loop=False,
            muted=False,
            idle_fallback_path=idle_video,
            idle_fallback_is_image=False,
            embed_video=True,
        )

    # uploaded image persists in idle and thinking states.
    if source_path:
        if stage == "thinking":
            return image_panel_html(source_path, "Thinking...", thinking=True)
        if stage == "voice":
            return image_panel_html(source_path, "Voice Reply", thinking=False)
        return image_panel_html(source_path, "Idle", thinking=False)

    # no uploaded image: fallback to idle/thinking videos.
    if stage == "thinking":
        return video_panel_html(get_thinking_video_path(), label="Thinking...", loop=True, muted=True)
    return video_panel_html(get_idle_video_path(), label="Idle", loop=True, muted=True)


def chatbot_from_history(history: List[Dict[str, str]]):
    return history


def get_recent_messages(history: List[Dict[str, str]], max_turns: int = 6):
    if not history:
        return []
    return history[-max_turns * 2:]


def clean_llm_reply(text: str) -> str:
    text = (text or "").strip()
    bad_prefixes = ["assistant:", "Assistant:", "AI:", "Answer:"]
    for p in bad_prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(parts) > 1:
        korean_parts = [p for p in parts if contains_hangul(p)]
        text = korean_parts[-1].strip() if korean_parts else parts[-1].strip()
    return text


def auto_select_tts_tags(reply_text: str) -> str:
    t = (reply_text or "").strip()
    if not t:
        return "<|emotion:contentment|><|prosody:expressive_high|>"

    if any(x in t for x in ["ㅋㅋ", "하하", "재밌", "웃", "funny"]):
        return "<|emotion:amusement|><|prosody:expressive_high|>"
    if any(x in t for x in ["와", "정말", "대박", "놀라", "우와"]):
        return "<|emotion:surprise|><|prosody:pitch_high|><|prosody:expressive_high|>"
    if any(x in t for x in ["미안", "죄송", "아쉽", "슬프"]):
        return "<|emotion:relief|><|prosody:speed_slow|>"
    if any(x in t for x in ["방법", "설명", "예를", "먼저", "다음", "정리"]):
        return "<|emotion:contemplation|><|prosody:speed_slow|>"
    if any(x in t for x in ["꼭", "할 수", "좋아요", "추천", "괜찮"]):
        return "<|emotion:determination|><|prosody:expressive_high|>"

    return "<|emotion:contentment|><|prosody:expressive_high|>"


# ============================================================
# LLM / TTS
# ============================================================
def local_fallback_reply(user_text: str) -> str:
    if "hello" in user_text.lower() or "안녕" in user_text:
        return "Hello. How can I help you today?"
    return "Sure. I will help based on what you shared."


def call_llm(user_text: str, chat_history: List[Dict[str, str]], system_prompt: str) -> str:
    system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    if not LLM_BASE_URL or not LLM_MODEL:
        return local_fallback_reply(user_text)

    url = f"{LLM_BASE_URL}/chat/completions"
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_recent_messages(chat_history, max_turns=6))
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"LLM API failed: {r.status_code}\n{r.text}")
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        text = clean_llm_reply(text)
        return text or local_fallback_reply(user_text)
    except Exception as e:
        print("LLM failed, using fallback reply.")
        print(e)
        return local_fallback_reply(user_text)


def boson_tts(text: str, out_mp3: Path) -> Path:
    check_boson_key()
    payload = {
        "model": "higgs-audio-v3-tts",
        "input": text,
        "voice": BOSON_VOICE or "default",
        "response_format": "mp3",
    }
    headers = {
        "Authorization": f"Bearer {BOSON_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }
    r = requests.post("https://api.boson.ai/v1/audio/speech", json=payload, headers=headers, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Boson TTS failed:\n{r.status_code}\n{r.text}")
    out_mp3.write_bytes(r.content)
    return out_mp3


def build_tts_input(reply_text: str) -> str:
    return f"{auto_select_tts_tags(reply_text)}{reply_text.strip()}"


def mp3_to_wav(mp3_path: Path, wav_path: Path) -> Path:
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-i",
        str(mp3_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    run_cmd(cmd)
    return wav_path


def get_media_duration_sec(media_path: Path) -> float:
    try:
        result = subprocess.check_output([
            "/usr/bin/ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ], text=True).strip()
        return max(float(result), 0.0)
    except Exception:
        return 3.0


# ============================================================
# MuseTalk
# ============================================================
def image_to_short_video(image_path: Path, out_video: Path, seconds: float = 3.0, fps: int = 25) -> Path:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    safe_seconds = max(3.0, float(seconds))
    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-t", f"{safe_seconds:.2f}",
        "-r", str(fps),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ]
    run_cmd(cmd)
    return out_video


def write_musetalk_config(video_or_image_path: Path, audio_path: Path, config_path: Path) -> None:
    data = {"task_0": {"video_path": str(video_or_image_path), "audio_path": str(audio_path)}}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def find_output_mp4(result_dir: Path) -> Path:
    mp4s = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise FileNotFoundError(f"No MuseTalk output mp4 was found in {result_dir}")
    non_temp = [p for p in mp4s if "temp" not in p.name.lower() and "tmp" not in p.name.lower()]
    return non_temp[0] if non_temp else mp4s[0]


def prepare_musetalk_inputs(source_image_file: Optional[str], wav_path: Path, job_id: str) -> Tuple[Path, Path, Path]:
    if not MUSETALK_DIR.exists():
        raise FileNotFoundError(f"MuseTalk directory was not found: {MUSETALK_DIR}")

    data_dir = MUSETALK_DIR / "data" / f"text_live_{job_id}"
    data_dir.mkdir(parents=True, exist_ok=True)

    if source_image_file:
        src = Path(source_image_file)
        suffix = src.suffix.lower()
        if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"]:
            raise ValueError(f"Unsupported source file type: {suffix}")
        if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            local_image = data_dir / f"source{suffix}"
            shutil.copy(src, local_image)
            local_visual = data_dir / "source_as_video.mp4"
            duration = get_media_duration_sec(wav_path) + 0.6
            image_to_short_video(local_image, local_visual, seconds=duration, fps=25)
        else:
            local_visual = data_dir / f"source_video{suffix}"
            shutil.copy(src, local_visual)
    else:
        default_motion = find_default_motion_video()
        local_visual = data_dir / "default_motion.mp4"
        shutil.copy(default_motion, local_visual)

    local_wav = data_dir / "reply.wav"
    shutil.copy(wav_path, local_wav)

    rel_visual = local_visual.relative_to(MUSETALK_DIR)
    rel_audio = local_wav.relative_to(MUSETALK_DIR)
    result_dir = MUSETALK_DIR / "results" / f"text_live_{job_id}"
    return rel_visual, rel_audio, result_dir


def run_musetalk(source_image_file: Optional[str], wav_path: Path, job_id: str) -> Path:
    rel_visual, rel_audio, result_dir = prepare_musetalk_inputs(source_image_file, wav_path, job_id)
    config_path = MUSETALK_DIR / "configs" / "inference" / f"text_live_{job_id}.yaml"
    write_musetalk_config(rel_visual, rel_audio, config_path)

    env = os.environ.copy()
    env["PATH"] = f"/usr/bin:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = MUSETALK_CUDA_VISIBLE_DEVICES
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "4"

    cmd = [
        sys.executable,
        "-m", "scripts.inference",
        "--inference_config", str(config_path.relative_to(MUSETALK_DIR)),
        "--result_dir", str(result_dir.relative_to(MUSETALK_DIR)),
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15",
    ]

    try:
        run_cmd(cmd, cwd=MUSETALK_DIR, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print("MuseTalk command returned non-zero exit code, trying recovery...")
        try:
            return find_output_mp4(result_dir)
        except Exception:
            raise e

    return find_output_mp4(result_dir)



# ============================================================
# MuseTalk Worker Client
# ============================================================

def call_musetalk_worker(
    source_image_file: Optional[str],
    wav_path: Path,
    job_id: str,
) -> Path:
    """
    Call persistent MuseTalk worker instead of running MuseTalk inside Gradio.
    This keeps the right-side UI responsive and centralizes MuseTalk GPU jobs.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file was not found: {wav_path}")

    payload = {
        "job_id": job_id,
        "wav_path": str(wav_path),
        "source_image_path": str(source_image_file) if source_image_file else None,
    }

    url = f"{MUSETALK_WORKER_URL}/generate"

    r = requests.post(
        url,
        json=payload,
        timeout=3600,
    )

    if r.status_code != 200:
        raise RuntimeError(f"MuseTalk Worker failed: {r.status_code}\n{r.text}")

    data = r.json()
    video_path = Path(data["video_path"])

    if not video_path.exists():
        raise FileNotFoundError(f"The video returned by the worker does not exist: {video_path}")

    return video_path


# ============================================================
# UI callbacks
# ============================================================
def refresh_right_panel(reply_mode: str, source_image_file: Optional[str]):
    return render_avatar_panel(source_image_file, reply_mode, stage="idle")


def clear_chat(reply_mode: str, source_image_file: Optional[str]):
    return [], [], None, render_avatar_panel(source_image_file, reply_mode, stage="idle"), ""


def generate_reply_stream(
    user_text: str,
    chat_history: List[Dict[str, str]],
    reply_mode: str,
    source_image_file: Optional[str],
):
    user_text = (user_text or "").strip()
    if not user_text:
        raise gr.Error("Please enter a message.")
    if chat_history is None:
        chat_history = []

    job_id = make_job_id()
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_mp3 = job_dir / "reply.mp3"
    out_wav = job_dir / "reply.wav"

    pending_history = list(chat_history)
    pending_history.append({"role": "user", "content": user_text})
    pending_history.append({"role": "assistant", "content": "Thinking..."})

    initial_stage = "thinking" if reply_mode == "Video Mode" else "voice"
    yield (
        chat_history,
        chatbot_from_history(pending_history),
        None,
        render_avatar_panel(source_image_file, reply_mode, stage=initial_stage),
        "",
    )

    try:
        assistant_reply = call_llm(user_text, chat_history, DEFAULT_SYSTEM_PROMPT)
        current_history = list(chat_history)
        current_history.append({"role": "user", "content": user_text})
        current_history.append({"role": "assistant", "content": assistant_reply})

        yield (
            current_history,
            chatbot_from_history(current_history),
            None,
            render_avatar_panel(source_image_file, reply_mode, stage=initial_stage),
            "",
        )

        final_tts_input = build_tts_input(assistant_reply)
        boson_tts(final_tts_input, out_mp3)

        if reply_mode == "Voice Mode":
            yield (
                current_history,
                chatbot_from_history(current_history),
                str(out_mp3),
                render_avatar_panel(source_image_file, reply_mode, stage="voice"),
                "",
            )
            return

        mp3_to_wav(out_mp3, out_wav)
        generated_mp4 = call_musetalk_worker(source_image_file, out_wav, job_id)
        final_video = OUTPUT_DIR / f"text_live_{job_id}.mp4"
        shutil.copy(generated_mp4, final_video)

        if not final_video.exists() or final_video.stat().st_size < 50000:
            raise RuntimeError("Video generation failed or the output file is too small.")

        # Video mode: return only video, not separate audio, to avoid audio-video desynchronization.
        yield (
            current_history,
            chatbot_from_history(current_history),
            None,
            render_avatar_panel(source_image_file, reply_mode, final_video=final_video),
            "",
        )

    except Exception as e:
        err_msg = f"An error occurred: {e}"
        error_history = list(chat_history)
        error_history.append({"role": "user", "content": user_text})
        error_history.append({"role": "assistant", "content": err_msg})
        yield (
            error_history,
            chatbot_from_history(error_history),
            None,
            render_avatar_panel(source_image_file, reply_mode, stage="error"),
            "",
        )


# ============================================================
# Build UI
# ============================================================
with gr.Blocks(
    title="Digital Human Chat",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue", neutral_hue="slate"),
) as demo:
    chat_state = gr.State([])

    gr.HTML(
        '''
        <div id="app-title">
            <h1>Digital Human Chat</h1>
            <p>Voice Mode plays audio only. Video Mode plays only the digital human video on the right, without an extra audio track.</p>
        </div>
        '''
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["panel"]):
                chatbot = gr.Chatbot(label="Conversation", type="messages", elem_id="chatbot", value=[], show_copy_button=True)
                user_text = gr.Textbox(label="Message", placeholder="Type a message, for example: 안녕하세요", lines=3)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary", scale=4)
                    clear_btn = gr.Button("Clear", scale=1)
                reply_mode = gr.Radio(label="Mode", choices=REPLY_MODES, value="Voice Mode")
                gr.HTML('<div class="mode-tip">Voice Mode: keeps the avatar image on the right and auto-plays audio. Video Mode: the avatar thinks first, then the generated video appears in the same position.</div>')

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                source_image = gr.File(
                    label="Upload Avatar Image",
                    file_types=["image"],
                    file_count="single",
                    type="filepath",
                    elem_classes=["upload-compact"],
                )
                avatar_html = gr.HTML(value=render_avatar_panel(None, "Voice Mode", stage="idle"))
                audio_out = gr.Audio(label="Voice Mode Only", type="filepath", autoplay=True)
                hidden_msg = gr.Textbox(visible=False)

    send_btn.click(
        fn=generate_reply_stream,
        inputs=[user_text, chat_state, reply_mode, source_image],
        outputs=[chat_state, chatbot, audio_out, avatar_html, hidden_msg],
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[user_text],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[reply_mode, source_image],
        outputs=[chat_state, chatbot, audio_out, avatar_html, user_text],
    )

    source_image.change(
        fn=refresh_right_panel,
        inputs=[reply_mode, source_image],
        outputs=[avatar_html],
    )

    reply_mode.change(
        fn=refresh_right_panel,
        inputs=[reply_mode, source_image],
        outputs=[avatar_html],
    )


demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_SERVER_PORT,
        share=False,
        allowed_paths=[
            str(APP_DIR),
            str(MUSETALK_DIR),
            str(TEST2_DIR),
            "/tmp",
            "/mnt/data",
        ],
    )
