import os
import sys
import time
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv


# ============================================================
# MuseTalk Persistent Worker API
# ============================================================
# This worker stays alive on a local port and serializes MuseTalk GPU jobs.
# The Gradio app calls this worker instead of running MuseTalk directly.
#
# Note:
# - This is the stable worker version: it keeps a persistent service and queue.
# - It still uses MuseTalk's official scripts.inference command per request for maximum compatibility.
# - The next optimization is to replace run_musetalk_subprocess() with direct model-preloaded inference
#   based on your exact MuseTalk/scripts/realtime_inference.py.
# ============================================================

APP_DIR = Path(os.getenv("DIGITAL_HUMAN_APP_DIR", "/workspace/digital_human_demo"))
ENV_PATH = APP_DIR / ".env"
load_dotenv(ENV_PATH)

MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", "/workspace/MuseTalk"))
TEST2_DIR = Path(os.getenv("TEST2_DIR", "/workspace/digital_human_demo/data/test2"))
DEFAULT_MOTION_VIDEO = Path(
    os.getenv(
        "DEFAULT_MOTION_VIDEO",
        "/workspace/digital_human_demo/data/test2/avatar--test2_25fps.mp4",
    )
)

MUSETALK_CUDA_VISIBLE_DEVICES = os.getenv("MUSETALK_CUDA_VISIBLE_DEVICES", "1").strip()
MUSETALK_WORKER_HOST = os.getenv("MUSETALK_WORKER_HOST", "127.0.0.1")
MUSETALK_WORKER_PORT = int(os.getenv("MUSETALK_WORKER_PORT", "8890"))

WORKER_DATA_DIR = MUSETALK_DIR / "data" / "worker_text_live"
WORKER_RESULT_DIR = MUSETALK_DIR / "results" / "worker_text_live"
WORKER_OUTPUT_DIR = APP_DIR / "worker_outputs"

WORKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
WORKER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
WORKER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gpu_lock = threading.Lock()
app = FastAPI(title="MuseTalk Persistent Worker")


class GenerateRequest(BaseModel):
    job_id: str
    wav_path: str
    source_image_path: Optional[str] = None


class GenerateResponse(BaseModel):
    ok: bool
    video_path: str
    raw_video_path: str
    elapsed_sec: float


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    print("\n========== WORKER RUN CMD ==========")
    print(" ".join(cmd))
    if cwd:
        print("CWD:", cwd)
    print("====================================\n")

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

    raise FileNotFoundError(
        f"没有找到默认 motion video，请检查 DEFAULT_MOTION_VIDEO 或 {TEST2_DIR}"
    )


def get_media_duration_sec(media_path: Path) -> float:
    try:
        result = subprocess.check_output(
            [
                "/usr/bin/ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            text=True,
        ).strip()
        return max(float(result), 0.0)
    except Exception:
        return 3.0


def image_to_short_video(
    image_path: Path,
    out_video: Path,
    seconds: float,
    fps: int = 25,
) -> Path:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    safe_seconds = max(3.0, float(seconds))

    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{safe_seconds:.2f}",
        "-r",
        str(fps),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-pix_fmt",
        "yuv420p",
        str(out_video),
    ]

    run_cmd(cmd)
    return out_video


def write_musetalk_config(
    video_or_image_path: Path,
    audio_path: Path,
    config_path: Path,
) -> None:
    data = {
        "task_0": {
            "video_path": str(video_or_image_path),
            "audio_path": str(audio_path),
        }
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def find_output_mp4(result_dir: Path) -> Path:
    """
    Prefer the real final MuseTalk result video instead of temp/intermediate mp4.
    MuseTalk usually writes:
      .../v15/source_reply.mp4
      .../v15/temp_source_reply.mp4
    The temp video can be silent/intermediate and sometimes renders black in browser.
    """
    mp4s = sorted(
        result_dir.rglob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not mp4s:
        raise FileNotFoundError(f"没有在 {result_dir} 找到 MuseTalk 输出 mp4")

    def good_size(p: Path) -> bool:
        try:
            return p.stat().st_size > 50_000
        except Exception:
            return False

    # 1. Best: final reply video, not temp.
    preferred = [
        p for p in mp4s
        if good_size(p)
        and "temp" not in p.name.lower()
        and "tmp" not in p.name.lower()
        and p.name.lower().endswith("_reply.mp4")
    ]
    if preferred:
        return preferred[0]

    # 2. Any non-temp mp4 with reasonable size.
    non_temp = [
        p for p in mp4s
        if good_size(p)
        and "temp" not in p.name.lower()
        and "tmp" not in p.name.lower()
    ]
    if non_temp:
        return non_temp[0]

    # 3. Last fallback.
    return mp4s[0]


def make_browser_playable_video(src_mp4: Path, dst_mp4: Path) -> Path:
    """
    Re-encode MuseTalk output to a browser-stable MP4 while keeping the audio track.
    Video mode uses only this final video, so audio must stay inside the mp4.
    """
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-i",
        str(src_mp4),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(dst_mp4),
    ]

    run_cmd(cmd)

    if not dst_mp4.exists() or dst_mp4.stat().st_size < 50_000:
        raise RuntimeError(f"浏览器兼容视频生成失败或文件过小: {dst_mp4}")

    # Check duration is readable.
    duration = get_media_duration_sec(dst_mp4)
    if duration <= 0.1:
        raise RuntimeError(f"浏览器兼容视频时长异常: {duration}, file={dst_mp4}")

    return dst_mp4


def prepare_inputs(
    source_image_path: Optional[str],
    wav_path: Path,
    job_id: str,
) -> Tuple[Path, Path, Path]:
    if not MUSETALK_DIR.exists():
        raise FileNotFoundError(f"找不到 MuseTalk 目录: {MUSETALK_DIR}")

    if not wav_path.exists():
        raise FileNotFoundError(f"找不到 wav 文件: {wav_path}")

    data_dir = WORKER_DATA_DIR / job_id
    data_dir.mkdir(parents=True, exist_ok=True)

    if source_image_path:
        src = Path(source_image_path)
        if not src.exists():
            raise FileNotFoundError(f"找不到上传图片/视频: {src}")

        suffix = src.suffix.lower()
        if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"]:
            raise ValueError(f"不支持的 Source 文件类型: {suffix}")

        if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            local_image = data_dir / f"source{suffix}"
            shutil.copy(src, local_image)

            local_visual = data_dir / "source_as_video.mp4"
            duration = get_media_duration_sec(wav_path) + 0.6
            image_to_short_video(
                image_path=local_image,
                out_video=local_visual,
                seconds=duration,
                fps=25,
            )
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
    result_dir = WORKER_RESULT_DIR / job_id

    return rel_visual, rel_audio, result_dir


def run_musetalk_subprocess(
    source_image_path: Optional[str],
    wav_path: Path,
    job_id: str,
) -> Path:
    rel_visual, rel_audio, result_dir = prepare_inputs(
        source_image_path=source_image_path,
        wav_path=wav_path,
        job_id=job_id,
    )

    config_path = MUSETALK_DIR / "configs" / "inference" / f"worker_text_live_{job_id}.yaml"

    write_musetalk_config(
        video_or_image_path=rel_visual,
        audio_path=rel_audio,
        config_path=config_path,
    )

    env = os.environ.copy()
    env["PATH"] = f"/usr/bin:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = MUSETALK_CUDA_VISIBLE_DEVICES
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "4"

    cmd = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path.relative_to(MUSETALK_DIR)),
        "--result_dir",
        str(result_dir.relative_to(MUSETALK_DIR)),
        "--unet_model_path",
        "models/musetalkV15/unet.pth",
        "--unet_config",
        "models/musetalkV15/musetalk.json",
        "--version",
        "v15",
    ]

    try:
        run_cmd(cmd, cwd=MUSETALK_DIR, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print("MuseTalk returned non-zero exit code. Trying to recover mp4...")
        try:
            return find_output_mp4(result_dir)
        except Exception:
            raise e

    return find_output_mp4(result_dir)


@app.get("/health")
def health():
    return {
        "ok": True,
        "musetalk_dir": str(MUSETALK_DIR),
        "gpu": MUSETALK_CUDA_VISIBLE_DEVICES,
        "worker_data_dir": str(WORKER_DATA_DIR),
        "worker_output_dir": str(WORKER_OUTPUT_DIR),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    start = time.time()

    try:
        with gpu_lock:
            raw_mp4 = run_musetalk_subprocess(
                source_image_path=req.source_image_path,
                wav_path=Path(req.wav_path),
                job_id=req.job_id,
            )

            final_video = WORKER_OUTPUT_DIR / f"text_live_{req.job_id}_browser.mp4"
            make_browser_playable_video(raw_mp4, final_video)

            if not final_video.exists() or final_video.stat().st_size < 50000:
                raise RuntimeError(f"视频文件生成失败或文件过小: {final_video}")

            return GenerateResponse(
                ok=True,
                video_path=str(final_video),
                raw_video_path=str(raw_mp4),
                elapsed_sec=time.time() - start,
            )

    except Exception as e:
        print("Worker generate failed:", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting MuseTalk worker...")
    print("MUSETALK_DIR:", MUSETALK_DIR)
    print("MUSETALK_CUDA_VISIBLE_DEVICES:", MUSETALK_CUDA_VISIBLE_DEVICES)
    print("Worker URL:", f"http://{MUSETALK_WORKER_HOST}:{MUSETALK_WORKER_PORT}")
    uvicorn.run(
        app,
        host=MUSETALK_WORKER_HOST,
        port=MUSETALK_WORKER_PORT,
        log_level="info",
    )
