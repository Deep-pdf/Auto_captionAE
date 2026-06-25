import os
import sys
import threading
import traceback
import json
import re
import importlib

import whisper
import ffmpeg
import torch
import customtkinter as ctk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

try:
    sanscript = importlib.import_module("indic_transliteration.sanscript")
    transliterate = getattr(sanscript, "transliterate")
    HAS_INDIC_TRANSLITERATION = True
except Exception:
    sanscript = None
    transliterate = None
    HAS_INDIC_TRANSLITERATION = False


SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
CONFIG_FILE = "autocaption_config.json"
OUTPUT_SCRIPT_OPTIONS = ["Auto", "English (Transliteration)"]

# Keep fallback stream handles alive for GUI/frozen runs where stdout/stderr may be None.
_FALLBACK_STD_STREAMS = []


# ---------- Utility functions ----------

def seconds_to_srt(ts: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = int(ts // 3600)
    minutes = int((ts % 3600) // 60)
    seconds = int(ts % 60)
    milliseconds = int((ts - int(ts)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_srt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt(seg["start"])
            end = seconds_to_srt(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def extract_audio(video_path: str) -> str:
    """Use ffmpeg to extract a WAV that Whisper can ingest."""
    base = os.path.splitext(video_path)[0]
    audio_path = f"{base}_temp.wav"
    # overwrite if exists
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar="16k", format="wav").run(overwrite_output=True)
    except FileNotFoundError as fe:
        # re-raise so caller can show a friendly message
        raise RuntimeError("FFmpeg executable not found. Please install FFmpeg and add it to your PATH.")
    return audio_path


def get_media_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe metadata."""
    try:
        info = ffmpeg.probe(path)
        duration = info.get("format", {}).get("duration")
        if duration is None:
            return 0.0
        return max(0.0, float(duration))
    except Exception:
        return 0.0


def extract_audio_segment(audio_path: str, start_sec: float, duration_sec: float, segment_path: str):
    """Extract a WAV segment from an existing WAV file."""
    ffmpeg.input(audio_path, ss=max(0.0, start_sec), t=max(0.0, duration_sec)).output(
        segment_path,
        ac=1,
        ar="16k",
        format="wav",
    ).run(overwrite_output=True)


def ensure_standard_streams():
    """
    Whisper writes progress to stderr during model download/loading.
    In some Windows GUI/frozen launches sys.stdout/sys.stderr may be None.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "write"):
            fallback = open(os.devnull, "w", encoding="utf-8")
            setattr(sys, stream_name, fallback)
            _FALLBACK_STD_STREAMS.append(fallback)


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_config(config: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def normalize_compute_choice(choice: str, gpu_available: bool) -> str:
    choice = (choice or "").strip().lower()
    if choice == "gpu" and gpu_available:
        return "GPU"
    if choice == "cpu":
        return "CPU"
    return "Auto"


def resolve_device(selection: str, gpu_available: bool) -> str:
    if selection == "GPU" and gpu_available:
        return "cuda"
    return "cpu"


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def transliterate_segment_text(text: str, output_script: str) -> str:
    if output_script != "English (Transliteration)":
        return text
    if not text.strip():
        return text
    if not contains_devanagari(text):
        return text
    if not HAS_INDIC_TRANSLITERATION:
        raise RuntimeError(
            "English transliteration requires 'indic-transliteration'. "
            "Install it with: pip install indic-transliteration"
        )
    return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


# ---------- Application GUI ----------

class AutoCaptionApp:
    def __init__(self):
        ensure_standard_streams()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.config = load_config()
        self.gpu_available = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else None
        self.compute_choice = normalize_compute_choice(self.config.get("compute_device", "Auto"), self.gpu_available)
        self.compute_device = resolve_device(self.compute_choice, self.gpu_available)

        # check ffmpeg availability early
        if not self._ffmpeg_available():
            messagebox.showerror("Missing FFmpeg", "FFmpeg not found in PATH. Please install FFmpeg and restart the app.")

        self.root = TkinterDnD.Tk()
        self.root.title("AutoCaption Studio")
        self.root.geometry("600x460")
        self.root.resizable(False, False)

        self.video_path = None

        self._build_widgets()

    def _ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available on the system PATH."""
        try:
            # run ffmpeg -version quietly
            import subprocess
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False

    def _build_widgets(self):
        frame = ctk.CTkFrame(self.root, corner_radius=8)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # drag/drop area
        self.drop_label = ctk.CTkLabel(
            frame,
            text="Drag video file here",
            width=400,
            height=150,
            fg_color=("#444", "#222"),
            justify="center",
            corner_radius=8,
        )
        self.drop_label.pack(pady=10)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        # model selector
        models = ["tiny", "base", "small", "medium", "large"]
        saved_model = self.config.get("model", "base")
        if saved_model not in models:
            saved_model = "base"
        self.model_var = ctk.StringVar(value=saved_model)
        model_frame = ctk.CTkFrame(frame)
        model_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(model_frame, text="Model:").pack(side="left", padx=(0, 5))
        self.model_menu = ctk.CTkOptionMenu(model_frame, values=models, variable=self.model_var)
        self.model_menu.pack(side="left")
        self.model_var.trace_add("write", self._on_model_change)

        # compute device selector
        compute_values = ["Auto", "CPU"]
        if self.gpu_available:
            compute_values.append("GPU")
        saved_compute_choice = normalize_compute_choice(self.config.get("compute_device", "Auto"), self.gpu_available)
        self.compute_var = ctk.StringVar(value=saved_compute_choice)
        ctk.CTkLabel(model_frame, text="Compute:").pack(side="left", padx=(15, 5))
        self.compute_menu = ctk.CTkOptionMenu(
            model_frame,
            values=compute_values,
            variable=self.compute_var,
            command=self._on_compute_change,
        )
        self.compute_menu.pack(side="left")

        # output script selector
        saved_output_script = self.config.get("output_script", "Auto")
        if saved_output_script not in OUTPUT_SCRIPT_OPTIONS:
            saved_output_script = "Auto"
        self.output_script_var = ctk.StringVar(value=saved_output_script)
        ctk.CTkLabel(model_frame, text="Output Script:").pack(side="left", padx=(15, 5))
        self.output_script_menu = ctk.CTkOptionMenu(
            model_frame,
            values=OUTPUT_SCRIPT_OPTIONS,
            variable=self.output_script_var,
            command=self._on_output_script_change,
        )
        self.output_script_menu.pack(side="left")

        # compute device indicator
        self.device_label = ctk.CTkLabel(frame, text=self._get_device_text(), text_color="#9ad0ff")
        self.device_label.pack(pady=(5, 10))

        # start button
        self.start_button = ctk.CTkButton(frame, text="Generate Captions", state="disabled", command=self._on_start)
        self.start_button.pack(pady=15)

        # progress
        self.status_label = ctk.CTkLabel(frame, text="Waiting for video...", text_color="#BBB")
        self.status_label.pack(pady=(10, 5))
        self.progress = ctk.CTkProgressBar(frame, width=400, height=25)
        self.progress.set(0)
        self.progress.pack(pady=(0, 10))
        
        # percentage label
        self.percent_label = ctk.CTkLabel(frame, text="0%", text_color="#FFF", font=("Arial", 12, "bold"))
        self.percent_label.pack()

    def _on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        if not files:
            return
        path = files[0]
        _, ext = os.path.splitext(path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            messagebox.showerror("Invalid file", "Please drop a supported video file (mp4, mov, mkv, avi).")
            return
        self.video_path = path
        self.drop_label.configure(text=os.path.basename(path))
        self.start_button.configure(state="normal")
        self.status_label.configure(text="Ready to generate captions")

    def _on_start(self):
        if not self.video_path:
            return
        # verify ffmpeg again in case PATH changed
        if not self._ffmpeg_available():
            messagebox.showerror("Missing FFmpeg", "FFmpeg not found in PATH. Please install FFmpeg and restart the app.")
            return
        self.start_button.configure(state="disabled")
        self.model_menu.configure(state="disabled")
        self.output_script_menu.configure(state="disabled")
        threading.Thread(target=self._process_file, daemon=True).start()

    def _on_output_script_change(self, selection):
        self.config["output_script"] = selection
        save_config(self.config)

    def _on_model_change(self, *_):
        self.config["model"] = self.model_var.get()
        save_config(self.config)

    def _on_compute_change(self, selection):
        self.compute_choice = normalize_compute_choice(selection, self.gpu_available)
        self.compute_device = resolve_device(self.compute_choice, self.gpu_available)
        self.config["compute_device"] = self.compute_choice
        save_config(self.config)
        self._update_device_label()

    def _get_device_text(self):
        if self.compute_choice == "Auto":
            if self.gpu_available:
                return f"Compute: Auto → GPU ({self.gpu_name})"
            return "Compute: Auto → CPU (no GPU detected)"
        if self.compute_choice == "GPU":
            if self.gpu_available:
                return f"Compute: GPU ({self.gpu_name})"
            return "Compute: GPU selected but not available, using CPU"
        return "Compute: CPU"

    def _update_device_label(self):
        self.device_label.configure(text=self._get_device_text())

    def _process_file(self):
        audio_path = None
        segment_audio_paths = []
        try:
            ensure_standard_streams()
            self._update_status("Loading model... (10%)", 0.1)
            self.compute_choice = normalize_compute_choice(self.compute_var.get(), self.gpu_available)
            self.compute_device = resolve_device(self.compute_choice, self.gpu_available)
            self._update_device_label()
            model_name = self.model_var.get()
            model = whisper.load_model(model_name, device=self.compute_device)
            selected_output_script = self.output_script_var.get()

            # whisper can handle video directly, but extract audio for reliability
            self._update_status("Extracting audio... (25%)", 0.25)
            audio_path = extract_audio(self.video_path)
            total_duration = get_media_duration(audio_path)
            if total_duration <= 0:
                total_duration = get_media_duration(self.video_path)

            self._update_status("Transcribing full audio... (40%)", 0.4)
            chunk_duration = 300.0
            min_tail_to_retry = 0.75
            segments = []
            current_start = 0.0
            base_progress = 0.4
            transcribe_progress_span = 0.5

            # Process all chunks from 0s to media end so silence does not stop transcription.
            while current_start < total_duration - 0.001:
                remaining = total_duration - current_start
                current_chunk_duration = min(chunk_duration, remaining)
                segment_audio_path = f"{os.path.splitext(audio_path)[0]}_chunk_{int(current_start * 1000)}.wav"
                extract_audio_segment(audio_path, current_start, current_chunk_duration, segment_audio_path)
                segment_audio_paths.append(segment_audio_path)

                result = model.transcribe(
                    segment_audio_path,
                    task="transcribe",
                    condition_on_previous_text=False,
                )
                for seg in result.get("segments", []):
                    seg_start = max(current_start, current_start + float(seg["start"]))
                    seg_end = min(total_duration, current_start + float(seg["end"]))
                    if seg_end > seg_start:
                        segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": transliterate_segment_text(seg["text"], selected_output_script),
                        })

                current_start += current_chunk_duration
                processed_ratio = min(1.0, current_start / total_duration) if total_duration > 0 else 1.0
                self._update_status(
                    f"Transcribing full audio... ({int((base_progress + transcribe_progress_span * processed_ratio) * 100)}%)",
                    base_progress + transcribe_progress_span * processed_ratio,
                )

            # Validate subtitle timeline vs input duration and auto-reprocess uncovered tail.
            last_end = max((seg["end"] for seg in segments), default=0.0)
            tail_gap = max(0.0, total_duration - last_end)
            if total_duration > 0 and tail_gap >= min_tail_to_retry:
                tail_start = max(0.0, last_end - 1.0)
                retry_duration = total_duration - tail_start
                retry_segment_audio_path = f"{os.path.splitext(audio_path)[0]}_tail_retry.wav"
                extract_audio_segment(audio_path, tail_start, retry_duration, retry_segment_audio_path)
                segment_audio_paths.append(retry_segment_audio_path)

                tail_result = model.transcribe(
                    retry_segment_audio_path,
                    task="transcribe",
                    condition_on_previous_text=False,
                )
                for seg in tail_result.get("segments", []):
                    seg_start = max(tail_start, tail_start + float(seg["start"]))
                    seg_end = min(total_duration, tail_start + float(seg["end"]))
                    if seg_end > seg_start:
                        segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": transliterate_segment_text(seg["text"], selected_output_script),
                        })

                last_end = max((seg["end"] for seg in segments), default=0.0)
                if total_duration - last_end >= min_tail_to_retry:
                    # Keep subtitle timeline aligned to full video duration when tail is silent.
                    segments.append({
                        "start": max(0.0, last_end),
                        "end": total_duration,
                        "text": "",
                    })

            segments.sort(key=lambda s: (s["start"], s["end"]))
            self._update_status("Generating captions... (95%)", 0.95)

            out_srt = os.path.splitext(self.video_path)[0] + ".srt"
            write_srt(segments, out_srt)

            self._update_status("Completed (100%)", 1.0)
            messagebox.showinfo("Done", f"Caption file created!\n\nLocation:\n{out_srt}\n\nFile size: {os.path.getsize(out_srt)} bytes")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self._update_status("Error", 0)
        finally:
            self.start_button.configure(state="normal")
            self.model_menu.configure(state="normal")
            self.output_script_menu.configure(state="normal")
            # remove temporary audio file if exists
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                for segment_audio_path in segment_audio_paths:
                    if os.path.exists(segment_audio_path):
                        os.remove(segment_audio_path)
            except Exception:
                pass

    def _update_status(self, message, progress_value=None):
        self.status_label.configure(text=message)
        if progress_value is not None:
            self.progress.set(progress_value)
            percent = int(progress_value * 100)
            self.percent_label.configure(text=f"{percent}%")
            self.root.update_idletasks()  # refresh UI immediately

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AutoCaptionApp()
    app.run()
