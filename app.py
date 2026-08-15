import os
import sys
import threading
import traceback
import json
import re
import importlib

try:
    import whisper
except Exception:
    whisper = None

try:
    import ffmpeg
except Exception:
    ffmpeg = None

try:
    import torch
except Exception:
    torch = None

try:
    import customtkinter as ctk
except Exception:
    ctk = None

from tkinter import Tk, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None

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
OUTPUT_SCRIPT_OPTIONS = ["Auto", "Hinglish", "English (Transliteration)"]

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


def wrap_text_for_srt(text: str, words_per_line: int) -> str:
    cleaned_text = " ".join((text or "").split())
    if not cleaned_text or words_per_line is None or words_per_line <= 0:
        return cleaned_text

    words = cleaned_text.split()
    if len(words) <= words_per_line:
        return cleaned_text

    wrapped_lines = []
    for index in range(0, len(words), words_per_line):
        wrapped_lines.append(" ".join(words[index:index + words_per_line]))
    return "\n".join(wrapped_lines)


def write_srt(segments, out_path, words_per_line: int = 0):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt(seg["start"])
            end = seconds_to_srt(seg["end"])
            text = wrap_text_for_srt(seg.get("text", ""), words_per_line)
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def extract_audio(video_path: str) -> str:
    """Use ffmpeg to extract a WAV that Whisper can ingest."""
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is not available. Please install it and restart the app.")

    base = os.path.splitext(video_path)[0]
    audio_path = f"{base}_temp.wav"
    # overwrite if exists
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar="16k", format="wav").run(overwrite_output=True)
    except FileNotFoundError:
        # re-raise so caller can show a friendly message
        raise RuntimeError("FFmpeg executable not found. Please install FFmpeg and add it to your PATH.")
    return audio_path


def get_media_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe metadata."""
    if ffmpeg is None:
        return 0.0

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
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is not available. Please install it and restart the app.")

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


HINGLISH_WORD_OVERRIDES = {
    "हम": "hum",
    "सब": "sab",
    "को": "ko",
    "एक": "ek",
    "साथ": "sath",
    "मैं": "main",
    "तुम": "tum",
    "आप": "aap",
    "आज": "aaj",
    "कभी": "kabhi",
    "यह": "yeh",
    "वह": "woh",
    "कुछ": "kuch",
    "है": "hai",
    "नहीं": "nahi",
    "हाँ": "haan",
    "बिल्कुल": "bilkul",
    "अच्छा": "accha",
    "शुक्रिया": "shukriya",
    "धन्यवाद": "dhanyavaad",
    "दोस्त": "dost",
    "किसी": "kisi",
    "नमस्ते": "namaste",
    "भाई": "bhai",
    "बहुत": "bahut",
    "सिर्फ": "sirf",
    "पर": "par",
    "मुझे": "mujhe",
    "हमें": "hamein",
    "तुझे": "tuje",
    "अपना": "apna",
    "सकते": "sakte",
    "कर": "kar",
    "लगे": "lage",
    "क्यों": "kyun",
    "क्योंकि": "kyunki",
    "लोग": "log",
}

DEVANAGARI_TO_HINGLISH = {
    "अ": "a",
    "आ": "a",
    "ा": "a",
    "इ": "i",
    "ि": "i",
    "ई": "i",
    "ी": "i",
    "उ": "u",
    "ु": "u",
    "ऊ": "u",
    "ू": "u",
    "ऋ": "ri",
    "ृ": "r",
    "ॠ": "ri",
    "ॢ": "r",
    "ए": "e",
    "े": "e",
    "ऐ": "ai",
    "ै": "ai",
    "ओ": "o",
    "ो": "o",
    "औ": "au",
    "ौ": "au",
    "ं": "n",
    "ँ": "n",
    "ः": "h",
    "़": "",
    "्": "",
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "क्ष": "ksh",
    "त्र": "tr",
    "ज्ञ": "gy",
    "श्र": "shr",
    "ज़": "z",
    "ड़": "r",
    "ढ़": "rh",
    "फ़": "f",
    "ॐ": "om",
}


def romanize_hinglish_text(text: str) -> str:
    if not text.strip() or not contains_devanagari(text):
        return text

    words = text.strip().split()
    romanized_words = []
    for word in words:
        if word in HINGLISH_WORD_OVERRIDES:
            romanized_words.append(HINGLISH_WORD_OVERRIDES[word])
            continue

        candidate = ""
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i:i + 2] in DEVANAGARI_TO_HINGLISH:
                candidate += DEVANAGARI_TO_HINGLISH[word[i:i + 2]]
                i += 2
                continue
            if word[i] in DEVANAGARI_TO_HINGLISH:
                candidate += DEVANAGARI_TO_HINGLISH[word[i]]
            i += 1

        if not candidate:
            romanized_words.append(word)
        else:
            romanized_words.append(candidate.lower())

    return " ".join(romanized_words)


def transliterate_segment_text(text: str, output_script: str) -> str:
    if not text.strip():
        return text
    if output_script == "Hinglish":
        return romanize_hinglish_text(text)
    if output_script != "English (Transliteration)":
        return text
    if not contains_devanagari(text):
        return text
    if not HAS_INDIC_TRANSLITERATION:
        raise RuntimeError(
            "English transliteration requires 'indic-transliteration'. "
            "Install it with: pip install indic-transliteration"
        )
    return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


def split_segment_by_words(seg, current_start: float, total_duration: float, words_per_line: int, selected_output_script: str) -> list:
    seg_start = max(current_start, current_start + float(seg["start"]))
    seg_end = min(total_duration, current_start + float(seg["end"]))
    
    if seg_end <= seg_start:
        return []

    raw_text = seg.get("text", "").strip()
    if not raw_text or words_per_line <= 0:
        return [{
            "start": seg_start,
            "end": seg_end,
            "text": transliterate_segment_text(raw_text, selected_output_script)
        }]

    words = seg.get("words", [])
    
    # If we have word timestamps, split based on them
    if words:
        sub_segments = []
        current_words_group = []
        for w in words:
            w_text = w.get("word", "").strip()
            if not w_text:
                continue
            
            # Resolve absolute timestamps
            w_start = max(seg_start, min(seg_end, current_start + float(w["start"])))
            w_end = max(seg_start, min(seg_end, current_start + float(w["end"])))
            
            current_words_group.append({
                "text": w_text,
                "start": w_start,
                "end": w_end
            })
            
            if len(current_words_group) == words_per_line:
                sub_text = " ".join(item["text"] for item in current_words_group)
                sub_start = current_words_group[0]["start"]
                sub_end = current_words_group[-1]["end"]
                if sub_end > sub_start:
                    sub_segments.append({
                        "start": sub_start,
                        "end": sub_end,
                        "text": transliterate_segment_text(sub_text, selected_output_script)
                    })
                current_words_group = []
        
        # Handle the remaining words
        if current_words_group:
            sub_text = " ".join(item["text"] for item in current_words_group)
            sub_start = current_words_group[0]["start"]
            sub_end = current_words_group[-1]["end"]
            if sub_end > sub_start:
                sub_segments.append({
                    "start": sub_start,
                    "end": sub_end,
                    "text": transliterate_segment_text(sub_text, selected_output_script)
                })
        
        if sub_segments:
            return sub_segments

    # Fallback to linear interpolation if word timestamps are missing or empty
    raw_words = raw_text.split()
    if not raw_words:
        return []
        
    sub_segments = []
    num_words = len(raw_words)
    groups = [raw_words[i:i + words_per_line] for i in range(0, num_words, words_per_line)]
    num_groups = len(groups)
    duration = seg_end - seg_start
    
    for i, gp in enumerate(groups):
        gp_text = " ".join(gp)
        gp_start = seg_start + (i / num_groups) * duration
        gp_end = seg_start + ((i + 1) / num_groups) * duration
        sub_segments.append({
            "start": gp_start,
            "end": gp_end,
            "text": transliterate_segment_text(gp_text, selected_output_script)
        })
        
    return sub_segments



# ---------- Application GUI ----------

class AutoCaptionApp:
    def __init__(self):
        ensure_standard_streams()
        if ctk is None:
            raise RuntimeError("CustomTkinter is required to run the GUI. Please install the requirements and try again.")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.config = load_config()
        self.gpu_available = False
        self.gpu_name = None
        if torch is not None:
            try:
                self.gpu_available = torch.cuda.is_available()
                self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else None
            except Exception:
                pass
        self.compute_choice = normalize_compute_choice(self.config.get("compute_device", "Auto"), self.gpu_available)
        self.compute_device = resolve_device(self.compute_choice, self.gpu_available)

        # check ffmpeg availability early
        if not self._ffmpeg_available():
            messagebox.showerror("Missing FFmpeg", "FFmpeg not found in PATH. Please install FFmpeg and restart the app.")

        self.root = TkinterDnD.Tk() if TkinterDnD is not None else Tk()
        self.root.title("AutoCaption Studio")
        self.root.geometry("700x520")
        self.root.resizable(False, False)

        self.video_path = None
        self.words_per_line = self._get_words_per_line_setting()

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
        if DND_FILES is not None:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        # model selector
        models = ["tiny", "base", "small", "medium", "large"]
        saved_model = self.config.get("model", "base")
        if saved_model not in models:
            saved_model = "base"
        self.model_var = ctk.StringVar(value=saved_model)
        controls_row_1 = ctk.CTkFrame(frame)
        controls_row_1.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(controls_row_1, text="Model:").pack(side="left", padx=(0, 5))
        self.model_menu = ctk.CTkOptionMenu(controls_row_1, values=models, variable=self.model_var)
        self.model_menu.pack(side="left")
        self.model_var.trace_add("write", self._on_model_change)

        # compute device selector
        compute_values = ["Auto", "CPU"]
        if self.gpu_available:
            compute_values.append("GPU")
        saved_compute_choice = normalize_compute_choice(self.config.get("compute_device", "Auto"), self.gpu_available)
        self.compute_var = ctk.StringVar(value=saved_compute_choice)
        ctk.CTkLabel(controls_row_1, text="Compute:").pack(side="left", padx=(15, 5))
        self.compute_menu = ctk.CTkOptionMenu(
            controls_row_1,
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
        ctk.CTkLabel(controls_row_1, text="Output Script:").pack(side="left", padx=(15, 5))
        self.output_script_menu = ctk.CTkOptionMenu(
            controls_row_1,
            values=OUTPUT_SCRIPT_OPTIONS,
            variable=self.output_script_var,
            command=self._on_output_script_change,
        )
        self.output_script_menu.pack(side="left")

        # words per line selector
        controls_row_2 = ctk.CTkFrame(frame)
        controls_row_2.pack(fill="x", pady=8)
        saved_words_per_line = self.config.get("words_per_line", 8)
        try:
            saved_words_per_line = max(1, int(saved_words_per_line))
        except Exception:
            saved_words_per_line = 8
        self.words_per_line_var = ctk.StringVar(value=str(saved_words_per_line))
        ctk.CTkLabel(controls_row_2, text="Words/line:").pack(side="left", padx=(0, 5))
        self.words_per_line_entry = ctk.CTkEntry(controls_row_2, width=90, textvariable=self.words_per_line_var)
        self.words_per_line_entry.pack(side="left")
        ctk.CTkLabel(controls_row_2, text="(applies to every subtitle line)", text_color="#9aa0a6").pack(side="left", padx=(8, 0))

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
        self.words_per_line = self._get_words_per_line_setting()
        # verify ffmpeg again in case PATH changed
        if not self._ffmpeg_available():
            messagebox.showerror("Missing FFmpeg", "FFmpeg not found in PATH. Please install FFmpeg and restart the app.")
            return
        self.start_button.configure(state="disabled")
        self.model_menu.configure(state="disabled")
        self.output_script_menu.configure(state="disabled")
        self.words_per_line_entry.configure(state="disabled")
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

    def _get_words_per_line_setting(self) -> int:
        raw_value = (self.words_per_line_var.get() if hasattr(self, "words_per_line_var") else "8").strip()
        try:
            parsed_value = int(raw_value)
        except Exception:
            parsed_value = 8
        parsed_value = max(1, parsed_value)
        self.config["words_per_line"] = parsed_value
        save_config(self.config)
        return parsed_value

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
            if whisper is None:
                raise RuntimeError("Whisper is not installed. Please install the project requirements and try again.")
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
                    word_timestamps=True,
                )
                for seg in result.get("segments", []):
                    split_segs = split_segment_by_words(
                        seg,
                        current_start=current_start,
                        total_duration=total_duration,
                        words_per_line=self.words_per_line,
                        selected_output_script=selected_output_script
                    )
                    segments.extend(split_segs)

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
                    word_timestamps=True,
                )
                for seg in tail_result.get("segments", []):
                    split_segs = split_segment_by_words(
                        seg,
                        current_start=tail_start,
                        total_duration=total_duration,
                        words_per_line=self.words_per_line,
                        selected_output_script=selected_output_script
                    )
                    segments.extend(split_segs)

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
            write_srt(segments, out_srt, words_per_line=self.words_per_line)

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
            if hasattr(self, "words_per_line_entry"):
                self.words_per_line_entry.configure(state="normal")
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
