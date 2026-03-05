import os
import threading
import traceback

import whisper
import ffmpeg
import customtkinter as ctk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD


SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")


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


# ---------- Application GUI ----------

class AutoCaptionApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # check ffmpeg availability early
        if not self._ffmpeg_available():
            messagebox.showerror("Missing FFmpeg", "FFmpeg not found in PATH. Please install FFmpeg and restart the app.")

        self.root = TkinterDnD.Tk()
        self.root.title("AutoCaption Studio")
        self.root.geometry("600x400")
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
        self.model_var = ctk.StringVar(value="base")
        model_frame = ctk.CTkFrame(frame)
        model_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(model_frame, text="Model:").pack(side="left", padx=(0, 5))
        self.model_menu = ctk.CTkOptionMenu(model_frame, values=models, variable=self.model_var)
        self.model_menu.pack(side="left")

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
        threading.Thread(target=self._process_file, daemon=True).start()

    def _process_file(self):
        audio_path = None
        try:
            self._update_status("Loading model... (10%)", 0.1)
            model_name = self.model_var.get()
            model = whisper.load_model(model_name)

            # whisper can handle video directly, but extract audio for reliability
            self._update_status("Extracting audio... (25%)", 0.25)
            audio_path = extract_audio(self.video_path)

            self._update_status("Transcribing... (40%)", 0.4)
            result = model.transcribe(audio_path)

            self._update_status("Generating captions... (75%)", 0.75)
            segments = result.get("segments", [])

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
            # remove temporary audio file if exists
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
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
