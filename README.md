# AutoCaption Studio

A desktop application for automatically generating `.srt` captions from videos using OpenAI's Whisper model.

## Overview

* Drag and drop video files (`.mp4`, `.mov`, `.mkv`, `.avi`).
* Select a Whisper model (tiny/base/small/medium/large).
* Generate `.srt` file saved alongside the video.
* Works offline and shows progress in a GUI.
* Built with Python, CustomTkinter, and FFmpeg.

## Setup

1. **Install Python 3.10+**
2. **Install FFmpeg** and add to your system `PATH`.
   ```bash
   ffmpeg -version
   ```
3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running

```bash
python app.py
```

Drop a video onto the window, choose a model, then click **Generate Captions**.

## Packaging

To create an executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed app.py
```

The result will be under `dist/app.exe`.

## Notes

This tool is designed for video editors using After Effects; Once captions are generated, import the `.srt` into AE via **File → Import → Captions (SRT)**.

Future enhancements can include word-level timestamps, character limits, filler removal, batch processing, and more.
