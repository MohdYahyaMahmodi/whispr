"""
Whispr - Open Source Transcription Studio
A web-based interface for OpenAI's Whisper speech recognition model.

Features:
- Real-time progress tracking with accurate progress bars
- GPU acceleration support with automatic fallback to CPU
- Multiple Whisper model sizes (tiny to turbo)
- Support for multiple audio/video formats
- Private local processing (no data sent to external servers)
- Export to TXT and SRT formats

Author: Mohd Mahmodi
License: MIT
GitHub: https://github.com/mohdmahmodi/whispr
"""

# Standard library imports
import os
import sys
import time
import uuid
import threading
import tempfile
import platform
import subprocess
from datetime import datetime
from pathlib import Path

# Third-party imports
import torch
import whisper
import librosa
import requests
import psutil
from flask import Flask, render_template, request, jsonify, send_file

# ================================
# Application Configuration
# ================================
app = Flask(__name__, static_folder='static')

# Directory paths
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
TRANSCRIPT_FOLDER = os.path.join(os.getcwd(), 'transcripts')
WHISPER_CACHE_DIR = os.path.join(os.getcwd(), 'whisper_models')

# File type restrictions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'avi', 'mov', 'flac', 'ogg', 'm4a'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)

# Global state management
transcription_tasks = {}  # Active transcription tasks
task_lock = threading.Lock()  # Thread-safe task access

# ================================
# Whisper Model Configurations
# ================================
WHISPER_MODELS = {
    'tiny': {
        'description': 'Fastest model (~10x realtime), 39M parameters',
        'parameters': '39M',
        'size_mb': 39,  # Actual model size
        'vram_gb': 1,
        'relative_speed': 10,
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt'
    },
    'base': {
        'description': 'Fast model (~7x realtime), 74M parameters',
        'parameters': '74M',
        'size_mb': 74,
        'vram_gb': 1,
        'relative_speed': 7,
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/4d70c13eb6c0d487dab0f48ecb2806ef7c5431dae0f4d1ea46e3d5119a9f60c7/base.en.pt'
    },
    'small': {
        'description': 'Balanced model (~4x realtime), 244M parameters',
        'parameters': '244M',
        'size_mb': 244,
        'vram_gb': 2,
        'relative_speed': 4,
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/55356645c2b361a969dfd0ef2c5a50d530afd8105a9f54e8e8c5cc79a6ef98/small.en.pt'
    },
    'medium': {
        'description': 'High accuracy model (~2x realtime), 769M parameters',
        'parameters': '769M',
        'size_mb': 769,
        'vram_gb': 5,
        'relative_speed': 2,
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.en.pt'
    },
    'large': {
        'description': 'Highest accuracy model (1x realtime), 1550M parameters',
        'parameters': '1550M',
        'size_mb': 1550,
        'vram_gb': 10,
        'relative_speed': 1,
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v3.pt'
    },
    'turbo': {
        'description': 'Optimized large-v3 model (~8x realtime), 809M parameters',
        'parameters': '809M',
        'size_mb': 809,
        'vram_gb': 6,
        'relative_speed': 8,
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638aa33c9e52e05b678c8c4d675b2/large-v3-turbo.pt'
    }
}

# ================================
# System Detection & Hardware Info
# ================================
def log_with_timestamp(message, level="INFO"):
    """Print log message with timestamp formatting."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def get_hardware_info():
    """
    Detect and return system hardware information including CPU, RAM, and GPU details.
    Supports Windows, macOS, and Linux platforms.
    """
    # GPU detection
    has_gpu = torch.cuda.is_available()
    
    # CPU information detection (platform-specific)
    cpu_info = ""
    try:
        if platform.system() == "Windows":
            # Windows-specific CPU detection using ctypes
            import ctypes
            from ctypes import wintypes
            
            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", wintypes.WORD),
                    ("wReserved", wintypes.WORD),
                    ("dwPageSize", wintypes.DWORD),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", wintypes.DWORD),
                    ("dwNumberOfProcessors", wintypes.DWORD),
                    ("dwProcessorType", wintypes.DWORD),
                    ("dwAllocationGranularity", wintypes.DWORD),
                    ("wProcessorLevel", wintypes.WORD),
                    ("wProcessorRevision", wintypes.WORD),
                ]
                
            si = SYSTEM_INFO()
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
            cpu_cores = si.dwNumberOfProcessors
            
            try:
                # Get CPU model from Windows registry
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                winreg.CloseKey(key)
                cpu_info = f"{cpu_model} ({cpu_cores} cores)"
            except:
                cpu_info = f"{cpu_cores} cores"
                
        elif platform.system() == "Darwin":
            # macOS CPU detection using sysctl
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            cpu_model = subprocess.check_output(cmd).decode().strip()
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_info = f"{cpu_model} ({cpu_cores} cores)"
            
        elif platform.system() == "Linux":
            # Linux CPU detection from /proc/cpuinfo
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":", 1)[1].strip()
                            break
                cpu_cores = psutil.cpu_count(logical=True)
                cpu_info = f"{cpu_model} ({cpu_cores} cores)"
            except:
                cpu_cores = psutil.cpu_count(logical=True)
                cpu_info = f"{cpu_cores} cores"
        else:
            # Fallback for other platforms
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_info = f"{cpu_cores} cores"
    except:
        # Final fallback
        cpu_info = f"{psutil.cpu_count(logical=True)} cores detected"
    
    # RAM detection
    total_ram = psutil.virtual_memory().total
    
    # GPU information structure
    if has_gpu:
        gpu_info = {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'count': torch.cuda.device_count(),
            'memory': torch.cuda.get_device_properties(0).total_memory,
            'cuda_version': torch.version.cuda
        }
    else:
        gpu_info = {
            'available': False,
            'name': None,
            'count': 0,
            'memory': 0,
            'cuda_version': None
        }
    
    return gpu_info, cpu_info, total_ram

# Initialize system information at startup
gpu_info, cpu_info, system_ram = get_hardware_info()

# Log system information
log_with_timestamp("==== SYSTEM DETECTION ====")
log_with_timestamp(f"PyTorch version: {torch.__version__}")
log_with_timestamp(f"CPU: {cpu_info}")
log_with_timestamp(f"RAM: {system_ram / (1024**3):.2f} GB")
log_with_timestamp(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_with_timestamp(f"CUDA version: {torch.version.cuda}")
    log_with_timestamp(f"GPU device name: {torch.cuda.get_device_name(0)}")
    log_with_timestamp(f"GPU device count: {torch.cuda.device_count()}")
    log_with_timestamp(f"GPU memory: {gpu_info['memory'] / (1024**3):.2f} GB")
else:
    log_with_timestamp("CUDA is not available. Using CPU mode (slower).")
log_with_timestamp("===========================")

# ================================
# Utility Functions
# ================================
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_audio_duration(file_path):
    """
    Get audio/video duration in seconds using librosa.
    Falls back to file size estimation if librosa fails.
    """
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        log_with_timestamp(f"Error getting audio duration: {e}", "WARNING")
        # Fallback: rough estimation based on file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb * 60  # Assume ~1MB per minute

def update_task_progress(task_id, progress, status, stage_info, detailed_status, console_message=None):
    """
    Thread-safe function to update transcription task progress.
    Calculates time estimates and logs progress messages.
    FIXED: Ensure progress is always a valid number between 0 and 100.
    """
    with task_lock:
        if task_id in transcription_tasks:
            # Ensure progress is a valid number between 0 and 100
            progress = max(0, min(100, float(progress) if progress is not None else 0))
            
            # Update basic progress information
            transcription_tasks[task_id]['progress'] = progress
            transcription_tasks[task_id]['status'] = status
            transcription_tasks[task_id]['stage_info'] = stage_info
            transcription_tasks[task_id]['detailed_status'] = detailed_status
            
            # Calculate time estimates
            start_time = transcription_tasks[task_id].get('start_time', time.time())
            elapsed = time.time() - start_time
            
            if progress > 5:  # Only estimate after 5% progress
                estimated_total = elapsed * (100 / progress)
                remaining = max(0, estimated_total - elapsed)
                transcription_tasks[task_id]['time_remaining'] = remaining
                transcription_tasks[task_id]['estimated_total'] = estimated_total
            
            # Log progress if message provided
            if console_message:
                log_with_timestamp(f"[Task {task_id[:8]}] {console_message}")

def format_time(seconds):
    """Convert seconds to SRT subtitle time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

# ================================
# Model Download Functions
# ================================
def download_model_with_progress(model_name, task_id):
    """
    Download Whisper model from OpenAI with real-time progress tracking.
    Skips download if model is already cached locally.
    FIXED: Better progress tracking and error handling.
    """
    model_path = os.path.join(WHISPER_CACHE_DIR, f"{model_name}.pt")
    
    # Check if model already exists
    if os.path.exists(model_path):
        log_with_timestamp(f"Model {model_name} already cached")
        update_task_progress(
            task_id, 15, 'model_ready', 'Model ready', 
            f"Model {model_name} is already downloaded",
            f"Model {model_name} found in cache, skipping download"
        )
        return True
    
    try:
        download_url = WHISPER_MODELS[model_name]['download_url']
        expected_size = WHISPER_MODELS[model_name]['size_mb'] * 1024 * 1024
        
        log_with_timestamp(f"Starting download of {model_name} model ({expected_size / (1024*1024):.0f} MB)")
        update_task_progress(
            task_id, 5, 'downloading_model', 'Downloading model', 
            f"Starting download of {model_name} model",
            f"Downloading {model_name} model from OpenAI servers"
        )
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Start streaming download
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', expected_size))
        
        # Initialize download tracking with valid numbers
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['total_size'] = total_size
                transcription_tasks[task_id]['downloaded'] = 0
        
        # Download with progress updates
        downloaded = 0
        chunk_size = 8192
        last_log_time = time.time()
        last_progress_update = time.time()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Calculate progress (5% to 15% of total progress)
                    download_progress = min(downloaded / total_size, 1.0)
                    total_progress = 5 + (download_progress * 10)
                    
                    # Update task progress every 100ms to avoid overwhelming the client
                    current_time = time.time()
                    if current_time - last_progress_update > 0.1:
                        with task_lock:
                            if task_id in transcription_tasks:
                                transcription_tasks[task_id]['progress'] = total_progress
                                transcription_tasks[task_id]['downloaded'] = downloaded
                                transcription_tasks[task_id]['detailed_status'] = f"Downloaded: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB"
                        last_progress_update = current_time
                    
                    # Periodic logging to console (every 2 seconds)
                    if current_time - last_log_time > 2.0:
                        log_with_timestamp(f"Download progress: {downloaded/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({download_progress*100:.1f}%)")
                        last_log_time = current_time
        
        # Move downloaded file to final location
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.rename(temp_path, model_path)
        
        log_with_timestamp(f"Model {model_name} downloaded successfully")
        update_task_progress(
            task_id, 15, 'model_ready', 'Model downloaded', 
            f"Model {model_name} downloaded and ready",
            f"Model {model_name} download completed successfully"
        )
        
        return True
        
    except Exception as e:
        log_with_timestamp(f"Error downloading model: {str(e)}", "ERROR")
        update_task_progress(
            task_id, 0, 'error', 'Download failed', 
            f"Failed to download model: {str(e)}",
            f"Model download failed: {str(e)}"
        )
        # Clean up partial download
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return False

# ================================
# Transcription Functions
# ================================
def monitor_transcription_progress(task_id, audio_duration, start_time):
    """
    Monitor transcription progress in separate thread.
    Estimates progress based on model speed and elapsed time.
    FIXED: Better progress calculation and smoother updates.
    """
    last_update = time.time()
    
    # Updated model speed estimates (realtime multipliers)
    model_speeds = {
        'tiny': 10,     # ~10x realtime
        'base': 7,      # ~7x realtime  
        'small': 4,     # ~4x realtime
        'medium': 2,    # ~2x realtime
        'large': 1,     # ~1x realtime
        'turbo': 8      # ~8x realtime
    }
    
    while True:
        time.sleep(0.5)  # Update every 500ms for smoother progress
        
        current_time = time.time()
        elapsed = current_time - start_time
        
        with task_lock:
            if task_id not in transcription_tasks:
                break
            
            task = transcription_tasks[task_id]
            if task['status'] != 'transcribing':
                break
            
            # Calculate expected progress
            model_name = task['model']
            expected_speed = model_speeds.get(model_name, 4)
            
            # Adjust for CPU vs GPU (CPU is ~3x slower)
            if not gpu_info['available']:
                expected_speed = expected_speed / 3
            
            # Calculate progress ratio with some smoothing
            expected_duration = audio_duration / expected_speed
            progress_ratio = min(elapsed / expected_duration, 0.95)  # Cap at 95%
            
            # Map to UI progress range (25% - 95%) - leaving room for final processing
            ui_progress = 25 + (progress_ratio * 70)
            
            # Only update if progress has increased by at least 1%
            current_progress = task.get('progress', 0)
            if ui_progress > current_progress + 1:
                transcription_tasks[task_id]['progress'] = ui_progress
                
                # Calculate time remaining
                if progress_ratio > 0.1:
                    estimated_total = elapsed / progress_ratio
                    remaining = max(0, estimated_total - elapsed)
                    transcription_tasks[task_id]['time_remaining'] = remaining
                    
                    # Update detailed status with time estimate
                    if remaining > 60:
                        time_str = f"{int(remaining/60)}m {int(remaining%60)}s"
                    else:
                        time_str = f"{int(remaining)}s"
                    
                    transcription_tasks[task_id]['detailed_status'] = f"Transcribing: {progress_ratio*100:.0f}% complete (Est. {time_str} remaining)"
        
        # Periodic console logging (every 5 seconds)
        if current_time - last_update > 5:
            with task_lock:
                if task_id in transcription_tasks:
                    current_progress = transcription_tasks[task_id].get('progress', 0)
                    log_with_timestamp(f"Transcription progress: {current_progress:.0f}% ({elapsed:.0f}s elapsed)")
                    last_update = current_time

def transcribe_file(file_path, model_name, task_id, transcription_options=None):
    """
    Main transcription function that processes audio/video files.
    Handles the complete transcription pipeline with progress tracking.
    FIXED: Better progress management and error handling.
    """
    transcription_options = transcription_options or {}
    
    try:
        log_with_timestamp(f"Starting transcription process for {os.path.basename(file_path)}")
        
        # Step 1: Analyze audio file (0% - 5%)
        update_task_progress(
            task_id, 2, 'analyzing', 'Analyzing audio', 
            "Analyzing audio file properties",
            "Analyzing audio file to determine duration and format"
        )
        
        audio_duration = get_audio_duration(file_path)
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['audio_duration'] = audio_duration
        
        log_with_timestamp(f"Audio duration: {audio_duration:.1f} seconds ({audio_duration/60:.1f} minutes)")
        
        # Step 2: Download model if needed (5% - 15%)
        if not download_model_with_progress(model_name, task_id):
            return
        
        # Step 3: Load model into memory (15% - 25%)
        log_with_timestamp(f"Loading {model_name} model into memory")
        update_task_progress(
            task_id, 20, 'loading_model', 'Loading model', 
            f"Loading {model_name} model into memory",
            f"Loading {model_name} model on {'GPU' if gpu_info['available'] else 'CPU'}"
        )
        
        device = "cuda" if gpu_info['available'] else "cpu"
        model = whisper.load_model(model_name, device=device, download_root=WHISPER_CACHE_DIR)
        log_with_timestamp(f"Model loaded on {device.upper()}")
        
        # Step 4: Start transcription (25% - 95%)
        update_task_progress(
            task_id, 25, 'transcribing', 'Transcribing audio', 
            f"Starting transcription on {device.upper()}",
            f"Beginning audio transcription using {model_name} model"
        )
        
        # Start progress monitoring thread
        processing_start_time = time.time()
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['processing_start_time'] = processing_start_time
        
        transcription_thread = threading.Thread(
            target=monitor_transcription_progress,
            args=(task_id, audio_duration, processing_start_time),
            daemon=True
        )
        transcription_thread.start()
        
        # Prepare transcription options
        transcribe_args = {
            'fp16': (device == "cuda"),
            'verbose': True
        }
        
        # Add advanced options if provided
        if transcription_options.get('multilingual'):
            transcribe_args['language'] = None  # Auto-detect language
        if transcription_options.get('word_timestamps'):
            transcribe_args['word_timestamps'] = True
        
        # Perform actual transcription
        result = model.transcribe(file_path, **transcribe_args)
        
        processing_time = time.time() - processing_start_time
        log_with_timestamp(f"Transcription completed in {processing_time:.1f} seconds")
        log_with_timestamp(f"Processing speed: {audio_duration/processing_time:.1f}x realtime")
        
        # Step 5: Generate text file (95% - 97%)
        update_task_progress(
            task_id, 96, 'generating_files', 'Creating transcript', 
            "Generating text transcript file",
            "Creating .txt transcript file"
        )
        
        txt_filename = os.path.join(TRANSCRIPT_FOLDER, f"{task_id}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        log_with_timestamp(f"Text transcript saved: {txt_filename}")
        
        # Step 6: Generate SRT subtitle file (97% - 99%)
        update_task_progress(
            task_id, 98, 'generating_files', 'Creating subtitles', 
            "Generating SRT subtitle file",
            "Creating .srt subtitle file with timestamps"
        )
        
        srt_filename = os.path.join(TRANSCRIPT_FOLDER, f"{task_id}.srt")
        with open(srt_filename, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"], start=1):
                f.write(f"{i}\n")
                start_time = format_time(segment["start"])
                end_time = format_time(segment["end"])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        log_with_timestamp(f"SRT subtitle file saved: {srt_filename}")
        
        # Step 7: Mark as completed (100%)
        total_time = time.time() - transcription_tasks[task_id]['start_time']
        update_task_progress(
            task_id, 100, 'completed', 'Transcription complete', 
            f"Completed in {total_time:.1f} seconds",
            f"Transcription completed successfully in {total_time:.1f} seconds"
        )
        
        # Store final results
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['transcript_txt'] = txt_filename
                transcription_tasks[task_id]['transcript_srt'] = srt_filename
                transcription_tasks[task_id]['completion_time'] = time.time()
                transcription_tasks[task_id]['processing_time'] = total_time
                transcription_tasks[task_id]['live_transcript'] = result["text"]
        
        log_with_timestamp(f"Transcription task {task_id[:8]} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        log_with_timestamp(f"Transcription error: {error_msg}", "ERROR")
        update_task_progress(
            task_id, 0, 'error', 'Error occurred', 
            f"Transcription failed: {error_msg}",
            f"Error during transcription: {error_msg}"
        )
        
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['error'] = error_msg
    
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                log_with_timestamp(f"Cleaned up uploaded file: {os.path.basename(file_path)}")
            except Exception as e:
                log_with_timestamp(f"Error cleaning up file: {str(e)}", "WARNING")

# ================================
# Flask Route Handlers
# ================================
@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html', models=WHISPER_MODELS, gpu_info=gpu_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initiate transcription process."""
    # Validate request has file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Validate file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if not file or not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique task ID and get model selection
    task_id = str(uuid.uuid4())
    model_name = request.form.get('model', 'base')
    
    # Validate model selection
    if model_name not in WHISPER_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    # Get advanced options
    transcription_options = {
        'multilingual': request.form.get('multilingual', 'false').lower() == 'true',
        'word_timestamps': request.form.get('word_timestamps', 'false').lower() == 'true',
        'speaker_detection': request.form.get('speaker_detection', 'false').lower() == 'true'
    }
    
    # Save uploaded file
    filename = os.path.join(UPLOAD_FOLDER, f"{task_id}_{file.filename}")
    try:
        file.save(filename)
        file_size = os.path.getsize(filename)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    log_with_timestamp(f"File uploaded: {file.filename} ({file_size / (1024*1024):.1f} MB)")
    log_with_timestamp(f"Task ID: {task_id}")
    log_with_timestamp(f"Selected model: {model_name}")
    log_with_timestamp(f"Options: {transcription_options}")
    
    # Initialize task tracking with better defaults
    with task_lock:
        transcription_tasks[task_id] = {
            'status': 'uploaded',
            'file': filename,
            'file_name': file.filename,
            'file_size': file_size,
            'model': model_name,
            'progress': 0,  # Start at 0%
            'transcript_txt': None,
            'transcript_srt': None,
            'error': None,
            'device_type': 'gpu' if gpu_info['available'] else 'cpu',
            'start_time': time.time(),
            'stage_info': 'File uploaded',
            'detailed_status': f"File {file.filename} uploaded successfully",
            'console_logs': [],
            'audio_duration': None,
            'processing_start_time': None,
            'downloaded': 0,
            'total_size': 0,
            'time_remaining': None,
            'estimated_total': None,
            'transcription_options': transcription_options
        }
    
    # Start transcription in background thread
    try:
        thread = threading.Thread(
            target=transcribe_file, 
            args=(filename, model_name, task_id, transcription_options),
            daemon=True
        )
        thread.start()
        time.sleep(0.1)  # Ensure thread starts
        
        return jsonify({
            'message': 'File uploaded successfully',
            'task_id': task_id,
            'file_name': file.filename,
            'file_size': file_size,
            'model': model_name,
            'options': transcription_options
        }), 200
    except Exception as e:
        log_with_timestamp(f"Error starting transcription: {str(e)}", "ERROR")
        return jsonify({'error': f'Failed to start transcription: {str(e)}'}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get current status and progress of a transcription task."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id].copy()
    
    # Build comprehensive status response with proper data validation
    response = {
        'status': task['status'],
        'progress': max(0, min(100, task['progress'])),  # Ensure valid progress
        'error': task.get('error'),
        'device_type': task.get('device_type', 'unknown'),
        'stage_info': task.get('stage_info', ''),
        'detailed_status': task.get('detailed_status', ''),
        'file_name': task.get('file_name', ''),
        'model': task.get('model', ''),
        'audio_duration': task.get('audio_duration'),
        'console_logs': task.get('console_logs', [])
    }
    
    # Add timing information if available
    if 'processing_time' in task:
        response['processing_time'] = task['processing_time']
    if 'time_remaining' in task and task['time_remaining'] is not None:
        response['time_remaining'] = max(0, task['time_remaining'])
    if 'estimated_total' in task and task['estimated_total'] is not None:
        response['estimated_total'] = task['estimated_total']
    
    # Add download progress for model downloads - ensure valid numbers
    if task['status'] == 'downloading_model':
        downloaded = task.get('downloaded', 0)
        total_size = task.get('total_size', 0)
        
        # Only include if we have valid data
        if downloaded >= 0 and total_size > 0:
            response['downloaded'] = downloaded
            response['total_size'] = total_size
    
    return jsonify(response), 200

@app.route('/live_transcript/<task_id>', methods=['GET'])
def get_live_transcript(task_id):
    """Get current live transcript for ongoing transcription."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id]
        return jsonify({
            'live_transcript': task.get('live_transcript', ''),
            'status': task['status'],
            'progress': task['progress']
        }), 200

@app.route('/system_info', methods=['GET'])
def get_system_info():
    """Get current system hardware information and cached models."""
    updated_gpu_info, updated_cpu_info, updated_ram = get_hardware_info()
    
    # Get list of cached Whisper models
    cached_models = []
    if os.path.exists(WHISPER_CACHE_DIR):
        cached_models = [f.name for f in os.scandir(WHISPER_CACHE_DIR) 
                        if f.is_file() and f.name.endswith('.pt')]
    
    return jsonify({
        'gpu': updated_gpu_info,
        'cpu_info': updated_cpu_info,
        'ram': updated_ram,
        'pytorch_version': torch.__version__,
        'cached_models': cached_models,
        'whisper_models': WHISPER_MODELS
    }), 200

@app.route('/transcript/<task_id>', methods=['GET'])
def get_transcript_text(task_id):
    """Get the final transcript text for a completed transcription."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id]
        
        if task['status'] != 'completed':
            return jsonify({'error': 'Transcription not completed yet'}), 400
    
    try:
        with open(task['transcript_txt'], 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        return jsonify({'transcript': transcript_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<task_id>/<format_type>', methods=['GET'])
def download_transcript(task_id, format_type):
    """Download completed transcript in specified format (txt or srt)."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id]
        
        if task['status'] != 'completed':
            return jsonify({'error': 'Transcription not completed yet'}), 400
    
    # Determine file path and MIME type based on format
    if format_type == 'txt':
        file_path = task['transcript_txt']
        mimetype = 'text/plain'
        filename = f"transcript_{task['file_name'].rsplit('.', 1)[0]}.txt"
    elif format_type == 'srt':
        file_path = task['transcript_srt']
        mimetype = 'text/srt'
        filename = f"subtitle_{task['file_name'].rsplit('.', 1)[0]}.srt"
    else:
        return jsonify({'error': 'Invalid format. Use "txt" or "srt"'}), 400
    
    return send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

@app.route('/cleanup/<task_id>', methods=['POST'])
def cleanup_task(task_id):
    """Clean up task resources after download."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id]
        
        # Remove transcript files
        for file_path in [task.get('transcript_txt'), task.get('transcript_srt')]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # Remove task from memory
        del transcription_tasks[task_id]
    
    log_with_timestamp(f"Task {task_id[:8]} cleaned up")
    return jsonify({'message': 'Task cleaned up successfully'}), 200

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about available Whisper models."""
    return jsonify(WHISPER_MODELS), 200

# ================================
# Application Entry Point
# ================================
if __name__ == '__main__':
    # Check for required dependencies
    try:
        import librosa
    except ImportError:
        log_with_timestamp("librosa not found. Install with: pip install librosa", "ERROR")
        sys.exit(1)
    
    log_with_timestamp("Starting Whispr Transcription Studio...")
    log_with_timestamp(f"Available models: {', '.join(WHISPER_MODELS.keys())}")
    app.run(debug=True, host='0.0.0.0', port=5000)