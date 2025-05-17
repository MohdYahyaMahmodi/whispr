from flask import Flask, render_template, request, jsonify, send_file
import os
import whisper
import time
import uuid
import threading
import json
import torch
import sys
import requests
import tempfile
import psutil
import platform
import subprocess
from tqdm import tqdm
from pathlib import Path

app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
TRANSCRIPT_FOLDER = os.path.join(os.getcwd(), 'transcripts')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'avi', 'mov', 'flac', 'ogg', 'm4a'}
WHISPER_CACHE_DIR = os.path.join(os.getcwd(), 'whisper_models')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'static', 'js'), exist_ok=True)

# Store ongoing transcription tasks
transcription_tasks = {}
task_lock = threading.Lock()  # Add a lock for thread safety

# Function to get system hardware information
def get_hardware_info():
    # Check if GPU is available
    has_gpu = torch.cuda.is_available()
    
    # Get CPU information
    cpu_info = ""
    try:
        if platform.system() == "Windows":
            # Get CPU info on Windows
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
            
            # Try to get CPU model
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                winreg.CloseKey(key)
                cpu_info = f"{cpu_model} ({cpu_cores} cores)"
            except:
                cpu_info = f"{cpu_cores} cores"
                
        elif platform.system() == "Darwin":  # macOS
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            cpu_model = subprocess.check_output(cmd).decode().strip()
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_info = f"{cpu_model} ({cpu_cores} cores)"
            
        elif platform.system() == "Linux":
            # Try to get CPU info from /proc/cpuinfo
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
        # Fallback if something goes wrong
        cpu_info = f"{psutil.cpu_count(logical=True)} cores detected"
    
    # Get total RAM
    total_ram = psutil.virtual_memory().total
    
    # GPU info
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

# Get system info at startup
gpu_info, cpu_info, system_ram = get_hardware_info()

# Print system info at startup
print("\n==== SYSTEM DETECTION ====")
print(f"PyTorch version: {torch.__version__}")
print(f"CPU: {cpu_info}")
print(f"RAM: {system_ram / (1024**3):.2f} GB")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Using CPU mode (slower).")
print("=========================\n")

# Whisper model information
WHISPER_MODELS = {
    'tiny': {
        'description': 'Fastest model (~32x realtime), lowest accuracy',
        'size_mb': 75,
        'req_cpu': 'Any modern CPU (2+ cores recommended)',
        'req_ram': '2 GB',
        'req_gpu': 'Not required, integrated GPU sufficient',
        'req_vram': '~1 GB VRAM (if using GPU)',
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt'
    },
    'base': {
        'description': 'Fast (~16x realtime), decent accuracy',
        'size_mb': 142,
        'req_cpu': 'Modern CPU with 4+ cores recommended',
        'req_ram': '4 GB',
        'req_gpu': 'Not required, but integrated GPU helpful',
        'req_vram': '~1 GB VRAM (if using GPU)',
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/4d70c13eb6c0d487dab0f48ecb2806ef7c5431dae0f4d1ea46e3d5119a9f60c7/base.en.pt'
    },
    'small': {
        'description': 'Good balance (~6x realtime), good accuracy',
        'size_mb': 466,
        'req_cpu': 'Modern multi-core CPU (6+ cores recommended)',
        'req_ram': '6 GB',
        'req_gpu': 'Recommended for decent performance',
        'req_vram': '~2 GB VRAM',
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/55356645c2b361a969dfd0ef2c5a50d530afd8105a9f54e8e8c8c5cc79a6ef98/small.en.pt'
    },
    'medium': {
        'description': 'Slower (~2x realtime), high accuracy',
        'size_mb': 1500,
        'req_cpu': 'High-end multi-core CPU',
        'req_ram': '8 GB',
        'req_gpu': 'Strongly recommended for reasonable speed',
        'req_vram': '~5 GB VRAM',
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.en.pt'
    },
    'large': {
        'description': 'Slowest (1x realtime), highest accuracy',
        'size_mb': 2900,
        'req_cpu': 'Very high-end multi-core CPU',
        'req_ram': '16 GB',
        'req_gpu': 'Required for reasonable performance',
        'req_vram': '~10 GB VRAM',
        'download_url': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v3.pt'
    }
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page with the upload form and model selection."""
    return render_template('index.html', models=WHISPER_MODELS, gpu_info=gpu_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload, check if the file is valid,
    and initiate the transcription process.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this transcription task
        task_id = str(uuid.uuid4())
        
        # Get selected model
        model_name = request.form.get('model', 'base')
        
        # Create a temporary file path
        filename = os.path.join(UPLOAD_FOLDER, f"{task_id}_{file.filename}")
        file.save(filename)
        
        # Get file size for better progress tracking
        file_size = os.path.getsize(filename)
        
        print(f"Processing file: {file.filename} (Size: {file_size/1024/1024:.2f} MB) with model: {model_name}")
        
        # Store task information with thread safety
        with task_lock:
            transcription_tasks[task_id] = {
                'status': 'preparing',
                'file': filename,
                'file_name': file.filename,
                'file_size': file_size,
                'model': model_name,
                'progress': 5,  # Start at 5% to show immediate feedback
                'transcript_txt': None,
                'transcript_srt': None,
                'error': None,
                'device_type': 'gpu' if gpu_info['available'] else 'cpu',
                'start_time': time.time(),
                'stage_info': 'Preparing transcription...',
                'detailed_status': f"Preparing to process {file.filename}"
            }
        
        try:
            # Start transcription in a background thread
            thread = threading.Thread(
                target=transcribe_file,
                args=(filename, model_name, task_id)
            )
            thread.daemon = True
            thread.start()
            
            # Small delay to ensure the thread has started
            time.sleep(0.1)
            
            return jsonify({'message': 'File uploaded successfully', 'task_id': task_id}), 200
        except Exception as e:
            print(f"Error starting transcription: {str(e)}")
            return jsonify({'error': f'Failed to start transcription: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

def download_model_with_progress(model_name, task_id):
    """
    Download a Whisper model with progress tracking.
    """
    # Check if model already exists in cache
    model_path = os.path.join(WHISPER_CACHE_DIR, f"{model_name}.pt")
    if os.path.exists(model_path):
        # Model already downloaded
        print(f"Model {model_name} already cached, skipping download")
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'model_ready'
                transcription_tasks[task_id]['progress'] = 30
                transcription_tasks[task_id]['stage_info'] = 'Model already downloaded'
                transcription_tasks[task_id]['detailed_status'] = f"Model {model_name} is already downloaded and ready to use"
        return True
    
    try:
        # Get model download URL
        download_url = WHISPER_MODELS[model_name]['download_url']
        model_size = WHISPER_MODELS[model_name]['size_mb'] * 1024 * 1024  # Convert to bytes
        
        print(f"Downloading model {model_name} ({WHISPER_MODELS[model_name]['size_mb']} MB) from {download_url}")
        
        # Create a temporary file for downloading
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download with progress tracking
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Update initial progress
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'downloading_model'
                transcription_tasks[task_id]['progress'] = 10
                transcription_tasks[task_id]['total_size'] = total_size
                transcription_tasks[task_id]['downloaded'] = 0
                transcription_tasks[task_id]['stage_info'] = 'Downloading model...'
                transcription_tasks[task_id]['detailed_status'] = f"Downloading {model_name} model ({WHISPER_MODELS[model_name]['size_mb']} MB)"
        
        # Download chunks with progress updates
        with open(temp_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress (10-30%)
                    progress = 10 + int(downloaded / total_size * 20) if total_size > 0 else 10
                    with task_lock:
                        if task_id in transcription_tasks:
                            transcription_tasks[task_id]['progress'] = progress
                            transcription_tasks[task_id]['downloaded'] = downloaded
                            transcription_tasks[task_id]['detailed_status'] = f"Downloading model: {downloaded/1024/1024:.1f} MB of {total_size/1024/1024:.1f} MB"
                    
                    # Print progress every 5MB
                    if downloaded % (5 * 1024 * 1024) < chunk_size:
                        print(f"Download progress: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB ({downloaded/total_size*100:.1f}%)")
        
        # Move the downloaded file to the cache directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.rename(temp_path, model_path)
        
        print(f"Model {model_name} downloaded successfully")
        
        # Update status after download
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'model_ready'
                transcription_tasks[task_id]['progress'] = 30
                transcription_tasks[task_id]['stage_info'] = 'Model downloaded'
                transcription_tasks[task_id]['detailed_status'] = f"Model {model_name} downloaded successfully"
        
        return True
        
    except Exception as e:
        # Handle download errors
        print(f"Error downloading model: {str(e)}")
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'error'
                transcription_tasks[task_id]['error'] = f"Model download error: {str(e)}"
                transcription_tasks[task_id]['detailed_status'] = f"Error downloading model: {str(e)}"
        return False

# Custom Whisper callback to track transcription progress
class ProgressCallback:
    def __init__(self, task_id):
        self.task_id = task_id
        self.last_progress_update = time.time()
    
    def __call__(self, progress):
        # Only update every 0.5 seconds to avoid excessive updates
        current_time = time.time()
        if current_time - self.last_progress_update >= 0.5:
            # Map from Whisper internal progress (0-1) to our progress range (50-95)
            ui_progress = 50 + int(progress * 45)
            
            with task_lock:
                if self.task_id in transcription_tasks:
                    transcription_tasks[self.task_id]['progress'] = ui_progress
                    transcription_tasks[self.task_id]['detailed_status'] = f"Transcribing: {progress*100:.1f}% complete"
                    
                    # Calculate remaining time estimate
                    if progress > 0.05:  # Only estimate after 5% progress
                        start_time = transcription_tasks[self.task_id].get('start_time', current_time)
                        elapsed = current_time - start_time
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                        if remaining > 0:
                            # Add time estimate to status
                            transcription_tasks[self.task_id]['time_remaining'] = remaining
                            if remaining < 60:
                                time_str = f"{int(remaining)} seconds"
                            else:
                                time_str = f"{int(remaining/60)} minutes {int(remaining%60)} seconds"
                            transcription_tasks[self.task_id]['detailed_status'] += f" (Est. {time_str} remaining)"
            
            # Print to console
            print(f"Transcription progress: {progress*100:.1f}% (UI progress: {ui_progress}%)")
            self.last_progress_update = current_time

def transcribe_file(file_path, model_name, task_id):
    """
    Background process to transcribe the audio/video file.
    Updates the task status throughout the process.
    """
    try:
        start_time = time.time()
        print(f"Starting transcription for task {task_id}, file: {file_path}, model: {model_name}")
        
        # Use the task_lock to prevent race conditions
        with task_lock:
            if task_id not in transcription_tasks:
                print(f"Error: Task ID {task_id} not found in transcription_tasks")
                return
            
            # Get file name for logging
            file_name = transcription_tasks[task_id].get('file_name', os.path.basename(file_path))
            
            # Update initial task status
            transcription_tasks[task_id]['status'] = 'preparing'
            transcription_tasks[task_id]['progress'] = 10
            transcription_tasks[task_id]['device_type'] = 'gpu' if gpu_info['available'] else 'cpu'
            transcription_tasks[task_id]['stage_info'] = 'Checking model availability'
            transcription_tasks[task_id]['detailed_status'] = f"Checking if {model_name} model is available"
        
        # Download model with progress tracking
        if not download_model_with_progress(model_name, task_id):
            # If download failed, the error is already set in the task
            return
        
        # Update task status - loading model
        print(f"Loading {model_name} model...")
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['status'] = 'loading_model'
            transcription_tasks[task_id]['progress'] = 35
            transcription_tasks[task_id]['stage_info'] = 'Loading model'
            transcription_tasks[task_id]['detailed_status'] = f"Loading {model_name} model into memory"
        
        # Load the model with appropriate device
        device = "cuda" if gpu_info['available'] else "cpu"
        model = whisper.load_model(model_name, device=device, download_root=WHISPER_CACHE_DIR)
        
        # Update task status - transcribing
        print(f"Starting transcription of {file_name}...")
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['status'] = 'transcribing'
            transcription_tasks[task_id]['progress'] = 50
            transcription_tasks[task_id]['stage_info'] = 'Transcribing audio'
            transcription_tasks[task_id]['detailed_status'] = f"Transcribing {file_name} using {device.upper()}"
        
        # Create progress callback
        progress_callback = ProgressCallback(task_id)
        
        # Perform transcription with callback for progress updates
        result = model.transcribe(
            file_path, 
            fp16=(device == "cuda"),  # Use fp16 on GPU for speed
            verbose=False,  # We handle progress reporting ourselves
            progress_callback=progress_callback
        )
        
        # Generate text transcript
        txt_filename = os.path.join(TRANSCRIPT_FOLDER, f"{task_id}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        # Update status to creating SRT
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['progress'] = 95
            transcription_tasks[task_id]['stage_info'] = 'Creating SRT file'
            transcription_tasks[task_id]['detailed_status'] = "Creating subtitle file in SRT format"
        
        # Generate SRT transcript
        srt_filename = os.path.join(TRANSCRIPT_FOLDER, f"{task_id}.srt")
        with open(srt_filename, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"], start=1):
                # Format: sequence number, time range, text
                f.write(f"{i}\n")
                start_time = format_time(segment["start"])
                end_time = format_time(segment["end"])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"Transcription completed in {total_time:.2f} seconds")
        
        # Update task status - completed
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['status'] = 'completed'
            transcription_tasks[task_id]['progress'] = 100
            transcription_tasks[task_id]['transcript_txt'] = txt_filename
            transcription_tasks[task_id]['transcript_srt'] = srt_filename
            transcription_tasks[task_id]['stage_info'] = 'Transcription complete'
            transcription_tasks[task_id]['detailed_status'] = f"Transcription completed in {total_time:.1f} seconds"
            transcription_tasks[task_id]['completion_time'] = time.time()
            transcription_tasks[task_id]['processing_time'] = total_time
        
    except Exception as e:
        # Update task status - error
        print(f"Transcription error: {str(e)}")
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'error'
                transcription_tasks[task_id]['error'] = str(e)
                transcription_tasks[task_id]['detailed_status'] = f"Error: {str(e)}"
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {str(e)}")

def format_time(seconds):
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get the status of a transcription task."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id].copy()  # Make a copy to avoid race conditions
    
    response = {
        'status': task['status'],
        'progress': task['progress'],
        'error': task.get('error'),
        'device_type': task.get('device_type', 'unknown'),
        'stage_info': task.get('stage_info', ''),
        'detailed_status': task.get('detailed_status', '')
    }
    
    # Add processing time if available
    if 'processing_time' in task:
        response['processing_time'] = task['processing_time']
    
    # Add remaining time estimate if available
    if 'time_remaining' in task:
        response['time_remaining'] = task['time_remaining']
    
    # Add download progress details if available and in download state
    if task['status'] == 'downloading_model' and 'downloaded' in task and 'total_size' in task:
        response['downloaded'] = task['downloaded']
        response['total_size'] = task['total_size']
        response['downloaded_mb'] = round(task['downloaded'] / (1024 * 1024), 2)
        response['total_mb'] = round(task['total_size'] / (1024 * 1024), 2)
    
    if task['status'] == 'completed':
        response['transcript'] = task['transcript_txt']
        
    return jsonify(response), 200

@app.route('/system_info', methods=['GET'])
def get_system_info():
    """Get information about the system capabilities."""
    # Use the hardware info function we defined at the top
    # This was causing recursion in the previous version
    updated_gpu_info, updated_cpu_info, updated_ram = get_hardware_info()
    
    # Get list of cached models
    cached_models = [f.name for f in os.scandir(WHISPER_CACHE_DIR) 
                     if f.is_file() and f.name.endswith('.pt')]
    
    return jsonify({
        'gpu': updated_gpu_info,
        'cpu_info': updated_cpu_info,
        'ram': updated_ram,
        'pytorch_version': torch.__version__,
        'whisper_cache_dir': WHISPER_CACHE_DIR,
        'cached_models': cached_models
    }), 200

@app.route('/transcript/<task_id>', methods=['GET'])
def get_transcript_text(task_id):
    """Get the transcript text content."""
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
    """Download the transcript in the requested format (txt or srt)."""
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id]
        
        if task['status'] != 'completed':
            return jsonify({'error': 'Transcription not completed yet'}), 400
        
        if format_type == 'txt':
            file_path = task['transcript_txt']
            mimetype = 'text/plain'
            attachment_filename = f"transcript_{task_id}.txt"
        elif format_type == 'srt':
            file_path = task['transcript_srt']
            mimetype = 'text/srt'
            attachment_filename = f"transcript_{task_id}.srt"
        else:
            return jsonify({'error': 'Invalid format type'}), 400
    
    return send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=attachment_filename
    )

@app.route('/cleanup/<task_id>', methods=['POST'])
def cleanup_task(task_id):
    """
    Clean up the transcription task files after the user has downloaded them.
    This helps free up server space.
    """
    with task_lock:
        if task_id not in transcription_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = transcription_tasks[task_id]
        
        # Remove transcript files if they exist
        if task['transcript_txt'] and os.path.exists(task['transcript_txt']):
            try:
                os.remove(task['transcript_txt'])
            except:
                pass
        
        if task['transcript_srt'] and os.path.exists(task['transcript_srt']):
            try:
                os.remove(task['transcript_srt'])
            except:
                pass
        
        # Remove the task from the dictionary
        del transcription_tasks[task_id]
    
    return jsonify({'message': 'Task resources cleaned up successfully'}), 200

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about all available Whisper models."""
    return jsonify(WHISPER_MODELS), 200
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
