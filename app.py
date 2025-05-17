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
from tqdm import tqdm
from pathlib import Path

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
TRANSCRIPT_FOLDER = os.path.join(os.getcwd(), 'transcripts')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'avi', 'mov', 'flac', 'ogg', 'm4a'}
WHISPER_CACHE_DIR = os.path.join(os.getcwd(), 'whisper_models')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)

# Store ongoing transcription tasks
transcription_tasks = {}
task_lock = threading.Lock()  # Add a lock for thread safety

# Check if GPU is available
has_gpu = torch.cuda.is_available()
gpu_info = {
    'available': has_gpu,
    'name': torch.cuda.get_device_name(0) if has_gpu else None,
    'count': torch.cuda.device_count() if has_gpu else 0,
    'memory': torch.cuda.get_device_properties(0).total_memory if has_gpu else 0,
    'cuda_version': torch.version.cuda if has_gpu else None
} if has_gpu else {
    'available': False,
    'name': None,
    'count': 0,
    'memory': 0,
    'cuda_version': None
}

# Print GPU info at startup
print("\n==== GPU DETECTION ====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. Using CPU mode (slower).")
print("======================\n")

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
        
        # Store task information with thread safety
        with task_lock:
            transcription_tasks[task_id] = {
                'status': 'starting',
                'file': filename,
                'model': model_name,
                'progress': 0,
                'transcript_txt': None,
                'transcript_srt': None,
                'error': None,
                'device_type': 'gpu' if has_gpu else 'cpu'
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
        return True
    
    try:
        # Get model download URL
        download_url = WHISPER_MODELS[model_name]['download_url']
        model_size = WHISPER_MODELS[model_name]['size_mb'] * 1024 * 1024  # Convert to bytes
        
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
                transcription_tasks[task_id]['progress'] = 0
                transcription_tasks[task_id]['total_size'] = total_size
                transcription_tasks[task_id]['downloaded'] = 0
        
        # Download chunks with progress updates
        with open(temp_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress (0-40%)
                    progress = int(downloaded / total_size * 40) if total_size > 0 else 0
                    with task_lock:
                        if task_id in transcription_tasks:
                            transcription_tasks[task_id]['progress'] = progress
                            transcription_tasks[task_id]['downloaded'] = downloaded
        
        # Move the downloaded file to the cache directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.rename(temp_path, model_path)
        
        return True
        
    except Exception as e:
        # Handle download errors
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'error'
                transcription_tasks[task_id]['error'] = f"Model download error: {str(e)}"
        return False

def transcribe_file(file_path, model_name, task_id):
    """
    Background process to transcribe the audio/video file.
    Updates the task status throughout the process.
    """
    try:
        # Use the task_lock to prevent race conditions
        with task_lock:
            if task_id not in transcription_tasks:
                print(f"Error: Task ID {task_id} not found in transcription_tasks")
                return
                
            # Update initial task status
            transcription_tasks[task_id]['status'] = 'preparing'
            transcription_tasks[task_id]['progress'] = 0
            transcription_tasks[task_id]['device_type'] = 'gpu' if has_gpu else 'cpu'
        
        # Download model with progress tracking
        if not download_model_with_progress(model_name, task_id):
            # If download failed, the error is already set in the task
            return
        
        # Update task status - loading model
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['status'] = 'loading_model'
            transcription_tasks[task_id]['progress'] = 45
        
        # Load the model with appropriate device
        device = "cuda" if has_gpu else "cpu"
        model = whisper.load_model(model_name, device=device, download_root=WHISPER_CACHE_DIR)
        
        # Update task status - transcribing
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['status'] = 'transcribing'
            transcription_tasks[task_id]['progress'] = 50
        
        # Perform transcription
        result = model.transcribe(file_path)
        
        # Generate text transcript
        txt_filename = os.path.join(TRANSCRIPT_FOLDER, f"{task_id}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
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
        
        # Update task status - completed
        with task_lock:
            if task_id not in transcription_tasks:
                return
            transcription_tasks[task_id]['status'] = 'completed'
            transcription_tasks[task_id]['progress'] = 100
            transcription_tasks[task_id]['transcript_txt'] = txt_filename
            transcription_tasks[task_id]['transcript_srt'] = srt_filename
        
    except Exception as e:
        # Update task status - error
        print(f"Transcription error: {str(e)}")
        with task_lock:
            if task_id in transcription_tasks:
                transcription_tasks[task_id]['status'] = 'error'
                transcription_tasks[task_id]['error'] = str(e)
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

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
        'device_type': task.get('device_type', 'unknown')
    }
    
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
    # Force check CUDA availability again
    cuda_available = torch.cuda.is_available()
    if cuda_available and not gpu_info['available']:
        # Update GPU info if it's now available but wasn't before
        gpu_info['available'] = True
        gpu_info['name'] = torch.cuda.get_device_name(0)
        gpu_info['count'] = torch.cuda.device_count()
        gpu_info['memory'] = torch.cuda.get_device_properties(0).total_memory
        gpu_info['cuda_version'] = torch.version.cuda
    
    return jsonify({
        'gpu': gpu_info,
        'pytorch_version': torch.__version__,
        'whisper_cache_dir': WHISPER_CACHE_DIR,
        'cached_models': [f.name for f in os.scandir(WHISPER_CACHE_DIR) if f.is_file() and f.name.endswith('.pt')]
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