
// main.js - Updated for better progress tracking and UI feedback
document.addEventListener("DOMContentLoaded", function () {
  // DOM elements - Main functionality
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("file");
  const dropArea = document.getElementById("drop-area");
  const filePrompt = document.getElementById("file-prompt");
  const fileInfo = document.getElementById("file-info");
  const filename = document.getElementById("filename");
  const filesize = document.getElementById("filesize");
  const removeFileBtn = document.getElementById("remove-file");
  const submitBtn = document.getElementById("submit-btn");
  const statusContainer = document.getElementById("status-container");
  const statusMessage = document.getElementById("status-message");
  const progressBar = document.getElementById("progress-bar");
  const progressPercentage = document.getElementById("progress-percentage-circle");
  const progressCircle = document.getElementById("progress-circle");
  const loadingSpinner = document.getElementById("loading-spinner");
  const resultsContainer = document.getElementById("results-container");
  const transcript = document.getElementById("transcript");
  const downloadTxtBtn = document.getElementById("download-txt");
  const downloadSrtBtn = document.getElementById("download-srt");
  const newTranscriptionBtn = document.getElementById("new-transcription");
  const errorContainer = document.getElementById("error-container");
  const errorMessage = document.getElementById("error-message");
  const tryAgainBtn = document.getElementById("try-again");
  const modelSelect = document.getElementById("model");
  const modelInfo = document.getElementById("model-info");
  const deviceType = document.getElementById("device-type");
  
  // DOM elements - New features
  const downloadProgressInfo = document.getElementById("download-progress-info");
  const downloadSize = document.getElementById("download-size");
  const totalSize = document.getElementById("total-size");
  const copyTranscriptBtn = document.getElementById("copy-transcript");
  const copySuccessMessage = document.getElementById("copy-success-message");
  const deviceIndicator = document.getElementById("device-indicator");
  const hardwareStatus = document.getElementById("hardware-status");
  const cpuInfo = document.getElementById("cpu-info");
  const ramInfo = document.getElementById("ram-info");
  const gpuInfo = document.getElementById("gpu-info");
  const vramInfo = document.getElementById("vram-info");
  const pytorchVersion = document.getElementById("pytorch-version");
  const cudaVersion = document.getElementById("cuda-version");
  const gpuTroubleshooting = document.getElementById("gpu-troubleshooting");

  // Circle progress setup
  const radius = 50;
  const circumference = radius * 2 * Math.PI;
  if (progressCircle) {
    progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    progressCircle.style.strokeDashoffset = `${circumference}`;
  }

  // Store the current task ID
  let currentTaskId = null;
  
  // Store the current status polling interval
  let statusInterval = null;
  
  // Store system info
  let systemInfo = null;
  
  // Enhanced logging
  const DEBUG = true; // Set to true to enable detailed console logging
  
  function log(message, type = 'info') {
    if (DEBUG) {
      switch (type) {
        case 'error':
          console.error(`[Whispr] ${message}`);
          break;
        case 'warn':
          console.warn(`[Whispr] ${message}`);
          break;
        case 'success':
          console.log(`%c[Whispr] ${message}`, 'color: #76B900');
          break;
        default:
          console.log(`[Whispr] ${message}`);
      }
    }
  }
  
  // Initialize system information
  fetchSystemInfo();
  
  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    if (dropArea) {
      dropArea.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    }
  });
  
  // Highlight drop area when dragging over it
  if (dropArea) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropArea.addEventListener(eventName, highlight, false);
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropArea.addEventListener("drop", handleDrop, false);
  }
  
  // Handle file input change
  if (fileInput) {
    fileInput.addEventListener("change", handleFileSelect);
  }
  
  // Handle remove file button
  if (removeFileBtn) {
    removeFileBtn.addEventListener("click", removeFile);
  }
  
  // Form submission
  if (uploadForm) {
    uploadForm.addEventListener("submit", handleSubmit);
  }
  
  // Download buttons
  if (downloadTxtBtn) {
    downloadTxtBtn.addEventListener("click", () => downloadTranscript("txt"));
  }
  
  if (downloadSrtBtn) {
    downloadSrtBtn.addEventListener("click", () => downloadTranscript("srt"));
  }
  
  // New transcription button
  if (newTranscriptionBtn) {
    newTranscriptionBtn.addEventListener("click", resetUI);
  }
  
  // Try again button
  if (tryAgainBtn) {
    tryAgainBtn.addEventListener("click", resetUI);
  }
  
  // Model selection change
  if (modelSelect) {
    modelSelect.addEventListener("change", updateModelInfo);
  }
  
  // Copy transcript button
  if (copyTranscriptBtn) {
    copyTranscriptBtn.addEventListener("click", copyTranscriptToClipboard);
  }
  
  // Update model info on page load
  if (modelInfo) {
    updateModelInfo();
  }
  
  // Prevent default drag behavior
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  
  // Highlight drop area
  function highlight() {
    dropArea.classList.add("border-nvidia-green", "bg-nvidia-900/10");
    dropArea.classList.remove("border-dark-300");
  }
  
  // Remove highlight from drop area
  function unhighlight() {
    dropArea.classList.remove("border-nvidia-green", "bg-nvidia-900/10");
    dropArea.classList.add("border-dark-300");
  }
  
  // Handle dropped files
  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      updateFileInfo(files[0]);
    }
  }
  
  // Handle file selection
  function handleFileSelect() {
    if (fileInput.files.length > 0) {
      updateFileInfo(fileInput.files[0]);
    }
  }
  
  // Update file information display
  function updateFileInfo(file) {
    filename.textContent = file.name;
    filesize.textContent = formatFileSize(file.size);
    filePrompt.classList.add("hidden");
    fileInfo.classList.remove("hidden");
    log(`File selected: ${file.name} (${formatFileSize(file.size)})`);
  }
  
  // Format file size (bytes to human readable)
  function formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }
  
  // Remove selected file
  function removeFile() {
    fileInput.value = "";
    filePrompt.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    log("File removed");
  }
  
  // Fetch system information
  function fetchSystemInfo() {
    fetch("/system_info")
      .then((response) => response.json())
      .then((info) => {
        log("System info received:", "success");
        log(JSON.stringify(info, null, 2));
        systemInfo = info;
        updateSystemInfo(info);
        updateModelCompatibility(info);
      })
      .catch((error) => {
        log(`Error fetching system info: ${error}`, "error");
      });
  }
  
  // Update system information display
  function updateSystemInfo(info) {
    const gpuAvailable = info.gpu.available;
    
    // Update PyTorch and CUDA info
    if (pytorchVersion) {
      pytorchVersion.textContent = info.pytorch_version || "Not installed";
    }
    
    if (cudaVersion) {
      cudaVersion.textContent = info.gpu.cuda_version || "Not available";
    }
    
    // Update RAM info
    if (ramInfo) {
      ramInfo.textContent = info.ram ? formatBytes(info.ram) : "Unknown";
    }
    
    // Update CPU info
    if (cpuInfo) {
      cpuInfo.textContent = info.cpu_info || "Unknown";
    }
    
    if (gpuAvailable) {
      // GPU is available
      if (gpuInfo) {
        gpuInfo.textContent = info.gpu.name || "Unknown GPU";
      }
      
      if (vramInfo) {
        vramInfo.textContent = formatBytes(info.gpu.memory);
      }
      
      if (gpuTroubleshooting) {
        gpuTroubleshooting.classList.add("hidden");
      }
      
      // Update device indicator
      if (deviceIndicator) {
        deviceIndicator.textContent = "GPU Enabled";
        deviceIndicator.className = "ml-4 px-3 py-1 text-xs font-medium rounded-full bg-nvidia-900/30 text-nvidia-green";
      }
      
      // Update hardware status
      if (hardwareStatus) {
        hardwareStatus.textContent = "GPU Detected: " + info.gpu.name;
        hardwareStatus.className = "inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-nvidia-900/30 text-nvidia-green";
      }
    } else {
      // No GPU available
      if (gpuInfo) {
        gpuInfo.textContent = "Not detected";
      }
      
      if (vramInfo) {
        vramInfo.textContent = "N/A";
      }
      
      if (gpuTroubleshooting) {
        gpuTroubleshooting.classList.remove("hidden");
      }
      
      // Update device indicator
      if (deviceIndicator) {
        deviceIndicator.textContent = "CPU Only Mode";
        deviceIndicator.className = "ml-4 px-3 py-1 text-xs font-medium rounded-full bg-yellow-900/30 text-yellow-400";
      }
      
      // Update hardware status
      if (hardwareStatus) {
        hardwareStatus.textContent = "⚠️ No GPU Detected - Running in CPU mode (slower)";
        hardwareStatus.className = "inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-yellow-900/30 text-yellow-400";
      }
    }
  }
  
  // Update model compatibility based on system info
  function updateModelCompatibility(info) {
    const models = ["tiny", "base", "small", "medium", "large"];
    const gpuAvailable = info.gpu.available;
    const cachedModels = info.cached_models || [];
    
    models.forEach((model) => {
      const statusElement = document.getElementById(`model-status-${model}`);
      if (!statusElement) return;
      
      let statusText = "";
      let statusClass = "";
      
      // Check if model is already cached
      const isModelCached = cachedModels.some((cache) => cache.includes(model));
      
      if (model === "tiny" || model === "base") {
        // These models work on almost any hardware
        statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
        statusClass = "bg-nvidia-900/30 text-nvidia-green";
      } else if (model === "small") {
        if (gpuAvailable) {
          statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
          statusClass = "bg-nvidia-900/30 text-nvidia-green";
        } else {
          statusText = "Compatible (Slow)";
          statusClass = "bg-yellow-900/30 text-yellow-400";
        }
      } else if (model === "medium") {
        if (gpuAvailable) {
          statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
          statusClass = "bg-nvidia-900/30 text-nvidia-green";
        } else {
          statusText = "CPU Only (Very Slow)";
          statusClass = "bg-yellow-900/30 text-yellow-400";
        }
      } else if (model === "large") {
        if (gpuAvailable) {
          statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
          statusClass = "bg-nvidia-900/30 text-nvidia-green";
        } else {
          statusText = "Not Recommended";
          statusClass = "bg-red-900/30 text-red-400";
        }
      }
      
      statusElement.textContent = statusText;
      statusElement.className = `px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${statusClass}`;
    });
  }
  
  // Format bytes
  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }
  
  // Format time remaining
  function formatTimeRemaining(seconds) {
    if (seconds < 60) {
      return `${Math.round(seconds)} seconds remaining`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      return `${minutes} min ${secs} sec remaining`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours} hr ${minutes} min remaining`;
    }
  }
  
  // Update model information based on selection
  function updateModelInfo() {
    const selectedModel = modelSelect.value;
    
    // Get model descriptions from the server
    fetch("/model_info")
      .then((response) => response.json())
      .then((models) => {
        const model = models[selectedModel];
        if (model) {
          const sizeInGB = (model.size_mb / 1024).toFixed(1);
          let warningText = "";
          
          // Add warning for large models
          if (model.size_mb > 1000) {
            warningText = ` ⚠️ This is a large model (${sizeInGB} GB) and may take a while to download if not already cached.`;
          }
          
          modelInfo.textContent = `${model.description}${warningText}`;
        }
      })
      .catch((error) => log(`Error fetching model info: ${error}`, "error"));
  }
  
  // Handle form submission
  function handleSubmit(e) {
    e.preventDefault();
    
    // Check if file is selected
    if (fileInput.files.length === 0) {
      showError("Please select a file to transcribe.");
      return;
    }
    
    // Prepare form data
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model", modelSelect.value);
    
    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.classList.add("opacity-50", "cursor-not-allowed");
    
    // Reset and show status container
    resetStatus();
    hideError();
    resultsContainer.classList.add("hidden");
    statusContainer.classList.remove("hidden");
    updateStatus("Uploading file...", 0, "Uploading");
    
    log(`Starting transcription with model: ${modelSelect.value}`, "success");
    
    // Send form data using XMLHttpRequest to track upload progress
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");

    xhr.upload.addEventListener("progress", (ev) => {
      if (ev.lengthComputable) {
        const percent = (ev.loaded / ev.total) * 100;
        const scaled = (percent / 100) * 10; // scale 0-10% of overall progress
        updateStatus(`Uploading file... (${percent.toFixed(0)}%)`, scaled);
      }
    });

    xhr.onload = function () {
      if (xhr.status === 200) {
        try {
          const data = JSON.parse(xhr.responseText);
          log("Upload successful: " + JSON.stringify(data), "success");
          currentTaskId = data.task_id;
          startStatusPolling();
        } catch (err) {
          showError("Unexpected server response.");
        }
      } else {
        showError(`Error uploading file: ${xhr.statusText}`);
      }
    };

    xhr.onerror = function () {
      showError("Network error during upload.");
      submitBtn.disabled = false;
      submitBtn.classList.remove("opacity-50", "cursor-not-allowed");
    };

    xhr.send(formData);
  }
  
  // Start polling for transcription status
  function startStatusPolling() {
    if (statusInterval) {
      clearInterval(statusInterval);
    }
    
    statusInterval = setInterval(() => {
      if (!currentTaskId) {
        clearInterval(statusInterval);
        return;
      }
      
      fetch(`/status/${currentTaskId}`)
        .then((response) => response.json())
        .then((data) => {
          updateStatusFromData(data);
          
          // If completed or error, stop polling
          if (data.status === "completed" || data.status === "error") {
            clearInterval(statusInterval);
            
            if (data.status === "completed") {
              fetchTranscript();
              log(`Transcription completed in ${data.processing_time ? data.processing_time.toFixed(1) : '?'} seconds`, "success");
            }
            
            if (data.status === "error") {
              showError(`Transcription error: ${data.error}`);
              log(`Transcription error: ${data.error}`, "error");
            }
          }
        })
        .catch((error) => {
          log(`Error checking status: ${error}`, "error");
          clearInterval(statusInterval);
          showError(`Error checking status: ${error.message}`);
        });
    }, 500); // Poll every 0.5 seconds for smoother updates
  }
  
  // Update status based on data from server
  function updateStatusFromData(data) {
    let progress = data.progress || 0;
    let stageInfo = data.stage_info || "";
    let detailedStatus = data.detailed_status || "";
    
    // Log status update
    log(`Status: ${data.status}, Progress: ${progress}%, Stage: ${stageInfo}`);
    
    // Update device type indicator
    if (deviceType) {
      if (data.device_type === "gpu") {
        deviceType.textContent = "Using GPU";
        deviceType.className = "text-sm font-medium px-3 py-1 rounded-full bg-nvidia-900/30 text-nvidia-green";
      } else {
        deviceType.textContent = "Using CPU";
        deviceType.className = "text-sm font-medium px-3 py-1 rounded-full bg-yellow-900/30 text-yellow-400";
      }
    }
    
    // Handle download progress display
    if (
      data.status === "downloading_model" &&
      data.downloaded !== undefined &&
      data.total_size !== undefined
    ) {
      downloadProgressInfo.classList.remove("hidden");
      downloadSize.textContent = formatBytes(data.downloaded);
      totalSize.textContent = formatBytes(data.total_size);
    } else {
      downloadProgressInfo.classList.add("hidden");
    }
    
    // Create status text based on stage_info and detailed_status
    let statusText = detailedStatus || stageInfo;
    
    // Add time remaining if available
    if (data.time_remaining && data.status === 'transcribing') {
      statusText += ` (${formatTimeRemaining(data.time_remaining)})`;
    }
    
    updateStatus(statusText, progress, stageInfo);
  }
  
  // Update status display
  function updateStatus(message, progressValue, stageInfo = "") {
    // Update main status message
    statusMessage.textContent = message;
    
    // Update progress bar
    progressBar.style.width = `${progressValue}%`;
    
    // Update circular progress indicator
    if (progressCircle) {
      const offset = circumference - (progressValue / 100) * circumference;
      progressCircle.style.strokeDashoffset = offset;
      progressPercentage.textContent = `${progressValue.toFixed(0)}%`;
    }
    
    // Hide loading spinner at 100%
    if (progressValue === 100) {
      loadingSpinner.classList.add("hidden");
    } else {
      loadingSpinner.classList.remove("hidden");
    }
  }
  
  // Reset status display
  function resetStatus() {
    updateStatus("Preparing...", 0);
  }
  
  // Fetch transcript when transcription is complete
  function fetchTranscript() {
    fetch(`/transcript/${currentTaskId}`)
      .then((response) => response.json())
      .then((data) => {
        log("Transcript fetched successfully");
        transcript.value = data.transcript;
        statusContainer.classList.add("hidden");
        resultsContainer.classList.remove("hidden");
      })
      .catch((error) => {
        log(`Error fetching transcript: ${error}`, "error");
        showError(`Error fetching transcript: ${error.message}`);
      });
  }
  
  // Download transcript
  function downloadTranscript(format) {
    if (!currentTaskId) return;
    
    const a = document.createElement("a");
    a.href = `/download/${currentTaskId}/${format}`;
    a.download = `transcript.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    log(`Downloading transcript as ${format}`, "success");
    
    // Clean up server resources
    setTimeout(() => {
      fetch(`/cleanup/${currentTaskId}`, { method: "POST" })
        .then((response) => response.json())
        .then((data) => log("Cleanup successful: " + JSON.stringify(data)))
        .catch((error) => log(`Error during cleanup: ${error}`, "error"));
    }, 1000);
  }
  
  // Copy transcript to clipboard
  function copyTranscriptToClipboard() {
    if (!transcript.value) return;
    
    navigator.clipboard
      .writeText(transcript.value)
      .then(() => {
        log("Transcript copied to clipboard", "success");
        copySuccessMessage.classList.remove("hidden");
        
        // Hide success message after 2 seconds
        setTimeout(() => {
          copySuccessMessage.classList.add("hidden");
        }, 2000);
      })
      .catch((error) => {
        log(`Error copying transcript: ${error}`, "error");
        showError(`Error copying transcript: ${error.message}`);
      });
  }
  
  // Show error message
  function showError(message) {
    errorMessage.textContent = message;
    errorContainer.classList.remove("hidden");
    statusContainer.classList.add("hidden");
    log(message, "error");
  }
  
  // Hide error message
  function hideError() {
    errorContainer.classList.add("hidden");
  }
  
  // Reset UI for new transcription
  function resetUI() {
    // Clear current task
    currentTaskId = null;
    
    // Clear interval if running
    if (statusInterval) {
      clearInterval(statusInterval);
      statusInterval = null;
    }
    
    // Reset form
    uploadForm.reset();
    filePrompt.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    
    // Hide containers
    statusContainer.classList.add("hidden");
    resultsContainer.classList.add("hidden");
    errorContainer.classList.add("hidden");
    
    // Enable submit button
    submitBtn.disabled = false;
    submitBtn.classList.remove("opacity-50", "cursor-not-allowed");
    
    log("UI reset for new transcription");
  }

  // Mobile menu toggle
  const mobileMenuButton = document.getElementById("mobile-menu-button");
  const mobileMenu = document.getElementById("mobile-menu");
  if (mobileMenuButton && mobileMenu) {
    mobileMenuButton.addEventListener("click", function () {
      mobileMenu.classList.toggle("hidden");
    });
  }
  
  // Toggle options panel
  const toggleOptions = document.getElementById("toggle-options");
  const optionsPanel = document.getElementById("options-panel");
  if (toggleOptions && optionsPanel) {
    toggleOptions.addEventListener("click", function () {
      if (optionsPanel.classList.contains("hidden")) {
        optionsPanel.classList.remove("hidden");
        toggleOptions.innerHTML = '<i class="fas fa-times mr-1"></i> Hide options';
      } else {
        optionsPanel.classList.add("hidden");
        toggleOptions.innerHTML = '<i class="fas fa-cog mr-1"></i> Show options';
      }
    });
  }
  
  // Accordion functionality for FAQs
  const accordionHeaders = document.querySelectorAll(".accordion-header");
  accordionHeaders.forEach((header) => {
    header.addEventListener("click", function () {
      const content = this.nextElementSibling;
      const icon = this.querySelector("i");
      
      // Close all other accordions
      accordionHeaders.forEach((otherHeader) => {
        if (otherHeader !== header) {
          const otherContent = otherHeader.nextElementSibling;
          const otherIcon = otherHeader.querySelector("i");
          otherContent.style.maxHeight = null;
          otherIcon.classList.remove("transform", "rotate-180");
        }
      });
      
      // Toggle current accordion
      if (content.style.maxHeight) {
        content.style.maxHeight = null;
        icon.classList.remove("transform", "rotate-180");
      } else {
        content.style.maxHeight = content.scrollHeight + "px";
        icon.classList.add("transform", "rotate-180");
      }
    });
  });
  
  // Open first FAQ by default
  if (accordionHeaders.length > 0) {
    const firstHeader = accordionHeaders[0];
    const firstContent = firstHeader.nextElementSibling;
    const firstIcon = firstHeader.querySelector("i");
    firstContent.style.maxHeight = firstContent.scrollHeight + "px";
    firstIcon.classList.add("transform", "rotate-180");
  }

  // Initialize
  log("Whispr Transcription Studio initialized", "success");
});
    