/*
 * Whispr - Open Source Transcription Studio
 * Frontend JavaScript for real-time transcription interface
 *
 * Features:
 * - Real-time progress tracking with accurate progress bars
 * - Live console output showing transcription status
 * - Drag and drop file upload with validation
 * - System hardware detection and model compatibility checking
 * - Download transcript in TXT/SRT formats
 *
 * Author: Mohd Mahmodi
 * License: MIT
 * GitHub: https://github.com/mohdmahmodi/whispr
 */

document.addEventListener("DOMContentLoaded", function () {
  // ================================
  // DOM Element References
  // ================================

  // File upload elements
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("file");
  const dropArea = document.getElementById("drop-area");
  const filePrompt = document.getElementById("file-prompt");
  const fileInfo = document.getElementById("file-info");
  const filename = document.getElementById("filename");
  const filesize = document.getElementById("filesize");
  const removeFileBtn = document.getElementById("remove-file");
  const submitBtn = document.getElementById("submit-btn");

  // Progress and status elements
  const statusContainer = document.getElementById("status-container");
  const statusMessage = document.getElementById("status-message");
  const progressBar = document.getElementById("progress-bar");
  const progressPercentageCircle = document.getElementById(
    "progress-percentage-circle"
  );
  const progressCircle = document.getElementById("progress-circle");
  const loadingSpinner = document.getElementById("loading-spinner");

  // Results and transcript elements
  const resultsContainer = document.getElementById("results-container");
  const transcript = document.getElementById("transcript");
  const downloadTxtBtn = document.getElementById("download-txt");
  const downloadSrtBtn = document.getElementById("download-srt");
  const newTranscriptionBtn = document.getElementById("new-transcription");
  const copyTranscriptBtn = document.getElementById("copy-transcript");
  const copySuccessMessage = document.getElementById("copy-success-message");

  // Error handling elements
  const errorContainer = document.getElementById("error-container");
  const errorMessage = document.getElementById("error-message");
  const tryAgainBtn = document.getElementById("try-again");

  // Model and device elements
  const modelSelect = document.getElementById("model");
  const modelInfo = document.getElementById("model-info");
  const deviceType = document.getElementById("device-type");
  const deviceIndicator = document.getElementById("device-indicator");

  // Download progress elements
  const downloadProgressInfo = document.getElementById(
    "download-progress-info"
  );
  const downloadSize = document.getElementById("download-size");
  const totalSize = document.getElementById("total-size");

  // Hardware information elements
  const hardwareStatus = document.getElementById("hardware-status");
  const cpuInfo = document.getElementById("cpu-info");
  const ramInfo = document.getElementById("ram-info");
  const gpuInfo = document.getElementById("gpu-info");
  const vramInfo = document.getElementById("vram-info");
  const pytorchVersion = document.getElementById("pytorch-version");
  const cudaVersion = document.getElementById("cuda-version");
  const gpuTroubleshooting = document.getElementById("gpu-troubleshooting");

  // ================================
  // Global State Variables
  // ================================

  let currentTaskId = null; // Current transcription task ID
  let statusInterval = null; // Status polling interval
  let systemInfo = null; // System hardware information
  let consoleDisplay = null; // Console display element

  // Progress circle configuration
  const radius = 50;
  const circumference = radius * 2 * Math.PI;

  // Application configuration
  const DEBUG = true; // Enable debug logging
  const ALLOWED_FILE_TYPES = [
    "mp3",
    "wav",
    "mp4",
    "avi",
    "mov",
    "flac",
    "ogg",
    "m4a",
  ];
  const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB limit

  // ================================
  // Utility Functions
  // ================================

  function log(message, type = "info") {
    /**
     * Enhanced logging with timestamps and color coding
     */
    if (DEBUG) {
      const timestamp = new Date().toLocaleTimeString();
      const colors = {
        error: "#FF6B6B",
        warn: "#FFE66D",
        success: "#76B900",
        info: "#4ECDC4",
      };
      console.log(
        `%c[${timestamp}] [Whispr] ${message}`,
        `color: ${colors[type] || colors.info}`
      );
    }
  }

  function formatFileSize(bytes) {
    /**
     * Convert bytes to human-readable file size
     */
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function formatBytes(bytes) {
    /**
     * Format bytes for system information display
     */
    if (!bytes) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function formatTimeRemaining(seconds) {
    /**
     * Format seconds into human-readable time remaining
     */
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

  // ================================
  // Initialization Functions
  // ================================

  function init() {
    /**
     * Initialize the application
     */
    log("Initializing Whispr Transcription Studio", "success");

    // Setup progress circle
    initializeProgressCircle();

    // Load system information
    fetchSystemInfo();

    // Setup all event listeners
    setupEventListeners();
    setupFileHandling();
    setupUIToggles();

    // Load model information
    updateModelInfo();

    log("Application initialized successfully", "success");
  }

  function initializeProgressCircle() {
    /**
     * Setup the circular progress indicator
     */
    if (progressCircle) {
      progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
      progressCircle.style.strokeDashoffset = `${circumference}`;
    }
  }

  // ================================
  // Event Listener Setup
  // ================================

  function setupEventListeners() {
    /**
     * Setup primary event listeners for form and buttons
     */
    // Form and file handling
    if (uploadForm) uploadForm.addEventListener("submit", handleSubmit);
    if (removeFileBtn) removeFileBtn.addEventListener("click", removeFile);

    // Download and action buttons
    if (downloadTxtBtn)
      downloadTxtBtn.addEventListener("click", () => downloadTranscript("txt"));
    if (downloadSrtBtn)
      downloadSrtBtn.addEventListener("click", () => downloadTranscript("srt"));
    if (newTranscriptionBtn)
      newTranscriptionBtn.addEventListener("click", resetUI);
    if (tryAgainBtn) tryAgainBtn.addEventListener("click", resetUI);
    if (copyTranscriptBtn)
      copyTranscriptBtn.addEventListener("click", copyTranscriptToClipboard);

    // Model selection
    if (modelSelect) modelSelect.addEventListener("change", updateModelInfo);
  }

  function setupFileHandling() {
    /**
     * Setup drag and drop file handling
     */
    if (!dropArea || !fileInput) return;

    // Prevent default drag behaviors
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      dropArea.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Visual feedback for drag operations
    ["dragenter", "dragover"].forEach((eventName) => {
      dropArea.addEventListener(eventName, highlight, false);
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropArea.addEventListener(eventName, unhighlight, false);
    });

    // File drop and selection handlers
    dropArea.addEventListener("drop", handleDrop, false);
    fileInput.addEventListener("change", handleFileSelect);
  }

  function setupUIToggles() {
    /**
     * Setup UI toggle functionality (mobile menu, options panel, FAQs)
     */
    // Mobile navigation menu
    const mobileMenuButton = document.getElementById("mobile-menu-button");
    const mobileMenu = document.getElementById("mobile-menu");
    if (mobileMenuButton && mobileMenu) {
      mobileMenuButton.addEventListener("click", () => {
        mobileMenu.classList.toggle("hidden");
      });
    }

    // Advanced options panel - FIXED
    const toggleOptions = document.getElementById("toggle-options");
    const optionsPanel = document.getElementById("options-panel");

    if (toggleOptions && optionsPanel) {
      // Check initial state and set proper text
      const isHidden = optionsPanel.classList.contains("hidden");
      if (isHidden) {
        toggleOptions.innerHTML =
          '<i class="fas fa-cog mr-1"></i> Show options';
      } else {
        toggleOptions.innerHTML =
          '<i class="fas fa-times mr-1"></i> Hide options';
      }

      toggleOptions.addEventListener("click", function (e) {
        e.preventDefault();

        if (optionsPanel.classList.contains("hidden")) {
          optionsPanel.classList.remove("hidden");
          toggleOptions.innerHTML =
            '<i class="fas fa-times mr-1"></i> Hide options';
          log("Advanced options panel opened");
        } else {
          optionsPanel.classList.add("hidden");
          toggleOptions.innerHTML =
            '<i class="fas fa-cog mr-1"></i> Show options';
          log("Advanced options panel closed");
        }
      });
    } else {
      log("Advanced options elements not found", "warn");
    }

    // FAQ accordions
    setupFAQAccordions();
  }

  function setupFAQAccordions() {
    /**
     * Setup FAQ accordion functionality
     */
    const accordionHeaders = document.querySelectorAll(".accordion-header");
    accordionHeaders.forEach((header) => {
      header.addEventListener("click", function () {
        const content = this.nextElementSibling;
        const icon = this.querySelector("i");

        // Close other open accordions
        accordionHeaders.forEach((otherHeader) => {
          if (otherHeader !== header) {
            const otherContent = otherHeader.nextElementSibling;
            const otherIcon = otherHeader.querySelector("i");
            if (otherContent) otherContent.style.maxHeight = null;
            if (otherIcon)
              otherIcon.classList.remove("transform", "rotate-180");
          }
        });

        // Toggle current accordion
        if (content && icon) {
          if (content.style.maxHeight) {
            content.style.maxHeight = null;
            icon.classList.remove("transform", "rotate-180");
          } else {
            content.style.maxHeight = content.scrollHeight + "px";
            icon.classList.add("transform", "rotate-180");
          }
        }
      });
    });

    // Open first FAQ by default
    if (accordionHeaders.length > 0) {
      const firstHeader = accordionHeaders[0];
      const firstContent = firstHeader.nextElementSibling;
      const firstIcon = firstHeader.querySelector("i");
      if (firstContent && firstIcon) {
        firstContent.style.maxHeight = firstContent.scrollHeight + "px";
        firstIcon.classList.add("transform", "rotate-180");
      }
    }
  }

  // ================================
  // File Handling Functions
  // ================================

  function preventDefaults(e) {
    /**
     * Prevent default drag and drop behaviors
     */
    e.preventDefault();
    e.stopPropagation();
  }

  function highlight() {
    /**
     * Add visual highlight to drop area during drag
     */
    if (dropArea) {
      dropArea.classList.add("border-nvidia-green", "bg-nvidia-900/10");
      dropArea.classList.remove("border-dark-300");
    }
  }

  function unhighlight() {
    /**
     * Remove visual highlight from drop area
     */
    if (dropArea) {
      dropArea.classList.remove("border-nvidia-green", "bg-nvidia-900/10");
      dropArea.classList.add("border-dark-300");
    }
  }

  function handleDrop(e) {
    /**
     * Handle file drop event
     */
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      updateFileInfo(files[0]);
    }
  }

  function handleFileSelect() {
    /**
     * Handle file selection from input
     */
    if (fileInput && fileInput.files.length > 0) {
      updateFileInfo(fileInput.files[0]);
    }
  }

  function updateFileInfo(file) {
    /**
     * Update UI with selected file information and validate file
     */
    if (!filename || !filesize || !filePrompt || !fileInfo) return;

    // Validate file type
    const fileExtension = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_FILE_TYPES.includes(fileExtension)) {
      showError(
        `Unsupported file type: .${fileExtension}. Please use: ${ALLOWED_FILE_TYPES.map(
          (ext) => "." + ext
        ).join(", ")}`
      );
      removeFile();
      return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      showError(
        `File too large: ${formatFileSize(
          file.size
        )}. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}.`
      );
      removeFile();
      return;
    }

    // Update UI with file information
    filename.textContent = file.name;
    filesize.textContent = formatFileSize(file.size);
    filePrompt.classList.add("hidden");
    fileInfo.classList.remove("hidden");

    // Warn about large files
    if (file.size > 50 * 1024 * 1024) {
      log(
        `Large file detected: ${formatFileSize(
          file.size
        )}. Transcription may take longer.`,
        "warn"
      );
    }

    log(`File selected: ${file.name} (${formatFileSize(file.size)})`);
    hideError();
  }

  function removeFile() {
    /**
     * Remove selected file and reset UI
     */
    if (fileInput) fileInput.value = "";
    if (filePrompt) filePrompt.classList.remove("hidden");
    if (fileInfo) fileInfo.classList.add("hidden");
    hideError();
    log("File removed");
  }

  // ================================
  // System Information Functions
  // ================================

  function fetchSystemInfo() {
    /**
     * Fetch system hardware information from server
     */
    log("Fetching system information...");

    fetch("/system_info")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((info) => {
        log("System info received", "success");
        systemInfo = info;
        updateSystemInfo(info);
        updateModelCompatibility(info);
      })
      .catch((error) => {
        log(`Error fetching system info: ${error}`, "error");
        updateHardwareStatus(
          "Unable to detect system capabilities",
          "bg-red-900/30 text-red-400"
        );
      });
  }

  function updateSystemInfo(info) {
    /**
     * Update UI with system hardware information
     */
    const gpuAvailable = info.gpu && info.gpu.available;

    // Update individual hardware components
    if (pytorchVersion)
      pytorchVersion.textContent = info.pytorch_version || "Not installed";
    if (cudaVersion)
      cudaVersion.textContent = info.gpu?.cuda_version || "Not available";
    if (ramInfo)
      ramInfo.textContent = info.ram ? formatBytes(info.ram) : "Unknown";
    if (cpuInfo) cpuInfo.textContent = info.cpu_info || "Unknown";

    // Update GPU-specific information
    if (gpuAvailable) {
      if (gpuInfo) gpuInfo.textContent = info.gpu.name || "Unknown GPU";
      if (vramInfo) vramInfo.textContent = formatBytes(info.gpu.memory);
      if (gpuTroubleshooting) gpuTroubleshooting.classList.add("hidden");

      updateDeviceIndicator(
        "GPU Enabled",
        "bg-nvidia-900/30 text-nvidia-green"
      );
      updateHardwareStatus(
        `GPU Detected: ${info.gpu.name}`,
        "bg-nvidia-900/30 text-nvidia-green"
      );
    } else {
      if (gpuInfo) gpuInfo.textContent = "Not detected";
      if (vramInfo) vramInfo.textContent = "N/A";
      if (gpuTroubleshooting) gpuTroubleshooting.classList.remove("hidden");

      updateDeviceIndicator(
        "CPU Only Mode",
        "bg-yellow-900/30 text-yellow-400"
      );
      updateHardwareStatus(
        "⚠️ No GPU Detected - Running in CPU mode (slower)",
        "bg-yellow-900/30 text-yellow-400"
      );
    }
  }

  function updateDeviceIndicator(text, className) {
    /**
     * Update device type indicator in navigation
     */
    if (deviceIndicator) {
      deviceIndicator.textContent = text;
      deviceIndicator.className = `ml-4 px-3 py-1 text-xs font-medium rounded-full ${className}`;
    }
  }

  function updateHardwareStatus(text, className) {
    /**
     * Update main hardware status display
     */
    if (hardwareStatus) {
      hardwareStatus.textContent = text;
      hardwareStatus.className = `inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${className}`;
    }
  }

  function updateModelCompatibility(info) {
    /**
     * Update model compatibility indicators based on system hardware
     */
    const models = ["tiny", "base", "small", "medium", "large"];
    const gpuAvailable = info.gpu && info.gpu.available;
    const cachedModels = info.cached_models || [];

    models.forEach((model) => {
      const statusElement = document.getElementById(`model-status-${model}`);
      if (!statusElement) return;

      const isModelCached = cachedModels.some((cache) => cache.includes(model));
      let statusText, statusClass;

      // Determine compatibility based on model requirements and available hardware
      if (model === "large" && !gpuAvailable) {
        statusText = "Not Recommended";
        statusClass = "bg-red-900/30 text-red-400";
      } else if ((model === "medium" || model === "small") && !gpuAvailable) {
        statusText = "CPU Only (Slow)";
        statusClass = "bg-yellow-900/30 text-yellow-400";
      } else {
        statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
        statusClass = "bg-nvidia-900/30 text-nvidia-green";
      }

      statusElement.textContent = statusText;
      statusElement.className = `px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${statusClass}`;
    });
  }

  // ================================
  // Model Information Functions
  // ================================

  function updateModelInfo() {
    /**
     * Update model information based on current selection
     */
    if (!modelSelect || !modelInfo) return;

    const selectedModel = modelSelect.value;

    fetch("/model_info")
      .then((response) => response.json())
      .then((models) => {
        const model = models[selectedModel];
        if (model) {
          let infoText = model.description;

          // Add warnings for large models
          if (model.size_mb > 1000) {
            const sizeInGB = (model.size_mb / 1024).toFixed(1);
            infoText += ` ⚠️ Large model (${sizeInGB} GB) - may take time to download.`;
          }

          // Add performance warnings for CPU-only systems
          if (
            systemInfo &&
            !systemInfo.gpu?.available &&
            ["medium", "large"].includes(selectedModel)
          ) {
            infoText += " Performance may be limited without GPU acceleration.";
          }

          modelInfo.textContent = infoText;
        }
      })
      .catch((error) => {
        log(`Error fetching model info: ${error}`, "error");
        if (modelInfo)
          modelInfo.textContent = "Unable to load model information.";
      });
  }

  // ================================
  // Transcription Process Functions
  // ================================

  function handleSubmit(e) {
    /**
     * Handle form submission and start transcription process
     */
    e.preventDefault();

    if (!validateSubmission()) return;

    // Prepare form data - FIXED to include advanced options
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model", modelSelect.value);

    // Add advanced options to form data
    const multilingualCheckbox = document.getElementById("multilingual");
    const wordTimestampsCheckbox = document.getElementById("word-timestamps");
    const speakerDetectionCheckbox =
      document.getElementById("speaker-detection");

    if (multilingualCheckbox) {
      formData.append("multilingual", multilingualCheckbox.checked);
    }
    if (wordTimestampsCheckbox) {
      formData.append("word_timestamps", wordTimestampsCheckbox.checked);
    }
    if (speakerDetectionCheckbox) {
      formData.append("speaker_detection", speakerDetectionCheckbox.checked);
    }

    // Update UI state
    setSubmissionState(true);
    hideError();
    if (resultsContainer) resultsContainer.classList.add("hidden");
    if (statusContainer) statusContainer.classList.remove("hidden");
    updateStatus("Uploading file...", 0);
    createProgressConsole();

    log(`Starting transcription with model: ${modelSelect.value}`, "success");

    // Submit to server
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (data.error) throw new Error(data.error);

        log(`Upload successful. Task ID: ${data.task_id}`, "success");
        currentTaskId = data.task_id;

        // Add initial console logs
        addConsoleLog(`File uploaded: ${data.file_name}`, "success");
        addConsoleLog(`Model selected: ${data.model}`, "info");
        addConsoleLog(`File size: ${formatFileSize(data.file_size)}`, "info");

        // Start status polling
        startStatusPolling();
      })
      .catch((error) => {
        log(`Upload error: ${error}`, "error");
        showError(`Upload failed: ${error.message}`);
        setSubmissionState(false);
      });
  }

  function validateSubmission() {
    /**
     * Validate form before submission
     */
    if (!fileInput || fileInput.files.length === 0) {
      showError("Please select a file to transcribe.");
      return false;
    }

    const file = fileInput.files[0];
    const fileExtension = file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_FILE_TYPES.includes(fileExtension)) {
      showError(`Unsupported file type: .${fileExtension}`);
      return false;
    }

    if (!modelSelect || !modelSelect.value) {
      showError("Please select a model.");
      return false;
    }

    return true;
  }

  function setSubmissionState(isSubmitting) {
    /**
     * Enable/disable form elements during submission
     */
    if (submitBtn) {
      submitBtn.disabled = isSubmitting;
      if (isSubmitting) {
        submitBtn.classList.add("opacity-50", "cursor-not-allowed");
      } else {
        submitBtn.classList.remove("opacity-50", "cursor-not-allowed");
      }
    }

    if (fileInput) fileInput.disabled = isSubmitting;
    if (modelSelect) modelSelect.disabled = isSubmitting;
  }

  // ================================
  // Progress Tracking Functions
  // ================================

  function createProgressConsole() {
    /**
     * Create live console output display
     */
    if (consoleDisplay) consoleDisplay.remove();

    consoleDisplay = document.createElement("div");
    consoleDisplay.id = "progress-console";
    consoleDisplay.className =
      "mt-6 p-4 bg-dark-700 rounded-lg border border-dark-400";
    consoleDisplay.innerHTML = `
      <div class="flex items-center justify-between mb-3">
        <h4 class="text-sm font-medium text-gray-300">Console Output</h4>
        <button id="clear-console" class="text-xs text-gray-500 hover:text-gray-400">Clear</button>
      </div>
      <div id="console-logs" class="h-32 overflow-y-auto bg-dark-800 rounded p-2 font-mono text-sm text-gray-300 space-y-1"></div>
    `;

    if (statusContainer) {
      statusContainer.appendChild(consoleDisplay);
    }

    // Setup clear button
    const clearBtn = document.getElementById("clear-console");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        const logsDiv = document.getElementById("console-logs");
        if (logsDiv) logsDiv.innerHTML = "";
      });
    }
  }

  function addConsoleLog(message, type = "info") {
    /**
     * Add message to live console output
     */
    const logsDiv = document.getElementById("console-logs");
    if (!logsDiv) return;

    const timestamp = new Date().toLocaleTimeString();
    const colors = {
      error: "text-red-400",
      warn: "text-yellow-400",
      success: "text-green-400",
      info: "text-gray-300",
    };

    const logEntry = document.createElement("div");
    logEntry.className = `text-xs ${colors[type] || colors.info}`;
    logEntry.textContent = `[${timestamp}] ${message}`;

    logsDiv.appendChild(logEntry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
  }

  function startStatusPolling() {
    /**
     * Start polling server for transcription status updates
     */
    if (statusInterval) clearInterval(statusInterval);

    statusInterval = setInterval(() => {
      if (!currentTaskId) {
        clearInterval(statusInterval);
        return;
      }

      fetch(`/status/${currentTaskId}`)
        .then((response) => response.json())
        .then((data) => {
          updateStatusFromData(data);

          // Handle completion or error
          if (data.status === "completed" || data.status === "error") {
            clearInterval(statusInterval);

            if (data.status === "completed") {
              fetchFinalTranscript();
              addConsoleLog(
                `Transcription completed in ${
                  data.processing_time?.toFixed(1) || "?"
                } seconds`,
                "success"
              );
              log(
                `Transcription completed in ${
                  data.processing_time?.toFixed(1) || "?"
                } seconds`,
                "success"
              );
            } else {
              showError(`Transcription error: ${data.error}`);
              addConsoleLog(`Error: ${data.error}`, "error");
              log(`Transcription error: ${data.error}`, "error");
            }
          }
        })
        .catch((error) => {
          log(`Error checking status: ${error}`, "error");
          clearInterval(statusInterval);
          showError(`Status check failed: ${error.message}`);
        });
    }, 800); // Poll every 800ms for smooth updates
  }

  function updateStatusFromData(data) {
    /**
     * Update UI with status data from server
     */
    const progress = Math.round(data.progress || 0);
    const stageInfo = data.stage_info || "";
    const detailedStatus = data.detailed_status || "";

    log(`Status: ${data.status}, Progress: ${progress}%, Stage: ${stageInfo}`);

    // Add console log for stage changes
    const lastStage = window.lastStage;
    if (stageInfo !== lastStage) {
      addConsoleLog(stageInfo, "info");
      window.lastStage = stageInfo;
    }

    // Update device type indicator
    updateDeviceTypeDisplay(data.device_type);

    // Update download progress display
    updateDownloadProgress(data);

    // Create status text with time estimate
    let statusText = detailedStatus || stageInfo;
    if (data.time_remaining && data.time_remaining > 0) {
      statusText += ` (${formatTimeRemaining(data.time_remaining)})`;
    }

    // Add audio duration information
    if (data.audio_duration && !window.audioDurationLogged) {
      addConsoleLog(
        `Audio duration: ${(data.audio_duration / 60).toFixed(1)} minutes`,
        "info"
      );
      window.audioDurationLogged = true;
    }

    updateStatus(statusText, progress);
  }

  function updateDeviceTypeDisplay(deviceTypeData) {
    /**
     * Update device type indicator during transcription
     */
    if (!deviceType || !deviceTypeData) return;

    if (deviceTypeData === "gpu") {
      deviceType.textContent = "Using GPU";
      deviceType.className =
        "text-sm font-medium px-3 py-1 rounded-full bg-nvidia-900/30 text-nvidia-green";
    } else {
      deviceType.textContent = "Using CPU";
      deviceType.className =
        "text-sm font-medium px-3 py-1 rounded-full bg-yellow-900/30 text-yellow-400";
    }
  }

  function updateDownloadProgress(data) {
    /**
     * Update model download progress display
     */
    if (
      data.status === "downloading_model" &&
      data.downloaded !== undefined &&
      data.total_size !== undefined
    ) {
      if (downloadProgressInfo) downloadProgressInfo.classList.remove("hidden");
      if (downloadSize) downloadSize.textContent = formatBytes(data.downloaded);
      if (totalSize) totalSize.textContent = formatBytes(data.total_size);

      // Add download progress to console
      const downloadPercent = (
        (data.downloaded / data.total_size) *
        100
      ).toFixed(1);
      addConsoleLog(
        `Download progress: ${downloadPercent}% (${formatBytes(
          data.downloaded
        )} / ${formatBytes(data.total_size)})`,
        "info"
      );
    } else {
      if (downloadProgressInfo) downloadProgressInfo.classList.add("hidden");
    }
  }

  function updateStatus(message, progressValue) {
    /**
     * Update progress bars and status message
     */
    // Update status message
    if (statusMessage) statusMessage.textContent = message;

    // Update linear progress bar
    if (progressBar) progressBar.style.width = `${progressValue}%`;

    // Update circular progress indicator
    if (progressCircle && progressPercentageCircle) {
      const offset = circumference - (progressValue / 100) * circumference;
      progressCircle.style.strokeDashoffset = offset;
      progressPercentageCircle.textContent = `${progressValue}%`;
    }

    // Toggle loading spinner
    if (loadingSpinner) {
      if (progressValue >= 100) {
        loadingSpinner.classList.add("hidden");
      } else {
        loadingSpinner.classList.remove("hidden");
      }
    }
  }

  // ================================
  // Results and Download Functions
  // ================================

  function fetchFinalTranscript() {
    /**
     * Fetch completed transcript from server
     */
    if (!currentTaskId) return;

    fetch(`/transcript/${currentTaskId}`)
      .then((response) => response.json())
      .then((data) => {
        log("Final transcript fetched successfully");
        if (transcript) transcript.value = data.transcript;

        // Show results, hide progress
        if (statusContainer) statusContainer.classList.add("hidden");
        if (resultsContainer) resultsContainer.classList.remove("hidden");

        // Re-enable form
        setSubmissionState(false);

        // Remove console display
        if (consoleDisplay) consoleDisplay.remove();
      })
      .catch((error) => {
        log(`Error fetching final transcript: ${error}`, "error");
        showError(`Failed to fetch transcript: ${error.message}`);
      });
  }

  function downloadTranscript(format) {
    /**
     * Download transcript in specified format - FIXED
     */
    if (!currentTaskId) {
      showError("No transcript available for download.");
      return;
    }

    if (!["txt", "srt"].includes(format)) {
      showError("Invalid download format.");
      return;
    }

    // Use window.open for better download handling
    try {
      const downloadUrl = `/download/${currentTaskId}/${format}`;

      // Create a temporary anchor element for download
      const anchor = document.createElement("a");
      anchor.href = downloadUrl;
      anchor.target = "_blank";
      anchor.download = `transcript.${format}`;

      // Append to body, click, and remove
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);

      log(`Downloading transcript as ${format}`, "success");

      // Clean up server resources after successful download
      setTimeout(() => {
        cleanupTask();
      }, 2000);
    } catch (error) {
      log(`Download error: ${error}`, "error");
      showError(
        `Failed to download ${format.toUpperCase()} file: ${error.message}`
      );
    }
  }

  function cleanupTask() {
    /**
     * Clean up server resources for completed task
     */
    if (!currentTaskId) return;

    fetch(`/cleanup/${currentTaskId}`, { method: "POST" })
      .then((response) => response.json())
      .then((data) => {
        log("Server cleanup completed", "success");
      })
      .catch((error) => {
        log(`Cleanup error: ${error}`, "warn");
      });
  }

  function copyTranscriptToClipboard() {
    /**
     * Copy transcript text to clipboard - FIXED
     */
    if (!transcript || !transcript.value) {
      showError("No transcript available to copy.");
      return;
    }

    const textToCopy = transcript.value;

    // Check if modern clipboard API is available
    if (navigator.clipboard && window.isSecureContext) {
      // Use modern Clipboard API
      navigator.clipboard
        .writeText(textToCopy)
        .then(() => {
          log("Transcript copied to clipboard", "success");
          showCopySuccess();
        })
        .catch((error) => {
          log(`Modern clipboard copy failed: ${error}`, "warn");
          fallbackCopyTextToClipboard(textToCopy);
        });
    } else {
      // Fallback to legacy method
      log("Using fallback clipboard method", "warn");
      fallbackCopyTextToClipboard(textToCopy);
    }
  }

  function showCopySuccess() {
    /**
     * Show copy success message
     */
    if (copySuccessMessage) {
      copySuccessMessage.classList.remove("hidden");
      setTimeout(() => {
        copySuccessMessage.classList.add("hidden");
      }, 3000);
    }
  }

  function fallbackCopyTextToClipboard(text) {
    /**
     * Fallback clipboard copy method for older browsers - FIXED
     */
    const textArea = document.createElement("textarea");
    textArea.value = text;

    // Make the textarea invisible but functional
    Object.assign(textArea.style, {
      position: "fixed",
      left: "-999999px",
      top: "-999999px",
      opacity: "0",
      pointerEvents: "none",
    });

    document.body.appendChild(textArea);

    try {
      // Focus and select the text
      textArea.focus();
      textArea.select();
      textArea.setSelectionRange(0, text.length);

      // Try to copy using execCommand
      const successful = document.execCommand("copy");

      if (successful) {
        log("Transcript copied (fallback method)", "success");
        showCopySuccess();
      } else {
        throw new Error("execCommand failed");
      }
    } catch (err) {
      log(`Fallback copy failed: ${err}`, "error");

      // Last resort - select the text in the textarea and show instruction
      textArea.style.position = "static";
      textArea.style.left = "auto";
      textArea.style.top = "auto";
      textArea.style.opacity = "1";
      textArea.style.pointerEvents = "auto";
      textArea.select();

      showError(
        "Unable to copy automatically. Please press Ctrl+C (or Cmd+C on Mac) to copy the selected text."
      );
    } finally {
      // Clean up - delay removal to allow manual copy
      setTimeout(() => {
        if (document.body.contains(textArea)) {
          document.body.removeChild(textArea);
        }
      }, 5000);
    }
  }

  // ================================
  // Error Handling Functions
  // ================================

  function showError(message) {
    /**
     * Display error message to user
     */
    if (errorMessage) errorMessage.textContent = message;
    if (errorContainer) errorContainer.classList.remove("hidden");
    if (statusContainer) statusContainer.classList.add("hidden");
    log(message, "error");
  }

  function hideError() {
    /**
     * Hide error message display
     */
    if (errorContainer) errorContainer.classList.add("hidden");
  }

  // ================================
  // UI Reset Functions
  // ================================

  function resetUI() {
    /**
     * Reset UI for new transcription
     */
    log("Resetting UI for new transcription");

    // Clear state variables
    currentTaskId = null;
    window.lastStage = null;
    window.audioDurationLogged = false;

    // Clear polling interval
    if (statusInterval) {
      clearInterval(statusInterval);
      statusInterval = null;
    }

    // Reset form and file selection
    if (uploadForm) uploadForm.reset();
    removeFile();

    // Hide all containers
    if (statusContainer) statusContainer.classList.add("hidden");
    if (resultsContainer) resultsContainer.classList.add("hidden");
    hideError();

    // Remove console display
    if (consoleDisplay) {
      consoleDisplay.remove();
      consoleDisplay = null;
    }

    // Re-enable form
    setSubmissionState(false);

    // Clear transcript
    if (transcript) transcript.value = "";
  }

  // ================================
  // Application Startup
  // ================================

  // Initialize the application when DOM is ready
  init();
});
