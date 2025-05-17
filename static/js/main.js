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
  const progressPercentage = document.getElementById(
    "progress-percentage-circle"
  );
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
  const downloadProgressInfo = document.getElementById(
    "download-progress-info"
  );
  const downloadSize = document.getElementById("download-size");
  const totalSize = document.getElementById("total-size");
  const copyTranscriptBtn = document.getElementById("copy-transcript");
  const copySuccessMessage = document.getElementById("copy-success-message");
  const mobileMenuButton = document.getElementById("mobile-menu-button");
  const mobileMenu = document.getElementById("mobile-menu");
  const deviceIndicator = document.getElementById("device-indicator");
  const hardwareStatus = document.getElementById("hardware-status");
  const cpuInfo = document.getElementById("cpu-info");
  const ramInfo = document.getElementById("ram-info");
  const gpuInfo = document.getElementById("gpu-info");
  const vramInfo = document.getElementById("vram-info");
  const pytorchVersion = document.getElementById("pytorch-version");
  const cudaVersion = document.getElementById("cuda-version");
  const gpuTroubleshooting = document.getElementById("gpu-troubleshooting");
  const toggleOptions = document.getElementById("toggle-options");
  const optionsPanel = document.getElementById("options-panel");

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

  // Initialize system information
  fetchSystemInfo();

  // Mobile menu toggle
  if (mobileMenuButton) {
    mobileMenuButton.addEventListener("click", function () {
      mobileMenu.classList.toggle("hidden");
    });
  }

  // Toggle options panel
  if (toggleOptions) {
    toggleOptions.addEventListener("click", function () {
      if (optionsPanel.classList.contains("hidden")) {
        optionsPanel.classList.remove("hidden");
        toggleOptions.innerHTML =
          '<i class="fas fa-times mr-1"></i> Hide options';
      } else {
        optionsPanel.classList.add("hidden");
        toggleOptions.innerHTML =
          '<i class="fas fa-cog mr-1"></i> Show options';
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
    dropArea.classList.add("active");
  }

  // Remove highlight from drop area
  function unhighlight() {
    dropArea.classList.remove("active");
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
    console.log(`File selected: ${file.name} (${formatFileSize(file.size)})`);
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
    console.log("File removed");
  }

  // Fetch system information
  function fetchSystemInfo() {
    fetch("/system_info")
      .then((response) => response.json())
      .then((info) => {
        console.log("System info:", info);
        systemInfo = info;
        updateSystemInfo(info);
        updateModelCompatibility(info);
      })
      .catch((error) => {
        console.error("Error fetching system info:", error);
      });
  }

  // Update system information display
  function updateSystemInfo(info) {
    // Update hardware detection section
    const gpuAvailable = info.gpu.available;

    if (pytorchVersion) {
      pytorchVersion.textContent = info.pytorch_version || "Unknown";
    }

    if (cudaVersion) {
      cudaVersion.textContent = info.gpu.cuda_version || "Not detected";
    }

    if (gpuAvailable) {
      if (gpuInfo) gpuInfo.textContent = info.gpu.name;
      if (vramInfo)
        vramInfo.textContent = formatBytes(info.gpu.memory) + " available";
      if (gpuTroubleshooting) gpuTroubleshooting.classList.add("hidden");

      // Update device indicator
      if (deviceIndicator) {
        deviceIndicator.textContent = "GPU Enabled";
        deviceIndicator.classList.add("bg-green-100", "text-green-800");
      }

      // Update hardware status
      if (hardwareStatus) {
        hardwareStatus.textContent = "GPU Detected: " + info.gpu.name;
        hardwareStatus.classList.add("bg-green-100", "text-green-800");
      }
    } else {
      if (gpuInfo) gpuInfo.textContent = "Not detected or not compatible";
      if (vramInfo) vramInfo.textContent = "N/A";
      if (gpuTroubleshooting) gpuTroubleshooting.classList.remove("hidden");

      // Update device indicator
      if (deviceIndicator) {
        deviceIndicator.textContent = "CPU Only Mode";
        deviceIndicator.classList.add("bg-yellow-100", "text-yellow-800");
      }

      // Update hardware status
      if (hardwareStatus) {
        hardwareStatus.textContent =
          "⚠️ No GPU Detected - Running in CPU mode (slower)";
        hardwareStatus.classList.add("bg-yellow-100", "text-yellow-800");
      }
    }

    // Fake CPU info (not available directly through the backend)
    const cpuCores = navigator.hardwareConcurrency || "Unknown";
    if (cpuInfo) cpuInfo.textContent = `${cpuCores} cores available`;
    if (ramInfo) ramInfo.textContent = "Check your system specifications";
  }

  // Update model compatibility based on system info
  function updateModelCompatibility(info) {
    const models = ["tiny", "base", "small", "medium", "large"];
    const gpuAvailable = info.gpu.available;
    const cachedModels = info.cached_models || [];

    models.forEach((model) => {
      const statusElement = document.getElementById(`model-status-${model}`);
      if (!statusElement) return;

      let compatible = true;
      let statusText = "";
      let statusClass = "";

      // Check if model is already cached
      const isModelCached = cachedModels.some((cache) => cache.includes(model));

      if (model === "tiny" || model === "base") {
        // These models work on almost any hardware
        statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
        statusClass = "bg-green-100 text-green-800";
      } else if (model === "small") {
        if (gpuAvailable) {
          statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
          statusClass = "bg-green-100 text-green-800";
        } else {
          statusText = "Compatible (Slow)";
          statusClass = "bg-yellow-100 text-yellow-800";
        }
      } else if (model === "medium") {
        if (gpuAvailable) {
          statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
          statusClass = "bg-green-100 text-green-800";
        } else {
          statusText = "CPU Only (Very Slow)";
          statusClass = "bg-yellow-100 text-yellow-800";
        }
      } else if (model === "large") {
        if (gpuAvailable) {
          statusText = isModelCached ? "Ready (Downloaded)" : "Compatible";
          statusClass = "bg-green-100 text-green-800";
        } else {
          statusText = "Not Recommended";
          statusClass = "bg-red-100 text-red-800";
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
      .catch((error) => console.error("Error fetching model info:", error));
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

    // Reset and show status container
    resetStatus();
    hideError();
    resultsContainer.classList.add("hidden");
    statusContainer.classList.remove("hidden");
    updateStatus("Uploading file...", 0);

    console.log(`Starting transcription with model: ${modelSelect.value}`);

    // Send form data
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        console.log("Upload successful:", data);
        currentTaskId = data.task_id;

        // Start polling for status
        startStatusPolling();
      })
      .catch((error) => {
        console.error("Error:", error);
        showError(`Error uploading file: ${error.message}`);
        submitBtn.disabled = false;
      });
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
            }

            if (data.status === "error") {
              showError(`Transcription error: ${data.error}`);
            }
          }
        })
        .catch((error) => {
          console.error("Error checking status:", error);
          clearInterval(statusInterval);
          showError(`Error checking status: ${error.message}`);
        });
    }, 1000); // Poll every 1 second
  }

  // Update status based on data from server
  function updateStatusFromData(data) {
    let statusText = "";
    let progress = data.progress || 0;

    // Update device type indicator
    if (deviceType) {
      if (data.device_type === "gpu") {
        deviceType.textContent = "Using GPU";
        deviceType.className =
          "text-sm font-medium px-3 py-1 rounded-full bg-green-100 text-green-800";
      } else {
        deviceType.textContent = "Using CPU";
        deviceType.className =
          "text-sm font-medium px-3 py-1 rounded-full bg-yellow-100 text-yellow-800";
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

    switch (data.status) {
      case "preparing":
        statusText = "Preparing transcription...";
        break;
      case "downloading_model":
        statusText = `Downloading Whisper ${modelSelect.value} model...`;
        break;
      case "loading_model":
        statusText = `Loading Whisper ${modelSelect.value} model...`;
        break;
      case "transcribing":
        statusText = `Transcribing audio using ${data.device_type.toUpperCase()}... This may take a while depending on file length and model size.`;
        break;
      case "completed":
        statusText = "Transcription completed!";
        submitBtn.disabled = false;
        break;
      case "error":
        statusText = `Error: ${data.error}`;
        submitBtn.disabled = false;
        break;
      default:
        statusText = "Processing...";
    }

    updateStatus(statusText, progress);
  }

  // Update status display
  function updateStatus(message, progressValue) {
    statusMessage.textContent = message;

    // Update progress bar
    progressBar.style.width = `${progressValue}%`;

    // Update circular progress indicator
    if (progressCircle) {
      const offset = circumference - (progressValue / 100) * circumference;
      progressCircle.style.strokeDashoffset = offset;
      progressPercentage.textContent = `${progressValue}%`;
    }

    if (progressValue === 100) {
      loadingSpinner.classList.add("hidden");
    } else {
      loadingSpinner.classList.remove("hidden");
    }

    console.log(`Status: ${message} (${progressValue}%)`);
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
        console.log("Transcript fetched successfully");
        transcript.value = data.transcript;
        statusContainer.classList.add("hidden");
        resultsContainer.classList.remove("hidden");
      })
      .catch((error) => {
        console.error("Error fetching transcript:", error);
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

    console.log(`Downloading transcript as ${format}`);

    // Clean up server resources
    setTimeout(() => {
      fetch(`/cleanup/${currentTaskId}`, { method: "POST" })
        .then((response) => response.json())
        .then((data) => console.log("Cleanup successful:", data))
        .catch((error) => console.error("Error during cleanup:", error));
    }, 1000);
  }

  // Copy transcript to clipboard
  function copyTranscriptToClipboard() {
    if (!transcript.value) return;

    navigator.clipboard
      .writeText(transcript.value)
      .then(() => {
        console.log("Transcript copied to clipboard");
        copySuccessMessage.classList.remove("hidden");

        // Hide success message after 2 seconds
        setTimeout(() => {
          copySuccessMessage.classList.add("hidden");
        }, 2000);
      })
      .catch((error) => {
        console.error("Error copying transcript:", error);
        showError(`Error copying transcript: ${error.message}`);
      });
  }

  // Show error message
  function showError(message) {
    errorMessage.textContent = message;
    errorContainer.classList.remove("hidden");
    statusContainer.classList.add("hidden");
    console.error(message);
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

    console.log("UI reset for new transcription");
  }
});
