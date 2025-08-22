# Configuration - Modify these parameters as needed
$Function = "yolo"                                          # Detection function: yolo, fall, climb
$InputSource = "rtsp://root:root@192.168.31.240/cam1/h264-1"  # RTSP stream or video file path
$LogFile = "log/detection.log"                             # Log file path

# Path to main_rtsp.exe directory
$ReleaseDir = "C:\Users\user\Desktop\Release"
$ExePath = "main_rtsp.exe"

# Check if Release directory exists
if (-not (Test-Path $ReleaseDir)) {
    Write-Host "Error: Release directory not found at $ReleaseDir" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if main_rtsp.exe exists in Release directory
$FullExePath = Join-Path $ReleaseDir $ExePath
if (-not (Test-Path $FullExePath)) {
    Write-Host "Error: main_rtsp.exe not found at $FullExePath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create log directory if needed
$logDir = Split-Path $LogFile -Parent
if ($logDir -and -not (Test-Path $logDir)) {
    Write-Host "Creating log directory: $logDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Execute main_rtsp.exe with the three parameters
Write-Host "Starting RTSP Detection System..." -ForegroundColor Green
Write-Host "Function: $Function | Input: $InputSource | Log: $LogFile" -ForegroundColor Yellow
Write-Host "Working Directory: $ReleaseDir" -ForegroundColor Cyan
Write-Host ""

# Save current directory
$OriginalDir = Get-Location

# Change to Release directory and execute
Set-Location $ReleaseDir
& ".\$ExePath" $Function $InputSource $LogFile

# Restore original directory
Set-Location $OriginalDir

# Wait for user input before closing (useful when double-clicking)
Write-Host ""
Write-Host "Detection system has stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
