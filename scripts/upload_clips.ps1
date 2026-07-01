param(
    [string]$InputVideoPath = "C:\Users\Admin\Downloads\Video Project.mp4",
    [string]$ServerUrl = "http://4.186.40.179:8000/api/clips",
    [string]$MatchId = "6a315d21b68f0f0bbf39d96b",
    [int]$Half = 1,
    [string]$Team0Name = "mnv",
    [string]$Team1Name = "mfc",
    [string]$Team0Colour = "#FFA500",
    [string]$Team1Colour = "#FFFF00",
    [string]$OutputDir = "$PSScriptRoot\..\data\clips_upload"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $InputVideoPath)) {
    throw "Input video not found: $InputVideoPath"
}

$ffmpeg = Get-Command ffmpeg -ErrorAction Stop
$ffprobe = Get-Command ffprobe -ErrorAction Stop
$curl = Get-Command curl.exe -ErrorAction Stop

$resolvedOutputDir = (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $OutputDir)).Path
$fileNameBase = [System.IO.Path]::GetFileNameWithoutExtension($InputVideoPath)

$durationText = & $ffprobe.Source -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $InputVideoPath
if (-not $durationText) {
    throw "Could not read video duration with ffprobe. Make sure ffmpeg and ffprobe are installed and in PATH."
}

$durationSeconds = [math]::Ceiling([double]$durationText.Trim())
$clipCount = [math]::Ceiling($durationSeconds / 60.0)

Write-Host "Video duration: $durationSeconds seconds"
Write-Host "Uploading $clipCount clips to $ServerUrl/api/clips"

for ($clipIndex = 0; $clipIndex -lt $clipCount; $clipIndex++) {
    $minute = $clipIndex + 1
    $startSeconds = $clipIndex * 60
    $clipPath = Join-Path $resolvedOutputDir ("{0}_h{1}_m{2:000}.mp4" -f $fileNameBase, $Half, $minute)

    $extractArgs = @(
        "-y",
        "-ss", $startSeconds.ToString(),
        "-i", $InputVideoPath,
        "-t", "60",
        "-c:v", "mpeg4",
        "-q:v", "5",
        "-c:a", "aac",
        "-movflags", "+faststart",
        $clipPath
    )

    & $ffmpeg.Source @extractArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg failed while creating clip $minute"
    }

    Write-Host ("Uploading clip {0} / {1}: {2}" -f $minute, $clipCount, $clipPath)

    $uploadArgs = @(
        "-sS",
        "-X", "POST",
        "-F", "file=@$clipPath;type=video/mp4",
        "-F", "match_id=$MatchId",
        "-F", "half=$Half",
        "-F", "minute=$minute",
        "-F", "team0_name=$Team0Name",
        "-F", "team1_name=$Team1Name",
        "-F", "team0_colour=$Team0Colour",
        "-F", "team1_colour=$Team1Colour",
        "$ServerUrl/api/clips"
    )

    $response = & $curl.Source @uploadArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe failed while uploading clip $minute"
    }

    Write-Host $response
}

Write-Host "Done. All clips uploaded."