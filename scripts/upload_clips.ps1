param(
    [string]$InputVideoPath = "C:\Users\Admin\Downloads\Video Project.mp4",
    [string]$ServerUrl = "http://4.186.40.179:8000/api/clips",
    [string]$MatchId = "6a315d21b68f0f0bbf39d96b",
    [int]$Half = 1,
    [string]$TeamAName = "mnv",
    [string]$TeamBName = "mfc",
    [string]$TeamAColour = "#FFA500",
    [string]$TeamBColour = "#FFFF00",
    [string]$TeamAGkColour = "#00FF00",
    [string]$TeamBGkColour = "#000000",
    [ValidateSet("classic", "futsal")]
    [string]$Ruleset = "classic",
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

$durationTotal = [double]$durationText.Trim()
$durationSeconds = [math]::Ceiling($durationTotal)
$clipCount = [math]::Ceiling($durationTotal / 60.0)

Write-Host "Video duration: $durationSeconds seconds"
Write-Host "Uploading $clipCount clips to $ServerUrl/api/clips"

for ($clipIndex = 0; $clipIndex -lt $clipCount; $clipIndex++) {
    $minute = $clipIndex + 1
    $startSeconds = $clipIndex * 60
    $clipDurationSeconds = [math]::Min(60.0, [math]::Max(0.0, $durationTotal - $startSeconds))
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
        "-F", "clip_duration_seconds=$clipDurationSeconds",
        "-F", "team_a_name=$TeamAName",
        "-F", "team_b_name=$TeamBName",
        "-F", "team_a_colour=$TeamAColour",
        "-F", "team_b_colour=$TeamBColour",
        "-F", "team_a_gk_colour=$TeamAGkColour",
        "-F", "team_b_gk_colour=$TeamBGkColour",
        "-F", "ruleset=$Ruleset",
        "$ServerUrl/api/clips"
    )

    $response = & $curl.Source @uploadArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe failed while uploading clip $minute"
    }

    Write-Host $response
}

Write-Host "Done. All clips uploaded."