# Run once on Windows. The notebook itself executes on the Colab kernel.
$labRoot = Split-Path -Parent $PSScriptRoot
$syncFolder = Join-Path $labRoot 'work/local-sync'
New-Item -ItemType Directory -Force -Path $syncFolder | Out-Null
$statusPath = Join-Path $syncFolder 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $syncStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    $nowSeconds = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    if ($syncStatus.running -and ($nowSeconds - $syncStatus.updated) -lt 10) {
        Write-Output 'Local notebook receiver is already running.'
        exit 0
    }
}
$stopPath = Join-Path $syncFolder 'STOP'
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$notebookPath = Join-Path $labRoot 'lab_day10.ipynb'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'lab.local_sync', '--notebook', $notebookPath) -WorkingDirectory $labRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $syncFolder 'stdout.log') -RedirectStandardError (Join-Path $syncFolder 'stderr.log')
$process.Id | Set-Content -LiteralPath (Join-Path $syncFolder 'process-id.txt')
Write-Output "Local receiver started (PID $($process.Id)). Enable File > Auto Save in VS Code."
