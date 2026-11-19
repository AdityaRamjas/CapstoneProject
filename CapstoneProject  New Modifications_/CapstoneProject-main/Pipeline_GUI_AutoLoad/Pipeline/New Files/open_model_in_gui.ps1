param(
    [Parameter(Mandatory=$true)][int]$ProcessId,
    [Parameter(Mandatory=$true)][string]$ModelPath
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName Microsoft.VisualBasic

try {
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw "Model file not found: $ModelPath"
    }
    $process = $null
    for ($i = 0; $i -lt 90; $i++) {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            $process.Refresh()
            if ($process.MainWindowHandle -ne [IntPtr]::Zero) { break }
        }
        Start-Sleep -Seconds 2
    }
    if ($null -eq $process -or $process.MainWindowHandle -eq [IntPtr]::Zero) {
        throw 'OpenSim GUI window did not become available within 3 minutes.'
    }
    Start-Sleep -Seconds 4
    $shell = New-Object -ComObject WScript.Shell
    if (-not $shell.AppActivate($ProcessId)) {
        throw 'Could not activate the OpenSim GUI window.'
    }
    Start-Sleep -Milliseconds 900
    # OpenSim GUI File > Open Model keyboard shortcut.
    [System.Windows.Forms.SendKeys]::SendWait('^o')
    Start-Sleep -Seconds 2
    # Use the clipboard to preserve spaces and parentheses in the path.
    [System.Windows.Forms.Clipboard]::SetText($ModelPath)
    [System.Windows.Forms.SendKeys]::SendWait('^v')
    Start-Sleep -Milliseconds 400
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    Write-Host "Sent Open Model request for: $ModelPath"
    Write-Host 'Confirm the model appears in the OpenSim Navigator.'
} catch {
    Write-Host "Automatic GUI loading failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Open the model manually using File > Open Model: $ModelPath"
    exit 1
}
