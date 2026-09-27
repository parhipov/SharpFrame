# Portable SharpFrame: embedded Python + the .venv's packages + the script, nothing to install.
#   powershell -ExecutionPolicy Bypass -File build_zip.ps1
# Writes dist\SharpFrame\ and dist\SharpFrame.zip.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$dist = "$root\dist\SharpFrame"
$zip = "$root\dist\SharpFrame.zip"
$pyver = '3.11.9'    # the last 3.11 with an embeddable build; the .venv is 3.11
$embed = "$env:TEMP\python-$pyver-embed-amd64.zip"

if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
if (Test-Path $zip) { Remove-Item $zip -Force }
if (-not (Test-Path $embed)) {
    [Net.ServicePointManager]::SecurityProtocol = 'Tls12'
    Invoke-WebRequest "https://www.python.org/ftp/python/$pyver/python-$pyver-embed-amd64.zip" -OutFile $embed
}
Expand-Archive $embed "$dist\python"
# the embedded build ignores site-packages unless its ._pth names it
Add-Content "$dist\python\python311._pth" 'Lib\site-packages' -Encoding ascii

# requirements.txt as installed in the .venv (same 3.11 ABI), less pip's own tooling
$src = "$root\.venv\Lib\site-packages"
if (-not (Test-Path $src)) { throw "no .venv: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" }
$site = "$dist\python\Lib\site-packages"
New-Item -ItemType Directory $site | Out-Null
Get-ChildItem $src | Where-Object { $_.Name -notmatch '^(pip|setuptools|_distutils_hack|distutils-precedence\.pth|__pycache__)' } |
    Copy-Item -Destination $site -Recurse
Get-ChildItem $site -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
# OpenCV's own ffmpeg (27 MB) is for cv2.VideoCapture; PyAV decodes here
Remove-Item "$site\cv2\opencv_videoio_ffmpeg*.dll" -ErrorAction SilentlyContinue

Copy-Item "$root\sharpframe.py", "$root\web.py", "$root\web.html", "$root\SharpFrame.bat", "$root\sharpframe.json", `
    "$root\README.txt" $dist
# Windows' own tar, not Compress-Archive: 5.1 writes backslashes into the zip's paths.
# By full path: from Git Bash its GNU tar comes first and takes E: for a remote host
& "$env:SystemRoot\System32\tar.exe" -a -c -f $zip -C "$root\dist" SharpFrame
if ($LASTEXITCODE) { throw 'tar failed' }
'{0}  {1:N0} MB' -f $zip, ((Get-Item $zip).Length / 1MB)
