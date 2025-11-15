<#
build_windows.ps1

간단한 빌드 스크립트: 가상환경 생성, 의존성 설치, PyInstaller로 exe 빌드

주의: pyrealsense2 네이티브 바이너리는 환경마다 위치가 다릅니다. 빌드 완료 후 생성된 exe를 테스트하고
필요시 PyInstaller spec에 pyrealsense2의 DLL 파일들을 `--add-binary`로 추가해야 합니다.
#>

param(
    [string]$VenvPath = ".venv",
    [string]$Entry = "ui.py",
    [string]$Name = "front_realsense"
)

Write-Host "== front_realsense: Windows build helper =="
if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating venv: $VenvPath"
    python -m venv $VenvPath
}

Write-Host "Activating venv and installing requirements..."
& $VenvPath\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Attempt to locate pyrealsense2 native binaries and prepare --add-binary arguments
$addBinaryArgs = @()
try {
    $pyModulePath = & python - <<'PY'
import pyrealsense2 as rs, os, json
print(os.path.dirname(rs.__file__))
PY
    $pyModulePath = $pyModulePath.Trim()
    if ($pyModulePath) {
        Write-Host "Found pyrealsense2 at: $pyModulePath"
        $files = Get-ChildItem -Path $pyModulePath -Include *.pyd,*.dll -File -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            # PyInstaller add-binary format: src;dest
            $src = $f.FullName
            $dest = "pyrealsense2"  # place into package folder
            $arg = "--add-binary `"$src;$dest`""
            $addBinaryArgs += $arg
            Write-Host "Including binary: $src"
        }
    }
} catch {
    Write-Host "pyrealsense2 not found or error while locating native files: $_"
}

Write-Host "Running PyInstaller..."

$args = @("--onefile","--noconfirm","--clean","-n", $Name, $Entry)
if ($addBinaryArgs.Count -gt 0) { $args += $addBinaryArgs }

Write-Host "pyinstaller $($args -join ' ')"
pyinstaller @args

if (Test-Path "dist\$Name.exe") {
    Write-Host "빌드 완료: dist\$Name.exe"
} else {
    Write-Host "빌드가 완료되었지만 dist\$Name.exe을 찾을 수 없습니다. 빌드 로그를 확인하세요."
}

Write-Host "Note: If the exe fails with a missing DLL/.pyd error, find the missing file on your system and re-run this script with that file added via --add-binary or adjust the spec file (pyinstaller.spec)."
