# update_and_run.ps1
# Đặt update_and_run.bat + update_and_run.ps1 vào thư mục "pdf2zh" (cạnh gui.py, translator.py...)
# (đặt ở thư mục cha pdf2zh-v1.9.11-with-assets-win64 cũng chạy được).
# Bấm đúp update_and_run.bat để: tắt exe cũ -> chép file .py đã đổi vào build\site-packages -> chạy lại exe.

$here = $PSScriptRoot
if (Test-Path (Join-Path $here "build\site-packages\pdf2zh")) {
    $src = $here                                   # script nằm trong thư mục pdf2zh
}
elseif (Test-Path (Join-Path $here "pdf2zh\build\site-packages\pdf2zh")) {
    $src = Join-Path $here "pdf2zh"                # script nằm ở thư mục cha
}
else {
    Write-Host "Không tìm thấy build\site-packages\pdf2zh. Hãy đặt script vào thư mục pdf2zh." -ForegroundColor Red
    exit 1
}

$build = Join-Path $src   "build"
$dst   = Join-Path $build "site-packages\pdf2zh"      # nơi exe thật sự đọc code
$exe   = Join-Path $build "pdf2zh.exe"
$py    = Join-Path $build "runtime\python.exe"        # chỉ để kiểm tra cú pháp

# 1. Tắt bản đang chạy
Get-Process -Name "pdf2zh", "_pystand_static*" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 800

# 2. Kiểm tra cú pháp tất cả file .py (lỗi thì dừng, không chép gì)
$files = Get-ChildItem $src -Filter "*.py" -File
if (Test-Path $py) {
    $bad = $false
    foreach ($f in $files) {
        $out = & $py -c "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8-sig').read())" $f.FullName 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Lỗi cú pháp trong $($f.Name):" -ForegroundColor Red
            $out | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            $bad = $true
        }
    }
    if ($bad) { Write-Host "Dừng lại, chưa chép file nào." -ForegroundColor Red; exit 1 }
}

# 3. Chép file đã đổi (backup .bak đúng 1 lần để không ghi đè bản gốc)
$n = 0
foreach ($f in $files) {
    $to = Join-Path $dst $f.Name
    if ((Test-Path $to) -and ((Get-FileHash $f.FullName).Hash -eq (Get-FileHash $to).Hash)) { continue }
    if ((Test-Path $to) -and -not (Test-Path "$to.bak")) { Copy-Item $to "$to.bak" }
    Copy-Item $f.FullName $to -Force
    Write-Host "  cập nhật $($f.Name)" -ForegroundColor Green
    $n++
}
if ($n -eq 0) { Write-Host "Không có file nào thay đổi." -ForegroundColor Yellow }

# 4. Kiểm tra dấu hiệu của bản sửa (thiếu = file nguồn chưa phải bản mới)
$markers = [ordered]@{
    "gui.py"        = "_can_spawn_process"
    "translator.py" = "set_cancel_event"
    "cache.py"      = "BUSY_TIMEOUT_MS"
}
foreach ($name in $markers.Keys) {
    $p = Join-Path $dst $name
    if ((Test-Path $p) -and (Select-String -Path $p -Pattern $markers[$name] -Quiet)) {
        Write-Host ("  [x] {0,-14} OK" -f $name)
    }
    else {
        Write-Host ("  [ ] {0,-14} THIẾU '{1}' -> file chưa phải bản mới" -f $name, $markers[$name]) -ForegroundColor Yellow
    }
}

# 5. Xóa cache biên dịch cũ
Remove-Item (Join-Path $dst "__pycache__") -Recurse -Force -ErrorAction SilentlyContinue

# 6. Chạy exe (thư mục làm việc = build, nên pdf2zh_files vẫn ở build\pdf2zh_files)
if (-not (Test-Path $exe)) { Write-Host "Không thấy $exe" -ForegroundColor Red; exit 1 }
Start-Process -FilePath $exe -WorkingDirectory $build
Write-Host "Đã chạy pdf2zh.exe - trình duyệt sẽ tự mở tại http://127.0.0.1:7860" -ForegroundColor Cyan
