@echo off
rem Builds dist\FaceMod.exe (single file, no Python needed to run it).
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller || goto :err
python -m PyInstaller --noconfirm --clean FaceMod.spec || goto :err
echo.
echo Built dist\FaceMod.exe
exit /b 0
:err
echo Build failed.
exit /b 1
