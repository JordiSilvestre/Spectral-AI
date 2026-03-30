@echo off
REM build_optix_win.bat -- Build OptiX Extension on Windows with MSVC
REM Usage: build_optix_win.bat [test]
REM   build_optix_win.bat        -- compile only
REM   build_optix_win.bat test   -- compile + run quick test

echo ======================================================================
echo SpectralAI OptiX RT Core Extension -- Windows Native Build
echo ======================================================================

REM Activate MSVC x64 environment
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%VCVARS%" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
)
if not exist "%VCVARS%" (
    echo [ERROR] vcvarsall.bat not found. Install Visual Studio Build Tools.
    exit /b 1
)

echo Activating MSVC x64 environment...
call "%VCVARS%" x64

REM Verify cl.exe is now available
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe not found after activating MSVC.
    exit /b 1
)

REM Set OptiX path
if not defined OPTIX_DIR (
    set "OPTIX_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
)
echo OptiX SDK: %OPTIX_DIR%

REM Set CUDA path
if not defined CUDA_HOME (
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
)
echo CUDA Home: %CUDA_HOME%

REM Ensure ninja and Python Scripts are in PATH
set "PATH=C:\Users\jsilv\AppData\Roaming\Python\Python314\Scripts;%PATH%"

REM Navigate to project root
cd /d "J:\Proyectos\SPECTRAL AI"

echo.
echo Building OptiX extension...
python cuda\v5\build_optix_ext.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

if "%1"=="test" (
    echo.
    echo Running quick test...
    python -c "import optix_training_ext as ext; print('is_ready:', ext.is_ready()); print('SUCCESS: Extension loaded!')"
)

echo.
echo Done.
