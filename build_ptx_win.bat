@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set OPTIX_INC=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include
set PROJ=J:\Proyectos\SPECTRAL AI
set PTX_OUT=%PROJ%\build\ptx

echo Compiling optix_router_raygen.ptx for sm_120...
nvcc --ptx -arch=compute_89 -I"%OPTIX_INC%" -I"%PROJ%\include" -I"%PROJ%\cuda" -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING "%PROJ%\cuda\optix_router_raygen.cu" -o "%PTX_OUT%\optix_router_raygen.ptx"
if errorlevel 1 echo RAYGEN FAILED

echo Compiling optix_router_hitgroup.ptx for sm_120...
nvcc --ptx -arch=compute_89 -I"%OPTIX_INC%" -I"%PROJ%\include" -I"%PROJ%\cuda" -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING "%PROJ%\cuda\optix_router_hitgroup.cu" -o "%PTX_OUT%\optix_router_hitgroup.ptx"
if errorlevel 1 echo HITGROUP FAILED

echo Done.
