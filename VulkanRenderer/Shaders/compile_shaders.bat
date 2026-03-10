@echo off
setlocal enabledelayedexpansion

for %%F in (*.hlsl) do (
    rem Split full filename: name.stage.hlsl
    for /f "tokens=1-3 delims=." %%a in ("%%F") do (
        set "base=%%a"
        set "stage=%%b"
        set "ext2=%%c"
    )

    rem Validate structure
    if /i "!ext2!"=="hlsl" (
        if /i "!stage!"=="frag" (
            set "profile=ps_6_6"
            set "compile=1"
        ) else if /i "!stage!"=="vert" (
            set "profile=vs_6_6"
            set "compile=1"
        ) else (
            set "compile=0"
        )
    ) else (
        set "compile=0"
    )

    if "!compile!"=="1" (
        echo Compiling %%F with !profile!...
        dxc.exe -spirv -T !profile! -E main -Fo "%%~nF.spv" "%%F"
    ) else (
        echo Skipping %%F
    )
)

echo Done.
