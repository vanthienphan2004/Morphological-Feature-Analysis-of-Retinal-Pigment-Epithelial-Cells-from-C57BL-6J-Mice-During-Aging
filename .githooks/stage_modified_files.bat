@echo off
for /f "tokens=*" %%i in ('git diff --name-only --cached -- "*.py"') do git add "%%i"