@echo off
cd /d "%~dp0"
echo ==========================================
echo   YouTube Downloader PRO (Docker Edition)
echo ==========================================
echo.

:: 1. Verificar Docker
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker no esta en ejecucion o no esta instalado.
    echo Por favor, abre Docker Desktop e intenta de nuevo.
    pause
    exit /b
)

:: 2. Copiar cookies si existen en la carpeta vieja
if exist "..\backend\cookies.txt" (
    echo [+] Copiando cookies previas para YouTube...
    copy "..\backend\cookies.txt" "backend\cookies.txt" >nul
)

:: 3. Levantar contenedores
echo [+] Construyendo e iniciando contenedores (esto puede tardar la primera vez)...
docker-compose up -d --build

:: 4. Esperar un poco
echo [+] Esperando a que el servidor este listo...
timeout /t 5 >nul

:: 5. Abrir en el navegador
echo [+] ¡Listo! Abriendo la interfaz...
start http://localhost

echo.
echo ==========================================
echo   INFO: El backend corre en el puerto 5000
echo   INFO: El frontend corre en el puerto 80
echo   Para apagar todo: docker-compose down
echo ==========================================
pause
