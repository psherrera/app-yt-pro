document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const videoUrlInput = document.getElementById('video-url');
    const fetchBtn = document.getElementById('fetch-info-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const qualityWarning = document.getElementById('quality-warning');
    const videoInfoCard = document.getElementById('video-info-card');
    const thumbnailImg = document.getElementById('video-thumbnail');
    const titleEl = document.getElementById('video-title');
    const uploaderEl = document.getElementById('video-uploader');
    const descriptionEl = document.getElementById('video-description');
    const formatSelect = document.getElementById('format-select');
    const downloadBtn = document.getElementById('download-btn');
    const downloadThumbBtn = document.getElementById('download-thumb-btn');
    const downloadProgress = document.getElementById('download-progress');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.getElementById('progress-text');
    const progressPercentage = document.getElementById('progress-percentage');
    const showTranscriptBtn = document.getElementById('show-transcript-btn');
    const transcriptSection = document.getElementById('transcript-section');
    const transcriptContent = document.getElementById('transcript-content');
    const copyTranscriptBtn = document.getElementById('copy-transcript-btn');
    const downloadTxtBtn = document.getElementById('download-txt-btn');
    const appSubtitle = document.getElementById('app-subtitle');
    const tabTitle = document.getElementById('tab-title');
    const inputIcon = document.getElementById('input-icon');

    const loginModal = document.getElementById('login-modal');
    const loginBtn = document.getElementById('login-btn');
    const passwordInput = document.getElementById('app-password');
    const loginError = document.getElementById('login-error');
    const clearUrlBtn = document.getElementById('clear-url-btn');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const pwaInstallBtn = document.getElementById('pwa-install-btn');

    // Navigation Logic
    const navItems = document.querySelectorAll('.nav-item');
    
    // API Configuration (Detección inteligente Local vs Render)
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
    const isSharedPort = window.location.port === '5000' || window.location.port === '10000';
    const API_BASE = (isLocal && !isSharedPort) ? 'http://localhost:5000/api' : '/api';
    const APP_PASSWORD = 'pablo'; 
    
    let currentTab = 'youtube';

    // Tab Logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            if (item.classList.contains('cursor-not-allowed')) return;

            navItems.forEach(i => {
                i.classList.remove('active', 'text-primary');
                i.classList.add('text-slate-500');
            });

            item.classList.add('active', 'text-primary');
            item.classList.remove('text-slate-500');
            
            currentTab = item.dataset.tab;
            
            // Update UI based on tab
            if (currentTab === 'youtube') {
                tabTitle.textContent = "YouTube Downloader";
                videoUrlInput.placeholder = "Pega el enlace de YouTube aquí...";
                appSubtitle.textContent = "Descarga entrevistas y videos de YouTube con la mejor calidad.";
                inputIcon.textContent = "link";
            } else if (currentTab === 'instagram') {
                tabTitle.textContent = "Instagram Downloader";
                videoUrlInput.placeholder = "Pega el enlace de Instagram (Reel o Video) aquí...";
                appSubtitle.textContent = "Descarga Reels y Videos de Instagram fácilmente.";
                inputIcon.textContent = "photo_camera";
            }

            // Reset results when switching tabs
            videoInfoCard.classList.add('hidden');
            videoUrlInput.value = '';
        });
    });

    // Login Logic
    loginBtn.addEventListener('click', () => {
        if (passwordInput.value === APP_PASSWORD) {
            loginModal.classList.add('opacity-0');
            setTimeout(() => loginModal.classList.add('hidden'), 500);
            localStorage.setItem('app_logged_in', 'true');
        } else {
            loginError.classList.remove('hidden');
            passwordInput.value = '';
            // Shake effect for error
            loginModal.querySelector('.glass').classList.add('animate-bounce');
            setTimeout(() => {
                loginModal.querySelector('.glass').classList.remove('animate-bounce');
            }, 500);
        }
    });

    passwordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loginBtn.click();
    });

    // Check session
    if (localStorage.getItem('app_logged_in') === 'true') {
        loginModal.classList.add('hidden');
    }

    // --- Server Status Monitor ---
    const updateStatusUI = (status) => {
        // Reset styles
        statusDot.className = 'w-2 h-2 rounded-full';
        statusText.className = 'text-xs font-medium';
        
        if (status === 'online') {
            statusDot.classList.add('bg-emerald-500', 'shadow-[0_0_8px_rgba(16,185,129,0.5)]');
            statusText.classList.add('text-emerald-500');
            statusText.textContent = 'Online';
        } else if (status === 'issues') {
            statusDot.classList.add('bg-amber-500', 'animate-pulse');
            statusText.classList.add('text-amber-500');
            statusText.textContent = 'Cookie Issues';
        } else {
            statusDot.classList.add('bg-slate-600');
            statusText.classList.add('text-slate-500');
            statusText.textContent = 'Offline';
        }
    };

    const checkServerStatus = async () => {
        try {
            const response = await fetch(`${API_BASE}/health/cookies`);
            if (response.ok) {
                const data = await response.json();
                updateStatusUI(data.status === 'ok' ? 'online' : 'issues');
            } else {
                updateStatusUI('offline');
            }
        } catch (error) {
            console.error("Status check failed:", error);
            updateStatusUI('offline');
        }
    };

    checkServerStatus();
    setInterval(checkServerStatus, 60000);

    // --- Clear URL Input Logic ---
    videoUrlInput.addEventListener('input', () => {
        if (videoUrlInput.value.trim().length > 0) {
            clearUrlBtn.classList.remove('hidden');
        } else {
            clearUrlBtn.classList.add('hidden');
        }
    });

    clearUrlBtn.addEventListener('click', () => {
        videoUrlInput.value = '';
        clearUrlBtn.classList.add('hidden');
        videoUrlInput.focus();
        // Also hide results if cleared
        videoInfoCard.classList.add('hidden');
    });

    // --- PWA Installation Logic ---
    let deferredPrompt;
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        pwaInstallBtn.classList.remove('hidden');
    });

    pwaInstallBtn.addEventListener('click', async () => {
        if (!deferredPrompt) return;
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        if (outcome === 'accepted') {
            pwaInstallBtn.classList.add('hidden');
        }
        deferredPrompt = null;
    });

    window.addEventListener('appinstalled', () => {
        pwaInstallBtn.classList.add('hidden');
        deferredPrompt = null;
    });

    // --- Web Share Target Handler ---
    const handleSharedContent = () => {
        const params = new URLSearchParams(window.location.search);
        const sharedUrl = params.get('url') || params.get('text') || params.get('title');
        
        if (sharedUrl) {
            // Clean up the URL (sometimes apps share text + URL together)
            const urlRegex = /(https?:\/\/[^\s]+)/g;
            const found = sharedUrl.match(urlRegex);
            const cleanUrl = found ? found[0] : sharedUrl;

            if (cleanUrl.startsWith('http')) {
                videoUrlInput.value = cleanUrl;
                // Si ya está logueado, disparamos el análisis automáticamente
                if (localStorage.getItem('app_logged_in') === 'true') {
                    setTimeout(() => fetchBtn.click(), 500);
                }
            }
        }
    };

    handleSharedContent();

    let currentMaxResThumbnail = '';

    // Fetch Video Info
    fetchBtn.addEventListener('click', async () => {
        const url = videoUrlInput.value.trim();
        if (!url) {
            const platform = currentTab === 'youtube' ? 'YouTube' : 'Instagram';
            alert(`Por favor, pega una URL válida de ${platform}`);
            return;
        }

        // Reset UI Components
        videoInfoCard.classList.add('hidden');
        qualityWarning.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        transcriptSection.classList.add('hidden');
        showTranscriptBtn.classList.add('hidden');
        downloadTxtBtn.classList.add('hidden');
        
        // Clear previous state
        titleEl.textContent = '';
        uploaderEl.textContent = '';
        descriptionEl.textContent = '';
        thumbnailImg.src = '';
        formatSelect.innerHTML = '';

        try {
            const response = await fetch(`${API_BASE}/video-info`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                const errMsg = errData.detail || errData.error || 'Error desconocido en el servidor';
                alert(`Error: ${errMsg}`);
                return;
            }

            const data = await response.json();

            if (data.error) {
                const cleanError = data.error.replace(/\u001b\[[0-9;]*m/g, '');
                alert(cleanError);
                return;
            }

            if (!data.formats || !Array.isArray(data.formats)) {
                alert('No se encontraron formatos disponibles para este video.');
                return;
            }

            // Populate Video Info
            if (data.thumbnail) {
                let thumbUrl = data.thumbnail;
                if (thumbUrl.startsWith('/')) {
                    thumbUrl = `${API_BASE}${thumbUrl}`;
                }
                thumbnailImg.src = thumbUrl;
                thumbnailImg.style.display = '';
            } else {
                thumbnailImg.style.display = 'none';
            }
            titleEl.textContent = data.title;
            uploaderEl.textContent = data.uploader;
            descriptionEl.textContent = data.description;
            currentMaxResThumbnail = data.max_res_thumbnail;
            
            const durationEl = document.getElementById('video-duration');
            if (durationEl && data.duration) {
                const mins = Math.floor(data.duration / 60);
                const secs = data.duration % 60;
                durationEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            }

            // Fill Formats and Selectors
            formatSelect.innerHTML = '';
            data.formats.forEach(f => {
                const option = document.createElement('option');
                option.value = f.format_id;
                const size = f.filesize ? `(${(f.filesize / (1024 * 1024)).toFixed(1)} MB)` : 'Size N/A';
                option.textContent = `${f.label} - ${size}`;
                formatSelect.appendChild(option);
            });

            if (data.formats.length === 0) {
                const option = document.createElement('option');
                option.value = 'best';
                option.textContent = 'Mejor calidad disponible';
                formatSelect.appendChild(option);
            }

            if (!data.has_ffmpeg && currentTab === 'youtube') {
                qualityWarning.classList.remove('hidden');
            }

            // Always show transcript button for smarter detection
            showTranscriptBtn.classList.remove('hidden');

            // Reveal result card with animation
            videoInfoCard.classList.remove('hidden');
            videoInfoCard.classList.add('fade-in');
        } catch (error) {
            console.error('Error:', error);
            alert('Error al conectar con el servidor. ¿Está el backend encendido?');
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });

    // Download Logic
    downloadBtn.addEventListener('click', async () => {
        const url = videoUrlInput.value.trim();
        const formatId = formatSelect.value;

        downloadBtn.disabled = true;
        downloadProgress.classList.remove('hidden');
        
        // Progress Simulation for visual feedback
        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 5;
                updateProgress(Math.floor(progress), 'Procesando descarga en el servidor...');
            }
        }, 800);

        function updateProgress(val, text) {
            progressFill.style.width = `${val}%`;
            progressPercentage.textContent = `${val}%`;
            if (text) progressText.textContent = text;
        }

        try {
            const response = await fetch(`${API_BASE}/download`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, format_id: formatId })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Error en la descarga');
            }

            clearInterval(interval);
            updateProgress(95, 'Preparando transferencia segura...');

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            
            const filename = `${titleEl.textContent.substring(0, 30).trim() || 'video'}.mp4`;
            a.download = filename;
            
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
            a.remove();

            updateProgress(100, '¡Descarga completada con éxito!');
            
            setTimeout(() => {
                downloadProgress.classList.add('hidden');
                downloadBtn.disabled = false;
            }, 5000);

        } catch (error) {
            clearInterval(interval);
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
            downloadProgress.classList.add('hidden');
        } finally {
            downloadBtn.disabled = false;
        }
    });

    // Transcript Extractor Logic
    showTranscriptBtn.addEventListener('click', async () => {
        const url = videoUrlInput.value.trim();
        transcriptSection.classList.remove('hidden');
        transcriptSection.classList.add('fade-in');
        transcriptContent.innerHTML = '<p class="animate-pulse">Analizando y extrayendo texto con IA... por favor espera.</p>';
        showTranscriptBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/transcript`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });

            const data = await response.json();

            if (data.error) {
                const cleanError = data.error.replace(/\u001b\[[0-9;]*m/g, '');
                transcriptContent.innerHTML = `<div class="bg-red-500/10 p-4 rounded-xl border border-red-500/20 text-red-400">Error: ${cleanError}</div>`;
            } else {
                const methodLabel = data.method === 'whisper' ? ' (Procesado con IA Whisper)' : ' (Subtítulos directos)';
                transcriptContent.innerHTML = `
                    <p class="mb-4">${data.transcript}</p>
                    <div class="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest bg-white/5 py-1 px-3 rounded-full w-fit">
                        <span class="material-symbols-outlined text-xs">info</span>
                        Metodo: ${methodLabel}
                    </div>
                `;
                downloadTxtBtn.classList.remove('hidden');
                downloadTxtBtn.classList.add('fade-in');
            }
        } catch (error) {
            transcriptContent.innerHTML = `<p class="text-red-400">Error al conectar con el servidor.</p>`;
        } finally {
            showTranscriptBtn.disabled = false;
        }
    });

    // Action Helpers
    downloadThumbBtn.addEventListener('click', () => {
        if (!currentMaxResThumbnail) return;
        window.open(currentMaxResThumbnail, '_blank');
    });

    copyTranscriptBtn.addEventListener('click', () => {
        const text = transcriptContent.textContent.split('Metodo:')[0].trim();
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = copyTranscriptBtn.innerHTML;
            copyTranscriptBtn.innerHTML = '<span class="material-symbols-outlined text-sm">check</span> COPIADO';
            setTimeout(() => {
                copyTranscriptBtn.innerHTML = originalHTML;
            }, 2000);
        });
    });

    downloadTxtBtn.addEventListener('click', () => {
        const text = transcriptContent.textContent.split('Metodo:')[0].trim();
        const blob = new Blob([text], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `transcripcion_${titleEl.textContent.substring(0, 20)}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    });
});
