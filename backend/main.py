import os
import uuid
import json
import gc
import re
import tempfile
import torch
import yt_dlp
import requests
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deep_translator import GoogleTranslator
import whisper
from fastapi import Response

# --- CONFIGURACIÓN DE ENTORNO ---
IS_RENDER = os.environ.get('RENDER') is not None
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    groq_client = None

app = FastAPI(title="YT Downloader Pro API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, 'downloads')
CACHE_FILE = os.path.join(BASE_DIR, 'transcripts_cache.json')

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# --- MODELO WHISPER (Ajustado a local PC) ---
# Usamos 'base' para mejor precisión en PC
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    # En PC priorizamos el modelo local 'base' para mejor calidad si no hay Groq o si se prefiere local
    if _whisper_model is None:
        try:
            print("Cargando modelo Whisper 'base' en memoria...")
            # Intentar cargar desde cache persistente si existe
            model_path = os.environ.get('WHISPER_CACHE_DIR', os.path.join(os.path.expanduser("~"), ".cache", "whisper"))
            _whisper_model = whisper.load_model("base", download_root=model_path)
        except Exception as e:
            print(f"Error cargando Whisper local: {e}")
            return None
    return _whisper_model

# --- CACHÉ ---
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# --- TRADUCCIÓN ---
def translate_to_spanish(text):
    if not text: return ""
    try:
        translator = GoogleTranslator(source='auto', target='es')
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            translated = [translator.translate(c) for c in chunks]
            return " ".join(translated)
        return translator.translate(text)
    except Exception as e:
        print(f"Error traducción: {e}")
        return text

# --- MODELOS DE DATOS ---
class VideoRequest(BaseModel):
    url: str
    format_id: Optional[str] = "best"

# --- ENDPOINTS ---

@app.post("/api/video-info")
async def get_video_info(req: VideoRequest, request: Request):
    url = req.url
    is_youtube = 'youtube.com' in url or 'youtu.be' in url
    
    # Opciones unificadas y robustas para evitar 403 Forbidden
    def get_robust_opts(target_url, extra={}):
        cookie_path = os.path.join(BASE_DIR, 'cookies.txt')
        opts = {
            'quiet': True,
            'no_warnings': True,
            'cachedir': False,
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            **extra
        }
        if os.path.exists(cookie_path):
            print(f"DEBUG: Cargando cookies desde {cookie_path}")
            opts['cookiefile'] = cookie_path
        else:
            print(f"DEBUG: No se encontró archivo de cookies en {cookie_path}")
        
        # Estrategia de clientes para YouTube
        if 'youtube.com' in target_url or 'youtu.be' in target_url:
            opts['extractor_args'] = {'youtube': {'player_client': ['android', 'ios', 'tv']}}
        return opts

    info = None
    last_error = ""
    
    # Intentos de extracción con opciones robustas
    try:
        opts = get_robust_opts(url)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        last_error = str(e)
        print(f"Error en extracción primaria: {last_error}")
        # Intento secundario con cliente web si falla
        try:
            opts = get_robust_opts(url)
            opts['extractor_args'] = {'youtube': {'player_client': ['web', 'tv']}}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e2:
            last_error = f"{last_error} | {str(e2)}"

    if not info:
        raise HTTPException(status_code=500, detail=f"No se pudo obtener información: {last_error}")

    # Procesar formatos
    formats = []
    seen_res = set()
    all_formats = info.get('formats', [])
    useful_formats = [f for f in all_formats if f.get('vcodec') != 'none']
    useful_formats.sort(key=lambda x: (x.get('height') or 0), reverse=True)

    for f in useful_formats:
        res = f.get('resolution') or f"{f.get('height')}p"
        if res == "Nonep" or not f.get('height'):
            res = f.get('format_note') or f.get('format_id') or "Calidad única"
        
        ext = f.get('ext', 'mp4')
        res_key = f"{res}_{ext}"
        if res_key not in seen_res:
            formats.append({
                'format_id': f.get('format_id'),
                'ext': ext,
                'resolution': res,
                'filesize': f.get('filesize') or f.get('filesize_approx'),
                'label': f"{res} (.{ext})"
            })
            seen_res.add(res_key)

    # Proxy para Instagram thumbnails (Usamos ruta relativa para que el frontend la complete)
    thumbnail = info.get('thumbnail')
    if 'instagram.com' in url and thumbnail:
        thumbnail = f"/proxy-thumbnail?url={thumbnail}"
        print(f"DEBUG: Instagram Thumbnail proxied (relative): {thumbnail}")

    return {
        'title': info.get('title'),
        'thumbnail': thumbnail,
        'max_res_thumbnail': thumbnail,
        'duration': info.get('duration'),
        'uploader': info.get('uploader') or "Desconocido",
        'description': (info.get('description') or 'Sin descripción')[:200] + '...',
        'formats': formats,
        'has_ffmpeg': True, # En Docker siempre tenemos FFmpeg
        'has_subtitles': bool(info.get('subtitles') or info.get('automatic_captions'))
    }

@app.post("/api/transcript")
async def get_transcript(req: VideoRequest):
    url = req.url
    cache = load_cache()
    if url in cache:
        return {"transcript": cache[url], "method": "cache"}

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 1. Intentar descargar subtítulos directos
            def get_robust_opts(target_url, extra={}):
                cookie_path = os.path.join(BASE_DIR, 'cookies.txt')
                opts = {
                    'skip_download': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['es.*', 'en.*'],
                    'outtmpl': os.path.join(tmpdir, 'sub.%(ext)s'),
                    'quiet': True,
                    'noplaylist': True,
                    'nocheckcertificate': True,
                    'ignoreerrors': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    **extra
                }
                if os.path.exists(cookie_path):
                    opts['cookiefile'] = cookie_path
                if 'youtube.com' in target_url or 'youtu.be' in target_url:
                    opts['extractor_args'] = {'youtube': {'player_client': ['android', 'ios', 'tv']}}
                return opts

            if 'youtube.com' in url or 'youtu.be' in url:
                ydl_opts_subs = get_robust_opts(url)
                with yt_dlp.YoutubeDL(ydl_opts_subs) as ydl:
                    ydl.extract_info(url, download=True)
                    sub_file = None
                    is_english = False
                    # Buscar español primero
                    for f in os.listdir(tmpdir):
                        if f.startswith('sub.') and ('.es' in f or '.es-419' in f):
                            sub_file = os.path.join(tmpdir, f)
                            break
                    # Si no, inglés
                    if not sub_file:
                        for f in os.listdir(tmpdir):
                            if f.startswith('sub.') and ('.en' in f or '.en-US' in f):
                                sub_file = os.path.join(tmpdir, f)
                                is_english = True
                                break
                    
                    if sub_file:
                        with open(sub_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Limpieza de VTT
                        content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
                        content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', content)
                        content = re.sub(r'^\d+\n', '', content, flags=re.MULTILINE)
                        content = re.sub(r'<[^>]*>', '', content)
                        
                        final_text = ' '.join([line.strip() for line in content.split('\n') if line.strip()])
                        if is_english: final_text = translate_to_spanish(final_text)
                        
                        cache[url] = final_text
                        save_cache(cache)
                        return {"transcript": final_text, "method": "subtitles"}

            raise Exception("No direct subtitles")

        except Exception:
            # 2. Descargar audio y usar Whisper
            try:
                def get_robust_opts(target_url, extra={}):
                    cookie_path = os.path.join(BASE_DIR, 'cookies.txt')
                    opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '128', 
                        }],
                        'quiet': True,
                        'nocheckcertificate': True,
                        'ignoreerrors': True,
                        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                        **extra
                    }
                    if os.path.exists(cookie_path):
                        opts['cookiefile'] = cookie_path
                    if 'youtube.com' in target_url or 'youtu.be' in target_url:
                        opts['extractor_args'] = {'youtube': {'player_client': ['android', 'ios', 'tv']}}
                    return opts
                
                audio_opts = get_robust_opts(url)
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    ydl.download([url])
                    audio_file = None
                    for f in os.listdir(tmpdir):
                        if f.startswith('audio.'):
                            audio_file = os.path.join(tmpdir, f)
                            break
                    
                    if not audio_file: raise Exception("No se pudo descargar audio")
                    
                    # 2.1 Intentar con Groq API (Más rápido y ligero)
                    if groq_client:
                        try:
                            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
                            if file_size_mb < 25:
                                print(f"Usando Groq API ({file_size_mb:.1f}MB)...")
                                with open(audio_file, "rb") as f:
                                    transcription = groq_client.audio.transcriptions.create(
                                        file=(audio_file, f.read()),
                                        model="whisper-large-v3",
                                        response_format="text",
                                        language="es"
                                    )
                                cache[url] = transcription
                                save_cache(cache)
                                return {"transcript": transcription, "method": "groq_whisper_v3"}
                            else:
                                print(f"Audio demasiado grande ({file_size_mb:.1f}MB) para Groq, usando Whisper local...")
                        except Exception as ge:
                            print(f"Error en Groq (usando fallback local): {ge}")

                    # 2.2 IA Local (Solo si no estamos en Render o Groq falló)
                    model = get_whisper_model()
                    if not model:
                        raise Exception("IA Local no disponible (Límite de RAM). Por favor configura GROQ_API_KEY.")
                    
                    result = model.transcribe(audio_file)
                    text = result['text'].strip()
                    if result.get('language') != 'es': text = translate_to_spanish(text)
                    
                    cache[url] = text
                    save_cache(cache)
                    return {"transcript": text, "method": "whisper_local"}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/download")
async def download_video(req: VideoRequest):
    url = req.url
    format_id = req.format_id
    uid = str(uuid.uuid4())
    output_template = os.path.join(DOWNLOAD_FOLDER, f'%(title)s_{uid}.%(ext)s')
    
    def get_robust_opts(target_url, extra={}):
        cookie_path = os.path.join(BASE_DIR, 'cookies.txt')
        opts = {
            'format': format_id,
            'outtmpl': output_template,
            'quiet': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            **extra
        }
        if os.path.exists(cookie_path):
            print(f"DEBUG: Cargando cookies desde {cookie_path}")
            opts['cookiefile'] = cookie_path
        else:
            print(f"DEBUG: No se encontró archivo de cookies en {cookie_path}")
        if 'youtube.com' in target_url or 'youtu.be' in target_url:
            opts['extractor_args'] = {'youtube': {'player_client': ['android', 'ios', 'tv']}}
        return opts

    opts = get_robust_opts(url)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
            # Encontrar archivo
            for f in os.listdir(DOWNLOAD_FOLDER):
                if uid in f:
                    return FileResponse(os.path.join(DOWNLOAD_FOLDER, f), filename=f)
            raise Exception("Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/proxy-thumbnail")
async def proxy_thumbnail(url: str):
    print(f"DEBUG: Proxy request for: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Referer': 'https://www.instagram.com/'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        print(f"DEBUG: Proxy success, Content-Type: {resp.headers.get('Content-Type')}")
        return Response(content=resp.content, media_type=resp.headers.get('Content-Type', 'image/jpeg'))
    except Exception as e:
        print(f"DEBUG: Proxy FAILED: {e}")
        return Response(status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
