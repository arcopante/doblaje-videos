@echo off
REM ================================================================
REM Configuracion para doblaje de videos - Windows
REM ================================================================

REM Directorio con videos a procesar
set INPUT_DIRECTORY=.\videos_a_transcribir

REM Modelo Whisper: tiny | base | small | medium | large
set WHISPER_MODEL=medium

REM Idioma destino (ISO 639-1)
set TARGET_LANGUAGE=es

REM Modo de salida:
REM   dub  -> Doblaje TTS + Demucs + remontaje
REM   burn -> Subtitulos quemados en imagen
REM   soft -> Pista suave seleccionable en MKV
REM   None -> Solo genera txt + srt
set SUBTITLE_MODE=dub

REM Modelo Coqui TTS (solo para SUBTITLE_MODE="dub")
REM   XTTS v2 (multilingue, clona voz): tts_models/multilingual/multi-dataset/xtts_v2
REM   VITS espanol (mas rapido): tts_models/es/css10/vits
set TTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2

REM Modelo Demucs (solo para SUBTITLE_MODE="dub"): htdemucs | htdemucs_ft
set DEMUCS_MODEL=htdemucs

REM Volumenes de mezcla (solo para SUBTITLE_MODE="dub")
REM 1.0 = volumen original sin cambio
set BACKGROUND_VOLUME=1.0
set VOICE_VOLUME=1.7

REM Backend de traduccion: ollama | google
set TRANSLATOR_BACKEND=google

REM Modelo Ollama (solo si TRANSLATOR_BACKEND=ollama)
set OLLAMA_MODEL=qwen3:4b

REM ================================================================
cd /d "%~dp0"
python transcribe_dobla_videos.py