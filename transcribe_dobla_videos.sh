#!/bin/bash
# ================================================================
# Configuración para doblaje de vídeos
# Edita los valores según tus necesidades
# ================================================================

# Directorio con vídeos a procesar
export INPUT_DIRECTORY="./videos_a_transcribir"

# Modelo Whisper: tiny | base | small | medium | large
export WHISPER_MODEL="medium"

# Idioma destino (ISO 639-1)
export TARGET_LANGUAGE="es"

# Modo de salida:
#   dub  -> Doblaje TTS + Demucs + remontaje
#   burn -> Subtítulos quemados en imagen
#   soft -> Pista suave seleccionable en MKV
#   None -> Solo genera srt
export SUBTITLE_MODE="dub"

# Modelo Coqui TTS (solo para SUBTITLE_MODE="dub")
#   XTTS v2 (multilingüe, clona voz): tts_models/multilingual/multi-dataset/xtts_v2
#   VITS español (más rápido): tts_models/es/css10/vits
export TTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

# Modelo Demucs (solo para SUBTITLE_MODE="dub"): htdemucs | htdemucs_ft
export DEMUCS_MODEL="htdemucs"

# Volúmenes de mezcla (solo para SUBTITLE_MODE="dub")
export BACKGROUND_VOLUME="1.0"
export VOICE_VOLUME="1.7"

# Modelo Ollama para traducción
export OLLAMA_MODEL="gemma3:4b"

# Backend de traducción: ollama | google
export TRANSLATOR_BACKEND="google"

# ================================================================
#cd "$(dirname "$0")"
# EJECUTO EL PYTHON DESDE DONDE SE ENCUENTRA EL SH, PERO MANTENIENDOME EN EL DIRECTORIO ACTUAL QUE ES DONDE SE ENCUENTRA EL VIDEO
python "$(dirname "$0")"/transcribe_dobla_videos.py
