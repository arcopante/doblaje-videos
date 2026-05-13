# Doblaje de Videos - Transcripcion, Traduccion y Doblaje con TTS

Pipeline completo para transcribir videos, traducir subtitulos y generar audio doblado usando Coqui TTS.

## Requisitos del Sistema

- Python 3.10+
- NVIDIA GPU con 8GB+ VRAM (para TTS y Demucs)
- ffmpeg instalado en el sistema
- Ollama (solo si usas traduccion con LLM)

## Instalacion

### 1. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install openai-whisper
pip install moviepy tqdm
pip install deep-translator
pip install pydub
pip install TTS
pip install demucs
```

### 3. Instalar ffmpeg

**Linux:**
```bash
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Windows:**
Descarga desde https://ffmpeg.org/download.html

### 4. (Opcional) Ollama para traduccion con LLM

Instala Ollama desde https://ollama.ai/, luego descarga el modelo:

```bash
ollama pull qwen3:4b
```

## Configuracion

Edita `transcribe_dobla_videos.sh` para ajustar los parametros:

```bash
# Directorio con videos a procesar
export INPUT_DIRECTORY="./videos_a_transcribir"

# Modelo Whisper: tiny | base | small | medium | large
export WHISPER_MODEL="medium"

# Idioma destino (ISO 639-1)
export TARGET_LANGUAGE="es"

# Modo de salida: dub | burn | soft | None
export SUBTITLE_MODE="dub"

# Modelo TTS
export TTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

# Modelo Demucs: htdemucs | htdemucs_ft
export DEMUCS_MODEL="htdemucs"

# Volumenes de mezcla
export BACKGROUND_VOLUME="1.0"
export VOICE_VOLUME="1.7"

# Backend de traduccion: ollama | google
export TRANSLATOR_BACKEND="google"

# Modelo Ollama (solo si TRANSLATOR_BACKEND=ollama)
export OLLAMA_MODEL="qwen3:4b"
```

## Uso

```bash
./transcribe_dobla_videos.sh
```

El script procesara todos los videos en el directorio configurado.

## Modelos TTS Disponibles

### XTTS v2 (recomendado)
- Multilingue con clonacion de voz
- ~1.8 GB
- Usa vocals.wav de Demucs como referencia para voz consistente

```bash
export TTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"
```

### VITS español
- Mas rapido, ~300 MB
- Sin clonacion de voz

```bash
export TTS_MODEL_NAME="tts_models/es/css10/vits"
```

## Modelos Whisper

| Modelo  | VRAM | Velocidad | Precision |
|---------|------|------------|-----------|
| tiny    | 1GB  | Muy rapido | Baja      |
| base    | 1GB  | Rapido     | Media     |
| small   | 2GB  | Media      | Buena     |
| medium  | 5GB  | Lenta      | Muy buena |
| large   | 10GB | Muy lenta  | Excelente |

## Modelos Demucs

- `htdemucs`: Modelo por defecto, buena relacion calidad/velocidad
- `htdemucs_ft`: Version fine-tuned, mejor calidad, mas lento

## Backends de Traduccion

### Google Translator (recomendado)
Traduccion rapida y gratuita.

```bash
export TRANSLATOR_BACKEND="google"
```

### Ollama (LLM)
Traduccion con modelo local, mas control pero mas lento.

```bash
export TRANSLATOR_BACKEND="ollama"
export OLLAMA_MODEL="qwen3:4b"
```

## Limpieza de VRAM

El script libera automaticamente la VRAM entre fases (Whisper -> Demucs -> TTS) para evitar errores de memoria. Si tienes menos de 8GB de VRAM, considera usar modelos mas pequenos.

## Solucion de Problemas

### CUDA out of memory
Usa modelos mas pequenos o reduce el numero de procesos paralelos.

### TTS no usa GPU
Verifica que PyTorch detecta CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### ffmpeg no encontrado
Instala ffmpeg segun las instrucciones de tu sistema operativo.