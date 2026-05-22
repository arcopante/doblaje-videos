# Doblaje de Videos — Transcripción, Traducción y Doblaje con IA

Pipeline completo para:

- Transcribir vídeos con Whisper
- Traducir subtítulos automáticamente
- Generar doblaje IA con clonación de voz
- Crear subtítulos quemados o pistas suaves

Compatible con:

- Linux (CPU y GPU NVIDIA)
- macOS Intel
- macOS Apple Silicon (M1/M2/M3/M4)

Usa:

- Whisper
- Coqui TTS — fork `coqui-tts` (XTTS v2)
- Demucs
- ffmpeg
- Ollama o Google Translate

------------------------------------------------------------
CARACTERÍSTICAS
------------------------------------------------------------

[1] Transcripción automática
Convierte el audio del vídeo a texto usando Whisper.

[2] Traducción automática
Dos modos disponibles:

- Google Translate (rápido y sencillo)
- Ollama local (más control y privacidad)

[3] Doblaje IA real
Genera voz sintética sincronizada con el vídeo usando:

- XTTS v2
- Clonación de voz automática
- Separación de voces con Demucs

[4] Tres modos de salida

Modo "dub"
Doblaje completo IA.

Salida:
video_es.mp4

Modo "burn"
Subtítulos incrustados permanentemente.

Salida:
video_es_sub.mp4

Modo "soft"
Pista de subtítulos seleccionable.

Salida:
video_es_soft.mkv

------------------------------------------------------------
REQUISITOS
------------------------------------------------------------

Requisitos generales:

- Python 3.11 recomendado (también funciona 3.12)
- ffmpeg instalado en el sistema
- 8 GB RAM mínimo
- GPU opcional pero recomendada

NOTA SOBRE VERSIONES DE PYTHON Y TTS:

El paquete original de Coqui TTS ("TTS" en PyPI) fue abandonado en
diciembre de 2023 y no soporta Python 3.12 ni publica wheels para
macOS Apple Silicon.

Se usa en su lugar el fork activo "coqui-tts", que:
- Soporta Python 3.11 y 3.12
- Publica wheels nativos para macOS (ARM64) desde v0.24.2
- Incluye correcciones y nuevas funcionalidades

IMPORTANTE:
Instala siempre "coqui-tts", no el paquete "TTS" original.

------------------------------------------------------------
INSTALACIÓN RÁPIDA CON REQUIREMENTS.TXT
------------------------------------------------------------

Se incluyen dos ficheros de dependencias listos para usar:

  requirements_macos_silicon.txt  →  macOS Apple Silicon (M1/M2/M3/M4)
  requirements_linux.txt          →  Linux (CPU o GPU NVIDIA)

Uso:

1. Crear y activar entorno virtual (ver secciones de instalación)
2. Instalar dependencias:

   macOS Silicon:
   pip install -r requirements_macos_silicon.txt

   Linux (con GPU NVIDIA, instalar PyTorch con CUDA antes — ver sección Linux):
   pip install -r requirements_linux.txt

------------------------------------------------------------
INSTALACIÓN — macOS Apple Silicon (M1/M2/M3/M4)
------------------------------------------------------------

1. Instalar Homebrew
https://brew.sh

2. Instalar Python 3.11

brew install python@3.11

3. Instalar ffmpeg

brew install ffmpeg

4. Crear entorno virtual

python3.11 -m venv .venv

5. Activar entorno

source .venv/bin/activate

6. Actualizar pip

pip install --upgrade pip setuptools wheel

7. Instalar PyTorch
El índice estándar ya incluye soporte MPS (Metal Performance Shaders)
para Apple Silicon:

pip install torch torchvision torchaudio

8. Instalar dependencias

pip install \
openai-whisper \
moviepy \
tqdm \
deep-translator \
pydub \
demucs \
requests \
numpy \
ffmpeg-python

9. Instalar Coqui TTS (FORK ACTIVO — no instalar el paquete "TTS" original)

pip install coqui-tts

------------------------------------------------------------
INSTALACIÓN — macOS Intel
------------------------------------------------------------

Mismos pasos que Apple Silicon. La única diferencia es que el chip
Intel no dispone de MPS, por lo que Whisper y Demucs correrán en CPU.

------------------------------------------------------------
INSTALACIÓN — Linux (Ubuntu/Debian)
------------------------------------------------------------

1. Instalar dependencias del sistema

sudo apt update

sudo apt install -y \
python3.11 \
python3.11-venv \
ffmpeg \
git

2. Crear entorno virtual

python3.11 -m venv .venv

3. Activar entorno

source .venv/bin/activate

4. Actualizar pip

pip install --upgrade pip setuptools wheel

5. Instalar PyTorch

GPU NVIDIA — elige según tu versión de CUDA:

CUDA 12.1:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

CUDA 11.8:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

CPU solamente:
pip install torch torchvision torchaudio

Comprueba tu versión de CUDA con: nvidia-smi

6. Instalar dependencias

pip install \
openai-whisper \
moviepy \
tqdm \
deep-translator \
pydub \
demucs \
requests \
numpy \
ffmpeg-python

7. Instalar Coqui TTS (FORK ACTIVO)

pip install coqui-tts

------------------------------------------------------------
INSTALACIÓN DE OLLAMA (OPCIONAL)
------------------------------------------------------------

Solo necesario si usas:

export TRANSLATOR_BACKEND="ollama"

Instalar:
https://ollama.com

Descargar modelo:

ollama pull qwen3:4b

------------------------------------------------------------
CONFIGURACIÓN
------------------------------------------------------------

Edita:

transcribe_dobla_videos.sh

Configuración básica:

# Directorio con vídeos
export INPUT_DIRECTORY="./videos_a_transcribir"

# Modelo Whisper
export WHISPER_MODEL="medium"

# Idioma destino
export TARGET_LANGUAGE="es"

# Modo:
# dub | burn | soft | None
export SUBTITLE_MODE="dub"

------------------------------------------------------------
TRADUCCIÓN
------------------------------------------------------------

Google Translate:

export TRANSLATOR_BACKEND="google"

Ollama local:

export TRANSLATOR_BACKEND="ollama"
export OLLAMA_MODEL="qwen3:4b"

------------------------------------------------------------
TTS
------------------------------------------------------------

XTTS v2 (recomendado):

export TTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

Características:

- Multilenguaje
- Clonación de voz automática
- Máxima calidad
- Usa Demucs automáticamente
- ~1.8 GB de descarga en el primer arranque

VITS español:

export TTS_MODEL_NAME="tts_models/es/css10/vits"

Características:

- Más rápido
- Menos RAM (~300 MB)
- Sin clonación de voz

------------------------------------------------------------
DEMUCS
------------------------------------------------------------

export DEMUCS_MODEL="htdemucs"

Opciones:

- htdemucs      → Muy buena calidad (recomendado)
- htdemucs_ft   → Mejor calidad pero más lento

------------------------------------------------------------
VOLÚMENES
------------------------------------------------------------

export BACKGROUND_VOLUME="1.0"
export VOICE_VOLUME="1.7"

------------------------------------------------------------
USO
------------------------------------------------------------

1. Coloca vídeos aquí:

videos_a_transcribir/

2. Ejecuta:

chmod +x transcribe_dobla_videos.sh

./transcribe_dobla_videos.sh

------------------------------------------------------------
FLUJO COMPLETO
------------------------------------------------------------

VIDEO
  ↓
WHISPER
  ↓
SRT ORIGINAL
  ↓
TRADUCCIÓN
  ↓
SRT TRADUCIDO
  ↓
DEMUX + DEMUCS
  ↓
XTTS v2
  ↓
DOBLAJE FINAL

------------------------------------------------------------
ARCHIVOS GENERADOS
------------------------------------------------------------

Transcripción:
video.srt

Traducción:
video_es.srt

Doblaje:
video_es.mp4

Subtítulos quemados:
video_es_sub.mp4

Subtítulos suaves:
video_es_soft.mkv

------------------------------------------------------------
MODELOS WHISPER
------------------------------------------------------------

Modelo   | RAM/VRAM | Velocidad  | Calidad
------------------------------------------------
tiny     | ~1 GB    | Muy rápida | Baja
base     | ~1 GB    | Rápida     | Media
small    | ~2 GB    | Media      | Buena
medium   | ~5 GB    | Lenta      | Muy buena
large    | ~10 GB   | Muy lenta  | Excelente

------------------------------------------------------------
RENDIMIENTO RECOMENDADO
------------------------------------------------------------

CPU solamente (Mac Intel / Linux sin GPU):

export WHISPER_MODEL="base"
export TTS_MODEL_NAME="tts_models/es/css10/vits"

macOS Apple Silicon (MPS):
Whisper y Demucs usan MPS automáticamente.
Coqui TTS corre en CPU (no tiene soporte MPS completo).

export WHISPER_MODEL="medium"
export TTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

GPU NVIDIA (CUDA):
Todo el pipeline corre en GPU.

export WHISPER_MODEL="medium"
export TTS_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"

------------------------------------------------------------
SOLUCIÓN DE PROBLEMAS
------------------------------------------------------------

ERROR: No matching distribution found for TTS
o
RuntimeError: TTS requires python >= 3.9 and < 3.12

Estás instalando el paquete original abandonado.
Solución: usar el fork activo.

pip uninstall TTS
pip install coqui-tts

------------------------------------------------------------

ffmpeg no encontrado

macOS:

brew install ffmpeg

Linux:

sudo apt install ffmpeg

------------------------------------------------------------

CUDA out of memory

Usa modelos más pequeños:

export WHISPER_MODEL="base"

------------------------------------------------------------

TTS no usa GPU en Linux

Comprueba que CUDA está disponible:

python -c "import torch; print(torch.cuda.is_available())"

Si devuelve False, reinstala PyTorch con el índice CUDA correcto
(ver sección de instalación Linux).

------------------------------------------------------------

TTS corre en CPU en macOS Apple Silicon

Es el comportamiento esperado.
Coqui TTS no tiene soporte completo de MPS.
Whisper y Demucs sí usan el chip Apple Silicon (MPS).

------------------------------------------------------------

ImportError: TorchCodec is required / instala torchcodec

Tienes moviepy 2.2 o superior instalado. La versión 2.2 introdujo
torchcodec como backend de vídeo, pero solo tiene wheels precompilados
para Linux — en macOS hay que compilarlo desde fuentes.

Solución: bajar moviepy a la versión anterior al cambio.

pip install "moviepy>=2.0.0,<2.2.0"

Si usas el requirements incluido, ya viene con este límite aplicado.

------------------------------------------------------------

NOTAS IMPORTANTES
------------------------------------------------------------

Primer arranque lento:

La primera vez se descargan automáticamente:

- Whisper medium: ~1.5 GB
- XTTS v2: ~1.8 GB
- Demucs htdemucs: ~80 MB

Las siguientes ejecuciones usan la caché local.

XTTS v2 usa clonación de voz automática:

El script:

1. Extrae las voces del vídeo con Demucs
2. Usa esas voces como referencia para XTTS v2
3. Mantiene una voz consistente durante todo el doblaje

------------------------------------------------------------
LICENCIAS Y CRÉDITOS
------------------------------------------------------------

Whisper:
https://github.com/openai/whisper

Coqui TTS (fork activo):
https://github.com/idiap/coqui-ai-TTS

Demucs:
https://github.com/facebookresearch/demucs

MoviePy:
https://zulko.github.io/moviepy/
