"""
Requisitos:
    pip install -U openai-whisper
    pip install moviepy tqdm
    pip install torch torchvision torchaudio
    pip install deep-translator
    pip install pydub
    pip install TTS               # Coqui TTS — descarga modelos la primera vez
    pip install demucs            # Separacion de voces/fondo (necesario para doblaje)
    # ffmpeg debe estar instalado en el sistema:
    #   Linux  : sudo apt install ffmpeg
    #   Mac    : brew install ffmpeg
    #   Windows: https://ffmpeg.org/download.html

Modelos Coqui TTS (TTS_MODEL_NAME en la config):
    Multilingue con clonacion de voz (recomendado):
        "tts_models/multilingual/multi-dataset/xtts_v2"   # ~1.8 GB, maxima calidad
        → Usa vocals.wav (extraido por Demucs) como referencia: voz siempre consistente
          y opcionalmente clona la voz del ponente original.
    Espanol especifico (mas rapido, ~300 MB, sin clonacion de voz):
        "tts_models/es/css10/vits"
    Lista completa: python -c "from TTS.api import TTS; TTS().list_models()"

═══════════════════════════════════════════════════════════════════════
FLUJO COMPLETO
═══════════════════════════════════════════════════════════════════════
1. Transcripcion  → video.mp4  ──► video.srt  (idioma original)
2. Traduccion     → video.srt  ──► video_es.srt
3. (Opcional) Una de estas tres salidas segun SUBTITLE_MODE:

   A) "dub"  DOBLAJE COMPLETO   : TTS por segmento con Demucs + XTTS v2
                                  Salida: video_es.mp4

   B) "burn" SUBTITULOS QUEMADOS: texto del SRT incrustado sobre la imagen (ffmpeg)
                                  Salida: video_es_sub.mp4

   C) "soft" PISTA SUAVE        : SRT como pista seleccionable en MKV (ffmpeg)
                                  Salida: video_es_soft.mkv
═══════════════════════════════════════════════════════════════════════
"""
import gc
import os
import re
import logging
import shutil
import subprocess
import sys
import time
import tempfile
import requests
from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

# Cache global para TTS y Ollama
TTS_MODEL = None
_TTS_MODEL_NAME_LOADED = None
OLLAMA_MODEL = "qwen3:4b"

# ── Constantes de ajuste de velocidad para doblaje ────────────────────
# Velocidad máxima que se pide al TTS antes de sonar distorsionada.
# Por encima de ~1.35 XTTS v2 puede producir artefactos de voz ("pito").
TTS_SPEED_MAX    = 1.30
# Velocidad máxima del post-proceso atempo (ffmpeg). Por encima de 1.35
# el time-stretching empieza a ser audible. Entre TTS_SPEED_MAX y
# ATEMPO_MAX el ajuste combinado cubre ratios de hasta ~1.30 × 1.35 ≈ 1.75.
ATEMPO_MAX       = 1.40
# Tasa de caracteres/segundo estimada para español con XTTS v2 a speed=1.0.
# Calibrado empíricamente; ajusta si tu material es muy diferente.
CHARS_PER_SEC_ES = 10.5

SPEED_SAFETY_MARGIN  = 1.04

# ── Whisper ───────────────────────────────────────────────────────────
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: 'whisper' no instalada. Ejecuta: pip install openai-whisper")
    WHISPER_AVAILABLE = False

# ── deep-translator ───────────────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: 'deep-translator' no instalada. Ejecuta: pip install deep-translator")
    TRANSLATOR_AVAILABLE = False

# ── pydub ─────────────────────────────────────────────────────────────
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: 'pydub' no instalada. Ejecuta: pip install pydub")
    PYDUB_AVAILABLE = False

# ── Coqui TTS ─────────────────────────────────────────────────────────
# Se usa `except Exception` en lugar de `except ImportError` porque en macOS
# Apple Silicon el import puede fallar con RuntimeError u otras excepciones
# al cargar dependencias internas de TTS. CoquiTTS se define siempre (None
# si falla) para evitar NameError en get_tts_model.
CoquiTTS = None
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except Exception as _tts_import_err:
    print(f"ADVERTENCIA: 'coqui-tts' no disponible ({_tts_import_err}). "
          "Ejecuta: pip install coqui-tts")
    COQUI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# 1. LOGGING
# ═══════════════════════════════════════════════════════════════════════

def setup_logging():
    """
    Configura el sistema de logging para escribir por stdout con nivel INFO.
    El formato incluye marca de tiempo, nivel de severidad y mensaje.
    Debe llamarse una sola vez al inicio del programa.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("--- Iniciando Sistema de Transcripción / Traducción / Doblaje ---")


# ═══════════════════════════════════════════════════════════════════════
# 2. HARDWARE
# ═══════════════════════════════════════════════════════════════════════

def get_compute_device() -> str:
    """
    Detecta el acelerador de hardware disponible para PyTorch/Whisper.
    Prueba en orden: GPU NVIDIA (CUDA) → Apple Silicon (MPS) → CPU.
    Devuelve la cadena de dispositivo que se pasa directamente a whisper.load_model().
    Si torch no está instalado, devuelve 'cpu' sin error.
    """
    logging.info("Detectando hardware...")
    try:
        import torch
        if torch.cuda.is_available():
            logging.info("GPU NVIDIA (CUDA) detectada.")
            return "cuda"
        if torch.backends.mps.is_available():
            # Whisper soporta MPS desde v20230918. En versiones anteriores o con
            # modelos grandes puede fallar; en ese caso process_batch captura el error.
            logging.info("Apple Silicon (MPS) detectado.")
            return "mps"
    except ImportError:
        pass
    logging.warning("Sin GPU. Usando CPU.")
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════
# 3. HELPERS DE TIEMPO SRT
# ═══════════════════════════════════════════════════════════════════════

def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convierte segundos (float) al formato de timestamp SRT: HH:MM:SS,mmm"""
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def srt_timestamp_to_seconds(ts: str) -> float:
    """Convierte un timestamp SRT (HH:MM:SS,mmm) a segundos en coma flotante."""
    ts = ts.strip().replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt_file(srt_path: Path) -> list[dict]:
    """
    Lee un fichero SRT y devuelve sus segmentos como lista de dicts.
    Cada dict contiene: {"index": int, "start": float, "end": float, "text": str}
    donde start y end son segundos en coma flotante.
    Los bloques con menos de 3 líneas o con timestamps malformados se omiten
    silenciosamente para no interrumpir el procesamiento del resto.
    """
    segments = []
    content = srt_path.read_text(encoding="utf-8").strip()
    for block in content.split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index    = int(lines[0].strip())
            ts_parts = lines[1].split(" --> ")
            start    = srt_timestamp_to_seconds(ts_parts[0])
            end      = srt_timestamp_to_seconds(ts_parts[1])
            text     = "\n".join(lines[2:]).strip()
            segments.append({"index": index, "start": start, "end": end, "text": text})
        except (ValueError, IndexError):
            continue
    return segments


# ═══════════════════════════════════════════════════════════════════════
# 4. TRANSCRIPCIÓN
# ═══════════════════════════════════════════════════════════════════════

def cleanup_torch():
    """
    Libera memoria Python + CUDA/MPS.
    Muy importante entre fases grandes:
    Whisper / Demucs / XTTS.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    except Exception as e:
        logging.debug(f"cleanup_torch(): {e}")


def transcribe_video(video_path: Path, model) -> dict | None:
    """
    Extrae el audio del vídeo como WAV mono a 16 kHz (formato nativo de Whisper)
    y lanza la transcripción. Devuelve el dict de resultados de Whisper, que incluye
    el texto completo en result["text"], los segmentos con timestamps en
    result["segments"] y el idioma detectado en result["language"].
    El fichero WAV temporal se elimina siempre en el bloque finally,
    haya error o no. Devuelve None si falla la extracción o la transcripción.
    """
    audio_path = video_path.with_suffix(".wav")
    try:
        logging.info("    -> Transcribiendo...")
        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            logging.error(f"    El vídeo {video_path.name} no tiene pista de audio.")
            clip.close()
            return None
        clip.audio.write_audiofile(
            str(audio_path),
            codec="pcm_s16le",
            fps=16000,
            logger=None
        )
        clip.close()
        result = model.transcribe(str(audio_path))
        logging.info("    -> Transcripcion completada.")
        return result
    except Exception as e:
        logging.error(f"    ERROR transcribiendo {video_path.name}: {e}")
        return None
    finally:
        if audio_path.exists():
            audio_path.unlink()



# ═══════════════════════════════════════════════════════════════════════
# DEMUCS — Separación de voces
# ═══════════════════════════════════════════════════════════════════════

def extract_audio_stems(
    video_path: Path,
    tmp_dir: Path,
    demucs_model: str = "htdemucs",
) -> tuple[Path | None, Path | None]:
    """
    Extrae el audio del vídeo a 44.1 kHz y lo procesa con Demucs en modo
    --two-stems=vocals, que genera exactamente dos ficheros:
        vocals.wav    — todo lo que Demucs identifica como voz humana
        no_vocals.wav — el resto: música, efectos de sala, ruido de fondo

    La calidad de la separación depende del modelo y del material; en vídeos
    con voz clara el resultado es muy limpio, pero en condiciones difíciles
    (reverb fuerte, música encima de la voz) puede haber artefactos en ambas pistas.

    vocals.wav se usa como audio de referencia en XTTS v2 para anclar el
    embedding de voz y garantizar consistencia entre segmentos. No es necesario
    que sea perfecto: XTTS v2 extrae el timbre general aunque haya algo de ruido.

    El parámetro demucs_model se pasa tanto al comando (--name) como a la
    construcción de la ruta de salida, asegurando que siempre coincidan.

    Devuelve (vocals_path, no_vocals_path). vocals_path puede ser None si
    Demucs no lo generó; no_vocals_path None indica fallo total de la separación.
    """
    logging.info("    -> Extrayendo audio original...")

    original_audio = tmp_dir / "original_audio.wav"

    try:
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(
            str(original_audio),
            codec="pcm_s16le",
            fps=44100,
            logger=None
        )
        video.close()
    except Exception as e:
        logging.error(f"    ERROR extrayendo audio: {e}")
        return None, None

    logging.info(f"    -> Separando voces con Demucs (modelo: {demucs_model})...")

    try:
        cmd = [
            "demucs",
            "--two-stems=vocals",
            f"--name={demucs_model}",   # explícito: evita depender del default de Demucs
            str(original_audio),
            "-o", str(tmp_dir),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"    ERROR ejecutando Demucs: {e.stderr.decode(errors='replace')}")
        return None, None
    except FileNotFoundError:
        logging.error("    Demucs no encontrado. Instala: pip install demucs")
        return None, None

    # La ruta de salida usa el mismo nombre de modelo que pasamos al comando
    stem_dir    = tmp_dir / demucs_model / original_audio.stem
    vocals_path    = stem_dir / "vocals.wav"
    no_vocals_path = stem_dir / "no_vocals.wav"

    if not no_vocals_path.exists():
        logging.error(f"    Demucs no genero no_vocals.wav en {stem_dir}")
        return None, None

    if not vocals_path.exists():
        logging.warning("    Demucs no genero vocals.wav; la clonacion de voz no estara disponible.")
        vocals_path = None

    logging.info("    -> Audio separado correctamente (vocals + no_vocals).")
    cleanup_torch()
    return vocals_path, no_vocals_path


# ═══════════════════════════════════════════════════════════════════════
# 5. TRADUCCION
# ═══════════════════════════════════════════════════════════════════════

def clean_llm_response(text: str) -> str:
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Nota:") or stripped.startswith("Reglas:") or stripped.startswith("Translation:"):
            continue
        if re.match(r"^\([^)]*\:", stripped) or stripped.startswith("(Con ") or stripped.startswith("(En "):
            continue
        clean_lines.append(stripped)
    result = " ".join(clean_lines).strip()
    return result if result else text.strip()


def translate_text(
    text: str,
    target_language: str,
    source_language: str = "auto",
) -> str | None:
    backend = os.environ.get("TRANSLATOR_BACKEND", "ollama").lower()

    if backend == "google":
        return _translate_google(text, target_language, source_language)
    else:
        return _translate_ollama(text, target_language, source_language)


def _translate_google(text: str, target: str, source: str = "auto") -> str | None:
    try:
        lang_map = {
            "es": "es", "en": "en", "fr": "fr",
            "de": "de", "it": "it", "pt": "pt",
        }
        src = lang_map.get(source, "auto")
        tgt = lang_map.get(target, target)

        result = GoogleTranslator(source=src, target=tgt).translate(text)
        if result:
            logging.debug(f"    Google translate: '{text[:50]}...' -> '{result[:50]}...'")
        return result
    except Exception as e:
        logging.error(f"    ERROR traduciendo con Google: {e}")
        return None


def _translate_ollama(text: str, target_language: str, source_language: str = "auto") -> str | None:
    try:
        lang_map = {
            "es": "español",
            "en": "inglés",
            "fr": "francés",
            "de": "alemán",
            "it": "italiano",
            "pt": "portugués",
        }

        target_lang_name = lang_map.get(target_language, target_language)

        prompt = f"Translate to {target_lang_name} for dubbing. Only return the translation, nothing else.\n\n{text}"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        return clean_llm_response(response.json()["response"])

    except Exception as e:
        logging.error(f"    ERROR traduciendo con Ollama: {e}")
        return None


def translate_srt(
    srt_content: str,
    target_language: str,
    source_language: str = "auto",
) -> str | None:
    """
    Traduce solo el texto del SRT usando translate_text()
    manteniendo timestamps intactos.
    """

    try:
        blocks = srt_content.strip().split("\n\n")
        translated_blocks = []

        for block in blocks:
            lines = block.strip().splitlines()

            if len(lines) < 3:
                translated_blocks.append(block)
                continue

            subtitle_id = lines[0]
            timestamp = lines[1]
            text = " ".join(lines[2:])

            translated = translate_text(
                text,
                target_language,
                source_language,
            )

            if not translated:
                translated = text

            translated_blocks.append(
                f"{subtitle_id}\n{timestamp}\n{translated}"
            )

        return "\n\n".join(translated_blocks)

    except Exception as e:
        logging.error(f"    ERROR traduciendo SRT: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# 6. GUARDAR TRANSCRIPCION + TRADUCCION
# ═══════════════════════════════════════════════════════════════════════

def save_results(video_path: Path, result: dict, target_language: str | None = None) -> bool:
    """
    Genera y guarda el fichero SRT a partir del dict de resultados de Whisper.

    Siempre escribe el fichero en el idioma original detectado:
        <stem>.srt  — subtítulos con timestamps construidos desde result["segments"]

    Si se proporciona target_language y el idioma detectado es distinto:
        <stem>_<lang>.srt  — SRT con solo las líneas de texto traducidas

    Devuelve False si falla la escritura del fichero original;
    los errores de traducción se registran pero no interrumpen la ejecución.
    """
    srt_blocks = []
    for i, seg in enumerate(result["segments"], start=1):
        start_ts = seconds_to_srt_timestamp(seg["start"])
        end_ts   = seconds_to_srt_timestamp(seg["end"])
        srt_blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{seg['text'].strip()}")

    srt_content   = "\n\n".join(srt_blocks)
    txt_content   = result["text"].strip()
    detected_lang = result.get("language", "auto")

    try:
        video_path.with_suffix(".srt").write_text(srt_content, encoding="utf-8")
        logging.info(f"    -> {video_path.stem}.srt guardado (idioma: {detected_lang})")
    except IOError as e:
        logging.error(f"    ERROR guardando ficheros originales: {e}")
        return False

    if target_language and TRANSLATOR_AVAILABLE:
        if detected_lang == target_language:
            logging.info(f"    -> Idioma detectado == destino ({target_language}). Sin traduccion.")
        else:
            logging.info(f"    -> Traduciendo a '{target_language}' (fuente: '{detected_lang}')...")
            backend = os.environ.get("TRANSLATOR_BACKEND", "ollama").lower()
            try:
                translated_srt = translate_srt(srt_content, target_language, detected_lang)
                if translated_srt:
                    video_path.with_name(f"{video_path.stem}_{target_language}.srt") \
                        .write_text(translated_srt, encoding="utf-8")
                    logging.info(f"    -> {video_path.stem}_{target_language}.srt guardado")
            finally:
                # Solo detener Ollama si se usó ese backend
                if backend == "ollama":
                    subprocess.run(["ollama", "stop", OLLAMA_MODEL], capture_output=True)

    return True


# ═══════════════════════════════════════════════════════════════════════
# 7. OPCION A — DOBLAJE COMPLETO (TTS + remontaje de video)
# ═══════════════════════════════════════════════════════════════════════

def get_tts_model(model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
    global TTS_MODEL, _TTS_MODEL_NAME_LOADED

    if not COQUI_AVAILABLE or CoquiTTS is None:
        raise RuntimeError(
            "Coqui TTS no está disponible. Instala el fork activo: pip install coqui-tts\n"
            "Si ya lo instalaste, revisa el ADVERTENCIA al inicio del log para ver "
            "qué excepción impidió cargarlo."
        )

    if TTS_MODEL is None or _TTS_MODEL_NAME_LOADED != model_name:
        logging.info(f"    -> Cargando modelo Coqui TTS: {model_name}")
        logging.info("       (primera vez descarga el modelo; las siguientes usan cache local)")

        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("    -> TTS usando GPU (CUDA)")
        elif torch.backends.mps.is_available():
            # Apple Silicon: Coqui TTS tiene soporte parcial de MPS;
            # se prueba MPS y si el modelo falla se cae a CPU automáticamente.
            device = "mps"
            logging.info("    -> TTS usando Apple Silicon (MPS)")
        else:
            device = "cpu"
            logging.info("    -> TTS usando CPU")

        TTS_MODEL = CoquiTTS(model_name)
        try:
            TTS_MODEL.to(device)
        except Exception as e:
            if device == "mps":
                logging.warning(f"    MPS no soportado por este modelo TTS ({e}). Usando CPU.")
                device = "cpu"
                TTS_MODEL.to(device)
            else:
                raise
        _TTS_MODEL_NAME_LOADED = model_name
        logging.info("    -> TTS cargado")

    return TTS_MODEL


def compute_tts_speed(text: str, slot_s: float) -> float:
    """
    Estima la velocidad TTS necesaria para que el audio generado quepa en slot_s.

    Usa CHARS_PER_SEC_ES como tasa de referencia calibrada para XTTS v2 en español
    a speed=1.0. El ratio resultante se acota en [1.0, TTS_SPEED_MAX] para no
    distorsionar la voz: por encima de TTS_SPEED_MAX el TTS empieza a sonar
    artificial. Si el texto es demasiado corto o el slot muy holgado, devuelve 1.0.

    El ajuste residual (cuando el TTS aun se pasa tras esta velocidad) lo gestiona
    adjust_audio_speed con atempo tras medir la duración real del WAV generado.
    """
    if slot_s <= 0 or not text.strip():
        return 1.0
    estimated_s = len(text.strip()) / CHARS_PER_SEC_ES
    ratio = (estimated_s / slot_s) * SPEED_SAFETY_MARGIN
    return max(1.0, min(TTS_SPEED_MAX, ratio))


def generate_segment_audio_coquitts(
    text: str,
    output_path: Path,
    language: str = "es",
    speaker_wav: str | None = None,
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    tts_speed: float = 1.0,
) -> bool:
    """
    Sintetiza un segmento de texto a WAV usando Coqui TTS y lo guarda en output_path.

    El parámetro tts_speed controla la cadencia de síntesis (1.0 = normal).
    Debe calcularse en assemble_dubbed_audio con compute_tts_speed() para que
    el audio generado se acerque al slot disponible sin necesitar post-proceso.
    Se acota a TTS_SPEED_MAX en esta función como salvaguarda adicional.

    Con XTTS v2 (modelo multilingüe) y speaker_wav:
        Pasa el audio de referencia como embedding de voz. Todos los segmentos
        sintetizados con el mismo speaker_wav suenan con la misma voz.

    Con modelos monolingüales (p.ej. css10/vits):
        speaker_wav y tts_speed se ignoran (la API no los soporta).

    Devuelve True solo si output_path existe y tiene tamaño mayor que cero.
    """
    try:
        tts   = get_tts_model(tts_model_name)
        speed = max(1.0, min(TTS_SPEED_MAX, tts_speed))

        if tts.is_multi_lingual:
            kwargs = dict(
                text=text,
                language=language,
                temperature=0.6,
                speed=speed,
                file_path=str(output_path),
            )
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            else:
                logging.warning(
                    "    XTTS v2 sin speaker_wav: la voz puede variar entre segmentos."
                )
            tts.tts_to_file(**kwargs)
        else:
            # Modelo monolingue — speed y language no aplican
            tts.tts_to_file(text=text, file_path=str(output_path))

        return output_path.exists() and output_path.stat().st_size > 0

    except Exception as e:
        logging.error(f"    ERROR en Coqui TTS: {e}")
        return False


def _build_atempo_chain(speed: float) -> str:
    """
    Construye la cadena de filtros atempo para ffmpeg.

    ffmpeg solo acepta atempo en el rango [0.5, 2.0] por aplicación.
    Para velocidades > 2.0 hay que encadenar varios filtros. En la práctica,
    con ATEMPO_MAX ≤ 1.35 nunca llegamos aquí, pero se incluye por robustez.
    """
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def adjust_audio_speed(audio_seg: "AudioSegment", target_ms: int) -> "AudioSegment":
    """
    Acelera audio_seg para que se aproxime a target_ms usando el filtro
    atempo de ffmpeg (time-stretching sin cambio de tono, sin efecto «pito»).

    La aceleración máxima aplicada es ATEMPO_MAX (por defecto 1.35×). Si el
    audio sigue siendo más largo tras ese límite, el exceso se gestiona en
    assemble_dubbed_audio con un truncado suave (fade-out), nunca aquí.

    NOTA: el bug original usaba atempo=1/speed (que RALENTIZABA el audio en
    lugar de acelerarlo). La corrección es atempo=speed.
    """
    current_ms = len(audio_seg)

    if current_ms == 0 or target_ms == 0:
        return audio_seg

    speed = current_ms / target_ms          # > 1 → hay que acelerar

    # Si el audio ya cabe o la diferencia es despreciable, no tocar
    if speed <= 1.02:
        return audio_seg

    # Acotar para no distorsionar la voz
    speed = min(speed, ATEMPO_MAX)

    input_path  = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        audio_seg.export(input_path, format="wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", _build_atempo_chain(speed),   # ✅ acelera (antes era 1/speed → ralentizaba)
            "-ar", "44100",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        result = AudioSegment.from_file(output_path)
        logging.debug(
            f"    atempo {speed:.3f}× : {current_ms}ms → {len(result)}ms "
            f"(objetivo {target_ms}ms)"
        )
        return result

    except Exception as e:
        logging.warning(f"    adjust_audio_speed fallido ({e}); se devuelve audio original.")
        return audio_seg

    finally:
        for p in [input_path, output_path]:
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def assemble_dubbed_audio(
    segments: list[dict],
    total_duration_s: float,
    output_path: Path,
    language: str = "es",
    speaker_wav: str | None = None,
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    tmp_dir: Path = Path("/tmp/dub_tmp"),
) -> bool:
    """
    Construye una pista de audio doblada completa a partir de los segmentos del SRT.

    Estrategia de ajuste de velocidad en tres capas (sin efecto «pito»):

      Capa 1 — TTS speed dinámico (mejor calidad):
        compute_tts_speed() estima cuánto texto hay que decir en el slot y
        le pide al TTS que sintetice a esa velocidad (máx. TTS_SPEED_MAX=1.30).
        La síntesis a velocidad ligeramente elevada es transparente al oído.

      Capa 2 — atempo residual (post-proceso):
        Si tras la síntesis el WAV real sigue siendo más largo que el slot,
        adjust_audio_speed aplica time-stretching con atempo (máx. ATEMPO_MAX=1.35).
        atempo conserva el tono; solo empieza a notarse por encima de ~1.4×.
        La combinación de ambas capas cubre ratios de hasta ~1.30×1.35 ≈ 1.75×.

      Capa 3 — truncado con fade-out (último recurso):
        Si el audio sigue desbordando el slot (texto muy largo, silencio entre
        segmentos casi nulo), se recorta con un fade-out de 120 ms para evitar
        el corte brusco. Se registra un WARNING para que puedas ajustar la
        traducción o los parámetros.

    Crea una pista de silencio de total_duration_s segundos y superpone cada
    segmento en su posición de inicio (start_ms). Los WAV temporales se eliminan
    en sus propios bloques finally. Al finalizar exporta la pista a output_path.
    Devuelve False si pydub no está disponible o si falla la exportación final.
    """
    if not PYDUB_AVAILABLE:
        logging.error("    pydub no disponible. Ejecuta: pip install pydub")
        return False

    tmp_dir.mkdir(parents=True, exist_ok=True)
    total_ms    = int(total_duration_s * 1000)
    final_audio = AudioSegment.silent(duration=total_ms)
    ok_count    = 0
    truncated   = 0

    for seg in segments:
        if not seg["text"].strip():
            continue

        start_ms       = int(seg["start"] * 1000)
        end_ms         = int(seg["end"]   * 1000)
        slot_ms        = end_ms - start_ms
        seg_audio_path = tmp_dir / f"seg_{seg['index']:04d}.wav"

        # ── Capa 1: velocidad dinámica en el TTS ──────────────────────
        slot_s    = slot_ms / 1000.0
        tts_speed = compute_tts_speed(seg["text"], slot_s)
        if tts_speed > 1.01:
            logging.debug(
                f"    Seg {seg['index']}: slot={slot_s:.2f}s  "
                f"tts_speed={tts_speed:.2f}×"
            )

        ok = generate_segment_audio_coquitts(
            text=seg["text"],
            output_path=seg_audio_path,
            language=language,
            speaker_wav=speaker_wav,
            tts_model_name=tts_model_name,
            tts_speed=tts_speed,
        )
        if not ok:
            logging.warning(f"    Segmento {seg['index']} sin audio; se inserta silencio.")
            continue

        try:
            seg_audio   = AudioSegment.from_file(str(seg_audio_path))
            actual_ms   = len(seg_audio)

            # ── Capa 2: atempo residual si aún se pasa ─────────────────
            if actual_ms > slot_ms:
                seg_audio = adjust_audio_speed(seg_audio, slot_ms)

            # ── Capa 3: truncado con fade-out como último recurso ───────
            if len(seg_audio) > slot_ms:
                fade_ms   = min(120, slot_ms // 4)
                seg_audio = seg_audio[:slot_ms].fade_out(fade_ms)
                truncated += 1
                logging.warning(
                    f"    Seg {seg['index']}: truncado con fade "
                    f"({actual_ms}ms → {slot_ms}ms). "
                    f"Considera alargar el slot o acortar la traducción."
                )

            final_audio = final_audio.overlay(seg_audio, position=start_ms)
            ok_count   += 1

        except Exception as e:
            logging.warning(f"    Error ensamblando segmento {seg['index']}: {e}")
        finally:
            if seg_audio_path.exists():
                seg_audio_path.unlink()

    logging.info(
        f"    -> {ok_count}/{len(segments)} segmentos sintetizados correctamente"
        + (f" ({truncated} truncados)" if truncated else "") + "."
    )

    try:
        final_audio.export(str(output_path), format="wav")
        logging.info(f"    -> Audio doblado exportado: {output_path.name}")
        return True
    except Exception as e:
        logging.error(f"    ERROR exportando audio final: {e}")
        return False


def merge_short_segments(segments, min_duration=3.0):
    merged = []
    buffer = None

    for seg in segments:
        if buffer is None:
            buffer = seg.copy()
            continue

        buffer_duration = buffer["end"] - buffer["start"]

        if buffer_duration < min_duration:
            buffer["text"] += " " + seg["text"]
            buffer["end"] = seg["end"]
        else:
            merged.append(buffer)
            buffer = seg.copy()

    if buffer:
        merged.append(buffer)

    return merged


def dub_video(
    video_path: Path,
    target_language: str,
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    demucs_model: str = "htdemucs",
    background_volume: float = 1.0,
    voice_volume: float = 1.7,
) -> bool:
    """
    Orquesta el doblaje completo de un vídeo al idioma destino.

    Pasos que ejecuta en orden:
      1. Comprueba que existe el SRT traducido (<stem>_<lang>.srt).
         Si el vídeo de salida ya existe, lo salta sin reprocesar (checkpoint).
      2. Llama a extract_audio_stems para separar vocals.wav y no_vocals.wav
         mediante Demucs. no_vocals.wav es el fondo musical; vocals.wav se
         usa como referencia de voz en XTTS v2.
      3. Llama a assemble_dubbed_audio para sintetizar cada segmento del SRT
         y ensamblarlos en una pista WAV alineada con los timestamps originales.
      4. Abre el vídeo original, aplica los factores de volumen a fondo y voces
         por separado, los mezcla con CompositeAudioClip y escribe el vídeo
         final con libx264 + AAC y la flag faststart (apto para streaming).
      5. Elimina todo el directorio temporal (_dub_tmp) en el bloque finally,
         haya error o no, para no dejar ficheros intermedios en disco.

    Salida: <stem>_<lang>.mp4 en el mismo directorio que el vídeo original.
    Devuelve True si el vídeo se generó correctamente, False en caso contrario.
    """
    srt_path = video_path.with_name(f"{video_path.stem}_{target_language}.srt")
    if not srt_path.exists():
        logging.error(f"    SRT traducido no encontrado: {srt_path.name}")
        return False

    output_path = video_path.with_name(f"{video_path.stem}_{target_language}.mp4")

    # ── Checkpoint ──────────────────────────────────────────────────────
    if output_path.exists():
        logging.warning(f"    [SALTO] Video doblado ya existe: {output_path.name}")
        return True

    logging.info(f"\n    ── OPCION A: Doblaje con Coqui TTS ({target_language}) ──")

    tmp_dir = video_path.parent / "_dub_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Duracion del video ───────────────────────────────────────
        clip           = VideoFileClip(str(video_path))
        total_duration = clip.duration
        clip.close()

        # ── Separar voces con Demucs ────────────────────────────────
        vocals_path, no_vocals_path = extract_audio_stems(video_path, tmp_dir, demucs_model)
        if no_vocals_path is None:
            logging.error("    No se pudo generar el audio de fondo.")
            return False

        # vocals_path se usa como referencia de voz para XTTS v2.
        # Si Demucs no lo generó, el TTS aun funciona pero sin consistencia garantizada.
        speaker_wav = str(vocals_path) if vocals_path else None
        if speaker_wav:
            logging.info("    -> vocals.wav disponible: se usara como referencia de voz (XTTS v2).")

        # ── Segmentos SRT ───────────────────────────────────────────
        segments = parse_srt_file(srt_path)
        segments = merge_short_segments(segments)
        if not segments:
            logging.error("    El SRT no contiene segmentos validos.")
            return False

        logging.info(f"    -> Sintetizando {len(segments)} segmentos con Coqui TTS...")

        # ── Generar audio doblado ───────────────────────────────────
        dubbed_audio_path = tmp_dir / f"{video_path.stem}_{target_language}_audio.wav"
        if not assemble_dubbed_audio(
            segments=segments,
            total_duration_s=total_duration,
            output_path=dubbed_audio_path,
            language=target_language,
            speaker_wav=speaker_wav,
            tts_model_name=tts_model_name,
            tmp_dir=tmp_dir,
        ):
            return False

        # ── Remontar video ──────────────────────────────────────────
        logging.info(f"    -> Remontando video: {output_path.name}")

        video            = VideoFileClip(str(video_path))
        background_audio = AudioFileClip(str(no_vocals_path))
        voice_audio      = AudioFileClip(str(dubbed_audio_path))

        # Limitar duraciones al video
        if voice_audio.duration > video.duration:
            voice_audio = voice_audio.subclipped(0, video.duration)
        if background_audio.duration > video.duration:
            background_audio = background_audio.subclipped(0, video.duration)

        # Ajuste de volumenes (configurables desde __main__)
        background_audio = background_audio.with_volume_scaled(background_volume)
        voice_audio      = voice_audio.with_volume_scaled(voice_volume)

        mixed_audio = CompositeAudioClip([background_audio, voice_audio])
        final       = video.with_audio(mixed_audio)

        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset="fast",          # un solo -preset (quitado el duplicado de ffmpeg_params)
            threads=os.cpu_count(),
            ffmpeg_params=["-movflags", "+faststart", "-crf", "23"],
            logger=None,
        )
        logging.info(f"    -> Video doblado guardado: {output_path.name}")

        for clip_obj in (video, background_audio, voice_audio, final):
            try:
                clip_obj.close()
            except Exception as e:
                logging.debug(f"    [close] {e}")

        global TTS_MODEL
        TTS_MODEL = None
        cleanup_torch()
        return True

    except Exception as e:
        logging.error(f"    ERROR en dub_video: {e}")
        return False

    finally:
        cleanup_torch()
        # Limpieza completa del directorio temporal (siempre, haya error o no)
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
                logging.info("    -> Directorio temporal eliminado.")
            except Exception as e:
                logging.warning(f"    No se pudo eliminar tmp_dir: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 8. OPCION B — SUBTITULOS QUEMADOS EN IMAGEN (burn-in con ffmpeg)
# ═══════════════════════════════════════════════════════════════════════

def burn_subtitles(video_path: Path, target_language: str) -> bool:
    """
    Renderiza el texto del SRT traducido directamente sobre los fotogramas del vídeo
    usando el filtro 'subtitles' de ffmpeg (tamaño de fuente 22, color blanco).
    El audio original se copia sin recodificar. Los subtítulos quedan incrustados
    permanentemente en la imagen y no se pueden desactivar desde el reproductor.
    Requiere que ffmpeg esté instalado en el sistema.
    Salida: <stem>_<lang>_sub<ext> en el mismo directorio que el vídeo original.
    """
    srt_path = video_path.with_name(f"{video_path.stem}_{target_language}.srt")
    if not srt_path.exists():
        logging.error(f"    SRT traducido no encontrado: {srt_path.name}")
        return False

    output_path  = video_path.with_name(
        f"{video_path.stem}_{target_language}_sub{video_path.suffix}"
    )
    # En Windows hay que escapar los dos puntos de la letra de unidad (C:\...).
    # En macOS/Linux las rutas absolutas empiezan por / y no necesitan ese escape.
    if sys.platform == "win32":
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    else:
        # En macOS/Linux solo escapar espacios y caracteres especiales del shell
        srt_escaped = str(srt_path).replace("'", "\\'").replace(" ", "\\ ")

    logging.info(f"\n    ── OPCION B: Subtitulos quemados -> {output_path.name} ──")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_escaped}':force_style='FontSize=22,PrimaryColour=&HFFFFFF&'",
        "-c:a", "copy",   # audio original sin recodificar
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"    -> Video con subtitulos quemados: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"    ffmpeg fallo: {e.stderr.decode(errors='replace')}")
        return False
    except FileNotFoundError:
        logging.error("    ffmpeg no encontrado. Instálalo: https://ffmpeg.org/download.html")
        return False


# ═══════════════════════════════════════════════════════════════════════
# 9. OPCION C — PISTA SUAVE DE SUBTITULOS (soft subtitles en MKV)
# ═══════════════════════════════════════════════════════════════════════

def add_soft_subtitles(video_path: Path, target_language: str) -> bool:
    """
    Empaqueta el SRT traducido como una pista de subtítulos seleccionable dentro
    del contenedor MKV, sin recodificar ni el vídeo ni el audio (copia directa).
    El espectador puede activar o desactivar los subtítulos desde el reproductor
    (VLC, MPV, etc.). La pista se etiqueta con el código de idioma y un título
    descriptivo en los metadatos del contenedor.
    Requiere que ffmpeg esté instalado en el sistema.
    Salida: <stem>_<lang>_soft.mkv en el mismo directorio que el vídeo original.
    """
    srt_path = video_path.with_name(f"{video_path.stem}_{target_language}.srt")
    if not srt_path.exists():
        logging.error(f"    SRT traducido no encontrado: {srt_path.name}")
        return False

    output_path = video_path.with_name(f"{video_path.stem}_{target_language}_soft.mkv")

    logging.info(f"\n    ── OPCION C: Pista suave -> {output_path.name} ──")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(srt_path),
        "-c",   "copy",     # copia video y audio sin recodificar
        "-c:s", "srt",
        "-metadata:s:s:0", f"language={target_language}",
        "-metadata:s:s:0", f"title=Subtitulos ({target_language})",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"    -> Video con pista suave: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"    ffmpeg fallo: {e.stderr.decode(errors='replace')}")
        return False
    except FileNotFoundError:
        logging.error("    ffmpeg no encontrado. Instala: https://ffmpeg.org/download.html")
        return False


# ═══════════════════════════════════════════════════════════════════════
# 10. PROCESAMIENTO POR LOTES
# ═══════════════════════════════════════════════════════════════════════

def process_batch(
    input_dir: str,
    model_name: str = "base",
    target_language: str | None = "es",
    subtitle_mode: str | None = "soft",
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    demucs_model: str = "htdemucs",
    background_volume: float = 1.0,
    voice_volume: float = 1.7,
):
    """
    Punto de entrada del procesamiento por lotes. Escanea input_dir en busca
    de ficheros de vídeo (.mp4, .mkv, .mov, .avi, .webm), carga el modelo
    Whisper una sola vez y procesa cada vídeo en orden.

    Por cada vídeo:
      - Transcripción: si ya existen <stem>.txt y <stem>.srt los salta (checkpoint).
        En caso contrario transcribe con Whisper y guarda los ficheros originales
        y, si target_language es distinto al idioma detectado, también los traducidos.
      - Subtítulos/doblaje: según subtitle_mode aplica una de estas acciones:
            "dub"  → dub_video():          doblaje con Coqui TTS + Demucs
            "burn" → burn_subtitles():     subtítulos quemados en imagen (ffmpeg)
            "soft" → add_soft_subtitles(): pista suave seleccionable en MKV (ffmpeg)
            None   → no hace nada con el vídeo, solo genera txt/srt

    Los parámetros tts_model_name, demucs_model, background_volume y voice_volume
    solo tienen efecto cuando subtitle_mode es "dub".
    Al terminar imprime un resumen con el conteo de vídeos procesados, saltados y con error.
    """
    if not WHISPER_AVAILABLE:
        sys.exit(1)

    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"'{input_dir}' no existe o no es un directorio.")
        return

    device = get_compute_device()

    video_extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
    video_files = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    )

    if not video_files:
        logging.warning(f"No se encontraron videos en '{input_dir}'.")
        return

    logging.info(f"\n{'='*64}")
    logging.info(f"Videos encontrados : {len(video_files)}")
    logging.info(f"Idioma destino     : {target_language or '(sin traduccion)'}")
    logging.info(f"Modo de salida     : {subtitle_mode or '(solo txt/srt)'}")
    if subtitle_mode == "dub":
        logging.info(f"Modelo TTS         : {tts_model_name}")
        logging.info(f"Modelo Demucs      : {demucs_model}")
        logging.info(f"Volumen fondo/voz  : {background_volume} / {voice_volume}")
    logging.info(f"\n{'='*64}\n")

    processed = skipped = errors = 0

    logging.info("Cargando modelo Whisper...")
    try:
        whisper_model = whisper.load_model(model_name, device=device)
    except Exception as e:
        if device == "mps":
            logging.warning(f"Whisper no pudo cargarse en MPS ({e}). Reintentando en CPU...")
            device = "cpu"
            whisper_model = whisper.load_model(model_name, device=device)
        else:
            raise

    for i, video_path in enumerate(video_files, start=1):
        logging.info(f"\n{'='*50}")
        logging.info(f"[{i}/{len(video_files)}] {video_path.name}")

        # ── Transcripcion con checkpoint ───────────────────────────
        txt_path = video_path.with_suffix(".txt")
        srt_path = video_path.with_suffix(".srt")

        if txt_path.exists() and srt_path.exists():
            logging.warning("    [SALTO] Transcripcion ya existente.")
            skipped += 1
        else:
            t0     = time.time()
            result = transcribe_video(video_path, whisper_model)
            if not result:
                errors += 1
                continue
            if not save_results(video_path, result, target_language):
                errors += 1
                continue
            processed += 1
            logging.info(f"    -> {time.time() - t0:.1f}s")

        # ── Subtitulos / Doblaje ────────────────────────────────────
        if subtitle_mode and target_language:
            if subtitle_mode == "dub":
                dub_video(
                    video_path,
                    target_language,
                    tts_model_name=tts_model_name,
                    demucs_model=demucs_model,
                    background_volume=background_volume,
                    voice_volume=voice_volume,
                )
            elif subtitle_mode == "burn":
                burn_subtitles(video_path, target_language)
            elif subtitle_mode == "soft":
                add_soft_subtitles(video_path, target_language)
            else:
                logging.warning(
                    f"    subtitle_mode '{subtitle_mode}' no reconocido. "
                    f"Valores validos: 'dub', 'burn', 'soft', None."
                )

    del whisper_model
    cleanup_torch()

    logging.info(f"\n{'='*64}")
    logging.info("LOTE FINALIZADO")
    logging.info(f"   Procesados : {processed}")
    logging.info(f"   Saltados   : {skipped}")
    logging.info(f"   Errores    : {errors}")
    logging.info(f"{'='*64}")


# ═══════════════════════════════════════════════════════════════════════
# 11. PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_logging()

    INPUT_DIRECTORY   = os.environ.get("INPUT_DIRECTORY",   "./videos_a_transcribir")
    WHISPER_MODEL     = os.environ.get("WHISPER_MODEL",     "medium")
    TARGET_LANGUAGE   = os.environ.get("TARGET_LANGUAGE",   "es")
    SUBTITLE_MODE     = os.environ.get("SUBTITLE_MODE",     "dub")
    TTS_MODEL_NAME    = os.environ.get("TTS_MODEL_NAME",    "tts_models/multilingual/multi-dataset/xtts_v2")
    DEMUCS_MODEL      = os.environ.get("DEMUCS_MODEL",      "htdemucs")
    BACKGROUND_VOLUME = float(os.environ.get("BACKGROUND_VOLUME", "1.0"))
    VOICE_VOLUME      = float(os.environ.get("VOICE_VOLUME",      "1.7"))
    OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "qwen3:4b")

    if not os.path.exists(INPUT_DIRECTORY):
        os.makedirs(INPUT_DIRECTORY)
        logging.info(
            f"Directorio '{INPUT_DIRECTORY}' creado. "
            f"Coloca tus videos ahi y vuelve a ejecutar."
        )
    else:
        process_batch(
            input_dir=INPUT_DIRECTORY,
            model_name=WHISPER_MODEL,
            target_language=TARGET_LANGUAGE,
            subtitle_mode=SUBTITLE_MODE,
            tts_model_name=TTS_MODEL_NAME,
            demucs_model=DEMUCS_MODEL,
            background_volume=BACKGROUND_VOLUME,
            voice_volume=VOICE_VOLUME,
        )
