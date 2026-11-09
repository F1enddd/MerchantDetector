
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import io
import soundfile as sf

from paths import BASE_DIR

from recognizer import recognize_audio_data, recognize_audio
from database import get_items_by_sound


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/images",
    StaticFiles(directory=BASE_DIR / "database" / "Images"),
    name="images"
)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "Frontend" / "index.html")

@app.get("/logo.svg")
def logo():
    return FileResponse(
        BASE_DIR / "Frontend" / "logo.svg",
        media_type="image/svg+xml"
    )

@app.get("/favicon.svg")
def favicon():
    return FileResponse(
        BASE_DIR / "Frontend" / "favicon.svg",
        media_type="image/svg+xml"
    )

@app.get("/recognize")
def recognize():
    recording = BASE_DIR / "test_audio.ogg"

    results = recognize_audio(recording)

    if not results:
        return {"recognized": False}

    best = results[0]

    if best["similarity"] < 0.55:
        return {
            "recognized": False,
            "candidates": results[:3]
        }

    items = get_items_by_sound(best["sound_id"])

    return {
        "recognized": True,
        "sound_id": best["sound_id"],
        "similarity": best["similarity"],
        "items": items
    }


@app.post("/recognize-live")
def recognize_live(file: UploadFile = File(...)):
    try:
        audio_data = file.file.read()

        recording, sample_rate = sf.read(
            io.BytesIO(audio_data),
            dtype="float32"
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать аудио: {error}"
        )

    results = recognize_audio_data(
        recording,
        sample_rate
    )

    if not results:
        return {"recognized": False}

    best = results[0]

    second_score = (
        results[1]["similarity"]
        if len(results) > 1
        else 0.0
    )

    difference = best["similarity"] - second_score

    # Не принимаем сомнительные совпадения.
    if best["similarity"] < 0.55 or difference < 0.08:
        return {
            "recognized": False,
            "candidates": results[:3]
        }

    items = get_items_by_sound(best["sound_id"])

    return {
        "recognized": True,
        "sound_id": best["sound_id"],
        "similarity": best["similarity"],
        "items": items
    }
