from potassium import Potassium, Request, Response
from transformers.utils.import_utils import subprocess

from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_write

from transformers import pipeline
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = MusicGen.get_pretrained('facebook/musicgen-melody')

    model.set_generation_params(duration=8)  # generate 8 seconds.

    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    descriptions = [prompt]

    wav = model.generate(descriptions)[0].cpu().numpy()

    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "f32le",
        "-ar", str(model.sample_rate),
        "-ac", "1",
        "-i", "-",
        "-ar", "44100",
        "-ac", "2",
        "-f", "wav",
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    wav = wav.astype("float32").tobytes()
    stdout, stderr = process.communicate(wav)

    return Response(
        body=stdout,
        headers={"Content-Type": "audio/wav"},
        status=200
    )


if __name__ == "__main__":
    app.serve()
