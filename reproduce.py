# Rebuild the six clips from the X thread: Qwen3-TTS-12Hz-0.6B-Base voice clone, the 400 at x8 (neurons.json + directions.json).
#
#   pip install qwen-tts soundfile
#   python reproduce.py        -> clips/*.wav  (reference voices are in refs/)
import os, json, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, MAX_NEW_TOKENS = 8, 400
REFS = {  # reference clip in refs/: (start sample at 24 kHz in the YouTube video's audio, transcript)
    "rKpltaOMFdc": (7463280, "You know, as we take on these experiences, I think we need to know that we are never, ever the finished product. And I think that's what I was always chasing as a young individual who wanted to be a TV presenter. I want to be Jameela Jamil."),
    "l7UkhsylS34": (4175280, "Nancy Drew did not exaggerate. Nearly everyone in River Heights shared the opinion that the Tophams were snobbish and arrogant, and the treatment they accorded old Josiah Crowley had aroused a"),
    "U4mQQc2XUOk": (16920480, "falling in love is something beautiful. I think that being in a relationship is also something that's hard. Based on the way you phrased your question though, it seems like you might just not have anybody in your life at the moment who is a particularly spicy lover."),
}
CALM = "Close your eyes, and breathe in slowly. Let the whole day melt away, one quiet moment at a time."
STORY = "So I walked in, and the whole room just went quiet. I mean, nobody said a word. And then my friend starts laughing, and honestly, I lost it too."
ADVICE = "Look, you do not have to get it perfect the first time. Just try it, see what happens, and be a little kind to yourself."
CLIPS = [  # name, reference voice, line, seed
    ("laugh", "rKpltaOMFdc", CALM, 0),
    ("whisper", "l7UkhsylS34", CALM, 0),
    ("question", "U4mQQc2XUOk", CALM, 0),
    ("yeah", "rKpltaOMFdc", STORY, 1),
    ("second_voice", "rKpltaOMFdc", STORY, 2),
    ("reply", "rKpltaOMFdc", ADVICE, 1),
]

def reference(vid): return f"{HERE}/refs/{vid}.wav"

def apply_400(tts, alpha):
    """down_proj out += (alpha - 1) * (sum_k a_k g_k) * d_L over the 400 in each layer L, g_k = d_L . W_down[:, k]."""
    neurons = json.load(open(f"{HERE}/neurons.json"))["neurons"]
    dirs = json.load(open(f"{HERE}/directions.json"))["layers"]
    layers = tts.model.talker.model.layers
    for l, d in dirs.items():
        down = layers[int(l)].mlp.down_proj; W = down.weight.float(); d = torch.tensor(d, device=W.device)
        idx = torch.tensor([n["index"] for n in neurons if n["group"] == "400" and n["layer"] == int(l)], device=W.device)
        g = d @ W[:, idx]
        down.register_forward_hook(lambda m, i, o, idx=idx, g=g, d=d:
                                   (o.float() + (alpha - 1) * (i[0][..., idx].float() @ g).unsqueeze(-1) * d).to(o.dtype))

if __name__ == "__main__":
    os.makedirs(f"{HERE}/clips", exist_ok=True)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0", dtype=torch.bfloat16)
    apply_400(tts, ALPHA)
    for name, vid, text, seed in CLIPS:
        torch.manual_seed(seed)
        ws, sr = tts.generate_voice_clone(text=text, language="English", ref_audio=reference(vid), ref_text=REFS[vid][1], max_new_tokens=MAX_NEW_TOKENS)
        sf.write(f"{HERE}/clips/{name}.wav", ws[0], sr); print(name, round(len(ws[0]) / sr, 2), "s", flush=True)
