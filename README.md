# Lipflow

**Wispr Flow for your lips.** Hold a key, silently mouth what you want to say, let go, and the
text shows up at your cursor in whatever app you're in. No microphone and no sound, just your webcam.

Everything runs locally on your Mac. An optional LLM pass fixes the words lip reading gets wrong.

```
 hold ⌥ (right)  ──►  webcam  ──►  face landmarks (live)  ──►  mouth crops, 25 fps
                                                                   │
   paste at cursor  ◄──  LLM cleanup  ◄──  beam search + LM  ◄──  VSR encoder (Apple GPU)
                                       ▲
                     live preview: greedy CTC every 0.45 s while you talk
```

## Russian (this fork)

This fork dictates in **Russian** by default. The English Auto-AVSR model only knows English, so
Russian uses Meta's AV-HuBERT fine-tuned on MuAViC-ru (about 49 hours of Russian TED talks),
re-implemented in plain PyTorch in `lipflow/ru.py` and loaded from
[nguyenvulebinh/AV-HuBERT-MuAViC-ru](https://huggingface.co/nguyenvulebinh/AV-HuBERT-MuAViC-ru)
(1.5 GB, CC-BY-NC 4.0, downloaded by `setup.sh`). It uses the same mouth crops as the English model,
so capture, the HUD, onboarding and face training work the same way.

Set expectations honestly: Russian lip reading alone is rough. On a filmed Russian address, lips
alone recover fragments ("…планы на будущее, конечно…"), while lips + audio read the sentence almost
word for word. So for Russian:

- Turn on **Settings → Whisper mode** and whisper softly while you mouth. The same model reads lips
  + audio; nothing extra is downloaded. When the mic hears only room noise, Lipflow falls back to lips.
- Do the practice round (24 Russian sentences, more rounds help). Face training on the Russian model
  takes about 3 minutes on an M2 Pro.
- Sentence repair goes to ChatGPT through the Lunori plugin's signed-in account (see below).
  Without it, the on-device fallback for Russian is Qwen3-1.7B (`LIPFLOW_LOCAL_MODEL` overrides
  it), which may only make small, lip-lookalike spelling fixes.

Switch language with `LIPFLOW_LANG=en` (in the `env` file or your shell), `"language": "en"` in
`~/Library/Application Support/Lipflow/settings.json`, or `uv run lipflow --lang en`. Test on a file:
`uv run lipflow file talk.mp4 --lang ru --start 60 --end 68 [--audio]`.

## Setup

Needs an Apple Silicon Mac on macOS 13 or later (macOS 26 for the Liquid Glass look), and about 2 GB of disk.

```sh
git clone https://github.com/SergeiNikolenko/lipflow.git ~/code/lipflow
cd ~/code/lipflow && ./setup.sh     # installs uv deps, ~1.2 GB of models, builds /Applications/Lipflow.app
open /Applications/Lipflow.app
```

The first launch opens setup, which takes about 8 minutes:

1. **Permissions:** Camera, Input Monitoring and Accessibility, each asked once for "Lipflow".
2. **Your words:** if you use Wispr Flow, it imports your dictation history (read locally, never
   uploaded) to learn your phrasing and the names you say.
3. **Practice:** you silently mouth 24 sentences (taken from your own history when available).
4. **Train:** it fine-tunes the language model on your phrasing and the lip reader on your face,
   on this Mac's GPU. Six practice sentences are held out, and the face model is only kept if it
   reads them better than the stock model. You see the before/after score.

Re-run it any time from the menu → *Set up / train on my face…*. More practice clips help more.
Everything personal (clips, phrases, trained models) lives in
`~/Library/Application Support/Lipflow/`. Delete that folder to start over.

To start Lipflow at login: System Settings → General → Login Items → add Lipflow. Logs are in
`~/Library/Logs/Lipflow.log`. Running from a terminal (`uv run lipflow`) also works, but then
macOS asks for permissions in the terminal's name.

### Most accurate: whisper mode

Settings → **Whisper mode**. While you hold the key, Lipflow also listens to a soft whisper and
reads lips + audio together with the Auto-AVSR audio-visual model (downloaded the first time,
1.8 GB). On test sentences, lips alone got 31.9% of words wrong; lips + audio got 6.9%. A real
whisper is less clear than those test clips, so expect somewhere in between. The mic is only on
while you hold the key.

### Learning from your corrections

When you fix a word Lipflow typed (within about 30 seconds), it saves that clip with your
corrected sentence, and the next *Practice & train more* round uses it. Only the text field it
pasted into is read. Switch it off in Settings.

### Better accuracy: turn on LLM cleanup

Lip reading can't tell apart words that look the same on the lips (p/b/m, f/v, t/d/n…), so the raw
model output reads like "WALLET OFFICER" when you said "while in office". Lipflow sends the model's
top-3 guesses plus your last few dictations to an LLM, which picks the sentence you meant and
fixes casing, punctuation and numbers. The first backend that's available is used:

1. **ChatGPT**: used when the Lunori Codex plugin is installed and signed in, plus `node` on the
   PATH. `lipflow/chatgpt.mjs` borrows Lunori's account client and calls the Responses API with your
   ChatGPT plan at low effort, so there is no API key to manage (`LIPFLOW_CHATGPT_MODEL` picks
   another model). Best at fixing badly mis-read sentences, a few seconds per sentence.
2. **Local** (the fallback): Qwen3-0.6B 4-bit running in-process on Apple Silicon
   via MLX. About 350 MB, downloaded on first launch, and about 0.2 s per sentence, fully offline.
   Tiny models copy the formatting they're shown, so this one gets lowercase guesses and a few
   worked examples (`SMALL_SHOTS` in `cleanup.py`). Override with `LIPFLOW_LOCAL_MODEL`.
3. **Offline rules**: sentence case, "I", end punctuation, "nineteen forty three" → 1943.

**Learn from your Wispr Flow history.** Most of what you'll mouth is stuff you already say.

```sh
uv run lipflow import-wispr     # or: --from-text my-writing.txt
```

This reads Wispr Flow's local database read-only, prints only counts, and saves your phrases to
`~/Library/Application Support/Lipflow/phrases.txt`. Nothing leaves your Mac. Lipflow then:
picks between the lip-reader's top 5 guesses using a small model of the word pairs you use;
shows the cleanup model your past sentences closest to what it read; and adds names you
capitalise often to your custom words for review. Restart Lipflow afterwards.

**Custom words.** Names are the hardest thing to lip-read (a name is just lip shapes). Put yours
in the Lipflow menu → *Edit custom words*, one per line (`~/Library/Application Support/Lipflow/words.txt`).
A guess that contains one of your words wins over the others and gets your capitalization, and
the LLM is told about them.

## Using it

| Do this | To |
|---|---|
| Hold **Right Option**, mouth the words, release | dictate |
| Double-tap **Right Option** … tap again | hands-free (up to 60 s) |
| **Esc** while listening | cancel |
| Lipflow menu → Copy last dictation / Open history | get text back |

Lipflow keeps filming for 0.4 s after you release the key, because the model needs the frames
after the last word to read it.

Options: `uv run lipflow --help`

```
--key {right_option,left_option,right_command,right_control,fn}
--cleanup {auto,chatgpt,local,basic}
--beam N          beam size (default 10)
--copy-only       copy to the clipboard instead of pasting
--camera N|FILE   camera index, or a video file to stand in for the webcam
```

Lip-read a video file: `uv run lipflow file talk.mp4 --start 10 --end 20`

## How it works

- **Model:** [Auto-AVSR](https://github.com/mpc001/auto_avsr) visual-only speech recognition trained
  on LRS3 (19.1% WER on the benchmark). A 3D-conv ResNet front end and a Conformer encoder feed a
  Transformer decoder plus CTC, with a subword Transformer language model in the beam search.
- **Preprocessing:** MediaPipe FaceLandmarker runs on every frame *while you're recording*, so
  there's no second detection pass afterwards. Eye, nose-base and mouth anchors are aligned to the
  training mean face, then 96×96 grayscale mouth crops are resampled to the 25 fps the model expects.
- **Speed (M4 Pro, 9 s utterance):** encoder 0.16 s on the Apple GPU (it's 1.7 s on CPU), beam
  search about 0.8–1.6 s on CPU (faster than MPS for thousands of tiny ops), for about 1–2 s from
  release to text. Two patches to the vendored ESPnet help: cross-attention keys/values are
  projected once per utterance instead of per hypothesis per step (25% faster beam search), and CTC
  scoring works on any device.

Accuracy on held-out news footage (public-domain White House addresses), raw model output:

| Said | Read |
|---|---|
| Born in New York City, and raised mostly in Chicago, Nancy Davis graduated from Smith College in 1943. | BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE IN NINETEEN FORTY THREE |
| …a real-life Hollywood romance with the love of her life, Ronald Reagan, whom she married in 1952. | ROMANCE WITH THE LOVE OF HER LIFE RONALD REAGAN WHOM SHE MARRIED IN NINETEEN FIFTY TWO |
| Presidents have delivered some form of final message while in office - a farewell address to the American people. | PRESIDENTS HAVE DELIVERED SOME FORM OF FINAL MESSAGE WHILE IN OFFICE FAREWELL ADDRESS TO THE AMERICAN PEOPLE |

Silently mouthed speech is harder than filmed speech (smaller lip movements), so expect more
errors on your own webcam. That's what the LLM cleanup is for.

## Development

```sh
./setup.sh --samples      # also fetch the public-domain test clips
uv run pytest             # the paste test is opt-in: LIPFLOW_TEST_PASTE=1
```

Code map: `lipflow/face.py` (landmarks → mouth crops), `vsr.py` (model), `camera.py`, `hotkey.py`
(Quartz event tap; pynput's macOS listener crashes on recent macOS), `paste.py`, `hud.py`,
`cleanup.py`, `app.py` (wiring + menu bar). See `NOTICE` for bundled code and model licensing.
The LRS3-trained weights are for non-commercial research use.

## License

MIT, see [LICENSE](LICENSE). Bundled third-party code keeps its own license, listed in
[NOTICE](NOTICE). The model weights that setup.sh downloads come from the LRS3 dataset, which is
for non-commercial research use only.
