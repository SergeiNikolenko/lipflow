"""Russian lip reading: AV-HuBERT (MuAViC-ru) re-implemented in plain PyTorch.

Weights: nguyenvulebinh/AV-HuBERT-MuAViC-ru (an HF port of Meta's MuAViC Russian AVSR checkpoint,
CC-BY-NC 4.0). The port's own code needs transformers 4.44 internals, so the network is rebuilt here
with the same parameter names and loaded straight from model.safetensors:

  video (T,88,88) ─ 3D-conv + ResNet-18 ─ proj ─┐
  audio log-fbank (T,104) ──────────── proj ────┴ concat ─ LN ─ proj ─ 24-layer Transformer ─┐
                                                 6-layer Transformer decoder (BPE, 1000) ◄──┘

One model reads lips alone (the audio half is zeros, as in its modality-dropout training) or lips +
a whisper. Mouth crops are the same 96x96 aligned patches Auto-AVSR uses (center 88x88, same
mean/std), so the capture pipeline is shared with the English model.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vsr import MODELS, LipReader, _MEAN, _STD, pick_encoder_device

RU_DIR = os.path.join(MODELS, "ru")
HF_REPO = "nguyenvulebinh/AV-HuBERT-MuAViC-ru"
BOS = EOS = 2  # fairseq-style: decoding starts from </s>
PAD, UNK = 1, 3
SAMPLES_PER_FRAME = 640  # 16 kHz / 25 fps


def available() -> bool:
    return os.path.exists(os.path.join(RU_DIR, "model.safetensors"))


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, local_dir=RU_DIR,
                      allow_patterns=["config.json", "model.safetensors", "sentencepiece.bpe.model", "vocab.json"])


# -- network ------------------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + (self.downsample(x) if self.downsample is not None else x))


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._layer(64, 1)
        self.layer2 = self._layer(128, 2)
        self.layer3 = self._layer(256, 2)
        self.layer4 = self._layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _layer(self, planes, stride):
        down = None
        if stride != 1 or self.inplanes != planes:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        blocks = [BasicBlock(self.inplanes, planes, stride, down), BasicBlock(planes, planes)]
        self.inplanes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)


class ResEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False), nn.BatchNorm3d(64), nn.PReLU(64),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        self.trunk = Trunk()

    def forward(self, x):  # (B, 1, T, 88, 88) -> (B, 512, T)
        x = self.frontend3D(x)
        B, C, T, H, W = x.shape
        x = self.trunk(x.transpose(1, 2).reshape(B * T, C, H, W))
        return x.view(B, T, -1).transpose(1, 2)


class SubModel(nn.Module):
    def __init__(self, resnet, input_dim, dim):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, x):  # (B, C, T) or video -> (B, D, T)
        if self.resnet is not None:
            x = self.resnet(x)
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads):
        super().__init__()
        self.h = heads
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)

    def kv(self, mem):
        B, S, _ = mem.shape
        k = self.k_proj(mem).view(B, S, self.h, -1).transpose(1, 2)
        v = self.v_proj(mem).view(B, S, self.h, -1).transpose(1, 2)
        return k, v

    def forward(self, x, mem=None, causal=False, kv=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.h, -1).transpose(1, 2)
        k, v = kv if kv is not None else self.kv(x if mem is None else mem)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal and T > 1)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.intermediate_dense = nn.Linear(dim, hidden)
        self.output_dense = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, dim, heads)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, hidden)

    def forward(self, x):
        x = x + self.attention(self.layer_norm(x))
        return x + self.feed_forward(self.final_layer_norm(x))


class PosConv(nn.Module):
    def __init__(self, dim, kernel, groups):
        super().__init__()
        conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=groups)
        self.conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
        self.trim = 1 if kernel % 2 == 0 else 0

    def forward(self, x):  # (B, T, D)
        h = self.conv(x.transpose(1, 2))
        if self.trim:
            h = h[:, :, :-self.trim]
        return F.gelu(h).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["encoder_hidden_size"]
        self.pos_conv_embed = PosConv(d, cfg["num_conv_pos_embeddings"], cfg["num_conv_pos_embedding_groups"])
        self.layers = nn.ModuleList([EncoderLayer(d, cfg["encoder_attention_heads"], cfg["intermediate_size"])
                                     for _ in range(cfg["num_hidden_layers"])])
        self.layer_norm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])

    def forward(self, x):
        x = x + self.pos_conv_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class AVHubert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["encoder_embed_dim"]
        self.d = d
        self.feature_extractor_audio = SubModel(None, cfg["audio_feat_dim"], d)
        self.feature_extractor_video = SubModel(ResEncoder(), 512, d)
        self.layer_norm = nn.LayerNorm(2 * d)
        self.post_extract_proj = nn.Linear(2 * d, d)
        self.encoder = Encoder(cfg)

    def forward(self, video=None, audio=None):
        """video (B,1,T,88,88) and/or audio (B,104,T) -> (B,T,D)."""
        fv = self.feature_extractor_video(video) if video is not None else None
        fa = self.feature_extractor_audio(audio) if audio is not None else None
        if fv is None:
            fv = fa.new_zeros(fa.shape)
        if fa is None:
            fa = fv.new_zeros(fv.shape)
        n = min(fa.shape[-1], fv.shape[-1])
        x = torch.cat([fa[..., :n], fv[..., :n]], dim=1).transpose(1, 2)
        return self.encoder(self.post_extract_proj(self.layer_norm(x)))


class DecoderLayer(nn.Module):
    def __init__(self, dim, enc_dim, heads, hidden):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, dim, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn = Attention(dim, enc_dim, heads)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, mem_kv):
        x = x + self.self_attn(self.self_attn_layer_norm(x), causal=True)
        x = x + self.encoder_attn(self.encoder_attn_layer_norm(x), kv=mem_kv)
        return x + self.fc2(F.relu(self.fc1(self.final_layer_norm(x))))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["decoder_hidden_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], d, padding_idx=PAD)
        self.register_buffer("embed_positions_weights", torch.zeros(cfg["max_target_positions"] + 2, d))
        self.scale = math.sqrt(d) if cfg.get("scale_embedding") else 1.0
        self.layers = nn.ModuleList([DecoderLayer(d, cfg["encoder_hidden_size"], cfg["decoder_attention_heads"],
                                                  cfg["decoder_ffn_dim"]) for _ in range(cfg["decoder_layers"])])
        self.layer_norm = nn.LayerNorm(d)

    def memory(self, enc):
        """Cross-attention keys/values, computed once per utterance and shared by every hypothesis."""
        return [layer.encoder_attn.kv(enc) for layer in self.layers]

    def forward(self, ys, mem):  # ys (B, L) without padding
        pos = torch.arange(ys.shape[1], device=ys.device) + PAD + 1
        x = self.embed_tokens(ys) * self.scale + self.embed_positions_weights[pos]
        for layer, kv in zip(self.layers, mem):
            x = layer(x, kv)
        return self.layer_norm(x) @ self.embed_tokens.weight.T


class AV2Text(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AVHubert(cfg)
        self.decoder = Decoder(cfg)

    def load(self, path):
        from safetensors.torch import load_file
        sd = {}
        for k, v in load_file(path).items():
            k = k.removeprefix("model.")
            if k == "decoder.embed_positions.weights":
                k = "decoder.embed_positions_weights"
            elif k.startswith(("encoder.label_embs", "encoder.mask_emb", "lm_head")):
                continue  # pre-training leftovers; the output layer is tied to embed_tokens
            sd[k] = v
        self.load_state_dict(sd)


# -- reader -------------------------------------------------------------------------------------

def log_fbank(wave: np.ndarray, rate: int = 16000) -> np.ndarray:
    """26 log mel filterbanks every 10 ms (python_speech_features.logfbank defaults)."""
    sig = np.append(wave[0], wave[1:] - 0.97 * wave[:-1]).astype(np.float64)
    flen, step, nfft = int(0.025 * rate), int(0.01 * rate), 512
    n = 1 + max(0, int(math.ceil((len(sig) - flen) / step)))
    sig = np.pad(sig, (0, max(0, (n - 1) * step + flen - len(sig))))
    idx = np.arange(flen)[None, :] + step * np.arange(n)[:, None]
    pow_spec = np.abs(np.fft.rfft(sig[idx], nfft)) ** 2 / nfft
    mel = lambda hz: 2595 * np.log10(1 + hz / 700.0)
    pts = 700 * (10 ** (np.linspace(mel(0), mel(rate / 2), 28) / 2595) - 1)
    bins = np.floor((nfft + 1) * pts / rate).astype(int)
    fb = np.zeros((26, nfft // 2 + 1))
    for j in range(26):
        a, b, c = bins[j], bins[j + 1], bins[j + 2]
        fb[j, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[j, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    feat = pow_spec @ fb.T
    return np.log(np.where(feat == 0, np.finfo(float).eps, feat)).astype(np.float32)


class RuReader:
    """Same interface as LipReader (encode / greedy / beam_search / read), plus whisper mode
    (encode_av) from the same weights."""
    lang = "ru"
    resample = staticmethod(LipReader.resample)
    to_tensor = staticmethod(LipReader.to_tensor)

    def __init__(self, device: str = "auto", beam_size: int = 5, personal: bool = True, **_):
        import sentencepiece as spm
        self.enc_device = pick_encoder_device(device)
        self.device = torch.device("cuda") if self.enc_device.type == "cuda" else torch.device("cpu")
        self.beam_size = beam_size
        cfg = json.load(open(os.path.join(RU_DIR, "config.json")))
        self.model = AV2Text(cfg)
        self.model.load(os.path.join(RU_DIR, "model.safetensors"))
        from .paths import PERSONAL_VSR_RU
        self.personal_vsr = False
        if personal and os.path.exists(PERSONAL_VSR_RU):
            self.model.load_state_dict(torch.load(PERSONAL_VSR_RU, map_location="cpu", weights_only=True),
                                       strict=False)
            self.personal_vsr = True
        self.model.eval()
        self.model.encoder.to(self.enc_device)
        self.model.decoder.to(self.device)
        vocab = json.load(open(os.path.join(RU_DIR, "vocab.json")))
        self.piece_id = vocab
        self.token_list = [None] * len(vocab)
        for p, i in vocab.items():
            self.token_list[i] = p
        self.spm = spm.SentencePieceProcessor(model_file=os.path.join(RU_DIR, "sentencepiece.bpe.model"))
        self.personal_lm = False

    # -- features ---------------------------------------------------------------------------
    @staticmethod
    def audio_tensor(wave: np.ndarray, n_frames: int) -> torch.Tensor:
        """16 kHz mono -> (104, n_frames): 4 stacked 10 ms filterbank frames per video frame,
        each normalised on its own (MuAViC's layer_norm over the feature axis)."""
        w = np.asarray(wave, dtype=np.float32)[:n_frames * SAMPLES_PER_FRAME]
        w = np.pad(w, (0, n_frames * SAMPLES_PER_FRAME - len(w)))
        f = log_fbank(w * 32768.0)
        f = np.pad(f, ((0, (-len(f)) % 4), (0, 0))).reshape(-1, 104)
        f = np.pad(f, ((0, max(0, n_frames - len(f))), (0, 0)))[:n_frames]
        x = torch.from_numpy(f)
        return F.layer_norm(x, x.shape[1:]).T

    @torch.inference_mode()
    def encode(self, rois: np.ndarray) -> torch.Tensor:
        v = self.to_tensor(rois).unsqueeze(0).to(self.enc_device)
        return self.model.encoder(video=v).squeeze(0).to(self.device)

    @torch.inference_mode()
    def encode_av(self, rois: np.ndarray, wave: np.ndarray) -> torch.Tensor:
        v = self.to_tensor(rois).unsqueeze(0).to(self.enc_device)
        a = self.audio_tensor(wave, rois.shape[0]).unsqueeze(0).to(self.enc_device)
        return self.model.encoder(video=v, audio=a).squeeze(0).to(self.device)

    # -- decoding ---------------------------------------------------------------------------
    def detok(self, ids: list[int]) -> str:
        pieces = [self.token_list[i] for i in ids if i > UNK]
        return " ".join("".join(pieces).replace("▁", " ").split())

    def encode_text(self, text: str) -> list[int]:
        return [self.piece_id.get(p, UNK) for p in self.spm.encode(text.lower(), out_type=str)]

    @torch.inference_mode()
    def beam_search(self, enc: torch.Tensor, nbest: int = 1, beam: "int | None" = None) -> "str | list[str]":
        beam = beam or self.beam_size
        dec = self.model.decoder
        mem1 = dec.memory(enc.unsqueeze(0))
        max_len = min(200, enc.shape[0] + 10)
        hyps = [([BOS], 0.0)]
        done: list[tuple[list[int], float]] = []
        for _ in range(max_len):
            ys = torch.tensor([h[0] for h in hyps], device=self.device)
            mem = [(k.expand(len(hyps), -1, -1, -1), v.expand(len(hyps), -1, -1, -1)) for k, v in mem1]
            logp = dec(ys, mem)[:, -1].log_softmax(-1)
            logp[:, PAD] = logp[:, UNK] = logp[:, 0] = -1e9
            scores = torch.tensor([h[1] for h in hyps], device=self.device)[:, None] + logp
            top = scores.view(-1).topk(min(2 * beam, scores.numel()))
            nxt = []
            for s, flat in zip(top.values.tolist(), top.indices.tolist()):
                b, tok = divmod(flat, logp.shape[1])
                seq = hyps[b][0] + [tok]
                if tok == EOS:
                    done.append((seq[1:-1], s / len(seq[1:])))  # length-normalised, as in fairseq
                else:
                    nxt.append((seq, s))
                if len(nxt) == beam:
                    break
            hyps = nxt
            if len(done) >= beam and max(d[1] for d in done) > max(h[1] / len(h[0]) for h in hyps):
                break
            if not hyps:
                break
        done += [(h[0][1:], h[1] / len(h[0])) for h in hyps] if not done else []
        texts = []
        for ids, _ in sorted(done, key=lambda d: -d[1]):
            t = self.detok(ids)
            if t not in texts:
                texts.append(t)
        texts = texts or [""]
        return texts[:max(nbest, 1)] if nbest > 1 else texts[0]

    def greedy(self, enc: torch.Tensor) -> str:
        """Live preview while you talk: beam 1."""
        return self.beam_search(enc, beam=1)

    def read(self, rois: np.ndarray, fast: bool = False) -> tuple[str, float]:
        t0 = time.time()
        enc = self.encode(rois)
        return (self.greedy(enc) if fast else self.beam_search(enc)), time.time() - t0

    def read_av(self, rois, wave) -> tuple[str, float]:
        t0 = time.time()
        return self.beam_search(self.encode_av(rois, wave)), time.time() - t0

    def warmup(self):
        self.read(np.zeros((25, 96, 96), dtype=np.uint8), fast=True)

    def warmup_av(self):
        self.encode_av(np.zeros((25, 96, 96), np.uint8), np.random.default_rng(0).normal(0, 0.01, 16000))

    # -- face adaptation (train_vsr) --------------------------------------------------------
    def targets(self, text: str) -> list[int]:
        return self.encode_text(" ".join(text.split()))

    def clip_loss(self, x: torch.Tensor, ys: list[int]) -> torch.Tensor:
        """Teacher-forced, label-smoothed cross-entropy (MuAViC's fine-tuning loss)."""
        dev = self.enc_device
        enc = self.model.encoder(video=x.unsqueeze(0).to(dev))
        dec = self.model.decoder
        inp = torch.tensor([[BOS] + ys], device=dev)
        out = torch.tensor([ys + [EOS]], device=dev)
        logits = dec(inp, dec.memory(enc))
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), out.view(-1), label_smoothing=0.1)

    def trainable(self, scope: str = "frontend+encoder1"):
        m = self.model
        for p in m.parameters():
            p.requires_grad_(False)
        params = list(m.encoder.feature_extractor_video.parameters())
        if scope == "frontend+encoder1":
            params += list(m.encoder.encoder.layers[0].parameters())
        for p in params:
            p.requires_grad_(True)
        return params

    def trained_prefixes(self, scope: str = "frontend+encoder1") -> list[str]:
        p = ["encoder.feature_extractor_video."]
        if scope == "frontend+encoder1":
            p.append("encoder.encoder.layers.0.")
        return p


__all__ = ["RuReader", "available", "download", "_MEAN", "_STD"]
