import torch

def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    elif att_mode == 2:
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = features[left:right]
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')
    

import torch
import torch.nn as nn
import numpy as np
from os.path import basename
import librosa
from scipy import signal

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram):
    global _mel_basis
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)

def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)

def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20

    return _normalize(S)

class AudDataset(object):
    def __init__(self, wavpath):
        wav = load_wav(wavpath, 16000)

        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # print(end_idx, spec.shape[0])
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel



class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out