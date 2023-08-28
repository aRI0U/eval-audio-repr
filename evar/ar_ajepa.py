"""Wrapper code for:

BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation

## Reference
- [1] https://arxiv.org/abs/2103.06695
- [2] https://github.com/nttcslab/byol-a
"""

from .ar_base import BaseAudioRepr, temporal_pooling
import logging
import torch
try:
    from src.helper import init_model, load_checkpoint
    from src.datasets.safe_audioset import LogMelSpectrogram
except (ImportError, ModuleNotFoundError) as e:
    logging.info("Didn't find local modules")
    try:
        from external.ajepa.src.helper import init_model, load_checkpoint
        from external.ajepa.src.datasets.safe_audioset import LogMelSpectrogram
    except (ModuleNotFoundError, ImportError) as e:
        logging.info(f'Make your copy of A-JEPA under external folder. Check Preparing-models.md for the details.')


class AR_AJEPA(BaseAudioRepr):
    def __init__(self, cfg, model=None, postnorm: str = None):
        super().__init__(cfg=cfg)
        n_mels, spec_length = cfg.crop_size
        self.to_feature = LogMelSpectrogram(spec_length=spec_length,
                                            convert_to_mono=False,
                                            sample_rate=cfg.sample_rate,
                                            n_fft=cfg.n_fft,
                                            win_length=cfg.win_length,
                                            hop_length=cfg.hop_length,
                                            f_min=cfg.f_min,
                                            f_max=cfg.f_max,
                                            n_mels=n_mels,
                                            normalized=True)

        if model is None:
            self.body, _ = init_model(
                device=torch.device("cpu"),
                patch_size=cfg.patch_size,
                model_name=cfg.model_name,
                crop_size=cfg.crop_size,
                in_chans=1
            )
            if cfg.weight_file is not None:
                load_checkpoint(cfg.weight_file, self.body, None)
        else:
            self.body = model

        postnorm = postnorm or cfg.get("postnorm")

        # post-normalization
        if postnorm == "naive":
            mean, std = -20, 40
        elif postnorm == "dataset":
            if n_mels == 128:
                mean, std = -31.2, 16.18
            elif n_mels == 80:
                mean, std = -28.8, 16.3  # ok
            elif n_mels == 81:
                mean, std = -32.87, 15.89  # ok (300000)
            else:
                raise ValueError
        elif postnorm is None:
            mean, std = 0, 1
        else:
            raise ValueError(f"Invalid post-normalization value: {postnorm}.")
        self.mean, self.std = mean, std

        print()
        print(postnorm, self.mean, self.std)
        print()

    def precompute(self, device, data_loader):
        self.to_feature = self.to_feature.to(device)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x.sub_(self.mean).div_(self.std)

        # x = normalize_spectrogram(self.norm_stats, x) # B,F,T
        x = x.unsqueeze(1)     # -> B,1,F,T
        x = self.body(x)       # -> B,T,D=C*F
        x = x.transpose(1, 2)  # -> B,D,T
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x
