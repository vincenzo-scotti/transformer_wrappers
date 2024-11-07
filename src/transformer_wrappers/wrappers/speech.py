import os
import logging
from datetime import datetime
import pickle

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers

import librosa
from sklearn.preprocessing import StandardScaler

from transformers import logging
from transformers import PreTrainedModel, BatchEncoding
from transformers import GPT2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from peft.peft_model import PeftModel
from transformers import logging as hf_logging
from transformer_wrappers.optim import optimizer_mapping, lr_scheduler_mapping

from typing import Type, Optional, Union, List, Iterable, Tuple, Dict

from .base import (
    SHARED_STRUCTURE_MODELS,
    ModuleWrapper,
    EmbeddingWrapper,
    TransformerWrapper,
    LMHeadWrapper,
    CausalLMWrapper
)
from.base.dtypes import *
from .base.constants import *


__all__ = ['AudioProcessor', 'SpeechTransformerWrapper', 'SpeechCausalLMWrapper']

logger = hf_logging.get_logger(__name__)

AUDIO_TOKEN: str = 'audio_token'
SPEECH_ENCODER_CONFIGS: str = 'speech_encoder'
SPEECH_DECODER_CONFIGS: str = 'speech_decoder'
POST_NET_CONFIGS: str = 'post_net'

INPUT_SPECTROGRAMS: str = 'input_spectrograms'
SPEECH_MASK: str = 'speech_mask'
APPEND_MASK: str = 'append_mask'

SPECTROGRAMS: str = 'spectrograms'
GENERATED_SPECTROGRAMS: str = 'generated_spectrograms'
OUTPUT_SPECTROGRAMS: str = 'output_spectrograms'

TOKEN_LABELS: str = 'token_labels'
TARGET_SPECTROGRAMS: str = 'target_spectrograms'

LOSS_VALUE: str = 'loss_value'
LOSS_COMPONENTS: str = 'loss_components'
LM_LOSS: str = 'language_modelling_loss'
SPEC_LOSS: str = 'spectrogram_generation_loss'

SR: str = 'sr'
WIN_SIZE: str = 'win_size'
HOP_SIZE: str = 'hop_size'
N_FFT: str = 'n_fft'
N_MEL: str = 'n_mel'
N_MFCC: str = 'n_mfcc'


class LayerNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)
        
    def forward(self, x):
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


class AudioProcessor:
    STANDARD_SCALER_FILE: str = 'audio_scaler.pickle'

    def __init__(
            self,
            sr: int = 16000,
            win_size: float = 0.025,  # In seconds
            hop_size: Optional[float] = 0.01,  # In seconds, defaults to window size
            n_fft: int = 512,
            n_mel: Optional[int] = 80,  # Typical value is 80 if not None, change to match speech embeddings requirements
            n_mfcc: Optional[int] = None  # Typical value is 12 if not None, change to match speech embedding requirements
    ):
        self.sr: int = sr
        self.win_size: float = win_size
        self.hop_size: float = hop_size if hop_size is not None else win_size
        self.n_fft: int = n_fft
        self.n_mel: Optional[int] = n_mel
        self.n_mfcc: Optional[int] = n_mfcc
        #
        self._win_size_samples: int = int(math.ceil(self.win_size * self.sr))
        self._hop_size_samples: int = int(math.ceil(self.hop_size * self.sr))
        #
        self._scaler: Optional[StandardScaler] = None

    @property
    def channels(self) -> int:
        if self.n_mfcc is not None:
            return self.n_mfcc
        elif self.n_mel is not None:
            return self.n_mel
        else:
            return self.n_fft


    def load_audio(self, path: str) -> np.ndarray:
        speech_data, _ = librosa.load(path, sr=self.sr)

        return speech_data

    def encode(
            self, speech_data: Union[Iterable[str], Iterable[np.ndarray], str, np.ndarray]
    ) -> Union[List[np.ndarray], np.ndarray]:
        # NOTE: output is channel first
        if isinstance(speech_data, str):
            return self.encode(self.load_audio(speech_data))
        elif isinstance(speech_data, Iterable) and all(
                isinstance(speech_data_, str) or isinstance(speech_data_, np.ndarray) for speech_data_ in speech_data
        ):
            return [self.encode(self.load_audio(speech_data_)) for speech_data_ in speech_data]
        elif isinstance(speech_data, np.ndarray):
            if self.n_mel is None and self.n_mfcc is None:
                spec = librosa.stft(
                    y=speech_data,
                    n_fft=self.n_fft - int(self.n_fft % 2 == 0),  # TODO check this
                    win_length=self._win_size_samples,
                    hop_length=self._hop_size_samples
                )
                spec = librosa.power_to_db(np.abs(spec) ** 2, ref=np.max)
                if self._scaler is not None:
                    spec = self._scaler.transform(spec)

                return spec
            elif self.n_mel is not None and self.n_mfcc is None:
                mel_spec = librosa.feature.melspectrogram(
                    y=speech_data,
                    n_fft=self.n_fft - int(self.n_fft % 2 == 0),  # TODO check this
                    win_length=self._win_size_samples,
                    hop_length=self._hop_size_samples,
                    n_mels=self.n_mel
                )
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                if self._scaler is not None:
                    mel_spec = self._scaler.transform(mel_spec)

                return mel_spec
            elif self.n_mel is not None and self.n_mfcc is not None:
                mfcc = librosa.feature.mfcc(
                    y=speech_data,
                    sr=self.sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft - int(self.n_fft % 2 == 0),  # TODO check this
                    win_length=self._win_size_samples,
                    hop_length=self._hop_size_samples,
                    n_mels=self.n_mel
                )
                if self._scaler is not None:
                    mfcc = self._scaler.transform(mfcc)

                return mfcc
            else:
                raise ValueError(
                    'Invalid configuration, `n_mel` attribute must be specified when `n_mfcc` is specified'
                )
        else:
            raise TypeError(f'Unsupported type {type(speech_data)}')

    def decode(self, *args, **kwargs):
        raise NotImplementedError(
            'Implement Griffin-Limm algorithm or Vocder DNN for this step '
            '(see: https://github.com/vincenzo-scotti/tts_mozilla_api and '
            'https://github.com/vincenzo-scotti/tts_mellotron_api)'
        )

    @staticmethod
    def get_encoded_length(speech_data: Union[np.ndarray, torch.Tensor], embedding_dim: int):
        return int(math.ceil(speech_data.numel() / embedding_dim))

    def fit_scaler(self, speech_data: Union[Iterable[str], Iterable[np.ndarray], str, np.ndarray]):
        #
        if self._scaler is not None:
            self._scaler = None
        speech_data = self.encode(speech_data)
        #
        if not isinstance(speech_data, np.ndarray):
            speech_data = np.hstack(speech_data)
        speech_data = speech_data.T

        self._scaler = StandardScaler().fit(speech_data)

    def load_scaler(self, path: str):
        with open(path, 'rb') as f:
            self._scaler = pickle.load(f)

    def serialise_scaler(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self._scaler, f)


class SpeechEmbeddingWrapper(EmbeddingWrapper):
    SPEECH_ENCODER_FILE: str = 'speech_encoder.pth'

    def __init__(self, module: nn.Module, speech_encoder: nn.Module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        #
        self._speech_encoder: nn.Module = speech_encoder

    @property
    def speech_encoder(self):
        return self._speech_encoder

    def _wrapped_forward(
            self,
            *args,
            input_spectrograms: Optional[torch.Tensor] = None,
            speech_mask: Optional[torch.BoolTensor] = None,
            **kwargs
    ):
        # Run base forward
        output = super()._wrapped_forward(*args, **kwargs)
        # Check whether there are spectrograms to embed
        if input_spectrograms is not None:
            #
            spectrogram_embeddings = self.speech_encoder.forward(input_spectrograms.to(output[self.module_output]))
            output[self.module_output][speech_mask] += spectrogram_embeddings.transpose(-1, -2)[speech_mask]
        #
        output |= {
            SPEECH_MASK: speech_mask
        }

        return output


class SpeechTransformerWrapper(TransformerWrapper):
    _embedding_dtype: Type[ModuleWrapper] = SpeechEmbeddingWrapper

    @property
    def wrapper_args(self):
        return super().wrapper_args | {INPUT_SPECTROGRAMS, GENERATED_SPECTROGRAMS}

    def _post_init_operations(
            self,
            audio_processor: AudioProcessor,
            speech_encoder: nn.Module,
            *args,
            **kwargs
    ):
        # Attribute names
        self._embedding_attr: TransformerEmbeddingAttr = self._get_embedding_attr()
        self._position_embedding_attr: Optional[TransformerPositionEmbeddingAttr] = self._get_position_embedding_attr()
        self._layers_attr: TransformerLayersAttr = self._get_layers_attr()
        self._norm_attr: TransformerNormAttr = self._get_norm_attr()
        # Wrappers
        self._embedding_wrapper: Tuple = self._embedding_dtype(
            getattr(self.base_model, self._embedding_attr.value),
            super_wrapper=self,
            position_embeddings=getattr(
                self.base_model, self._position_embedding_attr.value
            ) if self._position_embedding_attr is not None else None,
            speech_encoder=speech_encoder
        ),
        self._layers_wrapper: Tuple = self._layers_dtype(
            getattr(self.base_model, self._layers_attr.value), super_wrapper=self
        ),
        #
        self._audio_processor: AudioProcessor = audio_processor
        #
        self._audio_token: str = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(
            AUDIO_TOKEN, '<|audio|>'
        )
    @property
    def audio_processor(self) -> AudioProcessor:
        return self._audio_processor

    @property
    def audio_token(self) -> str:
        return self._audio_token

    @property
    def audio_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.audio_token)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            model_args: Optional[Tuple] = None,
            model_kwargs: Optional[Dict] = None,
            quantization_configs: Optional[BitsAndBytesConfig] = None,
            lora_configs: Optional[LoraConfig] = None,
            peft: bool = False,
            gradient_checkpointing: bool = False,
            tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
            tokenizer_args: Optional[Tuple] = None,
            tokenizer_kwargs: Optional[Dict] = None,
            **wrapper_kwargs
    ):
        model, tokenizer = cls._load_pretrained(
            pretrained_model_name_or_path,
            model_args=model_args,
            model_kwargs=model_kwargs,
            quantization_configs=quantization_configs,
            lora_configs=lora_configs,
            peft=peft,
            tokenizer_name_or_path=tokenizer_name_or_path,
            tokenizer_args=tokenizer_args,
            tokenizer_kwargs=tokenizer_kwargs,
            **wrapper_kwargs
        )
        if model.config.vocab_size != len(tokenizer):
            old_vocab_size = model.config.vocab_size
            model.resize_token_embeddings(len(tokenizer))
            model.get_input_embeddings().weight.data[old_vocab_size:] = 0.

        audio_processor = AudioProcessor(
            sr=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(SR, 16000),
            win_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(WIN_SIZE, 0.025),
            hop_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(HOP_SIZE, 0.01),
            n_fft=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_FFT, 512),
            n_mel=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MEL, 128),
            n_mfcc=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MFCC)
        )
        if os.path.exists(os.path.join(pretrained_model_name_or_path, AudioProcessor.STANDARD_SCALER_FILE)):
            audio_processor.load_scaler(os.path.join(pretrained_model_name_or_path, AudioProcessor.STANDARD_SCALER_FILE))

        act_key = 'activation_function' if isinstance(
            model if lora_configs is None else model.base_model.base_model, GPT2PreTrainedModel
        ) else 'hidden_act'
        speech_encoder_configs = model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(
            SPEECH_ENCODER_CONFIGS, {
                'in_channels': audio_processor.channels,
                'out_channels': model.config.hidden_size,
                'kernel_size': model.config.hidden_size // audio_processor.channels,
                'stride': model.config.hidden_size // audio_processor.channels
            }
        )
        if isinstance(speech_encoder_configs, dict):
            speech_encoder_configs = [speech_encoder_configs]
        for i in range(len(speech_encoder_configs)):
            if i == 0:
                speech_encoder_configs[i] |= {'in_channels': audio_processor.channels}
            else:
                speech_encoder_configs[i] |= {'in_channels': speech_encoder_configs[i - 1]['out_channels']}
            if i == len(speech_encoder_configs) - 1:
                speech_encoder_configs[i] |= {'out_channels': model.config.hidden_size}
            if 'stride' not in speech_encoder_configs[i]:
                speech_encoder_configs[i] |= {'stride': speech_encoder_configs[i]['kernel_size']}
        speech_encoder = nn.Sequential(
            *[
                module
                for configs in speech_encoder_configs
                for module in [
                    nn.Conv1d(
                        **configs,
                        bias=False,
                        padding='same' if configs.get('stride', 1) == 1 else 'valid',
                        dtype=model.dtype,
                        device=model.device
                    ),
                    ACT2FN.get(getattr(model.config, act_key), nn.GELU)(),
                    nn.Dropout(0.1),
                    LayerNorm1d(configs['out_channels'], device=model.device),
                ]
            ]
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE)
        ):
            speech_encoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE),
                weights_only=True
            ))

        wrapper = cls(model, tokenizer, audio_processor, speech_encoder)

        if gradient_checkpointing:
            wrapper.gradient_checkpointing_enable()

        wrapper.enable_wrapper()

        return wrapper

    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        is_wrapping = self.is_wrapping
        if is_wrapping:
            self.disable_wrapper()
        self.base_model.save_pretrained(save_directory, *args, **kwargs)
        if is_wrapping:
            self.enable_wrapper()
        torch.save(
            self.embedding_wrapper.speech_encoder.state_dict(),
            os.path.join(save_directory, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE)
        )
        self.audio_processor.serialise_scaler(os.path.join(save_directory, AudioProcessor.STANDARD_SCALER_FILE))

    def _pre_process_input(self, *args, speech_mask: Optional[torch.BoolTensor] = None, **kwargs):
        kwargs = super()._pre_process_input(*args, **kwargs)
        #
        if speech_mask is None:
            speech_mask = kwargs[INPUT_IDS] == self.audio_token_id if self.audio_token_id in kwargs[INPUT_IDS] else None
        #
        kwargs |= {SPEECH_MASK: speech_mask}

        return kwargs


class SpeechLMHeadWrapper(LMHeadWrapper):
    SPEECH_DECODER_FILE: str = 'speech_decoder.pth'
    MODALITY_SWITCH_FILE: str = 'modality_switch.pth'
    POST_NET_FILE: str = 'post_net.pth'

    def __init__(
            self, module: nn.Module,
            speech_decoder: nn.Module,
            modality_switch: nn.Module,
            *args,
            post_net: Optional[nn.Module] = None,
            **kwargs
    ):
        super().__init__(module, *args, **kwargs)
        #
        self._speech_decoder: nn.Module = speech_decoder
        self._post_net: Optional[nn.Module] = post_net
        self._modality_switch: nn.Module = modality_switch

    @property
    def speech_decoder(self) -> nn.Module:
        return self._speech_decoder

    @property
    def modality_switch(self):
        return self._modality_switch

    @property
    def post_net(self) -> Optional[nn.Module]:
        return self._post_net

    def _wrapped_forward(
            self,
            output_hidden_state: Optional[torch.tensor] = None,
            speech_mask: Optional[torch.BoolTensor] = None,
            **kwargs
    ):
        if output_hidden_state is None:
            raise ValueError()
        #
        logits = self.base_module.forward(output_hidden_state)
        logits[..., self.super_wrapper.audio_token_id] += self.modality_switch.forward(output_hidden_state).squeeze(-1)
        #
        spectrograms = self.speech_decoder(output_hidden_state.transpose(-1, -2))
        #
        output = kwargs | {
            self.module_output: {
                LOGITS: logits,
                SPECTROGRAMS: spectrograms
            },
            OUT_HIDDEN_STATE: output_hidden_state
        }

        return output

    def _post_process_output(self, *args, generated_spectrograms: Optional[List[torch.Tensor]] = None, **kwargs):
        #
        output = super()._post_process_output(*args, **kwargs)
        #
        if generated_spectrograms is not None:
            if len(generated_spectrograms) > 0:
                generated_spectrograms.append(output[self.module_output][SPECTROGRAMS])
            else:
                generated_spectrograms.extend([
                    kwargs.get(INPUT_SPECTROGRAMS, torch.full_like(output[self.module_output][SPECTROGRAMS], torch.nan)),
                    output[self.module_output][SPECTROGRAMS][..., -self.super_wrapper.speech_conversion_factor:]
                ])
            output |= {GENERATED_SPECTROGRAMS: generated_spectrograms}
        elif self.post_net is not None:
            output[self.module_output][SPECTROGRAMS] = output[self.module_output][SPECTROGRAMS] + self.post_net(output[self.module_output][SPECTROGRAMS])

        return output


class SpeechCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = SpeechTransformerWrapper
    _lm_head_dtype: Type[ModuleWrapper] = SpeechLMHeadWrapper

    def _post_init_operations(
            self,
            audio_processor: AudioProcessor,
            speech_encoder: nn.Module,
            speech_decoder: nn.Module,
            modality_switch: nn.Module,
            *args,
            post_net: Optional[nn.Module] = None,
            **kwargs
    ):
        # Attribute names
        self._transformer_attr = self._get_transformer_attr()
        self._lm_head_attr = self._get_lm_head_attr()
        # Wrappers
        self._transformer_wrapper = self._transformer_dtype(
            getattr(self.internal_model, self._transformer_attr.value),
            self._tokenizer,
            audio_processor,
            speech_encoder
        ),
        self._lm_head_wrapper = self._lm_head_dtype(
            getattr(self.internal_model, self._lm_head_attr.value),
            speech_decoder,
            modality_switch,
            post_net=post_net,
            super_wrapper=self
        ),

        # Lightning module parameters for fine-tuning
        self.optimiser_params = dict()
        self.lr_scheduler_params = dict()
        self.trainer_params = dict()
        self.data_loader_params = dict()
        self.metrics = None
        self._steps_per_epoch = None

    @property
    def audio_processor(self):
        return self.transformer_wrapper.audio_processor

    @property
    def audio_token(self) -> str:
        return self.transformer_wrapper.audio_token

    @property
    def audio_token_id(self) -> int:
        return self.transformer_wrapper.audio_token_id

    @property
    def speech_conversion_factor(self):
        factor = 1
        for module in self.transformer_wrapper.embedding_wrapper.speech_encoder:
            if isinstance(module, nn.Conv1d):
                factor *= module.kernel_size[0]

        return factor

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            model_args: Optional[Tuple] = None,
            model_kwargs: Optional[Dict] = None,
            quantization_configs: Optional[BitsAndBytesConfig] = None,
            lora_configs: Optional[LoraConfig] = None,
            peft: bool = False,
            gradient_checkpointing: bool = False,
            tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
            tokenizer_args: Optional[Tuple] = None,
            tokenizer_kwargs: Optional[Dict] = None,
            **wrapper_kwargs
    ):
        model, tokenizer = cls._load_pretrained(
            pretrained_model_name_or_path,
            model_args=model_args,
            model_kwargs=model_kwargs,
            quantization_configs=quantization_configs,
            lora_configs=lora_configs,
            peft=peft,
            tokenizer_name_or_path=tokenizer_name_or_path,
            tokenizer_args=tokenizer_args,
            tokenizer_kwargs=tokenizer_kwargs,
            **wrapper_kwargs
        )
        if model.config.vocab_size != len(tokenizer):
            old_vocab_size = model.config.vocab_size
            model.resize_token_embeddings(len(tokenizer))
            model.get_input_embeddings().weight.data[old_vocab_size:] = 0.
            model.get_output_embeddings().weight.data[old_vocab_size:] = 0.
            if model.get_output_embeddings().bias is not None:
                model.get_output_embeddings().bias.data[old_vocab_size:] = 0.

        audio_processor = AudioProcessor(
            sr=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(SR, 16000),
            win_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(WIN_SIZE, 0.025),
            hop_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(HOP_SIZE, 0.01),
            n_fft=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_FFT, 512),
            n_mel=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MEL, 128),
            n_mfcc=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MFCC)
        )
        if os.path.exists(os.path.join(pretrained_model_name_or_path, AudioProcessor.STANDARD_SCALER_FILE)):
            audio_processor.load_scaler(os.path.join(pretrained_model_name_or_path, AudioProcessor.STANDARD_SCALER_FILE))

        act_key = 'activation_function' if isinstance(
            model if lora_configs is None else model.base_model.base_model, GPT2PreTrainedModel
        ) else 'hidden_act'
        #
        speech_encoder_configs = model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(
            SPEECH_ENCODER_CONFIGS, {
                'in_channels': audio_processor.channels,
                'out_channels': model.config.hidden_size,
                'kernel_size': model.config.hidden_size // audio_processor.channels,
                'stride': model.config.hidden_size // audio_processor.channels
            }
        )
        if isinstance(speech_encoder_configs, dict):
            speech_encoder_configs = [speech_encoder_configs]
        for i in range(len(speech_encoder_configs)):
            if i == 0:
                speech_encoder_configs[i] |= {'in_channels': audio_processor.channels}
            else:
                speech_encoder_configs[i] |= {'in_channels': speech_encoder_configs[i-1]['out_channels']}
            if i == len(speech_encoder_configs) - 1:
                speech_encoder_configs[i] |= {'out_channels': model.config.hidden_size}
            if 'stride' not in speech_encoder_configs[i]:
                speech_encoder_configs[i] |= {'stride': speech_encoder_configs[i]['kernel_size']}
        speech_encoder = nn.Sequential(
            *[
                module
                for configs in speech_encoder_configs
                for module in [
                    nn.Conv1d(
                        **configs,
                        bias=False,
                        padding='same' if configs.get('stride', 1) == 1 else 'valid',
                        dtype=model.dtype,
                        device=model.device
                    ),
                    ACT2FN.get(getattr(model.config, act_key), nn.GELU)(),
                    nn.Dropout(0.1),
                    LayerNorm1d(configs['out_channels'], device=model.device),
                ]
            ]
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE)
        ):
            speech_encoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE),
                weights_only=True
            ))
        #
        speech_decoder_configs = model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(
            SPEECH_DECODER_CONFIGS, {
                'in_channels': audio_processor.channels,
                'out_channels': model.config.hidden_size,
                'kernel_size': model.config.hidden_size // audio_processor.channels,
                'stride': model.config.hidden_size // audio_processor.channels
            }
        )
        #
        if isinstance(speech_decoder_configs, dict):
            speech_decoder_configs = [speech_decoder_configs]
        for i in range(len(speech_decoder_configs)):
            if i == 0:
                speech_decoder_configs[i] |= {'in_channels': model.config.hidden_size}
            else:
                speech_decoder_configs[i] |= {'in_channels': speech_decoder_configs[i-1]['out_channels']}
            if i == len(speech_decoder_configs) - 1:
                speech_decoder_configs[i] |= {'out_channels': audio_processor.channels}
            if 'stride' not in speech_encoder_configs[i]:
                speech_decoder_configs[i] |= {'stride': speech_decoder_configs[i]['kernel_size']}
        speech_decoder = nn.Sequential(
            *[
                module
                for i, configs in enumerate(speech_decoder_configs)
                for module in [
                    nn.ConvTranspose1d(**configs, bias=False, dtype=model.dtype, device=model.device)
                ] + ([
                   ACT2FN.get(getattr(model.config, act_key), nn.GELU)(),
                   nn.Dropout(0.1),
                   LayerNorm1d(configs['out_channels'], device=model.device),
                ] if i < len(speech_decoder_configs) - 1 else [])
            ]
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.SPEECH_DECODER_FILE)
        ):
            speech_decoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.SPEECH_DECODER_FILE),
                weights_only=True
            ))
        #
        modality_switch = torch.nn.Linear(
            model.config.hidden_size, 1, dtype=model.dtype, device=model.device
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.MODALITY_SWITCH_FILE)
        ):
            modality_switch.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.MODALITY_SWITCH_FILE),
                weights_only=True
            ))
        #
        post_net_configs = model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(POST_NET_CONFIGS)
        if post_net_configs is not None:
            if isinstance(post_net_configs, dict):
                post_net_configs = [post_net_configs]
            for i in range(len(post_net_configs)):
                if i == 0:
                    post_net_configs[i] |= {'in_channels': audio_processor.channels}
                else:
                    post_net_configs[i] |= {'in_channels': post_net_configs[i - 1]['out_channels']}
                if i == len(post_net_configs) - 1:
                    post_net_configs[i] |= {'out_channels': audio_processor.channels}
            post_net = nn.Sequential(
                *[
                    module
                    for i, configs in enumerate(post_net_configs)
                    for module in [
                        nn.Conv1d(
                            **configs,
                            bias=False,
                            padding='same' if configs.get('stride', 1) == 1 else 'valid',
                            dtype=model.dtype,
                            device=model.device
                        )
                    ] + ([
                        ACT2FN.get(getattr(model.config, act_key), nn.GELU)(),
                        nn.Dropout(0.5),
                        LayerNorm1d(configs['out_channels'], device=model.device),
                    ] if i < len(post_net_configs) - 1 else [])
                ]
            )
            if os.path.exists(
                    os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.POST_NET_FILE)
            ):
                speech_decoder.load_state_dict(torch.load(
                    os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.POST_NET_FILE),
                    weights_only=True
                ))
        else:
            post_net = None

        wrapper = cls(
            model, tokenizer, audio_processor, speech_encoder, speech_decoder, modality_switch, post_net=post_net
        )

        if gradient_checkpointing:
            wrapper.gradient_checkpointing_enable()

        wrapper.enable_wrapper()

        return wrapper

    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        is_wrapping = self.is_wrapping
        if is_wrapping:
            self.disable_wrapper()
        self.base_model.save_pretrained(save_directory, *args, **kwargs)
        if is_wrapping:
            self.enable_wrapper()
        torch.save(
            self.transformer_wrapper.embedding_wrapper.speech_encoder.state_dict(),
            os.path.join(save_directory, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE)
        )
        torch.save(
            self.lm_head_wrapper.speech_decoder.state_dict(),
            os.path.join(save_directory, SpeechLMHeadWrapper.SPEECH_DECODER_FILE)
        )
        torch.save(
            self.lm_head_wrapper.modality_switch.state_dict(),
            os.path.join(save_directory, SpeechLMHeadWrapper.MODALITY_SWITCH_FILE)
        )
        if self.lm_head_wrapper.post_net is not None:
            torch.save(
                self.lm_head_wrapper.post_net.state_dict(),
                os.path.join(save_directory, SpeechLMHeadWrapper.POST_NET_FILE)
            )
        self.audio_processor.serialise_scaler(os.path.join(save_directory, AudioProcessor.STANDARD_SCALER_FILE))

    def _spectrogram_generation_loss(self, predicted: torch.Tensor, target: torch.Tensor):
        # Shift predictions to exclude the last element
        predicted = predicted[..., :-self.speech_conversion_factor]
        # shift targets to exclude the first element
        target = target[..., self.speech_conversion_factor:].to(predicted)
        # Get valid output maks
        mask = ~target.isnan()
        predicted = predicted[mask]
        target = target[mask]
        # Compute LM loss token-wise
        loss: torch.Tensor = F.mse_loss(predicted, target)

        return loss

    def _loss(
            self,
            token_logits: torch.Tensor,
            token_labels: torch.Tensor,
            predicted_spectrograms: Optional[torch.Tensor] = None,
            target_spectrograms: Optional[torch.Tensor] = None,
            return_components: bool = True
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        # LM loss
        lm_loss = CausalLMWrapper._loss(token_logits, token_labels)
        # Spectrogram generation loss
        spec_loss = self._spectrogram_generation_loss(
            predicted_spectrograms, target_spectrograms
        ) if predicted_spectrograms is not None and target_spectrograms is not None else 0.0
        # Total loss
        loss = lm_loss + spec_loss

        return loss, {LM_LOSS: lm_loss, SPEC_LOSS: spec_loss} if return_components else loss

    def _post_process_output(
            self,
            base_model_output: bool = False,
            labels: Optional[torch.LongTensor] = None,
            cache: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
            hidden_states: Optional[List[torch.FloatTensor]] = None,
            attention_weights: Optional[List[torch.FloatTensor]] = None,
            return_dict: bool = True,
            guided_generation: bool = False,
            **kwargs
    ):
        base_model_output = base_model_output or self.is_benchmarking
        #
        if base_model_output:
            if hidden_states is not None:
                logger.warning(
                    'Note: the last tensor in the output `hidden_states` is the non-normalised tensor `last_hidden_state`.'
                )
            if return_dict:
                if isinstance(self.internal_model, GPT2PreTrainedModel):
                    return CausalLMOutputWithCrossAttentions(
                        loss=kwargs.get(self.lm_loss),
                        logits=kwargs[self.model_output][LOGITS],
                        past_key_values=cache,
                        hidden_states=hidden_states,
                        attentions=attention_weights
                    )
                elif isinstance(self.internal_model, SHARED_STRUCTURE_MODELS):
                    return CausalLMOutputWithPast(
                        loss=kwargs.get(self.lm_loss),
                        logits=kwargs[self.model_output][LOGITS],
                        past_key_values=cache,
                        hidden_states=hidden_states,
                        attentions=attention_weights
                    )
                else:
                    raise NotImplementedError(f'Unsupported model type: `{type(self.internal_model)}`.')
            else:
                return tuple(
                    v for v in [
                        kwargs.get(self.lm_loss),
                        kwargs[self.model_output][LOGITS],
                        # kwargs[self.model_output].get(SPECTROGRAMS),
                        cache,
                        hidden_states,
                        attention_weights
                    ] if v is not None
                )
        else:
            # Extract output
            model_output = kwargs.pop(self.model_output)
            logits = model_output.pop(LOGITS)
            spectrograms = model_output.pop(SPECTROGRAMS)
            # Compute loss
            loss, components = self._loss(
                token_logits=logits,
                token_labels=labels,
                predicted_spectrograms=spectrograms,
                target_spectrograms=kwargs.get(INPUT_SPECTROGRAMS)
            ) if labels is not None else None, None
            # Update output dict
            kwargs |= {
                LOGITS: logits,
                SPECTROGRAMS: spectrograms,
                LOSS: {
                    LOSS_VALUE: loss,
                    LOSS_COMPONENTS: components
                },
                CACHE: cache,
                HIDDEN_STATES: hidden_states,
                ATTN_WEIGHTS: attention_weights,
                RETURN_DICT: return_dict
            }

            return kwargs

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        #
        if not self.is_wrapping:
            return self.base_model.generate(*args, **kwargs)
        #
        generated_spectrograms = list()
        generate_output = PreTrainedModel.generate(self, *args, generated_spectrograms=generated_spectrograms, **kwargs)
        generated_spectrograms = torch.cat(generated_spectrograms, dim=-1)
        # Re-run through layers to collect all data  # TODO find better solution
        return_inner_states |= any(
            kwargs.get(k, False) for k in [
                ADD_ATTN_RESIDUAL,
                RETURN_ATTENTION_OUTPUT,
                RETURN_INTERMEDIATE_HIDDEN_STATES,
                ADD_FFNN_RESIDUAL,
                RETURN_FFNN_UP_PROJ_OUTPUT,
                RETURN_FFNN_GATE_OUTPUT,
                RETURN_FFNN_INNER_ACTIVATIONS,
                RETURN_FFNN_OUTPUT,
                BASE_MODEL_OUTPUT
            ]
        )
        #
        if self.lm_head_wrapper.post_net is not None:
            generated_spectrograms = generated_spectrograms + self.lm_head_wrapper.post_net(generated_spectrograms)
        #
        if return_inner_states or not self.is_benchmarking:
            #
            return self.forward(
                input_ids=generate_output, input_spectrograms=generated_spectrograms, **kwargs
            ) | {OUTPUT_IDS: generate_output, OUTPUT_SPECTROGRAMS: generated_spectrograms}
        else:
            return generate_output, generated_spectrograms

    def prepare_inputs_for_generation(
            self,
            *args,
            input_spectrograms: Optional[List[...]] = None,
            generated_spectrograms: Optional[List[...]] = None,
            **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        #
        if len(generated_spectrograms) > 0:
            input_spectrograms = generated_spectrograms[-1]
        #
        inputs |= {INPUT_SPECTROGRAMS: input_spectrograms, GENERATED_SPECTROGRAMS: generated_spectrograms}

        return inputs

    def post_process_spectrograms(
            self, spectrograms: torch.Tensor, token_ids: torch.Tensor
    ) -> Union[List[List[torch.Tensor]], List[torch.Tensor]]:
        #
        if len(spectrograms.size()) == 3:
            return [self.post_process_spectrograms(spec, ids) for spec, ids in zip(spectrograms, token_ids)]
        #
        mask = torch.repeat_interleave(token_ids == self.audio_token_id, self.speech_conversion_factor, dim=-1)
        s_idxs, = torch.where(mask & ~ F.pad(mask, (1, 0), value=False)[..., :-1])
        e_idxs, = torch.where(mask & ~ F.pad(mask, (0, 1), value=False)[..., 1:])

        return [
            spectrograms[:, s_idx:e_idx] for s_idx, e_idx in zip(s_idxs, e_idxs + 1)
        ]

    # Lightning

    def _pad_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        # TODO do padding replicating side slices
        # TODO add support for cases where there isn't a single convolution with hop length equal to window length
        n_elements = spec.size(-1)
        expected_n_elements = int(math.ceil(n_elements / self.speech_conversion_factor)) * self.speech_conversion_factor
        pad_left = int(math.ceil((expected_n_elements - n_elements) / 2))
        pad_right = (expected_n_elements - n_elements) // 2
        spec = F.pad(spec, (pad_left, pad_right), value=spec.min())

        return spec

    def prepare_input(
            self,
            text: Optional[Union[Iterable[str], str]],
            audio_file_paths: Optional[Union[Iterable[Iterable[str]], Iterable[str], str]] = None
    ) -> BatchEncoding:
        # TODO rework checks on input
        if isinstance(text, str):
            return self.prepare_input([text], audio_file_paths=audio_file_paths)
        #
        if audio_file_paths is not None:
            #
            if isinstance(audio_file_paths, str):
                return self.prepare_input(text, audio_file_paths=[[audio_file_paths]])
            elif all(isinstance(elem, str) for elem in audio_file_paths):
                return self.prepare_input(text, audio_file_paths=[audio_file_paths])
            #
            spectrograms = [
                [
                    self._pad_spectrogram(torch.tensor(self.audio_processor.encode(file_path)))
                    for file_path in file_paths
                ]
                for file_paths in audio_file_paths
            ]
            text = [
                head + str().join(
                    self.audio_token * (spec.size(-1) // self.speech_conversion_factor) + split
                    for spec, split in zip(sequence_spectrograms, splits)
                )
                for (head, *splits), sequence_spectrograms in zip(
                    (sequence_text.split(self.audio_token) for sequence_text in text), spectrograms
                )
            ]
        else:
            spectrograms = None
        #
        input_encodings = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)  # , add_special_tokens=False)
        if spectrograms is not None:
            audio_stream = torch.full(
                (
                    input_encodings.input_ids.size(0),
                    self.audio_processor.channels,
                    input_encodings.input_ids.size(1) * self.speech_conversion_factor
                ),
                torch.nan
            )
            audio_stream[
                torch.repeat_interleave(
                    input_encodings.input_ids == self.audio_token_id,
                    self.speech_conversion_factor,
                    dim=-1
                ).unsqueeze(1).repeat((1, self.audio_processor.channels, 1))
            ] = torch.hstack([spec for sequence_spectrograms in spectrograms for spec in sequence_spectrograms]).ravel()
            if audio_stream.size(-1) > input_encodings.input_ids.size(-1) * self.speech_conversion_factor:  # Apply truncation
                audio_stream = audio_stream[..., :input_encodings.input_ids.size(-1) * self.speech_conversion_factor]
            input_encodings[INPUT_SPECTROGRAMS] = audio_stream

        return input_encodings


    def prepare_output(
            self,
            text: Optional[Union[Iterable[str], str]] = None,
            audio_file_paths: Optional[Union[Iterable[Iterable[str]], Iterable[str], str]] = None,
            input_data: Optional[BatchEncoding] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        if input_data is None:
            return self.prepare_output(input_data=self.prepare_input(text, audio_file_paths))
        #
        output_ids = input_data.input_ids.clone()
        output_ids[input_data.attention_mask == 0] = -100
        if input_data.get(INPUT_SPECTROGRAMS) is not None:
            target_spectrogram = input_data.input_spectrograms.clone()
            target_spectrogram[
                ~torch.repeat_interleave(
                    output_ids == self.audio_token_id, self.speech_conversion_factor, dim=-1
                ).unsqueeze(1).repeat((1, self.audio_processor.channels, 1))
            ] = torch.nan
        else:
            target_spectrogram = None

        return {TOKEN_LABELS: output_ids, TARGET_SPECTROGRAMS: target_spectrogram}

    def collate(self, samples: Iterable[Dict]) -> Tuple[BatchEncoding, Dict[str, Optional[torch.Tensor]]]:
        input_encodings = self.prepare_input(
            [sample['text'] for sample in samples],
            [sample.get('audio_file_paths', list()) for sample in samples]
        )
        target_output = self.prepare_output(input_data=input_encodings)

        return input_encodings, target_output

    def configure_optimizers(self):
        # Build optimiser
        optimiser_params = self.optimiser_params.copy()
        optimiser_dtype = optimiser_params.pop('dtype')
        if isinstance(self.base_model, PeftModel):
            params = [
                p for k, p in self.named_parameters()
                if 'lora' in k or 'speech' in k or 'modality_switch' in k or 'post_net' in k
            ]
        else:
            params = self.parameters()
        optimiser = optimizer_mapping[optimiser_dtype](params, **optimiser_params)
        # Check whether LR scheduling is required
        if len(self.lr_scheduler_params) > 0:
            lr_scheduler_params = self.lr_scheduler_params.copy()
            lr_scheduler_dtype = lr_scheduler_params.pop('dtype')
            lr_scheduler_interval = lr_scheduler_params.pop('interval')
            if lr_scheduler_params == 'step':
                lr_scheduler_params['steps_per_epoch'] = int(math.ceil(self._steps_per_epoch))
            lr_scheduler = lr_scheduler_mapping[lr_scheduler_dtype](optimiser, **lr_scheduler_params)
            return [optimiser], [{'scheduler': lr_scheduler, 'interval': lr_scheduler_interval}]
        else:
            return optimiser

    def _step(
            self,
            split: str,
            mini_batch: Tuple[BatchEncoding, Dict[str, Optional[torch.Tensor]]],
            mini_batch_idx: int
    ) -> Tuple[Dict, torch.Tensor]:
        # Unpack the encoding and the target labels
        input_encodings, target_output = mini_batch
        # Compute output
        wrapper_output = self.forward(**input_encodings)
        # Compute LM loss token-wise
        loss, loss_components = self._loss(
            token_logits=wrapper_output[LOGITS],
            predicted_spectrograms=wrapper_output[SPECTROGRAMS],
            **target_output
        )

        # Log LM loss
        self.log(f'Loss/{split.capitalize()}', loss)
        for k, v in loss_components.items():
            self.log(f'{k.capitalize()}/{split.capitalize()}', v)

        return wrapper_output, loss

    def _eval_step(self, split: str, mini_batch, mini_batch_idx: int):
        # Unpack the encoding and the target labels
        input_encodings, target_output = mini_batch
        # Run generic forward step
        output, loss = self._step(split, mini_batch, mini_batch_idx)
        # Take logits
        logits: torch.tensor = output[LOGITS]
        # Shift logits to exclude the last element
        logits = logits[..., :-1, :].contiguous()
        # shift labels to exclude the first element
        labels = target_output[TOKEN_LABELS][..., 1:].contiguous()

        # Log Perplexity
        for metric_id, metric in self.metrics.items():
            if metric_id == 'Perplexity':
                metric.update(logits, labels)
            else:
                # TODO manage generative metrics
                pass

        return loss

    def fine_tune(
            self,
            data_splits: Dict[str, Dataset],
            *_,
            dir_path: Optional[str] = None,
            callbacks: Optional[Dict[str, pl_callbacks.Callback]] = None,
            loggers: Optional[Iterable[pl_loggers.Logger]] = None
    ) -> 'CausalLMWrapper':
        # Fit audio scaler
        # self.audio_processor.fit_scaler(data_splits['train'].get_audio_scaling_samples())
        logger.info("Audio scaler fitting completed")
        # Create data loaders
        data_loaders: Dict[str, DataLoader] = {
            split: DataLoader(
                data,
                collate_fn=self.collate,
                shuffle=split == 'train' and len(data) < 10000,
                # TODO find better solution to shuffling large data sets
                **self.data_loader_params[split]
            )
            for split, data in data_splits.items()
        }
        logger.info("Data loaders instantiated")
        #
        self._steps_per_epoch = len(data_loaders['train']) / self.trainer_params.get('accumulate_grad_batches', 1)
        # Create Trainer
        self.configure_metrics()
        self.disable_benchmarking()
        trainer = L.Trainer(
            default_root_dir=dir_path,
            **self.trainer_params,
            callbacks=list(callbacks.values()),
            logger=loggers
        )
        logger.info("Trainer instantiated")
        # Train neural network
        self.enable_wrapper()
        self.train()
        start_time = datetime.now()
        logger.info("Training started")
        trainer.fit(self, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'])
        stop_time = datetime.now()
        logger.info(f"Training completed (elapsed time: {stop_time - start_time})")
        # Load torch checkpoint
        if 'model_checkpoint' in callbacks and isinstance(callbacks['model_checkpoint'], pl_callbacks.ModelCheckpoint):
            if os.path.exists(callbacks['model_checkpoint'].best_model_path):
                checkpoint = torch.load(callbacks['model_checkpoint'].best_model_path)
                self.load_state_dict(checkpoint['state_dict'])
                logger.info(f"Best checkpoint restored from {callbacks['model_checkpoint'].best_model_path}")
            else:
                logger.info(f"No checkpoint to restore")
        # Test neural network
        start_time = datetime.now()
        logger.info("Validation started")
        trainer.validate(self, dataloaders=data_loaders['validation'])
        stop_time = datetime.now()
        logger.info(f"Validation completed (elapsed time: {stop_time - start_time})")
        start_time = datetime.now()
        logger.info("Testing started")
        trainer.test(self, dataloaders=data_loaders['test'])
        stop_time = datetime.now()
        logger.info(f"Testing completed (elapsed time: {stop_time - start_time})")

        return self
