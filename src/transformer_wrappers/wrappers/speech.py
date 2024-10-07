import os
from copy import deepcopy

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MetricCollection

import librosa

from transformers import logging
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import BatchEncoding
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from transformers import BitsAndBytesConfig
from peft import LoraConfig

from typing import Type, Optional, Union, List, Iterable, Tuple, Dict

from .base import (
    SHARED_STRUCTURE_MODELS,
    ModuleWrapper,
    EmbeddingWrapper,
    TransformerWrapper,
    LMHeadWrapper,
    CausalLMWrapper,
    PreTrainedModelWrapper
)
from.base.dtypes import *
from .base.constants import *


__all__ = ['AudioProcessor', 'SpeechTransformerWrapper', 'SpeechCausalLMWrapper']

AUDIO_TOKEN: str = 'audio_token'

INPUT_SPECTROGRAMS: str = 'input_spectrograms'
SPEECH_MASK: str = 'speech_mask'

SPECTROGRAMS: str = 'spectrograms'
MODALITY_LOGITS: str = 'modality_logits'
LOSS_VALUE: str = 'loss_value'
LOSS_COMPONENTS: str = 'loss_components'

TOKEN_LABELS: str = 'token_labels'
TARGET_SPECTROGRAMS: str = 'target_spectrograms'
MODALITY_LABELS: str = 'modality_labels'

LM_LOSS: str = 'language_modelling_loss'
SPEC_LOSS: str = 'spectrogram_generation_loss'
MODALITY_LOSS: str = 'modality_prediction_loss'

SR: str = 'sr'
WIN_SIZE: str = 'win_size'
HOP_SIZE: str = 'hop_size'
N_FFT: str = 'n_fft'
N_MEL: str = 'n_mel'
N_MFCC: str = 'n_mfcc'

logger = logging.get_logger(__name__)


class AudioProcessor:
    def __init__(
            self,
            sr: int = 16000,
            win_size: float = 0.025,  # In seconds
            hop_size: Optional[float] = 0.01,  # In seconds, defaults to window size
            n_fft: int = 512,
            n_mel: Optional[int] = 128,  # Typical value is 80 if not None, change to match speech embeddings requirements
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



class SpeechEmbeddingWrapper(EmbeddingWrapper):
    SPEECH_ENCODER_FILE: str = 'speech_encoder.pth'

    def __init__(self, module: nn.Module, speech_encoder: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self._speech_encoder: nn.Module = speech_encoder

    @property
    def speech_encoder(self):
        return self._speech_encoder

    def _pad_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        # TODO do padding replicating side slices
        # TODO add support for cases where there isn't a single convolution with hop length equal to window length
        n_elements = spec.numel()
        expected_n_elements = int(math.ceil(n_elements / self.embedding_dim)) * self.embedding_dim
        pad_left = int(math.ceil((expected_n_elements - n_elements) / 2))
        pad_right = (expected_n_elements - n_elements) // 2
        spec = F.pad(spec, (pad_left, pad_right), value=spec.min())

        return spec

    def _pre_process_input(
            self, *args, input_spectrograms: Optional[Union[Iterable[torch.Tensor], torch.Tensor]] = None, **kwargs
    ):
        # Pad spectrograms to match required shape
        if input_spectrograms is not None:
            if isinstance(input_spectrograms, torch.Tensor):
                input_spectrograms = self._pad_spectrogram(input_spectrograms)[None, ...]
            else:
                input_spectrograms = [self._pad_spectrogram(spec)[None, ...] for spec in input_spectrograms]
        #
        kwargs |= {
            INPUT_SPECTROGRAMS: input_spectrograms
        }

        return kwargs

    def _wrapped_forward(
            self,
            *args,
            input_spectrograms: Optional[Union[Iterable[torch.Tensor], torch.Tensor]] = None,
            speech_mask: Optional[torch.BoolTensor] = None,
            **kwargs
    ):
        # Run base forward
        output = super()._wrapped_forward(*args, **kwargs)
        # Check whether there are spectrograms to embed
        if input_spectrograms is not None:
            #
            if isinstance(input_spectrograms, torch.Tensor):
                spectrogram_embedding = self.speech_encoder.forward(input_spectrograms)
            else:
                spectrogram_embedding = torch.cat(
                    [self.speech_encoder.forward(spec) for spec in input_spectrograms], dim=0
                )
            #
            output[self.module_output][speech_mask] = spectrogram_embedding.T
        #
        output |= {
            SPEECH_MASK: speech_mask
        }

        return output


class SpeechTransformerWrapper(TransformerWrapper):
    _embedding_dtype: Type[ModuleWrapper] = SpeechEmbeddingWrapper

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            audio_processor: AudioProcessor,
            speech_encoder: nn.Module,
            *args,
            **kwargs
    ):
        super(PreTrainedModelWrapper).__init__(model, tokenizer, *args, **kwargs)
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
        self._audio_token: str = self.config.task_specific_params[self.WRAPPER_CONFIGS_KEY].get(AUDIO_TOKEN, '<|audio|>')

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
        # TODO possibly extend model vocabulary to include also special audio token

        audio_processor = AudioProcessor(
            sr=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(SR, 16000),
            win_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(WIN_SIZE, 0.025),
            hop_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(HOP_SIZE, 0.01),
            n_fft=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_FFT, 512),
            n_mel=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MEL, 128),
            n_mfcc=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MFCC)
        )
        # TODO make speech embedding more configurable
        speech_encoder = torch.nn.Sequential(
            (torch.nn.RMSNorm if isinstance(model, SHARED_STRUCTURE_MODELS) else torch.nn.LayerNorm)(
                audio_processor.channels,
                eps=model.config.rms_norm_eps if isinstance(
                    model, SHARED_STRUCTURE_MODELS
                ) else model.config.layer_norm_epsilon
            ),
            torch.nn.Conv1d(
                audio_processor.channels,
                model.config.hidden_size,
                model.config.hidden_size // audio_processor.channels,
                stride=model.config.hidden_size // audio_processor.channels
            )
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


class SpeechLMHeadWrapper(LMHeadWrapper):
    SPEECH_DECODER_FILE: str = 'speech_decoder.pth'
    MODALITY_SWITCH_FILE: str = 'modality_switch.pth'

    def __init__(self, module: nn.Module, speech_decoder: nn.Module, modality_switch: nn.Module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        #
        self._speech_decoder: nn.Module = speech_decoder
        self._modality_switch: nn.Module = modality_switch

    @property
    def speech_decoder(self) -> nn.Module:
        return self._speech_decoder

    @property
    def modality_switch(self):
        return self._modality_switch

    @staticmethod
    def _get_spectrogram_splits(mask: torch.BoolTensor):
        # TODO: check this method
        _, s_idxs = torch.where(mask & ~ F.pad(mask, (1, 0), value=False)[..., :-1])
        _, e_idxs = torch.where(mask & ~ F.pad(mask, (0, 1), value=False)[..., 1:])
        idxs = (e_idxs + 1 - s_idxs).cumsum(dim=0)

        return [*zip([0] + idxs[:-1].cpu().tolist(), idxs.cpu().tolist())]

    def _wrapped_forward(
            self,
            output_hidden_state: Optional[torch.tensor] = None,
            speech_mask: Optional[torch.BoolTensor] = None,
            generating: bool = False,
            **kwargs
    ):
        if output_hidden_state is None:
            raise ValueError()
        #
        logits = self.base_module.forward(output_hidden_state)
        modality_logits = self.modality_switch.forward(output_hidden_state)
        modality_mask = (modality_logits > 0.0).view(output_hidden_state.size()[:-1])
        mask = F.pad(speech_mask, (0, 1), value=False)[..., 1:] if speech_mask is not None else modality_mask
        if mask is not None and torch.any(mask):
            spectrograms = [
                self.speech_decoder(output_hidden_state[mask][s_idx:e_idx].T)
                for s_idx, e_idx in self._get_spectrogram_splits(mask)
            ]
        else:
            spectrograms = None
        #
        output = kwargs | {
            self.module_output: {
                LOGITS: logits,
                SPECTROGRAMS: spectrograms,
                MODALITY_LOGITS: modality_logits
            },
            OUT_HIDDEN_STATE: output_hidden_state
        }

        return output


class SpeechCausalLMWrapper(CausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = SpeechTransformerWrapper
    _lm_head_dtype: Type[ModuleWrapper] = SpeechLMHeadWrapper

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            audio_processor: AudioProcessor,
            speech_encoder: nn.Module,
            speech_decoder: nn.Module,
            modality_switch: nn.Module,
            *args,
            **kwargs
    ):
        super(PreTrainedModelWrapper).__init__(model, tokenizer, *args, **kwargs)  # TODO fixme
        super(L.LightningModule).__init__(model, tokenizer, *args, **kwargs)  # TODO fixme
        # Attribute names
        self._transformer_attr: LMTransformerAttr = self._get_transformer_attr()
        self._lm_head_attr: LMHeadAttr = self._get_lm_head_attr()
        # Wrappers
        self._transformer_wrapper: Tuple[TransformerWrapper] = self._transformer_dtype(
            getattr(self.internal_model, self._transformer_attr.value),
            self._tokenizer,
            audio_processor,
            speech_encoder
        ),
        self._lm_head_wrapper: Tuple[LMHeadWrapper] = self._lm_head_dtype(
            getattr(self.internal_model, self._lm_head_attr.value),
            speech_decoder,
            modality_switch,
            super_wrapper=self
        ),

        # Lightning module parameters for fine-tuning
        self.optimiser_params: Dict = dict()
        self.lr_scheduler_params: Dict = dict()
        self.trainer_params: Dict = dict()
        self.data_loader_params: Dict = dict()
        self.metrics: Optional[MetricCollection] = None
        self._steps_per_epoch: Optional[int] = None

    @property
    def audio_processor(self):
        return self.transformer_wrapper.audio_processor

    @property
    def audio_token(self) -> str:
        return self.transformer_wrapper.audio_token

    @property
    def audio_token_id(self) -> int:
        return self.transformer_wrapper.audio_token_id

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
        # TODO possibly extend model vocabulary to include also special audio token

        audio_processor = AudioProcessor(
            sr=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(SR, 16000),
            win_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(WIN_SIZE, 0.025),
            hop_size=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(HOP_SIZE, 0.01),
            n_fft=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_FFT, 512),
            n_mel=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MEL, 128),
            n_mfcc=model.config.task_specific_params[cls.WRAPPER_CONFIGS_KEY].get(N_MFCC)
        )
        # TODO make speech embedding more configurable
        speech_encoder = torch.nn.Sequential(
            (torch.nn.RMSNorm if isinstance(model, SHARED_STRUCTURE_MODELS) else torch.nn.LayerNorm)(
                audio_processor.channels,
                eps=model.config.rms_norm_eps if isinstance(
                    model, SHARED_STRUCTURE_MODELS
                ) else model.config.layer_norm_epsilon
            ),
            torch.nn.Conv1d(
                audio_processor.channels,
                model.config.hidden_size,
                model.config.hidden_size // audio_processor.channels,
                stride=model.config.hidden_size // audio_processor.channels
            )
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE)
        ):
            speech_encoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE),
                weights_only=True
            ))
        speech_decoder = torch.nn.Sequential(
            (torch.nn.RMSNorm if isinstance(model, SHARED_STRUCTURE_MODELS) else torch.nn.LayerNorm)(
                audio_processor.channels,
                eps=model.config.rms_norm_eps if isinstance(
                    model, SHARED_STRUCTURE_MODELS
                ) else model.config.layer_norm_epsilon
            ),
            torch.nn.ConvTranspose1d(
                model.config.hidden_size,
                audio_processor.channels,
                model.config.hidden_size // audio_processor.channels,
                stride=model.config.hidden_size // audio_processor.channels
            )
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_DECODER_FILE)
        ):
            speech_decoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_DECODER_FILE),
                weights_only=True
            ))
        modality_switch = torch.nn.Linear(model.config.hidden_size, 1)
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.MODALITY_DECODER_FILE)
        ):
            modality_switch.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.MODALITY_DECODER_FILE),
                weights_only=True
            ))

        wrapper = cls(model, tokenizer, audio_processor, speech_encoder, speech_decoder, modality_switch)

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

    @staticmethod
    def _spectrogram_generation_loss(spectrograms: List[torch.Tensor], targets: List[torch.Tensor]):
        return torch.cat(
            [(target - spectrogram).norm().view(1) for spectrogram, target in zip(spectrograms, targets)]
        ).mean()

    @staticmethod
    def _modality_prediction_loss(modality_mask: torch.Tensor, speech_mask: torch.Tensor):
        return F.binary_cross_entropy_with_logits(modality_mask, speech_mask)

    @staticmethod
    def _loss(
            token_logits: torch.Tensor,
            token_labels: torch.Tensor,
            spectrograms: Optional[List[torch.Tensor]] = None,
            target_spectrograms: Optional[List[torch.Tensor]] = None,
            modality_logits: Optional[torch.Tensor] = None,
            modality_labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_components: bool = True
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        # LM loss
        lm_loss = super()._loss(token_logits, token_labels)
        # Spectrogram generation loss
        spec_loss = SpeechCausalLMWrapper._spectrogram_generation_loss(
            spectrograms, target_spectrograms
        ) if spectrograms is not None and target_spectrograms is not None else 0.0
        # Modality prediction loss
        modality_loss = SpeechCausalLMWrapper._modality_prediction_loss(
            modality_logits if attention_mask is None else modality_logits[attention_mask.bool()].view(-1),
            modality_labels if attention_mask is not None else modality_labels.to(modality_logits)[attention_mask.bool()].view(-1)
        ) if modality_logits is not None and modality_labels is not None else 0.0
        # Total loss
        loss = lm_loss + spec_loss + modality_loss

        return loss, {
            LM_LOSS: lm_loss, SPEC_LOSS: spec_loss, MODALITY_LOSS: modality_loss
        } if return_components else loss

    def _post_process_output(
            self,
            base_model_output: bool = False,
            labels: Optional[torch.LongTensor] = None,
            target_spectrograms: Optional[List[torch.Tensor]] = None,
            speech_mask: Optional[torch.BoolTensor] = None,
            cache: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
            hidden_states: Optional[List[torch.FloatTensor]] = None,
            attention_weights: Optional[List[torch.FloatTensor]] = None,
            return_dict: bool = True,
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
                        kwargs.get(self.lm_loss), kwargs[self.model_output], cache, hidden_states, attention_weights
                    ] if v is not None
                )
        else:
            #
            model_output = kwargs.pop(self.model_output)
            logits = model_output.pop(LOGITS)
            spectrograms = model_output.pop(SPECTROGRAMS)
            modality_logits = model_output.pop(MODALITY_LOGITS)
            loss, components = self._loss(
                logits, labels, spectrograms,
            ) if labels is not None else None
            #
            kwargs |= {
                LOGITS: logits,
                SPECTROGRAMS: spectrograms,
                MODALITY_LOGITS: modality_logits,
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

    # Lightning

    def prepare_input(
            self, text: Iterable[str], audio_file_paths: Optional[Iterable[Iterable[str]]] = None
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, Optional]]:
        if audio_file_paths is not None:
            spectrograms = [
                [torch.tensor(self.audio_processor.encode(file_path)) for file_path in file_paths]
                for file_paths in audio_file_paths
            ]
            text = [
                head + sum(
                    self.audio_token * int(math.ceil(spec.numel() / self.base_model.config.hidden_size)) + split
                    for spec, split in zip(spectrograms_, splits)
                )
                for (head, *splits), spectrograms_ in zip(
                    (text_.split(self.audio_token) for text_ in text), spectrograms
                )
            ]
            spectrograms = [spec for spectrograms_ in spectrograms for spec in spectrograms_]
        else:
            spectrograms = None
        #
        input_encodings = self.tokenizer(text, return_tensors='pt', padding=True)  # , add_special_tokens=False)
        speech_mask = input_encodings.input_ids == self.audio_token_id if spectrograms is not None else None

        return input_encodings.data | {INPUT_SPECTROGRAMS: spectrograms, SPEECH_MASK: speech_mask}


    def prepare_output(
            self,
            text: Optional[Iterable[str]] = None,
            audio_file_paths: Optional[Iterable[Iterable[str]]] = None,
            input_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, Optional]]:
        if input_data is None:
            if audio_file_paths is not None:
                spectrograms = [
                    [torch.tensor(self.audio_processor.encode(file_path)) for file_path in file_paths]
                    for file_paths in audio_file_paths
                ]
                text = [
                    head + sum(
                        self.audio_token * int(math.ceil(spec.numel() / self.base_model.config.hidden_size)) + split
                        for spec, split in zip(spectrograms_, splits)
                    )
                    for (head, *splits), spectrograms_ in zip(
                        (text_.split(self.audio_token) for text_ in text), spectrograms
                    )
                ]
                spectrograms = [spec for spectrograms_ in spectrograms for spec in spectrograms_]
            else:
                spectrograms = None
            #
            input_encodings = self.tokenizer(text, return_tensors='pt', padding=True)  # , add_special_tokens=False)
            speech_mask = input_encodings.input_ids == self.audio_token_id if spectrograms is not None else None

            output_ids = input_encodings.input_ids
            output_ids[~input_encodings.attention_mask.bool()] = -100

            return {TOKEN_LABELS: output_ids, TARGET_SPECTROGRAMS: spectrograms, MODALITY_LABELS: speech_mask}
        else:
            output_ids = torch.clone(input_data[INPUT_IDS])
            if input_data.get(SPEECH_MASK) is not None:
                output_ids[~input_data[ATTENTION_MASK].bool() & ~input_data[SPEECH_MASK]] = -100

            return {
                TOKEN_LABELS: output_ids,
                TARGET_SPECTROGRAMS: input_data.get(SPECTROGRAMS),
                MODALITY_LABELS: input_data.get(SPEECH_MASK)
            }

    def collate(self, samples: Iterable[Dict]) -> Tuple[
        Dict[str, Union[List[torch.Tensor], torch.Tensor, Optional]],
        Dict[str, Union[List[torch.Tensor], torch.Tensor, Optional]]
    ]:
        input_encodings = self.prepare_input(
            [sample['text'] for sample in samples], [sample['audio_file_paths'] for sample in samples]
        )
        target_output = self.prepare_output()
        return input_encodings, target_output

    def _step(
            self,
            split: str,
            mini_batch: Tuple[
                Dict[str, Union[List[torch.Tensor], torch.Tensor, Optional]],
                Dict[str, Union[List[torch.Tensor], torch.Tensor, Optional]]
            ],
            mini_batch_idx: int
    ) -> Tuple[Dict, torch.Tensor]:
        # Unpack the encoding and the target labels
        input_encodings, labels = mini_batch
        # Compute output
        wrapper_output = self.forward(**input_encodings)
        # Compute LM loss token-wise
        loss, loss_components = self._loss(
            token_logits=wrapper_output[LOGITS],
            spectrograms=wrapper_output[SPECTROGRAMS],
            modality_logits=wrapper_output[MODALITY_LOGITS],
            **labels,
            attention_mask=input_encodings[ATTENTION_MASK]
        )

        # Log LM loss
        self.log(f'Loss/{split.capitalize()}', loss)
        for k, v in loss_components.items():
            self.log(f'{k.split('_').capitalize()}/{split.capitalize()}', v)

        return wrapper_output, loss