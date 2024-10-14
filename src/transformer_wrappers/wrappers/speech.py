import os
import inspect

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa

from transformers import logging
from transformers import PreTrainedModel, BatchEncoding
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
    CausalLMWrapper
)
from.base.dtypes import *
from .base.constants import *


__all__ = ['AudioProcessor', 'SpeechTransformerWrapper', 'SpeechCausalLMWrapper']

AUDIO_TOKEN: str = 'audio_token'

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
            spectrogram_embeddings = self.speech_encoder.forward(input_spectrograms)
            output[self.module_output][speech_mask] += spectrogram_embeddings.transpose(-1, -2)[speech_mask]
        #
        output |= {
            SPEECH_MASK: speech_mask
        }

        return output


class SpeechTransformerWrapper(TransformerWrapper):
    _embedding_dtype: Type[ModuleWrapper] = SpeechEmbeddingWrapper

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
        # TODO make speech embedding more configurable
        speech_encoder = torch.nn.Conv1d(
            audio_processor.channels,
            model.config.hidden_size,
            model.config.hidden_size // audio_processor.channels,
            stride=model.config.hidden_size // audio_processor.channels,
            # dtype=model.base_model.dtype
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
        return self.config.hidden_size // self.audio_processor.channels

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
        # TODO make speech embedding more configurable
        speech_encoder = torch.nn.Conv1d(
            audio_processor.channels,
            model.config.hidden_size,
            model.config.hidden_size // audio_processor.channels,
            stride=model.config.hidden_size // audio_processor.channels,
            # dtype=model.base_model.dtype
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE)
        ):
            speech_encoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechEmbeddingWrapper.SPEECH_ENCODER_FILE),
                weights_only=True
            ))
        speech_decoder = torch.nn.ConvTranspose1d(
            model.config.hidden_size,
            audio_processor.channels,
            model.config.hidden_size // audio_processor.channels,
            stride=model.config.hidden_size // audio_processor.channels,
            dtype=model.base_model.dtype
        )
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.SPEECH_DECODER_FILE)
        ):
            speech_decoder.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.SPEECH_DECODER_FILE),
                weights_only=True
            ))
        modality_switch = torch.nn.Linear(model.config.hidden_size, 1, dtype=model.base_model.dtype)
        if os.path.exists(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.MODALITY_SWITCH_FILE)
        ):
            modality_switch.load_state_dict(torch.load(
                os.path.join(pretrained_model_name_or_path, SpeechLMHeadWrapper.MODALITY_SWITCH_FILE),
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

    def _spectrogram_generation_loss(self, predicted: torch.Tensor, target: torch.Tensor):
        # Get valid output maks
        mask = ~target.isnan()
        # Shift predictions to exclude the last element
        predicted = predicted[mask[..., :-self.speech_conversion_factor]]
        # shift targets to exclude the first element
        target = target[mask[..., :-self.speech_conversion_factor]]
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
            generated_spectrograms: Optional[List[torch.Tensor]] = None,
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
            #
            if generated_spectrograms is not None:
                if len(generated_spectrograms) > 0:
                    generated_spectrograms.append(spectrograms)
                else:
                    generated_spectrograms = [
                        kwargs.get(INPUT_SPECTROGRAMS, torch.full_like(spectrograms, torch.nan)),
                        spectrograms[..., -self.speech_conversion_factor:]
                    ]

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
                RETURN_DICT: return_dict,
                GENERATED_SPECTROGRAMS: generated_spectrograms
            }

            return kwargs

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        # NOTE: this generate won't work with multi-sequence approaches like beam-search
        if not self.is_wrapping:
            return self.base_model.generate(*args, **kwargs)
        # Make generation stateful by creating containers for input and output
        generated_spectrograms = list()
        #
        generate_output = super(PreTrainedModel).generate(
            *args, generated_spectrograms=generated_spectrograms, **kwargs
        )
        #
        # TODO: handle guided generation vs normal generation case
        generated_spectrograms = torch.cat(generated_spectrograms, dim=1)
        # Re-run through layers to collect all data  # TODO find better solution
        if return_inner_states or not self.is_benchmarking:
            #
            return self.forward(
                input_ids=generate_output,
                input_spectrograms=generated_spectrograms,
                **{
                    k: kwargs.get(k) for k in
                    set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())
                    if k not in {'args', 'kwargs', 'self', 'base_model_output'}
                },
                return_dict=True,
                output_attentions=True,
                use_cache=True,
                output_hidden_states=True,
                return_attention_output=True,  # Self-attention layer output
                return_feed_forward_output=True,
                return_intermediate_hidden_states=True
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
        mask = torch.repeat_interleave(token_ids == self.audio_token_id, self.speech_conversion_factor)
        s_idxs, = torch.where(mask & ~ F.pad(mask, (1, 0), value=False)[..., :-1])
        e_idxs, = torch.where(mask & ~ F.pad(mask, (0, 1), value=False)[..., 1:])
        idxs = (e_idxs + 1 - s_idxs).cumsum(dim=0)

        return [
            spectrograms[:, s_idx:e_idx] for s_idx, e_idx in zip([0] + idxs[:-1].cpu().tolist(), idxs.cpu().tolist())
        ]

    # Lightning

    def _pad_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        # TODO do padding replicating side slices
        # TODO add support for cases where there isn't a single convolution with hop length equal to window length
        n_elements = spec.numel()
        expected_n_elements = int(math.ceil(n_elements / self.config.hidden_size)) * self.config.hidden_size
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
                    self.audio_token * (spec.numel() // self.base_model.config.hidden_size) + split
                    for spec, split in zip(sequence_spectrograms, splits)
                )
                for (head, *splits), sequence_spectrograms in zip(
                    (sequence_text.split(self.audio_token) for sequence_text in text), spectrograms
                )
            ]
        else:
            spectrograms = None
        #
        input_encodings = self.tokenizer(text, return_tensors='pt', padding=True)  # , add_special_tokens=False)
        if spectrograms is not None:
            input_encodings[INPUT_SPECTROGRAMS] = torch.full(
                (input_encodings.size(0), input_encodings.size(1) * self.speech_conversion_factor),
                torch.nan
            )
            input_encodings.input_spectrograms[
                torch.repeat_interleave(input_encodings.input_ids == self.audio_token_id, self.speech_conversion_factor)
            ] = torch.hstack([spec for sequence_spectrograms in spectrograms for spec in sequence_spectrograms])

        return input_encodings


    def prepare_output(
            self,
            text: Optional[Union[Iterable[str], str]] = None,
            audio_file_paths: Optional[Union[Iterable[Iterable[str]], Iterable[str], str]] = None,
            input_data: Optional[BatchEncoding] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if input_data is None:
            return self.prepare_output(input_data=self.prepare_input(text, audio_file_paths))
        #
        output_ids = input_data.input_ids.clone()
        output_ids[input_data.attention_mask == 0] = -100
        if input_data.get(INPUT_SPECTROGRAMS) is not None:
            target_spectrogram = input_data.input_spectrograms.clone()
            target_spectrogram[
                torch.repeat_interleave(output_ids == self.audio_token_id, self.speech_conversion_factor)
            ] = torch.nan

            return output_ids, target_spectrogram
        else:
            return output_ids

    def collate(self, samples: Iterable[Dict]) -> Tuple[
        BatchEncoding, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ]:
        input_encodings = self.prepare_input(
            [sample['text'] for sample in samples],
            [sample.get('audio_file_paths', list()) for sample in samples]
        )
        target_output = self.prepare_output(input_data=input_encodings)
        return input_encodings, target_output

    def _step(
            self,
            split: str,
            mini_batch: Tuple[BatchEncoding, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
            mini_batch_idx: int
    ) -> Tuple[Dict, torch.Tensor]:
        # Unpack the encoding and the target labels
        input_encodings, targets = mini_batch
        if isinstance(input_encodings, torch.Tensor):
            token_labels = targets
            target_spectrograms = None
        else:
            token_labels, target_spectrograms = targets
        # Compute output
        wrapper_output = self.forward(**input_encodings)
        # Compute LM loss token-wise
        loss, loss_components = self._loss(
            token_logits=wrapper_output[LOGITS],
            token_labels=token_labels,
            predicted_spectrograms=wrapper_output[SPECTROGRAMS],
            target_spectrograms=target_spectrograms
        )

        # Log LM loss
        self.log(f'Loss/{split.capitalize()}', loss)
        for k, v in loss_components.items():
            self.log(f'{k.split('_').capitalize()}/{split.capitalize()}', v)

        return wrapper_output, loss
