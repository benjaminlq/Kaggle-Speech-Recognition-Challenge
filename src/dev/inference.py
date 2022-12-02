"""Decoder Module
"""
from pathlib import Path
from typing import Optional, Union

import torch
from torchaudio.models.decoder import ctc_decoder

import config
from dev.utils import find_best_label


class GreedyCTCDecoder:
    """Module for CTC Greedy Decoder"""

    def __init__(
        self,
        decode_dict: dict = config.ID2CHAR,
        blank_idx: int = 0,
        silence_idx: int = 1,
    ):
        """CTC Greedy Decoder. Tokens are selected independently at each time steps and combine together for predictions.

        Args:
            decode_dict (dict, optional): Decode Dictionary to convert character to index.
            blank_idx (int, optional): Index of blank token in token sequence. Defaults to 0.
            silence_idx (int, optional): Index of silence token in token sequence. Defaults to 1.
        """
        self.blank_idx = blank_idx
        self.silence_idx = silence_idx
        self.decode_dict = decode_dict
        self.blank_token = self.decode_dict[self.blank_idx]
        self.silence_token = self.decode_dict[self.silence_idx]

    def decode(self, emissions: torch.tensor):
        """Decode Sequence of output tokens based on matrix of log probabilities

        Args:
            emissions (torch.tensor): log probabilities of all tokens at each timestamp

        Returns:
            tuple: sequence of predicted idx, full token sequence, pred token sequence
        """
        ## Emission: (bs, seq_length, no_tokens)
        preds_id_full = torch.argmax(emissions, dim=2)  # (bs, seq_length)

        preds_ids = []
        preds_full = []
        preds = []

        for pred_id in preds_id_full:
            pred_collapsed = torch.unique_consecutive(pred_id).detach().cpu().numpy()
            preds_ids.append(pred_collapsed)
            pred_sequence_full = "".join(
                [self.decode_dict[idx.item()] for idx in pred_id]
            )
            preds_full.append(pred_sequence_full)
            pred_sequence = "".join(
                [
                    self.decode_dict[idx]
                    for idx in pred_collapsed
                    if self.decode_dict[idx]
                    not in [self.blank_token, self.silence_token]
                ]
            )
            preds.append(find_best_label(pred_sequence))
        return preds_ids, preds_full, preds


class BeamSearchCTCDecoder:
    """Module for BeamSearch Decoder"""

    def __init__(
        self,
        decode_dict: dict = config.ID2CHAR,
        lexicon_path: Optional[Union[str, Path]] = str(
            config.MODEL_PATH / "meta" / "lexicon.txt"
        ),
        beam_size: int = 50,
        blank_idx: int = 0,
        silence_idx: int = 1,
    ):
        """BeamSearch CTC Decoder. Tokens are selected independently at each time steps and combine together for predictions.

        Args:
            decode_dict (dict, optional): Decode Dictionary to convert character to index.
            lexicon_path (Optional[Union[str, Path]], optional): Path to lexicon dictionary.
            beam_size (int, optional): Number of sequences to keep at each step. Defaults to 50.
            blank_idx (int, optional): Index of blank token in token sequence. Defaults to 0.
            silence_idx (int, optional): Index of silence token in token sequence. Defaults to 1.
        """
        self.blank_idx = blank_idx
        self.silence_idx = silence_idx
        self.decode_dict = decode_dict
        self.tokens = config.CHAR_LIST
        self.blank_token = self.decode_dict[self.blank_idx]
        self.silence_token = self.decode_dict[self.silence_idx]

        self.decoder = ctc_decoder(
            lexicon=lexicon_path,
            tokens=self.tokens,
            beam_size=beam_size,
            blank_token=self.blank_token,
        )
        # self.decode_dict[len(self.tokens)-1] = "|"

    def decode(self, emissions: torch.tensor):
        """Decode Sequence of output tokens based on matrix of log probabilities

        Args:
            emissions (torch.tensor): log probabilities of all tokens at each timestamp

        Returns:
            tuple: sequence of predicted idx, full token sequence, pred token sequence
        """
        ## Emission = (bs, seq_len, token_number)
        _, seq_len, _ = emissions.shape
        hypotheses = self.decoder(emissions)

        preds_ids = []
        preds_full = []
        preds = []

        for hypothesis in hypotheses:
            best_beam = hypothesis[0]
            pred_collapsed = best_beam.tokens.detach().cpu().numpy()
            pred_collapsed = pred_collapsed[1 : (len(pred_collapsed) - 1)]
            preds_ids.append(pred_collapsed)
            full_token_list = [self.blank_token] * seq_len
            for idx in range(1, len(best_beam.timesteps) - 1):
                full_token_list[best_beam.timesteps[idx].item() - 1] = self.decode_dict[
                    best_beam.tokens[idx].item()
                ]
            preds_full.append("".join(full_token_list))
            pred_sequence = "".join(
                [
                    self.decode_dict[idx]
                    for idx in pred_collapsed
                    if self.decode_dict[idx] != self.silence_token
                ]
            )
            preds.append(find_best_label(pred_sequence))

        return preds_ids, preds_full, preds


if __name__ == "__main__":
    # greedy_decoder = GreedyCTCDecoder()
    # probs = torch.rand(5, 75, len(greedy_decoder.decode_dict))
    # log_probs = torch.softmax(probs, dim = 2)
    # print(greedy_decoder.decode(log_probs))

    beam_decoder = BeamSearchCTCDecoder()
    probs = torch.rand(5, 75, len(beam_decoder.decode_dict))
    log_probs = torch.softmax(probs, dim=2)
    print(beam_decoder.decode(log_probs))
