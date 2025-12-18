# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible penalties and temperature using PyTorch instead of Triton."""

import torch
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata


def apply_penalties_and_temperature(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> None:
    """PyTorch implementation of apply_penalties_and_temperature.

    Applies repetition, frequency, and presence penalties, as well as temperature
    scaling to the logits tensor.
    """
    num_reqs = logits.shape[0]

    for batch_idx in range(num_reqs):
        rep_penalty = sampling_metadata.repetition_penalty[batch_idx].item()
        freq_penalty = sampling_metadata.frequency_penalty[batch_idx].item()
        pres_penalty = sampling_metadata.presence_penalty[batch_idx].item()
        temperature = sampling_metadata.temperature[batch_idx].item()

        if temperature == 0.0:
            temperature = 1.0

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
        use_temperature = temperature != 1.0

        if not (use_penalty or use_temperature):
            continue

        # Get logits for this request
        req_logits = logits[batch_idx]

        if use_penalty:
            req_state_idx = int(sampling_metadata.idx_mapping[batch_idx].item())
            output_bin_counts = sampling_metadata.output_bin_counts[req_state_idx]
            output_bin_mask = output_bin_counts > 0

            # Apply repetition penalty
            if use_rep_penalty:
                prompt_bin_mask = sampling_metadata.prompt_bin_mask[req_state_idx]
                # Unpack the bitmask
                vocab_size = logits.shape[1]
                unpacked_mask = torch.zeros(
                    vocab_size, dtype=torch.bool, device=logits.device
                )

                # Unpack bits from the packed mask
                num_packed = prompt_bin_mask.shape[0]
                for i in range(num_packed):
                    packed_val = int(prompt_bin_mask[i].item())
                    for bit in range(32):
                        token_idx = i * 32 + bit
                        if token_idx < vocab_size:
                            if (packed_val >> bit) & 1:
                                unpacked_mask[token_idx] = True

                # Combine prompt and output masks
                combined_mask = unpacked_mask | output_bin_mask[:vocab_size].bool()

                # Apply repetition penalty
                # If logits are positive, divide by penalty; otherwise multiply by penalty
                positive_mask = req_logits > 0
                penalty_scale = torch.where(
                    positive_mask, 1.0 / rep_penalty, rep_penalty
                )
                penalty_scale = torch.where(
                    combined_mask, penalty_scale, torch.ones_like(penalty_scale)
                )
                req_logits.mul_(penalty_scale)

            # Apply frequency penalty
            if use_freq_penalty:
                req_logits.sub_(
                    freq_penalty * output_bin_counts[: req_logits.shape[0]].float()
                )

            # Apply presence penalty
            if use_pres_penalty:
                req_logits.sub_(
                    pres_penalty * output_bin_mask[: req_logits.shape[0]].float()
                )

        # Apply temperature
        if use_temperature:
            req_logits.div_(temperature)
