# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TLS Spectroscopy experiment with T1 scan."""

from typing import Optional

import numpy as np
from qiskit import pulse
from qiskit.pulse.builder import macro
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.library import continuous
from qiskit.pulse.library.samplers.strategies import midpoint_sample


@macro
def play_chunked_gaussian_square(
    duration: int,
    amp: complex,
    sigma: float,
    risefall_sigma_ratio: float,
    channel: PulseChannel,
    name: Optional[str] = None,
    chunk_size: int = 128,
    min_duration: int = 10,
) -> int:
    if duration < chunk_size * min_duration:
        pulse.play(
            pulse.GaussianSquare(
                duration=duration,
                amp=amp,
                sigma=sigma,
                risefall_sigma_ratio=risefall_sigma_ratio,
                name=name,
            ),
            channel=channel,
        )
        return duration

    if duration % chunk_size != 0:
        duration = chunk_size * int(duration / chunk_size)

    edge_length = int(sigma * risefall_sigma_ratio)

    rise_edge = midpoint_sample(
        continuous_pulse=continuous.gaussian,
        duration=edge_length,
        amp=amp,
        center=edge_length,
        sigma=sigma,
        zeroed_width=2 * edge_length + 2,
        rescale_amp=True,
    )

    fall_edge = midpoint_sample(
        continuous_pulse=continuous.gaussian,
        duration=edge_length,
        amp=amp,
        center=0,
        sigma=sigma,
        zeroed_width=2 * edge_length + 2,
        rescale_amp=True,
    )

    surplus = edge_length % chunk_size
    if surplus > 0:
        extra_flat_top_length = chunk_size - surplus
        rise_edge_waveform = np.r_[rise_edge, np.full(extra_flat_top_length, amp)]
        fall_edge_waveform = np.r_[np.full(extra_flat_top_length, amp), fall_edge]
    else:
        rise_edge_waveform = rise_edge
        fall_edge_waveform = fall_edge

    remainder = duration - (rise_edge_waveform.size + fall_edge_waveform.size)
    n_chunks = int(remainder / chunk_size)

    with pulse.align_sequential():
        pulse.play(
            rise_edge_waveform,
            channel=channel,
            name=name+"_rise" if name else None,
        )
        for _ in range(n_chunks):
            pulse.play(
                pulse.Constant(duration=chunk_size, amp=amp),
                channel=channel,
                name="_mid" if name else None,
            )
        pulse.play(
            fall_edge_waveform,
            channel=channel,
            name="_fall" if name else None,
        )

    return duration
