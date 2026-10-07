"""Normal-incidence transfer matrix method for lossless dielectric films.

Each layer uses the characteristic matrix

    [[cos(delta), 1j * sin(delta) / n],
     [1j * n * sin(delta), cos(delta)]]

where delta = 2*pi*n*d/lambda. Matrices are multiplied from the incident
medium toward the substrate.
"""

from collections.abc import Sequence
import math

Matrix2x2 = tuple[
    tuple[complex, complex],
    tuple[complex, complex],
]


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validate_stack(
    wavelength_nm: float,
    layer_indices: Sequence[float],
    thicknesses_nm: Sequence[float],
    incident_index: float,
    substrate_index: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...], float, float]:
    wavelength = _positive_finite(wavelength_nm, "wavelength_nm")
    incident = _positive_finite(incident_index, "incident_index")
    substrate = _positive_finite(substrate_index, "substrate_index")

    indices = tuple(
        _positive_finite(value, f"layer_indices[{index}]")
        for index, value in enumerate(layer_indices)
    )
    thicknesses = tuple(float(value) for value in thicknesses_nm)
    if len(indices) != len(thicknesses):
        raise ValueError("layer_indices and thicknesses_nm must have the same length")
    for index, value in enumerate(thicknesses):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"thicknesses_nm[{index}] must be a non-negative finite number"
            )
    return wavelength, indices, thicknesses, incident, substrate


def _multiply(left: Matrix2x2, right: Matrix2x2) -> Matrix2x2:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def _characteristic_matrix(
    refractive_index: float,
    thickness_nm: float,
    wavelength_nm: float,
) -> Matrix2x2:
    phase = 2.0 * math.pi * refractive_index * thickness_nm / wavelength_nm
    cosine = math.cos(phase)
    sine = math.sin(phase)
    return (
        (complex(cosine), 1j * sine / refractive_index),
        (1j * refractive_index * sine, complex(cosine)),
    )


def _amplitudes(
    wavelength_nm: float,
    layer_indices: Sequence[float],
    thicknesses_nm: Sequence[float],
    incident_index: float,
    substrate_index: float,
) -> tuple[complex, complex, float, float]:
    wavelength, indices, thicknesses, incident, substrate = _validate_stack(
        wavelength_nm,
        layer_indices,
        thicknesses_nm,
        incident_index,
        substrate_index,
    )

    matrix: Matrix2x2 = ((1.0 + 0j, 0j), (0j, 1.0 + 0j))
    for refractive_index, thickness_nm in zip(indices, thicknesses, strict=True):
        matrix = _multiply(
            matrix,
            _characteristic_matrix(refractive_index, thickness_nm, wavelength),
        )

    a, b = matrix[0]
    c, d = matrix[1]
    denominator = incident * a + incident * substrate * b + c + substrate * d
    reflection = (
        incident * a + incident * substrate * b - c - substrate * d
    ) / denominator
    transmission = (2.0 * incident) / denominator
    return reflection, transmission, incident, substrate


def reflectance(
    wavelength_nm: float,
    layer_indices: Sequence[float],
    thicknesses_nm: Sequence[float],
    incident_index: float = 1.0,
    substrate_index: float = 1.52,
) -> float:
    """Return power reflectance R for one wavelength."""

    reflection, _, _, _ = _amplitudes(
        wavelength_nm,
        layer_indices,
        thicknesses_nm,
        incident_index,
        substrate_index,
    )
    return abs(reflection) ** 2


def transmittance(
    wavelength_nm: float,
    layer_indices: Sequence[float],
    thicknesses_nm: Sequence[float],
    incident_index: float = 1.0,
    substrate_index: float = 1.52,
) -> float:
    """Return power transmittance T for one wavelength."""

    _, transmission, incident, substrate = _amplitudes(
        wavelength_nm,
        layer_indices,
        thicknesses_nm,
        incident_index,
        substrate_index,
    )
    return (substrate / incident) * abs(transmission) ** 2


def reflectance_spectrum(
    wavelengths_nm: Sequence[float],
    layer_indices: Sequence[float],
    thicknesses_nm: Sequence[float],
    incident_index: float = 1.0,
    substrate_index: float = 1.52,
) -> tuple[float, ...]:
    """Return one reflectance value for every wavelength in input order."""

    return tuple(
        reflectance(
            wavelength_nm,
            layer_indices,
            thicknesses_nm,
            incident_index,
            substrate_index,
        )
        for wavelength_nm in wavelengths_nm
    )

