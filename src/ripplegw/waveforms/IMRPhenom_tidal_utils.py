"""
Small utility script for shared functions between tidal waveforms, especially for NRTidalv2
"""

import jax
import jax.numpy as jnp
from ..typing import Array
from ..constants import gt


def universal_relation(coeffs: Array, x: float):
    """
    Applies the general formula of a universal relationship, which is a quartic polynomial.

    Args:
        coeffs (Array): Array of coefficients for the quartic polynomial, starting from the constant term and going to the fourth order.
        x (float): Variable of quartic polynomial

    Returns:
        float: Result of universal relation
    """
    return (
        coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * (x**2)
        + coeffs[3] * (x**3)
        + coeffs[4] * (x**4)
    )


def get_quadparam_octparam(lambda_: float) -> tuple[float, float]:
    """
    Compute the quadrupole and octupole parameter by checking the value of lambda and choosing the right subroutine.
    If lambda is smaller than 1, we make use of the fit formula as given by the LAL source code. Otherwise, we rely on the equations of
    the NRTidalv2 paper to get these parameters.

    Args:
        lambda_ (float): Tidal deformability of object.

    Returns:
        tuple[float, float]: Quadrupole and octupole parameters.
    """

    # Check if lambda is low or not, and choose right subroutine
    is_low_lambda = lambda_ < 1
    return jax.lax.cond(
        is_low_lambda,
        _get_quadparam_octparam_low,
        _get_quadparam_octparam_high,
        lambda_,
    )


def _get_quadparam_octparam_low(lambda_: float) -> tuple[float, float]:
    """
    Computes quadparameter following LALSimUniversalRelations.c of lalsuite

    Version for lambdas smaller than 1.

    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """

    # Coefficients of universal relation
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]

    # Extension of the fit in the range lambda2 = [0,1.] so that the BH limit is enforced, lambda2bar->0 gives quadparam->1. and the junction with the universal relation is smooth, of class C2
    quadparam = 1.0 + lambda_ * (
        0.427688866723244
        + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572)
    )
    log_quadparam = jnp.log(quadparam)

    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam


def _get_quadparam_octparam_high(lambda_: float) -> tuple[float, float]:
    """
    Computes quadparameter, following LALSimUniversalRelations.c of lalsuite

    Version for lambdas greater than 1.

    LALsuite has an extension where a separate formula is used for lambdas smaller than one, and another formula is used for lambdas larger than one.
    Args:
        lambda_: tidal deformability

    Returns:
        quadparam: Quadrupole coefficient called C_Q in NRTidalv2 paper
        octparam: Octupole coefficient called C_Oc in NRTidalv2 paper
    """

    # Coefficients of universal relation
    quad_coeffs = [0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4]
    oct_coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]

    # High lambda (above 1): use universal relation
    log_lambda = jnp.log(lambda_)
    log_quadparam = universal_relation(quad_coeffs, log_lambda)

    # Compute octparam:
    log_octparam = universal_relation(oct_coeffs, log_quadparam)

    quadparam = jnp.exp(log_quadparam)
    octparam = jnp.exp(log_octparam)

    return quadparam, octparam


def get_kappa(theta: Array) -> float:
    """
    Computes the tidal deformability parameter kappa according to equation (8) of the NRTidalv2 paper.

    Args:
        theta (Array): Intrinsic parameters m1, m2, chi1, chi2, lambda1, lambda2

    Returns:
        float: kappa_eff^T from equation (8) of NRTidalv2 paper.
    """

    # Auxiliary variables
    m1, m2, _, _, lambda1, lambda2 = theta
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M

    # Get kappa
    term1 = (1.0 + 12.0 * X2 / X1) * (X1**5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2**5.0) * lambda2
    kappa = (3.0 / 13.0) * (term1 + term2)

    return kappa


def get_merger_frequency(kappa, total_mass, q):
    # copied from lalsimulation.LALSimNRTunedTides.c
    # Constants
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4

    # Calculate kappa squared
    kappa2 = kappa * kappa

    # Numerator and denominator
    num = 1.0 + n_1 * kappa + n_2 * kappa2
    den = 1.0 + d_1 * kappa + d_2 * kappa2

    # Q_0
    Q_0 = a_0 / jnp.sqrt(q)

    # Dimensionless angular frequency of merger
    Momega_merger = Q_0 * (num / den)

    # Convert to frequency in Hz
    fHz_merger = Momega_merger / (jnp.pi * 2.0 * total_mass * gt)

    return fHz_merger


def get_tidal_phase(fHz, Xa, Xb, total_mass, kappa):
    # Constants
    c_Newt = 2.4375
    n_1 = -17.428
    n_3over2 = 31.867
    n_2 = -26.414
    n_5over2 = 62.362
    d_1 = n_1 - 2.496
    d_3over2 = 36.089

    # Dimensionless angular GW frequency
    M_omega = jnp.pi * fHz * (total_mass * gt)

    # Calculating powers of the frequency term
    PN_x = jnp.power(M_omega, 2.0 / 3.0)
    PN_x_3over2 = jnp.power(PN_x, 3.0 / 2.0)
    PN_x_5over2 = jnp.power(PN_x, 5.0 / 2.0)

    # Tidal phase calculation
    tidal_phase = -kappa * c_Newt / (Xa * Xb) * PN_x_5over2

    # Numerator and denominator for ratio
    num = (
        1.0
        + (n_1 * PN_x)
        + (n_3over2 * PN_x_3over2)
        + (n_2 * PN_x * PN_x)
        + (n_5over2 * PN_x_5over2)
    )
    den = 1.0 + (d_1 * PN_x) + (d_3over2 * PN_x_3over2)

    # Ratio
    ratio = num / den

    # Final tidal phase calculation
    tidal_phase *= ratio

    return tidal_phase


def get_planck_taper(t, t1, t2):
    taper = jnp.where(
        t <= t1,
        0.0,
        jnp.where(
            t >= t2,
            1.0,
            1.0 / (jnp.exp((t2 - t1) / (t - t1) + (t2 - t1) / (t - t2)) + 1.0),
        ),
    )
    return taper


def get_nr_tuned_tidal_phase_taper(fHz, m1, m2, lambda1, lambda2):
    """Note m1 and m2 here are in solar masses"""
    total_mass = m1 + m2
    q = m1 / m2

    # Xa and Xb are the masses normalized for total_mass = 1
    Xa = m1 / total_mass
    Xb = m2 / total_mass

    kappa = get_kappa([m1, m2, lambda1, lambda2])
    fHz_mrg = get_merger_frequency(kappa, total_mass, q)
    fHz_end_taper = 1.2 * fHz_mrg

    phi_tidal = get_tidal_phase(fHz, Xa, Xb, total_mass, kappa)
    planck_taper = 1.0 - get_planck_taper(fHz, fHz_mrg, fHz_end_taper)

    return phi_tidal, planck_taper
