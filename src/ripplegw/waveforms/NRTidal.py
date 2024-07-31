import jax
import jax.numpy as jnp

from ..constants import gt

# copying over functions into JAX
# copied from lalsimulation.LALSimNRTunedTides.c


def get_kappa2T(m1, m2, lambda1, lambda2):
    # Total mass
    total_mass = m1 + m2

    # Normalized masses
    Xa = m1 / total_mass
    Xb = m2 / total_mass

    # Calculate tidal coupling constant
    term1 = (1.0 + 12.0 * Xb / Xa) * (Xa**5.0) * lambda1
    term2 = (1.0 + 12.0 * Xa / Xb) * (Xb**5.0) * lambda2
    kappa2T = (3.0 / 13.0) * (term1 + term2)

    return kappa2T


def get_merger_frequency(kappa2T, total_mass, q):
    # Constants
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4

    # Calculate kappa2T squared
    kappa2T2 = kappa2T * kappa2T

    # Numerator and denominator
    num = 1.0 + n_1 * kappa2T + n_2 * kappa2T2
    den = 1.0 + d_1 * kappa2T + d_2 * kappa2T2

    # Q_0
    Q_0 = a_0 / jnp.sqrt(q)

    # Dimensionless angular frequency of merger
    Momega_merger = Q_0 * (num / den)

    # Convert to frequency in Hz
    fHz_merger = Momega_merger / (jnp.pi * 2.0 * total_mass * gt)

    return fHz_merger


def get_tidal_phase(fHz, Xa, Xb, total_mass, kappa2T):
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
    tidal_phase = -kappa2T * c_Newt / (Xa * Xb) * PN_x_5over2

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

    kappa2T = get_kappa2T(m1, m2, lambda1, lambda2)
    fHz_mrg = get_merger_frequency(kappa2T, total_mass, q)
    fHz_end_taper = 1.2 * fHz_mrg

    phi_tidal = get_tidal_phase(fHz, Xa, Xb, total_mass, kappa2T)
    planck_taper = 1.0 - get_planck_taper(fHz, fHz_mrg, fHz_end_taper)

    return phi_tidal, planck_taper
