"""This file implements the NRTidalv2 corrections that can be applied to any BBH baseline, see http://arxiv.org/abs/1905.06011"""

import jax
import jax.numpy as jnp

from ..constants import EulerGamma, gt, m_per_Mpc, C, PI, TWO_PI, MSUN, MRSUN
from ..typing import Array
from ripple import Mc_eta_to_ms, ms_to_Mc_eta
import sys
from .IMRPhenomD import get_Amp0
from .utils_tidal import *

def get_tidal_phase(x: Array, theta: Array, kappa: float) -> Array:
    """Computes the tidal phase psi_T from equation (17) of the NRTidalv2 paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)
        kappa (float): Tidal parameter kappa, precomputed in the main function.

    Returns:
        Array: Tidal phase correction.
    """
    
    # Mass variables
    m1, m2, _, _, _, _ = theta 
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute ratios
    X1 = m1_s / M_s
    X2 = m2_s / M_s

    # Compute powers
    x_2 = x ** (2.)
    x_3 = x ** (3.)
    x_3over2 = x ** (3.0/2.0)
    x_5over2 = x ** (5.0/2.0)

    # Initialize the coefficients
    c_Newt   =   2.4375
    n_1      = -12.615214237993088
    n_3over2 =  19.0537346970349
    n_2      = -21.166863146081035
    n_5over2 =  90.55082156324926
    n_3      = -60.25357801943598
    d_1      = -15.111207827736678
    d_3over2 =  22.195327350624694
    d_2      =   8.064109635305156

    # Pade approximant
    num = 1.0 + (n_1 * x) + (n_3over2 * x_3over2) + (n_2 * x_2) + (n_5over2 * x_5over2) + (n_3 * x_3)
    den = 1.0 + (d_1 * x) + (d_3over2 * x_3over2) + (d_2 * x_2)
    ratio = num / den
    
    # Assemble everything
    psi_T  = - kappa * c_Newt / (X1 * X2) * x_5over2
    psi_T *= ratio
    
    return psi_T

# TODO the spin corrections might not be 100% accurate for high spins
def get_spin_phase_correction(x: Array, theta: Array) -> Array:
    """Get the higher order spin corrections, as detailed in Section IIIC of the NRTidalv2 paper.

    Args:
        x (Array): Angular frequency, in particular, x = (pi M f)^(2/3)
        theta (Array): Intrinsic parameters (mass1, mass2, chi1, chi2, lambda1, lambda2)

    Returns:
        Array: Higher order spin corrections to the phase
    """
    
    m1, m2, chi1, chi2, lambda1, lambda2 = theta

    # Convert the mass variables
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    # Compute the auxiliary variables
    X1 = m1_s / M_s
    X2 = m2_s / M_s

    X1sq = X1 * X1
    X2sq = X2 * X2
    chi1_sq = chi1 * chi1
    chi2_sq = chi2 * chi2

    # Compute quadrupole parameters
    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)
    
    # Remember to remove 1 from quadrupole and octupole, for the BBH baseline
    
    SS_2 = - 50. * ((quadparam1 - 1) * chi1_sq * X1sq + (quadparam2 - 1) * chi2_sq * X2sq)
    SS_3 = 5.0/84.0 * (9407.0 + 8218.0 * X1 - 2016.0 * X1sq) * (quadparam1 - 1) * X1sq * chi1_sq \
         + 5.0/84.0 * (9407.0 + 8218.0 * X2 - 2016.0 * X2sq) * (quadparam2 - 1) * X2sq * chi2_sq
    
    SS_3p5 = - 400. * PI * (quadparam1 - 1) * chi1_sq * X1sq \
             - 400. * PI * (quadparam2 - 1) * chi2_sq * X2sq
    SSS_3p5 = 10. * ((X1sq + 308./3. * X1) * chi1 + (X2sq - 89./3. * X2) * chi2) * (quadparam1 - 1) * X1sq * chi1_sq \
            + 10. * ((X2sq + 308./3. * X2) * chi2 + (X1sq - 89./3. * X1) * chi1) * (quadparam2 - 1) * X2sq * chi1_sq \
                - 440. * (octparam1 - 1) * X1 * X1sq * chi1_sq * chi1 \
                - 440. * (octparam2 - 1) * X2 * X2sq * chi2_sq * chi2

    prefac = (3. / (128. * eta))
    psi_SS = prefac * (SS_2 * x ** (-1./2.) + SS_3 * x ** (1./2.) + (SS_3p5 + SSS_3p5) * x)

    return psi_SS



def _get_merger_frequency(theta: Array, kappa: float = None):
    """Computes the merger frequency in Hz of the given system, see equation (11) in https://arxiv.org/abs/1804.02235 or the lal source code.

    Args:
        theta (Array): Intrinsic parameters (m1, m2, chi1, chi2, lambda1, lambda2)
        kappa (float, optional): Tidal parameter kappa. Defaults to None, so that it is computed from the given parameters theta.

    Returns:
        float: The merger frequency in Hz.
    """
    
    # Get masses
    m1, m2, _, _, _, _ = theta 
    M = m1 + m2
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    q = m1_s / m2_s
    X1 = m1_s / M_s
    X2 = m2_s / M_s
    
    # If kappa was not given, compute it
    if kappa is None:
        kappa = get_kappa(theta)
        
    # Initialize coefficients
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4
    
    # Get ratio and prefactor
    kappa_2 = kappa * kappa
    num = 1.0 + n_1 * kappa + n_2 * kappa_2
    den = 1.0 + d_1 * kappa + d_2 * kappa_2
    Q_0 = a_0 * (q) ** (-1./2.)

    # Dimensionless angular frequency of merger
    Momega_merger = Q_0 * (num / den)

    # Convert from angular frequency to frequency (divide by 2*pi) and then convert from dimensionless frequency to Hz
    fHz_merger = Momega_merger / (M * gt) / (TWO_PI)

    return fHz_merger

def _gen_NRTidalv2(f: Array, theta_intrinsic: Array, theta_extrinsic: Array, h0_bbh: Array):
    """Master internal function to get the GW strain for given parameters. The function takes 
    a BBH strain, computed from an underlying BBH approximant, e.g. IMRPhenomD, and applies the
    tidal corrections to it afterwards, according to equation (25) of the NRTidalv2 paper.

    Args:
        f (Array): Frequencies in Hz
        theta_intrinsic (Array): Internal parameters of the system: m1, m2, chi1, chi2, lambda1, lambda2
        theta_extrinsic (Array): Extrinsic parameters of the system: d_L, tc and phi_c
        h0_bbh (Array): The BBH strain of the underlying model (i.e. before applying tidal corrections).

    Returns:
        Array: Final complex-valued strain of GW.
    """

    # Compute x: see NRTidalv2 paper for definition
    m1, m2, _, _, _, _ = theta_intrinsic
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    x = (PI * M_s * f) ** (2.0/3.0)
    # M = m1 + m2

    # Compute kappa
    kappa = get_kappa(theta=theta_intrinsic)

    # Get BBH amplitude and phase
    A_bbh = jnp.abs(h0_bbh)
    psi_bbh = jnp.log(h0_bbh / A_bbh) * 1.j
    
    # Get tidal amplitude and Planck taper
    A_T = get_tidal_amplitude(x, theta_intrinsic, kappa, distance=theta_extrinsic[0])
    f_merger = _get_merger_frequency(theta_intrinsic, kappa)
    A_P = jnp.ones_like(f) - get_planck_taper(f, f_merger)

    # Get tidal phase and spin corrections for BNS
    psi_T = get_tidal_phase(x, theta_intrinsic, kappa)
    psi_SS = get_spin_phase_correction(x, theta_intrinsic)
    
    # Assemble everything
    h0 = A_P * (A_bbh + A_T) * jnp.exp(1.j * -(psi_bbh + psi_T + psi_SS))

    return h0

def gen_NRTidalv2(f: Array, params: Array, f_ref: float, IMRphenom: str) -> Array:
    """
    Generate NRTidalv2 frequency domain waveform following NRTidalv2 paper.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    lambda1: Dimensionless tidal deformability of primary object
    lambda2: Dimensionless tidal deformability of secondary object
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence

    f_ref: Reference frequency for the waveform
    
    IMRphenom: string selecting the underlying BBH approximant

    Returns:
    --------
        h0 (array): Strain
    """
    
    # Get component masses
    m1, m2 = Mc_eta_to_ms(jnp.array([params[0], params[1]]))
    chi1, chi2 = params[2], params[3]
    lambda1, lambda2 = params[4], params[5]
    
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    theta_extrinsic = params[6:]

    # Get the parameters that are passed to the BBH waveform, i.e. remove lambdas
    bbh_params = jnp.concatenate((jnp.array([params[0], params[1], params[2], params[3]]), theta_extrinsic))

    # TODO - make compatible with other waveforms as well
    if IMRphenom == "IMRPhenomD":
        from ripple.waveforms.IMRPhenomD import (
            gen_IMRPhenomD as bbh_waveform_generator,
        )
    else:
        print("IMRPhenom string not recognized")
        return jnp.zeros_like(f)
    
    # Generate BBH waveform strain and get its amplitude and phase
    h0_bbh = bbh_waveform_generator(f, bbh_params, f_ref)

    # Use BBH waveform and add tidal corrections
    return _gen_NRTidalv2(f, theta_intrinsic, theta_extrinsic, h0_bbh)


def gen_NRTidalv2_hphc(f: Array, params: Array, f_ref: float, IMRphenom: str="IMRPhenomD"):
    """
    vars array contains both intrinsic and extrinsic variables
    
    IMRphenom denotes the name of the underlying BBH approximant used, before applying tidal corrections.
    
    theta = [Mchirp, eta, chi1, chi2, lambda1, lambda2, D, tc, phic, inclination]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
    --------
        hp (array): Strain of the plus polarization
        hc (array): Strain of the cross polarization
    """
    iota = params[-1]
    h0 = gen_NRTidalv2(f, params[:-1], f_ref, IMRphenom=IMRphenom)
    
    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)

    return hp, hc
