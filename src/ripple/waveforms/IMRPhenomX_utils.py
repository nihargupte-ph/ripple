from typing import Tuple

import jax.numpy as jnp
import jax

from ..constants import gt, PI
from ..typing import Array

# Dimensionless cutoff frequency for PhenomXAS
f_CUT = 0.3


def get_cutoff_fs(m1, m2, chi1, chi2):
    # This function returns a variety of frequencies needed for computing IMRPhenomXAS
    # In particular, we have fRD, fdamp, fMECO, FISCO
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)
    # m1Sq = m1_s * m1_s
    # m2Sq = m2_s * m2_s

    delta = jnp.sqrt(1.0 - 4.0 * eta_s)
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    chi_eff = mm1 * chi1 + mm2 * chi2

    eta2 = eta_s * eta_s
    eta3 = eta2 * eta_s
    eta4 = eta3 * eta_s
    S = (chi_eff - (38.0 / 113.0) * eta_s * (chi1 + chi2)) / (
        1.0 - (76.0 * eta_s / 113.0)
    )
    S2 = S * S
    S3 = S2 * S

    dchi = chi1 - chi2
    dchi2 = dchi * dchi

    StotR = (mm1**2.0 * chi1 + mm2**2.0 * chi2) / (mm1**2.0 + mm2**2.0)
    StotR2 = StotR * StotR
    StotR3 = StotR2 * StotR

    # First we need to calculate the dimensionless final spin and the radiated energy
    # From https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_utilities_8c_source.html
    # (((3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta2*eta)/(1 + 7.2388440419467335*eta)) + ((m1ByMSq + m2ByMSq)*totchi + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta2*eta)*totchi + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta2*eta)*totchi2 + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta2*eta)*totchi2*totchi) / (1 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta2*eta)*totchi)) + (0.3223660562764661*dchi*Seta*(1 + 9.332575956437443*eta)*eta2 - 0.059808322561702126*dchi*dchi*eta2*eta + 2.3170397514509933*dchi*Seta*(1 - 3.2624649875884852*eta)*eta2*eta*totchi))
    a = (
        (
            3.4641016151377544 * eta_s
            + 20.0830030082033 * eta2
            - 12.333573402277912 * eta3
        )
        / (1 + 7.2388440419467335 * eta_s)
        + (
            (mm1**2.0 + mm2**2.0) * StotR
            + (
                (
                    -0.8561951310209386 * eta_s
                    - 0.09939065676370885 * eta2
                    + 1.668810429851045 * eta3
                )
                * StotR
                + (
                    0.5881660363307388 * eta_s
                    - 2.149269067519131 * eta2
                    + 3.4768263932898678 * eta3
                )
                * StotR2
                + (
                    0.142443244743048 * eta_s
                    - 0.9598353840147513 * eta2
                    + 1.9595643107593743 * eta3
                )
                * StotR3
            )
            / (
                1
                + (
                    -0.9142232693081653
                    + 2.3191363426522633 * eta_s
                    - 9.710576749140989 * eta3
                )
                * StotR
            )
        )
        + (
            0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta_s) * eta2
            - 0.059808322561702126 * dchi2 * eta3
            + 2.3170397514509933
            * dchi
            * delta
            * (1 - 3.2624649875884852 * eta_s)
            * eta3
            * StotR
        )
    )

    Erad = (
        (
            (
                0.057190958417936644 * eta_s
                + 0.5609904135313374 * eta2
                - 0.84667563764404 * eta3
                + 3.145145224278187 * eta4
            )
            * (
                1
                + (
                    -0.13084389181783257
                    - 1.1387311580238488 * eta_s
                    + 5.49074464410971 * eta2
                )
                * StotR
                + (-0.17762802148331427 + 2.176667900182948 * eta2) * StotR2
                + (
                    -0.6320191645391563
                    + 4.952698546796005 * eta_s
                    - 10.023747993978121 * eta2
                )
                * StotR3
            )
        )
        / (
            1
            + (
                -0.9919475346968611
                + 0.367620218664352 * eta_s
                + 4.274567337924067 * eta2
            )
            * StotR
        )
    ) + (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta_s) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653
        * dchi
        * delta
        * (1 - 4.91667749015812 * eta_s)
        * eta_s
        * StotR
    )

    # Taken from https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_t_h_m__fits_8c_source.html

    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a
    a7 = a6 * a

    # First the ringdown frequency
    fRD = (
        (
            0.05947169566573468
            - 0.14989771215394762 * a
            + 0.09535606290986028 * a2
            + 0.02260924869042963 * a3
            - 0.02501704155363241 * a4
            - 0.005852438240997211 * a5
            + 0.0027489038393367993 * a6
            + 0.0005821983163192694 * a7
        )
        / (
            1
            - 2.8570126619966296 * a
            + 2.373335413978394 * a2
            - 0.6036964688511505 * a4
            + 0.0873798215084077 * a6
        )
    ) / (1.0 - Erad)

    # Then the damping frequency
    fdamp = (
        (
            0.014158792290965177
            - 0.036989395871554566 * a
            + 0.026822526296575368 * a2
            + 0.0008490933750566702 * a3
            - 0.004843996907020524 * a4
            - 0.00014745235759327472 * a5
            + 0.0001504546201236794 * a6
        )
        / (
            1
            - 2.5900842798681376 * a
            + 1.8952576220623967 * a2
            - 0.31416610693042507 * a4
            + 0.009002719412204133 * a6
        )
    ) / (1.0 - Erad)

    Z1 = 1.0 + jnp.cbrt((1.0 - a2)) * (jnp.cbrt(1 + a) + jnp.cbrt(1 - a))
    Z1 = jnp.where(Z1 > 3.0, 3.0, Z1)
    Z2 = jnp.sqrt(3.0 * a2 + Z1 * Z1)
    rISCO = 3.0 + Z2 - jnp.sign(a) * jnp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    rISCOsq = jnp.sqrt(rISCO)
    rISCO3o2 = rISCOsq * rISCOsq * rISCOsq
    OmegaISCO = 1.0 / (rISCO3o2 + a)
    fISCO = OmegaISCO / PI

    fMECO = (
        (
            0.018744340279608845
            + 0.0077903147004616865 * eta_s
            + 0.003940354686136861 * eta2
            - 0.00006693930988501673 * eta3
        )
        / (1.0 - 0.10423384680638834 * eta_s)
        + (
            (
                S
                * (
                    0.00027180386951683135
                    - 0.00002585252361022052 * S
                    + eta4
                    * (
                        -0.0006807631931297156
                        + 0.022386313074011715 * S
                        - 0.0230825153005985 * S2
                    )
                    + eta2
                    * (
                        0.00036556167661117023
                        - 0.000010021140796150737 * S
                        - 0.00038216081981505285 * S2
                    )
                    + eta_s
                    * (
                        0.00024422562796266645
                        - 0.00001049013062611254 * S
                        - 0.00035182990586857726 * S2
                    )
                    + eta3
                    * (
                        -0.0005418851224505745
                        + 0.000030679548774047616 * S
                        + 4.038390455349854e-6 * S2
                    )
                    - 0.00007547517256664526 * S2
                )
            )
            / (
                0.026666543809890402
                + (
                    -0.014590539285641243
                    - 0.012429476486138982 * eta_s
                    + 1.4861197211952053 * eta4
                    + 0.025066696514373803 * eta2
                    + 0.005146809717492324 * eta3
                )
                * S
                + (
                    -0.0058684526275074025
                    - 0.02876774751921441 * eta_s
                    - 2.551566872093786 * eta4
                    - 0.019641378027236502 * eta2
                    - 0.001956646166089053 * eta3
                )
                * S2
                + (
                    0.003507640638496499
                    + 0.014176504653145768 * eta_s
                    + 1.0 * eta4
                    + 0.012622225233586283 * eta2
                    - 0.00767768214056772 * eta3
                )
                * S3
            )
        )
        + (
            dchi2 * (0.00034375176678815234 + 0.000016343732281057392 * eta_s) * eta2
            + dchi
            * delta
            * eta_s
            * (
                0.08064665214195679 * eta2
                + eta_s * (-0.028476219509487793 - 0.005746537021035632 * S)
                - 0.0011713735642446144 * S
            )
        )
    )

    # return fRD / M_s, fdamp / M_s, fMECO / M_s, fISCO / M_s
    # print(fRD, fdamp, fMECO, fISCO)
    return fRD, fdamp, fMECO, fISCO


def nospin_CPvalue(NoSpin_coeffs, eta):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return (
        NoSpin_coeffs[0]
        + NoSpin_coeffs[1] * eta
        + NoSpin_coeffs[2] * eta2
        + NoSpin_coeffs[3] * eta3
        + NoSpin_coeffs[4] * eta4
        + NoSpin_coeffs[5] * eta5
    ) / (
        NoSpin_coeffs[6]
        + NoSpin_coeffs[7] * eta
        + NoSpin_coeffs[8] * eta2
        + NoSpin_coeffs[9] * eta3
    )


def Eqspin_CPvalue(EqSpin_coeffs, eta, S):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    numerator = S * (
        EqSpin_coeffs[0]
        + EqSpin_coeffs[1] * S
        + EqSpin_coeffs[2] * S2
        + EqSpin_coeffs[3] * S3
        + EqSpin_coeffs[4] * S4
        + eta
        * (
            EqSpin_coeffs[5]
            + EqSpin_coeffs[6] * S
            + EqSpin_coeffs[7] * S2
            + EqSpin_coeffs[8] * S3
            + EqSpin_coeffs[9] * S4
        )
        + eta2
        * (
            EqSpin_coeffs[10]
            + EqSpin_coeffs[11] * S
            + EqSpin_coeffs[12] * S2
            + EqSpin_coeffs[13] * S3
            + EqSpin_coeffs[14] * S4
        )
        + eta3
        * (
            EqSpin_coeffs[15]
            + EqSpin_coeffs[16] * S
            + EqSpin_coeffs[17] * S2
            + EqSpin_coeffs[18] * S3
            + EqSpin_coeffs[19] * S4
        )
        + eta4
        * (
            EqSpin_coeffs[20]
            + EqSpin_coeffs[21] * S
            + EqSpin_coeffs[22] * S2
            + EqSpin_coeffs[23] * S3
            + EqSpin_coeffs[24] * S4
        )
    )
    denominator = (
        EqSpin_coeffs[25]
        + EqSpin_coeffs[26] * S
        + EqSpin_coeffs[27] * S2
        + EqSpin_coeffs[28] * S3
    )
    return numerator / denominator


def Uneqspin_CPvalue(EqSpin_coeffs, eta, S, dchi):
    dchi2 = dchi * dchi
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return (
        dchi
        * delta
        * eta
        * (
            EqSpin_coeffs[0]
            + EqSpin_coeffs[1] * eta
            + EqSpin_coeffs[2] * eta2
            + EqSpin_coeffs[3] * eta4
            + EqSpin_coeffs[4] * eta5  # Added
            + EqSpin_coeffs[5] * S
            + EqSpin_coeffs[6] * S * eta2  # Added
            + EqSpin_coeffs[7] * S * eta3  # Added
        )
        + EqSpin_coeffs[8] * dchi2 * eta
    )


PhenomX_coeff_table = jnp.array(
    [
        [  # Coeffs collocation point 0 of the inspiral phase (ind 0)
            -17294.000000000007,  # No spin
            -19943.076428555978,
            483033.0998073767,
            0.0,
            0.0,
            0.0,
            1.0,
            4.460294035404433,
            0.0,
            0.0,
            68384.62786426462,  # Eq spin
            67663.42759836042,
            -2179.3505885609297,
            19703.894135534803,
            32614.091002011017,
            -58475.33302037833,
            62190.404951852535,
            18298.307770807573,
            -303141.1945565486,
            0.0,
            -148368.4954044637,
            -758386.5685734496,
            -137991.37032619823,
            1.0765877367729193e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0412979553629143,
            1.0,
            0.0,
            0.0,
            12017.062595934838,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the inspiral phase (ind 1)
            -7579.300000000004,  # No spin
            -120297.86185566607,
            1.1694356931282217e6,
            -557253.0066989232,
            0.0,
            0.0,
            1.0,
            18.53018618227582,
            0.0,
            0.0,
            -27089.36915061857,  # Eq spin
            -66228.9369155027,
            -44331.41741405198,
            0.0,
            0.0,
            50644.13475990821,
            157036.45676788126,
            126736.43159783827,
            0.0,
            0.0,
            150022.21343386435,
            -50166.382087278434,
            -399712.22891153296,
            0.0,
            0.0,
            -593633.5370110178,
            -325423.99477314285,
            +847483.2999508682,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5232497464826662,
            -3.062957826830017,
            -1.130185486082531,
            1.0,
            3843.083992827935,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the inspiral phase (ind 2)
            15415.000000000007,  # No spin
            873401.6255736464,
            376665.64637025696,
            -3.9719980569125614e6,
            8.913612508054944e6,
            0.0,
            1.0,
            46.83697749859996,
            0.0,
            0.0,
            397951.95299014193,  # Eq spin
            -207180.42746987,
            -130668.37221912303,
            0.0,
            0.0,
            -1.0053073129700898e6,
            1.235279439281927e6,
            -174952.69161683554,
            0.0,
            0.0,
            -1.9826323844247842e6,
            208349.45742548333,
            895372.155565861,
            0.0,
            0.0,
            4.662143741417853e6,
            -584728.050612325,
            -1.6894189124921719e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -9.675704197652225,
            3.5804521763363075,
            2.5298346636273306,
            1.0,
            -24708.109411857182,  # UnEq Spin
            24703.28267342699,
            0.0,
            0.0,
            0.0,
            47752.17032707405,
            0.0,
            0.0,
            -1296.9289110696955,
        ],
        [  # Coeffs collocation point 3 of the inspiral phase (ind 3)
            2439.000000000001,  # No spin
            -31133.52170083207,
            28867.73328134167,
            0.0,
            0.0,
            0.0,
            1.0,
            0.41143032589262585,
            0.0,
            0.0,
            16116.057657391262,  # Eq spin
            9861.635308837876,
            0.0,
            0.0,
            0.0,
            -82355.86732027541,
            -25843.06175439942,
            0.0,
            0.0,
            0.0,
            229284.04542668918,
            117410.37432997991,
            0.0,
            0.0,
            0.0,
            -375818.0132734753,
            -386247.80765802023,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.7385208695213668,
            0.25294420589064653,
            1.0,
            0.0,
            194.5554531509207,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 0 of the intermediate phase (ind 4)
            0.0,  # No Spin
            0.9951733419499662,
            101.21991715215253,
            632.4731389009143,
            0.0,
            0.0,
            0.00016803066316882238,
            0.11412314719189287,
            1.8413983770369362,
            1.0,
            18.694178521101332,  # Eq spin
            16.89845522539974,
            0.0,
            0.3612417066833153,
            0.0,
            -697.6773920613674,
            0.0,
            -147.53381808989846,
            0.0,
            0.0,
            0.0,
            4941.31613710257,
            0.0,
            0.0,
            0.0,
            3531.552143264721,
            -14302.70838220423,
            178.85850322465944,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.965640445745779,
            -2.7706595614504725,
            1.0,
            0.0,
            0.0,  # UnEq Spin
            356.74395864902294,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1693.326644293169,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the intermediate phase (ind 5)
            0.0,  # No Spin
            -5.126358906504587,
            -227.46830225846668,
            688.3609087244353,
            -751.4184178636324,
            0.0,
            -0.004551938711031158,
            -0.7811680872741462,
            1.0,
            0.0,
            0.1549280856660919,  # Eq spin
            -0.9539250460041732,
            -2.84311102369862,
            0.0,
            0.0,
            73.79645135116367,
            0.0,
            -8.13494176717772,
            0.0,
            0.0,
            0.0,
            -539.4071941841604,
            0.0,
            0.0,
            0.0,
            -936.3740515136005,
            1862.9097047992134,
            224.77581754671272,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5308507364054487,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            2993.3598520496153,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the intermediate phase (ind 6)
            -82.54500000000004,  # No Spin
            -5.58197349185435e6,
            -3.5225742421184325e8,
            1.4667258334378073e9,
            0.0,  #
            0.0,  #
            1.0,
            66757.12830903867,
            5.385164380400193e6,
            2.5176585751772933e6,
            19.416719811164853,  # Eq spin
            -36.066611959079935,
            -0.8612656616290079,
            5.95010003393006,
            4.984750041013893,
            207.69898051583655,
            -132.88417400679026,
            -17.671713040498304,
            29.071788188638315,
            37.462217031512786,
            170.97203068800542,
            -107.41099349364234,
            0.0,
            -647.8103976942541,
            0.0,
            -1365.1499998427248,
            1152.425940764218,
            415.7134909564443,
            1897.5444343138167,
            -866.283566780576,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.1492259468169692,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            7343.130973149263,
            -20486.813161100774,
            0.0,
            0.0,
            515.9898508588834,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the intermediate phase (ind 7)
            0.4248820426833804,  # No Spin
            -906.746595921514,
            -282820.39946006844,
            -967049.2793750163,
            670077.5414916876,
            0.0,
            1.0,
            1670.9440812294847,
            19783.077247023448,
            0.0,
            0.22814271667259703,  # Eq spin
            1.1366593671801855,
            0.4818323187946999,
            0.0,
            0.0,
            12.840649528989287,
            0.0,
            -61.17248283184154,
            941.6974723887743,
            0.0,
            -711.8532052499075,
            269.9234918621958,
            0.0,
            0.0,
            0.0,
            3499.432393555856,
            -877.8811492839261,
            -4974.189172654984,
            0.0,
            0.0,
            -4939.642457025497,
            -227.7672020783411,
            8745.201037897836,
            0.0,
            0.0,
            -1.2442293719740283,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            -514.8494071830514,
            1493.3851099678195,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 0 of the merger ringdown phase (ind 8)
            0.0,  # No spin
            0.7207992174994245,
            -1.237332073800276,
            6.086871214811216,
            0.0,
            0.0,
            0.006851189888541745,
            0.06099184229137391,
            -0.15500218299268662,
            1.0,
            0.06519048552628343,  # Eq spin
            0.0,
            0.20035146870472367,
            0.0,
            -0.2697933899920511,
            -25.25397971063995,
            -5.215945111216946,
            -0.28745205203100666,
            5.7756520242745735,
            +4.917070939324979,
            +58.59408241189781,
            +153.95945758807616,
            0.0,
            -43.97332874253772,
            -11.61488280763592,
            +160.14971486043524,
            -693.0504179144295,
            0.0,
            0.0,
            0.0,
            -308.62513664956975,
            +835.1725103648205,
            -47.56042058800358,
            +338.7263666984089,
            -22.384949087140086,
            1.0,
            -0.6628745847248266,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            -23.504907495268824,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the merger ringdown phase (ind 9)
            0.0,  # No spin
            -9.460253118496386,
            +9.429314399633007,
            +64.69109972468395,
            0.0,
            0.0,
            -0.0670554310666559,
            -0.09987544893382533,
            1.0,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.04497628581617564,
            0.0,
            17.36495157980372,
            0.0,
            0.0,
            0.0,
            0.0,
            -191.00932194869588,
            -62.997389062600035,
            +64.42947340363101,
            0.0,
            0.0,
            930.3458437154668,
            +808.457330742532,
            0.0,
            0.0,
            0.0,
            -774.3633787391745,
            -2177.554979351284,
            -1031.846477275069,
            0.0,
            0.0,
            1.0,
            -0.7267610313751913,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            -36.66374091965371,
            91.60477826830407,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the merger ringdown phase (ind 10)
            0.0,  # No spin
            -8.506898502692536,
            +13.936621412517798,
            0.0,
            0.0,
            0.0,
            -0.40919671232073945,
            1.0,
            0.0,
            0.0,
            0.0,  # Eq spin
            0.046849371468156265,
            0.0,
            0.0,
            0.0,
            1.7280582989361533,
            0.0,
            18.41570325463385,
            -13.743271480938104,
            0.0,
            73.8367329022058,
            0.0,
            -95.57802408341716,
            +215.78111099820157,
            0.0,
            -27.976989112929353,
            +6.404060932334562,
            +109.04824706217418,
            -633.1966645925428,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.6862449113932192,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            0.0,
            641.8965762829259,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 3 of the merger ringdown phase (ind 11)
            -85.86062966719405,  # No spin
            -4616.740713893726,
            -4925.756920247186,
            +7732.064464348168,
            +12828.269960300782,
            -39783.51698102803,
            1.0,
            +50.206318806624004,
            0.0,
            0.0,
            33.335857451144356,  # Eq spin
            -36.49019206094966,
            -3.835967351280833,
            2.302712009652155,
            1.6533417657003922,
            -69.19412903018717,
            26.580344399838758,
            -15.399770764623746,
            31.231253209893488,
            97.69027029734173,
            93.64156367505917,
            -18.184492163348665,
            423.48863373726243,
            -104.36120236420928,
            -719.8775484010988,
            0.0,
            1497.3545918387515,
            -101.72731770500685,
            0.0,
            0.0,
            1075.8686153198323,
            -3443.0233614187396,
            -4253.974688619423,
            -608.2901586790335,
            5064.173605639933,
            -1.3705601055555852,
            1.0,
            0.0,
            0.0,
            22.363215261437862,  # UnEq Spin
            156.08206945239374,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 4 of the merger ringdown phase (ind 12)
            0.0,  # No spin
            7.05731400277692,
            22.455288821807095,
            119.43820622871043,
            0.0,
            0.0,
            0.26026709603623255,
            1.0,
            0.0,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.0,
            -7.9407123129681425,
            9.486783128047414,
            0.0,
            0.0,
            0.0,
            134.88158268621922,
            -56.05992404859163,
            0.0,
            0.0,
            0.0,
            -316.26970506215554,
            90.31815139272628,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.7162058321905909,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            43.82713604567481,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
)
