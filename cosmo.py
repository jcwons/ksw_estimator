import numpy as np
import camb

def run_camb(lmax=2500, k_eta_fac=2.5, AccuracyBoost=2, lSampleBoost=50, lAccuracyBoost=2, verbose=True):
    transfer = {}
    cls = {}

    acc_opts = dict(AccuracyBoost=AccuracyBoost,
                    lSampleBoost=lSampleBoost,
                    lAccuracyBoost=lAccuracyBoost,
                    DoLateRadTruncation=False)

    pars = camb.CAMBparams()
    pars.set_accuracy(**acc_opts)

    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=1)
    pars.set_accuracy(lSampleBoost=50)

    pars.max_l = lmax
    pars.max_eta_k = k_eta_fac * lmax

    pars.AccurateBB = True
    pars.AccurateReionization = True
    pars.AccuratePolarization = True

    # calculate results for these parameters

    data = camb.get_transfer_functions(pars)
    transfer_s = data.get_cmb_transfer_data('scalar')

    data.calc_power_spectra()
    cls_camb = data.get_cmb_power_spectra(lmax=None, CMB_unit='muK',raw_cl=True)
    for key in cls_camb:
        cls_cm = cls_camb[key]
        n_ell, n_pol = cls_cm.shape
        temp = np.ascontiguousarray(cls_cm.transpose())
        # Remove monopole and dipole.
        cls_camb[key] = temp[:, 2:]

    ells_cls = np.arange(2, n_ell)
    cls['ells'] = ells_cls
    cls['cls'] = cls_camb

    # We need to modify scalar E-mode and tensor I transfer functions,
    # see Zaldarriaga 1997 eq. 18 and 39. (CAMB applies these factors
    # at a later stage).

    try:
        ells = transfer_s.l
    except AttributeError:
        ells = transfer_s.L
        # CAMB ells are in int32, gives nan in sqrt, so convert first.
    ells = np.array(ells,dtype="int64")
    prefactor = np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))

    transfer_s.delta_p_l_k[1, ...] *= prefactor[:, np.newaxis]
    transfer_s.delta_p_l_k *= (pars.TCMB * 1e6)


    transfer['scalar'] = transfer_s.delta_p_l_k
    transfer['k'] = transfer_s.q
    transfer['ells'] = ells  # sparse and might differ from cls['ells']
    return transfer, cls
