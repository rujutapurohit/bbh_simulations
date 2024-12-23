import numpy as np
import os
import sys
import random
from datetime import datetime
random.seed(datetime.now())

from scipy.stats import maxwell

# use maxwell distributions

def sample_maxwell(nins):

    dist = maxwell(loc=0.0, scale=nins)

    xmax = 3.0 * nins
    ymax = dist.pdf(np.sqrt(2.0) * nins)

    yran = 1.0
    yfun = 0.0
    while yran > yfun:
        xran = random.random() * xmax
        yran = random.random() * ymax
        yfun = dist.pdf(xran)

    return xran

# define function for mergine binary balck holes that takes in the masses of the bhs,
# their numbers and the slpoe of the bh function

def merging_bbh(mbh, numbhbin, slopebh, slopebhbin):

    numbh = len(mbh)

    m1_arr, m2_arr, j1_arr, j2_arr = [], [], [], []
    for i in range(numbh):
        if len(m1_arr) < numbhbin:
            j1 = numbh - 1 - i
            if j1 not in j1_arr and j1 not in j2_arr:
                m1 = mbh[j1]
                m1_arr.append(m1), j1_arr.append(j1)

                if i == 0:
                    j2 = j1
                    while j2 == j1:
                        mtot = mbh + np.ones_like(mbh) * m1
                        m2 = kc.sample_mass(min(mtot), max(mtot), slopebh) - m1
                        j2 = np.argmin(np.abs(mbh - m2))
                        m2 = mbh[j2]
                    m2_arr.append(m2), j2_arr.append(j2)
                else:
                    j12 = j1_arr + j2_arr
                    ind = np.arange(0, len(mbh), 1)
                    mask = np.ones(len(mbh), dtype=bool)
                    mask[j12] = False
                    mbhtmp = mbh[mask]
                    indtmp = ind[mask]

                    mtot = mbhtmp + np.ones_like(mbhtmp) * m1
                    m2 = kc.sample_mass(min(mtot), max(mtot), slopebh) - m1
                    j2 = np.argmin(np.abs(mbhtmp - m2))
                    m2 = mbhtmp[j2]
                    m2_arr.append(m2), j2_arr.append(indtmp[j2])

    mtot_arr = np.array(m1_arr) + np.array(m2_arr)
    mbin = kc.sample_mass(min(mtot_arr), max(mtot_arr), slopebhbin)
    jbin = np.argmin(np.abs(mtot_arr - mbin))

    j11, j22 = j1_arr[jbin], j2_arr[jbin]

    return(
        j11,
        j22,
        )

# define our new function that also takes in a parameter called frac which is the fraction of initial 
# cluster mass that undergoes gravitational runaway to form large-mass stars

def new_bbh(mcl, rh, mmin, mmax, slope, ispin, sspin, zeta, sigma, fbhbin, slopebh, slopebhbin, file_name, frac):

    mcl_ini, rh_ini = mcl, rh

    fdata = open(file_name, "w")

    # Define initial cluster quantities
    kmspcgyr = 1.e3

    lnk, mall, tsev, avt, grav, clight = 10., 0.638, 2.e-3, 10.0, 4.3e3, 3.e8                      # -, msun, Gyr, Gyr, pc^3/Msun/Gyr^2, pc/Gyr
    xi, nrh, a1, nu, beta, delta_mev = 0.1, 3.2, 1.47, 8.23e-2, 2.8e-3, 10 ** 5.4

    numcl = mcl / mall
    rhoh = mcl / (4.0 / 3.0 * np.pi * rh ** 3)                                      # Msun / pc^3
    vesct = 50. * (mcl / 10 ** 5) ** (1.0 / 3.0) * (rhoh / 10 ** 5) ** (1.0 / 6.0)  # km/s

    # Generate initial BH Mass spectrum

    dm = 0.5
    ms, msfin = sse.sse(mmin, mmax, dm, zeta, sigma)    # use sse to produce spectrum mbh mass

    numbh = int(0.003025 * mcl)                         # total number of black holes
    mbh, ss = np.array([]), np.array([])
    for i in range(numbh):                              # get black hole mass for each bh
        mms = kc.sample_mass(min(ms), max(ms), slope)
        Idm = np.argmin(np.abs(ms - mms))
        mm = msfin[Idm]
        vbirth = sample_maxwell(sigma) * 1.33 / mm

    # Apply natal kicks

        if vbirth < vesct:                              # check if black holes are retained after natal kicks
            mbh = np.append(mbh, mm)                    # array bh masses
            ssbh = kc.sample_spin(ispin, sspin, mm)
            ss = np.append(ss, ssbh)                    # array of black hole spin
    
    # total mass of runaway is frac times the initial cluster mass

    mrun = frac*mcl
    
    if mrun > 0:                                        #check if there is runaway
        mbh = np.append(mbh, mrun)                      #append bh mass with the runaway mass                    
        ssbh = kc.sample_spin(ispin, sspin, mrun)
        ss = np.append(ss, ssbh)  
    
    numbh = len(mbh)
    mbhcl = np.sum(mbh)
    ig = np.ones_like(mbh)                              # update generation number

    # Define additional initial cluster quantities
    psi = 1.0 + a1 * mbhcl / mcl / 0.01
    trh = 0.138 / mall / psi / lnk * np.sqrt(mcl * rh ** 3. / grav)     # Gyr
    ent = xi * (0.2 * grav * mcl ** 2. / rh) / trh                      # Msun pc^2 / Gyr^3
    tcc = nrh * trh

    ### merging bh binaries

    time, tmax = 0., 13.
    nir, nin, nout = 0, 0, 0
    while time < tmax:

        ### pairing black holes

        p = (-mbh).argsort()
        mbh, ig, ss = mbh[p], ig[p], ss[p]
        numbhbin = int(numbh / 2. * fbhbin)                                 # number of binary black holes

        mej = 0.
        if numbhbin >= 1 and time >= tcc:

            j11, j22 = merging_bbh2(mbh, numbhbin, slopebh, slopebhbin)      # get indicesof  merging bhs

            ### merging binary j11-j22

            mb1, mb2, ig1, ig2 = mbh[j11], mbh[j22], ig[j11], ig[j22]
            mbh[j11], mbh[j22] = 0., 0.
            mb12, mum = mb1 + mb2, mb1 * mb2 / (mb1 + mb2)
            vdispbh = vesct / 4.77                                          # km/s
            abin = grav * mb1 * mb2 / (mall * (vdispbh * kmspcgyr) ** 2.)   # pc - HS boundary

            inir, inin, inout, delta_t = 0, 0, 0, 0

            if mb12 > 1.e3:   # special treatment for massive black holes

                m3av = (mbhcl - mb12) / (len(mbh) - 2)
                dd = 1.0 - 6.0 / 9.0 * m3av / (mb12 + m3av)

                ecc = np.sqrt(random.random())

                while inin == 0:

                    delta_t += (1.0 / dd - 1) * grav * mb1 * mb2 / (2. * abin) / ent
                    abin = abin * dd                                    

                    ecc = min(1.0, ecc + 0.01 * (1.0 - dd)) #np.sqrt(random.random())
                    elle = np.sqrt(1. - ecc ** 2.)
                    lbsth = 1.3 * (grav ** 4. * (mb1 * mb2) ** 2. * mb12 / ent / clight ** 5.) ** (1. / 7.) * abin ** (-5. / 7.)
                    if elle < lbsth:
                        inin = 1
                        vkbin = 0.
                        tgw = 13.0 * 2000. / (mb1 * mb2 * mb12) * (abin * 2.e5 / 0.1) ** 4. * (1.0 - ecc ** 2) ** 3.5               # Gyr

            while inir == 0 and inin == 0 and inout == 0:

                m3 = 0.                                                                 # interacting bh
                while m3 == 0.:
                    m3 = kc.sample_mass(min(mbh), max(mbh), -1)
                    j3 = np.argmin(np.abs(mbh - m3))
                m3 = mbh[j3]
                mb123, q3, mum3 = mb12 + m3, m3 / mb12, m3 * mb12 / (m3 + mb12)

                #@ dd = 0.84
                dd = 1.0 - 6.0 / 9.0 * m3 / mb123
                delta_t += (1.0 / dd - 1) * grav * mb1 * mb2 / (2. * abin) / ent
                abin = abin * dd                                                        # sma after interaction

                #@ rsch = 2. * grav * mb12 / clight ** 2.                                   # pc - Schwarzschild radius                      
                rsch = 2. * grav * mb1 / clight ** 2.                                   # pc - Schwarzschild radius                      
                for ir in range(20):                                                    # compute resonant states
                    if inir == 0:
                        ecc = np.sqrt(random.random())
                        #@ elle = np.sqrt(1. - ecc ** 2.)
                        elle = 1. - ecc
                        #@ lirthr = 1.8 * (rsch / abin) ** (5. / 14.)
                        lirthr = 1.6 * (rsch / abin) ** (5. / 7.) * (mb2 / mb1) ** (2. / 7.) * (1. + mb2 / mb1) ** (1. / 7.)
                        if elle < lirthr:
                            inir = 1
                            vkbin, tgw = 0., 0.

                if inir == 0:        
                    vkbin = np.sqrt((1.0 / dd - 1) * grav * mum * m3 / mb123 /  abin) / kmspcgyr                  # km/s
                    vk3 = vkbin / q3
                    if vk3 > vesct:                                                                               # check if third BH ejected
                        mej += m3
                        mbh[j3] = 0.
                    if vkbin < vesct:                                                                             # check if mb12 ejected or merges
                        ecc = np.sqrt(random.random())
                        elle = np.sqrt(1. - ecc ** 2.)
                        lbsth = 1.3 * (grav ** 4. * (mb1 * mb2) ** 2. * mb12 / ent / clight ** 5.) ** (1. / 7.) * abin ** (-5. / 7.)
                        if elle < lbsth:
                            inin = 1 
                            tgw = 13.0 * 2000. / (mb1 * mb2 * mb12) * (abin * 2.e5 / 0.1) ** 4. * (1.0 - ecc ** 2) ** 3.5               # Gyr
                    else:
                        inout = 1
                        tgw = 13.0 * 2000. / (mb1 * mb2 * mb12) * (abin * 2.e5 / 0.1) ** 4. * (1.0 - ecc ** 2) ** 3.5                   # Gyr

            ### kick velocity, chieff, remnant mass and spin

            ecc = 0.
            vkick, chieff, chip, mfin, spinfin = kc.kick(mb1, mb2, ss[j11], ss[j22], ecc)

        else:           # not enough bh binaries or t<tcc

            j11, j22 = -1, -1
            tgw = 100.                              # Gyr
            delta_t = 0.05 * tcc                    # Gyr

        ### updating time

        time = time + delta_t
        tmerge = time + tgw

        # Evolve cluster quantities 

        mmbh = -mej / delta_t

        mev = 0.
        mev = -1.17e4 # Msun Gyr^-1
        if time < tsev:
            msev = 0.
        else:
            msev = - nu * (mcl - mbhcl) / time
        if time < tcc:
            mmtot = msev + mev + mmbh
            rrh = - msev / mcl * rh
        else:
            mmtot = msev + mev + mmbh
            rrh = (- msev / mcl + xi / trh + 2. * mmtot / mcl ) * rh

        # Update  cluster quantities

        mcl = mcl + mmtot * delta_t
        rh = rh + rrh * delta_t

        mall = mcl / numcl
        rhoh = mcl / (4.0 / 3.0 * np.pi * rh ** 3)
        vesct = 50. * (mcl / 10 ** 5) ** (1.0 / 3.0) * (rhoh / 10 ** 5) ** (1.0 / 6.0)      # km/s
        psi = 1.0 + a1 * (mbhcl - mej) / mcl / 0.01
        trh = 0.138 / mall / psi / lnk * np.sqrt(mcl * rh ** 3. / grav)                     # Gyr
        ent = xi * (0.2 * grav * mcl ** 2. / rh) / trh                                      # Msun pc^2 / Gyr^3


        if j11 > -1 and j22 > -1:
            (Idx,) = np.where(mbh > 0.)
            mbh, ig, ss = mbh[Idx], ig[Idx], ss[Idx]
            if vkbin < vesct and vkick < vesct:               # check if the merger remnant stays in the cluster
                mbh = np.append(mbh, mfin)
                ig = np.append(ig, max(ig1,ig2)+1)
                ss = np.append(ss, spinfin)

        # Update the number of bhs and their classes in the cluster
        numbh, mbhcl = len(mbh), np.sum(mbh)


    fdata.close()
    return