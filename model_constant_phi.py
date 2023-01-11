"""



"""

# Packages
import numpy as np
import pandas as pd
import tqdm

def poisson_process(n, lbd, alpha):
    """
    Creates rainfall series based on Poisson process.

    Parameters
    ----------
    n: array_like
       The number of events
    lbd: array_like
       The rain events mean distribution
    alpha: array_like
       Mean rain depth

    Returns
    -------
    samples: ndarray
       n Rain events in the same unit of alpha.
    """
    rain = np.zeros(n)
    for i in np.arange(n):
        n1 = np.random.uniform(0, 1, 1)
        if n1 < lbd:
            n2 = np.random.uniform(0, 1, 1)
            rain[i] = -alpha * np.log((1 - n2))
        else:
            rain[i] = 0
    return rain


def canopy(rain, rstar):
    """
    Returns the amount of water intercepted by the vegetation canopy
    """
    if rain <= rstar:
        return rain
    else:
        return rstar


def troughfall(rain, rstar):
    return rain - canopy(rain, rstar)


def evapotranspiration(s, sh, sw, sstar, emax, ew):
    """
    Computes evapotranspiration
    Parameters
    ----------
        s: soil moisture
        sh: soil moisture at hygroscopic point
        sw: soil moisture at wilting point
        sstar: soil moisture where ET decreases with s
        ew: evaporation rate
        emax: maximum evapotranspiration rate
    """
    if s <= sh:
        et = 0
    elif sh < s <= sw:
        et = ew * (s - sh) / (sw - sh)
    elif sw < s <= sstar:
        et = ew + (emax - ew) * (s - sw) / (sstar - sw)
    else:
        et = emax
    return et


def leakage_carbon(s, sh, phitex, phi, kstex, m_prime, ksstr, m):
    """
    Compute the total leakage given the soil conditions based on carbon dynamics
    Parameters
    ----------
    s: soil moisture
    sh: soil moisture at hygroscopic point
    phitex: textural porosity
    phi : total porosity
    kstex : textural hydraulic conductivity
    m_prime : shape parameter
    ksstr : structural hydraulic conductivity
    m : shape parameter

    """
    s_m = phitex / phi
    if s <= s_m:
        se_prime = (s - sh) / (s_m - sh)
        y = kstex * ((se_prime ** 0.5) * (1 - (1 - se_prime ** (1 / m_prime)) ** m_prime) ** 2)
    else:
        se = (s - s_m) / (1 - s_m)
        y = kstex + ksstr * ((se ** 0.5) * (1 - (1 - se ** (1 / m)) ** m) ** 2)
        pass
    return y


def swb_f(s, rain, sh, sw, sstar, emax, ew, phi, zr, rstar, phitex, kstex, m_prime, ksstr, m, dt):
    """
    Computes soil water balance for one time step
    """

    # Soil water storage capacity
    swc = phi * zr

    # Canopy interception
    ci = canopy(rain, rstar)
    tf = rain - ci  # Througfall

    # Add througfall
    s = s + tf / swc

    # Verify if there is runoff
    if s > 1:
        q = (s - 1) * swc
        s = 1
    else:
        s = s
        q = 0

    et = evapotranspiration(s, sh, sw, sstar, emax, ew) * dt
    lk = leakage_carbon(s, sh, phitex, phi, kstex, m_prime, ksstr, m) * dt

    s = s - (et + lk) / swc
    output = {'Rain': rain, 'CI': ci, 'Q': q, 's': s, 'ET': et, 'Lk': lk}
    return output


# Millennial model -------------------------------------------------------------

def st(t1, t2, t3, t4, t, t_ref):
    """
    Computes temperature scalar of daily time-step version of Century model
    Parameters
    ----------
        t1: x-axis location of inflection point
        t2: y-axis location of inflection point
        t3: distance from maximum point to minimum point
        t4: Slope of line at inflection point
         t:  Current temperature
        t_ref : Reference temperature of temperature scalar
    """
    stt = (t2 + (t3 / 3.14) * np.arctan(3.14 * (t - t1))) / \
        (t2 + (t3 / 3.14) * np.arctan(3.14 * t4 * (t_ref - t1)))
    return stt



def sw (w1,w2,s,sfc,sp):
    """
    Computes moisture scalar of daily time-step version of Century model
    Parameters
    ----------
        w1 : empirical parameter
        w2 : emperical parameter
    """
    sww = 1/(1 + w1 * np.exp(-w2*(s-sp)/(sfc-sp)))
    return sww


def agg_break_POM(kb, st, s, w1, w2, sfc, sp, A):
    """
    Computes aggregate Carbon breakdown to POM
    Parameters
    ----------
         Kb : rate of carbon breakdown
         A :  aggregated carbon
        st : temperature scalar taken from daily version of Century model
    """
    sww = sw(w1, w2, s, sfc, sp)
    Fa = kb * st * sww * A
    return Fa


def agg_form_POM(vpa, P, kpa, A, amax, st, s, w1, w2, sfc, sp):
    """
    Computes Formation of Aggregate C from POM
    Parameters
    ----------
        Vpa: maximum rate of aggregate formation
         P: POM
         Kpa : half-saturation constant of aggregate formation
         A : aggregated carbon
        Amax: Maximum capacity of Carbon in soil aggregates
        st : temperature scalar taken from daily version of Century model
    """
    sww = sw(w1, w2, s, sfc, sp)
    Fpa = vpa* st * sww * (P / (P+kpa)) * (1 - (A / amax))
    return Fpa


def POM_dec_DOC(vpl, P, kpl, kpe, B, st, s, w1, w2, sfc, sp):
    """
    Computes decomposition of POM into LMWC
    Parameters
    ----------
        Vpl: maximum rate of POM decomposition
         P: POM
         Kpl : half-saturation constant
         B : microbial biomass carbon
        Kpe: half-saturation constant of microbial control on POM mineralization
        st : temperature scalar
        sw : moisture scalar
    """
    sww = sw(w1, w2, s, sfc, sp)
    Fpl = vpl * st * sww * (P / (kpl + P)) * (B / (kpe + B))
    return Fpl


def A_dec_DOC(val, A, kal, kae, B, st, s, w1, w2, sfc, sp):
    """
    Computes depolymerization of A into DOC
    Parameters
    ----------
        Val: maximum rate of A depolymerization
         A : aggregated carbon 
         Kal : half-saturation constant
         B : microbial biomass carbon
        Kae: half-saturation constant of microbial control on A mineralization
        st : temperature scalar
        sw : moisture scalar
    """
    sww = sw(w1, w2, s, sfc, sp)
    Fal = val * st * sww * ((A*0.001) / (kal + (A*0.001))) * (B / (kae + B))
    return Fal



def DOC_leaching(s, DOC, lk, phi, zr):
    """
    Computes leaching loss of DOC
    Parameters
    ----------
    DOC : Dissolved organic carbon
    lk is the leakage from the soil water balance model
    phi : total porosity
    zr : root zone depth
    """
    Fl = lk * DOC / (phi * s * zr)
    return Fl


def adsorption_DOC(klm, DOC, M, Qmax, s, st, w1, w2, sfc, sp):
    """
    Computes adsorption of DOC to MAOM
    Parameters
    ----------
       klm : binding affinity based on pH
       Qmax : maximum sorption capacity
    """
    sww = sw(w1, w2, s, sfc, sp)
    Flm = st * sww * ((klm * Qmax * DOC - klm * DOC * M) / Qmax)
    return Flm

def desorption_DOC(M, Qmax):
    """
    Computes desorption of DOC to MAOM
    Parameters
    ----------
       M : mineral associated organic matter
       Qmax : maximum sorption capacity
    """
    Fld = M / Qmax
    return Fld

def DOC_biomass(s, vlm, st, a_star, ms_star, ns_star, mg_star, ng_star, DOC, KDOC, KO, phi, B ):
    """
    Computes microbial uptake of DOC
    Parameters
    ----------
        vlm: potential uptake rate of DOC
        st : temperature scalar taken from daily version of Century model
        B : microbial biomass
        DOC : dissolved organic carbon
        Phi : total porosity
        a_star : SOC-microorganisms collocation factor
        ms_star : cementation exponent for DOC
        ns_star : saturation exponent for DOC
        mg_star : cementation exponent for oxygen
        ng_star : saturation exponent for oxygen
        KDOC : half-saturation constant for DOC
        KO :   half-saturation constant for oxygen
        
    """

    Flb = vlm * st * B * (DOC * phi ** (a_star * (ms_star - ns_star)) * (s * phi) ** (a_star * ns_star)) / (
        (DOC * phi ** (a_star * (ms_star - ns_star)) * (s * phi) ** (a_star * ns_star)) + KDOC) * (
        (0.209 * (phi - s * phi) ** (4 / 3) * phi ** (mg_star - ng_star) * (s * phi - phi) ** ng_star) / (
            0.209 * (phi - s * phi) ** (4 / 3) * phi ** (mg_star - ng_star) * (s * phi - phi) ** ng_star + KO))    
    return Flb



def bio_MAOM(kmm, st, B):
    """
    Computes Carbon flow from microbial biomass B to MAOM
    Parameters
    ----------
        kmm: adsorption rate of microbial biomass
        st : temperature scalar taken from daily version of Century model
        B  : microbial biomass
    """
    Fbm = (kmm * st  * B)
    return Fbm

def pom_f(pi, Fi, kb, st, s, A, vpa, P, kpa, amax, vpl, kpl, kpe, B, w1, w2, sfc, sp,
          dt):  
    """
    Computes change in POM P with time
         pi : Proportion of C input allocated to POM
         Fi : C input from aboveground plant litter, root litter and root exudates
         pa : proportion of C in aggregate breakdown allocated to POM
         Fa : aggregate carbon formation from POM
         Fpa : aggregate carbon formation from POM
         Fpl : decomposition of POM into DOC
    """

    Fa = agg_break_POM(kb, st, s, w1, w2, sfc, sp, A)
    Fpa = agg_form_POM(vpa, P, kpa, A, amax, st, s, w1, w2, sfc, sp)
    Fpl = POM_dec_DOC(vpl, P, kpl, kpe, B, st, s, w1, w2, sfc, sp)
    P = (pi * Fi + Fa - Fpa - Fpl) * dt + P
    output = {'Fa': Fa, 'Fpa': Fpa, 'Fpl': Fpl,'P': P}
    return output


def doc_f(vpl, vlm, P, kpl, kpe, A, DOC, klm, M, Qmax, B, a_star, ms_star, ns_star, mg_star, ng_star, KDOC, KO,
          phi, pi, Fi, s, st, lk, zr, dt, kmm, w1, w2, sfc, sp, val, kal, kae):
    """
    Computes change in DOC with time
         pi : proportion of C input allocated to POM
         Fi : C input from aboveground plant litter, root litter and root exudates
         Fpl : decomposition of POM into DOC
         Fal : depolymerization of aggregate
         Fbm : turnover of microbial biomass
         Flb : uptake of DOC by microbial biomass
         Fl  : DOC leaching loss
         Flm : adsorption of DOC to MAOM 
         Fld : desorption of MAOM
    """

    Fl = DOC_leaching(s, DOC, lk, phi, zr)
    Fpl = POM_dec_DOC(vpl, P, kpl, kpe, B, st, s,w1, w2, sfc, sp)
    Fal = A_dec_DOC(val, A, kal, kae, B, st, s, w1, w2, sfc, sp)
    Flm = adsorption_DOC(klm, DOC, M, Qmax, s, st,w1, w2, sfc, sp)
    Flb = DOC_biomass(s, vlm, st, a_star, ms_star, ns_star,
                      mg_star, ng_star, DOC, KDOC, KO, phi, B)
    Fbm = bio_MAOM(kmm, st, B)
    Fld = desorption_DOC(M, Qmax)
    DOC = (Fi * (1 - pi) + Fpl + Fal + Fbm - Flb - Fl- Flm + Fld) * dt + DOC
    output = {'Fl': Fl, 'Fpl': Fpl, 'Fal': Fal, 'Flm': Flm, 'Flb': Flb, 'Fbm': Fbm, 'Fld': Fld,'DOC': DOC}
    return output

def agg_f(M, A, amax, s, vpa, P, kpa, kb, st, w1, w2, sfc, sp,val, kal, kae, B,
          dt):  
    """
    Computes change in A with time
         Fpa : decomposition of POM into DOC
         Fa : adsorption of DOC to MAOM
         Fal : uptake of DOC by microbial biomass
    """
    
    Fpa = agg_form_POM(vpa, P, kpa, A, amax, st, s,w1, w2, sfc, sp)
    Fa = agg_break_POM(kb, st, s, w1, w2, sfc, sp, A)
    Fal = A_dec_DOC(val, A, kal, kae, B, st, s, w1, w2, sfc, sp)
    A = (- Fa + Fpa - Fal) * dt + A
    output = {'Fpa': Fpa, 'Fa': Fa, 'Fal': Fal, 'A': A}
    return output


def maom_f(klm, DOC, M, Qmax, s, st, B,A, w1, w2, sfc, sp,
           dt):  
    """
    Computes change in MAOM with time
         Flm : adsorption of DOC to MAOM
         Fld : desorption of MAOM
    """
    Flm = adsorption_DOC(klm, DOC, M, Qmax, s, st, w1, w2, sfc, sp)
    Fld = desorption_DOC(M, Qmax)
    M = (Flm - Fld) * dt + M
    output = {'Flm': Flm, 'Fld': Fld, 'M': M}
    return output


def bio_f(DOC, vlm, B, a_star, ms_star, ns_star, mg_star, ng_star, KDOC, KO, kmm, st, s,
          phi, CUE, dt): 
    """
    Computes change in B with time
         CUE : carbon use efficiency
         Flb : adsorption of DOC to MAOM
         Fbm : carbon flow from microbial biomass to MAOM
    """

    Flb = DOC_biomass(s, vlm, st, a_star, ms_star, ns_star,
                      mg_star, ng_star, DOC, KDOC, KO, phi, B)
    Fbm = bio_MAOM(kmm, st, B)

    B = (CUE*Flb - Fbm) * dt + B
    output = {'Flb': Flb, 'Fbm': Fbm, 'B': B}
    return output


def ksstr(kstex, kstot, A, alpha, beta):
    """
    Returns the structural hydraulic conductivity
    Parameters
    ----------
    kstex : textural hydraulic conductivity
    kstot : total hydraulic conductivity
    A : aggregated carbon
    alpha : shape parameter
    beta : shape parameter

    """
    y = kstot - ((kstot - kstex) / (1 + (A / alpha) ** beta)) - kstex
    return y


def phistr(phitot, phitex, A, alpha, beta):
    """
    Returns the structural porosity
    Parameters
    ----------
    phitot : total porosity
    phitex : textural porosity
    A : aggregated carbon
    alpha : shape parameter
    beta : shape parameter

    """
    y = phitot - ((phitot - phitex) / (1 + (A / alpha) ** beta)) - phitex
    return y


def kstot(sand, kstex):
    """
    Returns maximum possible hydraulic conductivity (i.e., fully structured soil)
    Parameters
    ----------
    sand : sand fraction (%)
    kstex : textural hydraulic conductivity

    Returns
    -------

    """
    y = 10 ** (3.5 - 1.5 * sand ** 0.13) * kstex
    return y


def phitot(sand, phitex):
    """
    Returns maximum possible porosity (i.e., fully structured soil)
    Parameters
    ----------
    sand : sand fraction(%)
    phitex : textural porosity

    Returns
    -------

#     """
    y = phitex / (10 ** (3.5 - 1.5 * sand ** 0.13))
    return y

# ----------------------------------------------------------------------------
# Soil water balance coupled with Millennial
# ----------------------------------------------------------------------------

def swb_millennial(s, rain, npp, P0, A0, B0, M0, DOC0, alpha1, beta1, alpha2, beta2, soilp, w, u, v, h, c, dt):
    """
    Solve models together
    Returns
    -------
    """

    # Textural parameters
    # Soil parameters
    sh = soilp['sh']
    sw = soilp['sw']
    sstar = soilp['sstar']
    emax = soilp['emax']
    ew = soilp['ew']
    zr = soilp['zr']
    rstar = soilp['rstar']
    kstex = soilp['kstex']
    kstot = soilp['kstot']
    phitex = soilp['phitex']
    phitot = soilp['phitot']
    m_prime = soilp['m_prime']
    m = soilp['m']

    # Hydraulic parameterization
    ksstrp = ksstr(kstex, kstot, A0, alpha1, beta1)
    phistrp = phistr(phitot, phitex, A0, alpha2, beta2)
    phi = phitex + phistrp

    # Water balance
    sol_swb = swb_f(s, rain, sh, sw, sstar, emax, ew, phi, zr,
                    rstar, phitex, kstex, m_prime, ksstrp, m, dt)
    sr = sol_swb['s']
    lk = sol_swb['Lk']

    # Millennial
    pom = pom_f(s=sr, P=P0, A=A0, B=B0, dt=dt, Fi=npp, **w)
    doc = doc_f(s=sr, P=P0, M=M0, B=B0, DOC=DOC0, A=A0, lk=lk, dt=dt, zr=zr, phi=phi, Fi=npp, **u)
    agg = agg_f(s=sr, M=M0, P=P0, A=A0, B=B0, dt=dt, **v)
    maom = maom_f(s=sr, DOC=DOC0, A=A0, M=M0, B=B0, dt=dt, **h)
    bio = bio_f(s=sr, DOC=DOC0, B=B0, dt=dt, phi=phi, **c)

    P0 = pom['P']
    DOC0 = doc['DOC']
    A0 = agg['A']
    M0 = maom['M']
    B0 = bio['B']
    Fa0 = pom['Fa']
    Fpa0 = pom['Fpa']
    Flb0 = doc['Flb']
    Fl0 = doc['Fl']
    Flm0 = doc['Flm']
    Fpl0 = doc['Fpl']
    Fbm0 = doc['Fbm']
    Fld0 = maom['Fld']
    Fal0 = doc['Fal']
    Flm0 = maom['Flm']
    
    output = {'s': sr, 'phi': phi, 'ksstr': ksstrp, 'P': P0, 'DOC': DOC0, 'A': A0, 'M': M0, 'B': B0, 'Flb' : Flb0, 'Fl' : Fl0, 'Flm' : Flm0,               'Fpl' : Fpl0, 'Fbm' : Fbm0, 'lk':lk, 'Fa': Fa0, 'Fpa': Fpa0, 'Fld': Fld0, 'Fal':Fal0, 'Flm':Flm0}
    return output


def long_run(s, rain, npp, P0, A0, B0, M0, DOC0, alpha1, beta1, alpha2, beta2, soilp, w, u, v, h, c, dt):
    """
    Run the coupled swb and millennial model for long-term
    Parameters
    -------

    """
    # print(f"Long run A0: {A0}")
    nr = len(rain)
    s_out = np.zeros(nr)
    phi_out = np.zeros(nr)
    ksstr_out = np.zeros(nr)
    P_out = np.zeros(nr)
    DOC_out = np.zeros(nr)
    A_out = np.zeros(nr)
    M_out = np.zeros(nr)
    B_out = np.zeros(nr)
    Fa_out = np.zeros(nr)
    Fpa_out = np.zeros(nr)
    Flb_out = np.zeros(nr)
    Fl_out = np.zeros(nr)
    Flm_out= np.zeros(nr)
    Fpl_out = np.zeros(nr)
    Fbm_out = np.zeros(nr)
    Fld_out = np.zeros(nr)
    Fal_out = np.zeros(nr)
    Flm_out = np.zeros(nr)
    lk_out = np.zeros(nr)
    
    
    # for i in range(nr):
    for i in tqdm.tqdm(range(nr)):
        y = swb_millennial(s, rain[i], npp[i], P0, A0, B0, M0, DOC0, alpha1, beta1, alpha2, beta2, soilp, w, u, v, h, c,
                           dt)
        s_out[i] = y['s']
        phi_out[i] = y['phi']
        ksstr_out[i] = y['ksstr']
        P_out[i] = y['P']
        DOC_out[i] = y['DOC']
        A_out[i] = y['A']
        M_out[i] = y['M']
        B_out[i] = y['B']
        Fa_out[i] = y['Fa']
        Fpa_out[i] = y['Fpa']
        Flb_out[i] = y['Flb']
        Fl_out[i] = y['Fl']
        Flm_out[i] = y['Flm']
        Fpl_out[i] = y['Fpl']
        Fbm_out[i] = y['Fbm']
        Fld_out[i] = y['Fld']
        Fal_out[i] = y['Fal']
        Flm_out[i] = y['Flm']
        lk_out[i] = y['lk']
        
        s = y['s']
        P0 = y['P']
        DOC0 = y['DOC']
        A0 = y['A']
        M0 = y['M']
        B0 = y['B']
        Fa = y['Fa']
        Fpa = y['Fpa']
        Flb = y['Flb']
        Fl = y['Fl']
        Flm = y['Flm']
        Fpl = y['Fpl']
        Fbm = y['Fbm']
        Fld = y['Fld']
        Fal = y['Fal']
        Flm = y['Flm']
        lk = y['lk']
        
        pass

    output = {'s': s_out, 'phi': phi_out, 'ksstr': ksstr_out,
              'P': P_out, 'DOC': DOC_out, 'A': A_out, 'M': M_out,
              'B': B_out, 'Fa': Fa_out, 'Fpa': Fpa_out, 'Flb': Flb_out, 'Fl': Fl_out, 'Flm': Flm_out, 'Fpl': Fpl_out, 'Fbm': Fbm_out, 'Fld':                Fld_out, 'Fal': Fal_out, 'Flm': Flm_out, 'lk': lk_out}
    output = pd.DataFrame(output)

    y1 = int(len(output) / (365 * 1/dt))
    years_index = np.repeat(np.arange(1, y1+1), len(output) / y1)
    output["Years"] = years_index
    return output

def run_julia(rain, npp, kstex, kstot, phitex, phitot, alpha1, beta1, alpha2, beta2):
    # Initial conditions
    s = 0.3
    P0 = 139.416
    DOC0 = 46.094
    A0 = 786.332
    M0 = 1140.25
    B0 = 124.242
    dt = 1 / 24
    # Parameters
    # POM parameters ---------------------------------
    w = {'pi': 0.66,
         'kb': 0.0004,
         'st': 1.1,
         'w1': 30,
         'w2': 9,
         'sfc': 0.55,
         'sp': 0.05,
         'vpa': 5,
         'kpa': 100,
         'amax': 1000,
         'vpl': 3,
         'kpl': 100,
         'kpe': 12}

    # DOC parameters -----------------------------------------
    u = {'vpl': 3,
         'vlm': 0.38,
         'kpl': 100,
         'kpe': 12,
         'val': 3,
         'kal': 100,
         'kae': 12,
         'klm': 0.25,
         'Qmax': 1239.20,
         'a_star': 0.794,
         'ms_star': 1.5,
         'ns_star': 2,
         'mg_star': 1.5,
         'ng_star': 2,
         'KDOC': 190,
         'KO': 0.005,
         'pi': 0.66,
         'st': 1.1,
         'w1': 30,
         'w2': 9,
         'sfc': 0.55,
         'kmm': 0.005,
         'sp': 0.05}

    # Aggregate parameters --------------------------
    v = {'amax': 1000,
         'vpa': 5,
         'kpa': 100,
         'val': 3,
         'kal': 100,
         'kae': 12,
         'kb': 0.0004,
         'st': 1.1,
         'w1': 30,
         'w2': 9,
         'sfc': 0.55,
         'sp': 0.05}

    # MAOM parameters -------------------------------------
    h = {'klm': 0.25,
         'Qmax': 1239.20,
         'st': 1.1,
         'w1': 30,
         'w2': 9,
         'sfc': 0.55,
         'sp': 0.05}

    # BIO parameters -----------------------------------
    c = {'vlm': 0.38,
         'a_star': 0.794,
         'ms_star': 1.5,
         'ns_star': 2,
         'mg_star': 1.5,
         'ng_star': 2,
         'KDOC': 190,
         'KO': 0.005,
         'st': 1.1,
         'CUE': 0.35,
         'kmm': 0.005}

    # Water balance model -------------------------------
    soilp = {'sh': 0.19,
             'sw': 0.24,
             'sstar': 0.57,
             'emax': 0.5 / 100,
             'ew': 0.05 / 100,
             'zr': 20 / 100,
             'rstar': 0.0,
             'kstex': kstex,  
             'kstot': kstot,  
             'phitex': phitex,  
             'phitot': phitot, 
             'm_prime': 0.286,  
             'm': 1.5  
             }
    x = long_run(s, rain, npp, P0, A0, B0, M0, DOC0, alpha1,
                 beta1, alpha2, beta2, soilp, w, u, v, h, c, dt)
    return x
