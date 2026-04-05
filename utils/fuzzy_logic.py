import numpy as np
import skfuzzy as fuzz

def evaluate_fuzzy_risk(ndvi, vh_norm):
    """
    Evaluates the fuzzy risk of deforestation given NDVI and VH backscatter (normalized).
    vectorized for fast execution over images.
    
    Returns a matrix of risk scores (0 to 100).
    """
    # Define universes
    u_ndvi = np.linspace(-1, 1, 100)
    u_vh = np.linspace(0, 1, 100)
    u_risk = np.linspace(0, 100, 100)

    # Membership functions for NDVI
    ndvi_lo = fuzz.trapmf(u_ndvi, [-1, -1, 0.1, 0.3])
    ndvi_md = fuzz.trimf(u_ndvi, [0.1, 0.4, 0.6])
    ndvi_hi = fuzz.trapmf(u_ndvi, [0.4, 0.7, 1.0, 1.0])

    # Membership functions for VH
    vh_lo = fuzz.trapmf(u_vh, [0.0, 0.0, 0.2, 0.4])
    vh_md = fuzz.trimf(u_vh, [0.2, 0.5, 0.7])
    vh_hi = fuzz.trapmf(u_vh, [0.5, 0.8, 1.0, 1.0])

    # Membership functions for Risk
    risk_low = fuzz.trapmf(u_risk, [0, 0, 20, 40])
    risk_med = fuzz.trimf(u_risk, [20, 50, 80])
    risk_high = fuzz.trapmf(u_risk, [60, 80, 100, 100])

    # Interpolate input values to get membership levels
    # Since ndvi and vh_norm are 2D arrays, this is vectorized
    ndvi_lo_val = fuzz.interp_membership(u_ndvi, ndvi_lo, ndvi)
    ndvi_md_val = fuzz.interp_membership(u_ndvi, ndvi_md, ndvi)
    ndvi_hi_val = fuzz.interp_membership(u_ndvi, ndvi_hi, ndvi)

    vh_lo_val = fuzz.interp_membership(u_vh, vh_lo, vh_norm)
    vh_md_val = fuzz.interp_membership(u_vh, vh_md, vh_norm)
    vh_hi_val = fuzz.interp_membership(u_vh, vh_hi, vh_norm)

    # Apply Rule Base (Mamdani operations - min for AND, max for OR)
    
    # Rule 1: IF NDVI Low AND VH Low THEN Risk High
    r1 = np.fmin(ndvi_lo_val, vh_lo_val)
    # Rule 2: IF NDVI Low AND VH Medium THEN Risk High
    r2 = np.fmin(ndvi_lo_val, vh_md_val)
    # Risk High Activation
    act_high = np.fmax(r1, r2)
    
    # Rule 3: IF NDVI Medium AND VH Low THEN Risk Medium
    r3 = np.fmin(ndvi_md_val, vh_lo_val)
    # Rule 4: IF NDVI Medium AND VH Medium THEN Risk Medium
    r4 = np.fmin(ndvi_md_val, vh_md_val)
    # Risk Medium Activation
    act_med = np.fmax(r3, r4)
    
    # Rule 5: IF NDVI High OR VH High THEN Risk Low
    r5 = np.fmax(ndvi_hi_val, vh_hi_val)
    # Risk Low Activation
    act_low = r5
    
    # Aggregate and Defuzzify
    # We will use a fast defuzzification approach based on weighted average of centers
    # Centers of the risk MFs
    c_low = 20
    c_med = 50
    c_high = 80
    
    sum_weights = act_low + act_med + act_high
    # Avoid division by zero
    sum_weights[sum_weights == 0] = 1e-5
    
    risk_matrix = (act_low * c_low + act_med * c_med + act_high * c_high) / sum_weights
    
    return risk_matrix
