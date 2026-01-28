import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ogstools as ot
from scipy.special import exp1

def calc_u(r, S, T, t):
    """Calculate and return the dimensionless time parameter, u."""
    return r**2 * S / 4 / T / t


def theis_drawdown(t, S, T, Q, r):
    """Calculate and return the drawdown s(r,t) for parameters S, T.

    This version uses the Theis equation, s(r,t) = Q * W(u) / (4.pi.T),
    where W(u) is the Well function for u = Sr^2 / (4Tt).
    S is the aquifer storage coefficient,
    T is the transmissivity (m2/day),
    r is the distance from the well (m), and
    Q is the pumping rate (m3/day).
    """
    u = calc_u(r, S, T, t)
    return Q / 4 / np.pi / T * exp1(u)


# Analytical solution setup
Q = 2  # Pumping rate from well (m3/day)
r = 1  # Distance from well (m)
S, T = 1e-7, 1e-2 # meter and days
t = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4])
# Hydraulic head over time at given distance
s = theis_drawdown(t, S, T, Q, r)
p_conversion = 1/98.1

# Plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(t, s*p_conversion, label=f"r = {r} m")
ax.set(
    xlim=(0,4),
    #ylim=(1.6, 2.6),
    title="Theis: Analytical solution",
    xlabel=r"$t\;/\;\mathrm{days}$",
    ylabel=r"$Pressure\;/\;\mathrm{MPa}$",
)
ax.legend()
ax.grid()
plt.savefig("theis-stimtec-time.png", dpi=600)
plt.show()

# Hydraulic head over radius at given time
t = 1  # Time in days
r = np.arange(0, 51, 1) # radius in 
u = calc_u(r, S, T, t)
s = theis_drawdown(t, S, T, Q, r)

# Plotting
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(r, s*p_conversion, label=f"t = {t} day(s)")
ax.set(
    #xlim=(1, 40),
    title="Theis: Analytical solution",
    xlabel=r"$r\;/\mathrm{m}$",
    ylabel=r"$Pressure\;/\;\mathrm{MPa}$",
)
ax.legend()
ax.grid()
plt.savefig("theis-stimtec-radial.png", dpi=600)
plt.show()