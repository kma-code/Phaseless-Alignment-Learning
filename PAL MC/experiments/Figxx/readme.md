This experiment demonstrates how PAL relies on prospective coding to learn top-down weights.
We do so by overriding the prospective_voltage function defined in microcircuit.py, see runner.py.
The parameter files in the subfolders of this directory contain the parameter taur, which if non-zero controls the 'prospectivity' of our model via:

prospective_voltage = uvec_old + taur * tau * (uvec - uvec_old) / dt

