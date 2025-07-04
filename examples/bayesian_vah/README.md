# Emulation using PCSK

Refer to the Jupyter notebook `emulation_and_calibration/emulator_validation.ipynb` for the construction of a PCSK 
emulator for the VAH model.  This notebook is adapted from the repository <https://github.com/danOSU/Bayesian_parameter_inferece_for_VAH> 
and only concerns the construction of the emulator.  The full emulation and 
calibration example should be referred to the original repository. 

Additional requirements for this example include:

```bash
seaborn==0.13.2
```

Upon running the notebook, an `emulated_vs_simulated.png` figure, 
and an `PCSK/R2.png` figure will be produced and saved.