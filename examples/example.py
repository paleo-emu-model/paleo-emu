"""
Example: train a PCA+GP emulator and predict under scenario forcing.

All scenarios are defined in example_PCA_GP.yml.

Usage:
  runner.predict()                                           # all scenarios
  runner.predict("SSP585")                                   # case 1: single file
  runner.predict("past800ka_ens")                            # case 2: all 90 members
  runner.predict("past800ka_ens", member="1-10")             # case 2: members 1–10
  runner.predict("past800ka_var")                            # case 3: all vars
  runner.predict("past800ka_var", var="sst")                 # case 3: one var
  runner.predict("past800ka_var", var=["sst", "precip"])     # case 3: subset
"""

from paleo_emu import PaleoEmuRunner

runner = PaleoEmuRunner("example_PCA_GP.yml")

runner.train()

runner.predict("SSP585")
