# Regional demo bank: 20,000 simulations

Generated with the seeded synthetic 80-region connectome (demo seed 42). WF and WNF vary independently in [2,4], and G in [1.5,2.5], with b=0.16. Contains six raw directional ATM features, signal statistics, avalanche counts, simulation IDs and noise seeds.

The parameter design and generation manifest accompany the complete CSV. No empirical features, clinical data or subject posteriors are included. This bank illustrates the regional pipeline and does not reproduce the original-connectome study results.

The regional notebook loads this bank by default in demo mode when RUN_BANK=False. To generate a new bank instead, set RUN_BANK=True and choose a distinct output folder.
