#!/bin/env python3
"""
Modules to define useful constants
"""

import numpy as np

class PhysicsConstants:
    def __init__(self):
        self.proton_mass = 0.938272081 # target mass - previously: M  NOTE!: aao_gen system uses 0.938 for proton mass
        self.neutral_pion_mass = 0.1349766

        self.electron_mass = 0.5109989461 * 0.001 # electron mass- previously: me
        self.electron_beam_energy = 10.604 # beam energy- previously: ebeam

        self.electron_beam_momentum_magnitude = np.sqrt(self.electron_beam_energy ** 2 - self.electron_mass ** 2) # beam electron momentum - previously: pbeam
        # electron beam 4 momentum
        self.electron_beam_4_vector = [self.electron_beam_energy, 0, 0, self.electron_beam_momentum_magnitude] # beam 4 vector - previously: beam4
        self.electron_beam_3_vector = [0, 0, self.electron_beam_momentum_magnitude] # beam vector - previously: beam
        self.target_3_vector = [0, 0, 0] # target vector - previously: target
        self.target_4_vector = [self.proton_mass, 0, 0, 0] # target 4 vector - previously: target4




