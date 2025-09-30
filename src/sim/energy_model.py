"""Radio Energy Model for Wireless Sensor Networks.

Implements the energy consumption model for wireless communication
in sensor networks based on transmission distance and packet size.
"""

import math
from typing import Union


class RadioEnergyModel:
    """Radio energy consumption model for wireless sensor networks.
    
    Implements the first-order radio model with distance-dependent
    energy consumption for transmission and fixed energy for reception.
    
    Args:
        e_elec: Electronics energy per bit (J/bit)
        eps_amp: Amplifier energy per bit per meter^n (J/bit/m^n)
        n: Path loss exponent (typically 2 for free space)
        packet_bits: Packet size in bits
    """
    
    def __init__(self, e_elec: float = 50e-9, eps_amp: float = 100e-12, 
                 n: int = 2, packet_bits: int = 512):
        self.e_elec = e_elec  # Electronics energy (J/bit)
        self.eps_amp = eps_amp  # Amplifier energy (J/bit/m^n)
        self.n = n  # Path loss exponent
        self.packet_bits = packet_bits  # Packet size (bits)
    
    def tx_energy(self, distance_m: float) -> float:
        """Calculate transmission energy for given distance.
        
        Args:
            distance_m: Transmission distance in meters
            
        Returns:
            Energy consumed for transmission in Joules
        """
        electronics_energy = self.e_elec * self.packet_bits
        amplifier_energy = self.eps_amp * self.packet_bits * (distance_m ** self.n)
        return electronics_energy + amplifier_energy
    
    def rx_energy(self) -> float:
        """Calculate reception energy (distance-independent).
        
        Returns:
            Energy consumed for reception in Joules
        """
        return self.e_elec * self.packet_bits
    
    def __repr__(self) -> str:
        return (f"RadioEnergyModel(e_elec={self.e_elec}, eps_amp={self.eps_amp}, "
                f"n={self.n}, packet_bits={self.packet_bits})")
