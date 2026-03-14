import math

class Motor:

    def __init__(self, KV, voltage, max_current):
        self.KV = KV
        self.voltage = voltage
        self.max_current = max_current
    
    def torque_available(self, rpm):
        """Calculates the motor torque available at a given RPM."""
        KV = self.KV
        V  = self.voltage
        Imax = self.max_current

        # torque constant (N·m/A)
        kt = 1.0 / (KV * (2 * math.pi / 60))

        T_stall = kt * Imax
        RPM_max = KV * V

        return max(0.0, T_stall * (1.0 - rpm / RPM_max))