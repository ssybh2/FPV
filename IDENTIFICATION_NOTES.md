# Motor identification notes

Source: the uploaded ZIP containing three repeated 1405 + 3-inch propeller + 30 A ESC test runs.

The bundled `motor_lut.csv` averages the three `mean_step_data_by_PWM.csv` files at each PWM point from 1000 to 1800 microseconds.

Mean identified coefficients:

- Kt = 1.3287717252618608e-6 N/(rad/s)^2
- sample standard deviation across three fits = 1.3017266303062117e-8
- Kq = 1.772957417327994e-8 N m/(rad/s)^2
- sample standard deviation across three fits = 1.0199367644005286e-10

The corresponding mean Kq/Kt ratio is approximately 0.01334 m.

## Important limitation: motor time constant

The files named as high-frequency recordings in the supplied archive do not contain a usable high-frequency time series. The low-frequency records are too slow for precise identification of a small multirotor motor/propeller transient.

Therefore `motor_tau_s = 0.10` is a placeholder nominal value, not a measured fact.

For the final digital twin, record PWM/command, RPM and preferably thrust at >= 500 Hz during both rising and falling steps. We should then identify at least:

- pure command/ESC delay Td,
- spin-up tau_up,
- spin-down tau_down,
- possible dependence of tau on operating point.
