# [PLPAK] : Package for Low-temperature Plasma And Kinetics

## What is PLPAK ?
PLPAK is a library built for zero-dimensional analysis of plasma-kinetic systems based on provided reaction mechanism set. For non-equilibrium reactions multiple temperatures might be required to specify the rates of reactions and on top of that rate equations for energy exchanges in different modes are also essential. PLPAK is built to streamline the process of building such reaction systems, testing them and solving them.

## Capabilities
The reaction mechanisms are provided in format compatible to Cantera. The reaction rates with function of various temperatures in the system is specififed in the reaction mechanim file. The multi-mode energy exchange for specific reactions is also specified in the reaction mechanism file. Based on this, PLPAK can solve the system and also produce the ODE system in different programming languages, viz. Python, C++, Fortran, Matlab, Julia, etc.

### Sample Mechanism
```yaml
- equation: 2 ele + N4p => ele + 2 N2 
  rate-constant: {A: 0, b: 0, Ea: 1}
  type: elementary
  newAttribute: (7.0e-32*(300/Te)**4.5)
  energyExchange: {type: ratios, R: [0.0, 0.0, 1.0]}

- equation: ele + N3p => N + N2 
  rate-constant: {A: 0, b: 0, Ea: 1}
  type: elementary
  newAttribute: (2.0e-13*sqrt(300/Te))

- equation: ele + N2p => 2 N 
  rate-constant: {A: 0, b: 0, Ea: 1}
  type: elementary
  newAttribute: (1.8e-13*(300.0/Te)**0.39)
  energyExchange: {type: valueseV, val: [-3.5, 0.0, 0.0], fix: False}

```

