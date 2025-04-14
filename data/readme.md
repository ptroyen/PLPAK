PLPAK Reaction Mechanism Format

```yaml
newAttributeSet:
  unit: m # Default unit for new attributes
  vars: {T: 'T',Tv: 'Tv', Te: 'Te'} # Var mappings

reactions:
  # equation: 'R1 => P1 + P2'
  # rate-constant: {A: ..., b: ..., Ea: ...} # Arrhenius (placeholder if newAttribute used)
  # type: elementary/three-body/...
  # duplicate: True/False # Same reaction can appear twice
  # newAttribute: # Optional, new reaction expression
  #   unit: ...
  #   Bolsig: [...] # List of coefficients
  #   K: '(expression)' # Rate constant expression using vars
  #   (expression) # Directly using vars, unit from newAttributeSet
  # energyExchange: # Optional, energy exchange details
  #   type: ratios/valueseV/...
  #   R: [...] # Ratios for energy distribution
  #   val: [...] # Energy values (eV)
  #   fix: True # Fix specified values


  - equation: ele + N2 => ele + N2_A
    rate-constant: {A: 0, b: 0, Ea: 1}
    newAttribute: {unit: m, Bolsig: [-24.62, ...]}
    energyExchange: {type: valueseV, val: [0, 0, 6.17], fix: True}

  - equation: N2 + N2_ap => N2_B + N2
    rate-constant: {A: 0, b: 0, Ea: 1}
    newAttribute: {unit: cm, K: '(2.8e-13)'}

```
