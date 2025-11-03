## PLPAK Reaction Mechanism Format (Short Description)

### This format extends Cantera for custom reaction attributes.


**1. `newAttributeSet` (Global Settings):**
   - `unit`: Default unit for `newAttribute` if not specified there.
   - `vars`: Maps short names (e.g., `Te`) to variable names used in expressions.

**2. `reactions` (List of Reactions):**
   - Each reaction is a dictionary.
   - **`equation`**: Chemical reaction string (e.g., `A + B => C`).
   - **`rate-constant`**: Arrhenius parameters `{A: ..., b: ..., Ea: ...}`. *Always required*, acts as a placeholder if `newAttribute` is used.
   - **`type`**: Reaction type (e.g., `elementary`).
   - **`duplicate`**: `True` if the same reaction can appear multiple times.
   - **`newAttribute`** (Custom Reaction Property - Optional):
     - Can be a dictionary with `unit` and an attribute name (`Bolsig`, `K`, etc.).
       - `Bolsig`: List of numerical coefficients.
       - `K`: String expression for rate constant using `vars`.
     - Or a direct string expression using `vars` (unit from `newAttributeSet`).
   - **`energyExchange`** (Energy Transfer - Optional):
     - `type`: How energy is exchanged (`ratios`, `valueseV`) between the three tempearture modes.
     - `R`: List of ratios for product energy distribution.
     - `val`: List of energy values (eV) for each energy mode.
     - `fix`: `True` to prevent automatic energy balancing. If `True` uses given energy exchange values and doesnt overwride by energy balance.

```yaml
newAttributeSet:
unit: m
vars: {T: 'T',Tv: 'Tv', Te: 'Te'}

reactions: 
- equation: ele + N2 => ele + N2_A
  rate-constant: {A: 0, b: 0, Ea: 1}
  newAttribute: {unit: m, Bolsig: [-24.62, ...]}
  energyExchange: {type: valueseV, val: [0, 0, 6.17], fix: True}

- equation: N2 + N2_ap => N2_B + N2
  rate-constant: {A: 0, b: 0, Ea: 1}
  newAttribute: {unit: cm, K: '(2.8e-13)'}

```
