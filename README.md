# PLPAK
Package for Low Temperature Plasma Kinetics

PLPAK is a library built for zero-dimensional analysis of plasma-kinetic systems based on provided reaction mechanism set. For non-equilibrium reactions multiple temperatures might be required to specify the rates of reactions and on top of that rate equations for energy exchanges in different modes are also essential. PLPAK is built to streamline the process of building such reaction systems, testing them and solving them.

Contact Details :
* Sagar Pokharel : pokharel_sagar@tamu.edu / pokharelsagar1015@gmail.com
* Albina Tropina : atropina@tamu.edu

## Usage
If you utilize the package in your research, please cite the following papers:
- Pokharel, S., & Tropina, A. A. (2023). Self-consistent model and numerical approach for laser-induced non-equilibrium plasma. Journal of Applied Physics, 134(22), 223301. https://doi.org/10.1063/5.0175177




# Installation
- Tested on the following
    - Python 3.9.20
    - Cantera==2.6.0
    - matplotlib==3.9.2
    - Scipy and Sympy (for full functionality) 

- Clone the repository or download the zip file.
- Add path to the directory containing the package to the environment variable and import.
    ```python
    dir_plpak = os.path.dirname("path_to/PLPAK/src/plpak")
    sys.path.append(dir_plpak)
    import PLPAK as pl
    ```

# **[More Details](./src/readme.md)**
For more details, see the **[src/readme.md](./src/readme.md)**.

## Directory Structure

- `src/` contains the source code for the package.
- `example/` contains examples of how to use the package and relevant .py script to load the library and run the simulations.
- `data/` contains the data files required for the examples. Reaction mechanisms for $N_2$-plasma and experimental data for comparison are included. Please contact developers for use of air-plasma model.


### Test
- Run the example/solveWithMech.py to run the simulation and save the data.
- The comparison of simulated results against experiments of Paper

    <img src="./examples/compModelsN2.png" alt="Comparison of Models" width="400"/>

    *Figure: Comparison of Models*
