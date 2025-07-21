# DFL-Tutorial-2025

## Installation

### Julia part
1. Install julia if not already installed. See instructions at [Julia Installation](https://julialang.org/downloads/).
2. Install Pluto
   ```julia
   using Pkg
   Pkg.add("Pluto")
   ```

### Python part
1. Install Python if not already installed.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Tutorial(s)
### Julia Tutorial
1. Open Julia and run the following commands:
   ```julia
   using Pluto
   Pluto.run()
   ```
2. In the Pluto interface, open the notebooks

### Python Tutorial
1. Open a terminal
2. Run the notebook with marimo
   ```bash
   marimo run pyEPO/notebook.py
   ```

