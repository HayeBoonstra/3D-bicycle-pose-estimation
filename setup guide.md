# Physically Plausible Bicycle Motion Trajectories with MuJoCo

This repository generates physically plausible bicycle motion trajectories using [MuJoCo](https://mujoco.readthedocs.io/en/stable/).

## Setup

### Linux

1. **Activate the virtual environment**

   This repository was written for **Python 3.12**. Source the virtual environment named `test-env`:

   ```bash
   source test-env/bin/activate
   ```

   > When the virtual environment is active, its name will appear at the beginning of your terminal prompt.

2. **Install required Python packages**  
   Run the following command _inside the activated environment_:

   ```bash
   pip install -r test-requirements.txt
   ```

---

### Windows

*MuJoCo requires the Visual C++ Redistributable packages.*  
If these are not installed, you can find a complete installer here:  
[Visual C++ Redistributable Runtime Package All-in-One](https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/)

*Python 3.12* can be downloaded from the [Microsoft Store](https://www.microsoft.com/store/productId/9PJPW5LDXLZ5).

1. **Activate the virtual environment**  
   Open a terminal and navigate to the repository directory. Then run:

   ```powershell
   .\win-env\Scripts\Activate.ps1
   ```
   > When the virtual environment is active, its name will appear at the beginning of your terminal prompt.

2. **Install required Python packages**

   ```powershell
   pip install -r test-requirements.txt
   ```

---

You're all set! 🎉  
Proceed to running the simulations or tests as described in the repo.