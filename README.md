# LSM and Etching Simulation

## 🛠️ Dependencies

* **Python** 3.10.19
* **NumPy** 2.2.6
* **Matplotlib** 3.10.8
* **Scikit-image** 0.25.2
* **Tqdm** 4.67.1

### Installation
```bash
git clone [https://github.com/leftover211/FSPS.git](https://github.com/leftover211/FSPS.git)
cd FSPS
pip install numpy matplotlib scikit-image tqdm
```

## 📂 File Structure

```bash
.
├── etching_simulation.py     
├── surface_visualization.py   
├── simulation_images/       
└── README.md
```

## 💻 Usage

### 1. Run Etching Simulation
To simulate the etching profile evolution (Wet vs Dry) and generate result images:
```bash
python etching_simulation.py
```
* It generates 3D visualization results (`.svg`) for both Isotropic and Anisotropic cases.
* **Output Location:** `simulation_images/` directory.

### 2. Run Concept Visualization
To visualize the fundamental mathematical concept of the Level Set Method :
```bash
python surface_visualization.py
```
* This script visualizes the 3D level set function $\phi$ and its zero-level contour.
* **Output Location:** `levelset_images/` directory.

---


## 📝 Author's Note
This project was used in the presentation for the **Micro/Nano Mechanical Engineering** course.
