# Computational Fluid Dynamics (CFD) Solvers: From Scratch

**Course:** NPTEL - Computational Fluid Dynamics (IIT Madras)
**Syllabus Focus:** FDM, Finite Volume Method (FVM), and Navier-Stokes Solvers.
**Status:** 🟢 Active (Jan 2026)

---

## 📌 Research Objective
To replace "Black Box" commercial software dependence with a First-Principles understanding of Fluid Dynamics. This repository contains custom Python implementations of the NPTEL course modules, focusing on:
1.  **Numerical Stability:** Analyzing consistency and convergence (Weeks 5-6).
2.  **Solver Architecture:** Writing custom linear algebraic solvers (Weeks 9-10).
3.  **Complex Flows:** Modeling Turbulent and Compressible flows (Weeks 7, 12).

---

## 🛠️ Tech Stack
* **Language:** Python 3.x (replacing the course's standard MATLAB approach).
* **Core Libraries:** `NumPy` (Matrix Operations), `Matplotlib` (Vector Field Visualization).
* **Applied Math:** Finite Difference (FDM), Finite Volume (FVM), TDMA Solvers.

---

## 📂 Implementation Modules (Syllabus Mapped)

### 🔹 Phase 1: Analytical Benchmarks (Weeks 1-4)
* **Duct Flow Solvers:**
  * Calculating fully developed flow in **Rectangular Ducts** (Week 1).
  * Calculating flow in **Triangular Ducts** (Week 2).
* **Governing Equations:** Python scripts demonstrating Conservation of Mass/Momentum principles.

### 🔹 Phase 2: The Numerical Engine (Weeks 5-6)
* **Discretization:** Implementing Finite Difference Approximations (Forward, Backward, Central).
* **Stability Analysis:** Code demonstrations of **Consistency, Stability, and Convergence**.
  * *Project:* Visualizing the "CFL Condition" (Showcasing when a simulation blows up).

### 🔹 Phase 3: The Navier-Stokes Solvers (Weeks 7-8)
* **Compressible Flow:** Solvers for high-speed flow regimes.
* **Incompressible Flow:** Solving standard low-speed aerodynamics problems.

### 🔹 Phase 4: Advanced Matrix Solvers (Weeks 9-10)
* **Linear Algebra from Scratch:** Custom implementation of:
  * Gaussian Elimination.
  * TDMA (Tri-Diagonal Matrix Algorithm).
  * Iterative Methods (Jacobi, Gauss-Seidel, SOR).
* *Note: These matrix operations form the backbone of both CFD and Deep Learning architectures.*

### 🔹 Phase 5: Modern Techniques (Weeks 11-12)
* **Finite Volume Method (FVM):** Moving beyond grids to control volumes (Industry Standard).
* **Turbulence Modelling:** Basic implementation of RANS (Reynolds-Averaged Navier-Stokes) concepts.

---

## 🚀 Future Integration (AI + CFD)
* This codebase serves as the "Ground Truth" generator for my future work in **Physics-Informed Neural Networks (PINNs)**.
* Goal: Use data generated here (Weeks 7-8) to train AI models that predict flow fields 100x faster.

---

### 👤 Author
**Suhail Majeed Sheikh** | *Mechanical Engineer & AI Researcher*
*NIT Srinagar '2024*