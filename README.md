Hybrid Optimization and Surrogate Modeling for HJT-PV/PEM Systems
This repository contains the computational framework for optimizing and controlling a reconfigurable Heterojunction (HJT) PV–PEM hydrogen production system. The project transitions from high-fidelity, metaheuristic multi-objective optimization to a high-speed hybrid surrogate model capable of real-time operational decision-making.

🚀 Overview
Directly coupling PV arrays with PEM electrolyzers requires precise configuration to maintain high efficiency under fluctuating weather conditions. This framework addresses two core challenges:

Sizing and Trade-off Analysis: Using NSGA-II to explore the feasible design space (PV strings and PEM temperature).

Real-Time Control: Replacing computationally expensive physics-based loops with a Hybrid Surrogate Model that is up to 900x faster for daily forecasting.

📂 Repository Structure
1. NSGA-II_application_1MW.py (Metaheuristic Optimization)
This script implements the NSGA-II algorithm to identify the Pareto-optimal front for the system.

Degrees of Freedom: Number of parallel HJT PV strings and PEM operating temperature.

Objectives: Maximizing hydrogen production and STH efficiency while minimizing energy conversion losses and configuration costs.

Purpose: Systematic exploration of the search space without requiring predefined weighting factors.

2. NSGA-II plots_1MW.py (Visualizations)
A dedicated module for generating high-quality visualizations of the optimization results, including:

3D/4D Pareto Fronts: Illustrating the trade-offs between competing objectives.

Correlation Matrices: Analyzing the relationships between environmental inputs (irradiance) and system performance.

MCDA Weight Distributions: Boxplots showing CRITIC-derived weight variability.

3. Better_prediction.py (Hybrid Surrogate Model)
The core of the real-time decision framework. It maps environmental features directly to optimal system configurations.

Machine Learning: Utilizes Random Forest Regression to predict PV configurations and PEM temperatures.

Reliability Filter: Employs a k-d tree-based nearest-neighbor search to verify prediction validity against the training database.

Physical Logic: Incorporates rule-based feasibility checks to ensure valid operation under low irradiance or out-of-distribution conditions.

🛠 Methodology
The workflow follows a two-stage process:

Offline Optimization: NSGA-II evaluates thousands of candidate configurations using physics-based simulations to generate a robust database of optimal solutions.

Online Deployment: The hybrid model learns from this database to provide near-instantaneous recommendations. This avoids the "prohibitively slow" iterative cycles of traditional physics-based methods, enabling responsive control under continuously changing weather.

💻 Performance Baseline
All execution times and benchmarks were recorded on the following hardware:

Processor: AMD Ryzen 9 7950X3D 16-Core Processor (4.20 GHz).

Optimization Speedup: The surrogate model achieves an 18x speedup for annual horizons and a 900x speedup for next-day forecasts compared to the full physics-based model.

📖 Citation
If you use this code or methodology in your research, please cite:

[Insert your paper title and DOI here]
