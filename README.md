[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/1_SYVk3e)
#  RL-Class-Assignment Documentation

## Project Introduction

### Repository Structure

The repository is organized as follows:
```
├── README.md 
├── requirement.txt # Lists all the necessary dependencies to set up your development environment.
├── best_hiv_agent.pth # Weights of the best agent. 
└── src
    ├── env_hiv.py # Defines the simulation environment your agent will interact with. 
    ├── evaluate.py # Contains the evaluation function to assess your agent's performance. 
    ├── interface.py # Provides the agent interface, outlining the methods your agent must implement. 
    ├── train.py # Training script. 
    └── main.py # Serves as the entry point for automatically evaluating your agent. 
```

### The problem to solve

This challenge is inspired by the 2006 paper by Ernst et al. titled "[Clinical data based optimal STI strategies for HIV: a reinforcement learning approach](https://ieeexplore.ieee.org/abstract/document/4177178)". It is based on the 2004 simulation model of Adams et al. published in the "[Dynamic Multidrung Therapies for HIV: Optimal and STI Control Approaches](https://www.aimspress.com/fileOther/PDF/MBE/1551-0018_2004_2_223.pdf)" paper.  
You don't have to read these papers (although the first might be a great inspiration for your work, while the second might provide better insights as to the physical quantities manipulated). In particular, your agent doesn't have to mimic that of Ernst et al.! Feel free to be creative and develop your own solution (FQI, DQN, evolutionary search, policy gradients: your choice).

You are provided with the simulator class `HIVPatient` of an HIV infected patient's immune system. This simulator follows the Gymnasium interface and provides the usual functions. Your goal is to design a closed-loop control strategy which performs structured treatment interruption, ie. a control strategy which keeps the patient healthy while avoiding prescribing drugs at every time step. 

The `HIVPatient` class implements a simulator of the patient's immune system through 6 state variables, which are observed every 5 days (one time step):
- `T1` number of healthy type 1 cells (CD4+ T-lymphocytes), 
- `T1star` number of infected type 1 cells,
- `T2` number of healthy type 2 cells (macrophages),
- `T2star` number of infected type 2 cells,
- `V` number of free virus particles,
- `E` number of HIV-specific cytotoxic cells (CD8 T-lymphocytes).

The physician can prescribe two types of drugs:
- Reverse transcriptase inhibitors, which prevent the virus from modifying an infected cell's DNA,
- Protease inhibitors, which prevent the cell replication mechanism.

Giving these drugs systematically is not desirable. First, it prevents the immune system from naturally fighting the infection. Second, it might induce drug resistance by the infection. Third, it causes many pharmaceutical side effects which may lead the patient to abandon treatment.  
There are four possible choices for the physician at each time step: prescribe nothing, one drug, or both.

The reward model encourages high values of `E` and low values of `V`, while penalizing giving drugs.

The patient's immune system is simulated via a system of deterministic non-linear equations.

The goals is to write the 'train.py' script to develop an `Agent` class that conforms to the protocol defined in `interface.py`. This class should encapsulate the functionality for your agent, including the ability to:
  - Decide on actions based on observations from the environment.
  - Save what it has learned to disk.
  - Load a previously saved model for further inference or evaluation.

The final grade, out of nine points, is made of two parts.  
First, your agent will be evaluated on the default patient. There are six score thresholds: every time your agent passes a threshold, you gain one point.  
Then, your agent will be evaluated on the population of patients. There are three score thresholds: every time your agent passes a threshold, you gain another point.  
