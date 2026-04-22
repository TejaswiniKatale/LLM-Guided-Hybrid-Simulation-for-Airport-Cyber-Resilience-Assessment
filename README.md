# LLM-Guided-Hybrid-Simulation-for-Airport-Cyber-Resilience-Assessment

Airport systems rely on tightly connected digital and physical components. When cyber disruptions occur, they can impact both service performance and passenger movement. Traditional airport simulation studies often model either service queues or pedestrian flow, but not both together in a way that supports cyber-resilience analysis.

This project combines:

- **Discrete-Event Simulation (DES)** for service operations
- **Microscopic pedestrian simulation** for passenger movement
- **LLM-based behavior modeling** for passenger decisions under disruption

## Main LLM Contribution

The core contribution of this project is the integration of an LLM to generate **bounded passenger behavior parameters** during degraded information scenarios.

Instead of assigning fixed assumptions manually, the LLM is used to produce realistic behavior-related parameters such as:

- **Misrouting probability**
- **Dwell probability**
- **Dwell-time range**
- **Passenger response under disrupted guidance/display conditions**

These outputs are constrained within predefined bounds so that the simulation remains realistic, stable, and suitable for research analysis.

## Simulation Framework

The framework has 3 main layers:

### 1. DES Layer
Models airport service processes such as:
- Check-in
- Information desk
- Security screening

### 2. Pedestrian Layer
Models passenger movement inside the terminal, including:
- Walking paths
- Congestion
- Route choice
- Waiting behavior

### 3. LLM Layer
Adds adaptive passenger behavior under cyber-disrupted conditions by generating bounded behavioral parameters that influence movement and decision-making.

## Research Objective

The goal of this project is to explore how cyber disruptions propagate through airport operations and affect both:

- **Operational performance**
- **Passenger behavior**

This helps support **exploratory cyber-resilience analysis** for airport systems.

## Scenario Analysis

The framework is designed to compare:
- A **baseline scenario**
- Multiple **cyber disruption scenarios**

Performance is evaluated using:
- Throughput
- Queue length
- Waiting time
- Completion time
- Resource utilization

## Key Idea

The LLM is not used as a general chatbot inside the simulation.  
Instead, it is used as a **behavior generation module** that provides realistic, bounded parameters for disrupted passenger behavior.

This makes the framework:
- More adaptive than fixed-rule models
- More behavior-aware than standard queue simulations
- More suitable for studying cyber-physical impacts in airports

## Applications

This project can support research in:
- Airport cyber resilience
- Passenger flow analysis
- Human behavior under disrupted information systems
- Hybrid simulation of cyber-physical infrastructure

## Future Work

Possible extensions include:
- Larger airport layouts
- More passenger behavior types
- Real-world calibration
- Additional disruption scenarios
- Multi-agent intelligent decision modeling
