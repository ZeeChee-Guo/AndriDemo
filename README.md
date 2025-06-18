# Andri Demo

## 1. Overview

<<<<<<< HEAD
A system for Anomaly detection in the presence of Drift, and enables users to interactively co-explore the interaction
of anomalies and drift.

## 2. Quick Start

### 1. Prerequisites
Python 3.9.0

### 2. Clone the repository and enter the project directory

```bash
git clone https://github.com/ZeeChee-Guo/AndriDemo.git
cd AndriDemo
```
### 3. Clone the repository and enter the project directory:
On **Windows**:
 ```
 python -m venv venv
 venv\Scripts\activate
 ```

On **macOS/Linux**:
```
python3 -m venv venv
source venv/bin/activate
```
### 4. Run the Flask server

```bash
python app.py
```

Once the server is running, open your browser and navigate to http://localhost:5000/ to start using AnDriDemo.


## 3. Architecture
The architecture of AnDri consists of three main modules: The system is designed as a web application with a Python Flask backend and a JavaScript-based frontend for interactive visualization and user labeling.
![System Architecture](static/img/architecture.png)
