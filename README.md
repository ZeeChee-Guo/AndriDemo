# Andri Demo

## 1 Overview

A system for Anomaly detection in the presence of Drift, and enables users to interactively co-explore the interaction
of anomalies and drift.

## 2. Quick Start

### 2.1 Prerequisites
Python 3.9.0

### 2.2 Clone the repository and enter the project directory

```bash
git clone https://github.com/ZeeChee-Guo/AndriDemo.git
cd AndriDemo
```
### 2.3 Clone the repository and enter the project directory:
On Windows:
 ```
 python -m venv venv
 venv\Scripts\activate
 ```

On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```
### 2.4 Run the Flask server

```bash
python app.py
```

Once the server is running, open your browser and navigate to http://localhost:5000/ to start using AnDriDemo.


## 3. Usage
The system is designed as a web application with a Python Flask backend and a JavaScript-based frontend for interactive visualization and user labeling.
![System Architecture](static/img/architecture.png)

### 3.1 Upload Data
![System Architecture](static/img/upload.png)
Go to http://localhost:5000/. Choose one of our example datasets or upload your own dataset. Then set α, which means that the proportion of outliers cannot exceed this value. 


### 3.2 Select Training Set(s)
![System Architecture](static/img/select.png)
Click on the "chart" on the upper right to add the training set, and adjust the size of the training set in the lower "chart".


### 3.3 Mark Anomalies
![System Architecture](static/img/mark.png)
Click on a certain area on the upper right chart to navigate, and drag on the lower right chart to mark the abnormal points.

### 3.4 Evaluate Points
![System Architecture](static/img/repeat.png)
Based on the Normal Pattern other information, determine whether some points are anomalies or normal points.


### 3.5 Repeat With Other Algorithms
Click on the other algorithms in the tab bar, repeat the process of active learning, and find the thresholds for the other algorithms.


### 3.6 View Results
![System Architecture](static/img/view.png)
Examine the accuracy rates of each algorithm, and also understand why Andri achieved better results