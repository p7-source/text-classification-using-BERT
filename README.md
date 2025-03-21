# text-classification-using-BERT :rocket:
![Project Logo](https://github.com/p7-source/text-classification-using-BERT/blob/main/ClassificationImg.png?raw=true)

Welcome, every one!
I just build an end to end text classification project using BERT/distilBert to classify cutomers' sentiments.
## Happy Coding all :smile:




# Problem Statement
The goal of the project is to analyze the the customer feedback from social media surveys, support tickets, or customer executive chats to determine the overall sentiment towards a company's products and services. The sentiment will classify into three catagories.



## 📝 Description
This project demonstrates **Text Classification using BERT**, a state-of-the-art transformer model from Hugging Face. It is designed as a **Minimum Viable Product (MVP)** with a focus on modularity, scalability, and ease of integration into existing systems. The project is structured into **three pipelines**:
1. **Feature Engineering**
2. **Model Training**
3. **Inference Pipeline**

---

## 🛠️ Key Features
- **Modular Codebase**: Notebooks converted into production-ready modular code.
- **Experiment Tracking**: Integrated with **Weights & Biases (Wandb)** or **MLflow**, and **DVC**.
- **Deployment Ready**: Includes Docker and AWS EC2/ECR setup for seamless deployment.
- **Unique Pipeline Design**: Three pipelines for feature engineering, model training, and inference.

---
## Packages'installation and Project setup
```bash
conda create -n tcenv
conda activate tcenv
conda install python=3.9
pip install -r .\requirements.txt
```
---
## Project setup 
```bash
git clone https://github.com/p7-source/text-classification-using-BERT.git
cd text-classification-using-BERT
```
## Project setup 
```bash
git clone https://github.com/p7-source/text-classification-using-BERT.git
cd text-classification-using-BERT
python app.py
```
## 🏗️ Project Structure

```plaintext
text-classification-using-BERT/
├───.github
│   └───workflows
├───artifacts
│   ├───data_ingestion
│   ├───feature_engineering
│   │   └───datasets
│   ├───prepare_model
│   └───training
│       ├───testing_dir
│       └───trained_model
├───config
├───logs
├───model
├───research
├───src
│   ├───textClassifier
│   │   ├───components
│   │   ├───config
│   │   ├───constants
│   │   ├───entity
│   │   ├───logs
│   │   ├───pipeline
│   │   ├───utils
├───templates
└───wandb


