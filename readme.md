Research Project: Effort Estimation in Textual Requirements

This repository contains the files developed as part of a research project focused on effort estimation in textual requirements using Natural Language Processing (NLP) techniques and machine learning.

The project aims to explore language models and deep neural networks to predict the effort required based on user stories, contributing to the field of software engineering.
Repository Structure

    data/
    Contains the datasets used in the project, including preprocessed and organized data for training and testing.

    Fine-tuning_data/
    Includes specific data used in the fine-tuning process of the language models.

    convert_embeddings.ipynb
    Notebook responsible for converting user stories into embeddings using pre-trained models such as FastText and XLNet.

    ESApp.ipynb
    Main notebook of the project, which implements the deep learning model for effort estimation, including training, testing, and evaluation phases.

Technologies Used

    Programming Language: Python
    Development Environment: Visual Studio Code
    Language Models: FastText and XLNet
    Libraries:
        NumPy
        Pandas
        Scikit-learn
        TensorFlow/Keras
        Matplotlib
        Seaborn

Reproducibility

This repository is organized to ensure the reproducibility of the study. All data, notebooks, and scripts necessary to replicate the experiments are included.
How to Reproduce

    Clone this repository:

git clone https://github.com/your-username/repository-name.git

Ensure all dependencies are installed:

    pip install -r requirements.txt

    Run the notebooks in the following order:
        convert_embeddings.ipynb
        ESApp.ipynb

Contribution

This repository is part of an academic project, and external contributions are not being accepted at the moment. However, suggestions and feedback are welcome!
License

This project is licensed under the MIT License, allowing its use for academic and research purposes.