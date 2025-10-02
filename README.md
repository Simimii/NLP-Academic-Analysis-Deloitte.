# NLP-Powered Analysis of Academic Publications
*Project for the "Data Science in Action" course in collaboration with Deloitte.*

## Overview
This project is an end-to-end NLP system designed to help researchers efficiently navigate academic literature. It retrieves articles, identifies key topics, provides summaries, and recommends related papers. The final product is an interactive web application built with Streamlit.

## Key Features
*   **Topic Modeling:** Utilizes BERTopic with Bayesian optimization to identify key themes in a dataset of over 6,400 academic papers.
*   **Recommendation System:** A hybrid model that combines semantic similarity (using Sentence Transformers) with topic awareness to provide highly relevant article suggestions.
*   **Text Summarization:** Implements both extractive and abstractive methods to generate concise summaries of research papers.
*   **Interactive UI:** A user-friendly web application built with Streamlit that allows for article analysis, topic exploration, and dataset browsing.

## Technologies Used
*   **Programming:** Python
*   **Core Libraries:** pandas, Scikit-learn, Sentence Transformers, BERTopic
*   **Deployment/UI:** Streamlit
*   **Data Retrieval:** OpenAlex API

## How to View
*   The final **Technical Report** can be found in the `/report/` folder.
*   The primary **Jupyter Notebooks** detailing the analysis and model development are in the `/Codes/` folder.
*   The code for the Streamlit application is in the `/Codes/` folder.

## Final Results
*   The topic model successfully identified distinct research clusters with high coherence and diversity scores.
*   The recommendation system achieved a Precision@k score of 0.80, demonstrating its effectiveness.
