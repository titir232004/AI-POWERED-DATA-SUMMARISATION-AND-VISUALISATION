# AI Document Summarizer

## Overview
AI Document Summarizer is a web-based application built using Python and Streamlit, capable of generating both abstractive and extractive summaries for input documents. The application also provides visualizations such as word clouds, bar charts of word frequencies, network graphs, and timelines for text analysis. 

## Features
- **Abstractive Summarization**: Summarizes the document using the `facebook/bart-large-xsum` model from Hugging Face Transformers.
- **Extractive Summarization**: Uses the LexRank algorithm for extractive summary generation.
- **Word Cloud**: Visualizes the most common words in the text.
- **Bar Chart**: Displays the top 10 most frequent words.
- **Network Graph**: Shows the relationships between consecutive words.
- **Timeline**: Generates a simple timeline based on detected date-like words in the text.
- **CSV Export**: Saves the original text, abstractive summary, and extractive summary to a CSV file for later reference.

## Requirements

### Prerequisites
Ensure you have `Python 3.8+` installed on your system. It is recommended to create and use a virtual environment for this project.

### Setting Up the Virtual Environment
1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### `requirements.txt` File
Create a `requirements.txt` file with the following content:

```plaintext
streamlit==1.40.2
transformers==4.46.3
datasets==3.1.0
sumy==0.11.0
wordcloud==1.9.4
matplotlib==3.9.2
pandas==2.2.3
nltk==3.9.1
networkx==3.4.2
```

### NLTK Resource Download
Run the following command in your Python environment to download the necessary NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Running the Application
To launch the Streamlit app, use the following command in your terminal:

```bash
streamlit run app.py
```

Where `app.py` is the name of the Python script containing the provided code.

## How to Use
1. **Input Document**: Paste your document directly into the text area or upload a `.txt` file.
2. **Generate Summaries**: Click the **Summarize** button to generate summaries and visualizations.
3. **View Results**: The application will display the original text, abstractive summary, extractive summary, and visualizations.
4. **Download Summaries**: The summaries are saved to a CSV file named `summaries.csv` in the current working directory.

## Project Structure
```plaintext
ai-document-summarizer/
│
├── app.py                # Main application script
├── requirements.txt      # Python package dependencies
└── summaries.csv         # CSV file for saving summaries (auto-created)
```

## Key Libraries Used
- **Streamlit**: Framework for building web applications.
- **Hugging Face Transformers**: For pre-trained models for abstractive summarization.
- **Sumy**: For extractive summarization.
- **NLTK**: For natural language processing tasks like tokenization.
- **WordCloud**: For generating word clouds.
- **Matplotlib**: For plotting bar charts and visualizations.
- **NetworkX**: For generating network graphs.
- **Pandas**: For data handling and CSV export.

## Acknowledgements
This project utilizes the following open-source libraries:
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Sumy](https://github.com/miso-belica/sumy)
- [WordCloud](https://github.com/amueller/word_cloud)
- [Streamlit](https://streamlit.io)
