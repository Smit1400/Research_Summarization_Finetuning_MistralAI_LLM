from PyPDF2 import PdfReader
import re
import pandas as pd
import nltk
from textblob import TextBlob
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load spacy's NER model
nlp = spacy.load('en_core_web_sm')

patterns_to_remove = [
    r'\(see Figure \d+\)',  # More specific pattern to remove references to figures
    r'\(see Table \d+\)',   # More specific pattern to remove references to tables
    # Add more specific patterns as needed
]
# Function to clean text using regular expressions
def clean_text(text):
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Remove any standalone mathematical symbols and numbers not part of words
    text = re.sub(r'\b[\d.]+\b', '', text)
    # Remove any additional unwanted characters, keep hyphens and apostrophes within words
    text = re.sub(r'(?<!\w)[-]', ' ', text)
    text = re.sub(r'(?<!\w)[^\w\s\'-]+', '', text)
    text = re.sub(r'(?<=\w)[^\w\s\'-]+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\[a-z]*', ' ', text)
    return text


def find_nearest_section(text, keywords):
    nearest_index = len(text)
    for keyword in keywords:
        index = text.lower().find(keyword)
        if 0 <= index < nearest_index:
            nearest_index = index
    return nearest_index

def extract_conclusion(text):
    conclusion_index = find_nearest_section(text, ['conclusion'])
    conclusion = text[conclusion_index:conclusion_index+1500].strip()
    ref_index = conclusion.lower().find("references")
    conclusion = conclusion[:ref_index]
    return conclusion
    
    
def text_processing(text: str) -> str:
    # Lowercasing
    text = text.lower()

    # Removing HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Replace URLs and email addresses with a space
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', ' ', text)

    # Carefully replace or remove non-standard characters
    # This regex will replace non-alphanumeric characters that are not within a word with a space
    text = re.sub(r'(?<=\W)\W+|\W+(?=\W)', ' ', text)

    # Tokenization and further cleaning
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]

    return ' '.join(cleaned_words)
    
if __name__=='__main__':
    data_dict = {'text': [], 'summary': []}
    for i in range(1, 2):
        reader = PdfReader(f'./research_papers/{i}.pdf')
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        
        text = clean_text(text)
        conclusion = extract_conclusion(text)
        
        data_dict['text'].append(text)
        data_dict['summary'].append(conclusion)
    
    df = pd.DataFrame(data=data_dict)
    df['processed_text'] = df['text'].apply(text_processing)
    
    df.to_csv('data.csv', index=False)
    
    print(df['processed_text'][0])
    print(df.shape)