from PyPDF2 import PdfReader
import re

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
    return text

def clean_extracted_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove common LaTeX artifacts like \n, \x0b, etc.
    text = re.sub(r'\\[a-z]*', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
    
    
if __name__=='__main__':
    reader = PdfReader('./research_papers/2.pdf')
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
      
    text = clean_text(text)
    
    cleaned_text = clean_extracted_text(text)
    
    conclusion = extract_conclusion(cleaned_text)