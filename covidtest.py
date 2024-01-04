import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import difflib

Translation = {
    'AUA': 'I', 'AUC': 'I', 'AUU': 'I', 'AUG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACU': 'T',
    'AAC': 'N', 'AAU': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGU': 'S', 'AGA': 'R', 'AGG': 'R',
    'CUA': 'L', 'CUC': 'L', 'CUG': 'L', 'CUU': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCU': 'P',
    'CAC': 'H', 'CAU': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGU': 'R',
    'GUA': 'V', 'GUC': 'V', 'GUG': 'V', 'GUU': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCU': 'A',
    'GAC': 'D', 'GAU': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGU': 'G',
    'UCA': 'S', 'UCC': 'S', 'UCG': 'S', 'UCU': 'S',
    'UUC': 'F', 'UUU': 'F', 'UUA': 'L', 'UUG': 'L',
    'UAC': 'Y', 'UAU': 'Y', 'UAA': '*', 'UAG': '*',
    'UGC': 'C', 'UGU': 'C', 'UGA': '*', 'UGG': 'W',
}


def Funkmer(DNA, k):
    arr = []
    length = len(DNA)
    for i in range(0, length, 3):
        flag = True
        if len(DNA[i:i + k]) == k:
            for j in DNA[i:i + k]:
                if j not in ("A", "U", "G", "C"):
                    flag = False
            if flag:
                arr.append(DNA[i:i + k])
    return arr


def DNAProtein(sequence):
    DNA = sequence
    RNA = DNA.replace('T', 'U')
    k = 3
    kmers = Funkmer(RNA, k)
    Protein = ''
    count1 = 0
    Arr = []
    flag = False
    for i in kmers:
        if i == 'AUG' and flag == False:
            flag = True
            Arr.append('')
        elif (i == 'UAA' or i == 'UAG' or i == 'UGA'):
            flag = False
        if flag:
            Arr[len(Arr) - 1] = Arr[len(Arr) - 1] + Translation[i]
    return list(filter(lambda el: len(el) >= 10, Arr))


def fileread(filename):
    gene_sequence = ""
    with open(filename, 'r') as f:
        for line in f:
            if not line[0] == '>':
                gene_sequence += line.rstrip()
    return gene_sequence


def create_dataset():
    # Prepare dataset for training the classification model
    covid_variants = [
        ("original", fileread(r"C:\Users\heman\OneDrive\Documents\BIO[1]\BIO\Covid\covid_varients\originalcovid.txt")),
        ("Alpha", fileread(r"C:\Users\heman\OneDrive\Documents\BIO[1]\BIO\Covid\covid_varients\Alpha.txt")),
        ("Beta", fileread(r"C:\Users\heman\OneDrive\Documents\BIO[1]\BIO\Covid\covid_varients\Beta.txt")),
        ("Gamma", fileread(r"C:\Users\heman\OneDrive\Documents\BIO[1]\BIO\Covid\covid_varients\Gamma.txt")),
        ("Omicron", fileread(r"C:\Users\heman\OneDrive\Documents\BIO[1]\BIO\Covid\covid_varients\omicron.txt")),
    ]
    
    # Replace 'path_to_non_covid_sample.txt' with the correct path
    non_covid_sample_path = r"C:\Users\heman\OneDrive\Documents\BIO[1]\BIO\Covid\test cases\test2.txt"
    non_covid_sequence = [("non_covid", fileread(non_covid_sample_path))]

    # Combine the datasets
    dataset = covid_variants + non_covid_sequence
    labels, sequences = zip(*dataset)

    return sequences, labels



def train_classifier(X, y):
    # Use a simple CountVectorizer to convert DNA sequences into numerical features
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X_vectorized = vectorizer.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    return classifier, vectorizer

def main():
    st.title("COVID Variant Detection")

    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])

    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')
        st.text("Uploaded content:")
        st.text(content)

        # Load the trained classifier and vectorizer
        classifier, vectorizer = train_classifier(*create_dataset())

        # Convert the uploaded DNA sequence into numerical features
        content_vectorized = vectorizer.transform([content])

        # Predict if the sequence is a COVID variant or not
        prediction = classifier.predict(content_vectorized)[0]

        # Display the prediction
        if prediction == 'non_covid':
            st.text("The uploaded genome sequence is classified as: Non-COVID variant")
        else:
            st.success("The uploaded genome sequence is classified as: COVID variant.")
            st.info("Please maintain social distancing, stay hygiened, and wear a mask.")

if __name__ == "__main__":
    main()