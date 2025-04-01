#__import__('pysqlite3')
#import sys
#gisys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#import sqlite3

import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Utilisation de st.secrets pour récupérer les clés
OPENAI_API_KEY = st.secrets["openai_api_key"]


# Mise en page Streamlit
st.set_page_config(page_title="Assistant Fiscal IA", page_icon="", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Assistant Juridique Fiscal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Veuillez télécharger votre fichier pour recevoir un résumé.</p>",
            unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource
def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="chroma_fiscalite", embedding_function=embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

qa_chain = load_qa_chain()

# Initialisation de l'historique
if "history" not in st.session_state:
    st.session_state.history = []

# Téléchargement du fichier
uploaded_file = st.file_uploader("Téléchargez un fichier (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

# Enregistrement du fichier et du résumé dans le session_state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# Fonction pour générer un résumé du document
def summarize_document(file, file_type):
    temp_filename = "temp_uploaded_file." + file_type
    with open(temp_filename, "wb") as f:
        f.write(file.getbuffer())

    # Chargement du fichier selon son type
    if file_type == "pdf":
        loader = PyPDFLoader(temp_filename)
    elif file_type == "txt":
        loader = TextLoader(temp_filename)
    elif file_type == "docx":
        loader = Docx2txtLoader(temp_filename)
    else:
        return "Format de fichier non pris en charge."

    # Découpage du texte en morceaux et génération du résumé
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    summary = llm.predict("Résume ce document en quelques phrases : " + " ".join([t.page_content for t in texts]))
    os.remove(temp_filename)  # Supprimer le fichier temporaire
    return summary

# Affichage du résumé uniquement si le bouton est pressé
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    # Ajouter un bouton pour générer le résumé
    if st.button("Générer le résumé du document"):
        with st.spinner("Génération du résumé..."):
            summary = summarize_document(uploaded_file, file_type)
            st.session_state.uploaded_file = uploaded_file  # Sauvegarder le fichier téléchargé
            st.session_state.summary = summary  # Sauvegarder le résumé
            st.markdown("### Résumé du fichier :")
            st.write(summary)

# Entrée utilisateur
user_input = st.chat_input(" Posez votre question ici...")

# Traitement de la requête
if user_input:
    with st.spinner("Recherche en cours..."):
        result = qa_chain.run(user_input)
        st.session_state.history.append({"user": user_input, "bot": result})

# Affichage de la conversation
for i, entry in enumerate(st.session_state.history):
    with st.container():
        st.markdown(
            f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Vous :</strong><br>{entry['user']}</div>",
            unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin-bottom: 20px;'><strong> Assistant :</strong><br>{entry['bot']}</div>",
            unsafe_allow_html=True)

# Bouton pour réinitialiser la conversation
with st.sidebar:
    if st.button(" Effacer la conversation"):
        # Réinitialiser l'historique et les fichiers
        st.session_state.history = []
        st.session_state.uploaded_file = None  # Supprimer le fichier téléchargé
        st.session_state.summary = None  # Supprimer le résumé
        st.session_state['uploaded_file'] = None  # Effacer les données du fichier
        st.session_state['summary'] = None  # Effacer le résumé
        st.rerun()  # Recharger l'application comme au premier démarrage
