import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import base64

# Configuration de la clé API
API_KEY = st.secrets["API_KEY"]
API_URL = "https://api.mistral.ai/v1/ocr"

# Fonction pour encoder le fichier PDF en Base64
def encode_file_to_base64(file):
    return base64.b64encode(file.read()).decode('utf-8')

# Fonction pour appeler l'API OCR avec fichier en Base64
def call_ocr_api_with_base64(file_base64, file_name, model="default"):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "document": {
            "type": "document_base64",
            "document_base64": file_base64,
            "document_name": file_name
        },
        "include_image_base64": False
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur API: {response.status_code} - {response.text}")
        return None

# Interface Streamlit
st.title("OCR Extracteur de Texte avec Mistral AI")
st.markdown("Upload un fichier PDF pour extraire le texte.")

uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

if uploaded_file:
    st.write(f"Fichier uploadé: {uploaded_file.name}")
    st.info("Traitement du fichier en cours...")

    # Encoder le fichier en Base64
    file_base64 = encode_file_to_base64(uploaded_file)

    # Appel à l'API OCR avec le fichier encodé
    ocr_result = call_ocr_api_with_base64(file_base64, uploaded_file.name)
    
    if ocr_result:
        # Extraction du texte des pages
        pages_text = [page.get("text", "") for page in ocr_result.get("pages", [])]
        full_text = "\n".join(pages_text)
        
        # Affichage du texte extrait
        st.subheader("Texte extrait")
        st.text_area("", full_text, height=300)
        
        # Téléchargement au format TXT
        txt_bytes = BytesIO(full_text.encode("utf-8"))
        st.download_button(
            label="Télécharger en TXT",
            data=txt_bytes,
            file_name=f"{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain"
        )
        
        # Téléchargement au format CSV
        df = pd.DataFrame({"Texte": pages_text})
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger en CSV",
            data=csv_bytes,
            file_name=f"{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
else:
    st.info("Veuillez uploader un fichier PDF.")
