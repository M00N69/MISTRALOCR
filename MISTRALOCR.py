import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from mistralai import Mistral
import os

# Configuration de la clé API
API_KEY = st.secrets["API_KEY"]

# Initialiser le client Mistral
client = Mistral(api_key=API_KEY)

# Fonction pour uploader le fichier PDF directement via l'API Mistral
def upload_pdf_to_mistral(file):
    try:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": file.name,
                "content": file.read(),
            },
            purpose="ocr"
        )
        return uploaded_pdf.id
    except Exception as e:
        st.error(f"Erreur lors de l'upload: {e}")
        st.stop()

# Fonction pour obtenir une URL signée pour le fichier uploadé
def get_signed_url(file_id):
    try:
        signed_url = client.files.get_signed_url(file_id=file_id)
        return signed_url.url
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'URL signée: {e}")
        st.stop()

# Fonction pour appeler l'API OCR avec l'URL signée
def call_ocr_api(signed_url, model="mistral-ocr-latest"):
    try:
        ocr_response = client.ocr.process(
            model=model,
            document={
                "type": "document_url",
                "document_url": signed_url,
            },
            include_image_base64=False
        )
        return ocr_response
    except Exception as e:
        st.error(f"Erreur API: {e}")
        st.stop()

# Interface Streamlit
st.title("OCR Extracteur de Texte avec Mistral AI")
st.markdown("Upload un fichier PDF pour extraire le texte.")

uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

if uploaded_file:
    st.write(f"Fichier uploadé: {uploaded_file.name}")
    st.info("Upload du fichier en cours...")

    file_id = upload_pdf_to_mistral(uploaded_file)

    if file_id:
        st.success(f"Fichier uploadé avec succès. ID: {file_id}")
        signed_url = get_signed_url(file_id)

        if signed_url:
            st.success("URL signée obtenue avec succès.")
            ocr_result = call_ocr_api(signed_url)

            # ➡️ Afficher le type et le contenu brut de la réponse pour déboguer
            st.write("Type de ocr_result:", type(ocr_result))
            st.write("Contenu brut de ocr_result:", ocr_result)

            if ocr_result:
                try:
                    # 🔄 Utiliser .dict() pour convertir en dictionnaire si nécessaire
                    if hasattr(ocr_result, "pages"):
                        # Extraire le texte en Markdown depuis chaque page
                        pages_text = [page.markdown for page in ocr_result.pages]
                    else:
                        # Si .pages échoue, essayer d'accéder via .dict()
                        ocr_result_dict = ocr_result.dict()
                        pages_text = [page["markdown"] for page in ocr_result_dict.get("pages", [])]

                    full_text = "\n".join(pages_text)

                except AttributeError:
                    st.error("Erreur : Impossible d'accéder à l'attribut 'pages'.")
                    st.stop()

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
