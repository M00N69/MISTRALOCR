import streamlit as st
import requests
from mistralai import Mistral
import os
import json
import pandas as pd
from io import BytesIO
import base64

# Configuration de la clé API
# Utilisez st.secrets pour une gestion sécurisée des clés
API_KEY = st.secrets["API_KEY"]

# Initialiser le client Mistral
client = Mistral(api_key=API_KEY)

# --- Fonctions existantes pour l'OCR ---

# Fonction pour uploader le fichier PDF directement via l'API Mistral
def upload_pdf_to_mistral(file):
    """Uploads a file to Mistral AI for processing."""
    try:
        # Use file.getvalue() to get bytes content for upload
        file_content = file.getvalue()
        uploaded_pdf = client.files.upload(
            file={
                "file_name": file.name,
                "content": file_content,
            },
            purpose="ocr"
        )
        st.success(f"Fichier '{file.name}' uploadé avec succès (ID: {uploaded_pdf.id}).")
        return uploaded_pdf.id
    except Exception as e:
        st.error(f"Erreur lors de l'upload du fichier: {e}")
        return None

# Fonction pour obtenir une URL signée pour le fichier uploadé
def get_signed_url(file_id):
    """Gets a signed URL for an uploaded file ID."""
    try:
        signed_url = client.files.get_signed_url(file_id=file_id)
        st.success("URL signée récupérée.")
        return signed_url.url
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'URL signée: {e}")
        return None

# Fonction pour appeler l'API OCR avec l'URL signée
def call_ocr_api(signed_url, model="mistral-ocr-latest"):
    """Calls the Mistral OCR API to process the document via URL."""
    try:
        with st.spinner("Traitement OCR en cours..."):
            ocr_response = client.ocr.process(
                model=model,
                document={
                    "type": "document_url",
                    "document_url": signed_url,
                },
                include_image_base64=False # Set to True if you need images, but generally not for text extraction
            )
        st.success("OCR terminé.")
        return ocr_response
    except Exception as e:
        st.error(f"Erreur lors de l'appel de l'API OCR: {e}")
        return None

# --- Nouvelle fonction pour l'extraction IA ---

def extract_info_with_llm(ocr_text):
    """Uses a Mistral LLM to extract structured information from OCR text."""
    # Define the desired JSON structure
    json_schema = {
      "report_info": {
        "lab_name": "string",
        "report_id": "string",
        "issue_date": "string", # e.g., "DD/MM/YYYY" or "YYYY-MM-DD"
        "validation_date": "string" # e.g., "DD/MM/YYYY" or "YYYY-MM-DD"
      },
      "client_info": {
        "client_name": "string"
      },
      "sample_info": {
        "product_name": "string",
        "lot_number": "string",
        "sample_id": "string", # Internal lab sample ID
        "date_received": "string", # e.g., "DD/MM/YYYY" or "YYYY-MM-DD"
        "date_analyzed": "string", # e.g., "DD/MM/YYYY" or "YYYY-MM-DD"
        "date_collected": "string"  # Optional, e.g., "DD/MM/YYYY" or "YYYY-MM-DD"
      },
      "analysis_results": [ # This should be an array of objects, one for each row in the results table
        {
          "parameter": "string", # e.g., "Humidité", "Matière grasse libre", "Protéines", "Sucres solubles", "HPD", etc.
          "result": "string", # The measured value, can be number or text like "<0.1"
          "unit": "string", # e.g., "g/100g", "%", "mg/100g"
          "specification": "string", # The specification or regulatory limit (e.g., "<=82", "<35", "Conforme")
          "uncertainty": "string", # The uncertainty value
          "method": "string" # The analytical method used (e.g., "Mi105", "NF V 04-403")
        }
        # ... more analysis results objects
      ],
      "conclusion": "string" # The overall conclusion (e.g., "Conforme")
    }

    # Craft a detailed prompt for the LLM
    prompt = f"""
    You are an expert in analyzing laboratory food analysis reports (bulletins d'analyse).
    I will provide you with the full text extracted from such a report.
    Your task is to extract the key information and structure it into a JSON object based on the following schema.
    Pay close attention to the tables containing analysis results. Extract each row from these tables into the 'analysis_results' array.

    Ensure the JSON output is strictly valid and only contains the JSON object within a markdown code block.

    JSON Schema:
    {json.dumps(json_schema, indent=2)}

    If a specific piece of information (like date_collected, method, or certain report/client fields) is not present or cannot be reliably identified in the text, set the corresponding value to `null`.
    For the 'analysis_results', if a column value (like unit, specification, uncertainty, or method for a specific parameter) is not present in that row, set it to `null` for that result object.
    Extract numerical results and uncertainties as strings, preserving the original format (e.g., "59.9", "1.2", "<0.1").

    Here is the text from the analysis report:

    ---TEXT_START---
    {ocr_text}
    ---TEXT_END---

    Please provide the extracted information as a single JSON object inside a markdown code block like this:
    ```json
    {{...}}
    ```
    """

    try:
        with st.spinner("Analyse IA des résultats en cours..."):
            # Use a capable generative model like mistral-large-latest or mistral-medium
            chat_response = client.chat(
                model="mistral-large-latest", # Or "mistral-medium", "mistral-small"
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"} # Request JSON output directly
            )
        st.success("Analyse IA terminée.")

        # Attempt to parse the JSON response
        # Mistral's response_format="json_object" should return a JSON directly
        # Sometimes it might still wrap it, so we'll try both direct parsing and finding the markdown block
        response_content = chat_response.choices[0].message.content.strip()

        try:
            # Try direct parsing first
            extracted_data = json.loads(response_content)
            return extracted_data
        except json.JSONDecodeError:
            # If direct parsing fails, try to find the JSON block in markdown
            st.warning("Le modèle n'a pas retourné de JSON direct. Recherche du bloc de code markdown.")
            json_string = None
            if response_content.startswith("```json"):
                json_string = response_content[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()
            elif "```json" in response_content:
                 # Look for the first ```json block
                 start = response_content.find("```json") + len("```json")
                 end = response_content.find("```", start)
                 if start != -1 and end != -1:
                     json_string = response_content[start:end].strip()

            if json_string:
                try:
                    extracted_data = json.loads(json_string)
                    return extracted_data
                except json.JSONDecodeError:
                    st.error("L'IA a généré une réponse qui ne contient pas de JSON valide même dans un bloc de code.")
                    st.text(response_content) # Show user what the AI returned
                    return None
            else:
                 st.error("L'IA a généré une réponse inattendue qui ne contient pas de JSON valide.")
                 st.text(response_content) # Show user what the AI returned
                 return None


    except Exception as e:
        st.error(f"Erreur lors de l'appel de l'API IA pour l'extraction: {e}")
        return None

# --- Helper function for Excel download ---

def to_excel(df):
    """Saves a DataFrame to an Excel file in memory."""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Analysis Results')
    writer.close() # Use close() instead of save() for newer pandas/xlsxwriter
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit Interface ---

st.title("🧪 Extracteur d'Informations de Bulletins d'Analyses (IA)")
st.markdown("""
Upload un fichier PDF de bulletin d'analyse pour:
1.  Extraire le texte via OCR.
2.  Utiliser une IA (Mistral Large) pour identifier et structurer les informations clés (détails du rapport, du produit, et surtout les résultats d'analyse).
3.  Afficher les résultats structurés sous forme de tableau.
4.  Permettre le téléchargement des résultats d'analyse au format Excel.
""")

uploaded_file = st.file_uploader("📥 Choisissez un fichier PDF de bulletin d'analyse", type=["pdf"])

extracted_data = None # Variable pour stocker les données extraites par l'IA

if uploaded_file:
    st.write(f"Fichier sélectionné: **{uploaded_file.name}**")

    # Step 1: Upload and OCR
    st.subheader("Étape 1: OCR du Document")
    file_id = upload_pdf_to_mistral(uploaded_file)

    if file_id:
        signed_url = get_signed_url(file_id)

        if signed_url:
            ocr_result = call_ocr_api(signed_url)

            if ocr_result:
                try:
                    # Extract the markdown text from all pages
                    pages_text = [page.markdown for page in ocr_result.pages]
                    full_text = "\n\n==NEW_PAGE==\n\n".join(pages_text) # Add separator for clarity to LLM

                    # Optional: Display the raw OCR text for debugging/verification
                    # with st.expander("Voir le texte OCR brut"):
                    #    st.text(full_text)

                    # Step 2: AI Extraction
                    st.subheader("Étape 2: Extraction des Informations par IA")
                    extracted_data = extract_info_with_llm(full_text)

                    if extracted_data:
                        st.subheader("Étape 3: Informations Extraites")

                        # Display top-level info
                        st.markdown("#### Détails du Rapport et de l'Échantillon")
                        report_info_df = pd.DataFrame({
                            "Champ": ["Laboratoire", "ID Rapport", "Date d'émission", "Date de Validation", "Client", "Produit", "Lot", "ID Échantillon", "Date réception", "Date analyse", "Date collecte", "Conclusion"],
                            "Valeur": [
                                extracted_data.get('report_info', {}).get('lab_name'),
                                extracted_data.get('report_info', {}).get('report_id'),
                                extracted_data.get('report_info', {}).get('issue_date'),
                                extracted_data.get('report_info', {}).get('validation_date'),
                                extracted_data.get('client_info', {}).get('client_name'),
                                extracted_data.get('sample_info', {}).get('product_name'),
                                extracted_data.get('sample_info', {}).get('lot_number'),
                                extracted_data.get('sample_info', {}).get('sample_id'),
                                extracted_data.get('sample_info', {}).get('date_received'),
                                extracted_data.get('sample_info', {}).get('date_analyzed'),
                                extracted_data.get('sample_info', {}).get('date_collected'),
                                extracted_data.get('conclusion')
                            ]
                        })
                        st.table(report_info_df.set_index("Champ"))


                        # Display analysis results in a DataFrame
                        st.markdown("#### Résultats d'Analyse Détaillés")
                        analysis_results = extracted_data.get('analysis_results')

                        if analysis_results:
                            results_df = pd.DataFrame(analysis_results)
                            # Reorder columns for better readability if needed
                            display_cols = ['parameter', 'result', 'unit', 'specification', 'uncertainty', 'method']
                            results_df = results_df[display_cols].fillna('') # Fill NaN for better display

                            st.dataframe(results_df, use_container_width=True)

                            # Step 4: Download as Excel
                            st.subheader("Étape 4: Télécharger les Résultats")
                            excel_data = to_excel(results_df)
                            st.download_button(
                                label="Télécharger les résultats en Excel",
                                data=excel_data,
                                file_name=f"resultats_analyse_{uploaded_file.name.replace('.pdf', '')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning("Aucun résultat d'analyse n'a pu être extrait par l'IA.")

                    else:
                        st.error("Échec de l'extraction des informations par l'IA.")

                except AttributeError:
                    st.error("Erreur : Impossible d'accéder à l'attribut 'pages' dans la réponse OCR.")
                except Exception as e:
                    st.error(f"Une erreur inattendue s'est produite lors du traitement des données extraites : {e}")

else:
    st.info("Veuillez uploader un fichier PDF pour commencer l'analyse.")

st.markdown("---")
st.markdown("Développé avec Mistral AI 🧠")
