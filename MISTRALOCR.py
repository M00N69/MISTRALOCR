import streamlit as st
import requests
from mistralai import Mistral
import os
import json
import pandas as pd
from io import BytesIO
import base64

# --- Configuration ---

# Utiliser st.secrets pour une gestion sécurisée des clés API
# Assurez-vous que votre fichier .streamlit/secrets.toml contient [mistral_api] API_KEY="votre_cle_api"
# Ou configurez-le via l'interface Streamlit Cloud Secrets.
try:
    API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("Clé API Mistral non trouvée. Assurez-vous que 'API_KEY' est définie dans les secrets Streamlit.")
    st.stop() # Arrête l'exécution si la clé API n'est pas trouvée

# Initialiser le client Mistral
# Le base_url peut être spécifié si nécessaire, sinon l'API publique par défaut est utilisée
try:
    client = Mistral(api_key=API_KEY)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    st.stop()

# Modèles à utiliser
OCR_MODEL = "mistral-ocr-latest"
# Utiliser un modèle plus performant pour l'extraction de données structurées complexes
# mistral-large-latest est recommandé, mistral-medium peut être une alternative
LLM_MODEL = "mistral-large-latest" # ou "mistral-medium-latest"

# --- Fonctions pour l'OCR et l'Upload (Utilisation de l'API Mistral Files et OCR) ---

@st.cache_data(show_spinner=False) # Cache les résultats si le fichier est le même
def upload_pdf_to_mistral(_file_content, file_name):
    """Uploads file content to Mistral AI for processing.
    _file_content is the actual bytes data, file_name is for the API."""
    try:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": file_name,
                "content": _file_content, # Pass the bytes content
            },
            purpose="ocr" # Specify purpose for OCR
        )
        st.success(f"Fichier '{file_name}' uploadé avec succès (ID: {uploaded_pdf.id}).")
        return uploaded_pdf.id
    except Exception as e:
        st.error(f"Erreur lors de l'upload du fichier à Mistral API: {e}")
        return None

@st.cache_data(show_spinner=False) # Cache les résultats
def get_signed_url(_file_id):
    """Gets a signed URL for an uploaded file ID."""
    try:
        signed_url = client.files.get_signed_url(file_id=_file_id)
        st.success("URL signée pour l'OCR récupérée.")
        return signed_url.url
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'URL signée: {e}")
        return None

@st.cache_data(show_spinner=False) # Cache les résultats
def call_ocr_api(_signed_url):
    """Calls the Mistral OCR API to process the document via URL."""
    try:
        with st.spinner(f"Traitement OCR en cours avec le modèle '{OCR_MODEL}'..."):
            ocr_response = client.ocr.process(
                model=OCR_MODEL,
                document={
                    "type": "document_url",
                    "document_url": _signed_url,
                },
                include_image_base64=False # Usually not needed for text extraction
            )
        st.success("OCR terminé.")
        return ocr_response
    except Exception as e:
        st.error(f"Erreur lors de l'appel de l'API OCR: {e}")
        return None

# --- Fonction pour l'Extraction Structurée par IA (Utilisation de l'API Mistral Chat) ---

# Important: Cette fonction ne peut pas être directement cachée avec @st.cache_data
# car la réponse de l'API Chat dépend du modèle, des paramètres, et potentiellement de l'évolution du modèle lui-même,
# ce qui n'est pas basé uniquement sur l'entrée ocr_text de manière déterministe et stable sur le long terme.
# De plus, les appels API externes ne sont généralement pas mis en cache côté Streamlit.
def extract_info_with_llm(ocr_text):
    """Uses a Mistral LLM to extract structured information from OCR text."""

    # Define the desired JSON structure (Schema)
    # This schema guides the LLM on what to extract and how to format it.
    json_schema = {
      "report_info": {
        "lab_name": "string or null",
        "report_id": "string or null",
        "issue_date": "string or null", # e.g., "DD/MM/YYYY", "YYYY-MM-DD", or "Month Day, Year" - keep as string to handle various formats
        "validation_date": "string or null", # e.g., "DD/MM/YYYY" - keep as string
        "validator_name": "string or null" # Name of the person who validated the report
      },
      "client_info": {
        "client_name": "string or null",
        "client_address": "string or null", # Optional, but useful if present
        "client_id": "string or null" # Client code/number if available
      },
      "sample_info": {
        "product_name": "string or null",
        "lot_number": "string or null",
        "sample_id": "string or null", # Internal lab sample ID
        "date_received": "string or null", # e.g., "DD/MM/YYYY" - keep as string
        "date_analyzed": "string or null", # e.g., "DD/MM/YYYY" - keep as string
        "date_collected": "string or null", # Optional, e.g., "DD/MM/YYYY" - keep as string
        "product_format": "string or null", # e.g., "375G (3X125G)"
        "best_before_date": "string or null", # DLC/DLUO
        "supplier": "string or null", # Fournisseur
        "ean_code": "string or null" # Gencod/Code Barre
      },
      "analysis_results": [ # This must be an array of objects, one for each row in the results table
        {
          "parameter": "string or null", # e.g., "Humidité", "Matière grasse libre", "Protéines", "Sucres solubles", "HPD", etc.
          "result": "string or null", # The measured value, can be number or text like "<0.1", "Conforme"
          "unit": "string or null", # e.g., "g/100g", "%", "mg/100g", "Unité"
          "specification": "string or null", # The specification or regulatory limit (e.g., "<=82", "<35", "Conforme", "N/A")
          "uncertainty": "string or null", # The uncertainty value (e.g., "1.2", "0.4")
          "method": "string or null" # The analytical method used (e.g., "Mi105", "NF V 04-403", "Calcul")
        }
        # ... more analysis results objects as found in the document tables
      ],
      "conclusion": "string or null" # The overall conclusion (e.g., "Conforme")
      # Could add more fields like notes, comments, etc. if relevant
    }

    # Craft a detailed prompt for the LLM
    # The prompt is key to getting good structured output from the LLM.
    prompt = f"""
    You are an expert in analyzing laboratory food analysis reports (bulletins d'analyse).
    I will provide you with the full text extracted from such a report using OCR.
    The text may contain information from multiple pages, separated by '==NEW_PAGE=='.
    Your primary task is to extract the key information and structure it into a JSON object based strictly on the following schema.

    Pay very close attention to the tables containing analysis results. You MUST extract EACH ROW from ALL analysis results tables into the 'analysis_results' array. Each object in the array should correspond to one row and contain the 'parameter', 'result', 'unit', 'specification', 'uncertainty', and 'method' columns as listed in the schema, extracting the value for that row.

    Ensure the JSON output is strictly valid and only contains the JSON object within a markdown code block formatted as ````json```.

    JSON Schema:
    {json.dumps(json_schema, indent=2)}

    If a specific piece of information for any field (including any column for any row in 'analysis_results') is not present or cannot be reliably identified in the text, set the corresponding JSON value to `null`.
    Extract numerical results, specifications, and uncertainties as strings, preserving the original format (e.g., "59.9", "1.2", "<0.1", "<=82"). Keep dates and other text fields as strings.

    Here is the text from the analysis report:

    ---REPORT_TEXT_START---
    {ocr_text}
    ---REPORT_TEXT_END---

    Please provide the extracted information as a single JSON object inside a markdown code block like this:
    ````json
    {{...}}
    ````
    """

    try:
        with st.spinner(f"Analyse IA des résultats en cours avec le modèle '{LLM_MODEL}'..."):
            # Use client.chat.completions.create for chat completions (correct syntax for v1+)
            chat_response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # Use response_format to strongly encourage JSON output
                response_format={"type": "json_object"},
                temperature=0 # Set temperature to 0 for deterministic output
            )
        st.success("Analyse IA terminée.")

        # Attempt to parse the JSON response
        # Mistral's response_format="json_object" should return a JSON directly in .content
        # However, robust parsing logic is still good practice.
        response_content = chat_response.choices[0].message.content.strip()

        extracted_data = None
        json_string = None

        try:
            # Try direct parsing first
            extracted_data = json.loads(response_content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find the JSON block in markdown
            st.warning("Le modèle n'a pas retourné de JSON direct. Recherche du bloc de code markdown...")
            if response_content.startswith("```json"):
                json_string = response_content[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()
            elif "```json" in response_content: # Handle cases where text might precede the block
                 start = response_content.find("```json") + len("```json")
                 end = response_content.find("```", start)
                 if start != -1 and end != -1:
                     json_string = response_content[start:end].strip()

            if json_string:
                try:
                    extracted_data = json.loads(json_string)
                except json.JSONDecodeError:
                    st.error("L'IA a généré une réponse qui ne contient pas de JSON valide même dans un bloc de code.")
                    st.text("Réponse brute de l'IA:")
                    st.text(response_content) # Show user what the AI returned for debugging
                    return None
            else:
                 st.error("L'IA a généré une réponse inattendue qui ne contient pas de JSON valide ou un bloc markdown JSON.")
                 st.text("Réponse brute de l'IA:")
                 st.text(response_content) # Show user what the AI returned for debugging
                 return None

        # Validate basic structure
        if not isinstance(extracted_data, dict):
             st.error("L'IA n'a pas retourné un objet JSON de niveau supérieur valide.")
             st.text("Réponse brute de l'IA:")
             st.text(response_content) # Show user what the AI returned for debugging
             return None

        return extracted_data

    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'appel ou du traitement de la réponse de l'IA : {e}")
        # Optionally print traceback for debugging
        # import traceback
        # st.text(traceback.format_exc())
        return None

# --- Helper function for Excel download ---

def to_excel(df):
    """Saves a DataFrame to an Excel file in memory."""
    output = BytesIO()
    try:
        # Use 'xlsxwriter' engine explicitly
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Resultats Analyse')
        processed_data = output.getvalue()
        return processed_data
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier Excel : {e}")
        return None


# --- Streamlit Interface ---

st.set_page_config(page_title="Extracteur de Bulletins d'Analyse", layout="wide")

st.title("🔬 Extracteur Automatisé de Bulletins d'Analyses (IA)")
st.markdown("""
Uploadez un fichier PDF de bulletin d'analyse. Cette application utilisera l'OCR de Mistral AI pour lire le document,
puis un grand modèle de langage (LLM) pour extraire les informations clés et les résultats d'analyse sous forme structurée.
Les résultats seront affichés et pourront être téléchargés au format Excel.
""")

st.sidebar.header("Paramètres et Informations")
st.sidebar.info(f"Modèle OCR utilisé : `{OCR_MODEL}`")
st.sidebar.info(f"Modèle LLM utilisé pour l'extraction : `{LLM_MODEL}`")
# Option pour montrer le texte OCR brut (utile pour débugger le prompt si l'extraction échoue)
show_raw_ocr = st.sidebar.checkbox("Afficher le texte OCR brut", value=False)


uploaded_file = st.file_uploader("📥 Choisissez un fichier PDF de bulletin d'analyse", type=["pdf"])

extracted_data = None # Variable pour stocker les données extraites par l'IA

if uploaded_file:
    st.write(f"Fichier sélectionné: **{uploaded_file.name}**")

    # Lire le contenu du fichier une seule fois
    file_content = uploaded_file.getvalue()

    # Step 1: Upload and OCR
    st.subheader("Étape 1: OCR du Document")
    # Pass content and name to caching function
    file_id = upload_pdf_to_mistral(file_content, uploaded_file.name)

    if file_id:
        # Step 1.1: Get Signed URL
        signed_url = get_signed_url(file_id)

        if signed_url:
            # Step 1.2: Perform OCR
            ocr_result = call_ocr_api(signed_url)

            if ocr_result and ocr_result.pages:
                try:
                    # Extract the markdown text from all pages
                    # Add a separator between pages to help the LLM differentiate
                    pages_text = [page.markdown for page in ocr_result.pages]
                    full_text = "\n\n==NEW_PAGE==\n\n".join(pages_text)

                    if show_raw_ocr:
                         with st.expander("Voir le texte OCR brut extrait"):
                            st.text(full_text) # Use st.text for preformatted text

                    # Step 2: AI Extraction
                    st.subheader("Étape 2: Extraction des Informations Structurées par IA")
                    extracted_data = extract_info_with_llm(full_text)

                    if extracted_data:
                        st.subheader("Étape 3: Informations Extraites")

                        # Display top-level info
                        st.markdown("#### Détails du Rapport, Client et Échantillon")
                        report_client_sample_info = {
                            "Laboratoire": extracted_data.get('report_info', {}).get('lab_name'),
                            "ID Rapport": extracted_data.get('report_info', {}).get('report_id'),
                            "Date d'émission": extracted_data.get('report_info', {}).get('issue_date'),
                            "Date de Validation": extracted_data.get('report_info', {}).get('validation_date'),
                            "Validateur": extracted_data.get('report_info', {}).get('validator_name'),
                            "Nom Client": extracted_data.get('client_info', {}).get('client_name'),
                            "Adresse Client": extracted_data.get('client_info', {}).get('client_address'),
                            "ID Client": extracted_data.get('client_info', {}).get('client_id'),
                            "Nom Produit": extracted_data.get('sample_info', {}).get('product_name'),
                            "Lot": extracted_data.get('sample_info', {}).get('lot_number'),
                            "ID Échantillon": extracted_data.get('sample_info', {}).get('sample_id'),
                            "Date réception": extracted_data.get('sample_info', {}).get('date_received'),
                            "Date analyse": extracted_data.get('sample_info', {}).get('date_analyzed'),
                            "Date collecte": extracted_data.get('sample_info', {}).get('date_collected'),
                            "Format Produit": extracted_data.get('sample_info', {}).get('product_format'),
                            "DLC/DLUO": extracted_data.get('sample_info', {}).get('best_before_date'),
                            "Fournisseur": extracted_data.get('sample_info', {}).get('supplier'),
                            "Code EAN": extracted_data.get('sample_info', {}).get('ean_code'),
                            "Conclusion Générale": extracted_data.get('conclusion')
                        }

                        # Convert to DataFrame for display, handling None/null
                        info_df_data = [{"Champ": key, "Valeur": value if value is not None else "N/A"} for key, value in report_client_sample_info.items()]
                        info_df = pd.DataFrame(info_df_data).set_index("Champ")
                        st.table(info_df)


                        # Display analysis results in a DataFrame
                        st.markdown("#### Résultats d'Analyse Détaillés")
                        analysis_results = extracted_data.get('analysis_results')

                        if analysis_results:
                            # Ensure analysis_results is a list of dictionaries
                            if isinstance(analysis_results, list) and all(isinstance(item, dict) for item in analysis_results):
                                results_df = pd.DataFrame(analysis_results)
                                # Define the desired column order
                                display_cols = ['parameter', 'result', 'unit', 'specification', 'uncertainty', 'method']
                                # Reorder columns and fill potential NaN values with empty strings for display
                                results_df = results_df.reindex(columns=display_cols).fillna('')

                                st.dataframe(results_df, use_container_width=True)

                                # Step 4: Download as Excel
                                st.subheader("Étape 4: Télécharger les Résultats")
                                excel_data = to_excel(results_df)
                                if excel_data:
                                    st.download_button(
                                        label="⬇️ Télécharger les résultats d'analyse en Excel",
                                        data=excel_data,
                                        file_name=f"resultats_analyse_{uploaded_file.name.replace('.pdf', '')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            else:
                                st.error("L'IA a retourné les résultats d'analyse dans un format inattendu (pas une liste de dictionnaires).")
                                st.json(analysis_results) # Show the raw result for debugging
                        else:
                            st.warning("Aucun résultat d'analyse n'a pu être extrait par l'IA.")

                    else:
                        st.error("Échec de l'extraction des informations structurées par l'IA.")

                except AttributeError:
                    st.error("Erreur : Le résultat OCR n'a pas le format attendu (attribut 'pages' manquant).")
                except Exception as e:
                    st.error(f"Une erreur inattendue s'est produite lors du traitement des données extraites : {e}")
                    # import traceback
                    # st.text(traceback.format_exc()) # Show full traceback for debugging

            elif ocr_result is not None and not ocr_result.pages:
                 st.warning("L'OCR n'a pas pu extraire de pages de texte de ce document.")
            elif ocr_result is None:
                 st.error("Échec de l'OCR.") # Message déjà géré par call_ocr_api, mais redondance pour clarté.

else:
    st.info("Veuillez uploader un fichier PDF pour commencer l'analyse.")

st.markdown("---")
st.markdown("Développé avec ❤️ et Mistral AI 🧠")
st.markdown("[Code source sur GitHub](<Lien vers votre dépôt si publié>)") # Optionnel : ajoutez un lien vers le code
