import streamlit as st
from mistralai import Mistral
import os
import json
import pandas as pd
from io import BytesIO

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Extracteur de Bulletins d'Analyse", layout="wide")

# --- Configuration API Mistral ---
try:
    API_KEY = st.secrets["API_KEY"]
    if not API_KEY:
        st.error("Clé API Mistral non trouvée dans Streamlit Secrets (API_KEY). Assurez-vous qu'elle est définie.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lors de l'accès à Streamlit Secrets : {e}")
    st.stop()

# Initialiser le client Mistral
try:
    client = Mistral(api_key=API_KEY)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    st.stop()

# Modèles à utiliser
OCR_MODEL = "mistral-ocr-latest"
LLM_MODEL = "mistral-large-latest"

# --- Fonctions pour l'OCR et l'Upload ---
@st.cache_data(show_spinner=False)
def upload_pdf_to_mistral(_file_content, file_name):
    try:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": file_name,
                "content": _file_content,
            },
            purpose="ocr"
        )
        st.success(f"Fichier '{file_name}' uploadé avec succès (ID: {uploaded_pdf.id}).")
        return uploaded_pdf.id
    except Exception as e:
        st.error(f"Erreur lors de l'upload du fichier à Mistral API: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_signed_url(_file_id):
    try:
        signed_url = client.files.get_signed_url(file_id=_file_id)
        st.success("URL signée pour l'OCR récupérée.")
        return signed_url.url
    except Exception as e:
        st.error(f"Erreur lors de la récupération de l'URL signée: {e}")
        return None

@st.cache_data(show_spinner=False)
def call_ocr_api(_signed_url):
    try:
        with st.spinner(f"Traitement OCR en cours avec le modèle '{OCR_MODEL}'..."):
            ocr_response = client.ocr.process(
                model=OCR_MODEL,
                document={
                    "type": "document_url",
                    "document_url": _signed_url,
                },
                include_image_base64=False
            )
        st.success("OCR terminé.")
        return ocr_response
    except Exception as e:
        st.error(f"Erreur lors de l'appel de l'API OCR: {e}")
        return None

# --- Fonction pour l'Extraction Structurée par IA ---
def extract_info_with_llm(ocr_text):
    json_schema = {
      "report_info": {
        "lab_name": "string or null",
        "report_id": "string or null",
        "issue_date": "string or null",
        "validation_date": "string or null",
        "validator_name": "string or null"
      },
      "client_info": {
        "client_name": "string or null",
        "client_address": "string or null",
        "client_id": "string or null"
      },
      "sample_info": {
        "product_name": "string or null",
        "lot_number": "string or null",
        "sample_id": "string or null",
        "date_received": "string or null",
        "date_analyzed": "string or null",
        "date_collected": "string or null",
        "product_format": "string or null",
        "best_before_date": "string or null",
        "supplier": "string or null",
        "ean_code": "string or null"
      },
      "analysis_results": [
        {
          "parameter": "string or null",
          "result": "string or null",
          "unit": "string or null",
          "specification": "string or null",
          "uncertainty": "string or null",
          "method": "string or null"
        }
      ],
      "conclusion": "string or null"
    }

    prompt = f"""
    You are an expert in analyzing laboratory food analysis reports (bulletins d'analyse).
    I will provide you with the full text extracted from such a report using OCR.
    The text may contain information from multiple pages, separated by '==NEW_PAGE=='.
    Your task is to extract the key information and structure it into a JSON object based strictly on the following schema.

    Pay close attention to the tables containing analysis results. You MUST extract EACH ROW from ALL analysis results tables into the 'analysis_results' array. Each object in the array should correspond to one row and contain the 'parameter', 'result', 'unit', 'specification', 'uncertainty', and 'method' columns as listed in the schema, extracting the value for that row.

    Ensure the JSON output is strictly valid and only contains the JSON object within a markdown code block formatted as ````json__BLOCK_CODE_BLOCK_1__`json
    {{...}}
    ````
    """

    try:
        with st.spinner(f"Analyse IA des résultats en cours avec le modèle '{LLM_MODEL}'..."):
            chat_response = client.chat.complete(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
        st.success("Analyse IA terminée.")

        response_content = chat_response.choices[0].message.content.strip()

        extracted_data = None
        json_string = None

        try:
            extracted_data = json.loads(response_content)
        except json.JSONDecodeError:
            st.warning("Le modèle n'a pas retourné de JSON direct. Recherche du bloc de code markdown...")
            if response_content.startswith("```json"):
                json_string = response_content[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()
            elif "```json" in response_content:
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
                    st.text(response_content)
                    return None
            else:
                st.error("L'IA a généré une réponse inattendue qui ne contient pas de JSON valide ou un bloc markdown JSON.")
                st.text("Réponse brute de l'IA:")
                st.text(response_content)
                return None

        if not isinstance(extracted_data, dict):
            st.error("L'IA n'a pas retourné un objet JSON de niveau supérieur valide.")
            st.text("Réponse brute de l'IA:")
            st.text(response_content)
            return None

        return extracted_data

    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'appel ou du traitement de la réponse de l'IA : {e}")
        return None

# --- Helper function for Excel download ---
def to_excel(df):
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Resultats Analyse')
        processed_data = output.getvalue()
        return processed_data
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier Excel : {e}")
        return None

# --- Interface Streamlit Principale ---
st.title("🔬 Extracteur Automatisé de Bulletins d'Analyses (IA)")
st.markdown("""
Uploadez un fichier PDF de bulletin d'analyse. Cette application utilisera l'OCR de Mistral AI pour lire le document,
puis un grand modèle de langage (LLM) pour extraire les informations clés et les résultats d'analyse sous forme structurée.
Les résultats seront affichés et pourront être téléchargés au format Excel.
""")

st.sidebar.header("Paramètres et Informations")
st.sidebar.info(f"Modèle OCR utilisé : `{OCR_MODEL}`")
st.sidebar.info(f"Modèle LLM utilisé pour l'extraction : `{LLM_MODEL}`")
show_raw_ocr = st.sidebar.checkbox("Afficher le texte OCR brut", value=False)

uploaded_file = st.file_uploader("📥 Choisissez un fichier PDF de bulletin d'analyse", type=["pdf"])

extracted_data = None

if uploaded_file:
    st.write(f"Fichier sélectionné: **{uploaded_file.name}**")

    file_content = uploaded_file.getvalue()

    st.subheader("Étape 1: OCR du Document")
    file_id = upload_pdf_to_mistral(file_content, uploaded_file.name)

    if file_id:
        signed_url = get_signed_url(file_id)

        if signed_url:
            ocr_result = call_ocr_api(signed_url)

            if ocr_result and ocr_result.pages:
                try:
                    pages_text = [page.markdown for page in ocr_result.pages]
                    full_text = "\n\n==NEW_PAGE==\n\n".join(pages_text)

                    if show_raw_ocr:
                        st.subheader("Texte OCR brut extrait")
                        st.text(full_text)

                    st.subheader("Étape 2: Extraction des Informations Structurées par IA")
                    extracted_data = extract_info_with_llm(full_text)

                    if extracted_data:
                        st.subheader("Étape 3: Informations Extraites")

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

                        info_df_data = [{"Champ": key, "Valeur": value if value is not None else "N/A"} for key, value in report_client_sample_info.items()]
                        info_df = pd.DataFrame(info_df_data).set_index("Champ")
                        st.table(info_df)

                        st.markdown("#### Résultats d'Analyse Détaillés")
                        analysis_results = extracted_data.get('analysis_results')

                        if analysis_results:
                            if isinstance(analysis_results, list) and all(isinstance(item, dict) for item in analysis_results):
                                results_df = pd.DataFrame(analysis_results)
                                display_cols = ['parameter', 'result', 'unit', 'specification', 'uncertainty', 'method']
                                results_df = results_df.reindex(columns=display_cols).fillna('')

                                st.dataframe(results_df, use_container_width=True)

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
                                st.json(analysis_results)
                        else:
                            st.warning("Aucun résultat d'analyse n'a pu être extrait par l'IA.")

                    else:
                        st.error("Échec de l'extraction des informations structurées par l'IA.")

                except AttributeError:
                    st.error("Erreur : Le résultat OCR n'a pas le format attendu (attribut 'pages' manquant).")
                except Exception as e:
                    st.error(f"Une erreur inattendue s'est produite lors du traitement des données extraites : {e}")

            elif ocr_result is not None and not ocr_result.pages:
                st.warning("L'OCR n'a pas pu extraire de pages de texte de ce document.")
            elif ocr_result is None:
                st.error("Échec de l'OCR (voir les logs ou les messages d'erreur précédents).")

else:
    st.info("Veuillez uploader un fichier PDF pour commencer l'analyse.")

st.markdown("---")
st.markdown("Développé avec ❤️ et Mistral AI 🧠")
st.markdown("[Code source sur GitHub](<Lien vers votre dépôt si publié>)")
