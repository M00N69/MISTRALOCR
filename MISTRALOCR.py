def extract_info_with_llm(ocr_text):
    """Uses a Mistral LLM to extract structured information from OCR text."""
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
    You are an expert in analyzing laboratory food analysis reports.
    I will provide you with the full text extracted from such a report using OCR.
    The text may contain information from multiple pages, separated by '==NEW_PAGE=='.
    Your task is to extract the following key information and structure it into a JSON object:

    1. Laboratory Name
    2. Report ID
    3. Issue Date
    4. Client Name
    5. Client Address
    6. Product Name
    7. Lot Number
    8. Sample ID
    9. Date Received
    10. Date Analyzed
    11. Date Collected
    12. Product Format
    13. Best Before Date
    14. Supplier
    15. EAN Code
    16. Conclusion

    Additionally, extract each row from the analysis results tables into the 'analysis_results' array. Each object in the array should contain the 'parameter', 'result', 'unit', 'specification', 'uncertainty', and 'method' columns.

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
