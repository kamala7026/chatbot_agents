# core/adservice_manager.py
import requests
import json
from typing import Dict, Any
import pandas as pd
from datetime import datetime

from utils.logger_config import logger
from core.common.config import ADSERVICE_API_URL

class AdserviceApiService:
    """Manages the business logic for interacting with the Adservice API."""

    def get_lifecycles_data(self, node_id: str, buids: str, file_id: str, start_date: str, end_date: str) -> str:
        """
        Calls the Lifecycles API, fetches data, and returns a formatted Markdown table.
        """
        logger.info("Calling the Lifecycles API via AdserviceManager.")
        
        headers = {
            "x-ottg-principal-userid": "GCP9827VMW877",
            "x-ottg-principal-orgid": "GC07495337UA",
            "x-ottg-caller-application": "adweb",
            "x-ottg-caller-application-node": node_id,
            "x-requested-buids": buids,
            "Content-Type": "application/json"
        }

        params = {
            "where_ContentFile.standardCustomerAttribute.ContentFile.fileID": file_id,
            "where_startDate": start_date,
            "where_endDate": end_date,
            "returnFormattedDate": "false",
            "where_searchPerspective": "ContentFile",
            "where_searchType": "active",
            "where_Process.standardCustomerAttribute.Process.direction": "both",
            "totalCount": "true",
            "distinctSummary": "true"
        }

        try:
            req = requests.Request('GET', ADSERVICE_API_URL, params=params, headers=headers)
            prepared_req = req.prepare()
            logger.debug(f"Calling Lifecycles API. Final URL: {prepared_req.url}")

            response = requests.get(ADSERVICE_API_URL, headers=headers, params=params, timeout=30)
            logger.debug(f"API Response Status Code: {response.status_code}")
            logger.debug(f"API Raw Response Text: {response.text}")
            response.raise_for_status()
            
            result_json = response.json()
            return self._format_json_to_markdown(result_json)

        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err} - Response: {http_err.response.text}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during API call: {e}", exc_info=True)
            return f"An unexpected error occurred: {e}"

    def _format_json_to_markdown(self, data: Dict[str, Any]) -> str:
        """Formats the JSON response from the Lifecycles API into a Markdown table using pandas."""
        
        lifecycles = data.get("lifecycle", [])
        if not lifecycles:
            return "No lifecycle data found with the current parameters."

        all_rows_data = []

        for lifecycle_item in lifecycles:
            extracted_data = {}

            if data.get('resultSetStatus'):
                if data['resultSetStatus'].get('statusCode') is not None:
                    extracted_data['Status Code'] = data['resultSetStatus']['statusCode']
                if data['resultSetStatus'].get('statusMessage') is not None:
                    extracted_data['Status Message'] = data['resultSetStatus']['statusMessage']

            if data.get('totalCount') is not None:
                extracted_data['Total Count'] = data['totalCount']

            if data.get('distinctSummary'):
                summary = data['distinctSummary']
                if summary.get('partnerInfo') and summary['partnerInfo']:
                    partner_info = summary['partnerInfo'][0]
                    if partner_info.get('buReference') and partner_info['buReference'].get('companyName') is not None:
                        extracted_data['Company Name'] = partner_info['buReference']['companyName']
                    if partner_info.get('totalCount') is not None:
                        extracted_data['Partner Total Count'] = partner_info['totalCount']
                if summary.get('senderAddresses') and summary['senderAddresses']:
                    extracted_data['Sender Address'] = summary['senderAddresses'][0].get('address')
                if summary.get('receiverAddreses') and summary['receiverAddreses']:
                    extracted_data['Receiver Address'] = summary['receiverAddreses'][0].get('address')
                if summary.get('processingDirection') and summary['processingDirection']:
                    extracted_data['Processing Direction'] = summary['processingDirection'][0].get('processingDirection')
                if summary.get('documentType') and summary['documentType']:
                    extracted_data['Document Type'] = summary['documentType'][0].get('documentType')
                if summary.get('reconStatus') and summary['reconStatus']:
                    extracted_data['Recon Status'] = summary['reconStatus'][0].get('reconStatus')
                if summary.get('reprocessedStatus') and summary['reprocessedStatus']:
                    extracted_data['Reprocessed Status'] = summary['reprocessedStatus'][0].get('reprocessed')

            if lifecycle_item.get('perspectiveType') is not None:
                extracted_data['Perspective Type'] = lifecycle_item['perspectiveType']
            if lifecycle_item.get('lifecycleCreateDate') is not None:
                timestamp_ms = lifecycle_item['lifecycleCreateDate']
                dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
                extracted_data['Lifecycle Create Date'] = dt_object.strftime('%Y-%m-%d %H:%M:%S')

            if lifecycle_item.get('contentFile'):
                content_file = lifecycle_item['contentFile']
                if content_file.get('sender') and content_file['sender'].get('ediAddress') is not None:
                    extracted_data['Content File Sender EDI Address'] = content_file['sender']['ediAddress']
                if content_file.get('receiver') and content_file['receiver'].get('ediAddress') is not None:
                    extracted_data['Content File Receiver EDI Address'] = content_file['receiver']['ediAddress']
                if content_file.get('snrf') is not None:
                    extracted_data['Content File SNRF'] = content_file['snrf']
                if content_file.get('contentId') is not None:
                    extracted_data['Content File Content ID'] = content_file['contentId']
                if content_file.get('docType') is not None:
                    extracted_data['Content File Document Type'] = content_file['docType']

            if lifecycle_item.get('processReference') and lifecycle_item['processReference']:
                extracted_data['Process ID'] = lifecycle_item['processReference'][0].get('processId')

            if lifecycle_item.get('selectedFields') and lifecycle_item['selectedFields'].get('selectedField'):
                for field in lifecycle_item['selectedFields']['selectedField']:
                    if field.get('fieldName') and field.get('fieldValue') is not None:
                        field_name = field['fieldName']
                        field_value = field['fieldValue']
                        column_name = f'Selected Field: {field_name}'

                        if field_name == 'fileCreateDate' and field.get('fieldValueType') == 'DATE':
                            timestamp_ms = int(field_value)
                            dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
                            extracted_data[column_name] = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            if column_name not in extracted_data:
                                extracted_data[column_name] = field_value
            
            all_rows_data.append(extracted_data)

        if not all_rows_data:
            return "Could not parse any data from the API response."

        df = pd.DataFrame(all_rows_data)

        desired_columns = [
            'Status Code', 'Status Message', 'Total Count', 'Company Name', 'Partner Total Count',
            'Sender Address', 'Receiver Address', 'Processing Direction', 'Document Type', 'Recon Status',
            'Reprocessed Status', 'Perspective Type', 'Lifecycle Create Date', 'Content File Sender EDI Address',
            'Content File Receiver EDI Address', 'Content File SNRF', 'Content File Content ID',
            'Content File Document Type', 'Process ID', 'Selected Field: snrf', 'Selected Field: docType',
            'Selected Field: fileCreateDate', 'Selected Field: partnerName', 'Selected Field: fileDirection'
        ]
        
        existing_columns = [col for col in desired_columns if col in df.columns]
        df_filtered = df[existing_columns]

        preamble = "The JSON data has been successfully formatted into a tabular form, as you requested.\n\nHere is the table:"
        markdown_table = df_filtered.to_markdown(index=False)
        return preamble + "\n\n" + str(markdown_table) 