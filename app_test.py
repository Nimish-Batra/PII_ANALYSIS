# import re
# import streamlit as st
# import tempfile
# import json
# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
# from faker import Faker
# import spacy
# import pandas as pd
# from io import BytesIO
# from langchain.schema import Document
# from langchain.chat_models import AzureChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnableLambda
# from langchain.schema import StrOutputParser
# import zipfile
# import os
#
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ea-openai.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "2355a247f79f4b8ea2adaa0929cd32c2"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
#
# fake = Faker()
#
# nlp = spacy.load("en_core_web_sm")
#
#
# mappings = {}
#
#
# def process_regex_input(input_regex):
#     # Unescape any incorrectly escaped backslashes
#     return input_regex.replace('\\\\', '\\')
#
#
# def save_mappings():
#     with open("mappings.json", "w") as file:
#         json.dump(mappings, file, indent=4)
#
# def delete_previous_files():
#     files_to_delete = ["mappings.json", "anonymized_text.txt", "anonymized_file.zip"]
#     for file in files_to_delete:
#         try:
#             if os.path.exists(file):
#                 os.remove(file)
#         except PermissionError as e:
#             st.warning(f"Could not delete {file}: {e}")
#
# def validate_entity(entity_type, value):
#     """
#     Validate detected entity to ensure it belongs to the correct category.
#     """
#     doc = nlp(value)
#     for ent in doc.ents:
#         if entity_type == "Name" and ent.label_ == "PERSON":
#             return True
#         if entity_type == "Location" and ent.label_ in ["GPE", "LOC"]:
#             return True
#     return False
#
#
# def replace_with_fake_data(text,patterns):
#
#     processed_text = text
#
#     for label, pattern in patterns.items():
#         matches = re.finditer(pattern, text)
#         for match in matches:
#             value = match.group(0)
#
#             if label == "Name" and not validate_entity("Name", value):
#                 continue  # Skip replacement if not validated as a name
#
#             if label == "Location" and not validate_entity("Location", value):
#                 continue  # Skip replacement if not validated as a location
#
#             if value not in mappings:
#                 # Generate a fake value and store it in mappings
#                 if label == "Name":
#                     fake_value = fake.name()
#                 elif label == "Location":
#                     fake_value = fake.city()
#                 elif label == "US_Driving_License":
#                     fake_value = fake.bothify(text="??#######")
#                 elif label == "US_SSN":
#                     fake_value = fake.ssn()
#                 elif label == "DOB":
#                     fake_value = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%m/%d/%Y')
#                 elif label == "Address":
#                     fake_value = fake.address().replace('\n', ', ')
#                 elif label == "Employee_ID":
#                     fake_value = fake.bothify(text="EID######")
#                 elif label == "IP_Address":
#                     fake_value = fake.ipv4()
#                 elif label == "Credit_Card":
#                     fake_value = fake.credit_card_number()
#                 elif label == "CVV":
#                     fake_value = fake.credit_card_security_code()
#                 elif label == "Email":
#                     fake_value = fake.email()
#                 elif label == "Phone_Number":
#                     fake_value = fake.phone_number()
#                 elif label == "Credit_Expiry":
#                     fake_value = fake.credit_card_expire()
#                 elif label == "IBAN_Code":
#                     fake_value = fake.iban()
#                 elif label == "Crypto_Wallet":
#                     fake_value = fake.bothify(text="1#########################")
#                 elif label == "Passport":
#                     fake_value = fake.bothify(text="#########")
#                 elif label == "US_ITIN":
#                     fake_value = fake.ssn()
#                 elif label == "URLs":
#                     fake_value = fake.url()
#                 elif label == "NRIC":
#                     fake_value = fake.bothify(text="S#######D")
#
#                 else:
#                      fake_value = fake.bothify(text="???######??")    #11 hexa-numeric random pattern for custom fields
#
#                 mappings[value] = fake_value
#             else:
#                 fake_value = mappings[value]
#
#             # Replace the value in the processed_text
#             processed_text = processed_text.replace(value, fake_value)
#
#     save_mappings()  # Save mappings to file
#     return processed_text
#
# def get_text_chunks(row_text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=2000,
#         chunk_overlap=100,
#         length_function=len
#     )
#
#     chunks = text_splitter.split_text(row_text)
#     documents = [Document(page_content=chunk) for chunk in chunks]
#     return documents
#
# def get_vectostore(text_chunks):
#     embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding')
#     vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
#     return vectorstore
#
# def create_anonymizer_chain(vectorstore, question, anonymizer_mapping):
#     template = """Answer the question based only on the following context:
#     {context}
#
#     Question: {anonymized_question}
#     """
#     # Anonymize the question using the provided mapping
#     anonymized_question = anonymize_question(question, anonymizer_mapping)
#     st.write("Anonymized Question:", anonymized_question)
#
#     # Retrieve relevant documents based on the anonymized question
#     retriever = vectorstore.as_retriever()
#     relevant_docs = retriever.get_relevant_documents(anonymized_question)
#     context = "\n".join([doc.page_content for doc in relevant_docs])
#
#     # Create the prompt with context and anonymized question
#     formatted_prompt = template.format(context=context, anonymized_question=anonymized_question)
#
#     # Initialize the LLM and prompt
#     llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo", model_name="gpt-4", temperature=0.50)
#     prompt = ChatPromptTemplate.from_template(formatted_prompt)
#
#     # Build the chain
#     chain =prompt | llm | StrOutputParser()
#     # Deanonymize the response
#     response_chain = chain | RunnableLambda(lambda x: replace_fake_data_with_real(x, mappings))
#
#     # Invoke the chain with the question
#     response = response_chain.invoke({"question": anonymized_question})
#
#     # Print the answer (this will be the deanonymized response)
#     print("Answer------->", response)
#     return response
#
#
# # def process_excel_for_qna(file_obj):
# #     df = pd.read_excel(file_obj, nrows=1000)
# #     # Convert the DataFrame into a text format
# #     text = df.to_string(index=False)
# #     return text
#
# def anonymize_question(question, mapping):
#     for real, fake in mapping.items():
#         question = question.replace(real, fake)
#     return question
#
# def replace_fake_data_with_real(response, mappings):
#     for real, fake in mappings.items():
#         response = response.replace(fake, real)
#     return response
#
#
#
# def process_csv_or_excel(file_obj, file_type):
#     if file_type == 'csv':
#         df = pd.read_csv(file_obj, nrows=1000)
#     elif file_type == 'xlsx':
#         df = pd.read_excel(file_obj, nrows=1000)
#
#     # Convert numeric-like columns to strings
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col] = df[col].astype(str)
#         elif df[col].dtype in ['int64', 'float64']:
#             df[col] = df[col].apply(lambda x: str(x) if not pd.isnull(x) else x)
#
#     # anonymization process
#     df_anonymized = df.applymap(lambda x: replace_with_fake_data(str(x), combined_patterns) if isinstance(x, str) else x)
#
#     return df_anonymized
#
#
# # Save anonymized CSV/Excel
# def save_anonymized_file(df=None, text=None, original_file_name=None, file_type=None):
#     with BytesIO() as buffer:
#         with zipfile.ZipFile(buffer, 'w') as zip_file:
#             if df is not None:
#                 if file_type == 'csv':
#                     anonymized_file_name = f"anonymized_{original_file_name}"
#                     df.to_csv(anonymized_file_name, index=False)
#                     zip_file.writestr(anonymized_file_name, df.to_csv(index=False))
#                 elif file_type == 'xlsx':
#                     anonymized_file_name = f"anonymized_{original_file_name}.xlsx"
#                     with BytesIO() as excel_buffer:
#                         df.to_excel(excel_buffer, index=False, engine='openpyxl')
#                         zip_file.writestr(anonymized_file_name,excel_buffer.getvalue())
#
#             if text is not None:
#                 zip_file.writestr("anonymized_text.txt", text)
#
#             # Save the mappings.json file inside the zip
#             save_mappings()
#             with open("mappings.json", "r") as mappings_file:
#                 zip_file.writestr("mappings.json", mappings_file.read())
#
#         # Set the buffer position to the beginning
#         buffer.seek(0)
#
#         # Provide a download button for the zip file
#         st.download_button(
#             label="Download Anonymized File and Mappings",
#             data=buffer.getvalue(),
#             file_name=f"anonymized_{original_file_name}.zip",
#             mime="application/zip"
#         )
#
# def get_chunks(file_obj, file_type):
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_path = temp_file.name
#         temp_file.write(file_obj.read())
#
#     if file_type == 'pdf':
#         loader = PyPDFLoader(temp_path)
#     elif file_type in ['docx', 'doc']:
#         loader = Docx2txtLoader(temp_path)
#     # elif file_type in ['xlsx', 'xls']:
#     #     loader = UnstructuredExcelLoader(temp_path)
#     else:
#         return []
#
#     chunks = loader.load_and_split()
#     return chunks
#
# def get_combined_patterns(selected_state):
#     combined_patterns =st.session_state.patterns.copy()
#     if selected_state in state_specific_patterns:
#         combined_patterns.update(state_specific_patterns[selected_state])
#     return combined_patterns
#
# if 'patterns' not in st.session_state:
#     st.session_state.patterns = {
#         # Process address patterns first to avoid name overriding
#         "Address": r"\b(\d+\s[A-Za-z0-9\s,.]+(?:St|Rd|Ave|Blvd|Ln|Dr|Ct|Pl|Sq|Ter|Way)(?:,\s[A-Za-z\s]+,\s[A-Z]{2}\s\d{5}))\b",  # Improved Address pattern
#         "Name": r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b",  # Adjusted regex for names (first, middle, last)
#         "US_Driving_License": r"\b[A-Z]{1,2}\d{7,8}\b",  # US driving license number (simplified)
#         "US_SSN": r"\b(\d{3}-\d{2}-\d{4})\b",  # US Social Security Number
#         "DOB": r"\b((?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/\d{4})\b",  # Date of Birth in MM/DD/YYYY format
#         "Employee_ID": r"\b(?:Employee ID[:\s\-]*|EID[:\s\-]*|EMP[:\s\-]*)(\w{2,10}-?\d{3,8})\b",  # Employee ID (various formats)
#         "IP_Address": r"\b((?:\d{1,3}\.){3}\d{1,3})\b",  # IPv4 address
#         "Credit_Card": r"\b(?:\d{4}[-\s]?){3}\d{4}|\d{16}\b", # Credit Card number
#         "CVV": r"\bCVV[:\s\-]*(\d{3,4})\b",  # CVV code (3-4 digits after 'CVV')
#         "Email": r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7})\b",  # Email address
#         "Phone_Number": r"\b((?:\(?\d{3}\)?[-\s.]?)?\d{3}[-\s.]?\d{4})\b",  # US Phone number
#         "Credit_Expiry": r"\b(0[1-9]|1[0-2])/([0-9]{2}|[0-9]{4})\b",  # Credit Card Expiry Date
#         "IBAN_Code": r"\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{0,4}\b",  # Improved IBAN Code pattern
#         "Crypto_Wallet": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",  # Cryptocurrency Wallet Address
#         "Passport": r"\b[A-Z0-9]{9}\b",  # Passport number (simplified)
#         "US_ITIN": r"\b\d{3}-\d{2}-\d{4}\b",  # US ITIN
#         "URLs": r"\bhttps?:\/\/[^\s/$.?#].[^\s]*\b",  # URLs
#         "NRIC": r"\b[A-Z]{1}\d{7}[A-Z]{1}\b",  # NRIC number (Singapore example)
#     }
#
# specific_states = [
#     "California", "Illinois", "Nevada", "Virginia",
#     "Washington", "Texas", "New York", "Massachusetts"
# ]
#
# # Calculate other states not included in specific_states
# all_states = [
#     "Alabama", "Alaska", "Arizona", "Arkansas", "Colorado",
#     "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
#     "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
#     "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
#     "New Hampshire", "New Jersey", "New Mexico", "North Carolina", "North Dakota",
#     "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
#     "South Dakota", "Tennessee", "Utah", "Vermont", "West Virginia", "Wisconsin", "Wyoming"
# ]
#
# # Determine the list of other states
# other_states = [state for state in all_states if state not in specific_states]
#
# # Add "Other States" as an option in the dropdown
# dropdown_options = specific_states + ["Other States"]
#
#
# state_specific_patterns = {
#     "California": {
#         "Internet_Browsing_History": r"https?://(?:www\.)?[^\s/$.?#].[^\s]*",
#         "Geolocation_Data":r"\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b|\b\d{1,3}°\s?[NS],?\s?\d{1,3}°\s?[EW]\b",
#         "Biometric_Data": r"(\d{1,3}[\s]*cm|\d{2,3}[\s]*kg|[\d]{2,3}/[\d]{2,3}/[\d]{2,4})",
#         "Audio_Data": r"\b(Audio|Sound|Recording|Voice)[^\n]*",
#         "Microphone_Data": r"\b(Microphone|Mic)[^\n]*",
#         "Camera_Data": r"\b(Camera|Video|Image|Picture|Photo|Snapshot)[^\n]*",
#         "Sensor_Data": r"\b(Sensor|Motion|Temperature|Pressure|Proximity|Light|Accelerometer|Gyroscope)[^\n]*",
#         "Inferences": r"\b(infer|inferred|assume|assumed|predict|predicted|likely|probability|probable|tend|tendency|behavior|pattern|insight|suggest|suggested|hypothesis|indicative|correlation|analysis)[^\n]*"
#     },
#     "New_York": {
#         "Biometric_Data": r"(\d{1,3}[\s]*cm|\d{2,3}[\s]*kg|[\d]{2,3}/[\d]{2,3}/[\d]{2,4})",
#         "User_Credentials": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,7}[\s]*[\w!@#$%^&*()_+-=]*"
#     },
#     "Texas": {
#         "Voter_ID": r"\b\d{10}\b"
#     },
#     "Massachusetts": {
#         "Account_Numbers": r"\b\d{12,16}\b",
#         "Encryption_Requirement": r"(encrypted|encryption|secure|ssl|tls)[^\n]*"
#     },
#     "Illinois": {
#         "Biometric_Data": r"(fingerprint|retina|iris)[^\n]*"
#     },
#     "Nevada": {
#         "userID": r"\b(userID[_-]?[a-zA-Z0-9]+|[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,7})\b",
#         "sessionID": r"\b(sessionID[_-]?[a-zA-Z0-9]+|[a-fA-F0-9]{16,})\b",
#         "trackingID": r"\b(trackingID[_-]?[a-zA-Z0-9]+|[a-zA-Z0-9]{10,})\b",
#         "cookieID": r"\b(cookieID[_-]?[a-zA-Z0-9]+|[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})\b",
#         "deviceID": r"\b(deviceID[_-]?[a-zA-Z0-9]+|[a-fA-F0-9]{8,}|[A-Z0-9]{8,12})\b",
#         "Internet_Activity": r"https?://(?:www\.)?[^\s/$.?#].[^\s]*",
#     },
#     "Virginia": {
#         "Race": r"\b(Asian|Black|White|African American|Caucasian|Latino|Hispanic|Native American|Pacific Islander|Mixed Race|Other Race|Race: [A-Za-z]+)\b",
#         "Ethnicity": r"\b(Hispanic|Latino|Non-Hispanic|Jewish|Arab|European|African|Asian|Ethnicity: [A-Za-z]+)\b",
#         "Religion": r"\b(Christian|Muslim|Jewish|Buddhist|Hindu|Atheist|Agnostic|Catholic|Protestant|Evangelical|Orthodox|Religion: [A-Za-z]+)\b",
#         "Sexual_Orientation": r"\b(Heterosexual|Homosexual|Bisexual|Asexual|Pansexual|Queer|Gay|Lesbian|Transgender|LGBTQ|Sexual Orientation: [A-Za-z]+)\b",
#         "Citizenship_Status": r"\b(Citizen|Non-Citizen|Permanent Resident|Green Card Holder|Visa Holder|Immigrant|Refugee|Asylum Seeker|Undocumented|Naturalized Citizen|Citizenship Status: [A-Za-z]+)\b"
#     },
#     "Washington": {
#         "Genetic_Data": r"(DNA|RNA|gene|chromosome|genetic)[^\n]*",
#         "User_Generated_Content": r"(post|comment|review|upload|content)[^\n]*"
#     },
# }
#
#
# # Streamlit UI
# st.title("PII Detection")
#
# final_chunks = []
# docs = st.file_uploader("Upload your files here and click on Process",
#                         type=['docx', 'doc', 'pdf', 'csv', 'xlsx', 'xls'], accept_multiple_files=True)
#
#
#
# # Handling state selection logic in Python
# selected_state = st.selectbox("Select a State", options=dropdown_options)
#
# if selected_state == "Other States":
#     st.info(f"Other States Include:\n\n " + "   ,   ".join(other_states))
# combined_patterns = get_combined_patterns(selected_state)
# # print("Combined Patterns---->",combined_patterns)
#
# with st.sidebar:
#     st.header("Add Custom Field for Masking")
#     custom_field_name = st.text_input("Field Name")
#     if custom_field_name:
#         st.info(
#             f"Write a regular expression for the {custom_field_name}. If you don't have one, simply [click here](https://regex-generator.olafneumann.org/?sampleText=GB29%20NWBK%206016%201331%209268%2019.&flags=i&selection=0%7CCombination%20%5BAlphanumeric%20characters%20%2B%20Character%5D) to create one.\n\n**Important:** Your pattern must start and end with \\b. For example, \\b[0-9]{{12}}\\b.")
#         custom_field_regex = st.text_input("Regular Expression for Field")
#
#     if st.button("Add Custom Field"):
#         if custom_field_name and custom_field_regex:
#             try:
#                 # Compile the regular expression to validate it and ensure it's correct
#                 processed_regex = process_regex_input(custom_field_regex)
#                 re.compile(processed_regex)
#                 # Store the regex directly as it is inputted, ensuring it's correctly interpreted
#                 if custom_field_name not in combined_patterns:
#                     combined_patterns[custom_field_name] = custom_field_regex
#                     st.success(f"Custom field '{custom_field_name}' added successfully!")
#                     # st.write("Current Patterns:")
#                     # st.info("\n".join([f"- {key}" for key in combined_patterns]))
#                 else:
#                     st.error(f"Field name '{custom_field_name}' already exists.")
#             except re.error:
#                 st.error("Invalid regular expression. Please check your input.")
#         else:
#             st.error("Please provide both a field name and a regular expression.")
#     if st.expander("Current patterns"):
#         st.write("Current Patterns:")
#         st.info("\n".join([f"- {key}" for key in combined_patterns]))
#
#
#
# if st.button("Process"):
#     if docs:
#         with st.spinner("Processing files..."):
#             for f_obj in docs:
#                 file_type = f_obj.name.split('.')[-1].lower()
#
#                 if file_type in ['pdf', 'docx', 'doc']:
#                     chunks = get_chunks(f_obj, file_type)
#                     final_chunks.extend(chunks)
#                     full_text = "\n".join([chunk.page_content for chunk in final_chunks])
#                     print("Pattern---------->", combined_patterns)
#                     mask = replace_with_fake_data(full_text, combined_patterns)
#                     st.subheader("Anonymized Text")
#                     st.text_area("Anonymized Text", mask, height=300)
#                     # Save and download anonymized text along with mappings.json
#                     save_anonymized_file(text=mask, original_file_name=f"anonymized_{f_obj.name}")
#
#                     # Store processed chunks and vectorstore
#                     chunks = get_text_chunks(mask)
#                     vectorstore = get_vectostore(chunks)
#                     st.session_state["vectorstore"] = vectorstore
#                     st.session_state["mappings"] = mappings
#                     st.success("Files processed successfully!")
#
#                 elif file_type in ['csv', 'xlsx', 'xls']:
#                     df_anonymized = process_csv_or_excel(f_obj, file_type)
#                     st.subheader(f"Anonymized Data - {f_obj.name}")
#                     st.dataframe(df_anonymized)
#                     save_anonymized_file(df=df_anonymized, original_file_name=f_obj.name, file_type=file_type)
#                     # Convert Excel content to text for QnA
#                     excel_text = df_anonymized.to_string()
#                     print("Excel Text------->",excel_text)
#                     chunks = get_text_chunks(excel_text)
#                     print("Chunks------>", chunks)
#                     vectorstore = get_vectostore(chunks)
#                     st.session_state["vectorstore"] = vectorstore
#                     st.session_state["mappings"] = mappings
#                     st.success("Files processed successfully!")
#
#
#
#     else:
#         st.error("Please upload at least one document.")
#
# user_question = st.text_input("Ask a question about your documents:")
#
# if st.button("Submit Question"):
#     if user_question:
#         if "vectorstore" in st.session_state:
#             vectorstore = st.session_state["vectorstore"]
#             mappings = st.session_state["mappings"]
#             with st.spinner("Processing your question..."):
#                 try:
#                     response = create_anonymizer_chain(vectorstore, user_question, mappings)
#                     st.subheader("Response")
#                     st.write(response)
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#         else:
#             st.error("Please process the documents first.")
#     else:
#         st.warning("Please enter a question before submitting.")


import re
import streamlit as st
import tempfile
import json
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from faker import Faker
import spacy
import pandas as pd
from io import BytesIO
from langchain.schema import Document
from langchain.chat_models import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema import StrOutputParser
import zipfile
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ea-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "2355a247f79f4b8ea2adaa0929cd32c2"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

fake = Faker()

nlp = spacy.load("en_core_web_sm")

def process_regex_input(input_regex):
    return input_regex.replace('\\\\', '\\')

def save_mappings(mappings, filename):
    with open(filename, "w") as file:
        json.dump(mappings, file, indent=4)

def delete_previous_files():
    files_to_delete = ["mappings.json", "anonymized_text.txt", "anonymized_file.zip"]
    for file in files_to_delete:
        try:
            if os.path.exists(file):
                os.remove(file)
        except PermissionError as e:
            st.warning(f"Could not delete {file}: {e}")

def validate_entity(entity_type, value):
    doc = nlp(value)
    for ent in doc.ents:
        if entity_type == "Name" and ent.label_ == "PERSON":
            return True
        if entity_type == "Location" and ent.label_ in ["GPE", "LOC"]:
            return True
    return False

def replace_with_fake_data(text, patterns, mappings):
    processed_text = text
    for label, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            value = match.group(0)

            if label == "Name" and not validate_entity("Name", value):
                continue

            if label == "Location" and not validate_entity("Location", value):
                continue

            if value not in mappings:
                fake_value = generate_fake_value(label)
                mappings[value] = fake_value
            else:
                fake_value = mappings[value]

            processed_text = processed_text.replace(value, fake_value)

    return processed_text

def generate_fake_value(label):
    if label == "Name":
        return fake.name()
    elif label == "Location":
        return fake.city()
    elif label == "US_Driving_License":
        return fake.bothify(text="??#######")
    elif label == "US_SSN":
        return fake.ssn()
    elif label == "DOB":
        return fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%m/%d/%Y')
    elif label == "Address":
        return fake.address().replace('\n', ', ')
    elif label == "Employee_ID":
        return fake.bothify(text="EID######")
    elif label == "IP_Address":
        return fake.ipv4()
    elif label == "Credit_Card":
        return fake.credit_card_number()
    elif label == "CVV":
        return fake.credit_card_security_code()
    elif label == "Email":
        return fake.email()
    elif label == "Phone_Number":
        return fake.phone_number()
    elif label == "Credit_Expiry":
        return fake.credit_card_expire()
    elif label == "IBAN_Code":
        return fake.iban()
    elif label == "Crypto_Wallet":
        return fake.bothify(text="1#########################")
    elif label == "Passport":
        return fake.bothify(text="#########")
    elif label == "US_ITIN":
        return fake.ssn()
    elif label == "URLs":
        return fake.url()
    elif label == "NRIC":
        return fake.bothify(text="S#######D")
    else:
        return fake.bothify(text="???######??")

def get_text_chunks(row_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(row_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def get_vectostore(text_chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding')
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def save_anonymized_file(df=None, text=None, original_file_name=None, file_type=None, mappings=None):
    with BytesIO() as buffer:
        with zipfile.ZipFile(buffer, 'w') as zip_file:
            if df is not None:
                if file_type == 'csv':
                    anonymized_file_name = f"anonymized_{original_file_name}"
                    df.to_csv(anonymized_file_name, index=False)
                    zip_file.writestr(anonymized_file_name, df.to_csv(index=False))
                elif file_type == 'xlsx':
                    anonymized_file_name = f"anonymized_{original_file_name}.xlsx"
                    with BytesIO() as excel_buffer:
                        df.to_excel(excel_buffer, index=False, engine='openpyxl')
                        zip_file.writestr(anonymized_file_name, excel_buffer.getvalue())

            if text is not None:
                zip_file.writestr(f"anonymized_{original_file_name}.txt", text)

            if mappings is not None:
                mappings_file_name = f"{original_file_name}_mappings.json"
                save_mappings(mappings, mappings_file_name)
                with open(mappings_file_name, "r") as mappings_file:
                    zip_file.writestr(mappings_file_name, mappings_file.read())

        buffer.seek(0)

        st.download_button(
            label="Download Anonymized File and Mappings",
            data=buffer.getvalue(),
            file_name=f"anonymized_{original_file_name}.zip",
            mime="application/zip"
        )

def get_chunks(file_obj, file_type):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(file_obj.read())

    if file_type == 'pdf':
        loader = PyPDFLoader(temp_path)
    elif file_type in ['docx', 'doc']:
        loader = Docx2txtLoader(temp_path)
    else:
        return []

    chunks = loader.load_and_split()
    return chunks

def get_combined_patterns(selected_state):
    combined_patterns = st.session_state.patterns.copy()
    if selected_state in state_specific_patterns:
        combined_patterns.update(state_specific_patterns[selected_state])
    return combined_patterns

def process_files(docs, combined_patterns):
    for f_obj in docs:
        file_type = f_obj.name.split('.')[-1].lower()
        mappings = {}

        if file_type in ['pdf', 'docx', 'doc']:
            chunks = get_chunks(f_obj, file_type)
            full_text = "\n".join([chunk.page_content for chunk in chunks])
            mask = replace_with_fake_data(full_text, combined_patterns, mappings)

            st.subheader(f"Anonymized Text - {f_obj.name}")
            st.text_area(f"Anonymized Text - {f_obj.name}", mask, height=300)

            # Save and download anonymized text along with mappings.json
            save_anonymized_file(text=mask, original_file_name=f_obj.name, mappings=mappings)

            # Store processed chunks and vectorstore
            chunks = get_text_chunks(mask)
            vectorstore = get_vectostore(chunks)
            st.session_state[f"vectorstore_{f_obj.name}"] = vectorstore
            st.session_state[f"mappings_{f_obj.name}"] = mappings
            st.success(f"File {f_obj.name} processed successfully!")

        elif file_type in ['csv', 'xlsx', 'xls']:
            df_anonymized = process_csv_or_excel(f_obj, file_type, combined_patterns, mappings)
            st.subheader(f"Anonymized Data - {f_obj.name}")
            st.dataframe(df_anonymized)
            save_anonymized_file(df=df_anonymized, original_file_name=f_obj.name, file_type=file_type, mappings=mappings)

            # Convert Excel content to text for QnA
            excel_text = df_anonymized.to_string()
            chunks = get_text_chunks(excel_text)
            vectorstore = get_vectostore(chunks)
            st.session_state[f"vectorstore_{f_obj.name}"] = vectorstore
            st.session_state[f"mappings_{f_obj.name}"] = mappings
            st.success(f"File {f_obj.name} processed successfully!")

    # After processing, update the file names in the session state
    st.session_state["vectorstore_options"] = [f_obj.name for f_obj in docs]

def process_csv_or_excel(file_obj, file_type, combined_patterns, mappings):
    if file_type == 'csv':
        df = pd.read_csv(file_obj, nrows=1000)
    elif file_type == 'xlsx':
        df = pd.read_excel(file_obj, nrows=1000)

    # Convert numeric-like columns to strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].apply(lambda x: str(x) if not pd.isnull(x) else x)

    # Anonymization process
    df_anonymized = df.applymap(lambda x: replace_with_fake_data(str(x), combined_patterns, mappings) if isinstance(x, str) else x)

    return df_anonymized

def anonymize_question(question, mapping):
    for real, fake in mapping.items():
        question = question.replace(real, fake)
    return question

def replace_fake_data_with_real(response, mappings):
    for real, fake in mappings.items():
        response = response.replace(fake, real)
    return response

def create_anonymizer_chain(vectorstore, question, anonymizer_mapping):
    template = """Answer the question based only on the following context:
    {context}

    Question: {anonymized_question}
    """
    anonymized_question = anonymize_question(question, anonymizer_mapping)
    st.write("Anonymized Question:", anonymized_question)

    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(anonymized_question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    formatted_prompt = template.format(context=context, anonymized_question=anonymized_question)

    llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo", model_name="gpt-4", temperature=0.50)
    prompt = ChatPromptTemplate.from_template(formatted_prompt)

    chain = prompt | llm | StrOutputParser()
    response_chain = chain | RunnableLambda(lambda x: replace_fake_data_with_real(x, anonymizer_mapping))

    response = response_chain.invoke({"question": anonymized_question})
    return response

if 'patterns' not in st.session_state:
    st.session_state.patterns = {
        "Address": r"\b(\d+\s[A-Za-z0-9\s,.]+(?:St|Rd|Ave|Blvd|Ln|Dr|Ct|Pl|Sq|Ter|Way)(?:,\s[A-Za-z\s]+,\s[A-Z]{2}\s\d{5}))\b",
        "Name": r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b",
        "US_Driving_License": r"\b[A-Z]{1,2}\d{7,8}\b",
        "US_SSN": r"\b(\d{3}-\d{2}-\d{4})\b",
        "DOB": r"\b((?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/\d{4})\b",
        "Employee_ID": r"\b(?:Employee ID[:\s\-]*|EID[:\s\-]*|EMP[:\s\-]*)(\w{2,10}-?\d{3,8})\b",
        "IP_Address": r"\b((?:\d{1,3}\.){3}\d{1,3})\b",
        "Credit_Card": r"\b(?:\d{4}[-\s]?){3}\d{4}|\d{16}\b",
        "CVV": r"\bCVV[:\s\-]*(\d{3,4})\b",
        "Email": r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7})\b",
        "Phone_Number": r"\b((?:\(?\d{3}\)?[-\s.]?)?\d{3}[-\s.]?\d{4})\b",
        "Credit_Expiry": r"\b(0[1-9]|1[0-2])/([0-9]{2}|[0-9]{4})\b",
        "IBAN_Code": r"\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{0,4}\b",
        "Crypto_Wallet": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
        "Passport": r"\b[A-Z0-9]{9}\b",
        "US_ITIN": r"\b\d{3}-\d{2}-\d{4}\b",
        "URLs": r"\bhttps?:\/\/[^\s/$.?#].[^\s]*\b",
        "NRIC": r"\b[A-Z]{1}\d{7}[A-Z]{1}\b",
    }

specific_states = [
    "California", "Illinois", "Nevada", "Virginia",
    "Washington", "Texas", "New York", "Massachusetts"
]

all_states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "New Hampshire", "New Jersey", "New Mexico", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Utah", "Vermont", "West Virginia", "Wisconsin", "Wyoming"
]

other_states = [state for state in all_states if state not in specific_states]

dropdown_options = specific_states + ["Other States"]

state_specific_patterns = {
    "California": {
        "Internet_Browsing_History": r"https?://(?:www\.)?[^\s/$.?#].[^\s]*",
        "Geolocation_Data":r"\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b|\b\d{1,3}°\s?[NS],?\s?\d{1,3}°\s?[EW]\b",
        "Biometric_Data": r"(\d{1,3}[\s]*cm|\d{2,3}[\s]*kg|[\d]{2,3}/[\d]{2,3}/[\d]{2,4})",
        "Audio_Data": r"\b(Audio|Sound|Recording|Voice)[^\n]*",
        "Microphone_Data": r"\b(Microphone|Mic)[^\n]*",
        "Camera_Data": r"\b(Camera|Video|Image|Picture|Photo|Snapshot)[^\n]*",
        "Sensor_Data": r"\b(Sensor|Motion|Temperature|Pressure|Proximity|Light|Accelerometer|Gyroscope)[^\n]*",
        "Inferences": r"\b(infer|inferred|assume|assumed|predict|predicted|likely|probability|probable|tend|tendency|behavior|pattern|insight|suggest|suggested|hypothesis|indicative|correlation|analysis)[^\n]*"
    },
    "New_York": {
        "Biometric_Data": r"(\d{1,3}[\s]*cm|\d{2,3}[\s]*kg|[\d]{2,3}/[\d]{2,3}/[\d]{2,4})",
        "User_Credentials": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,7}[\s]*[\w!@#$%^&*()_+-=]*"
    },
    "Texas": {
        "Voter_ID": r"\b\d{10}\b"
    },
    "Massachusetts": {
        "Account_Numbers": r"\b\d{12,16}\b",
        "Encryption_Requirement": r"(encrypted|encryption|secure|ssl|tls)[^\n]*"
    },
    "Illinois": {
        "Biometric_Data": r"(fingerprint|retina|iris)[^\n]*"
    },
    "Nevada": {
        "userID": r"\b(userID[_-]?[a-zA-Z0-9]+|[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,7})\b",
        "sessionID": r"\b(sessionID[_-]?[a-zA-Z0-9]+|[a-fA-F0-9]{16,})\b",
        "trackingID": r"\b(trackingID[_-]?[a-zA-Z0-9]+|[a-zA-Z0-9]{10,})\b",
        "cookieID": r"\b(cookieID[_-]?[a-zA-Z0-9]+|[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})\b",
        "deviceID": r"\b(deviceID[_-]?[a-zA-Z0-9]+|[a-fA-F0-9]{8,}|[A-Z0-9]{8,12})\b",
        "Internet_Activity": r"https?://(?:www\.)?[^\s/$.?#].[^\s]*",
    },
    "Virginia": {
        "Race": r"\b(Asian|Black|White|African American|Caucasian|Latino|Hispanic|Native American|Pacific Islander|Mixed Race|Other Race|Race: [A-Za-z]+)\b",
        "Ethnicity": r"\b(Hispanic|Latino|Non-Hispanic|Jewish|Arab|European|African|Asian|Ethnicity: [A-Za-z]+)\b",
        "Religion": r"\b(Christian|Muslim|Jewish|Buddhist|Hindu|Atheist|Agnostic|Catholic|Protestant|Evangelical|Orthodox|Religion: [A-Za-z]+)\b",
        "Sexual_Orientation": r"\b(Heterosexual|Homosexual|Bisexual|Asexual|Pansexual|Queer|Gay|Lesbian|Transgender|LGBTQ|Sexual Orientation: [A-Za-z]+)\b",
        "Citizenship_Status": r"\b(Citizen|Non-Citizen|Permanent Resident|Green Card Holder|Visa Holder|Immigrant|Refugee|Asylum Seeker|Undocumented|Naturalized Citizen|Citizenship Status: [A-Za-z]+)\b"
    },
    "Washington": {
        "Genetic_Data": r"(DNA|RNA|gene|chromosome|genetic)[^\n]*",
        "User_Generated_Content": r"(post|comment|review|upload|content)[^\n]*"
    },
}

# Streamlit UI
st.title("PII Detection")

docs = st.file_uploader("Upload your files here and click on Process",
                        type=['docx', 'doc', 'pdf', 'csv', 'xlsx', 'xls'], accept_multiple_files=True)

selected_state = st.selectbox("Select a State", options=dropdown_options)

if selected_state == "Other States":
    st.info(f"Other States Include:\n\n " + "   ,   ".join(other_states))
combined_patterns = get_combined_patterns(selected_state)

with st.sidebar:
    st.header("Add Custom Field for Masking")
    custom_field_name = st.text_input("Field Name")
    if custom_field_name:
        st.info(
            f"Write a regular expression for the {custom_field_name}. If you don't have one, simply [click here](https://regex-generator.olafneumann.org/?sampleText=GB29%20NWBK%206016%201331%209268%2019.&flags=i&selection=0%7CCombination%20%5BAlphanumeric%20characters%20%2B%20Character%5D) to create one.\n\n**Important:** Your pattern must start and end with `\\b`. For example, `\\b[0-9]{{12}}\\b`.")
        custom_field_regex = st.text_input("Regular Expression for Field")

    if st.button("Add Custom Field"):
        if custom_field_name and custom_field_regex:
            try:
                processed_regex = process_regex_input(custom_field_regex)
                re.compile(processed_regex)
                if custom_field_name not in combined_patterns:
                    combined_patterns[custom_field_name] = custom_field_regex
                    st.success(f"Custom field '{custom_field_name}' added successfully!")
                else:
                    st.error(f"Field name '{custom_field_name}' already exists.")
            except re.error:
                st.error("Invalid regular expression. Please check your input.")
        else:
            st.error("Please provide both a field name and a regular expression.")
    if st.expander("Current patterns"):
        st.write("Current Patterns:")
        st.info("\n".join([f"- {key}" for key in combined_patterns]))

if st.button("Process"):
    if docs:
        with st.spinner("Processing files..."):
            process_files(docs, combined_patterns)
    else:
        st.error("Please upload at least one document.")

# File selection dropdown
if "vectorstore_options" in st.session_state and st.session_state["vectorstore_options"]:
    selected_file = st.selectbox("Select the file to query", st.session_state["vectorstore_options"])

    user_question = st.text_input("Ask a question about your documents:")

    if st.button("Submit Question"):
        if user_question:
            if selected_file and f"vectorstore_{selected_file}" in st.session_state:
                vectorstore = st.session_state[f"vectorstore_{selected_file}"]
                mappings = st.session_state[f"mappings_{selected_file}"]

                with st.spinner("Processing your question..."):
                    try:
                        response = create_anonymizer_chain(vectorstore, user_question, mappings)
                        st.subheader("Response")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.error("Please process the documents first.")
        else:
            st.warning("Please enter a question before submitting.")
else:
    st.info("No files processed yet. Please upload and process files first.")
