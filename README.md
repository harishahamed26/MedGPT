**Description**

This repository contains a Python script called app.py that allows users to summarize a PDF, Word and .TXT document using natural language processing. The script uses the gradio library for building a simple web interface for users to input the PDF, Word and .TXT file path and a custom prompt for summarization.


Installing dependencies for running this project locally
To get started, you need to have Python installed on your machine and then create a virtual environment
to avoid any conflicts with other projects that may be present in your system. Here are some steps to
install all necessary packages before starting development or deployment of our application:

1. To create a new virtual environment in mac or windows or linux refer the below link 

https://realpython.com/python-virtual-environments-a-primer/


2. virtualenv venv # creates a new python env called' .venv' inside 'MEDGPT/' folder
source./venv/bin/activate   # activate the newly created virtual environment

3. Clone the repository 

4. Navigate into MEDGPT directory ( cd /path/to/medgpt )

5. Now let's move over to installing the required packages 

You can install the required libraries using the following command:

pip install -r requirement.txt

6. It is necessary to execute the below command as well to perform the nltk text splitting
    
python -m nltk.downloader all 

7. Create a .env file and paste your OpenAI API key 

OPENAI_API_KEY = ''

    To get your OpenAi key Click on the below link:
    
    https://beta.openai.com/account/api-keys

8. Once all the required package get completed Just execute the below command in the terminal:

streamlit run app.py



**Necessary Packages:**

pip install streamlit openai tiktoken langchain nltk python-docx python-dotenv PyPDF2 faiss-cpu



For Google Collab

!npm install localtunnel

!pip install -q streamlit openai tiktoken langchain nltk python-docx python-dotenv PyPDF2 faiss-cpu pip install audio-recorder-streamlit SpeechRecognition transformers

!python -m nltk.downloader all 

Steps:

1. Import all the files

2. create a .env file and include the OpenAI API key 

3. Write the app.py using the below command

%%writefile app.py

4. Execute the streamlit command along with npx command to start the local server 

!streamlit run /content/app.py & npx localtunnel --port 8501

5. Once the server is connected copy the external url IP address and submit to the server



