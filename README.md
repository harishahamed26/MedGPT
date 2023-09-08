**#MedGPT Bot Based on Large Language Models**


**Description**

This repository contains a Python script called appStreamlit.py that allows users to summarize a PDF, Word dot TXT, and Audio document using natural language processing. The script uses the Streamlit library to build a simple web interface for users to input the PDF, Word, dot TXT, and Audio file path and a custom prompt for summarization.

Website: [https://medgpt-raq-bot.streamlit.app/]([url](https://medgpt-raq-bot.streamlit.app/))

Architecture:

![MedGPT Architecture](https://github.com/harishahamed26/MedGPT/assets/36252984/d1c55d20-e537-4272-8f2d-ba10b9efd869)


![MedGPT Architecture Description](https://github.com/harishahamed26/MedGPT/assets/36252984/502a08de-e948-42ef-8bcb-18cd5f145357)


Installing dependencies for running this project locally
To get started, you need to have Python installed on your machine and then create a virtual environment
to avoid any conflicts with other projects that may be present in your system. Here are some steps to
install all necessary packages before starting development or deployment of our application:

1. To create a new virtual environment in Mac or windows or Linux refer to the below link 

https://realpython.com/python-virtual-environments-a-primer/


2. virtualenv venv # creates a new python env called' .venv' inside 'MEDGPT/' folder
source./venv/bin/activate   # activate the newly created virtual environment

3. Clone the repository 

4. Navigate into the MEDGPT directory ( cd /path/to/medgpt )

5. Now let's move over to installing the required packages 

You can install the required libraries using the following command:

pip install -r requirement.txt

6. It is necessary to execute the below command as well to perform the nltk text splitting
    
python -m nltk.downloader all 

7. Create a .env file and paste your OpenAI API key 

OPENAI_API_KEY = ''

    To get your OpenAi key Click on the below link:
    
    https://beta.openai.com/account/api-keys

8. Once all the required packages are completed Just execute the below command in the terminal:

streamlit run app.py



**Necessary Packages:**

!pip install -r requirements.txt


**For Google Collab**

!npm install localtunnel

!python -m nltk.downloader all 


!streamlit run /content/app.py & npx localtunnel --port 8501

Once the server is connected copy the external URL IP address and submit it to the server




