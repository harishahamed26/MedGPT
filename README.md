# MedGPT Bot Based on Large Language Models
----

Blog: [https://medium.com/@harishahamed26/medgpt-bot-based-on-large-language-models-using-langchain-streamlit-702cb135ea8d](https://medium.com/@harishahamed26/medgpt-bot-based-on-large-language-models-using-langchain-streamlit-702cb135ea8d)


[s22141508-PGXPO23-Poster (60.96 × 142.24 cm).pdf](https://github.com/harishahamed26/MedGPT/files/12566897/s22141508-PGXPO23-Poster.60.96.x.142.24.cm.pdf)


**Description**

Welcome to the MedGPT Bot repository! This project contains a Python script appStreamlit.py, that harnesses the power of natural language processing to summarize various types of documents, including PDFs, Word files (.doc and .docx), text files (.txt), and audio documents. The user interface for this script is built using the Streamlit library, allowing users to input the file path and a custom prompt for summarization.

https://medgpt-raq-bot.streamlit.app/

----

**Architecture**

![MedGPT Architecture](https://github.com/harishahamed26/MedGPT/assets/36252984/553d3c2d-1366-4022-abfa-05bb6f718b41)



![MedGPT Architecture Description](https://github.com/harishahamed26/MedGPT/assets/36252984/502a08de-e948-42ef-8bcb-18cd5f145357)


**Future Work Architecture**

![Future Architecture png](https://github.com/harishahamed26/MedGPT/assets/36252984/671ed58c-b27f-403b-8a1d-6820dd823663)


**STS Integration**

![STS](https://github.com/harishahamed26/MedGPT/assets/36252984/23a9b19a-f79f-4dc2-ae08-4de805aefcae)


-----

**Home Page of the MedGPT Bot**

<img width="1440" alt="Homepage-MedGPT" src="https://github.com/harishahamed26/MedGPT/assets/36252984/871e3c50-2271-49fc-8825-e1b9eca0c459">


-----

**English Conversation of the MedGPT Bot**

<img width="1440" alt="English-MedGPT" src="https://github.com/harishahamed26/MedGPT/assets/36252984/136cd070-3881-40fc-afe7-3476bcd9a871">


-----

**French Conversation of the MedGPT Bot**

<img width="1440" alt="French-MedGPT" src="https://github.com/harishahamed26/MedGPT/assets/36252984/b7d2febb-63e6-4757-a37f-9df8bb0f5236">

-----


### Getting Started

Before you can run this project locally, make sure you have Python installed on your machine. It's also recommended to create a virtual environment to manage project dependencies and avoid conflicts with other projects. Here are the steps to set up and run MedGPT Bot:

1. Create a new virtual environment using the instructions provided in this [link](https://realpython.com/python-virtual-environments-a-primer/).

2. Clone this repository to your local machine.

3. Navigate to the `MEDGPT` directory using the command `cd /path/to/medgpt`.

4. Install the required Python packages by running:

   ```
   pip install -r requirements.txt
   ```

5. Execute the following command to download the necessary NLTK data for text splitting:

   ```
   python -m nltk.downloader all
   ```

6. Create a `.env` file and paste your OpenAI API key:

   ```
   OPENAI_API_KEY = 'your_openai_api_key_here'
   ```

   To obtain your OpenAI API key, visit [this link](https://beta.openai.com/account/api-keys).

7. Once you have all the required packages and settings in place, run the application with the following command:

   ```
   streamlit run appColab.py
   ```

Necessary Packages

You can install the necessary Python packages using the following command:

```
pip install -r requirements.txt
```

For Google Colab

If you plan to run this project on Google Colab, follow these additional steps:

1. Install the localtunnel package using npm:

   ```
   !npm install localtunnel
   ```

2. Download the necessary NLTK data:

   ```
   !python -m nltk.downloader all
   ```

3. Run the application and expose it using localtunnel:

   ```
   !streamlit run /content/appColab.py & npx localtunnel --port 8501
   ```

   Once the server is connected, copy the external URL IP address and submit it to the server.

---

Feel free to modify and expand upon this README file as needed to provide additional information, usage instructions, or any other details specific to your project.
