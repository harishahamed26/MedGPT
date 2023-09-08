# Importing the necessary Library and package
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter #python -m nltk.downloader all 
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from io import StringIO
from ChatTemplate import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from audio_recorder_streamlit import audio_recorder
import docx
import streamlit as st
import openai
import os
from io import BytesIO
import speech_recognition as sr
import pickle
from transformers import pipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
from transformers import SpeechT5HifiGan
import nltk

@st.cache
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
st.session_state.ConvoHistory = None

#load_dotenv()
my_secret_key = st.secrets["OPENAI_API_KEY"]
#my_secret_key = os.getenv("OPENAI_API_KEY")
SampleRate = 16000

# Getting the PDF File 
def get_PDF_File():
    PDF_Documents = st.file_uploader(label = 'Upload your pdf here', accept_multiple_files = True, type = 'pdf', label_visibility='hidden')
    if st.button('Upload PDF'):
        if len(PDF_Documents) != 0:
            file = PDF_text_preprocessing(PDF_Documents)
            st.success('Uploaded')
            return file
        else:
            st.error('Upload the file')

# Getting the Word File
def get_Word_File():
    word_Documents = st.file_uploader(label = 'Upload your word here', accept_multiple_files = True, type = 'docx', label_visibility='hidden')
    if st.button('Upload WORD'):
        if len(word_Documents) != 0:
            file = Word_text_preprocessing(word_Documents)
            st.success('Uploaded')
            return file
        else:
            st.error('Upload the file')

# Getting the txt File
def get_txt_File():
    txt_Documents = st.file_uploader(label = 'Upload your txt here', accept_multiple_files = True, type = 'txt', label_visibility='hidden')
    if st.button('Upload TXT'):
        if len(txt_Documents) != 0:
            file = Txt_text_preprocessing(txt_Documents)
            st.success('Uploaded')
            return file
        else:
            st.error('Upload the file')

# Getting the Audio File
def get_Audio_File():
    Audio_Files = st.file_uploader(label = 'Upload your Audio here', type = ['mp3', 'm4a', 'wav'], label_visibility='hidden')
    if st.button('Upload Audio'):
        if Audio_Files is not None:
            file = AudioToText_preprocessing(Audio_Files)
            st.success('Uploaded')
            return file
        else:
            st.error('Upload the file')

# Extracting of PDF Files
def PDF_text_preprocessing(PDF_Documents):
    text = ""
    PDF_Split = ""
    for pdf in PDF_Documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()   

# Chunking of the PDF Files
    PDF_Split = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 1000,
        separator = '\n', 
        length_function = len
    )
    PDF_chunks = PDF_Split.split_text(text)
    return PDF_chunks

# Extracting of Word Files
def Word_text_preprocessing(Word_Documents):
    text = ""
    Word_Split = ""
    for word in Word_Documents:
        word_reader = docx.Document(word)
        for para in word_reader.paragraphs:
            text += para.text
    
# Chunking of the Word Files
    Word_Split = NLTKTextSplitter(
        chunk_size = 500,
        chunk_overlap = 500,
        separator = '\n', 
        length_function = len 
    )

    Word_chunks = Word_Split.split_text(text)
    return Word_chunks

# Extracting of .txt Files
def Txt_text_preprocessing(txt_Documents):
    text = ""
    txt_Split = ""
    for i in range(0, len(txt_Documents)):
        txt_reader = txt_Documents[i].getvalue()
        txt_reader_io = StringIO(txt_reader.decode(("utf-8")))
        for txt in txt_reader_io:
            text += txt
    
# Chunking of the .txt Files
    txt_Split = NLTKTextSplitter(
        chunk_size = 500,
        chunk_overlap = 500,
        separator = '\n', 
        length_function = len
    )

    txt_chunks = txt_Split.split_text(text)
    return txt_chunks

# Extracting the Audio files 
def AudioToText_preprocessing(Audio_Files):
    audioTxt = openai.Audio.transcribe("whisper-1", Audio_Files, api_key= my_secret_key)
    audioTxtString = str(audioTxt)

    # Chunking of the transcripted  Files
    txt_Split = NLTKTextSplitter(
        chunk_size = 500,
        chunk_overlap = 500,
        separator = '\n', 
        length_function = len
    )

    audio_chunks = txt_Split.split_text(audioTxtString)
    return audio_chunks

# Embedding and Storing of Chunks into the Vector DB (FAISS)
def Chunk_Embedding(Combined_Chunks):
    Embedding = OpenAIEmbeddings() # "model": "text-embedding-ada-002"
    VectoreDB = FAISS.from_texts(texts = Combined_Chunks, embedding = Embedding)
    return VectoreDB

# Processing of the LLM, Chain and Memory
def Process_LLM(VectorStorage):
    Memory =  ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    LLM = OpenAI(temperature = 0.75)
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you information is not present in the uploaded document, don't try to make up an answer.
    If the question is hello or hi or hey then just answer Hello!  how may i help you.

    {context}

    Question: {question}
    Answer in english:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    Conversational_Chain = RetrievalQA.from_chain_type( 
        llm = LLM,
        memory = Memory, 
        chain_type = "stuff",
        retriever = VectorStorage.as_retriever(), 
        chain_type_kwargs = {"prompt": PROMPT}, 
        verbose = True
    )
    return Conversational_Chain

# Creating fucntion for connversation session
def ConvoPlace(queries, ConversationModel, target_language):
    source_language='English'
    response = ConversationModel({'query': queries})
    
    st.session_state.ConvoHistory = response['chat_history']
    print(f'checking the value of response {response}')
  
    for i, msg in enumerate(st.session_state.ConvoHistory):
        if i % 2 == 0:
          if target_language == 'English':
            st.write(user_template.replace('{{msg}}', msg.content), unsafe_allow_html = True)
          else:
            LanguagePrompt = f"Translate the following '{source_language}' text to '{target_language}': {msg.content}"
            TranslatorResponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": LanguagePrompt}
            ],
            max_tokens=3500,
            n=1,
            stop=None,
            temperature=0.75, api_key = my_secret_key
            )

            translation = TranslatorResponse.choices[0].message.content.strip()
            st.write(user_template.replace('{{msg}}', translation), unsafe_allow_html = True)
            

            #inputs = processor(text= msg.content, return_tensors="pt")
            #speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            #st.audio(speech.numpy(), sample_rate = SampleRate)
        else:
          if target_language == 'English':
            st.write(bot_template.replace('{{msg}}', msg.content), unsafe_allow_html = True)
          else:
            LanguagePrompt = f"Translate the following '{source_language}' text to '{target_language}': {msg.content}"
            TranslatorResponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": LanguagePrompt}
            ],
            max_tokens=3500,
            n=1,
            stop=None,
            temperature=0.75, api_key = my_secret_key
            )

            translation = TranslatorResponse.choices[0].message.content.strip()
            st.write(user_template.replace('{{msg}}', translation), unsafe_allow_html = True)
            


            
    with st.sidebar:
      st.header('Click here Reset the conversation')
      if st.button('Reset Chat'):
          st.session_state.ConvoHistory = None
          st.success('Chat Cleared')
      st.write('------')

      response = st.session_state.ConvoHistory[len(st.session_state.ConvoHistory)-1].content
      inputs = processor(text= response, return_tensors="pt")
      speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
      st.header('Last Response generated')
      st.audio(speech.numpy(), sample_rate = SampleRate)
            #st.audio(speech.numpy(), sample_rate = SampleRate)
    

def main():
    st.set_page_config(page_title = 'MedGPT', page_icon='📚', layout='wide')
    st.header('MedGPT Chatbot 🤖')
    st.write(css, unsafe_allow_html = True)
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    # Style.css to remove footer and make some css changes
    with open('style.css', 'r') as style:
        st.markdown(f'<style> {style.read()} </style>', unsafe_allow_html= True)

    # Initialization of the session state 
    if "ConversationModel"  not in st.session_state:
        st.session_state.ConversationModel = None

    if "ConvoHistory"  not in st.session_state:
        st.session_state.ConvoHistory = None

    # Tabs to select he file uploading method
    col1, col2, col3, col4 = st.tabs(['PDF', 'Word', 'txt', 'audio'])
    with col1:
       PDF_chunk =  get_PDF_File()
            
    with col2:
       Word_chunk = get_Word_File()
    
    with col3:
        Txt_chunk = get_txt_File()
    
    with col4:
        Audio_chunk = get_Audio_File()

    # Creating a process button to proceed with the embedding of the chucks   
    if PDF_chunk is not None or Word_chunk is not None or Txt_chunk is not None or Audio_chunk is not None:
        Combined_Chunks = []
        if PDF_chunk is not None:
            Combined_Chunks.extend(PDF_chunk)   
        if Word_chunk is not None:
            Combined_Chunks.extend(Word_chunk) 
        if Txt_chunk is not None:
            Combined_Chunks.extend(Txt_chunk)
        if Audio_chunk is not None:
            Combined_Chunks.extend(Audio_chunk)    
    
        VectorStorage =  Chunk_Embedding(Combined_Chunks)
        st.session_state.ConversationModel = Process_LLM(VectorStorage)  
    
    with st.sidebar:

      st.write('------')
      st.header('Translation')
      target_language = st.selectbox('Select the language', 
                              ("English"
                              ,"French"
                              )
                              ,label_visibility = "hidden"
      )

      st.write('------')
  
      # Audio Speech recognition
      st.header('Click here for audio query')
      Recording = audio_recorder(text='', icon_size="1.5x", neutral_color= '#06EC02', recording_color='#EC2C02')
      
        
    if Recording is not None:  
      ConvoPlace(whisper(Recording)['text'], st.session_state.ConversationModel, target_language)

    queries = st.chat_input(placeholder = 'Ask about your document')
    
    
    if queries:
      ConvoPlace(queries, st.session_state.ConversationModel, target_language)
      
    
    

if __name__ == '__main__':
    main()
    
    