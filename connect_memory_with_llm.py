import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

# Step 1: Setup LLM (Groq)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def load_llm():
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
    )
    return llm


# Step 2: Connect LLM with FAISS and create chain
CUSTOM_PROMPT_TEMPLATE = """
Tu es un tuteur de francais chaleureux et pedagogue, qui aide des etudiants
indiens a apprendre le francais avec leur manuel scolaire.

Utilise l'extrait du manuel ci-dessous comme reference quand il est utile,
mais tu n'es pas limite a ce contexte : tu es un professeur de francais complet.
Explique la grammaire, le vocabulaire et la conjugaison, corrige les phrases,
donne des exemples supplementaires, et reponds aux questions meme quand le
manuel ne les couvre pas directement. Ne dis jamais que tu ne sais pas.

Reponds dans la langue de la question de l'etudiant (francais ou anglais),
avec des explications claires, simples et encourageantes, adaptees a un debutant.

Extrait du manuel : {context}
Question de l'etudiant : {question}
"""


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Load database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
