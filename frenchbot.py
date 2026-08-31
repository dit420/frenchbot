import os
import streamlit as st

from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def get_groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
        available = list(st.secrets.keys())
        raise RuntimeError(f"st.secrets has no GROQ_API_KEY. Keys found: {available}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"st.secrets could not be read: {type(e).__name__}: {e}")


def load_llm():
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env locally, "
            "or to the app's Secrets on Streamlit Community Cloud."
        )
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.3,
        groq_api_key=groq_api_key,
    )
    return llm


def main():
    st.title("Tuteur de francais")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pose ta question de francais")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

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

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )

            response = qa_chain.invoke({"query": prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            pages = sorted({doc.metadata.get("page_label", "?") for doc in source_documents})
            result_to_show = result
            if pages:
                result_to_show += f"\n\n*Manuel, page(s) {', '.join(pages)}*"
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
