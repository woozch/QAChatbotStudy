import random
import gradio as gr
import os

from dotenv import load_dotenv


load_dotenv()

from langchain_openai import ChatOpenAI

# from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ChatMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.chains import create_tagging_chain

from typing import List
from scipy.spatial import distance


# load models
llama = LlamaAPI(os.environ.get("LLAMA_API_KEY"))
chat_openai = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

AVAILABLE_DB = ["None"]

try:
    answer_faiss_db = FAISS.load_local(
        "db/law_stackexchange/sentence_transformer/faiss_index_answer", embeddings_model
    )
    answer_retriever = answer_faiss_db.as_retriever()
    AVAILABLE_DB.append("Answer")
except:
    answer_faiss_db = None
    answer_retriever = None

try:
    qna_faiss_db = FAISS.load_local(
        "db/law_stackexchange/sentence_transformer/faiss_index_concat", embeddings_model
    )
    qna_retriever = qna_faiss_db.as_retriever()
    AVAILABLE_DB.append("Question&Answer")
except:
    qna_faiss_db = None
    qna_retriever = None

# experimental implementation
DEBUG = False  # debug print flag

PROMPT_TEMPLATE = """Please consider the content of the "Context" and take into account the "History" conversation when responding.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{history}

{context}

Question: {question}
Helpful Answer:"""

DEFAULT_MODEL_TYPE = "openai"
HISTORY_CONTEXT_SIZE = 10


def build_prompt_template(message: str, history: List[List[str]], context=""):
    history = [
        dialog
        for dialog in history
        if not dialog[1].startswith("Error: ") or dialog[1] != ""
    ]  # filter out responses with errors

    history_document = ""
    for dialog in history:
        history_document += ". ".join([elem.strip() for elem in dialog])

    return PROMPT_TEMPLATE.format(
        context="Context:\n" + context,
        history="History:\n" + history_document,
        question=message,
    )


def chat_response(message, history, *additional_args):
    model_type, temperature, db_type = additional_args

    if model_type == "llama":
        # Custom prompt with llama
        # TODO: message preprocessing, if question is not in detail, elaborate it by retrieving relevent context
        L2_THRESHOLD = 0.5
        # filter relevent history
        embeded_message = embeddings_model.embed_query(message)
        relevent_history = [
            dialog
            for dialog in history
            if distance.euclidean(
                embeddings_model.embed_query(
                    ". ".join([elem.strip() for elem in dialog])
                ),
                embeded_message,
            )
            < L2_THRESHOLD
        ]

        # retrieve context
        if db_type == "Answer":
            faiss_db = answer_faiss_db
        elif db_type == "Question&Answer":
            faiss_db = qna_faiss_db
        else:
            faiss_db = None

        if faiss_db is not None:
            retrived_docs_with_score = faiss_db.similarity_search_with_score(
                message, k=1
            )
            # use less token.. but several strategies can be used
            retrived_doc, l2_score = retrived_docs_with_score[0]
            if l2_score < L2_THRESHOLD:
                context = retrived_doc.page_content
            else:
                context = ""
        else:
            context = ""

        # TODO: add check logic for relevent history
        prompt = build_prompt_template(
            message, relevent_history[-HISTORY_CONTEXT_SIZE:], context
        )
        if DEBUG:
            print("########################  CONTEXT  ############################")
            print(retrived_docs_with_score)
            print("###############################################################")
            print("########################  PROMPT  ############################")
            print(prompt)
            print("###############################################################")
        api_request_json = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "function_call": "none",
        }
        response = llama.run(api_request_json)
        if response.status_code == 200:
            response_dict = response.json()
            response_message = response_dict["choices"][0]["message"]["content"]
        else:
            response_message = "Error: " + str(response.status_code)

    elif model_type == "openai":
        if db_type == "Answer":
            retriever = answer_retriever
        elif db_type == "Question&Answer":
            retriever = qna_retriever
        else:
            retriever = None
        # use langchain QA prompt
        chat_openai.temperature = float(temperature)  # assign temperature
        # print("chat_openai.temperature", chat_openai.temperature)
        if retriever is not None:
            qa = RetrievalQA.from_chain_type(
                llm=chat_openai, chain_type="stuff", retriever=retriever
            )  # not sure what the prompt that langchain using. how it works..
            response_message = qa.run(message)
        else:
            # without any retreival process
            chat_messages = [
                HumanMessage(content=message),
            ]
            response = chat_openai.invoke(chat_messages)
            response_message = response.content

    return response_message


def model_type_radio_response(model_type):
    return model_type


def temperature_slider_response(temperature):
    return temperature


def db_type_radio_response(db_type):
    return db_type


def build_interface() -> gr.Blocks:
    # construct the interface
    # demo = gr.ChatInterface(model_response)
    # gr.themes.builder()
    theme = None
    theme = gr.themes.Soft(
        # primary_hue="teal",
    )
    with gr.Blocks(theme=theme, title="Law News Bot", head="Chat") as demo:
        # demo.title = "Law News Bot"
        model_type_state = gr.State(DEFAULT_MODEL_TYPE)
        temperature_type_state = gr.State(0.7)  # chatgpt default
        db_type_state = gr.State(AVAILABLE_DB[-1])
        with gr.Row():
            with gr.Column(scale=4):
                chat_interface = gr.ChatInterface(
                    chat_response,
                    additional_inputs=[
                        model_type_state,
                        temperature_type_state,
                        db_type_state,
                    ],
                )
            with gr.Column(scale=1):
                model_type_radio = gr.Radio(
                    ["llama", "openai"],
                    value=DEFAULT_MODEL_TYPE,
                    label="Model Type",
                    interactive=True,
                )
                # set event for radio
                model_type_radio.select(
                    model_type_radio_response,
                    inputs=[model_type_radio],
                    outputs=[model_type_state],
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    interactive=True,
                )
                # set event for slider
                temperature_slider.change(
                    temperature_slider_response,
                    inputs=[temperature_slider],
                    outputs=[temperature_type_state],
                )
                db_type_radio = gr.Radio(
                    AVAILABLE_DB,
                    value=AVAILABLE_DB[-1],
                    label="Retreval DB Type",
                    interactive=True,
                )
                # set event for db radio
                db_type_radio.select(
                    db_type_radio_response,
                    inputs=[db_type_radio],
                    outputs=[db_type_state],
                )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
