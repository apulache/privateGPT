#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse

from langchain.embeddings import HuggingFaceInstructEmbeddings

import torch

load_dotenv()

runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
print("++++++++ " + runtime_device.upper() + " is available and will be use for Embeddings +++++++++")

if runtime_device == "cuda":
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME_GPU')
    local_persist_directory = os.environ.get('PERSIST_DIRECTORY_GPU')
else:
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME_CPU')
    local_persist_directory = os.environ.get('PERSIST_DIRECTORY_CPU')

print("++++++++ " + embeddings_model_name.upper() + " will be use +++++++++")
print("++++++++ " + local_persist_directory.upper() + " db repo will be use +++++++++" + "\n")



persist_directory = local_persist_directory

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    #embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': runtime_device})
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    retriever.search_kwargs['distance_metric'] = 'cos'
    
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)

        case "GPT4All":
            # llm = GPT4All(model=model_path, n_ctx=model_n_ctx, 
            #               callbacks=callbacks, verbose=False, n_threads=8, use_mlock=True)  # backend='gptj'

            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, 
                          verbose=False, n_threads=6, n_predict=640, embedding=True, temp=0.3, use_mlock=True)
            
            # llm = GPT4All(model=model_path, n_ctx=model_n_ctx,
            #       callbacks=callbacks, verbose=False,
            #       n_threads=8, use_mlock=True, backend='gptj',
            #       embedding=True, n_predict=512,
            #       temp=0.3, repeat_penalty=3, n_parts=4) # backend='gptj'
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, 
                                     return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        #query = input("\nEnter a query: ")
        query = input("\n" + '\U00002753' + " Enter a query: " + "\n")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Print the result
        # print("\n\n> Question:")
        # print(query)
        # print("\n> Answer:")
        # print(answer)

        print("\n\n> Question: " + '\U00002753') # \U0002753 -- red question mark
        print(query + "\n")
        print("\n> Answer: " + '\U0001F4AC')
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            #print("\n> " + document.metadata["source"] + ":")
            print("\n\n> " + '\U0001F4DA' + ' ' + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
