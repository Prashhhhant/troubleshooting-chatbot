import os
import re
from pyngrok import ngrok
from langchain.chat_models import ChatOpenAI
import json
from dotenv import load_dotenv
from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from flask import Flask, render_template, request, jsonify

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

load_dotenv()
chat_history = []
ngrok_key = os.getenv("NGROK_KEY")

app = Flask(__name__)
ngrok.set_auth_token(ngrok_key)
public_url = ngrok.connect(5000).public_url


class YouTubeSearchTool(BaseTool):
    name = "YouTubeSearch"
    description = (
        "search for youtube videos associated with computer diagnosis. "
        "the input to this tool should be a comma separated list, "
        "the first part contains a issue and the second a "
        "number that is the maximum number of video results "
        "to return aka num_results. the second part is optional"
    )

    def _search(self, person: str, num_results: int) -> str:
        from youtube_search import YoutubeSearch

        results = YoutubeSearch(person, num_results).to_json()
        data = json.loads(results)
        url_suffix_list = [video["url_suffix"] for video in data["videos"]]
        return str(url_suffix_list)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        values = query.split(",")
        person = values[0]
        if len(values) > 1:
            num_results = int(values[1])
        else:
            num_results = 2
        return self._search(person, num_results)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(
            "YouTubeSearchTool  does not yet support async")


@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        question = request.form.get('question')
        response = generate_response(question)

        return jsonify({'response': response})
    else:
        return render_template('bot.html')


def generate_response(question):
    if "youtube" in question:
        tool = YouTubeSearchTool()
        query = f"{question},1"

        result = tool.run(query)
        print(result)
        clean_string = re.sub(r'[\[\]\'"]', '', result)
        clean_string = re.sub(r'&.*', '', clean_string)
        prefix = 'https://www.youtube.com'

        link = prefix + str(clean_string)
        return f"Here is a YouTube video that may help you:<br><a href='{link}' target='_blank'>{link}</a>"

    query = question
    try:
        # create embeddings of the PDF
        embeddings = OpenAIEmbeddings()
        persist_directory = 'db'
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)  # loading pre trained vectors
    except:
        return "PLease train data to start QNA..."

    prompt_template = """Use the following pieces of context to answer the users question. You are a chatbot your name is 'ComputerGini'. Your tasks is to help your users solve problems they are having with their Computers/Laptops.
    The User will share what issues they are having You will help them find the cause and best solution of the issue..
    ----------------
    {context}

    Human: {question}
    Chatbot:"""

    PROMPT = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template
    )

    # Modify model_name if you have access to GPT-4
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.5, max_tokens=500)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    chain = ConversationalRetrievalChain(
        retriever=vectordb.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )

    ans = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, ans["answer"]))
    print(chat_history)
    answer = ans["answer"].replace('\n', '<br>')
    return answer


print(f"To access the Global link please click {public_url}")

app.run(port=5000)
