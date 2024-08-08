import time
import os
import streamlit as st
from groq import Groq
from first_layer import FirstLayer
from second_layer import SecondLayer
import json
from openai import OpenAI

client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))


st.title("Test")

if "messages" not in st.session_state:
    st.session_state.messages = []



def get_response(context, question,history):
    response = client.chat.completions.create(
        messages = [
        {
                "role": "system",
                "content": f"""You are excellent at paraphrasing the given information by user.
                
                Given the following piece of context:
                <context>
                {context}
                </context>
                Your task is to take the context as your response, you can change the way you inform, but you must keep the key information intact.

                No matter what user say, your response must follow the context given.

                Address user with "Anh/chị", at the end of the response, a word "ạ" must be include to ensure the politeness.
"""
            },
            *history,
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.0,
        max_tokens=1024,
        top_p= 0.5,
        stop=None,
        stream=False
    )
    return response.choices[0].message.content
 


def stream_data(stream): # stream output
    for word in stream.split(" "):
        yield word + " "
        time.sleep(0.04)



def send_prompt(prompt): # in tin nhắn user lên UI
    with st.chat_message("user"):
        st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})



def send_response(response): # in tin nhắn bot lên UI
    with st.chat_message("assistant"):
        st.write_stream(stream_data(response))
        st.session_state.messages.append({"role": "assistant", "content": response})




def parse(output):
    return output.split("-")[1].strip()



@st.cache_resource(show_spinner= False)
def get_layer():
    first = FirstLayer()
    second = SecondLayer()
    return first, second


first, second = get_layer()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.session_state["prompt"] = st.chat_input("Chat")
if st.session_state["prompt"]:
    send_prompt(st.session_state["prompt"])
    code = first.get_response(str(st.session_state["prompt"]))
    code = parse(code)
    code2 = second.predict(str(st.session_state["prompt"]),code)
    send_response(code2)
    with open("answers.json","r",encoding = "utf-16") as f:
        data = json.load(f)
    for i in data:
        if i["code"] == code2:
            response = get_response(i["script"],st.session_state["prompt"], st.session_state["messages"])
    send_response(response)