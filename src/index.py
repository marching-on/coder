

import streamlit as st
from codegen_intel import Coder

# 使用 st.session_state 存储 g_coder
if 'g_coder' not in st.session_state:
    st.session_state['g_coder'] = Coder()
g_coder = st.session_state['g_coder']

if "history_data" not in st.session_state:
    st.session_state["history_data"] = []
history_data = st.session_state["history_data"]

#if g_coder == None:
#    g_coder = Coder()
st.title("coder 🖥️")

if "messages_human" not in st.session_state:
    st.session_state["messages_human"] = []
if "messages_assistant" not in st.session_state:
    st.session_state["messages_assistant"] = []

messages_human = st.session_state["messages_human"]
messages_assistant = st.session_state["messages_assistant"]
max_length = max(len(messages_human), len(messages_assistant))

for i in range(max_length):
    if i < len(messages_human) and messages_human[i] is not None:
        with st.chat_message("user", avatar='🧐'):
            st.markdown(messages_human[i])
    if i < len(messages_assistant) and messages_assistant[i] is not None:
        with st.chat_message("assistant", avatar='🖥️'):
            st.code(messages_assistant[i], language="python")

question = st.chat_input("Shift+Enter to breaklines, Enter to send")

if question:
    st.session_state["messages_human"].append(question)
    with st.chat_message("user", avatar='🧐'):
        st.markdown(question)

    answer, st.session_state["history_data"] = g_coder.infer(question, history_data)

    st.session_state["messages_assistant"].append(answer)
    with st.chat_message("assistant", avatar='🖥️'):
        #st.markdown(answer)
        st.code(answer, language='python')


container = st.container()
container.write("")

if container.button("清空对话"):
    st.session_state["messages_human"] = []
    st.session_state["messages_assistant"] = []
    st.session_state["history_data"] = []
