import openai
import gradio as gr

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a professor who is teching advanced machine learning algorithms"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply
#css_code='body{background-image:url("https://picsum.photos/seed/picsum/200/300");}'
#demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Online teacher",css=css_code)

#demo.launch(share=True)
demo = gr.Interface(fn=CustomChatGPT, inputs=gr.inputs.Textbox(label="Student's Question"), \
    outputs=[gr.outputs.Textbox(label='Annser from Teacher'),\
      gr.Plot()],\
        css='div {margin-left: auto; margin-right: auto; width: 100%;\
            background-image: url("https://drive.google.com/uc?export=view&id=12345678900"); repeat 0 0;}')\
              .launch(share=True)