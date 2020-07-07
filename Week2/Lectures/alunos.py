import streamlit as st
import pandas as pd

def main():
    st.title("Hello world")
    st.header("This is a header")
    st.subheader("This is a subheader")
    st.text("This is a comment")
    # st.image('logo.png')
    # st.audio
    # st.video("sentiment_motion.mov")
    st.markdown("Botao")
    botao = st.button("Botao")
    if botao:
        st.markdown("Clicado")
    
    check = st.checkbox("Checkbox")
    if check:
        st.markdown("Check")

    radio = st.radio("Choose wisely", ["Opt 1", 'Opt 2'])

    if radio == "Opt 1":
        st.markdown("Opt 1")
    if radio == "Opt 2":
        st.markdown("Opt 2")

    select = st.selectbox("Choose opt", ["Opt 1", 'Opt 2'])

    
    if select == "Opt 1":
        st.markdown("Opt 1")
    if select == "Opt 2":
        st.markdown("Opt 2")

    multi = st.multiselect("Choose opt", ["Opt 1", 'Opt 2'])

    
    if "Opt 1" in multi:
        st.markdown("Opt 1")
    if "Opt 2" in multi:
        st.markdown("Opt 2")

    file = st.file_uploader("Send your file", type="csv")
    if file is not None:
        sl = st.slider("Valores", 0, 100)
        st.markdown("Not empty")
        df2 = pd.read_csv(file)
        st.dataframe(df2.head(sl))
        st.table(df2.head(sl))
        st.write(df2.columns)
        st.dataframe(df2.groupby("species").mean())

    #df = pd.read_csv("IRIS.csv")
    #st.dataframe(df.head(5))



if __name__ == "__main__":
    main()