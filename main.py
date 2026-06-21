
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="Analisis Sentimen Danantara", layout="wide")

st.title("📊 Dashboard Analisis Sentimen Danantara")
st.markdown("Upload file hasil proses model untuk menampilkan visualisasi.")

uploaded = st.file_uploader("Upload hasil_prediksi_kfold_valid.csv", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    if "label_logistic_regression" in df.columns:
        sentimen = df["label_logistic_regression"].value_counts().reset_index()
        sentimen.columns = ["Sentimen", "Jumlah"]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                sentimen,
                values="Jumlah",
                names="Sentimen",
                title="Distribusi Sentimen"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(
                sentimen,
                x="Sentimen",
                y="Jumlah",
                title="Jumlah Data per Sentimen",
                text="Jumlah"
            )
            st.plotly_chart(fig2, use_container_width=True)

    if "topic" in df.columns and "label_logistic_regression" in df.columns:
        st.subheader("Distribusi Sentimen per Topik")

        topic_df = (
            df.groupby(["topic", "label_logistic_regression"])
            .size()
            .reset_index(name="Jumlah")
        )

        fig3 = px.bar(
            topic_df,
            x="topic",
            y="Jumlah",
            color="label_logistic_regression",
            barmode="group",
            title="Sentimen per Topik"
        )
        st.plotly_chart(fig3, use_container_width=True)

    if "prediksi_sesuai" in df.columns:
        st.subheader("Kesesuaian Prediksi")

        cocok = df["prediksi_sesuai"].value_counts().reset_index()
        cocok.columns = ["Status", "Jumlah"]

        fig4 = px.pie(
            cocok,
            values="Jumlah",
            names="Status",
            title="Kesesuaian Prediksi Logistic Regression vs IndoBERT"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Statistik Ringkas")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Data", len(df))

    if "label_logistic_regression" in df.columns:
        mayoritas = df["label_logistic_regression"].value_counts().idxmax()
        c2.metric("Sentimen Dominan", mayoritas)

    if "prediksi_sesuai" in df.columns:
        akurasi = (
            (df["prediksi_sesuai"] == "Sesuai").sum()
            / len(df)
        ) * 100
        c3.metric("Kesesuaian", f"{akurasi:.2f}%")

    st.subheader("Download Data")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_kfold_valid.csv",
        mime="text/csv"
    )

else:
    st.info("Upload file CSV hasil prediksi terlebih dahulu.")
