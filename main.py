import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Analisis Sentimen Danantara",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Sentimen Danantara")
st.markdown(
    "Dashboard visualisasi hasil analisis sentimen Danantara menggunakan "
    "IndoBERT, BERTopic, dan Logistic Regression."
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Upload Dataset")

pred_file = st.sidebar.file_uploader(
    "hasil_prediksi_kfold_valid.csv",
    type=["csv"]
)

topic_file = st.sidebar.file_uploader(
    "hasil_topic_modeling.csv",
    type=["csv"]
)

metric_file = st.sidebar.file_uploader(
    "ringkasan_hasil_kfold.csv",
    type=["csv"]
)

if pred_file and topic_file and metric_file:

    df = pd.read_csv(pred_file)
    topic_df = pd.read_csv(topic_file)
    metric_df = pd.read_csv(metric_file)

    st.success("Semua file berhasil dimuat.")

    # =========================
    # KPI
    # =========================

    st.header("📈 Ringkasan Performa Model")

    metric_map = dict(
        zip(
            metric_df["Metrik"],
            metric_df["Rata-rata"]
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metric_map.get('Accuracy', 0):.4f}"
    )

    col2.metric(
        "Precision",
        f"{metric_map.get('Precision (macro)', 0):.4f}"
    )

    col3.metric(
        "Recall",
        f"{metric_map.get('Recall (macro)', 0):.4f}"
    )

    col4.metric(
        "F1 Score",
        f"{metric_map.get('F1-Score (macro)', 0):.4f}"
    )

    st.divider()

    # =========================
    # DISTRIBUSI SENTIMEN
    # =========================

    st.header("🥧 Distribusi Sentimen")

    sentimen = (
        df["label_logistic_regression"]
        .value_counts()
        .reset_index()
    )

    sentimen.columns = ["Sentimen", "Jumlah"]

    c1, c2 = st.columns(2)

    with c1:
        fig_pie = px.pie(
            sentimen,
            names="Sentimen",
            values="Jumlah",
            title="Distribusi Sentimen"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with c2:
        fig_bar = px.bar(
            sentimen,
            x="Sentimen",
            y="Jumlah",
            color="Sentimen",
            text="Jumlah",
            title="Jumlah Data per Sentimen"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    st.divider()

    # =========================
    # Kesesuaian Prediksi
    # =========================

    st.header("🎯 Kesesuaian Prediksi")

    cocok = (
        df["prediksi_sesuai"]
        .value_counts()
        .reset_index()
    )

    cocok.columns = ["Status", "Jumlah"]

    fig = px.pie(
        cocok,
        names="Status",
        values="Jumlah",
        title="Kesesuaian Prediksi Logistic Regression terhadap Label IndoBERT"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =========================
    # WORDCLOUD
    # =========================

    st.header("☁️ WordCloud Sentimen")

    tabs = st.tabs([
        "Positif",
        "Netral",
        "Negatif"
    ])

    labels = [
        "Positif",
        "Netral",
        "Negatif"
    ]

    for i, label in enumerate(labels):

        with tabs[i]:

            temp = df[
                df["label_logistic_regression"] == label
            ]

            text = " ".join(
                temp["dokumen_preprocessing"]
                .astype(str)
            )

            if len(text.strip()) > 0:

                wc = WordCloud(
                    width=1200,
                    height=600,
                    background_color="white"
                ).generate(text)

                fig, ax = plt.subplots(
                    figsize=(12, 6)
                )

                ax.imshow(wc)
                ax.axis("off")

                st.pyplot(fig)

            else:
                st.warning(
                    f"Tidak ada data {label}"
                )

    st.divider()

    # =========================
    # TOPIC MODELING
    # =========================

    st.header("🧩 Topic Modeling BERTopic")

    fig_topic = px.bar(
        topic_df,
        x="Topic",
        y="Count",
        title="Distribusi Topic"
    )

    st.plotly_chart(
        fig_topic,
        use_container_width=True
    )

    st.subheader("Informasi Topic")

    st.dataframe(
        topic_df,
        use_container_width=True
    )

    st.divider()

    # =========================
    # SENTIMEN PER TOPIC
    # =========================

    st.header("📊 Sentimen per Topic")

    topic_sentiment = (
        df.groupby(
            [
                "topic",
                "label_logistic_regression"
            ]
        )
        .size()
        .reset_index(name="Jumlah")
    )

    fig_topic_sent = px.bar(
        topic_sentiment,
        x="topic",
        y="Jumlah",
        color="label_logistic_regression",
        barmode="group",
        title="Distribusi Sentimen pada Setiap Topic"
    )

    st.plotly_chart(
        fig_topic_sent,
        use_container_width=True
    )

    st.divider()

    # =========================
    # DATA EXPLORER
    # =========================

    st.header("📋 Data Explorer")

    pilihan_sentimen = st.multiselect(
        "Filter Sentimen",
        options=df[
            "label_logistic_regression"
        ].unique(),
        default=df[
            "label_logistic_regression"
        ].unique()
    )

    filtered_df = df[
        df[
            "label_logistic_regression"
        ].isin(
            pilihan_sentimen
        )
    ]

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Data Hasil Filter",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="hasil_filter.csv",
        mime="text/csv"
    )

else:

    st.info(
        """
        Upload ketiga file berikut:

        1. hasil_prediksi_kfold_valid.csv
        2. hasil_topic_modeling.csv
        3. ringkasan_hasil_kfold.csv
        """
    )
