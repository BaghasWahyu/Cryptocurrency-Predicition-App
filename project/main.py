import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scrape import symbol
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Prediksi Performa Cryptocurrencies")

st.markdown(
    "Disini terdapat 5 Cryptocurrency berdasarkan kapitalisasi pasar tertinggi")

st.session_state


def create_sequence(dataset):
    sequences = []
    labels = []

    start_idx = 0

    for stop_idx in range(1, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequences), np.array(labels))


MMS = MinMaxScaler(feature_range=(0, 1))


@st.cache_data
def convert_df(df):
    return df.to_excel()


if 'input_crypto' not in st.session_state:
    st.session_state['input_crypto'] = 'BTC'

if 'input_start' not in st.session_state:
    st.session_state['input_start'] = pd.to_datetime("2022-12-31")

if 'input_end' not in st.session_state:
    st.session_state['input_end'] = pd.to_datetime("today")

if 'chart_crypto_1' not in st.session_state:
    st.session_state['chart_crypto_1'] = ['Open']

if 'chart_crypto_2' not in st.session_state:
    st.session_state['chart_crypto_2'] = ['Volume']

if 'chart_predict' not in st.session_state:
    st.session_state['chart_predict'] = ['Open', 'open_predicted']

if 'chart_next_predict' not in st.session_state:
    st.session_state['chart_next_predict'] = ['Open']


@st.cache_resource
def load_trained_model(path_to_model):
    model = load_model(
        path_to_model)
    return model


dropdown = st.selectbox(
    "Pilih Salah Satu Cryptocurrency", symbol, key='input_crypto')
# st.write(st.session_state.input_crypto)

start_predict = st.date_input(
    "Tanggal Awal Prediksi", value=pd.to_datetime("2022-12-31"), min_value=pd.to_datetime("2022-12-31"), key='input_start')
# st.write(st.session_state.input_start)

end_predict = st.date_input("Tanggal Akhir Prediksi",
                            value=pd.to_datetime("today"), key='input_end')
# st.write(st.session_state.input_end)

periode = (start_predict - end_predict).days - 1

if len(dropdown) > 0:
    st.subheader(f'Berikut data historis {dropdown} 2019-2022')
    data_historis = pd.read_excel(
        f"/prediksi_crypto/Cryptocurrency-Predicition-App/data_historis/{dropdown}_data_historis.xlsx", index_col=0, parse_dates=True)
    df = pd.DataFrame(data_historis)

    cols1 = df.columns.tolist()

    cols1.remove("Date")
    cols1.remove("Volume")
    cols1.remove("Market Cap")

    st.dataframe(df, use_container_width=True)
    pilihan1 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols1, key='chart_crypto_1', default=['Open'])
    # st.write(st.session_state.chart_crypto_1)
    data1 = df[pilihan1 + ["Date"]]
    st.line_chart(data1, x="Date", y=pilihan1)

    cols1_5 = df.columns.tolist()

    cols1_5.remove("Date")
    cols1_5.remove("Open")
    cols1_5.remove("High")
    cols1_5.remove("Low")
    cols1_5.remove("Close")

    pilihan1_5 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols1_5, default=["Volume"], key='chart_crypto_2')
    # st.write(st.session_state.chart_crypto_2)
    data1_5 = df[pilihan1_5 + ["Date"]]
    st.line_chart(data1_5, x="Date", y=pilihan1_5)

    crypto_data = df[['Date', 'Open', 'High', 'Low', 'Close']]
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date'])
    crypto_data.set_index('Date', drop=True, inplace=True)
    crypto_data[crypto_data.columns] = MMS.fit_transform(
        crypto_data)

    training_size = round(len(crypto_data) * 0.90)
    train_data = crypto_data[:training_size]
    test_data = crypto_data[training_size:]

    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)

    new_model = load_trained_model(
        f'/prediksi_crypto/Cryptocurrency-Predicition-App/model/{dropdown}_model')

    test_predicted = new_model.predict(test_seq)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    test_inverse = MMS.inverse_transform(test_label)
    rmse = np.sqrt(
        np.mean(((test_inverse_predicted - test_inverse)**2)))

    st.markdown(f"Skor RMSE yang dihasilkan adalah {rmse}")

    data_value = -test_inverse_predicted.shape[0]

    new_data = pd.concat([crypto_data.iloc[data_value:].copy(), pd.DataFrame(test_inverse_predicted, columns=[
        'open_predicted', 'high_predicted', 'low_predicted', 'close_predicted'], index=crypto_data.iloc[data_value:].index)], axis=1)
    new_data[['Open', 'High', 'Low', 'Close']] = MMS.inverse_transform(
        new_data[['Open', 'High', 'Low', 'Close']])

    cols2 = new_data.columns.tolist()
    st.subheader(f'Berikut data {dropdown} Terkini dan yang Teprediksi')
    st.dataframe(new_data, use_container_width=True)
    pilihan2 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols2, default=["Open", "open_predicted"], key="chart_predict")
    data = new_data[pilihan2]
    st.line_chart(data, y=pilihan2)

    if st.button("Mulai Prediksi"):

        new_rows = pd.DataFrame(index=pd.date_range(
            start=start_predict,  end=end_predict, freq='D', inclusive='right'), columns=new_data.columns[:4])
        new_pred_data = pd.concat([new_data.drop(columns=['open_predicted', 'high_predicted',
                                                          'low_predicted', 'close_predicted'], axis=0), new_rows], axis=0)
        upcoming_prediction = pd.DataFrame(
            columns=['Open', 'High', 'Low', 'Close'], index=new_pred_data.index)
        upcoming_prediction.index = pd.to_datetime(
            upcoming_prediction.index)

        curr_seq = test_seq[-1:]

        for i in range(periode, 0):
            up_pred = new_model.predict(curr_seq)
            upcoming_prediction.iloc[i] = up_pred
            curr_seq = np.append(curr_seq[0][1:], up_pred, axis=0)
            curr_seq = curr_seq.reshape(test_seq[-1:].shape)

        upcoming_prediction[['Open', 'High', 'Low', 'Close']] = MMS.inverse_transform(
            upcoming_prediction[['Open', 'High', 'Low', 'Close']])

        cols3 = upcoming_prediction.columns.tolist()

        pilihan3 = st.multiselect(
            "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols3, default=["Open"], key='chart_next_predict')
        data_prediction = upcoming_prediction[pilihan3]
        data_prediction = data_prediction[start_predict:]
        st.subheader(f"Berikut data {dropdown} {pilihan3} yang akan datang")
        st.dataframe(
            data_prediction, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(new_pred_data.loc['2022-01-01':,
                pilihan3], label=f'Harga {pilihan3} Terkini')
        ax.plot(upcoming_prediction.loc['2023-01-01':,
                pilihan3], label=f'Harga {pilihan3} yang akan datang')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.set_xlabel('Tanggal', size=15)
        ax.set_ylabel(f'{dropdown} Price', size=15)
        ax.set_title(
            f'Peramalan harga {dropdown} {pilihan3} yang akan datang', size=15)
        ax.legend()
        st.pyplot(fig)
