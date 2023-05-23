import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scrape import symbolCrypto
import numpy as np
import io
from cryptocmd import CmcScraper
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Prediksi Performa Cryptocurrencies")

st.markdown(
    "Disini terdapat 5 Cryptocurrency berdasarkan kapitalisasi pasar tertinggi")

# st.session_state


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
def convert_df_to_excel(dataframe):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False)


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
    st.session_state['chart_next_predict'] = 'Open'


@st.cache_resource
def load_trained_model(path_to_model):
    model = load_model(
        path_to_model)
    return model


dropdown = st.selectbox(
    "Pilih Salah Satu Cryptocurrency", symbolCrypto, key='input_crypto')

if dropdown:
    start_predict = st.date_input(
        "Tanggal Awal Prediksi", value=pd.to_datetime("2022-12-31"), min_value=pd.to_datetime("2022-12-31"), key='input_start')

    end_predict = st.date_input("Tanggal Akhir Prediksi",
                                value=pd.to_datetime("today"), key='input_end')


periode = (start_predict - end_predict).days - 1

if len(dropdown) > 0:
    st.subheader(f'Berikut data historis {dropdown} 2019-2022')
    data_historis = pd.read_excel(
        f"./data_historis/{dropdown}_data_historis.xlsx", index_col=0, parse_dates=True)
    df = pd.DataFrame(data_historis)

    with open(f"./data_historis/{dropdown}_data_historis.xlsx", "rb") as file_historis:
        st.download_button(
            label="Download Data Historis",
            data=file_historis,
            file_name=f'{dropdown}_data_historis.xlsx',
            mime='application/vnd.ms-excel',
        )

    cols1 = df.columns.tolist()

    cols1.remove("Date")
    cols1.remove("Volume")
    cols1.remove("Market Cap")

    st.dataframe(df, use_container_width=True)
    pilihan1 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols1, key='chart_crypto_1', default=['Open'])
    # st.write(st.session_state.chart_crypto_1)
    if pilihan1:
        data1 = df[pilihan1 + ["Date"]]
        st.line_chart(data1, x="Date", y=pilihan1)

        cols1_5 = df.columns.tolist()

        cols1_5.remove("Date")
        cols1_5.remove("Open")
        cols1_5.remove("High")
        cols1_5.remove("Low")
        cols1_5.remove("Close")

    else:
        st.warning('Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!')

    pilihan1_5 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols1_5, default=["Volume"], key='chart_crypto_2')
    # st.write(st.session_state.chart_crypto_2)
    if pilihan1_5:
        data1_5 = df[pilihan1_5 + ["Date"]]
        st.line_chart(data1_5, x="Date", y=pilihan1_5)
    else:
        st.warning('Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!')

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
        f'./model/{dropdown}_model')

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
    if pilihan2:
        data = new_data[pilihan2]
        st.line_chart(data, y=pilihan2)
    else:
        st.warning('Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!')
    new_rows = pd.DataFrame(index=pd.date_range(
        start=start_predict,  end=end_predict, freq='D', inclusive='right'), columns=new_data.columns[:4])
    new_pred_data = pd.concat([new_data.drop(columns=['open_predicted', 'high_predicted',
                                                      'low_predicted', 'close_predicted'], axis=0), new_rows], axis=0)
    upcoming_prediction = pd.DataFrame(
        columns=['Open', 'High', 'Low', 'Close'], index=new_pred_data.index)
    upcoming_prediction.index = pd.to_datetime(
        upcoming_prediction.index)

    current_seq = test_seq[-1:]

    for i in range(periode, 0):
        up_pred = new_model.predict(current_seq)
        upcoming_prediction.iloc[i] = up_pred
        current_seq = np.append(current_seq[0][1:], up_pred, axis=0)
        current_seq = current_seq.reshape(test_seq[-1:].shape)

    upcoming_prediction[['Open', 'High', 'Low', 'Close']] = MMS.inverse_transform(
        upcoming_prediction[['Open', 'High', 'Low', 'Close']])

    cols3 = upcoming_prediction.columns.tolist()

    @st.cache_data
    def get_latest_price():
        latest_date_start = start_predict.strftime("%d-%m-%Y")

        latest_date_end = end_predict.strftime("%d-%m-%Y")

        latest_scraper = CmcScraper(
            dropdown, latest_date_start, latest_date_end)
        latest_price = latest_scraper.get_dataframe()

        latest_price['Open'] = latest_price['Open'].apply(
            lambda x: round(x, 2))
        latest_price['High'] = latest_price['High'].apply(
            lambda x: round(x, 2))
        latest_price['Low'] = latest_price['Low'].apply(lambda x: round(x, 2))
        latest_price['Close'] = latest_price['Close'].apply(
            lambda x: round(x, 2))

        latest_price = latest_price[::-1]
        latest_price = latest_price.reset_index()

        latest_price = latest_price[['Date', 'Open', 'High', 'Low', 'Close']]
        latest_price['Date'] = pd.to_datetime(latest_price['Date'])
        latest_price.set_index('Date', drop=True, inplace=True)

        return latest_price

    latest_price = get_latest_price()

    diff_col_name_latest_price = latest_price.rename(
        columns={'Open': 'Open_Latest', 'High': 'High_Latest', 'Low': 'Low_Latest', 'Close': 'Close_Latest'})

    pilihan3 = st.selectbox(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols3, key='chart_next_predict')
    pilihan3_str = str(pilihan3)
    pilihan3_latest = (f'{pilihan3}_Latest')
    data_prediction = upcoming_prediction[pilihan3]
    data_prediction = data_prediction[start_predict:]
    # print(type(data_prediction))
    data_combined = pd.concat(
        [data_prediction, diff_col_name_latest_price[pilihan3_latest]], axis=1)

    st.subheader(f"Berikut data {dropdown} {pilihan3_str} yang akan datang")
    st.dataframe(
        data_combined, use_container_width=True)

    download_btn_all, download_btn_pred, download_btn_latest = st.columns(3)
    excel_pred_data = convert_df_to_excel(upcoming_prediction)
    with download_btn_pred:
        st.download_button(
            label="Download All Data",
            data=excel_pred_data,
            file_name=f'{dropdown}_prediction.xlsx',
            mime='application/vnd.ms-excel',
        )

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(new_pred_data.loc['2022-01-01':,
            pilihan3], label=f'Harga {pilihan3_str} Terkini')
    ax.plot(upcoming_prediction.loc['2023-01-01':,
            pilihan3], label=f'Harga {pilihan3_str} yang akan datang')
    ax.plot(latest_price.loc['2023-01-01':, pilihan3],
            label=f'Harga {pilihan3_str} terbaru')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_xlabel('Tanggal', size=15)
    ax.set_ylabel(f'{dropdown} Price', size=15)
    ax.set_title(
        f'Peramalan harga {dropdown} {pilihan3_str} yang akan datang', size=15)
    ax.legend()

    st.pyplot(fig)
