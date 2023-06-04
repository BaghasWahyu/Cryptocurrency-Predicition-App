import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scrape import symbolCrypto
import numpy as np
import io
from cryptocmd import CmcScraper
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Tugas Akhir", layout="wide")

st.title("Prediksi Performa Cryptocurrencies")

st.markdown(
    "5 Cryptocurrency berdasarkan kapitalisasi pasar tertinggi")

MMS = MinMaxScaler(feature_range=(0, 1))

if 'input_crypto' not in st.session_state:
    st.session_state['input_crypto'] = 'BTC'

if 'input_start' not in st.session_state:
    st.session_state['input_start'] = pd.to_datetime("2023-01-01")

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
        "Tanggal Awal Prediksi", value=pd.to_datetime("2023-01-01"), min_value=pd.to_datetime("2023-01-01"), key='input_start')

    end_predict = st.date_input("Tanggal Akhir Prediksi",
                                value=pd.to_datetime("today"), key='input_end')

period = (start_predict - end_predict).days - 1

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
    option1 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols1, key='chart_crypto_1', default=['Open'])
    if option1:
        data1 = df[option1 + ["Date"]]
        st.line_chart(data1, x="Date", y=option1)

        cols1_5 = df.columns.tolist()

        cols1_5.remove("Date")
        cols1_5.remove("Open")
        cols1_5.remove("High")
        cols1_5.remove("Low")
        cols1_5.remove("Close")

    else:
        st.warning('Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!')

    option1_5 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols1_5, default=["Volume"], key='chart_crypto_2')
    if option1_5:
        data1_5 = df[option1_5 + ["Date"]]
        st.line_chart(data1_5, x="Date", y=option1_5)
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

    total_train_data, total_test_data = st.columns(2)
    with total_train_data:
        st.markdown(f"Jumlah Data Latih adalah: {len(train_data)}")
        st.write(f"Berikut Data Latih yang sudah dilakukan proses Min Max Scaling",
                 pd.DataFrame(train_data))

    with total_test_data:
        st.markdown(f"Jumlah Data Uji adalah: {len(test_data)}")
        st.write(
            f"Berikut Data Uji yang sudah dilakukan proses Min Max Scaling", pd.DataFrame(test_data))

    def create_sequence(dataset):
        sequences = []
        labels = []

        start_idx = 0

        for stop_idx in range(10, len(dataset)):
            sequences.append(dataset.iloc[start_idx:stop_idx])
            labels.append(dataset.iloc[stop_idx])
            start_idx += 1
        return (np.array(sequences), np.array(labels))

    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)
    train_col_1, train_col_2, train_col_3 = st.columns(3)
    with train_col_1:
        st.write("Train Sequence", train_seq.shape)
        st.write(train_seq[0])
    with train_col_2:
        st.write(train_seq[1])
    with train_col_3:
        st.write("Train label", train_label.shape)
        st.write(train_label)
    test_col_1, test_col_2, test_col_3 = st.columns(3)
    with test_col_1:
        st.write("Test Sequence", test_seq.shape)
        st.write(test_seq[0])
    with test_col_2:
        st.write(test_seq[1])
    with test_col_3:
        st.write("Test Label", test_label.shape)
        st.write(test_label)

    loaded_model = load_trained_model(
        f'./model/{dropdown}_model')

    test_predicted = loaded_model.predict(test_seq)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    test_inverse = MMS.inverse_transform(test_label)

    RMSE = np.sqrt(
        np.mean(((test_inverse_predicted - test_inverse)**2)))

    st.markdown(f"Skor RMSE yang dihasilkan adalah {RMSE}")

    test_inverse_predicted_shape_negative = -test_inverse_predicted.shape[0]

    new_data = pd.concat([crypto_data.iloc[test_inverse_predicted_shape_negative:].copy(), pd.DataFrame(test_inverse_predicted, columns=[
        'open_predicted', 'high_predicted', 'low_predicted', 'close_predicted'], index=crypto_data.iloc[test_inverse_predicted_shape_negative:].index)], axis=1)
    new_data[['Open', 'High', 'Low', 'Close']] = MMS.inverse_transform(
        new_data[['Open', 'High', 'Low', 'Close']])

    cols2 = new_data.columns.tolist()
    st.subheader(f'Berikut data {dropdown} Terkini dan yang Teprediksi')
    st.dataframe(new_data, use_container_width=True)
    option2 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols2, default=["Open", "open_predicted"], key="chart_predict")
    if option2:
        data = new_data[option2]
        st.line_chart(data, y=option2)
    else:
        st.warning('Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!')
    new_rows = pd.DataFrame(index=pd.date_range(
        start=new_data.index[-1],  end=end_predict, freq='D', inclusive='right'), columns=new_data.columns[:4])
    new_pred_data = pd.concat([new_data.drop(columns=['open_predicted', 'high_predicted',
                                                      'low_predicted', 'close_predicted'], axis=0), new_rows], axis=0)
    upcoming_prediction = pd.DataFrame(
        columns=['Open', 'High', 'Low', 'Close'], index=new_pred_data.index)
    upcoming_prediction.index = pd.to_datetime(
        upcoming_prediction.index)

    current_seq = test_seq[-1:]

    for i in range(period, 0):
        up_pred = loaded_model.predict(current_seq)
        upcoming_prediction.iloc[i] = up_pred
        current_seq = np.append(current_seq[0][1:], up_pred, axis=0)
        current_seq = current_seq.reshape(test_seq[-1:].shape)

    upcoming_prediction[['Open', 'High', 'Low', 'Close']] = MMS.inverse_transform(
        upcoming_prediction[['Open', 'High', 'Low', 'Close']])

    cols3 = upcoming_prediction.columns.tolist()

    @st.cache_data
    def get_latest_price(cryptocurrency, start_predict, end_predict):
        latest_date_start = start_predict.strftime("%d-%m-%Y")

        latest_date_end = end_predict.strftime("%d-%m-%Y")

        latest_scraper = CmcScraper(
            cryptocurrency, latest_date_start, latest_date_end)
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

    latest_price = get_latest_price(dropdown, start_predict, end_predict)

    diff_col_name_latest_price = latest_price.rename(
        columns={'Open': 'Open_Latest', 'High': 'High_Latest', 'Low': 'Low_Latest', 'Close': 'Close_Latest'})
    diff_col_name_upcoming_prediction = upcoming_prediction.rename(
        columns={'Open': 'Open_Future', 'High': 'High_Future', 'Low': 'Low_Future', 'Close': 'Close_Future'})

    option3 = st.selectbox(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart", cols3, key='chart_next_predict')
    option3_str = str(option3)
    option3_latest = (f'{option3}_Latest')
    option3_future = (f'{option3}_Future')
    data_prediction = upcoming_prediction[option3]
    data_prediction = data_prediction[start_predict:]
    data_combined = pd.concat(
        [data_prediction, diff_col_name_latest_price[option3_latest]], axis=1)
    all_data_combined = pd.concat(
        [diff_col_name_upcoming_prediction[start_predict:], diff_col_name_latest_price], axis=1)

    st.subheader(f"Berikut harga {dropdown} {option3_str} yang akan datang")
    st.dataframe(
        data_combined, use_container_width=True)

    download_btn_all, download_btn_pred, download_btn_latest = st.columns(3)

    all_data_excel = io.BytesIO()
    with pd.ExcelWriter(all_data_excel, engine='xlsxwriter') as writer:
        all_data_combined.to_excel(writer)
    with download_btn_all:
        st.download_button(
            label=f"Download {dropdown} All Data",
            data=all_data_excel,
            file_name=f'{dropdown}_all.xlsx',
            mime='application/vnd.ms-excel',
        )

    pred_data_excel = io.BytesIO()
    with pd.ExcelWriter(pred_data_excel, engine='xlsxwriter') as writer:
        upcoming_prediction[start_predict:].to_excel(writer)
    with download_btn_pred:
        st.download_button(
            label=f"Download {dropdown} Prediction Data",
            data=pred_data_excel,
            file_name=f'{dropdown}_prediction.xlsx',
            mime='application/vnd.ms-excel',
        )

    diff_col_name_latest_price_excel = io.BytesIO()
    with pd.ExcelWriter(diff_col_name_latest_price_excel, engine='xlsxwriter') as writer:
        diff_col_name_latest_price.to_excel(writer)
    with download_btn_latest:
        st.download_button(
            label=f"Download {dropdown} Latest Data",
            data=diff_col_name_latest_price_excel,
            file_name=f'{dropdown}_latest.xlsx',
            mime='application/vnd.ms-excel',
        )

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(new_pred_data.loc['2022-01-01':,
            option3], label=f'Harga {option3_str} Terkini')
    ax.plot(upcoming_prediction.loc['2023-01-01':,
            option3], label=f'Harga {option3_str} yang akan datang')
    ax.plot(latest_price.loc['2023-01-01':, option3],
            label=f'Harga {option3_str} terbaru')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_xlabel('Tanggal', size=15)
    ax.set_ylabel(f'{dropdown} Price', size=15)
    ax.set_title(
        f'Peramalan harga {dropdown} {option3_str} yang akan datang', size=15)
    ax.legend()

    st.pyplot(fig)
