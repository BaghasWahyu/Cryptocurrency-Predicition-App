import io as io

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd
import streamlit as st
from cryptocmd import CmcScraper
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta, datetime

from scrape import symbolCrypto, list_crypto, namaCrypto

st.title("Aplikasi Prediksi Harga Cryptocurrencies")

MMS = MinMaxScaler(feature_range=(0, 1))

if "input_crypto" not in st.session_state:
    st.session_state["input_crypto"] = "Bitcoin"

if "input_epoch" not in st.session_state:
    st.session_state["input_epoch"] = 25

if "input_neuron" not in st.session_state:
    st.session_state["input_neuron"] = 50

if "input_batch_size" not in st.session_state:
    st.session_state["input_batch_size"] = 16

if "chart_crypto_1" not in st.session_state:
    st.session_state["chart_crypto_1"] = ["Open"]

if "chart_crypto_2" not in st.session_state:
    st.session_state["chart_crypto_2"] = ["Volume"]

if "chart_predict" not in st.session_state:
    st.session_state["chart_predict"] = ["Open", "open_predicted"]

if "chart_next_predict" not in st.session_state:
    st.session_state["chart_next_predict"] = "Open"


@st.cache_resource
def load_trained_model(path_to_model):
    model = load_model(path_to_model)
    return model


@st.cache_data
def create_sequence(dataset):
    sequences = []
    labels = []

    start_idx = 0
    for stop_idx in range(5, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1

    return np.array(sequences), np.array(labels)


@st.cache_data
def plot_actual_vs_predicted(dataframe, opsi, crypto_name):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xticklabels(dataframe.index, rotation=45)
    for item in opsi:
        dataframe[[item]].plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Crypto Price")
    ax.set_title(f"Actual vs Predicted {crypto_name} price")
    ax.legend()
    return fig


@st.cache_data
def get_latest_price(cryptocurrency, start_predict, end_predict):
    latest_date_start = start_predict.strftime("%d-%m-%Y")

    latest_date_end = end_predict.strftime("%d-%m-%Y")

    latest_scraper = CmcScraper(cryptocurrency, latest_date_start, latest_date_end)
    latest_price = latest_scraper.get_dataframe()

    latest_price["Open"] = latest_price["Open"].apply(lambda x: round(x, 2))
    latest_price["High"] = latest_price["High"].apply(lambda x: round(x, 2))
    latest_price["Low"] = latest_price["Low"].apply(lambda x: round(x, 2))
    latest_price["Close"] = latest_price["Close"].apply(lambda x: round(x, 2))

    latest_price = latest_price[::-1]
    latest_price = latest_price.reset_index()
    latest_price = latest_price.drop(columns=["index"])

    return latest_price


with st.sidebar:
    dropdown_index = st.selectbox(
        "Pilih Salah Satu Cryptocurrency", namaCrypto, key="input_crypto"
    )

    epochs_list = [25, 50]
    neurons_list = [50, 100]
    batch_sizes = [16, 32]

    epoch_option = st.selectbox(
        "Pilih Salah Satu Epoch", epochs_list, key="input_epoch"
    )
    neurons_option = st.selectbox(
        "Pilih Salah Satu Neurons", neurons_list, key="input_neuron"
    )
    batch_size_option = st.selectbox(
        "Pilih Salah Satu Batch Size", batch_sizes, key="input_batch_size"
    )
    dropdown = symbolCrypto[namaCrypto.index(dropdown_index)]

st.subheader(
    "Berikut 5 Cryoptocurrency tertinggi berdasarkan Market Capitalization per 31 Desember 2023 menurut situs web coinmarketcap.com"
)
st.dataframe(list_crypto, use_container_width=True)

if len(dropdown) > 0:
    st.subheader(f"Berikut data historis {dropdown_index}")

    startdate = "2019-01-01"
    startdate = datetime.strptime(startdate, "%Y-%m-%d").date()

    enddate = datetime.now()
    enddate = enddate.date()

    df_latest_price = get_latest_price(dropdown, startdate, enddate)
    data_historis = pd.read_excel(
        f"./data_historis/{dropdown}_data_historis.xlsx", index_col=0, parse_dates=True
    )
    if not df_latest_price.equals(data_historis):
        df_latest_price.to_excel(f"./data_historis/{dropdown}_data_historis.xlsx")

    df = df_latest_price
    df = df.drop(columns=["Time Open", "Time High", "Time Low", "Time Close"], axis=1)

    with open(f"./data_historis/{dropdown}_data_historis.xlsx", "rb") as file_historis:
        st.download_button(
            label="Download Data Historis",
            data=file_historis,
            file_name=f"{dropdown}_data_historis.xlsx",
            mime="application/vnd.ms-excel",
        )

    cols1 = df.columns.tolist()
    cols1.remove("Date")
    cols1.remove("Volume")
    cols1.remove("Market Cap")

    st.dataframe(df, use_container_width=True)
    option1 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart",
        cols1,
        key="chart_crypto_1",
        default=["Open"],
    )
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
        st.warning("Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!")

    option1_5 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart",
        cols1_5,
        default=["Volume"],
        key="chart_crypto_2",
    )
    if option1_5:
        data1_5 = df[option1_5 + ["Date"]]
        st.line_chart(data1_5, x="Date", y=option1_5)
    else:
        st.warning("Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!")

    crypto_data = df[["Date", "Open", "High", "Low", "Close"]]
    crypto_data["Date"] = pd.to_datetime(crypto_data["Date"])
    crypto_data.set_index("Date", drop=True, inplace=True)
    crypto_data_real = crypto_data.copy()
    crypto_data[crypto_data.columns] = MMS.fit_transform(crypto_data)

    st.write("Data Historis setelah dilakukan proses Min-Max Scaling")
    st.dataframe(crypto_data, use_container_width=True)

    # Pembagian Data Latih dan Data Uji
    end_date = pd.to_datetime("2023-12-31")
    start_date_test = end_date + timedelta(days=1)
    train_data = crypto_data[:end_date]
    test_data = crypto_data[start_date_test:]

    total_train_data, total_test_data = st.columns(2)
    with total_train_data:
        st.markdown(f"Jumlah Data Latih adalah: {len(train_data)}")
        st.write(
            f"Berikut Data Latih yang sudah dilakukan proses Min Max Scaling",
            pd.DataFrame(train_data),
        )

    with total_test_data:
        st.markdown(f"Jumlah Data Uji adalah: {len(test_data)}")
        st.write(
            f"Berikut Data Uji yang sudah dilakukan proses Min Max Scaling",
            pd.DataFrame(test_data),
        )

    train_seq, train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)

    with st.expander("Data Latih dan Data Uji"):
        train_column_1, train_column_2, train_column_3 = st.columns(3)
        with train_column_1:
            st.write("Train Sequence", train_seq.shape)
            st.write(train_seq[0])
        with train_column_2:
            st.write(train_seq[1])
        with train_column_3:
            st.write("Train label", train_label.shape)
            st.write(train_label)
        test_column_1, test_column_2, test_column_3 = st.columns(3)
        with test_column_1:
            st.write("Test Sequence", test_seq.shape)
            st.write(test_seq[0])
        with test_column_2:
            st.write(test_seq[1])
        with test_column_3:
            st.write("Test Label", test_label.shape)
            st.write(test_label)

    loaded_model = load_trained_model(
        f"./model/Model_{dropdown}_Epochs_{epoch_option}_Neurons_{neurons_option}_Batch_{batch_size_option}.keras"
    )
    with st.expander("Ringkasan Model"):
        loaded_model.summary(print_fn=st.write)
        for layer in loaded_model.layers:
            if len(layer.get_weights()) > 0:
                layer_weights = layer.get_weights()[0]
                layer_bias = layer.get_weights()[1]
                st.write(layer.name)
                st.write("Weights:")
                st.write(layer_weights)
                st.write("Bias:")
                st.write(layer_bias)
                st.write("--------")

    test_predicted = loaded_model.predict(test_seq)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    test_inverse_label = MMS.inverse_transform(test_label)

    st.write(f"Test sequence shape {test_seq.shape}")
    st.write(f"Test label shape {test_label.shape}")
    st.write(f"Test predicted shape {test_predicted.shape}")

    test_inverse_predicted_shape_negative = -(test_inverse_predicted.shape[0])

    new_data = pd.concat(
        [
            crypto_data.iloc[test_inverse_predicted_shape_negative:].copy(),
            pd.DataFrame(
                test_inverse_predicted,
                columns=[
                    "open_predicted",
                    "high_predicted",
                    "low_predicted",
                    "close_predicted",
                ],
                index=crypto_data.iloc[test_inverse_predicted_shape_negative:].index,
            ),
        ],
        axis=1,
    )

    new_data[["Open", "High", "Low", "Close"]] = MMS.inverse_transform(
        new_data[["Open", "High", "Low", "Close"]]
    )
    # Memeriksa kolom yang memiliki tipe data float32 pada DataFrame new_data
    float32_columns = new_data.select_dtypes(include="float32").columns

    # Mengubah tipe data kolom tersebut menjadi float64
    new_data[float32_columns] = new_data[float32_columns].astype("float64")

    new_data["open_margin_of_error"] = new_data["Open"] - new_data["open_predicted"]
    # Menghitung margin of error dalam bentuk persentase untuk kolom open
    new_data["open_margin_of_error_percent"] = (
        (new_data["Open"] - new_data["open_predicted"]) / new_data["Open"]
    ) * 100

    # Menghitung margin of error untuk kolom high
    new_data["high_margin_of_error"] = new_data["High"] - new_data["high_predicted"]
    # Menghitung margin of error dalam bentuk persentase untuk kolom high
    new_data["high_margin_of_error_percent"] = (
        (new_data["High"] - new_data["high_predicted"]) / new_data["High"]
    ) * 100

    # Menghitung margin of error untuk kolom low
    new_data["low_margin_of_error"] = new_data["Low"] - new_data["low_predicted"]
    # Menghitung margin of error dalam bentuk persentase untuk kolom low
    new_data["low_margin_of_error_percent"] = (
        (new_data["Low"] - new_data["low_predicted"]) / new_data["Low"]
    ) * 100

    # Menghitung margin of error untuk kolom close
    new_data["close_margin_of_error"] = new_data["Close"] - new_data["close_predicted"]
    # Menghitung margin of error dalam bentuk persentase untuk kolom close
    new_data["close_margin_of_error_percent"] = (
        (new_data["Close"] - new_data["close_predicted"]) / new_data["Close"]
    ) * 100

    st.subheader(f"Berikut data {dropdown_index} Terkini dan yang Teprediksi")
    if dropdown == "USDT":
        number = st.number_input(
            "Masukkan angka untuk pembulatan di belakang koma",
            value=4,
            placeholder="Ketikan Angka....",
        )
    else:
        number = st.number_input(
            "Masukkan angka untuk pembulatan di belakang koma",
            value=2,
            placeholder="Ketikan Angka....",
        )
    # Pembulatan nilai pada dataframe
    new_data = new_data.round(number)

    cols2 = new_data.columns.tolist()

    st.text("Harian")
    new_data_daily_io = io.BytesIO()
    with pd.ExcelWriter(new_data_daily_io, engine="xlsxwriter") as writer:
        new_data.to_excel(writer)
    st.download_button(
        label=f"Download Data Harian {dropdown_index}",
        data=new_data_daily_io,
        file_name=f"{dropdown}_Epoch{epoch_option}_Neuron{neurons_option}_BatchSize{batch_size_option}_daily.xlsx",
        mime="application/vnd.ms-excel",
    )
    st.dataframe(new_data, use_container_width=True)

    mean_open_margin_of_error = np.mean(new_data["open_margin_of_error"])
    mean_high_margin_of_error = np.mean(new_data["high_margin_of_error"])
    mean_low_margin_of_error = np.mean(new_data["low_margin_of_error"])
    mean_close_margin_of_error = np.mean(new_data["close_margin_of_error"])

    mean_open_margin_of_error_percent = np.mean(
        new_data["open_margin_of_error_percent"]
    )
    mean_high_margin_of_error_percent = np.mean(
        new_data["high_margin_of_error_percent"]
    )
    mean_low_margin_of_error_percent = np.mean(new_data["low_margin_of_error_percent"])
    mean_close_margin_of_error_percent = np.mean(
        new_data["close_margin_of_error_percent"]
    )

    if dropdown == "USDT":
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} Open Harian : {mean_open_margin_of_error:.5f} atau {mean_open_margin_of_error_percent:.5f}%"
        )
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} High Harian : {mean_high_margin_of_error:.5f} atau {mean_high_margin_of_error_percent:.5f}%"
        )
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} Low Harian : {mean_low_margin_of_error:.5f} atau {mean_low_margin_of_error_percent:.5f}%"
        )
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} Close Harian : {mean_close_margin_of_error:.5f} atau {mean_close_margin_of_error_percent:.5f}%"
        )
        st.write("--------")
        # RMSE Open Daily
        MSE_open = mean_squared_error(new_data["Open"], new_data["open_predicted"])
        RMSE_open = np.sqrt(MSE_open)
        RMSE_open_percentage = (RMSE_open / np.mean(new_data["Open"])) * 100
        st.write(
            f"RMSE Open Harian {dropdown} : {RMSE_open:.5f} atau {RMSE_open_percentage:.5f}%"
        )

        # RMSE High Daily
        MSE_high = mean_squared_error(new_data["High"], new_data["high_predicted"])
        RMSE_high = np.sqrt(MSE_high)
        RMSE_high_percentage = (RMSE_high / np.mean(new_data["High"])) * 100
        st.write(
            f"RMSE High Harian {dropdown} : {RMSE_high:.5f} atau {RMSE_high_percentage:.5f}%"
        )

        # RMSE Low Daily
        MSE_low = mean_squared_error(new_data["Low"], new_data["low_predicted"])
        RMSE_low = np.sqrt(MSE_low)
        RMSE_low_percentage = (RMSE_low / np.mean(new_data["Low"])) * 100
        st.write(
            f"RMSE Low Harian {dropdown} : {RMSE_low:.5f} atau {RMSE_low_percentage:.5f}%"
        )

        # RMSE Close Daily
        MSE_close = mean_squared_error(new_data["Close"], new_data["close_predicted"])
        RMSE_close = np.sqrt(MSE_close)
        RMSE_close_percentage = (RMSE_close / np.mean(new_data["Close"])) * 100
        st.write(
            f"RMSE Close Harian {dropdown} : {RMSE_close:.5f} atau {RMSE_close_percentage:.5f}%"
        )
        st.write("--------")
    else:
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} Open Harian : {mean_open_margin_of_error:.3f} atau {mean_open_margin_of_error_percent:.3f}%"
        )
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} High Harian : {mean_high_margin_of_error:.3f} atau {mean_high_margin_of_error_percent:.3f}%"
        )
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} Low Harian : {mean_low_margin_of_error:.3f} atau {mean_low_margin_of_error_percent:.3f}%"
        )
        st.write(
            f"Rata-rata Margin of Error {dropdown_index} Close Harian : {mean_close_margin_of_error:.3f} atau {mean_close_margin_of_error_percent:.3f}%"
        )

        st.write("--------")

        # RMSE Open Daily
        MSE_open = mean_squared_error(new_data["Open"], new_data["open_predicted"])
        RMSE_open = np.sqrt(MSE_open)
        RMSE_open_percentage = (RMSE_open / np.mean(new_data["Open"])) * 100
        st.write(
            f"RMSE Open Harian {dropdown} : {RMSE_open:.3f} atau {RMSE_open_percentage:.3f}%"
        )

        # RMSE High Daily
        MSE_high = mean_squared_error(new_data["High"], new_data["high_predicted"])
        RMSE_high = np.sqrt(MSE_high)
        RMSE_high_percentage = (RMSE_high / np.mean(new_data["High"])) * 100
        st.write(
            f"RMSE High Harian {dropdown} : {RMSE_high:.3f} atau {RMSE_high_percentage:.3f}%"
        )

        # RMSE Low Daily
        MSE_low = mean_squared_error(new_data["Low"], new_data["low_predicted"])
        RMSE_low = np.sqrt(MSE_low)
        RMSE_low_percentage = (RMSE_low / np.mean(new_data["Low"])) * 100
        st.write(
            f"RMSE Low Harian {dropdown} : {RMSE_low:.3f} atau {RMSE_low_percentage:.3f}%"
        )

        # RMSE Close Daily
        MSE_close = mean_squared_error(new_data["Close"], new_data["close_predicted"])
        RMSE_close = np.sqrt(MSE_close)
        RMSE_close_percentage = (RMSE_close / np.mean(new_data["Close"])) * 100
        st.write(
            f"RMSE Close Harian {dropdown} : {RMSE_close:.3f} atau {RMSE_close_percentage:.3f}%"
        )
        st.write("--------")

    option2 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart",
        cols2,
        default=["Open", "open_predicted"],
        key="chart_predict",
    )

    if option2:
        st.subheader("Grafik Harian")
        data_daily = new_data[option2]
        st.line_chart(data_daily, y=option2)

        # Memanggil fungsi untuk membuat plot
        fig_daily = plot_actual_vs_predicted(new_data, option2, dropdown_index)
        img_daily = io.BytesIO()
        fig_daily.savefig(img_daily, format="png")
        btn_daily = st.download_button(
            label="Unduh Gambar",
            data=img_daily,
            file_name=f"{dropdown}_Epoch{epoch_option}_Neuron{neurons_option}_BatchSize{batch_size_option}_daily.png",
            mime="image/png",
        )

        # Menampilkan plot di aplikasi Streamlit
        st.pyplot(fig_daily)
        st.write("--------")

    else:
        st.warning("Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!")

    with st.form("first_form"):
        start_predict = new_data.index[-1].date()
        today = pd.to_datetime(start_predict)
        next_month = today + timedelta(days=30)
        end_predict = st.date_input(
            "Tanggal Akhir Prediksi", value=next_month, key="input_end"
        )
        periode = (start_predict - end_predict).days
        st.form_submit_button("Submit")

    new_rows = pd.DataFrame(
        index=pd.date_range(
            start=new_data.index[-1], end=end_predict, freq="D", inclusive="right"
        ),
        columns=new_data.columns[:4],
    )

    new_prediction_data = pd.concat(
        [
            new_data[["Open", "High", "Low", "Close"]],
            new_rows,
        ],
        axis=0,
    )
    st.write(new_prediction_data)
    upcoming_trend = pd.DataFrame(
        columns=["Open", "High", "Low", "Close"], index=new_prediction_data.index
    )

    upcoming_trend.index = pd.to_datetime(upcoming_trend.index)

    current_seq = test_seq[-1:]

    for i in range(periode, 0):
        trend_prediction = loaded_model.predict(current_seq)
        upcoming_trend.iloc[i] = trend_prediction
        current_seq = np.append(current_seq[0][1:], trend_prediction, axis=0)
        current_seq = current_seq.reshape(test_seq[-1:].shape)

    upcoming_trend[["Open", "High", "Low", "Close"]] = MMS.inverse_transform(
        upcoming_trend[["Open", "High", "Low", "Close"]]
    )

    cols3 = upcoming_trend.columns.tolist()

    upcoming_trend = upcoming_trend.rename(
        columns={
            "Open": "Open_Trend",
            "High": "High_Trend",
            "Low": "Low_Trend",
            "Close": "Close_Trend",
        }
    )

    option3 = st.selectbox(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart",
        cols3,
        key="chart_next_predict",
    )

    option3_str = str(option3)
    option3_trend = f"{option3}_Trend"
    data_trend = upcoming_trend[option3_trend]
    data_trend = data_trend[start_predict + timedelta(days=1) :]

    st.subheader(f"Berikut Tren harga {dropdown_index} {option3_str} yang akan datang")
    st.dataframe(data_trend, use_container_width=True)

    trend_data_excel = io.BytesIO()
    with pd.ExcelWriter(trend_data_excel, engine="xlsxwriter") as writer:
        data_trend.to_excel(writer)
        st.download_button(
            label=f"Download {dropdown_index} Trend Data",
            data=trend_data_excel,
            file_name=f"{dropdown}_trend.xlsx",
            mime="application/vnd.ms-excel",
        )

    fig_tren, ax = plt.subplots(figsize=(15, 7.5))
    ax.plot(
        new_prediction_data.loc[:, option3],
        label=f"Harga {option3_str} Terkini",
    )
    ax.plot(
        upcoming_trend.loc[:, option3_trend],
        label=f"Tren harga {option3_str} yang akan datang",
    )
    ax.xaxis.set_major_formatter(
        DateFormatter("%Y-%m-%d")
    )  # Menyesuaikan formatter sumbu x
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=45, ha="right"
    )  # Menetapkan label pada sumbu x
    ax.set_xlabel("Tanggal", size=15)
    ax.set_ylabel(f"{dropdown_index} Price", size=15)
    ax.set_title(
        f"Peramalan tren harga {dropdown_index} {option3_str} yang akan datang", size=15
    )
    ax.legend()

    img_tren = io.BytesIO()
    fig_tren.savefig(img_tren, format="png")
    btn_tren = st.download_button(
        label="Unduh Gambar",
        data=img_tren,
        file_name=f"{dropdown}_{option3_str}_Epoch{epoch_option}_Neuron{neurons_option}_BatchSize{batch_size_option}_trend.png",
        mime="image/png",
    )

    st.pyplot(fig_tren)
