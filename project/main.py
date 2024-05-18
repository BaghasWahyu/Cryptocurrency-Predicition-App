import io as io
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd
import streamlit as st
from cryptocmd import CmcScraper
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta, datetime
from scrape import symbolCrypto, list_crypto, namaCrypto

st.title("Aplikasi Prediksi Harga Cryptocurrencies")

MMS = MinMaxScaler(feature_range=(0, 1))

# Set nilai kondisi default session
defaults = {
    "input_crypto": "Bitcoin",
    "input_epoch": 25,
    "input_neuron": 50,
    "input_batch_size": 16,
    "chart_crypto_1": ["Open"],
    "chart_crypto_2": ["Volume"],
    "chart_predict": [
        "Open",
        "o_predicted",
        "High",
        "h_predicted",
        "Low",
        "l_predicted",
        "Close",
        "c_predicted",
    ],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def load_trained_model(path_to_model):
    model = load_model(path_to_model)
    return model


@st.cache_data
def calculate_MAPE(actual, predicted):
    MAPE = np.mean((np.abs(np.subtract(actual, predicted) / actual))) * 100
    return MAPE


@st.cache_data
def calculate_RMSE(actual, predicted):
    MSE = mean_squared_error(actual, predicted)
    RMSE = np.sqrt(MSE)
    RMSE_percentage = (RMSE / np.mean(actual)) * 100
    return RMSE, RMSE_percentage


@st.cache_data
def margin_of_error(actual, predicted):
    difference = actual - predicted
    APE = (abs(difference) / actual) * 100
    return difference, APE


@st.cache_data
def create_sequence(dataset):
    sequences, labels = [], []
    for start_idx in range(len(dataset) - 5):
        sequences.append(dataset.iloc[start_idx : start_idx + 5])
        labels.append(dataset.iloc[start_idx + 5])
    return np.array(sequences), np.array(labels)


@st.cache_data
def plot_actual_vs_predicted(dataframe, opsi, crypto_name, epoch, neuron, batchSize):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xticklabels(dataframe.index, rotation=45)
    for item in opsi:
        dataframe[[item]].plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Crypto Price")
    ax.set_title(
        f"Actual vs Predicted {crypto_name} Epoch {epoch} Neuron {100} Batch Size {batchSize} price"
    )
    ax.legend()
    return fig


@st.cache_data
def get_latest_price(cryptocurrency, start_predict, end_predict):
    scraper = CmcScraper(
        cryptocurrency,
        start_predict.strftime("%d-%m-%Y"),
        end_predict.strftime("%d-%m-%Y"),
    )
    data = scraper.get_dataframe()
    data[["Open", "High", "Low", "Close"]] = data[
        ["Open", "High", "Low", "Close"]
    ].round(2)
    return data[::-1].reset_index(drop=True)


with st.sidebar:
    dropdown_index = st.selectbox(
        "Pilih Salah Satu Cryptocurrency", namaCrypto, key="input_crypto"
    )
    epoch_option = st.selectbox("Pilih Salah Satu Epoch", [25, 50], key="input_epoch")
    neurons_option = st.selectbox(
        "Pilih Salah Satu Neurons", [50, 100], key="input_neuron"
    )
    batch_size_option = st.selectbox(
        "Pilih Salah Satu Batch Size", [16, 32], key="input_batch_size"
    )
    dropdown = symbolCrypto[namaCrypto.index(dropdown_index)]

st.subheader(
    "Berikut 5 Cryoptocurrency tertinggi berdasarkan Market Capitalization per 31 Desember 2023 menurut situs web coinmarketcap.com"
)
st.dataframe(list_crypto, use_container_width=True)

if len(dropdown) > 0:
    st.subheader(f"Berikut data historis {dropdown_index}")

    startdate = datetime(2019, 1, 1).date()
    enddate = datetime(2024, 5, 12).date()

    df_latest_price = get_latest_price(dropdown, startdate, enddate)
    data_historis = pd.read_excel(
        f"./data_historis/{dropdown}_data_historis.xlsx",
        index_col=0,
        parse_dates=True,
        engine="openpyxl",
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

    cols1 = [
        col
        for col in df.columns.tolist()
        if col not in ["Date", "Volume", "Market Cap"]
    ]

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
        cols1_5 = [
            col
            for col in df.columns.tolist()
            if col not in ["Date", "Open", "High", "Low", "Close"]
        ]

    else:
        st.warning("Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!")

    option1_5 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart",
        cols1_5,
        default=["Volume"],
        key="chart_crypto_2",
    )
    if option1_5:
        st.line_chart(df[["Date"] + option1_5].set_index("Date"))

    else:
        st.warning("Silahkan Pilih Aspek yang akan Ditampilkan Terlebih Dahulu!")

    crypto_data = df.set_index("Date").copy()
    crypto_data = crypto_data.drop(columns=["Market Cap", "Volume"])
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

    with st.expander("Shape"):
        test_predicted = loaded_model.predict(test_seq)
        test_inverse_predicted = MMS.inverse_transform(test_predicted)
        test_inverse_label = MMS.inverse_transform(test_label)

    st.write(f"Test sequence shape {test_seq.shape}")
    st.write(f"Test label shape {test_label.shape}")
    st.write(f"Test predicted shape {test_predicted.shape}")

    new_data = pd.concat(
        [
            crypto_data.iloc[-test_inverse_predicted.shape[0] :].copy(),
            pd.DataFrame(
                test_inverse_predicted,
                columns=[
                    "o_predicted",
                    "h_predicted",
                    "l_predicted",
                    "c_predicted",
                ],
                index=crypto_data.iloc[-test_inverse_predicted.shape[0] :].index,
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

    for col in ["open", "high", "low", "close"]:
        (
            new_data[f"{col[0]}_difference"],
            _,
        ) = margin_of_error(new_data[col.capitalize()], new_data[f"{col[0]}_predicted"])

    for col in ["open", "high", "low", "close"]:
        (
            _,
            new_data[f"{col[0]}_APE"],
        ) = margin_of_error(new_data[col.capitalize()], new_data[f"{col[0]}_predicted"])

    st.subheader(f"Berikut data {dropdown_index} Terkini dan yang Teprediksi")
    number = st.number_input(
        "Masukkan angka untuk pembulatan di belakang koma",
        value=4 if dropdown == "USDT" else 2,
        placeholder="Ketikan Angka....",
    )

    # Pembulatan nilai pada dataframe
    new_data = new_data.round(number)

    st.text("Harian")
    new_data_daily_io = io.BytesIO()
    with pd.ExcelWriter(new_data_daily_io, engine="xlsxwriter") as writer1:
        new_data.to_excel(writer1)
        workbook = writer1.book
        worksheet = writer1.sheets["Sheet1"]
        # adjust the column widths based on the content
        for i, col in enumerate(new_data.columns):
            width = max(new_data[col].apply(lambda x: len(str(x))).max(), len(col))
            worksheet.set_column(i, i, width)
    new_data_daily_io.seek(0)
    st.download_button(
        label=f"Download Data Harian {dropdown_index}",
        data=new_data_daily_io.getvalue(),
        file_name=f"{dropdown}_Epoch{epoch_option}_Neuron{neurons_option}_BatchSize{batch_size_option}_daily.xlsx",
        mime="application/vnd.ms-excel",
    )
    st.dataframe(new_data, use_container_width=True)

    mape_rmse_cols = ["Open", "High", "Low", "Close"]
    for col in mape_rmse_cols:
        st.write("--------")
        if dropdown == "USDT":
            st.write(
                f"MAPE {col}: {calculate_MAPE(new_data[col], new_data[f'{col[0].lower()}_predicted']):.5f}%"
            )
            st.write(
                f"RMSE {col}: {calculate_RMSE(new_data[col], new_data[f'{col[0].lower()}_predicted'])[0]:.5f} atau {calculate_RMSE(new_data[col], new_data[f'{col[0].lower()}_predicted'])[1]:.5f}%"
            )
        else:
            st.write(
                f"MAPE {col}: {calculate_MAPE(new_data[col], new_data[f'{col[0].lower()}_predicted']):.3f}%"
            )
            st.write(
                f"RMSE {col}: {calculate_RMSE(new_data[col], new_data[f'{col[0].lower()}_predicted'])[0]:.3f} atau {calculate_RMSE(new_data[col], new_data[f'{col[0].lower()}_predicted'])[1]:.3f}%"
            )
    st.write("--------")
    RMSE_total, RMSE_total_percentage = calculate_RMSE(
        new_data[["Open", "High", "Low", "Close"]],
        new_data[["o_predicted", "h_predicted", "l_predicted", "c_predicted"]],
    )
    st.write(
        f"RMSE Total Harian {dropdown} : {RMSE_total:.5f} atau {RMSE_total_percentage:.5f}%"
        if dropdown == "USDT"
        else f"RMSE Total {dropdown} : {RMSE_total:.3f} atau {RMSE_total_percentage:.3f}%"
    )
    st.write("--------")
    option2 = st.multiselect(
        "Pilih Aspek untuk ditampilkan dalam bentuk Line Chart",
        new_data.columns.tolist(),
        default=[
            "Open",
            "o_predicted",
            "High",
            "h_predicted",
            "Low",
            "l_predicted",
            "Close",
            "c_predicted",
        ],
        key="chart_predict",
    )

    if option2:
        st.subheader("Grafik Harian")
        st.line_chart(new_data[option2], y=option2)
        # Memanggil fungsi untuk membuat plot
        fig_daily = plot_actual_vs_predicted(
            new_data,
            option2,
            dropdown_index,
            epoch_option,
            neurons_option,
            batch_size_option,
        )
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
        date = "2024-05-12"
        today = pd.to_datetime(date)
        next_days = today + timedelta(days=30)
        end_predict = st.date_input(
            "Tanggal Akhir Prediksi", value=next_days, key="input_end"
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
            "Open": "open_trend",
            "High": "high_trend",
            "Low": "low_trend",
            "Close": "close_trend",
        }
    )

    trend_data_excel = io.BytesIO()
    with pd.ExcelWriter(trend_data_excel, engine="xlsxwriter") as writer2:
        upcoming_trend[start_predict + timedelta(days=1) :].to_excel(writer2)
    trend_data_excel.seek(0)
    st.download_button(
        label=f"Download {dropdown_index} Trend Data",
        data=trend_data_excel.getvalue(),
        file_name=f"{dropdown}_trend.xlsx",
        mime="application/vnd.ms-excel",
    )
    next_days_cols = ["Open", "High", "Low", "Close"]
    for col in next_days_cols:
        option3_trend = f"{col.lower()}_trend"
        data_trend = upcoming_trend[option3_trend]
        data_trend = data_trend[start_predict + timedelta(days=1) :]

        st.subheader(f"Berikut Tren harga {dropdown_index} {col} yang akan datang")

        st.dataframe(data_trend, use_container_width=True)

        fig_tren, ax = plt.subplots(figsize=(15, 7.5))
        ax.plot(
            new_prediction_data.loc[:, col],
            label=f"Harga {col} Terkini",
        )
        ax.plot(
            upcoming_trend.loc[:, option3_trend],
            label=f"Tren harga {col} yang akan datang",
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
            f"Peramalan tren harga {dropdown_index} {col} yang akan datang",
            size=15,
        )
        ax.legend()

        img_tren = io.BytesIO()
        fig_tren.savefig(img_tren, format="png")
        btn_tren = st.download_button(
            label="Unduh Gambar",
            data=img_tren,
            file_name=f"{dropdown}_{col}_Epoch{epoch_option}_Neuron{neurons_option}_BatchSize{batch_size_option}_trend.png",
            mime="image/png",
        )

        st.pyplot(fig_tren)
