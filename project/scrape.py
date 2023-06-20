from bs4 import BeautifulSoup
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Tugas Akhir", layout="wide")

df = pd.DataFrame()

crypto_name_list = []
crypto_market_cap_list = []
crypto_price_list = []
crypto_circulating_supply_list = []
crypro_symbol_list = []


@st.cache_data
def scrape(date):

    URL = 'https://coinmarketcap.com/historical/'+date
    webpage = requests.get(URL)
    soup = BeautifulSoup(webpage.text, 'html.parser')
    tr = soup.find_all('tr', attrs={'class': 'cmc-table-row'})
    count = 0
    for row in tr:
        if count == 10:
            break
        count = count + 1
        name_column = row.find('td', attrs={
                               'class': 'cmc-table__cell cmc-table__cell--sticky cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__name'})
        crypto_name = name_column.find(
            'a', attrs={'class': 'cmc-table__column-name--name cmc-link'}).text.strip()
        coin_market_cap = row.find('td', attrs={
                                   'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__market-cap'}).text.strip()
        crypto_price = row.find('td', attrs={
                                'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__price'}).text.strip()
        crypto_circulating_supply_symbol = row.find('td', attrs={
                                                    'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__circulating-supply'}).text.strip()
        crypto_circulating_supply = crypto_circulating_supply_symbol.split(' ')[
            0]
        crypto_symbol = crypto_circulating_supply_symbol.split(' ')[1]

        crypto_name_list.append(crypto_name)
        crypto_market_cap_list.append(coin_market_cap)
        crypto_price_list.append(crypto_price)
        crypto_circulating_supply_list.append(crypto_circulating_supply)
        crypro_symbol_list.append(crypto_symbol)


@st.cache_data
def get_symbol():
    scrape(date='20221231/')
    df['Name'] = crypto_name_list
    df['Market Capitalization'] = crypto_market_cap_list
    df['Price'] = crypto_price_list
    df['Circulating Supply'] = crypto_circulating_supply_list
    df['Symbol'] = crypro_symbol_list

    list_crypto = df[:5]

    nama_crypto = list_crypto['Name']
    symbol_crypto = list_crypto['Symbol']
    return nama_crypto, symbol_crypto


if __name__ == "__main__":
    nama_crypto, symbol_crypto = get_symbol()

    symbolCrypto = []

    for i in symbol_crypto:
        j = i.replace(' ', '-')
        symbolCrypto.append(j)
