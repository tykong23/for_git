import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from mlflow.pyfunc import load_model
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import os

demand_file_path = 'data/demand2.csv'
inventory_file_path = 'data/inventory.csv'
stock_file_path = 'data/stock.csv'
lead_time_file_path = 'data/average_lead_time.csv'
mean_stock_file_path = 'data/mean_stock.csv'

mlflow.set_tracking_uri("http://127.0.0.1:5000/") 

def load_model_from_mlflow(model_name):
    model_uri = f"models:/{model_name}/latest"
    return load_model(model_uri)

def remove_punctuation(column):
    return column.str.replace(',', '', regex=True).replace('.00', '', regex=True).str.strip()

def add_data_to_csv(file_path, data):
    df = pd.read_csv(file_path)
    new_data = pd.DataFrame(data)
    updated_df = pd.concat([df, new_data], ignore_index=True)
    updated_df.to_csv(file_path, index=False)
    return updated_df

def update_stock_csv():
    demand_df = pd.read_csv(demand_file_path)
    inventory_df = pd.read_csv(inventory_file_path)
    stock_df = make_stock_frame(inventory_df, demand_df)
    stock_df.to_csv(stock_file_path, index=False)
    return stock_df

def make_stock_frame(inventory, demand):
    inventory = inventory.copy()
    demand = demand.copy()
    inventory['y'] = inventory['y'].fillna(0)
    demand['y'] = demand['y'].fillna(0)
    combined = pd.concat([inventory, demand])
    combined['y'] = combined.apply(lambda row: -row['y'] if row.name in demand.index else row['y'], axis=1)
    stock = combined.groupby(['i', 't']).sum().groupby('i').cumsum().reset_index()
    stock0 = stock.groupby('i')['y'].transform('min')
    stock['y'] = stock['y'] - stock0
    return stock

def calc_safety_stock(demand, avg_lead_time, service_level):
    L = avg_lead_time / np.timedelta64(1, 'D')
    Z = stats.norm.ppf(service_level)
    sigma_d = demand.groupby('i').std()['y']
    ss = Z * sigma_d * np.sqrt(L)
    ss = ss.fillna(0)
    return ss

def calc_reorder_point(demand, avg_lead_time, safety_stock):
    L = avg_lead_time / np.timedelta64(1, 'D')
    mean_d = demand.groupby('i').mean()['y']
    return np.ceil(mean_d * L + safety_stock)

def plot_product_demand(product_id, test, y_test, y_pred):
    idx = f'i_{product_id}'
    test_i = test[test[idx] == 1]
    plt.figure(figsize=(14, 7))
    plt.plot(test_i['t'], y_test.loc[test_i.index], label='실제값')
    plt.plot(test_i['t'], y_pred[test_i.index.values], label='예측값', alpha=0.7)
    plt.xlabel('날짜')
    plt.ylabel('수요')
    plt.title(f'상품 ID: {product_id}의 실제 수요와 LightGBM 예측 수요')
    plt.legend()
    st.pyplot()
    
def simulate_stock(product_id):
    product_id = str(product_id)
    
    lead_time_df = pd.read_csv(lead_time_file_path)
    mean_stock_df = pd.read_csv(mean_stock_file_path)
    avg_lead_time = lead_time_df.set_index('상품코드_WMS_')['average_lead_time'].drop_duplicates()
    avg_order_quantity = mean_stock_df.set_index('상품코드_WMS_')['mean_stock']
    demand_df = pd.read_csv(demand_file_path)
    stock_df = pd.read_csv(stock_file_path)

    s = demand_df[demand_df['i'] == product_id].std()['y']
    lt = pd.to_timedelta(avg_lead_time[product_id]).days
    ss = stats.norm.ppf(0.95) * s * np.sqrt(lt)

    mean_d = demand_df[demand_df['i'] == product_id].mean()['y']
    ROP = mean_d * lt + ss
    q = avg_order_quantity[product_id]

    init_stock = stock_df[(stock_df['i'] == product_id)]['y'].values[-1]

    current_stock = init_stock
    inventory_levels = []
    order_dates = []

    rows_as_tuples = [tuple(row) for row in demand_df[demand_df.i == product_id][['t', 'y']].itertuples(index=False, name=None)]
    for date, d in rows_as_tuples:
        current_stock -= d
        inventory_levels.append(current_stock)

        if current_stock <= ROP:
            current_stock += q
            order_dates.append(date)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot([date for date, _ in rows_as_tuples], inventory_levels, label='최적화된 재고 수준', color='blue')
    ax.plot(stock_df[stock_df['i'] == product_id]['t'], stock_df[stock_df['i'] == product_id]['y'], label='현재 재고 수준', color='orange')
    for order_date in order_dates:
        ax.axvline(x=order_date, color='red', linestyle='--', label='주문 발생' if order_date == order_dates[0] else "")
    ax.set_xlabel('날짜')
    ax.set_ylabel('재고 수준')
    ax.set_title(f'상품 ID: {product_id}의 재고 수준 시뮬레이션')
    ax.legend()
    st.pyplot(fig)

    st.write({'재주문점': ROP, '안전 재고': ss, '평균 주문량': q, '평균 리드 타임': lt})

st.sidebar.title("Dashboard")
option = st.sidebar.selectbox("섹션 선택", ["재고 추가", "재고 현황 및 주요 지표", "수요 예측 결과", "재고 시뮬레이션"])

if option == "재고 추가":
    st.title('입/출고 입력')
    st.header('입/출고 데이터 추가')
    csv_choice = st.selectbox('입고/출고 선택', ['출고', '입고'])

    if csv_choice == '출고':
        st.subheader("출고 데이터 입력")
        i = st.text_input("상품 코드 (i)")
        t = st.text_input("날짜 (t, 예: 2024-06-21)")
        y = st.number_input("수량 (y)", step=1.0)
        if st.button('출고에 데이터 추가'):
            new_data = {'i': [i], 't': [t], 'y': [y]}
            updated_demand_df = add_data_to_csv(demand_file_path, new_data)
            st.write("출고에 데이터가 성공적으로 추가되었습니다.")
            st.write(updated_demand_df.tail())
            stock_df = update_stock_csv()
            st.write("재고가 성공적으로 업데이트되었습니다.")
    else:
        st.subheader("입고 데이터 입력")
        i = st.text_input("상품 코드 (i)")
        t = st.text_input("날짜 (t, 예: 2024-06-21)")
        y = st.number_input("수량 (y)", step=1.0)
        if st.button('입고에 데이터 추가'):
            new_data = {'i': [i], 't': [t], 'y': [y]}
            updated_inventory_df = add_data_to_csv(inventory_file_path, new_data)
            st.write("입고에 데이터가 성공적으로 추가되었습니다.")
            st.write(updated_inventory_df.tail())
            stock_df = update_stock_csv()
            st.write("재고가 성공적으로 업데이트되었습니다.")

elif option == "재고 현황 및 주요 지표":
    st.title('재고 현황 및 주요 지표')

    st.header('전체 현재 재고량')
    stock_df = pd.read_csv(stock_file_path)
    stock_df['i'] = stock_df['i'].astype(str)
    last_stock = pd.DataFrame(stock_df.pivot_table(index='i', columns='t', values='y').iloc[:, -1]).reset_index()
    fig = px.bar(last_stock, x=last_stock.columns[1], y='i', title='상품별 현재 재고량', orientation='h', labels={'i': '상품 ID', 'current_stock': '현재 재고량'})
    st.plotly_chart(fig)

    st.header('지정 상품의 현재 재고량')
    items = st.text_input('상품 ID 입력 (예: 10011,730211)')
    if st.button('재고량 확인'):
        items_list = items.split(',')
        st_df = last_stock.loc[last_stock['i'].isin(items_list), ['i', last_stock.columns[1]]].reset_index(drop=True)
        st_df.columns = ['상품', '현재 재고량']
        fig = px.bar(st_df, x='상품', y='현재 재고량', title=f'상품 {items}의 현재 재고량')
        st.plotly_chart(fig)

    st.header('상품의 실시간 재고 수준 추이')
    product_id = st.text_input('상품 ID 입력 (예: 10011)')
    if st.button('실시간 재고 추이 확인'):
        product_df = stock_df[stock_df['i'] == product_id]
        fig = px.line(product_df, x='t', y='y', title=f'상품 : {product_id} 재고 수준 추이')
        st.plotly_chart(fig)

    st.header('안전 재고 및 재주문점 계산')
    demand_df = pd.read_csv(demand_file_path)
    inventory_df = pd.read_csv(inventory_file_path)
    lead_time_df = pd.read_csv(lead_time_file_path)
    
    unique_lead_time_df = lead_time_df[['상품코드_WMS_', 'average_lead_time']].drop_duplicates()
    unique_lead_time_df['average_lead_time'] = pd.to_timedelta(unique_lead_time_df['average_lead_time'])
    inventory_df['입고량_pcs_'] = inventory_df['y']

    if 'y' in demand_df.columns and 'y' in inventory_df.columns:
        avg_lead_time = unique_lead_time_df.groupby('상품코드_WMS_')['average_lead_time'].mean()
        avg_lead_time.name = 'lt_bar'

        order_quantity_mean = np.round(inventory_df.groupby('i')['y'].mean())
        order_quantity_mean.name = 'q_bar'

        service_level = st.number_input('서비스 수준 입력 (예: 0.9 = 90%, 0.95 = 95%)', min_value=0.01, max_value=0.99, value=0.95, step=0.01)

        if st.button('계산'):
            safety_stock = calc_safety_stock(demand_df, avg_lead_time, service_level)
            reorder_point = calc_reorder_point(demand_df, avg_lead_time, safety_stock)

            st.header('안전 재고')
            st.write(safety_stock)

            st.header('재주문점')
            st.write(reorder_point)
    else:
        st.write("리드 타임 및 입고량에 대한 필요한 컬럼이 CSV 파일에 없습니다.")

elif option == "수요 예측 결과":
    st.title('수요 예측 결과')

    product_id_input = st.text_input('상품 ID 입력 (예: 10011)')

    if st.button('예측 결과 확인'):
        product_id = str(product_id_input).strip()

        if product_id:
            prophet_model = load_model_from_mlflow(f'Prophet_{product_id}')
            if prophet_model:
                demand_df = pd.read_csv(demand_file_path)
                demand_df = demand_df[demand_df['i'] == product_id]
                demand_df['t'] = pd.to_datetime(demand_df['t'])
                future = pd.DataFrame({'ds': pd.date_range(start=demand_df['t'].max(), periods=52, freq='W')})
                forecast = prophet_model.predict(future)
                
                st.write(f"상품 ID {product_id}에 대한 Prophet 예측 결과")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(forecast['ds'], forecast['yhat'], label='예측값', color='blue')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)
                ax.set_title(f'상품 ID: {product_id}의 Prophet 예측')
                ax.set_xlabel('날짜')
                ax.set_ylabel('예측 값')
                ax.legend()
                st.pyplot(fig)
            else:
                st.write(f'상품 ID {product_id}에 대한 Prophet 모델이 없습니다.')

            # Load LightGBM model and predict
            lgbm_model = load_model_from_mlflow('lightgbm')
            if lgbm_model:
                barcode_df = pd.read_csv('data/cv_barcode.csv')
                item = barcode_df['i'].tolist()

                demand = pd.read_csv(demand_file_path)
                demand = demand[demand['i'].isin(item)]
                demand['t'] = pd.to_datetime(demand['t'])
                demand['day'] = demand['t'].dt.day
                demand['month'] = demand['t'].dt.month
                demand['year'] = demand['t'].dt.year
                demand['dayofweek'] = demand['t'].dt.dayofweek

                for lag in range(1, 7):
                    demand[f'lag_{lag}'] = demand.groupby('i')['y'].shift(lag)

                demand = demand.dropna()
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(demand[['i']])
                encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['i']))
                data_encoded = pd.concat([demand.drop(columns=['i']).reset_index(drop=True), encoded_df], axis=1)

                test = data_encoded.loc[data_encoded['year'] > 2022].reset_index(drop=True)
                X_test = test.drop(columns=['y', 't'])
                y_test = test['y']
                y_pred = lgbm_model.predict(X_test)

                st.write(f"상품 ID {product_id}에 대한 LightGBM 예측 결과")
                plot_product_demand(product_id, test, y_test, y_pred)
            else:
                st.write(f'LightGBM 모델이 없습니다.')
        else:
            st.write("상품 ID를 입력해주세요.")

elif option == "재고 시뮬레이션":
    st.title('재고 시뮬레이션')

    product_id = st.text_input('상품 ID 입력 (예: 10011)')
    
    if st.button('시뮬레이션 시작'):
        if product_id:
            simulate_stock(product_id)
        else:
            st.write("상품 ID를 입력해주세요.")
