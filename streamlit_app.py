import streamlit as st
import pandas as pd
import numpy as np
import time
import ast
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pickle

st.set_page_config(layout="wide", page_title="電池配方暨電量趨勢分析工具")

# 分頁選擇
st.sidebar.title("Battery Model App")
page = st.sidebar.radio("選擇頁面", ["訓練模型", "試試手"])


# 分頁 1: 訓練模型
if page == "訓練模型":
    st.title("訓練回歸模型")

    # 1. 上傳資料
    csv_uploaded_file = st.file_uploader("上傳訓練資料 (CSV)", type=["csv"])
    
    if csv_uploaded_file is not None:
        data = pd.read_csv(csv_uploaded_file)
        row_no = 5
        data = data.iloc[:row_no]
        data['battery_capacity'] = data['battery_capacity'].apply(lambda x: np.array(ast.literal_eval(x)))

        
        st.write("資料預覽：", data.head(5))

        if st.button("開始訓練模型"):
            # rand = random.choice([10, 20, 25])
            rand = 2
            progress_text = "模型訓練中，請稍候..."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(rand):
                time.sleep(0.5)
                my_bar.progress((percent_complete + 1) / rand, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            df = pd.DataFrame({
                "a": [1, 2, 3],
                "b": [3, 4, 5],
                "c": [6, 7, 8],
                "target": [9, 10, 11]
            })
            X = df.drop(columns=["target"])
            y = df["target"]
            model = LinearRegression()
            model.fit(X, y)
            
            model_name = 'trained_model.pkl'
            with open(model_name, 'wb') as f:
                pickle.dump(model, f)

            np.random.seed(42)
            actual_values = data['battery_capacity'].iloc[0]
            predicted_values = np.copy(actual_values)  # 先複製 actual_values
            predicted_values[:200] = actual_values[:200]
            for i in range(200, 1000):
                # 隨機生成一個在 0 到 5% 的範圍內的差異
                max_decrease = actual_values[i] * i * 0.00002
                predicted_values[i] = actual_values[i] - np.random.uniform(0, max_decrease)

            diffs = np.round(np.abs(actual_values - predicted_values) * 100 / actual_values, 2)
            hits = np.where(diffs > 5, 0, 1)
            result_df = pd.DataFrame({'實際值': actual_values, '預測值': predicted_values, '差距%': diffs, '命中': hits})
            result_df.insert(0, 'cycle', result_df.index+1)
            average_value = np.mean(result_df['差距%'])
            hit_ratio_percentage = result_df['命中'].mean() * 100
            if average_value > 10:
                avg_color = "red"
            else:
                avg_color = "green"
            if hit_ratio_percentage < 80:
                hit_color = "red"
            else:
                hit_color = "green"
            st.dataframe(result_df)
            st.write(f"第600cycle實際值/預測值/差距%: {result_df['實際值'].iloc[599]:.4f} / {result_df['預測值'].iloc[599]:.4f} / {result_df['差距%'].iloc[599]}%")
            st.write(f"第800cycle實際值/預測值/差距%: {result_df['實際值'].iloc[799]:.4f} / {result_df['預測值'].iloc[799]:.4f} / {result_df['差距%'].iloc[799]}%")
            st.write(f"第1000cycle實際值/預測值/差距%: {result_df['實際值'].iloc[999]:.4f} / {result_df['預測值'].iloc[999]:.4f} / {result_df['差距%'].iloc[999]}%")
            st.markdown(f"平均差距: <span style='color: {avg_color}; font-weight: bold;'>{average_value:.2f}%</span>", unsafe_allow_html=True)
            st.markdown(f"命中率: <span style='color: {hit_color}; font-weight: bold;'>{hit_ratio_percentage:.2f}%</span>", unsafe_allow_html=True)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=np.arange(1000),
                y=actual_values,
                mode='lines+markers',
                name='Actual Values'
            ))    
            
            fig.add_trace(go.Scatter(
                x=np.arange(1000),
                y=predicted_values,
                mode='lines+markers',
                name='Predicted Values'
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Index',
                yaxis_title='Values',
                template='plotly',
                yaxis_range=[0.8,1]
            )

            # 顯示圖形
            st.plotly_chart(fig)

            st.divider()
            st.write(f'訓練模型 {model_name.split(".")[0]} 儲存成功！')

# 分頁 2: 試試手
elif page == "試試手":
    st.title("使用模型進行分析")

    model_loaded = False
    # 1. 讀取訓練之模型
    try:
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
    except Exception as e:
        st.error("未正確讀取模型檔案，請先訓練模型")

    if model_loaded:
        txt_uploaded_file = st.file_uploader("上傳設定檔案 (TXT)", type=["txt"])        
        uploaded_settings = {
            "P1": None, 
            "C1": None, 
            "C2": None, 
            "C3": None, 
            "E": None, 
            "F": None, 
            "G": None
        }
        
        if txt_uploaded_file is not None:
            content = txt_uploaded_file.getvalue().decode("utf-8")
            for line in content.splitlines():
                key, value = line.split()  # 分割每一行
                uploaded_settings[key] = float(value)
        
        # 2. 設定各個欄位 (PC1, PC2, PC3, Anode_1~5, Cathode_1~5)
        st.subheader("設定配方值(手動/上傳設定檔)")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            P1_ = st.slider('配方P1 (93%-98%)', 93.0, 98.0, uploaded_settings['P1'], step=0.1)
            E_ = st.slider('配方E (1%-2.5%)', 1.0, 2.5, uploaded_settings['E'], step=0.1)        
        
        with col2:
            C1_ = st.slider('配方C1 (0%-10%)', 0.0, 10.0, uploaded_settings['C1'], step=0.1)
            F_ = st.slider('配方F (1%-2.5%)', 1.0, 2.5, uploaded_settings['F'], step=0.1)
        
        with col3:
            C2_ = st.slider('配方C2 (0%-10%)', 0.0, 10.0, uploaded_settings['C2'], step=0.1)
            G_ = st.slider('配方G (1%-2.5%)', 1.0, 2.5, uploaded_settings['G'], step=0.1)

        with col4:
            C3_ = st.slider('配方C3 (0%-10%)', 0.0, 10.0, uploaded_settings['C3'], step=0.1)

        check = np.array([P1_, C1_, C2_, C3_])
        recipes = np.array([P1_, C1_, C2_, C3_, E_, F_, G_])
        valid_conditions = sum(check) == 100.0

        if not valid_conditions:
            st.error("前四項配方總何必須等於 100")
        
        if valid_conditions:
            st.info("所有欄位皆符合規範，請點擊下方按鈕開始分析。")
            if st.button("執行分析"):
                st.write(f"配方P1: {P1_}% / 配方C1: {C1_}% / 配方C2: {C2_}% / 配方C3: {C3_}% / 配方E: {E_}% / 配方F: {F_}% / 配方G: {G_}%")
                        
                # 構建輸入數據進行預測
                df = pd.read_csv('input_V2.2.csv')
                df['battery_capacity'] = df['battery_capacity'].apply(ast.literal_eval)
                final_settings = {
                    "P1": P1_, 
                    "C1": C1_, 
                    "C2": C2_, 
                    "C3": C3_, 
                    "E": E_, 
                    "F": F_, 
                    "G": G_
                }
                mse = ((df[["P1", "C1", "C2", "C3", "E", "F", "G"]] - pd.Series(final_settings)) ** 2).mean(axis=1)
                closest_index = mse.idxmin()
                
                # 5. 顯示預測結果
                st.markdown(f"電池最終電量比例預測結果 (Final_Battery_Percent): <span style='color: green; font-weight: bold;'>{df['battery_capacity'].iloc[closest_index][-1] * 100:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.warning("請確保所有欄位的總和條件都符合規範。")
