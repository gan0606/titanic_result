import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from catboost import CatBoost,Pool
from catboost import CatBoostClassifier

import shap
import os
# 訓練したモデルを保存したfileを読み込むmodule
import pickle

# pyplotを使用する際に注記が出ないようにする文
st.set_option("deprecation.showPyplotGlobalUse", False)

# 関数化する
def main():
    # タイトル
    st.title("Taitanic号の生存者分析")

    # サイドバーのmenu
    menu = ["分析の目的", "生存確率の分布", "生存の要因", "結論"]
    # サイドバーの作成
    chosen_menu = st.sidebar.selectbox(
        "menu選択", menu
    )

    # ファイルの設定
    # 訓練済みのモデルファイル
    model_file = "./catmodel.pkl"
    # 読み込めているかを確認
    is_model_file = os.path.isfile(model_file)

    # 前処理済みのすべてのデータ
    # all_file = "./dst/all_data.csv"
    # 説明変数のtraindata
    x_train_file = "./X_train.csv"
    # 説明変数のtestndata
    x_test_file = "./X_test.csv"
    # 説明変数のtraindata
    y_train_file = "./y_train.csv"
    # 説明変数のtraindata
    # y_test_file = "./dst/y_test.csv"
    # 読み込めているかを確認
    # is_all_file = os.path.isfile(all_file)
    is_x_train_file = os.path.isfile(x_train_file)
    is_x_test_file = os.path.isfile(x_test_file)
    is_y_train_file = os.path.isfile(y_train_file)
    # is_y_test_file = os.path.isfile(y_test_file)

    # モデルを再学習するかどうか
    # 再学習しないことを宣言
    no_update = True

    # printで出力すると、ターミナルに出る
    # st.writeだとブラウザ上に出る
    # print(is_all_file)
    print(is_x_train_file)
    print(is_x_test_file)
    print(is_y_train_file)
    # print(is_y_test_file)
    print(no_update)
    

    X_train = pd.read_csv("./X_train.csv")
    X_test = pd.read_csv("./X_test.csv")
    y_train = pd.read_csv("./y_train.csv")
    print("データを読み込みました")


    cat_model = pickle.load(open('./catmodel.pkl', 'rb'))
    print("モデルが読み込まれました")
    

    # # 読み込んだ、または作成したモデルで生存確率を予測
    pred_prob = cat_model.predict(X_test, prediction_type="Probability")
    pred_prob = np.round(pred_prob, 3)

    # menuの中身
    if chosen_menu == "分析の目的":
        st.subheader("分析目的")
        st.write("・乗客の属性情報 (年齢、性別、社会階級など) を基に、生存率を予測すること")
        st.write("・生存率に影響を与えた要因を分析すること")

    elif chosen_menu == "生存確率の分布":
        st.subheader("訓練データの生存確率")
        type_ = ["No Survived=0", "Survived=1"]

        fig = make_subplots(rows=1, cols=1)

        # 円グラフの設定
        fig.add_trace(go.Pie(labels=type_, values=y_train.value_counts(), name="Survived"))

        # 円グラフにholeとhover機能をつける
        fig.update_traces(hole=0.4, hoverinfo="label+percent", textfont_size=16)

        # pieのレイアウトを設定
        fig.update_layout(
        # タイトル
        title_text = "<b>Survivedの分布<b>",
        # pieの中央にsurvivedと表示させる
        annotations = [dict(text="Survived", x=0.5, y=0.5, font_size=20, showarrow=False)])
        # streamlitで表示
        st.plotly_chart(fig)

        st.subheader("予測したデータの生存確率分布")
        fig = px.histogram(pred_prob[:, 1])

        # 各図をsubplotsで表示
        fig = go.Figure(data=fig)
        fig.update_layout(height=600, showlegend=False, title="<b>生存確率の分布<b>")
        # streamlitで表示
        st.plotly_chart(fig)

    elif chosen_menu == "生存の要因":
        st.subheader("SHAP分析の結果")
        st.write("SHAP値とは、乗客の属性情報 (年齢、性別、社会階級など)がモデルの予測にどの程度貢献しているかを数字で表しています。")

        # shap分析の結果を表示
        st.write("中央線より左側が死亡を表し、右側が生存を表します。")
        st.write("例えば、Age(年齢)のSHAP値からは、高齢であればあるほど生存したことが読み取れます。")
        st.write("また、Family(同行者数)のSHAP値からは、同行した人数が多いほど死亡したことが読み取れます。")
        st.write("[注意]SHAP値は、定量的な属性情報の数字的な強さを表せますが、定性的な属性情報の数字的な強さは表せません。")
        # shapの可視化
        pool = Pool(X_train, y_train)
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(pool)
        fig = shap.summary_plot(shap_values, X_train)
        st.pyplot(fig)

    elif chosen_menu == "結論":
        st.subheader("結論")
        st.write("・Sex(性別)やPclass(船室料金)が生死に大きな要因を与えました。")
        st.write("・ご高齢の方が優先的に助けられたと考えられます。")
        st.write("・同行者数が多いほど、救助は後回しにされたと考えられます。")


# streamlitを実行したときにmain()を実行するという表記
if __name__ == "__main__":
    main()
