import os
import streamlit as st
from dotenv import load_dotenv

# --- ローカル用 .env 読み込み ---
load_dotenv()

# --- 環境変数または Secrets からAPIキーを取得 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# --- LangChainやOpenAIが環境変数から参照できるように設定 ---
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    st.error("❌ OPENAI_API_KEY が見つかりません。.env または Streamlit Secrets に設定してください。")
    st.stop()


st.set_page_config(page_title="LLMプロンプト送信デモ", page_icon="🤖", layout="centered")

st.title("🤖 LLM プロンプト送信デモ（LangChain + Streamlit）")
st.caption("ラジオボタンで専門家ロールを選び、入力内容をもとにLLMが回答します。")

# APIキー確認（未設定なら警告）
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が読み込めませんでした。.env の設定を確認してください。")
    st.stop()

# ====== 専門家ロール（A/B/C…）の定義 ======
# 必要に応じて増やせます。Aはご提示の例に合わせて「健康アドバイザー」を採用。
EXPERT_ROLES = {
    "A：健康アドバイザー": (
        "あなたは健康に関するアドバイザーです。"
        "医学的に安全で一般向けに配慮された助言のみを行い、"
        "必要に応じて医療機関の受診を案内してください。"
    ),
    "B：教育コンサルタント": (
        "あなたは教育コンサルタントです。"
        "学習計画や学習方法、モチベーション維持の観点から、"
        "具体的で実践的な提案をしてください。"
    ),
    "C：ビジネス戦略家": (
        "あなたはビジネス戦略の専門家です。"
        "MECEや3C/4P/5Forces等の枠組みを適宜用い、"
        "実行可能な打ち手と優先順位を提示してください。"
    ),
}

# ====== UI（左：設定 / 右：結果） ======
with st.sidebar:
    st.subheader("⚙️ 設定")
    role_label = st.radio(
        "専門家ロールを選択",
        options=list(EXPERT_ROLES.keys()),
        index=0,
        help="選択に応じてSystemメッセージが切り替わります。",
    )

    model_name = st.selectbox(
        "モデル",
        options=[
            "gpt-4o-mini",   # 例に合わせて既定
            "gpt-4o",        # お好みで
            "gpt-4.1-mini",  # 参考
        ],
        index=0,
        help="OpenAIのチャットモデルを選択します。",
    )

    temperature = st.slider(
        "Temperature（創造性）", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

# 入力フォーム
with st.form(key="prompt_form", clear_on_submit=False):
    user_text = st.text_area(
        "📝 質問 / 相談内容を入力",
        placeholder="例）最近眠れないのですが、どうしたらいいですか？",
        height=160,
    )
    submitted = st.form_submit_button("送信する ▶")

# ====== 送信処理 ======
if submitted:
    if not user_text.strip():
        st.warning("入力が空です。質問や相談内容を入力してください。")
        st.stop()

    # LangChainのLLMクライアント
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        # api_keyは環境変数OPENAI_API_KEYから自動取得
    )

    # 選択ロールに応じたSystemプロンプト
    system_prompt = EXPERT_ROLES[role_label]

    # ChatPromptTemplateでメッセージを組み立て
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm

    with st.spinner("LLMが回答を作成中…"):
        try:
            result = chain.invoke({"user_input": user_text})
            answer = result.content  # ChatMessageのcontent
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.stop()

    # ====== 表示 ======
    st.markdown("### ✅ 回答")
    st.write(answer)

    # デバッグ用（必要に応じて展開）
    with st.expander("🔍 使用したSystemプロンプト（参考）"):
        st.code(system_prompt, language="markdown")

