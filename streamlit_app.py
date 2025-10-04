import streamlit as st
import os
from dotenv import load_dotenv

# We assume retriever.py is in a 'core' directory
# Make sure you have an __init__.py file in the 'core' directory
from core.retriever import InformationRetriever

# --- Page Configuration ---
st.set_page_config(
    page_title="KSU Engineering Q&A",
    page_icon="🤖",
    layout="centered"
)

# --- Environment Variables & Secrets ---
# For local development, create a .env file with your OPENAI_API_KEY
load_dotenv()
# For Streamlit Cloud, set the secret in the app settings
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("مفتاح OpenAI API غير موجود. الرجاء إضافته لملف .env أو في إعدادات التطبيق السحابية.")
    st.stop()


# --- Load Retriever (Cached for Performance) ---
@st.cache_resource
def load_retriever_system():
    """
    Loads the InformationRetriever instance using Streamlit's caching.
    The retriever will be loaded only once.
    """
    print("Initializing InformationRetriever for the first time...")
    # --- Configuration ---
    # Make sure these paths are correct within your project structure
    excel_file_path = os.path.join("data", "crawled_chunks_final.xlsx")
    embeddings_cache_path = os.path.join("data", "ksa_engineering_embeddings.npy")
    openai_model = "gpt-4o" # Recommended model for quality
    # Use your fine-tuned model from Hugging Face
    embedding_model_name = "TheMohanad1/Fine-Tuned-E5"

    # Ensure data file exists before initializing
    if not os.path.exists(excel_file_path):
        st.error(f"ملف البيانات غير موجود. تأكد من وجود '{excel_file_path}'.")
        # Stop the app if data files are missing
        st.stop()

    retriever_instance = InformationRetriever(
        openai_model=openai_model,
        excel_path=excel_file_path,
        embeddings_path=embeddings_cache_path,
        embedding_model_name=embedding_model_name
    )
    print("InformationRetriever loaded successfully.")
    return retriever_instance

try:
    retriever = load_retriever_system()
except Exception as e:
    st.error(f"حدث خطأ فادح أثناء تهيئة النظام: {e}")
    st.stop()


# --- Chat Interface ---
st.title("🤖 مساعد كلية الهندسة بجامعة الملك سعود")
st.caption("اطرح سؤالك حول البرامج الأكاديمية، شروط القبول، أو أي معلومة تخص الكلية.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("ما هو سؤالك؟"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("...جاري البحث عن أفضل إجابة"):
            # Prepare chat history for the retriever
            history_for_retriever = [
                msg for msg in st.session_state.messages if msg["role"] in ['user', 'assistant']
            ]
            
            # Get the result from your powerful backend
            result = retriever.search_and_answer(
                query=prompt,
                history=history_for_retriever,
                top_k=7
            )
            
            response_text = result.get("answer", "عفواً، لم أتمكن من إيجاد إجابة.")
            sources = result.get("sources", [])

            st.markdown(response_text)
            
            # Display sources in an expander
            if sources:
                with st.expander("📚 عرض المصادر المستخدمة"):
                    for i, source in enumerate(sources):
                        st.write(f"**المصدر رقم {i+1}:**")
                        st.info(source.get('text', 'لا يوجد نص للمصدر.'))
                        st.write(f"[رابط المصدر]({source.get('source_url', '#')})")
                        st.markdown("---")


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})


