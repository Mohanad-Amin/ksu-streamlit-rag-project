🤖 KSU Engineering Q&A Assistant
تطبيق ويب تفاعلي مبني باستخدام Streamlit، يعمل كمساعد ذكي للإجابة على الأسئلة المتعلقة بكلية الهندسة في جامعة الملك سعود. يستخدم التطبيق بنية RAG (Retrieval-Augmented Generation) المتقدمة لتقديم إجابات دقيقة بناءً على قاعدة معرفة تم جمعها من الموقع الرسمي للكلية.

🔗 رابط النسخة الحية من التطبيق (سيتم إضافته لاحقًا)

🖼️ لقطة شاشة للتطبيق
(هنا يمكنك إضافة لقطة شاشة لواجهة التطبيق بعد تشغيله)

✨ الميزات الرئيسية
واجهة محادثة تفاعلية: واجهة مستخدم سهلة وبديهية مبنية باستخدام Streamlit.

استرجاع معلومات دقيق (RAG): يستخدم نموذج TheMohanad1/Fine-Tuned-E5 الذي تم ضبطه خصيصًا لتحويل النصوص إلى متجهات واسترجاع المعلومات الأكثر صلة من قاعدة بيانات FAISS.

إدراك سياق المحادثة: يقوم بإعادة صياغة سؤال المستخدم بناءً على سجل المحادثة لضمان دقة البحث.

إجابات مدعومة بالمصادر: يعرض النصوص الأصلية التي تم استخدامها لتوليد الإجابة لضمان الشفافية والموثوقية.

أداء عالي: يستخدم st.cache_resource لتحميل النماذج وقواعد البيانات مرة واحدة فقط، مما يضمن استجابة سريعة.

🛠️ التقنيات المستخدمة
إطار العمل: Streamlit

نماذج الذكاء الاصطناعي:

Retriever: TheMohanad1/Fine-Tuned-E5 (نموذج مضبوط من e5-base-v2)

Generator: OpenAI gpt-4o

فهرسة المتجهات: FAISS (Facebook AI Similarity Search)

المكتبات الرئيسية: Sentence-Transformers, Pandas, OpenAI

🚀 كيفية تشغيل المشروع محليًا
استنسخ المستودع:

git clone [https://github.com/Mohanad-Amin/ksu-streamlit-rag-project.git](https://github.com/Mohanad-Amin/ksu-streamlit-rag-project.git)
cd ksu-streamlit-rag-project

أنشئ وشغّل البيئة الافتراضية:

python3 -m venv venv
source venv/bin/activate

ثبّت المكتبات المطلوبة:

pip install -r requirements.txt

أنشئ ملف الأسرار:

أنشئ ملفًا جديدًا باسم .env في المجلد الرئيسي للمشروع.

أضف مفتاح OpenAI الخاص بك بداخله بالشكل التالي:

OPENAI_API_KEY="sk-..."

شغّل تطبيق Streamlit:

streamlit run streamlit_app.py

📂 هيكل المشروع
ksa-rag-project/
├── core/
│   ├── __init__.py      # يجعل المجلد حزمة بايثون
│   └── retriever.py     # يحتوي على منطق RAG والاتصال بـ OpenAI
├── data/
│   ├── crawled_chunks_final.xlsx  # قاعدة المعرفة النصية
│   └── ksa_engineering_embeddings.npy # المتجهات المحسوبة مسبقًا
├── .env                 # (محلي فقط) لتخزين مفتاح API
├── .gitignore           # لمنع رفع الملفات الحساسة
├── streamlit_app.py     # الكود الخاص بواجهة المستخدم
└── requirements.txt     # المكتبات اللازمة للمشروع
