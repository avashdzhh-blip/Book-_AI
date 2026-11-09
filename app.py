
import streamlit as st
import fitz
import pytesseract
import numpy as np
import re

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Book AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book AI")
st.caption("دستیار هوشمند کتاب‌های شما")


def normalize_text(text):
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    progress = st.progress(0)
    status = st.empty()

    total = len(doc)

    for i, page in enumerate(doc):
        text = page.get_text("text")

        # OCR برای صفحات اسکن شده
        if len(text.strip()) < 40:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            text = pytesseract.image_to_string(
                image,
                lang="fas+eng",
                config="--psm 6"
            )

        text = normalize_text(text)

        if text:
            pages.append({
                "page": i + 1,
                "text": text
            })

        progress.progress((i + 1) / total)
        status.write(f"در حال خواندن صفحه {i + 1} از {total}")

    progress.empty()
    status.empty()
    doc.close()

    return pages


def create_chunks(pages, chunk_size=900, overlap=120):
    chunks = []

    for item in pages:
        text = item["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            part = text[start:end].strip()

            if len(part) > 30:
                chunks.append({
                    "page": item["page"],
                    "text": part
                })

            start += chunk_size - overlap

    return chunks


def find_answer(question, chunks, vectorizer, matrix):
    question = normalize_text(question)

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        matrix
    ).flatten()

    best_indexes = np.argsort(scores)[::-1]

    results = []

    for index in best_indexes[:5]:
        if scores[index] <= 0:
            continue

        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "matrix" not in st.session_state:
    st.session_state.matrix = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None


st.subheader("📖 آپلود کتاب")

uploaded_file = st.file_uploader(
    "فایل PDF کتاب را انتخاب کنید",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("پردازش کتاب", type="primary"):

        try:
            with st.spinner("در حال پردازش کتاب..."):

                pdf_bytes = uploaded_file.getvalue()

                pages = extract_pdf(pdf_bytes)

                if not pages:
                    st.error("متنی از کتاب استخراج نشد.")
                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:
                    st.error("متن کافی برای جست‌وجو پیدا نشد.")
                    st.stop()

                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=50000,
                    sublinear_tf=True
                )

                matrix = vectorizer.fit_transform(
                    [item["text"] for item in chunks]
                )

                st.session_state.chunks = chunks
                st.session_state.vectorizer = vectorizer
                st.session_state.matrix = matrix
                st.session_state.book_name = uploaded_file.name

            st.success("کتاب با موفقیت پردازش شد! 🎉")
            st.write(f"تعداد صفحات خوانده‌شده: {len(pages)}")
            st.write(f"تعداد بخش‌های آماده جست‌وجو: {len(chunks)}")

        except Exception as e:
            st.error("هنگام پردازش کتاب خطایی رخ داد.")
            st.exception(e)


st.divider()

st.subheader("💬 از کتاب سؤال بپرس")

if st.session_state.chunks is None:

    st.info("ابتدا یک کتاب PDF آپلود و پردازش کنید.")

else:

    st.caption(f"کتاب فعلی: {st.session_state.book_name}")

    question = st.text_input(
        "سؤال خود را بنویسید:",
        placeholder="مثلاً فتوسنتز چیست؟"
    )

    if st.button("پیدا کردن پاسخ"):

        if not question.strip():
            st.warning("لطفاً سؤال خود را بنویسید.")

        else:
            results = find_answer(
                question,
                st.session_state.chunks,
                st.session_state.vectorizer,
                st.session_state.matrix
            )

            if not results:
                st.warning(
                    "قسمت مرتبطی در متن کتاب پیدا نشد."
                )

            else:
                st.subheader("📚 پاسخ‌های پیدا شده از کتاب")

                for i, result in enumerate(results, 1):

                    with st.expander(
                        f"نتیجه {i} - صفحه {result['page']}",
                        expanded=(i == 1)
                    ):

                        st.write(result["text"])

                        st.caption(
                            f"صفحه {result['page']}"
                        )

                st.info(
                    "این بخش‌ها مستقیماً از متن کتاب استخراج شده‌اند. "
                    "برای پاسخ دقیق‌تر، متن صفحات نمایش‌داده‌شده را بررسی کنید."
                )
                

              
