
import streamlit as st
import fitz
import pytesseract
import re

from PIL import Image
from rank_bm25 import BM25Okapi


st.set_page_config(
    page_title="Book AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book AI")
st.caption("دستیار هوشمند کتاب‌های شما")


# یکسان‌سازی متن فارسی
def normalize_text(text):
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("ۀ", "ه")
    text = text.replace("ة", "ه")
    text = text.replace("\u200c", " ")
    text = re.sub(r"[ـ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# تبدیل متن به کلمات
def tokenize(text):
    text = normalize_text(text).lower()
    return re.findall(r"[a-zA-Z0-9آ-ی]+", text)


# استخراج متن PDF
def extract_pdf(pdf_bytes):
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []
    progress = st.progress(0)
    status = st.empty()

    total = len(doc)

    for i, page in enumerate(doc):
        text = page.get_text("text")

        # OCR فقط برای صفحات بدون متن کافی
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
        status.write(
            f"در حال خواندن صفحه {i + 1} از {total}"
        )

    progress.empty()
    status.empty()
    doc.close()

    return pages


# تقسیم متن به بخش‌های کوچک‌تر
def create_chunks(pages):
    chunks = []

    for item in pages:
        sentences = re.split(
            r"(?<=[.!؟?])\s+",
            item["text"]
        )

        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(current) + len(sentence) < 700:
                current += " " + sentence
            else:
                if len(current.strip()) > 20:
                    chunks.append({
                        "page": item["page"],
                        "text": current.strip()
                    })

                current = sentence

        if len(current.strip()) > 20:
            chunks.append({
                "page": item["page"],
                "text": current.strip()
            })

    return chunks


# ساخت موتور جست‌وجو
def build_search(chunks):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


# پیدا کردن قسمت‌های مرتبط
def find_answer(question, chunks, search_engine):
    question_tokens = tokenize(question)

    if not question_tokens:
        return []

    scores = search_engine.get_scores(question_tokens)

    indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in indexes[:5]:
        if scores[index] <= 0:
            continue

        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


# اطلاعات کاربران در نشست جداگانه
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "search_engine" not in st.session_state:
    st.session_state.search_engine = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None


# آپلود کتاب
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

                search_engine = build_search(chunks)

                st.session_state.chunks = chunks
                st.session_state.search_engine = search_engine
                st.session_state.book_name = uploaded_file.name

            st.success("کتاب با موفقیت پردازش شد! 🎉")

            st.write(f"تعداد صفحات خوانده‌شده: {len(pages)}")
            st.write(f"تعداد بخش‌های کتاب: {len(chunks)}")

        except Exception as e:
            st.error("خطا هنگام پردازش کتاب:")
            st.exception(e)


st.divider()

# پرسش از کتاب
st.subheader("💬 از کتاب سؤال بپرس")

if st.session_state.chunks is None:

    st.info("ابتدا یک کتاب آپلود و پردازش کنید.")

else:

    st.caption(
        f"کتاب فعلی: {st.session_state.book_name}"
    )

    question = st.text_input(
        "سؤال خود را بنویسید:",
        placeholder="مثلاً یاخته چیست؟"
    )

    if st.button("پیدا کردن پاسخ"):

        if not question.strip():
            st.warning("لطفاً سؤال خود را بنویسید.")

        else:
            results = find_answer(
                question,
                st.session_state.chunks,
                st.session_state.search_engine
            )

            if not results:
                st.warning(
                    "قسمت مرتبطی در کتاب پیدا نشد."
                )

            else:
                st.subheader("📚 قسمت‌های مرتبط کتاب")

                for i, result in enumerate(results, 1):

                    with st.expander(
                        f"نتیجه {i} | صفحه {result['page']}",
                        expanded=(i == 1)
                    ):

                        st.write(result["text"])

                        st.caption(
                            f"شماره صفحه: {result['page']}"
                        )

                st.info(
                    "متن‌های بالا از کتاب استخراج شده‌اند. "
                    "پاسخ نهایی را با توجه به متن کتاب بررسی کنید."
                )
                

              
