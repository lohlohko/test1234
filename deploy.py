from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_numeric
import tensorflow as tf
import joblib
import magic
import fitz

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

vectorizer = joblib.load('tfidf_vectorizer_fin.joblib')
model = tf.keras.models.load_model('calculate_similarity_model_fin_V10.h5')

# ===========================Routing=======================
@app.get("/")
def read_root():
    return {"message": "Success get"}

@app.post("/predict")
async def predict(cv: UploadFile = File(...), job_description: UploadFile = File(...)):
    try:
        new_cv_text = await read_and_preprocess_pdf(cv)
        new_job_description = await read_and_preprocess_pdf(job_description)

        # Use the loaded vectorizer to transform the text
        new_cv_vector = vectorizer.transform([new_cv_text]).toarray()
        new_job_vector = vectorizer.transform([new_job_description]).toarray()

        prediction = model.predict({
            'cv_input': new_cv_vector,
            'job_input': new_job_vector
        })

        rounded_similarity_score = round(prediction[0][0] * 100, 2)
        return {"similarity_score": rounded_similarity_score}

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ===========================Process=======================
async def read_and_preprocess_pdf(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="File not provided.")

    try:
        mime_type = magic.Magic()
        file_type = mime_type.from_buffer(await file.read(1024))

        if 'pdf' in file_type.lower():
            pdf_document = fitz.open(stream=await file.read(), filetype="pdf")
            text = ''
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            pdf_document.close()
        else:
            text = (await file.read()).decode('utf-8')

        preprocessed_text = ' '.join(preprocess_string(text, [strip_tags, strip_numeric]))
        return preprocessed_text

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {file.filename}")
    except IsADirectoryError as e:
        raise HTTPException(status_code=400, detail=f"Expected a file, but got a directory: {file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
