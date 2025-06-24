# 🍷 Wine Quality Classifier API

An interactive FastAPI microservice that classifies red wine samples as **good** or **not good** based on physicochemical properties, complete with a browser-based UI for real-time predictions.

---

## 🚀 Features

- **REST API** (`/predict`): Accepts JSON input of 11 features, returns class label and confidence score.  
- **Web Interface**: A simple, clean HTML form served at `/` to input feature values and display results instantly.  
- **Model**: PyTorch-based fully connected network (`WineNet`) trained on the UCI Wine Quality dataset.  
- **Easy Deployment**: Ready for Render (or Heroku), with a `Procfile` and minimal configuration.

---

## 📦 Technologies Used

- **Python 3.9+**  
- **FastAPI** for the web service and static file serving  
- **Uvicorn** as the ASGI server  
- **PyTorch** for model inference  
- **Pydantic** for input validation  
- **HTML/CSS/JS** for the front-end  

---

## 📂 Repository Structure

```bash
wine-quality-api/
├── app.py               # FastAPI application
├── model.pth            # Trained PyTorch model weights
├── requirements.txt     # Python dependencies
├── Procfile             # Render start command
└── static/
    └── index.html       # Interactive front-end
