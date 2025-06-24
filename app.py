from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, torch, torch.nn as nn, torch.nn.functional as F

# 1) Define your request schema
class WineRequest(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# 2) Rebuild your network
class WineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11,64)
        self.fc2 = nn.Linear(64,32)
        self.out = nn.Linear(32,2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# 3) Load weights
model = WineNet()
model_file = os.path.join(os.path.dirname(__file__), "model.pth")
model.load_state_dict(torch.load(model_file, map_location="cpu"))
model.eval()

# 4) Create FastAPI app
app = FastAPI(title="Wine Quality Classifier")

# 5) Serve the static directory at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 6) Root endpoint serves our webpage
@app.get("/")
def read_index():
    return FileResponse(os.path.join("static", "index.html"))

# 7) Prediction endpoint
@app.post("/predict")
def predict(data: WineRequest):
    x = torch.tensor([[ 
        data.fixed_acidity, data.volatile_acidity, data.citric_acid,
        data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
        data.total_sulfur_dioxide, data.density, data.pH,
        data.sulphates, data.alcohol
    ]], dtype=torch.float)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
        idx    = int(probs.argmax())
    return {
        "prediction": "good" if idx==1 else "not good",
        "probability": float(probs[idx])
    }

# 8) Run via Uvicorn
if __name__=="__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
