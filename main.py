from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

# Initialize the FastAPI app
app = FastAPI()

# Load the YOLOv8 model (replace 'best.pt' with your model's path)
model = YOLO('best.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Make predictions using YOLO model
    results = model(image)

    # Initialize variables to track the highest confidence prediction
    best_prediction_name = None
    max_confidence = 0

    # Extract and process data from YOLOv8 results
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf.item())
            if confidence > max_confidence:
                max_confidence = confidence
                best_prediction_name = result.names[int(box.cls.item())]  # Class name from YOLO model

    # Return the name of the most confident prediction as JSON response
    return JSONResponse(content={"most_confident_class_name": best_prediction_name})

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
