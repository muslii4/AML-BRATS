import io

import h5py
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from AML_BRATS.data.data_loading import preprocess
from AML_BRATS.models.unet import UNet

MODEL_PATH = "models/UNET_HYD_25EPOCHS_adam_BNORM_LR0.0001_WD0.01_bce1_NOAUG_BS64_FOLD1_final.pkl"
MODEL_BATCH_NORM = "BNORM" in MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3, MODEL_BATCH_NORM)
model.load_state_dict(
    torch.load(MODEL_PATH, weights_only=True, map_location=device)
)
model.eval()


class HealthResponse(BaseModel):
    message: str


def load_h5_input(file: UploadFile = File(...)) -> np.ndarray:
    """
    Extracts data from a h5 file.

    If the file is not h5, raise error.
    """
    try:
        with h5py.File(file.file, "r") as f:
            key = list(f.keys())[0]
            if len(key) == 0:
                raise HTTPException(
                    status_code=400, detail="No scan found in this file"
                )
            input_data = np.array(f[key])
            return input_data
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid '.h5' file.")
    except Exception as e:
        raise HTTPException(
            status_code=415, detail=f"Invalid '.h5' file: {str(e)}"
        )


def segmentation_to_png(
    prediction: torch.Tensor,
    background: np.ndarray,
    contrast_channel: int = 0,
    threshold: float = 0.5,
):
    prediction_arr: np.ndarray = prediction.detach().cpu().numpy()
    binary_mask = prediction_arr >= threshold

    mri_channel = background[:, :, contrast_channel]
    mri_min = mri_channel.min()
    mri_max = mri_channel.max()
    if mri_max > mri_min:
        mri_normalized = (
            (mri_channel - mri_min) / (mri_max - mri_min) * 255
        ).astype(np.uint8)
    else:
        mri_normalized = np.zeros_like(mri_channel, dtype=np.uint8)

    rgb = np.stack([mri_normalized, mri_normalized, mri_normalized], axis=2)

    rgb[binary_mask[0]] = [255, 0, 0]
    rgb[binary_mask[1]] = [0, 255, 0]
    rgb[binary_mask[2]] = [0, 0, 255]

    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


app = FastAPI(
    title="Brain Tumor Segmentation",
    description="""API for Brain Tumor Segmentation

    Our model performs segmentation on MRI scan slices. We trained (model info).

    It was trained on BraTS2020 dataset, which included MRI scan slices with segmentations performed manually by neuro-radiologists.

    User guide:
    1. Scroll to "Predict" section. 
    2. Click on "Try it out" button.
    3. Click on "Browse..." and select the scan file to be uploaded from your computer.
    4. Click on the blue button "Execute".
    5. Scroll to Response Body: you may visualize or save the '.png' file with the regions of the tumor from the uploaded scan.
    
    Important!
    This API is for educational and testing purposes only.
    This is not a diagnostic tool!

    Input:
    Upload a '.h5' file with a slice from a MRI scan.

    Output:
    PNG file with the FLAIR MRI contrast with the predicted tumor mask overlayed on top:
    - red, green, blue = regions of the tumor:
        * red - the necrotic and non-enhancing tumor core 
        * green - the peritumoral edema
        * blue - the GD-enhancing tumor
    - black = background
    - shades of gray = healthy tissue
    """,
    version="alpha",
)


@app.get("/", description="Root endpoint")
async def root():
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    description="Check if the API is running.",
)
async def health():
    return {"message": "API is running"}


@app.post(
    "/predict",
    summary="Segment MRI tumor regions",
    description=(
        "Upload a single MRI scan slice in HDF5 format (.h5). "
        "The model performs multi-class segmentation and returns a PNG mask "
        "with 3 color-coded tumor sub-regions overlaid on the scan. "
        "Blue - D-enhancing tumor, green - the peritumoral edema, red - necrotic and non-enhancing tumor core"
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "PNG segmentation mask with 3 labeled tumor regions.",
        },
        400: {
            "description": "No file provided, the file is not in the right format (.h5), or it does not contain proper data.",
        },
    },
)
async def predict(scan: UploadFile):
    if scan.filename is None:
        raise HTTPException(
            status_code=400, detail="Invalid or no file uploaded"
        )
    if not scan.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="File must be '.h5'")
    raw_input = load_h5_input(scan)
    input_data = preprocess(raw_input)
    input_tensor = torch.from_numpy(input_data).unsqueeze(0)
    input_tensor = input_tensor.permute(0, 3, 1, 2).float()

    with torch.no_grad():
        prediction = model(input_tensor).sigmoid()[0]

    user_image = segmentation_to_png(
        prediction, background=raw_input, contrast_channel=0
    )
    return StreamingResponse(user_image, media_type="image/png")


def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
