# AML-BRATS

AML-BRATS is a project about segmentation models, with a focus on the tumor segmentation in slices of MRI scans. It provides a UNet model and includes a Streamlit demo interface for interactive use.

The API is deployed at https://amlbrats.muslii.top/. The demo can be accessed at https://frontend-aml.onrender.com/.

## Instructions on how to launch the API locally

To clonde the repository:

`git clone https://github.com/muslii4/AML-BRATS.git`

`cd AML-BRATS`

In the AML-BRATS directory, in the terminal, use the following command to launch the API:

    uv run main.py

If you do not have uv installed, then run:

```
pip install -r requirements.txt

python main.py
```

The API runs locally, open your browser at http://127.0.0.1:8000/docs.

To get the app restarted:

    uvicorn api:app --reload

### To get the demo running, in another terminal, at the same time with the API running:

    streamlit run streamlit.py

## Features
* Users do not need to perform any special preprocessing steps
* Both the API and demo have instructions about how to be used to get predictions, step by step
* The prediction mask may be saved by the users in '.png' format
