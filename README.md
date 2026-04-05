# road-segmentation

This application automatically identifies geometric shapes of road pathways and segments them using a trained neural network.

## How to run

### Backend
1. Move to the backend folder 
```bash
cd backend
```
2. Install the python dependencies
```bash
pip install -r requirements.txt
```
3. Run the uvicorn server on port 5000
```bash
uvicorn main:app --reload --port 5000
```

### Frontend
1. Move to the frontend folder
```bash
cd frontend
```
2. Install the javascript dependencies
```bash
npm install
```
3. Run the next.js server
```bash
npm run dev
```
