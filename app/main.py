from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Medical Evidence API", version="0.1.0", description="Auditable retrieval of medical evidence with strict abstention.")
app.include_router(router)
