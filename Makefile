.PHONY: db backend frontend ingest all stop

db:
	docker compose up qdrant -d

ingest:
	source venv/bin/activate && python ingest.py

backend:
	source venv/bin/activate && cd backend && uvicorn main:app --reload --port 8000

frontend:
	source venv/bin/activate && cd frontend && streamlit run app.py

all:
	make db
	@echo "Starting backend..."
	source venv/bin/activate && cd backend && uvicorn main:app --reload --port 8000 &
	@echo "Starting frontend..."
	source venv/bin/activate && cd frontend && streamlit run app.py

stop:
	docker compose down
	pkill -f uvicorn || true
	pkill -f streamlit || true