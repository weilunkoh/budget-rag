set version_num=%1
set image_name=budget-rag
ECHO "Running docker container for budget-rag version %version_num%"
docker run -d -p 8501:8501 --rm --name budget-rag --env-file .env %image_name%:%version_num% 