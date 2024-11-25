set version_num=%1
set image_name=budget-rag
ECHO "Building budget-rag version %version_num%"
docker build -t %image_name%:%version_num% .
