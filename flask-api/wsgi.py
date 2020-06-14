from app import app

if __name__ == "__main__":
    app.run(port=5000) # Will run gunicorn on port 5000 if nothing is specify in config file
