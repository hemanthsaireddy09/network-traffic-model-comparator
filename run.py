from app import app

# This is only used when running locally. On PythonAnywhere, the WSGI file handles this
if __name__ == '__main__':
    # Only use debug mode in development
    app.run(host='0.0.0.0', debug=False)