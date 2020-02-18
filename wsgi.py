from app import app

print('wsgi:asd')

if __name__ == "__main__":
    print('wsgi:pre run')
    app.run()
    print('wsdi:post run')