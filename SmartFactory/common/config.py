import os

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(ROOT_DIR, 'app.db'))
SQLALCHEMY_TRACK_MODIFICATION = False



print(f"BASE_DIR={BASE_DIR}")
print(f"ROOT_DIR={ROOT_DIR}")
