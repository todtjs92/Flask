from flask import Flask ,jsonify , request

app = Flask(__name__)
app.users = {}
app.id_count = 1

from flask.json import JSONEncoder

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

        return JSONEncoder.default(self,obj)

app.json_encoder = CustomJSONEncoder


@app.route("/ping", methods = ['GET'])
def ping():
    return "pong"



@app.route("/sign-up",methods= ['POST'])
def sign_up():
    new_user = request.json
    new_user["id"] = app.id_count
    app.users[app.id_count] = new_user
    app.id_count = app.id_count + 1

    return jsonify(new_user)


@app.route('/follow', methods = ['POST'])
def follow():
    payload = request.json
    user_id = int(payload['id'])
    user_id_to_follow = int(payload['follow'])

    if  user_id not in app.users or user_id_to_follow not in app.users:
        return "The user is not exist" , 400

    user = app.users[user_id]
    user.setdefault('follow',set()).add(user_id_to_follow)

    return jsonify(user)

