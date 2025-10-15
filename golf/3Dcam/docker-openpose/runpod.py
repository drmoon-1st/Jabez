from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import os
import base64
import pyopenpose as op
import json
import numpy as np
import cv2
import traceback

app = Flask(__name__)

# OpenPose parameters
params = {
    "model_folder": "models/",
    "model_pose" : "BODY_25",
    "net_resolution": "-1x208",
    "output_resolution": "-1x-1",
    "num_gpu": 1,
    "num_gpu_start": 0,
    "disable_blending": False,
    "scale_number": 1,
    "scale_gap": 0.4,
    "render_threshold": 0.3,
    "number_people_max": 1,
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def OpenPoseImageProcessing(base64_image_string):
    """Process a base64 image with OpenPose and return keypoints (list) and the rendered image as base64.
    Returns (pose_keypoints_list, output_image_base64)
    pose_keypoints_list is an empty list if no people detected.
    """
    try:
        # Decode Base64 image string
        image_data = base64.b64decode(base64_image_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from base64")

        # Create OpenPose datum
        datum = op.Datum()
        datum.cvInputData = image

        # Process image
        datums = [datum]
        opWrapper.emplaceAndPop(op.VectorDatum(datums))

        # Extract keypoints safely
        if datum.poseKeypoints is None:
            pose_keypoints = []
        else:
            # poseKeypoints is a numpy array shaped (people, keypoints, 3)
            pose_keypoints = datum.poseKeypoints.tolist()

        # Get output image with skeletons
        output_image = datum.cvOutputData
        _, buffer = cv2.imencode('.jpg', output_image)
        output_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Debug print sizes
        print(f"OpenPose processed: people={len(pose_keypoints)}; output_image_bytes={len(buffer)}")

        return pose_keypoints, output_image_base64

    except Exception as e:
        print("Exception in OpenPoseImageProcessing:")
        traceback.print_exc()
        return [], ""


# The rest of your app logic (k-means etc.) can remain the same; minimal adjustments below to integrate returned keypoints.

# Keep previously defined functions that depend on the shape of openpose_json
# (I will include your existing functions that were referenced in your original file: kepfeldolgoz, dontes, etc.)

# --- BEGIN pasted helper functions (kept as in your original script) ---

# Note: these functions reference and mutate global variables; keep the same behavior.

pontok = []
klaszteradatKonyvtar = "klaszteradat"
kezdetimintak = []
centroidok = []
besorolas = []
valtozott = True
K = 5


def dontes():
    global centroidok
    global pontok
    global K

    almafa = []
    for a in pontok:
        almafa.append(a[0])
        almafa.append(a[1])

    mini = 0
    for i in range(K):
        if (tavolsagSzurt(centroidok[mini], almafa) > tavolsagSzurt(centroidok[i], almafa)):
            mini = i
    return mini


def kepfeldolgoz(openpose_json):
    global pontok
    pontok = []

    # openpose_json expected as list of people; we use first person if present
    if not openpose_json or len(openpose_json) == 0:
        print("No people detected by OpenPose")
        return

    kulcspontok = openpose_json[0]
    noembededlist = []
    for a in kulcspontok:
        noembededlist.append(a[0])
        noembededlist.append(a[1])
        noembededlist.append(a[2])
    kulcspontok = noembededlist

    print("Kulcspontok: " + str(kulcspontok))
    # 2 adatpont generálása, valószínűséget elhagyom
    i = 0
    for szam in kulcspontok:
        if i % 3 == 2:
            if (kulcspontok[i] <= 0.3): #ha kicsi a valószínűsége, akkor 0,0
                pontok.append([0, 0])
            else:
                pontok.append([kulcspontok[i - 2], kulcspontok[i - 1]])
        i = i + 1

    # origó a 1.ik pont -> ennek megfelelően pontok transzformálása
    origoX = pontok[1][0] if len(pontok) > 1 else 0
    origoY = pontok[1][1] if len(pontok) > 1 else 0
    for a in pontok:
        if ((a[0] != 0) and (a[1] != 1)):
            a[0] = a[0] - origoX
            a[1] = a[1] - origoY

    # Normalizálás---------
    maxY = -1e9
    maxX = -1e9
    minX = 1e9
    minY = 1e9
    for a in pontok:
        maxX = max(maxX, a[0])
        minX = min(minX, a[0])
        maxY = max(maxY, a[1])
        minY = min(minY, a[1])

    # Normalizálás x as képpé-------------
    # Guard against division by zero
    dx = (maxX - minX) if (maxX - minX) != 0 else 1.0
    dy = (maxY - minY) if (maxY - minY) != 0 else 1.0
    xkicsinyites = 1.0 / dx
    ykicsinyites = 1.0 / dy
    for a in pontok:
        a[0] = a[0] * xkicsinyites
        a[1] = a[1] * ykicsinyites


def tavolsagSzurt(a, b):
    tav = 0
    sulyozotttSzamitas = [1, 1, 3, 40, 50, 3, 40, 50]
    for x in range(50):
        if ((b[x] != 0) and (x < 16)):
            tav = tav + ((a[x] - b[x]) * sulyozotttSzamitas[int(x/2)]) ** 2
    return (tav) ** (1/2)


def tavolsag(a, b):
    tav = 0
    for x in range(50):
        if ((b[x] != 0)):
            tav = tav + (a[x] - b[x]) ** 2
    return tav ** (1 / 2)


def besorol():
    global besorolas
    global kezdetimintak
    global centroidok

    for i in range(len(kezdetimintak)):
        legjobbj = 0
        for j in range(len(centroidok)):
            if (tavolsag(centroidok[legjobbj], kezdetimintak[i]) > (tavolsag(centroidok[j], kezdetimintak[i]))):
                legjobbj = j
        besorolas[i] = legjobbj


def ujcentroid():
    global besorolas
    global kezdetimintak
    global centroidok
    global valtozott
    valtozott = False

    for i in range(len(centroidok)):
        aktujatlag = [0] * 50
        db = 0
        for j in range(len(kezdetimintak)):
            if (besorolas[j] == i):
                for k in range(50):
                    aktujatlag[k] = aktujatlag[k] + kezdetimintak[j][k]
                db = db + 1

        if db == 0:
            db = 1

        for k in range(50):
            aktujatlag[k] = aktujatlag[k] / db

        if (aktujatlag != centroidok[i]):
            valtozott = True
        centroidok[i] = aktujatlag


def kezdetimintakolvas():
    global kezdetimintak
    global besorolas
    global klaszteradatKonyvtar

    for path in (os.listdir(klaszteradatKonyvtar)):
        if os.path.isfile(os.path.join(klaszteradatKonyvtar, path)):
            print(klaszteradatKonyvtar + path)
            aktfile = []
            file = open(klaszteradatKonyvtar + path, "r")
            for i in range(50):
                aktfile.append(float(file.readline().strip()))
            file.close()
            kezdetimintak.append(aktfile)
            besorolas.append(0)


def EleresiUtBeallito():
    global klaszteradatKonyvtar
    workdir = os.getcwd()
    rootdir = os.path.dirname(workdir)
    klaszteradatKonyvtar = rootdir + "/" + klaszteradatKonyvtar + "/"
    print("RootDir: " + rootdir)
    print("KlaszteradKonyvtar: " + klaszteradatKonyvtar)

# Initialize clustering data the same way you had it in your original file.
EleresiUtBeallito()
try:
    kezdetimintakolvas()
    for i in range(K):
        centroidok.append(kezdetimintak[int(((len(kezdetimintak))/K)*i)])

    while valtozott:
        besorol()
        ujcentroid()
    print("Klaszter kész.")
except Exception:
    print("Warning: failed to initialize clustering (maybe klaszteradat folder missing). Continuing...")
    traceback.print_exc()

# --- END helpers ---


@app.route('/openpose_predict', methods=['POST'])
def flask_get_image():
    try:
        if not request.is_json:
            return jsonify({'error': 'Expected JSON body'}), 400

        data = request.get_json()
        if 'img' not in data:
            return jsonify({'error': 'Missing image data (img)'}), 400
        img_base64 = data['img']
        turbo_without_skeleton = data.get('turbo_without_skeleton', True)

        pose_json, img_base64_rendered = OpenPoseImageProcessing(img_base64)

        # ensure pose_json is a proper list (empty list if no detections)
        if pose_json is None:
            people_list = []
        else:
            people_list = pose_json

        # K-means klaszterezés (기존 로직)
        kepfeldolgoz(people_list)
        pose_id = dontes()
        print("Képfeldolgozás kész.")

        # Respond: include keypoints so clients can use them immediately.
        # Keep pose_id if you still want to provide it.
        # Normalize and return people and a compatibility field pose_keypoints_2d
        print(f"DEBUG: people_list len = {len(people_list)}")
        if people_list:
            try:
                print(f"DEBUG: first person sample: {people_list[0][:6]}")
            except Exception:
                pass

        response_payload = {
            'message': 'OK',
            'pose_id': pose_id,
            'people': people_list,
            'pose_keypoints_2d': people_list[0] if people_list else [],
            'openposeimg': img_base64_rendered
        }

        if turbo_without_skeleton:
            response_payload['openposeimg'] = ""

        return jsonify(response_payload), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 19030))
    app.run(host='0.0.0.0', port=port)
