from flask import Flask, render_template, request, jsonify
import cv2
import torch
import numpy as np
import time
import os
from train import UNet

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# Load model
model = UNet()
model.load_state_dict(torch.load("tree_model.pth", map_location="cpu"))
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", result=False)

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", result=False)

        filename = f"{int(time.time())}.jpg"
        filepath = os.path.join("static", filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        h, w = image.shape[:2]

        # Preprocess
        img = cv2.resize(image, (256, 256)) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        with torch.no_grad():
            output = model(img)
            output = torch.sigmoid(output)[0][0].numpy()

        mask = (output > 0.3).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                cnt = cnt.squeeze()
                if len(cnt.shape) == 2:
                    polygons.append(cnt.tolist())

        return render_template(
            "index.html",
            result=True,
            image_file=filename,
            polygons=polygons,
            width=w,
            height=h
        )

    return render_template("index.html", result=False)


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()

    polygons = data["polygons"]
    image_file = data["image_file"]

    img_path = os.path.join("static", image_file)
    image = cv2.imread(img_path)

    for poly in polygons:
        pts = np.array(
            [[int(round(p["x"])), int(round(p["y"]))] for p in poly],
            np.int32
        )
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(image, [pts], True, (0, 0, 255), 3)

    out_name = f"edited_{int(time.time())}.jpg"
    cv2.imwrite(os.path.join("static", out_name), image)

    return jsonify({"status": "success", "image": out_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)