import subprocess
import json
import statistics
import base64

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

PREDICT = ROOT / "src" / "predict.py"
RUNS = ROOT / "runs" / "predict"

CONFIGS = [
    ("standard", []),
    ("tta", ["--tta"]),
    ("sahi", ["--sahi"]),
    ("tta_sahi", ["--tta", "--sahi"]),
]

def run_prediction(image):

    before = set(RUNS.iterdir()) if RUNS.exists() else set()

    for name, flags in CONFIGS:
        print(f"\n{name}")
        subprocess.run(
            ["python", str(PREDICT), "--image", str(image), *flags],
            check=True
        )
    after = set(RUNS.iterdir())

    return sorted(after - before)

def collect_results(run_dirs):

    data = []

    for run in run_dirs:

        img_dir = next(run.glob("img_*"))

        with open(img_dir / "results.json", encoding="utf8") as f:
            result = json.load(f)

        data.append({
            "name": run.name,
            "image": img_dir / "image_clean.jpg",
            "json": result
        })

    return data


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_html(results):

    order = {
        "standard": 0,
        "tta_sahi": 1,
        "tta": 2,
        "sahi": 3
    }

    def get_type(name):
        if "tta_sahi" in name:
            return "tta_sahi"
        if "standard" in name:
            return "standard"
        if "_tta_" in name:
            return "tta"
        return "sahi"


    results = sorted(
        results,
        key=lambda x: order[get_type(x["name"])]
    )


    rows = []
    images = []

    names = []
    scores = []
    times = []
    counts = []
    confs = []


    for r in results:

        j = r["json"]
        detections = j["detections"]

        confidence = [
            d["confidence"]
            for d in detections
        ]

        bbox = [
            d["bbox"][2] * d["bbox"][3]
            for d in detections
        ]


        avg_conf = (
            sum(confidence) / len(confidence)
            if confidence else 0
        )

        max_conf = max(confidence) if confidence else 0
        min_conf = min(confidence) if confidence else 0

        avg_bbox = (
            sum(bbox) / len(bbox)
            if bbox else 0
        )


        if "tta_sahi" in r["name"]:
            config = "TTA + SAHI"
        elif "standard" in r["name"]:
            config = "Standard"
        elif "_tta_" in r["name"]:
            config = "TTA"
        else:
            config = "SAHI"


        names.append(config)
        scores.append(j["score"])
        times.append(j["time"])
        counts.append(len(detections))
        confs.append(round(avg_conf, 3))


        rows.append(f"""
        <tr>
            <td>{config}</td>
            <td>{j['score']}</td>
            <td>{j['time']}s</td>
            <td>{len(detections)}</td>
            <td>{avg_conf:.3f}</td>
            <td>{max_conf:.3f}</td>
            <td>{min_conf:.3f}</td>
            <td>{avg_bbox:.0f}</td>
        </tr>
        """)


        img64 = image_to_base64(r["image"])

        images.append(f"""
        <div class="card">
            <b>{config}</b>
            <img src="data:image/jpeg;base64,{img64}">
        </div>
        """)



    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>YOLO Acne Comparison</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


<style>

body {{
    font-family: Arial, sans-serif;
    margin: 15px;
    font-size: 14px;
}}

h1, h2 {{
    margin: 10px 0;
}}

h3 {{
    margin: 5px 0;
    font-size: 14px;
}}


.grid {{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:12px;
}}


.card {{
    border:1px solid #ddd;
    padding:8px;
}}


.card img {{
    width:100%;
    display:block;
    margin-top:5px;
}}


table {{
    border-collapse:collapse;
    width:100%;
    margin-top:15px;
    font-size:13px;
}}


th, td {{
    border:1px solid #ddd;
    padding:5px;
    text-align:center;
}}


.charts {{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:15px;
    margin-top:15px;
}}


.chart {{
    height:260px;
}}


</style>

</head>


<body>


<h1>Acne detection comparison</h1>

Generated:
{datetime.now():%Y-%m-%d %H:%M}


<h2>Images</h2>


<div class="grid">

{"".join(images)}

</div>



<h2>Metrics</h2>


<table>

<tr>
<th>Config</th>
<th>Score</th>
<th>Time</th>
<th>Detections</th>
<th>Avg conf</th>
<th>Max conf</th>
<th>Min conf</th>
<th>Avg bbox</th>
</tr>


{"".join(rows)}


</table>



<h2>Charts</h2>


<div class="charts">


<div class="chart">
<h3>Number of detections</h3>
<canvas id="count"></canvas>
</div>


<div class="chart">
<h3>Average confidence</h3>
<canvas id="conf"></canvas>
</div>


<div class="chart">
<h3>Inference time</h3>
<canvas id="time"></canvas>
</div>


<div class="chart">
<h3>Acne score</h3>
<canvas id="score"></canvas>
</div>


</div>




<script>


const labels = {names};

const navy = "#0b1f3a";


function makeChart(id, title, data) {{

    new Chart(
        document.getElementById(id),
        {{

            type:"bar",

            data: {{

                labels: labels,

                datasets:[{{

                    label:title,

                    data:data,

                    backgroundColor:navy

                }}]

            }},

            options:{{

                responsive:true,

                maintainAspectRatio:false,

                plugins:{{

                    legend:{{

                        display:true,

                        position:"bottom"

                    }}

                }}

            }}

        }}
    );

}}



makeChart(
    "count",
    "Detections",
    {counts}
);


makeChart(
    "conf",
    "Average confidence",
    {confs}
);


makeChart(
    "time",
    "Inference time [s]",
    {times}
);


makeChart(
    "score",
    "Acne score",
    {scores}
);



</script>


</body>

</html>
"""


    path = ROOT / "notebooks" / "predictions_comparison.html"


    with open(path, "w", encoding="utf8") as f:
        f.write(html)


    print(f"Saved: {path}")


def main():

    image = Path("Models/yolo/images/raw/levle1_271.jpg")
    run_dirs = run_prediction(image)
    results = collect_results(run_dirs)

    make_html(results)

if __name__ == "__main__":
    main()