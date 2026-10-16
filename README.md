# Face Recognition Attendance API

A FastAPI backend for the Flutter attendance app. The phone captures camera
frames and sends them over HTTP. The server does face detection, liveness
(anti-spoofing), recognition and attendance records.

**Flow:** 1. registration, then 2. detection, then 3. attendance, for one
person or for a group in one frame.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export API_KEY=change-me          # every /api call must send X-API-Key: change-me
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Interactive docs, with every request and response field, are at
`http://<host>:8000/docs`. Run a single worker, because sessions live in
memory. Put HTTPS in front of it in production.

## Test with your webcam

`test_client/index.html` is a browser page that uses your webcam and calls
the API the same way the Flutter app does. With the API running, start a
second terminal:

```bash
python -m http.server 8080 -d test_client
```

Open `http://localhost:8080/portal.html` for the role-based test portal.
Sign in as Admin to register, check, run detection tests, mark attendance,
list and delete people, and see attendance records with photos. Sign in as
Employee, pick your name, and only attendance is available, as a 1:1 check
against that employee. Roles are enforced by the page only, not by the API.

`http://localhost:8080` is the simpler tester with every mode on one screen.
Allow the camera when the browser asks. Pick a mode, then press
Start. Check registered, Register, Detect face, Liveness, Verify and
Attendance each call their endpoint and show the message, spoof flag, scores
and face box. Browsers only allow the camera on `localhost` or HTTPS.

## Project layout

```text
app/
  main.py                 FastAPI app, startup, router wiring
  core/config.py          settings: defaults < config.toml < environment
  core/runtime.py         ONNX / OpenCV thread budget
  ml/face_engine.py       face detection + recognition (insightface buffalo_sc)
  ml/antispoof.py         liveness model (MiniFASNet ensemble)
  db/store.py             SQLite: people, embeddings, face images, attendance
  services/analysis.py    the gate sequence every frame goes through
  services/pose.py        head pose from face keypoints
  services/registration.py  guided center / left / right registration
  services/attendance.py  multi-frame attendance check
  services/sessions.py    in-memory session registry
  schemas/                request and response models
  api/deps.py             shared dependencies: models, image upload, API key
  api/routes/             one router per step: registration, detection,
                          attendance, people, health
test_client/portal.html   webcam test portal with Admin and Employee roles
test_client/index.html    simpler webcam tester
models/                   liveness weights
data/                     the database (biometric data, never commit)
config.toml               every threshold, commented
```

## What every frame is checked for

`app/services/analysis.py` runs these gates in order. The first failure
decides the response `status` and `message`.

| status | message shown to the user |
|---|---|
| `image_too_small` | Image too small - send the full camera frame |
| `no_face` | No face detected - look at the camera |
| `multiple_faces` | Only one person should be in view |
| `too_close` | Please keep distance from the camera |
| `too_far` | Please move a little closer to the camera |
| `off_centre` | Keep your face in the centre of the frame |
| `low_detection` | Face not clear - improve lighting and hold still |
| `blurry` | Image is blurry - hold the phone still |
| `spoof` | Spoofing detected - use your live face, not a photo or screen |
| `liveness_low` | Liveness not confirmed - face the light and hold still |
| `not_recognized` | Face not recognized - please register first |
| `ok` | Live face verified |
| `attendance_marked` | Attendance marked for NAME |

Once a person's attendance is marked, their face reports `attendance_marked`
instead of a problem such as `too_close` or `liveness_low`. In a group session
this holds for every later frame too. A spoof is always reported as `spoof`.

Every response carries `spoof`, `liveness_score`, `detection_score` and
`face_ratio`. When the face is too close or too far, only the distance
message is returned and the scores are `null`. The liveness model needs to
see around the face to catch a screen or paper edge, so its score is not
trustworthy at the wrong distance.

## API

Image uploads are `multipart/form-data` with the file in the `image` field.
Send the full camera frame, not a face crop, at least 480 px on the short
side.

### 1. Registration

| Call | Body |
|---|---|
| `POST /api/enroll/check` | file `image`: is this face already registered? |
| `POST /api/enroll/start` | JSON `{"name", "employee_id", "employee_number"}` |
| `POST /api/enroll/frame` | form `session_id`, file `image` |
| `GET /api/enroll/status/{session_id}` | none |
| `POST /api/enroll/cancel` | JSON `{"session_id"}` |

Send frames 3 to 4 times a second. Show `instruction`, and `frame.problem`
when it is set. The person looks straight, then turns left, then right, and
5 samples are kept per pose. `state` ends as `complete`, `timeout`,
`cancelled` or `failed`.

Registration allows the face closer than attendance, up to 60% of the
image's shorter side instead of 35%, because a big, sharp face gives better
samples. Set it with `[enrollment] min_face_ratio` and `max_face_ratio`.

**Already-registered people are detected.** After the first 3 frontal
samples, the face is compared with everyone registered. On a match, `state`
becomes `already_registered`, nothing is saved, and `existing_person` holds
their `name`, `employee_id`, `employee_number`, `created_at`, `samples`,
`has_image` and `similarity`. `POST /api/enroll/check` does the same with one
image before a session starts. It returns `already_registered` as true,
false, or null when the image could not be checked, with `person` and
`message`. Starting with an `employee_id` or `employee_number` that is
already in use returns 409, with the same details. To re-register someone
on purpose, delete them first. The match threshold is
`[enrollment] duplicate_sim` in `config.toml`.

### 2. Detection, one function per endpoint

| Call | What it checks |
|---|---|
| `POST /api/detect/face` | face, distance, centring, detection score, sharpness |
| `POST /api/detect/liveness` | the above, plus live or spoof |
| `POST /api/verify` | the above, plus who it is |
| `POST /api/verify/group` | every face in the frame, each checked on its own |

`/api/verify` also returns `reason`, `label`, `score` and `liveness`, the
field names the first app version used.

### 3. Attendance

| Call | Body |
|---|---|
| `POST /api/attendance/start` | JSON `{"employee_id"}`, optional, for a 1:1 check |
| `POST /api/attendance/frame` | form `session_id`, file `image` |
| `GET /api/attendance/status/{session_id}` | none |
| `POST /api/attendance/cancel` | JSON `{"session_id"}` |
| `GET /api/attendance?day=YYYY-MM-DD&employee_id=` | records |
| `GET /api/attendance/{id}` and `/{id}/image` | one record, its face photo |

A session needs 1 frontal frame that passes the strict bars, then a slight
head turn to either side (left or right). Set
`challenge_sides_required = 2` in `[attendance]` to require both. Identity is checked
on every frame, so there is no return to face-front. Several spoof frames
reject the session. A second mark within 5
minutes returns the first record with `already_marked: true`.

### 3b. Group attendance

| Call | Body |
|---|---|
| `POST /api/attendance/group/start` | none |
| `POST /api/attendance/group/frame` | form `session_id`, file `image` |
| `GET /api/attendance/group/status/{session_id}` | none |
| `POST /api/attendance/group/finish` | JSON `{"session_id"}`, ends it and lists who was marked |
| `POST /api/attendance/group/cancel` | JSON `{"session_id"}` |

Every face in each frame is checked on its own, at the same strict liveness,
detection and match bars as single attendance. Each frame response lists
every face, left to right, with its status, message, scores, name and
progress. A person is marked once their own face passes on 5 frames over at
least a second. Others in the frame, including a spoof, do not hold them up,
and a spoof face is never marked. The same person appearing twice in one
frame counts for neither, because one of them is a photo or screen.

Group attendance has no head-turn challenge, because it cannot be aimed at
one person in a group. Group marks are stored with `mode` set to `group`, so
they can be told apart. Distance limits are in `[group_attendance]`: people
may stand further back, down to a face 80 px wide.

### People

`GET /api/people`, `GET /api/people/{employee_id}`,
`GET /api/people/{employee_id}/image`, `DELETE /api/people/{name}`.

## Tuning

All thresholds are in `config.toml`. The liveness values (`threshold` 0.48,
`min_liveness` 0.98) and `min_det_score` 0.75 should be checked on your
phones. Log `liveness_score` and `detection_score` from real users and from
photos or videos shown on another phone, then set the thresholds between
the two groups.

## Licences

The insightface `buffalo_sc` models are licensed for non-commercial research.
Check their terms, or buy a licence, before a commercial release. The
liveness weights are Apache-2.0, from minivision-ai via yakhyo's ONNX exports.
Registered faces are biometric data: collect consent, and delete people who
leave with `DELETE /api/people/{name}`.
