import os
import cv2
import pickle
import base64
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

#uses dlib's pre-trained CNN
import face_recognition

from gaze_detector import GazeDetector

# App Setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gaze_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

ENCODINGS_FILE = 'face_encodings.pkl'

# Database Models
class Person(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False, unique=True)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)
    sessions      = db.relationship('GazeSession', backref='person', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        sessions = GazeSession.query.filter_by(person_id=self.id).all()
        avg_attention = (
            round(sum(s.attention_score for s in sessions) / len(sessions), 1)
            if sessions else 0
        )
        return {
            'id':             self.id,
            'name':           self.name,
            'registered_at':  self.registered_at.strftime('%Y-%m-%d %H:%M'),
            'total_sessions': len(sessions),
            'avg_attention':  avg_attention,
        }


class GazeSession(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    person_id       = db.Column(db.Integer, db.ForeignKey('person.id'), nullable=False)
    started_at      = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at        = db.Column(db.DateTime, nullable=True)
    total_frames    = db.Column(db.Integer, default=0)
    attentive_frames= db.Column(db.Integer, default=0)

    @property
    def attention_score(self):
        if self.total_frames == 0:
            return 0
        return round((self.attentive_frames / self.total_frames) * 100, 1)

    @property
    def duration_seconds(self):
        end = self.ended_at or datetime.utcnow()
        return max(0, int((end - self.started_at).total_seconds()))

    def to_dict(self):
        return {
            'id':               self.id,
            'person_id':        self.person_id,
            'person_name':      self.person.name if self.person else 'Unknown',
            'started_at':       self.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            'ended_at':         self.ended_at.strftime('%Y-%m-%d %H:%M:%S') if self.ended_at else None,
            'total_frames':     self.total_frames,
            'attentive_frames': self.attentive_frames,
            'attention_score':  self.attention_score,
            'duration_seconds': self.duration_seconds,
        }


# Face Encoding Store  (pickle file — no dataset needed)
def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}   # person_name: [encoding, ...]

def save_encodings(data):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

def identify_face(face_enc, threshold=0.55):
    """Return person name or None."""
    store = load_encodings()
    best_name, best_dist = None, threshold
    for name, encodings in store.items():
        dists = face_recognition.face_distance(encodings, face_enc)
        if len(dists) and dists.min() < best_dist:
            best_dist = dists.min()
            best_name = name
    return best_name


# Live Tracking State  (shared across threads)
class LiveState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.running        = False
        self.cap            = None
        self.gaze           = GazeDetector(max_faces=4)
        self.active_sessions= {}   # person_name -> session_id
        self.frame_counters = {}   # person_name -> {total, attentive}
        self.latest_frame   = None
        self.latest_stats   = []

state = LiveState()

# Metrics Store 
class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.frame_count = 0
        self.start_time = datetime.utcnow()

    def update(self, predicted, actual):
        # predicted: "Known" / "Unknown"
        # actual: "Known" / "Unknown"
        if predicted == "Known" and actual == "Known":
            self.tp += 1
        elif predicted == "Known" and actual == "Unknown":
            self.fp += 1
        elif predicted == "Unknown" and actual == "Known":
            self.fn += 1
        elif predicted == "Unknown" and actual == "Unknown":
            self.tn += 1

        self.frame_count += 1

    def compute(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall    = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        accuracy  = (self.tp + self.tn) / max(1, (self.tp + self.fp + self.fn + self.tn))
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "fps": round(fps, 2)
        }

metrics = Metrics()

# Background capture thread
def capture_loop():
    with app.app_context():
        state.cap = cv2.VideoCapture(0)
        state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while state.running:
            ret, frame = state.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = state.gaze.detect(frame)
            stats_out = []

            # Face recognition — run every 10 frames
            face_locs  = face_recognition.face_locations(frame, model='hog')
            face_encs  = face_recognition.face_encodings(frame, face_locs)

            # Map tracker object_ids → person names via proximity
            id_to_name = {}
            for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                name = identify_face(enc) or 'Unknown'
                cx   = (left + right) // 2
                cy   = (top  + bottom) // 2
                id_to_name[(cx, cy)] = name

                # Face box
                color = (0, 200, 100) if name != 'Unknown' else (100, 100, 100)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            def closest_name(nose_pt):
                if not id_to_name:
                    return 'Unknown'
                best, best_d = 'Unknown', float('inf')
                for (cx, cy), n in id_to_name.items():
                    d = ((nose_pt[0]-cx)**2 + (nose_pt[1]-cy)**2)**0.5
                    if d < best_d:
                        best_d, best = d, n
                return best if best_d < 120 else 'Unknown'

            for res in results:
                if res.get('status') == 'calibrating':
                    prog = res['progress']
                    tot  = res['total']
                    cv2.putText(frame, f'Calibrating... {prog}/{tot}',
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,220), 2)
                    stats_out.append({'status': 'calibrating', 'progress': prog, 'total': tot})
                    continue

                oid      = res['object_id']
                looking  = res['looking']
                conf     = res['confidence']
                nose_pt  = res['nose_pt']
                rolling  = res['rolling_attention']
                session  = res['session_attention']

                name = closest_name(nose_pt)
                # Registered person = Known, else Unknown
                actual = "Known" if name != "Unknown" else "Unknown"
                predicted = actual  

                metrics.update(predicted, actual)

                # Session management 
                if name != 'Unknown':
                    if name not in state.active_sessions:
                        # Open new DB session
                        new_sess = GazeSession(
                            person_id  = Person.query.filter_by(name=name).first().id,
                            started_at = datetime.utcnow()
                        )
                        db.session.add(new_sess)
                        db.session.commit()
                        state.active_sessions[name] = new_sess.id
                        state.frame_counters[name]  = {'total': 0, 'attentive': 0}

                    state.frame_counters[name]['total'] += 1
                    if looking:
                        state.frame_counters[name]['attentive'] += 1

                    # Flush counters to DB every 30 frames
                    if state.frame_counters[name]['total'] % 30 == 0:
                        sess_row = GazeSession.query.get(state.active_sessions[name])
                        if sess_row:
                            sess_row.total_frames     = state.frame_counters[name]['total']
                            sess_row.attentive_frames = state.frame_counters[name]['attentive']
                            db.session.commit()

                # ── Draw gaze overlay ───────────────────────────
                color = (0, 255, 80) if looking else (0, 60, 255)
                label = f"{'FOCUSED' if looking else 'AWAY'} {conf*100:.0f}% | R:{rolling:.0f}% S:{session:.0f}%"
                cv2.circle(frame, nose_pt, 6, color, -1)
                cv2.putText(frame, label, (nose_pt[0] - 60, nose_pt[1] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                stats_out.append({
                    'status':           'detected',
                    'object_id':        oid,
                    'name':             name,
                    'looking':          looking,
                    'confidence':       round(conf * 100),
                    'rolling_attention':round(rolling, 1),
                    'session_attention':round(session, 1),
                })

            with state.lock:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                state.latest_frame = buf.tobytes()
                state.latest_stats = stats_out

        # Cleanup 
        for name, sess_id in state.active_sessions.items():
            sess_row = GazeSession.query.get(sess_id)
            if sess_row and not sess_row.ended_at:
                if name in state.frame_counters:
                    sess_row.total_frames     = state.frame_counters[name]['total']
                    sess_row.attentive_frames = state.frame_counters[name]['attentive']
                sess_row.ended_at = datetime.utcnow()
        db.session.commit()

        state.active_sessions.clear()
        state.frame_counters.clear()

        if state.cap:
            state.cap.release()
        state.cap     = None
        state.running = False


def gen_frames():
    while state.running:
        with state.lock:
            frame = state.latest_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Routes — Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/track')
def track_page():
    return render_template('track.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')


# Routes — API

# Face Registration
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    name   = data.get('name', '').strip()
    images = data.get('images', [])   # list of base64 JPEG strings

    if not name:
        return jsonify({'error': 'Name is required'}), 400
    if len(images) < 3:
        return jsonify({'error': 'At least 3 face captures required'}), 400

    encodings = []
    errors = []
    for i, b64 in enumerate(images):
        try:
            # Strip data URL header 
            if ',' in b64:
                b64 = b64.split(',', 1)[1]

            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)

            # IMREAD_UNCHANGED preserves alpha channel so we can detect format
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                errors.append(f"Image {i}: failed to decode")
                continue

            # Normalise to 3-channel BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Scale up small frames so HOG can find the face
            h, w = img.shape[:2]
            if max(h, w) < 600:
                scale = 600 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            # Convert to contiguous uint8 RGB array for face_recognition
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.uint8)

            locs = face_recognition.face_locations(rgb, model='hog', number_of_times_to_upsample=2)
            if not locs:
                errors.append(f"Image {i}: no face detected")
                continue

            encs = face_recognition.face_encodings(rgb, locs)
            if encs:
                encodings.append(encs[0])
            else:
                errors.append(f"Image {i}: location found but encoding failed")

        except Exception as e:
            errors.append(f"Image {i}: {str(e)}")
            continue

    if len(encodings) < 1:
        return jsonify({
            'error': f'Could not detect a face. Debug: {"; ".join(errors)}'
        }), 400

    # Save to pickle store
    store = load_encodings()
    store[name] = encodings
    save_encodings(store)

    # Save to DB
    existing = Person.query.filter_by(name=name).first()
    if not existing:
        db.session.add(Person(name=name))
        db.session.commit()

    return jsonify({'success': True, 'name': name, 'encodings_captured': len(encodings)})


@app.route('/api/persons', methods=['GET'])
def api_persons():
    persons = Person.query.order_by(Person.registered_at.desc()).all()
    return jsonify([p.to_dict() for p in persons])


@app.route('/api/persons/<int:person_id>', methods=['DELETE'])
def api_delete_person(person_id):
    person = Person.query.get_or_404(person_id)
    # Remove from encodings file
    store = load_encodings()
    store.pop(person.name, None)
    save_encodings(store)
    db.session.delete(person)
    db.session.commit()
    return jsonify({'success': True})


# Session Data 
@app.route('/api/sessions', methods=['GET'])
def api_sessions():
    person_id = request.args.get('person_id', type=int)
    q = GazeSession.query.order_by(GazeSession.started_at.desc())
    if person_id:
        q = q.filter_by(person_id=person_id)
    sessions = q.limit(100).all()
    return jsonify([s.to_dict() for s in sessions])


@app.route('/api/stats/summary', methods=['GET'])
def api_stats_summary():
    total_persons  = Person.query.count()
    total_sessions = GazeSession.query.count()
    sessions       = GazeSession.query.filter(GazeSession.total_frames > 0).all()
    avg_attention  = (
        round(sum(s.attention_score for s in sessions) / len(sessions), 1)
        if sessions else 0
    )
    return jsonify({
        'total_persons':  total_persons,
        'total_sessions': total_sessions,
        'avg_attention':  avg_attention,
    })


# Live Tracking 
@app.route('/api/tracking/start', methods=['POST'])
def api_start_tracking():
    if state.running:
        return jsonify({'error': 'Already running'}), 400
    state.running = True
    state.gaze    = GazeDetector(max_faces=4)   # fresh calibration
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    return jsonify({'success': True})


@app.route('/api/tracking/stop', methods=['POST'])
def api_stop_tracking():
    state.running = False
    return jsonify({'success': True})


@app.route('/api/tracking/status', methods=['GET'])
def api_tracking_status():
    with state.lock:
        stats = list(state.latest_stats)
    return jsonify({'running': state.running, 'detections': stats})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Entry Point

@app.route('/api/debug_image', methods=['POST'])
def debug_image():
    import base64, numpy as np, cv2
    data = request.get_json()
    b64 = data.get('image', '')
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return jsonify({'error': 'decode failed'})
    return jsonify({
        'shape': list(img.shape),
        'dtype': str(img.dtype),
        'bytes_len': len(img_bytes)
    })

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    return jsonify(metrics.compute())

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, threaded=True, port=5000)
