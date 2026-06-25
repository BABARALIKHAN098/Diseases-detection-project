import os
import sqlite3
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory,
)
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'app.db')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
IMAGE_DIR = os.path.join(BASE_DIR, 'img')

os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-secret'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

MODEL_PIPELINE = None

def get_model_pipeline():
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        try:
            MODEL_PIPELINE = PredictPipeline()
        except Exception as e:
            app.logger.error(f"Model pipeline initialization failed: {e}")
            MODEL_PIPELINE = None
    return MODEL_PIPELINE

FEATURE_NAMES = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
    'cad', 'appet', 'pe', 'ane', 'urine_abnormality', 'urine_protein_score',
    'renal_function_avg', 'renal_function_load', 'electrolyte_gap',
    'hemo_pcv_ratio', 'wbc_rc_ratio', 'age_bp_interaction',
    'comorbidity_flag', 'senior_age_flag'
]

FIELD_LABELS = {
    'age': 'Age',
    'bp': 'Blood pressure',
    'sg': 'Specific gravity',
    'al': 'Albumin',
    'su': 'Sugar',
    'rbc': 'Red blood cells',
    'pc': 'Pus cell',
    'pcc': 'Pus cell clumps',
    'ba': 'Ba',
    'bgr': 'Blood glucose random',
    'bu': 'Blood urea',
    'sc': 'Serum creatinine',
    'sod': 'Sodium',
    'pot': 'Potassium',
    'hemo': 'Hemoglobin',
    'pcv': 'PCV',
    'wc': 'WBC count',
    'rc': 'RBC count',
    'htn': 'Hypertension',
    'dm': 'Diabetes mellitus',
    'cad': 'Coronary artery disease',
    'appet': 'Appetite',
    'pe': 'Pedal edema',
    'ane': 'Anemia',
    'urine_abnormality': 'Urine abnormality',
    'urine_protein_score': 'Urine protein score',
    'renal_function_avg': 'Renal function average',
    'renal_function_load': 'Renal function load',
    'electrolyte_gap': 'Electrolyte gap',
    'hemo_pcv_ratio': 'Hemoglobin/PCV ratio',
    'wbc_rc_ratio': 'WBC/RBC ratio',
    'age_bp_interaction': 'Age-BP interaction',
    'comorbidity_flag': 'Comorbidity flag',
    'senior_age_flag': 'Senior age flag',
}

MODEL_PIPELINE = PredictPipeline()


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        '''
    )
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS patient_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            customer_name TEXT NOT NULL,
            contact_number TEXT NOT NULL,
            prediction TEXT NOT NULL,
            patient_data TEXT NOT NULL,
            report_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        '''
    )
    connection.commit()
    connection.close()


init_db()


@app.context_processor
def inject_year():
    return {'current_year': datetime.utcnow().year}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/img/<path:filename>')
def serve_img(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip().lower()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not fullname or not email or not username or not password:
            flash('All fields are required.', 'error')
            return redirect(url_for('signup'))

        password_hash = generate_password_hash(password)
        connection = get_db_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
                (fullname, email, username, password_hash),
            )
            connection.commit()
            flash('Your account has been created. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
            return redirect(url_for('signup'))
        finally:
            connection.close()

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        connection.close()

        if user and check_password_hash(user['password'], password):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful.', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid username or password.', 'error')
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))


@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('dashboard.html', prediction=None, report_url=None)


def create_pdf_report(report_path, customer_name, contact_number, prediction, patient_data):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib import colors

    canvas_obj = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    margin = inch
    y = height - margin
    canvas_obj.setFont('Helvetica-Bold', 20)
    canvas_obj.drawString(margin, y, 'D-Health AI Prediction Report')

    y -= 28
    canvas_obj.setFont('Helvetica', 11)
    canvas_obj.drawString(margin, y, f'Patient: {customer_name}')
    y -= 16
    canvas_obj.drawString(margin, y, f'Contact: {contact_number}')
    y -= 16
    canvas_obj.drawString(margin, y, f'Date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')

    y -= 30
    canvas_obj.setFont('Helvetica-Bold', 14)
    canvas_obj.drawString(margin, y, 'Prediction summary')
    y -= 22
    canvas_obj.setFont('Helvetica-Bold', 12)
    if prediction == 'High risk':
        canvas_obj.setFillColor(colors.red)
    else:
        canvas_obj.setFillColor(colors.green)
    canvas_obj.drawString(margin, y, prediction)
    canvas_obj.setFillColor(colors.black)

    y -= 28
    canvas_obj.setFont('Helvetica-Bold', 13)
    canvas_obj.drawString(margin, y, 'Submitted patient details')
    y -= 20
    canvas_obj.setFont('Helvetica', 10)

    for label, value in patient_data.items():
        if y < margin + 60:
            canvas_obj.showPage()
            y = height - margin
            canvas_obj.setFont('Helvetica', 10)

        canvas_obj.drawString(margin, y, f'{label}: {value}')
        y -= 14

    y -= 18
    if y >= margin:
        canvas_obj.setFont('Helvetica-Oblique', 10)
        canvas_obj.drawString(margin, y, 'This report is an automated prediction summary and does not replace medical advice.')

    canvas_obj.save()


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    raw_data = {}
    for field in FEATURE_NAMES:
        raw_value = request.form.get(field)
        if raw_value is None:
            flash(f'Missing required field: {FIELD_LABELS.get(field, field)}', 'error')
            return redirect(url_for('dashboard'))
        raw_data[field] = float(raw_value)

    customer_name = request.form.get('customer_name', '').strip()
    contact_number = request.form.get('contact_number', '').strip()
    if not customer_name or not contact_number:
        flash('Customer name and contact number are required.', 'error')
        return redirect(url_for('dashboard'))

    input_frame = pd.DataFrame([raw_data], columns=FEATURE_NAMES)
    pipeline = get_model_pipeline()
    if pipeline is None:
        flash('Prediction model is unavailable. Please contact the administrator.', 'error')
        return redirect(url_for('dashboard'))

    prediction_value = pipeline.predict(input_frame)[0]
    prediction_text = 'High risk' if int(prediction_value) == 1 else 'Low risk'

    patient_data = {
        FIELD_LABELS[field]: raw_data[field]
        for field in FEATURE_NAMES
    }

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    safe_name = ''.join(c for c in customer_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    report_filename = f'report_{safe_name}_{timestamp}.pdf'
    report_path = os.path.join(REPORTS_DIR, report_filename)

    create_pdf_report(report_path, customer_name, contact_number, prediction_text, patient_data)

    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        'INSERT INTO patient_reports (user_id, customer_name, contact_number, prediction, patient_data, report_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (
            session['user_id'],
            customer_name,
            contact_number,
            prediction_text,
            str(patient_data),
            report_filename,
            datetime.utcnow().isoformat(),
        ),
    )
    connection.commit()
    connection.close()

    report_url = url_for('download_report', filename=report_filename)
    return render_template('dashboard.html', prediction=prediction_text, report_url=report_url)


@app.route('/reports/<path:filename>')
def download_report(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
